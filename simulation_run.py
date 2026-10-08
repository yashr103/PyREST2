"""
simulation_run.py - Simulation Run backend (EM/NVT/NPT equilibration + REST2-REMD production)

Callable entry point: run_rest2_remd(...). Driven from the "Simulation run"
GUI page.

Based on the official OpenMM Cookbook REST tutorial:
https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/Running_a_REST_simulation.html

Pure OpenMM + openmmtools + ParmEd.

HOW IT WORKS
------------
1. Build a "vanilla" system from the Amber prmtop (ff19SB+GAFF2+OPC, etc.)
2. Convert it to a REST-capable system by replacing:
     HarmonicBondForce    -> CustomBondForce    (lambda_rest_bonds)
     HarmonicAngleForce   -> CustomAngleForce   (lambda_rest_angles)
     PeriodicTorsionForce -> CustomTorsionForce (lambda_rest_torsions)
     NonbondedForce       -> NonbondedForce     (lambda_rest_electrostatics,
                                                  lambda_rest_sterics via offsets)
   PME is preserved - the NonbondedForce is kept, just with parameter offsets.
3. Define RESTState (openmmtools GlobalParameterState) mapping one effective
   temperature to all five lambda parameters.
4. Create N thermodynamic states (one per replica), each at T_real=T_min but
   with different REST scaling (different T_eff).
5. Run REMD using openmmtools ReplicaExchangeSampler.

INSTALLATION
------------
    conda install -c conda-forge openmmtools
    # or
    pip install openmmtools

REFERENCE
---------
Wang et al. JCTC 2011, DOI: 10.1021/jp204407d
"""

import os
import sys
import json
import math
import copy
import pickle
import logging
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*openmmtools\.multistate.*experimental.*",
)

import numpy as np
import parmed as pmd

import openmm
from openmm import app, unit
from openmm.unit import (
    nanometer,
    kelvin,
    atmosphere,
    picosecond,
    femtoseconds,
    kilojoules_per_mole,
)

try:
    from openmmtools.constants import kB as kB_unit
    from openmmtools import mcmc
    from openmmtools import cache as openmmtools_cache
    from openmmtools.multistate import ReplicaExchangeSampler, MultiStateReporter
    from openmmtools.states import (
        GlobalParameterState,
        SamplerState,
        ThermodynamicState,
        CompoundThermodynamicState,
    )

    OPENMMTOOLS_AVAILABLE = True
except ImportError:
    OPENMMTOOLS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Fixed amino-acid resname set (not user-configurable)
# ═══════════════════════════════════════════════════════════════════════════

# Shared with residue_utils so the REST solute and the "Detect ligand" scan
# agree on what counts as protein. This includes the Amber protonation /
# disulfide variants (CYX, CYM, ASH, GLH, LYN) - tleap renames every
# disulfide-bonded cysteine to CYX, and leaving those out silently excluded
# them from REST scaling (bonded terms across them were treated as "inter").
from residue_utils import STANDARD_AA_RESNAMES, EXCLUDED_RESNAMES

AA_RESNAMES = set(STANDARD_AA_RESNAMES)


# ═══════════════════════════════════════════════════════════════════════════
# REST state class (verbatim from the official tutorial)
# ═══════════════════════════════════════════════════════════════════════════

if OPENMMTOOLS_AVAILABLE:

    class RESTState(GlobalParameterState):
        """
        OpenMMTools thermodynamic state that maps one effective temperature
        to all five REST lambda parameters. Copied verbatim from the official
        OpenMM Cookbook REST tutorial.
        """

        lambda_rest_bonds = GlobalParameterState.GlobalParameter(
            "lambda_rest_bonds", standard_value=1.0
        )
        lambda_rest_angles = GlobalParameterState.GlobalParameter(
            "lambda_rest_angles", standard_value=1.0
        )
        lambda_rest_torsions = GlobalParameterState.GlobalParameter(
            "lambda_rest_torsions", standard_value=1.0
        )
        lambda_rest_electrostatics = GlobalParameterState.GlobalParameter(
            "lambda_rest_electrostatics", standard_value=0.0
        )
        lambda_rest_sterics = GlobalParameterState.GlobalParameter(
            "lambda_rest_sterics", standard_value=0.0
        )

        def set_rest_parameters(self, beta_m, beta_0):
            """Set all lambda parameters given beta_m (1/kT_eff) and beta_0 (1/kT_ref)."""
            lambda_functions = {
                "lambda_rest_bonds": lambda bm, b0: float(np.sqrt(bm / b0)),
                "lambda_rest_angles": lambda bm, b0: float(np.sqrt(bm / b0)),
                "lambda_rest_torsions": lambda bm, b0: float(np.sqrt(bm / b0)),
                "lambda_rest_electrostatics": lambda bm, b0: float(np.sqrt(bm / b0) - 1),
                "lambda_rest_sterics": lambda bm, b0: float(bm / b0 - 1),
            }
            for param_name in self._parameters:
                if self._parameters[param_name] is not None:
                    setattr(
                        self, param_name, lambda_functions[param_name](beta_m, beta_0)
                    )

    def _patch_replica_exchange_sampler():
        """Work around an openmmtools + NumPy>=2.4 incompatibility in the
        neighbor-swap (Metropolis) exchange path.

        openmmtools' _mix_neighboring_replicas() locates replicas with

            replica_i = np.where(self._replica_thermodynamic_states == state)

        and passes the resulting 1-tuple (array([idx]),) straight into
        _attempt_swap(). Fancy indexing with tuple-of-array indices turns
        every energy - and therefore log_p_accept - into a shape-(1,) numpy
        array, and since NumPy 2.4, math.exp() on such an array raises

            TypeError: only 0-dimensional arrays can be converted to Python scalars

        Coercing the np.where() results to plain scalar ints fixes it without
        changing the exchange scheme or the acceptance statistics.
        """
        from openmmtools.multistate.replicaexchange import (
            ReplicaExchangeSampler as _ReplicaExchangeSampler,
        )

        _original_attempt_swap = _ReplicaExchangeSampler._attempt_swap

        def _safe_attempt_swap(self, replica_i, replica_j):
            if isinstance(replica_i, tuple):
                replica_i = int(replica_i[0][0])
            if isinstance(replica_j, tuple):
                replica_j = int(replica_j[0][0])
            return _original_attempt_swap(self, replica_i, replica_j)

        _ReplicaExchangeSampler._attempt_swap = _safe_attempt_swap

    _patch_replica_exchange_sampler()


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


def load_amber(prmtop, inpcrd):
    return pmd.load_file(prmtop, inpcrd)


def get_rest_atoms(topology, solute_resnames):
    """Return sorted list of REST atom indices (protein + ligand)."""
    return sorted(
        a.index for a in topology.atoms() if a.residue.name in solute_resnames
    )


# Production needs a periodic box (NPT equilibration, solvated system), so the
# non-periodic choices offered for plain system creation are rejected here.
PERIODIC_NONBONDED_METHODS = {
    "PME": app.PME,
    "Ewald": app.Ewald,
    "CutoffPeriodic": app.CutoffPeriodic,
}
CONSTRAINT_OPTIONS = {
    "HBonds": app.HBonds,
    "AllBonds": app.AllBonds,
    "HAngles": app.HAngles,
    "None": None,
}


def build_vanilla_system(
    amber,
    hydrogen_mass,
    temperature,
    with_barostat=False,
    pressure=1.0,
    barostat="Monte Carlo barostat",
    barostat_interval=25,
    nonbonded_method="PME",
    nonbonded_cutoff=1.0,
    constraints="HBonds",
    rigid_water=True,
    ewald_error_tolerance=0.0005,
):
    """Build a standard OpenMM system from an Amber prmtop/inpcrd (via ParmEd).
    hydrogen_mass <= 0 disables hydrogen mass repartitioning. The defaults are
    the settings every earlier version of this script hard-coded."""
    if nonbonded_method not in PERIODIC_NONBONDED_METHODS:
        raise ValueError(
            f"Nonbonded method '{nonbonded_method}' can't be used for REST2 "
            f"production - it needs a periodic method: "
            f"{', '.join(PERIODIC_NONBONDED_METHODS)}."
        )
    if constraints not in CONSTRAINT_OPTIONS:
        raise ValueError(f"Unsupported constraints option: {constraints}")
    system = amber.createSystem(
        nonbondedMethod=PERIODIC_NONBONDED_METHODS[nonbonded_method],
        nonbondedCutoff=nonbonded_cutoff * nanometer,
        constraints=CONSTRAINT_OPTIONS[constraints],
        rigidWater=rigid_water,
        hydrogenMass=hydrogen_mass * unit.amu if hydrogen_mass > 0 else None,
        ewaldErrorTolerance=ewald_error_tolerance,
        removeCMMotion=False,
    )
    if with_barostat:
        if barostat == "Monte Carlo membrane barostat":
            # No membrane-building step exists anywhere in this pipeline
            # (PDB Preprocessing / System Generation only build solvated
            # protein-ligand-in-water systems) - zero surface tension makes
            # this behave like the isotropic barostat, so it's harmless to
            # select, just not scientifically meaningful for anything this
            # app actually builds.
            system.addForce(
                openmm.MonteCarloMembraneBarostat(
                    pressure * atmosphere,
                    0.0 * openmm.unit.bar * openmm.unit.nanometer,
                    temperature * kelvin,
                    openmm.MonteCarloMembraneBarostat.XYIsotropic,
                    openmm.MonteCarloMembraneBarostat.ZFree,
                    barostat_interval,
                )
            )
        else:
            system.addForce(
                openmm.MonteCarloBarostat(
                    pressure * atmosphere, temperature * kelvin, barostat_interval
                )
            )
    return system


def get_rest_identifier(atoms, rest_atoms_set):
    """
    Return rest_id = [is_rest, is_inter, is_nonrest] for a set of atoms,
    or              [is_rest, is_nonrest]             for a single atom.
    Copied verbatim from the official OpenMM Cookbook REST tutorial.
    """
    if isinstance(atoms, int):
        if atoms in rest_atoms_set:
            return [1, 0]
        return [0, 1]
    elif isinstance(atoms, set):
        if atoms.intersection(rest_atoms_set):
            if atoms.issubset(rest_atoms_set):
                return [1, 0, 0]  # all REST
            else:
                return [0, 1, 0]  # inter
        return [0, 0, 1]  # all non-REST
    else:
        raise TypeError(f"atoms must be int or set, got {type(atoms)}")


def build_rest_cmap_correction_forces(cmap_force, rest_atoms_set):
    """
    REST scaling for CMAPTorsionForce. OpenMM has no parameter offsets for
    CMAP, so the original force is kept unchanged and one
    CustomCompoundBondForce per map adds

        (rest_scale - 1) * CMAP(phi, psi)

    with the same rest_scale convention as the torsions (lambda_rest_torsions).
    Total = rest_scale * CMAP, and at lambda = 1 (the reference state) the
    correction is exactly zero, so state 0 reproduces the vanilla energy
    exactly; scaled states only see spline-interpolation differences on the
    (rest_scale - 1) fraction. Terms that touch no REST atoms are skipped.
    """
    two_pi = 2.0 * math.pi
    terms_by_map = {}
    for i in range(cmap_force.getNumTorsions()):
        map_idx, a1, a2, a3, a4, b1, b2, b3, b4 = cmap_force.getTorsionParameters(i)
        rid = get_rest_identifier({a1, a2, a3, a4, b1, b2, b3, b4}, rest_atoms_set)
        if rid == [0, 0, 1]:
            continue
        terms_by_map.setdefault(map_idx, []).append(
            ((a1, a2, a3, a4, b1, b2, b3, b4), rid)
        )

    forces = []
    for map_idx, terms in sorted(terms_by_map.items()):
        size, energy = cmap_force.getMapParameters(map_idx)
        energy = np.asarray(
            [e.value_in_unit(kilojoules_per_mole) if unit.is_quantity(e) else e
             for e in energy],
            dtype=float,
        ).reshape(size, size, order="F")  # energy[i, j]: phi_i, psi_j
        # CMAP grid points sit at k*2pi/size for k = 0..size-1; append the
        # wrapped first row/column so the periodic spline spans [0, 2pi].
        table = np.zeros((size + 1, size + 1))
        table[:size, :size] = energy
        table[size, :] = table[0, :]
        table[:, size] = table[:, 0]
        fn_name = f"cmap{map_idx}"

        expr = (
            f"(rest_scale - 1) * {fn_name}(phi_w, psi_w);"
            "rest_scale = is_rest*lambda_rest_torsions*lambda_rest_torsions"
            " + is_inter*lambda_rest_torsions + is_nonrest;"
            f"phi_w = phi + {two_pi}*step(-phi);"
            f"psi_w = psi + {two_pi}*step(-psi);"
            "phi = dihedral(p1, p2, p3, p4);"
            "psi = dihedral(p5, p6, p7, p8);"
        )
        force = openmm.CustomCompoundBondForce(8, expr)
        force.addGlobalParameter("lambda_rest_torsions", 1.0)
        for p in ["is_rest", "is_inter", "is_nonrest"]:
            force.addPerBondParameter(p)
        force.addTabulatedFunction(
            fn_name,
            openmm.Continuous2DFunction(
                size + 1, size + 1, table.flatten(order="F"),
                0.0, two_pi, 0.0, two_pi, True,
            ),
        )
        if cmap_force.usesPeriodicBoundaryConditions():
            force.setUsesPeriodicBoundaryConditions(True)
        for particles, rid in terms:
            force.addBond(list(particles), rid)
        forces.append(force)
    return forces


def build_rest_system(vanilla_system, rest_atoms):
    """
    Convert a vanilla OpenMM system to a REST-capable system, following the
    official OpenMM Cookbook tutorial exactly. PME is fully preserved.
    """
    rest_atoms_set = set(rest_atoms)

    rest_system = openmm.System()
    vanilla_forces = {type(f).__name__: f for f in vanilla_system.getForces()}

    for i in range(vanilla_system.getNumParticles()):
        rest_system.addParticle(vanilla_system.getParticleMass(i))

    if "MonteCarloBarostat" in vanilla_forces:
        rest_system.addForce(copy.deepcopy(vanilla_forces["MonteCarloBarostat"]))

    rest_system.setDefaultPeriodicBoxVectors(
        *vanilla_system.getDefaultPeriodicBoxVectors()
    )

    for i in range(vanilla_system.getNumConstraints()):
        a1, a2, length = vanilla_system.getConstraintParameters(i)
        rest_system.addConstraint(a1, a2, length)

    # ── BONDS ────────────────────────────────────────────────────────────
    bond_expr = "rest_scale * (K/2) * (r-length)^2;"
    bond_expr += (
        "rest_scale = is_rest*lambda_rest_bonds*lambda_rest_bonds"
        " + is_inter*lambda_rest_bonds + is_nonrest;"
    )

    rest_bond = openmm.CustomBondForce(bond_expr)
    rest_bond.addGlobalParameter("lambda_rest_bonds", 1.0)
    for p in ["is_rest", "is_inter", "is_nonrest", "length", "K"]:
        rest_bond.addPerBondParameter(p)

    vf = vanilla_forces["HarmonicBondForce"]
    if vf.usesPeriodicBoundaryConditions():
        rest_bond.setUsesPeriodicBoundaryConditions(True)
    for i in range(vf.getNumBonds()):
        p1, p2, r0, k = vf.getBondParameters(i)
        rid = get_rest_identifier({p1, p2}, rest_atoms_set)
        rest_bond.addBond(p1, p2, rid + [r0, k])
    rest_system.addForce(rest_bond)

    # ── ANGLES ───────────────────────────────────────────────────────────
    angle_expr = "rest_scale * (K/2) * (theta-theta0)^2;"
    angle_expr += (
        "rest_scale = is_rest*lambda_rest_angles*lambda_rest_angles"
        " + is_inter*lambda_rest_angles + is_nonrest;"
    )

    rest_angle = openmm.CustomAngleForce(angle_expr)
    rest_angle.addGlobalParameter("lambda_rest_angles", 1.0)
    for p in ["is_rest", "is_inter", "is_nonrest", "theta0", "K"]:
        rest_angle.addPerAngleParameter(p)

    vf = vanilla_forces["HarmonicAngleForce"]
    if vf.usesPeriodicBoundaryConditions():
        rest_angle.setUsesPeriodicBoundaryConditions(True)
    for i in range(vf.getNumAngles()):
        p1, p2, p3, t0, k = vf.getAngleParameters(i)
        rid = get_rest_identifier({p1, p2, p3}, rest_atoms_set)
        rest_angle.addAngle(p1, p2, p3, rid + [t0, k])
    rest_system.addForce(rest_angle)

    # ── TORSIONS ─────────────────────────────────────────────────────────
    torsion_expr = "rest_scale * U;"
    torsion_expr += (
        "rest_scale = is_rest*lambda_rest_torsions*lambda_rest_torsions"
        " + is_inter*lambda_rest_torsions + is_nonrest;"
    )
    torsion_expr += "U = K*(1+cos(periodicity*theta-phase));"

    rest_torsion = openmm.CustomTorsionForce(torsion_expr)
    rest_torsion.addGlobalParameter("lambda_rest_torsions", 1.0)
    for p in ["is_rest", "is_inter", "is_nonrest", "periodicity", "phase", "K"]:
        rest_torsion.addPerTorsionParameter(p)

    vf = vanilla_forces["PeriodicTorsionForce"]
    if vf.usesPeriodicBoundaryConditions():
        rest_torsion.setUsesPeriodicBoundaryConditions(True)
    for i in range(vf.getNumTorsions()):
        p1, p2, p3, p4, per, phase, K = vf.getTorsionParameters(i)
        rid = get_rest_identifier({p1, p2, p3, p4}, rest_atoms_set)
        rest_torsion.addTorsion(p1, p2, p3, p4, rid + [per, phase, K])
    rest_system.addForce(rest_torsion)

    # ── NONBONDED (PME preserved via offsets) ───────────────────────────
    rest_nb = openmm.NonbondedForce()

    vf = vanilla_forces["NonbondedForce"]
    nb_method = vf.getNonbondedMethod()
    rest_nb.setNonbondedMethod(nb_method)

    if nb_method != openmm.NonbondedForce.NoCutoff:
        rest_nb.setReactionFieldDielectric(vf.getReactionFieldDielectric())
        rest_nb.setCutoffDistance(vf.getCutoffDistance())

    if nb_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
        alpha, nx, ny, nz = vf.getPMEParameters()
        rest_nb.setPMEParameters(alpha, nx, ny, nz)
        rest_nb.setEwaldErrorTolerance(vf.getEwaldErrorTolerance())

    rest_nb.setUseSwitchingFunction(vf.getUseSwitchingFunction())
    if vf.getUseSwitchingFunction():
        rest_nb.setSwitchingDistance(vf.getSwitchingDistance())

    rest_nb.setUseDispersionCorrection(vf.getUseDispersionCorrection())

    rest_nb.addGlobalParameter("lambda_rest_electrostatics", 0.0)
    rest_nb.addGlobalParameter("lambda_rest_sterics", 0.0)

    for i in range(vf.getNumParticles()):
        q, sigma, eps = vf.getParticleParameters(i)
        rid = get_rest_identifier(i, rest_atoms_set)
        rest_nb.addParticle(q, sigma, eps)
        if rid == [1, 0]:  # REST atom - add offsets
            rest_nb.addParticleParameterOffset(
                "lambda_rest_electrostatics", i, q, 0.0 * sigma, eps * 0.0
            )
            rest_nb.addParticleParameterOffset(
                "lambda_rest_sterics", i, q * 0.0, 0.0 * sigma, eps
            )

    for i in range(vf.getNumExceptions()):
        p1, p2, chargeProd, sigma, eps = vf.getExceptionParameters(i)
        rid = get_rest_identifier({p1, p2}, rest_atoms_set)
        exc_idx = rest_nb.addException(p1, p2, chargeProd, sigma, eps)

        if rid == [1, 0, 0]:  # both REST
            rest_nb.addExceptionParameterOffset(
                "lambda_rest_sterics", exc_idx, chargeProd, 0.0 * sigma, eps
            )
        elif rid == [0, 1, 0]:  # inter
            rest_nb.addExceptionParameterOffset(
                "lambda_rest_electrostatics", exc_idx, chargeProd, 0.0 * sigma, eps
            )

    rest_system.addForce(rest_nb)

    # ── CMAP (ff19SB / ff14SB-style backbone correction maps) ────────────
    # CMAP is a torsion-type term and must be scaled like the torsions above;
    # leaving it unscaled holds the backbone phi/psi surface at full strength
    # in every replica, so the heated replicas barely sample new backbone
    # conformations.
    if "CMAPTorsionForce" in vanilla_forces:
        rest_system.addForce(copy.deepcopy(vanilla_forces["CMAPTorsionForce"]))
        for f in build_rest_cmap_correction_forces(
            vanilla_forces["CMAPTorsionForce"], rest_atoms_set
        ):
            rest_system.addForce(f)

    skip = {
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
        "MonteCarloBarostat",
        "CMAPTorsionForce",
    }
    for f in vanilla_system.getForces():
        if type(f).__name__ not in skip:
            rest_system.addForce(copy.deepcopy(f))

    return rest_system


def _exponential_temperature_ladder(t_min, t_max, n_replicas):
    """Exponential spacing between t_min and t_max (official tutorial convention)."""
    if n_replicas == 1:
        return [t_min]
    return [
        t_min
        + (t_max - t_min)
        * (math.exp(float(i) / float(n_replicas - 1)) - 1.0)
        / (math.e - 1.0)
        for i in range(n_replicas)
    ]


def _linear_temperature_ladder(t_min, t_max, n_replicas):
    if n_replicas == 1:
        return [t_min]
    step = (t_max - t_min) / (n_replicas - 1)
    return [t_min + i * step for i in range(n_replicas)]


def _as_float(value, name):
    """Coerce a number, numpy scalar, or size-1 numpy array to a plain float.
    GUI frameworks sometimes hand us numpy scalars or size-1 arrays instead of
    plain Python numbers. On NumPy >= 2.4, passing those on to OpenMM's C++
    layer (or to float()) raises
        TypeError: only 0-dimensional arrays can be converted to Python scalars
    so every scalar coming in from outside is normalised here."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{name} must be a single number, got {value!r}")
        return float(value.ravel()[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"{name} must be a single number, got {value!r}")
    return float(arr.ravel()[0])


def _as_int(value, name):
    """Coerce a number, numpy scalar, or size-1 numpy array to a plain int."""
    return int(round(_as_float(value, name)))


def _resume_sampler(storage_path, n_iterations, n_replicas, temperatures, timestep,
                    n_steps_per_iter, checkpoint_interval, log):
    """
    Reopens an existing openmmtools storage file and returns
    (sampler, stored_settings). The sampler is rewound to its last checkpoint.

    from_storage() is a classmethod that RETURNS a new sampler; its result must
    be used rather than calling it on an existing instance.

    Everything that defines the simulation (REST system, temperature ladder,
    timestep, steps per iteration, exchange scheme, checkpoint interval) is
    read back from storage, not from the arguments. Arguments that disagree
    are reported, and the stored values are returned so the log and
    run_metadata.json describe what the resumed run actually uses.
    """
    sampler = ReplicaExchangeSampler.from_storage(storage_path)
    log.info(f"Resumed from storage at iteration {sampler.iteration} (last checkpoint).")

    if sampler.n_states != n_replicas:
        raise ValueError(
            f"The storage file has {sampler.n_states} thermodynamic states but "
            f"Number of replicas is {n_replicas}. A resumed run always continues "
            f"with the stored ladder - set the replica count back to "
            f"{sampler.n_states}, or start a new run in a different output directory."
        )

    stored = {}
    try:
        stored_temps = []
        for state in sampler._thermodynamic_states:
            lam = float(state.lambda_rest_bonds)  # lambda = sqrt(T_min / T_eff)
            stored_temps.append(state.temperature.value_in_unit(kelvin) / (lam * lam))
        stored["temperatures"] = stored_temps
        if max(abs(a - b) for a, b in zip(stored_temps, temperatures)) > 0.05:
            log.warning(
                "  Temperature ladder in the GUI/arguments differs from the stored "
                "one - continuing with the stored ladder: "
                + ", ".join(f"{t:.1f}" for t in stored_temps) + " K"
            )
    except Exception as exc:  # noqa: BLE001 - informational only
        log.warning(f"  Could not read the stored temperature ladder ({exc}).")

    try:
        moves = sampler.mcmc_moves
        move = moves[0] if isinstance(moves, (list, tuple)) else moves
        stored["timestep_fs"] = float(move.timestep.value_in_unit(femtoseconds))
        stored["n_steps_per_iter"] = int(move.n_steps)
        if (abs(stored["timestep_fs"] - timestep) > 1e-6
                or stored["n_steps_per_iter"] != n_steps_per_iter):
            log.warning(
                f"  Timestep/exchange interval in the GUI/arguments ({timestep} fs, "
                f"{n_steps_per_iter} steps) differ from the stored run - continuing "
                f"with {stored['timestep_fs']} fs, {stored['n_steps_per_iter']} steps."
            )
    except Exception as exc:  # noqa: BLE001 - informational only
        log.warning(f"  Could not read the stored MCMC move ({exc}).")

    try:
        stored["checkpoint_interval"] = int(sampler._reporter.checkpoint_interval)
        if stored["checkpoint_interval"] != checkpoint_interval:
            log.warning(
                f"  Checkpoint interval {checkpoint_interval} differs from the stored "
                f"run - continuing with {stored['checkpoint_interval']}."
            )
    except Exception:  # noqa: BLE001 - informational only
        pass

    # openmmtools' online free-energy (MBAR) analysis isn't used by REST2 and
    # re-analyses the whole growing dataset every 200 iterations; runs created
    # by older versions of this script stored it as enabled.
    if sampler.online_analysis_interval is not None:
        sampler.online_analysis_interval = None

    if sampler.number_of_iterations != n_iterations:
        log.info(
            f"  Total iterations: {sampler.number_of_iterations} (stored) -> "
            f"{n_iterations} (requested)."
        )
        # Must go through the property so the new limit is written to storage;
        # run() never goes past it.
        sampler.number_of_iterations = n_iterations
    return sampler, stored


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def run_rest2_remd(
    prmtop,
    inpcrd,
    ligand_resnames=None,
    protein_only=False,
    n_replicas=16,
    t_min=300.0,
    t_max=400.0,
    temperature_distribution="Exponential spacing",
    custom_temperatures=None,
    timestep=2.0,
    friction=1.0,
    hydrogen_mass=1.5,
    nonbonded_method="PME",
    nonbonded_cutoff=1.0,
    constraints="HBonds",
    rigid_water=True,
    ewald_error_tolerance=0.0005,
    em_max_iter=5000,
    em_tolerance=10.0,
    npt_equil_steps=500000,
    npt_pressure=1.0,
    npt_barostat="Monte Carlo barostat",
    npt_barostat_interval=25,
    nvt_equil_steps=500000,
    nvt_temperature=None,
    nvt_thermostat="Langevin middle integrator",
    nvt_restrain_heavy_atoms=False,
    nvt_restraint_force_constant=4184.0,
    n_iterations=1000,
    n_steps_per_iter=1000,
    checkpoint_interval=200,
    exchange_scheme="Neighbor swap (Metropolis)",
    output_dir="rest2_output",
    platform_name="CUDA",
    restart=False,
    log_callback=None,
):
    """
    Runs EM -> NPT -> NVT equilibration, then REST2 replica exchange production.

    protein_only: apo / ligand-free run. The REST solute is the protein alone,
    ligand_resnames must be empty, and any non-standard residues left in the
    topology are reported (they stay unscaled, like solvent). Without it,
    every name in ligand_resnames must exist in the topology - a mistyped
    name used to be silently ignored, leaving the ligand untempered.

    Writes <output_dir>/run_metadata.json (system type, exact temperature
    ladder, timing) so analysis.py can reproduce the production settings.

    custom_temperatures: optional explicit list of effective temperatures (K),
    one per replica, used instead of computing a ladder from t_min/t_max/
    temperature_distribution. Must have exactly n_replicas values, sorted
    ascending, with the first value equal to t_min - state 0 is always the
    unscaled reference state (T_eff = t_min = the real simulation temperature),
    so a custom list that doesn't start at t_min would misrepresent which
    state is actually your target ensemble.

    nvt_temperature: target temperature for the NVT equilibration stage only.
    Defaults to t_min if not given. If you explicitly set it to something
    other than t_min, a warning is logged: production always runs at the
    real temperature t_min (REST2 only scales the Hamiltonian, not the real
    temperature), so equilibrating NVT at a different temperature leaves the
    system out of equilibrium the moment production starts.

    nvt_thermostat: "Langevin middle integrator" (default), "Nose-Hoover", or
    "Andersen" - applies ONLY to the NVT equilibration stage. EM+NPT and
    production always use LangevinMiddleIntegrator / openmmtools'
    LangevinDynamicsMove respectively, since those are the standard, well-
    validated choices for barostatted equilibration and replica exchange.

    nvt_restrain_heavy_atoms: if True, restrains solute (protein+ligand)
    heavy atoms to their starting positions during NVT with a harmonic
    force (nvt_restraint_force_constant, kJ/mol/nm^2, ~10 kcal/mol/A^2 by
    default) - lets solvent/hydrogens relax without the solute drifting
    during initial equilibration. Water and hydrogens are never restrained.

    npt_barostat: "Monte Carlo barostat" (default, isotropic) or "Monte
    Carlo membrane barostat" (anisotropic, for membrane systems). NOTE: this
    pipeline has no membrane-building step anywhere (PDB Preprocessing /
    System Generation only build solvated protein-ligand-in-water systems),
    so the membrane option is implemented for completeness but isn't
    scientifically meaningful for anything this app actually builds - it
    defaults to zero surface tension, which makes it behave like the
    isotropic barostat anyway.

    exchange_scheme: "Neighbor swap (Metropolis)" (default) or "Gibbs
    sampling", passed to openmmtools' replica_mixing_scheme if the installed
    openmmtools version supports it; falls back to the library default with
    a warning if not.

    log_callback(str): optional callable invoked with each log line, so a GUI
    can stream progress instead of only reading the log file.

    Returns the path to the openmmtools NetCDF storage file.
    """
    # Normalise every scalar coming from the caller (the GUI may pass numpy
    # scalars or size-1 arrays, which crash OpenMM's C++ layer on NumPy >= 2.4
    # with "only 0-dimensional arrays can be converted to Python scalars").
    n_replicas = _as_int(n_replicas, "n_replicas")
    t_min = _as_float(t_min, "t_min")
    t_max = _as_float(t_max, "t_max")
    timestep = _as_float(timestep, "timestep")
    friction = _as_float(friction, "friction")
    hydrogen_mass = _as_float(hydrogen_mass, "hydrogen_mass")
    nonbonded_cutoff = _as_float(nonbonded_cutoff, "nonbonded_cutoff")
    ewald_error_tolerance = _as_float(ewald_error_tolerance, "ewald_error_tolerance")
    em_max_iter = _as_int(em_max_iter, "em_max_iter")
    em_tolerance = _as_float(em_tolerance, "em_tolerance")
    npt_equil_steps = _as_int(npt_equil_steps, "npt_equil_steps")
    npt_pressure = _as_float(npt_pressure, "npt_pressure")
    npt_barostat_interval = _as_int(npt_barostat_interval, "npt_barostat_interval")
    nvt_equil_steps = _as_int(nvt_equil_steps, "nvt_equil_steps")
    if nvt_temperature is not None:
        nvt_temperature = _as_float(nvt_temperature, "nvt_temperature")
    nvt_restraint_force_constant = _as_float(
        nvt_restraint_force_constant, "nvt_restraint_force_constant"
    )
    n_iterations = _as_int(n_iterations, "n_iterations")
    n_steps_per_iter = _as_int(n_steps_per_iter, "n_steps_per_iter")
    checkpoint_interval = _as_int(checkpoint_interval, "checkpoint_interval")

    if not OPENMMTOOLS_AVAILABLE:
        raise ImportError(
            "openmmtools is not installed. Install with:\n"
            "  conda install -c conda-forge openmmtools\n"
            "or:\n"
            "  pip install openmmtools"
        )

    if n_replicas < 1:
        raise ValueError(f"n_replicas must be >= 1, got {n_replicas}")
    if not os.path.isfile(prmtop):
        raise FileNotFoundError(f"prmtop not found: {prmtop}")
    if not os.path.isfile(inpcrd):
        raise FileNotFoundError(f"inpcrd not found: {inpcrd}")

    if nvt_temperature is None:
        nvt_temperature = t_min

    if temperature_distribution == "Custom list" and not custom_temperatures:
        raise ValueError(
            "Temperature distribution is 'Custom list' but no custom temperatures "
            "were given."
        )
    if n_replicas > 1 and not custom_temperatures and t_max <= t_min:
        raise ValueError(
            f"T max ({t_max} K) must be higher than T min ({t_min} K) - otherwise "
            f"the 'hot' replicas would be scaled towards a colder solute."
        )

    # Output-format settings, applied identically to NPT, NVT and production.
    system_kwargs = dict(
        nonbonded_method=nonbonded_method,
        nonbonded_cutoff=nonbonded_cutoff,
        constraints=constraints,
        rigid_water=bool(rigid_water),
        ewald_error_tolerance=ewald_error_tolerance,
    )
    if nonbonded_method not in PERIODIC_NONBONDED_METHODS:
        raise ValueError(
            f"Nonbonded method '{nonbonded_method}' can't be used for REST2 "
            f"production - choose {', '.join(PERIODIC_NONBONDED_METHODS)} on the "
            f"Output format page."
        )
    if constraints not in CONSTRAINT_OPTIONS:
        raise ValueError(f"Unsupported constraints option: {constraints}")

    ligand_resnames = {s.strip().upper() for s in (ligand_resnames or set()) if s.strip()}
    if protein_only and ligand_resnames:
        raise ValueError(
            f"Protein-only run requested, but ligand residue names were also "
            f"given ({', '.join(sorted(ligand_resnames))}). Clear them, or switch "
            f"System type to 'Protein-ligand complex'."
        )
    solute_resnames = AA_RESNAMES | ligand_resnames
    solute_desc = "protein" if not ligand_resnames else "protein + ligand"

    os.makedirs(output_dir, exist_ok=True)
    storage_path = os.path.join(output_dir, "rest2_remd.nc")
    equil_pkl = os.path.join(output_dir, "equil_state.pkl")

    if restart:
        # Resuming only needs the production storage (it holds the REST system,
        # ladder, MCMC move and replica positions at every checkpoint) - not
        # equil_state.pkl, which is just the pre-production structure.
        checkpoint_path = os.path.join(output_dir, "rest2_remd_checkpoint.nc")
        missing = [p for p in (storage_path, checkpoint_path) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"--restart was given but the production storage is incomplete - "
                f"missing: {', '.join(missing)}\nResume needs a production run that "
                f"reached its first checkpoint in this output directory. Otherwise "
                f"uncheck 'Resume from previous run' and start fresh."
            )

    log = logging.getLogger(f"simulation_run.{id(object())}")
    log.setLevel(logging.INFO)
    log.propagate = False
    log.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(message)s")
    file_handler = logging.FileHandler(os.path.join(output_dir, "rest2_remd.log"))
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    if log_callback is not None:

        class _CallbackHandler(logging.Handler):
            def emit(self_inner, record):
                log_callback(self_inner.format(record))

        cb_handler = _CallbackHandler()
        cb_handler.setFormatter(formatter)
        log.addHandler(cb_handler)
    else:
        log.addHandler(logging.StreamHandler(sys.stdout))

    # Forward openmmtools' own logger into this run's log, count NaN restarts
    # for the progress line, and drop the "API is experimental" notice it
    # repeats on every analyser instantiation.
    nan_restarts = {"count": 0}

    class _OpenmmtoolsForwarder(logging.Handler):
        def emit(self_inner, record):
            message = record.getMessage()
            if "API is experimental" in message:
                return
            if "Potential energy is NaN" in message:
                nan_restarts["count"] += 1
            log.log(record.levelno, f"openmmtools: {message}")

    openmmtools_logger = logging.getLogger("openmmtools")
    for handler in list(openmmtools_logger.handlers):
        if isinstance(handler, logging.Handler) and type(handler).__name__ == "_OpenmmtoolsForwarder":
            openmmtools_logger.removeHandler(handler)
    openmmtools_logger.addHandler(_OpenmmtoolsForwarder(level=logging.WARNING))

    if abs(nvt_temperature - t_min) > 1e-6:
        log.warning(
            f"NVT target temperature ({nvt_temperature} K) differs from T min "
            f"({t_min} K). Production always runs at the real temperature "
            f"T min - equilibrating NVT at a different temperature will leave "
            f"the system out of equilibrium the moment production starts."
        )

    # ── Constants + platform ────────────────────────────────────────────
    kB_val = (
        openmm.unit.BOLTZMANN_CONSTANT_kB * openmm.unit.AVOGADRO_CONSTANT_NA
    ).value_in_unit(kilojoules_per_mole / kelvin)

    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name in ("CUDA", "OpenCL", "HIP"):
            platform_props = {"DeviceIndex": "0", "Precision": "mixed"}
        else:
            platform_props = {}
        log.info(f"Platform: {platform_name}")
    except Exception:
        log.warning(f"{platform_name} not available - using CPU.")
        platform = openmm.Platform.getPlatformByName("CPU")
        platform_props = {}

    # Tell openmmtools which platform to use for its OWN internal Contexts.
    # Without this, ReplicaExchangeSampler/LangevinDynamicsMove use
    # openmmtools' default platform selection during production - completely
    # ignoring the platform chosen above, which otherwise only reaches the
    # EM/NPT/NVT equilibration Simulation objects further down.
    try:
        openmmtools_cache.global_context_cache = openmmtools_cache.ContextCache(
            platform=platform,
            platform_properties=platform_props if platform_props else None,
        )
        log.info(f"  openmmtools context cache platform set to: {platform.getName()}")
    except Exception as e:
        log.warning(
            f"Could not set openmmtools' platform for production ({e}) - "
            f"it may fall back to its own default platform selection for "
            f"the REST2-REMD run, even though equilibration used {platform_name}."
        )

    # ── Load topology + identify REST atoms ─────────────────────────────
    log.info("Loading topology...")
    _amber_ref = load_amber(prmtop, inpcrd)
    topology_ref = _amber_ref.topology
    n_atoms = _amber_ref.ptr("natom")

    present_resnames = {r.name.strip().upper() for r in topology_ref.residues()}
    hetero_resnames = sorted(present_resnames - EXCLUDED_RESNAMES)

    missing = sorted(ligand_resnames - present_resnames)
    if missing:
        raise ValueError(
            f"Ligand residue name(s) not found in the topology: {', '.join(missing)}. "
            f"Non-standard residues present: {', '.join(hetero_resnames) or 'none'}"
            f"{' - for an apo system use the protein-only mode.' if not hetero_resnames else '.'}"
        )

    rest_atoms = get_rest_atoms(topology_ref, solute_resnames)

    log.info(f"  System type  : {'protein only' if protein_only else 'protein-ligand complex'}")
    log.info(f"  Total atoms  : {n_atoms}")
    log.info(f"  REST atoms   : {len(rest_atoms)}  ({solute_desc} = solute)")
    log.info(f"  Solvent atoms: {n_atoms - len(rest_atoms)}")

    unscaled_hetero = [r for r in hetero_resnames if r not in ligand_resnames]
    if unscaled_hetero:
        log.warning(
            f"  Non-standard residue(s) {', '.join(unscaled_hetero)} are NOT part of "
            f"the REST solute and will be simulated unscaled."
        )

    if len(rest_atoms) == 0:
        raise ValueError(
            "No REST atoms found - the topology contains no standard amino-acid "
            "residues and no matching ligand residues."
        )

    # ── Temperature ladder ───────────────────────────────────────────────
    if custom_temperatures:
        temperatures = [float(t) for t in custom_temperatures]
        if len(temperatures) != n_replicas:
            raise ValueError(
                f"custom_temperatures has {len(temperatures)} value(s) but "
                f"n_replicas={n_replicas} - they must match exactly."
            )
        if temperatures != sorted(temperatures):
            raise ValueError(
                "custom_temperatures must be sorted ascending (state 0 = "
                "lowest/reference temperature)."
            )
        if abs(temperatures[0] - t_min) > 1e-6:
            raise ValueError(
                f"custom_temperatures[0] ({temperatures[0]} K) must equal t_min "
                f"({t_min} K) - state 0 is always the unscaled reference state "
                f"at the real simulation temperature."
            )
        log.info(f"Using custom temperature ladder: {temperatures}")
    elif temperature_distribution == "Linear spacing":
        temperatures = _linear_temperature_ladder(t_min, t_max, n_replicas)
    else:
        temperatures = _exponential_temperature_ladder(t_min, t_max, n_replicas)

    log.info("Temperature ladder (K):")
    for i, T in enumerate(temperatures):
        beta_m = 1.0 / (kB_val * T)
        beta_0 = 1.0 / (kB_val * t_min)
        lam = np.sqrt(beta_m / beta_0)
        log.info(f"  Replica {i:2d}: T_eff={T:.1f} K  lambda={lam:.4f}")

    metadata = {
        "system_type": "protein_only" if protein_only else "protein_ligand",
        "protein_only": bool(protein_only),
        "ligand_resnames": sorted(ligand_resnames),
        "n_replicas": n_replicas,
        "t_min": t_min,
        "t_max": t_max,
        "temperature_distribution": (
            "Custom list" if custom_temperatures else temperature_distribution
        ),
        "temperatures": [float(T) for T in temperatures],
        "timestep_fs": timestep,
        "n_steps_per_iter": n_steps_per_iter,
        "n_iterations": n_iterations,
        "checkpoint_interval": checkpoint_interval,
        "hydrogen_mass_amu": hydrogen_mass if hydrogen_mass > 0 else None,
        "nonbonded_method": nonbonded_method,
        "nonbonded_cutoff_nm": nonbonded_cutoff,
        "constraints": constraints,
        "rigid_water": bool(rigid_water),
        "ewald_error_tolerance": ewald_error_tolerance,
        "prmtop": os.path.abspath(prmtop),
        "inpcrd": os.path.abspath(inpcrd),
        "n_atoms": int(n_atoms),
        "n_rest_atoms": len(rest_atoms),
    }
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    if not restart:
        # On restart this is written after the stored settings are read back.
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

    def _make_integrator(seed=42):
        integ = openmm.LangevinMiddleIntegrator(
            t_min * kelvin, friction / picosecond, timestep * femtoseconds
        )
        integ.setRandomNumberSeed(seed)
        return integ

    def _make_nvt_integrator_and_force(seed=200):
        """Returns (integrator, extra_force_or_None) for the chosen NVT
        thermostat. Andersen needs a plain Verlet integrator plus a separate
        AndersenThermostat Force; the other two are self-contained integrators."""
        if nvt_thermostat == "Nose-Hoover":
            integ = openmm.NoseHooverIntegrator(
                nvt_temperature * kelvin, friction / picosecond, timestep * femtoseconds
            )
            try:
                integ.setRandomNumberSeed(seed)
            except AttributeError:
                pass
            return integ, None
        elif nvt_thermostat == "Andersen":
            integ = openmm.VerletIntegrator(timestep * femtoseconds)
            force = openmm.AndersenThermostat(
                nvt_temperature * kelvin, friction / picosecond
            )
            try:
                force.setRandomNumberSeed(seed)
            except AttributeError:
                pass
            return integ, force
        else:  # "Langevin middle integrator" (default)
            integ = openmm.LangevinMiddleIntegrator(
                nvt_temperature * kelvin, friction / picosecond, timestep * femtoseconds
            )
            integ.setRandomNumberSeed(seed)
            return integ, None

    def _build_heavy_atom_restraint_force(
        amber_structure, positions_nm, atom_indices, force_constant
    ):
        """Harmonic positional restraint on the given atoms' CURRENT starting
        positions - used to restrain solute heavy atoms during NVT so
        solvent/hydrogens can relax without the solute drifting.

        positions_nm must be a plain (unit-stripped) numpy array in nanometers,
        matching whatever coordinates the simulation is actually about to
        start from - NOT the original pre-equilibration input structure,
        which would anchor atoms to the wrong reference point entirely."""
        heavy_indices = [
            i for i in atom_indices if amber_structure.atoms[i].atomic_number != 1
        ]
        restraint_force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint_force.addGlobalParameter(
            "k", force_constant * kilojoules_per_mole / nanometer**2
        )
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")
        for i in heavy_indices:
            x0, y0, z0 = positions_nm[i]
            restraint_force.addParticle(i, [float(x0), float(y0), float(z0)])
        return restraint_force, len(heavy_indices)

    # ── Equilibration (EM -> NPT -> NVT, vanilla system) ─────────────────
    if not restart:
        log.info("EQUILIBRATION (EM -> NPT -> NVT)")
        log.info(
            f"  System: {nonbonded_method}, cutoff {nonbonded_cutoff} nm, constraints "
            f"{constraints}, rigid water {bool(rigid_water)}, Ewald tolerance "
            f"{ewald_error_tolerance}, "
            + (f"HMR {hydrogen_mass} amu" if hydrogen_mass > 0 else "no HMR")
        )

        log.info(
            f"[1/3] EM + NPT ({npt_equil_steps} steps, target {npt_pressure} atm, "
            f"{npt_barostat}, update every {npt_barostat_interval} steps)..."
        )
        amber_eq = load_amber(prmtop, inpcrd)
        sys_npt = build_vanilla_system(
            amber_eq,
            hydrogen_mass,
            t_min,
            with_barostat=True,
            pressure=npt_pressure,
            barostat=npt_barostat,
            barostat_interval=npt_barostat_interval,
            **system_kwargs,
        )
        sim_npt = app.Simulation(
            topology_ref, sys_npt, _make_integrator(100), platform, platform_props
        )
        sim_npt.context.setPositions(amber_eq.positions)
        sim_npt.minimizeEnergy(
            tolerance=em_tolerance * kilojoules_per_mole / nanometer,
            maxIterations=em_max_iter,
        )
        sim_npt.context.setVelocitiesToTemperature(t_min * kelvin)
        sim_npt.step(npt_equil_steps)
        st = sim_npt.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos_npt = st.getPositions(asNumpy=True)
        box_npt = st.getPeriodicBoxVectors()
        del sim_npt
        log.info("  NPT done.")

        log.info(
            f"[2/3] NVT ({nvt_equil_steps} steps, target {nvt_temperature} K, "
            f"thermostat: {nvt_thermostat})..."
        )
        amber_eq2 = load_amber(prmtop, inpcrd)
        sys_nvt = build_vanilla_system(
            amber_eq2, hydrogen_mass, t_min, with_barostat=False, **system_kwargs
        )

        if nvt_restrain_heavy_atoms:
            pos_npt_nm = pos_npt.value_in_unit(nanometer)
            restraint_force, n_restrained = _build_heavy_atom_restraint_force(
                amber_eq2, pos_npt_nm, rest_atoms, nvt_restraint_force_constant
            )
            sys_nvt.addForce(restraint_force)
            log.info(
                f"  Restraining {n_restrained} solute heavy atoms "
                f"(k={nvt_restraint_force_constant} kJ/mol/nm^2)."
            )

        nvt_integrator, nvt_extra_force = _make_nvt_integrator_and_force(200)
        if nvt_extra_force is not None:
            sys_nvt.addForce(nvt_extra_force)

        sim_nvt = app.Simulation(
            topology_ref, sys_nvt, nvt_integrator, platform, platform_props
        )
        sim_nvt.context.setPositions(pos_npt)
        sim_nvt.context.setPeriodicBoxVectors(*box_npt)
        sim_nvt.context.setVelocitiesToTemperature(nvt_temperature * kelvin)
        sim_nvt.step(nvt_equil_steps)
        st = sim_nvt.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos_eq = st.getPositions(asNumpy=True)
        box_eq = st.getPeriodicBoxVectors()
        del sim_nvt
        log.info("[3/3] Equilibration complete.")

        with open(equil_pkl, "wb") as fh:
            pickle.dump((pos_eq, box_eq), fh)
    if restart:
        # ── Resume: everything comes from the storage file ───────────────
        log.info("RESUMING REST2-REMD PRODUCTION (equilibration skipped)")
        sampler, stored = _resume_sampler(
            storage_path, n_iterations, n_replicas, temperatures, timestep,
            n_steps_per_iter, checkpoint_interval, log,
        )
        timestep = stored.get("timestep_fs", timestep)
        n_steps_per_iter = stored.get("n_steps_per_iter", n_steps_per_iter)
        if stored.get("temperatures"):
            t_max = stored["temperatures"][-1]

        previous = {}
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r") as fh:
                    previous = json.load(fh)
            except (OSError, ValueError):
                previous = {}
        metadata.update(stored)
        metadata["t_max"] = t_max
        metadata = {**previous, **metadata}
        metadata["resumed_at_iterations"] = previous.get("resumed_at_iterations", []) + [
            int(sampler.iteration)
        ]
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
    else:
        # ── Build REST system (shared by all replicas) ───────────────────
        log.info("Building REST system...")
        amber_rest = load_amber(prmtop, inpcrd)
        vanilla_sys = build_vanilla_system(
            amber_rest, hydrogen_mass, t_min, with_barostat=False, **system_kwargs
        )
        rest_sys = build_rest_system(vanilla_sys, rest_atoms)

        force_names = [type(f).__name__ for f in rest_sys.getForces()]
        log.info(f"  REST system forces: {force_names}")
        log.info(f"  REST atoms: {len(rest_atoms)}")

        # ── Build openmmtools thermodynamic states ────────────────────────
        log.info("Building thermodynamic states...")

        beta_0 = 1.0 / (kB_val * t_min)

        rest_state_ref = RESTState.from_system(rest_sys)
        thermostate_ref = ThermodynamicState(rest_sys, temperature=t_min * kelvin)
        compound_state_ref = CompoundThermodynamicState(
            thermostate_ref, composable_states=[rest_state_ref]
        )

        compound_states = []
        sampler_states = []

        for i, T in enumerate(temperatures):
            beta_m = 1.0 / (kB_val * T)

            cs = copy.deepcopy(compound_state_ref)
            cs.set_rest_parameters(beta_m, beta_0)
            compound_states.append(cs)

            ss = SamplerState(positions=pos_eq, box_vectors=box_eq)
            sampler_states.append(ss)

            log.info(
                f"  State {i:2d}: T_eff={T:.1f} K  "
                f"lambda_bonds={cs.lambda_rest_bonds:.4f}  "
                f"lambda_elec={cs.lambda_rest_electrostatics:.4f}  "
                f"lambda_ster={cs.lambda_rest_sterics:.4f}"
            )

        log.info("Minimising each thermodynamic state...")
        for i, (cs, ss) in enumerate(zip(compound_states, sampler_states)):
            context = openmm.Context(
                rest_sys, _make_integrator(400 + i), platform, platform_props
            )
            cs.apply_to_context(context)
            context.setPositions(ss.positions)
            context.setPeriodicBoxVectors(*ss.box_vectors)
            openmm.LocalEnergyMinimizer.minimize(context, maxIterations=em_max_iter)
            st = context.getState(getPositions=True, enforcePeriodicBox=True)
            ss.positions = st.getPositions(asNumpy=True)
            ss.box_vectors = st.getPeriodicBoxVectors()
            del context
            log.info(f"  State {i:2d}: minimised.")

        # ── Replica exchange sampler ─────────────────────────────────────
        log.info("Setting up ReplicaExchangeSampler...")

        mcmc_move = mcmc.LangevinDynamicsMove(
            timestep=timestep * femtoseconds,
            collision_rate=friction / picosecond,
            n_steps=n_steps_per_iter,
            reassign_velocities=True,
            n_restart_attempts=6,
        )

        _mixing_scheme_map = {
            "Neighbor swap (Metropolis)": "swap-neighbors",
            "Gibbs sampling": "swap-all",
        }
        _mixing_scheme = _mixing_scheme_map.get(exchange_scheme, "swap-neighbors")
        # online_analysis_interval=None: openmmtools otherwise re-runs MBAR on
        # the whole growing dataset every 200 iterations, which REST2 doesn't use.
        try:
            sampler = ReplicaExchangeSampler(
                mcmc_moves=mcmc_move,
                number_of_iterations=n_iterations,
                replica_mixing_scheme=_mixing_scheme,
                online_analysis_interval=None,
            )
            log.info(f"  Exchange scheme: {exchange_scheme} ({_mixing_scheme})")
        except TypeError:
            log.warning(
                "This openmmtools version doesn't support choosing the exchange "
                "scheme (replica_mixing_scheme) - using its default (neighbor swap)."
            )
            sampler = ReplicaExchangeSampler(
                mcmc_moves=mcmc_move,
                number_of_iterations=n_iterations,
                online_analysis_interval=None,
            )

        reporter = MultiStateReporter(
            storage=storage_path,
            checkpoint_interval=checkpoint_interval,
        )
        sampler.create(
            thermodynamic_states=compound_states,
            sampler_states=sampler_states,
            storage=reporter,
        )

    log.info(f"  Replicas: {sampler.n_states}")
    log.info(f"  Iterations: {n_iterations}")
    log.info(f"  Steps/iteration: {n_steps_per_iter}")
    log.info(f"  Storage: {storage_path}")

    # ── Run in chunks so we can report progress ─────────────────────────
    # A single sampler.run() call blocks with zero output until the entire
    # production is done. run(n_iterations=chunk) runs at most `chunk` more
    # iterations and never past sampler.number_of_iterations, so progress is
    # always read back from sampler.iteration rather than counted here.
    log.info("REST2-REMD PRODUCTION")
    log.info(f"  T range         : {t_min:.0f} - {t_max:.0f} K (effective solute)")
    log.info(f"  REST atoms      : {len(rest_atoms)}  ({solute_desc})")
    log.info(f"  Iterations      : {n_iterations}  x  {n_steps_per_iter} steps")
    total_ns = n_iterations * n_steps_per_iter * timestep / 1e6
    log.info(f"  Total MD/replica: {total_ns:.1f} ns")

    chunk_size = max(1, n_iterations // 50)  # ~50 progress updates over the whole run
    start_iteration = int(sampler.iteration)
    completed = start_iteration
    start_time = time.time()
    if completed >= n_iterations:
        log.info(
            f"  Storage is already at iteration {completed} - nothing to run. "
            f"Increase the total production steps to extend this run."
        )

    while completed < n_iterations:
        this_chunk = min(chunk_size, n_iterations - completed)
        sampler.run(n_iterations=this_chunk)
        if int(sampler.iteration) <= completed:
            raise RuntimeError(
                f"openmmtools made no progress past iteration {completed} "
                f"(stored limit: {sampler.number_of_iterations})."
            )
        completed = int(sampler.iteration)

        elapsed = time.time() - start_time
        pct = 100.0 * completed / n_iterations
        # Rate from this session only - on a resumed run the iterations done
        # before the restart took no time in this process.
        rate = (completed - start_iteration) / elapsed if elapsed > 0 else 0.0
        eta = (n_iterations - completed) / rate if rate > 0 else float("nan")
        nan_note = (
            f", NaN restarts so far: {nan_restarts['count']}"
            if nan_restarts["count"]
            else ""
        )

        # Kept in a fixed, parseable format (also used by the GUI to drive a
        # real progress bar) while still being readable in a plain log file.
        log.info(
            f"  Progress: {completed}/{n_iterations} iterations ({pct:.1f}%) "
            f"- elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min{nan_note}"
        )

    log.info("REST2-REMD COMPLETE")
    log.info(f"Output: {storage_path}")

    return storage_path


def main(argv=None):
    """Command-line entry point.

    The GUI launches this as `python -c "import simulation_run;
    simulation_run.main(...)"` rather than `python simulation_run.py`, because
    the distributed builds ship simulation_run only as a compiled extension
    (.pyd / .so) - there is no .py file to run, and compiled modules cannot be
    executed as scripts. A separate process is still required: openmmtools
    registers signal handlers, which only works in a process's main thread.
    """
    import argparse

    # openmmtools stores the module + class name of every composable state in
    # the .nc file and re-imports it on resume. Runs started by the old
    # launcher (`python simulation_run.py`) recorded RESTState under
    # '__main__'; exposing it there too keeps those runs resumable now that
    # __main__ is the `-c` launcher instead of this module.
    if OPENMMTOOLS_AVAILABLE:
        main_module = sys.modules.get("__main__")
        if main_module is not None and not hasattr(main_module, "RESTState"):
            main_module.RESTState = RESTState

    parser = argparse.ArgumentParser(
        description="Run EM/NPT/NVT equilibration + REST2-REMD production. "
        "Runs in its own process (the GUI launches it as a subprocess) "
        "because openmmtools registers signal handlers, which only works "
        "in a process's main thread - not in a GUI background thread."
    )
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--inpcrd", required=True)
    parser.add_argument(
        "--ligand-resnames", default="", help="comma-separated residue names, e.g. MOL"
    )
    parser.add_argument(
        "--protein-only",
        action="store_true",
        help="apo / ligand-free run: REST solute = protein only "
        "(--ligand-resnames must be empty)",
    )
    parser.add_argument("--n-replicas", type=int, default=16)
    parser.add_argument("--t-min", type=float, default=300.0)
    parser.add_argument("--t-max", type=float, default=400.0)
    parser.add_argument(
        "--temp-distribution",
        default="Exponential spacing",
        choices=["Exponential spacing", "Linear spacing", "Custom list"],
    )
    parser.add_argument(
        "--custom-temperatures",
        default="",
        help="comma-separated list, required if --temp-distribution='Custom list', "
        "e.g. '300,308,317,327,340' - must have exactly --n-replicas values, "
        "sorted ascending, starting at --t-min",
    )
    parser.add_argument("--timestep", type=float, default=2.0, help="fs")
    parser.add_argument("--friction", type=float, default=1.0, help="/ps")
    parser.add_argument(
        "--hydrogen-mass", type=float, default=1.5,
        help="amu; 0 disables hydrogen mass repartitioning",
    )
    parser.add_argument(
        "--nonbonded-method", default="PME",
        choices=list(PERIODIC_NONBONDED_METHODS),
    )
    parser.add_argument("--nonbonded-cutoff", type=float, default=1.0, help="nm")
    parser.add_argument(
        "--constraints", default="HBonds", choices=list(CONSTRAINT_OPTIONS)
    )
    parser.add_argument("--no-rigid-water", action="store_true")
    parser.add_argument("--ewald-error-tolerance", type=float, default=0.0005)
    parser.add_argument("--em-max-iter", type=int, default=5000)
    parser.add_argument("--em-tolerance", type=float, default=10.0, help="kJ/mol/nm")
    parser.add_argument("--npt-steps", type=int, default=500000)
    parser.add_argument("--npt-pressure", type=float, default=1.0, help="atm")
    parser.add_argument(
        "--npt-barostat",
        default="Monte Carlo barostat",
        choices=["Monte Carlo barostat", "Monte Carlo membrane barostat"],
    )
    parser.add_argument("--npt-barostat-interval", type=int, default=25, help="steps")
    parser.add_argument("--nvt-steps", type=int, default=500000)
    parser.add_argument(
        "--nvt-temperature",
        type=float,
        default=None,
        help="defaults to --t-min if not given",
    )
    parser.add_argument(
        "--nvt-thermostat",
        default="Langevin middle integrator",
        choices=["Langevin middle integrator", "Nose-Hoover", "Andersen"],
    )
    parser.add_argument("--nvt-restrain-heavy-atoms", action="store_true")
    parser.add_argument(
        "--nvt-restraint-force-constant",
        type=float,
        default=4184.0,
        help="kJ/mol/nm^2, ~10 kcal/mol/A^2 by default",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=1000, help="replica exchange cycles"
    )
    parser.add_argument(
        "--n-steps-per-iter", type=int, default=1000, help="MD steps per cycle"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=200,
        help="cycles between checkpoints",
    )
    parser.add_argument(
        "--exchange-scheme",
        default="Neighbor swap (Metropolis)",
        choices=["Neighbor swap (Metropolis)", "Gibbs sampling"],
    )
    parser.add_argument("--output-dir", default="rest2_output")
    parser.add_argument(
        "--platform", default="CUDA", choices=["CUDA", "OpenCL", "HIP", "CPU"]
    )
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(argv)

    ligand_resnames = {
        s.strip().upper() for s in args.ligand_resnames.split(",") if s.strip()
    }
    custom_temperatures = [
        float(t) for t in args.custom_temperatures.split(",") if t.strip()
    ] or None

    try:
        run_rest2_remd(
            prmtop=args.prmtop,
            inpcrd=args.inpcrd,
            ligand_resnames=ligand_resnames,
            protein_only=args.protein_only,
            n_replicas=args.n_replicas,
            t_min=args.t_min,
            t_max=args.t_max,
            temperature_distribution=args.temp_distribution,
            custom_temperatures=custom_temperatures,
            timestep=args.timestep,
            friction=args.friction,
            hydrogen_mass=args.hydrogen_mass,
            nonbonded_method=args.nonbonded_method,
            nonbonded_cutoff=args.nonbonded_cutoff,
            constraints=args.constraints,
            rigid_water=not args.no_rigid_water,
            ewald_error_tolerance=args.ewald_error_tolerance,
            em_max_iter=args.em_max_iter,
            em_tolerance=args.em_tolerance,
            npt_equil_steps=args.npt_steps,
            npt_pressure=args.npt_pressure,
            npt_barostat=args.npt_barostat,
            npt_barostat_interval=args.npt_barostat_interval,
            nvt_equil_steps=args.nvt_steps,
            nvt_temperature=args.nvt_temperature,
            nvt_thermostat=args.nvt_thermostat,
            nvt_restrain_heavy_atoms=args.nvt_restrain_heavy_atoms,
            nvt_restraint_force_constant=args.nvt_restraint_force_constant,
            n_iterations=args.n_iterations,
            n_steps_per_iter=args.n_steps_per_iter,
            checkpoint_interval=args.checkpoint_interval,
            exchange_scheme=args.exchange_scheme,
            output_dir=args.output_dir,
            platform_name=args.platform,
            restart=args.restart,
        )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - make sure the GUI's subprocess wrapper sees a clear message
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
