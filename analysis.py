#!/usr/bin/env python3
"""
analysis.py - Analysis backend (publication-subset REST2 analysis)

Refactored from a flat, argparse-driven script into run_rest2_analysis(...),
a callable function, so it can be run from the "Analysis" GUI page instead
of only from the command line. The CLI entry point still works the same way:

    python analysis.py --dir rest2_output --out analysis_output ...

Analyses produced (see the "publication subset" discussion):
  - REMD diagnostics : pairwise acceptance rates, replica temperature walk,
                        per-state energy distributions (overlap check)
  - Structural        : backbone/protein RMSD, per-residue RMSF, Rg time series
  - Ligand binding     : ligand RMSD (site-fit), ligand-receptor COM distance
  - Free energy        : 2D RMSD-Rg landscape (ground state, T_min)

Note on units:
  - `energies` read from the NetCDF is openmmtools' REDUCED potential
    (dimensionless, kT units), not kJ/mol. Since REST2 keeps the real
    temperature at T_min for every thermodynamic state (only the Hamiltonian
    is scaled), the conversion is a constant factor: U[kJ/mol] =
    u_reduced * kB * T_min.

Dependencies: mdtraj, netCDF4, numpy, pandas, matplotlib, scipy
    conda install -c conda-forge mdtraj scipy pandas matplotlib netcdf4
"""

import os
import glob
import json
import argparse
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

try:
    import mdtraj as md

    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False

try:
    import netCDF4 as nc4

    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

try:
    import parmed as pmd
    import openmm
    from openmm.app import CutoffPeriodic
    from openmm.unit import nanometer, kilojoules_per_mole

    HAS_OPENMM_PARMED = True
except ImportError:
    HAS_OPENMM_PARMED = False


# Constants that don't depend on any particular run

KB_KJ_PER_MOL_K = 0.0083144621  # kJ/mol/K, used to convert reduced potentials -> kJ/mol

CA_SEL = "name CA"
BB_SEL = "backbone"
PROT_SEL = "protein"

BG = "white"
PANEL = "white"
GRID = "#b0b0b0"
TEXT = "#1a1a1a"
ACCENT = "#1f77b4"  # matplotlib's standard blue - readable and familiar in print
ACCENT3 = "#2ca02c"  # standard green, used for the "ideal mixing target" band
CMAP_MAIN = "viridis"  # colorblind-friendly, standard in current MD literature
CMAP_DIV = "RdBu"  # diverging, reads cleanly in both color and grayscale print

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "axes.titlepad": 12,
        "axes.labelsize": 12,
        "axes.labelpad": 7,
        "axes.grid": True,
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": "white",
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        "legend.fontsize": 10,
        "legend.framealpha": 0.95,
        "legend.frameon": True,
        "legend.title_fontsize": 11,
        "text.color": TEXT,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)



def grid_shape(n):
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def state_color(i, n):
    return plt.get_cmap(CMAP_MAIN)(i / max(n - 1, 1))


def moving_average(y, window):
    """Centered rolling mean, edge-padded so the output keeps the same length
    as the input (min_periods=1 avoids NaNs at the ends)."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < 2:
        return y
    return (
        pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().values
    )


def lighten_color(color, amount=0.55):
    """Blend a color toward white by `amount` (0 = unchanged, 1 = white).
    Used to derive a pale 'raw data' shade from a bold 'moving average'
    color of the same hue."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)



RMSD_SERIES_COLORS = {
    "backbone": "#000000",  # black
    "lig_fit_prot": "#d62728",  # red
    "lig_pose": "#1f77b4",  # blue
}
RG_COLOR = "#000000"

# RMSF plot color - solid per-residue data, no smoothing overlay.
RMSF_MAIN_COLOR = "#7B3294"  # purple


def styled_colorbar(mappable, ax, label=""):
    cb = plt.colorbar(mappable, ax=ax, pad=0.02)
    cb.set_label(label, color=TEXT)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    cb.outline.set_edgecolor(GRID)
    return cb


def robust_outlier_mask(values, z_cut):
    """Boolean mask, True for points within z_cut robust z-scores
    (|v - median| / (1.4826 * MAD)) of the bulk. A zero MAD (all-identical
    data) flags nothing."""
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        return np.isfinite(v)
    return np.abs(v - med) / mad <= z_cut


def _get_nc_var(ds, name):
    """Robustly fetch a variable from a NetCDF dataset or any nested group."""
    if name in ds.variables:
        return ds.variables[name]
    for grp in ds.groups.values():
        if name in grp.variables:
            return grp.variables[name]
    return None



# Main entry point


def run_rest2_analysis(
    input_dir="rest2_output",
    output_dir="analysis_output",
    topology="complex.prmtop",
    ref_pdb="complex.pdb",
    nc_file=None,
    nc_checkpoint=None,
    n_replicas=7,
    t_min=300.0,
    t_max=328.0,
    ligand_sel="resname MOL",
    receptor_sel="protein",
    timestep=2.0,
    steps_per_iter=1000,
    checkpoint_interval=200,
    stride=1,
    skip_structural=False,
    image_molecules=False,
    skip_energy_decomposition=False,
    energy_decomposition_stride=10,
    n_equil_frames=1,
    protein_only=None,
    fes_outlier_z=5.0,
    log_callback=None,
):
    """
    Runs the publication-subset REST2 analysis. Returns a dict with
    'output_dir', 'plots' (list of PNG paths written), 'csvs' (list of
    CSV paths written) and 'protein_only' (the mode actually used).

    protein_only: True analyses an apo / ligand-free system - ligand_sel is
    ignored, every ligand metric/plot/CSV column is skipped, and the REST
    solute is labelled as "Protein". False forces protein-ligand mode. None
    (default) takes the value recorded by simulation_run.py in
    <input_dir>/run_metadata.json, else falls back to protein-only when
    ligand_sel matches no atoms.

    fes_outlier_z: before binning the RMSD-Rg free-energy surface, frames
    further than this many robust z-scores (median/MAD) from the bulk in
    either coordinate are dropped - isolated single frames otherwise stretch
    the axes and show up as a fake minimum. Removal only happens if the
    flagged frames are at most 1% of the data; a larger group is treated as
    a genuine sub-population and kept. 0 or None disables the filter. The
    CSV exports always keep every frame.

    If <input_dir>/run_metadata.json exists (written by simulation_run.py),
    its replica count, exact temperature ladder (including linear/custom
    ladders) and timing values take precedence over n_replicas / t_min /
    t_max / timestep / steps_per_iter / checkpoint_interval, with a warning
    for every value that disagrees.

    energy_decomposition_stride: extra thinning applied ON TOP OF `stride`,
    just for the E_solute/E_water-interaction decomposition - this step
    recomputes energies from scratch per frame via OpenMM and is by far the
    slowest part of this script, so it defaults to using far fewer frames
    than the structural analyses. Set skip_energy_decomposition=True to
    disable it entirely.

    n_equil_frames: number of leading (post-stride) checkpoint frames to
    discard from EVERY reconstructed state-continuous trajectory before any
    structural or energy analysis runs. Frame 0 of the checkpoint is
    frequently the pre-production equilibrated/minimized configuration
    (written before the first exchange/production step), not a real draw
    from any thermodynamic state's equilibrium ensemble - it typically shows
    up as an anomalously low-energy outlier (esp. in the protein
    intramolecular term) relative to the bulk of the trajectory. Default of
    1 drops just that frame; set to 0 to disable, or higher if more burn-in
    should be discarded.
    """
    if not HAS_MDTRAJ:
        raise ImportError(
            "mdtraj is not installed. Install with: conda install -c conda-forge mdtraj"
        )
    if not HAS_NETCDF:
        raise ImportError(
            "netCDF4 is not installed. Install with: conda install -c conda-forge netcdf4"
        )

    def log(msg):
        print(msg)
        if log_callback is not None:
            log_callback(msg)

    os.makedirs(output_dir, exist_ok=True)

    # ── Run metadata recorded by simulation_run.py ──────────────────────
    run_meta = None
    meta_path = os.path.join(input_dir, "run_metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as fh:
                run_meta = json.load(fh)
            log(f"  Run metadata    : {meta_path}")
        except (OSError, ValueError) as e:
            log(f"WARNING: could not read {meta_path} ({e}) - using given settings.")

    meta_temperatures = None
    if run_meta:
        overrides = [
            ("n_replicas", "n_replicas", n_replicas, int),
            ("t_min", "t_min", t_min, float),
            ("t_max", "t_max", t_max, float),
            ("timestep", "timestep_fs", timestep, float),
            ("steps_per_iter", "n_steps_per_iter", steps_per_iter, int),
            ("checkpoint_interval", "checkpoint_interval", checkpoint_interval, int),
        ]
        resolved = {}
        for name, key, given, cast in overrides:
            if run_meta.get(key) is None:
                resolved[name] = given
                continue
            value = cast(run_meta[key])
            if abs(float(value) - float(given)) > 1e-6:
                log(
                    f"WARNING: {name}={given} differs from the production run "
                    f"({value}) - using the production value."
                )
            resolved[name] = value
        n_replicas = resolved["n_replicas"]
        t_min = resolved["t_min"]
        t_max = resolved["t_max"]
        timestep = resolved["timestep"]
        steps_per_iter = resolved["steps_per_iter"]
        checkpoint_interval = resolved["checkpoint_interval"]

        temps = run_meta.get("temperatures")
        if temps and len(temps) == n_replicas:
            meta_temperatures = [float(t) for t in temps]

        if protein_only is None and run_meta.get("protein_only") is not None:
            protein_only = bool(run_meta["protein_only"])

    if nc_file is None:
        nc_candidates = glob.glob(os.path.join(input_dir, "*.nc"))
        nc_candidates = [
            f for f in nc_candidates if "checkpoint" not in os.path.basename(f).lower()
        ]
        nc_file = nc_candidates[0] if nc_candidates else None

    if nc_checkpoint is None:
        chk_candidates = glob.glob(os.path.join(input_dir, "*checkpoint*.nc"))
        nc_checkpoint = chk_candidates[0] if chk_candidates else None

    if meta_temperatures is not None:
        # Exact ladder used in production (exponential, linear or custom).
        temperatures = meta_temperatures
    elif n_replicas > 1:
        temperatures = [
            t_min
            + (t_max - t_min)
            * (math.exp(float(i) / float(n_replicas - 1)) - 1.0)
            / (math.e - 1.0)
            for i in range(n_replicas)
        ]
    else:
        temperatures = [t_min]

    time_per_frame = (
        checkpoint_interval * steps_per_iter * timestep / 1e6
    )  # ns per checkpoint frame
    time_per_iter = steps_per_iter * timestep / 1e6  # ns per iteration
    kT_min = KB_KJ_PER_MOL_K * t_min  # kJ/mol, constant across states in REST2

    plots_written = []
    csvs_written = []

    def savefig(name):
        for ax in plt.gcf().get_axes():
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(GRID)
                ax.spines[side].set_linewidth(0.9)

        path = os.path.join(output_dir, name)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG, edgecolor=BG)
        plt.close()
        log(f"  -> {path}")
        plots_written.append(path)

    log(f"\n{'='*65}\nREST2 PUBLICATION-SUBSET ANALYSIS\n{'='*65}")
    log(f"  Target File     : {nc_file}")
    log(f"  Checkpoint File : {nc_checkpoint}")
    log(f"  Time Per Frame  : {time_per_frame:.4f} ns  (checkpoint)")
    log(f"  Time Per Iter   : {time_per_iter:.6f} ns  (energy)")

    # ── Shared mutable state across the nested steps below ─────────────
    S = {
        "replica_states": None,
        "state_replicas": None,
        "energies": None,
        "n_attempts": None,
        "n_accepts": None,
        "state_energy_dfs": [],
        "state_trajs": {},
        "ref_struct": None,
        "equil_frame": None,
        "all_rmsd": {},
        "all_rg": {},
        "all_lig_rmsd": {},
        "all_lig_fit_prot_rmsd": {},
        "all_lig_dist": {},
    }

    # ── 1. NetCDF parsing (raw netCDF4, no openmmtools dependency) ──────
    def load_netcdf_analysis():
        if not nc_file:
            log("ERROR: MultiState NetCDF target could not be resolved.")
            return
        try:
            with nc4.Dataset(nc_file, "r") as ds:
                states_var = _get_nc_var(ds, "states")
                if states_var is not None:
                    replica_states = np.array(states_var[:])
                    n_iter, n_rep = replica_states.shape
                    log(
                        f"  Parsed steps : {n_iter} exchange iterations over {n_rep} states."
                    )

                    replica_states_int = replica_states.astype(int)
                    if replica_states_int.size:
                        log(
                            f"  states values : min={replica_states_int.min()}, "
                            f"max={replica_states_int.max()}"
                        )
                    state_replicas = np.zeros_like(replica_states_int)
                    iter_idx = np.arange(n_iter)[:, None]
                    replica_idx = np.arange(n_rep)[None, :]
                    if replica_states_int.size and (
                        replica_states_int.min() < 0
                        or replica_states_int.max() >= n_rep
                    ):
                        log(
                            f"  WARNING: 'states' values outside [0, {n_rep - 1}] - "
                            f"clipping out-of-range entries (data may be inconsistent!)."
                        )
                        replica_states_int = np.clip(replica_states_int, 0, n_rep - 1)
                    state_replicas[iter_idx, replica_states_int] = replica_idx

                    S["replica_states"] = replica_states
                    S["state_replicas"] = state_replicas
                else:
                    log("WARNING: 'states' variable not found in NetCDF.")

                energies_var = _get_nc_var(ds, "energies")
                if energies_var is not None:
                    S["energies"] = np.array(energies_var[:])
                    log(f"  Energies shape (reduced, kT units): {S['energies'].shape}")
                else:
                    log("WARNING: 'energies' variable not found in NetCDF.")

                accepted_var = _get_nc_var(ds, "accepted")
                proposed_var = _get_nc_var(ds, "proposed")
                if accepted_var is not None and proposed_var is not None:
                    accepted_arr = np.array(accepted_var[:])
                    proposed_arr = np.array(proposed_var[:])

                    if accepted_arr.ndim == 3:
                        accepted_arr = accepted_arr.sum(axis=0)
                        proposed_arr = proposed_arr.sum(axis=0)

                    if accepted_arr.ndim == 2 and proposed_arr.ndim == 2:
                        S["n_accepts"] = np.array(
                            [
                                accepted_arr[i, i + 1]
                                for i in range(
                                    min(n_replicas - 1, accepted_arr.shape[0] - 1)
                                )
                            ]
                        )
                        S["n_attempts"] = np.array(
                            [
                                proposed_arr[i, i + 1]
                                for i in range(
                                    min(n_replicas - 1, proposed_arr.shape[0] - 1)
                                )
                            ]
                        )
                        log(
                            f"  Acceptance matrix  : parsed {len(S['n_accepts'])} neighbor pairs"
                        )
                    else:
                        log(
                            f"WARNING: unexpected acceptance array dims: {accepted_arr.ndim}, {proposed_arr.ndim}"
                        )
                else:
                    log("WARNING: 'accepted'/'proposed' variables not found.")
        except Exception as e:
            log(f"WARNING: NetCDF structure mapping failed: {e}")

    def extract_state_energies():
        if S["energies"] is None or S["state_replicas"] is None:
            log("  Missing energies/state mapping - skipping energy extraction.")
            return

        energies = S["energies"]
        state_replicas = S["state_replicas"]
        replica_states = S["replica_states"]
        n_iter, _ = replica_states.shape

        for s in range(
            min(n_replicas, energies.shape[2] if energies.ndim >= 3 else n_replicas)
        ):
            E_reduced = []
            for t in range(n_iter):
                r = int(state_replicas[t, s])
                r = max(0, min(r, energies.shape[1] - 1))
                E_reduced.append(energies[t, r, s])

            E_kJmol = np.array(E_reduced) * kT_min

            df = pd.DataFrame(
                {
                    "iteration": np.arange(n_iter),
                    "time_ns": np.arange(n_iter) * time_per_iter,
                    "potential_kJ_mol": E_kJmol,
                    "state": s,
                    "temperature_K": temperatures[s],
                }
            )
            S["state_energy_dfs"].append(df)

    load_netcdf_analysis()
    extract_state_energies()

    # ── 2. State-continuous trajectory reconstruction ───────────────────
    def load_state_continuous_trajectories():
        if not nc_checkpoint or not os.path.exists(topology):
            log("Skipping structural analysis - trajectory or topology file missing.")
            return

        log("Reassembling trajectories for thermodynamic (physical) states...")

        state_replicas = S["state_replicas"]
        if state_replicas is None:
            log(
                "Skipping structural analysis - no replica/state mapping available "
                "(the 'states' variable wasn't found when parsing the main NetCDF file)."
            )
            return

        try:
            with nc4.Dataset(nc_checkpoint, "r") as ds:
                positions = ds.variables["positions"]
                n_chk_frames, n_rep, n_atoms, _ = positions.shape

                box_vecs = None
                if "box_vectors" in ds.variables:
                    box_vecs = ds.variables["box_vectors"][:]
                    log(f"  Box vectors found : shape {box_vecs.shape}")

                top = (
                    md.load_prmtop(topology)
                    if topology.endswith(".prmtop")
                    else md.load_topology(topology)
                )
                if top.n_atoms != n_atoms:
                    log(
                        f"  WARNING: topology has {top.n_atoms} atoms but checkpoint "
                        f"positions have {n_atoms} atoms - structural metrics may be "
                        f"inconsistent."
                    )
                frame_indices = list(range(0, n_chk_frames, stride))

                # Checkpoint frame k belongs to iteration k * checkpoint_interval.
                # If that exceeds the stored iterations every frame clamps to the
                # last one, freezing the replica->state map: each "state" then
                # degenerates into a replica-continuous trajectory that wanders
                # the whole ladder, making all states look identical.
                n_iterations_stored = state_replicas.shape[0]
                last_needed = frame_indices[-1] * checkpoint_interval if frame_indices else 0
                if last_needed > n_iterations_stored - 1:
                    log(
                        f"  WARNING: checkpoint interval {checkpoint_interval} x "
                        f"{n_chk_frames} stored frames needs iteration {last_needed}, "
                        f"but only {n_iterations_stored} iterations are stored. The "
                        f"checkpoint interval is probably wrong - with a too-large "
                        f"value every thermodynamic state collapses onto a single "
                        f"replica and all states look identical. Set it to "
                        f"{max(1, (n_iterations_stored - 1) // max(1, n_chk_frames - 1))} "
                        f"(or let run_metadata.json supply it)."
                    )

                replica_sources = {}
                for s in range(min(n_replicas, n_rep)):
                    state_coords = []
                    state_boxes = [] if box_vecs is not None else None
                    used_replicas = set()

                    for k in frame_indices:
                        t = min(k * checkpoint_interval, n_iterations_stored - 1)
                        r = int(state_replicas[t, s])
                        r = max(0, min(r, n_rep - 1))
                        used_replicas.add(r)
                        state_coords.append(positions[k, r, :, :])
                        if box_vecs is not None:
                            state_boxes.append(box_vecs[k, r, :, :])
                    replica_sources[s] = used_replicas

                    if len(state_coords) == 0:
                        log(f"  -> State {s:02d}: NO FRAMES (skipped)")
                        continue

                    time_arr = np.arange(len(state_coords)) * stride * time_per_frame
                    traj = md.Trajectory(
                        xyz=np.array(state_coords), topology=top, time=time_arr
                    )

                    if state_boxes is not None:
                        traj.unitcell_vectors = np.array(state_boxes)

                    if image_molecules and traj.unitcell_vectors is not None:
                        try:
                            traj.image_molecules(inplace=True)
                            log(f"  -> State {s:02d}: PBC imaging applied.")
                        except Exception as img_err:
                            log(f"  -> State {s:02d}: PBC imaging failed ({img_err})")

                    S["state_trajs"][s] = traj
                    log(
                        f"  -> State {s:02d} ({temperatures[s]:.1f} K): {len(traj)} frames, "
                        f"{time_arr[-1]:.2f} ns, drawn from {len(replica_sources[s])} "
                        f"replica(s)"
                    )

                # Demultiplexing sanity check: with working replica exchange
                # each thermodynamic state is visited by many replicas. If a
                # state draws all of its frames from a single replica, the
                # state-continuous reconstruction has degenerated into a
                # replica-continuous one and every state will look alike.
                stuck = [s for s, reps in replica_sources.items() if len(reps) == 1]
                if stuck and len(frame_indices) > 1:
                    log(
                        f"  WARNING: state(s) {stuck} took every frame from one replica. "
                        f"Either the replicas never exchanged, or the checkpoint interval "
                        f"is wrong - in both cases the per-state structural metrics are "
                        f"not state-resolved and all states will look identical."
                    )
        except Exception as e:
            log(f"CRITICAL ERROR tracking state coords: {e}")

    if not skip_structural:
        load_state_continuous_trajectories()

    # ── Drop burn-in frames from every reconstructed trajectory ─────────
    # Must happen AFTER reconstruction (needs full trajs to slice) and
    # BEFORE ref_struct fallback / structural / energy-decomposition blocks
    # (all of which consume S["state_trajs"]), so the exclusion is applied
    # uniformly everywhere downstream.
    if n_equil_frames > 0 and S["state_trajs"]:
        log(
            f"\nDiscarding first {n_equil_frames} frame(s) (burn-in / "
            f"pre-production structure) from each reconstructed trajectory..."
        )
        for s in list(S["state_trajs"].keys()):
            traj = S["state_trajs"][s]
            if len(traj) > n_equil_frames:
                if s == 0:
                    # Kept as the fallback RMSD reference: it's the
                    # equilibrated structure, and it is NOT in the analysed
                    # frames (a reference taken from inside the trajectory
                    # scores RMSD = 0 against itself - a fake FES point).
                    S["equil_frame"] = traj[n_equil_frames - 1]
                S["state_trajs"][s] = traj[n_equil_frames:]
            else:
                log(
                    f"  WARNING: state {s} trajectory has only {len(traj)} frame(s) - "
                    f"cannot discard {n_equil_frames} burn-in frame(s), leaving as-is."
                )

    if os.path.exists(ref_pdb):
        S["ref_struct"] = md.load(ref_pdb)
        log(f"  Reference structure: {ref_pdb}")
    elif S["equil_frame"] is not None:
        S["ref_struct"] = S["equil_frame"]
        log(
            f"  Reference structure: '{ref_pdb}' not found - using the last "
            f"burn-in frame of state 0 (excluded from the analysed frames)"
        )
    elif S["state_trajs"]:
        S["ref_struct"] = S["state_trajs"][0][0]
        log(
            f"  Reference structure: '{ref_pdb}' not found - using the first "
            f"analysed frame of state 0 (that frame will have RMSD = 0)"
        )

    # ── Resolve protein-only vs protein-ligand mode ──────────────────────
    # Done once, up front, so the structural and energy-decomposition blocks
    # agree on the ligand atoms (and so an invalid ligand selection can't
    # abort the whole structural block in a protein-only run).
    ligand_indices_global = np.array([], dtype=int)
    if S["state_trajs"] and 0 in S["state_trajs"] and protein_only is not True:
        try:
            ligand_indices_global = S["state_trajs"][0].topology.select(ligand_sel)
        except Exception as sel_err:
            log(f"WARNING: invalid ligand selection '{ligand_sel}' ({sel_err}).")
        if len(ligand_indices_global) == 0:
            if protein_only is None:
                log(
                    f"  Ligand selection '{ligand_sel}' matched no atoms - "
                    f"analysing as a protein-only system."
                )
                protein_only = True
            else:
                log(
                    f"WARNING: ligand selection '{ligand_sel}' matched no atoms "
                    f"- ligand metrics will be empty. Check the residue name."
                )
    if protein_only is None:
        protein_only = False
    solute_label = "Protein" if protein_only else "Solute (Protein+Ligand)"
    LIGAND_COLUMNS = [
        "ligand_site_fit_protein_rmsd_nm",
        "ligand_rmsd_nm",
        "ligand_rec_com_dist_nm",
        "E_protein_kcal_mol",
        "E_ligand_kcal_mol",
        "E_protein_ligand_interaction_kcal_mol",
    ]

    def drop_ligand_columns(df):
        """Protein-only CSVs shouldn't carry all-NaN ligand columns (and in
        protein-only mode E_protein is just E_solute, so it's dropped too)."""
        if not protein_only:
            return df
        return df.drop(columns=LIGAND_COLUMNS, errors="ignore")

    log(f"  System type     : {'protein only' if protein_only else 'protein-ligand complex'}")

    # ── 3. REMD diagnostic plots ─────────────────────────────────────────
    def plot_acceptance():
        if S["n_attempts"] is None or len(S["n_attempts"]) == 0:
            log("WARNING: no acceptance data to plot.")
            return
        n_attempts, n_accepts = S["n_attempts"], S["n_accepts"]
        n = len(n_attempts)
        rates = np.where(n_attempts > 0, 100.0 * n_accepts / n_attempts, 0.0)
        labels = [
            f"{temperatures[j]:.0f}<->{temperatures[j+1]:.0f} K" for j in range(n)
        ]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(
            range(n),
            rates,
            color=[state_color(j, n) for j in range(n)],
            edgecolor=GRID,
            lw=0.5,
            width=0.6,
        )
        for bar, r, at, ac in zip(bars, rates, n_attempts, n_accepts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{r:.1f}%\n({ac}/{at})",
                ha="center",
                va="bottom",
            )
        ax.axhspan(
            20, 40, color=ACCENT3, alpha=0.15, label="Ideal mixing target (20-40%)"
        )
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Acceptance percentage (%)")
        ax.set_title("Pairwise Replica Exchange Rates")
        ax.legend()
        plt.tight_layout()
        savefig("01_acceptance_rates.png")

    def plot_replica_walk():
        if S["replica_states"] is None:
            log("WARNING: no replica-state walk data to plot.")
            return
        replica_states = S["replica_states"]
        n_iter, n_rep = replica_states.shape
        nrows, ncols = grid_shape(n_rep)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharey=True
        )
        axes_flat = np.array(axes).flatten()

        for r in range(n_rep):
            ax = axes_flat[r]
            n_T = len(temperatures)
            T_walk = []
            for s in replica_states[:, r]:
                si = int(s)
                T_walk.append(temperatures[si] if 0 <= si < n_T else np.nan)
            ax.plot(T_walk, lw=0.3, color=state_color(r, n_rep), alpha=0.8)
            ax.set_title(f"Replica {r} mixing walk")
            ax.set_ylim(t_min - 5, t_max + 5)
        for ax in axes_flat[n_rep:]:
            ax.set_visible(False)
        fig.suptitle(
            "Replica Path History Across Effective Temperatures", y=1.0, fontsize=14
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        savefig("02_replica_walk.png")

    def plot_energy_distributions():
        if not S["state_energy_dfs"]:
            log("WARNING: no energy distributions to plot.")
            return
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, df in enumerate(S["state_energy_dfs"]):
            E = df["potential_kJ_mol"].dropna().values
            if len(E) < 3:
                continue
            try:
                kde = gaussian_kde(E, bw_method=0.25)
            except (np.linalg.LinAlgError, ValueError):
                log(
                    f"  WARNING: state {i} energy series is degenerate (constant) - "
                    f"skipping KDE for this state."
                )
                continue
            x = np.linspace(E.min() - 1.5 * E.std(), E.max() + 1.5 * E.std(), 300)
            c = state_color(i, len(S["state_energy_dfs"]))
            ax.fill_between(x, kde(x), alpha=0.12, color=c)
            ax.plot(
                x, kde(x), color=c, lw=1.8, label=f"State {i} ({temperatures[i]:.1f} K)"
            )
        ax.set_xlabel("Potential Energy (kJ/mol)")
        ax.set_title("State-Continuous Energy Distributions (neighbor overlap check)")
        ax.legend(ncol=2, frameon=True)
        plt.tight_layout()
        savefig("03_energy_distributions.png")

    plot_acceptance()
    plot_replica_walk()
    plot_energy_distributions()

    # ── 4. Structural + ligand-binding metrics ──────────────────────────
    def run_structural_block():
        state_trajs = S["state_trajs"]
        ref_struct = S["ref_struct"]
        if not state_trajs:
            return
        first_traj = state_trajs.get(0)
        if first_traj is None:
            log(
                "WARNING: state 0 trajectory missing - cannot perform structural analysis."
            )
            return

        # Guard against a topology/atom-count mismatch between the reference
        # structure (ref_pdb, e.g. complex.pdb) and the trajectories (which are
        # built from the prmtop topology). If they differ - e.g. the PDB lacks
        # ions/waters that tleap added to the prmtop - selections computed from
        # the trajectory topology would index out of bounds on ref_struct
        # ("index N is out of bounds for axis 1 with size M").
        if ref_struct is not None and ref_struct.n_atoms != first_traj.n_atoms:
            equil_frame = S["equil_frame"]
            log(
                f"WARNING: reference '{ref_pdb}' has {ref_struct.n_atoms} atoms "
                f"but trajectories have {first_traj.n_atoms}. Using the "
                + (
                    "last burn-in frame of state 0 as the reference instead."
                    if equil_frame is not None
                    else "first analysed frame as the reference instead "
                    "(it will have RMSD = 0)."
                )
            )
            ref_struct = equil_frame if equil_frame is not None else first_traj[0]
            S["ref_struct"] = ref_struct

        ca_indices = first_traj.topology.select(CA_SEL)
        bb_indices = first_traj.topology.select(BB_SEL)
        prot_indices = first_traj.topology.select(PROT_SEL)
        lig_indices = ligand_indices_global
        rec_indices = first_traj.topology.select(receptor_sel)

        log(
            f"  Selection counts: CA={len(ca_indices)}, BB={len(bb_indices)}, "
            f"PROT={len(prot_indices)}, LIG={len(lig_indices)}, REC={len(rec_indices)}"
        )

        ref_bb = (
            ref_struct.atom_slice(bb_indices)
            if bb_indices is not None and len(bb_indices) > 0
            else None
        )
        ref_prot = (
            ref_struct.atom_slice(prot_indices)
            if prot_indices is not None and len(prot_indices) > 0
            else None
        )
        ref_lig = (
            ref_struct.atom_slice(lig_indices)
            if lig_indices is not None and len(lig_indices) > 0
            else None
        )
        ref_rec = (
            ref_struct.atom_slice(rec_indices)
            if rec_indices is not None and len(rec_indices) > 0
            else None
        )
        ref_ca = (
            ref_struct.atom_slice(ca_indices)
            if ca_indices is not None and len(ca_indices) > 0
            else None
        )

        all_rmsd, all_rg = S["all_rmsd"], S["all_rg"]
        all_lig_rmsd, all_lig_fit_prot_rmsd = (
            S["all_lig_rmsd"],
            S["all_lig_fit_prot_rmsd"],
        )
        all_lig_dist = S["all_lig_dist"]

        for s, traj in sorted(state_trajs.items()):
            if len(traj) == 0:
                continue

            try:
                traj_bb = (
                    traj.atom_slice(bb_indices)
                    if bb_indices is not None and len(bb_indices) > 0
                    else traj
                )
                traj_prot = (
                    traj.atom_slice(prot_indices)
                    if prot_indices is not None and len(prot_indices) > 0
                    else traj
                )
                traj_lig = (
                    traj.atom_slice(lig_indices)
                    if lig_indices is not None and len(lig_indices) > 0
                    else None
                )

                if ref_bb is not None and len(traj_bb) > 0:
                    all_rmsd[s] = md.rmsd(traj_bb.superpose(ref_bb), ref_bb)  # nm

                if len(traj_prot) > 0:
                    all_rg[s] = md.compute_rg(traj_prot)  # nm

                if (
                    ref_rec is not None
                    and traj_lig is not None
                    and ref_lig is not None
                    and len(traj_lig) > 0
                ):
                    # Superpose a COPY, not the shared state_trajs[s] object -
                    # traj.superpose() mutates its target in place, and
                    # state_trajs[s] is reused elsewhere.
                    traj_aligned = traj.slice(slice(None), copy=True).superpose(
                        ref_struct, atom_indices=rec_indices
                    )
                    all_lig_rmsd[s] = md.rmsd(
                        traj_aligned.atom_slice(lig_indices), ref_lig
                    )  # nm
                    all_lig_fit_prot_rmsd[s] = md.rmsd(
                        traj_aligned.atom_slice(prot_indices), ref_prot
                    )  # nm

                    lig_com = md.compute_center_of_mass(
                        traj_aligned.atom_slice(lig_indices)
                    )
                    rec_com = md.compute_center_of_mass(
                        traj_aligned.atom_slice(rec_indices)
                    )
                    delta = lig_com - rec_com
                    # Minimum-image correction: without this, if the ligand or
                    # receptor COM sits near a periodic box edge, the raw
                    # coordinate difference can spike to a spuriously large
                    # value instead of the true (shortest, wrapped) distance.
                    # This is an orthorhombic approximation (per-axis wrapping
                    # using box lengths) - exact for cubic/rectangular boxes,
                    # an approximation for non-orthorhombic boxes (e.g. the
                    # truncated octahedron used in system_generation.py), but
                    # still substantially more correct than no PBC treatment.
                    if traj_aligned.unitcell_lengths is not None:
                        box_lengths = traj_aligned.unitcell_lengths  # (n_frames, 3), nm
                        delta = delta - box_lengths * np.round(delta / box_lengths)
                    else:
                        log(
                            "WARNING: no box vectors available for ligand-receptor COM "
                            "distance - skipping periodic-image correction for this state."
                        )
                    all_lig_dist[s] = np.sqrt((delta**2).sum(axis=1))  # nm
            except Exception as state_err:
                log(
                    f"  WARNING: structural metrics failed for state {s} "
                    f"({temperatures[s]:.0f} K): {state_err}"
                )

        # ── Export full per-frame, per-state structural metrics to CSV ──
        all_states = sorted(state_trajs.keys())
        combined_rows = []
        for s in all_states:
            traj = state_trajs[s]
            n_frames = len(traj)
            if n_frames == 0:
                continue
            t_ns = traj.time

            df = pd.DataFrame(
                {
                    "state": s,
                    "temperature_K": (
                        temperatures[s] if s < len(temperatures) else np.nan
                    ),
                    "frame": np.arange(n_frames),
                    "time_ns": t_ns,
                }
            )
            df["backbone_rmsd_nm"] = all_rmsd.get(s, np.full(n_frames, np.nan))
            df["ligand_site_fit_protein_rmsd_nm"] = all_lig_fit_prot_rmsd.get(
                s, np.full(n_frames, np.nan)
            )
            df["ligand_rmsd_nm"] = all_lig_rmsd.get(s, np.full(n_frames, np.nan))
            df["rg_nm"] = all_rg.get(s, np.full(n_frames, np.nan))
            df["ligand_rec_com_dist_nm"] = all_lig_dist.get(
                s, np.full(n_frames, np.nan)
            )

            df = drop_ligand_columns(df)
            out_name = f"rmsd_state_{s:02d}.csv"
            df.to_csv(os.path.join(output_dir, out_name), index=False)
            log(f"  -> {os.path.join(output_dir, out_name)}")
            csvs_written.append(os.path.join(output_dir, out_name))
            combined_rows.append(df)

        if combined_rows:
            combined = pd.concat(combined_rows, ignore_index=True)
            combined_path = os.path.join(output_dir, "all_replica_rmsd.csv")
            combined.to_csv(combined_path, index=False)
            log(f"  -> {combined_path}  (all states, long format)")
            csvs_written.append(combined_path)

        # ── Plots ────────────────────────────────────────────────────
        def _plot_raw(ax, t_ns, y, color, label):
            """Plots only the raw per-frame series, no smoothing."""
            ax.plot(t_ns, y, color=color, lw=0.8, alpha=0.9, label=label, zorder=2)

        def plot_rmsd():
            metrics = [
                (all_rmsd, "Backbone RMSD", RMSD_SERIES_COLORS["backbone"]),
                (
                    all_lig_fit_prot_rmsd,
                    "Ligand-Site Aligned Protein RMSD",
                    RMSD_SERIES_COLORS["lig_fit_prot"],
                ),
                (all_lig_rmsd, "Ligand Binding-Pose RMSD", RMSD_SERIES_COLORS["lig_pose"]),
            ]
            metrics = [
                (d, label, c)
                for d, label, c in metrics
                if d and 0 in d and d[0] is not None and len(d[0]) > 0
            ]
            if not metrics:
                log("WARNING: no RMSD data to plot.")
                return

            fig, ax = plt.subplots(figsize=(12, 6))
            s = 0
            t_ns = state_trajs[s].time
            for data_dict, label, color in metrics:
                _plot_raw(ax, t_ns, data_dict[s], color, label)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("RMSD (nm)")
            ax.set_title(f"RMSD vs. Time ({temperatures[0]:.0f} K)")
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            plt.tight_layout()
            savefig("04_rmsd_time.png")

        def plot_rg():
            if 0 not in all_rg or all_rg[0] is None or len(all_rg[0]) == 0:
                log("WARNING: no Rg data to plot.")
                return
            fig, ax = plt.subplots(figsize=(12, 6))
            s = 0
            t_ns = state_trajs[s].time
            _plot_raw(
                ax,
                t_ns,
                all_rg[s],
                RG_COLOR,
                f"Radius of Gyration ({temperatures[0]:.0f} K)",
            )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Rg (nm)")
            ax.set_title("Protein Radius of Gyration")
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            plt.tight_layout()
            savefig("05_rg_time.png")

        def plot_ligand_distance():
            if (
                0 not in all_lig_dist
                or all_lig_dist[0] is None
                or len(all_lig_dist[0]) == 0
            ):
                log("WARNING: no ligand-receptor distance data to plot.")
                return
            fig, ax = plt.subplots(figsize=(11, 5))
            s = 0
            t_ns = state_trajs[s].time
            ax.plot(
                t_ns,
                all_lig_dist[s],
                lw=1.5,
                color=state_color(s, n_replicas),
                label=f"State 0 ({temperatures[0]:.0f} K)",
            )
            ax.set_xlabel("Physical Time (ns)")
            ax.set_ylabel("Ligand-Receptor COM Distance (nm)")
            ax.set_title("Ligand-Receptor Center-of-Mass Distance")
            ax.legend(loc="upper right", frameon=True)
            plt.tight_layout()
            savefig("06_ligand_com_distance.png")

        def plot_fes_rmsd_rg():
            if 0 not in all_rmsd or 0 not in all_rg:
                log("WARNING: missing state-0 RMSD/Rg for FES.")
                return
            x, y = np.asarray(all_rmsd[0]), np.asarray(all_rg[0])
            if len(x) == 0 or len(y) == 0:
                return

            if fes_outlier_z:
                keep = robust_outlier_mask(x, fes_outlier_z) & robust_outlier_mask(
                    y, fes_outlier_z
                )
                n_out = int((~keep).sum())
                max_out = max(1, int(0.01 * len(x)))
                if 0 < n_out <= max_out:
                    t_ns = state_trajs[0].time
                    dropped = ", ".join(
                        f"frame {i} ({t_ns[i]:.2f} ns: RMSD={x[i]:.4f}, Rg={y[i]:.4f} nm)"
                        for i in np.where(~keep)[0][:20]
                    )
                    log(
                        f"  FES: removed {n_out} outlier frame(s) beyond "
                        f"{fes_outlier_z:g} robust z-scores: {dropped}"
                        f"{' ...' if n_out > 20 else ''} (CSV files keep all frames)"
                    )
                    x, y = x[keep], y[keep]
                elif n_out > max_out:
                    log(
                        f"  FES: {n_out} frames ({100.0 * n_out / len(x):.1f}%) lie beyond "
                        f"{fes_outlier_z:g} robust z-scores - too many to be noise, "
                        f"kept as a genuine sub-population."
                    )
            H, xe, ye = np.histogram2d(x, y, bins=45)
            G = -KB_KJ_PER_MOL_K * temperatures[0] * np.log(H.T / H.max() + 1e-5)
            G_smooth = gaussian_filter(
                np.where(np.isinf(G), np.nanmax(G[~np.isinf(G)]), G), sigma=1.1
            )

            fig, ax = plt.subplots(figsize=(8, 7))
            extent = [xe[0], xe[-1], ye[0], ye[-1]]
            im = ax.imshow(
                G_smooth,
                cmap="inferno",
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
            )
            ax.grid(False)

            cbar = plt.colorbar(im, ax=ax, shrink=0.75, aspect=35, pad=0.02)
            cbar.set_label("Gibbs Free Energy (kJ/mol)")

            ax.set_xlabel("Unscaled State 0 Backbone RMSD (nm)")
            ax.set_ylabel("Radius of Gyration Rg (nm)")
            ax.set_title(f"Ground Ensemble FES ({temperatures[0]:.0f} K)")
            plt.tight_layout()
            savefig("07_fes_rmsd_rg.png")

        def plot_rmsf():
            if 0 not in state_trajs:
                return
            traj0 = state_trajs[0]
            ca_sel = traj0.topology.select(CA_SEL)
            if ca_sel is None or len(ca_sel) == 0:
                return
            traj_ca = traj0.atom_slice(ca_sel)
            if ref_ca is None or len(ref_ca) == 0:
                return

            traj_ca_super = traj_ca.superpose(ref_ca)
            rmsf_nm = md.rmsf(traj_ca_super, ref_ca)  # nm
            resids = [traj0.topology.atom(i).residue.resSeq for i in ca_sel]

            fig, ax = plt.subplots(figsize=(9, 5))
            # Original per-residue RMSF only - no smoothing overlay.
            ax.plot(resids, rmsf_nm, color=RMSF_MAIN_COLOR, lw=1.6, zorder=3)
            ax.set_title(
                f"Ground State Per-Residue C\u03b1 RMSF ({temperatures[0]:.1f} K)"
            )
            ax.set_xlabel("Residue")
            ax.set_ylabel("RMSF (nm)")
            plt.tight_layout()
            savefig("08_rmsf.png")

        def plot_state_comparison():
            """Structural metrics vs effective temperature, across ALL states.

            Every other structural plot shows state 0 only; this one answers
            'do the hot replicas actually differ from the ground state?'. Flat
            curves mean the ladder is too weak, or the run too short, to change
            the conformational ensemble."""
            states = sorted(s for s in state_trajs if len(state_trajs[s]) > 0)
            if len(states) < 2:
                return
            temps = [temperatures[s] if s < len(temperatures) else np.nan for s in states]

            panels = [
                (all_rmsd, "Backbone RMSD (nm)", RMSD_SERIES_COLORS["backbone"]),
                (all_rg, "Rg (nm)", RG_COLOR),
            ]
            if not protein_only and any(s in all_lig_rmsd for s in states):
                panels.append(
                    (all_lig_rmsd, "Ligand RMSD (nm)", RMSD_SERIES_COLORS["lig_pose"])
                )

            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
            axes = np.atleast_1d(axes)
            for ax, (data, label, color) in zip(axes, panels):
                means, errs, xs = [], [], []
                for s, T in zip(states, temps):
                    values = np.asarray(data.get(s, []), dtype=float)
                    values = values[np.isfinite(values)]
                    if len(values) == 0:
                        continue
                    xs.append(T)
                    means.append(values.mean())
                    # Standard error; frames are correlated, so this is a floor
                    # on the true uncertainty, not a rigorous error bar.
                    errs.append(values.std() / max(1.0, math.sqrt(len(values))))
                if not xs:
                    continue
                ax.errorbar(xs, means, yerr=errs, marker="o", color=color,
                            lw=1.6, capsize=3, zorder=3)
                ax.set_xlabel("Effective temperature (K)")
                ax.set_ylabel(label)
                ax.set_title(f"{label.split(' (')[0]} vs state")

            fig.suptitle(
                "Per-State Structural Metrics (flat = no enhancement from the ladder)",
                y=1.02,
            )
            plt.tight_layout()
            savefig("09_state_comparison.png")

        def plot_rmsf_by_state():
            """Per-residue RMSF for every state, coldest to hottest."""
            states = sorted(s for s in state_trajs if len(state_trajs[s]) > 1)
            if len(states) < 2 or ref_ca is None or len(ref_ca) == 0:
                return
            ca_sel = state_trajs[states[0]].topology.select(CA_SEL)
            if ca_sel is None or len(ca_sel) == 0:
                return
            resids = [
                state_trajs[states[0]].topology.atom(i).residue.resSeq for i in ca_sel
            ]

            fig, ax = plt.subplots(figsize=(11, 6))
            plotted = 0
            for s in states:
                try:
                    traj_ca = state_trajs[s].atom_slice(ca_sel)
                    rmsf_nm = md.rmsf(traj_ca.superpose(ref_ca), ref_ca)
                except Exception as exc:  # noqa: BLE001 - one bad state shouldn't kill the plot
                    log(f"  WARNING: RMSF failed for state {s} ({exc}).")
                    continue
                ax.plot(resids, rmsf_nm, lw=1.2, color=state_color(s, len(states)),
                        label=f"State {s} ({temperatures[s]:.0f} K)")
                plotted += 1
            if plotted < 2:
                plt.close()
                return
            ax.set_xlabel("Residue")
            ax.set_ylabel("RMSF (nm)")
            ax.set_title("Per-Residue Cα RMSF Across the Ladder")
            ax.legend(ncol=2, frameon=True, fontsize=8)
            plt.tight_layout()
            savefig("10_rmsf_by_state.png")

        plot_rmsd()
        plot_rg()
        plot_ligand_distance()
        plot_fes_rmsd_rg()
        plot_rmsf()
        plot_state_comparison()
        plot_rmsf_by_state()

    if S["state_trajs"] and not skip_structural:
        run_structural_block()

    # ── 4b. Energy decomposition: E_solute (intramolecular) vs
    #        E_solute-water (interaction) - the classic REST validation
    #        diagnostic (Wang/Terakawa-style figures), generalized from
    #        "protein" to "solute" (protein+ligand) since that's what's
    #        actually REST-scaled in this pipeline (see simulation_run.py's
    #        rest_atoms = protein U ligand), not just the bare protein the
    #        original reference figures used (which had no ligand at all).
    def run_energy_decomposition_block():
        if not HAS_OPENMM_PARMED:
            log(
                "WARNING: openmm/parmed not installed - skipping E_solute/E_water "
                "energy decomposition. Install with: conda install -c conda-forge openmm parmed"
            )
            return
        if not os.path.exists(topology):
            log("WARNING: topology file not found - skipping energy decomposition.")
            return

        state_trajs = S["state_trajs"]
        if not state_trajs:
            return
        first_traj = state_trajs.get(0)
        if first_traj is None:
            return

        protein_indices = sorted(set(first_traj.topology.select(PROT_SEL)))
        ligand_indices = sorted(set(int(i) for i in ligand_indices_global))
        solute_indices = sorted(set(protein_indices) | set(ligand_indices))
        water_indices = sorted(set(first_traj.topology.select("water")))

        if not solute_indices:
            log(
                f"WARNING: no {solute_label.lower()} atoms found - skipping energy decomposition."
            )
            return
        if not water_indices:
            log("WARNING: no water atoms found - skipping energy decomposition.")
            return
        if protein_only:
            log("  Protein-only system: E_solute = E_protein (no ligand split).")
        elif not ligand_indices:
            log(
                "WARNING: no ligand atoms matched selection "
                f"'{ligand_sel}' - protein/ligand split will be skipped, only "
                "combined E_solute will be reported."
            )

        log(
            "Building energy-decomposition systems (solute-only, water-only, combined)..."
        )
        log(
            "  NOTE: solute/water/combined/protein/ligand sub-systems ALL use the "
            "same plain periodic cutoff (not PME), so that both "
            "E_combined = E_solute + E_water + E_interaction and "
            "E_solute = E_protein + E_ligand + E_protein_ligand_interaction "
            "are exactly additive - this only holds if every term uses an "
            "identical nonbonded treatment. This is a deliberate simplification "
            "for this diagnostic only; it does not affect the production run itself."
        )

        try:
            structure = pmd.load_file(topology)
            combined_indices = sorted(solute_indices + water_indices)

            # Every sub-system must use the identical nonbonded treatment, or
            # the decomposition is not additive: E_combined = E_solute +
            # E_water + E_interaction only holds when each term is computed the
            # same way. The component energies are therefore cutoff-truncated,
            # which is the right trade for this diagnostic and does not affect
            # the PME production run.
            decomp_cutoff_kwargs = dict(
                nonbondedMethod=CutoffPeriodic,
                nonbondedCutoff=1.0 * nanometer,
                constraints=None,
            )

            solute_system = structure[solute_indices].createSystem(
                rigidWater=False, **decomp_cutoff_kwargs
            )
            water_system = structure[water_indices].createSystem(
                rigidWater=True, **decomp_cutoff_kwargs
            )
            combined_system = structure[combined_indices].createSystem(
                rigidWater=True, **decomp_cutoff_kwargs
            )

            # Extra sub-systems used ONLY to split the combined solute energy
            # into protein-only / ligand-only / protein-ligand-interaction
            # pieces for diagnostic purposes. Same CutoffPeriodic treatment
            # as above, for the same additivity reason.
            protein_system = None
            ligand_system = None
            if ligand_indices and protein_indices:
                protein_system = structure[protein_indices].createSystem(
                    rigidWater=False, **decomp_cutoff_kwargs
                )
                ligand_system = structure[ligand_indices].createSystem(
                    rigidWater=False, **decomp_cutoff_kwargs
                )
        except Exception as e:
            log(
                f"WARNING: could not build energy-decomposition systems ({e}) - skipping."
            )
            return

        platform = openmm.Platform.getPlatformByName("CPU")
        ctx_solute = openmm.Context(
            solute_system, openmm.VerletIntegrator(1.0), platform
        )
        ctx_water = openmm.Context(water_system, openmm.VerletIntegrator(1.0), platform)
        ctx_combined = openmm.Context(
            combined_system, openmm.VerletIntegrator(1.0), platform
        )
        ctx_protein = (
            openmm.Context(protein_system, openmm.VerletIntegrator(1.0), platform)
            if protein_system is not None
            else None
        )
        ctx_ligand = (
            openmm.Context(ligand_system, openmm.VerletIntegrator(1.0), platform)
            if ligand_system is not None
            else None
        )

        # Map global atom index -> local index within solute_indices, so we
        # can pull the ligand-only / protein-only position blocks out of the
        # positions already sliced for the solute context.
        solute_pos_of = {atom: i for i, atom in enumerate(solute_indices)}
        protein_local = (
            np.array([solute_pos_of[a] for a in protein_indices])
            if protein_system is not None
            else None
        )
        ligand_local = (
            np.array([solute_pos_of[a] for a in ligand_indices])
            if ligand_system is not None
            else None
        )

        KJ_TO_KCAL = 1.0 / 4.184
        # Outlier threshold (kcal/mol) for flagging individual frames whose
        # E_solute is far from the bulk of the distribution - a large,
        # discrete jump like this usually signals a structural/numerical
        # artifact in that specific frame rather than real physics.
        OUTLIER_Z = 5.0

        all_e_solute = {}
        all_e_water_int = {}
        all_e_protein = {}
        all_e_ligand = {}
        all_e_protein_ligand_int = {}
        outlier_rows = []
        eff_stride = max(1, energy_decomposition_stride)

        for s, traj in sorted(state_trajs.items()):
            n_frames_total = len(traj)
            if n_frames_total == 0:
                continue
            frame_indices = list(range(0, n_frames_total, eff_stride))
            e_solute_vals = []
            e_water_int_vals = []
            e_protein_vals = []
            e_ligand_vals = []
            e_pl_int_vals = []

            for fi in frame_indices:
                full_pos_nm = traj.xyz[fi]  # (n_atoms, 3), nm
                box_nm = (
                    traj.unitcell_vectors[fi]
                    if traj.unitcell_vectors is not None
                    else None
                )

                solute_pos_nm = full_pos_nm[solute_indices]
                ctx_solute.setPositions(solute_pos_nm * nanometer)
                if box_nm is not None:
                    ctx_solute.setPeriodicBoxVectors(*(box_nm * nanometer))
                E_solute = (
                    ctx_solute.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(kilojoules_per_mole)
                )

                ctx_water.setPositions(full_pos_nm[water_indices] * nanometer)
                if box_nm is not None:
                    ctx_water.setPeriodicBoxVectors(*(box_nm * nanometer))
                E_water = (
                    ctx_water.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(kilojoules_per_mole)
                )

                ctx_combined.setPositions(full_pos_nm[combined_indices] * nanometer)
                if box_nm is not None:
                    ctx_combined.setPeriodicBoxVectors(*(box_nm * nanometer))
                E_combined = (
                    ctx_combined.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(kilojoules_per_mole)
                )

                E_water_interaction = E_combined - E_solute - E_water

                e_solute_vals.append(E_solute * KJ_TO_KCAL)
                e_water_int_vals.append(E_water_interaction * KJ_TO_KCAL)

                E_protein = np.nan
                E_ligand = np.nan
                E_pl_int = np.nan
                if ctx_protein is not None and ctx_ligand is not None:
                    ctx_protein.setPositions(solute_pos_nm[protein_local] * nanometer)
                    if box_nm is not None:
                        ctx_protein.setPeriodicBoxVectors(*(box_nm * nanometer))
                    E_protein = (
                        ctx_protein.getState(getEnergy=True)
                        .getPotentialEnergy()
                        .value_in_unit(kilojoules_per_mole)
                    )

                    ctx_ligand.setPositions(solute_pos_nm[ligand_local] * nanometer)
                    if box_nm is not None:
                        ctx_ligand.setPeriodicBoxVectors(*(box_nm * nanometer))
                    E_ligand = (
                        ctx_ligand.getState(getEnergy=True)
                        .getPotentialEnergy()
                        .value_in_unit(kilojoules_per_mole)
                    )

                    E_pl_int = E_solute - E_protein - E_ligand

                e_protein_vals.append(E_protein * KJ_TO_KCAL if np.isfinite(E_protein) else np.nan)
                e_ligand_vals.append(E_ligand * KJ_TO_KCAL if np.isfinite(E_ligand) else np.nan)
                e_pl_int_vals.append(E_pl_int * KJ_TO_KCAL if np.isfinite(E_pl_int) else np.nan)

            all_e_solute[s] = np.array(e_solute_vals)
            all_e_water_int[s] = np.array(e_water_int_vals)
            all_e_protein[s] = np.array(e_protein_vals)
            all_e_ligand[s] = np.array(e_ligand_vals)
            all_e_protein_ligand_int[s] = np.array(e_pl_int_vals)

            # ── Flag outlier frames for this state ──────────────────────
            E_arr = all_e_solute[s]
            if len(E_arr) >= 5 and np.nanstd(E_arr) > 0:
                mu, sigma = np.nanmean(E_arr), np.nanstd(E_arr)
                z = (E_arr - mu) / sigma
                bad = np.where(np.abs(z) > OUTLIER_Z)[0]
                for local_i in bad:
                    fi = frame_indices[local_i]
                    outlier_rows.append(
                        {
                            "state": s,
                            "temperature_K": temperatures[s],
                            "frame": fi,
                            "time_ns": traj.time[fi] if fi < len(traj.time) else np.nan,
                            "E_solute_kcal_mol": E_arr[local_i],
                            "E_protein_kcal_mol": all_e_protein[s][local_i],
                            "E_ligand_kcal_mol": all_e_ligand[s][local_i],
                            "E_protein_ligand_interaction_kcal_mol": all_e_protein_ligand_int[s][local_i],
                            "z_score": z[local_i],
                        }
                    )

            log(
                f"  State {s:02d}: {len(frame_indices)} frames processed for energy decomposition "
                f"({len([b for b in outlier_rows if b['state'] == s])} outlier frame(s) flagged)."
            )

        del ctx_solute, ctx_water, ctx_combined
        if ctx_protein is not None:
            del ctx_protein
        if ctx_ligand is not None:
            del ctx_ligand

        if not all_e_solute:
            return

        # ── Outlier-frame report ─────────────────────────────────────────
        if outlier_rows:
            outlier_df = drop_ligand_columns(
                pd.DataFrame(outlier_rows).sort_values(
                    "z_score", key=lambda c: c.abs(), ascending=False
                )
            )
            outlier_csv = os.path.join(output_dir, "energy_decomposition_outliers.csv")
            outlier_df.to_csv(outlier_csv, index=False)
            csvs_written.append(outlier_csv)
            log(
                f"\n  WARNING: {len(outlier_df)} outlier frame(s) flagged "
                f"(|z| > {OUTLIER_Z:.0f} on E_solute) - see {outlier_csv}"
            )
            log(
                "  Inspect E_protein_kcal_mol / E_ligand_kcal_mol / "
                "E_protein_ligand_interaction_kcal_mol columns in that file to see "
                "which component is actually driving each outlier, and pull the "
                "listed (state, frame) pairs out with mdtraj to check geometry."
            )
        else:
            log(
                f"\n  No E_solute outlier frames flagged (|z| > {OUTLIER_Z:.0f} within any state)."
            )

        rows = []
        for s in sorted(all_e_solute.keys()):
            # beta_m/beta_0 = T_min/T_m (since beta = 1/kT). This MUST match
            # simulation_run.py's set_rest_parameters(), which scales solute
            # electrostatics by sqrt(beta_m/beta_0) - using T_m/T_min here
            # instead (the reciprocal) would apply a factor that grows with
            # temperature when the real one shrinks, inverting the state
            # ordering in the scaled/combination plots below.
            beta_ratio = t_min / temperatures[s]
            half_sqrt_beta_ratio = 0.5 * math.sqrt(beta_ratio)
            n = len(all_e_solute[s])
            e_prot = all_e_protein[s] if s in all_e_protein else np.full(n, np.nan)
            e_lig = all_e_ligand[s] if s in all_e_ligand else np.full(n, np.nan)
            e_pl = (
                all_e_protein_ligand_int[s]
                if s in all_e_protein_ligand_int
                else np.full(n, np.nan)
            )
            for e_sol, e_int, ep, el, epl in zip(
                all_e_solute[s], all_e_water_int[s], e_prot, e_lig, e_pl
            ):
                rows.append(
                    {
                        "state": s,
                        "temperature_K": temperatures[s],
                        "beta_m_over_beta_0": beta_ratio,
                        "E_solute_kcal_mol": e_sol,
                        "E_protein_kcal_mol": ep,
                        "E_ligand_kcal_mol": el,
                        "E_protein_ligand_interaction_kcal_mol": epl,
                        "E_solute_water_interaction_kcal_mol": e_int,
                        "scaled_pw_kcal_mol": half_sqrt_beta_ratio * e_int,
                        "chi_kcal_mol": e_sol + half_sqrt_beta_ratio * e_int,
                    }
                )
        decomp_df = drop_ligand_columns(pd.DataFrame(rows))
        decomp_csv = os.path.join(output_dir, "energy_decomposition.csv")
        decomp_df.to_csv(decomp_csv, index=False)
        csvs_written.append(decomp_csv)
        log(f"  -> {decomp_csv}")

        fig, ax = plt.subplots(figsize=(11, 6))
        for s in sorted(all_e_solute.keys()):
            E = all_e_solute[s]
            if len(E) < 3:
                continue
            try:
                kde = gaussian_kde(E, bw_method=0.25)
            except (np.linalg.LinAlgError, ValueError):
                log(
                    f"  WARNING: E_solute series for state {s} is degenerate "
                    f"(constant) - skipping KDE."
                )
                continue
            x = np.linspace(E.min() - 1.5 * E.std(), E.max() + 1.5 * E.std(), 300)
            c = state_color(s, n_replicas)
            ax.fill_between(x, kde(x), alpha=0.12, color=c)
            ax.plot(
                x, kde(x), color=c, lw=1.8, label=f"State {s} ({temperatures[s]:.1f} K)"
            )
        ax.set_xlabel("E$_{solute}$ (kcal/mol)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{solute_label} Intramolecular Energy Distributions")
        ax.legend(ncol=2, frameon=True)
        plt.tight_layout()
        savefig("11_e_solute_distributions.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        for s in sorted(all_e_water_int.keys()):
            E = all_e_water_int[s]
            if len(E) < 3:
                continue
            try:
                kde = gaussian_kde(E, bw_method=0.25)
            except (np.linalg.LinAlgError, ValueError):
                log(
                    f"  WARNING: E_solute-water series for state {s} is degenerate "
                    f"(constant) - skipping KDE."
                )
                continue
            x = np.linspace(E.min() - 1.5 * E.std(), E.max() + 1.5 * E.std(), 300)
            c = state_color(s, n_replicas)
            ax.fill_between(x, kde(x), alpha=0.12, color=c)
            ax.plot(
                x, kde(x), color=c, lw=1.8, label=f"State {s} ({temperatures[s]:.1f} K)"
            )
        ax.set_xlabel("E$_{solute-water}$ (kcal/mol)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"{solute_label}-Water Interaction Energy Distributions")
        ax.legend(ncol=2, frameon=True)
        plt.tight_layout()
        savefig("12_e_solute_water_distributions.png")

        fig, ax = plt.subplots(figsize=(8, 7))
        for s in sorted(all_e_solute.keys()):
            ax.scatter(
                all_e_solute[s],
                all_e_water_int[s],
                s=6,
                alpha=0.35,
                color=state_color(s, n_replicas),
                label=f"State {s} ({temperatures[s]:.0f} K)",
            )
        ax.set_xlabel("E$_{solute}$ (kcal/mol)")
        ax.set_ylabel("E$_{solute-water}$ (kcal/mol)")
        ax.set_title("Energy Component Overlap Across the REST Ladder")
        ax.legend(markerscale=2, ncol=2, frameon=True)
        plt.tight_layout()
        savefig("13_energy_component_scatter.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        for s in sorted(all_e_water_int.keys()):
            E = all_e_water_int[s]
            if len(E) < 3:
                continue
            lam = t_min / temperatures[s]  # beta_m/beta_0, matches simulation_run.py
            scaled = 0.5 * math.sqrt(lam) * E
            try:
                kde = gaussian_kde(scaled, bw_method=0.25)
            except (np.linalg.LinAlgError, ValueError):
                log(
                    f"  WARNING: scaled E_solute-water series for state {s} is "
                    f"degenerate (constant) - skipping KDE."
                )
                continue
            x = np.linspace(
                scaled.min() - 1.5 * scaled.std(),
                scaled.max() + 1.5 * scaled.std(),
                300,
            )
            c = state_color(s, n_replicas)
            ax.fill_between(x, kde(x), alpha=0.12, color=c)
            ax.plot(
                x, kde(x), color=c, lw=1.8, label=f"State {s} ({temperatures[s]:.1f} K)"
            )
        ax.set_xlabel(r"$(1/2)\sqrt{\beta_m/\beta_0}\ E_{pw}$ (kcal/mol)")
        ax.set_ylabel("Probability Density")
        ax.set_title("Scaled Solute-Water Interaction Energy Distributions")
        ax.legend(ncol=2, frameon=True)
        plt.tight_layout()
        savefig("14_scaled_e_solute_water_distributions.png")

        fig, ax = plt.subplots(figsize=(11, 6))
        for s in sorted(all_e_solute.keys()):
            E_sol = all_e_solute[s]
            E_int = all_e_water_int[s]
            if len(E_sol) < 3:
                continue
            lam = t_min / temperatures[s]  # beta_m/beta_0, matches simulation_run.py
            chi = E_sol + 0.5 * math.sqrt(lam) * E_int
            try:
                kde = gaussian_kde(chi, bw_method=0.25)
            except (np.linalg.LinAlgError, ValueError):
                log(
                    f"  WARNING: E_solute + (1/2)√(βm/β0) E_pw series for "
                    f"state {s} is degenerate (constant) - skipping KDE."
                )
                continue
            x = np.linspace(
                chi.min() - 1.5 * chi.std(), chi.max() + 1.5 * chi.std(), 300
            )
            c = state_color(s, n_replicas)
            ax.fill_between(x, kde(x), alpha=0.12, color=c)
            ax.plot(
                x, kde(x), color=c, lw=1.8, label=f"State {s} ({temperatures[s]:.1f} K)"
            )
        ax.set_xlabel(
            r"$E_{pp} + (1/2)\sqrt{\beta_m/\beta_0}\ E_{pw}$ (kcal/mol)"
        )
        ax.set_ylabel("Probability Density")
        ax.set_title("REST Exchange-Scope Combination Test Distributions")
        ax.legend(ncol=2, frameon=True)
        plt.tight_layout()
        savefig("15_rest_combination_distributions.png")

        # ── New: protein vs ligand intramolecular energy overlap ─────────
        def plot_protein_ligand_split():
            if protein_only:
                return
            has_split = any(
                np.any(np.isfinite(all_e_protein.get(s, np.array([]))))
                for s in all_e_protein
            )
            if not has_split:
                return

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            ax = axes[0]
            for s in sorted(all_e_protein.keys()):
                E = all_e_protein[s]
                E = E[np.isfinite(E)]
                if len(E) < 3:
                    continue
                try:
                    kde = gaussian_kde(E, bw_method=0.25)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                x = np.linspace(E.min() - 1.5 * E.std(), E.max() + 1.5 * E.std(), 300)
                c = state_color(s, n_replicas)
                ax.fill_between(x, kde(x), alpha=0.12, color=c)
                ax.plot(x, kde(x), color=c, lw=1.8, label=f"State {s} ({temperatures[s]:.1f} K)")
            ax.set_xlabel("E$_{protein}$ (kcal/mol)")
            ax.set_ylabel("Probability Density")
            ax.set_title("Protein-Only Intramolecular Energy")
            ax.legend(ncol=1, frameon=True, fontsize=8)

            ax = axes[1]
            for s in sorted(all_e_ligand.keys()):
                E = all_e_ligand[s]
                E = E[np.isfinite(E)]
                if len(E) < 3:
                    continue
                try:
                    kde = gaussian_kde(E, bw_method=0.25)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                x = np.linspace(E.min() - 1.5 * E.std(), E.max() + 1.5 * E.std(), 300)
                c = state_color(s, n_replicas)
                ax.fill_between(x, kde(x), alpha=0.12, color=c)
                ax.plot(x, kde(x), color=c, lw=1.8, label=f"State {s} ({temperatures[s]:.1f} K)")
            ax.set_xlabel("E$_{ligand}$ (kcal/mol)")
            ax.set_ylabel("Probability Density")
            ax.set_title("Ligand-Only Intramolecular Energy")
            ax.legend(ncol=1, frameon=True, fontsize=8)

            plt.tight_layout()
            savefig("16_protein_ligand_energy_split.png")

        plot_protein_ligand_split()

    if S["state_trajs"] and not skip_structural and not skip_energy_decomposition:
        run_energy_decomposition_block()

    # ── 5. Summary export ────────────────────────────────────────────────
    def write_summary():
        rows = []
        for s in range(n_replicas):
            row = {"thermodynamic_state": s, "temperature_K": temperatures[s]}
            if s < len(S["state_energy_dfs"]):
                E = S["state_energy_dfs"][s]["potential_kJ_mol"].dropna().values
                if len(E) > 0:
                    row.update(
                        {
                            "E_potential_mean_kJmol": E.mean(),
                            "E_potential_std_kJmol": E.std(),
                        }
                    )
            if (
                s in S["all_rmsd"]
                and S["all_rmsd"][s] is not None
                and len(S["all_rmsd"][s]) > 0
            ):
                row["backbone_rmsd_mean_nm"] = S["all_rmsd"][s].mean()
            if (
                s in S["all_lig_rmsd"]
                and S["all_lig_rmsd"][s] is not None
                and len(S["all_lig_rmsd"][s]) > 0
            ):
                row["ligand_rmsd_mean_nm"] = S["all_lig_rmsd"][s].mean()
            if (
                s in S["all_rg"]
                and S["all_rg"][s] is not None
                and len(S["all_rg"][s]) > 0
            ):
                row["rg_mean_nm"] = S["all_rg"][s].mean()
            if (
                s in S["all_lig_dist"]
                and S["all_lig_dist"][s] is not None
                and len(S["all_lig_dist"][s]) > 0
            ):
                row["ligand_rec_com_dist_mean_nm"] = S["all_lig_dist"][s].mean()
            rows.append(row)

        df_out = pd.DataFrame(rows)
        out_csv = os.path.join(output_dir, "summary_statistics.csv")
        df_out.to_csv(out_csv, index=False)
        log("\nSummary metrics:")
        log(df_out.to_string(index=False))
        log(f"\n  -> {out_csv}")
        csvs_written.append(out_csv)

    write_summary()
    log(f"\n{'='*65}\nANALYSIS COMPLETE: results exported to {output_dir}/\n{'='*65}\n")

    return {
        "output_dir": output_dir,
        "plots": plots_written,
        "csvs": csvs_written,
        "protein_only": protein_only,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REST2 publication-subset analysis")
    parser.add_argument(
        "--dir", default="rest2_output", help="Input production directory"
    )
    parser.add_argument(
        "--out", default="analysis_output", help="Output analysis directory"
    )
    parser.add_argument("--top", default="complex.prmtop", help="Amber topology file")
    parser.add_argument("--ref", default="complex.pdb", help="Reference PDB structure")
    parser.add_argument("--nc", default=None, help="NetCDF file (auto-detected)")
    parser.add_argument(
        "--nc-checkpoint", default=None, help="Checkpoint NetCDF (auto-detected)"
    )
    parser.add_argument("--replicas", type=int, default=7, help="Number of replicas")
    parser.add_argument(
        "--t-min", type=float, default=300.0, help="Minimum/reference temperature (K)"
    )
    parser.add_argument(
        "--t-max", type=float, default=328.0, help="Maximum effective temperature (K)"
    )
    parser.add_argument(
        "--ligand", default="resname MOL", help="Ligand MDTraj selection expression"
    )
    parser.add_argument(
        "--receptor", default="protein", help="Receptor MDTraj selection expression"
    )
    parser.add_argument(
        "--timestep", type=float, default=2.0, help="Simulation timestep (fs)"
    )
    parser.add_argument(
        "--steps-per-iter",
        type=int,
        default=1000,
        help="MD steps per exchange iteration",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=200,
        help="Cycles between saved checkpoints",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Frame stride for trajectory processing"
    )
    parser.add_argument(
        "--skip-structural", action="store_true", help="Skip structural parsing"
    )
    parser.add_argument(
        "--image-molecules",
        action="store_true",
        help="Run PBC image_molecules() on trajectories",
    )
    parser.add_argument(
        "--skip-energy-decomposition",
        action="store_true",
        help="Skip the E_solute/E_solute-water energy decomposition analysis",
    )
    parser.add_argument(
        "--energy-decomposition-stride",
        type=int,
        default=10,
        help="Extra frame thinning (on top of --stride) for the energy decomposition step - "
        "it's the slowest part of this script since it recomputes energies via OpenMM",
    )
    parser.add_argument(
        "--n-equil-frames",
        type=int,
        default=1,
        help="Number of leading (post-stride) frames to discard from each "
        "reconstructed trajectory as burn-in/equilibration (default: 1, "
        "which drops the pre-production frame at t=0). Set to 0 to disable.",
    )
    parser.add_argument(
        "--fes-outlier-z",
        type=float,
        default=5.0,
        help="Drop isolated frames beyond this robust z-score from the RMSD-Rg "
        "FES (only if they are <=1%% of frames). 0 disables.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--protein-only",
        dest="protein_only",
        action="store_const",
        const=True,
        default=None,
        help="Analyse an apo / ligand-free system (ignores --ligand). Default: "
        "read from run_metadata.json, else auto-detect from --ligand.",
    )
    mode.add_argument(
        "--protein-ligand",
        dest="protein_only",
        action="store_const",
        const=False,
        help="Force protein-ligand mode.",
    )
    args, _ = parser.parse_known_args()

    run_rest2_analysis(
        input_dir=args.dir,
        output_dir=args.out,
        topology=args.top,
        ref_pdb=args.ref,
        nc_file=args.nc,
        nc_checkpoint=args.nc_checkpoint,
        n_replicas=args.replicas,
        t_min=args.t_min,
        t_max=args.t_max,
        ligand_sel=args.ligand,
        receptor_sel=args.receptor,
        timestep=args.timestep,
        steps_per_iter=args.steps_per_iter,
        checkpoint_interval=args.checkpoint_interval,
        stride=args.stride,
        skip_structural=args.skip_structural,
        image_molecules=args.image_molecules,
        skip_energy_decomposition=args.skip_energy_decomposition,
        energy_decomposition_stride=args.energy_decomposition_stride,
        n_equil_frames=args.n_equil_frames,
        protein_only=args.protein_only,
        fes_outlier_z=args.fes_outlier_z,
    )
