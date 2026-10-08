"""
system_generation.py - System Generation backend

Two-stage pipeline, matching the user's original script:

  1. run_tleap(...)          -> builds a tleap.in from the chosen force fields /
                                 water model / box / ion settings and runs tleap,
                                 producing complex.prmtop / complex.inpcrd / complex.pdb

  2. build_openmm_system(...) -> loads the resulting prmtop/inpcrd with ParmEd and
                                 creates the OpenMM System with the nonbonded /
                                 constraint / HMR settings from the "Output format"
                                 page, then re-saves the (HMR-adjusted) prmtop/inpcrd.

Equivalent by hand to:

    tleap -f tleap.in
    # then in Python:
    amber = pmd.load_file('complex.prmtop', 'complex.inpcrd')
    system = amber.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometers,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=1.5*amu,
        ewaldErrorTolerance=0.0005,
    )
"""

import math
import os
import subprocess
import shutil

import parmed as pmd
from openmm.app import PME, CutoffPeriodic, CutoffNonPeriodic, NoCutoff, Ewald
from openmm.app import HBonds, AllBonds, HAngles
from openmm.unit import nanometers, amu

# GUI label -> leaprc source line
PROTEIN_FF_MAP = {
    "leaprc.protein.ff19SB": "leaprc.protein.ff19SB",
    "leaprc.protein.ff14SB": "leaprc.protein.ff14SB",
    "leaprc.protein.ff99SBildn": "leaprc.protein.ff99SBildn",
}

LIGAND_FF_MAP = {
    "leaprc.gaff2": "leaprc.gaff2",
    "leaprc.gaff": "leaprc.gaff",
}

# GUI label -> (leaprc source line, tleap solvent box keyword)
WATER_MODEL_MAP = {
    "leaprc.water.opc": ("leaprc.water.opc", "OPCBOX"),
    "leaprc.water.tip3p": ("leaprc.water.tip3p", "TIP3PBOX"),
    "leaprc.water.tip4pew": ("leaprc.water.tip4pew", "TIP4PEWBOX"),
    "leaprc.water.spce": ("leaprc.water.spce", "SPCBOX"),
    "leaprc.water.opc3": ("leaprc.water.opc3", "OPC3BOX"),
}

# GUI label -> tleap solvate command
BOX_SHAPE_MAP = {
    "Truncated octahedron (solvateoct)": "solvateoct",
    "Cubic (solvatebox)": "solvatebox",
}

# GUI label -> OpenMM nonbonded method constant
NONBONDED_METHOD_MAP = {
    "PME": PME,
    "CutoffPeriodic": CutoffPeriodic,
    "CutoffNonPeriodic": CutoffNonPeriodic,
    "NoCutoff": NoCutoff,
    "Ewald": Ewald,
}

# GUI label -> OpenMM constraints constant ("None" -> None)
CONSTRAINTS_MAP = {
    "HBonds": HBonds,
    "AllBonds": AllBonds,
    "HAngles": HAngles,
    "None": None,
}


def _find_tleap(tleap_path=None):
    """Resolve the tleap executable: explicit path > PATH lookup."""
    if tleap_path:
        if not os.path.isfile(tleap_path):
            raise FileNotFoundError(f"tleap not found at: {tleap_path}")
        return tleap_path
    found = shutil.which("tleap")
    if not found:
        raise FileNotFoundError(
            "tleap executable not found on PATH. Activate your AmberTools "
            "environment first, or provide an explicit tleap path."
        )
    return found


# Common install locations to check if `conda` itself isn't on PATH - this
# happens a lot for GUI apps launched from a desktop icon or file manager,
# which typically don't inherit the shell's PATH the way a terminal does.
_COMMON_CONDA_LOCATIONS = [
    "~/miniconda3/bin/conda",
    "~/miniforge3/bin/conda",
    "~/anaconda3/bin/conda",
    "~/mambaforge/bin/conda",
    "/opt/conda/bin/conda",
    "/opt/miniconda3/bin/conda",
    "~/miniconda3/Scripts/conda.exe",
    "~/anaconda3/Scripts/conda.exe",
    "C:/ProgramData/miniconda3/Scripts/conda.exe",
    "C:/ProgramData/anaconda3/Scripts/conda.exe",
]


def _find_conda_executable(conda_exe=None):
    """Resolve the conda/mamba executable: explicit path > PATH > common install locations."""
    if conda_exe:
        if not os.path.isfile(conda_exe):
            raise FileNotFoundError(f"conda executable not found at: {conda_exe}")
        return conda_exe

    found = shutil.which("conda") or shutil.which("mamba")
    if found:
        return found

    for candidate in _COMMON_CONDA_LOCATIONS:
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded):
            return expanded

    raise FileNotFoundError(
        "Could not find a conda/mamba executable (not on PATH, and not in any "
        "common install location). This usually happens when the app is launched "
        "from a desktop icon rather than an already-activated terminal. Provide "
        "an explicit path to conda/conda.exe in the 'Conda executable' field."
    )


def build_tleap_script(
    protein_pdb,
    protein_ff="leaprc.protein.ff19SB",
    ligand_ff="leaprc.gaff2",
    water_model="leaprc.water.opc",
    ligand_mol2=None,
    ligand_frcmod=None,
    box_shape="Truncated octahedron (solvateoct)",
    box_padding=10.0,
    neutralize=True,
    cation="Na+",
    anion="Cl-",
    ion_counts=None,
    output_prefix="complex",
    save_parm=True,
    disulfide_bonds=None,
):
    """Builds the tleap.in text. Returns a str.

    ion_counts: optional (n_cation, n_anion) pair of ION COUNTS for
    addIonsRand. tleap has no concentration option - its third argument is a
    number of ions - so run_tleap() solvates once to count waters and converts
    the requested molarity into counts before calling this with ion_counts set.

    disulfide_bonds: optional list of (index_i, index_j, description) using
    tleap's 1-based sequential residue numbering. tleap never forms S-S bonds
    on its own, so these become explicit `bond` commands.
    """
    if protein_ff not in PROTEIN_FF_MAP:
        raise ValueError(f"Unsupported protein force field: {protein_ff}")
    if ligand_ff not in LIGAND_FF_MAP:
        raise ValueError(f"Unsupported ligand force field: {ligand_ff}")
    if water_model not in WATER_MODEL_MAP:
        raise ValueError(f"Unsupported water model: {water_model}")
    if box_shape not in BOX_SHAPE_MAP:
        raise ValueError(f"Unsupported box shape: {box_shape}")

    water_line, water_box = WATER_MODEL_MAP[water_model]
    solvate_cmd = BOX_SHAPE_MAP[box_shape]

    has_ligand = bool(ligand_mol2)

    lines = [f"source {PROTEIN_FF_MAP[protein_ff]}"]
    if has_ligand:
        # GAFF is only needed to type a ligand - skip it for protein-only boxes.
        lines.append(f"source {LIGAND_FF_MAP[ligand_ff]}")
    lines.append(f"source {water_line}")

    if has_ligand:
        lines.append(f'LIG = loadmol2 "{ligand_mol2}"')
        if ligand_frcmod:
            lines.append(f'loadamberparams "{ligand_frcmod}"')
        lines.append(f'PROT = loadpdb "{protein_pdb}"')
        lines.append("COMPLEX = combine {PROT LIG}")
    else:
        lines.append(f'COMPLEX = loadpdb "{protein_pdb}"')

    # Disulfides must be bonded while COMPLEX still holds only the solute -
    # the indices come from the protein PDB's residue order, and solvation
    # would append thousands of waters after them.
    for index_i, index_j, description in disulfide_bonds or []:
        lines.append(f"bond COMPLEX.{index_i}.SG COMPLEX.{index_j}.SG  # {description}")

    lines.append(f"{solvate_cmd} COMPLEX {water_box} {box_padding}")

    if neutralize:
        lines.append(f"addions COMPLEX {cation} 0")
        lines.append(f"addions COMPLEX {anion} 0")

    if ion_counts:
        n_cation, n_anion = ion_counts
        # addIonsRand takes a COUNT, not a concentration: a molarity here
        # either aborts tleap or silently adds zero salt.
        if n_cation > 0:
            lines.append(f"addionsrand COMPLEX {cation} {int(n_cation)}")
        if n_anion > 0:
            lines.append(f"addionsrand COMPLEX {anion} {int(n_anion)}")

    if save_parm:
        lines.append(
            f'saveamberparm COMPLEX "{output_prefix}.prmtop" "{output_prefix}.inpcrd"'
        )
    lines.append(f'savepdb COMPLEX "{output_prefix}.pdb"')
    lines.append("quit")

    return "\n".join(lines) + "\n"


def find_disulfides(pdb_path, cutoff=2.5):
    """Finds disulfide-bonded cysteine pairs by SG-SG distance.

    Returns a list of (index_i, index_j, description). The indices are 1-based
    in TLEAP's numbering, which is simply the order residues appear in the
    file - tleap renumbers sequentially from 1 on loadpdb and ignores the PDB
    resSeq column, so PDB residue numbers must not be used in bond commands.

    A typical disulfide has SG-SG ~2.03 A; 2.5 A is the usual detection
    cutoff (same value pdb4amber uses). Each cysteine can take part in at
    most one bond, closest pair first.
    """
    order, index_of, sg_xyz, resname_of = [], {}, {}, {}
    try:
        with open(pdb_path, "r", errors="replace") as fh:
            for line in fh:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                key = line[21:27]
                if key not in index_of:
                    index_of[key] = len(order) + 1  # 1-based, tleap ordering
                    order.append(key)
                    resname_of[key] = line[17:20].strip().upper()
                if line[12:16].strip() == "SG" and resname_of.get(key, "").startswith("CY"):
                    try:
                        sg_xyz[key] = (
                            float(line[30:38]), float(line[38:46]), float(line[46:54])
                        )
                    except ValueError:
                        pass
    except OSError:
        return []

    candidates = []
    keys = list(sg_xyz)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            distance = math.dist(sg_xyz[ka], sg_xyz[kb])
            if distance <= cutoff:
                candidates.append((distance, ka, kb))

    pairs, used = [], set()
    for distance, ka, kb in sorted(candidates):
        if ka in used or kb in used:
            continue
        used.update((ka, kb))
        pairs.append(
            (index_of[ka], index_of[kb], f"{ka.strip()}-{kb.strip()} ({distance:.2f} A)")
        )
    return sorted(pairs)


def sanitize_pdb_for_tleap(src_pdb, dest_pdb, disulfide_cutoff=2.5):
    """Fixes N-terminal backbone hydrogen naming and disulfide cysteines,
    writing the result to dest_pdb.

    Amber's N-terminal residue templates (NMET, NALA, ...) name the three
    backbone amine hydrogens H1/H2/H3 and define no atom called plain 'H'.
    PDBFixer (and most other tools) write the internal-residue name 'H' for
    the backbone amide hydrogen, so tleap cannot type it on residue 1 and
    aborts with

        FATAL:  Atom .R<NMET 1>.A<H 20> does not have a type.

    The bare 'H' on the first residue of each chain is therefore renamed to
    H1 when H1 is absent, or dropped when H1/H2/H3 are already present.
    Everything else - including protonation-state hydrogens elsewhere in the
    structure - is passed through untouched.

    Disulfides get the same treatment: tleap only forms an S-S bond when both
    residues are named CYX *and* an explicit `bond` command is given. PDBFixer
    leaves them as CYS, so without this they silently end up as four free
    thiols instead of two disulfides. Bonded cysteines are therefore renamed
    CYS -> CYX and their HG thiol hydrogen removed (the CYX template has no
    HG, so leaving it would trigger the same 'does not have a type' abort).

    Returns (changes, disulfides): a list of human-readable changes, and the
    (index_i, index_j, description) pairs for run_tleap's bond commands.
    """
    with open(src_pdb, "r", errors="replace") as fh:
        lines = fh.readlines()

    def residue_key(line):
        return line[21:27]  # chain + resSeq + insertion code

    # Disulfides, located before anything is rewritten.
    disulfides = find_disulfides(src_pdb, cutoff=disulfide_cutoff)
    ss_residue_keys = set()
    if disulfides:
        seen_keys, index_to_key = set(), {}
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                key = residue_key(line)
                if key not in seen_keys:
                    seen_keys.add(key)
                    index_to_key[len(seen_keys)] = key
        for index_i, index_j, _ in disulfides:
            ss_residue_keys.add(index_to_key.get(index_i))
            ss_residue_keys.add(index_to_key.get(index_j))
        ss_residue_keys.discard(None)

    # First residue of each chain; a TER record starts a new chain segment.
    first_residue_keys, seen_chains = set(), set()
    fresh_chain = True
    for line in lines:
        if line.startswith("TER"):
            fresh_chain = True
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21]
        if fresh_chain or chain not in seen_chains:
            first_residue_keys.add(residue_key(line))
            seen_chains.add(chain)
            fresh_chain = False

    # Atom names present in each of those first residues.
    names_by_residue = {}
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            key = residue_key(line)
            if key in first_residue_keys:
                names_by_residue.setdefault(key, set()).add(line[12:16].strip())

    changes, out, renamed_cyx = [], [], set()
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            key = residue_key(line)
            atom_name = line[12:16].strip()

            if key in ss_residue_keys:
                # CYX has no thiol hydrogen - drop it, then rename the residue.
                if atom_name in ("HG", "HG1"):
                    changes.append(f"removed {atom_name} from disulfide CYS {key.strip()}")
                    continue
                if line[17:20].strip().upper() == "CYS":
                    line = line[:17] + "CYX" + line[20:]
                    if key not in renamed_cyx:
                        renamed_cyx.add(key)
                        changes.append(f"CYS -> CYX on {key.strip()} (disulfide)")

            if key in first_residue_keys and atom_name == "H":
                resname = line[17:20].strip()
                if "H1" in names_by_residue.get(key, set()):
                    changes.append(f"removed stray H from N-terminal {resname} {key.strip()}")
                    continue
                line = line[:12] + " H1 " + line[16:]
                changes.append(f"renamed H -> H1 on N-terminal {resname} {key.strip()}")
        out.append(line)

    for _, _, description in disulfides:
        changes.append(f"disulfide bond {description}")

    with open(dest_pdb, "w") as fh:
        fh.writelines(out)
    return changes, disulfides


WATER_MOLARITY = 55.56  # mol/L of pure water, used to turn a salt
# concentration into a number of ions given the number of water molecules.


def count_waters_in_pdb(pdb_path, water_resnames=("WAT", "HOH", "SOL")):
    """Counts water MOLECULES (not atoms) in a tleap-written PDB."""
    residues = set()
    try:
        with open(pdb_path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith(("ATOM  ", "HETATM")):
                    if line[17:20].strip().upper() in water_resnames:
                        residues.add(line[21:27])  # chain + resSeq + insertion code
    except OSError:
        return 0
    return len(residues)


def ions_for_concentration(ion_conc, n_waters):
    """Number of ion pairs giving `ion_conc` (mol/L) among `n_waters` waters.

    n_ions = conc * n_waters / 55.56 - the standard 'salt per water' estimate
    (as used by e.g. gmx genion -conc), which avoids needing the box volume.
    """
    if ion_conc <= 0 or n_waters <= 0:
        return 0
    return int(round(ion_conc * n_waters / WATER_MOLARITY))


def _tail(text, max_lines=40):
    lines = (text or "").strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "...\n" + "\n".join(lines[-max_lines:])


def _read_leap_log(workdir, max_lines=40):
    """tleap's real diagnostics go to stdout and leap.log. When it is launched
    through `conda run`, conda only adds its own '<command> failed' line on
    stderr, so leap.log is usually the only place the actual reason survives."""
    log_path = os.path.join(workdir, "leap.log")
    if not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, "r", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ""

    keywords = ("FATAL", "Fatal", "ERROR", "Error:", "could not", "Could not",
                "not found", "Unknown", "missing", "Missing")
    flagged = [ln for ln in lines if any(k in ln for k in keywords)]
    sections = []
    if flagged:
        sections.append("leap.log errors:\n" + _tail("\n".join(flagged), max_lines))
    sections.append("leap.log tail:\n" + _tail("\n".join(lines), max_lines))
    return "\n\n".join(sections)


def _tleap_failure_message(headline, result, workdir):
    """Assembles every source of diagnostics. Reporting only
    `result.stderr or result.stdout` is not enough: under `conda run` stderr
    carries conda's generic 'command failed' line while tleap's own output -
    the part that says what actually went wrong - goes to stdout."""
    sections = [headline]
    for label, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        trimmed = _tail(text)
        if trimmed:
            sections.append(f"tleap {label}:\n{trimmed}")
    leap_log = _read_leap_log(workdir)
    if leap_log:
        sections.append(leap_log)
    sections.append(
        f"Full details in:\n  {os.path.join(workdir, 'leap.log')}\n"
        f"  {os.path.join(workdir, 'tleap.in')}  (the generated script)"
    )
    return "\n\n".join(sections)


def _derive_conda_exe_from_env_path(env_path):
    """Given an env path like <BASE>/envs/<name>, find <BASE>/bin/conda or
    <BASE>/condabin/conda - this guarantees using the SAME conda installation
    that owns the environment, which name-based lookup (-n) can't guarantee
    if the machine has more than one conda installation."""
    base = os.path.dirname(
        os.path.dirname(os.path.abspath(env_path))
    )  # strip .../envs/<name>
    candidates = [
        os.path.join(base, "bin", "conda"),
        os.path.join(base, "condabin", "conda"),
        os.path.join(base, "Scripts", "conda.exe"),
        os.path.join(base, "condabin", "conda.bat"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_tleap(
    protein_pdb,
    protein_ff="leaprc.protein.ff19SB",
    ligand_ff="leaprc.gaff2",
    water_model="leaprc.water.opc",
    ligand_mol2=None,
    ligand_frcmod=None,
    box_shape="Truncated octahedron (solvateoct)",
    box_padding=10.0,
    neutralize=True,
    cation="Na+",
    anion="Cl-",
    ion_conc=0.15,
    output_prefix="complex",
    workdir=None,
    tleap_path=None,
    conda_env=None,
    conda_env_path=None,
    conda_exe=None,
):
    """
    Writes tleap.in and runs tleap. Returns dict with paths to prmtop/inpcrd/pdb,
    plus the captured stdout for display in the GUI.

    Three ways to run tleap, in priority order:
      1. conda_env_path (recommended if you have multiple conda installations):
         an explicit path to the environment, e.g.
         "/home/user/miniforge3/envs/amber-env". Run via `conda run -p <path>`.
         Unambiguous - doesn't depend on environment *names* being unique
         across different conda installations on the same machine.
      2. conda_env: just the environment *name*, e.g. "amber-env". Run via
         `conda run -n <name>`. Can silently resolve to the wrong environment
         if more than one conda installation on the machine has an env with
         that name - if you hit "command not found" errors despite the name
         being correct, switch to conda_env_path instead.
      3. Neither given: tleap_path (explicit binary path) or PATH lookup,
         invoked directly (no conda activation at all).

    Options 1/2 matter because AmberTools (from conda-forge) typically needs
    more than just its own directory on PATH to work correctly - it needs
    LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on macOS) and other environment
    variables that only get set by properly activating the conda environment.
    Just pointing at the tleap binary's absolute path (tleap_path) skips that
    activation and can fail with missing-library or missing-data-file errors,
    even though the exact same binary works fine from an activated terminal.
    """
    workdir = workdir or os.path.dirname(os.path.abspath(protein_pdb)) or "."
    os.makedirs(workdir, exist_ok=True)

    # Amber's N-terminal templates have no atom named plain 'H'; fix that up
    # front rather than letting tleap die on "does not have a type".
    sanitized_pdb = os.path.join(
        workdir, os.path.splitext(os.path.basename(protein_pdb))[0] + "_tleap.pdb"
    )
    pdb_fixes, disulfides = sanitize_pdb_for_tleap(protein_pdb, sanitized_pdb)
    if pdb_fixes:
        protein_pdb = sanitized_pdb

    script = build_tleap_script(
        protein_pdb=protein_pdb,
        protein_ff=protein_ff,
        ligand_ff=ligand_ff,
        water_model=water_model,
        ligand_mol2=ligand_mol2,
        ligand_frcmod=ligand_frcmod,
        box_shape=box_shape,
        box_padding=box_padding,
        neutralize=neutralize,
        cation=cation,
        anion=anion,
        output_prefix=output_prefix,
        disulfide_bonds=disulfides,
    )

    def _execute(script_text, tleap_in_name):
        """Writes a tleap input file into workdir and runs tleap on it."""
        with open(os.path.join(workdir, tleap_in_name), "w") as f:
            f.write(script_text)

        if conda_env_path:
            if not os.path.isdir(conda_env_path):
                raise FileNotFoundError(
                    f"Conda environment path not found: {conda_env_path}"
                )
            conda_exe_resolved = (
                conda_exe
                or _derive_conda_exe_from_env_path(conda_env_path)
                or _find_conda_executable(None)
            )
            cmd = [conda_exe_resolved, "run", "-p", conda_env_path,
                   "--no-capture-output", "tleap", "-f", tleap_in_name]
        elif conda_env:
            conda_exe_resolved = _find_conda_executable(conda_exe)
            cmd = [conda_exe_resolved, "run", "-n", conda_env,
                   "--no-capture-output", "tleap", "-f", tleap_in_name]
        else:
            tleap_exe = _find_tleap(tleap_path)
            env = os.environ.copy()
            tleap_dir = os.path.dirname(tleap_exe)
            if tleap_dir:
                env["PATH"] = tleap_dir + os.pathsep + env.get("PATH", "")
            return subprocess.run([tleap_exe, "-f", tleap_in_name],
                                  capture_output=True, text=True, env=env, cwd=workdir)
        return subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)

    # ── Salt concentration -> ion count ──────────────────────────────────
    # tleap's addIonsRand wants a NUMBER OF IONS, so solvate once with only
    # the neutralising counterions, count the waters tleap added, and convert.
    n_waters, n_ions = 0, 0
    if ion_conc and ion_conc > 0:
        probe_prefix = f"{output_prefix}_watercount"
        probe_script = build_tleap_script(
            protein_pdb=protein_pdb,
            protein_ff=protein_ff,
            ligand_ff=ligand_ff,
            water_model=water_model,
            ligand_mol2=ligand_mol2,
            ligand_frcmod=ligand_frcmod,
            box_shape=box_shape,
            box_padding=box_padding,
            neutralize=neutralize,
            cation=cation,
            anion=anion,
            output_prefix=probe_prefix,
            save_parm=False,  # only need the PDB to count waters
            disulfide_bonds=disulfides,
        )
        probe = _execute(probe_script, "tleap_watercount.in")
        probe_pdb = os.path.join(workdir, f"{probe_prefix}.pdb")
        if probe.returncode != 0 or not os.path.isfile(probe_pdb):
            raise RuntimeError(
                _tleap_failure_message(
                    "tleap failed while solvating the system.", probe, workdir
                )
            )
        n_waters = count_waters_in_pdb(probe_pdb)
        n_ions = ions_for_concentration(ion_conc, n_waters)
        try:
            os.remove(probe_pdb)
        except OSError:
            pass

        script = build_tleap_script(
            protein_pdb=protein_pdb,
            protein_ff=protein_ff,
            ligand_ff=ligand_ff,
            water_model=water_model,
            ligand_mol2=ligand_mol2,
            ligand_frcmod=ligand_frcmod,
            box_shape=box_shape,
            box_padding=box_padding,
            neutralize=neutralize,
            cation=cation,
            anion=anion,
            ion_counts=(n_ions, n_ions),
            output_prefix=output_prefix,
            disulfide_bonds=disulfides,
        )

    result = _execute(script, "tleap.in")

    if result.returncode != 0:
        raise RuntimeError(_tleap_failure_message("tleap failed.", result, workdir))

    prmtop_path = os.path.join(workdir, f"{output_prefix}.prmtop")
    inpcrd_path = os.path.join(workdir, f"{output_prefix}.inpcrd")
    pdb_path = os.path.join(workdir, f"{output_prefix}.pdb")

    if not os.path.exists(prmtop_path) or not os.path.exists(inpcrd_path):
        # tleap often exits 0 even after a FATAL error, simply skipping
        # saveamberparm - so this path needs the same full diagnostics.
        raise RuntimeError(
            _tleap_failure_message(
                f"tleap ran but did not write {output_prefix}.prmtop / "
                f"{output_prefix}.inpcrd.",
                result,
                workdir,
            )
        )

    return {
        "prmtop": prmtop_path,
        "inpcrd": inpcrd_path,
        "pdb": pdb_path,
        "stdout": result.stdout,
        "n_waters": n_waters,
        "n_ion_pairs": n_ions,
        "ion_conc": ion_conc,
        "pdb_fixes": pdb_fixes,
        "disulfides": disulfides,
    }


def build_openmm_system(
    prmtop_path,
    inpcrd_path,
    nonbonded_method="PME",
    nonbonded_cutoff=1.0,
    constraints="HBonds",
    rigid_water=True,
    hmr_enabled=True,
    hydrogen_mass=1.5,
    ewald_error_tolerance=0.0005,
    resave=True,
):
    """
    Loads prmtop/inpcrd via ParmEd, builds the OpenMM System with the given
    nonbonded/constraint/HMR settings, and (optionally) re-saves the prmtop/inpcrd
    so the HMR-adjusted masses are baked in for the simulation stage.

    Returns dict with the System object, atom/residue counts, and file paths.
    """
    if nonbonded_method not in NONBONDED_METHOD_MAP:
        raise ValueError(f"Unsupported nonbonded method: {nonbonded_method}")
    if constraints not in CONSTRAINTS_MAP:
        raise ValueError(f"Unsupported constraints option: {constraints}")

    amber = pmd.load_file(prmtop_path, inpcrd_path)

    kwargs = dict(
        nonbondedMethod=NONBONDED_METHOD_MAP[nonbonded_method],
        nonbondedCutoff=nonbonded_cutoff * nanometers,
        constraints=CONSTRAINTS_MAP[constraints],
        rigidWater=rigid_water,
        ewaldErrorTolerance=ewald_error_tolerance,
    )
    if hmr_enabled:
        kwargs["hydrogenMass"] = hydrogen_mass * amu

    system = amber.createSystem(**kwargs)

    natom = amber.ptr("natom")
    nres = amber.ptr("nres")

    if resave:
        amber.save(prmtop_path, overwrite=True)
        amber.save(inpcrd_path, overwrite=True)

    return {
        "system": system,
        "natom": natom,
        "nres": nres,
        "prmtop": prmtop_path,
        "inpcrd": inpcrd_path,
    }


if __name__ == "__main__":
    # Example / manual test run (mirrors the original script's defaults)
    tleap_result = run_tleap(
        protein_pdb="prot.pdb",
        ligand_mol2="ligand.mol2",
        ligand_frcmod="ligand.frcmod",
    )
    print(tleap_result["stdout"])

    sys_result = build_openmm_system(
        tleap_result["prmtop"],
        tleap_result["inpcrd"],
    )
    print("System built successfully!")
    print(f"  Atoms   : {sys_result['natom']}")
    print(f"  Residues: {sys_result['nres']}")
