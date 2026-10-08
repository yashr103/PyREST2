"""
ligand_prep.py - Ligand Prep backend

Builds and runs the antechamber / parmchk2 commands used to parameterize
a ligand for tleap, based on the options chosen on the "Ligand Prep" page:

    - input ligand file (.sdf, .mol2, .pdb, ...)
    - partial charge method   (AM1-BCC / RESP / Gasteiger)
    - atom type / force field (gaff2 / gaff)
    - net charge
    - spin multiplicity
    - whether to generate the frcmod file

Equivalent to:

    antechamber -i ligand.sdf -fi sdf -o ligand.mol2 -fo mol2 -c bcc -s 2 -at gaff2
    parmchk2 -i ligand.mol2 -f mol2 -o ligand.frcmod -s gaff2

"""

import os
import shutil
import subprocess

# GUI label -> antechamber -c flag
CHARGE_METHOD_MAP = {
    "AM1-BCC": "bcc",
    "RESP": "resp",
    "Gasteiger": "gas",
}

# GUI label -> antechamber -at / parmchk2 -s flag
ATOM_TYPE_MAP = {
    "gaff2": "gaff2",
    "gaff": "gaff",
}

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


def _derive_conda_exe_from_env_path(env_path):
    """Given an env path like <BASE>/envs/<name>, find <BASE>/bin/conda or
    <BASE>/condabin/conda - guarantees using the SAME conda installation that
    owns the environment, which name-based lookup (-n) can't guarantee if the
    machine has more than one conda installation."""
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


def _find_on_path(exe_name):
    found = shutil.which(exe_name)
    if not found:
        raise FileNotFoundError(
            f"{exe_name} not found on PATH. Activate your AmberTools environment "
            f"first, or provide a conda environment name/path."
        )
    return found


def _tail(text, max_lines=40):
    lines = (text or "").strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "...\n" + "\n".join(lines[-max_lines:])


def _tool_failure_message(tool, result, workdir):
    """Collects every diagnostic AmberTools leaves behind. Reporting only
    `result.stderr or result.stdout` hides the real cause when the tool runs
    through `conda run`: conda writes its own generic 'command failed' line to
    stderr, shadowing the tool's own output on stdout.

    sqm.out matters most: antechamber's AM1-BCC charge step runs sqm, and a
    failure there (bad valence, wrong net charge, unconverged SCF) is by far
    the most common way ligand preparation dies."""
    sections = [f"{tool} failed."]
    for label, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        trimmed = _tail(text)
        if trimmed:
            sections.append(f"{tool} {label}:\n{trimmed}")

    for helper in ("sqm.out", "ANTECHAMBER.FRCMOD", "ATOMTYPE.INF"):
        path = os.path.join(workdir, helper)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", errors="replace") as fh:
                content = fh.read()
        except OSError:
            continue
        if helper == "sqm.out":
            failed = [ln for ln in content.splitlines()
                      if "ERROR" in ln or "Error" in ln or "not converge" in ln]
            if failed:
                sections.append("sqm.out errors:\n" + _tail("\n".join(failed)))
            sections.append("sqm.out tail:\n" + _tail(content))

    sections.append(f"Working directory (check the files above): {workdir}")
    return "\n\n".join(sections)


def _run(cmd_tail, workdir, conda_env=None, conda_env_path=None, conda_exe=None):
    """
    Runs cmd_tail (e.g. ["antechamber", "-i", ...]) either directly (PATH
    lookup) or via `conda run` inside a specific environment. Same priority
    order as system_generation.py's run_tleap: conda_env_path > conda_env >
    direct PATH lookup.
    """
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
        cmd = [
            conda_exe_resolved,
            "run",
            "-p",
            conda_env_path,
            "--no-capture-output",
        ] + cmd_tail
    elif conda_env:
        conda_exe_resolved = _find_conda_executable(conda_exe)
        cmd = [
            conda_exe_resolved,
            "run",
            "-n",
            conda_env,
            "--no-capture-output",
        ] + cmd_tail
    else:
        exe = _find_on_path(cmd_tail[0])
        cmd = [exe] + cmd_tail[1:]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    if result.returncode != 0:
        raise RuntimeError(_tool_failure_message(cmd_tail[0], result, workdir))
    return result


def prepare_ligand(
    input_file,
    charge_method="AM1-BCC",
    atom_type="gaff2",
    net_charge=0,
    multiplicity=1,
    output_name="ligand",
    generate_frcmod=True,
    workdir=None,
    conda_env=None,
    conda_env_path=None,
    conda_exe=None,
):
    """
    Run antechamber (+ optionally parmchk2) on a ligand file.

    Returns the paths of the files produced: (mol2_path, frcmod_path_or_None)
    """
    if atom_type not in ATOM_TYPE_MAP:
        raise ValueError(f"Unsupported atom type/force field: {atom_type}")
    if charge_method not in CHARGE_METHOD_MAP:
        raise ValueError(f"Unsupported charge method: {charge_method}")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Ligand input file not found: {input_file}")

    workdir = workdir or os.path.dirname(os.path.abspath(input_file)) or "."

    input_ext = os.path.splitext(input_file)[1].lstrip(".").lower()
    if not input_ext:
        raise ValueError(f"Could not determine input file format from: {input_file}")

    mol2_out = os.path.join(workdir, f"{output_name}.mol2")
    frcmod_out = os.path.join(workdir, f"{output_name}.frcmod")

    at_flag = ATOM_TYPE_MAP[atom_type]
    c_flag = CHARGE_METHOD_MAP[charge_method]

    antechamber_cmd = [
        "antechamber",
        "-i",
        input_file,
        "-fi",
        input_ext,
        "-o",
        mol2_out,
        "-fo",
        "mol2",
        "-c",
        c_flag,
        "-s",
        "2",
        "-at",
        at_flag,
        "-nc",
        str(net_charge),
        "-m",
        str(multiplicity),
    ]
    _run(antechamber_cmd, workdir, conda_env, conda_env_path, conda_exe)

    frcmod_path = None
    if generate_frcmod:
        parmchk2_cmd = [
            "parmchk2",
            "-i",
            mol2_out,
            "-f",
            "mol2",
            "-o",
            frcmod_out,
            "-s",
            at_flag,
        ]
        _run(parmchk2_cmd, workdir, conda_env, conda_env_path, conda_exe)
        frcmod_path = frcmod_out

    if not os.path.exists(mol2_out):
        raise RuntimeError(
            "antechamber ran but did not produce the expected .mol2 output. "
            "Check the ligand input file and charge/atom-type settings."
        )

    return mol2_out, frcmod_path


if __name__ == "__main__":
    # Example / manual test run (mirrors your original snippet's defaults)
    mol2_path, frcmod_path = prepare_ligand(
        input_file="ligand.sdf",
        charge_method="AM1-BCC",
        atom_type="gaff2",
        net_charge=0,
        multiplicity=1,
        output_name="ligand",
        generate_frcmod=True,
    )
    print("mol2:", mol2_path)
    print("frcmod:", frcmod_path)
