"""
residue_utils.py - Auto-detect ligand/hetero residue names from a topology.

Used so the user doesn't have to already know/remember their ligand's residue
name (e.g. "MOL") to fill in the Simulation Run / Analysis pages - scans the
topology for residues that aren't standard amino acids, water, or common
ions, and reports whatever's left as candidate ligand/hetero residues.

Note: protein *selection* doesn't need this kind of detection at all - MDTraj's
"protein" keyword already recognizes standard amino acids generically, with no
name required. Ligand residue names are arbitrary per-project labels (MOL,
LIG, UNK, whatever antechamber or the original PDB happened to assign), which
is exactly what needs detecting.
"""

import parmed as pmd

STANDARD_AA_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "HIE", "HID", "HIP", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ACE", "NME", "NHE", "CYX", "CYM", "ASH", "GLH", "LYN", "HYP",
}

WATER_RESNAMES = {"WAT", "HOH", "TIP3", "TIP4", "TIP5", "SPC", "OPC", "T3P", "T4P"}

COMMON_ION_RESNAMES = {
    "NA", "CL", "K", "MG", "CA", "ZN", "F", "BR", "I", "LI", "CS", "RB",
    "NA+", "CL-", "K+", "MG2+", "CA2+", "ZN2+", "F-", "BR-", "I-",
    "SOD", "CLA", "POT", "MN", "FE", "CU", "NI", "CO",
}

EXCLUDED_RESNAMES = STANDARD_AA_RESNAMES | WATER_RESNAMES | COMMON_ION_RESNAMES


def detect_hetero_resnames(structure_file):
    """
    Loads a structure (anything ParmEd can read - PDB, prmtop, mol2, etc.)
    and returns a sorted list of residue names that are NOT standard amino
    acids, water, or common ions. In a typical protein-ligand system, what's
    left over is the ligand (and any other genuine heteroatoms/cofactors).

    Raises whatever exception ParmEd raises if the file can't be loaded/parsed.
    """
    structure = pmd.load_file(structure_file)
    found = set()
    for residue in structure.residues:
        name = residue.name.strip().upper()
        if name not in EXCLUDED_RESNAMES:
            found.add(name)
    return sorted(found)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python residue_utils.py <structure_file>")
        sys.exit(1)
    candidates = detect_hetero_resnames(sys.argv[1])
    if candidates:
        print("Candidate ligand/hetero residues:", ", ".join(candidates))
    else:
        print("No non-standard (hetero) residues found.")
