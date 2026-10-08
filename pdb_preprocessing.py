"""
pdb_preprocessing.py - PDB Preprocessing backend

Wraps PDBFixer to clean up an input PDB the way the "PDB Preprocessing" page
expects, based on:

    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename='prot.pdb')
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)       # remove waters/ligands from protein file
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)     # physiological pH
    with open('protein_fixed.pdb', 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

Notes on what PDBFixer can/can't do, mapped to the GUI's options:

  - Remove waters / remove heterogens: PDBFixer.removeHeterogens(keepWater)
    is a single all-or-nothing call for non-water heterogens; there's no
    native "keep ligands but drop only water" mode. If either checkbox is
    on, we call it once with keepWater = (not remove_waters).
  - Chain selection: PDBFixer.removeChains(chainIds=...) is used directly.
  - Missing residues (loop modeling): PDBFixer.findMissingResidues() always
    detects gaps; if the GUI option is off we clear fixer.missingResidues
    afterward so addMissingAtoms() doesn't try to build loops.
  - Nonstandard residues are always standardized (e.g. MSE -> MET) - this
    is safe/standard practice and isn't exposed as a toggle.
  - Missing atoms are always completed; missing *hydrogens* are only added
    if the GUI checkbox is on, using the chosen pH.
  - Alternate-location selection: NOT implemented here - PDBFixer has no
    native altloc API. If your input has altlocs, pre-filter it (e.g. with
    pdb-tools' `pdb_selaltloc`) before running this.
  - Disulfide bonds: PDBFixer doesn't auto-detect/add SS bonds. This module
    does a simple geometric scan (SG-SG distance) on the fixed structure and
    reports candidate pairs; it does NOT add explicit bonds to the tleap
    script. If you need that, the pairs returned here can be fed into
    system_generation.py as extra 'bond' lines - ask if you want that wired up.
"""

import os

from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openmm.unit import nanometers


def fix_pdb(
    input_pdb,
    output_pdb=None,
    remove_waters=True,
    remove_heterogens=True,
    chain_select="",
    model_missing_residues=False,
    add_hydrogens=True,
    ph=7.0,
    detect_disulfides=True,
    disulfide_cutoff=2.5,
):
    """
    Runs the PDBFixer cleanup pipeline. Returns a dict with:
        output_pdb            - path to the written, fixed PDB
        n_missing_residues     - number of missing-residue gaps found
        n_missing_atoms_added  - number of atoms added to existing residues
        chains_kept             - list of chain IDs kept
        disulfide_pairs         - list of (resid1, resid2) CYS pairs within cutoff
    """
    if not os.path.isfile(input_pdb):
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    output_pdb = output_pdb or _default_output_path(input_pdb)

    fixer = PDBFixer(filename=input_pdb)

    # ── Chain selection ──────────────────────────────────────────────
    requested_chains = {c.strip().upper() for c in chain_select.split(",") if c.strip()}
    all_chain_ids = {chain.id for chain in fixer.topology.chains()}
    if requested_chains:
        matched_chains = {
            cid for cid in all_chain_ids if cid.upper() in requested_chains
        }
        if not matched_chains:
            raise ValueError(
                f"Chain selection '{chain_select}' didn't match any chain in the "
                f"input PDB. Available chains: {sorted(all_chain_ids)}"
            )
        remove_ids = [
            cid for cid in all_chain_ids if cid.upper() not in requested_chains
        ]
        if remove_ids:
            fixer.removeChains(chainIds=remove_ids)
    chains_kept = sorted({chain.id for chain in fixer.topology.chains()})

    # ── Missing residues (loop modeling) ────────────────────────────
    fixer.findMissingResidues()
    n_missing_residues = sum(len(v) for v in fixer.missingResidues.values())
    if not model_missing_residues:
        fixer.missingResidues = {}

    # ── Nonstandard residues (always standardized) ──────────────────
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # ── Heterogens / waters ──────────────────────────────────────────
    if remove_heterogens or remove_waters:
        keep_water = not remove_waters
        fixer.removeHeterogens(keepWater=keep_water)

    # ── Missing atoms (always completed) + hydrogens (optional) ─────
    fixer.findMissingAtoms()
    n_missing_atoms_added = sum(len(v) for v in fixer.missingAtoms.values()) + len(
        getattr(fixer, "missingTerminals", {})
    )
    fixer.addMissingAtoms()

    if add_hydrogens:
        fixer.addMissingHydrogens(ph)

    with open(output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    disulfide_pairs = []
    if detect_disulfides:
        disulfide_pairs = _detect_disulfides(
            fixer.topology, fixer.positions, disulfide_cutoff
        )

    return {
        "output_pdb": output_pdb,
        "n_missing_residues": n_missing_residues,
        "n_missing_atoms_added": n_missing_atoms_added,
        "chains_kept": chains_kept,
        "disulfide_pairs": disulfide_pairs,
    }


def _default_output_path(input_pdb):
    base, _ = os.path.splitext(input_pdb)
    return f"{base}_fixed.pdb"


def _detect_disulfides(topology, positions, cutoff_angstrom=2.5):
    """Simple geometric scan for candidate disulfide bonds: any two CYS SG
    atoms within cutoff_angstrom of each other. Returns a list of
    ((chain_id, resnum), (chain_id, resnum)) tuples. Does not modify anything."""
    cutoff_nm = cutoff_angstrom / 10.0  # OpenMM positions are in nm

    # `positions` (e.g. fixer.positions) is a unit-wrapped Quantity, not a
    # plain list of Vec3 - indexing it still returns Quantity objects, which
    # don't expose .x/.y/.z. Strip units once up front to get plain Vec3s.
    positions_nm = positions.value_in_unit(nanometers)

    sg_atoms = []
    for atom in topology.atoms():
        if atom.residue.name in ("CYS", "CYX") and atom.name == "SG":
            sg_atoms.append(atom)

    pairs = []
    for i in range(len(sg_atoms)):
        for j in range(i + 1, len(sg_atoms)):
            a1, a2 = sg_atoms[i], sg_atoms[j]
            p1 = positions_nm[a1.index]
            p2 = positions_nm[a2.index]
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dz = p1.z - p2.z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist <= cutoff_nm:
                pairs.append(
                    (
                        (a1.residue.chain.id, a1.residue.id),
                        (a2.residue.chain.id, a2.residue.id),
                    )
                )
    return pairs


if __name__ == "__main__":
    result = fix_pdb("prot.pdb", ph=7.4)
    print("Output:", result["output_pdb"])
    print("Missing residues found:", result["n_missing_residues"])
    print("Missing atoms added:", result["n_missing_atoms_added"])
    print("Chains kept:", result["chains_kept"])
    print("Candidate disulfides:", result["disulfide_pairs"])
