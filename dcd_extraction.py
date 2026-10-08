"""
Standalone DCD extraction from an openmmtools REMD .nc storage file
=====================================================================
Run this AT ANY TIME - during a run, after a crash, or after a clean
completion. It only reads whatever checkpoints currently exist in the
.nc file; it does not require the simulation to have finished.

Usage
-----
    python dcd_extraction.py

Edit the CONFIG block below to match your run.

Requires: openmmtools, mdtraj, numpy
    pip install mdtraj   (openmmtools should already be installed)
"""

import os

# Avoid HDF5 file-lock errors if the .nc file was left open by a crashed
# process - read-only access doesn't need exclusive locking.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
import numpy as np

try:
    import mdtraj as md
except ImportError:
    print("ERROR: mdtraj not installed.\n  pip install mdtraj")
    sys.exit(1)

try:
    from openmmtools.multistate import MultiStateReporter
except ImportError:
    print("ERROR: openmmtools not installed.")
    sys.exit(1)

from openmm.unit import nanometers

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG - edit these
# ═══════════════════════════════════════════════════════════════════════════

NC_PATH = "rest2_output/rest2_remd.nc"  # path to the .nc storage file
TOPOLOGY_FILE = "complex.prmtop"  # Amber topology
DCD_DIR = "rest2_output/dcd"  # output directory for DCDs

# Atom selection (MDTraj selection syntax). "not water" = solute only.
ATOM_SELECTION = "not water"

# Extract every Nth available checkpoint (1 = every checkpoint).
DCD_INTERVAL = 1

# Write a DCD for every replica (replica_0.dcd ... replica_N.dcd)?
# If False, only replica_0.dcd (T_MIN, unscaled - recommended for analysis)
# is written.
WRITE_ALL_REPLICAS = True

# Write state0_demuxed.dcd - follows whichever physical replica currently
# occupies thermodynamic state 0 (T_MIN) at each checkpoint. Usually the
# best trajectory for analysis since it pools data from all replicas that
# visit state 0.
WRITE_DEMUXED_DCD = True


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════


def main():
    if not os.path.exists(NC_PATH):
        print(f"ERROR: {NC_PATH} not found.")
        sys.exit(1)

    os.makedirs(DCD_DIR, exist_ok=True)

    print("=" * 65)
    print("DCD EXTRACTION (standalone)")
    print("=" * 65)

    reporter = MultiStateReporter(NC_PATH, open_mode="r")
    try:
        _extract(reporter)
    finally:
        reporter.close()


def _extract(reporter):
    # ── Basic info ────────────────────────────────────────────────────────
    n_replicas = reporter.n_replicas
    checkpoint_interval = reporter.checkpoint_interval
    last_iter_total = reporter.read_last_iteration(last_checkpoint=False)
    last_iter_ckpt = reporter.read_last_iteration(last_checkpoint=True)

    print(f"  .nc file            : {NC_PATH}")
    print(f"  Replicas            : {n_replicas}")
    print(f"  Checkpoint interval : {checkpoint_interval}")
    print(
        f"  Last iteration      : {last_iter_total}  "
        f"(last checkpoint: {last_iter_ckpt})"
    )

    # ── Available checkpoint iterations ─────────────────────────────────────
    checkpoint_iters = reporter.read_checkpoint_iterations()
    print(
        f"  Checkpoints found   : {len(checkpoint_iters)}  "
        f"-> iterations {list(checkpoint_iters[:5])}"
        f"{' ...' if len(checkpoint_iters) > 5 else ''}"
    )

    if len(checkpoint_iters) == 0:
        print("  No checkpoints available yet - nothing to extract.")
        return

    frames_to_read = checkpoint_iters[:: max(1, DCD_INTERVAL)]
    n_frames = len(frames_to_read)
    print(
        f"  DCD_INTERVAL        : {DCD_INTERVAL}  -> {n_frames} frames will be written"
    )

    # ── Topology + atom selection ───────────────────────────────────────────
    traj_top = md.load_topology(TOPOLOGY_FILE)
    try:
        sel_indices = traj_top.select(ATOM_SELECTION)
        if len(sel_indices) == 0:
            print(
                f"  WARNING: selection '{ATOM_SELECTION}' matched 0 atoms, using 'all'"
            )
            sel_indices = traj_top.select("all")
    except Exception as e:
        print(f"  WARNING: selection error ({e}), using 'all'")
        sel_indices = traj_top.select("all")

    print(f"  Atom selection      : '{ATOM_SELECTION}'  ({len(sel_indices)} atoms)")
    sub_top = traj_top.subset(sel_indices)

    # ── Read frames ──────────────────────────────────────────────────────────
    n_sel = len(sel_indices)
    pos_arr = np.zeros((n_frames, n_replicas, n_sel, 3), dtype=np.float32)
    box_arr = np.zeros((n_frames, n_replicas, 3, 3), dtype=np.float32)
    replica_index_arr = np.zeros((n_frames, n_replicas), dtype=np.int32)

    print("\n  Reading frames ...")
    for fi, it in enumerate(frames_to_read):
        sampler_states = reporter.read_sampler_states(iteration=int(it))
        if sampler_states is None:
            print(f"    iteration {it}: no positions stored, skipping")
            continue

        for rep_i, ss in enumerate(sampler_states):
            pos_arr[fi, rep_i] = ss.positions.value_in_unit(nanometers)[sel_indices]
            box_arr[fi, rep_i] = np.array(ss.box_vectors.value_in_unit(nanometers))

        # replica_indices[rep] = state index that replica `rep` currently occupies.
        # np.argsort() of this permutation gives its inverse: state_to_rep[state].
        replica_indices = reporter.read_replica_thermodynamic_states(iteration=int(it))
        replica_index_arr[fi] = np.argsort(replica_indices)

        print(f"    {fi+1}/{n_frames}  (iteration {it})")

    print("  Done reading.")

    # ── Write per-replica DCDs ────────────────────────────────────────────
    replicas_to_write = range(n_replicas) if WRITE_ALL_REPLICAS else [0]
    for rep_i in replicas_to_write:
        dcd_path = os.path.join(DCD_DIR, f"replica_{rep_i}.dcd")
        traj = md.Trajectory(
            xyz=pos_arr[:, rep_i, :, :],
            topology=sub_top,
            unitcell_vectors=box_arr[:, rep_i, :, :],
        )
        traj.save_dcd(dcd_path)
        label = "  <- T_MIN (unscaled, use for analysis)" if rep_i == 0 else ""
        print(f"  wrote {dcd_path}  [{n_frames} frames]{label}")

    # ── Demuxed state-0 DCD ──────────────────────────────────────────────
    if WRITE_DEMUXED_DCD:
        pos_demux = np.zeros((n_frames, n_sel, 3), dtype=np.float32)
        box_demux = np.zeros((n_frames, 3, 3), dtype=np.float32)
        for fi in range(n_frames):
            rep0 = replica_index_arr[fi, 0]
            pos_demux[fi] = pos_arr[fi, rep0]
            box_demux[fi] = box_arr[fi, rep0]

        traj_demux = md.Trajectory(
            xyz=pos_demux, topology=sub_top, unitcell_vectors=box_demux
        )
        dcd_path = os.path.join(DCD_DIR, "state0_demuxed.dcd")
        traj_demux.save_dcd(dcd_path)
        print(f"  wrote {dcd_path}  [{n_frames} frames]  <- demuxed T_MIN ensemble")

    print(f"\nDone. DCDs are in {DCD_DIR}/")


if __name__ == "__main__":
    main()
