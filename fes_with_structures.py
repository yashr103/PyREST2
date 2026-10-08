"""Standalone FES + representative-structure figure for REST2 output.

Output: fes_with_structures.png (or the path given with --output)
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc4
import mdtraj as md
from scipy.ndimage import gaussian_filter

KB_KJ_PER_MOL_K = 0.0083144621  # kJ/mol/K

BG = "white"
PANEL = "white"
GRID = "#b0b0b0"
TEXT = "#1a1a1a"
ACCENT = "#1f77b4"

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


def _get_nc_var(ds, name):
    if name in ds.variables:
        return ds.variables[name]
    for grp in ds.groups.values():
        if name in grp.variables:
            return grp.variables[name]
    return None


def load_state0_trajectory(
    nc_file,
    nc_checkpoint,
    topology,
    checkpoint_interval,
    timestep,
    steps_per_iter,
    stride,
):
    with nc4.Dataset(nc_file, "r") as ds:
        sv = _get_nc_var(ds, "states")
        if sv is None:
            raise RuntimeError(f"'states' variable not found in {nc_file}")
        replica_states = np.array(sv[:])
        n_iter, n_rep = replica_states.shape
        print(f"  Parsed steps : {n_iter} exchange iterations over {n_rep} states.")

        rst_int = replica_states.astype(int)
        if rst_int.size and (rst_int.min() < 0 or rst_int.max() >= n_rep):
            print(f"  WARNING: 'states' values outside [0, {n_rep - 1}] - clipping.")
            rst_int = np.clip(rst_int, 0, n_rep - 1)
        state_replicas = np.zeros_like(rst_int)
        iter_idx = np.arange(n_iter)[:, None]
        replica_idx = np.arange(n_rep)[None, :]
        state_replicas[iter_idx, rst_int] = replica_idx

    top = (
        md.load_prmtop(topology)
        if topology.endswith(".prmtop")
        else md.load_topology(topology)
    )

    with nc4.Dataset(nc_checkpoint, "r") as ds:
        pos_var = ds.variables["positions"]
        n_chk, n_rep_chk, n_atoms, _ = pos_var.shape
        if top.n_atoms != n_atoms:
            raise RuntimeError(
                f"Topology has {top.n_atoms} atoms but checkpoint positions have "
                f"{n_atoms} atoms."
            )
        print(f"  Checkpoint frames: {n_chk}, {n_rep_chk} replicas, {n_atoms} atoms.")
        box_var = ds.variables["box_vectors"] if "box_vectors" in ds.variables else None

        frame_indices = list(range(0, n_chk, stride))
        coords = []
        boxes = [] if box_var is not None else None
        for k in frame_indices:
            t = min(k * checkpoint_interval, n_iter - 1)
            r = int(state_replicas[t, 0])
            r = max(0, min(r, n_rep_chk - 1))
            coords.append(np.asarray(pos_var[k, r, :, :], dtype=np.float64))
            if box_var is not None:
                boxes.append(np.asarray(box_var[k, r, :, :], dtype=np.float64))

    time_per_frame = checkpoint_interval * steps_per_iter * timestep / 1e6  # ns
    time_arr = np.arange(len(coords)) * time_per_frame
    traj = md.Trajectory(
        xyz=np.array(coords, dtype=np.float64), topology=top, time=time_arr
    )
    if boxes:
        traj.unitcell_vectors = np.array(boxes, dtype=np.float64)

    print(f"  State 0 trajectory: {len(traj)} frames, {time_arr[-1]:.2f} ns")
    return traj


def _ca_residues(t):
    return [(a.residue.name, a.residue.index) for a in t.atoms if a.name == "CA"]


def _state0_frame0(traj):
    return traj.slice(0).slice(slice(None), copy=True)


def compute_cvs(traj, ref_pdb, image_molecules=False):
    """Backbone RMSD (vs a validated reference) and protein Rg.

    The RMSD reference PDB is only trusted if its atom count AND its
    backbone residue order match the trajectory topology; otherwise the
    first state-0 trajectory frame is used as the reference. A
    topology/order mismatch (common when the prmtop was regenerated from
    the PDB, e.g. tleap reordering/renumbering) silently scrambles the
    atom correspondence and produces nonsensical per-frame RMSD values.
    Superposition is applied on a COPY so the trajectory used later for
    PDB export / Rg is never mutated in place.
    """
    if image_molecules:
        try:
            traj = traj.image_molecules(inplace=False)
            print("  PBC imaging applied before computing CVs.")
        except Exception as img_err:
            print(f"  WARNING: PBC imaging failed ({img_err}) - using raw coords.")

    bb = traj.topology.select("backbone")
    prot = traj.topology.select("protein")

    ref = None
    if ref_pdb and os.path.exists(ref_pdb):
        cand = md.load(ref_pdb)
        if cand.n_atoms != traj.n_atoms:
            print(
                f"  WARNING: reference '{ref_pdb}' has {cand.n_atoms} atoms vs "
                f"trajectory {traj.n_atoms} - using the first trajectory frame as "
                f"the RMSD reference instead."
            )
        elif _ca_residues(cand) != _ca_residues(traj):
            print(
                f"  WARNING: reference '{ref_pdb}' has a different backbone residue "
                f"order than the trajectory - using the first trajectory frame as "
                f"the RMSD reference instead."
            )
        else:
            ref = cand

    if ref is None:
        ref = _state0_frame0(traj)
        print("  Reference for RMSD: first frame of the state-0 trajectory.")

    traj_bb = traj.atom_slice(bb).slice(slice(None), copy=True)
    ref_bb = ref.atom_slice(bb).slice(slice(None), copy=True)
    traj_bb.superpose(ref_bb)
    rmsd = md.rmsd(traj_bb, ref_bb) * 10.0  # nm -> A

    if prot.size:
        rg = md.compute_rg(traj.atom_slice(prot)) * 10.0
    else:
        rg = md.compute_rg(traj) * 10.0

    print(
        f"  CV ranges: RMSD {rmsd.min():.2f}-{rmsd.max():.2f} A, "
        f"Rg {rg.min():.2f}-{rg.max():.2f} A"
    )
    return rmsd, rg


def build_fes(
    rmsd, rg, t_min, bins=45, sigma=1.1, pad_frac=0.05, range_percentiles=(0.5, 99.5)
):
    """Build the 2D RMSD-Rg free-energy histogram.

    Bin edges are taken from percentiles of the data (default 0.5-99.5%)
    rather than raw min/max. A handful of outlier frames (e.g. early,
    not-yet-equilibrated frames sitting very close to the reference
    structure) can otherwise stretch histogram2d's default range out far
    beyond where the trajectory actually spends its time, squashing the
    populated basin into a thin sliver at one edge of the plot instead of
    filling the frame. A small padding fraction is added back on top so
    the basin isn't cropped right at the plot edge.
    """
    lo_pct, hi_pct = range_percentiles
    x_lo, x_hi = np.percentile(rmsd, [lo_pct, hi_pct])
    y_lo, y_hi = np.percentile(rg, [lo_pct, hi_pct])
    x_pad = (x_hi - x_lo) * pad_frac
    y_pad = (y_hi - y_lo) * pad_frac
    hist_range = [[x_lo - x_pad, x_hi + x_pad], [y_lo - y_pad, y_hi + y_pad]]

    H, xe, ye = np.histogram2d(rmsd, rg, bins=bins, range=hist_range)
    G = -KB_KJ_PER_MOL_K * t_min * np.log(H.T / H.max() + 1e-5)
    finite = G[np.isfinite(G)]
    G = np.where(np.isinf(G), finite.max() if finite.size else 0.0, G)
    G = gaussian_filter(G, sigma=sigma)
    return G, xe, ye


def select_structures(G, xe, ye, rmsd, rg, n_structures=10):
    """Pick `n_structures` frames whose (RMSD, Rg) points span the full
    free-energy range, ordered HIGHEST energy first -> LOWEST (global
    minimum) last. Each frame's energy is looked up from the FES grid."""
    gi = np.clip(np.searchsorted(ye, rg, side="right") - 1, 0, G.shape[0] - 1)
    gj = np.clip(np.searchsorted(xe, rmsd, side="right") - 1, 0, G.shape[1] - 1)
    frame_energy = G[gi, gj]

    order = np.argsort(frame_energy)[::-1]  # highest energy first
    n_frames = len(order)
    positions = np.unique(np.linspace(0, n_frames - 1, n_structures).astype(int))
    selected = order[positions]

    energies = frame_energy[selected]
    print(f"  Selected {len(selected)} structures (high -> low energy):")
    for k, e in zip(selected, energies):
        print(
            f"    frame {k:6d}   E={e:8.2f} kJ/mol   "
            f"RMSD={rmsd[k]:6.2f} A   Rg={rg[k]:6.2f} A"
        )
    return selected, energies


def export_pdbs(traj, selected, energies, rmsd, rg, export_dir, prefix="structure"):
    """Write each selected frame to an individual PDB (for PyMOL etc.) plus
    a CSV mapping label -> frame -> energy/RMSD/Rg."""
    os.makedirs(export_dir, exist_ok=True)
    rows = []
    info_path = os.path.join(export_dir, "structures_info.csv")
    with open(info_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "frame",
                "filename",
                "E_kJmol",
                "E_rel_min_kJmol",
                "RMSD_A",
                "Rg_A",
            ],
        )
        writer.writeheader()
        emin = energies.min()
        for idx, fr in enumerate(selected):
            name = f"{prefix}_{idx + 1:02d}.pdb"
            path = os.path.join(export_dir, name)
            traj[int(fr)].save_pdb(path)
            row = {
                "label": idx + 1,
                "frame": int(fr),
                "filename": name,
                "E_kJmol": round(float(energies[idx]), 3),
                "E_rel_min_kJmol": round(float(energies[idx] - emin), 3),
                "RMSD_A": round(float(rmsd[fr]), 3),
                "Rg_A": round(float(rg[fr]), 3),
            }
            writer.writerow(row)
            rows.append(row)
    for r in rows:
        print(
            f"    -> {os.path.join(export_dir, r['filename'])}  "
            f"(frame {r['frame']}, E_rel {r['E_rel_min_kJmol']:.1f} kJ/mol)"
        )
    print(f"    -> {info_path}")
    return info_path


def main():
    parser = argparse.ArgumentParser(
        description="FES (RMSD vs Rg) with representative protein structures "
        "extracted from the state-0 trajectory."
    )
    parser.add_argument("--input-dir", default="rest2_output")
    parser.add_argument("--nc-file", default=None, help="Overrides input-dir globbing")
    parser.add_argument("--nc-checkpoint", default=None)
    parser.add_argument("--topology", default="complex.prmtop")
    parser.add_argument("--ref-pdb", default="complex.pdb")
    parser.add_argument("--n-replicas", type=int, default=16)
    parser.add_argument("--t-min", type=float, default=300.0)
    parser.add_argument("--timestep", type=float, default=2.0, help="fs")
    parser.add_argument("--steps-per-iter", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-molecules", action="store_true",
                        help="Apply mdtraj image_molecules() (needs box vectors) "
                             "before computing RMSD/Rg")
    parser.add_argument("--ligand-sel", default="resname MOL")
    parser.add_argument("--output", default="fes_with_structures.png")
    parser.add_argument("--bins", type=int, default=45)
    parser.add_argument("--sigma", type=float, default=1.1)
    parser.add_argument(
        "--n-structures",
        type=int,
        default=10,
        help="Number of representative structures to draw, "
        "spanning highest to lowest free energy",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="If set, write each selected structure frame to its "
        "own PDB in this directory (for e.g. PyMOL rendering)",
    )
    parser.add_argument(
        "--prefix",
        default="structure",
        help="PDB filename prefix used with --export-dir",
    )
    args = parser.parse_args()

    nc_file = args.nc_file
    if nc_file is None:
        candidates = glob.glob(os.path.join(args.input_dir, "*.nc"))
        candidates = [
            f for f in candidates if "checkpoint" not in os.path.basename(f).lower()
        ]
        nc_file = candidates[0] if candidates else None
    if nc_file is None:
        sys.exit("ERROR: no main NetCDF file found. Pass --nc-file explicitly.")

    nc_checkpoint = args.nc_checkpoint
    if nc_checkpoint is None:
        candidates = glob.glob(os.path.join(args.input_dir, "*checkpoint*.nc"))
        nc_checkpoint = candidates[0] if candidates else None
    if nc_checkpoint is None:
        sys.exit(
            "ERROR: no checkpoint NetCDF file found. Pass --nc-checkpoint explicitly."
        )

    print(f"  Target File     : {nc_file}")
    print(f"  Checkpoint File : {nc_checkpoint}")

    traj = load_state0_trajectory(
        nc_file,
        nc_checkpoint,
        args.topology,
        args.checkpoint_interval,
        args.timestep,
        args.steps_per_iter,
        args.stride,
    )
    rmsd, rg = compute_cvs(traj, args.ref_pdb, image_molecules=args.image_molecules)
    G, xe, ye = build_fes(rmsd, rg, args.t_min, bins=args.bins, sigma=args.sigma)

    selected, energies = select_structures(
        G, xe, ye, rmsd, rg, n_structures=args.n_structures
    )

    if args.export_dir:
        export_pdbs(
            traj, selected, energies, rmsd, rg, args.export_dir, prefix=args.prefix
        )

    # Inferno pixel-grid theme: black/purple = low (favorable) free energy,
 
    cmap = plt.get_cmap("inferno")
    vmin, vmax = G.min(), G.max()

    fig = plt.figure(figsize=(11, 8))
    fes_ax = fig.add_axes([0.08, 0.11, 0.52, 0.77])
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    im = fes_ax.imshow(
        G,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    fes_ax.grid(False)

    cbar = fig.colorbar(im, ax=fes_ax, shrink=0.75, aspect=35, pad=0.02)
    cbar.set_label("Gibbs Free Energy (kJ/mol)")
    fes_ax.set_xlabel("Backbone RMSD (\u00c5)")
    fes_ax.set_ylabel("Radius of Gyration Rg (\u00c5)")
    fes_ax.set_title(f"Ground Ensemble FES ({args.t_min:.0f} K)")

    n_sel = len(selected)
    if n_sel:
        emin = energies.min()
        legend_lines = [
            f"{idx + 1}   {energies[idx] - emin:5.2f} kJ/mol" for idx in range(n_sel)
        ]
        legend_body = "\n".join(legend_lines)
        fig.text(
            0.70,
            0.5,
            "Structure   \u0394E (kJ/mol)\n" + legend_body,
            ha="left",
            va="center",
            color=TEXT,
            linespacing=1.7,
            zorder=30,
        )

        for idx, fr in enumerate(selected):
            e = energies[idx]
            color = cmap((e - vmin) / (vmax - vmin))
            rx, ry = rmsd[fr], rg[fr]
            if not (xe[0] <= rx <= xe[-1] and ye[0] <= ry <= ye[-1]):
                print(
                    f"  NOTE: structure {idx + 1} (frame {fr}) at RMSD={rx:.2f} A, "
                    f"Rg={ry:.2f} A falls outside the plotted axis range "
                    f"(RMSD [{xe[0]:.2f}, {xe[-1]:.2f}], Rg [{ye[0]:.2f}, {ye[-1]:.2f}]) "
                    f"- marker will render pinned to the plot edge."
                )
            fes_ax.scatter(
                rx,
                ry,
                s=130,
                marker="o",
                facecolor="white",
                edgecolor=color,
                linewidths=1.6,
                zorder=25,
            )
            fes_ax.text(
                rx,
                ry,
                str(idx + 1),
                color=ACCENT,
                ha="center",
                va="center",
                fontsize=8,
                zorder=26,
            )
    else:
        print("  WARNING: no structures selected - plotting FES without structures.")

    # Lock the view to the histogram extent. Without this, scatter() will
    # auto-expand the axes to fit every marker plotted on it - if a
    # selected structure's (RMSD, Rg) point falls outside the percentile-
    # clipped FES range (an outlier frame), the view stretches to include
    # it, leaving a large empty band around the actual heatmap.
    fes_ax.set_xlim(xe[0], xe[-1])
    fes_ax.set_ylim(ye[0], ye[-1])

    # Force all four spines visible (top/right can otherwise be dropped
    # depending on how the axes was constructed/backend renders it) so the
    # FES panel gets a full box border, matching the rest of the figure
    # styling instead of only left/bottom lines.
    for side in ("top", "right", "left", "bottom"):
        fes_ax.spines[side].set_visible(True)
        fes_ax.spines[side].set_color(GRID)
        fes_ax.spines[side].set_linewidth(0.9)

    out_path = args.output
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
