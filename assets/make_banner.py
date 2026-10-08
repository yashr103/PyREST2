"""Builds the repository header: app mark, wordmark and the ViStA Lab logo.

Run from the repository root:  python assets/make_banner.py
Sources are the two logos the application itself ships with, i.png (app) and
v.png (lab), so the repository header and the running app always agree.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP_LOGO = os.path.join(ROOT, "i.png")
LAB_LOGO = os.path.join(ROOT, "v.png")

NAVY, INK, GREY, WHITE = "#101A33", "#101828", "#5A6478", "#FFFFFF"
W, H = 1200, 300

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})


def trimmed(path):
    """Crop a transparent PNG to its visible content."""
    im = Image.open(path).convert("RGBA")
    box = im.split()[-1].getbbox()
    return im.crop(box) if box else im


def place(ax, im, cx, cy, height, z=6):
    """Draw `im` centred on (cx, cy) at the given pixel height."""
    w = height * im.width / im.height
    ax.imshow(im, extent=(cx - w / 2, cx + w / 2, cy - height / 2,
                          cy + height / 2), zorder=z, interpolation="antialiased")
    return w


def build():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.patch.set_alpha(0.0)

    r = 34
    ax.add_patch(FancyBboxPatch((2, 2), W - 4, H - 4,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=WHITE, edgecolor="#E6E9EF", lw=2, zorder=1))
    # dark square tile on the left, carrying the app mark
    ax.add_patch(FancyBboxPatch((2, 2), H - 4, H - 4,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=NAVY, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((H / 2, 2), H / 2 - 2, H - 4, facecolor=NAVY,
                           edgecolor="none", zorder=2))
    place(ax, trimmed(APP_LOGO), H / 2, H / 2, 150)

    ax.text(H + 54, H / 2 + 26, "PyREST2", fontsize=52, fontweight="bold",
            color=INK, ha="left", va="center", zorder=5)
    ax.text(H + 58, H / 2 - 40, "replica exchange with solute tempering",
            fontsize=18, color=GREY, ha="left", va="center", zorder=5)
    ax.text(H + 58, H / 2 - 74, "a desktop application for OpenMM",
            fontsize=18, color=GREY, ha="left", va="center", zorder=5)

    # lab logo on the right, with a hairline separating it from the wordmark
    ax.plot([900, 900], [70, 230], color="#E6E9EF", lw=2, zorder=4)
    lab = trimmed(LAB_LOGO)
    place(ax, lab, 1038, 140, 62)
    ax.text(1038, 196, "developed at", fontsize=13, color=GREY, ha="center",
            va="center", zorder=5)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"banner.{ext}"), dpi=300, transparent=True)

    # stand-alone marks, for the README and for anyone reusing them.
    # Downscaled: the full-resolution originals stay in the repository root.
    def shrink(im, height):
        w = max(1, round(height * im.width / im.height))
        return im.resize((w, height), Image.LANCZOS)

    shrink(trimmed(APP_LOGO), 512).save(os.path.join(HERE, "logo_app.png"))
    shrink(lab, 220).save(os.path.join(HERE, "logo_vista.png"))
    print("wrote banner.png/.svg/.pdf, logo_app.png, logo_vista.png ->", HERE)


if __name__ == "__main__":
    build()
