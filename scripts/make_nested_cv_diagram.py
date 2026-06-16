"""Generate paper/figures/nested_cv_splits.png.

A static schematic of the repeated nested cross-validation with *visit-grouped*
splits used in Section "Regression Modelling". The unit of splitting is one visit
(a contiguous 60-second RPM hold); all sweeps within a hold are kept together so
that near-duplicate windows never straddle a train/validation boundary. Each cell
below is one visit; the thin ticks inside a cell are its individual sweeps.

This figure is illustrative (data-independent). Re-run after changing the CV
design constants (R repeats, k folds) below to match config.yaml / the macros.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "paper" / "figures" / "nested_cv_splits.png"

DPI = 150
R_REPEATS = 5   # config.yaml modelling.n_repeats  (\resNrepeats)
K_FOLDS = 5     # config.yaml modelling.cv_n_splits (\resNfoldsCv)

plt.rcParams.update({"font.size": 11})

# Palette (matched to the previous schematic)
TEST = dict(fc="#fdeee2", ec="#c15a2a", tc="#8a3a12")    # hold-out test / outer validation
POOL = dict(fc="#e8e8f8", ec="#5a5aa0", tc="#3b3b8f")     # training pool / inner pool
IVAL = dict(fc="#d9efde", ec="#2e7d4f", tc="#1f5a38")     # inner validation
ITRN = dict(fc="#e6f1fb", ec="#3a6a9a", tc="#28557e")     # inner training

fig, ax = plt.subplots(figsize=(11.5, 7.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.invert_yaxis()  # draw top-to-bottom


def cell(x, y, w, h, style, ticks=3, lw=1.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
        fc=style["fc"], ec=style["ec"], lw=lw, mutation_aspect=0.5, zorder=2))
    for i in range(ticks):
        tx = x + w * (i + 1) / (ticks + 1)
        ax.plot([tx, tx], [y + h * 0.2, y + h * 0.8],
                color=style["ec"], lw=0.6, alpha=0.5, zorder=3)


def bar(x0, y, total_w, h, assignments, ticks=3, gap=0.5):
    n = len(assignments)
    cw = (total_w - gap * (n - 1)) / n
    for i, st in enumerate(assignments):
        cell(x0 + i * (cw + gap), y, cw, h, st, ticks=ticks)


def section(label, y):
    ax.text(2, y, label, ha="left", va="center", fontsize=12.5, fontweight="bold")


def rlabel(x, y, txt):
    ax.text(x, y, txt, ha="left", va="center", fontsize=10.5, color="#444444")


def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=14, color="#666666", lw=1.4))


X0, W, H = 8, 78, 6.5
N = 12  # number of visit groups drawn

# Title + note
ax.text(50, 3, "Repeated nested cross-validation with visit-grouped splits",
        ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(50, 8.5, "unit of splitting = one visit (contiguous 60-s RPM hold); "
        "all sweeps in a hold stay together", ha="center", va="center",
        fontsize=10, color="#555555", style="italic")

# --- Section A: group-level hold-out split ---
section("Visit-grouped hold-out split  (GroupShuffleSplit)", 15)
assign_a = [POOL] * N
for i in (3, 8):  # two whole visits held out
    assign_a[i] = TEST
bar(X0, 19, W, H, assign_a)
rlabel(X0 + W + 2, 22.3, "Train 80%")
rlabel(X0 + W + 2, 25.0, "Test 20%")
ax.text(50, 30.5, "whole holds assigned together — sweeps never straddle the split",
        ha="center", va="center", fontsize=9.5, color="#777777")
arrow(50, 33, 37.5)

# --- Section B: outer folds on the training visits ---
section(f"Outer folds  (group {K_FOLDS}-fold, × R={R_REPEATS} re-seedings)", 40)
train_idx = [i for i in range(N) if assign_a[i] is POOL]  # 10 training visits
nb = len(train_idx)
# fold 1: first 2 visits validation; fold 2: next 2
folds = [
    (43.5, [0, 1], "Fold 1"),
    (51.5, [2, 3], "Fold 2"),
]
for yy, val_cells, lab in folds:
    assign_b = [POOL] * nb
    for c in val_cells:
        assign_b[c] = TEST
    bar(X0, yy, W, H, assign_b)
    rlabel(X0 + W + 2, yy + H / 2, lab)
ax.text(X0 + W * 0.18, 59.5, "…", ha="center", va="center", fontsize=16)
rlabel(X0 + W + 2, 59.5, f"Folds 3–{K_FOLDS}")
arrow(50, 62.5, 67)

# --- Section C: inner folds (example: outer fold 1 inner pool) ---
section("Inner folds  (example: outer fold 1 inner pool)", 69.5)
ni = nb - 2  # inner pool size for fold 1
inner = [
    (73, [0, 1], "Inner fold 1"),
    (81, [2, 3], "Inner fold 2"),
]
XI = X0 + W * 0.16
WI = W * 0.84
for yy, val_cells, lab in inner:
    assign_c = [ITRN] * ni
    for c in val_cells:
        assign_c[c] = IVAL
    bar(XI, yy, WI, H, assign_c)
    rlabel(XI + WI + 2, yy + H / 2, lab)
ax.text(XI + WI * 0.2, 89, "…", ha="center", va="center", fontsize=16)
rlabel(XI + WI + 2, 89, f"Inner folds 3–{K_FOLDS}")

# --- Legend (evenly spaced across the width) ---
leg = [("Hold-out test", TEST), ("Inner pool", POOL),
       ("Inner validation", IVAL), ("Inner training", ITRN)]
for (name, st), lx in zip(leg, (6, 30, 52, 76)):
    ax.add_patch(FancyBboxPatch((lx, 95), 3.0, 3.0,
                 boxstyle="round,pad=0,rounding_size=0.6",
                 fc=st["fc"], ec=st["ec"], lw=1.3, mutation_aspect=0.5))
    ax.text(lx + 4.0, 96.5, name, ha="left", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")