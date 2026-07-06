"""
18_inspect_results.py
=====================
Inspection figures for the new pipeline:

  1. P-value tables (rendered):  complementarity and full-vs-selected
     Diebold-Mariano results from 13_group_paired_tests.py, as images with
     significant group-level p-values highlighted.
  2. Grouping / fold visualisation:  reproduces the exact grouping used by
     11_featureset_comparison.py (contiguous acquisition holds -> operating-
     point twin merge -> GroupKFold) and shows
       (a) operating-point space (RPM vs temperature) coloured by fold,
       (b) chronological sweep order vs RPM coloured by fold (twins/up-down
           sweeps share a fold),
       (c) groups-per-fold and sweeps-per-fold summary.

Outputs (outputs/18_inspect_results/)
  pvalues_complementarity.png, pvalues_full_vs_selected.png
  grouping_folds.png

Usage
-----
    python scripts/18_inspect_results.py
    python scripts/18_inspect_results.py --n-folds 5
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GroupKFold

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.grouping import derive_hold_groups, merge_twin_groups

_p = argparse.ArgumentParser()
_p.add_argument("--config", type=str, default=None)
_p.add_argument("--n-folds", type=int, default=None)
args, _ = _p.parse_known_args()

cfg = load_config(args.config)
OUTPUT_DIR = get_output_dir(cfg)
NEW_DIR = OUTPUT_DIR / "new"
TESTS_DIR = NEW_DIR / "group_paired_tests"
SCRIPT_DIR = NEW_DIR / "inspect_results"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

D_PW = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RPM_MIN = cfg["filters"].get("rpm_min", 0.0)  # drop startup/standstill transients
TEMP_MIN = cfg["filters"].get("temp_min", None)  # drop sub-floor cold-start
N_FOLDS = args.n_folds or cfg.get("modelling", {}).get("cv_n_splits", 5)
DPI = 150


# %%
# =============================================================================
# 1. P-value tables (rendered images)
# =============================================================================
def _render_pvalues(csv_name, keep_cols, fname, title):
    fp = TESTS_DIR / csv_name
    if not fp.exists():
        print(f"  (skip) {csv_name} not found — run 13_group_paired_tests.py")
        return
    df = pd.read_csv(fp)
    cols = [c for c in keep_cols if c in df.columns]
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.4g}")
    fig, ax = plt.subplots(figsize=(1.55 * len(cols), 0.45 * len(d) + 1.1))
    ax.axis("off")
    t = ax.table(cellText=d.values, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.4)
    for j in range(len(cols)):
        t[(0, j)].set_facecolor("#1f4e79")
        t[(0, j)].set_text_props(color="white", weight="bold")
    # highlight significant group-level p
    if "p_group" in cols:
        pj = cols.index("p_group")
        for i, v in enumerate(df["p_group"].values, start=1):
            t[(i, pj)].set_facecolor("#c7e9c0" if v < 0.05 else "#fdd0a2")
    ax.set_title(title, pad=12)
    fig.savefig(SCRIPT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


_render_pvalues(
    "complementarity_tests.csv",
    ["model", "contrast", "n_groups",
     "mean_dMSE(A-B)", "dm_stat", "p_group", "p_window_naive", "better"],
    "pvalues_complementarity.png",
    "Complementarity — group-paired Diebold-Mariano (green: p_group < 0.05)")


# %%
# =============================================================================
# 2. Grouping / fold visualisation (reproduces 11's grouping exactly)
# =============================================================================
_raw_feat, _raw_meta = load_parquet_pair(NEW_DIR)
_, metadata = filter_by_metadata(_raw_feat, _raw_meta, rpm_max=RPM_MAX,
                                 rpm_min=RPM_MIN, temp_min=TEMP_MIN)
metadata = metadata.reset_index(drop=True)
metadata["kappa"] = metadata.apply(
    lambda r: calculate_kappa(rpm=r["rpm"], temp_c=r["temperature_c"], d_pw=D_PW,
                              nu_40=r["viscosity_40c_cst"], nu_100=r["viscosity_100c_cst"]),
    axis=1)

sweeps = metadata.drop_duplicates(["file", "sweep"]).reset_index(drop=True)
base = derive_hold_groups(sweeps)
merged = merge_twin_groups(sweeps, base, rpm_bin_width=100.0, temp_bin_width=1.0)

gkf = GroupKFold(n_splits=N_FOLDS)
fold = np.empty(len(sweeps), dtype=int)
for fi, (_, te) in enumerate(gkf.split(np.arange(len(sweeps)), groups=merged)):
    fold[te] = fi
sweeps["group"] = merged
sweeps["fold"] = fold

# chronological rank
sweep_no = sweeps["sweep"].str.split("_").str[1].astype(int).values
order = np.lexsort((sweep_no, sweeps["file"].values))
chron = np.empty(len(sweeps), dtype=int); chron[order] = np.arange(len(sweeps))
sweeps["chron"] = chron

n_hold, n_merged = len(np.unique(base)), len(np.unique(merged))
print(f"{n_hold} acquisition holds -> {n_merged} operating-point groups "
      f"({n_hold - n_merged} merged as twins) over {len(sweeps)} sweeps")

cmap = ListedColormap(plt.colormaps["tab10"].colors[:N_FOLDS])
fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 0.8])

# (a) operating-point space coloured by fold
axA = fig.add_subplot(gs[0])
sc = axA.scatter(sweeps["rpm"], sweeps["temperature_c"], c=sweeps["fold"],
                 cmap=cmap, s=10, alpha=0.7, vmin=-0.5, vmax=N_FOLDS - 0.5)
axA.set_xlabel("RPM"); axA.set_ylabel("temperature [°C]")
axA.set_title("(a) Operating-point space by fold")
cb = fig.colorbar(sc, ax=axA, ticks=range(N_FOLDS)); cb.set_label("fold")

# (b) chronological order vs RPM coloured by fold
axB = fig.add_subplot(gs[1])
axB.scatter(sweeps["chron"], sweeps["rpm"], c=sweeps["fold"], cmap=cmap,
            s=8, alpha=0.7, vmin=-0.5, vmax=N_FOLDS - 0.5)
axB.set_xlabel("chronological sweep order"); axB.set_ylabel("RPM")
axB.set_title("(b) Staircase over time by fold\n(twins / up–down sweeps share a fold)")

# (c) groups and sweeps per fold
axC = fig.add_subplot(gs[2])
gpf = sweeps.groupby("fold")["group"].nunique()
spf = sweeps.groupby("fold").size()
x = np.arange(N_FOLDS)
axC.bar(x - 0.2, gpf.reindex(range(N_FOLDS)).values, 0.4, label="groups", color="#1f4e79")
axC.bar(x + 0.2, spf.reindex(range(N_FOLDS)).values / 10, 0.4,
        label="sweeps /10", color="#c55a11")
axC.set_xticks(x); axC.set_xlabel("fold")
axC.set_title("(c) Fold sizes")
axC.legend(fontsize=8); axC.grid(axis="y", ls=":", alpha=0.5)

fig.suptitle(f"Leak-free grouping: {n_hold} holds → {n_merged} operating-point "
             f"groups → {N_FOLDS}-fold GroupKFold", fontsize=13)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "grouping_folds.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: grouping_folds.png")

# group-size distribution summary
gsize = sweeps.groupby("group").size()
print(f"group size (sweeps): min {gsize.min()}, median {int(gsize.median())}, "
      f"max {gsize.max()}; n_groups={n_merged}")
print(f"\nAll outputs -> {SCRIPT_DIR}")

if __name__ == "__main__":
    print("\n18_inspect_results complete.")
