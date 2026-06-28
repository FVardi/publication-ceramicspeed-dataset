"""
14_new_method_figures.py
========================
Presentation figures + tables for the new (leak-free) pipeline, for visual
inspection. Reads outputs of 11_featureset_comparison.py and
12_fullset_decomposition.py. Run those first (or run_new_pipeline.py).

Produces (outputs/14_new_method_figures/):
  comparison_table.png            rendered model x feature-set summary table
  holdout_r2_bars.png             holdout R2 bar chart (full vs selected)
  cv_vs_holdout_r2.png            CV(grouped) vs holdout R2, all configs
  predvactual_<model>_<split>.png 2x3 predicted-vs-actual grids (rows full/
                                  selected, cols AE/US/Combined), coloured by RPM
  marginal_vs_conditional_full.png  paper-style 2-panel figure, ALL features

Usage
-----
    python scripts/14_new_method_figures.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from ceramicspeed.config import load_config, get_output_dir

cfg = load_config()
OUTPUT_DIR = get_output_dir(cfg)
PRED_DIR = OUTPUT_DIR / "11_featureset_comparison" / "predictions"
CM_PATH = OUTPUT_DIR / "12_fullset_decomposition" / "tables" / "cond_vs_marginal_full.csv"
FIG_DIR = OUTPUT_DIR / "14_new_method_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["ElasticNet", "LightGBM"]
TARGETS = ["AE", "US", "Combined"]
MODES = ["full", "selected"]
DPI = 150


# %%
# =============================================================================
# 1. Summary table (rendered) + comparison bars
# =============================================================================
comp = pd.read_csv(OUTPUT_DIR / "11_featureset_comparison" / "featureset_comparison.csv")
comp = comp.sort_values(["model", "target", "feature_set"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 0.45 * len(comp) + 1.2))
ax.axis("off")
show_cols = ["model", "target", "feature_set", "n_features",
             "cv_r2_grouped", "holdout_r2", "holdout_mae", "holdout_rmse"]
tbl = ax.table(cellText=comp[show_cols].round(4).values,
               colLabels=show_cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
for j in range(len(show_cols)):
    tbl[(0, j)].set_facecolor("#1f4e79")
    tbl[(0, j)].set_text_props(color="white", weight="bold")
ax.set_title("New pipeline — CV (grouped) and holdout performance", pad=12)
fig.savefig(FIG_DIR / "comparison_table.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: comparison_table.png")

# Holdout R2 bars, grouped by target
fig, ax = plt.subplots(figsize=(9, 5))
bar_keys = [(m, fs) for m in MODELS for fs in MODES]
colors = {"ElasticNet": ["#9ecae1", "#3182bd"], "LightGBM": ["#fdae6b", "#e6550d"]}
width = 0.2
x = np.arange(len(TARGETS))
def _r2(m, t, fs):
    sub = comp[(comp.model == m) & (comp.target == t) & (comp.feature_set == fs)]
    return float(sub["holdout_r2"].values[0]) if len(sub) else np.nan


for i, (m, fs) in enumerate(bar_keys):
    vals = [_r2(m, t, fs) for t in TARGETS]
    ax.bar(x + (i - 1.5) * width, vals, width,
           label=f"{m} ({fs})", color=colors[m][MODES.index(fs)])
ax.set_xticks(x); ax.set_xticklabels(TARGETS)
ax.set_ylabel("Holdout R²"); ax.set_ylim(0, 1)
ax.set_title("Holdout R² — full vs selected feature set")
ax.legend(fontsize=8); ax.grid(axis="y", ls=":", alpha=0.5)
fig.tight_layout(); fig.savefig(FIG_DIR / "holdout_r2_bars.png", dpi=DPI)
plt.close()
print("Saved: holdout_r2_bars.png")

# CV vs holdout scatter (optimism check)
fig, ax = plt.subplots(figsize=(6.2, 6))
for m, mk in zip(MODELS, ["o", "s"]):
    s = comp[comp.model == m]
    ax.scatter(s["cv_r2_grouped"], s["holdout_r2"], marker=mk, s=70,
               edgecolor="k", lw=0.5, label=m)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("CV (grouped) R²"); ax.set_ylabel("Holdout R²")
ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)
ax.set_title("CV vs holdout R² (points below line = CV optimism)")
ax.legend(); ax.grid(ls=":", alpha=0.5)
fig.tight_layout(); fig.savefig(FIG_DIR / "cv_vs_holdout_r2.png", dpi=DPI)
plt.close()
print("Saved: cv_vs_holdout_r2.png")


# %%
# =============================================================================
# 2. Predicted-vs-actual grids (CV + holdout), coloured by RPM
# =============================================================================
def _pred_grid(split):
    prefix = "cv" if split == "cv" else "holdout"
    for model in MODELS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
        for r, mode in enumerate(MODES):
            for c, target in enumerate(TARGETS):
                ax = axes[r][c]
                fp = PRED_DIR / f"{prefix}_{model}_{target}_{mode}.csv"
                if not fp.exists():
                    ax.set_visible(False); continue
                d = pd.read_csv(fp)
                r2 = r2_score(d["y_true"], d["y_pred"])
                sc = ax.scatter(d["y_true"], d["y_pred"], c=d["rpm"], cmap="viridis",
                                s=8, alpha=0.5, edgecolors="none")
                lim = [0, max(d["y_true"].max(), d["y_pred"].max()) * 1.05]
                ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_title(f"{target} ({mode})  R²={r2:.3f}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{mode}\npredicted κ")
                if r == 1:
                    ax.set_xlabel("true κ")
        fig.colorbar(sc, ax=axes, label="RPM", shrink=0.6, pad=0.01)
        ttl = "out-of-fold CV" if split == "cv" else "holdout"
        fig.suptitle(f"{model} — predicted vs true κ ({ttl})", fontsize=13)
        fig.savefig(FIG_DIR / f"predvactual_{model.lower()}_{split}.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved: predvactual_{model.lower()}_{split}.png")


_pred_grid("holdout")
_pred_grid("cv")


# %%
# =============================================================================
# 3. Marginal vs conditional correlation figure (paper style) — ALL features
# =============================================================================
R = pd.read_csv(CM_PATH)

_ABBR = {"mobility": "mob", "complexity": "cplx", "spectral_skewness": "sp.skew",
         "spectral_kurtosis": "sp.kurt", "spectral_bandwidth": "sp.bw", "skewness": "skew",
         "kurtosis": "kurt", "crest_factor": "crest", "dominant_frequency": "domfreq",
         "margin_factor": "margin", "shape_factor": "shape", "rms": "rms",
         "center_frequency": "cfreq", "spectral_flatness": "sp.flat"}


def _short(name):
    n = name.replace("AE__", "").replace("US__", "").replace("AE_", "").replace("US_", "")
    band, _, stat = n.partition("__")
    if not stat:
        return _ABBR.get(band, band)
    band = (band.replace("500-1000kHz", "0.5-1M").replace("1000-2000kHz", "1-2M")
                .replace("20-500kHz", "20-500k").replace("20-100kHz", "20-100k")
                .replace("10-20kHz", "10-20k").replace("0-10kHz", "0-10k"))
    return f"{band} {_ABBR.get(stat, stat)}"


_STY = {"AE": dict(marker="o", color="#1f4e79", label="AE"),
        "US": dict(marker="s", color="#c55a11", label="US")}
has_iqr = {"cond_temp_lo", "cond_temp_hi", "cond_rpm_lo", "cond_rpm_hi"}.issubset(R.columns)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
panels = [("marg_temp", "marg_rpm", "Marginal correlation (all features)", None),
          ("cond_temp", "cond_rpm", "Conditional (partition-based)",
           ("cond_temp_lo", "cond_temp_hi", "cond_rpm_lo", "cond_rpm_hi") if has_iqr else None)]
for ax, (xc, yc, ttl, errs) in zip(axes, panels):
    for sensor, st in _STY.items():
        s = R[R["sensor"] == sensor]
        if errs:
            ax.errorbar(s[xc], s[yc],
                        xerr=[s[xc] - s[errs[0]], s[errs[1]] - s[xc]],
                        yerr=[s[yc] - s[errs[2]], s[errs[3]] - s[yc]],
                        fmt="none", ecolor=st["color"], alpha=0.2, lw=0.7)
        ax.scatter(s[xc], s[yc], marker=st["marker"], color=st["color"],
                   s=45, edgecolor="k", lw=0.4, label=st["label"], zorder=3)
    _texts = [ax.text(row[xc], row[yc], _short(row["feature"]), fontsize=5.2,
                      color=_STY[row["sensor"]]["color"], zorder=4)
              for _, row in R.iterrows() if np.isfinite(row[xc]) and np.isfinite(row[yc])]
    try:
        from adjustText import adjust_text
        adjust_text(_texts, ax=ax, expand=(1.1, 1.3),
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.3))
    except ImportError:
        pass
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel(r"Spearman $\rho$ with temperature")
    ax.set_title(ttl)
axes[0].set_ylabel(r"Spearman $\rho$ with RPM (speed)")
axes[0].legend(loc="lower left")
fig.tight_layout()
fig.savefig(FIG_DIR / "marginal_vs_conditional_full.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: marginal_vs_conditional_full.png")

print(f"\nAll figures saved to {FIG_DIR}")

if __name__ == "__main__":
    print("\n14_new_method_figures complete.")
