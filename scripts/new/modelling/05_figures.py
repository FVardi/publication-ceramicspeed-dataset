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
  predvactual_<model>.png         2x3 predicted-vs-actual grids (rows full/
                                  selected, cols AE/US/Combined), coloured by
                                  RPM, over the pooled held-out predictions
  marginal_vs_conditional_full.png  paper-style 2-panel figure, ALL features

Reads the DEFAULT (pooled GroupKFold, operating-point-merged) outputs of
11_featureset_comparison.py. Run that (and 12_fullset_decomposition.py) first.

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
NEW_DIR = OUTPUT_DIR / "new"
PRED_DIR = NEW_DIR / "regression" / "predictions"
CM_PATH = NEW_DIR / "correlations" / "tables" / "cond_vs_marginal_full.csv"
FIG_DIR = NEW_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["ElasticNet", "LightGBM"]
TARGETS = ["AE", "US", "Combined"]
MODES = ["full"]  # selected feature set removed 2026-06-29
DPI = 150


# %%
# =============================================================================
# 1. Summary table (rendered) + comparison bars
# =============================================================================
comp = pd.read_csv(NEW_DIR / "regression" / "featureset_comparison.csv")
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

# Holdout R2 bars, grouped by target (full feature set, per model)
fig, ax = plt.subplots(figsize=(9, 5))
colors = {"ElasticNet": "#3182bd", "LightGBM": "#e6550d"}
width = 0.35
x = np.arange(len(TARGETS))
def _r2(m, t):
    sub = comp[(comp.model == m) & (comp.target == t)]
    return float(sub["holdout_r2"].values[0]) if len(sub) else np.nan


for i, m in enumerate(MODELS):
    vals = [_r2(m, t) for t in TARGETS]
    ax.bar(x + (i - 0.5) * width, vals, width, label=m, color=colors[m])
ax.set_xticks(x); ax.set_xticklabels(TARGETS)
ax.set_ylabel("Holdout R²"); ax.set_ylim(0, 1)
ax.set_title("Holdout R² (full feature set)")
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
def _pred_grid():
    """Predicted-vs-actual grid over the pooled held-out predictions (every
    group held out exactly once across all folds, by a model that never saw
    it during selection/tuning/training)."""
    for model in MODELS:
        fig, axes = plt.subplots(len(MODES), 3, figsize=(15, 5 * len(MODES)),
                                 sharex=True, sharey=True, squeeze=False)
        for r, mode in enumerate(MODES):
            for c, target in enumerate(TARGETS):
                ax = axes[r][c]
                fp = PRED_DIR / f"holdout_pooled_{model}_{target}_{mode}.csv"
                if not fp.exists():
                    ax.set_visible(False); continue
                d = pd.read_csv(fp)
                r2 = r2_score(d["y_true"], d["y_pred"])
                sc = ax.scatter(d["y_true"], d["y_pred"], c=d["rpm"], cmap="viridis",
                                s=8, alpha=0.5, edgecolors="none")
                lim = [0, max(d["y_true"].max(), d["y_pred"].max()) * 1.05]
                ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
                ax.set_xlim(lim); ax.set_ylim(lim)
                ax.set_title(f"{target}  R²={r2:.3f}", fontsize=10)
                if c == 0:
                    ax.set_ylabel("predicted κ")
                ax.set_xlabel("true κ")
        fig.colorbar(sc, ax=axes, label="RPM", shrink=0.6, pad=0.01)
        fig.suptitle(f"{model} — predicted vs true κ (pooled held-out)", fontsize=13)
        fig.savefig(FIG_DIR / f"predvactual_{model.lower()}.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"Saved: predvactual_{model.lower()}.png")


_pred_grid()


def _pred_grid_combined():
    """Single 2x3 grid (models x sensors) over the pooled held-out predictions,
    for the paper's predicted-vs-actual figure."""
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(15, 5 * len(MODELS)),
                             sharex=True, sharey=True, squeeze=False)
    sc = None
    for r, model in enumerate(MODELS):
        for c, target in enumerate(TARGETS):
            ax = axes[r][c]
            fp = PRED_DIR / f"holdout_pooled_{model}_{target}_full.csv"
            if not fp.exists():
                ax.set_visible(False); continue
            d = pd.read_csv(fp)
            r2 = r2_score(d["y_true"], d["y_pred"])
            sc = ax.scatter(d["y_true"], d["y_pred"], c=d["rpm"], cmap="viridis",
                            s=8, alpha=0.5, edgecolors="none")
            lim = [0, max(d["y_true"].max(), d["y_pred"].max()) * 1.05]
            ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_title(f"{target}  R²={r2:.3f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{model}\npredicted κ")
            if r == len(MODELS) - 1:
                ax.set_xlabel("true κ")
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="RPM", shrink=0.6, pad=0.01)
    fig.suptitle("Predicted vs true κ (pooled held-out)", fontsize=13)
    fig.savefig(FIG_DIR / "predvactual_combined.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved: predvactual_combined.png")


_pred_grid_combined()


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

# Label only the top-N features per sensor per conditional axis -- gives a
# fixed, readable number regardless of the overall correlation magnitudes.
TOP_N = 5

notable = set()
for sensor in R["sensor"].unique():
    s = R[R["sensor"] == sensor]
    notable.update(s.reindex(s["cond_rpm"].abs().nlargest(TOP_N).index)["feature"])
    notable.update(s.reindex(s["cond_temp"].abs().nlargest(TOP_N).index)["feature"])

R["_notable"] = R["feature"].isin(notable)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharex=True, sharey=True)
panels = [("marg_temp", "marg_rpm", "Marginal correlation (all features)", None),
          ("cond_temp", "cond_rpm", "Conditional (partition-based)", None)]
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
    _texts = [ax.text(row[xc], row[yc], _short(row["feature"]), fontsize=7.5,
                      color=_STY[row["sensor"]]["color"], zorder=5)
              for _, row in R.iterrows()
              if row["_notable"] and np.isfinite(row[xc]) and np.isfinite(row[yc])]
    try:
        from adjustText import adjust_text
        adjust_text(_texts, ax=ax,
                    expand=(2.0, 2.5),
                    force_text=(0.5, 0.8),
                    arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0,
                                    mutation_scale=8, shrinkB=3))
    except (ImportError, Exception):
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
