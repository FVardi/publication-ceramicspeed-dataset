"""
06_plots.py
===========
Regenerate all publication figures from saved outputs — no model retraining needed.

Pipeline position: run after 03_evaluation.py and 04_modelling.py have completed.

Reads from:
  outputs/03_evaluation/predictions/repeated_cv_scores.csv
  outputs/03_evaluation/tables/performance_table_cv.csv
  outputs/04_modelling/predictions/model_cv_{tag}.csv
  outputs/04_modelling/predictions/model_holdout_{tag}.csv
  outputs/04_modelling/shap/shap_importance_{tag}.csv
  outputs/04_modelling/shap/shap_values_{tag}.csv
  outputs/04_modelling/shap/shap_sensor_contribution_{name}.csv
  outputs/04_modelling/tables/model_weights_{tag}.csv
  outputs/04_modelling/tables/model_folds_{tag}.csv
  outputs/features.parquet
  outputs/metadata.parquet
  outputs/feature_selection.json

Writes to:
  outputs/06_plots/

Usage
-----
    python scripts/06_plots.py
    python scripts/06_plots.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import math
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.loading import load_parquet_pair

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = get_output_dir(cfg)
SCRIPT_DIR = OUTPUT_DIR / "06_plots"
SCRIPT_DIR.mkdir(exist_ok=True)

CV_DIR        = OUTPUT_DIR / "03_evaluation"
MODEL_DIR     = OUTPUT_DIR / "04_modelling"
PRED_DIR      = MODEL_DIR / "predictions"
SHAP_DIR      = MODEL_DIR / "shap"
TABLES_DIR    = MODEL_DIR / "tables"
CV_PRED_DIR   = CV_DIR / "predictions"
CV_TABLE_DIR  = CV_DIR / "tables"

# %%
# =============================================================================
# Aesthetics  — edit this section to change the look of all plots
# =============================================================================

# --- General ---
DPI         = 150
FONT_SIZE   = 10
TITLE_SIZE  = 12
LABEL_SIZE  = 10
TICK_SIZE   = 9

plt.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   TITLE_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "figure.dpi":       DPI,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# --- Colors ---
COLOR_BAR       = "#4878CF"   # main bar / violin fill
COLOR_ERROR     = "#2c2c2c"   # error bar caps
COLOR_IDEAL     = "#333333"   # 1:1 diagonal line
COLOR_ZERO      = "#333333"   # residual zero line
ALPHA_SCATTER   = 0.6
ALPHA_VIOLIN    = 0.7
RPM_CMAP        = "tab10"
RPM_STEP        = 1000        # bin width for RPM colormap

MODEL_COLORS = {
    "ElasticNet":  "#4878CF",
    "Polynomial":  "#6ACC65",
    "LightGBM":    "#D65F5F",
}

# %%
# =============================================================================
# Load shared data
# =============================================================================

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)

feat_sel_path = OUTPUT_DIR / "feature_selection.json"
with open(feat_sel_path) as fh:
    feature_selection: dict = json.load(fh)

sensor_names  = list(feature_selection.keys())
model_types   = ["ElasticNet", "Polynomial", "LightGBM"]
_SENSOR_LABEL = {"UL": "US"}
display_sensor_names = [_SENSOR_LABEL.get(s, s) for s in sensor_names]

# Build the list of all model tags from saved holdout CSVs
all_tags: list[str] = sorted(
    p.stem.replace("model_holdout_", "")
    for p in PRED_DIR.glob("model_holdout_*.csv")
)

# Decode tag → model type and feature set label
def _tag_parts(tag: str) -> tuple[str, str]:
    """Return (model_type, feature_set) from a lowercase model tag."""
    for mt in model_types:
        if tag.startswith(mt.lower() + "_"):
            fs = tag[len(mt) + 1:]
            return mt, fs.upper() if fs != "combined" else "Combined"
    return tag, "?"

# RPM colormap (shared across all scatter plots, derived from metadata)
_rpm_data_max  = float(raw_metadata_df["rpm"].max())
_rpm_ceil      = math.ceil(_rpm_data_max / RPM_STEP) * RPM_STEP
_rpm_boundaries = np.arange(0, _rpm_ceil + RPM_STEP, RPM_STEP)
_rpm_n         = len(_rpm_boundaries) - 1
_rpm_cmap      = plt.colormaps[RPM_CMAP].resampled(_rpm_n)
_rpm_norm      = mcolors.BoundaryNorm(_rpm_boundaries, _rpm_n)

# Metadata indexed by (file, sweep) for RPM lookups
# Deduplicate first — raw_metadata_df has one row per sensor × sweep
_meta_idx = (
    raw_metadata_df.drop_duplicates(subset=["file", "sweep"])
    .set_index(["file", "sweep"])["rpm"]
)


def _get_rpm(ho_df: pd.DataFrame) -> np.ndarray | None:
    """Return RPM array aligned with rows of a holdout/cv prediction DataFrame."""
    if "rpm" in ho_df.columns:
        return ho_df["rpm"].values
    if "file" in ho_df.columns and "sweep" in ho_df.columns:
        keys = list(zip(ho_df["file"], ho_df["sweep"]))
        rpm = np.array([_meta_idx.get(k, np.nan) for k in keys])
        return rpm if not np.all(np.isnan(rpm)) else None
    return None


# CV scores
cv_scores_df  = pd.read_csv(CV_PRED_DIR / "repeated_cv_scores.csv")
cv_scores     = {col: cv_scores_df[col].values for col in cv_scores_df.columns}
perf_df       = pd.read_csv(CV_TABLE_DIR / "performance_table_cv.csv")

feature_sets  = display_sensor_names + (["Combined"] if any("Combined" in t for t in cv_scores) else [])

# %%
# =============================================================================
# Plot E1 — CV score distributions (violin, one per feature set)
# =============================================================================

for fs in feature_sets:
    names     = [f"{mt}_{_SENSOR_LABEL.get(fs, fs)}" if fs not in ("Combined",) else f"{mt}_Combined"
                 for mt in model_types]
    available = [n for n in names if n in cv_scores]
    if not available:
        continue

    fig, ax = plt.subplots(figsize=(6, 4))
    data   = [cv_scores[n] for n in available]
    labels = [n.split("_")[0] for n in available]
    colors = [MODEL_COLORS.get(lbl, COLOR_BAR) for lbl in labels]

    vp = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=True)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(ALPHA_VIOLIN)
    for part in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part in vp:
            vp[part].set_color("#333333")
            vp[part].set_linewidth(1.2)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE (outer CV fold)")
    ax.set_title(f"CV score distribution — {fs} features\n"
                 f"(R={len(data[0]) // cfg.get('modelling', {}).get('cv_n_splits', 5)} "
                 f"× k={cfg.get('modelling', {}).get('cv_n_splits', 5)} folds)")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig_path = SCRIPT_DIR / f"eval_cv_distribution_{fs.lower()}.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")

# %%
# =============================================================================
# Plot E2 — Mean ± std RMSE bar chart (all models)
# =============================================================================

fig, ax = plt.subplots(figsize=(max(8, len(perf_df) * 0.9), 5))
x = np.arange(len(perf_df))
ax.bar(x, perf_df["mean_rmse"], yerr=perf_df["std_rmse"],
       capsize=4, color=COLOR_BAR, alpha=0.85,
       error_kw={"linewidth": 1.2, "ecolor": COLOR_ERROR})
ax.set_xticks(x)
ax.set_xticklabels(perf_df["model"], rotation=40, ha="right", fontsize=TICK_SIZE)
ax.set_ylabel("Mean RMSE (repeated nested CV)")
ax.set_title("Model performance — mean ± std RMSE")
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
plt.savefig(SCRIPT_DIR / "eval_performance_bar.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: eval_performance_bar.png")

# %%
# =============================================================================
# Load holdout + CV prediction DataFrames
# =============================================================================

holdout: dict[str, pd.DataFrame] = {}
cv_preds: dict[str, pd.DataFrame] = {}

for tag in all_tags:
    ho_path = PRED_DIR / f"model_holdout_{tag}.csv"
    cv_path = PRED_DIR / f"model_cv_{tag}.csv"
    if ho_path.exists():
        holdout[tag] = pd.read_csv(ho_path)
    if cv_path.exists():
        cv_preds[tag] = pd.read_csv(cv_path)

# %%
# =============================================================================
# Plot M1a — CV out-of-fold: predicted vs actual (all models)
# =============================================================================

tags_with_cv = [t for t in all_tags if t in cv_preds]
if tags_with_cv:
    n = len(tags_with_cv)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in flat[n:]:
        ax.set_visible(False)

    for ax, tag in zip(flat, tags_with_cv):
        df   = cv_preds[tag]
        rpm  = _get_rpm(df)
        mt, fs = _tag_parts(tag)
        sc = ax.scatter(df["y_true"], df["y_pred"],
                        c=rpm if rpm is not None else COLOR_BAR,
                        cmap=_rpm_cmap if rpm is not None else None,
                        norm=_rpm_norm if rpm is not None else None,
                        s=12, alpha=ALPHA_SCATTER, edgecolors="none")
        if rpm is not None:
            plt.colorbar(sc, ax=ax, label="RPM")
        lims = [min(df["y_true"].min(), df["y_pred"].min()),
                max(df["y_true"].max(), df["y_pred"].max())]
        margin = 0.05 * (lims[1] - lims[0])
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, "--", color=COLOR_IDEAL, lw=1, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        r2   = float(1 - np.sum((df["y_true"] - df["y_pred"])**2) /
                         np.sum((df["y_true"] - df["y_true"].mean())**2))
        rmse = float(np.sqrt(((df["y_true"] - df["y_pred"])**2).mean()))
        ax.set_title(f"{mt} — {fs}\nR²={r2:.3f}  RMSE={rmse:.3f}")
        ax.set_xlabel("True κ"); ax.set_ylabel("Predicted κ")
        ax.grid(ls=":", alpha=0.4)

    fig.suptitle("CV Out-of-Fold Predictions vs True κ (Training Set)", fontsize=TITLE_SIZE)
    fig.tight_layout()
    plt.savefig(SCRIPT_DIR / "model_pred_vs_actual_cv.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved: model_pred_vs_actual_cv.png")

# %%
# =============================================================================
# Plot M1b — Holdout: predicted vs actual (all models)
# =============================================================================

n = len(all_tags)
ncols = min(3, n)
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
for ax in flat[n:]:
    ax.set_visible(False)

for ax, tag in zip(flat, all_tags):
    if tag not in holdout:
        ax.set_visible(False)
        continue
    df  = holdout[tag]
    rpm = _get_rpm(df)
    mt, fs = _tag_parts(tag)
    sc = ax.scatter(df["y_true"], df["y_pred"],
                    c=rpm if rpm is not None else COLOR_BAR,
                    cmap=_rpm_cmap if rpm is not None else None,
                    norm=_rpm_norm if rpm is not None else None,
                    s=14, alpha=ALPHA_SCATTER, edgecolors="none")
    if rpm is not None:
        plt.colorbar(sc, ax=ax, label="RPM")
    lims = [min(df["y_true"].min(), df["y_pred"].min()),
            max(df["y_true"].max(), df["y_pred"].max())]
    margin = 0.05 * (lims[1] - lims[0])
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "--", color=COLOR_IDEAL, lw=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2   = float(1 - np.sum((df["y_true"] - df["y_pred"])**2) /
                     np.sum((df["y_true"] - df["y_true"].mean())**2))
    rmse = float(np.sqrt(((df["y_true"] - df["y_pred"])**2).mean()))
    mae  = float((df["y_true"] - df["y_pred"]).abs().mean())
    ax.set_title(f"{mt} — {fs}\nR²={r2:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    ax.set_xlabel("True κ"); ax.set_ylabel("Predicted κ")
    ax.grid(ls=":", alpha=0.4)

fig.suptitle("Hold-Out Test Set: Predicted vs True κ", fontsize=TITLE_SIZE)
fig.tight_layout()
plt.savefig(SCRIPT_DIR / "model_pred_vs_actual_holdout.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: model_pred_vs_actual_holdout.png")

# %%
# =============================================================================
# Plot M2 — Coefficient / importance bar charts (log scale, per model)
# =============================================================================

for tag in all_tags:
    weight_path = TABLES_DIR / f"model_weights_{tag}.csv"
    if not weight_path.exists():
        continue
    wdf = pd.read_csv(weight_path)
    # Detect weight column (coefficient or importance)
    weight_col = next((c for c in wdf.columns if c not in ("feature",) and not c.startswith("abs_")), None)
    if weight_col is None:
        continue
    top_n = min(20, len(wdf))
    wdf   = wdf.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    vals   = wdf[weight_col].values
    labels = wdf["feature"].values
    colors = [MODEL_COLORS.get(_tag_parts(tag)[0], COLOR_BAR)] * len(vals)

    # For coefficients (can be negative): use signed bar; for importance: always positive
    if np.any(vals < 0):
        ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel(weight_col.replace("_", " ").title())
    else:
        ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(f"{weight_col.replace('_', ' ').title()} (log scale)")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=max(6, TICK_SIZE - 1))
    ax.invert_yaxis()
    mt, fs = _tag_parts(tag)
    ax.set_title(f"{mt} — {fs}")
    fig.tight_layout()
    plt.savefig(SCRIPT_DIR / f"model_coefs_log_{tag}.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: model_coefs_log_{tag}.png")

# %%
# =============================================================================
# Plot M3 — CV fold metrics (R² per fold, all models)
# =============================================================================

n = len(all_tags)
ncols = min(3, n)
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
for ax in flat[n:]:
    ax.set_visible(False)

for ax, tag in zip(flat, all_tags):
    fold_path = TABLES_DIR / f"model_folds_{tag}.csv"
    if not fold_path.exists():
        ax.set_visible(False)
        continue
    fdf = pd.read_csv(fold_path)
    mt, fs = _tag_parts(tag)
    color = MODEL_COLORS.get(mt, COLOR_BAR)
    folds = fdf["fold"].values if "fold" in fdf.columns else np.arange(len(fdf))
    ax.bar(folds, fdf["r2"].values, color=color, alpha=0.8)
    ax.axhline(fdf["r2"].mean(), color="#333333", ls="--", lw=1, label=f"mean={fdf['r2'].mean():.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title(f"{mt} — {fs}")
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(axis="y", ls=":", alpha=0.4)

fig.suptitle("R² per Cross-Validation Fold (Training Set)", fontsize=TITLE_SIZE)
fig.tight_layout()
plt.savefig(SCRIPT_DIR / "model_cv_fold_r2.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: model_cv_fold_r2.png")

# %%
# =============================================================================
# Plot M4 — Holdout residuals (all models)
# =============================================================================

fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
for ax in flat[n:]:
    ax.set_visible(False)

for ax, tag in zip(flat, all_tags):
    if tag not in holdout:
        ax.set_visible(False)
        continue
    df  = holdout[tag]
    rpm = _get_rpm(df)
    mt, fs = _tag_parts(tag)
    sc = ax.scatter(df["y_pred"], df["residual"],
                    c=rpm if rpm is not None else MODEL_COLORS.get(mt, COLOR_BAR),
                    cmap=_rpm_cmap if rpm is not None else None,
                    norm=_rpm_norm if rpm is not None else None,
                    s=12, alpha=ALPHA_SCATTER, edgecolors="none")
    if rpm is not None:
        plt.colorbar(sc, ax=ax, label="RPM")
    ax.axhline(0, color=COLOR_ZERO, ls="--", lw=0.8)
    ax.set_xlabel("Predicted κ")
    ax.set_ylabel("Residual (true − pred)")
    ax.set_title(f"{mt} — {fs}")
    ax.grid(ls=":", alpha=0.4)

fig.suptitle("Residual Analysis (Hold-Out Test Set)", fontsize=TITLE_SIZE)
fig.tight_layout()
plt.savefig(SCRIPT_DIR / "model_residuals_holdout.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: model_residuals_holdout.png")

# %%
# =============================================================================
# Plot S1 — SHAP importance bar (per model, ElasticNet + LightGBM only)
# =============================================================================

for tag in all_tags:
    imp_path = SHAP_DIR / f"shap_importance_{tag}.csv"
    if not imp_path.exists():
        continue
    imp_df = pd.read_csv(imp_path, index_col=0, header=0)
    imp_df.columns = ["mean_abs_shap"]
    top_n  = min(20, len(imp_df))
    imp_df = imp_df.head(top_n)

    mt, fs = _tag_parts(tag)
    color  = MODEL_COLORS.get(mt, COLOR_BAR)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(top_n), imp_df["mean_abs_shap"].values, color=color, alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(imp_df.index.tolist(), fontsize=max(6, TICK_SIZE - 1))
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP importance — {mt} / {fs}")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    plt.savefig(SCRIPT_DIR / f"shap_importance_{tag}.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: shap_importance_{tag}.png")

# %%
# =============================================================================
# Plot S2 — SHAP beeswarm (per model)
# =============================================================================

try:
    import shap as shap_lib

    # Build feature data lookup: (file, sweep) → feature row, per sensor
    _feat_by_sensor: dict[str, pd.DataFrame] = {}
    for sname in sensor_names:
        mask = raw_feature_df["sensor"] == sname
        _feat_by_sensor[sname] = raw_feature_df[mask].set_index(["file", "sweep"])

    for tag in all_tags:
        shap_val_path = SHAP_DIR / f"shap_values_{tag}.csv"
        ho_path       = PRED_DIR / f"model_holdout_{tag}.csv"
        if not shap_val_path.exists() or tag not in holdout:
            continue

        shap_df = pd.read_csv(shap_val_path, index_col=0)
        ho_df   = holdout[tag]
        feature_names = shap_df.columns.tolist()
        shap_vals = shap_df.values

        # Reconstruct feature data for beeswarm colouring
        mt, fs = _tag_parts(tag)
        X_data: np.ndarray | None = None
        try:
            if fs == "Combined":
                # Combined model: columns are prefixed SENSOR__feature
                parts = []
                for sname in sensor_names:
                    label = _SENSOR_LABEL.get(sname, sname)
                    pfx   = f"{label}__"
                    cols  = [c for c in feature_names if c.startswith(pfx) or not any(
                        c.startswith(f"{_SENSOR_LABEL.get(s, s)}__") for s in sensor_names
                    )]
                    # simpler: just find which columns belong to this sensor
                    sensor_cols = [c for c in feature_names if c.startswith(pfx)]
                    if sensor_cols and "file" in ho_df.columns:
                        keys   = list(zip(ho_df["file"], ho_df["sweep"]))
                        feat_s = _feat_by_sensor.get(sname)
                        if feat_s is not None:
                            raw_cols = [c.replace(pfx, "") for c in sensor_cols]
                            sub = feat_s.reindex(keys)[raw_cols].values
                            parts.append((sensor_cols, sub))
                if parts:
                    X_data = np.empty((len(ho_df), len(feature_names)))
                    for sensor_cols, sub in parts:
                        idxs = [feature_names.index(c) for c in sensor_cols]
                        for i, idx in enumerate(idxs):
                            X_data[:, idx] = sub[:, i]
            else:
                # Per-sensor model
                sname = next((s for s in sensor_names
                              if _SENSOR_LABEL.get(s, s).lower() == fs.lower()), None)
                if sname and "file" in ho_df.columns:
                    keys     = list(zip(ho_df["file"], ho_df["sweep"]))
                    feat_s   = _feat_by_sensor.get(sname)
                    if feat_s is not None:
                        X_data = feat_s.reindex(keys)[feature_names].values
        except Exception:
            X_data = None

        base_val = float(ho_df["y_pred"].mean())
        explanation = shap_lib.Explanation(
            values=shap_vals,
            base_values=np.full(len(shap_vals), base_val),
            data=X_data,
            feature_names=feature_names,
        )
        try:
            shap_lib.plots.beeswarm(
                explanation,
                max_display=min(20, len(feature_names)),
                show=False,
            )
            bw_path = SCRIPT_DIR / f"shap_beeswarm_{tag}.png"
            plt.savefig(bw_path, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"Saved: shap_beeswarm_{tag}.png")
        except Exception as exc:
            plt.close("all")
            print(f"Beeswarm skipped for {tag}: {exc}")

except ImportError:
    print("shap not installed — beeswarm plots skipped")

# %%
# =============================================================================
# Plot S3 — SHAP sensor contribution (combined models)
# =============================================================================

for contrib_path in sorted(SHAP_DIR.glob("shap_sensor_contribution_*.csv")):
    contrib_df = pd.read_csv(contrib_path)
    name       = contrib_path.stem.replace("shap_sensor_contribution_", "")

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = [MODEL_COLORS.get(s, COLOR_BAR) for s in contrib_df["sensor"]]
    ax.barh(contrib_df["sensor"], contrib_df["total_mean_abs_shap"], color=colors, alpha=0.85)
    ax.set_xlabel("Sum of mean |SHAP|")
    ax.set_title(f"Sensor contribution — {name}")
    ax.invert_yaxis()
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig_path = SCRIPT_DIR / f"shap_sensor_contribution_{name}.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")

# %%
# =============================================================================
# Entry point
# =============================================================================

print(f"\nAll figures saved to: {SCRIPT_DIR}")

if __name__ == "__main__":
    print("\n06_plots complete.")
