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
# Fall back to the repo-local outputs/ when the configured machine profile
# does not match this machine (same guard as scripts 07-10).
from pathlib import Path as _P
_REPO_OUT = _P(__file__).resolve().parent.parent / "outputs"
if not (OUTPUT_DIR / "features.parquet").exists() and (_REPO_OUT / "features.parquet").exists():
    print(f"NOTE: {OUTPUT_DIR} has no pipeline outputs; falling back to {_REPO_OUT}")
    OUTPUT_DIR = _REPO_OUT
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
    feature_selection: dict = json.loads(fh.read().rstrip("\x00"))

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
_RPM_MAX_FILTER = float(cfg.get("filters", {}).get("rpm_max", np.inf))
_rpm_data_max  = float(
    raw_metadata_df.loc[raw_metadata_df["rpm"] <= _RPM_MAX_FILTER, "rpm"].max())
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
# Plot S2b — AE | US beeswarm 2-panel (composed from the per-model PNGs)
# =============================================================================

import matplotlib.image as _mpimg

_ae_bw = SCRIPT_DIR / "shap_beeswarm_lightgbm_ae.png"
_us_bw = SCRIPT_DIR / "shap_beeswarm_lightgbm_us.png"
if _ae_bw.exists() and _us_bw.exists():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, _png, _ttl in zip(axes, [_ae_bw, _us_bw], ["(a) AE", "(b) US"]):
        ax.imshow(_mpimg.imread(_png))
        ax.set_title(_ttl, fontsize=12, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "shap_beeswarm_ae_us.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved: shap_beeswarm_ae_us.png")
else:
    print("Skipped S2b (per-model AE/US beeswarm PNGs not found)")

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
# Plot D1 — Band validation: stationary vs running PSDs (from 08)
# =============================================================================

BAND_DIR = OUTPUT_DIR / "08_band_validation"
_psd_path = BAND_DIR / "tables" / "psd_curves.npz"
if _psd_path.exists():
    _npz = np.load(_psd_path)
    freq = _npz["freq"]
    curves = {}
    for key in _npz.files:
        if not key.startswith("psd__"):
            continue
        _, tb, group, ntag = key.split("__")
        curves.setdefault(tb, []).append((group, int(ntag[1:]), _npz[key]))
    bands_cfg = cfg["frequency_bands"]["AE"]
    for tb, items in curves.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for group, n, arr in sorted(items):
            is_still = group == "standstill"
            ax.loglog(freq, arr, lw=1.1,
                      color="#2c2c2c" if is_still else None,
                      alpha=0.95 if is_still else 0.75,
                      label=f"{'stationary' if is_still else group} (n={n})")
        for b in bands_cfg:
            ax.axvspan(max(b["f_lo"], 1), b["f_hi"], alpha=0.06, color=COLOR_BAR)
        ax.set_xlim(1e3, freq.max())
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"PSD [V$^2$/Hz]")
        ax.set_title(f"AE PSD, temperature bin {tb}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = tb.replace(" ", "").replace(",", "_").replace("(", "").replace("]", "")
        fig_path = SCRIPT_DIR / f"band_validation_psd_{safe}.png"
        plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
        # stable alias used by the paper: the coolest bin (contains the
        # stationary reference)
        _lo = float(tb.split(",")[0].strip("( ").strip())
        if _lo == min(float(t.split(",")[0].strip("( ").strip()) for t in curves):
            plt.savefig(SCRIPT_DIR / "band_validation_psd.png", dpi=DPI, bbox_inches="tight")
            print("Saved: band_validation_psd.png (paper alias)")
        plt.close()
        print(f"Saved: {fig_path.name}")
else:
    print("Skipped D1 (run scripts/08_band_validation.py first)")

# %%
# =============================================================================
# Plot D2 — Headline feature at standstill vs running (from 08)
# =============================================================================

_sf_path = BAND_DIR / "tables" / "standstill_features.csv"
if _sf_path.exists():
    sf = pd.read_csv(_sf_path)
    hl = "AE_1000-2000kHz__complexity"
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for is_still, lbl, col in [(False, "running", COLOR_BAR), (True, "stationary", "#D65F5F")]:
        sub = sf[(sf["group"] == "standstill") == is_still]
        ax.scatter(sub["temp"], sub[hl], s=18, alpha=0.8, label=lbl, color=col)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel(hl.replace("__", " "))
    ax.legend()
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig_path = SCRIPT_DIR / "band_validation_standstill_feature.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")
else:
    print("Skipped D2 (run scripts/08_band_validation.py first)")

# %%
# =============================================================================
# Plot D3 — Comb-strip band power vs RPM, per temperature window (from 10)
# =============================================================================

MECH_DIR = OUTPUT_DIR / "10_band_mechanism"
for tw_dir in sorted(MECH_DIR.glob("tw_*")) if MECH_DIR.exists() else []:
    cs_path = tw_dir / "tables" / "comb_strip.csv"
    if not cs_path.exists():
        continue
    cs = pd.read_csv(cs_path)
    band_labels = [b["label"] for b in cfg["frequency_bands"]["AE"]]
    band_labels += [b["label"] for b in cfg.get("validation_extra_bands", {}).get("AE", [])]
    fig, axes = plt.subplots(1, len(band_labels), figsize=(4.6 * len(band_labels), 3.8),
                             sharex=True)
    for ax, label in zip(np.atleast_1d(axes), band_labels):
        sub = cs[cs["band"] == label]
        run, still = sub[sub["rpm"] >= 60], sub[sub["rpm"] < 60]
        ax.semilogy(run["rpm"], run["p_broad"], "o", ms=4.5, color=COLOR_BAR,
                    label="broadband residual")
        ax.semilogy(run["rpm"], run["p_line"], "s", ms=4.5, color="#D65F5F",
                    alpha=0.75, label="line (comb) component")
        if not still.empty:
            ax.axhline(still["p_broad"].median(), color="#2c2c2c", ls="--", lw=1,
                       label="stationary broadband")
        ax.set_title(label)
        ax.set_xlabel("RPM")
        ax.grid(ls=":", alpha=0.4)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel(r"Band power [V$^2$]")
    ax0.legend(fontsize=8)
    fig.suptitle(f"Comb-stripped AE band power ({tw_dir.name.replace('tw_', '')})", y=1.02)
    fig.tight_layout()
    fig_path = SCRIPT_DIR / f"comb_strip_vs_rpm_{tw_dir.name}.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    # stable alias used by the paper: the coolest window
    _all_tw = sorted(MECH_DIR.glob("tw_*"), key=lambda d: float(d.name[3:].split("-")[0]))
    if tw_dir == _all_tw[0]:
        plt.savefig(SCRIPT_DIR / "comb_strip_cool.png", dpi=DPI, bbox_inches="tight")
        print("Saved: comb_strip_cool.png (paper alias)")
    plt.close()
    print(f"Saved: {fig_path.name}")
if not MECH_DIR.exists():
    print("Skipped D3 (run scripts/10_band_mechanism.py first)")

# %%
# =============================================================================
# Plot D4 — Within-RPM-step temperature sensitivity (from 10)
# =============================================================================

_ws_candidates = sorted(MECH_DIR.glob("tw_*/tables/within_step_rho.csv")) if MECH_DIR.exists() else []
if _ws_candidates:
    ws = pd.read_csv(_ws_candidates[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for feat_name, sub in ws.groupby("feature"):
        ax.plot(sub["step"], sub["rho_temp"], marker="o", ms=3, lw=1,
                label=feat_name.replace("__", " "))
    ax.axhline(0, color="#2c2c2c", lw=0.8)
    ax.set_xlabel("RPM step")
    ax.set_ylabel(r"Within-step Spearman $\rho$(feature, temperature)")
    ax.legend(fontsize=7)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig_path = SCRIPT_DIR / "within_step_rho.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")
else:
    print("Skipped D4 (run scripts/10_band_mechanism.py first)")

# %%
# =============================================================================
# Plot D4c — Marginal vs conditional correlation structure (from 09)
# =============================================================================

_cm_path = OUTPUT_DIR / "09_proxy_diagnostics" / "tables" / "cond_vs_marginal.csv"
if _cm_path.exists():
    R = pd.read_csv(_cm_path)
    _ABBR = {"mobility": "mob", "complexity": "cplx", "spectral_skewness": "sp.skew",
             "spectral_kurtosis": "sp.kurt", "spectral_bandwidth": "sp.bw", "skewness": "skew",
             "kurtosis": "kurt", "crest_factor": "crest", "dominant_frequency": "domfreq",
             "margin_factor": "margin", "shape_factor": "shape", "rms": "rms"}

    def _short(name):
        n = name.replace("AE_", "").replace("US_", "").replace("UL_", "")
        band, _, stat = n.partition("__")
        if not stat:
            return _ABBR.get(band, band)
        band = (band.replace("500-1000kHz", "0.5-1M").replace("1000-2000kHz", "1-2M")
                    .replace("20-500kHz", "20-500k").replace("20-100kHz", "20-100k")
                    .replace("10-20kHz", "10-20k").replace("0-10kHz", "0-10k"))
        return f"{band} {_ABBR.get(stat, stat)}"

    _STY = {"AE": dict(marker="o", color="#1f4e79", label="AE"),
            "US": dict(marker="s", color="#c55a11", label="US")}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    for ax, (xc, yc, ttl, errs) in zip(axes, [
            ("marg_temp", "marg_rpm", "Marginal correlation", None),
            ("cond_temp", "cond_rpm", "Conditional (partition-based)",
             ("cond_temp_lo", "cond_temp_hi", "cond_rpm_lo", "cond_rpm_hi"))]):
        for sensor, st in _STY.items():
            s = R[R["sensor"] == sensor]
            if errs:
                ax.errorbar(s[xc], s[yc],
                            xerr=[s[xc] - s[errs[0]], s[errs[1]] - s[xc]],
                            yerr=[s[yc] - s[errs[2]], s[errs[3]] - s[yc]],
                            fmt="none", ecolor=st["color"], alpha=0.22, lw=0.8)
            ax.scatter(s[xc], s[yc], marker=st["marker"], color=st["color"],
                       s=50, edgecolor="k", lw=0.4, label=st["label"], zorder=3)
        _texts = [ax.text(r[xc], r[yc], _short(r["feature"]), fontsize=5.6,
                          color=_STY[r["sensor"]]["color"], zorder=4)
                  for _, r in R.iterrows()]
        try:
            from adjustText import adjust_text
            adjust_text(_texts, ax=ax, expand=(1.15, 1.4),
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4))
        except ImportError:
            pass  # plain labels if adjustText not installed
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r"Spearman $\rho$ with temperature")
        ax.set_title(ttl)
    axes[0].set_ylabel(r"Spearman $\rho$ with RPM (speed)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig_path = SCRIPT_DIR / "marginal_vs_conditional.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")
else:
    print("Skipped D4c (run scripts/09_proxy_diagnostics.py first)")

# %%
# =============================================================================
# Plot D5 — Two-stage vs direct kappa prediction (from 09 + 04)
# =============================================================================

PROXY_DIR = OUTPUT_DIR / "09_proxy_diagnostics"
_ts_path = PROXY_DIR / "tables" / "two_stage_predictions.csv"
_direct_path = PRED_DIR / "model_holdout_lightgbm_ae.csv"
if _ts_path.exists() and _direct_path.exists():
    ts = pd.read_csv(_ts_path)
    direct = pd.read_csv(_direct_path).dropna(subset=["file"])

    def _r2_rmse(y, yh):
        y, yh = np.asarray(y), np.asarray(yh)
        ss = 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
        return ss, float(np.sqrt(np.mean((y - yh) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    panels = [
        (ts["y_true"], ts["y_pred_two_stage"],
         "Two-stage: features → (R̂PM, T̂) → κ"),
        (direct["y_true"], direct["y_pred"],
         "Direct: features → κ (LightGBM)"),
    ]
    for ax, (y, yh, ttl) in zip(axes, panels):
        r2, rmse = _r2_rmse(y, yh)
        ax.scatter(y, yh, s=4, alpha=0.3, color=COLOR_BAR)
        lim = [0, float(max(y.max(), yh.max())) * 1.05]
        ax.plot(lim, lim, ls="--", lw=1, color=COLOR_IDEAL)
        ax.set_xlabel(r"True $\kappa$")
        ax.set_title(f"{ttl}\n$R^2$={r2:.3f}, RMSE={rmse:.3f}")
        ax.grid(ls=":", alpha=0.4)
    axes[0].set_ylabel(r"Predicted $\kappa$")
    fig.tight_layout()
    fig_path = SCRIPT_DIR / "two_stage_vs_direct.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")
else:
    print("Skipped D5 (run scripts/09_proxy_diagnostics.py first)")

# %%
# =============================================================================
# Plot D6 — VIF of Stage-1 survivors, log axis (from 02)
# =============================================================================

_vif_paths = {n: OUTPUT_DIR / "02_feature_analysis" / "tables" / f"vif_{n}.csv"
              for n in ("ae", "us")}
if all(pp.exists() for pp in _vif_paths.values()):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, (n, pp) in zip(axes, _vif_paths.items()):
        v = pd.read_csv(pp).sort_values("vif", ascending=False).reset_index(drop=True)
        v["vif"] = v["vif"].clip(upper=1e6)  # display cap for (near-)collinear features
        colors = [COLOR_BAR if r else "#bbbbbb" for r in v["retained"]]
        ax.bar(range(len(v)), v["vif"], color=colors)
        ax.set_yscale("log")
        _vif_thresh = float(cfg.get("feature_selection", {}).get("vif_threshold", 5))
        ax.axhline(_vif_thresh, color="#2c2c2c", ls="--", lw=1)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([f.replace("__", "\n") for f in v["feature"]],
                           rotation=90, fontsize=5)
        ax.set_title(f"{n.upper()} — VIF (log scale)")
        ax.set_ylabel("VIF" if n == "ae" else "")
        ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig_path = SCRIPT_DIR / "vif_log.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")
else:
    print("Skipped D6 (re-run scripts/02_feature_analysis.py to export vif_ae/us.csv)")

# %%
# =============================================================================
# Plot D7 — Total band power vs RPM at fixed temperatures (from 10)
# =============================================================================

_tw_dirs = sorted(MECH_DIR.glob("tw_*")) if MECH_DIR.exists() else []
if _tw_dirs:
    # Validation figure (fig:band_power) shows only the retained AE bands; the
    # >1 MHz (1--2 MHz) validation_extra_bands are excluded from feature
    # extraction and modelling, so they are no longer plotted here.
    band_labels = [b["label"] for b in cfg["frequency_bands"]["AE"]]
    _tw_colors = ["#4878CF", "#D65F5F", "#6ACC65"]
    fig, axes = plt.subplots(1, len(band_labels), figsize=(4.4 * len(band_labels), 3.8),
                             sharex=True)
    for ax, label in zip(np.atleast_1d(axes), band_labels):
        for color, tw_dir in zip(_tw_colors, _tw_dirs):
            cs_path = tw_dir / "tables" / "comb_strip.csv"
            if not cs_path.exists():
                continue
            cs = pd.read_csv(cs_path)
            sub = cs[(cs["band"] == label) & (cs["rpm"] >= 60)]
            tw_label = tw_dir.name.replace("tw_", "").replace("C", "°C")
            ax.semilogy(sub["rpm"], sub["p_total"], "o", ms=4.5, color=color,
                        alpha=0.8, label=tw_label)
        ax.set_title(label)
        ax.set_xlabel("RPM")
        ax.grid(ls=":", alpha=0.4)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel(r"Total band power [V$^2$]")
    ax0.legend(fontsize=8, title="temperature")
    fig.tight_layout()
    fig_path = SCRIPT_DIR / "band_power_vs_rpm.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path.name}")
else:
    print("Skipped D7 (run scripts/10_band_mechanism.py first)")

# %%
# =============================================================================
# Entry point
# =============================================================================

print(f"\nAll figures saved to: {SCRIPT_DIR}")

if __name__ == "__main__":
    print("\n06_plots complete.")
