"""
04_evaluation.py
================
Statistical evaluation of the nine model–feature-set combinations.

Pipeline position: 5th script — reads outputs of 03_modelling.py
(features.parquet, metadata.parquet, feature_selection.json, best_params.json,
and model_holdout_*.csv files).

What this script does
---------------------
1. Runs repeated nested cross-validation (R=10 repeats × k=5 folds) on the
   training pool to obtain a *distribution* of performance scores per model.
   - Elastic Net / Polynomial: inner HP selection is re-run per outer fold
     (ElasticNetCV / RidgeCV, fast).
   - LightGBM: uses fixed hyperparameters from best_params.json (no Optuna).
2. Reports mean ± std RMSE from the repeated CV as the primary performance table.
3. Loads the paired holdout predictions saved by 03_modelling.py.
4. Runs two levels of pairwise significance tests:
   - Level 1 (CV scores): corrected repeated k-fold t-test (Nadeau & Bengio 2003)
     with Holm-Bonferroni correction.
   - Level 2 (holdout residuals): Wilcoxon signed-rank + Diebold-Mariano +
     bootstrap CIs on ΔRMSE.
5. Computes SHAP-based cross-model feature agreement tables.
6. Saves all results as CSV tables and summary plots.

Usage
-----
    python scripts/04_evaluation.py
    python scripts/04_evaluation.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm.auto import tqdm

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.evaluation import (
    corrected_repeated_kfold_t_test,
    holm_bonferroni_correction,
    wilcoxon_test,
    diebold_mariano_test,
    bootstrap_rmse_diff_ci,
    cross_model_agreement,
    pairwise_tests_dataframe,
)

warnings.filterwarnings("ignore", category=UserWarning)

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Fast mode: fewer CV repeats and bootstrap resamples for quick sanity checks.",
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

DEBUG: bool = args.debug or bool(cfg.get("debug", {}).get("enabled", False))
if DEBUG:
    print("*** DEBUG MODE — reduced parameters for fast testing ***")

OUTPUT_DIR = get_output_dir(cfg)
D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]
RANDOM_STATE: int = cfg.get("random_state", 42)

model_cfg = cfg.get("modelling", {})
CV_N_SPLITS: int = model_cfg.get("cv_n_splits", 5)
TEST_SIZE: float = model_cfg.get("test_size", 0.2)

_dbg = cfg.get("debug", {})
CV_N_REPEATS: int = 10        # R — number of CV repeats
ALPHA: float = 0.05           # significance level
N_BOOT: int = 10_000          # bootstrap resamples for ΔRMSE CI

if DEBUG:
    CV_N_SPLITS  = _dbg.get("cv_n_splits", 3)
    CV_N_REPEATS = _dbg.get("n_repeats", 2)
    N_BOOT       = _dbg.get("n_boot", 200)
    print(f"  cv_n_splits={CV_N_SPLITS}, n_repeats={CV_N_REPEATS}, n_boot={N_BOOT}")

enet_cfg = model_cfg.get("elastic_net", {})
ENET_ALPHAS: list[float] | None = enet_cfg.get("alphas")
ENET_L1_RATIOS: list[float] = enet_cfg.get("l1_ratios") or [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ENET_MAX_ITER: int = enet_cfg.get("max_iter", 10_000)

poly_cfg = model_cfg.get("polynomial", {})
POLY_DEGREE: int = poly_cfg.get("degree", 2)
POLY_TOP_K: int | None = poly_cfg.get("top_k")
POLY_ALPHAS: list[float] = poly_cfg.get("alphas") or list(np.logspace(-3, 4, 15))

# %%
# =============================================================================
# Load data + reproduce the same holdout split as 03_modelling.py
# =============================================================================

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)

feat_sel_path = OUTPUT_DIR / "feature_selection.json"
if not feat_sel_path.exists():
    raise FileNotFoundError(f"{feat_sel_path} not found. Run 02_feature_analysis.py first.")
with open(feat_sel_path) as fh:
    feature_selection: dict = json.load(fh)

params_path = OUTPUT_DIR / "best_params.json"
if not params_path.exists():
    raise FileNotFoundError(f"{params_path} not found. Run 03_modelling.py first.")
with open(params_path) as fh:
    best_params: dict = json.load(fh)

# Optional feature override (mirrors 03_modelling.py)
FEATURE_OVERRIDE: dict[str, list[str]] = cfg.get("feature_override") or {}

df, metadata = filter_by_metadata(raw_feature_df, raw_metadata_df, rpm_max=RPM_MAX)
df = df.reset_index(drop=True)
metadata = metadata.reset_index(drop=True)

metadata["kappa"] = metadata.apply(
    lambda row: calculate_kappa(
        rpm=row["rpm"],
        temp_c=row["temperature_c"],
        d_pw=D_PW_MM,
        nu_40=row["viscosity_40c_cst"],
        nu_100=row["viscosity_100c_cst"],
    ),
    axis=1,
)

# Reproduce the identical 80/20 sweep-level split from 03_modelling.py
from sklearn.model_selection import train_test_split

sweep_keys = df[["file", "sweep"]].drop_duplicates().reset_index(drop=True)
train_sweep_idx, _ = train_test_split(
    np.arange(len(sweep_keys)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)
train_sweeps = set(
    zip(sweep_keys.iloc[train_sweep_idx]["file"], sweep_keys.iloc[train_sweep_idx]["sweep"])
)
row_in_train = df.apply(lambda r: (r["file"], r["sweep"]) in train_sweeps, axis=1)
train_idx = np.where(row_in_train)[0]

df_train = df.iloc[train_idx].reset_index(drop=True)
meta_train = metadata.iloc[train_idx].reset_index(drop=True)

print(f"Training pool: {len(df_train)} rows ({len(train_sweep_idx)} sweeps)")

# %%
# =============================================================================
# Prepare per-sensor training datasets
# =============================================================================

sensor_train: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

for sensor_name, sel_info in feature_selection.items():
    retained = FEATURE_OVERRIDE.get(sensor_name) or sel_info["retained"]
    tr_mask = df_train["sensor"] == sensor_name
    X_tr = df_train.loc[tr_mask, retained].reset_index(drop=True)
    y_tr = meta_train.loc[tr_mask, "kappa"].reset_index(drop=True)
    valid = X_tr.notna().all(axis=1)
    X_tr = X_tr[valid].reset_index(drop=True)
    y_tr = y_tr[valid].values
    sensor_train[sensor_name] = (X_tr, y_tr)
    print(f"  {sensor_name}: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")


def _build_keyed(df_src, meta_src, sensor_name, retained):
    mask = df_src["sensor"] == sensor_name
    X = df_src.loc[mask, ["file", "sweep"] + retained].copy()
    km = meta_src.loc[mask, ["kappa"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    X["kappa"] = km["kappa"].values
    valid = X[retained].notna().all(axis=1)
    X = X[valid].set_index(["file", "sweep"])
    return X.rename(columns=lambda c: f"{sensor_name}__{c}" if c != "kappa" else c)


retained_map = {
    sname: (FEATURE_OVERRIDE.get(sname) or feature_selection[sname]["retained"])
    for sname in feature_selection
}

X_combined_train: pd.DataFrame | None = None
y_combined_train: np.ndarray | None = None

try:
    parts = [_build_keyed(df_train, meta_train, sname, ret) for sname, ret in retained_map.items()]
    merged = parts[0]
    for part in parts[1:]:
        feat_cols = [c for c in part.columns if c != "kappa"]
        merged = merged.join(part[feat_cols], how="inner")
    feat_cols = [c for c in merged.columns if c != "kappa"]
    X_combined_train = pd.DataFrame(merged[feat_cols].values, columns=feat_cols)
    y_combined_train = merged["kappa"].values
    print(f"  Combined: {X_combined_train.shape[0]} samples, {X_combined_train.shape[1]} features")
except Exception as exc:
    print(f"WARNING: Could not build combined training set: {exc}")

# %%
# =============================================================================
# Repeated nested CV helpers
# =============================================================================


def _enet_fold_score(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, val: np.ndarray) -> float:
    """One outer fold of nested Elastic Net CV — returns RMSE on val."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X.values[tr])
    X_val_s = scaler.transform(X.values[val])
    inner = ElasticNetCV(
        l1_ratio=ENET_L1_RATIOS,
        alphas=ENET_ALPHAS,
        cv=5,
        max_iter=ENET_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    inner.fit(X_tr_s, y[tr])
    model = ElasticNet(alpha=inner.alpha_, l1_ratio=inner.l1_ratio_, max_iter=ENET_MAX_ITER)
    model.fit(X_tr_s, y[tr])
    y_pred = model.predict(X_val_s)
    return float(np.sqrt(mean_squared_error(y[val], y_pred)))


def _poly_fold_score(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, val: np.ndarray) -> float:
    """One outer fold of nested Polynomial (Ridge) CV — returns RMSE on val."""
    X_vals = X.values
    if POLY_TOP_K is not None and POLY_TOP_K < X.shape[1]:
        corr = np.abs([np.corrcoef(X_vals[tr, j], y[tr])[0, 1] for j in range(X.shape[1])])
        top_idx = np.argsort(corr)[::-1][:POLY_TOP_K]
        X_vals = X_vals[:, top_idx]

    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    X_tr_poly = poly.fit_transform(scaler.fit_transform(X_vals[tr]))
    X_val_poly = poly.transform(scaler.transform(X_vals[val]))

    inner_ridge = RidgeCV(alphas=POLY_ALPHAS)
    inner_ridge.fit(X_tr_poly, y[tr])
    model = Ridge(alpha=float(inner_ridge.alpha_))
    model.fit(X_tr_poly, y[tr])
    y_pred = model.predict(X_val_poly)
    return float(np.sqrt(mean_squared_error(y[val], y_pred)))


def _lgb_fold_score(
    X: pd.DataFrame,
    y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    params: dict,
) -> float:
    """One outer fold of LightGBM with fixed params — returns RMSE on val.

    Early stopping is intentionally disabled: the val set should only be used
    for evaluation, not for stopping decisions.  n_estimators comes from the
    best_params.json selected by 03_modelling.py.
    """
    model = lgb.LGBMRegressor(**params)
    model.fit(X.iloc[tr], y[tr], callbacks=[lgb.log_evaluation(period=0)])
    y_pred = model.predict(X.iloc[val])
    return float(np.sqrt(mean_squared_error(y[val], y_pred)))


def repeated_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    fold_score_fn,
    n_splits: int = CV_N_SPLITS,
    n_repeats: int = CV_N_REPEATS,
    desc: str = "",
) -> np.ndarray:
    """Run R×k repeated nested CV; return array of shape (R*k,) RMSE scores."""
    scores = []
    for r in tqdm(range(n_repeats), desc=desc or "repeats", leave=False):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + r)
        for tr_idx, val_idx in cv.split(X):
            scores.append(fold_score_fn(X, y, tr_idx, val_idx))
    return np.array(scores)


# %%
# =============================================================================
# Run repeated nested CV for all 9 model–feature-set combinations
# =============================================================================

# Model names must match the keys in best_params.json (from 03_modelling.py)
# Pattern: {ModelType}_{SensorName} e.g. "ElasticNet_AE", "LightGBM_Combined"

cv_scores: dict[str, np.ndarray] = {}  # model_name → R*k RMSE scores

sensor_names = list(feature_selection.keys())

# --- Elastic Net ---
for sensor_name in sensor_names:
    X_tr, y_tr = sensor_train[sensor_name]
    model_name = f"ElasticNet_{sensor_name}"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_tr, y_tr,
        fold_score_fn=_enet_fold_score,
        desc=model_name,
    )

if X_combined_train is not None:
    model_name = "ElasticNet_Combined"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_combined_train, y_combined_train,
        fold_score_fn=_enet_fold_score,
        desc=model_name,
    )

# --- Polynomial ---
for sensor_name in sensor_names:
    X_tr, y_tr = sensor_train[sensor_name]
    model_name = f"Polynomial_{sensor_name}"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_tr, y_tr,
        fold_score_fn=_poly_fold_score,
        desc=model_name,
    )

if X_combined_train is not None:
    model_name = "Polynomial_Combined"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_combined_train, y_combined_train,
        fold_score_fn=_poly_fold_score,
        desc=model_name,
    )

# --- LightGBM (fixed params from best_params.json) ---
_lgb_base = dict(
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbosity=-1,
)
_lgb_keep_keys = {
    "n_estimators", "learning_rate", "max_depth", "num_leaves",
    "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
}

for sensor_name in sensor_names:
    X_tr, y_tr = sensor_train[sensor_name]
    model_name = f"LightGBM_{sensor_name}"
    lgb_params = {**_lgb_base, **{k: v for k, v in best_params.get(model_name, {}).items()
                                   if k in _lgb_keep_keys}}
    print(f"\nRepeated nested CV: {model_name}  params={lgb_params}")
    cv_scores[model_name] = repeated_nested_cv(
        X_tr, y_tr,
        fold_score_fn=lambda X, y, tr, val, p=lgb_params: _lgb_fold_score(X, y, tr, val, p),
        desc=model_name,
    )

if X_combined_train is not None:
    model_name = "LightGBM_Combined"
    lgb_params = {**_lgb_base, **{k: v for k, v in best_params.get(model_name, {}).items()
                                   if k in _lgb_keep_keys}}
    print(f"\nRepeated nested CV: {model_name}  params={lgb_params}")
    cv_scores[model_name] = repeated_nested_cv(
        X_combined_train, y_combined_train,
        fold_score_fn=lambda X, y, tr, val, p=lgb_params: _lgb_fold_score(X, y, tr, val, p),
        desc=model_name,
    )

# Save raw CV score distributions
cv_scores_df = pd.DataFrame(cv_scores)
cv_scores_path = OUTPUT_DIR / "repeated_cv_scores.csv"
cv_scores_df.to_csv(cv_scores_path, index=False)
print(f"\nSaved: {cv_scores_path.name}")

# %%
# =============================================================================
# Performance summary table — mean ± std RMSE from repeated nested CV
# =============================================================================

perf_rows = []
for model_name, scores in cv_scores.items():
    parts = model_name.rsplit("_", 1)
    model_type = parts[0] if len(parts) == 2 else model_name
    feature_set = parts[1] if len(parts) == 2 else "?"
    perf_rows.append({
        "model": model_name,
        "model_type": model_type,
        "feature_set": feature_set,
        "mean_rmse": float(scores.mean()),
        "std_rmse": float(scores.std(ddof=1)),
        "mean_minus_std": float(scores.mean() - scores.std(ddof=1)),
        "n_scores": len(scores),
    })

perf_df = pd.DataFrame(perf_rows).sort_values("mean_rmse")
perf_path = OUTPUT_DIR / "performance_table_cv.csv"
perf_df.to_csv(perf_path, index=False)
print(f"Saved: {perf_path.name}")

print("\n" + "=" * 70)
print("PERFORMANCE (repeated nested CV, mean ± std RMSE)")
print("=" * 70)
for _, row in perf_df.iterrows():
    print(f"  {row['model']:35s}  {row['mean_rmse']:.4f} ± {row['std_rmse']:.4f}")

# %%
# =============================================================================
# Load holdout predictions from 03_modelling.py
# =============================================================================

holdout_y_true: dict[str, np.ndarray] = {}
holdout_y_pred: dict[str, np.ndarray] = {}
holdout_residuals: dict[str, np.ndarray] = {}  # y_true - y_pred

for model_name in cv_scores:
    ho_path = OUTPUT_DIR / f"model_holdout_{model_name.lower()}.csv"
    if not ho_path.exists():
        print(f"WARNING: {ho_path.name} not found — skipping holdout tests for {model_name}")
        continue
    ho_df = pd.read_csv(ho_path)
    holdout_y_true[model_name] = ho_df["y_true"].values
    holdout_y_pred[model_name] = ho_df["y_pred"].values
    holdout_residuals[model_name] = ho_df["residual"].values

print(f"\nLoaded holdout predictions for {len(holdout_y_true)} models")

# %%
# =============================================================================
# Define comparison pairs
# =============================================================================

# Within each feature set — model architecture comparison (3 pairs × 3 sets)
model_types = ["ElasticNet", "Polynomial", "LightGBM"]
feature_sets = sensor_names + (["Combined"] if X_combined_train is not None else [])

within_pairs: list[tuple[str, str]] = []
for fs in feature_sets:
    names = [f"{mt}_{fs}" for mt in model_types]
    available = [n for n in names if n in cv_scores]
    within_pairs.extend(combinations(available, 2))

# Cross-feature-set — best model per feature set (best = lowest mean CV RMSE)
best_per_fs: dict[str, str] = {}
for fs in feature_sets:
    candidates = {n: cv_scores[n].mean() for n in cv_scores if n.endswith(f"_{fs}")}
    if candidates:
        best_per_fs[fs] = min(candidates, key=candidates.get)

cross_pairs: list[tuple[str, str]] = []
fs_list = list(best_per_fs.keys())
for fs_a, fs_b in combinations(fs_list, 2):
    cross_pairs.append((best_per_fs[fs_a], best_per_fs[fs_b]))

print("\nWithin-feature-set pairs:")
for a, b in within_pairs:
    print(f"  {a}  vs  {b}")
print("\nCross-feature-set pairs (best per set):")
for a, b in cross_pairs:
    print(f"  {a}  vs  {b}")

# %%
# =============================================================================
# Level 1 — CV-score significance tests (within feature sets)
# =============================================================================

# For within-feature-set pairs: holdout sets align within a feature set.
# Pass holdout_y_true as a dict — pairwise_tests_dataframe looks up model_a's y_true per pair.
within_results = pairwise_tests_dataframe(
    cv_scores=cv_scores,
    holdout_residuals=holdout_residuals,
    holdout_y_true=holdout_y_true,
    holdout_y_pred=holdout_y_pred,
    pairs=within_pairs,
    k=CV_N_SPLITS,
    alpha=ALPHA,
    n_boot=N_BOOT,
)
within_results.insert(0, "comparison_type", "within_feature_set")
within_results_path = OUTPUT_DIR / "stat_tests_within_featureset.csv"
within_results.to_csv(within_results_path, index=False)
print(f"\nSaved: {within_results_path.name}")

# %%
# =============================================================================
# Level 1+2 — Cross-feature-set tests
# =============================================================================

# Cross-feature-set: holdout sets may differ in size for different sensors,
# so only run observation-level tests for models sharing the same holdout set.
# Within this experiment all sensor models share the same sweeps (inner join),
# but combined models have potentially fewer windows.

def _shared_true(model_a: str, model_b: str) -> np.ndarray | None:
    ya = holdout_y_true.get(model_a)
    yb = holdout_y_true.get(model_b)
    if ya is not None and yb is not None and len(ya) == len(yb) and np.allclose(ya, yb):
        return ya
    return None


cross_rows = []
raw_p_cross_cv = []
for model_a, model_b in cross_pairs:
    row: dict = {
        "comparison_type": "cross_feature_set",
        "model_a": model_a,
        "model_b": model_b,
    }
    # CV level
    if model_a in cv_scores and model_b in cv_scores:
        t_stat, p_cv = corrected_repeated_kfold_t_test(
            cv_scores[model_a], cv_scores[model_b], k=CV_N_SPLITS
        )
        row["cv_t_stat"] = t_stat
        row["cv_p_value"] = p_cv
    raw_p_cross_cv.append(row.get("cv_p_value", np.nan))

    # Holdout level — only if y_true arrays match
    shared_true = _shared_true(model_a, model_b)
    if shared_true is not None:
        e_a = holdout_residuals[model_a]
        e_b = holdout_residuals[model_b]
        w_stat, p_wilcox = wilcoxon_test(e_a, e_b)
        dm_stat, p_dm = diebold_mariano_test(e_a, e_b)
        mean_diff, ci_lo, ci_hi = bootstrap_rmse_diff_ci(
            shared_true, holdout_y_pred[model_a], holdout_y_pred[model_b], n_boot=N_BOOT
        )
        row.update({
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcox,
            "dm_stat": dm_stat,
            "dm_p": p_dm,
            "delta_rmse": mean_diff,
            f"ci_{int((1-ALPHA)*100)}_lo": ci_lo,
            f"ci_{int((1-ALPHA)*100)}_hi": ci_hi,
        })
    cross_rows.append(row)

cross_results = pd.DataFrame(cross_rows)

# Holm-Bonferroni on cross-pair CV p-values
valid = ~pd.isna(cross_results.get("cv_p_value", pd.Series(dtype=float)))
if valid.any():
    reject, _ = holm_bonferroni_correction(
        cross_results.loc[valid, "cv_p_value"].values, alpha=ALPHA
    )
    cross_results.loc[valid, "cv_reject_holm"] = reject

cross_results_path = OUTPUT_DIR / "stat_tests_cross_featureset.csv"
cross_results.to_csv(cross_results_path, index=False)
print(f"Saved: {cross_results_path.name}")

# %%
# =============================================================================
# Combined significance table (all pairs)
# =============================================================================

all_results = pd.concat([within_results, cross_results], ignore_index=True)
all_path = OUTPUT_DIR / "stat_tests_all.csv"
all_results.to_csv(all_path, index=False)
print(f"Saved: {all_path.name}")

# %%
# =============================================================================
# Cross-model feature agreement (SHAP-based)
# =============================================================================

for fs in feature_sets:
    imp_map: dict[str, pd.Series] = {}
    for mt in model_types:
        tag = f"{mt}_{fs}".lower()
        imp_path = OUTPUT_DIR / f"shap_importance_{tag}.csv"
        if imp_path.exists():
            imp_series = pd.read_csv(imp_path, index_col=0, header=None).iloc[:, 0]
            imp_map[mt] = imp_series
    if len(imp_map) < 2:
        continue
    top_k = min(10, min(len(s) for s in imp_map.values()))
    agree_df = cross_model_agreement(imp_map, top_k=top_k)
    agree_path = OUTPUT_DIR / f"shap_agreement_{fs.lower()}.csv"
    agree_df.to_csv(agree_path, index=False)
    print(f"Saved: {agree_path.name}")

# %%
# =============================================================================
# Figure E1 — CV score distributions (violin plot per feature set)
# =============================================================================

for fs in feature_sets:
    names = [f"{mt}_{fs}" for mt in model_types]
    available = [n for n in names if n in cv_scores]
    if not available:
        continue

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [cv_scores[n] for n in available]
    labels = [n.replace(f"_{fs}", "") for n in available]
    vp = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=True)
    for body in vp["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE (outer CV fold)")
    ax.set_title(f"CV score distribution — {fs} feature set\n"
                 f"(R={CV_N_REPEATS} repeats × k={CV_N_SPLITS} folds)")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig_path = OUTPUT_DIR / f"eval_cv_distribution_{fs.lower()}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Saved: {fig_path.name}")

# %%
# =============================================================================
# Figure E2 — Mean ± std RMSE comparison (grouped bar chart)
# =============================================================================

fig, ax = plt.subplots(figsize=(max(8, len(perf_df) * 0.9), 5))
x = np.arange(len(perf_df))
bars = ax.bar(x, perf_df["mean_rmse"], yerr=perf_df["std_rmse"],
              capsize=4, color="#4878CF", alpha=0.8, error_kw={"linewidth": 1.2})
ax.set_xticks(x)
ax.set_xticklabels(perf_df["model"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Mean RMSE (repeated nested CV)")
ax.set_title(f"Model performance — mean ± std RMSE\n"
             f"(R={CV_N_REPEATS}×k={CV_N_SPLITS} outer folds)")
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
bar_path = OUTPUT_DIR / "eval_performance_bar.png"
plt.savefig(bar_path, dpi=150)
plt.show()
print(f"Saved: {bar_path.name}")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print("\n--- Performance (mean ± std RMSE, repeated nested CV) ---")
for _, row in perf_df.iterrows():
    print(f"  {row['model']:35s}  {row['mean_rmse']:.4f} ± {row['std_rmse']:.4f}")

print("\n--- Within-feature-set significance (Nadeau-Bengio, Holm-Bonferroni) ---")
for _, row in within_results.iterrows():
    flag = "*" if row.get("cv_reject_holm", False) else " "
    p = row.get("cv_p_value", float("nan"))
    print(f"  {flag} {row['model_a']:30s} vs {row['model_b']:30s}  p={p:.4f}")

print("\n--- Cross-feature-set significance ---")
for _, row in cross_results.iterrows():
    flag = "*" if row.get("cv_reject_holm", False) else " "
    p = row.get("cv_p_value", float("nan"))
    diff = row.get("delta_rmse", float("nan"))
    ci_lo = row.get("ci_95_lo", float("nan"))
    ci_hi = row.get("ci_95_hi", float("nan"))
    print(f"  {flag} {row['model_a']:30s} vs {row['model_b']:30s}  "
          f"p={p:.4f}  ΔRMSE={diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n04_evaluation complete.")
