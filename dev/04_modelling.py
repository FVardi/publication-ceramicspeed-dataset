"""
04_modelling.py
===============
Train and evaluate regression models for predicting kappa from acoustic
emission and ultrasound features.

Pipeline position: 4th script — reads features.parquet + metadata.parquet
from 01_feature_generation and feature_selection.json from 02_feature_analysis.

Evaluation strategy
-------------------
1. 80/20 random hold-out split (stratified by sensor).
2. Nested KFold CV on the 80% training set:
   - Outer loop: unbiased out-of-fold evaluation.
   - Inner loop: hyperparameter selection per outer fold.
3. Final model refit on the full training set.
4. Final evaluation on the 20% hold-out test set.

Models: Elastic Net, Bayesian Ridge, LightGBM (per-sensor + combined).

Usage
-----
    python 04_modelling.py
    python 04_modelling.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from own_utils.loading import load_parquet_pair
from own_utils.cleaning import filter_by_metadata
from own_utils.calculate_kappa import calculate_kappa
from own_utils.config import load_config, get_output_dir
from own_utils.modelling import (
    train_elastic_net_cv,
    train_bayesian_ridge_cv,
    train_polynomial_cv,
    train_lightgbm_cv,
    evaluate_on_holdout,
    results_summary_table,
    get_feature_weights,
    ModelResult,
)
from own_utils.visualization import (
    plot_coefficients,
    plot_coefficients_log,
    plot_cv_fold_metrics,
    plot_residuals,
    plot_bayesian_uncertainty,
)

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
D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]
RANDOM_STATE: int = cfg.get("random_state", 42)

# Modelling config
model_cfg = cfg.get("modelling", {})
CV_N_SPLITS: int = model_cfg.get("cv_n_splits", 5)
TEST_SIZE: float = model_cfg.get("test_size", 0.2)

enet_cfg = model_cfg.get("elastic_net", {})
ENET_ALPHAS: list[float] | None = enet_cfg.get("alphas")
ENET_L1_RATIOS: list[float] | None = enet_cfg.get("l1_ratios")
ENET_MAX_ITER: int = enet_cfg.get("max_iter", 10_000)

bayes_cfg = model_cfg.get("bayesian_ridge", {})
BAYES_MAX_ITER: int = bayes_cfg.get("max_iter", 300)

poly_cfg = model_cfg.get("polynomial", {})
POLY_DEGREE: int = poly_cfg.get("degree", 2)
POLY_ALPHAS: list[float] | None = poly_cfg.get("alphas")

lgb_cfg = model_cfg.get("lightgbm", {})
LGB_N_ESTIMATORS: int = lgb_cfg.get("n_estimators", 500)
LGB_LEARNING_RATE: float = lgb_cfg.get("learning_rate", 0.05)
LGB_MAX_DEPTH: int = lgb_cfg.get("max_depth", 6)
LGB_NUM_LEAVES: int = lgb_cfg.get("num_leaves", 31)
LGB_MIN_CHILD_SAMPLES: int = lgb_cfg.get("min_child_samples", 10)
LGB_SUBSAMPLE: float = lgb_cfg.get("subsample", 0.8)
LGB_COLSAMPLE: float = lgb_cfg.get("colsample_bytree", 0.8)
LGB_REG_ALPHA: float = lgb_cfg.get("reg_alpha", 0.0)
LGB_REG_LAMBDA: float = lgb_cfg.get("reg_lambda", 1.0)
LGB_EARLY_STOP: int = lgb_cfg.get("early_stopping_rounds", 50)


# %%
# =============================================================================
# Load intermediate data
# =============================================================================

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)

# Load retained feature lists from 02_feature_analysis
feat_sel_path = OUTPUT_DIR / "feature_selection.json"
if not feat_sel_path.exists():
    raise FileNotFoundError(
        f"{feat_sel_path} not found. Run 02_feature_analysis.py first."
    )
with open(feat_sel_path) as fh:
    feature_selection = json.load(fh)

print(f"Loaded {len(raw_feature_df)} rows from features.parquet")
for sensor, info in feature_selection.items():
    print(f"  {sensor}: {len(info['retained'])} / {len(info['all_columns'])} features retained")

# Optional per-sensor feature override from config.yaml
FEATURE_OVERRIDE: dict[str, list[str]] = cfg.get("feature_override") or {}

# %%
# =============================================================================
# Filtering + kappa calculation
# =============================================================================

df, metadata = filter_by_metadata(
    raw_feature_df, raw_metadata_df, rpm_max=RPM_MAX
)
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

print(f"After RPM filter (<{RPM_MAX}): {len(df)} rows")
print(f"Kappa range: [{metadata['kappa'].min():.2f}, {metadata['kappa'].max():.2f}]")

# %%
# =============================================================================
# Train / hold-out split (80/20)
# =============================================================================

# Split at the row level — same indices apply to both df and metadata.
all_indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)
meta_train = metadata.iloc[train_idx].reset_index(drop=True)
meta_test = metadata.iloc[test_idx].reset_index(drop=True)

print(f"\nTrain / hold-out split: {len(df_train)} train, {len(df_test)} test "
      f"({TEST_SIZE:.0%} hold-out)")
print(f"  Train kappa: [{meta_train['kappa'].min():.2f}, {meta_train['kappa'].max():.2f}]")
print(f"  Test  kappa: [{meta_test['kappa'].min():.2f}, {meta_test['kappa'].max():.2f}]")

# %%
# =============================================================================
# Prepare per-sensor datasets (train + test)
# =============================================================================

sensor_train: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
sensor_test: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

for sensor_name, sel_info in feature_selection.items():
    if sensor_name in FEATURE_OVERRIDE:
        retained = FEATURE_OVERRIDE[sensor_name]
        print(f"\n{sensor_name}: using {len(retained)} override features (from config)")
    else:
        retained = sel_info["retained"]
        print(f"\n{sensor_name}: using {len(retained)} auto-retained features")

    # Train split
    tr_mask = df_train["sensor"] == sensor_name
    X_tr = df_train.loc[tr_mask, retained].reset_index(drop=True)
    y_tr = meta_train.loc[tr_mask, "kappa"].values

    # Test split
    te_mask = df_test["sensor"] == sensor_name
    X_te = df_test.loc[te_mask, retained].reset_index(drop=True)
    y_te = meta_test.loc[te_mask, "kappa"].values

    sensor_train[sensor_name] = (X_tr, y_tr)
    sensor_test[sensor_name] = (X_te, y_te)

    print(f"  Features: {retained}")
    print(f"  Train: {X_tr.shape}  |  Test: {X_te.shape}")

# %%
# =============================================================================
# Build combined feature matrices (train + test)
# =============================================================================

combined_train_parts: list[pd.DataFrame] = []
combined_test_parts: list[pd.DataFrame] = []
combined_y_train_parts: list[np.ndarray] = []
combined_y_test_parts: list[np.ndarray] = []

for sensor_name in feature_selection:
    X_tr, y_tr = sensor_train[sensor_name]
    X_te, y_te = sensor_test[sensor_name]
    combined_train_parts.append(
        X_tr.rename(columns=lambda c: f"{sensor_name}__{c}").reset_index(drop=True)
    )
    combined_test_parts.append(
        X_te.rename(columns=lambda c: f"{sensor_name}__{c}").reset_index(drop=True)
    )
    combined_y_train_parts.append(y_tr)
    combined_y_test_parts.append(y_te)

if len(combined_y_train_parts[0]) == len(combined_y_train_parts[1]):
    X_combined_train = pd.concat(combined_train_parts, axis=1)
    y_combined_train = combined_y_train_parts[0]
    X_combined_test = pd.concat(combined_test_parts, axis=1)
    y_combined_test = combined_y_test_parts[0]
else:
    print("WARNING: Sensor row counts differ. Skipping combined models.")
    X_combined_train = None

# %%
# =============================================================================
# Helper: train + evaluate one model
# =============================================================================


def _train_and_eval(train_fn, X_tr, y_tr, X_te, y_te, **kwargs) -> ModelResult:
    """Train via CV on training set, then evaluate on hold-out."""
    result = train_fn(X_tr, y_tr, **kwargs)
    evaluate_on_holdout(result, X_te, y_te)
    print(result.summary())
    return result


# %%
# =============================================================================
# Train all models
# =============================================================================

results: list[ModelResult] = []

# --- Elastic Net (per-sensor) ------------------------------------------------
for sensor_name in feature_selection:
    X_tr, y_tr = sensor_train[sensor_name]
    X_te, y_te = sensor_test[sensor_name]

    print(f"\n{'='*60}")
    print(f"Training Elastic Net — {sensor_name}")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_elastic_net_cv, X_tr, y_tr, X_te, y_te,
        n_splits=CV_N_SPLITS,
        alphas=ENET_ALPHAS,
        l1_ratios=ENET_L1_RATIOS,
        max_iter=ENET_MAX_ITER,
        random_state=RANDOM_STATE,
        name=f"ElasticNet_{sensor_name}",
        sensor=sensor_name,
    )
    results.append(result)

# --- Elastic Net (combined) ---------------------------------------------------
if X_combined_train is not None:
    print(f"\n{'='*60}")
    print("Training Elastic Net — Combined (AE + Ultrasound)")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_elastic_net_cv,
        X_combined_train, y_combined_train,
        X_combined_test, y_combined_test,
        n_splits=CV_N_SPLITS,
        alphas=ENET_ALPHAS,
        l1_ratios=ENET_L1_RATIOS,
        max_iter=ENET_MAX_ITER,
        random_state=RANDOM_STATE,
        name="ElasticNet_Combined",
        sensor="combined",
    )
    results.append(result)

# --- Bayesian Ridge (per-sensor) ----------------------------------------------
for sensor_name in feature_selection:
    X_tr, y_tr = sensor_train[sensor_name]
    X_te, y_te = sensor_test[sensor_name]

    print(f"\n{'='*60}")
    print(f"Training Bayesian Ridge — {sensor_name}")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_bayesian_ridge_cv, X_tr, y_tr, X_te, y_te,
        n_splits=CV_N_SPLITS,
        max_iter=BAYES_MAX_ITER,
        random_state=RANDOM_STATE,
        name=f"BayesianRidge_{sensor_name}",
        sensor=sensor_name,
    )
    results.append(result)

# --- Bayesian Ridge (combined) ------------------------------------------------
if X_combined_train is not None:
    print(f"\n{'='*60}")
    print("Training Bayesian Ridge — Combined (AE + Ultrasound)")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_bayesian_ridge_cv,
        X_combined_train, y_combined_train,
        X_combined_test, y_combined_test,
        n_splits=CV_N_SPLITS,
        max_iter=BAYES_MAX_ITER,
        random_state=RANDOM_STATE,
        name="BayesianRidge_Combined",
        sensor="combined",
    )
    results.append(result)

# --- Polynomial Regression (per-sensor) --------------------------------------
for sensor_name in feature_selection:
    X_tr, y_tr = sensor_train[sensor_name]
    X_te, y_te = sensor_test[sensor_name]

    print(f"\n{'='*60}")
    print(f"Training Polynomial Regression (degree={POLY_DEGREE}) — {sensor_name}")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_polynomial_cv, X_tr, y_tr, X_te, y_te,
        degree=POLY_DEGREE,
        n_splits=CV_N_SPLITS,
        alphas=POLY_ALPHAS,
        random_state=RANDOM_STATE,
        name=f"Polynomial_{sensor_name}",
        sensor=sensor_name,
    )
    results.append(result)

# --- Polynomial Regression (combined) ----------------------------------------
if X_combined_train is not None:
    print(f"\n{'='*60}")
    print(f"Training Polynomial Regression (degree={POLY_DEGREE}) — Combined (AE + Ultrasound)")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_polynomial_cv,
        X_combined_train, y_combined_train,
        X_combined_test, y_combined_test,
        degree=POLY_DEGREE,
        n_splits=CV_N_SPLITS,
        alphas=POLY_ALPHAS,
        random_state=RANDOM_STATE,
        name="Polynomial_Combined",
        sensor="combined",
    )
    results.append(result)

# --- LightGBM (per-sensor) ---------------------------------------------------
for sensor_name in feature_selection:
    X_tr, y_tr = sensor_train[sensor_name]
    X_te, y_te = sensor_test[sensor_name]

    print(f"\n{'='*60}")
    print(f"Training LightGBM — {sensor_name}")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_lightgbm_cv, X_tr, y_tr, X_te, y_te,
        n_splits=CV_N_SPLITS,
        n_estimators=LGB_N_ESTIMATORS,
        learning_rate=LGB_LEARNING_RATE,
        max_depth=LGB_MAX_DEPTH,
        num_leaves=LGB_NUM_LEAVES,
        min_child_samples=LGB_MIN_CHILD_SAMPLES,
        subsample=LGB_SUBSAMPLE,
        colsample_bytree=LGB_COLSAMPLE,
        reg_alpha=LGB_REG_ALPHA,
        reg_lambda=LGB_REG_LAMBDA,
        early_stopping_rounds=LGB_EARLY_STOP,
        random_state=RANDOM_STATE,
        name=f"LightGBM_{sensor_name}",
        sensor=sensor_name,
    )
    results.append(result)

# --- LightGBM (combined) -----------------------------------------------------
if X_combined_train is not None:
    print(f"\n{'='*60}")
    print("Training LightGBM — Combined (AE + Ultrasound)")
    print(f"{'='*60}")

    result = _train_and_eval(
        train_lightgbm_cv,
        X_combined_train, y_combined_train,
        X_combined_test, y_combined_test,
        n_splits=CV_N_SPLITS,
        n_estimators=LGB_N_ESTIMATORS,
        learning_rate=LGB_LEARNING_RATE,
        max_depth=LGB_MAX_DEPTH,
        num_leaves=LGB_NUM_LEAVES,
        min_child_samples=LGB_MIN_CHILD_SAMPLES,
        subsample=LGB_SUBSAMPLE,
        colsample_bytree=LGB_COLSAMPLE,
        reg_alpha=LGB_REG_ALPHA,
        reg_lambda=LGB_REG_LAMBDA,
        early_stopping_rounds=LGB_EARLY_STOP,
        random_state=RANDOM_STATE,
        name="LightGBM_Combined",
        sensor="combined",
    )
    results.append(result)

# %%
# =============================================================================
# Summary comparison table
# =============================================================================

summary_df = results_summary_table(results)
print("\n" + "=" * 80)
print("MODEL COMPARISON  (CV = cross-validation on train set, HO = hold-out test set)")
print("=" * 80)
print(summary_df.to_string(index=False))

summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")

# %%
# =============================================================================
# Figure M1 — Predicted vs actual: CV (train) and hold-out (test)
# =============================================================================

import math

n_models = len(results)

# RPM arrays for the holdout set, keyed by sensor name
_holdout_rpm: dict[str, np.ndarray] = {}
for _sname in feature_selection:
    _te_mask = df_test["sensor"] == _sname
    _holdout_rpm[_sname] = meta_test.loc[_te_mask, "rpm"].values
# combined model shares row order with the first sensor
_first_sensor = next(iter(feature_selection))
_holdout_rpm["combined"] = _holdout_rpm[_first_sensor]

# CV (training) RPM, keyed by sensor name
_cv_rpm: dict[str, np.ndarray] = {}
for _sname in feature_selection:
    _tr_mask = df_train["sensor"] == _sname
    _cv_rpm[_sname] = meta_train.loc[_tr_mask, "rpm"].values
_cv_rpm["combined"] = _cv_rpm[_first_sensor]
_ncols = min(3, n_models)
_nrows = math.ceil(n_models / _ncols)

# CV predictions (out-of-fold on training set)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 6 * _nrows), squeeze=False)
_axes_flat = [axes[r][c] for r in range(_nrows) for c in range(_ncols)]
for ax in _axes_flat[n_models:]:
    ax.set_visible(False)
for ax, result in zip(_axes_flat, results):
    _rpm = _cv_rpm.get(result.sensor, None)
    sc = ax.scatter(result.y_true, result.y_pred, c=_rpm, cmap="viridis",
                    s=14, alpha=0.6, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="RPM")
    lims = [min(result.y_true.min(), result.y_pred.min()),
            max(result.y_true.max(), result.y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="ideal")
    ax.set_xlabel("True κ")
    ax.set_ylabel("Predicted κ")
    ax.set_title(f"{result.name}\nR² = {result.r2:.3f}   MAE = {result.mae:.3f}   RMSE = {result.rmse:.3f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(ls=":", alpha=0.4)
fig.suptitle("CV Out-of-Fold Predictions vs True κ (Training Set)", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "model_pred_vs_actual_cv.png", dpi=150)
plt.show()
print("Saved: model_pred_vs_actual_cv.png")

# Hold-out predictions
fig, axes = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 6 * _nrows), squeeze=False)
_axes_flat = [axes[r][c] for r in range(_nrows) for c in range(_ncols)]
for ax in _axes_flat[n_models:]:
    ax.set_visible(False)
for ax, result in zip(_axes_flat, results):
    if result.holdout_y_true is not None:
        _rpm = _holdout_rpm.get(result.sensor, None)
        sc = ax.scatter(result.holdout_y_true, result.holdout_y_pred,
                        c=_rpm, cmap="viridis", s=14, alpha=0.6, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="RPM")
        lims = [
            min(result.holdout_y_true.min(), result.holdout_y_pred.min()),
            max(result.holdout_y_true.max(), result.holdout_y_pred.max()),
        ]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="ideal")
        h = result.holdout_metrics
        ax.set_title(f"{result.name}\nR² = {h['r2']:.3f}   MAE = {h['mae']:.3f}   RMSE = {h['rmse']:.3f}")
        ax.set_xlabel("True κ")
        ax.set_ylabel("Predicted κ")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(ls=":", alpha=0.4)
fig.suptitle("Hold-Out Test Set: Predicted vs True κ", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "model_pred_vs_actual_holdout.png", dpi=150)
plt.show()
print("Saved: model_pred_vs_actual_holdout.png")

# %%
# =============================================================================
# Figure M2 — Coefficient bar charts (log scale, per model)
# =============================================================================

for result in results:
    n_features = len(result.feature_names)
    fig, ax = plot_coefficients_log(result, top_n=min(20, n_features))
    fig.tight_layout()
    fname = f"model_coefs_log_{result.name.lower()}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

# %%
# =============================================================================
# Figure M3 — CV fold metrics (R² per fold)
# =============================================================================

fig, axes = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 4 * _nrows), squeeze=False)
_axes_flat = [axes[r][c] for r in range(_nrows) for c in range(_ncols)]
for ax in _axes_flat[n_models:]:
    ax.set_visible(False)

for ax, result in zip(_axes_flat, results):
    plot_cv_fold_metrics(result, metric="r2", ax=ax)

fig.suptitle("R² per Cross-Validation Fold (Training Set)", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "model_cv_fold_r2.png", dpi=150)
plt.show()
print("Saved: model_cv_fold_r2.png")

# %%
# =============================================================================
# Figure M4 — Residual plots (hold-out)
# =============================================================================

fig, axes = plt.subplots(_nrows, _ncols, figsize=(7 * _ncols, 4 * _nrows), squeeze=False)
_axes_flat = [axes[r][c] for r in range(_nrows) for c in range(_ncols)]
for ax in _axes_flat[n_models:]:
    ax.set_visible(False)

for ax, result in zip(_axes_flat, results):
    if result.holdout_y_true is not None:
        residuals = result.holdout_y_true - result.holdout_y_pred
        _rpm = _holdout_rpm.get(result.sensor, None)
        sc = ax.scatter(result.holdout_y_pred, residuals, c=_rpm, cmap="viridis",
                        s=12, alpha=0.6, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="RPM")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Predicted κ")
        ax.set_ylabel("Residual (true − pred)")
        ax.set_title(f"{result.name} — Hold-Out Residuals")
        ax.grid(ls=":", alpha=0.4)

fig.suptitle("Residual Analysis (Hold-Out Test Set)", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "model_residuals_holdout.png", dpi=150)
plt.show()
print("Saved: model_residuals_holdout.png")

# %%
# =============================================================================
# Save detailed results
# =============================================================================

for result in results:
    tag = result.name.lower()
    weights, is_importance = get_feature_weights(result)

    weight_label = "importance" if is_importance else "coefficient"
    weight_df = pd.DataFrame({
        "feature": result.feature_names,
        weight_label: weights,
        f"abs_{weight_label}": np.abs(weights),
    }).sort_values(f"abs_{weight_label}", ascending=False)
    weight_path = OUTPUT_DIR / f"model_weights_{tag}.csv"
    weight_df.to_csv(weight_path, index=False)
    print(f"Saved: {weight_path.name}")

    fold_path = OUTPUT_DIR / f"model_folds_{tag}.csv"
    pd.DataFrame(result.fold_metrics).to_csv(fold_path, index=False)
    print(f"Saved: {fold_path.name}")

    # Save hold-out predictions
    if result.holdout_y_true is not None:
        ho_df = pd.DataFrame({
            "y_true": result.holdout_y_true,
            "y_pred": result.holdout_y_pred,
            "residual": result.holdout_y_true - result.holdout_y_pred,
        })
        ho_path = OUTPUT_DIR / f"model_holdout_{tag}.csv"
        ho_df.to_csv(ho_path, index=False)
        print(f"Saved: {ho_path.name}")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 80)
print("MODELLING — SUMMARY")
print("=" * 80)
for result in results:
    est = result.estimator
    weights, is_importance = get_feature_weights(result)
    print(f"\n{result.name} ({result.sensor})")
    print(f"  CV:       R² = {result.r2:.3f}   MAE = {result.mae:.3f}   RMSE = {result.rmse:.3f}")
    if result.holdout_metrics:
        h = result.holdout_metrics
        print(f"  Hold-out: R² = {h['r2']:.3f}   MAE = {h['mae']:.3f}   RMSE = {h['rmse']:.3f}")

    if hasattr(est, "l1_ratio"):
        alpha = getattr(est, "alpha_", None) or est.alpha
        l1 = getattr(est, "l1_ratio_", None) or est.l1_ratio
        print(f"  α = {alpha:.4f}   l1_ratio = {l1:.2f}")
    elif hasattr(est, "lambda_"):
        print(f"  α (noise) = {est.alpha_:.2e}   λ (weights) = {est.lambda_:.2e}")
    elif hasattr(est, "n_estimators"):
        print(f"  n_estimators = {est.n_estimators}   lr = {est.learning_rate}")

    n_nz = int(np.sum(np.abs(weights) > 1e-10))
    print(f"  Features: {n_nz} non-zero / {len(result.feature_names)} total")

    top_idx = np.argsort(np.abs(weights))[::-1][:5]
    weight_label = "importance" if is_importance else "coef"
    print(f"  Top 5 features ({weight_label}):")
    for i in top_idx:
        print(f"    {result.feature_names[i]:40s}  {weight_label} = {weights[i]:+.4f}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n04_modelling complete.")
