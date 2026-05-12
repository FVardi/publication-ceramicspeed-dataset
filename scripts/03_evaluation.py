"""
03_evaluation.py
================
Repeated nested cross-validation for unbiased model performance estimation.

Pipeline position: 3rd script — reads features.parquet + metadata.parquet
from 01_feature_generation and feature_selection.json from 02_feature_analysis.
Runs before 04_modelling.py and does not depend on any fitted models.

What this script does
---------------------
Runs repeated nested CV (R=10 repeats × k=5 folds) on the training pool for
all model–feature-set combinations. HP selection is performed inside each outer
fold, so the outer val set is never seen during HP search:

  - Elastic Net:   ElasticNetCV (alpha, l1_ratio) re-run per outer fold
  - Polynomial:    RidgeCV (alpha) re-run per outer fold
  - LightGBM:      Optuna (learning_rate, num_leaves, max_depth) with inner
                   k-fold + early stopping per outer fold

The 50 RMSE scores per model are saved and used by 05_holdout_tests.py for
statistical significance testing (Nadeau-Bengio corrected t-test).

Usage
-----
    python scripts/03_evaluation.py
    python scripts/03_evaluation.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import warnings

import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm.auto import tqdm

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir

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
        help="Fast mode: fewer CV repeats and Optuna trials for quick sanity checks.",
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

DEBUG: bool = args.debug or bool(cfg.get("debug", {}).get("enabled", False))
if DEBUG:
    print("*** DEBUG MODE — reduced parameters for fast testing ***")

OUTPUT_DIR = get_output_dir(cfg)
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
for _d in [FIGURES_DIR, TABLES_DIR, PREDICTIONS_DIR]:
    _d.mkdir(exist_ok=True)

D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]
RANDOM_STATE: int = cfg.get("random_state", 42)

model_cfg = cfg.get("modelling", {})
CV_N_SPLITS: int = model_cfg.get("cv_n_splits", 5)
TEST_SIZE: float = model_cfg.get("test_size", 0.2)

_dbg = cfg.get("debug", {})
CV_N_REPEATS: int = 10   # R — number of CV repeats

if DEBUG:
    CV_N_SPLITS  = _dbg.get("cv_n_splits", 3)
    CV_N_REPEATS = _dbg.get("n_repeats", 2)
    print(f"  cv_n_splits={CV_N_SPLITS}, n_repeats={CV_N_REPEATS}")

enet_cfg = model_cfg.get("elastic_net", {})
ENET_ALPHAS: list[float] | None = enet_cfg.get("alphas")
ENET_L1_RATIOS: list[float] = enet_cfg.get("l1_ratios") or [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ENET_MAX_ITER: int = enet_cfg.get("max_iter", 10_000)

poly_cfg = model_cfg.get("polynomial", {})
POLY_DEGREE: int = poly_cfg.get("degree", 2)
POLY_TOP_K: int | None = poly_cfg.get("top_k")
POLY_ALPHAS: list[float] = poly_cfg.get("alphas") or list(np.logspace(-3, 4, 15))

lgb_cfg = model_cfg.get("lightgbm", {})
LGB_N_ESTIMATORS: int = lgb_cfg.get("n_estimators", 500)
LGB_EARLY_STOP: int = lgb_cfg.get("early_stopping_rounds", 50)
LGB_MIN_CHILD_SAMPLES: int = lgb_cfg.get("min_child_samples", 10)
LGB_SUBSAMPLE: float = lgb_cfg.get("subsample", 0.8)
LGB_COLSAMPLE: float = lgb_cfg.get("colsample_bytree", 0.8)
LGB_REG_ALPHA: float = lgb_cfg.get("reg_alpha", 0.0)
LGB_REG_LAMBDA: float = lgb_cfg.get("reg_lambda", 1.0)
LGB_N_TRIALS_NESTED: int = lgb_cfg.get("n_trials", 50)
# Continuous search bounds for the nested CV inner Optuna loop
LGB_LR_BOUNDS: tuple[float, float] = (0.005, 0.3)
LGB_LEAVES_BOUNDS: tuple[int, int] = (15, 127)
LGB_DEPTH_BOUNDS: tuple[int, int] = (3, 12)

if DEBUG:
    LGB_N_ESTIMATORS    = _dbg.get("n_estimators", 40)
    LGB_EARLY_STOP      = _dbg.get("early_stopping_rounds", 10)
    LGB_N_TRIALS_NESTED = _dbg.get("n_trials", 2)
    print(f"  n_estimators={LGB_N_ESTIMATORS}, early_stop={LGB_EARLY_STOP}, "
          f"n_trials_nested={LGB_N_TRIALS_NESTED}")

# %%
# =============================================================================
# Load data + reproduce the same holdout split as 04_modelling.py
# =============================================================================

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)

feat_sel_path = OUTPUT_DIR / "feature_selection.json"
if not feat_sel_path.exists():
    raise FileNotFoundError(f"{feat_sel_path} not found. Run 02_feature_analysis.py first.")
with open(feat_sel_path) as fh:
    feature_selection: dict = json.load(fh)

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
# Nested CV fold score functions
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
    return float(np.sqrt(mean_squared_error(y[val], model.predict(X_val_s))))


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
    return float(np.sqrt(mean_squared_error(y[val], model.predict(X_val_poly))))


def _lgb_fold_score(
    X: pd.DataFrame,
    y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
) -> float:
    """One outer fold of fully nested LightGBM CV.

    Inner loop: Optuna searches HP on the outer fold's training data.
    Outer loop: final model fit on full tr, evaluated on val (never seen during HP search).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_tr, y_tr = X.iloc[tr], y[tr]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=LGB_N_ESTIMATORS,
            learning_rate=trial.suggest_float(
                "learning_rate", *LGB_LR_BOUNDS, log=True),
            num_leaves=trial.suggest_int(
                "num_leaves", *LGB_LEAVES_BOUNDS),
            max_depth=trial.suggest_int(
                "max_depth", *LGB_DEPTH_BOUNDS),
            min_child_samples=LGB_MIN_CHILD_SAMPLES,
            subsample=LGB_SUBSAMPLE,
            colsample_bytree=LGB_COLSAMPLE,
            reg_alpha=LGB_REG_ALPHA,
            reg_lambda=LGB_REG_LAMBDA,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        )
        inner_cv = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        rmses = []
        for in_tr, in_val in inner_cv.split(X_tr):
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_tr.iloc[in_tr], y_tr[in_tr],
                eval_set=[(X_tr.iloc[in_val], y_tr[in_val])],
                callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            rmses.append(float(np.sqrt(mean_squared_error(y_tr[in_val], m.predict(X_tr.iloc[in_val])))))
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=LGB_N_TRIALS_NESTED, show_progress_bar=False)

    best = study.best_params
    final_model = lgb.LGBMRegressor(
        n_estimators=LGB_N_ESTIMATORS,
        learning_rate=best["learning_rate"],
        num_leaves=best["num_leaves"],
        max_depth=best["max_depth"],
        min_child_samples=LGB_MIN_CHILD_SAMPLES,
        subsample=LGB_SUBSAMPLE,
        colsample_bytree=LGB_COLSAMPLE,
        reg_alpha=LGB_REG_ALPHA,
        reg_lambda=LGB_REG_LAMBDA,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    final_model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=0)])
    return float(np.sqrt(mean_squared_error(y[val], final_model.predict(X.iloc[val]))))


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
# Run repeated nested CV for all model–feature-set combinations
# =============================================================================

# Pattern: {ModelType}_{SensorName} e.g. "ElasticNet_AE", "LightGBM_Combined"

cv_scores: dict[str, np.ndarray] = {}

sensor_names = list(feature_selection.keys())

# --- Elastic Net ---
for sensor_name in sensor_names:
    X_tr, y_tr = sensor_train[sensor_name]
    model_name = f"ElasticNet_{sensor_name}"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(X_tr, y_tr, _enet_fold_score, desc=model_name)

if X_combined_train is not None:
    model_name = "ElasticNet_Combined"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_combined_train, y_combined_train, _enet_fold_score, desc=model_name)

# --- Polynomial ---
for sensor_name in sensor_names:
    X_tr, y_tr = sensor_train[sensor_name]
    model_name = f"Polynomial_{sensor_name}"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(X_tr, y_tr, _poly_fold_score, desc=model_name)

if X_combined_train is not None:
    model_name = "Polynomial_Combined"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_combined_train, y_combined_train, _poly_fold_score, desc=model_name)

# --- LightGBM (nested Optuna HP search per outer fold) ---
for sensor_name in sensor_names:
    X_tr, y_tr = sensor_train[sensor_name]
    model_name = f"LightGBM_{sensor_name}"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(X_tr, y_tr, _lgb_fold_score, desc=model_name)

if X_combined_train is not None:
    model_name = "LightGBM_Combined"
    print(f"\nRepeated nested CV: {model_name}")
    cv_scores[model_name] = repeated_nested_cv(
        X_combined_train, y_combined_train, _lgb_fold_score, desc=model_name)

# Save raw CV score distributions
cv_scores_df = pd.DataFrame(cv_scores)
cv_scores_path = PREDICTIONS_DIR / "repeated_cv_scores.csv"
cv_scores_df.to_csv(cv_scores_path, index=False)
print(f"\nSaved: {cv_scores_path.name}")

# %%
# =============================================================================
# Performance summary table — mean ± std RMSE from repeated nested CV
# =============================================================================

perf_rows = []
for model_name, scores in cv_scores.items():
    parts = model_name.rsplit("_", 1)
    perf_rows.append({
        "model": model_name,
        "model_type": parts[0] if len(parts) == 2 else model_name,
        "feature_set": parts[1] if len(parts) == 2 else "?",
        "mean_rmse": float(scores.mean()),
        "std_rmse": float(scores.std(ddof=1)),
        "mean_minus_std": float(scores.mean() - scores.std(ddof=1)),
        "n_scores": len(scores),
    })

perf_df = pd.DataFrame(perf_rows).sort_values("mean_rmse")
perf_path = TABLES_DIR / "performance_table_cv.csv"
perf_df.to_csv(perf_path, index=False)
print(f"Saved: {perf_path.name}")

# %%
# =============================================================================
# Figure E1 — CV score distributions (violin plot per feature set)
# =============================================================================

model_types = ["ElasticNet", "Polynomial", "LightGBM"]
feature_sets = sensor_names + (["Combined"] if X_combined_train is not None else [])

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
    fig_path = FIGURES_DIR / f"eval_cv_distribution_{fs.lower()}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Saved: {fig_path.name}")

# %%
# =============================================================================
# Figure E2 — Mean ± std RMSE comparison (grouped bar chart)
# =============================================================================

fig, ax = plt.subplots(figsize=(max(8, len(perf_df) * 0.9), 5))
x = np.arange(len(perf_df))
ax.bar(x, perf_df["mean_rmse"], yerr=perf_df["std_rmse"],
       capsize=4, color="#4878CF", alpha=0.8, error_kw={"linewidth": 1.2})
ax.set_xticks(x)
ax.set_xticklabels(perf_df["model"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Mean RMSE (repeated nested CV)")
ax.set_title(f"Model performance — mean ± std RMSE\n"
             f"(R={CV_N_REPEATS}×k={CV_N_SPLITS} outer folds)")
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
bar_path = FIGURES_DIR / "eval_performance_bar.png"
plt.savefig(bar_path, dpi=150)
plt.show()
print(f"Saved: {bar_path.name}")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 80)
print("NESTED CV — PERFORMANCE SUMMARY")
print("=" * 80)
print(f"(R={CV_N_REPEATS} repeats × k={CV_N_SPLITS} folds = {CV_N_REPEATS * CV_N_SPLITS} scores per model)")
print()
for _, row in perf_df.iterrows():
    print(f"  {row['model']:35s}  {row['mean_rmse']:.4f} ± {row['std_rmse']:.4f}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Run 04_modelling.py next to fit the final models.")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n03_evaluation complete.")
