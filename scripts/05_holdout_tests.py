"""
05_holdout_tests.py
===================
Statistical significance testing on holdout predictions.

Pipeline position: 5th and final script — reads outputs of 03_evaluation.py
(repeated_cv_scores.csv) and 04_modelling.py (model_holdout_*.csv,
shap_importance_*.csv).

What this script does
---------------------
1. Loads the repeated nested CV scores from 03_evaluation.py and the paired
   holdout predictions from 04_modelling.py.
2. Runs two levels of pairwise significance tests:
   - Level 1 (CV scores): corrected repeated k-fold t-test (Nadeau & Bengio
     2003) with Holm-Bonferroni correction — for within-feature-set pairs.
   - Level 2 (holdout residuals): Wilcoxon signed-rank + Diebold-Mariano +
     bootstrap CIs on ΔRMSE — for cross-feature-set pairs.
3. Computes SHAP-based cross-model feature agreement tables.
4. Saves all results as CSV tables.

Usage
-----
    python scripts/05_holdout_tests.py
    python scripts/05_holdout_tests.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

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
    parser.add_argument("--debug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

DEBUG: bool = args.debug or bool(cfg.get("debug", {}).get("enabled", False))
if DEBUG:
    print("*** DEBUG MODE ***")

OUTPUT_DIR = get_output_dir(cfg)
TABLES_DIR = OUTPUT_DIR / "tables"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SHAP_DIR = OUTPUT_DIR / "shap"
for _d in [TABLES_DIR, PREDICTIONS_DIR, SHAP_DIR]:
    _d.mkdir(exist_ok=True)

model_cfg = cfg.get("modelling", {})
CV_N_SPLITS: int = model_cfg.get("cv_n_splits", 5)

_dbg = cfg.get("debug", {})
ALPHA: float = 0.05
N_BOOT: int = 10_000

if DEBUG:
    CV_N_SPLITS = _dbg.get("cv_n_splits", 3)
    N_BOOT      = _dbg.get("n_boot", 200)
    print(f"  cv_n_splits={CV_N_SPLITS}, n_boot={N_BOOT}")

# %%
# =============================================================================
# Load feature selection (for sensor names and feature sets)
# =============================================================================

feat_sel_path = OUTPUT_DIR / "feature_selection.json"
if not feat_sel_path.exists():
    raise FileNotFoundError(f"{feat_sel_path} not found. Run 02_feature_analysis.py first.")
with open(feat_sel_path) as fh:
    feature_selection: dict = json.load(fh)

sensor_names = list(feature_selection.keys())

# %%
# =============================================================================
# Load repeated CV scores from 03_evaluation.py
# =============================================================================

cv_scores_path = PREDICTIONS_DIR / "repeated_cv_scores.csv"
if not cv_scores_path.exists():
    raise FileNotFoundError(
        f"{cv_scores_path} not found. Run 03_evaluation.py first."
    )

cv_scores_df = pd.read_csv(cv_scores_path)
cv_scores: dict[str, np.ndarray] = {col: cv_scores_df[col].values for col in cv_scores_df.columns}

has_combined = any("_Combined" in n for n in cv_scores)
feature_sets = sensor_names + (["Combined"] if has_combined else [])
model_types = ["ElasticNet", "Polynomial", "LightGBM"]

print(f"Loaded CV scores for {len(cv_scores)} models "
      f"({len(next(iter(cv_scores.values())))} scores each)")

# %%
# =============================================================================
# Load holdout predictions from 04_modelling.py
# =============================================================================

holdout_y_true: dict[str, np.ndarray] = {}
holdout_y_pred: dict[str, np.ndarray] = {}
holdout_residuals: dict[str, np.ndarray] = {}

for model_name in cv_scores:
    ho_path = PREDICTIONS_DIR / f"model_holdout_{model_name.lower()}.csv"
    if not ho_path.exists():
        print(f"WARNING: {ho_path.name} not found — holdout tests skipped for {model_name}")
        continue
    ho_df = pd.read_csv(ho_path)
    holdout_y_true[model_name] = ho_df["y_true"].values
    holdout_y_pred[model_name] = ho_df["y_pred"].values
    holdout_residuals[model_name] = ho_df["residual"].values

print(f"Loaded holdout predictions for {len(holdout_y_true)} / {len(cv_scores)} models")

# %%
# =============================================================================
# Define comparison pairs
# =============================================================================

# Within each feature set — compare architectures (ElasticNet vs Poly vs LightGBM)
within_pairs: list[tuple[str, str]] = []
for fs in feature_sets:
    names = [f"{mt}_{fs}" for mt in model_types]
    available = [n for n in names if n in cv_scores]
    within_pairs.extend(combinations(available, 2))

# Cross-feature-set — best model per feature set (lowest mean CV RMSE)
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
within_results_path = TABLES_DIR / "stat_tests_within_featureset.csv"
within_results.to_csv(within_results_path, index=False)
print(f"\nSaved: {within_results_path.name}")

# %%
# =============================================================================
# Level 1+2 — Cross-feature-set tests
# =============================================================================


def _shared_true(model_a: str, model_b: str) -> np.ndarray | None:
    ya = holdout_y_true.get(model_a)
    yb = holdout_y_true.get(model_b)
    if ya is not None and yb is not None and len(ya) == len(yb) and np.allclose(ya, yb):
        return ya
    return None


cross_rows = []
for model_a, model_b in cross_pairs:
    row: dict = {
        "comparison_type": "cross_feature_set",
        "model_a": model_a,
        "model_b": model_b,
    }
    if model_a in cv_scores and model_b in cv_scores:
        t_stat, p_cv = corrected_repeated_kfold_t_test(
            cv_scores[model_a], cv_scores[model_b], k=CV_N_SPLITS
        )
        row["cv_t_stat"] = t_stat
        row["cv_p_value"] = p_cv

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
            f"ci_{int((1 - ALPHA) * 100)}_lo": ci_lo,
            f"ci_{int((1 - ALPHA) * 100)}_hi": ci_hi,
        })
    cross_rows.append(row)

cross_results = pd.DataFrame(cross_rows)

valid = ~pd.isna(cross_results.get("cv_p_value", pd.Series(dtype=float)))
if valid.any():
    reject, _ = holm_bonferroni_correction(
        cross_results.loc[valid, "cv_p_value"].values, alpha=ALPHA
    )
    cross_results.loc[valid, "cv_reject_holm"] = reject

cross_results_path = TABLES_DIR / "stat_tests_cross_featureset.csv"
cross_results.to_csv(cross_results_path, index=False)
print(f"Saved: {cross_results_path.name}")

# %%
# =============================================================================
# Combined significance table (all pairs)
# =============================================================================

all_results = pd.concat([within_results, cross_results], ignore_index=True)
all_path = TABLES_DIR / "stat_tests_all.csv"
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
        imp_path = SHAP_DIR / f"shap_importance_{tag}.csv"
        if imp_path.exists():
            imp_series = pd.read_csv(imp_path, index_col=0, header=0).iloc[:, 0]
            imp_map[mt] = imp_series
    if len(imp_map) < 2:
        continue
    top_k = min(10, min(len(s) for s in imp_map.values()))
    agree_df = cross_model_agreement(imp_map, top_k=top_k)
    agree_path = SHAP_DIR / f"shap_agreement_{fs.lower()}.csv"
    agree_df.to_csv(agree_path, index=False)
    print(f"Saved: {agree_path.name}")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 80)
print("HOLDOUT TESTS — SUMMARY")
print("=" * 80)

print("\n--- Within-feature-set (Nadeau-Bengio, Holm-Bonferroni) ---")
for _, row in within_results.iterrows():
    flag = "*" if row.get("cv_reject_holm", False) else " "
    p = row.get("cv_p_value", float("nan"))
    print(f"  {flag} {row['model_a']:30s} vs {row['model_b']:30s}  p={p:.4f}")

print("\n--- Cross-feature-set significance ---")
for _, row in cross_results.iterrows():
    flag = "*" if row.get("cv_reject_holm", False) else " "
    p = row.get("cv_p_value", float("nan"))
    diff = row.get("delta_rmse", float("nan"))
    ci_lo = row.get(f"ci_{int((1 - ALPHA) * 100)}_lo", float("nan"))
    ci_hi = row.get(f"ci_{int((1 - ALPHA) * 100)}_hi", float("nan"))
    print(f"  {flag} {row['model_a']:30s} vs {row['model_b']:30s}  "
          f"p={p:.4f}  ΔRMSE={diff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n05_holdout_tests complete.")
