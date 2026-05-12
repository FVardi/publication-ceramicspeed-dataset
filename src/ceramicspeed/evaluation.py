"""
evaluation.py
=============
Statistical evaluation utilities for comparing regression models.

Functions
---------
corrected_repeated_kfold_t_test
    Nadeau & Bengio (2003) corrected paired t-test for repeated k-fold CV.
holm_bonferroni_correction
    Family-wise error rate control via Holm's step-down procedure.
wilcoxon_test
    Non-parametric paired test on per-observation absolute error differences.
diebold_mariano_test
    Harvey et al. (1997) modified Diebold-Mariano test for paired forecast accuracy.
bootstrap_rmse_diff_ci
    Bootstrap confidence interval on the difference in mean RMSE between two models.
cross_model_agreement
    Top-k feature overlap table across models within a feature set.
pairwise_tests_dataframe
    Run all pairwise comparisons and return a tidy results DataFrame.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "corrected_repeated_kfold_t_test",
    "holm_bonferroni_correction",
    "wilcoxon_test",
    "diebold_mariano_test",
    "bootstrap_rmse_diff_ci",
    "cross_model_agreement",
    "pairwise_tests_dataframe",
]


# ---------------------------------------------------------------------------
# Corrected repeated k-fold t-test  (Nadeau & Bengio, 2003)
# ---------------------------------------------------------------------------


def corrected_repeated_kfold_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Paired t-test corrected for the non-independence of repeated k-fold CV folds.

    Parameters
    ----------
    scores_a, scores_b:
        1-D arrays of per-fold test scores (e.g. RMSE) of length R*k, where R
        is the number of repeats and k the number of folds.  Both arrays must
        be of equal length and paired fold-by-fold.
    k:
        Number of folds used in the CV (required to compute the correction).

    Returns
    -------
    (t_stat, p_value) : tuple[float, float]
        Two-sided p-value from the Student t(R*k − 1) distribution.

    References
    ----------
    Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
    Machine Learning, 52(3), 239–281.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have equal length")

    n = len(scores_a)  # R * k
    d = scores_a - scores_b
    d_bar = d.mean()
    s2 = d.var(ddof=1)

    # Correction term: accounts for positive correlation among folds that share
    # training data.  (1/(k-1)) is the ratio n_test/n_train for balanced folds.
    sigma2_corr = (1.0 / n + 1.0 / (k - 1)) * s2

    if sigma2_corr <= 0.0:
        return 0.0, 1.0

    t_stat = d_bar / np.sqrt(sigma2_corr)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    return float(t_stat), p_value


# ---------------------------------------------------------------------------
# Holm–Bonferroni correction
# ---------------------------------------------------------------------------


def holm_bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down multiple comparison correction.

    Parameters
    ----------
    p_values:
        Sequence of raw p-values.
    alpha:
        Family-wise error rate (default 0.05).

    Returns
    -------
    (reject, adjusted_alpha) : tuple[np.ndarray, np.ndarray]
        *reject* is a boolean array (True = reject H0 after correction).
        *adjusted_alpha* gives the Holm-corrected threshold for each p-value
        in the *sorted* order, useful for reporting.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    # Holm threshold for the i-th smallest p-value: alpha / (m - i)
    thresholds = alpha / (m - np.arange(m))

    # Reject up to the first non-rejection
    reject_sorted = np.zeros(m, dtype=bool)
    for i in range(m):
        if sorted_p[i] <= thresholds[i]:
            reject_sorted[i] = True
        else:
            break  # all subsequent p-values also fail to reject

    # Map back to original order
    reject = np.empty(m, dtype=bool)
    reject[order] = reject_sorted
    adj_alpha = np.empty(m, dtype=float)
    adj_alpha[order] = thresholds

    return reject, adj_alpha


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Wilcoxon signed-rank test on per-observation absolute errors.

    H0: the median of (|e_A| - |e_B|) is zero — no systematic difference
    in absolute error between the two models.

    Parameters
    ----------
    errors_a, errors_b:
        Per-observation *signed* prediction errors (y_true − y_pred) for
        models A and B respectively.  Absolute errors are computed internally.
    alternative:
        'two-sided' (default), 'greater', or 'less'.

    Returns
    -------
    (statistic, p_value) : tuple[float, float]
    """
    abs_diff = np.abs(np.asarray(errors_a, dtype=float)) - np.abs(
        np.asarray(errors_b, dtype=float)
    )
    # Drop exact ties (|e_A| == |e_B|) — scipy handles this but warns otherwise
    abs_diff = abs_diff[abs_diff != 0.0]
    if len(abs_diff) == 0:
        return 0.0, 1.0
    result = stats.wilcoxon(abs_diff, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Diebold–Mariano test  (Harvey, Leybourne & Newbold 1997 modification)
# ---------------------------------------------------------------------------


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    loss: str = "squared",
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Harvey et al. (1997) modified Diebold–Mariano test.

    Tests H0: equal predictive accuracy between models A and B.

    Parameters
    ----------
    errors_a, errors_b:
        Per-observation signed prediction errors (y_true − y_pred).
    loss:
        Loss function applied to errors before differencing.
        'squared' (default) or 'absolute'.
    alternative:
        'two-sided' (default), 'greater' (A worse than B), 'less' (A better).

    Returns
    -------
    (dm_stat, p_value) : tuple[float, float]
        The modified DM statistic and its p-value from t(T-1).

    References
    ----------
    Diebold, F. X., & Mariano, R. S. (1995). J. Business & Econ. Stat., 13, 253–263.
    Harvey, D., Leybourne, S., & Newbold, P. (1997). Int. J. Forecasting, 13, 281–291.
    """
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)

    if loss == "squared":
        d = e_a**2 - e_b**2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss!r}")

    T = len(d)
    d_bar = d.mean()
    var_d = d.var(ddof=1)

    if var_d <= 0.0:
        return 0.0, 1.0

    # Standard DM statistic
    dm = d_bar / np.sqrt(var_d / T)

    # Harvey et al. small-sample correction (h=1 forecast horizon)
    correction = np.sqrt((T + 1 - 2 * 1 + 1 * (1 - 1) / T) / T)
    dm_modified = dm * correction

    if alternative == "two-sided":
        p_value = float(2 * stats.t.sf(abs(dm_modified), df=T - 1))
    elif alternative == "greater":
        p_value = float(stats.t.sf(dm_modified, df=T - 1))
    elif alternative == "less":
        p_value = float(stats.t.cdf(dm_modified, df=T - 1))
    else:
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less'")

    return float(dm_modified), p_value


# ---------------------------------------------------------------------------
# Bootstrap CI on ΔRMSE
# ---------------------------------------------------------------------------


def bootstrap_rmse_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval on (RMSE_A − RMSE_B).

    A positive value means model A has higher (worse) RMSE than model B.

    Parameters
    ----------
    y_true:
        True target values.
    y_pred_a, y_pred_b:
        Predictions from models A and B (must be aligned with y_true).
    n_boot:
        Number of bootstrap resamples (default 10,000).
    alpha:
        CI level: returns (1−alpha)×100% interval (default 0.05 → 95% CI).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    (mean_diff, lower_ci, upper_ci) : tuple[float, float, float]
        Observed mean ΔRMSE and the percentile bootstrap CI bounds.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    n = len(y_true)

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmse_a = np.sqrt(np.mean((y_true[idx] - y_pred_a[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((y_true[idx] - y_pred_b[idx]) ** 2))
        diffs[i] = rmse_a - rmse_b

    observed = np.sqrt(np.mean((y_true - y_pred_a) ** 2)) - np.sqrt(
        np.mean((y_true - y_pred_b) ** 2)
    )
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return float(observed), lower, upper


# ---------------------------------------------------------------------------
# Cross-model feature agreement
# ---------------------------------------------------------------------------


def cross_model_agreement(
    importance_map: dict[str, pd.Series],
    top_k: int = 10,
) -> pd.DataFrame:
    """Top-k feature overlap table across models within a feature set.

    Parameters
    ----------
    importance_map:
        Dict mapping model name → pd.Series of mean absolute SHAP importance
        (index = feature name, values = importance).
    top_k:
        Number of top features to consider per model.

    Returns
    -------
    pd.DataFrame
        Rows = features that appear in the top-k of at least one model.
        Columns = model names + 'n_models_in_top_k' (count of models that
        rank this feature in their top-k).
        Sorted by n_models_in_top_k descending, then by the first model's rank.
    """
    model_names = list(importance_map.keys())
    top_k_sets: dict[str, set] = {
        name: set(imp.nlargest(top_k).index) for name, imp in importance_map.items()
    }

    all_features = set()
    for s in top_k_sets.values():
        all_features |= s

    rows = []
    for feat in all_features:
        row: dict = {"feature": feat}
        count = 0
        for name in model_names:
            in_top = feat in top_k_sets[name]
            row[f"{name}_rank"] = (
                int(importance_map[name].rank(ascending=False)[feat])
                if feat in importance_map[name].index
                else None
            )
            row[f"{name}_in_top{top_k}"] = in_top
            if in_top:
                count += 1
        row["n_models_in_top_k"] = count
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["n_models_in_top_k", model_names[0] + f"_rank"],
        ascending=[False, True],
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: run all pairwise tests and return a tidy DataFrame
# ---------------------------------------------------------------------------


def pairwise_tests_dataframe(
    cv_scores: dict[str, np.ndarray],
    holdout_residuals: dict[str, np.ndarray],
    holdout_y_true: dict[str, np.ndarray],
    holdout_y_pred: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    k: int,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    run_dm: bool = True,
    run_bootstrap: bool = True,
) -> pd.DataFrame:
    """Run all Level-1 (CV) and Level-2 (holdout) pairwise tests.

    Parameters
    ----------
    cv_scores:
        Dict mapping model name → array of shape (R*k,) RMSE from repeated CV.
    holdout_residuals:
        Dict mapping model name → array of signed residuals (y_true − y_pred)
        on the holdout set.
    holdout_y_true:
        Dict mapping model name → true holdout values.  The entry for model_a
        is used when running bootstrap CIs for a given pair.
    holdout_y_pred:
        Dict mapping model name → holdout predictions.
    pairs:
        List of (model_a, model_b) tuples to test.
    k:
        Number of CV folds (for Nadeau-Bengio correction).
    alpha:
        Significance threshold for Holm-Bonferroni correction.
    n_boot:
        Bootstrap resamples for ΔRMSE CI.
    run_dm:
        Whether to run Diebold-Mariano test.
    run_bootstrap:
        Whether to compute bootstrap CIs.

    Returns
    -------
    pd.DataFrame
        One row per pair, columns include all test results and the Holm-corrected
        rejection decisions for the CV-level t-test.
    """
    rows = []
    raw_p_cv = []

    for model_a, model_b in pairs:
        row: dict = {"model_a": model_a, "model_b": model_b}

        # Level 1 — corrected repeated k-fold t-test
        if model_a in cv_scores and model_b in cv_scores:
            t_stat, p_cv = corrected_repeated_kfold_t_test(
                cv_scores[model_a], cv_scores[model_b], k=k
            )
            row["cv_t_stat"] = t_stat
            row["cv_p_value"] = p_cv
        else:
            row["cv_t_stat"] = np.nan
            row["cv_p_value"] = np.nan
        raw_p_cv.append(row.get("cv_p_value", np.nan))

        # Level 2 — Wilcoxon + DM + bootstrap CI
        if model_a in holdout_residuals and model_b in holdout_residuals:
            e_a = holdout_residuals[model_a]
            e_b = holdout_residuals[model_b]
            if len(e_a) == len(e_b):
                w_stat, p_wilcox = wilcoxon_test(e_a, e_b)
                row["wilcoxon_stat"] = w_stat
                row["wilcoxon_p"] = p_wilcox

                if run_dm:
                    dm_stat, p_dm = diebold_mariano_test(e_a, e_b)
                    row["dm_stat"] = dm_stat
                    row["dm_p"] = p_dm

                y_true = holdout_y_true.get(model_a) or holdout_y_true.get(model_b)
                if run_bootstrap and y_true is not None and model_a in holdout_y_pred and model_b in holdout_y_pred:
                    mean_diff, ci_lo, ci_hi = bootstrap_rmse_diff_ci(
                        y_true,
                        holdout_y_pred[model_a],
                        holdout_y_pred[model_b],
                        n_boot=n_boot,
                        alpha=alpha,
                    )
                    row["delta_rmse"] = mean_diff
                    row[f"ci_{int((1-alpha)*100)}_lo"] = ci_lo
                    row[f"ci_{int((1-alpha)*100)}_hi"] = ci_hi

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Holm-Bonferroni correction on CV-level p-values
    raw_p_arr = np.array(raw_p_cv, dtype=float)
    valid_mask = ~np.isnan(raw_p_arr)
    if valid_mask.any():
        valid_p = raw_p_arr[valid_mask]
        reject, _ = holm_bonferroni_correction(valid_p, alpha=alpha)
        reject_col = np.full(len(raw_p_cv), False)
        reject_col[np.where(valid_mask)[0]] = reject
        result_df["cv_reject_holm"] = reject_col

    return result_df
