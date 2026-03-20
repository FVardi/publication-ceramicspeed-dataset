"""
analysis.py
===========
Feature analysis utilities for CeramicSpeed bearing analysis.

Provides correlation metrics (Spearman, Pearson, mutual information),
combined feature ranking, redundancy detection (VIF, inter-feature
correlation), and greedy feature selection.

All functions are stateless: they accept DataFrames / arrays and return
results.  Plotting is handled separately in :mod:`visualization`.

Functions
---------
spearman_correlation(df, target)
pearson_correlation(df, target)
mutual_information(df, target, ...)
feature_ranking(spearman_df, pearson_df, mi_df)
correlation_matrix(df, method="spearman")
mi_matrix(df, ...)
variance_inflation_factors(df)
identify_redundant_features(corr_matrix, mi_mat, vif_df, ...)
reduce_redundant_features(df, target, corr_matrix, vif_df, ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

__all__ = [
    "spearman_correlation",
    "pearson_correlation",
    "mutual_information",
    "feature_ranking",
    "correlation_matrix",
    "mi_matrix",
    "variance_inflation_factors",
    "identify_redundant_features",
    "reduce_redundant_features",
    "pca_transform",
]


# ---------------------------------------------------------------------------
# Single-target correlation / MI
# ---------------------------------------------------------------------------


def spearman_correlation(
    df: pd.DataFrame,
    target: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Spearman rank-correlation between each feature column and a target.

    Parameters
    ----------
    df:
        Feature matrix (samples × features).
    target:
        Target variable (e.g. kappa), aligned with *df* rows.

    Returns
    -------
    pd.DataFrame
        Indexed by feature name with columns ``"rho"`` and ``"p_value"``,
        sorted by absolute correlation descending.
    """
    records: list[dict[str, float | str]] = []
    for col in df.columns:
        rho, p = stats.spearmanr(df[col], target)
        records.append({"feature": col, "rho": rho, "p_value": p})

    return (
        pd.DataFrame(records)
        .set_index("feature")
        .sort_values("rho", key=np.abs, ascending=False)
    )


def pearson_correlation(
    df: pd.DataFrame,
    target: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Pearson correlation between each feature column and a target.

    Returns
    -------
    pd.DataFrame
        Indexed by feature name with columns ``"r"`` and ``"p_value"``,
        sorted by absolute correlation descending.
    """
    records: list[dict[str, float | str]] = []
    for col in df.columns:
        r, p = stats.pearsonr(df[col], target)
        records.append({"feature": col, "r": r, "p_value": p})

    return (
        pd.DataFrame(records)
        .set_index("feature")
        .sort_values("r", key=np.abs, ascending=False)
    )


def mutual_information(
    df: pd.DataFrame,
    target: np.ndarray | pd.Series,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Mutual information between each feature and a target.

    Captures non-monotonic and non-linear relationships that
    Spearman / Pearson miss.

    Returns
    -------
    pd.DataFrame
        Indexed by feature with column ``"mi"``, sorted descending.
    """
    mi = mutual_info_regression(
        df.values,
        np.asarray(target),
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return (
        pd.DataFrame({"feature": df.columns, "mi": mi})
        .set_index("feature")
        .sort_values("mi", ascending=False)
    )


# ---------------------------------------------------------------------------
# Combined ranking
# ---------------------------------------------------------------------------


def feature_ranking(
    spearman_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank features by Spearman |rho|, sorted descending."""
    ranking = pd.DataFrame(index=spearman_df.index)
    ranking["|rho|"] = spearman_df["rho"].abs()
    ranking["rank_rho"] = ranking["|rho|"].rank(ascending=False)
    return ranking.sort_values("rank_rho")


# ---------------------------------------------------------------------------
# Inter-feature redundancy analysis
# ---------------------------------------------------------------------------


def correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute the inter-feature correlation matrix.

    Parameters
    ----------
    df:
        Feature matrix (samples × features).
    method:
        ``"spearman"`` or ``"pearson"``.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    return df.corr(method=method)


def mi_matrix(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pairwise mutual information matrix between all features.

    Uses ``mutual_info_regression`` with each feature as target in turn.
    The result is symmetrised by averaging MI(i,j) and MI(j,i).
    """
    cols = df.columns.tolist()
    n = len(cols)
    mat = np.zeros((n, n))

    for j in range(n):
        mi_vals = mutual_info_regression(
            df.values,
            df.iloc[:, j].values,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        mat[:, j] = mi_vals

    mat = (mat + mat.T) / 2.0
    return pd.DataFrame(mat, index=cols, columns=cols)


def variance_inflation_factors(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Variance Inflation Factor for each feature.

    VIF_j = 1 / (1 - R²_j) where R²_j is the coefficient of
    determination from regressing feature j on all remaining features.
    Features are standardised first so scale does not affect R².

    Returns
    -------
    pd.DataFrame
        Indexed by feature with column ``"vif"``, sorted descending.
    """
    X = StandardScaler().fit_transform(df.values)
    n_features = X.shape[1]
    vifs = np.empty(n_features)

    for j in range(n_features):
        y_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)
        coef, *_ = np.linalg.lstsq(X_rest, y_j, rcond=None)
        y_hat = X_rest @ coef
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf

    return (
        pd.DataFrame({"feature": df.columns, "vif": vifs})
        .set_index("feature")
        .sort_values("vif", ascending=False)
    )


def identify_redundant_features(
    corr_matrix: pd.DataFrame,
    vif_df: pd.DataFrame,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
) -> pd.DataFrame:
    """Flag redundant features based on Spearman correlation and VIF.

    For each feature reports:

    - ``high_vif``:  VIF above *vif_threshold*.
    - ``n_corr_partners``:  number of other features with
      |correlation| >= *corr_threshold*.
    - ``redundant``:  ``True`` when flagged by both criteria.

    Returns
    -------
    pd.DataFrame
        Indexed by feature, sorted by VIF descending.
    """
    features = corr_matrix.columns.tolist()

    abs_corr = corr_matrix.abs()
    np.fill_diagonal(abs_corr.values, 0.0)
    n_corr = (abs_corr >= corr_threshold).sum(axis=1)

    vif_series = vif_df["vif"].reindex(features)

    result = pd.DataFrame(
        {
            "vif": vif_series,
            "high_vif": vif_series >= vif_threshold,
            "n_corr_partners": n_corr,
            "redundant": (vif_series >= vif_threshold) & (n_corr > 0),
        },
        index=features,
    )

    return result.sort_values("vif", ascending=False)


def reduce_redundant_features(
    df: pd.DataFrame,
    target: np.ndarray | pd.Series,
    corr_matrix: pd.DataFrame,
    vif_df: pd.DataFrame,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
) -> list[str]:
    """Greedy removal of redundant features, keeping the best predictor.

    Within each cluster of highly-correlated features (|r| >= threshold),
    the feature with the highest absolute Spearman correlation to the
    target is retained and the rest are dropped, provided they also
    exceed the VIF threshold.

    Returns
    -------
    list[str]
        Column names of the retained (non-redundant) feature subset.
    """
    spear = spearman_correlation(df, target)
    relevance = spear["rho"].abs()

    abs_corr = corr_matrix.abs().copy()
    np.fill_diagonal(abs_corr.values, 0.0)

    dropped: set[str] = set()

    for feat in relevance.sort_values().index:
        if feat in dropped:
            continue
        partners = abs_corr.loc[feat]
        redundant_with = [
            p
            for p in partners[partners >= corr_threshold].index
            if p not in dropped
            and p != feat
            and vif_df.loc[p, "vif"] >= vif_threshold
        ]
        for partner in redundant_with:
            if relevance[feat] >= relevance[partner]:
                dropped.add(partner)
            else:
                dropped.add(feat)
                break

    return [f for f in df.columns if f not in dropped]


# ---------------------------------------------------------------------------
# PCA (computation only — plotting in visualization.py)
# ---------------------------------------------------------------------------


def pca_transform(
    df: pd.DataFrame,
    n_components: int = 2,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """Standardise features and project onto principal components.

    Parameters
    ----------
    df:
        Feature matrix (samples × features).
    n_components:
        Number of PCA components.

    Returns
    -------
    tuple[np.ndarray, PCA, StandardScaler]
        ``(X_pca, pca_object, scaler)``
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler
