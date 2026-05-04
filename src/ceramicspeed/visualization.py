"""
visualization.py
================
Consolidated plotting functions for CeramicSpeed bearing analysis.

Gathers all visualisation routines from feature analysis and modelling
into a single module.  Every function returns ``(fig, ax, ...)`` tuples
and accepts an optional ``ax`` parameter so callers can embed plots in
multi-panel figures.

Feature analysis plots
----------------------
plot_pca_kappa(X_pca, kappa, pca, ...)
plot_correlation_matrix(df, method="spearman", ...)
plot_mi_matrix(mi_mat, ...)
plot_vif(vif_df, ...)

Model evaluation plots
----------------------
plot_predicted_vs_actual(result, ...)
plot_coefficients(result, top_n=20, ...)
plot_cv_fold_metrics(result, metric="r2", ...)
plot_residuals(result, ...)
plot_bayesian_uncertainty(result, X, ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge

from .modelling import ModelResult, get_feature_weights

__all__ = [
    # Feature analysis
    "plot_pca_kappa",
    "plot_correlation_matrix",
    "plot_mi_matrix",
    "plot_vif",
    # Model evaluation
    "plot_predicted_vs_actual",
    "plot_coefficients",
    "plot_coefficients_log",
    "plot_cv_fold_metrics",
    "plot_residuals",
    "plot_bayesian_uncertainty",
]


# ======================================================================
# Feature analysis plots
# ======================================================================


def plot_pca_kappa(
    X_pca: np.ndarray,
    kappa: np.ndarray | pd.Series,
    pca: PCA,
    *,
    kappa_thresholds: tuple[float, float] = (1.0, 4.0),
    regime_labels: tuple[str, str, str] = (
        "boundary (κ<1)",
        "mixed (1≤κ<4)",
        "full-film (κ≥4)",
    ),
    regime_colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
    title: str = "PCA of features colored by κ regime",
) -> tuple[plt.Figure, plt.Axes]:
    """2-D PCA scatter plot colored by kappa lubrication regime.

    Parameters
    ----------
    X_pca:
        PCA-transformed coordinates (n_samples × 2).
    kappa:
        Kappa values aligned with *X_pca* rows.
    pca:
        Fitted PCA object (for explained variance labels).
    kappa_thresholds:
        (low, high) boundaries separating the three regimes.
    regime_labels:
        Labels for the three regimes.
    regime_colors:
        Optional mapping from regime label to colour string.
    ax:
        Matplotlib axes.  Created if ``None``.
    title:
        Plot title.

    Returns
    -------
    (fig, ax)
    """
    
    kappa_arr = np.asarray(kappa)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    sc = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kappa_arr,
        cmap="viridis",
        alpha=0.6,
        s=10,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("κ")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax.set_title(title)

    return fig, ax

    # kappa_arr = np.asarray(kappa)
    # low, high = kappa_thresholds

    # regime = np.where(
    #     kappa_arr < low,
    #     regime_labels[0],
    #     np.where(kappa_arr < high, regime_labels[1], regime_labels[2]),
    # )

    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(8, 6))
    # else:
    #     fig = ax.get_figure()

    # colors = regime_colors or {
    #     "boundary (κ<1)": "#d62728",
    #     "mixed (1≤κ<4)": "#ff7f0e",
    #     "full-film (κ≥4)": "#2ca02c",
    # }

    # for label in regime_labels:
    #     mask = regime == label
    #     if not mask.any():
    #         continue
    #     ax.scatter(
    #         X_pca[mask, 0],
    #         X_pca[mask, 1],
    #         label=label,
    #         alpha=0.5,
    #         s=10,
    #         color=colors.get(label),
    #     )

    # ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    # ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    # ax.set_title(title)
    # ax.legend()

    # return fig, ax


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    ax: plt.Axes | None = None,
    title: str | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Inter-feature correlation matrix heatmap.

    Parameters
    ----------
    df:
        Feature matrix (samples × features).
    method:
        ``"spearman"`` or ``"pearson"``.

    Returns
    -------
    (fig, ax, corr_matrix)
    """
    corr = df.corr(method=method)

    if ax is None:
        size = max(8, len(corr.columns) * 0.45)
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.get_figure()

    im = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title(title or f"Inter-feature {method.title()} correlation")

    return fig, ax, corr


def plot_mi_matrix(
    mi_mat: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str = "Inter-feature mutual information",
    cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of a pre-computed mutual information matrix."""
    if ax is None:
        size = max(8, len(mi_mat.columns) * 0.45)
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.get_figure()

    im = ax.imshow(mi_mat.values, cmap=cmap, aspect="equal", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(mi_mat.columns)))
    ax.set_yticks(range(len(mi_mat.columns)))
    ax.set_xticklabels(mi_mat.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(mi_mat.columns, fontsize=8)
    ax.set_title(title)

    return fig, ax


def plot_vif(
    vif_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str = "Variance Inflation Factor",
    thresholds: tuple[float, ...] = (5.0, 10.0),
    threshold_colors: tuple[str, ...] = ("orange", "red"),
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of Variance Inflation Factors.

    Parameters
    ----------
    vif_df:
        DataFrame indexed by feature with column ``"vif"``.
    thresholds:
        Reference lines to draw on the x-axis.
    threshold_colors:
        Colours for the reference lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(vif_df) * 0.3)))
    else:
        fig = ax.get_figure()

    ax.barh(vif_df.index, vif_df["vif"])
    for thresh, color in zip(thresholds, threshold_colors):
        ax.axvline(x=thresh, color=color, linestyle="--", label=f"VIF = {thresh}")
    ax.set_xlabel("VIF")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()

    return fig, ax


# ======================================================================
# Model evaluation plots
# ======================================================================


def plot_predicted_vs_actual(
    result: ModelResult,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of cross-validated predictions vs true kappa."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.scatter(
        result.y_true, result.y_pred, s=14, alpha=0.5, edgecolors="none"
    )

    lims = [
        min(result.y_true.min(), result.y_pred.min()),
        max(result.y_true.max(), result.y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="ideal")

    ax.set_xlabel("True κ")
    ax.set_ylabel("Predicted κ")
    ax.set_title(
        f"{result.name}\n"
        f"R² = {result.r2:.3f}   MAE = {result.mae:.3f}   "
        f"RMSE = {result.rmse:.3f}"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(ls=":", alpha=0.4)

    return fig, ax


def plot_coefficients(
    result: ModelResult,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of model coefficients or feature importances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    else:
        fig = ax.get_figure()

    weights, is_importance = get_feature_weights(result)
    series = pd.Series(weights, index=result.feature_names)

    if is_importance:
        sorted_imp = series.sort_values(ascending=False).head(top_n)
        y_pos = np.arange(len(sorted_imp))
        ax.barh(
            y_pos,
            sorted_imp.values,
            color="#2ca02c",
            edgecolor="white",
            lw=0.5,
        )
        ax.set_xlabel("Feature importance (split-based)")
    else:
        abs_vals = series.abs().sort_values(ascending=False).head(top_n)
        signed = series[abs_vals.index]
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in signed]
        y_pos = np.arange(len(abs_vals))
        ax.barh(
            y_pos,
            abs_vals.values,
            color=colors,
            edgecolor="white",
            lw=0.5,
        )
        ax.set_xlabel("|Coefficient| (standardised features)")
        sorted_imp = abs_vals

        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="#1f77b4", label="positive"),
                Patch(color="#d62728", label="negative"),
            ],
            fontsize=8,
            loc="lower right",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_imp.index, fontsize=8)
    ax.invert_yaxis()

    subtitle_parts: list[str] = []
    est = result.estimator
    if hasattr(est, "l1_ratio"):
        alpha = getattr(est, "alpha_", None) or est.alpha
        subtitle_parts.append(f"α={alpha:.4f}, l1={est.l1_ratio:.2f}")
    elif hasattr(est, "lambda_"):
        subtitle_parts.append(f"α={est.alpha_:.2e}, λ={est.lambda_:.2e}")
    elif hasattr(est, "n_estimators"):
        subtitle_parts.append(
            f"n_trees={est.n_estimators}, lr={est.learning_rate}"
        )
    subtitle = f"  ({', '.join(subtitle_parts)})" if subtitle_parts else ""
    chart_label = "importances" if is_importance else "coefficients"
    ax.set_title(f"{result.name} — Top {top_n} {chart_label}{subtitle}")
    ax.grid(axis="x", ls=":", alpha=0.5)

    return fig, ax


def plot_coefficients_log(
    result: ModelResult,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of |coefficients| or importances on a log x-axis.

    Bars are coloured by sign (positive = blue, negative = red) for linear
    models, and green for tree importances.  Coefficients equal to zero are
    excluded.  A dashed reference line is drawn at |coef| = 1.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
    else:
        fig = ax.get_figure()

    weights, is_importance = get_feature_weights(result)
    series = pd.Series(weights, index=result.feature_names)

    # Drop exact zeros before ranking
    series = series[series != 0]

    if is_importance:
        sorted_s = series.sort_values(ascending=False).head(top_n)
        colors = ["#2ca02c"] * len(sorted_s)
        xlabel = "Feature importance (log scale)"
    else:
        sorted_s = series.abs().sort_values(ascending=False).head(top_n)
        colors = [
            "#d62728" if series[name] < 0 else "#1f77b4"
            for name in sorted_s.index
        ]
        xlabel = "|Coefficient| (log scale, standardised features)"

    y_pos = np.arange(len(sorted_s))
    ax.barh(
        y_pos,
        sorted_s.values,
        color=colors,
        edgecolor="white",
        lw=0.5,
    )

    ax.set_xscale("log")
    ax.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.7, label="|coef| = 1")
    ax.set_xlabel(xlabel)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_s.index, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", which="both", ls=":", alpha=0.4)

    if not is_importance:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(color="#1f77b4", label="positive"),
                Patch(color="#d62728", label="negative"),
                plt.Line2D([0], [0], color="grey", ls="--", lw=0.8, label="|coef| = 1"),
            ],
            fontsize=8,
            loc="lower right",
        )
    else:
        ax.legend(fontsize=8)

    subtitle_parts: list[str] = []
    est = result.estimator
    inner = est[-1] if hasattr(est, "__len__") else est  # unwrap Pipeline
    if hasattr(est, "l1_ratio"):
        alpha = getattr(est, "alpha_", None) or est.alpha
        subtitle_parts.append(f"α={alpha:.4f}, l1={est.l1_ratio:.2f}")
    elif hasattr(est, "lambda_"):
        subtitle_parts.append(f"α={est.alpha_:.2e}, λ={est.lambda_:.2e}")
    elif hasattr(est, "n_estimators"):
        subtitle_parts.append(f"n_trees={est.n_estimators}, lr={est.learning_rate}")
    elif hasattr(inner, "alpha") and inner is not est:
        subtitle_parts.append(f"α={inner.alpha:.4f}")
    subtitle = f"  ({', '.join(subtitle_parts)})" if subtitle_parts else ""
    chart_label = "importances" if is_importance else "coefficients"
    ax.set_title(f"{result.name} — Top {top_n} {chart_label} (log){subtitle}")

    return fig, ax


def plot_cv_fold_metrics(
    result: ModelResult,
    metric: str = "r2",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of a metric across CV folds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    fold_df = pd.DataFrame(result.fold_metrics)
    x = np.arange(len(fold_df))
    vals = fold_df[metric].values

    ax.bar(x, vals, color="#1f77b4", edgecolor="white", lw=0.5)
    ax.axhline(
        np.mean(vals),
        color="red",
        ls="--",
        lw=1,
        label=f"mean = {np.mean(vals):.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in fold_df.index], fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{result.name} — {metric.upper()} per CV fold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)

    return fig, ax


def plot_residuals(
    result: ModelResult,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Residual plot (predicted vs residual) to check for patterns."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    residuals = result.y_true - result.y_pred
    ax.scatter(result.y_pred, residuals, s=12, alpha=0.5, edgecolors="none")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Predicted κ")
    ax.set_ylabel("Residual (true − pred)")
    ax.set_title(f"{result.name} — Residuals")
    ax.grid(ls=":", alpha=0.4)

    return fig, ax


def plot_bayesian_uncertainty(
    result: ModelResult,
    X: pd.DataFrame,
    y: np.ndarray | None = None,
    ci_level: float = 0.95,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Bayesian Ridge predictions with confidence and prediction intervals.

    Parameters
    ----------
    result:
        Must wrap a fitted ``BayesianRidge`` estimator.
    X:
        Feature matrix (unscaled — the function applies result.scaler).
    y:
        True target.  Defaults to ``result.y_true``.
    ci_level:
        Confidence level (default 0.95 → 95% intervals).
    ax:
        Matplotlib axes.  Created if ``None``.

    Returns
    -------
    (fig, ax)
    """
    from scipy import stats as sp_stats

    est = result.estimator
    if not isinstance(est, BayesianRidge):
        raise TypeError(
            f"plot_bayesian_uncertainty requires a BayesianRidge estimator, "
            f"got {type(est).__name__}"
        )

    if y is None:
        y = result.y_true

    X_scaled = result.scaler.transform(X.values)

    y_mean, y_pred_std = est.predict(X_scaled, return_std=True)

    Sigma = est.sigma_
    y_conf_var = np.einsum("ij,jk,ik->i", X_scaled, Sigma, X_scaled)
    y_conf_std = np.sqrt(np.maximum(y_conf_var, 0.0))

    z = sp_stats.norm.ppf(0.5 + ci_level / 2.0)

    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    mean_sorted = y_mean[sort_idx]
    pred_lo = mean_sorted - z * y_pred_std[sort_idx]
    pred_hi = mean_sorted + z * y_pred_std[sort_idx]
    conf_lo = mean_sorted - z * y_conf_std[sort_idx]
    conf_hi = mean_sorted + z * y_conf_std[sort_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    x_axis = np.arange(len(y_sorted))
    ci_pct = int(ci_level * 100)

    ax.fill_between(
        x_axis,
        pred_lo,
        pred_hi,
        alpha=0.18,
        color="#1f77b4",
        label=f"{ci_pct}% prediction interval",
    )
    ax.fill_between(
        x_axis,
        conf_lo,
        conf_hi,
        alpha=0.35,
        color="#ff7f0e",
        label=f"{ci_pct}% confidence interval (mean)",
    )
    ax.plot(
        x_axis, mean_sorted, lw=1.2, color="#1f77b4", label="predicted mean"
    )
    ax.scatter(
        x_axis, y_sorted, s=10, color="k", zorder=5, alpha=0.6, label="true κ"
    )

    ax.set_xlabel("Observation (sorted by true κ)")
    ax.set_ylabel("κ")
    ax.set_title(f"{result.name} — Bayesian Predictive Distribution")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(ls=":", alpha=0.4)

    return fig, ax
