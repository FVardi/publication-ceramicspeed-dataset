"""
modelling.py
============
Model training and evaluation utilities for CeramicSpeed bearing analysis.

Training approach
-----------------
1. **Hold-out split** — the caller splits data into train and test sets
   before calling any training function (typically 80/20 via
   ``train_test_split`` in the pipeline script).
2. **Single-level KFold CV on the train set** — each fold trains a model
   on the fold's training data and predicts on the fold's validation data,
   producing out-of-fold (OOF) predictions for the full training set.
3. **Final model** — refit on the full training set.
4. **Hold-out evaluation** — the caller evaluates the returned model on
   the held-out test set via :func:`evaluate_on_holdout`.

All functions are stateless — they accept data and config parameters
rather than reading global state — so they can be unit-tested and
reused from notebooks.

Classes
-------
ModelResult
    Container for a trained model with CV metrics and optional hold-out
    metrics.

Functions
---------
train_elastic_net_cv(X_train, y_train, ...)
train_bayesian_ridge_cv(X_train, y_train, ...)
train_lightgbm_cv(X_train, y_train, ...)
evaluate_on_holdout(result, X_test, y_test)
results_summary_table(results)
get_feature_weights(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

__all__ = [
    "ModelResult",
    "train_elastic_net_cv",
    "train_bayesian_ridge_cv",
    "train_polynomial_cv",
    "train_lightgbm_cv",
    "evaluate_on_holdout",
    "clip_predictions",
    "results_summary_table",
    "get_feature_weights",
    "compute_shap_values",
    "get_shap_feature_importance",
]


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return R², MAE, RMSE as a dict."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Container for a trained model, CV metrics, and hold-out metrics.

    Attributes
    ----------
    name, sensor : str
        Human-readable identifiers.
    feature_names : list[str]
        Column names used during training.
    estimator : Any
        Fitted sklearn / LightGBM estimator.
    scaler : StandardScaler
        Fitted scaler (for later prediction on new data).
    y_true, y_pred : np.ndarray
        Out-of-fold CV predictions on the *training* set.
    r2, mae, rmse : float
        Aggregate CV metrics (computed from y_true / y_pred).
    fold_metrics : list[dict]
        Per-fold CV metrics.
    holdout_metrics : dict | None
        Metrics on the hold-out test set (populated by
        :func:`evaluate_on_holdout`).
    holdout_y_true, holdout_y_pred : np.ndarray | None
        Hold-out predictions (populated by :func:`evaluate_on_holdout`).
    """

    name: str
    sensor: str
    feature_names: list[str]
    estimator: Any
    scaler: StandardScaler

    # Cross-validated outputs (aligned with training data)
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)

    # Aggregate CV metrics
    r2: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0

    # Per-fold CV metrics
    fold_metrics: list[dict[str, float]] = field(default_factory=list)

    # Hold-out test set (populated after training via evaluate_on_holdout)
    holdout_metrics: dict[str, float] | None = None
    holdout_y_true: np.ndarray | None = field(default=None, repr=False)
    holdout_y_pred: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.r2 = float(r2_score(self.y_true, self.y_pred))
        self.mae = float(mean_absolute_error(self.y_true, self.y_pred))
        self.rmse = float(
            np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        )

    def summary(self) -> str:
        s = (
            f"{self.name}  |  CV: R² = {self.r2:.3f}  "
            f"MAE = {self.mae:.3f}  RMSE = {self.rmse:.3f}  "
            f"({len(self.feature_names)} features)"
        )
        if self.holdout_metrics:
            h = self.holdout_metrics
            s += (
                f"\n{'':>{len(self.name)}}  |  Hold-out: R² = {h['r2']:.3f}  "
                f"MAE = {h['mae']:.3f}  RMSE = {h['rmse']:.3f}"
            )
        return s


# ---------------------------------------------------------------------------
# Hold-out evaluation
# ---------------------------------------------------------------------------


def evaluate_on_holdout(
    result: ModelResult,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> ModelResult:
    """Evaluate a trained model on the hold-out test set.

    Scales *X_test* with the scaler stored in *result*, predicts, and
    stores the hold-out metrics and predictions in the result object.

    Parameters
    ----------
    result:
        A ModelResult returned by one of the ``train_*_cv`` functions.
    X_test:
        Hold-out feature matrix (unscaled).
    y_test:
        Hold-out target values.

    Returns
    -------
    ModelResult
        The same object, mutated with hold-out fields populated.
    """
    y_test = np.asarray(y_test, dtype=float)

    # LightGBM: pass DataFrame directly so feature names match those stored
    # during fit — avoids sklearn's feature-name-mismatch warning.
    # Pipeline models handle scaling internally.
    # Linear models use the stored scaler.
    if isinstance(result.estimator, lgb.LGBMRegressor):
        X_scaled = X_test
    elif isinstance(result.estimator, Pipeline):
        X_scaled = X_test.values
    else:
        X_scaled = result.scaler.transform(X_test.values)

    y_pred_test = result.estimator.predict(X_scaled)

    result.holdout_y_true = y_test
    result.holdout_y_pred = y_pred_test
    result.holdout_metrics = _compute_metrics(y_test, y_pred_test)

    return result


# ---------------------------------------------------------------------------
# Prediction clipping
# ---------------------------------------------------------------------------


def clip_predictions(
    result: ModelResult,
    lo: float = 0.0,
    hi: float | None = None,
) -> ModelResult:
    """Clip OOF and hold-out predictions to [lo, hi] and recompute metrics.

    Applied after training and hold-out evaluation — does not affect the
    model weights or the training objective.

    Parameters
    ----------
    result:
        A ModelResult (hold-out evaluation must already have been run).
    lo:
        Lower bound.  Default ``0.0`` — kappa is physically non-negative.
    hi:
        Upper bound.  ``None`` (default) uses ``max(result.y_true)``, i.e.
        the highest kappa seen in the training set.
    """
    _hi = hi if hi is not None else float(result.y_true.max())

    result.y_pred = np.clip(result.y_pred, lo, _hi)
    m = _compute_metrics(result.y_true, result.y_pred)
    result.r2, result.mae, result.rmse = m["r2"], m["mae"], m["rmse"]

    if result.holdout_y_pred is not None and result.holdout_y_true is not None:
        result.holdout_y_pred = np.clip(result.holdout_y_pred, lo, _hi)
        result.holdout_metrics = _compute_metrics(
            result.holdout_y_true, result.holdout_y_pred
        )

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_feature_weights(result: ModelResult) -> tuple[np.ndarray, bool]:
    """Return (weights_array, is_importance).

    For linear models: returns coef_ (signed).
    For tree-based models: returns feature_importances_ (unsigned).
    For Pipeline estimators: inspects the final step.
    """
    est = result.estimator
    if isinstance(est, Pipeline):
        est = est[-1]
    if hasattr(est, "coef_"):
        return np.asarray(est.coef_), False
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_), True
    raise AttributeError(
        f"{type(est).__name__} has neither coef_ nor feature_importances_"
    )


def _unwrap_estimator(est: Any) -> Any:
    if isinstance(est, Pipeline):
        return est.steps[-1][1]
    return est


def compute_shap_values(
    result: ModelResult,
    X: pd.DataFrame,
    *,
    return_explanation: bool = False,
) -> "pd.DataFrame | shap.Explanation":
    """Compute SHAP values for a fitted model.

    Supports LightGBM (TreeExplainer), Elastic Net (LinearExplainer), and
    Polynomial Regression (LinearExplainer on the Ridge step with poly-expanded
    features).

    Parameters
    ----------
    result:
        A trained ModelResult.
    X:
        Feature matrix matching result.feature_names for linear/tree models,
        or the *original* feature matrix for Pipeline (Polynomial) models —
        the pipeline's internal steps handle the transformation.
    return_explanation:
        If True, return a shap.Explanation object (needed for beeswarm plots).
        If False (default), return a plain DataFrame of SHAP values.

    Returns
    -------
    pd.DataFrame or shap.Explanation
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required. Install with `pip install shap`."
        ) from exc

    est = _unwrap_estimator(result.estimator)

    # ------------------------------------------------------------------
    # LightGBM — TreeExplainer
    # ------------------------------------------------------------------
    if isinstance(est, lgb.LGBMRegressor):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=result.feature_names)
        X_input = X.loc[:, result.feature_names]
        explainer = shap.TreeExplainer(est)
        explanation = explainer(X_input)
        if return_explanation:
            return explanation
        vals = explanation.values
        if isinstance(vals, list):
            vals = vals[0]
        return pd.DataFrame(vals, columns=result.feature_names, index=X_input.index)

    # ------------------------------------------------------------------
    # Pipeline (Polynomial) — LinearExplainer on the Ridge step
    # ------------------------------------------------------------------
    if isinstance(result.estimator, Pipeline):
        pipeline = result.estimator
        # Transform X through all steps except the final Ridge
        X_transformed = X.values if isinstance(X, pd.DataFrame) else X
        for _, step in pipeline.steps[:-1]:
            X_transformed = step.transform(X_transformed)
        ridge = pipeline.steps[-1][1]
        explainer = shap.LinearExplainer(ridge, X_transformed)
        explanation = explainer(X_transformed)
        # Attach feature names so beeswarm labels work
        explanation.feature_names = result.feature_names
        if return_explanation:
            return explanation
        return pd.DataFrame(
            explanation.values,
            columns=result.feature_names,
        )

    # ------------------------------------------------------------------
    # Elastic Net — LinearExplainer (scaler applied externally)
    # ------------------------------------------------------------------
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=result.feature_names)
    X_input = X.loc[:, result.feature_names]
    X_scaled = result.scaler.transform(X_input.values)
    explainer = shap.LinearExplainer(est, X_scaled)
    explanation = explainer(X_scaled)
    explanation.feature_names = result.feature_names
    if return_explanation:
        return explanation
    return pd.DataFrame(
        explanation.values,
        columns=result.feature_names,
        index=X_input.index,
    )


def get_shap_feature_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Return mean absolute SHAP value importance for each feature."""
    return shap_df.abs().mean(axis=0).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Elastic Net training
# ---------------------------------------------------------------------------


def train_elastic_net_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_splits: int = 5,
    alphas: list[float] | None = None,
    l1_ratios: list[float] | None = None,
    max_iter: int = 10_000,
    random_state: int = 42,
    name: str = "ElasticNet",
    sensor: str = "unknown",
) -> ModelResult:
    """Train an Elastic Net with single-level KFold CV on the training set.

    ``ElasticNetCV`` is fit once on the full training set to select
    alpha and l1_ratio.  A single KFold loop then trains plain
    ``ElasticNet`` models at those fixed hyperparameters to collect
    out-of-fold predictions.  The ``ElasticNetCV`` fit is returned as
    the final model.

    Parameters
    ----------
    X_train:
        Training feature matrix (samples × features).
    y_train:
        Training target values (kappa).
    n_splits:
        Number of KFold splits.
    alphas, l1_ratios:
        Hyperparameter grids.  ``None`` uses defaults.
    max_iter:
        Maximum solver iterations.
    random_state:
        Random seed.
    name, sensor:
        Identifiers for plots and summaries.

    Returns
    -------
    ModelResult
        Contains the final model (fit on full training set), scaler,
        OOF predictions, and per-fold metrics.  Call
        :func:`evaluate_on_holdout` to add hold-out metrics.
    """
    from sklearn.linear_model import ElasticNet

    feature_names = X_train.columns.tolist()
    y_train = np.asarray(y_train, dtype=float)
    _l1_ratios = l1_ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)

    # Nested CV: HP selection is local to each outer fold's training split so
    # the OOF validation data never influences HP choice.
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred_oof = np.empty_like(y_train)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_scaled)):
        # Inner CV: select alpha / l1_ratio on this fold's training data only
        inner_cv = ElasticNetCV(
            l1_ratio=_l1_ratios,
            alphas=alphas,
            cv=n_splits,
            max_iter=max_iter,
            random_state=random_state,
        )
        inner_cv.fit(X_scaled[tr_idx], y_train[tr_idx])
        fold_alpha = inner_cv.alpha_
        fold_l1_ratio = inner_cv.l1_ratio_

        fold_model = ElasticNet(
            alpha=fold_alpha,
            l1_ratio=fold_l1_ratio,
            max_iter=max_iter,
        )
        fold_model.fit(X_scaled[tr_idx], y_train[tr_idx])
        y_pred_oof[val_idx] = fold_model.predict(X_scaled[val_idx])

        m = _compute_metrics(y_train[val_idx], y_pred_oof[val_idx])
        fold_metrics.append({
            "fold": fold_idx,
            **m,
            "n_val": len(val_idx),
            "alpha": fold_alpha,
            "l1_ratio": fold_l1_ratio,
        })

    # Final model: HP selection on the full training set
    final_model = ElasticNetCV(
        l1_ratio=_l1_ratios,
        alphas=alphas,
        cv=n_splits,
        max_iter=max_iter,
        random_state=random_state,
    )
    final_model.fit(X_scaled, y_train)

    return ModelResult(
        name=name,
        sensor=sensor,
        feature_names=feature_names,
        estimator=final_model,
        scaler=scaler,
        y_true=y_train,
        y_pred=y_pred_oof,
        fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# Bayesian Ridge training
# ---------------------------------------------------------------------------


def train_bayesian_ridge_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_splits: int = 5,
    max_iter: int = 300,
    random_state: int = 42,
    name: str = "BayesianRidge",
    sensor: str = "unknown",
) -> ModelResult:
    """Train a Bayesian Ridge with single-level KFold CV on the training set.

    BayesianRidge is an empirical Bayes model that tunes its own
    regularisation hyperparameters (alpha, lambda) during fitting, so no
    explicit hyperparameter grid search is needed.  Each CV fold fits a
    fresh BayesianRidge on the fold's training data and predicts on the
    fold's validation data.  A final model is refit on the full training set.

    Parameters
    ----------
    X_train:
        Training feature matrix (samples × features).
    y_train:
        Training target values (kappa).
    n_splits:
        Number of KFold splits.
    max_iter:
        Maximum number of EM iterations.
    random_state:
        Random seed.
    name, sensor:
        Identifiers for plots and summaries.

    Returns
    -------
    ModelResult
    """
    feature_names = X_train.columns.tolist()
    y_train = np.asarray(y_train, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_pred_oof = np.empty_like(y_train)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_scaled)):
        model = BayesianRidge(max_iter=max_iter)
        model.fit(X_scaled[tr_idx], y_train[tr_idx])
        y_pred_oof[val_idx] = model.predict(X_scaled[val_idx])

        m = _compute_metrics(y_train[val_idx], y_pred_oof[val_idx])
        fold_metrics.append({
            "fold": fold_idx,
            **m,
            "n_val": len(val_idx),
            "alpha_": float(model.alpha_),
            "lambda_": float(model.lambda_),
        })

    # Final model: refit on full training set
    final_model = BayesianRidge(max_iter=max_iter)
    final_model.fit(X_scaled, y_train)

    return ModelResult(
        name=name,
        sensor=sensor,
        feature_names=feature_names,
        estimator=final_model,
        scaler=scaler,
        y_true=y_train,
        y_pred=y_pred_oof,
        fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# Polynomial regression training
# ---------------------------------------------------------------------------


def train_polynomial_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    degree: int = 2,
    top_k: int | None = None,
    n_splits: int = 5,
    alphas: list[float] | None = None,
    random_state: int = 42,
    name: str = "Polynomial",
    sensor: str = "unknown",
) -> ModelResult:
    """Train a polynomial regression (PolynomialFeatures + Ridge) with KFold CV.

    Uses RidgeCV on the full training set to select the regularisation
    strength, then runs a single KFold loop at that fixed alpha to collect
    out-of-fold predictions.  The final model is a sklearn Pipeline
    (StandardScaler → PolynomialFeatures → Ridge) refit on the full
    training set.

    Parameters
    ----------
    X_train:
        Training feature matrix (samples × features).
    y_train:
        Training target values (kappa).
    degree:
        Polynomial degree (default 2).
    top_k:
        If set, restrict input to the *top_k* features ranked by absolute
        Pearson correlation with *y_train* before polynomial expansion.
        Degree-2 expansion of N features produces ~N²/2 terms, so keeping
        N small avoids an overparameterised model.  ``None`` uses all
        features.
    n_splits:
        Number of KFold splits.
    alphas:
        Ridge regularisation grid.  ``None`` uses a log-spaced default.
    random_state:
        Random seed.
    name, sensor:
        Identifiers for plots and summaries.

    Returns
    -------
    ModelResult
        The estimator is a fitted sklearn Pipeline; feature_names are the
        expanded polynomial feature names.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import PolynomialFeatures

    class _ColumnSelector(BaseEstimator, TransformerMixin):
        """Select columns by integer index — used as a Pipeline step."""
        def __init__(self, indices: list[int]) -> None:
            self.indices = indices
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            if hasattr(X, "iloc"):
                return X.iloc[:, self.indices].values
            return X[:, self.indices]

    y_train = np.asarray(y_train, dtype=float)

    # Determine which columns to use (all, or top-k by |Pearson r|)
    if top_k is not None and top_k < X_train.shape[1]:
        corr = np.abs(
            np.array([np.corrcoef(X_train.iloc[:, j], y_train)[0, 1]
                      for j in range(X_train.shape[1])])
        )
        top_idx = np.argsort(corr)[::-1][:top_k].tolist()
    else:
        top_idx = list(range(X_train.shape[1]))

    col_names = [X_train.columns[i] for i in top_idx]
    X_sel = X_train.iloc[:, top_idx]
    _alphas = alphas or np.logspace(-3, 4, 15)

    # Nested CV: alpha selection is local to each outer fold's training split.
    # RidgeCV with cv=None uses GCV (efficient leave-one-out) as the inner CV.
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred_oof = np.empty_like(y_train)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_sel.values)):
        fold_scaler = StandardScaler()
        fold_poly = PolynomialFeatures(degree=degree, include_bias=False)

        X_tr_poly = fold_poly.fit_transform(
            fold_scaler.fit_transform(X_sel.values[tr_idx])
        )
        X_val_poly = fold_poly.transform(
            fold_scaler.transform(X_sel.values[val_idx])
        )

        # Inner HP selection on this fold's training data only (GCV)
        inner_ridge_cv = RidgeCV(alphas=_alphas)
        inner_ridge_cv.fit(X_tr_poly, y_train[tr_idx])
        fold_alpha = float(inner_ridge_cv.alpha_)

        fold_model = Ridge(alpha=fold_alpha)
        fold_model.fit(X_tr_poly, y_train[tr_idx])
        y_pred_oof[val_idx] = fold_model.predict(X_val_poly)

        m = _compute_metrics(y_train[val_idx], y_pred_oof[val_idx])
        fold_metrics.append({
            "fold": fold_idx,
            **m,
            "n_val": len(val_idx),
            "alpha": fold_alpha,
        })

    # Select alpha on the full training set for the final model only
    scaler_ref = StandardScaler()
    poly_ref = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_ref = poly_ref.fit_transform(scaler_ref.fit_transform(X_sel.values))
    best_alpha = float(RidgeCV(alphas=_alphas).fit(X_poly_ref, y_train).alpha_)

    # Final model: Pipeline refit on full training set.
    # _ColumnSelector is first so the full feature matrix can be passed in
    # at predict time — it selects the same top_k columns used during training.
    pipeline_steps = [
        ("selector", _ColumnSelector(top_idx)),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=best_alpha)),
    ]
    final_model = Pipeline(pipeline_steps)
    final_model.fit(X_train, y_train)

    # Derive feature names from the fitted pipeline's own poly step so names
    # are guaranteed to match the Ridge's coef_.
    feature_names = (
        final_model.named_steps["poly"]
        .get_feature_names_out(col_names)
        .tolist()
    )

    # Dummy scaler kept for ModelResult API compatibility
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(X_train.values)

    return ModelResult(
        name=name,
        sensor=sensor,
        feature_names=feature_names,
        estimator=final_model,
        scaler=dummy_scaler,
        y_true=y_train,
        y_pred=y_pred_oof,
        fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------


def train_lightgbm_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_splits: int = 5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    min_child_samples: int = 10,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 50,
    param_grid: dict[str, list] | None = None,
    n_trials: int = 50,
    random_state: int = 42,
    name: str = "LightGBM",
    sensor: str = "unknown",
) -> ModelResult:
    """Train a LightGBM regressor with nested KFold CV on the training set.

    If ``param_grid`` is provided, an inner 3-fold ``GridSearchCV`` runs on
    each outer fold's training slice to select ``learning_rate``,
    ``num_leaves``, ``max_depth``, or any other keys in the grid.  A fast
    fixed ``n_estimators`` (200) is used for screening; early stopping then
    determines the true number of rounds for the actual fold training.

    Without ``param_grid`` the behaviour is unchanged (fixed hyperparameters
    from the call arguments, early stopping only).

    Parameters
    ----------
    X_train:
        Training feature matrix (samples × features).
    y_train:
        Training target values (kappa).
    n_splits:
        Number of KFold splits.
    n_estimators, learning_rate, max_depth, num_leaves, ... : various
        LightGBM base hyperparameters (used as defaults / fixed values when
        the corresponding key is absent from ``param_grid``).
    early_stopping_rounds:
        Stop if validation metric doesn't improve for this many rounds.
    param_grid:
        Dict mapping hyperparameter names to lists of candidate values,
        e.g. ``{"learning_rate": [0.01, 0.05], "num_leaves": [31, 63]}``.
        Each list becomes ``suggest_categorical`` choices in the Optuna
        study.  ``None`` skips tuning entirely.
    n_trials:
        Number of Optuna trials per study (default 50).
    random_state:
        Random seed.
    name, sensor:
        Identifiers for plots and summaries.

    Returns
    -------
    ModelResult
    """
    import optuna

    feature_names = X_train.columns.tolist()
    y_train = np.asarray(y_train, dtype=float)

    base_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )

    # LightGBM doesn't need scaling — store a fitted scaler for
    # ModelResult API compatibility only.
    scaler = StandardScaler()
    scaler.fit(X_train.values)

    X = X_train  # keep as DataFrame so feature names propagate to all LGBMRegressor fits
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # n_estimators used for fast Optuna screening (no early stopping)
    _screen_n = min(200, n_estimators)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _make_objective(X_tr: pd.DataFrame, y_tr: np.ndarray) -> object:
        """Return an Optuna objective that evaluates HPs via 3-fold inner CV."""
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

        def objective(trial: optuna.Trial) -> float:
            params = {
                **base_params,
                "n_estimators": _screen_n,
                **{k: trial.suggest_categorical(k, v) for k, v in param_grid.items()},
            }
            scores = []
            for itr, ival in inner_cv.split(X_tr):
                m = lgb.LGBMRegressor(**params)
                m.fit(X_tr.iloc[itr], y_tr[itr], callbacks=[lgb.log_evaluation(period=0)])
                scores.append(float(np.sqrt(mean_squared_error(y_tr[ival], m.predict(X_tr.iloc[ival])))))
            return float(np.mean(scores))

        return objective

    y_pred_oof = np.empty_like(y_train)
    fold_metrics: list[dict[str, float]] = []
    best_iters: list[int] = []

    fold_iter = tqdm(
        enumerate(cv.split(X)),
        total=n_splits,
        desc=f"{name} — CV folds",
        unit="fold",
    )
    for fold_idx, (tr_idx, val_idx) in fold_iter:
        fold_params = dict(base_params)

        if param_grid:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state + fold_idx),
            )
            with tqdm(total=n_trials, desc=f"  fold {fold_idx+1}/{n_splits} — Optuna", unit="trial", leave=False) as pbar:
                def _optuna_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                    pbar.update(1)
                    pbar.set_postfix(best_rmse=f"{study.best_value:.4f}")
                study.optimize(
                    _make_objective(X.iloc[tr_idx], y_train[tr_idx]),
                    n_trials=n_trials,
                    callbacks=[_optuna_cb],
                )
            fold_params.update(study.best_params)

        model = lgb.LGBMRegressor(**fold_params)
        model.fit(
            X.iloc[tr_idx],
            y_train[tr_idx],
            eval_set=[(X.iloc[val_idx], y_train[val_idx])],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        y_pred_oof[val_idx] = model.predict(X.iloc[val_idx])
        best_iters.append(model.best_iteration_)

        m = _compute_metrics(y_train[val_idx], y_pred_oof[val_idx])
        fold_iter.set_postfix(r2=f"{m['r2']:.3f}", rmse=f"{m['rmse']:.3f}")
        fold_metrics.append({
            "fold": fold_idx,
            **m,
            "n_val": len(val_idx),
            "best_iteration": model.best_iteration_,
            **({"best_params": str(study.best_params)} if param_grid else {}),
        })

    # Final model: Optuna study on the full training set, then refit
    final_params = dict(base_params)
    if param_grid:
        final_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        with tqdm(total=n_trials, desc="  final model — Optuna", unit="trial", leave=False) as pbar:
            def _final_optuna_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                pbar.update(1)
                pbar.set_postfix(best_rmse=f"{study.best_value:.4f}")
            final_study.optimize(
                _make_objective(X, y_train),
                n_trials=n_trials,
                callbacks=[_final_optuna_cb],
            )
        final_params.update(final_study.best_params)

    mean_best_iter = max(1, int(np.mean(best_iters)))
    final_params["n_estimators"] = mean_best_iter
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X, y_train)

    return ModelResult(
        name=name,
        sensor=sensor,
        feature_names=feature_names,
        estimator=final_model,
        scaler=scaler,
        y_true=y_train,
        y_pred=y_pred_oof,
        fold_metrics=fold_metrics,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def results_summary_table(results: list[ModelResult]) -> pd.DataFrame:
    """Combine multiple ModelResults into a comparison DataFrame."""
    rows = []
    for r in results:
        est = r.estimator
        weights, is_importance = get_feature_weights(r)
        row: dict[str, Any] = {
            "model": r.name,
            "sensor": r.sensor,
            "n_features": len(r.feature_names),
            "n_nonzero": int(np.sum(np.abs(weights) > 1e-10)),
            "CV_R²": r.r2,
            "CV_MAE": r.mae,
            "CV_RMSE": r.rmse,
        }

        # Hold-out metrics (if available)
        if r.holdout_metrics:
            row["HO_R²"] = r.holdout_metrics["r2"]
            row["HO_MAE"] = r.holdout_metrics["mae"]
            row["HO_RMSE"] = r.holdout_metrics["rmse"]

        # Model-specific hyperparameters
        if hasattr(est, "l1_ratio"):
            row["alpha"] = getattr(est, "alpha_", None) or est.alpha
            row["l1_ratio"] = getattr(est, "l1_ratio_", None) or est.l1_ratio
        elif hasattr(est, "lambda_"):
            row["alpha_noise"] = est.alpha_
            row["lambda_weights"] = est.lambda_
        elif hasattr(est, "n_estimators"):
            row["n_estimators"] = est.n_estimators
            row["learning_rate"] = getattr(est, "learning_rate", None)

        rows.append(row)
    return pd.DataFrame(rows)
