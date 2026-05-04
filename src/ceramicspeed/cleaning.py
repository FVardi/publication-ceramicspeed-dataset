"""
cleaning.py
===========
Data cleaning utilities for CeramicSpeed bearing analysis.

Two levels of cleaning
----------------------

**Signal-level** (applied to raw voltage arrays *before* feature
extraction):

- ``clean_signal``            — one-call pipeline for all signal checks.
- ``remove_signal_nan_inf``   — replace NaN/Inf samples.
- ``detect_clipping``         — flag signals that hit the DAQ rails.
- ``detect_saturation``       — flag signals with suspiciously flat regions.
- ``remove_signal_outliers``  — z-score spike removal + interpolation.
- ``validate_signal``         — reject sweeps too short or all-zero.

**Feature-level** (applied to the feature DataFrame *after* extraction):

- ``remove_nan_inf``          — handle NaN/Inf feature values.
- ``remove_constant_features``— drop zero-variance columns.
- ``remove_outliers``         — IQR / z-score row-wise outlier removal.
- ``filter_by_metadata``      — RPM / temperature operational filters.
- ``clean_features``          — one-call pipeline for feature cleaning.

All functions are pure — they return new arrays/DataFrames and never
modify input data in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    # Signal-level
    "clean_signal",
    "remove_signal_nan_inf",
    "detect_clipping",
    "detect_saturation",
    "remove_signal_outliers",
    "validate_signal",
    "SignalQualityReport",
    # Feature-level
    "remove_nan_inf",
    "remove_constant_features",
    "remove_outliers",
    "filter_by_metadata",
    "clean_features",
]


# =====================================================================
#  SIGNAL-LEVEL CLEANING  (raw voltage → clean voltage)
# =====================================================================


@dataclass
class SignalQualityReport:
    """Summary of signal-level cleaning applied to a single sweep/sensor."""

    is_valid: bool = True
    rejection_reason: str | None = None

    n_nan_inf_replaced: int = 0
    is_clipped: bool = False
    clipping_fraction: float = 0.0
    is_saturated: bool = False
    saturation_longest_run: int = 0
    n_outlier_samples_replaced: int = 0

    def __str__(self) -> str:
        if not self.is_valid:
            return f"REJECTED: {self.rejection_reason}"
        parts: list[str] = []
        if self.n_nan_inf_replaced:
            parts.append(f"nan/inf={self.n_nan_inf_replaced}")
        if self.is_clipped:
            parts.append(f"clipped ({self.clipping_fraction:.1%})")
        if self.is_saturated:
            parts.append(f"saturated (run={self.saturation_longest_run})")
        if self.n_outlier_samples_replaced:
            parts.append(f"outliers={self.n_outlier_samples_replaced}")
        return "OK" + (f" [{', '.join(parts)}]" if parts else "")


# ---------------------------------------------------------------------------
# Individual signal-cleaning steps
# ---------------------------------------------------------------------------


def validate_signal(
    signal: np.ndarray,
    *,
    min_length: int = 64,
) -> tuple[bool, str | None]:
    """Check whether a signal is usable at all.

    Parameters
    ----------
    signal:
        1-D voltage array.
    min_length:
        Minimum number of samples required.

    Returns
    -------
    tuple[bool, str | None]
        ``(is_valid, reason)`` — reason is ``None`` when valid.
    """
    if signal.size < min_length:
        return False, f"too short ({signal.size} < {min_length})"
    if np.all(signal == 0):
        return False, "all-zero signal"
    if np.all(np.isnan(signal)):
        return False, "all-NaN signal"
    return True, None


def remove_signal_nan_inf(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN and Inf values with linear interpolation.

    For leading/trailing NaN/Inf, nearest-neighbour extrapolation is used.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(cleaned_signal, n_replaced)``
    """
    bad = ~np.isfinite(signal)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return signal.copy(), 0

    out = signal.copy()
    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        # Entire signal is bad — replace with zeros
        return np.zeros_like(signal), n_bad

    out[bad] = np.interp(
        np.where(bad)[0], good_idx, signal[good_idx]
    )
    return out, n_bad


def detect_clipping(
    signal: np.ndarray,
    *,
    clip_fraction_threshold: float = 0.01,
    rail_percentile: float = 99.9,
) -> tuple[bool, float]:
    """Detect whether a signal is clipped (hitting DAQ voltage rails).

    A signal is considered clipped if more than
    *clip_fraction_threshold* of samples sit at the min or max rail
    value (estimated from the *rail_percentile*).

    Returns
    -------
    tuple[bool, float]
        ``(is_clipped, fraction_clipped)``
    """
    if signal.size == 0:
        return False, 0.0

    lo = np.percentile(signal, 100 - rail_percentile)
    hi = np.percentile(signal, rail_percentile)

    at_rail = np.sum((signal <= lo) | (signal >= hi))
    frac = float(at_rail / signal.size)
    return frac >= clip_fraction_threshold, frac


def detect_saturation(
    signal: np.ndarray,
    *,
    flat_tolerance: float = 1e-10,
    min_run_length: int = 50,
) -> tuple[bool, int]:
    """Detect suspiciously flat (saturated) regions in a signal.

    A "flat run" is a contiguous stretch of samples where consecutive
    differences are below *flat_tolerance*.

    Returns
    -------
    tuple[bool, int]
        ``(is_saturated, longest_run_length)``
    """
    if signal.size < 2:
        return False, 0

    diffs = np.abs(np.diff(signal))
    flat = diffs < flat_tolerance

    # Find longest run of True in `flat`
    if not flat.any():
        return False, 0

    # Efficient run-length encoding
    changes = np.diff(flat.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases (run starts at index 0 or ends at last index)
    if flat[0]:
        starts = np.concatenate([[0], starts])
    if flat[-1]:
        ends = np.concatenate([ends, [len(flat)]])

    if len(starts) == 0:
        return False, 0

    run_lengths = ends - starts
    longest = int(run_lengths.max())
    return longest >= min_run_length, longest


def remove_signal_outliers(
    signal: np.ndarray,
    *,
    z_threshold: float = 6.0,
    window: int = 5,
) -> tuple[np.ndarray, int]:
    """Remove spike outliers from a signal by z-score thresholding.

    Outlier samples are replaced with the local median within a
    *window*-wide neighbourhood.

    Parameters
    ----------
    signal:
        1-D voltage array.
    z_threshold:
        Number of standard deviations to consider a sample an outlier.
        Default ``6.0`` targets only extreme spikes.
    window:
        Half-width of the local median window used to replace outliers.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(cleaned_signal, n_replaced)``
    """
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma < 1e-30:
        return signal.copy(), 0

    z = np.abs((signal - mu) / sigma)
    outlier_mask = z > z_threshold
    n_outliers = int(outlier_mask.sum())

    if n_outliers == 0:
        return signal.copy(), 0

    out = signal.copy()
    outlier_indices = np.where(outlier_mask)[0]

    for idx in outlier_indices:
        lo = max(0, idx - window)
        hi = min(len(signal), idx + window + 1)
        neighbourhood = signal[lo:hi]
        # Use median of non-outlier neighbours
        local_good = neighbourhood[~outlier_mask[lo:hi]]
        if len(local_good) > 0:
            out[idx] = np.median(local_good)
        else:
            out[idx] = mu  # fallback

    return out, n_outliers

# ---------------------------------------------------------------------------
# One-call signal cleaning pipeline
# ---------------------------------------------------------------------------


def clean_signal(
    signal: np.ndarray,
    *,
    min_length: int = 64,
    fix_nan_inf: bool = True,
    check_clipping: bool = True,
    clip_fraction_threshold: float = 0.01,
    check_saturation: bool = True,
    saturation_min_run: int = 50,
    remove_outliers_z: float | None = 6.0,
    outlier_window: int = 5,
    reject_clipped: bool = False,
    reject_saturated: bool = False,
) -> tuple[np.ndarray, SignalQualityReport]:
    """Apply all signal-level cleaning steps in sequence.

    The pipeline order is:

    1. Validate (length, all-zero, all-NaN).
    2. Replace NaN/Inf with interpolation.
    3. Check for clipping (optionally reject).
    4. Check for saturation (optionally reject).
    5. Remove spike outliers via z-score.

    Parameters
    ----------
    signal:
        Raw 1-D voltage array.
    min_length:
        Minimum signal length; shorter signals are rejected.
    fix_nan_inf:
        Replace NaN/Inf samples with interpolated values.
    check_clipping:
        Run clipping detection.
    clip_fraction_threshold:
        Fraction of samples at the rail to flag clipping.
    check_saturation:
        Run saturation detection.
    saturation_min_run:
        Minimum flat-run length to flag saturation.
    remove_outliers_z:
        Z-score threshold for spike removal.  ``None`` skips.
    outlier_window:
        Local median window half-width for outlier replacement.
    reject_clipped:
        If ``True``, clipped signals are marked invalid.
    reject_saturated:
        If ``True``, saturated signals are marked invalid.

    Returns
    -------
    tuple[np.ndarray, SignalQualityReport]
        ``(cleaned_signal, report)``
    """
    report = SignalQualityReport()
    out = np.asarray(signal, dtype=float)

    # 1. Validate
    is_valid, reason = validate_signal(out, min_length=min_length)
    if not is_valid:
        report.is_valid = False
        report.rejection_reason = reason
        return out, report

    # 2. NaN / Inf
    if fix_nan_inf:
        out, n_replaced = remove_signal_nan_inf(out)
        report.n_nan_inf_replaced = n_replaced

    # 3. Clipping
    if check_clipping:
        is_clipped, clip_frac = detect_clipping(
            out, clip_fraction_threshold=clip_fraction_threshold
        )
        report.is_clipped = is_clipped
        report.clipping_fraction = clip_frac
        if reject_clipped and is_clipped:
            report.is_valid = False
            report.rejection_reason = (
                f"clipping detected ({clip_frac:.1%} at rail)"
            )
            return out, report

    # 4. Saturation
    if check_saturation:
        is_sat, longest_run = detect_saturation(
            out, min_run_length=saturation_min_run
        )
        report.is_saturated = is_sat
        report.saturation_longest_run = longest_run
        if reject_saturated and is_sat:
            report.is_valid = False
            report.rejection_reason = (
                f"saturation detected (flat run = {longest_run} samples)"
            )
            return out, report

    # 5. Spike outliers
    if remove_outliers_z is not None:
        out, n_spikes = remove_signal_outliers(
            out, z_threshold=remove_outliers_z, window=outlier_window
        )
        report.n_outlier_samples_replaced = n_spikes

    return out, report


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


def remove_nan_inf(
    df: pd.DataFrame,
    strategy: Literal["drop", "median", "mean", "zero"] = "drop",
) -> pd.DataFrame:
    """Handle NaN and Inf values in a feature DataFrame.

    Parameters
    ----------
    df:
        Feature DataFrame (samples × features).
    strategy:
        How to handle problematic values:

        - ``"drop"``   — drop any row containing NaN or Inf.
        - ``"median"`` — replace NaN/Inf with the column median.
        - ``"mean"``   — replace NaN/Inf with the column mean.
        - ``"zero"``   — replace NaN/Inf with 0.0.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (same columns, potentially fewer rows).
    """
    # Replace Inf/-Inf with NaN so all strategies work uniformly
    out = df.replace([np.inf, -np.inf], np.nan)

    if strategy == "drop":
        return out.dropna()
    elif strategy == "median":
        return out.fillna(out.median())
    elif strategy == "mean":
        return out.fillna(out.mean())
    elif strategy == "zero":
        return out.fillna(0.0)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: 'drop', 'median', 'mean', 'zero'."
        )


# ---------------------------------------------------------------------------
# Constant / near-constant feature removal
# ---------------------------------------------------------------------------


def remove_constant_features(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features with zero (or near-zero) variance.

    Parameters
    ----------
    df:
        Feature DataFrame (samples × features).
    threshold:
        Minimum variance to keep a feature.  Use ``0.0`` to only drop
        perfectly constant columns.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(cleaned_df, dropped_columns)`` — the cleaned DataFrame and a
        list of column names that were removed.
    """
    numeric = df.select_dtypes(include=[np.number])
    variances = numeric.var()
    to_drop = variances[variances <= threshold].index.tolist()
    return df.drop(columns=to_drop), to_drop


# ---------------------------------------------------------------------------
# Outlier detection and removal
# ---------------------------------------------------------------------------


def remove_outliers(
    df: pd.DataFrame,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 3.0,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Detect and remove rows containing outlier values.

    Parameters
    ----------
    df:
        Feature DataFrame (samples × features).
    method:
        Detection method:

        - ``"iqr"`` — Interquartile range.  A value is an outlier if it
          falls more than ``threshold × IQR`` below Q1 or above Q3.
        - ``"zscore"`` — Z-score.  A value is an outlier if
          ``|z| > threshold``.
    threshold:
        Sensitivity parameter.  For IQR typically 1.5 (mild) or 3.0
        (extreme).  For z-score typically 3.0.
    columns:
        Subset of columns to check.  When ``None``, all numeric columns
        are checked.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(cleaned_df, outlier_mask)`` — the cleaned DataFrame and a
        boolean Series indicating which rows were flagged as outliers.
    """
    numeric = df.select_dtypes(include=[np.number])
    if columns is not None:
        numeric = numeric[columns]

    if method == "iqr":
        Q1 = numeric.quantile(0.25)
        Q3 = numeric.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outlier_mask = ((numeric < lower) | (numeric > upper)).any(axis=1)

    elif method == "zscore":
        means = numeric.mean()
        stds = numeric.std()
        # Avoid division by zero for constant columns
        stds = stds.replace(0, np.nan)
        z_scores = (numeric - means) / stds
        outlier_mask = (z_scores.abs() > threshold).any(axis=1)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'iqr', 'zscore'."
        )

    return df[~outlier_mask], outlier_mask


# ---------------------------------------------------------------------------
# Metadata-based operational filters
# ---------------------------------------------------------------------------


def filter_by_metadata(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    rpm_max: float | None = None,
    rpm_min: float | None = None,
    temp_min: float | None = None,
    temp_max: float | None = None,
    rpm_col: str = "rpm",
    temp_col: str = "temperature_c",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply operational filters based on metadata columns.

    Filters both the feature DataFrame and the metadata DataFrame using
    the same boolean mask, keeping them aligned.

    Parameters
    ----------
    df:
        Feature DataFrame.
    metadata:
        Metadata DataFrame (must have same row count as *df*).
    rpm_max, rpm_min:
        Upper/lower bounds on RPM.  ``None`` means no limit.
    temp_min, temp_max:
        Upper/lower bounds on temperature [°C].  ``None`` means no limit.
    rpm_col, temp_col:
        Column names in *metadata* for RPM and temperature.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(filtered_df, filtered_metadata)``
    """
    mask = pd.Series(True, index=metadata.index)

    if rpm_max is not None and rpm_col in metadata.columns:
        mask &= metadata[rpm_col] < rpm_max
    if rpm_min is not None and rpm_col in metadata.columns:
        mask &= metadata[rpm_col] >= rpm_min
    if temp_min is not None and temp_col in metadata.columns:
        mask &= metadata[temp_col] >= temp_min
    if temp_max is not None and temp_col in metadata.columns:
        mask &= metadata[temp_col] <= temp_max

    return df[mask], metadata[mask]


# ---------------------------------------------------------------------------
# Convenience: chain common cleaning steps
# ---------------------------------------------------------------------------


def clean_features(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    rpm_max: float | None = None,
    nan_strategy: Literal["drop", "median", "mean", "zero"] = "drop",
    drop_constant: bool = True,
    remove_outliers_method: Literal["iqr", "zscore"] | None = None,
    outlier_threshold: float = 3.0,
    outlier_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Chain common cleaning steps in one call.

    Applies the following pipeline in order:

    1. Metadata-based operational filters (RPM).
    2. NaN/Inf handling.
    3. Constant feature removal.
    4. Outlier removal (optional).

    Parameters
    ----------
    df:
        Feature DataFrame.
    metadata:
        Metadata DataFrame (same row count as *df*).
    rpm_max:
        Maximum RPM threshold.  ``None`` skips RPM filtering.
    nan_strategy:
        Strategy for NaN/Inf values (see :func:`remove_nan_inf`).
    drop_constant:
        Whether to remove constant features.
    remove_outliers_method:
        Outlier detection method.  ``None`` skips outlier removal.
    outlier_threshold:
        Threshold for outlier detection.
    outlier_columns:
        Subset of columns to check for outliers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        ``(cleaned_df, cleaned_metadata, report)`` where *report* is a
        dict summarising what was removed at each step.
    """
    report: dict = {
        "initial_rows": len(df),
        "initial_features": len(df.columns),
    }

    # Identify feature columns (exclude index-like columns)
    index_cols = [c for c in ["file", "sweep", "sensor"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in index_cols]

    # 1. Metadata filter
    if rpm_max is not None:
        df, metadata = filter_by_metadata(df, metadata, rpm_max=rpm_max)
        report["after_rpm_filter"] = len(df)

    # 2. NaN / Inf handling (only on feature columns)
    feature_part = remove_nan_inf(df[feature_cols], strategy=nan_strategy)
    # Align metadata with surviving rows
    if nan_strategy == "drop":
        surviving_idx = feature_part.index
        df = pd.concat([df[index_cols].loc[surviving_idx], feature_part], axis=1)
        metadata = metadata.loc[surviving_idx]
    else:
        df = pd.concat([df[index_cols], feature_part], axis=1)
    report["after_nan_handling"] = len(df)

    # 3. Constant feature removal
    dropped_cols: list[str] = []
    if drop_constant:
        cleaned_features, dropped_cols = remove_constant_features(
            df[feature_cols].reindex(df.index)
        )
        remaining_feature_cols = cleaned_features.columns.tolist()
        df = pd.concat(
            [df[index_cols].reindex(df.index), cleaned_features], axis=1
        )
        report["dropped_constant_features"] = dropped_cols
    else:
        remaining_feature_cols = feature_cols

    # 4. Outlier removal
    if remove_outliers_method is not None:
        check_cols = outlier_columns or remaining_feature_cols
        check_cols = [c for c in check_cols if c in df.columns]
        df, outlier_mask = remove_outliers(
            df,
            method=remove_outliers_method,
            threshold=outlier_threshold,
            columns=check_cols,
        )
        metadata = metadata.loc[df.index]
        report["outliers_removed"] = int(outlier_mask.sum())
        report["after_outlier_removal"] = len(df)

    report["final_rows"] = len(df)
    report["final_features"] = len(
        [c for c in df.columns if c not in index_cols]
    )

    return df, metadata, report
