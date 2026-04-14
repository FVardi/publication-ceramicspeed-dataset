"""
loading.py
==========
Data loading utilities for CeramicSpeed bearing analysis.

Loads raw measurement data from HDF5 files, applies signal-level cleaning
*before* feature extraction, and returns structured feature rows and
metadata rows ready for downstream analysis.

Functions
---------
load_hdf5_file(file_path, sensors=None)
    Load raw signals and metadata from a single HDF5 measurement file.

load_and_process_file(file_path, ..., signal_clean_cfg=None)
    Load an HDF5 file, clean each signal, then extract features for every
    sweep/sensor pair.

load_and_process_files_parallel(file_paths, ..., signal_clean_cfg=None)
    Parallel wrapper around ``load_and_process_file``.

load_parquet_pair(output_dir)
    Load features.parquet and metadata.parquet from a pipeline output
    directory.

discover_hdf5_files(input_dir)
    Return a sorted list of HDF5 file paths in a directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import extract_features, bandpass_filter
from .cleaning import clean_signal, SignalQualityReport

logger = logging.getLogger(__name__)

__all__ = [
    "load_hdf5_file",
    "load_and_process_file",
    "load_and_process_files_parallel",
    "load_parquet_pair",
    "discover_hdf5_files",
]

#: Default sensor channels present in every sweep group.
DEFAULT_SENSORS: tuple[str, ...] = ("AE", "Ultrasound")

# Older files use "Ultrasound"; newer scope files renamed it "UL".
# These mappings normalise both to the canonical names used everywhere else.
_CHANNEL_ALIASES: dict[str, str] = {"UL": "Ultrasound"}
_SWEEP_ATTR_ALIASES: dict[str, str] = {
    "telem_rpm_meas":   "rpm",
    "telem_omron_pv_c": "temperature_c",
    "telem_mass_g":     "load_g",
}

# Fallback viscosity data keyed on lubricant product_name, for files that
# omit viscosity_40c_cst / viscosity_100c_cst from their metadata.
_VISCOSITY_FALLBACK: dict[str, dict[str, float]] = {
    "Keratech 22": {"viscosity_40c_cst": 22.0, "viscosity_100c_cst": 4.1},
}

#: Default signal-cleaning settings.  Callers can override individual
#: keys via the ``signal_clean_cfg`` parameter.
DEFAULT_SIGNAL_CLEAN_CFG: dict[str, Any] = {
    "enabled": True,
    "min_length": 64,
    "fix_nan_inf": True,
    "check_clipping": True,
    "clip_fraction_threshold": 0.01,
    "check_saturation": True,
    "saturation_min_run": 50,
    "remove_outliers_z": 6.0,
    "outlier_window": 5,
    "apply_detrend": False,
    "reject_clipped": False,
    "reject_saturated": False,
}


def _resolve_signal_cfg(
    user_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge user overrides into the default signal-cleaning config."""
    cfg = dict(DEFAULT_SIGNAL_CLEAN_CFG)
    if user_cfg is not None:
        cfg.update(user_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Low-level HDF5 access
# ---------------------------------------------------------------------------


def load_hdf5_file(
    file_path: str | Path,
    sensors: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load raw signals and metadata from a single HDF5 measurement file.

    Parameters
    ----------
    file_path:
        Path to the ``.hdf5`` measurement file.
    sensors:
        Sensor channel names to read.  Defaults to ``("AE", "Ultrasound")``.

    Returns
    -------
    dict
        Keys: ``"file_name"``, ``"fs"`` (sampling frequency),
        ``"lubricant_metadata"``, ``"bearing_metadata"``,
        ``"sweeps"`` (list of dicts with ``"name"``, ``"test_parameters"``,
        and per-sensor ``"voltage"`` arrays).
    """
    file_path = Path(file_path)
    sensors = sensors or DEFAULT_SENSORS

    with h5py.File(file_path, "r") as f:
        sweeps_grp = f["sweeps"]

        # Build a reverse alias map so we can find each canonical sensor in the
        # file regardless of whether it uses the old or new channel name.
        # e.g. canonical "Ultrasound" may be stored as "UL".
        _alias_to_canonical = {v: k for k, v in _CHANNEL_ALIASES.items()}  # {"Ultrasound": "UL"}

        def _file_channel(canonical: str, sweep_grp) -> str:
            """Return the name actually present in *sweep_grp* for *canonical*."""
            if canonical in sweep_grp:
                return canonical
            alias = _alias_to_canonical.get(canonical)
            if alias and alias in sweep_grp:
                return alias
            raise KeyError(f"Channel '{canonical}' (or alias) not found in sweep")

        # Sampling frequency — identical across all sweeps and sensors
        first_sweep = sweeps_grp[list(sweeps_grp.keys())[0]]
        first_ch = _file_channel(sensors[0], first_sweep)
        time_axis: np.ndarray = first_sweep[first_ch]["time"][()]
        fs: float = 1.0 / float(np.mean(np.diff(time_axis)))

        lubricant_metadata = dict(f["metadata"]["lubricant"].attrs)
        if "viscosity_40c_cst" not in lubricant_metadata:
            product = lubricant_metadata.get("product_name", "")
            fallback = _VISCOSITY_FALLBACK.get(product, {})
            if fallback:
                lubricant_metadata.update(fallback)
                logger.info(
                    "%s: viscosity data missing for '%s', using fallback values",
                    file_path.name, product,
                )
            else:
                logger.warning(
                    "%s: viscosity data missing for '%s' and no fallback found",
                    file_path.name, product,
                )
        bearing_metadata = dict(f["metadata"]["bearing"].attrs)

        sweep_list: list[dict[str, Any]] = []
        for sweep_name, sweep in sweeps_grp.items():
            # Normalise sweep-level telemetry attribute names
            raw_attrs = dict(sweep.attrs)
            norm_attrs = {
                _SWEEP_ATTR_ALIASES.get(k, k): v for k, v in raw_attrs.items()
            }
            sweep_data: dict[str, Any] = {
                "name": sweep_name,
                "test_parameters": norm_attrs,
            }
            for sensor_name in sensors:
                file_ch = _file_channel(sensor_name, sweep)
                sweep_data[sensor_name] = sweep[file_ch]["voltage"][()]
            sweep_list.append(sweep_data)

    return {
        "file_name": file_path.stem,
        "fs": fs,
        "lubricant_metadata": lubricant_metadata,
        "bearing_metadata": bearing_metadata,
        "sweeps": sweep_list,
    }


# ---------------------------------------------------------------------------
# Feature extraction from HDF5 (loading + signal cleaning + features)
# ---------------------------------------------------------------------------


def load_and_process_file(
    file_path: str | Path,
    frequency_bands: dict[str, list[tuple[float, float, str]]] | None = None,
    sensors: tuple[str, ...] | None = None,
    signal_clean_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load an HDF5 file, clean signals, and extract features.

    For every sweep × sensor pair:

    1. **Signal cleaning** — NaN/Inf interpolation, spike removal,
       clipping/saturation detection (see :func:`cleaning.clean_signal`).
       Sweeps that fail validation are skipped entirely.
    2. **Feature extraction** — broadband and (optionally) per-band
       features from the cleaned signal.

    Parameters
    ----------
    file_path:
        Path to the ``.hdf5`` measurement file.
    frequency_bands:
        Optional dict mapping sensor name to a list of
        ``(f_lo, f_hi, label)`` tuples.  When ``None`` only broadband
        features are extracted.
    sensors:
        Sensor channel names to process.  Defaults to
        ``("AE", "Ultrasound")``.
    signal_clean_cfg:
        Dict of keyword overrides for :func:`cleaning.clean_signal`.
        Set ``{"enabled": False}`` to skip signal cleaning entirely
        (replicates the old behaviour).

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(feature_rows, metadata_rows)`` — one dict per (sweep, sensor)
        pair that passed validation.
    """
    sensors = sensors or DEFAULT_SENSORS
    scfg = _resolve_signal_cfg(signal_clean_cfg)
    do_clean = scfg.pop("enabled", True)

    hdf5_data = load_hdf5_file(file_path, sensors=sensors)

    file_name = hdf5_data["file_name"]
    fs = hdf5_data["fs"]
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    n_rejected = 0

    for sweep_data in hdf5_data["sweeps"]:
        sweep_name = sweep_data["name"]
        test_parameters = sweep_data["test_parameters"]

        for sensor_name in sensors:
            voltage: np.ndarray = sweep_data[sensor_name]

            # ---- Signal-level cleaning (before features) ----
            if do_clean:
                voltage, sig_report = clean_signal(voltage, **scfg)
                if not sig_report.is_valid:
                    logger.info(
                        "Skipping %s / %s / %s: %s",
                        file_name, sweep_name, sensor_name,
                        sig_report.rejection_reason,
                    )
                    n_rejected += 1
                    continue
            else:
                sig_report = SignalQualityReport()

            # ---- Broadband features (no prefix) ----
            features = extract_features(voltage, fs)

            # ---- Band-specific features (prefixed) ----
            if frequency_bands and sensor_name in frequency_bands:
                for f_lo, f_hi, band_label in frequency_bands[sensor_name]:
                    filtered = bandpass_filter(voltage, fs, f_lo, f_hi)
                    band_features = extract_features(filtered, fs)
                    for key, val in band_features.items():
                        features[f"{band_label}__{key}"] = val

            # ---- Signal quality flags as metadata columns ----
            quality_cols = {
                "sig_clipped": sig_report.is_clipped,
                "sig_clipping_frac": sig_report.clipping_fraction,
                "sig_saturated": sig_report.is_saturated,
                "sig_nan_inf_replaced": sig_report.n_nan_inf_replaced,
                "sig_outliers_replaced": sig_report.n_outlier_samples_replaced,
            }

            row: dict[str, Any] = {
                "file": file_name,
                "sweep": sweep_name,
                "sensor": sensor_name,
                **features,
            }

            metadata_row: dict[str, Any] = {
                "file": file_name,
                "sweep": sweep_name,
                **test_parameters,
                **hdf5_data["lubricant_metadata"],
                **hdf5_data["bearing_metadata"],
                **quality_cols,
            }

            rows.append(row)
            metadata.append(metadata_row)

    if n_rejected:
        logger.warning(
            "%s: %d sweep/sensor pair(s) rejected by signal cleaning",
            file_name, n_rejected,
        )

    return rows, metadata


def load_and_process_files_parallel(
    file_paths: list[str | Path],
    frequency_bands: dict[str, list[tuple[float, float, str]]] | None = None,
    sensors: tuple[str, ...] | None = None,
    signal_clean_cfg: dict[str, Any] | None = None,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process multiple HDF5 files in parallel.

    Convenience wrapper around :func:`load_and_process_file` that uses
    ``joblib.Parallel`` and assembles the results into DataFrames.

    Parameters
    ----------
    file_paths:
        List of paths to ``.hdf5`` files.
    frequency_bands:
        Optional per-sensor frequency band definitions.
    sensors:
        Sensor channel names.  Defaults to ``("AE", "Ultrasound")``.
    signal_clean_cfg:
        Signal-cleaning overrides (passed to each worker).
    n_jobs:
        Number of parallel workers (default ``-1`` = all CPUs).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(features_df, metadata_df)``
    """
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(load_and_process_file)(
            fp, frequency_bands, sensors, signal_clean_cfg
        )
        for fp in file_paths
    )

    all_features = [row for feat_rows, _ in results for row in feat_rows]
    all_metadata = [row for _, meta_rows in results for row in meta_rows]

    return pd.DataFrame(all_features), pd.DataFrame(all_metadata)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------


def load_parquet_pair(
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features.parquet and metadata.parquet from a directory.

    Parameters
    ----------
    output_dir:
        Directory containing the parquet files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(features_df, metadata_df)``

    Raises
    ------
    FileNotFoundError
        If either parquet file is missing.
    """
    output_dir = Path(output_dir)
    feat_path = output_dir / "features.parquet"
    meta_path = output_dir / "metadata.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    return (
        pd.read_parquet(feat_path),
        pd.read_parquet(meta_path),
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_hdf5_files(
    input_dir: str | Path,
    file_patterns: list[str] | None = None,
) -> list[Path]:
    """Return a sorted list of HDF5 file paths in a directory.

    Parameters
    ----------
    input_dir:
        Directory to search for ``.hdf5`` files.
    file_patterns:
        Optional list of glob patterns matched against file *stems*
        (filename without extension), e.g. ``["scope_*", "firstOil_T75_*"]``.
        When ``None`` or empty all ``.hdf5`` files are returned.

    Returns
    -------
    list[Path]
        Sorted list of file paths.

    Raises
    ------
    FileNotFoundError
        If no HDF5 files are found (after filtering).
    """
    import fnmatch

    input_dir = Path(input_dir)
    all_files = sorted(input_dir.glob("*.hdf5"))

    if file_patterns:
        all_files = [
            f for f in all_files
            if any(fnmatch.fnmatch(f.stem, pat) for pat in file_patterns)
        ]

    if not all_files:
        raise FileNotFoundError(
            f"No .hdf5 files found in {input_dir}"
            + (f" matching patterns {file_patterns}" if file_patterns else "")
        )
    return all_files
