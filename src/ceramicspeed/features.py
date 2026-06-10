"""
features.py
===========
Signal feature extraction for CeramicSpeed bearing analysis.

Computes time-domain and frequency-domain features from 1-D sensor signals
(acoustic emission and ultrasound).

The feature set is the canonical, de-duplicated set (14 features): every
feature is mathematically independent of the others.  Exact transforms and
products of retained features (std, variance, peak, impulse_factor,
rms_frequency, frequency_weighted_std, normalized_frequency_std,
frequency_skewness/kurtosis, spectral_mean/std, peak_frequency,
normalized_bandwidth) were removed — see paper/feature_audit.md for the
derivations.

Functions
---------
extract_features(signal_data, fs)
    Compute a comprehensive set of statistical and spectral features.

bandpass_filter(signal_data, fs, f_lo, f_hi, order=5)
    Apply a zero-phase Butterworth bandpass filter.
"""

from __future__ import annotations

import numpy as np
import antropy as ant
from scipy.signal import butter, sosfiltfilt

__all__ = ["extract_features", "bandpass_filter", "FEATURE_NAMES"]


#: Canonical feature names, in output order.
FEATURE_NAMES: list[str] = [
    # Time-domain (8)
    "rms",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "margin_factor",
    "mobility",
    "complexity",
    # Frequency-domain (6)
    "dominant_frequency",
    "center_frequency",
    "spectral_bandwidth",
    "spectral_skewness",
    "spectral_kurtosis",
    "spectral_flatness",
]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def bandpass_filter(
    signal_data: np.ndarray,
    fs: float,
    f_lo: float,
    f_hi: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to a 1-D signal.

    Parameters
    ----------
    signal_data:
        Raw voltage signal as a 1-D NumPy array.
    fs:
        Sampling frequency in Hz.
    f_lo:
        Lower cutoff frequency in Hz.
    f_hi:
        Upper cutoff frequency in Hz.  Must be < fs / 2.
    order:
        Butterworth filter order (default 5).

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal (same length as input).
    """
    nyq = fs / 2.0
    low = f_lo / nyq
    high = min(f_hi / nyq, 0.9999)  # guard against Nyquist

    if low >= 1.0:
        # Band entirely above Nyquist — no signal content; return zeros.
        return np.zeros_like(signal_data)
    if low <= 0:
        sos = butter(order, high, btype="low", output="sos")
    elif low >= high:
        # f_lo clipped to Nyquist makes low >= high — treat as highpass.
        sos = butter(order, low, btype="high", output="sos")
    else:
        sos = butter(order, [low, high], btype="band", output="sos")

    return sosfiltfilt(sos, signal_data)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(signal_data: np.ndarray, fs: float) -> dict[str, float]:
    """Extract time-domain and frequency-domain features from a 1-D signal.

    Parameters
    ----------
    signal_data:
        Raw voltage signal as a 1-D NumPy array.  If a non-array object is
        passed it will be converted via ``numpy.asarray``.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    dict[str, float]
        Dictionary of computed feature values (see :data:`FEATURE_NAMES`).

        *Time-domain* (8)
            ``rms``
                Root mean square amplitude.
            ``skewness``, ``kurtosis``
                Third and fourth standardised central moments.
            ``crest_factor``
                Half peak-to-peak amplitude divided by RMS.
            ``shape_factor``
                RMS divided by mean absolute value.
            ``margin_factor``
                Half peak-to-peak amplitude divided by squared mean
                square-root amplitude.
            ``mobility``, ``complexity``
                Hjorth parameters (power-weighted spectral spread proxies).

        *Frequency-domain* (6) — computed from the one-sided FFT magnitude
        spectrum :math:`S_k = |X_k|` with frequencies :math:`f_k`.
            ``dominant_frequency``
                Frequency of maximum spectral power.
            ``center_frequency``
                Magnitude-weighted mean frequency (spectral centroid),
                :math:`f_c = \\sum_k f_k S_k / \\sum_k S_k`.
            ``spectral_bandwidth``
                Magnitude-weighted spectral spread,
                :math:`\\sigma_w = (\\sum_k (f_k - f_c)^2 S_k / \\sum_k S_k)^{1/2}`.
            ``spectral_skewness``, ``spectral_kurtosis``
                Standardised (dimensionless) third and fourth
                magnitude-weighted spectral moments,
                :math:`\\sum_k (f_k - f_c)^p S_k / (\\sigma_w^p \\sum_k S_k)`.
            ``spectral_flatness``
                Geometric-to-arithmetic mean ratio of the magnitude spectrum.

    Notes
    -----
    Removed (recoverable) features relative to the legacy 26-feature set:
    ``std`` and ``variance`` equal ``rms`` (and its square) for zero-mean
    band-filtered signals; ``peak`` = ``crest_factor`` × ``rms``;
    ``impulse_factor`` = ``crest_factor`` × ``shape_factor``;
    ``rms_frequency``² = ``center_frequency``² + ``spectral_bandwidth``²;
    the legacy unweighted ``frequency_weighted_std`` and
    ``normalized_frequency_std`` were deterministic functions of
    ``center_frequency``.
    """
    x: np.ndarray = np.asarray(signal_data, dtype=float)

    # ------------------------------------------------------------------
    # Shared quantities
    # ------------------------------------------------------------------
    N: int = len(x)
    mean: float = np.mean(x)
    deviation: np.ndarray = x - mean
    abs_mean: float = float(np.mean(np.abs(x)))

    # One-sided FFT magnitude spectrum
    fft_coeffs: np.ndarray = np.fft.fft(x)[: N // 2]
    fft_mag: np.ndarray = np.abs(fft_coeffs)
    freq: np.ndarray = np.fft.fftfreq(N, d=1.0 / fs)[: len(fft_mag)]

    _fft_sum = float(np.sum(fft_mag))
    _fft_empty = _fft_sum < 1e-30  # band is silent (e.g. above Nyquist)

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------
    peak: float = float((np.max(x) - np.min(x)) / 2.0)  # internal only
    rms: float = float(np.sqrt(np.mean(x**2)))
    std: float = float(np.sqrt(np.mean(deviation**2)))  # internal only

    if std < 1e-30:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(np.mean((deviation / std) ** 3))
        kurtosis = float(np.mean((deviation / std) ** 4))

    crest_factor: float = peak / rms if rms > 0 else 0.0
    shape_factor: float = rms / abs_mean if abs_mean > 0 else 0.0
    sqrt_mean: float = float(np.mean(np.sqrt(np.abs(x))))
    margin_factor: float = peak / sqrt_mean**2 if sqrt_mean > 0 else 0.0

    if std < 1e-30:
        mobility: float = 0.0
        complexity: float = 0.0
    else:
        hjorth_mobility, hjorth_complexity = ant.hjorth_params(x, axis=0)
        mobility = float(hjorth_mobility)
        complexity = float(hjorth_complexity)

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------
    if _fft_empty:
        dominant_frequency = 0.0
        center_frequency = 0.0
        spectral_bandwidth = 0.0
        spectral_skewness = 0.0
        spectral_kurtosis = 0.0
        spectral_flatness = 0.0
    else:
        dominant_frequency = float(freq[int(np.argmax(fft_mag**2))])
        center_frequency = float(np.sum(freq * fft_mag) / _fft_sum)

        # Magnitude-weighted spectral spread (true bandwidth)
        spectral_bandwidth = float(
            np.sqrt(np.sum((freq - center_frequency) ** 2 * fft_mag) / _fft_sum)
        )

        # Standardised (dimensionless) weighted spectral shape moments
        if spectral_bandwidth < 1e-30:
            spectral_skewness = 0.0
            spectral_kurtosis = 0.0
        else:
            spectral_skewness = float(
                np.sum((freq - center_frequency) ** 3 * fft_mag)
                / (spectral_bandwidth**3 * _fft_sum)
            )
            spectral_kurtosis = float(
                np.sum((freq - center_frequency) ** 4 * fft_mag)
                / (spectral_bandwidth**4 * _fft_sum)
            )

        # Geometric / arithmetic mean; guard against log(0) from silent bins
        _pos = fft_mag[fft_mag > 0]
        _spec_mean = float(np.mean(fft_mag))
        spectral_flatness = (
            float(np.exp(np.mean(np.log(_pos))) / _spec_mean)
            if len(_pos) > 0 and _spec_mean > 0
            else 0.0
        )

    # ------------------------------------------------------------------
    # Assemble feature dictionary
    # ------------------------------------------------------------------
    return {
        # Time-domain
        "rms": rms,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
        "margin_factor": margin_factor,
        "mobility": mobility,
        "complexity": complexity,
        # Frequency-domain
        "dominant_frequency": dominant_frequency,
        "center_frequency": center_frequency,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_skewness": spectral_skewness,
        "spectral_kurtosis": spectral_kurtosis,
        "spectral_flatness": spectral_flatness,
    }
