"""
features.py
===========
Signal feature extraction for CeramicSpeed bearing analysis.

Computes time-domain and frequency-domain features from 1-D sensor signals
(acoustic emission and ultrasound).

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

__all__ = ["extract_features", "bandpass_filter"]


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
        Dictionary of computed feature values.  Keys are grouped as:

        *Time-domain*
            ``peak``, ``rms``, ``std``, ``variance``, ``skewness``,
            ``crest_factor``, ``kurtosis``, ``shape_factor``,
            ``impulse_factor``, ``margin_factor``, ``mobility``,
            ``complexity``

        *Frequency-domain*
            ``dominant_frequency``, ``spectral_mean``, ``spectral_std``,
            ``spectral_skewness``, ``spectral_kurtosis``,
            ``center_frequency``, ``rms_frequency``, ``spectral_flatness``,
            ``frequency_weighted_std``, ``peak_frequency``,
            ``normalized_frequency_std``, ``frequency_skewness``,
            ``frequency_kurtosis``, ``normalized_bandwidth``
    """
    x: np.ndarray = np.asarray(signal_data, dtype=float)

    # ------------------------------------------------------------------
    # Shared quantities
    # ------------------------------------------------------------------
    N: int = len(x)
    mean: float = np.mean(x)
    deviation: np.ndarray = x - mean
    abs_sum: float = float(np.sum(np.abs(x)))

    # One-sided FFT magnitude spectrum
    fft_coeffs: np.ndarray = np.fft.fft(x)[: N // 2]
    fft_mag: np.ndarray = np.abs(fft_coeffs)
    K: int = len(fft_mag)
    freq: np.ndarray = np.fft.fftfreq(N, d=1.0 / fs)[:K].reshape(-1, 1)

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------
    peak: float = float((np.max(x) - np.min(x)) / 2.0)
    rms: float = float(np.sqrt(np.sum(x**2) / N))
    std: float = float(np.sqrt(np.sum(deviation**2) / N))
    variance: float = float(np.var(x))

    if std < 1e-30:
        skewness = 0.0
        kurtosis = 0.0
    else:
        skewness = float(np.sum((deviation / std) ** 3) / N)
        kurtosis = float(np.sum((deviation / std) ** 4) / N)

    crest_factor: float = peak / rms if rms > 0 else 0.0
    shape_factor: float = rms / (abs_sum / N) if abs_sum > 0 else 0.0
    impulse_factor: float = peak / (abs_sum / N) if abs_sum > 0 else 0.0
    sqrt_sum: float = float(np.sum(np.sqrt(np.abs(x)))) / N
    margin_factor: float = peak / sqrt_sum**2 if sqrt_sum > 0 else 0.0

    hjorth_mobility, hjorth_complexity = ant.hjorth_params(x, axis=0)
    mobility: float = float(hjorth_mobility)
    complexity: float = float(hjorth_complexity)

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------
    dominant_frequency: float = float(np.argmax((fft_mag / K) ** 2) * (fs / K))
    spectral_mean: float = float(np.mean(fft_mag))
    spectral_std: float = float(np.std(fft_mag))
    center_frequency: float = float(
        np.sum(freq * fft_mag) / np.sum(fft_mag)
    )

    if spectral_std < 1e-30:
        spectral_skewness = 0.0
        spectral_kurtosis = 0.0
    else:
        # spectral_skewness = float(
        #     np.sum(((fft_mag - spectral_mean) / spectral_std) ** 3) / K
        # )
        spectral_skewness = float(
            np.sum(freq - center_frequency ** 3 * fft_mag)
            / (spectral_std ** 3 * np.sum(fft_mag))
        )
        # spectral_kurtosis = float(
        #     np.sum(((fft_mag - spectral_mean) / spectral_std) ** 4) / K
        # )
        spectral_kurtosis = float(
            np.sum(freq - center_frequency ** 4 * fft_mag)
            / (spectral_std ** 4 * np.sum(fft_mag))
        )

    rms_frequency: float = float(
        np.sqrt(np.sum((freq**2) * fft_mag) / np.sum(fft_mag))
    )
    # spectral_flatness: float = float(
    #     np.sum(freq**2 * fft_mag)
    #     / np.sqrt(np.sum(fft_mag) * np.sum(freq**4 * fft_mag))
    # )
    spectral_flatness: float = float(
        np.exp(np.mean(np.log(fft_mag))) / np.mean(fft_mag)
    )
    frequency_weighted_std: float = float(
        np.sqrt(np.sum((freq - center_frequency) ** 2) / K)
    )
    peak_frequency: float = float(
        np.sqrt(np.sum(freq**4 * fft_mag) / np.sum(freq**2 * fft_mag))
    )
    normalized_frequency_std: float = frequency_weighted_std / center_frequency
    frequency_skewness: float = float(
        np.sum((freq - center_frequency) ** 3 * fft_mag)
        / (K * frequency_weighted_std**3)
    )
    frequency_kurtosis: float = float(
        np.sum((freq - center_frequency) ** 4 * fft_mag)
        / (K * frequency_weighted_std**4)
    )
    normalized_bandwidth: float = float(
        np.sum(np.abs(freq - center_frequency) ** 0.5 * fft_mag)
        / (K * frequency_weighted_std**0.5)
    )

    # ------------------------------------------------------------------
    # Assemble feature dictionary
    # ------------------------------------------------------------------
    return {
        # Time-domain
        "peak": peak,
        "rms": rms,
        "std": std,
        "variance": variance,
        "skewness": skewness,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "margin_factor": margin_factor,
        "mobility": mobility,
        "complexity": complexity,
        # Frequency-domain
        "dominant_frequency": dominant_frequency,
        "spectral_mean": spectral_mean,
        "spectral_std": spectral_std,
        "spectral_skewness": spectral_skewness,
        "spectral_kurtosis": spectral_kurtosis,
        "center_frequency": center_frequency,
        "rms_frequency": rms_frequency,
        "spectral_flatness": spectral_flatness,
        "frequency_weighted_std": frequency_weighted_std,
        "peak_frequency": peak_frequency,
        "normalized_frequency_std": normalized_frequency_std,
        "frequency_skewness": frequency_skewness,
        "frequency_kurtosis": frequency_kurtosis,
        "normalized_bandwidth": normalized_bandwidth,
    }
