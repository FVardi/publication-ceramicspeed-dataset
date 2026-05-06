"""
infogram.py
===========
Infogram (Antoni 2016) — spectral negentropy of the squared envelope spectrum
mapped over a grid of bandpass centre-frequency × bandwidth.

High negentropy indicates that the demodulated band carries structured,
impulsive content consistent with a bearing fault signature.

The right-hand panel shows the gradient magnitude of the negentropy map,
highlighting sharp boundaries between low- and high-negentropy regions.
"""

# %%
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter

from ceramicspeed.config import get_input_dir, load_config
from ceramicspeed.loading import discover_hdf5_files
from ceramicspeed.features import bandpass_filter
from ceramicspeed import eda as _eda

# %%
# -----------------------------------------------------------------------------
# Configuration — edit here
# -----------------------------------------------------------------------------

CONFIG_PATH = None
N_SWEEPS    = 1
SENSORS     = ("AE",)   # UL is too low-frequency for this analysis
WAVEFORM_MS = 20.0
ENV_SHOW_MS = 20.0

# Sigma for Gaussian smoothing before gradient (pixels); 0 = no smoothing
GRADIENT_SMOOTH_SIGMA = 1.5

# Infogram grid per sensor  [Hz]
INFOGRAM_CFG = {
    "AE": dict(f_lo=20_000, f_hi=1_900_000, n_centers=40, n_bws=30),
}

# -----------------------------------------------------------------------------

cfg       = load_config(CONFIG_PATH)
INPUT_DIR = get_input_dir(cfg)

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

# %%
# -----------------------------------------------------------------------------
# Load sweeps (same selection as eda_spectra.py)
# -----------------------------------------------------------------------------

sweep_names = cfg.get("sweep_selection") or None

sweeps = _eda.load_sweeps(
    files, cfg,
    sensors=list(SENSORS), waveform_ms=WAVEFORM_MS, env_show_ms=ENV_SHOW_MS,
    skip_stats=True,
    max_sweeps=None if sweep_names else N_SWEEPS,
    sweep_names=sweep_names,
)
print(f"Loaded {len(sweeps)} sweeps")
sweeps.sort(key=lambda r: r.get("kappa", float("inf")))


# %%
# -----------------------------------------------------------------------------
# Infogram computation
# -----------------------------------------------------------------------------

def _negentropy_ses(signal: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Negentropy of the squared envelope spectrum for the given band."""
    filtered = bandpass_filter(signal, fs, f_lo, f_hi)
    env2 = np.abs(hilbert(filtered)) ** 2
    ses = np.abs(np.fft.rfft(env2 - env2.mean()))
    total = ses.sum()
    if total == 0:
        return float("nan")
    p = ses / total
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p)))
    var_ses = float(np.var(ses))
    H_gauss = 0.5 * (1.0 + np.log(2.0 * np.pi * var_ses)) if var_ses > 0 else 0.0
    return H_gauss - H


def compute_infogram(
    signal: np.ndarray,
    fs: float,
    f_lo: float,
    f_hi: float,
    n_centers: int = 40,
    n_bws: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centres_Hz, bandwidths_Hz, negentropy_matrix[bw, centre])."""
    centres    = np.linspace(f_lo, f_hi, n_centers)
    bw_min     = 2.0 * fs / len(signal)
    bw_max     = (f_hi - f_lo) / 2.0
    bandwidths = np.linspace(bw_min, bw_max, n_bws)

    neg = np.full((n_bws, n_centers), np.nan)
    for i, bw in enumerate(bandwidths):
        half = bw / 2.0
        for j, fc in enumerate(centres):
            lo = fc - half
            hi = fc + half
            if lo < 1.0 or hi >= fs / 2.0:
                continue
            neg[i, j] = _negentropy_ses(signal, fs, lo, hi)

    return centres, bandwidths, neg


def gradient_magnitude(Z: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """2-D gradient magnitude of Z, optionally pre-smoothed."""
    src = gaussian_filter(Z, sigma=sigma) if sigma > 0 else Z.copy()
    # Replace NaN with local mean so the gradient doesn't blow up at edges
    nan_mask = np.isnan(src)
    src[nan_mask] = np.nanmean(src)
    dy, dx = np.gradient(src)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    grad[nan_mask] = np.nan
    return grad


# %%
# -----------------------------------------------------------------------------
# Plot infograms + gradient
# -----------------------------------------------------------------------------

for sensor in SENSORS:
    icfg = INFOGRAM_CFG.get(sensor, {})
    fig = make_subplots(
        rows=len(sweeps), cols=2,
        column_titles=["Negentropy", "Gradient magnitude"],
        subplot_titles=[
            f"{r['sweep']}  RPM={r['rpm']:.0f}  κ={r.get('kappa', float('nan')):.2f}"
            for r in sweeps
            for _ in range(2)
        ],
        shared_xaxes=False,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for row_i, rec in enumerate(sweeps):
        v = rec["waveform"].get(sensor)
        if v is None:
            continue
        fs = rec["fs"]

        centres, bandwidths, neg = compute_infogram(v, fs, **icfg)
        grad = gradient_magnitude(neg, sigma=GRADIENT_SMOOTH_SIGMA)

        x_khz = centres    / 1e3
        y_khz = bandwidths / 1e3

        cb_y = 1 - (row_i + 0.5) / len(sweeps)
        cb_len = 0.9 / len(sweeps)

        fig.add_trace(
            go.Heatmap(
                x=x_khz, y=y_khz, z=neg,
                colorscale="Viridis",
                colorbar=dict(title="Negentropy", len=cb_len, y=cb_y, x=0.45),
                showscale=True,
            ),
            row=row_i + 1, col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=x_khz, y=y_khz, z=grad,
                colorscale="Hot",
                colorbar=dict(title="|∇|", len=cb_len, y=cb_y, x=1.0),
                showscale=True,
            ),
            row=row_i + 1, col=2,
        )

        for col in (1, 2):
            fig.update_xaxes(title_text="Centre frequency [kHz]", row=row_i + 1, col=col)
            fig.update_yaxes(title_text="Bandwidth [kHz]",        row=row_i + 1, col=col)

    fig.update_layout(
        title_text=f"Infogram — {sensor}",
        height=420 * len(sweeps),
    )
    fig.show()

# %%
