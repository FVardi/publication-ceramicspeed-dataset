"""
eda_wavelet.py
==============
Laplace-wavelet scalogram.

The complex analytic Laplace wavelet is:
    ψ(t) = exp(-(α + i·2π·f₀)·t)  for t ≥ 0

Parameters
----------
f₀  — centre frequency [Hz]   → x-axis
α   — decay rate [rad/s]       → y-axis  (related to Q = π·f₀/α)

For each (f₀, α) pair the signal is cross-correlated with the wavelet and
the chosen statistic (RMS or kurtosis of the envelope) is plotted as a
heatmap.  Time is not shown — the statistic collapses the time axis.
"""

# %%
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import kurtosis as scipy_kurtosis

from ceramicspeed.config import get_input_dir, load_config
from ceramicspeed.loading import discover_hdf5_files
from ceramicspeed import eda as _eda

# %%
# -----------------------------------------------------------------------------
# Configuration — edit here
# -----------------------------------------------------------------------------

CONFIG_PATH = None
N_SWEEPS    = 1
SENSORS     = ("AE",)   # UL too low-frequency for this range
WAVEFORM_MS = 20.0
ENV_SHOW_MS = 20.0

# Colour statistic: "kurtosis" (impulsiveness) or "rms" (energy)
STATISTIC = "kurtosis"

# Wavelet parameter grid per sensor
WAVELET_CFG = {
    "AE": dict(
        f0_lo  =  20_000,    # Hz  — lowest centre frequency
        f0_hi  = 1_900_000,  # Hz  — highest centre frequency
        n_f0   = 40,         # grid points along f₀
        alpha_lo =  2_000,   # rad/s  (high Q, narrow band)
        alpha_hi = 500_000,  # rad/s  (low Q, broad band)
        n_alpha  = 30,       # grid points along α
    ),
}

# -----------------------------------------------------------------------------

cfg       = load_config(CONFIG_PATH)
INPUT_DIR = get_input_dir(cfg)

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

# %%
# -----------------------------------------------------------------------------
# Load sweeps
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
# Wavelet helpers
# -----------------------------------------------------------------------------

def _make_wavelet(f0: float, alpha: float, fs: float, n_tau: int = 5) -> np.ndarray:
    """Complex analytic Laplace wavelet, truncated at n_tau time constants."""
    tau = n_tau / alpha          # seconds to 99.3% decay
    t   = np.arange(0, tau, 1.0 / fs)
    psi = np.exp(-(alpha + 1j * 2.0 * np.pi * f0) * t)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))   # unit energy
    return psi


def _scalogram_value(signal: np.ndarray, fs: float, f0: float, alpha: float) -> float:
    """Cross-correlate signal with wavelet, return chosen statistic of |output|."""
    psi = _make_wavelet(f0, alpha, fs)
    # Frequency-domain cross-correlation: conj(Ψ) × S
    N   = len(signal) + len(psi) - 1
    S   = np.fft.fft(signal, n=N)
    P   = np.fft.fft(psi,    n=N)
    env = np.abs(np.fft.ifft(np.conj(P) * S)[: len(signal)])
    if STATISTIC == "kurtosis":
        return float(scipy_kurtosis(env, fisher=False))   # 4th moment / σ⁴
    return float(np.sqrt(np.mean(env ** 2)))              # RMS


def compute_scalogram(
    signal: np.ndarray,
    fs: float,
    f0_lo: float, f0_hi: float, n_f0: int,
    alpha_lo: float, alpha_hi: float, n_alpha: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f0_grid_Hz, alpha_grid, Z[alpha, f0])."""
    f0s    = np.linspace(f0_lo,    f0_hi,    n_f0)
    alphas = np.linspace(alpha_lo, alpha_hi, n_alpha)
    Z      = np.full((n_alpha, n_f0), np.nan)
    for i, alpha in enumerate(alphas):
        for j, f0 in enumerate(f0s):
            # Skip if wavelet would be shorter than 4 samples
            if alpha / fs > len(signal) / 4:
                continue
            Z[i, j] = _scalogram_value(signal, fs, f0, alpha)
    return f0s, alphas, Z


# %%
# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

for sensor in SENSORS:
    wcfg = WAVELET_CFG.get(sensor)
    if wcfg is None:
        continue

    fig = make_subplots(
        rows=len(sweeps), cols=1,
        subplot_titles=[
            f"{sensor}  {r['sweep']}  RPM={r['rpm']:.0f}  κ={r.get('kappa', float('nan')):.2f}"
            for r in sweeps
        ],
        shared_xaxes=False,
        shared_yaxes=False,
    )

    for row_i, rec in enumerate(sweeps):
        v = rec["waveform"].get(sensor)
        if v is None:
            continue
        fs = rec["fs"]

        f0s, alphas, Z = compute_scalogram(v, fs, **wcfg)

        # Q-factor for secondary axis label: Q = π·f₀_mean / α
        q_lo = np.pi * f0s.mean() / wcfg["alpha_hi"]
        q_hi = np.pi * f0s.mean() / wcfg["alpha_lo"]

        fig.add_trace(
            go.Heatmap(
                x=f0s / 1e3,     # kHz
                y=alphas,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(
                    title=STATISTIC.capitalize(),
                    len=1 / len(sweeps),
                    y=1 - (row_i + 0.5) / len(sweeps),
                ),
                showscale=True,
            ),
            row=row_i + 1, col=1,
        )
        fig.update_xaxes(title_text="Centre frequency f₀ [kHz]", row=row_i + 1, col=1)
        fig.update_yaxes(title_text="Decay rate α [rad/s]",       row=row_i + 1, col=1)

    fig.update_layout(
        title_text=f"Laplace-wavelet scalogram ({STATISTIC}) — {sensor}",
        height=450 * len(sweeps),
    )
    fig.show()
