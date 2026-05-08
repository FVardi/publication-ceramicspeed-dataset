"""
eda_spectra.py
================
Quick inspection of a small number of sweeps — waveforms and FFT spectra.
"""

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files
from ceramicspeed import eda as _eda

# %%
# -----------------------------------------------------------------------------
# Configuration — edit here
# -----------------------------------------------------------------------------

CONFIG_PATH = None
N_SWEEPS    = 1      # number of sweeps to load
SENSORS     = ("AE", "UL")
WAVEFORM_MS = 20.0
ENV_SHOW_MS = 20.0

AE_BANDS_KHZ = [(20, 500), (500, 1000), (1000, 2000)]
UL_FMAX_KHZ  = 20.0

# -----------------------------------------------------------------------------

cfg        = load_config(CONFIG_PATH)
INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)

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
    sensors=SENSORS, waveform_ms=WAVEFORM_MS, env_show_ms=ENV_SHOW_MS,
    skip_stats=True,
    max_sweeps=None if sweep_names else N_SWEEPS,
    sweep_names=sweep_names,
)
print(f"Loaded {len(sweeps)} sweeps")
sweeps.sort(key=lambda r: r.get("kappa", float("inf")))

n_cols = 2
n_rows = math.ceil(len(sweeps) / n_cols)


def _plot_grid(sweeps, sensor, fmin_khz, fmax_khz, y_data_fn, ylabel, color, title, fname):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), squeeze=False)
    for i, rec in enumerate(sweeps):
        ax = axes[i // n_cols][i % n_cols]
        v = rec["waveform"].get(sensor)
        if v is None:
            ax.set_visible(False)
            continue
        fs = rec["fs"]
        N  = len(v)
        mag   = y_data_fn(v, fs)
        f_khz = np.fft.rfftfreq(N, d=1.0 / fs) / 1e3
        if len(mag) > len(f_khz):
            mag = mag[: len(f_khz)]
        mask = (f_khz >= fmin_khz) & (f_khz <= fmax_khz)
        ax.plot(f_khz[mask], mag[mask], lw=0.8, color=color)
        kappa = rec.get("kappa", float("nan"))
        ax.set_title(f"{rec['sweep']}  RPM={rec['rpm']:.0f}  κ={kappa:.2f}", fontsize=8)
        ax.set_xlabel("Frequency [kHz]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(fmin_khz, fmax_khz)
        ax.grid(ls=":", alpha=0.3)
    for i in range(len(sweeps), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def _fft_mag(v, fs):
    return np.abs(np.fft.rfft(v)) / len(v)


def _env_mag(v, fs):
    mag = np.abs(np.fft.rfft(np.abs(hilbert(v)))) / len(v)
    mag[0] = 0.0   # zero DC instead of dropping (keeps index alignment)
    return mag


def _plot_waveform_grid(sweeps, sensor, color, title, fname):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), squeeze=False)
    for i, rec in enumerate(sweeps):
        ax = axes[i // n_cols][i % n_cols]
        v = rec["waveform"].get(sensor)
        if v is None:
            ax.set_visible(False)
            continue
        fs  = rec["fs"]
        t_ms = np.arange(len(v)) / fs * 1e3
        ax.plot(t_ms, v, lw=0.5, color=color)
        kappa = rec.get("kappa", float("nan"))
        ax.set_title(f"{rec['sweep']}  RPM={rec['rpm']:.0f}  κ={kappa:.2f}", fontsize=8)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Amplitude [V]")
        ax.grid(ls=":", alpha=0.3)
    for i in range(len(sweeps), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# %%
# -----------------------------------------------------------------------------
# Waveform — AE
# -----------------------------------------------------------------------------

_plot_waveform_grid(sweeps, "AE", "steelblue", "Waveform — AE", "eda_waveform_ae.png")

# %%
# -----------------------------------------------------------------------------
# Waveform — UL
# -----------------------------------------------------------------------------

_plot_waveform_grid(sweeps, "UL", "steelblue", "Waveform — UL", "eda_waveform_ul.png")

# %%
# -----------------------------------------------------------------------------
# Spectrum — AE (full range)
# -----------------------------------------------------------------------------

_plot_grid(sweeps, "AE", 0, max(fmax for _, fmax in AE_BANDS_KHZ), _fft_mag, "|FFT| [V]", "steelblue",
           "Spectrum — AE  full range",
           "eda_spectrum_ae_full.png")

# %%
# -----------------------------------------------------------------------------
# Spectrum — AE (per band)
# -----------------------------------------------------------------------------

for fmin, fmax in AE_BANDS_KHZ:
    _plot_grid(sweeps, "AE", fmin, fmax, _fft_mag, "|FFT| [V]", "steelblue",
               f"Spectrum — AE  {fmin}–{fmax} kHz",
               f"eda_spectrum_ae_{fmin}-{fmax}khz.png")

# %%
# -----------------------------------------------------------------------------
# Spectrum — UL
# -----------------------------------------------------------------------------

_plot_grid(sweeps, "UL", 0, UL_FMAX_KHZ, _fft_mag, "|FFT| [V]", "steelblue",
           "Spectrum — UL  0–20 kHz",
           "eda_spectrum_ul.png")

# %%
# -----------------------------------------------------------------------------
# Envelope spectrum — AE (per band)
# -----------------------------------------------------------------------------

for fmin, fmax in AE_BANDS_KHZ:
    _plot_grid(sweeps, "AE", fmin, fmax, _env_mag, "|FFT env| [V]", "darkorange",
               f"Envelope spectrum — AE  {fmin}–{fmax} kHz",
               f"eda_envelope_spectrum_ae_{fmin}-{fmax}khz.png")

# %%
# -----------------------------------------------------------------------------
# Envelope spectrum — UL
# -----------------------------------------------------------------------------

_plot_grid(sweeps, "UL", 0, UL_FMAX_KHZ, _env_mag, "|FFT env| [V]", "darkorange",
           "Envelope spectrum — UL  0–20 kHz",
           "eda_envelope_spectrum_ul.png")

# %%