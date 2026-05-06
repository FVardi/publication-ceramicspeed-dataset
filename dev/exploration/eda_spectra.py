"""
eda_few_files.py
================
Quick inspection of a small number of sweeps — waveforms and FFT spectra.
"""

# %%
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import hilbert

from ceramicspeed.config import get_input_dir, load_config
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

SENSOR_FMAX_KHZ = {"UL": 20.0}    # x-axis upper cap per sensor (kHz); None = full range
SENSOR_FMIN_KHZ = {"AE": 0.0}     # x-axis lower cap per sensor (kHz); None = start from DC

# Spectral flatness subbands [kHz]
FLATNESS_BANDS = {
    "AE": [(20, 500), (500, 1000), (1000, 2000)],  # None = Nyquist
}

# -----------------------------------------------------------------------------

cfg       = load_config(CONFIG_PATH)
INPUT_DIR = get_input_dir(cfg)

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

# %%
# -----------------------------------------------------------------------------
# Load N_SWEEPS sweeps
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


def spectral_flatness(mag: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of magnitude bins."""
    m = mag[mag > 0]
    if len(m) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(m))) / np.mean(m))


def flatness_annotation(sensor: str, f_khz: np.ndarray, mag: np.ndarray, fs: float) -> str:
    bands = FLATNESS_BANDS.get(sensor)
    if not bands:
        return ""
    nyq = fs / 2e3  # kHz
    parts = []
    for lo, hi in bands:
        hi_eff = hi if hi is not None else nyq
        band = (f_khz >= lo) & (f_khz < hi_eff)
        label = f"{lo}-{int(hi_eff)}kHz" if hi is not None else f"{lo}kHz-Nyq"
        sf = spectral_flatness(mag[band])
        parts.append(f"SF {label}={sf:.3f}")
    return "<br>" + "<br>".join(parts)


# %%
# -----------------------------------------------------------------------------
# Raw FFT spectra
# -----------------------------------------------------------------------------

for sensor in SENSORS:
    fmax = SENSOR_FMAX_KHZ.get(sensor)
    fmin = SENSOR_FMIN_KHZ.get(sensor)
    fig = make_subplots(
        rows=len(sweeps), cols=1,
        subplot_titles=[
            f"{sensor}<br>RPM={r['rpm']:.0f}  κ={r.get('kappa', float('nan')):.2f}"
            for r in sweeps
        ],
        shared_xaxes=False,
        shared_yaxes=True,
    )
    for row_i, rec in enumerate(sweeps):
        v = rec["waveform"].get(sensor)
        if v is None:
            continue
        fs  = rec["fs"]
        N   = len(v)
        mag = np.abs(np.fft.rfft(v)) / N
        f   = np.fft.rfftfreq(N, d=1.0 / fs) / 1e3   # kHz
        mask = f > fmin if fmin else slice(None)
        mag, f = mag[mask], f[mask]
        fig.add_trace(
            go.Scatter(x=f, y=mag, mode="lines", line=dict(width=0.8),
                       name=rec["sweep"], showlegend=False),
            row=row_i + 1, col=1,
        )
        fig.update_xaxes(title_text="Frequency [kHz]",
                         range=[0, fmax] if fmax else None,
                         row=row_i + 1, col=1)
        fig.update_yaxes(title_text="|FFT|", row=row_i + 1, col=1)
        fig.layout.annotations[row_i].text += flatness_annotation(sensor, f, mag, fs)
    fig.update_layout(title_text=f"Raw FFT spectra — {sensor}", height=400 * len(sweeps))
    fig.show()


# %%
# -----------------------------------------------------------------------------
# Envelope FFT spectra
# -----------------------------------------------------------------------------

for sensor in SENSORS:
    fmax = SENSOR_FMAX_KHZ.get(sensor)
    fmin = SENSOR_FMIN_KHZ.get(sensor)
    fig = make_subplots(
        rows=len(sweeps), cols=1,
        subplot_titles=[
            f"{sensor}<br>RPM={r['rpm']:.0f}  κ={r.get('kappa', float('nan')):.2f}"
            for r in sweeps
        ],
        shared_xaxes=False,
        shared_yaxes=True,
    )
    for row_i, rec in enumerate(sweeps):
        v = rec["waveform"].get(sensor)
        if v is None:
            continue
        fs      = rec["fs"]
        N       = len(v)
        env     = np.abs(hilbert(v))
        mag     = np.abs(np.fft.rfft(env)) / N
        f       = np.fft.rfftfreq(N, d=1.0 / fs) / 1e3   # kHz
        mask = f > fmin if fmin else slice(None)
        mag, f = mag[mask], f[mask]
        fig.add_trace(
            go.Scatter(x=f, y=mag, mode="lines", line=dict(width=0.8),
                       name=rec["sweep"], showlegend=False),
            row=row_i + 1, col=1,
        )
        fig.update_xaxes(title_text="Frequency [kHz]",
                         range=[0, fmax] if fmax else None,
                         row=row_i + 1, col=1)
        fig.update_yaxes(title_text="|FFT|", type="log", row=row_i + 1, col=1)
        fig.layout.annotations[row_i].text += flatness_annotation(sensor, f, mag, fs)
    fig.update_layout(title_text=f"Envelope spectra — {sensor}", height=400 * len(sweeps))
    fig.show()
