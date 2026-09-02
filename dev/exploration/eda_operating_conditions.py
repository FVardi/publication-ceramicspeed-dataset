"""
eda_operating_conditions.py
===========================
RPM / temperature operating-conditions scatter (coloured by κ), κ histogram,
and time-series of RPM, temperature, and κ over sweep index.

Reads from outputs/sweep_conditions.csv (produced by operating_condition_mapping.py).
Run that script first if the CSV does not exist yet.

Both figures are also copied to paper/figures/ (same convention as
scripts/new/signal_processing/06_feature_kappa_figure.py) so the paper
compiles standalone.

Usage
-----
    python dev/exploration/eda_operating_conditions.py
    python dev/exploration/eda_operating_conditions.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import pathlib
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ceramicspeed import eda as _eda
from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files

# %%
# =============================================================================
# Configuration
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

INPUT_DIR    = get_input_dir(cfg)
OUTPUT_DIR   = get_output_dir(cfg)
EDA_DIR      = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = pathlib.Path(__file__).resolve().parents[2] / "paper" / "figures"

RPM_MAX: float = 3000.0  # filter out end-of-test high-RPM transients

KAPPA_BOUNDS  = cfg.get("kappa", {}).get("boundaries", [0.5, 1.0])
KAPPA_COLORS  = cfg.get("kappa", {}).get("colors", ["#d62728", "#ff7f0e", "#2ca02c"])
KAPPA_LABELS  = cfg.get("kappa", {}).get("labels", [f"κ < {KAPPA_BOUNDS[0]}", f"{KAPPA_BOUNDS[0]}–{KAPPA_BOUNDS[1]}", f"κ ≥ {KAPPA_BOUNDS[1]}"])

# %%
# =============================================================================
# Load data from sweep_conditions.csv
# =============================================================================

csv_path = OUTPUT_DIR / "sweep_conditions.csv"
if not csv_path.exists():
    raise FileNotFoundError(
        f"{csv_path} not found. Run operating_condition_mapping.py first."
    )

df = pd.read_csv(csv_path).dropna(subset=["rpm", "temperature_c", "kappa"])
n_before = len(df)
df = df[df["rpm"] <= RPM_MAX].reset_index(drop=True)
print(f"Loaded {n_before} sweeps from {csv_path.name}; {n_before - len(df)} removed (rpm > {RPM_MAX:.0f}); {len(df)} retained")

# %%
# =============================================================================
# Acquisition timing statistics
# =============================================================================

# Window collection interval: derived from timestamps on the first few sweeps
_hdf5_files = discover_hdf5_files(INPUT_DIR, file_patterns=cfg.get("filters", {}).get("file_patterns"))

_window_interval_s: float = float("nan")
_waveform_ms: float = float("nan")
_sample_rate_hz: float = float("nan")
if _hdf5_files:
    with h5py.File(_hdf5_files[0], "r") as _hf:
        _sweeps_grp = _hf["sweeps"]
        _names = sorted(_sweeps_grp.keys(), key=lambda s: int(s.split("_")[1]))[:101]
        _elapsed = [float(_sweeps_grp[n].attrs["telem_elapsed_sec"]) for n in _names]
        _intervals = np.diff(_elapsed)
        _window_interval_s = float(np.mean(_intervals))
        _window_interval_std_s = float(np.std(_intervals))
        # Waveform duration and sample rate from the time axis of the first sweep
        # (read directly, not hardcoded -- acquisition settings differ by file:
        # 12.5 MHz/20ms in the earlier captures, 2.5 MHz/200ms from Aug 2026 on).
        _ae = _sweeps_grp[_names[0]]["AE"]
        _t = _ae["time"][()]
        _waveform_ms = (_t[-1] - _t[0]) * 1e3
        _sample_rate_hz = 1.0 / float(np.mean(np.diff(_t)))

# Operating condition dwell: run-length encode RPM setpoints
_ts_sorted = df.copy()
_ts_sorted["sweep_idx"] = _ts_sorted["sweep"].str.split("_").str[1].astype(int)
_ts_sorted = _ts_sorted.sort_values("sweep_idx").reset_index(drop=True)
_ts_sorted["rpm_set"] = (_ts_sorted["rpm"] / 100).round() * 100

_runs, _i = [], 0
while _i < len(_ts_sorted):
    _rpm = _ts_sorted.loc[_i, "rpm_set"]
    _j = _i
    while _j < len(_ts_sorted) and _ts_sorted.loc[_j, "rpm_set"] == _rpm:
        _j += 1
    _runs.append(_j - _i)
    _i = _j

_runs_arr = np.array(_runs)
_peak_mask = _runs_arr >= 100   # 3000 RPM holds
_dwell_windows_median = float(np.median(_runs_arr[~_peak_mask]))
_dwell_windows_mean   = float(np.mean(_runs_arr[~_peak_mask]))
_peak_windows_mean    = float(np.mean(_runs_arr[_peak_mask])) if _peak_mask.any() else float("nan")

print("\n--- Acquisition timing ---")
print(f"  Waveform duration       : {_waveform_ms:.1f} ms  "
      f"({int(round(_waveform_ms / 1e3 * _sample_rate_hz)):,} samples @ {_sample_rate_hz / 1e6:.2f} MHz)")
print(f"  Window collection interval: {_window_interval_s:.2f} s ± {_window_interval_std_s:.3f} s  (~{1/_window_interval_s:.3f} Hz)")
print(f"  RPM setpoint dwell (ramp): median {_dwell_windows_median:.0f} windows  "
      f"({_dwell_windows_median * _window_interval_s:.0f} s), "
      f"mean {_dwell_windows_mean:.1f} windows ({_dwell_windows_mean * _window_interval_s:.0f} s)")
print(f"  Peak-RPM dwell (3000 RPM): mean {_peak_windows_mean:.0f} windows  "
      f"({_peak_windows_mean * _window_interval_s / 60:.1f} min)")

# %%
# =============================================================================
# Save statistics to JSON
# =============================================================================

_kappa_regime_counts = {
    f"kappa_lt_{KAPPA_BOUNDS[0]}":  int((df["kappa"] < KAPPA_BOUNDS[0]).sum()),
    f"kappa_{KAPPA_BOUNDS[0]}_to_{KAPPA_BOUNDS[1]}": int(
        ((df["kappa"] >= KAPPA_BOUNDS[0]) & (df["kappa"] < KAPPA_BOUNDS[1])).sum()
    ),
    f"kappa_gte_{KAPPA_BOUNDS[1]}": int((df["kappa"] >= KAPPA_BOUNDS[1]).sum()),
}

stats = {
    "dataset": {
        "n_sweeps_raw":      n_before,
        "n_sweeps_removed":  n_before - len(df),
        "n_sweeps_retained": len(df),
        "rpm_max_filter":    RPM_MAX,
    },
    "acquisition": {
        "waveform_duration_ms":      round(_waveform_ms, 2),
        "sample_rate_mhz":           round(_sample_rate_hz / 1e6, 4),
        "n_samples_per_window":      int(round(_waveform_ms / 1e3 * _sample_rate_hz)),
        "window_interval_s":         round(_window_interval_s, 3),
        "window_interval_std_s":     round(_window_interval_std_s, 4),
        "window_rate_hz":            round(1.0 / _window_interval_s, 4),
        "ramp_dwell_windows_median": int(_dwell_windows_median),
        "ramp_dwell_windows_mean":   round(_dwell_windows_mean, 1),
        "ramp_dwell_s_median":       round(_dwell_windows_median * _window_interval_s, 1),
        "ramp_dwell_s_mean":         round(_dwell_windows_mean * _window_interval_s, 1),
        "peak_dwell_windows_mean":   round(_peak_windows_mean, 1),
        "peak_dwell_min_mean":       round(_peak_windows_mean * _window_interval_s / 60.0, 2),
    },
    "operating_conditions": {
        "rpm_min":         float(df["rpm"].min()),
        "rpm_max":         float(df["rpm"].max()),
        "temperature_min": float(df["temperature_c"].min()),
        "temperature_max": float(df["temperature_c"].max()),
        "kappa_min":       round(float(df["kappa"].min()), 4),
        "kappa_max":       round(float(df["kappa"].max()), 4),
        "kappa_mean":      round(float(df["kappa"].mean()), 4),
        "kappa_std":       round(float(df["kappa"].std()), 4),
        "kappa_regimes":   _kappa_regime_counts,
    },
}

_stats_path = EDA_DIR / "operating_conditions_stats.json"
with open(_stats_path, "w", encoding="utf-8") as _fh:
    json.dump(stats, _fh, indent=2)
print(f"Saved: {_stats_path.name}")

# %%
# =============================================================================
# Scatter: RPM vs temperature coloured by κ, + κ histogram
# =============================================================================

fig = _eda.plot_operating_conditions(df, KAPPA_BOUNDS)
fig.savefig(EDA_DIR / "eda_operating_conditions.png", dpi=150)
plt.show()
plt.close(fig)
print("Saved: eda_operating_conditions.png")

# Copy into the paper so it compiles standalone (same convention as
# scripts/new/signal_processing/06_feature_kappa_figure.py).
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(EDA_DIR / "eda_operating_conditions.png",
             PAPER_FIG_DIR / "eda_operating_conditions.png")
print(f"Copied -> {PAPER_FIG_DIR / 'eda_operating_conditions.png'}")

# %%
# =============================================================================
# Time-series: RPM, temperature, κ over sweep index
# =============================================================================

ts = df.copy()
ts["sweep_idx"] = ts["sweep"].str.split("_").str[1].astype(int)
ts = ts.sort_values("sweep_idx").reset_index(drop=True)

def _kappa_color(k):
    if k < KAPPA_BOUNDS[0]:
        return KAPPA_COLORS[0]
    elif k < KAPPA_BOUNDS[1]:
        return KAPPA_COLORS[1]
    return KAPPA_COLORS[2]

point_colors = ts["kappa"].map(_kappa_color)

fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

axes[0].scatter(ts["sweep_idx"], ts["rpm"], c=point_colors, s=4, alpha=0.6, linewidths=0)
axes[0].set_ylabel("RPM")
_eda._style_axes(axes[0])

axes[1].scatter(ts["sweep_idx"], ts["temperature_c"], c=point_colors, s=4, alpha=0.6, linewidths=0)
axes[1].set_ylabel("Temperature [°C]")
_eda._style_axes(axes[1])

axes[2].scatter(ts["sweep_idx"], ts["kappa"], c=point_colors, s=4, alpha=0.6, linewidths=0)
for bound in KAPPA_BOUNDS:
    axes[2].axhline(bound, color=_eda._AXIS_COLOR, ls="--", lw=0.8, alpha=0.8)
axes[2].set_ylabel("κ")
axes[2].set_xlabel("Sweep index")
_eda._style_axes(axes[2])
axes[2].legend(
    handles=[Patch(color=c, label=l) for c, l in zip(KAPPA_COLORS, KAPPA_LABELS)],
    fontsize=8, loc="upper right", frameon=False,
)

fig.suptitle("Operating conditions over time", fontsize=12)
fig.tight_layout()
fig.savefig(EDA_DIR / "eda_conditions_timeseries.png", dpi=150)
plt.show()
plt.close(fig)
print("Saved: eda_conditions_timeseries.png")

# Copy into the paper so it compiles standalone (same convention as
# scripts/new/signal_processing/06_feature_kappa_figure.py).
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(EDA_DIR / "eda_conditions_timeseries.png",
             PAPER_FIG_DIR / "eda_conditions_timeseries.png")
print(f"Copied -> {PAPER_FIG_DIR / 'eda_conditions_timeseries.png'}")

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\neda_operating_conditions complete.")

# %%
