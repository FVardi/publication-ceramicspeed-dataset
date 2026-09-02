"""
eda_waveforms.py
================
Time-domain waveforms for the sweeps listed in ``sweep_selection`` (config.yaml).
Layout: rows = sensors (AE, US, SP), columns = sweeps sorted by ascending κ.

Usage
-----
    python dev/exploration/eda_waveforms.py
    python dev/exploration/eda_waveforms.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import pathlib
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import _normalize_sweep_params

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

INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

WAVEFORM_MS: float = 10.0

_SENSOR_LABEL = {"UL": "US"}
_SENSOR_COLOR = {"AE": "#1f77b4", "UL": "#ff7f0e"}
_SENSOR_ORDER = ("AE", "UL")

SWEEP_SELECTION: list[str] = cfg.get("sweep_selection", [])
if not SWEEP_SELECTION:
    raise ValueError("No sweep_selection defined in config.yaml")

_VISCOSITY_FALLBACK = {"viscosity_40c_cst": 22.0, "viscosity_100c_cst": 4.1}

# %%
# =============================================================================
# Discover HDF5 files
# =============================================================================

_input = pathlib.Path(INPUT_DIR)
_all_files = sorted(list(_input.glob("*.hdf5")) + list(_input.glob("*.h5")))

_patterns: list[str] = cfg.get("filters", {}).get("file_patterns", [])
hdf5_files = (
    [fp for fp in _all_files if any(p in fp.stem for p in _patterns)]
    if _patterns else _all_files
)
if not hdf5_files:
    raise FileNotFoundError(
        f"No HDF5 files matched file_patterns={_patterns} in {INPUT_DIR}"
    )

print(f"Found {len(hdf5_files)} file(s) (file_patterns={_patterns or 'all'}):")
for fp in hdf5_files:
    print(f"  {fp.name}")

# %%
# =============================================================================
# Load selected sweeps
# =============================================================================

records: list[dict] = []

for fp in hdf5_files:
    with h5py.File(fp, "r") as hf:
        sweeps_grp = hf["sweeps"]
        lm = dict(hf["metadata"]["lubricant"].attrs)
        for k, v in _VISCOSITY_FALLBACK.items():
            lm.setdefault(k, v)

        for sweep_name in SWEEP_SELECTION:
            if sweep_name not in sweeps_grp:
                continue
            sweep = sweeps_grp[sweep_name]
            tp = _normalize_sweep_params(dict(sweep.attrs))
            rpm  = float(tp.get("rpm", np.nan))
            temp = float(tp.get("temperature_c", np.nan))
            try:
                kap = calculate_kappa(
                    rpm=rpm, temp_c=temp,
                    d_pw=float(cfg["bearing"]["d_pw_mm"]),
                    nu_40=float(lm["viscosity_40c_cst"]),
                    nu_100=float(lm["viscosity_100c_cst"]),
                )
            except Exception:
                kap = np.nan

            # Determine AE sample rate once for sensors that lack a time axis
            ae_fs: float = 250_000.0
            if "AE" in sweep and "time" in sweep["AE"]:
                t0, t1 = float(sweep["AE"]["time"][0]), float(sweep["AE"]["time"][1])
                ae_fs = 1.0 / (t1 - t0)

            rec: dict = {
                "sweep": sweep_name,
                "rpm": rpm,
                "temperature_c": temp,
                "kappa": kap,
                "waveform": {},
                "fs": {},
            }

            for sensor in _SENSOR_ORDER:
                if sensor not in sweep:
                    continue
                grp = sweep[sensor]
                if "voltage" not in grp:
                    continue
                if "time" in grp:
                    t0, t1 = float(grp["time"][0]), float(grp["time"][1])
                    fs = 1.0 / (t1 - t0)
                else:
                    fs = ae_fs
                n = int(WAVEFORM_MS * 1e-3 * fs)
                rec["waveform"][sensor] = grp["voltage"][:n].copy()
                rec["fs"][sensor] = fs

            records.append(rec)

records.sort(key=lambda r: r["kappa"] if not np.isnan(r["kappa"]) else float("inf"))
print(f"Loaded {len(records)} sweeps:")
for r in records:
    print(
        f"  {r['sweep']:20s}  κ={r['kappa']:.3f}"
        f"  RPM={r['rpm']:.0f}  T={r['temperature_c']:.0f}°C"
    )

# %%
# =============================================================================
# Plot
# =============================================================================

present_sensors = [s for s in _SENSOR_ORDER if any(s in r["waveform"] for r in records)]
n_rows = len(present_sensors)
n_cols = len(records)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.5 * n_cols, 3.5 * n_rows),
    sharey="row",
    squeeze=False,
)

for row_i, sensor in enumerate(present_sensors):
    display = _SENSOR_LABEL.get(sensor, sensor)
    color   = _SENSOR_COLOR.get(sensor, f"C{row_i}")

    for col_i, rec in enumerate(records):
        ax = axes[row_i, col_i]

        if sensor not in rec["waveform"]:
            ax.set_visible(False)
            continue

        v  = rec["waveform"][sensor]
        fs = rec["fs"][sensor]
        t_ms = np.arange(len(v)) / fs * 1e3

        ax.plot(t_ms, v, lw=0.5, color=color)
        ax.set_xlabel("Time [ms]")
        ax.grid(ls=":", alpha=0.3)

        if col_i == 0:
            ax.set_ylabel(f"{display}\nVoltage [V]")

        if row_i == 0:
            ax.set_title(
                f"{rec['sweep']}\n"
                f"κ = {rec['kappa']:.3f}   RPM = {rec['rpm']:.0f}"
                f"   T = {rec['temperature_c']:.0f}°C",
                fontsize=9,
            )

fig.suptitle(
    f"Time-domain waveforms — first {WAVEFORM_MS:.0f} ms  "
    f"({len(records)} sweeps, sorted by κ)",
    fontsize=12,
)
fig.tight_layout()

out_path = EDA_DIR / "eda_waveforms.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()

# Copy into the paper so it compiles standalone (same convention as
# scripts/new/signal_processing/06_feature_kappa_figure.py).
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(out_path, PAPER_FIG_DIR / "eda_waveforms.png")
print(f"Copied -> {PAPER_FIG_DIR / 'eda_waveforms.png'}")

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\neda_waveforms complete.")
