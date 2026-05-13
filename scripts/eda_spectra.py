"""
eda_spectra.py
==============
Visualise selected individual sweep spectra for qualitative inspection.

Plots PSD (Welch) for each sweep in ``sweep_selection`` (config.yaml).
Layout: rows = sensors, columns = selected sweeps.  Y-axis is shared
within each sensor row so absolute PSD levels are comparable across
sweeps; sensors use independent y-scales.

Usage
-----
    python scripts/eda_spectra.py
    python scripts/eda_spectra.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.eda import load_sweeps

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

OUTPUT_DIR = get_output_dir(cfg)
INPUT_DIR  = get_input_dir(cfg)
EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)

SENSORS: tuple[str, ...] = ("AE", "UL")
SWEEP_SELECTION: list[str] = cfg.get("sweep_selection", [])
SENSOR_LIMITS: dict = cfg.get("sensors", {})

if not SWEEP_SELECTION:
    raise ValueError("No sweep_selection defined in config.yaml")

# %%
# =============================================================================
# Load selected sweeps
# =============================================================================

hdf5_files = sorted(pathlib.Path(INPUT_DIR).glob("*.hdf5"))
if not hdf5_files:
    raise FileNotFoundError(f"No .hdf5 files found in {INPUT_DIR}")

sweeps = load_sweeps(
    hdf5_files,
    cfg,
    sensors=SENSORS,
    waveform_ms=10.0,
    env_show_ms=10.0,
    skip_stats=True,
    sweep_names=SWEEP_SELECTION,
)

sweeps.sort(key=lambda r: r.get("kappa", float("nan")))
print(f"Loaded {len(sweeps)} sweep records")
for r in sweeps:
    print(f"  {r['sweep']:20s}  κ={r['kappa']:.3f}  RPM={r['rpm']:.0f}  T={r['temperature_c']:.0f}°C")

# %%
# =============================================================================
# PSD spectra — shared y-axis per sensor, independent across sensors
# =============================================================================

n_cols = len(sweeps)
n_rows = len(SENSORS)

kappa_vals = [r["kappa"] for r in sweeps]
norm  = mcolors.Normalize(vmin=min(kappa_vals), vmax=max(kappa_vals))
cmap  = plt.cm.viridis

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.5 * n_cols, 4 * n_rows),
    sharey="row",   # shared within each sensor row, independent between sensors
    squeeze=False,
)

for row_i, sensor in enumerate(SENSORS):
    f_max = SENSOR_LIMITS.get(sensor, {}).get("f_max", None)

    for col_i, rec in enumerate(sweeps):
        ax = axes[row_i, col_i]

        if sensor not in rec.get("psd", {}):
            ax.set_visible(False)
            continue

        f, p = rec["psd"][sensor]
        if f_max is not None:
            mask = f <= f_max
            f, p = f[mask], p[mask]

        color = cmap(norm(rec["kappa"]))
        ax.semilogy(f / 1e3, p, lw=1.0, color=color)

        ax.set_title(
            f"{rec['sweep']}\nκ={rec['kappa']:.3f}  RPM={rec['rpm']:.0f}  T={rec['temperature_c']:.0f}°C",
            fontsize=8,
        )
        ax.set_xlabel("Frequency [kHz]")
        ax.grid(ls=":", which="both", alpha=0.3)

        if col_i == 0:
            ax.set_ylabel(f"{sensor}\nPSD  [V² / Hz]")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=axes, label="κ", shrink=0.6)
fig.suptitle("Individual sweep spectra", fontsize=12)
fig.tight_layout()
plt.savefig(EDA_DIR / "eda_spectra.png", dpi=150)
plt.show()
print("Saved: eda_spectra.png")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\neda_spectra complete.")
