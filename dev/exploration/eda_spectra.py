"""
eda_spectra.py
==============
Visualise selected individual sweep spectra for qualitative inspection.

Plots the RAW one-sided FFT magnitude (no Welch averaging, no pre-filtering)
of the AE and US channels for each sweep in ``sweep_selection`` (config.yaml).
AE is shown over its full acquired bandwidth (to Nyquist); US up to 40 kHz so
that content above the heterodyned 0--20 kHz baseband is visible.

Layout: rows = sensors, columns = selected sweeps. Y-axis shared within each
sensor row; sensors use independent y-scales.

Usage
-----
    python dev/exploration/eda_spectra.py
    python dev/exploration/eda_spectra.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import pathlib
import sys

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except NameError:
    ROOT = pathlib.Path.cwd()
    while not (ROOT / "config.yaml").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))

from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import get_input_dir, get_output_dir, load_config

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

INPUT_DIR  = pathlib.Path(get_input_dir(cfg))
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)

SENSORS: tuple[str, ...] = ("AE", "UL")
US_F_MAX = 40_000.0          # show US up to 40 kHz (above the 0-20 kHz baseband)

SWEEP_SELECTION: list[str] = cfg.get("sweep_selection", [])
if not SWEEP_SELECTION:
    raise ValueError("No sweep_selection defined in config.yaml")

pats = cfg.get("filters", {}).get("file_patterns") or ["*"]
hdf5_files = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix in (".hdf5", ".h5") and any(s in p.stem for s in pats))
if not hdf5_files:
    raise FileNotFoundError(f"No matching HDF5 files in {INPUT_DIR}")

print(f"Found {len(hdf5_files)} HDF5 file(s); sweep_selection: {SWEEP_SELECTION}")

# %%
# =============================================================================
# Load raw voltages + metadata for the selected sweeps
# =============================================================================

_VIS_FALLBACK = {"viscosity_40c_cst": 22.0, "viscosity_100c_cst": 4.1}

records = []
for fp in hdf5_files:
    with h5py.File(fp, "r") as f:
        grp = f["sweeps"]
        present = [s for s in SWEEP_SELECTION if s in grp]
        if not present:
            continue
        lm = dict(f["metadata"]["lubricant"].attrs)
        for k, v in _VIS_FALLBACK.items():
            lm.setdefault(k, v)
        first = grp[present[0]][SENSORS[0]]["time"][()]
        fs = 1.0 / float(first[1] - first[0])
        for name in present:
            attrs = dict(grp[name].attrs)
            rpm  = float(attrs.get("rpm", attrs.get("telem_rpm_meas", np.nan)))
            temp = float(attrs.get("temperature_c", attrs.get("telem_omron_pv_c", np.nan)))
            rec = {
                "sweep": name, "rpm": rpm, "temperature_c": temp, "fs": fs,
                "kappa": calculate_kappa(
                    rpm=max(rpm, 1.0), temp_c=temp,
                    d_pw=cfg["bearing"]["d_pw_mm"],
                    nu_40=float(lm["viscosity_40c_cst"]),
                    nu_100=float(lm["viscosity_100c_cst"]),
                ),
            }
            for sensor in SENSORS:
                if sensor in grp[name]:
                    rec[sensor] = grp[name][sensor]["voltage"][()].astype(float)
            records.append(rec)

records.sort(key=lambda r: r.get("kappa", float("nan")))
print(f"Loaded {len(records)} sweep records (fs = {records[0]['fs']/1e6:.3f} MHz)")
for r in records:
    print(f"  {r['sweep']:14s}  κ={r['kappa']:.3f}  RPM={r['rpm']:.0f}  T={r['temperature_c']:.0f}°C")

# %%
# =============================================================================
# Raw FFT spectra
# =============================================================================

n_cols     = len(records)
kappa_vals = [r["kappa"] for r in records]
norm       = mcolors.Normalize(vmin=min(kappa_vals), vmax=max(kappa_vals))
cmap       = plt.cm.viridis

fig, axes = plt.subplots(len(SENSORS), n_cols,
                         figsize=(4.5 * n_cols, 4 * len(SENSORS)),
                         sharey="row", squeeze=False)

for row_i, sensor in enumerate(SENSORS):
    for col_i, rec in enumerate(records):
        ax = axes[row_i, col_i]
        if sensor not in rec:
            ax.set_visible(False)
            continue
        x    = rec[sensor]
        fs   = rec["fs"]
        mag  = np.abs(np.fft.rfft(x)) / len(x)
        freq = np.fft.rfftfreq(len(x), d=1.0 / fs)
        color = cmap(norm(rec["kappa"]))
        if sensor == "AE":
            ax.loglog(freq[1:] / 1e6, mag[1:], lw=0.4, color=color)
            ax.set_xlim(1e-3, fs / 2 / 1e6)   # 0.001–6.25 MHz
            ax.set_xlabel("Frequency [MHz]")
        else:
            m = freq <= US_F_MAX
            ax.semilogy(freq[m] / 1e3, mag[m], lw=0.4, color=color)
            ax.set_xlim(0, US_F_MAX / 1e3)
            ax.set_xlabel("Frequency [kHz]")
        ax.set_title(f"{rec['sweep']}\nκ={rec['kappa']:.3f}  RPM={rec['rpm']:.0f}  "
                     f"T={rec['temperature_c']:.0f}°C", fontsize=8)
        ax.grid(ls=":", which="both", alpha=0.3)
        if col_i == 0:
            ax.set_ylabel(f"{'US' if sensor == 'UL' else sensor}\nraw |FFT|  [V]")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=axes, label="κ", shrink=0.6)
fig.suptitle("Individual sweep raw FFT spectra (unfiltered)", fontsize=12)
fig.tight_layout()
plt.savefig(EDA_DIR / "eda_spectra.png", dpi=150)
plt.show()
print(f"Saved: {EDA_DIR / 'eda_spectra.png'}")

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\neda_spectra complete.")

# %%
