"""
eda.py
======
EDA orchestrator — configure, load data, generate and save all figures.

Usage
-----
    python dev/exploration/eda.py
    python dev/exploration/eda.py --config alt.yaml
    python dev/exploration/eda.py --max-files 2   # quick test on fewer files
"""

import argparse
import matplotlib.pyplot as plt

from ceramicspeed import eda as _eda
from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--max-files", type=int, default=None)
    args, _ = p.parse_known_args()
    return args


# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

args = parse_args()
cfg = load_config(args.config)

INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

SENSORS     = ("AE", "UL")
WAVEFORM_MS = 20.0
ENV_SHOW_MS = 5.0
BP_LOW_HZ   = 10e3
BP_HIGH_HZ  = 200e3
BP_LABEL    = f"{int(BP_LOW_HZ / 1e3)}-{int(BP_HIGH_HZ / 1e3)}kHz"

KAPPA_BOUNDS = cfg.get("kappa", {}).get("boundaries", [0.5, 1.0])
RPM_MIN      = cfg["filters"].get("rpm_min", 0.0)
RPM_MAX      = cfg["filters"]["rpm_max"]

kappa_ivs = _eda.make_kappa_intervals(KAPPA_BOUNDS)
iv_colors = [f"C{i}" for i in range(len(kappa_ivs))]

# %%
# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
if args.max_files:
    files = files[: args.max_files]
print(f"Found {len(files)} HDF5 file(s)")

_load_kw = dict(
    sensors=SENSORS, waveform_ms=WAVEFORM_MS, env_show_ms=ENV_SHOW_MS,
    bp_low_hz=BP_LOW_HZ, bp_high_hz=BP_HIGH_HZ,
)

cached = _eda.try_load_parquet(OUTPUT_DIR, EDA_DIR)
if cached is not None:
    stats_df, hp_df = _eda.build_stats_from_parquet(
        *cached, d_pw_mm=cfg["bearing"]["d_pw_mm"],
        rpm_min=RPM_MIN, rpm_max=RPM_MAX, bp_band_label=BP_LABEL,
    )
    sweeps = _eda.load_sweeps(files, cfg, **_load_kw, skip_stats=True)
else:
    sweeps = _eda.load_sweeps(files, cfg, **_load_kw)
    stats_df, hp_df = _eda.build_stats_from_sweeps(
        sweeps, sensors=SENSORS, bp_band_label=BP_LABEL
    )
    _eda.save_parquet_cache(sweeps, eda_dir=EDA_DIR, sensors=SENSORS, bp_band_label=BP_LABEL)

psd_rows = _eda.collect_psd_rows(sweeps, SENSORS)
print(f"Sweeps: {len(sweeps)}  |  Feature rows: {len(stats_df)}")


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(EDA_DIR / name, dpi=600)
    plt.show()
    plt.close(fig)
    print(f"Saved: {name}")


# %%
# -----------------------------------------------------------------------------
# Overview — operating conditions
# -----------------------------------------------------------------------------

_save(_eda.plot_operating_conditions(stats_df, KAPPA_BOUNDS),
      "eda_operating_conditions.png")

# %%
# -----------------------------------------------------------------------------
# Time domain
# -----------------------------------------------------------------------------

_save(_eda.plot_waveforms(sweeps, SENSORS, waveform_ms=WAVEFORM_MS),
      "eda_waveforms.png")

_save(_eda.plot_time_domain_stats(stats_df, SENSORS),
      "eda_time_domain_stats.png")

if hp_df is not None:
    _save(_eda.plot_time_domain_stats(hp_df, SENSORS, bp=True, bp_label=BP_LABEL),
          f"eda_time_domain_stats_{BP_LABEL}.png")

_save(_eda.plot_envelope(sweeps, SENSORS, kappa_ivs, iv_colors),
      "eda_envelope.png")

# %%
# -----------------------------------------------------------------------------
# Frequency domain
# -----------------------------------------------------------------------------

_save(_eda.plot_psd_kappa_regimes(psd_rows, SENSORS, kappa_ivs, iv_colors),
      "eda_psd_kappa_regimes.png")

_save(_eda.plot_envelope_psd(sweeps, SENSORS, kappa_ivs, iv_colors),
      "eda_envelope_psd.png")

_save(_eda.plot_envelope_spectral_flatness(sweeps, SENSORS),
      "eda_envelope_spectral_flatness.png")

# %%
# -----------------------------------------------------------------------------

print(f"\nAll outputs saved to: {EDA_DIR}")

if __name__ == "__main__":
    print("\nEDA complete.")
