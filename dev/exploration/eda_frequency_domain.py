"""
eda_frequency_domain.py
=======================
Frequency-domain EDA: mean PSD per κ regime, Hilbert envelope PSD, and
envelope spectral flatness vs κ.

Usage
-----
    python dev/exploration/eda_frequency_domain.py
    python dev/exploration/eda_frequency_domain.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse

import matplotlib.pyplot as plt

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

INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

SENSORS     = ("AE", "UL")
WAVEFORM_MS = 20.0
ENV_SHOW_MS = 20.0

RPM_MIN  = cfg["filters"].get("rpm_min", 0.0)
RPM_MAX  = cfg["filters"]["rpm_max"]
KAPPA_BOUNDS = cfg.get("kappa", {}).get("boundaries", [0.5, 1.0])

kappa_ivs = _eda.make_kappa_intervals(KAPPA_BOUNDS)
iv_colors  = [f"C{i}" for i in range(len(kappa_ivs))]

# %%
# =============================================================================
# Load data
# =============================================================================

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

sweeps = _eda.load_sweeps(
    files, cfg,
    sensors=SENSORS, waveform_ms=WAVEFORM_MS, env_show_ms=ENV_SHOW_MS,
    skip_stats=True,
)
psd_rows = _eda.collect_psd_rows(sweeps, SENSORS)
print(f"Sweeps: {len(sweeps)}")


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(EDA_DIR / name, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved: {name}")


# %%
# =============================================================================
# Mean PSD per κ regime
# =============================================================================

_save(
    _eda.plot_psd_kappa_regimes(psd_rows, SENSORS, kappa_ivs, iv_colors),
    "eda_psd_kappa_regimes.png",
)

# %%
# =============================================================================
# Envelope PSD per κ regime
# =============================================================================

_save(
    _eda.plot_envelope_psd(sweeps, SENSORS, kappa_ivs, iv_colors),
    "eda_envelope_psd.png",
)

# %%
# =============================================================================
# Envelope spectral flatness vs κ
# =============================================================================

_save(
    _eda.plot_envelope_spectral_flatness(sweeps, SENSORS),
    "eda_envelope_spectral_flatness.png",
)

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print(f"\nAll outputs saved to: {EDA_DIR}")
    print("eda_frequency_domain complete.")
