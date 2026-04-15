"""
01_feature_generation.py
========================
Extract time-domain and frequency-domain features from raw HDF5 measurement
files and save them as Parquet files for downstream analysis.

Pipeline position: 1st script — reads raw HDF5, writes features.parquet and
metadata.parquet.

Usage
-----
    python 01_feature_generation.py                   # uses config.yaml
    python 01_feature_generation.py --config alt.yaml # uses alternate config
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from own_utils.loading import discover_hdf5_files, load_and_process_files_parallel
from own_utils.config import (
    load_config,
    get_input_dir,
    get_output_dir,
    get_sensor_prefilter,
    get_frequency_bands_config,
)

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: config.yaml in project root)",
    )
    # When running as interactive cells (# %%), sys.argv may contain junk.
    # parse_known_args avoids crashing in that case.
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

INPUT_DIR = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
SIGNAL_CLEAN_CFG: dict = cfg.get("signal_cleaning", {})

# Pre-filter: restrict each sensor to its effective bandwidth before
# computing broadband features (prevents out-of-band noise from biasing
# spectral features — especially important for the heterodyned UL probe).
SENSOR_PREFILTER = get_sensor_prefilter(cfg)

# Physics-motivated bandpass bands: produces additional prefixed feature
# columns (e.g. "AE_50-200kHz__mobility") alongside the broadband set.
FREQUENCY_BANDS = get_frequency_bands_config(cfg)

# %%
# =============================================================================
# Discover HDF5 files
# =============================================================================

FILE_PATTERNS: list[str] | None = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s) in {INPUT_DIR}")

if SIGNAL_CLEAN_CFG.get("enabled", False):
    print("Signal cleaning: ENABLED")
    for k, v in SIGNAL_CLEAN_CFG.items():
        if k != "enabled":
            print(f"  {k}: {v}")
else:
    print("Signal cleaning: DISABLED")

if SENSOR_PREFILTER:
    print("Sensor pre-filter: ENABLED")
    for sensor, (f_lo, f_hi) in SENSOR_PREFILTER.items():
        print(f"  {sensor}: {f_lo/1e3:.0f}–{f_hi/1e3:.0f} kHz")
else:
    print("Sensor pre-filter: DISABLED (broadband features from raw signal)")

if FREQUENCY_BANDS:
    print("Frequency bands: ENABLED")
    for sensor, bands in FREQUENCY_BANDS.items():
        labels = [label for _, _, label in bands]
        print(f"  {sensor}: {labels}")
else:
    print("Frequency bands: DISABLED (broadband features only)")

# %%
# =============================================================================
# Clean signals + extract features in parallel
# =============================================================================

df, metadata_df = load_and_process_files_parallel(
    files,
    signal_clean_cfg=SIGNAL_CLEAN_CFG,
    sensor_prefilter=SENSOR_PREFILTER,
    frequency_bands=FREQUENCY_BANDS,
)

# %%
# =============================================================================
# Save to Parquet
# =============================================================================

df.to_parquet(OUTPUT_DIR / "features.parquet", engine="pyarrow")
print(f"Saved: {OUTPUT_DIR / 'features.parquet'}  ({len(df)} rows)")

metadata_df.to_parquet(OUTPUT_DIR / "metadata.parquet", engine="pyarrow")
print(f"Saved: {OUTPUT_DIR / 'metadata.parquet'}  ({len(metadata_df)} rows)")


# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n01_feature_generation complete.")

