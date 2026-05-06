"""
operating_condition_mapping.py
================================
Read all sweeps from the HDF5 files and write a CSV table of operating
conditions so that specific sweeps can be hand-picked for inspection.

Output: outputs/sweep_conditions.csv
Columns: file, sweep, rpm, temperature_c, kappa, viscosity_40c_cst, viscosity_100c_cst
"""

# %%
import pandas as pd

from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files
from ceramicspeed import eda as _eda

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CONFIG_PATH = None

cfg        = load_config(CONFIG_PATH)
INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)

# %%
# -----------------------------------------------------------------------------
# Load sweep metadata (no waveforms, no stats)
# -----------------------------------------------------------------------------

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

sweeps = _eda.load_sweeps(
    files, cfg,
    sensors=("AE",),     # one sensor is enough to get metadata
    waveform_ms=0.1,     # minimal waveform to keep memory low
    env_show_ms=0.1,
    skip_stats=True,
)
print(f"Loaded {len(sweeps)} sweeps")

# %%
# -----------------------------------------------------------------------------
# Build and save table
# -----------------------------------------------------------------------------

df = pd.DataFrame([
    {
        "file":               r["file"],
        "sweep":              r["sweep"],
        "rpm":                round(r["rpm"], 1),
        "temperature_c":      round(r["temperature_c"], 1),
        "kappa":              round(r["kappa"], 4) if r["kappa"] == r["kappa"] else None,
        "viscosity_40c_cst":  r["viscosity_40c_cst"],
        "viscosity_100c_cst": r["viscosity_100c_cst"],
    }
    for r in sweeps
])

out_path = OUTPUT_DIR / "sweep_conditions.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows → {out_path}")

df

