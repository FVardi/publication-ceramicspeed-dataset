"""
kappa_vs_slipring.py
====================
Compare the kappa lubrication parameter with the slipring electrical
conductance measurement (SR channel).

kappa is derived from operating conditions (RPM, temperature, viscosity).
SR voltage measures electrical conductance across the bearing film:
    ~0 V  →  non-conducting oil film (good EHD film, expect high kappa)
    ~5 V  →  metallic contact (film breakdown, expect low kappa)

Loads SR stats and telemetry directly from HDF5 — no need to run the
pipeline first.

Usage
-----
    python dev/exploration/kappa_vs_slipring.py
    python dev/exploration/kappa_vs_slipring.py --config alt.yaml
"""

# %%
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

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

D_PW_MM    = cfg["bearing"]["d_pw_mm"]
RPM_MIN    = cfg["filters"].get("rpm_min", 50)

KAPPA_BOUNDS = cfg.get("kappa", {}).get("boundaries", [0.5, 1.0])
KAPPA_COLORS = cfg.get("kappa", {}).get("colors", ["#d62728", "#ff7f0e", "#2ca02c"])
KAPPA_LABELS = cfg.get("kappa", {}).get("labels", [f"κ < {KAPPA_BOUNDS[0]}",
                                                     f"{KAPPA_BOUNDS[0]}–{KAPPA_BOUNDS[1]}",
                                                     f"κ ≥ {KAPPA_BOUNDS[1]}"])

# Viscosity fallback — used when not present in HDF5 lubricant metadata
_NU_40_FALLBACK  = 22.0
_NU_100_FALLBACK = 4.1


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(EDA_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {name}")


# %%
# -----------------------------------------------------------------------------
# Load SR stats + telemetry directly from HDF5
# (reads only SR voltage and sweep attrs — skips AE/UL waveforms)
# -----------------------------------------------------------------------------

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

rows = []
for fpath in files:
    with h5py.File(fpath, "r") as f:
        lube = dict(f["metadata"]["lubricant"].attrs)
        nu_40  = float(lube.get("viscosity_40c_cst",  _NU_40_FALLBACK))
        nu_100 = float(lube.get("viscosity_100c_cst", _NU_100_FALLBACK))

        sweep_keys = sorted(f["sweeps"].keys(), key=lambda n: int(n.split("_")[1]))
        for seq_idx, sweep_name in enumerate(sweep_keys):
            sw = f["sweeps"][sweep_name]
            if "SP" not in sw:
                continue
            attrs = dict(sw.attrs)
            rpm  = attrs.get("telem_rpm_meas")
            temp = attrs.get("telem_omron_pv_c")
            if rpm is None or temp is None or float(rpm) < RPM_MIN:
                continue

            sr_v  = sw["SP"]["voltage"][()]
            kappa = calculate_kappa(
                rpm=float(rpm), temp_c=float(temp),
                d_pw=D_PW_MM, nu_40=nu_40, nu_100=nu_100,
            )
            rows.append({
                "file":          fpath.stem,
                "sweep":         sweep_name,
                "seq_idx":       seq_idx,      # numerical acquisition order
                "rpm":           float(rpm),
                "temperature_c": float(temp),
                "kappa":         kappa,
                "sr_mean_v":     float(np.mean(sr_v)),
                "sr_std_v":      float(np.std(sr_v)),
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} sweeps with SR data (RPM >= {RPM_MIN})")
print(f"  kappa:     {df['kappa'].min():.3f} - {df['kappa'].max():.3f}")
print(f"  sr_mean_v: {df['sr_mean_v'].min():.3f} - {df['sr_mean_v'].max():.3f} V")


# %%
# -----------------------------------------------------------------------------
# Plot 1: κ vs SR mean — coloured by RPM (left) and temperature (right)
# -----------------------------------------------------------------------------

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

sc = ax_left.scatter(df["kappa"], df["sr_mean_v"],
                     c=df["rpm"], cmap="plasma", s=8, alpha=0.4, edgecolors="none")
fig.colorbar(sc, ax=ax_left, label="RPM")
ax_left.set_xlabel("κ")
ax_left.set_ylabel("SR mean voltage [V]")
ax_left.set_title("κ vs SR mean — coloured by RPM")
ax_left.grid(ls=":", alpha=0.4)

sc = ax_right.scatter(df["kappa"], df["sr_mean_v"],
                      c=df["temperature_c"], cmap="coolwarm", s=8, alpha=0.4, edgecolors="none")
fig.colorbar(sc, ax=ax_right, label="Temperature [°C]")
ax_right.set_xlabel("κ")
ax_right.set_ylabel("SR mean voltage [V]")
ax_right.set_title("κ vs SR mean — coloured by temperature")
ax_right.grid(ls=":", alpha=0.4)

fig.suptitle("Kappa vs slipring conductance (SR mean)", fontsize=13)
fig.tight_layout()
_save(fig, "kappa_vs_slipring_mean.png")

# %%
# -----------------------------------------------------------------------------
# Plot 2: κ vs SR std — coloured by RPM (left) and temperature (right)
# (low std = stably good or stably bad film; high std = intermittent contact)
# -----------------------------------------------------------------------------

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

sc = ax_left.scatter(df["kappa"], df["sr_std_v"],
                     c=df["rpm"], cmap="plasma", s=8, alpha=0.4, edgecolors="none")
fig.colorbar(sc, ax=ax_left, label="RPM")
ax_left.set_xlabel("κ")
ax_left.set_ylabel("SR std [V]")
ax_left.set_title("κ vs SR std — coloured by RPM")
ax_left.grid(ls=":", alpha=0.4)

sc = ax_right.scatter(df["kappa"], df["sr_std_v"],
                      c=df["temperature_c"], cmap="coolwarm", s=8, alpha=0.4, edgecolors="none")
fig.colorbar(sc, ax=ax_right, label="Temperature [°C]")
ax_right.set_xlabel("κ")
ax_right.set_ylabel("SR std [V]")
ax_right.set_title("κ vs SR std — coloured by temperature")
ax_right.grid(ls=":", alpha=0.4)

fig.suptitle("Kappa vs slipring signal variability (SR std)", fontsize=13)
fig.tight_layout()
_save(fig, "kappa_vs_slipring_std.png")

# %%
# -----------------------------------------------------------------------------
# Plot 3: SR mean and κ over acquisition order — lubrication trend during test
# (only meaningful if the file covers a continuous test run)
# -----------------------------------------------------------------------------

if len(df["file"].unique()) == 1:
    fig, (ax_sp, ax_kappa) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    sc_sp = ax_sp.scatter(df["seq_idx"], df["sr_mean_v"],
                          c=df["rpm"], cmap="plasma", s=4, alpha=0.5, edgecolors="none")
    fig.colorbar(sc_sp, ax=ax_sp, label="RPM")
    ax_sp.set_ylabel("SR mean voltage [V]")
    ax_sp.set_title("Slipring conductance and κ over test run")
    ax_sp.grid(ls=":", alpha=0.4)

    sc_k = ax_kappa.scatter(df["seq_idx"], df["kappa"],
                            c=df["rpm"], cmap="plasma", s=4, alpha=0.5, edgecolors="none")
    fig.colorbar(sc_k, ax=ax_kappa, label="RPM")
    ax_kappa.set_ylabel("κ")
    ax_kappa.set_xlabel("Sweep index (acquisition order)")
    ax_kappa.grid(ls=":", alpha=0.4)

    fig.tight_layout()
    _save(fig, "kappa_vs_slipring_over_time.png")
else:
    print("Skipping time-series plot — multiple files loaded, seq_idx not comparable across files.")

# %%
# -----------------------------------------------------------------------------
# Plot 4: SR mean vs SR std, coloured by κ (continuous)
# Shows the operating space in signal-space: low mean + low std = stable good
# film; high mean + low std = stable poor film; high std = intermittent contact
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(df["sr_mean_v"], df["sr_std_v"],
                c=df["kappa"], cmap="viridis", s=8, alpha=0.5, edgecolors="none")
fig.colorbar(sc, ax=ax, label="κ")
ax.set_xlabel("SR mean voltage [V]")
ax.set_ylabel("SR std [V]")
ax.set_title("SR signal space — coloured by κ")
ax.grid(ls=":", alpha=0.4)
fig.tight_layout()
_save(fig, "slipring_mean_vs_std.png")

# %%
print("\nDone.")

