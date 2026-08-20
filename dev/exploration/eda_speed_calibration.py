"""
eda_speed_calibration.py
=========================
Compare the three speed-related telemetry channels recorded per sweep:

    telem_rpm_target   -- nominal protocol setpoint
    telem_vfd_cmd_hz   -- VFD drive frequency actually commanded
    telem_rpm_meas     -- "measured" RPM (OGT5000 sensor per setup.tex,
                           suspected faulty)

Purpose: characterise how these three disagree, and check whether a fixed
linear conversion `rpm_reconstructed = telem_vfd_cmd_hz * FACTOR` tracks the
protocol target better than telem_rpm_meas does. Does NOT assume any
particular factor is correct -- it fits one from steady-state data and
reports goodness of fit so the analyst can judge.

Steady-state detection: a sweep is "steady" when telem_vfd_cmd_hz has
stopped changing between consecutive sweeps (VFD has settled) and the VFD
is actually running. This avoids contaminating the fit with ramp/transient
sweeps where cmd_hz and meas are both still moving and not comparable.

Usage
-----
    python dev/exploration/eda_speed_calibration.py
    python dev/exploration/eda_speed_calibration.py --config alt.yaml
    python dev/exploration/eda_speed_calibration.py --cmd-delta 0.01
"""

# %%
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cmd-delta", type=float, default=0.01,
                        help="Max |change in cmd_hz| between consecutive sweeps "
                             "to call a sweep steady-state (default 0.01 Hz)")
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
cfg = load_config(args.config)

INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

CMD_DELTA = args.cmd_delta


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(EDA_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {name}")


# %%
# -----------------------------------------------------------------------------
# Load telemetry: target, cmd_hz, meas, running flag, per file
# -----------------------------------------------------------------------------

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")

rows = []
for fpath in files:
    with h5py.File(fpath, "r") as f:
        if "sweeps" not in f:
            continue
        sweep_keys = sorted(f["sweeps"].keys(), key=lambda n: int(n.split("_")[1]))
        for seq_idx, sweep_name in enumerate(sweep_keys):
            attrs = f["sweeps"][sweep_name].attrs
            if "telem_rpm_meas" not in attrs or "telem_vfd_cmd_hz" not in attrs:
                continue
            rows.append({
                "file":      fpath.stem,
                "sweep":     sweep_name,
                "seq_idx":   seq_idx,
                "meas":      float(attrs["telem_rpm_meas"]),
                "target":    float(attrs.get("telem_rpm_target", np.nan)),
                "cmd_hz":    float(attrs["telem_vfd_cmd_hz"]),
                "running":   bool(attrs.get("telem_vfd_is_running", False)),
                "elapsed_s": float(attrs.get("telem_elapsed_sec", np.nan)),
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} sweeps total across {df['file'].nunique()} file(s)")

# %%
# -----------------------------------------------------------------------------
# Steady-state detection: cmd_hz stopped changing, VFD running, cmd_hz > 0
# -----------------------------------------------------------------------------

df = df.sort_values(["file", "seq_idx"]).reset_index(drop=True)
df["d_cmd_hz"] = df.groupby("file")["cmd_hz"].diff().abs().fillna(np.inf)
df["steady"] = df["running"] & (df["d_cmd_hz"] < CMD_DELTA) & (df["cmd_hz"] > 0.01)

steady = df[df["steady"]].copy()
print(f"Steady-state sweeps: {len(steady)} / {len(df)} "
      f"({100 * len(steady) / max(len(df), 1):.1f}%)")

# %%
# -----------------------------------------------------------------------------
# Aggregate by nominal target level
# -----------------------------------------------------------------------------

agg = (
    steady.groupby("target")
    .agg(n=("target", "size"),
         cmd_hz_mean=("cmd_hz", "mean"),
         meas_mean=("meas", "mean"),
         meas_std=("meas", "std"))
    .reset_index()
)
agg["meas_over_target"] = agg["meas_mean"] / agg["target"]
agg["meas_over_cmd_hz"] = agg["meas_mean"] / agg["cmd_hz_mean"]

print("\nSteady-state summary by protocol target RPM:")
print(agg.to_string(index=False, float_format=lambda x: f"{x:10.3f}"))

agg.to_csv(EDA_DIR / "speed_calibration_by_target.csv", index=False)
print(f"\nSaved: speed_calibration_by_target.csv")

# %%
# -----------------------------------------------------------------------------
# Fit meas = a * cmd_hz + b  and  target = a2 * cmd_hz + b2  on steady points
# -----------------------------------------------------------------------------

def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y = a*x + b. Returns (a, b, r2)."""
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (a * x + b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


x = steady["cmd_hz"].to_numpy()
a_meas, b_meas, r2_meas = _fit(x, steady["meas"].to_numpy())
a_targ, b_targ, r2_targ = _fit(x, steady["target"].to_numpy())

print("\nLinear fits on steady-state sweeps (y = a * cmd_hz + b):")
print(f"  meas   = {a_meas:.3f} * cmd_hz + {b_meas:.3f}   (R^2 = {r2_meas:.4f})")
print(f"  target = {a_targ:.3f} * cmd_hz + {b_targ:.3f}   (R^2 = {r2_targ:.4f})")
print("\n(A clean frequency->RPM relationship should have b ~= 0 and R^2 ~= 1;")
print(" a large nonzero intercept or low R^2 indicates the two aren't linked")
print(" by a simple factor over this speed range.)")

# %%
# -----------------------------------------------------------------------------
# Plot 1: meas & target vs cmd_hz (steady-state only), with linear fits
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(steady["cmd_hz"], steady["meas"], s=8, alpha=0.4, label="telem_rpm_meas")
xs = np.linspace(0, steady["cmd_hz"].max(), 50)
axes[0].plot(xs, a_meas * xs + b_meas, "r--", lw=1.5,
             label=f"fit: {a_meas:.2f}x + {b_meas:.1f} (R²={r2_meas:.3f})")
axes[0].set_xlabel("VFD cmd_hz [Hz]")
axes[0].set_ylabel("telem_rpm_meas [RPM]")
axes[0].set_title("Measured RPM vs commanded VFD frequency")
axes[0].legend(fontsize=8)
axes[0].grid(ls=":", alpha=0.4)

axes[1].scatter(steady["cmd_hz"], steady["target"], s=8, alpha=0.4, color="C1",
                 label="telem_rpm_target")
axes[1].plot(xs, a_targ * xs + b_targ, "r--", lw=1.5,
             label=f"fit: {a_targ:.2f}x + {b_targ:.1f} (R²={r2_targ:.3f})")
axes[1].set_xlabel("VFD cmd_hz [Hz]")
axes[1].set_ylabel("telem_rpm_target [RPM]")
axes[1].set_title("Protocol target RPM vs commanded VFD frequency")
axes[1].legend(fontsize=8)
axes[1].grid(ls=":", alpha=0.4)

fig.suptitle("Steady-state speed channels vs VFD command frequency", fontsize=13)
fig.tight_layout()
_save(fig, "speed_calibration_vs_cmd_hz.png")

# %%
# -----------------------------------------------------------------------------
# Plot 2: meas/cmd_hz and target/cmd_hz ratio vs target level
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(agg["target"], agg["meas_over_cmd_hz"], "o-", label="meas / cmd_hz")
ax.plot(agg["target"], agg["target"] / agg["cmd_hz_mean"], "s-", label="target / cmd_hz")
ax.set_xlabel("Protocol target RPM")
ax.set_ylabel("Ratio to cmd_hz [RPM/Hz]")
ax.set_title("Speed/frequency ratio across the protocol staircase\n"
             "(flat line = simple conversion factor holds; drift = it doesn't)")
ax.legend()
ax.grid(ls=":", alpha=0.4)
fig.tight_layout()
_save(fig, "speed_calibration_ratio_vs_target.png")

# %%
# -----------------------------------------------------------------------------
# Plot 3: full time series -- target, cmd_hz-implied, and meas, chronologically
# -----------------------------------------------------------------------------

for fname in df["file"].unique():
    sub = df[df["file"] == fname].sort_values("seq_idx")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub["seq_idx"], sub["target"], lw=1, alpha=0.6, label="telem_rpm_target")
    ax.plot(sub["seq_idx"], sub["meas"], lw=0.6, alpha=0.6, label="telem_rpm_meas")
    ax.plot(sub["seq_idx"], sub["cmd_hz"] * a_meas, lw=0.6, alpha=0.6,
            label=f"cmd_hz * {a_meas:.2f} (fitted factor)")
    ax.set_xlabel("Sweep index (acquisition order)")
    ax.set_ylabel("RPM")
    ax.set_title(f"Speed channels over time -- {fname}")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"speed_calibration_timeseries_{fname}.png")

print("\nDone.")
