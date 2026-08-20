"""
speed_measurement_check.py
==========================
Investigates the speed-measurement artifact: for rpm targets above ~2000, the
recorded measured rpm (OGT5000, `rpm_ogt` == `telem_rpm_meas`) reads >=2962,
i.e. far above target, for essentially every segment.

Evidence sought
---------------
1. Timeline: rpm target vs measured rpm, with the VFD command frequency below.
   If the VFD command tracks the staircase while measured rpm diverges, the
   shaft speed was almost certainly correct and only the measurement is bad.
2. Target vs measured scatter (y = x reference) to locate the divergence onset.
3. Measured rpm per VFD Hz: for a VFD-driven induction motor at steady state
   this ratio should be nearly constant (approx. 60 rpm/Hz for a 2-pole motor,
   minus slip); the sensor's failure region shows up as ratio excursions.

Runs as a plain script or as interactive cells (# %%). Reads the raw telem_*
columns, which the speed reconstruction leaves untouched.

Outputs: dev/speed_measurement_check_timeline.png
         dev/speed_measurement_check_scatter.png
         plus a per-target-step summary printed to stdout.
"""

# %%
# =============================================================================
# Load
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_TARGET = "#2a78d6"   # blue
_MEAS = "#e34948"     # red
_VFD = "#1baf7a"      # aqua
_INK = "#333333"

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive cell: __file__ undefined
    ROOT = Path.cwd()

m = pd.read_parquet(ROOT / "outputs" / "new" / "metadata.parquet")
ae = (m.drop_duplicates(subset=["file", "sweep"])
        .sort_values("timestamp_utc").reset_index(drop=True))
ae["t_h"] = (pd.to_datetime(ae["timestamp_utc"])
             - pd.to_datetime(ae["timestamp_utc"]).min()).dt.total_seconds() / 3600.0
run = ae[ae["telem_rpm_target"] > 0].copy()  # drop standstill transitions
print(f"{len(ae)} segments, {len(run)} with nonzero target")

# %%
# =============================================================================
# 1. Timeline: target vs measured rpm, VFD command
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.5), sharex=True,
                         gridspec_kw={"hspace": 0.12})
ax = axes[0]
ax.scatter(run["t_h"], run["telem_rpm_meas"], s=3, color=_MEAS, linewidths=0,
           label="measured rpm (OGT5000)")
ax.scatter(run["t_h"], run["telem_rpm_target"], s=2, color=_TARGET,
           linewidths=0, label="target rpm")
ax.set_ylabel("Rotational speed [rpm]", fontsize=9, color=_INK)
ax.legend(loc="upper right", fontsize=8, frameon=False, markerscale=3)
ax = axes[1]
ax.scatter(run["t_h"], run["telem_vfd_cmd_hz"], s=2, color=_VFD, linewidths=0)
ax.set_ylabel("VFD command [Hz]", fontsize=9, color=_INK)
ax.set_xlabel("Time [h]", fontsize=9, color=_INK)
for ax in axes:
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8, colors=_INK, length=2)
fig.savefig(Path(ROOT) / "dev" / "speed_measurement_check_timeline.png",
            dpi=200, bbox_inches="tight")
plt.show()

# %%
# =============================================================================
# 2. Target vs measured, and rpm-per-Hz
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
ax = axes[0]
ax.scatter(run["telem_rpm_target"], run["telem_rpm_meas"], s=3, color=_MEAS,
           linewidths=0, alpha=0.4)
lim = max(run["telem_rpm_meas"].max(), 3000) * 1.05
ax.plot([0, lim], [0, lim], color=_INK, lw=0.8, ls="--")
ax.set_xlabel("Target rpm", fontsize=9, color=_INK)
ax.set_ylabel("Measured rpm", fontsize=9, color=_INK)
ax = axes[1]
ratio = run["telem_rpm_meas"] / run["telem_vfd_cmd_hz"]
ax.scatter(run["telem_rpm_target"], ratio, s=3, color=_VFD, linewidths=0,
           alpha=0.4)
ax.set_xlabel("Target rpm", fontsize=9, color=_INK)
ax.set_ylabel("Measured rpm per VFD Hz", fontsize=9, color=_INK)
for ax in axes:
    ax.grid(color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8, colors=_INK, length=2)
fig.tight_layout()
fig.savefig(Path(ROOT) / "dev" / "speed_measurement_check_scatter.png",
            dpi=200, bbox_inches="tight")
plt.show()

# %%
# =============================================================================
# 3. Per-target summary
# =============================================================================
run["rel_err"] = (run["telem_rpm_meas"] - run["telem_rpm_target"]).abs() / run["telem_rpm_target"]
summary = (run.groupby("telem_rpm_target")
              .agg(n=("sweep", "count"),
                   med_meas=("telem_rpm_meas", "median"),
                   med_vfd_hz=("telem_vfd_cmd_hz", "median"),
                   frac_err_gt10pct=("rel_err", lambda s: (s > 0.10).mean())))
summary["med_rpm_per_hz"] = summary["med_meas"] / summary["med_vfd_hz"]
pd.set_option("display.width", 120)
print(summary.round(2).to_string())

# %%

x = run["sweep"].values
y = run["telem_rpm_meas"].values

# %%
plt.plot(x, y)