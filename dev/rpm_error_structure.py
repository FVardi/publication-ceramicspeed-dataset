"""
rpm_error_structure.py
======================
Examines whether the OGT5000 rpm-measurement error has exploitable structure:

  A. Static transfer curve: measured rpm vs target, per temperature block.
     If the 13 block curves coincide, the distortion is a stable transfer
     function (invertible in its monotone region), not drift.
  B. Relative error (measured/target) vs target, per block.
  C. Hysteresis: up-sweep vs down-sweep measured rpm at the same target.
  D. Within-hold drift: slope of measured rpm across each ~60 s hold.

The corrupted final block is analysed separately (flagged, not mixed in).
Runs as a plain script or as interactive cells (# %%). Reads the raw telem_*
columns, which the speed reconstruction leaves untouched.

Outputs: dev/rpm_error_structure.png + summary statistics to stdout.
"""

# %%
# =============================================================================
# Load and derive holds / visits
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_INK = "#333333"

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # interactive cell: __file__ undefined
    ROOT = Path.cwd()

m = pd.read_parquet(ROOT / "outputs" / "new" / "metadata.parquet")
ae = (m.drop_duplicates(subset=["file", "sweep"])
        .sort_values("timestamp_utc").reset_index(drop=True))
ae["t_s"] = (pd.to_datetime(ae["timestamp_utc"])
             - pd.to_datetime(ae["timestamp_utc"]).min()).dt.total_seconds()
run = ae[ae["telem_rpm_target"] > 0].copy()
run["block"] = run["telem_temp_target"]

# hold id: consecutive same rpm target
run["hold_id"] = (run["telem_rpm_target"] != run["telem_rpm_target"].shift()).cumsum()
# up/down: position of the hold within its block's staircase
hold_first = run.groupby("hold_id").agg(
    block=("block", "first"), target=("telem_rpm_target", "first"),
    t0=("t_s", "min")).sort_values("t0")
hold_first["visit"] = hold_first.groupby(["block", "target"]).cumcount()  # 0=up, 1=down
run = run.merge(hold_first[["visit"]], left_on="hold_id", right_index=True)

last_block = run["block"].max()
good = run[run["block"] < last_block]
bad = run[run["block"] == last_block]
print(f"{len(good)} segments in blocks < {last_block:.0f} degC; "
      f"{len(bad)} in the corrupted final block (excluded)")

# %%
# =============================================================================
# A. Cross-block stability of the transfer curve
# =============================================================================
med = (good.groupby(["block", "telem_rpm_target"])["telem_rpm_meas"]
           .median().unstack(0))
cross_block_std = med.std(axis=1)
cross_block_mean = med.mean(axis=1)
print("=== A. cross-block stability of the transfer curve ===")
print(pd.DataFrame({"mean_meas": cross_block_mean.round(1),
                    "std_across_blocks": cross_block_std.round(2),
                    "rel_std_pct": (100 * cross_block_std / cross_block_mean).round(2)}
                   ).to_string())

# %%
# =============================================================================
# C. Hysteresis up vs down
# =============================================================================
hv = (good.groupby(["block", "telem_rpm_target", "visit"])["telem_rpm_meas"]
          .median().unstack("visit"))
hv = hv.dropna()
hv["updown_diff"] = hv[1] - hv[0]
print("=== C. up- vs down-sweep hysteresis (median per target, all blocks) ===")
hyst = hv.groupby("telem_rpm_target")["updown_diff"].agg(["median", "std"]).round(2)
print(hyst.to_string())

# %%
# =============================================================================
# D. Within-hold drift
# =============================================================================
def _slope(g):
    if len(g) < 4:
        return np.nan
    t = g["t_s"].values - g["t_s"].values[0]
    return np.polyfit(t, g["telem_rpm_meas"].values, 1)[0]

slopes = good.groupby("hold_id").apply(_slope)
sl = hold_first.join(slopes.rename("slope"), how="inner")
sl = sl[sl.index.isin(good["hold_id"])]
print("=== D. within-hold drift (rpm/s), distribution over all holds ===")
print(sl["slope"].describe().round(3).loc[["mean", "50%", "std", "min", "max"]].to_string())

# %%
# =============================================================================
# Figure: A/B/C/D panels
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
cmap = plt.get_cmap("viridis")
blocks = sorted(good["block"].unique())

ax = axes[0, 0]
for b in blocks:
    s = med[b].dropna()
    ax.plot(s.index, s.values, color=cmap((b - min(blocks)) / (max(blocks) - min(blocks))),
            lw=1.0, alpha=0.8)
ax.plot([0, 3000], [0, 3000], color=_INK, lw=0.8, ls="--")
ax.set_xlabel("Target rpm"); ax.set_ylabel("Median measured rpm")
ax.set_title("A. Transfer curve per block (colour = block)", fontsize=9)

ax = axes[0, 1]
for b in blocks:
    s = (med[b] / med[b].index).dropna()
    ax.plot(s.index, s.values, color=cmap((b - min(blocks)) / (max(blocks) - min(blocks))),
            lw=1.0, alpha=0.8)
ax.axhline(1.0, color=_INK, lw=0.8, ls="--")
ax.set_xlabel("Target rpm"); ax.set_ylabel("Measured / target")
ax.set_title("B. Relative error per block", fontsize=9)

ax = axes[1, 0]
ax.scatter(hv.index.get_level_values("telem_rpm_target"), hv["updown_diff"],
           s=8, color="#2a78d6", linewidths=0, alpha=0.6)
ax.axhline(0, color=_INK, lw=0.8, ls="--")
ax.set_xlabel("Target rpm"); ax.set_ylabel("Down-sweep $-$ up-sweep [rpm]")
ax.set_title("C. Hysteresis per (block, target)", fontsize=9)

ax = axes[1, 1]
ax.scatter(sl["target"], sl["slope"], s=8, color="#1baf7a", linewidths=0, alpha=0.6)
ax.axhline(0, color=_INK, lw=0.8, ls="--")
ax.set_xlabel("Target rpm"); ax.set_ylabel("Within-hold slope [rpm/s]")
ax.set_title("D. Drift within holds", fontsize=9)

for ax in axes.flat:
    ax.grid(color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8, colors=_INK, length=2)
fig.tight_layout()
fig.savefig(Path(ROOT) / "dev" / "rpm_error_structure.png", dpi=200,
            bbox_inches="tight")
plt.show()
