"""
operating_condition_varying_protocol.py
=======================================
Illustrates the full nominal operating-condition protocol as three stacked
scatter plots sharing a time axis:

  1. Rotational speed: 13 pyramid staircase sweeps (100 -> 3000 -> 100 rpm in
     100 rpm steps, 60 s holds, extended hold at the 3000 rpm peak).
  2. Housing temperature: 13 blocks from 40 to 100 degC in 5 degC increments,
     one block per pyramid.
  3. The resulting nominal viscosity ratio kappa, computed per segment from
     the nominal speed and temperature via the ISO 281 / ASTM D341 relations
     (ceramicspeed.calculate_kappa), for Keratech 22 on the SYJ25 bearing.

Everything is nominal -- no data is loaded. Temperature transitions between
blocks are drawn as instantaneous.

Output: dev/operating_condition_varying_protocol.png
"""
# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ceramicspeed.calculate_kappa import calculate_kappa
# %%
RPM_MIN = 100
RPM_MAX = 3000
RPM_STEP = 100
HOLD_S = 60.0             # nominal hold per staircase step
PEAK_HOLD_S = 20.0 * 60   # extended hold at the 3000 rpm peak
SEGMENT_INTERVAL_S = 5.0  # one acquired segment every ~5 s within a hold

TEMPS_C = np.arange(40, 105, 5)  # 13 temperature blocks
D_PW_MM = 38.0                   # SYJ25 pitch diameter
NU_40 = 22.0                     # Keratech 22 [cSt]
NU_100 = 4.1

_SPEED = "#2a78d6"   # blue
_TEMP = "#eb6834"    # orange
_KAPPA = "#1baf7a"   # aqua
_INK = "#333333"
# %%
# nominal staircase: up 100..3000, down 2900..100 (peak visited once)
rpms_up = np.arange(RPM_MIN, RPM_MAX + RPM_STEP, RPM_STEP)
rpms_down = np.arange(RPM_MAX - RPM_STEP, RPM_MIN - RPM_STEP, -RPM_STEP)
pyramid = np.concatenate([rpms_up, rpms_down])

t, rpm_v, temp_v = [], [], []
t0 = 0.0
for temp in TEMPS_C:
    for rpm in pyramid:
        hold = PEAK_HOLD_S if rpm == RPM_MAX else HOLD_S
        seg_times = np.arange(0.0, hold, SEGMENT_INTERVAL_S)
        t.extend(t0 + seg_times)
        rpm_v.extend([rpm] * len(seg_times))
        temp_v.extend([temp] * len(seg_times))
        t0 += hold

t = np.asarray(t) / 3600.0  # hours
rpm_v = np.asarray(rpm_v, dtype=float)
temp_v = np.asarray(temp_v, dtype=float)
kappa_v = np.asarray([
    calculate_kappa(rpm=r, temp_c=c, d_pw=D_PW_MM, nu_40=NU_40, nu_100=NU_100)
    for r, c in zip(rpm_v, temp_v)])

fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True,
                         gridspec_kw={"hspace": 0.12})
panels = [
    (rpm_v, _SPEED, "Nominal speed [rpm]", (0, RPM_MAX + 200)),
    (temp_v, _TEMP, "Nominal temperature [$^\\circ$C]", (35, 105)),
    (kappa_v, _KAPPA, "Nominal $\\kappa$ [-]", (0, float(kappa_v.max()) * 1.1)),
]
for ax, (vals, color, label, ylim) in zip(axes, panels):
    ax.scatter(t, vals, s=2, color=color, linewidths=0)
    ax.set_ylabel(label, fontsize=9, color=_INK)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_INK)
    ax.tick_params(labelsize=8, colors=_INK, length=2)
axes[-1].set_xlabel("Time [h]", fontsize=9, color=_INK)
axes[-1].set_xlim(0, t[-1])

try:
    out = Path(__file__).with_suffix(".png")
except NameError:  # interactive cell: __file__ undefined
    out = Path.cwd() / "dev" / "operating_condition_varying_protocol.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {out}")
print(f"{len(TEMPS_C)} blocks x {len(pyramid)} holds, {len(t)} nominal segments, "
      f"duration {t[-1]:.1f} h, kappa {kappa_v.min():.3f}-{kappa_v.max():.2f}")

# %%
