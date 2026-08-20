"""
00_speed_reconstruction.py
==========================
Reconstruct the true shaft speed from the VFD command frequency and write a
corrected metadata.parquet, replacing the faulty OGT5000 measurement.

Background (see dev/speed_measurement_check.py and dev/rpm_error_structure.py):
the recorded OGT rpm is a static distortion of the true speed -- a constant
phantom offset of ~365 rpm (visible at standstill), faithful shaft tracking in
between, and a hard ceiling at ~2970 rpm (max count rate). The VFD command
frequency is a clean per-segment actuator record; in the sensor's healthy band
the OGT satisfies  meas ~= K_RPM_PER_HZ * f_vfd + OFFSET  (R^2 ~ 0.995), with
K_RPM_PER_HZ ~= 59.5 consistent with a 2-pole induction motor (60 rpm/Hz
synchronous, ~1% slip).

Reconstruction:  rpm := K_RPM_PER_HZ * telem_vfd_cmd_hz
  * valid wherever the drive was running, including the OGT ceiling zone, the
    60 Hz rail (~3570 rpm), and the final temperature block (whose OGT channel
    is corrupted but whose VFD channel is clean);
  * stationary segments (f_vfd ~ 0: the bottom staircase steps and the
    between-block transitions) get rpm = 0 and are excluded downstream by the
    config rpm_min filter;
  * the original OGT value is preserved as `rpm_ogt`, and a per-segment
    `speed_source` flag records provenance.

Validation: in the OGT's healthy band the reconstruction must agree with
(rpm_ogt - OGT_OFFSET); the agreement statistics are printed.

K_RPM_PER_HZ = 59.5 is empirically calibrated: in the June 2026 sessions
(scope_20260619, scope_20260625), which log both OGT rpm and VFD Hz, the
unsaturated regulated band (17 < Hz < 49, meas < 2900) fits
meas = 59.57 x Hz - 12 and meas = 59.36 x Hz - 2 respectively (R^2 ~ 0.992),
i.e. k = 59.5 +/- 0.1 with ~zero offset -- consistent with a 2-pole induction
motor at ~0.8% slip. Pass --k to override.

Usage
-----
    python scripts/new/signal_processing/00_speed_reconstruction.py
    python scripts/new/signal_processing/00_speed_reconstruction.py --k 59.5
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from ceramicspeed.config import load_config, get_output_dir

K_RPM_PER_HZ_DEFAULT = 59.5   # provisional: 2-pole induction, ~1% slip
OGT_OFFSET = 365.0            # phantom counts, measured at standstill
OGT_CEILING = 2900.0          # treat OGT readings above this as saturated
HZ_STATIONARY = 0.5           # below this the drive is considered off
# OGT healthy band used for validation (reconstruction vs meas - offset)
VALID_BAND_HZ = (10.0, 42.0)

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--config", type=str, default=None)
_p.add_argument("--k", type=float, default=K_RPM_PER_HZ_DEFAULT,
                help="rpm per VFD Hz (default %(default)s, provisional).")
args, _ = _p.parse_known_args()

cfg = load_config(args.config)
NEW_DIR = get_output_dir(cfg) / "new"
META = NEW_DIR / "metadata.parquet"
BACKUP = NEW_DIR / "metadata_ogt_original.parquet"

m = pd.read_parquet(META)

if "rpm_ogt" in m.columns:
    print("metadata.parquet already reconstructed (rpm_ogt present); "
          "re-deriving from rpm_ogt.")
    ogt = m["rpm_ogt"]
else:
    if not BACKUP.exists():
        shutil.copy2(META, BACKUP)
        print(f"Original metadata backed up -> {BACKUP.name}")
    ogt = m["rpm"]
    m["rpm_ogt"] = ogt

hz = m["telem_vfd_cmd_hz"].astype(float)
recon = args.k * hz
stationary = hz < HZ_STATIONARY

m["rpm"] = np.where(stationary, 0.0, recon)
m["speed_source"] = np.where(stationary, "stationary_excluded", "vfd_hz")

# ---- validation against the OGT healthy band --------------------------------
band = (~stationary) & hz.between(*VALID_BAND_HZ) & (ogt < OGT_CEILING)
diff = (recon[band] - (ogt[band] - OGT_OFFSET))
rel = diff / recon[band]
print(f"k = {args.k} rpm/Hz (PROVISIONAL)")
print(f"validation band {VALID_BAND_HZ} Hz: n = {band.sum()} segments")
print(f"  recon vs (OGT - {OGT_OFFSET:.0f}): median diff = {diff.median():+.1f} rpm, "
      f"median |rel| = {rel.abs().median()*100:.2f}%, p95 |rel| = {rel.abs().quantile(0.95)*100:.2f}%")

# ---- summary ------------------------------------------------------------------
seg = m.drop_duplicates(subset=["file", "sweep"])
n_stat = (seg["speed_source"] == "stationary_excluded").sum()
run = seg[seg["speed_source"] == "vfd_hz"]
print(f"\nsegments: {len(seg)} total; {n_stat} stationary (rpm=0, dropped by "
      f"rpm_min filter); {len(run)} running")
print(f"reconstructed speed range (running): "
      f"{run['rpm'].min():.0f}-{run['rpm'].max():.0f} rpm")

m.to_parquet(META, engine="pyarrow")
print(f"\nWrote corrected {META.name} (original OGT values kept in rpm_ogt)")

if __name__ == "__main__":
    print("\n00_speed_reconstruction complete.")
