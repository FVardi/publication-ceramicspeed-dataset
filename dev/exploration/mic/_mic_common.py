"""
_mic_common.py
===============
Shared loader for the new microphone sensor (oe_samples/mic_amb, mic_mch),
used by the eda_mic_*.py scripts in this folder.

Exploratory only -- not wired into the main pipeline or src/ceramicspeed/.
The mic channel currently exists in exactly one file
(scope_20260901_112732.h5, 151 windows out of 3964 sweeps), stored under a
top-level oe_samples/ group that's structurally separate from sweeps/AE|UL|SP:

  oe_samples/oe_NNN/{mic_amb, mic_mch}   1-D voltage arrays, ~74k samples
                                          each (~0.74-0.93 s depending on
                                          the window's own sample_rate_hz
                                          attr -- 80 kHz in this file, was
                                          100 kHz in the earlier 5-window
                                          pilot capture; always read fs
                                          per-window, never assume a value)
  oe_NNN attrs                           telem_* fields already embedded
                                          directly (rpm/temp/cmd_hz -- no
                                          need to look up a sweep for the
                                          label), plus tick_start (seconds,
                                          same clock as sweeps' own `tick`)

Each mic_amb, and its neighbour mic_mch, is one instrument -- an "ambient"
(background reference) and a "machine" (bearing-mounted) microphone that
fire together, so records here return both waveforms per window, aligned by
construction (no per-window matching needed for the mic pair itself).

Speed: telem_rpm_meas is unreliable throughout this dataset (see
dev/exploration/eda_speed_calibration.py) -- rpm is reconstructed here via
the same telem_vfd_cmd_hz * 59.5 correction and the same "target==100 means
not actually rotating" exclusion the main pipeline uses
(ceramicspeed.loading._normalize_sweep_params), applied directly to each
oe_NNN's own embedded telemetry.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import get_input_dir
from ceramicspeed.loading import _normalize_sweep_params

#: The only file this sensor currently exists in.
MIC_FILE_PATTERN = "scope_20260901_112732"

#: Keratech 22 (Kerax) viscosity constants -- same fallback the main
#: pipeline uses (this file's lubricant metadata omits viscosity, like
#: every scope-format file so far).
_VISCOSITY_FALLBACK = {"viscosity_40c_cst": 22.0, "viscosity_100c_cst": 4.1}


def find_mic_file(cfg: dict, file_pattern: str = MIC_FILE_PATTERN) -> Path:
    """Return the path to the mic-carrying HDF5 file (ignores cfg's own
    filters.file_patterns, which currently points elsewhere)."""
    input_dir = Path(get_input_dir(cfg))
    matches = sorted(input_dir.glob(f"{file_pattern}*"))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{file_pattern}*' in {input_dir}"
        )
    return matches[0]


def load_mic_records(
    cfg: dict,
    d_pw_mm: float | None = None,
    file_pattern: str = MIC_FILE_PATTERN,
) -> list[dict]:
    """Load every oe_samples window: both mic waveforms, telemetry, and kappa.

    Returns
    -------
    list[dict]
        One record per oe_NNN window still present after the rotation
        filter, each with keys: ``name``, ``tick_start``, ``fs``,
        ``mic_amb`` (voltage array), ``mic_mch`` (voltage array), ``rpm``,
        ``temperature_c``, ``kappa``.
    """
    d_pw_mm = d_pw_mm if d_pw_mm is not None else cfg["bearing"]["d_pw_mm"]
    fpath = find_mic_file(cfg, file_pattern)

    records: list[dict] = []
    with h5py.File(fpath, "r") as f:
        lm = dict(f["metadata"]["lubricant"].attrs)
        for k, v in _VISCOSITY_FALLBACK.items():
            lm.setdefault(k, v)

        oe_grp = f["oe_samples"]
        for name in sorted(oe_grp.keys(), key=lambda s: int(s.split("_")[1])):
            win = oe_grp[name]
            tp = _normalize_sweep_params(dict(win.attrs))
            rpm = float(tp.get("rpm", np.nan))
            temp_c = float(tp.get("temperature_c", np.nan))

            if not np.isfinite(rpm) or rpm <= 0:
                continue  # not actually rotating (target=100, or genuinely idle)
            if not np.isfinite(temp_c):
                continue

            try:
                kap = calculate_kappa(
                    rpm=rpm, temp_c=temp_c, d_pw=d_pw_mm,
                    nu_40=float(lm["viscosity_40c_cst"]),
                    nu_100=float(lm["viscosity_100c_cst"]),
                )
            except Exception:
                continue
            if not np.isfinite(kap):
                continue

            mic_amb = win["mic_amb"][()]
            mic_mch = win["mic_mch"][()]
            fs = float(win["mic_amb"].attrs["sample_rate_hz"])

            records.append({
                "name": name,
                "tick_start": float(win.attrs["tick_start"]),
                "fs": fs,
                "mic_amb": mic_amb,
                "mic_mch": mic_mch,
                "rpm": rpm,
                "temperature_c": temp_c,
                "kappa": kap,
            })

    return records
