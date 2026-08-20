"""
eda_speed_verification.py
===========================
Independent check of whether the ~2968 RPM plateau seen in telem_rpm_meas
at the top of the speed staircase (see eda_speed_calibration.py) reflects
a genuine physical speed ceiling, or a sensor artifact -- using a signal
that doesn't depend on any of the three disputed telemetry channels:
the vibration content of the AE waveform itself.

Method
------
1. Calibrate: at LOW/MID target levels (well below the disputed high
   range, and where telem_vfd_cmd_hz is nowhere near any saturation),
   find the dominant low-frequency peak in the averaged envelope
   spectrum of the AE signal, and check that it scales linearly with
   cmd_hz. This establishes an order (peak_freq / cmd_hz) with no
   reliance on telem_rpm_meas.
2. Extrapolate that same order into the disputed HIGH target range
   (2400-3000 RPM, where telem_rpm_meas plateaus at ~2968). If the
   vibration peak keeps climbing in lock-step with cmd_hz's prediction,
   the physical shaft speed is still increasing -- telem_rpm_meas is
   the one that's wrong (sensor saturation). If the vibration peak
   ALSO plateaus at the same point as telem_rpm_meas, the shaft speed
   itself is genuinely capping (e.g. motor slip / torque limit) and
   telem_rpm_meas would be closer to correct there.

Caveat: mains hum (50 Hz, and its harmonics) sits inside the frequency
band of interest at the top of the range investigated here, since
cmd_hz reaches ~40-50 Hz -- printed output flags any peak landing
suspiciously close to a mains harmonic so it isn't over-interpreted.

Usage
-----
    python dev/exploration/eda_speed_verification.py
    python dev/exploration/eda_speed_verification.py --config alt.yaml
"""

# %%
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files

# %%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sensor", type=str, default="AE", choices=["AE", "UL"])
    parser.add_argument("--max-sweeps-per-target", type=int, default=30)
    parser.add_argument("--fmax", type=float, default=120.0,
                        help="Upper bound of the peak-search band [Hz]")
    parser.add_argument("--fmin", type=float, default=3.0,
                        help="Lower bound of the peak-search band [Hz], "
                             "excludes DC/very-low-freq drift")
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
cfg = load_config(args.config)

INPUT_DIR  = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

SENSOR      = args.sensor
N_MAX       = args.max_sweeps_per_target
FMIN, FMAX  = args.fmin, args.fmax

# Targets spanning uncontested low/mid range (calibration) through the
# disputed high range where telem_rpm_meas plateaus.
TARGET_LEVELS = [400, 800, 1200, 1600, 2000, 2200, 2400, 2600, 2800, 3000]
CMD_DELTA = 0.01

MAINS_HZ = 50.0
MAINS_TOL = 1.5  # Hz -- flag peaks within this of a mains harmonic


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(EDA_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {name}")


# %%
# -----------------------------------------------------------------------------
# Discover file(s) and index steady-state sweeps per target level
# -----------------------------------------------------------------------------

FILE_PATTERNS = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
print(f"Found {len(files)} HDF5 file(s)")
fpath = files[0]
if len(files) > 1:
    print(f"  Using first file only: {fpath.name}")

with h5py.File(fpath, "r") as f:
    sweep_keys = sorted(f["sweeps"].keys(), key=lambda n: int(n.split("_")[1]))
    prev_cmd = None
    index: dict[float, list[str]] = {t: [] for t in TARGET_LEVELS}
    cmd_hz_by_target: dict[float, list[float]] = {t: [] for t in TARGET_LEVELS}
    meas_by_target: dict[float, list[float]] = {t: [] for t in TARGET_LEVELS}

    for name in sweep_keys:
        attrs = f["sweeps"][name].attrs
        if "telem_vfd_cmd_hz" not in attrs or "telem_rpm_target" not in attrs:
            continue
        cmd_hz = float(attrs["telem_vfd_cmd_hz"])
        target = float(attrs["telem_rpm_target"])
        running = bool(attrs.get("telem_vfd_is_running", False))
        steady = running and prev_cmd is not None and abs(cmd_hz - prev_cmd) < CMD_DELTA and cmd_hz > 0.01
        prev_cmd = cmd_hz

        if steady and target in index and len(index[target]) < N_MAX:
            if SENSOR in f["sweeps"][name] and "voltage" in f["sweeps"][name][SENSOR]:
                index[target].append(name)
                cmd_hz_by_target[target].append(cmd_hz)
                meas_by_target[target].append(float(attrs.get("telem_rpm_meas", np.nan)))

    for t in TARGET_LEVELS:
        print(f"  target={t:5.0f}  n_sweeps_selected={len(index[t]):3d}")

    # %%
    # -------------------------------------------------------------------------
    # For each target level: average envelope spectrum across selected sweeps
    # -------------------------------------------------------------------------

    results = []
    for t in TARGET_LEVELS:
        names = index[t]
        if not names:
            continue

        psd_sum = None
        freqs = None
        for name in names:
            grp = f["sweeps"][name][SENSOR]
            v = grp["voltage"][()]
            time = grp["time"][()]
            fs = 1.0 / (time[1] - time[0])

            v = v - np.mean(v)
            envelope = np.abs(hilbert(v))
            envelope = envelope - np.mean(envelope)

            spectrum = np.abs(np.fft.rfft(envelope)) ** 2
            f_axis = np.fft.rfftfreq(len(envelope), d=1.0 / fs)

            if psd_sum is None:
                psd_sum = spectrum
                freqs = f_axis
            else:
                psd_sum = psd_sum + spectrum

        psd_avg = psd_sum / len(names)

        band_mask = (freqs >= FMIN) & (freqs <= FMAX)
        band_freqs = freqs[band_mask]
        band_psd = psd_avg[band_mask]
        peak_idx = int(np.argmax(band_psd))
        peak_freq = float(band_freqs[peak_idx])

        mains_flag = any(abs(peak_freq - k * MAINS_HZ) < MAINS_TOL for k in (1, 2))

        cmd_hz_mean = float(np.mean(cmd_hz_by_target[t]))
        meas_mean = float(np.nanmean(meas_by_target[t]))

        results.append({
            "target": t,
            "n": len(names),
            "cmd_hz_mean": cmd_hz_mean,
            "meas_mean": meas_mean,
            "peak_freq_hz": peak_freq,
            "mains_suspect": mains_flag,
            "band_freqs": band_freqs,
            "band_psd": band_psd,
        })

        flag = "  <-- near mains harmonic, interpret with caution" if mains_flag else ""
        print(f"  target={t:5.0f}  cmd_hz={cmd_hz_mean:7.3f}  "
              f"meas={meas_mean:8.1f}  peak_freq={peak_freq:6.2f} Hz{flag}")

# %%
# -----------------------------------------------------------------------------
# Calibrate order (peak_freq / cmd_hz) on the low/mid, uncontested range
# -----------------------------------------------------------------------------

CALIB_MAX_TARGET = 2000.0  # below this, no dispute about cmd_hz validity

calib = [r for r in results if r["target"] <= CALIB_MAX_TARGET and not r["mains_suspect"]]
if len(calib) >= 2:
    x = np.array([r["cmd_hz_mean"] for r in calib])
    y = np.array([r["peak_freq_hz"] for r in calib])
    order = float(np.sum(x * y) / np.sum(x * x))  # proportional fit through origin
    print(f"\nCalibrated order (peak_freq / cmd_hz) from targets <= {CALIB_MAX_TARGET:.0f}: "
          f"{order:.4f}")
else:
    order = float("nan")
    print("\nNot enough clean calibration points to fit an order.")

# %%
# -----------------------------------------------------------------------------
# Plot: predicted vs observed peak frequency across the full range
# -----------------------------------------------------------------------------

targets      = [r["target"] for r in results]
cmd_hz_means = [r["cmd_hz_mean"] for r in results]
meas_means   = [r["meas_mean"] for r in results]
peak_freqs   = [r["peak_freq_hz"] for r in results]
mains_flags  = [r["mains_suspect"] for r in results]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

ax1.plot(targets, [c * order for c in cmd_hz_means], "o--", color="C0",
         label=f"predicted from cmd_hz (order={order:.3f}, calibrated <= {CALIB_MAX_TARGET:.0f} RPM)")
colors = ["red" if m else "C2" for m in mains_flags]
ax1.scatter(targets, peak_freqs, c=colors, s=60, zorder=5,
            label="observed AE envelope peak (red = near mains harmonic)")
ax1.set_ylabel("Frequency [Hz]")
ax1.set_title(f"Independent check: AE envelope peak vs cmd_hz-predicted trend ({SENSOR} sensor)")
ax1.legend(fontsize=8)
ax1.grid(ls=":", alpha=0.4)
ax1.axvline(CALIB_MAX_TARGET, color="gray", ls=":", lw=1)
ax1.text(CALIB_MAX_TARGET, ax1.get_ylim()[1] * 0.95, " calibration | extrapolation ",
          fontsize=7, color="gray", ha="left")

ax2.plot(targets, meas_means, "s-", color="C3", label="telem_rpm_meas")
ax2.plot(targets, targets, "k:", lw=1, label="target = target (reference)")
ax2.set_xlabel("Protocol target RPM")
ax2.set_ylabel("telem_rpm_meas [RPM]")
ax2.legend(fontsize=8)
ax2.grid(ls=":", alpha=0.4)

fig.tight_layout()
_save(fig, f"speed_verification_{SENSOR}_envelope_peak.png")

# %%
# -----------------------------------------------------------------------------
# Plot: overlaid envelope spectra per target (visual sanity check)
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.get_cmap("viridis")
for i, r in enumerate(results):
    color = cmap(i / max(len(results) - 1, 1))
    psd_db = 10 * np.log10(r["band_psd"] + 1e-30)
    ax.plot(r["band_freqs"], psd_db, color=color, lw=1,
            label=f"target={r['target']:.0f} (cmd_hz={r['cmd_hz_mean']:.1f})")
    ax.axvline(r["peak_freq_hz"], color=color, ls=":", lw=0.8, alpha=0.6)

ax.axvline(MAINS_HZ, color="red", ls="--", lw=1, alpha=0.5, label="50 Hz mains")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Envelope PSD [dB]")
ax.set_title(f"Averaged {SENSOR} envelope spectra by target level")
ax.legend(fontsize=7, ncol=2)
ax.grid(ls=":", alpha=0.4)
fig.tight_layout()
_save(fig, f"speed_verification_{SENSOR}_spectra_overlay.png")

print("\nDone.")
