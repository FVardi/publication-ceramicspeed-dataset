"""
eda_mic_waveforms.py
=====================
Time-domain waveforms for the new microphone sensor (ambient + machine
mic), for a handful of windows spread across the kappa range.

Layout: rows = mic_amb (ambient reference) / mic_mch (bearing-mounted),
columns = windows sorted by ascending kappa.

Exploratory only -- see _mic_common.py for how this sensor is loaded
(151 windows total in scope_20260901_112732.h5, ~142 survive the
not-actually-rotating filter). Not wired into the main pipeline.

Usage
-----
    python dev/exploration/mic/eda_mic_waveforms.py
    python dev/exploration/mic/eda_mic_waveforms.py --n 6
"""

# %%
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mic_common import load_mic_records  # noqa: E402

from ceramicspeed.config import get_output_dir, load_config  # noqa: E402

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n", type=int, default=5,
                        help="Number of windows to show, spread evenly across kappa (default 5).")
    parser.add_argument("--ms", type=float, default=20.0,
                        help="Milliseconds of each waveform to plot (default 20 ms).")
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR = OUTPUT_DIR / "eda" / "mic"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %%
records = load_mic_records(cfg)
records.sort(key=lambda r: r["kappa"])
print(f"Loaded {len(records)} usable windows (kappa {records[0]['kappa']:.3f}-{records[-1]['kappa']:.3f})")

idx = np.round(np.linspace(0, len(records) - 1, args.n)).astype(int)
chosen = [records[i] for i in idx]
for r in chosen:
    print(f"  {r['name']:8s}  kappa={r['kappa']:.3f}  RPM={r['rpm']:.0f}  T={r['temperature_c']:.0f}C  fs={r['fs']/1e3:.0f}kHz")

# %%
_CHANNELS = [("mic_amb", "Ambient mic", "#2a78d6"), ("mic_mch", "Machine mic", "#e34948")]

fig, axes = plt.subplots(2, len(chosen), figsize=(4.5 * len(chosen), 6), sharey="row")
for row, (key, label, color) in enumerate(_CHANNELS):
    for col, r in enumerate(chosen):
        ax = axes[row, col]
        n = int(args.ms * 1e-3 * r["fs"])
        v = r[key][:n]
        t_ms = np.arange(len(v)) / r["fs"] * 1e3
        ax.plot(t_ms, v, lw=0.5, color=color)
        ax.set_xlabel("Time [ms]")
        if col == 0:
            ax.set_ylabel(f"{label}\nVoltage [V]")
        if row == 0:
            ax.set_title(f"{r['name']}\nκ={r['kappa']:.3f}  RPM={r['rpm']:.0f}  T={r['temperature_c']:.0f}°C",
                         fontsize=9)

fig.suptitle(f"Microphone waveforms -- first {args.ms:.0f} ms ({len(chosen)} windows, sorted by κ)", fontsize=12)
fig.tight_layout()
out_path = EDA_DIR / "eda_mic_waveforms.png"
fig.savefig(out_path, dpi=150)
plt.show()
print(f"Saved: {out_path}")

# %%
if __name__ == "__main__":
    print("\neda_mic_waveforms complete.")
