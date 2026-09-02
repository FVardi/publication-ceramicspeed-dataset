"""
eda_mic_spectrogram.py
========================
Raw/unfiltered spectral heatmap for the microphone sensor (ambient +
machine mic): frequency (y-axis) vs. window sorted by kappa (x-axis).

No prefiltering or band assumptions -- the point is to see where real,
kappa-dependent content actually lives before ever choosing sub-bands
(same process used for AE/UL earlier: dev/exploration/eda_spectrogram.py).

Exploratory only -- see _mic_common.py. Not wired into the main pipeline.

Usage
-----
    python dev/exploration/mic/eda_mic_spectrogram.py
    python dev/exploration/mic/eda_mic_spectrogram.py --fmax 20000
"""

# %%
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mic_common import load_mic_records  # noqa: E402

from ceramicspeed.config import get_output_dir, load_config  # noqa: E402

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fmax", type=float, default=None,
                        help="Upper frequency to display [Hz] (default: full Nyquist).")
    parser.add_argument("--nperseg", type=int, default=4096,
                        help="Welch segment length (default 4096).")
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

kappas = np.array([r["kappa"] for r in records])

# %%
_CHANNELS = [("mic_amb", "Ambient mic"), ("mic_mch", "Machine mic")]

fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"hspace": 0.35})

for ax, (key, label) in zip(axes, _CHANNELS):
    fs = records[0]["fs"]  # constant across this file
    nperseg = min(args.nperseg, min(len(r[key]) for r in records))
    mats = []
    f_axis = None
    for r in records:
        f_axis, p = welch(r[key], fs=r["fs"], nperseg=nperseg,
                          noverlap=nperseg // 2, window="hann")
        mats.append(p)
    mat_db = 10 * np.log10(np.maximum(np.stack(mats), 1e-30))

    fmax = args.fmax or f_axis[-1]
    fmask = f_axis <= fmax
    f_khz = f_axis[fmask] / 1e3
    m = mat_db[:, fmask]

    vlo, vhi = np.percentile(m, [2, 98])
    im = ax.imshow(
        m.T, aspect="auto", origin="lower",
        extent=[0, len(records) - 1, f_khz[0], f_khz[-1]],
        cmap="viridis", vmin=vlo, vmax=vhi, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="PSD [dB]", pad=0.01)
    ax.set_ylabel("Frequency [kHz]")
    ax.set_title(f"{label} -- spectral heatmap ({len(records)} windows, 0-{f_khz[-1]:.1f} kHz, Nyquist={fs/2/1e3:.0f} kHz)",
                 fontsize=10)
    n_ticks = 10
    tick_idx = np.round(np.linspace(0, len(records) - 1, n_ticks)).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{kappas[i]:.2f}" for i in tick_idx], fontsize=8)

axes[-1].set_xlabel("κ (lubrication ratio) -- windows sorted left→right by κ")
fig.suptitle("Microphone spectral heatmaps: PSD vs κ (raw, unfiltered, welch)", fontsize=12)

out_path = EDA_DIR / "eda_mic_spectrogram.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")

# %%
if __name__ == "__main__":
    print("\neda_mic_spectrogram complete.")
