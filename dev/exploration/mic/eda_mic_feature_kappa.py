"""
eda_mic_feature_kappa.py
==========================
Broadband feature extraction for the microphone sensor (ambient + machine
mic) and Spearman correlation of each feature with kappa -- the mic
equivalent of scripts/new/signal_processing/06_feature_kappa_figure.py.

Uses ceramicspeed.features.extract_features unchanged (it's generic in
signal + sample rate, no mic-specific code needed) to get the same 14
broadband features used everywhere else in this project: rms, skewness,
kurtosis, crest_factor, shape_factor, margin_factor, mobility, complexity,
dominant_frequency, center_frequency, spectral_bandwidth, spectral_skewness,
spectral_kurtosis, spectral_flatness.

No sub-bands yet -- eda_mic_spectrogram.py hasn't been used to pick any,
and this is deliberately broadband-only until that's done.

Exploratory only -- see _mic_common.py. Not wired into the main pipeline.

Usage
-----
    python dev/exploration/mic/eda_mic_feature_kappa.py
"""

# %%
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mic_common import load_mic_records  # noqa: E402

from ceramicspeed.config import get_output_dir, load_config  # noqa: E402
from ceramicspeed.features import extract_features  # noqa: E402

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)
OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR = OUTPUT_DIR / "eda" / "mic"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %%
records = load_mic_records(cfg)
print(f"Loaded {len(records)} usable windows (kappa {min(r['kappa'] for r in records):.3f}-"
      f"{max(r['kappa'] for r in records):.3f})")

# %%
# -----------------------------------------------------------------------------
# Feature extraction -- broadband only, both channels
# -----------------------------------------------------------------------------

_CHANNELS = {"mic_amb": "MIC_AMB", "mic_mch": "MIC_MCH"}

rows = []
for r in records:
    row = {"name": r["name"], "kappa": r["kappa"], "rpm": r["rpm"],
           "temperature_c": r["temperature_c"]}
    for key, prefix in _CHANNELS.items():
        feats = extract_features(r[key], r["fs"])
        for fname, val in feats.items():
            row[f"{prefix}__{fname}"] = val
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Extracted {len(_CHANNELS) * 14} features x {len(df)} windows")

feature_cols = [c for c in df.columns if "__" in c]

# %%
# -----------------------------------------------------------------------------
# Spearman correlation with kappa
# -----------------------------------------------------------------------------

corr_rows = []
for col in feature_cols:
    rho, p = spearmanr(df[col], df["kappa"])
    channel, fname = col.split("__", 1)
    corr_rows.append({"channel": channel, "feature": fname, "column": col,
                       "rho": rho, "p_value": p})

corr_df = pd.DataFrame(corr_rows).sort_values("rho", key=np.abs, ascending=False)
csv_path = EDA_DIR / "mic_feature_kappa_correlation.csv"
corr_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
print()
print(corr_df.to_string(index=False))

# %%
# -----------------------------------------------------------------------------
# Bar chart, sorted by |rho|, coloured by channel
# -----------------------------------------------------------------------------

_COLOR = {"MIC_AMB": "#2a78d6", "MIC_MCH": "#e34948"}
_LABEL = {"MIC_AMB": "Ambient mic", "MIC_MCH": "Machine mic"}

plot_df = corr_df.sort_values("rho")
labels = [f"{_LABEL[r.channel]} · {r.feature.replace('_', ' ')}" for r in plot_df.itertuples()]
colors = [_COLOR[c] for c in plot_df["channel"]]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(range(len(plot_df)), plot_df["rho"], color=colors, edgecolor="k", lw=0.4)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Spearman ρ with κ")
ax.set_title(f"Microphone broadband features vs κ ({len(records)} windows, no sub-bands)", fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=_COLOR[c], label=_LABEL[c]) for c in _COLOR],
          fontsize=8, loc="lower right")
fig.tight_layout()

out_path = EDA_DIR / "mic_feature_kappa_correlation.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {out_path}")

# %%
if __name__ == "__main__":
    print("\neda_mic_feature_kappa complete.")
