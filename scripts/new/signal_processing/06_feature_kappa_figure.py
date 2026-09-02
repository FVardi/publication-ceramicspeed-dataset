"""
06_feature_kappa_figure.py
==========================
Per-feature correlation with kappa, as a figure (replaces the appendix
ranking longtables).

For every candidate feature of both channels with |rho| >= PLOT_THRESHOLD,
plots the signed marginal Spearman rho with kappa, sorted by |rho| within
channel: AE (top panel) and US (bottom panel). Bar colour encodes the
frequency sub-band. The CSV output keeps all features, unfiltered.

Reads 12_fullset_decomposition's cond_vs_marginal_full.csv (marg_kappa column).

Outputs (outputs/new/feature_kappa_figure/)
  feature_kappa_correlation.png    paper figure (also copied to paper/figures/)
  feature_kappa_correlation.csv    tidy per-feature values as plotted

Usage
-----
    python scripts/new/signal_processing/06_feature_kappa_figure.py
"""

import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ceramicspeed.config import load_config, get_output_dir

cfg = load_config()
OUTPUT_DIR = get_output_dir(cfg)
NEW_DIR = OUTPUT_DIR / "new"
CM_PATH = NEW_DIR / "correlations" / "tables" / "cond_vs_marginal_full.csv"
SCRIPT_DIR = NEW_DIR / "feature_kappa_figure"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = Path(__file__).resolve().parents[3] / "paper" / "figures"

_BAND_ORDER = {"broadband": 0, "50-200kHz": 1, "200-600kHz": 2,
               "0-16kHz": 3, "16-40kHz": 4}
# Categorical band colours: fixed assignment, one hue per band (CVD-validated set)
#: AE bands: 50-200kHz/200-600kHz (config.yaml frequency_bands.AE, set after
#: the 2026-08-18 spectrogram inspection -- see config.yaml's AE comment).
#: UL bands: 0-16kHz/16-40kHz (config.yaml frequency_bands.UL, same pass).
_BAND_COLOR = {"broadband": "#2a78d6", "50-200kHz": "#1baf7a",
               "200-600kHz": "#eda100", "0-16kHz": "#008300",
               "16-40kHz": "#e34948"}
_BAND_LABEL = {"broadband": "Broadband", "50-200kHz": "50–200 kHz",
               "200-600kHz": "200–600 kHz", "0-16kHz": "0–16 kHz",
               "16-40kHz": "16–40 kHz"}
_INK = "#333333"
PLOT_THRESHOLD = 0.4  # only features with |rho(kappa)| >= this are drawn


def parse_feat(sensor, name):
    rest = name[len(sensor):] if name.startswith(sensor) else name
    left, _, stat = rest.partition("__")
    band = left[1:] if left.startswith("_") else "broadband"
    return (band or "broadband"), stat


# %%
cm = pd.read_csv(CM_PATH)
parsed = [parse_feat(s, f) for s, f in zip(cm["sensor"], cm["feature"])]
cm["band"] = [p[0] for p in parsed]
cm["stat"] = [p[1] for p in parsed]
cm["abs_kappa"] = cm["marg_kappa"].abs()

tidy = cm[["sensor", "band", "stat", "marg_kappa"]].copy()
tidy.to_csv(SCRIPT_DIR / "feature_kappa_correlation.csv", index=False)
print(f"Wrote {len(tidy)} rows -> feature_kappa_correlation.csv")


# %%
# ---- Figure: signed Spearman rho with kappa, sorted by |rho| per channel ----
plot_cm = cm[cm["abs_kappa"] >= PLOT_THRESHOLD]
n_dropped = len(cm) - len(plot_cm)
print(f"Plotting {len(plot_cm)} of {len(cm)} features "
      f"(|rho| >= {PLOT_THRESHOLD}; {n_dropped} below threshold omitted)")
panels = [("AE", plot_cm[plot_cm["sensor"] == "AE"]),
          ("US", plot_cm[plot_cm["sensor"] == "US"])]
n_rows = [len(s) for _, s in panels]

fig, axes = plt.subplots(
    2, 1, figsize=(7.0, 0.082 * sum(n_rows) + 1.3), sharex=True,
    gridspec_kw={"height_ratios": n_rows, "hspace": 0.18})

for ax, (sensor, s) in zip(axes, panels):
    s = s.sort_values("abs_kappa", ascending=True).reset_index(drop=True)
    labels = [f"{_BAND_LABEL[b]} · {st.replace('_', ' ')}"
              for b, st in zip(s["band"], s["stat"])]
    colors = [_BAND_COLOR[b] for b in s["band"]]

    ax.barh(range(len(s)), s["marg_kappa"], height=0.62, color=colors)
    ax.axvline(0, color=_INK, lw=0.8)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(labels, fontsize=5.8, color=_INK)
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.set_xlim(-1, 1)
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_INK)
    ax.tick_params(axis="both", labelsize=7, colors=_INK, length=2)
    ax.set_title(sensor, loc="left", fontsize=9, color=_INK, fontweight="bold")

    bands = sorted(s["band"].unique(), key=lambda b: _BAND_ORDER[b])
    ax.legend(handles=[Patch(facecolor=_BAND_COLOR[b], label=_BAND_LABEL[b])
                       for b in bands],
              loc="lower right", fontsize=6.5, frameon=False,
              labelcolor=_INK, handlelength=1.2, handleheight=1.0)

axes[1].set_xlabel("Spearman $\\rho$ with $\\kappa$", fontsize=8, color=_INK)

fig.savefig(SCRIPT_DIR / "feature_kappa_correlation.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
print("Saved: feature_kappa_correlation.png")

# Copy into the paper so it compiles standalone (same convention as
# scripts/copy_figures.py).
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(SCRIPT_DIR / "feature_kappa_correlation.png",
             PAPER_FIG_DIR / "feature_kappa_correlation.png")
print(f"Copied -> {PAPER_FIG_DIR / 'feature_kappa_correlation.png'}")

if __name__ == "__main__":
    print("\n06_feature_kappa_figure complete.")
