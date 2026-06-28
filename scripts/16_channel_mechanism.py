"""
16_channel_mechanism.py
=======================
Mechanistic deepening of the AE/US operating-point encoding (Option 2).

Answers WHERE (which sub-bands) and WHAT KIND (which feature types) of each
channel's operating-point information lives, and ties it to RPM vs temperature.
Builds on 12_fullset_decomposition.py.

Part A -- coupling structure (descriptive, all sweeps):
  Reads cond_vs_marginal_full.csv, tags every feature with its sub-band and
  feature type, and aggregates the *conditional* |Spearman rho| with RPM and
  with temperature by (channel, band) and (channel, feature-type). The
  conditional correlation is used because the marginal one is dominated by the
  speed sweep and hides temperature coupling.

Part B -- SHAP on the operating-condition models:
  Fits LightGBM features->RPM and features->temperature per channel and
  aggregates mean |SHAP| by sub-band and feature type, showing which bands/types
  drive each channel's inference of speed vs temperature.

Outputs (outputs/16_channel_mechanism/)
  tables/coupling_by_band.csv, coupling_by_type.csv, shap_oc_aggregated.csv
  figures/coupling_heatmap.png, coupling_by_type.png, shap_oc_by_band.png

Usage
-----
    python scripts/16_channel_mechanism.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir

cfg = load_config()
OUTPUT_DIR = get_output_dir(cfg)
CM_PATH = OUTPUT_DIR / "12_fullset_decomposition" / "tables" / "cond_vs_marginal_full.csv"
SCRIPT_DIR = OUTPUT_DIR / "16_channel_mechanism"
TABLES_DIR = SCRIPT_DIR / "tables"
FIG_DIR = SCRIPT_DIR / "figures"
for d in (TABLES_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

D_PW = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RANDOM_STATE = cfg.get("random_state", 42)
DPI = 150

# --- feature-type taxonomy --------------------------------------------------
_TYPE = {
    "rms": "amplitude", "peak": "amplitude", "std": "amplitude", "variance": "amplitude",
    "crest_factor": "shape", "shape_factor": "shape", "impulse_factor": "shape",
    "margin_factor": "shape", "skewness": "shape", "kurtosis": "shape",
    "mobility": "complexity", "complexity": "complexity",
}
_SPECTRAL = {
    "dominant_frequency", "center_frequency", "rms_frequency", "peak_frequency",
    "spectral_mean", "spectral_std", "spectral_skewness", "spectral_kurtosis",
    "spectral_flatness", "spectral_bandwidth", "frequency_weighted_std",
    "normalized_frequency_std", "frequency_skewness", "frequency_kurtosis",
    "normalized_bandwidth",
}
for _s in _SPECTRAL:
    _TYPE[_s] = "spectral"

_BAND_ORDER = ["broadband", "20-500kHz", "500-1000kHz",
               "0-10kHz", "10-20kHz", "20-100kHz"]
_TYPE_ORDER = ["amplitude", "shape", "spectral", "complexity"]


def parse_feat(sensor: str, name: str) -> tuple[str, str]:
    """Return (band, feature_type) from a renamed feature column."""
    rest = name[len(sensor):] if name.startswith(sensor) else name
    left, _, stat = rest.partition("__")
    band = left[1:] if left.startswith("_") else "broadband"
    if band == "":
        band = "broadband"
    return band, _TYPE.get(stat, "other")


# %%
# =============================================================================
# Part A -- conditional coupling by band and by feature type
# =============================================================================
cm = pd.read_csv(CM_PATH)
cm["band"] = [parse_feat(s, f)[0] for s, f in zip(cm["sensor"], cm["feature"])]
cm["ftype"] = [parse_feat(s, f)[1] for s, f in zip(cm["sensor"], cm["feature"])]
cm["abs_cond_rpm"] = cm["cond_rpm"].abs()
cm["abs_cond_temp"] = cm["cond_temp"].abs()

by_band = (cm.groupby(["sensor", "band"])
           .agg(n=("feature", "size"),
                cond_rpm=("abs_cond_rpm", "mean"),
                cond_temp=("abs_cond_temp", "mean"))
           .reset_index())
by_type = (cm.groupby(["sensor", "ftype"])
           .agg(n=("feature", "size"),
                cond_rpm=("abs_cond_rpm", "mean"),
                cond_temp=("abs_cond_temp", "mean"))
           .reset_index())
by_band.to_csv(TABLES_DIR / "coupling_by_band.csv", index=False)
by_type.to_csv(TABLES_DIR / "coupling_by_type.csv", index=False)
print("Conditional |rho| coupling by (channel, band):")
print(by_band.to_string(index=False))

# Heatmap: rows = channel-band, cols = [RPM, Temp]
by_band["row"] = by_band["sensor"] + " · " + by_band["band"]
ordkey = {b: i for i, b in enumerate(_BAND_ORDER)}
by_band = by_band.sort_values(["sensor", "band"], key=lambda c: c.map(
    lambda v: ordkey.get(v, 99)) if c.name == "band" else c)
M = by_band[["cond_rpm", "cond_temp"]].values
fig, ax = plt.subplots(figsize=(5.2, 0.5 * len(by_band) + 1.5))
im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(0.6, M.max()), aspect="auto")
ax.set_xticks([0, 1]); ax.set_xticklabels(["|ρ| RPM", "|ρ| Temp"])
ax.set_yticks(range(len(by_band))); ax.set_yticklabels(by_band["row"])
for i in range(len(by_band)):
    for j, v in enumerate(M[i]):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.4 else "black", fontsize=8)
ax.set_title("Conditional coupling by channel × sub-band")
fig.colorbar(im, ax=ax, label="mean |conditional ρ|", shrink=0.7)
fig.tight_layout(); fig.savefig(FIG_DIR / "coupling_heatmap.png", dpi=DPI)
plt.close()
print("Saved: coupling_heatmap.png")

# Grouped bars by feature type
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, sensor in zip(axes, ["AE", "US"]):
    s = by_type[by_type["sensor"] == sensor].set_index("ftype").reindex(_TYPE_ORDER).dropna(how="all")
    x = np.arange(len(s))
    ax.bar(x - 0.2, s["cond_rpm"], 0.4, label="RPM", color="#1f4e79")
    ax.bar(x + 0.2, s["cond_temp"], 0.4, label="Temperature", color="#c55a11")
    ax.set_xticks(x); ax.set_xticklabels(s.index, rotation=15)
    ax.set_title(f"{sensor}"); ax.grid(axis="y", ls=":", alpha=0.5)
axes[0].set_ylabel("mean |conditional ρ|"); axes[0].legend()
fig.suptitle("Conditional coupling by feature type (RPM vs temperature)")
fig.tight_layout(); fig.savefig(FIG_DIR / "coupling_by_type.png", dpi=DPI)
plt.close()
print("Saved: coupling_by_type.png")


# %%
# =============================================================================
# Part B -- SHAP on features->RPM and features->temperature, per channel
# =============================================================================
import json

raw_feat, raw_meta = load_parquet_pair(OUTPUT_DIR)
with open(OUTPUT_DIR / "feature_selection.json") as fh:
    feature_selection = json.load(fh)
df, metadata = filter_by_metadata(raw_feat, raw_meta, rpm_max=RPM_MAX)
df = df.reset_index(drop=True); metadata = metadata.reset_index(drop=True)

_SENSOR_LABEL = {"UL": "US"}
shap_rows = []
for sensor, info in feature_selection.items():
    label = _SENSOR_LABEL.get(sensor, sensor)
    all_cols = info["all_columns"]
    m = df["sensor"] == sensor
    X = df.loc[m, all_cols].reset_index(drop=True)
    valid = X.notna().all(axis=1)
    X = X[valid].reset_index(drop=True)
    md = metadata.loc[m].reset_index(drop=True)[valid.values].reset_index(drop=True)

    for target_name, y in (("RPM", md["rpm"].values), ("Temp", md["temperature_c"].values)):
        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=63, random_state=RANDOM_STATE, verbose=-1)
        model.fit(X, y)
        sv = shap.TreeExplainer(model).shap_values(X)
        imp = np.abs(sv).mean(axis=0)  # mean |SHAP| per feature
        for col, val in zip(all_cols, imp):
            ren = col if col.startswith(f"{label}_") else f"{label}__{col}"
            band, ftype = parse_feat(label, ren)
            shap_rows.append({"channel": label, "target": target_name,
                              "feature": ren, "band": band, "ftype": ftype,
                              "mean_abs_shap": float(val)})
    print(f"{label}: SHAP computed for ->RPM and ->Temp ({len(all_cols)} features)")

sh = pd.DataFrame(shap_rows)
# normalise to fraction of total importance within each (channel, target)
sh["frac"] = sh.groupby(["channel", "target"])["mean_abs_shap"].transform(lambda s: s / s.sum())
sh.to_csv(TABLES_DIR / "shap_oc_aggregated.csv", index=False)

# Aggregate by band, plot per (channel, target)
agg = sh.groupby(["channel", "target", "band"])["frac"].sum().reset_index()
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for r, channel in enumerate(["AE", "US"]):
    for c, target in enumerate(["RPM", "Temp"]):
        ax = axes[r][c]
        s = agg[(agg.channel == channel) & (agg.target == target)]
        s = s.set_index("band").reindex([b for b in _BAND_ORDER if b in s["band"].values]).dropna()
        ax.bar(range(len(s)), s["frac"], color="#1f4e79" if target == "RPM" else "#c55a11")
        ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=20)
        ax.set_title(f"{channel} → {target}")
        ax.grid(axis="y", ls=":", alpha=0.5)
        if c == 0:
            ax.set_ylabel("fraction of SHAP importance")
fig.suptitle("Where each channel's speed/temperature information lives (SHAP by sub-band)")
fig.tight_layout(); fig.savefig(FIG_DIR / "shap_oc_by_band.png", dpi=DPI)
plt.close()
print("Saved: shap_oc_by_band.png")

# Console summary: dominant band/type for temperature per channel
print("\nTemperature-information concentration (SHAP fraction):")
for channel in ["AE", "US"]:
    s = sh[(sh.channel == channel) & (sh.target == "Temp")]
    top_band = s.groupby("band")["frac"].sum().sort_values(ascending=False)
    top_type = s.groupby("ftype")["frac"].sum().sort_values(ascending=False)
    print(f"  {channel}: top band {top_band.index[0]} ({top_band.iloc[0]:.2f}), "
          f"top type {top_type.index[0]} ({top_type.iloc[0]:.2f})")

if __name__ == "__main__":
    print("\n16_channel_mechanism complete.")
