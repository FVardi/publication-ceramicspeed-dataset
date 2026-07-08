"""
19_clustered_shap.py
====================
Redundancy-aware (clustered) SHAP feature importance for the kappa regression,
on the FULL candidate feature set (no selection gate).

Why clustered: with the full set, near-duplicate features split SHAP credit, so
per-feature importances within a correlated family are diluted/unstable. We
therefore cluster features by |Spearman correlation| (merge when |rho| >= --corr),
sum mean|SHAP| within each cluster, and name the cluster by its single strongest
member. This recovers honest, nameable importance ("the amplitude cluster, led by
rms") without a separate VIF-selected feature set, and keeps genuinely distinct
features (e.g. mobility, complexity) as their own singletons.

Per channel (AE, US, Combined): fit LightGBM features->kappa, TreeSHAP, cluster,
plot top clusters.

Outputs (outputs/19_clustered_shap/)
  clustered_shap.png            top clusters per channel (bar = summed mean|SHAP|)
  clustered_shap_<channel>.csv  cluster membership + importances

Usage
-----
    python scripts/19_clustered_shap.py
    python scripts/19_clustered_shap.py --corr 0.85 --top 12
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata, true_candidate_columns
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir

_p = argparse.ArgumentParser()
_p.add_argument("--config", type=str, default=None)
_p.add_argument("--corr", type=float, default=0.8,
                help="Cluster features with |Spearman rho| >= this (default 0.8).")
_p.add_argument("--top", type=int, default=12, help="Top clusters to plot per channel.")
args, _ = _p.parse_known_args()

cfg = load_config(args.config)
OUTPUT_DIR = get_output_dir(cfg)
NEW_DIR = OUTPUT_DIR / "new"
SCRIPT_DIR = NEW_DIR / "clustered_shap"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
D_PW = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RPM_MIN = cfg["filters"].get("rpm_min", 0.0)  # drop startup/standstill transients
TEMP_MIN = cfg["filters"].get("temp_min", None)  # drop sub-floor cold-start
RANDOM_STATE = cfg.get("random_state", 42)
DPI = 150
_SENSOR_LABEL = {"UL": "US"}

_ABBR = {"mobility": "mob", "complexity": "cplx", "spectral_skewness": "sp.skew",
         "spectral_kurtosis": "sp.kurt", "spectral_bandwidth": "sp.bw", "skewness": "skew",
         "kurtosis": "kurt", "crest_factor": "crest", "dominant_frequency": "domfreq",
         "margin_factor": "margin", "shape_factor": "shape", "rms": "rms",
         "center_frequency": "cfreq", "spectral_flatness": "sp.flat"}


def _short(name):
    # keep the channel prefix so broadband AE/US features stay distinguishable
    ch = "AE" if name.startswith("AE") else ("US" if name.startswith("US") else "")
    n = name.replace("AE__", "").replace("US__", "").replace("AE_", "").replace("US_", "")
    band, _, stat = n.partition("__")
    if not stat:  # broadband: the leading token is the stat, no sub-band
        return f"{ch} {_ABBR.get(band, band)}".strip()
    band = (band.replace("500-1000kHz", "0.5-1M").replace("20-500kHz", "20-500k")
                .replace("20-100kHz", "20-100k").replace("10-20kHz", "10-20k")
                .replace("0-10kHz", "0-10k"))
    return f"{ch} {band} {_ABBR.get(stat, stat)}".strip()


# %%
raw_feat, raw_meta = load_parquet_pair(NEW_DIR)
df, metadata = filter_by_metadata(raw_feat, raw_meta, rpm_max=RPM_MAX,
                                  rpm_min=RPM_MIN, temp_min=TEMP_MIN)
df = df.reset_index(drop=True); metadata = metadata.reset_index(drop=True)
metadata["kappa"] = metadata.apply(
    lambda r: calculate_kappa(rpm=r["rpm"], temp_c=r["temperature_c"], d_pw=D_PW,
                              nu_40=r["viscosity_40c_cst"], nu_100=r["viscosity_100c_cst"]),
    axis=1)


def _rename(c, label):
    return c if c.startswith(f"{label}_") else f"{label}__{c}"


def _channel_matrix(channel):
    """Return (X, y) for AE / US / Combined on the full candidate set."""
    if channel != "Combined":
        sensor = "AE" if channel == "AE" else "UL"
        label = _SENSOR_LABEL.get(sensor, sensor)
        cols = true_candidate_columns(raw_feat, raw_meta, sensor, RPM_MAX)
        m = df["sensor"] == sensor
        X = df.loc[m, cols].reset_index(drop=True)
        y = metadata.loc[m, "kappa"].reset_index(drop=True)
        valid = X.notna().all(axis=1)
        X = X[valid].reset_index(drop=True); y = y[valid].values
        X.columns = [_rename(c, label) for c in cols]
        return X, y
    # Combined: join AE + US on (file, sweep)
    parts = {}
    for sensor in ("AE", "UL"):
        label = _SENSOR_LABEL.get(sensor, sensor)
        cols = true_candidate_columns(raw_feat, raw_meta, sensor, RPM_MAX)
        m = df["sensor"] == sensor
        Xk = df.loc[m, ["file", "sweep"] + cols].reset_index(drop=True)
        Xk["kappa"] = metadata.loc[m, "kappa"].values
        Xk = Xk[Xk[cols].notna().all(axis=1)].set_index(["file", "sweep"])
        Xk = Xk.rename(columns=lambda c: c if c == "kappa" else _rename(c, label))
        parts[sensor] = Xk
    ae, us = parts["AE"], parts["UL"]
    fae = [c for c in ae.columns if c != "kappa"]
    fus = [c for c in us.columns if c != "kappa"]
    merged = ae[fae + ["kappa"]].join(us[fus], how="inner")
    return merged.drop(columns=["kappa"]), merged["kappa"].values


def _clustered_shap(channel):
    X, y = _channel_matrix(channel)
    # drop zero-variance columns (corr undefined)
    X = X.loc[:, X.std() > 0]
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                              random_state=RANDOM_STATE, verbose=-1)
    model.fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

    # cluster by |Spearman corr|
    corr = X.corr(method="spearman").abs().fillna(0.0)
    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - args.corr, criterion="distance")

    rows = []
    for cl in np.unique(labels):
        members = X.columns[labels == cl]
        sub = imp[members].sort_values(ascending=False)
        rows.append({
            "channel": channel, "cluster": int(cl), "size": len(members),
            "total_shap": float(sub.sum()), "rep": sub.index[0],
            "members": ";".join(sub.index),
        })
    cdf = pd.DataFrame(rows).sort_values("total_shap", ascending=False)
    cdf.to_csv(SCRIPT_DIR / f"clustered_shap_{channel.lower()}.csv", index=False)
    return cdf


# %%
channels = ["AE", "US", "Combined"]
results = {ch: _clustered_shap(ch) for ch in channels}
for ch in channels:
    top = results[ch].head(3)
    print(f"{ch}: top clusters -> " + ", ".join(
        f"{_short(r.rep)}(+{r.size-1}, {r.total_shap:.3f})" for r in top.itertuples()))

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for ax, ch in zip(axes, channels):
    c = results[ch].head(args.top).iloc[::-1]
    labels = [f"{_short(r.rep)}" + (f"  (+{r.size-1})" if r.size > 1 else "")
              for r in c.itertuples()]
    colors = ["#1f4e79" if r.rep.startswith("AE") else "#c55a11" for r in c.itertuples()]
    ax.barh(range(len(c)), c["total_shap"], color=colors, edgecolor="k", lw=0.4)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("summed mean |SHAP| in cluster")
    ax.set_title(f"{ch}")
    ax.grid(axis="x", ls=":", alpha=0.5)
fig.suptitle(f"Clustered SHAP importance for κ (full feature set; clusters at |ρ|≥{args.corr}; "
             f"label = strongest member, +N redundant partners)", fontsize=12)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "clustered_shap.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: clustered_shap.png -> {SCRIPT_DIR}")

if __name__ == "__main__":
    print("\n19_clustered_shap complete.")
