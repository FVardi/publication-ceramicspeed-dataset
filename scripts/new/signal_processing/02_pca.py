"""
02_pca.py
=========
PCA of the full candidate feature set (signal-processing characterisation, not
modelling). Shows the low-dimensional structure of each channel's feature space
and how it separates by kappa / operating point.

Per channel (AE, US, Combined): standardise the full candidate features, PCA,
and plot PC1 vs PC2 coloured by kappa, plus a scree (explained-variance) panel.

Reads features from outputs/new/ (produced by 01_feature_generation.py).
Outputs (outputs/new/pca/): pca_scatter.png, pca_scree.png, pca_explained.csv

Usage
-----
    python scripts/new/signal_processing/02_pca.py
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata, true_candidate_columns
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir

_p = argparse.ArgumentParser()
_p.add_argument("--config", type=str, default=None)
args, _ = _p.parse_known_args()

cfg = load_config(args.config)
OUTPUT_DIR = get_output_dir(cfg)
NEW_DIR = OUTPUT_DIR / "new"
SCRIPT_DIR = NEW_DIR / "pca"
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

D_PW = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RPM_MIN = cfg["filters"].get("rpm_min", 0.0)
TEMP_MIN = cfg["filters"].get("temp_min", None)
RANDOM_STATE = cfg.get("random_state", 42)
DPI = 150
_SENSOR_LABEL = {"UL": "US"}

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
    """Return (X, kappa) for AE / US / Combined on the full candidate set."""
    if channel != "Combined":
        sensor = "AE" if channel == "AE" else "UL"
        label = _SENSOR_LABEL.get(sensor, sensor)
        cols = true_candidate_columns(raw_feat, raw_meta, sensor, RPM_MAX)
        m = df["sensor"] == sensor
        X = df.loc[m, cols].reset_index(drop=True)
        k = metadata.loc[m, "kappa"].reset_index(drop=True)
        valid = X.notna().all(axis=1)
        X = X[valid].reset_index(drop=True); k = k[valid].values
        X.columns = [_rename(c, label) for c in cols]
        return X, k
    parts = {}
    for sensor in ("AE", "UL"):
        label = _SENSOR_LABEL.get(sensor, sensor)
        cols = true_candidate_columns(raw_feat, raw_meta, sensor, RPM_MAX)
        m = df["sensor"] == sensor
        Xk = df.loc[m, ["file", "sweep"] + cols].copy()
        Xk["kappa"] = metadata.loc[m, "kappa"].values
        Xk = Xk[Xk[cols].notna().all(axis=1)].set_index(["file", "sweep"])
        Xk = Xk.rename(columns=lambda c: c if c == "kappa" else _rename(c, label))
        parts[sensor] = Xk
    ae, us = parts["AE"], parts["UL"]
    fae = [c for c in ae.columns if c != "kappa"]
    fus = [c for c in us.columns if c != "kappa"]
    merged = ae[fae + ["kappa"]].join(us[fus], how="inner")
    return merged.drop(columns=["kappa"]), merged["kappa"].values


# %%
channels = ["AE", "US", "Combined"]
pcas = {}
expl_rows = []
for ch in channels:
    X, k = _channel_matrix(ch)
    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=min(10, Xs.shape[1]), random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    pcas[ch] = (coords, k, pca, X.shape[1])
    for i, ev in enumerate(pca.explained_variance_ratio_, start=1):
        expl_rows.append({"channel": ch, "PC": i, "explained_var_ratio": round(float(ev), 4)})
    print(f"{ch}: {X.shape[1]} features, PC1+PC2 explain "
          f"{100*pca.explained_variance_ratio_[:2].sum():.1f}%")

pd.DataFrame(expl_rows).to_csv(SCRIPT_DIR / "pca_explained.csv", index=False)

# --- scatter: PC1 vs PC2 coloured by kappa ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, ch in zip(axes, channels):
    coords, k, pca, nf = pcas[ch]
    ev = pca.explained_variance_ratio_
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=k, cmap="viridis", s=8, alpha=0.6,
                    edgecolors="none")
    ax.set_xlabel(f"PC1 ({100*ev[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100*ev[1]:.0f}%)")
    ax.set_title(f"{ch} ({nf} features)")
    fig.colorbar(sc, ax=ax, label="κ", shrink=0.85)
fig.suptitle("PCA of the full feature set — coloured by κ", fontsize=13)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "pca_scatter.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved: pca_scatter.png")

# --- scree ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for ch in channels:
    ev = pcas[ch][2].explained_variance_ratio_
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev), marker="o", label=ch)
ax.set_xlabel("number of components"); ax.set_ylabel("cumulative explained variance")
ax.set_title("PCA scree (full feature set)")
ax.grid(ls=":", alpha=0.5); ax.legend()
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "pca_scree.png", dpi=DPI)
plt.close()
print(f"Saved: pca_scree.png -> {SCRIPT_DIR}")

if __name__ == "__main__":
    print("\n02_pca complete.")
