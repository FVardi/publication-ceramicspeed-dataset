"""
eda_features.py
===============
Exploratory data analysis of the extracted features from 01_feature_generation.py.

Pipeline position: optional EDA step — reads features.parquet + metadata.parquet
from 01_feature_generation.

Usage
-----
    python scripts/eda_features.py
    python scripts/eda_features.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import clean_features
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed — UMAP will be skipped (pip install umap-learn)")

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(exist_ok=True)
D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]

# %%
# =============================================================================
# Load features + metadata
# =============================================================================

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)
print(f"Loaded {len(raw_feature_df)} rows from features.parquet")
print(f"Sensors present: {raw_feature_df['sensor'].unique().tolist()}")


def _split_sensor(
    feat_df: pd.DataFrame, meta_df: pd.DataFrame, sensor: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = feat_df["sensor"] == sensor
    f = feat_df[mask].reset_index(drop=True)
    m = meta_df[mask].reset_index(drop=True)
    f = f.dropna(axis=1, how="all")   # drop columns that belong to other sensors
    return f, m


ae_raw, ae_meta_raw = _split_sensor(raw_feature_df, raw_metadata_df, "AE")
ul_raw, ul_meta_raw = _split_sensor(raw_feature_df, raw_metadata_df, "UL")

# %%
# =============================================================================
# Clean features
# =============================================================================

_clean_kwargs = dict(rpm_max=RPM_MAX, nan_strategy="drop", drop_constant=True)
ae_df, ae_metadata, ae_report = clean_features(ae_raw, ae_meta_raw, **_clean_kwargs)
ul_df, ul_metadata, ul_report = clean_features(ul_raw, ul_meta_raw, **_clean_kwargs)

ae_df = ae_df.set_index(["file", "sweep", "sensor"])
ul_df = ul_df.set_index(["file", "sweep", "sensor"])

for label, report in [("AE", ae_report), ("UL", ul_report)]:
    print(f"\n{label} after cleaning: {report['final_rows']} rows, {report['final_features']} features")
    if report.get("dropped_constant_features"):
        print(f"  dropped constant: {report['dropped_constant_features']}")

# %%
# =============================================================================
# Calculate kappa
# =============================================================================


def _add_kappa(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["kappa"] = metadata.apply(
        lambda row: calculate_kappa(
            rpm=row["rpm"],
            temp_c=row["temperature_c"],
            d_pw=D_PW_MM,
            nu_40=row["viscosity_40c_cst"],
            nu_100=row["viscosity_100c_cst"],
        ),
        axis=1,
    )
    return metadata


ae_metadata = _add_kappa(ae_metadata)
ul_metadata = _add_kappa(ul_metadata)

# %%
# =============================================================================
# Apply feature selection from 02_feature_analysis.py
# =============================================================================

feat_sel_path = OUTPUT_DIR / "feature_selection.json"
if not feat_sel_path.exists():
    raise FileNotFoundError(
        f"{feat_sel_path} not found. Run 02_feature_analysis.py first."
    )
with open(feat_sel_path) as fh:
    feature_selection = json.load(fh)

ae_df_all = ae_df  # preserve full feature set for plots that override selection
ul_df_all = ul_df

ae_df = ae_df[feature_selection["AE"]["retained"]]
ul_df = ul_df[feature_selection["UL"]["retained"]]

# _sensors is defined here so it always reflects the filtered DataFrames.
# Defining it in the visualizations cell would cause stale references if
# cells are re-run out of order.
_sensors = [
    ("AE", ae_df, ae_metadata),
    ("UL", ul_df, ul_metadata),
]

for label, feat_df, sel in [("AE", ae_df, feature_selection["AE"]), ("UL", ul_df, feature_selection["UL"])]:
    print(f"{label}: using {len(sel['retained'])} / {len(sel['all_columns'])} selected features: {sel['retained']}")

# %%
# =============================================================================
# Summary printout
# =============================================================================

for label, feat_df, meta in [("AE", ae_df, ae_metadata), ("UL", ul_df, ul_metadata)]:
    feat_cols = feat_df.columns.tolist()
    print(f"\n{'='*60}")
    print(f"Sensor: {label}")
    print(f"  Rows: {len(feat_df)}")
    print(f"  Features ({len(feat_cols)}): {feat_cols}")
    print(f"  Kappa: min={meta['kappa'].min():.3f}  max={meta['kappa'].max():.3f}"
          f"  mean={meta['kappa'].mean():.3f}  std={meta['kappa'].std():.3f}")
    print(f"  RPM:   min={meta['rpm'].min():.0f}  max={meta['rpm'].max():.0f}"
          f"  mean={meta['rpm'].mean():.0f}")
    print(f"  Temp:  min={meta['temperature_c'].min():.1f}°C  max={meta['temperature_c'].max():.1f}°C"
          f"  mean={meta['temperature_c'].mean():.1f}°C")

# %%
# =============================================================================
# Feature vs κ  (scatter coloured by RPM + binned mean)
# =============================================================================

HARDCODED_AE_FEATURES = [
    "AE_1000-2000kHz__complexity",
    "AE_500-1000kHz__complexity",
    "AE_500-1000kHz__frequency_weighted_std",
    "AE_500-1000kHz__dominant_frequency",
    "AE_500-1000kHz__spectral_std",
    "frequency_skewness",
]

_plot_sensors = [
    ("AE", ae_df_all[HARDCODED_AE_FEATURES], ae_metadata),
    ("UL", ul_df,                             ul_metadata),
]

for label, feat_df, meta in _plot_sensors:
    features = feat_df.columns.tolist()
    n_feat = len(features)
    rpm = meta["rpm"].values
    kappa = meta["kappa"].values

    ncols = min(n_feat, 2)
    nrows = math.ceil(n_feat / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    for ax, feat in zip(axes_flat, features):
        y = feat_df[feat].values
        sc = ax.scatter(kappa, y, c=rpm, cmap="plasma", s=10, alpha=0.5, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="RPM")

        bins = pd.cut(pd.Series(kappa), bins=12)
        trend = pd.Series(y).groupby(bins, observed=True).mean()
        bin_mids = [iv.mid for iv in trend.index]
        ax.plot(bin_mids, trend.values, "r-o", ms=4, lw=1.5, label="bin mean")

        ax.set_xlabel("κ")
        ax.set_ylabel(feat)
        ax.set_title(f"{label} — {feat}")
        ax.legend(fontsize=8)
        ax.grid(ls=":", alpha=0.4)

    fig.suptitle(f"Features vs κ — {label} sensor", fontsize=13)
    fig.tight_layout()
    fname = f"eda_feature_vs_kappa_{label.lower()}.png"
    plt.savefig(EDA_DIR / fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\neda_features complete.")
