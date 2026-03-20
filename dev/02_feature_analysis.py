"""
02_feature_analysis.py
======================
Analyse extracted features: correlation with kappa, redundancy detection,
PCA visualisation, and feature selection.

Pipeline position: 2nd script — reads features.parquet + metadata.parquet
from 01_feature_generation, writes analysis outputs.

Usage
-----
    python 02_feature_analysis.py
    python 02_feature_analysis.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from own_utils.loading import load_parquet_pair
from own_utils.cleaning import clean_features
from own_utils.calculate_kappa import calculate_kappa
from own_utils.analysis import (
    spearman_correlation,
    feature_ranking,
    correlation_matrix,
    variance_inflation_factors,
    identify_redundant_features,
    reduce_redundant_features,
    pca_transform,
)
from own_utils.visualization import (
    plot_pca_kappa,
    plot_correlation_matrix,
    plot_vif,
)
from own_utils.config import load_config, get_output_dir

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
D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]

# %%
# =============================================================================
# Load and clean data
# =============================================================================

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)
print(f"Loaded {len(raw_feature_df)} rows from features.parquet")

df, metadata, cleaning_report = clean_features(
    raw_feature_df,
    raw_metadata_df,
    rpm_max=RPM_MAX,
    nan_strategy="drop",
    drop_constant=True,
)
print(f"After cleaning: {cleaning_report['final_rows']} rows, "
      f"{cleaning_report['final_features']} features")
if cleaning_report.get("dropped_constant_features"):
    print(f"  Dropped constant features: {cleaning_report['dropped_constant_features']}")

df = df.set_index(["file", "sweep", "sensor"])

# %%
# =============================================================================
# Calculate kappa
# =============================================================================

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

# %%
# =============================================================================
# Split by sensor
# =============================================================================

ae_mask = df.index.get_level_values("sensor") == "AE"
us_mask = df.index.get_level_values("sensor") == "Ultrasound"

ae_df = df[ae_mask]
us_df = df[us_mask]
ae_kappa = metadata["kappa"][ae_mask]
us_kappa = metadata["kappa"][us_mask]

# %%
# =============================================================================
# Correlation analysis: Spearman
# =============================================================================

ae_spearman = spearman_correlation(ae_df, ae_kappa)
us_spearman = spearman_correlation(us_df, us_kappa)

# %%
# =============================================================================
# Feature ranking by |ρ|
# =============================================================================

ae_ranking = feature_ranking(ae_spearman)
us_ranking = feature_ranking(us_spearman)

# %%
print("AE — top 10 features by combined rank:")
print(ae_ranking.head(10).to_string())
# %%
print("\nUltrasound — top 10 features by combined rank:")
print(us_ranking.head(10).to_string())

# %%
# =============================================================================
# Feature correlation bar plot — ranked by |ρ| per sensor
# =============================================================================

import numpy as np

for sensor_label, spearman in [("AE", ae_spearman), ("Ultrasound (US)", us_spearman)]:
    _rho = spearman["rho"].sort_values(key=np.abs, ascending=True)  # ascending for horizontal bar
    _colors = ["C0" if v >= 0 else "C3" for v in _rho]

    fig_r, ax_r = plt.subplots(figsize=(8, max(4, 0.35 * len(_rho))))
    ax_r.barh(_rho.index, _rho.values, color=_colors)
    ax_r.axvline(0, color="k", lw=0.8)
    ax_r.set_xlabel("Spearman ρ with κ")
    ax_r.set_title(f"{sensor_label} — feature correlation ranking")
    ax_r.grid(ls=":", axis="x", alpha=0.4)
    fig_r.tight_layout()
    _fname = f"feature_ranking_{'ae' if 'AE' in sensor_label else 'us'}_barplot.png"
    plt.savefig(OUTPUT_DIR / _fname, dpi=150)
    plt.show()
    print(f"Saved: {_fname}")

# %%
# =============================================================================
# PCA visualisation coloured by kappa regime
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ae_pca_coords, ae_pca, _ = pca_transform(ae_df)
us_pca_coords, us_pca, _ = pca_transform(us_df)

plot_pca_kappa(ae_pca_coords, ae_kappa, ae_pca,
               ax=axes[0], title="AE — PCA colored by κ regime")
plot_pca_kappa(us_pca_coords, us_kappa, us_pca,
               ax=axes[1], title="Ultrasound — PCA colored by κ regime")

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_kappa_regimes.png", dpi=150)
plt.show()
print("Saved: pca_kappa_regimes.png")

# %%
# =============================================================================
# Redundancy analysis — inter-feature correlation matrix
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

_, _, ae_corr_mat = plot_correlation_matrix(
    ae_df, method="spearman", ax=axes[0], title="AE — Spearman inter-feature |ρ|"
)
_, _, us_corr_mat = plot_correlation_matrix(
    us_df, method="spearman", ax=axes[1], title="US — Spearman inter-feature |ρ|"
)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "inter_feature_correlation.png", dpi=150)
plt.show()
print("Saved: inter_feature_correlation.png")

# %%
# =============================================================================
# Variance Inflation Factor
# =============================================================================

ae_vif = variance_inflation_factors(ae_df)
us_vif = variance_inflation_factors(us_df)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_vif(ae_vif, ax=axes[0], title="AE — Variance Inflation Factor")
plot_vif(us_vif, ax=axes[1], title="US — Variance Inflation Factor")
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "vif_barplot.png", dpi=150)
plt.show()
print("Saved: vif_barplot.png")

# %%
# =============================================================================
# Redundancy summary — flag features by all three criteria
# =============================================================================

ae_redundancy = identify_redundant_features(ae_corr_mat, ae_vif)
us_redundancy = identify_redundant_features(us_corr_mat, us_vif)

# %%
print("AE redundancy flags:")
print(ae_redundancy.to_string())
# %%
print("\nUS redundancy flags:")
print(us_redundancy.to_string())

# %%
# =============================================================================
# Greedy redundancy reduction — retained feature subsets
# =============================================================================

ae_retained = reduce_redundant_features(ae_df, ae_kappa, ae_corr_mat, ae_vif)
us_retained = reduce_redundant_features(us_df, us_kappa, us_corr_mat, us_vif)

# %%
print(f"\nAE: {len(ae_df.columns)} → {len(ae_retained)} features retained")
print("Retained:", ae_retained)
print(f"\nUS: {len(us_df.columns)} → {len(us_retained)} features retained")
print("Retained:", us_retained)

# %%
# =============================================================================
# Save retained feature lists + rankings for downstream scripts
# =============================================================================

feature_selection_output = {
    "AE": {
        "retained": ae_retained,
        "all_columns": ae_df.columns.tolist(),
    },
    "Ultrasound": {
        "retained": us_retained,
        "all_columns": us_df.columns.tolist(),
    },
}
out_path = OUTPUT_DIR / "feature_selection.json"
with open(out_path, "w") as fh:
    json.dump(feature_selection_output, fh, indent=2)
print(f"Saved: {out_path.name}")

ae_ranking.to_csv(OUTPUT_DIR / "feature_ranking_ae.csv")
us_ranking.to_csv(OUTPUT_DIR / "feature_ranking_us.csv")
print("Saved: feature_ranking_ae.csv, feature_ranking_us.csv")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n02_feature_analysis complete.")

