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

ae_df = ae_df[feature_selection["AE"]["retained"]]
ul_df = ul_df[feature_selection["UL"]["retained"]]

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
# Visualizations — shared setup
# =============================================================================

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

RANDOM_STATE: int = cfg.get("random_state", 42)

# Sensor bundles iterated in every plot cell below
_sensors = [
    ("AE", ae_df, ae_metadata),
    ("UL", ul_df, ul_metadata),
]


_kappa_cfg = cfg.get("kappa", {})
_kappa_bounds = _kappa_cfg.get("boundaries", [0.5, 1.0])
_kappa_labels = _kappa_cfg.get("labels", ["κ < 0.5", "0.5 ≤ κ < 1", "1 ≤ κ"])
_kappa_colors = _kappa_cfg.get("colors", ["#d62728", "#ff7f0e", "#2ca02c"])


def _scatter_regime(ax: plt.Axes, x: np.ndarray, y: np.ndarray, kappa: np.ndarray) -> None:
    """Scatter coloured by kappa regime (< 0.5 / 0.5–1 / ≥ 1) with a legend."""
    lo, hi = _kappa_bounds
    masks = [kappa < lo, (kappa >= lo) & (kappa < hi), kappa >= hi]
    for mask, label, color in zip(masks, _kappa_labels, _kappa_colors):
        if mask.any():
            ax.scatter(x[mask], y[mask], c=color, label=label,
                       s=10, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=8, markerscale=2)


# %%
# =============================================================================
# PCA
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (label, feat_df, meta) in zip(axes, _sensors):
    X_scaled = StandardScaler().fit_transform(feat_df.values)
    kappa = meta["kappa"].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    _scatter_regime(ax, X_pca[:, 0], X_pca[:, 1], kappa)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax.set_title(f"PCA — {label}")
    ax.grid(ls=":", alpha=0.4)

fig.suptitle("PCA", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_pca.png", dpi=150)
plt.show()
print("Saved: eda_pca.png")

# %%
# =============================================================================
# t-SNE
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (label, feat_df, meta) in zip(axes, _sensors):
    X_scaled = StandardScaler().fit_transform(feat_df.values)
    kappa = meta["kappa"].values
    perplexity = min(30, max(5, len(feat_df) // 5))
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE).fit_transform(X_scaled)
    _scatter_regime(ax, X_tsne[:, 0], X_tsne[:, 1], kappa)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE — {label}  (perplexity={perplexity})")
    ax.grid(ls=":", alpha=0.4)

fig.suptitle("t-SNE", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_tsne.png", dpi=150)
plt.show()
print("Saved: eda_tsne.png")

# %%
# =============================================================================
# UMAP
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (label, feat_df, meta) in zip(axes, _sensors):
    kappa = meta["kappa"].values
    if _UMAP_AVAILABLE:
        X_scaled = StandardScaler().fit_transform(feat_df.values)
        X_umap = umap_lib.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
        _scatter_regime(ax, X_umap[:, 0], X_umap[:, 1], kappa)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(ls=":", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "umap-learn not installed\npip install umap-learn",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title(f"UMAP — {label}")

fig.suptitle("UMAP", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_umap.png", dpi=150)
plt.show()
print("Saved: eda_umap.png")

# %%
# =============================================================================
# RadViz  (feature anchors sorted by hierarchical clustering of |correlation|)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (label, feat_df, meta) in zip(axes, _sensors):
    kappa = meta["kappa"].values
    feature_names = feat_df.columns.tolist()
    # Step 1 — quantile transform: map each feature to uniform [0, 1].
    # Step 2 — symmetric power stretch with exponent < 1, which creates a
    # U-shaped distribution (density ∝ (y−0.5)²): values are pushed toward
    # 0 and 1 so each sample is pulled strongly by a few anchors rather than
    # weakly by all, spreading the projection away from the centre.
    # Output stays in [0, 1] by construction — no rescaling needed.
    X_uniform = QuantileTransformer(
        output_distribution="uniform", random_state=RANDOM_STATE
    ).fit_transform(feat_df.values)
    X_norm = 0.5 + 0.5 * np.sign(X_uniform - 0.5) * np.abs(2 * X_uniform - 1) ** (1 / 3)

    # Sort features so that correlated neighbours are adjacent on the circle,
    # minimising visual crossing of the RadViz springs.
    if len(feature_names) > 2:
        corr_abs = np.abs(np.corrcoef(X_norm.T))
        np.fill_diagonal(corr_abs, 1.0)
        dist = np.clip(1.0 - corr_abs, 0.0, None)
        dist = (dist + dist.T) / 2          # enforce exact symmetry
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist), method="ward")
        order = leaves_list(Z)
    else:
        order = np.arange(len(feature_names))

    feature_names_sorted = [feature_names[i] for i in order]
    X_sorted = X_norm[:, order]

    # Project onto unit circle
    n_feat = len(feature_names_sorted)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False)
    anchors = np.column_stack([np.cos(angles), np.sin(angles)])
    weight_sum = X_sorted.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
    coords = (X_sorted @ anchors) / weight_sum

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.6, alpha=0.25)
    for i, name in enumerate(feature_names_sorted):
        ax.plot(*anchors[i], "k.", ms=5, alpha=0.7)
        ax.annotate(name, xy=anchors[i], xytext=1.13 * anchors[i],
                    ha="center", va="center", fontsize=7)
    _scatter_regime(ax, coords[:, 0], coords[:, 1], kappa)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.set_title(f"RadViz — {label}")
    ax.grid(ls=":", alpha=0.4)

fig.suptitle("RadViz  (anchors sorted by hierarchical clustering of |r|)", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_radviz.png", dpi=150)
plt.show()
print("Saved: eda_radviz.png")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\neda_features complete.")
