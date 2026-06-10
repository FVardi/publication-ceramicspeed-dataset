"""
02_feature_analysis.py
======================
Analyse extracted features: correlation with kappa, redundancy detection,
PCA visualisation, and feature selection.

Pipeline position: 2nd script — reads features.parquet + metadata.parquet
from 01_feature_generation, writes analysis outputs.

Usage
-----
    python scripts/02_feature_analysis.py
    python scripts/02_feature_analysis.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import clean_features
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.analysis import (
    spearman_correlation,
    pearson_correlation,
    feature_ranking,
    correlation_matrix,
    variance_inflation_factors,
    identify_redundant_features,
    reduce_redundant_features,
    pca_transform,
)
from ceramicspeed.visualization import (
    plot_pca_kappa,
    plot_correlation_matrix,
    plot_vif,
)
from ceramicspeed.config import load_config, get_output_dir

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
SCRIPT_DIR = OUTPUT_DIR / "02_feature_analysis"
FIGURES_DIR = SCRIPT_DIR / "figures"
TABLES_DIR = SCRIPT_DIR / "tables"
for _d in [SCRIPT_DIR, FIGURES_DIR, TABLES_DIR]:
    _d.mkdir(exist_ok=True)

D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]

feat_sel_cfg = cfg.get("feature_selection", {})
CORR_MIN: float = feat_sel_cfg.get("corr_min", 0.1)
VIF_THRESHOLD: float = feat_sel_cfg.get("vif_threshold", 10.0)

# %%
# =============================================================================
# Load data and split by sensor before cleaning
# =============================================================================
# Band-specific columns exist for both sensors (e.g. AE_20-500kHz__* for AE,
# US_0-10kHz__* for UL).  Cleaning the combined DataFrame with nan_strategy="drop"
# would drop all rows of one sensor because they have NaN in the other sensor's
# band columns.  Split first so each sensor is cleaned against only its own columns.

raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)
print(f"Loaded {len(raw_feature_df)} rows from features.parquet")

def _split_sensor(feat_df, meta_df, sensor):
    mask = feat_df["sensor"] == sensor
    f = feat_df[mask].reset_index(drop=True)
    m = meta_df[mask].reset_index(drop=True)
    # Drop columns that are entirely NaN for this sensor (other sensors' bands)
    f = f.dropna(axis=1, how="all")
    return f, m

ae_raw, ae_meta_raw = _split_sensor(raw_feature_df, raw_metadata_df, "AE")
us_raw, us_meta_raw = _split_sensor(raw_feature_df, raw_metadata_df, "UL")

_clean_kwargs = dict(rpm_max=RPM_MAX, nan_strategy="drop", drop_constant=True)
ae_df, ae_metadata, ae_report = clean_features(ae_raw, ae_meta_raw, **_clean_kwargs)
us_df, us_metadata, us_report = clean_features(us_raw, us_meta_raw, **_clean_kwargs)

print(f"AE after cleaning: {ae_report['final_rows']} rows, {ae_report['final_features']} features")
print(f"UL after cleaning: {us_report['final_rows']} rows, {us_report['final_features']} features")
for label, report in [("AE", ae_report), ("UL", us_report)]:
    if report.get("dropped_constant_features"):
        print(f"  {label} dropped constant: {report['dropped_constant_features']}")

ae_df = ae_df.set_index(["file", "sweep", "sensor"])
us_df = us_df.set_index(["file", "sweep", "sensor"])

# %%
# =============================================================================
# Calculate kappa
# =============================================================================

def _calc_kappa(metadata: "pd.DataFrame") -> "pd.Series":
    return metadata.apply(
        lambda row: calculate_kappa(
            rpm=row["rpm"],
            temp_c=row["temperature_c"],
            d_pw=D_PW_MM,
            nu_40=row["viscosity_40c_cst"],
            nu_100=row["viscosity_100c_cst"],
        ),
        axis=1,
    )

ae_target = _calc_kappa(ae_metadata)
us_target = _calc_kappa(us_metadata)

# %%
# =============================================================================
# Correlation analysis: Spearman + Pearson
# =============================================================================

ae_spearman = spearman_correlation(ae_df, ae_target)
us_spearman = spearman_correlation(us_df, us_target)

ae_pearson = pearson_correlation(ae_df, ae_target)
us_pearson = pearson_correlation(us_df, us_target)

# %%
# =============================================================================
# Threshold filter — both |ρ| ≥ CORR_MIN and |r| ≥ CORR_MIN required
# (configured via feature_selection.corr_min in config.yaml)
# =============================================================================

ae_keep = (ae_spearman["rho"].abs() >= CORR_MIN) & (ae_pearson["r"].abs() >= CORR_MIN)
us_keep = (us_spearman["rho"].abs() >= CORR_MIN) & (us_pearson["r"].abs() >= CORR_MIN)

ae_df       = ae_df[ae_keep[ae_keep].index]
us_df       = us_df[us_keep[us_keep].index]
ae_spearman = ae_spearman.loc[ae_keep[ae_keep].index]
ae_pearson  = ae_pearson.loc[ae_keep[ae_keep].index]
us_spearman = us_spearman.loc[us_keep[us_keep].index]
us_pearson  = us_pearson.loc[us_keep[us_keep].index]

print(f"AE: {ae_keep.sum()} / {len(ae_keep)} features pass |ρ| ≥ {CORR_MIN} and |r| ≥ {CORR_MIN}")
print(f"UL: {us_keep.sum()} / {len(us_keep)} features pass |ρ| ≥ {CORR_MIN} and |r| ≥ {CORR_MIN}")

# %%
# =============================================================================
# Feature ranking by combined |ρ| + |r|
# =============================================================================

ae_ranking = feature_ranking(ae_spearman, ae_pearson)
us_ranking = feature_ranking(us_spearman, us_pearson)

# %%
print("AE — top 10 features by combined rank:")
print(ae_ranking.head(10).to_string())
# %%
print("\nUltrasound — top 10 features by combined rank:")
print(us_ranking.head(10).to_string())

# %%
# =============================================================================
# Feature correlation bar plot — Spearman ρ and Pearson r, sorted by combined rank
# =============================================================================

import numpy as np

for sensor_label, ranking, spearman, pearson in [
    ("AE", ae_ranking, ae_spearman, ae_pearson),
    ("UL", us_ranking, us_spearman, us_pearson),
]:
    feature_order = ranking.sort_values("rank", ascending=False).index.tolist()
    rho = spearman["rho"].reindex(feature_order).values
    r   = pearson["r"].reindex(feature_order).values

    y = np.arange(len(feature_order))
    h = 0.35

    fig_r, ax_r = plt.subplots(figsize=(8, max(4, 0.4 * len(feature_order))))
    ax_r.barh(y + h / 2, rho, h, label="Spearman ρ", color="C0", alpha=0.85)
    ax_r.barh(y - h / 2, r,   h, label="Pearson r",  color="C1", alpha=0.85)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(feature_order)
    ax_r.axvline(0, color="k", lw=0.8)
    ax_r.set_xlabel("Correlation with κ")
    ax_r.set_title(f"{sensor_label} — feature correlation ranking")
    ax_r.legend(loc="lower right")
    ax_r.grid(ls=":", axis="x", alpha=0.4)
    fig_r.tight_layout()
    _fname = f"feature_ranking_{'ae' if sensor_label == 'AE' else 'us'}_barplot.png"
    plt.savefig(FIGURES_DIR / _fname, dpi=150)
    plt.show()
    print(f"Saved: {_fname}")

# %%
# =============================================================================
# PCA visualisation coloured by kappa regime
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ae_pca_coords, ae_pca, _ = pca_transform(ae_df)
us_pca_coords, us_pca, _ = pca_transform(us_df)

plot_pca_kappa(ae_pca_coords, ae_target, ae_pca,
               ax=axes[0], title="AE — PCA colored by κ")
plot_pca_kappa(us_pca_coords, us_target, us_pca,
               ax=axes[1], title="Ultrasound — PCA colored by κ")

fig.tight_layout()
plt.savefig(FIGURES_DIR / "pca_kappa_regimes.png", dpi=600)
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
plt.savefig(FIGURES_DIR / "inter_feature_correlation.png", dpi=150)
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
plt.savefig(FIGURES_DIR / "vif_barplot.png", dpi=150)
plt.show()
print("Saved: vif_barplot.png")

# %%
# =============================================================================
# Redundancy summary — flag features by all three criteria
# =============================================================================

ae_redundancy = identify_redundant_features(ae_corr_mat, ae_vif, vif_threshold=VIF_THRESHOLD)
us_redundancy = identify_redundant_features(us_corr_mat, us_vif, vif_threshold=VIF_THRESHOLD)

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

ae_retained = reduce_redundant_features(ae_df, ae_target, ae_corr_mat, ae_vif, vif_threshold=VIF_THRESHOLD)
us_retained = reduce_redundant_features(us_df, us_target, us_corr_mat, us_vif, vif_threshold=VIF_THRESHOLD)

# %%
print(f"\nAE: {len(ae_df.columns)} → {len(ae_retained)} features retained")
print("Retained:", ae_retained)
print(f"\nUS: {len(us_df.columns)} → {len(us_retained)} features retained")
print("Retained:", us_retained)

# %%
# =============================================================================
# Trimmed bar plot for paper — top 20 AE features only, retained highlighted
# =============================================================================

TOP_N_AE = 20

ae_top20_order = ae_ranking.head(TOP_N_AE).index.tolist()[::-1]  # best at top
rho_top = ae_spearman["rho"].abs().reindex(ae_top20_order).values
r_top   = ae_pearson["r"].abs().reindex(ae_top20_order).values
y20 = np.arange(TOP_N_AE)
h   = 0.35

fig20, ax20 = plt.subplots(figsize=(8, 0.38 * TOP_N_AE + 1))
for i, feat in enumerate(ae_top20_order):
    ec = "k" if feat in ae_retained else "none"
    lw = 0.9 if feat in ae_retained else 0
    ax20.barh(i + h / 2, rho_top[i], h, color="C0", alpha=0.85, edgecolor=ec, linewidth=lw)
    ax20.barh(i - h / 2, r_top[i],   h, color="C1", alpha=0.85, edgecolor=ec, linewidth=lw)

ax20.set_yticks(y20)
ax20.set_yticklabels(ae_top20_order, fontsize=8)
ax20.axvline(0, color="k", lw=0.8)
ax20.set_xlabel("|Correlation with κ|")
ax20.set_title(f"AE — top {TOP_N_AE} features by combined rank (retained outlined)")
ax20.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.85, label="|Spearman ρ|"),
        plt.Rectangle((0, 0), 1, 1, color="C1", alpha=0.85, label="|Pearson r|"),
        plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.5, ec="k", lw=0.9, label="Retained"),
    ],
    loc="lower right",
    fontsize=8,
)
ax20.grid(ls=":", axis="x", alpha=0.4)
fig20.tight_layout()
plt.savefig(FIGURES_DIR / "feature_ranking_ae_barplot_top20.png", dpi=150)
plt.close()
print("Saved: feature_ranking_ae_barplot_top20.png")

# %%
# =============================================================================
# LaTeX longtable for appendix — full AE and UL rankings
# =============================================================================

def _parse_col(col: str) -> tuple[str, str]:
    """Return (sub_band_label, feature_name) from a column name string."""
    if "__" in col:
        prefix, feat = col.split("__", 1)
        parts = prefix.split("_", 1)
        band = parts[1] if len(parts) > 1 else ""
    else:
        band, feat = "", col
    if not band:
        band_label = "Broadband"
    else:
        band_label = (
            band.replace("20-500kHz", "20--500\\,kHz")
                .replace("500kHz-1MHz", "500\\,kHz--1\\,MHz")
                .replace("1-2MHz", "1--2\\,MHz")
                .replace("-", "--")
        )
    feat_label = feat.replace("_", " ")
    return band_label, feat_label


def _ranking_to_latex(ranking, spearman, pearson, retained, sensor, label):
    lines = [
        f"% Auto-generated by 02_feature_analysis.py — {sensor} feature ranking",
        r"\begin{longtable}{@{} r l l r r c @{}}",
        r"\caption{Full " + sensor + r" feature correlation ranking with $\kappa$. "
        r"$|\rho|$: absolute Spearman correlation; $|r|$: absolute Pearson correlation. "
        r"Retained: selected after Stage~2 redundancy reduction.}",
        r"\label{tab:app_" + label + r"_ranking} \\",
        r"\toprule",
        r"Rank & Sub-band & Feature & {$|\rho|$} & {$|r|$} & Ret. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{6}{l}{\small\itshape (continued)} \\",
        r"\toprule",
        r"Rank & Sub-band & Feature & {$|\rho|$} & {$|r|$} & Ret. \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, row in ranking.iterrows():
        feat = row.name
        band_label, feat_label = _parse_col(feat)
        rho_val = abs(spearman.loc[feat, "rho"]) if feat in spearman.index else float("nan")
        r_val   = abs(pearson.loc[feat, "r"])    if feat in pearson.index   else float("nan")
        ret = r"\checkmark" if feat in retained else "--"
        rank = int(row["rank"])
        lines.append(
            f"{rank} & {band_label} & {feat_label} & {rho_val:.3f} & {r_val:.3f} & {ret} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


ae_table_tex = _ranking_to_latex(ae_ranking, ae_spearman, ae_pearson, ae_retained, "AE", "ae")
us_table_tex = _ranking_to_latex(us_ranking, us_spearman, us_pearson, us_retained, "US", "us")

(TABLES_DIR / "feature_ranking_ae_appendix.tex").write_text(ae_table_tex, encoding="utf-8")
(TABLES_DIR / "feature_ranking_us_appendix.tex").write_text(us_table_tex, encoding="utf-8")
print("Saved: feature_ranking_ae_appendix.tex, feature_ranking_us_appendix.tex")

# %%
# =============================================================================
# Save retained feature lists + rankings for downstream scripts
# =============================================================================

feature_selection_output = {
    "AE": {
        "retained": ae_retained,
        "all_columns": ae_df.columns.tolist(),
    },
    "UL": {
        "retained": us_retained,
        "all_columns": us_df.columns.tolist(),
    },
}
out_path = OUTPUT_DIR / "feature_selection.json"
with open(out_path, "w") as fh:
    json.dump(feature_selection_output, fh, indent=2)
print(f"Saved: {out_path.name}")

ae_ranking.to_csv(TABLES_DIR / "feature_ranking_ae.csv")
us_ranking.to_csv(TABLES_DIR / "feature_ranking_us.csv")
print("Saved: feature_ranking_ae.csv, feature_ranking_us.csv")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n02_feature_analysis complete.")

