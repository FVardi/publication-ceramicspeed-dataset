"""
03_frequency_band_analysis.py
=============================
Identifies which frequency bands in the AE and Ultrasound signals
carry the most information about lubrication condition (kappa).

Approach
--------
1.  For every sweep in every HDF5 file, compute the Welch power spectral
    density (PSD) for both the AE and the Ultrasound channels.
2.  Partition the spectrum into fixed-width frequency bands.  For each band,
    compute the integrated band power (dB re RMS² / Hz).
3.  Compute kappa for every sweep from the lubricant / bearing metadata
    stored inside the HDF5 file.
4.  Analyse the relationship between each band's power and kappa via:
        - Spearman rank correlation (monotonic)
        - Mutual information        (non-linear / non-parametric)
5.  Confound analysis:  partial Spearman controlling for RPM & temperature.
6.  Visualise:
        - Mean PSD per sensor, overlaid by kappa regime
        - Bar charts of |Spearman ρ| and MI per band
        - Band-power heat-map vs sweep, sorted by kappa
        - Confound scatter / partial comparison plots
        - Combined ranking summary (dot-plot)

Pipeline position: 3rd script — reads raw HDF5 directly (PSD-level analysis).

Usage
-----
    python 03_frequency_band_analysis.py
    python 03_frequency_band_analysis.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr, rankdata
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from own_utils import calculate_kappa
from own_utils.config import (
    load_config,
    get_input_dir,
    get_output_dir,
    get_sensor_names,
    get_sensor_freq_limits,
    make_frequency_bands,
)
from own_utils.loading import discover_hdf5_files

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

INPUT_DIR = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
SENSOR_NAMES = get_sensor_names(cfg)
SENSOR_FLIMS = get_sensor_freq_limits(cfg)
BANDS = make_frequency_bands(cfg)

D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]
MI_RANDOM_STATE: int = cfg.get("random_state", 42)

# Welch PSD parameters
welch_cfg = cfg.get("welch", {})
WELCH_NPERSEG: int = welch_cfg.get("nperseg", 4096)
WELCH_NOVERLAP: int = welch_cfg.get("noverlap", 2048)
WELCH_WINDOW: str = welch_cfg.get("window", "hann")

# Kappa regime settings
kappa_cfg = cfg.get("kappa", {})
KAPPA_BOUNDARIES: list[float] = kappa_cfg.get("boundaries", [0.5, 1.0, 2.0])
KAPPA_LABELS: list[str] = kappa_cfg.get(
    "labels", ["κ < 0.5", "0.5 < κ < 1", "1 ≤ κ < 2", "κ ≥ 2"]
)
KAPPA_COLORS: list[str] = kappa_cfg.get(
    "colors", ["Blue", "#d62728", "#ff7f0e", "#2ca02c"]
)

# Band-power column names (shared order per sensor)
BAND_COLS: dict[str, list[str]] = {
    sensor: [label for _, _, label in BANDS[sensor]] for sensor in SENSOR_NAMES
}


# %%
# =============================================================================
# Helper functions
# =============================================================================


def band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Integrate PSD over [f_lo, f_hi] using the trapezoidal rule."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(psd[mask], freqs[mask]))


def kappa_regime(k: float) -> int:
    """Map a kappa value to a regime index."""
    for i, boundary in enumerate(KAPPA_BOUNDARIES):
        if k < boundary:
            return i
    return len(KAPPA_BOUNDARIES)


def partial_spearman(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray
) -> float:
    """Spearman partial correlation controlling for covariates."""
    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack(
        [rankdata(covariates[:, j]) for j in range(covariates.shape[1])]
    )
    rc_pinv = np.linalg.pinv(rc)
    res_x = rx - rc @ (rc_pinv @ rx)
    res_y = ry - rc @ (rc_pinv @ ry)
    num = np.sum(res_x * res_y)
    den = np.sqrt(np.sum(res_x**2) * np.sum(res_y**2))
    return float(num / den) if den > 0 else 0.0


def analyse_bands(
    df: pd.DataFrame, band_cols: list[str], kappa: pd.Series
) -> pd.DataFrame:
    """Return a DataFrame with Spearman ρ and MI for each band."""
    rows = []
    X = df[band_cols].values
    y = kappa.values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi_scores = mutual_info_regression(X, y, random_state=MI_RANDOM_STATE)

    for i, col in enumerate(band_cols):
        valid = ~np.isnan(X[:, i])
        spear, _ = spearmanr(X[valid, i], y[valid])
        rows.append({"band": col, "spearman": spear, "MI": mi_scores[i]})

    result = pd.DataFrame(rows).set_index("band")
    result["abs_spearman"] = result["spearman"].abs()
    for metric in ["abs_spearman", "MI"]:
        result[f"rank_{metric}"] = result[metric].rank(ascending=False)
    result["combined_rank"] = result[["rank_abs_spearman", "rank_MI"]].mean(axis=1)
    return result.sort_values("combined_rank")


# %%
# =============================================================================
# Data loading
# =============================================================================

FILE_PATTERNS: list[str] | None = cfg.get("filters", {}).get("file_patterns") or None
files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
if not files:
    raise FileNotFoundError(f"No matching HDF5 files found in {INPUT_DIR}")

print(f"Found {len(files)} HDF5 file(s).")
print(f"\nBand width: {cfg['band_width_hz'] / 1e3:.0f} kHz")
for sensor, bands in BANDS.items():
    print(f"  {sensor}: {len(bands)} bands  ({bands[0][2]} … {bands[-1][2]})")

# %%
# ---------------------------------------------------------------------------
# Main loading loop
# ---------------------------------------------------------------------------

records: list[dict] = []

for file_path in tqdm(files, desc="Loading files"):
    with h5py.File(file_path, "r") as f:
        lubricant_meta = dict(f["metadata"]["lubricant"].attrs)
        bearing_meta = dict(f["metadata"]["bearing"].attrs)
        sweeps = f["sweeps"]

        first_sweep = sweeps[list(sweeps.keys())[0]]
        time_axis = first_sweep["AE"]["time"][()]
        fs: float = 1.0 / float(np.mean(np.diff(time_axis)))

        for sweep_name, sweep in sweeps.items():
            test_params = dict(sweep.attrs)
            # Support both the old format (rpm, temperature_c) and the new
            # scope format (telem_rpm_meas, telem_omron_pv_c).
            rpm = float(
                test_params.get("rpm", test_params.get("telem_rpm_meas", 0))
            )
            temp_c = float(
                test_params.get("temperature_c", test_params.get("telem_omron_pv_c", 25))
            )

            if rpm >= RPM_MAX:
                continue

            # Keratech 22 (Kerax) viscosity fallback for files that omit it.
            nu_40 = float(lubricant_meta.get("viscosity_40c_cst", 22.0))
            nu_100 = float(lubricant_meta.get("viscosity_100c_cst", 4.1))

            kappa_val = calculate_kappa.calculate_kappa(
                rpm=rpm,
                temp_c=temp_c,
                d_pw=D_PW_MM,
                nu_40=nu_40,
                nu_100=nu_100,
            )

            for sensor_name in SENSOR_NAMES:
                if sensor_name not in sweep:
                    continue

                voltage: np.ndarray = sweep[sensor_name]["voltage"][()]

                freqs, psd = signal.welch(
                    voltage,
                    fs=fs,
                    window=WELCH_WINDOW,
                    nperseg=WELCH_NPERSEG,
                    noverlap=WELCH_NOVERLAP,
                )

                band_powers = {
                    label: band_power(freqs, psd, f_lo, f_hi)
                    for f_lo, f_hi, label in BANDS[sensor_name]
                }

                row = {
                    "file": file_path.stem,
                    "sweep": sweep_name,
                    "sensor": sensor_name,
                    "kappa": kappa_val,
                    "rpm": rpm,
                    "temp_c": temp_c,
                    "_freqs": freqs,
                    "_psd": psd,
                    **band_powers,
                }
                records.append(row)

print(f"\nLoaded {len(records)} (sweep × sensor) records.")

# %%
# =============================================================================
# Organise into per-sensor DataFrames
# =============================================================================

full_df = pd.DataFrame(records)

sensor_dfs: dict[str, pd.DataFrame] = {}
for sensor_name in SENSOR_NAMES:
    sdf = full_df[full_df["sensor"] == sensor_name].copy()
    sdf = sdf.dropna(
        subset=[label for _, _, label in BANDS[sensor_name]]
    )
    sensor_dfs[sensor_name] = sdf
    print(f"{sensor_name}: {len(sdf)} rows after dropping NaN bands")

# %%
# =============================================================================
# Correlation & MI analysis
# =============================================================================

print("\nRunning correlation & MI analysis…")
analysis: dict[str, pd.DataFrame] = {}
for sensor_name in SENSOR_NAMES:
    sdf = sensor_dfs[sensor_name]
    band_cols = BAND_COLS[sensor_name]
    kappa_vals = sdf["kappa"]
    analysis[sensor_name] = analyse_bands(sdf, band_cols, kappa_vals)
    print(f"\n{sensor_name} — top 5 bands by combined rank:")
    print(
        analysis[sensor_name]
        .head(5)[["spearman", "MI", "combined_rank"]]
        .to_string()
    )

# %%
# =============================================================================
# Confound analysis — RPM and temperature as predictors
# =============================================================================

print("\n--- Confound analysis: RPM & temperature ---")

confound_cols = ["rpm", "temp_c"]

for sensor_name in SENSOR_NAMES:
    sdf = sensor_dfs[sensor_name]
    kappa = sdf["kappa"].values
    print(f"\n{sensor_name}:")
    for cname in confound_cols:
        c = sdf[cname].values
        rho, _ = spearmanr(c, kappa)
        mi = float(
            mutual_info_regression(
                c.reshape(-1, 1), kappa, random_state=MI_RANDOM_STATE
            )[0]
        )
        print(f"  {cname:8s} vs κ  →  Spearman ρ = {rho:+.3f}   MI = {mi:.3f}")

confound_analysis: dict[str, pd.DataFrame] = {}

for sensor_name in SENSOR_NAMES:
    sdf = sensor_dfs[sensor_name]
    band_cols = BAND_COLS[sensor_name]
    kappa = sdf["kappa"].values
    covariates = sdf[confound_cols].values

    rows_c = []
    for col in band_cols:
        bp = sdf[col].values.astype(float)
        valid = ~np.isnan(bp)
        rho_rpm, _ = spearmanr(bp[valid], sdf["rpm"].values[valid])
        rho_temp, _ = spearmanr(bp[valid], sdf["temp_c"].values[valid])
        partial_rho = partial_spearman(bp[valid], kappa[valid], covariates[valid])
        rows_c.append(
            {
                "band": col,
                "spearman_rpm": rho_rpm,
                "spearman_temp": rho_temp,
                "partial_spearman_kappa": partial_rho,
            }
        )

    cdf = pd.DataFrame(rows_c).set_index("band")
    cdf["abs_partial_spearman"] = cdf["partial_spearman_kappa"].abs()
    confound_analysis[sensor_name] = cdf

    print(f"\n{sensor_name} — partial Spearman (band vs κ | RPM, temp), top 5:")
    print(
        cdf.sort_values("abs_partial_spearman", ascending=False)
        .head(5)[["spearman_rpm", "spearman_temp", "partial_spearman_kappa"]]
        .to_string()
    )

# %%
# =============================================================================
# Figure C1 — Band power vs κ coloured by RPM (top-3 bands per sensor)
# =============================================================================

for sensor_name in SENSOR_NAMES:
    sdf = sensor_dfs[sensor_name]
    df_an = analysis[sensor_name]
    top_bands = df_an.index[:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, band_label in zip(axes, top_bands):
        sc = ax.scatter(
            sdf["kappa"],
            sdf[band_label],
            c=sdf["rpm"],
            cmap="viridis",
            s=18,
            alpha=0.7,
            edgecolors="none",
        )
        ax.set_xlabel("κ")
        ax.set_ylabel(f"Band power  ({band_label})")
        ax.set_title(band_label, fontsize=10)
        ax.grid(ls=":", alpha=0.4)
    fig.colorbar(sc, ax=axes, label="RPM", shrink=0.8)
    fig.suptitle(f"{sensor_name} — Band power vs κ  (coloured by RPM)", fontsize=12)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / f"band_confound_rpm_{sensor_name.lower()}.png", dpi=150)
    plt.show()
    print(f"Saved: band_confound_rpm_{sensor_name.lower()}.png")

# %%
# =============================================================================
# Figure C2 — Band power vs κ coloured by temperature (top-3 bands)
# =============================================================================

for sensor_name in SENSOR_NAMES:
    sdf = sensor_dfs[sensor_name]
    df_an = analysis[sensor_name]
    top_bands = df_an.index[:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, band_label in zip(axes, top_bands):
        sc = ax.scatter(
            sdf["kappa"],
            sdf[band_label],
            c=sdf["temp_c"],
            cmap="magma",
            s=18,
            alpha=0.7,
            edgecolors="none",
        )
        ax.set_xlabel("κ")
        ax.set_ylabel(f"Band power  ({band_label})")
        ax.set_title(band_label, fontsize=10)
        ax.grid(ls=":", alpha=0.4)
    fig.colorbar(sc, ax=axes, label="Temperature (°C)", shrink=0.8)
    fig.suptitle(
        f"{sensor_name} — Band power vs κ  (coloured by temperature)", fontsize=12
    )
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / f"band_confound_temp_{sensor_name.lower()}.png", dpi=150)
    plt.show()
    print(f"Saved: band_confound_temp_{sensor_name.lower()}.png")

# %%
# =============================================================================
# Figure C3 — Partial Spearman vs raw Spearman (bar chart comparison)
# =============================================================================

fig, axes = plt.subplots(
    1,
    len(SENSOR_NAMES),
    figsize=(max(10, len(BAND_COLS[SENSOR_NAMES[0]]) * 0.7), 5),
)
if len(SENSOR_NAMES) == 1:
    axes = [axes]

for ax, sensor_name in zip(axes, SENSOR_NAMES):
    df_an = analysis[sensor_name]
    cdf = confound_analysis[sensor_name]

    band_order = df_an.index.tolist()
    raw_vals = df_an.loc[band_order, "abs_spearman"].values
    partial_vals = cdf.loc[band_order, "abs_partial_spearman"].values
    xpos = np.arange(len(band_order))
    w = 0.35

    ax.bar(xpos - w / 2, raw_vals, width=w, color="#1f77b4", label="|ρ| raw")
    ax.bar(
        xpos + w / 2,
        partial_vals,
        width=w,
        color="#e377c2",
        label="|ρ| partial (ctrl RPM+T)",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(band_order, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title(f"{sensor_name} — Raw vs partial Spearman with κ")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=0.5)

fig.suptitle(
    "Effect of controlling for RPM & temperature on band–κ correlation", fontsize=12
)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "band_confound_partial_comparison.png", dpi=150)
plt.show()
print("Saved: band_confound_partial_comparison.png")

# %%
# =============================================================================
# Figure 1 — Mean PSD per sensor coloured by kappa regime
# =============================================================================

fig, axes = plt.subplots(1, len(SENSOR_NAMES), figsize=(7 * len(SENSOR_NAMES), 5))
if len(SENSOR_NAMES) == 1:
    axes = [axes]

for ax, sensor_name in zip(axes, SENSOR_NAMES):
    sdf = sensor_dfs[sensor_name]
    f_min, f_max = SENSOR_FLIMS[sensor_name]
    n_regimes = len(KAPPA_LABELS)

    N_INTERP = 2048
    common_freq = np.linspace(f_min, f_max, N_INTERP)

    regime_psds: list[list[np.ndarray]] = [[] for _ in range(n_regimes)]

    for _, row in sdf.iterrows():
        regime = kappa_regime(row["kappa"])
        freqs = row["_freqs"]
        psd = row["_psd"]
        psd_interp = np.interp(common_freq, freqs, psd)
        regime_psds[regime].append(psd_interp)

    for i, (label, color) in enumerate(zip(KAPPA_LABELS, KAPPA_COLORS)):
        if not regime_psds[i]:
            continue
        stack = np.vstack(regime_psds[i])
        mean_psd = np.mean(stack, axis=0)
        std_psd = np.std(stack, axis=0)
        ax.semilogy(
            common_freq / 1e3,
            mean_psd,
            color=color,
            lw=1.8,
            label=f"{label}  (n={len(regime_psds[i])})",
        )
        ax.fill_between(
            common_freq / 1e3,
            np.maximum(mean_psd - std_psd, 1e-20),
            mean_psd + std_psd,
            color=color,
            alpha=0.15,
        )

    for f_lo, f_hi, _ in BANDS[sensor_name]:
        ax.axvline(f_lo / 1e3, color="grey", lw=0.4, alpha=0.4, ls="--")

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD  (V² / Hz)")
    ax.set_title(f"{sensor_name} — Mean PSD by κ regime")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)

fig.suptitle("Mean Power Spectral Density per Lubrication Regime", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "band_analysis_mean_psd.png", dpi=150)
plt.show()
print("Saved: band_analysis_mean_psd.png")

# %%
# =============================================================================
# Figure 2 — Correlation & MI per frequency band (bar charts)
# =============================================================================

n_metrics = 2
fig, axes = plt.subplots(
    n_metrics,
    len(SENSOR_NAMES),
    figsize=(
        max(10, len(BAND_COLS[SENSOR_NAMES[0]]) * 0.55) * len(SENSOR_NAMES),
        4 * n_metrics,
    ),
    sharey="row",
)
if len(SENSOR_NAMES) == 1:
    axes = axes.reshape(-1, 1)

metric_info = [
    ("abs_spearman", "|Spearman ρ|", "#1f77b4"),
    ("MI", "Mutual Information", "#2ca02c"),
]

for col_idx, sensor_name in enumerate(SENSOR_NAMES):
    df_an = analysis[sensor_name]
    band_order = df_an.index.tolist()

    for row_idx, (metric, ylabel, bar_color) in enumerate(metric_info):
        ax = axes[row_idx, col_idx]
        vals = df_an.loc[band_order, metric]
        xpos = np.arange(len(band_order))

        bars = ax.bar(xpos, vals, color=bar_color, edgecolor="white", lw=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(band_order, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{sensor_name} — {ylabel}", fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.5)

        top3_idx = vals.nlargest(3).index
        for bar, band_label in zip(bars, band_order):
            if band_label in top3_idx:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)

fig.suptitle("Frequency Band Relevance for κ (sorted by combined rank)", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "band_analysis_correlations.png", dpi=150)
plt.show()
print("Saved: band_analysis_correlations.png")

# %%
# =============================================================================
# Figure 3 — Band-power heat-map sorted by kappa
# =============================================================================

fig, axes = plt.subplots(1, len(SENSOR_NAMES), figsize=(12 * len(SENSOR_NAMES), 6))
if len(SENSOR_NAMES) == 1:
    axes = [axes]

for ax, sensor_name in zip(axes, SENSOR_NAMES):
    sdf = sensor_dfs[sensor_name]
    band_cols = BAND_COLS[sensor_name]

    sdf_sorted = sdf.sort_values("kappa")
    X_raw = sdf_sorted[band_cols].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_db = 10 * np.log10(np.maximum(X_raw, 1e-30))

    im = ax.imshow(
        X_db.T, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest"
    )
    plt.colorbar(im, ax=ax, label="Band power  (dB re V²·Hz)")

    ax.set_yticks(np.arange(len(band_cols)))
    ax.set_yticklabels(band_cols, fontsize=7)
    ax.set_xlabel("Sweeps (sorted by κ →)")
    ax.set_title(f"{sensor_name} — Band power heat-map (sorted by κ)")

    ax2 = ax.twinx()
    ax2.plot(
        np.arange(len(sdf_sorted)),
        sdf_sorted["kappa"].values,
        color="cyan",
        lw=1.2,
        alpha=0.9,
        label="κ",
    )
    ax2.set_ylabel("κ", color="cyan")
    ax2.tick_params(axis="y", colors="cyan")
    ax2.legend(loc="upper left", fontsize=8)

fig.suptitle("Band Power Heat-map vs. Lubrication Ratio κ", fontsize=13)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "band_analysis_heatmap.png", dpi=150)
plt.show()
print("Saved: band_analysis_heatmap.png")

# %%
# =============================================================================
# Figure 5 — Combined ranking summary (dot-plot)
# =============================================================================

fig, axes = plt.subplots(1, len(SENSOR_NAMES), figsize=(12, 5))
if len(SENSOR_NAMES) == 1:
    axes = [axes]

for ax, sensor_name in zip(axes, SENSOR_NAMES):
    df_an = analysis[sensor_name]
    df_plot = df_an[["abs_spearman", "MI"]].copy()

    for col in df_plot.columns:
        col_min, col_max = df_plot[col].min(), df_plot[col].max()
        df_plot[col] = (df_plot[col] - col_min) / (col_max - col_min + 1e-30)

    df_plot["mean_score"] = df_plot.mean(axis=1)
    df_plot = df_plot.sort_values("mean_score", ascending=True)

    metric_colors = {"abs_spearman": "#1f77b4", "MI": "#2ca02c"}
    metric_labels = {"abs_spearman": "|Spearman ρ|", "MI": "Mutual Info"}

    ypos = np.arange(len(df_plot))
    for metric, color in metric_colors.items():
        ax.scatter(
            df_plot[metric],
            ypos,
            color=color,
            label=metric_labels[metric],
            s=30,
            zorder=3,
            alpha=0.85,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(df_plot.index, fontsize=7)
    ax.set_xlabel("Normalised score  (0 = weakest, 1 = strongest)")
    ax.set_title(f"{sensor_name} — Band relevance summary")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8, alpha=0.6)

fig.suptitle(
    "Overall Band Relevance for Lubrication Ratio κ\n"
    "(sorted by mean normalised score — higher = more relevant)",
    fontsize=12,
)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "band_analysis_summary_ranking.png", dpi=150)
plt.show()
print("Saved: band_analysis_summary_ranking.png")

# %%
# =============================================================================
# Save numeric results to CSV
# =============================================================================

for sensor_name in SENSOR_NAMES:
    df_out = analysis[sensor_name].copy()
    cdf = confound_analysis[sensor_name]
    for col in [
        "spearman_rpm",
        "spearman_temp",
        "partial_spearman_kappa",
        "abs_partial_spearman",
    ]:
        df_out[col] = cdf[col].reindex(df_out.index)
    out_path = OUTPUT_DIR / f"band_analysis_{sensor_name.lower()}.csv"
    df_out.to_csv(out_path)
    print(f"Saved: {out_path.name}")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 70)
print("FREQUENCY BAND ANALYSIS — SUMMARY")
print("=" * 70)
for sensor_name in SENSOR_NAMES:
    df_an = analysis[sensor_name]
    cdf = confound_analysis[sensor_name]
    top_band = df_an.index[0]
    partial_rho = cdf.loc[top_band, "partial_spearman_kappa"]
    print(f"\n{sensor_name}:")
    print(f"  Most informative band (raw combined rank): {top_band}")
    print(f"    |Spearman ρ| (raw)     = {df_an.loc[top_band, 'abs_spearman']:.3f}")
    print(f"    MI                     = {df_an.loc[top_band, 'MI']:.3f}")
    print(f"    Spearman ρ (partial)   = {partial_rho:+.3f}  (ctrl RPM + temp)")
    print(f"    ρ with RPM             = {cdf.loc[top_band, 'spearman_rpm']:+.3f}")
    print(f"    ρ with temperature     = {cdf.loc[top_band, 'spearman_temp']:+.3f}")
print("\nAll plots and CSVs saved to:", OUTPUT_DIR)

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n03_frequency_band_analysis complete.")
