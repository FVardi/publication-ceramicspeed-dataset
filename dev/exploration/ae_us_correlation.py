"""
ae_us_correlation.py
====================
Correlation analysis between raw AE and Ultrasound time-series signals
recorded simultaneously under the same operating conditions (same file /
sweep pair).

Analyses
--------
1. Per-sweep correlation  — raw voltage and Hilbert amplitude envelope of AE
   and US are interpolated onto a common time grid; Pearson and Spearman r
   are computed for both.
2. Sweep-level RMS correlation  — scatter of AE RMS vs US RMS across all
   sweeps, coloured by operating condition (RPM / kappa).
3. Correlation vs operating conditions — how raw and envelope r changes with
   RPM, temperature, and kappa.

Usage
-----
    python exploration/ae_us_correlation.py
    python exploration/ae_us_correlation.py --config alt.yaml
    python exploration/ae_us_correlation.py --max-files 10   # quick test
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from own_utils.config import load_config, get_input_dir, get_output_dir
from own_utils.calculate_kappa import calculate_kappa
from own_utils.loading import discover_hdf5_files

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of HDF5 files (for quick testing)",
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

INPUT_DIR = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]
SENSORS = ("AE", "Ultrasound")

# %%
# =============================================================================
# Raw signal loader (loads voltage + time axis for both sensors per sweep)
# =============================================================================


def load_sweep_signals(
    file_path: Path,
    sensors: tuple[str, ...] = SENSORS,
) -> list[dict]:
    """Load voltage and time arrays for every sweep in an HDF5 file.

    Returns
    -------
    list of dict, one per sweep, with keys:
        ``file``, ``sweep``, ``test_parameters``,
        ``lubricant_metadata``, ``bearing_metadata``,
        and for each sensor: ``{sensor: {"t": np.ndarray, "v": np.ndarray}}``.
    """
    records = []
    with h5py.File(file_path, "r") as f:
        lubricant_meta = dict(f["metadata"]["lubricant"].attrs)
        bearing_meta = dict(f["metadata"]["bearing"].attrs)

        for sweep_name, sweep_grp in f["sweeps"].items():
            test_params = dict(sweep_grp.attrs)
            rec: dict = {
                "file": file_path.stem,
                "sweep": sweep_name,
                "test_parameters": test_params,
                "lubricant_metadata": lubricant_meta,
                "bearing_metadata": bearing_meta,
            }
            for sensor in sensors:
                if sensor not in sweep_grp:
                    continue
                rec[sensor] = {
                    "t": sweep_grp[sensor]["time"][()],
                    "v": sweep_grp[sensor]["voltage"][()],
                }
            records.append(rec)
    return records


# %%
# =============================================================================
# Per-sweep correlation helpers
# =============================================================================


def _envelope(v: np.ndarray) -> np.ndarray:
    """Hilbert amplitude envelope of a real-valued signal."""
    return np.abs(hilbert(v - v.mean()))


def _common_time_grid(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """Return a uniform time grid covering the overlap of two time axes."""
    t_start = max(t_a[0], t_b[0])
    t_end = min(t_a[-1], t_b[-1])
    # Use the coarser sampling rate to avoid unnecessary up-sampling
    dt = max(float(np.mean(np.diff(t_a))), float(np.mean(np.diff(t_b))))
    return np.arange(t_start, t_end, dt)


def analyse_sweep(
    t_ae: np.ndarray,
    v_ae: np.ndarray,
    t_us: np.ndarray,
    v_us: np.ndarray,
) -> dict:
    """Compute correlation metrics between AE and US for one sweep.

    Computes both raw-signal and envelope-based metrics on a common time grid.

    Returns
    -------
    dict with keys:
        ``ae_rms``, ``us_rms``,
        ``raw_pearson_r``, ``raw_pearson_p``, ``raw_spearman_r``,
        ``env_pearson_r``, ``env_pearson_p``, ``env_spearman_r``,
        ``n_samples_common``.
    """
    t_grid = _common_time_grid(t_ae, t_us)

    # --- Interpolate raw signals onto common grid ---
    v_ae_i = np.interp(t_grid, t_ae, v_ae)
    v_us_i = np.interp(t_grid, t_us, v_us)

    raw_pearson_r, raw_pearson_p = pearsonr(v_ae_i, v_us_i)
    raw_spearman_r, _ = spearmanr(v_ae_i, v_us_i)

    # --- Envelope correlation ---
    env_ae_i = _envelope(v_ae_i)
    env_us_i = _envelope(v_us_i)

    env_pearson_r, env_pearson_p = pearsonr(env_ae_i, env_us_i)
    env_spearman_r, _ = spearmanr(env_ae_i, env_us_i)

    def _hjorth_mobility(x: np.ndarray) -> float:
        return float(np.sqrt(np.var(np.diff(x)) / np.var(x))) if np.var(x) > 0 else np.nan

    return {
        "ae_rms": float(np.sqrt(np.mean(v_ae**2))),
        "us_rms": float(np.sqrt(np.mean(v_us**2))),
        "ae_hjorth_mobility": _hjorth_mobility(v_ae),
        "us_hjorth_mobility": _hjorth_mobility(v_us),
        # raw signal
        "raw_pearson_r": float(raw_pearson_r),
        "raw_pearson_p": float(raw_pearson_p),
        "raw_spearman_r": float(raw_spearman_r),
        # envelope
        "env_pearson_r": float(env_pearson_r),
        "env_pearson_p": float(env_pearson_p),
        "env_spearman_r": float(env_spearman_r),
        "n_samples_common": len(t_grid),
        # stored arrays for example plots
        "_t_grid": t_grid,
        "_v_ae_i": v_ae_i,
        "_v_us_i": v_us_i,
        "_env_ae": env_ae_i,
        "_env_us": env_us_i,
    }


# %%
# =============================================================================
# Load all files and collect per-sweep results
# =============================================================================

FILE_PATTERNS: list[str] | None = cfg.get("filters", {}).get("file_patterns") or None
hdf5_files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
if args.max_files:
    hdf5_files = hdf5_files[: args.max_files]

print(f"Processing {len(hdf5_files)} HDF5 file(s) from {INPUT_DIR}")

rows = []
example_sweeps: list[dict] = []  # keep a few for the overlay plot

for fp in hdf5_files:
    try:
        sweeps = load_sweep_signals(fp)
    except Exception as exc:
        print(f"  WARNING: could not load {fp.name}: {exc}")
        continue

    for sweep in sweeps:
        tp = sweep["test_parameters"]
        lm = sweep["lubricant_metadata"]
        bm = sweep["bearing_metadata"]

        # Both sensors must be present
        if SENSORS[0] not in sweep or SENSORS[1] not in sweep:
            continue

        rpm = float(tp.get("rpm", np.nan))
        if rpm > RPM_MAX:
            continue

        temp_c = float(tp.get("temperature_c", np.nan))
        nu_40 = float(lm.get("viscosity_40c_cst", np.nan))
        nu_100 = float(lm.get("viscosity_100c_cst", np.nan))

        try:
            kappa = calculate_kappa(
                rpm=rpm, temp_c=temp_c, d_pw=D_PW_MM,
                nu_40=nu_40, nu_100=nu_100,
            )
        except Exception:
            kappa = np.nan

        metrics = analyse_sweep(
            sweep["AE"]["t"], sweep["AE"]["v"],
            sweep["Ultrasound"]["t"], sweep["Ultrasound"]["v"],
        )

        # Save a handful of sweeps for cross-correlation example plots
        if len(example_sweeps) < 6:
            example_sweeps.append({
                "label": f"{sweep['file']} / {sweep['sweep']}",
                "rpm": rpm,
                "kappa": kappa,
                **metrics,
            })

        rows.append({
            "file": sweep["file"],
            "sweep": sweep["sweep"],
            "rpm": rpm,
            "temperature_c": temp_c,
            "kappa": kappa,
            **{k: v for k, v in metrics.items() if not k.startswith("_")},
        })

df = pd.DataFrame(rows)
print(f"\nCollected {len(df)} sweep-pairs")
print(df[["ae_rms", "us_rms",
          "raw_pearson_r", "raw_spearman_r",
          "env_pearson_r", "env_spearman_r"]].describe().round(4))

# %%
# =============================================================================
# Figure 1 — Distribution of per-sweep correlation per κ regime
# =============================================================================

_kappa_edges = df["kappa"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
_kappa_labels = [
    f"{_kappa_edges[i]:.2f} ≤ κ < {_kappa_edges[i+1]:.2f}"
    for i in range(len(_kappa_edges) - 1)
]
df["kappa_bin"] = pd.cut(
    df["kappa"], bins=_kappa_edges, labels=_kappa_labels,
    include_lowest=True,
)

_corr_panels = [
    ("raw_pearson_r", "Raw signal correlation (Pearson r)", "Pearson r  (AE vs US raw)"),
    ("env_pearson_r", "Envelope correlation (Pearson r)",   "Pearson r  (AE vs US envelope)"),
]

n_bins = len(_kappa_labels)
fig, axes = plt.subplots(n_bins, 2, figsize=(12, 4 * n_bins), squeeze=False)

for row_i, label in enumerate(_kappa_labels):
    sub = df[df["kappa_bin"] == label]
    for col_i, (col, title, xlabel) in enumerate(_corr_panels):
        ax = axes[row_i, col_i]
        ax.hist(sub[col].dropna(), bins=25, color=f"C{row_i}", edgecolor="none")
        med = sub[col].median()
        ax.axvline(med, color="red", ls="--", lw=1.2, label=f"median = {med:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\n{label}")
        ax.legend(fontsize=8)
        ax.grid(ls=":", alpha=0.3)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "ae_us_envelope_correlation_dist.png", dpi=150)
plt.show()
print("Saved: ae_us_envelope_correlation_dist.png")

# %%
# =============================================================================
# Figure 2 — RMS scatter: AE vs US, coloured by RPM and kappa
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, label in [
    (axes[0], "rpm", "RPM"),
    (axes[1], "kappa", "κ"),
]:
    valid = df.dropna(subset=[col])
    sc = ax.scatter(
        valid["ae_rms"], valid["us_rms"],
        c=valid[col], cmap="viridis", s=18, alpha=0.7, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=label)

    # Overall Spearman r across sweeps
    r, p = spearmanr(valid["ae_rms"], valid["us_rms"])
    ax.set_xlabel("AE RMS  [V]")
    ax.set_ylabel("US RMS  [V]")
    ax.set_title(f"AE vs US RMS — coloured by {label}\nSpearman ρ = {r:.3f}  (p = {p:.2e})")
    ax.grid(ls=":", alpha=0.4)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "ae_us_rms_scatter.png", dpi=150)
plt.show()
print("Saved: ae_us_rms_scatter.png")

# %%
# =============================================================================
# Figure 2b — Hjorth mobility scatter: AE vs US, coloured by RPM and kappa
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, label in [
    (axes[0], "rpm", "RPM"),
    (axes[1], "kappa", "κ"),
]:
    valid = df.dropna(subset=[col, "ae_hjorth_mobility", "us_hjorth_mobility"])
    sc = ax.scatter(
        valid["ae_hjorth_mobility"], valid["us_hjorth_mobility"],
        c=valid[col], cmap="viridis", s=18, alpha=0.7, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=label)

    r, p = spearmanr(valid["ae_hjorth_mobility"], valid["us_hjorth_mobility"])
    ax.set_xlabel("AE Hjorth mobility")
    ax.set_ylabel("US Hjorth mobility")
    ax.set_title(f"AE vs US Hjorth mobility — coloured by {label}\nSpearman ρ = {r:.3f}  (p = {p:.2e})")
    ax.grid(ls=":", alpha=0.4)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "ae_us_hjorth_mobility_scatter.png", dpi=150)
plt.show()
print("Saved: ae_us_hjorth_mobility_scatter.png")

# %%
# =============================================================================
# Figure 3 — Raw and envelope correlation vs operating conditions
# =============================================================================

cond_cols = [
    ("rpm",           "RPM"),
    ("temperature_c", "Temperature [°C]"),
    ("kappa",         "κ (lubrication ratio)"),
]
corr_cols = [
    ("raw_pearson_r", "Raw Pearson r",      "C0"),
    ("env_pearson_r", "Envelope Pearson r", "C2"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")

for row_i, (ycol, ylabel, color) in enumerate(corr_cols):
    for col_i, (xcol, xlabel) in enumerate(cond_cols):
        ax = axes[row_i][col_i]
        valid = df.dropna(subset=[xcol, ycol])
        ax.scatter(valid[xcol], valid[ycol],
                   s=12, alpha=0.5, color=color, edgecolors="none")
        bins = pd.cut(valid[xcol], bins=10)
        trend = valid.groupby(bins, observed=True)[ycol].mean()
        bin_centres = [iv.mid for iv in trend.index]
        ax.plot(bin_centres, trend.values, "r-o", ms=4, lw=1.5, label="bin mean")
        ax.axhline(0, color="k", ls="--", lw=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}  vs  {xlabel}")
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=8)
        if row_i == 1:
            ax.set_xlabel(xlabel)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "ae_us_correlation_vs_conditions.png", dpi=150)
plt.show()
print("Saved: ae_us_correlation_vs_conditions.png")

# %%
# =============================================================================
# Figure 5 — Raw signal and envelope overlay (best and worst correlated sweep)
# =============================================================================

df_sorted = df.sort_values("env_pearson_r")
best_idx = df_sorted.index[-1]
worst_idx = df_sorted.index[0]

fig, axes = plt.subplots(2, 2, figsize=(16, 8))

for col_i, (idx, title) in enumerate([
    (best_idx,  "Best-correlated sweep"),
    (worst_idx, "Worst-correlated sweep"),
]):
    row = df.loc[idx]
    match = next(
        (ex for ex in example_sweeps
         if ex["label"].startswith(str(row["file"]))),
        None,
    )
    if match is None:
        for r in range(2):
            axes[r][col_i].set_title(
                f"{title}\n(no stored arrays — increase --max-files)")
        continue

    t_ms = match["_t_grid"] * 1e3

    # Top row: raw signals (normalised)
    ax_raw = axes[0][col_i]
    ae_n = match["_v_ae_i"] / (np.abs(match["_v_ae_i"]).max() or 1)
    us_n = match["_v_us_i"] / (np.abs(match["_v_us_i"]).max() or 1)
    ax_raw.plot(t_ms, ae_n, lw=0.5, color="C0", alpha=0.8, label="AE raw (norm.)")
    ax_raw.plot(t_ms, us_n, lw=0.5, color="C3", alpha=0.8, label="US raw (norm.)")
    ax_raw.set_xlabel("Time [ms]")
    ax_raw.set_ylabel("Normalised voltage")
    ax_raw.set_title(
        f"{title} — raw signals\n"
        f"raw r = {row['raw_pearson_r']:.3f}  "
        f"RPM={row['rpm']:.0f}  κ={row['kappa']:.2f}"
    )
    ax_raw.legend(fontsize=8)
    ax_raw.grid(ls=":", alpha=0.4)

    # Bottom row: envelopes (normalised)
    ax_env = axes[1][col_i]
    ae_env_n = match["_env_ae"] / (match["_env_ae"].max() or 1)
    us_env_n = match["_env_us"] / (match["_env_us"].max() or 1)
    ax_env.plot(t_ms, ae_env_n, lw=0.8, color="C0", alpha=0.8, label="AE envelope (norm.)")
    ax_env.plot(t_ms, us_env_n, lw=0.8, color="C3", alpha=0.8, label="US envelope (norm.)")
    ax_env.set_xlabel("Time [ms]")
    ax_env.set_ylabel("Normalised envelope")
    ax_env.set_title(
        f"{title} — envelopes\n"
        f"env r = {row['env_pearson_r']:.3f}"
    )
    ax_env.legend(fontsize=8)
    ax_env.grid(ls=":", alpha=0.4)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "ae_us_envelope_overlay.png", dpi=150)
plt.show()
print("Saved: ae_us_envelope_overlay.png")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 70)
print("AE — US CORRELATION SUMMARY")
print("=" * 70)

r_rms, p_rms = spearmanr(df["ae_rms"].dropna(), df["us_rms"].dropna())
print(f"\nRMS level (Spearman):      ρ = {r_rms:.3f}  p = {p_rms:.2e}")

for kind, col in [("Raw signal", "raw_pearson_r"), ("Envelope", "env_pearson_r")]:
    print(f"\n{kind} correlation (Pearson r per sweep):")
    print(f"  mean={df[col].mean():.3f}  std={df[col].std():.3f}  "
          f"min={df[col].min():.3f}  median={df[col].median():.3f}  "
          f"max={df[col].max():.3f}")

print("\nCorrelation vs operating conditions (Spearman):")
print(f"  {'':20s}  {'raw r':>12}   {'env r':>12}")
for cond_col, cond_label in [("rpm", "RPM"), ("temperature_c", "Temperature"),
                              ("kappa", "κ")]:
    for ycol in ["raw_pearson_r", "env_pearson_r"]:
        valid = df.dropna(subset=[cond_col, ycol])
        r, p = spearmanr(valid[cond_col], valid[ycol])
        tag = "raw" if "raw" in ycol else "env"
        print(f"  vs {cond_label:15s} ({tag}):  ρ = {r:+.3f}  p = {p:.2e}  (n={len(valid)})")

df_out = df.drop(columns=["_t_grid", "_env_ae", "_env_us", "_xcorr", "_lags"],
                 errors="ignore")
out_path = OUTPUT_DIR / "ae_us_correlation_results.csv"
df_out.to_csv(out_path, index=False)
print(f"\nSaved per-sweep results: {out_path.name}")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\nae_us_correlation complete.")
