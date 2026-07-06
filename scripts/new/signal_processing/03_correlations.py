"""
12_fullset_decomposition.py
===========================
Operating-point decomposition + marginal/conditional correlation structure,
computed on the FULL candidate feature sets (no selection gate), aligned to
the same leak-free protocol as 11_featureset_comparison.py.

Why
---
kappa is a deterministic function of (RPM, T) at fixed oil/geometry, so any
features -> kappa model can, at best, infer the operating point. This script
quantifies how much of each channel's kappa-regression performance is
operating-point soft-sensing:

  Part A -- decomposition (per channel AE / US / Combined, full features):
    * features -> RPM            (R2: how well the channel soft-senses speed)
    * features -> temperature    (R2: how well it soft-senses temperature)
    * two-stage  features -> (RPM_hat, T_hat) -> ISO 281 -> kappa   (R2)
    * direct     features -> kappa  (R2)
    The two-stage R2 is the share of kappa prediction reachable purely by
    inferring the operating point; direct - two_stage is the residual beyond
    explicit operating-point inference. By default this is computed via
    pooled GroupKFold (operating-point-merged groups) over the whole dataset,
    matching 11_featureset_comparison.py -- see that script's docstring for
    why (single-split noise, operating-point-twin leakage). Use
    --single-split / --allow-twin-split to deviate, for comparison.

  Part B -- marginal vs conditional correlations (per channel, full features):
    * marginal     : Spearman(feature, RPM) and (feature, T) over all sweeps
    * conditional  : within-RPM-step Spearman(feature, T)   [speed held]
                     within-temp-block Spearman(feature, RPM)[temp held]
    Answers: are each channel's features correlated with RPM, temperature, or
    both? Saved as a table + per-channel proxy maps. This is purely
    descriptive (uses all data, no train/test split), so it is unaffected by
    --single-split / --allow-twin-split.

Candidate features are derived fresh via target-independent cleaning (see
ceramicspeed.cleaning.true_candidate_columns) rather than reusing
feature_selection.json's "all_columns", for the same reason as
11_featureset_comparison.py: that field is captured *after* a whole-dataset
correlation filter against kappa and is not actually the full candidate set.

Usage
-----
    python scripts/12_fullset_decomposition.py
    python scripts/12_fullset_decomposition.py --single-split
    python scripts/12_fullset_decomposition.py --allow-twin-split
"""

# %%
import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import r2_score
import lightgbm as lgb

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata, true_candidate_columns
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.grouping import derive_hold_groups, merge_twin_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--single-split", action="store_true",
                        help="Use a single 80/20 GroupShuffleSplit instead of the "
                             "default pooled GroupKFold over the whole dataset.")
    parser.add_argument("--n-folds", type=int, default=None,
                        help="Number of folds for the default pooled GroupKFold "
                             "(default: modelling.cv_n_splits from config, or 5).")
    parser.add_argument("--allow-twin-split", action="store_true",
                        help="Do not merge operating-point twin holds before "
                             "splitting/CV (see 11_featureset_comparison.py).")
    parser.add_argument("--rpm-bin-width", type=float, default=100.0)
    parser.add_argument("--temp-bin-width", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)
OUTPUT_DIR = get_output_dir(cfg)
NEW_DIR = OUTPUT_DIR / "new"
SCRIPT_DIR = NEW_DIR / "correlations"
TABLES_DIR = SCRIPT_DIR / "tables"
FIGURES_DIR = SCRIPT_DIR / "figures"
for d in (SCRIPT_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

D_PW = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RPM_MIN = cfg["filters"].get("rpm_min", 0.0)  # drop startup/standstill transients
TEMP_MIN = cfg["filters"].get("temp_min", None)  # drop sub-floor cold-start
RANDOM_STATE = cfg.get("random_state", 42)
model_cfg = cfg.get("modelling", {})
TEST_SIZE = model_cfg.get("test_size", 0.2)
GROUPED_SPLIT = bool(model_cfg.get("grouped_split", True))
N_FOLDS = args.n_folds or model_cfg.get("cv_n_splits", 5)
_SENSOR_LABEL = {"UL": "US"}

_suffix_parts = []
if args.single_split:
    _suffix_parts.append("singlesplit")
if args.allow_twin_split:
    _suffix_parts.append("twinsplit")
_suffix = "" if not _suffix_parts else "_" + "_".join(_suffix_parts)

# %%
# =============================================================================
# Load + filter + kappa
# =============================================================================
raw_feat, raw_meta = load_parquet_pair(NEW_DIR)

true_all_columns = {
    sensor: true_candidate_columns(raw_feat, raw_meta, sensor, RPM_MAX)
    for sensor in ("AE", "UL")
}
print(f"True candidate columns (target-independent cleaning only): "
      f"AE={len(true_all_columns['AE'])}, UL={len(true_all_columns['UL'])}")

df, metadata = filter_by_metadata(raw_feat, raw_meta, rpm_max=RPM_MAX,
                                  rpm_min=RPM_MIN, temp_min=TEMP_MIN)
df = df.reset_index(drop=True)
metadata = metadata.reset_index(drop=True)
metadata["kappa"] = metadata.apply(
    lambda r: calculate_kappa(rpm=r["rpm"], temp_c=r["temperature_c"], d_pw=D_PW,
                              nu_40=r["viscosity_40c_cst"], nu_100=r["viscosity_100c_cst"]),
    axis=1,
)
NU40 = float(metadata["viscosity_40c_cst"].iloc[0])
NU100 = float(metadata["viscosity_100c_cst"].iloc[0])


def _rename(c, label):
    return c if (c in ("kappa", "rpm", "temp") or c.startswith(f"{label}_")) else f"{label}__{c}"


# %%
# =============================================================================
# PART B -- marginal vs conditional correlations (full features, all sweeps)
# Descriptive only (uses all data, no split) -- unaffected by --single-split /
# --allow-twin-split, so always saved under the plain (no-suffix) name.
# =============================================================================
TBLOCK = 2.0
MIN_N_PART = 30
MIN_LEVELS = 8


def _partition_corr(d, feats, by, target, level_col):
    """Per-feature within-partition Spearman: returns (median, q25, q75)."""
    out = {}
    for c in feats:
        rhos = []
        for _, sub in d.groupby(by):
            if sub[level_col].nunique() < MIN_LEVELS or len(sub) < MIN_N_PART:
                continue
            r = spearmanr(sub[c], sub[target])[0]
            if np.isfinite(r):
                rhos.append(r)
        if rhos:
            rhos = np.array(rhos)
            out[c] = (float(np.median(rhos)),
                      float(np.percentile(rhos, 25)),
                      float(np.percentile(rhos, 75)))
    return out


def _short(name):
    return (name.replace("AE__", "").replace("US__", "").replace("AE_", "")
                .replace("US_", "").replace("__", " ").replace("_", " ").strip())


cond_rows = []
for sensor, label in (("AE", "AE"), ("UL", "US")):
    all_cols = true_all_columns[sensor]
    ren = [_rename(c, label) for c in all_cols]
    m = df["sensor"] == sensor
    d = df.loc[m, all_cols].reset_index(drop=True)
    d.columns = ren
    d["rpm"] = metadata.loc[m, "rpm"].values
    d["temp"] = metadata.loc[m, "temperature_c"].values
    d["kappa"] = metadata.loc[m, "kappa"].values
    d = d[d[ren].notna().all(axis=1)]
    d = d[d["rpm"] >= 60].reset_index(drop=True)
    d["rstep"] = (d["rpm"] / 100).round() * 100
    d["tblock"] = (d["temp"] / TBLOCK).round() * TBLOCK

    cond_t = _partition_corr(d, ren, "rstep", "temp", "temp")
    cond_r = _partition_corr(d, ren, "tblock", "rpm", "rstep")
    for c in ren:
        ct = cond_t.get(c, (np.nan, np.nan, np.nan))
        cr = cond_r.get(c, (np.nan, np.nan, np.nan))
        cond_rows.append({
            "sensor": label, "feature": c,
            "marg_rpm": spearmanr(d[c], d["rpm"])[0],
            "marg_temp": spearmanr(d[c], d["temp"])[0],
            "marg_kappa": spearmanr(d[c], d["kappa"])[0],
            "cond_rpm": cr[0], "cond_rpm_lo": cr[1], "cond_rpm_hi": cr[2],
            "cond_temp": ct[0], "cond_temp_lo": ct[1], "cond_temp_hi": ct[2],
        })
    print(f"[{label}] marginal/conditional over {len(d)} sweeps, {len(ren)} features")

cm = pd.DataFrame(cond_rows)
cm.to_csv(TABLES_DIR / "cond_vs_marginal_full.csv", index=False)
print(f"Saved: {TABLES_DIR / 'cond_vs_marginal_full.csv'}")


# %%
# =============================================================================
# PART B figures -- proxy map per channel: marginal rho(RPM) vs rho(T),
# coloured by |rho(kappa)|. Shows whether features track speed, temp, or both.
# =============================================================================
for label in ("AE", "US"):
    sub = cm[cm["sensor"] == label]
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    sc = ax.scatter(sub["marg_temp"], sub["marg_rpm"], c=sub["marg_kappa"].abs(),
                    cmap="viridis", vmin=0, vmax=1, s=70, edgecolor="k", linewidth=0.5, zorder=3)
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
    for i, (_, row) in enumerate(sub.iterrows()):
        dy = 4 if i % 2 == 0 else -9
        ax.annotate(_short(row["feature"]), (row["marg_temp"], row["marg_rpm"]),
                    fontsize=6.5, xytext=(5, dy), textcoords="offset points", zorder=4)
    ax.set_xlabel(r"marginal Spearman $\rho$ with temperature")
    ax.set_ylabel(r"marginal Spearman $\rho$ with RPM (speed)")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_title(f"Operating-point proxy map -- all {label} candidate features")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$|\rho|$ with $\kappa$")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"proxy_map_full_{label.lower()}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: proxy_map_full_{label.lower()}.png")

# Quadrant summary: how many features track speed-only / temp-only / both
print("\nFeature operating-point coupling (|marginal rho| >= 0.3):")
for label in ("AE", "US"):
    sub = cm[cm["sensor"] == label]
    spd = sub["marg_rpm"].abs() >= 0.3
    tmp = sub["marg_temp"].abs() >= 0.3
    print(f"  {label}: speed-only={int((spd & ~tmp).sum())}  "
          f"temp-only={int((~spd & tmp).sum())}  "
          f"both={int((spd & tmp).sum())}  neither={int((~spd & ~tmp).sum())}  "
          f"(of {len(sub)})")

if __name__ == "__main__":
    print("\ncorrelations complete.")
