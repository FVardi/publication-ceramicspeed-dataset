"""
09_proxy_diagnostics.py
=======================
Operating-condition proxy diagnostics for the kappa regression.

Pipeline position: standalone diagnostic -- reads features.parquet +
metadata.parquet + feature_selection.json; does not modify any pipeline
output. Run after 04_modelling.py (uses its holdout predictions for the
direct-model comparison).

What this script does
---------------------
kappa is a deterministic function of (RPM, T), so the acoustic model's
performance must be interpreted against how much operating-point information
the features carry. This script quantifies that:

1. Per-feature |Spearman| of every retained feature with RPM, with
   temperature, and with kappa.
2. Set-level proxy models: LightGBM on the retained AE features predicting
   RPM and predicting temperature (same 80/20 split as 04_modelling.py).
3. Two-stage model: predicted (RPM_hat, T_hat) fed through the ISO 281 /
   Walther formula (calculate_kappa) -> kappa_hat, compared on the holdout
   against the direct features->kappa LightGBM from 04_modelling.py.

Outputs (outputs/09_proxy_diagnostics/)
---------------------------------------
tables/feature_oc_correlations.csv   per-feature rho with RPM / T / kappa
tables/proxy_summary.csv             set-level R2/RMSE for all models
proxy_stats.json                     headline numbers for 07_paper_export
figures/two_stage_vs_direct.png      holdout scatter comparison

Usage
-----
    python scripts/09_proxy_diagnostics.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util as _ilu


def _load_module(name, rel):
    spec = _ilu.spec_from_file_location(name, ROOT / rel)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_config = _load_module("cs_config", "src/ceramicspeed/config.py")
_ck = _load_module("cs_kappa", "src/ceramicspeed/calculate_kappa.py")
calculate_kappa = _ck.calculate_kappa

try:
    cfg = _config.load_config()
except Exception:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8").rstrip("\x00"))

_args, _ = argparse.ArgumentParser().parse_known_args()
_p = argparse.ArgumentParser()
_p.add_argument("--output-dir", type=str, default=None)
_args, _ = _p.parse_known_args()
if _args.output_dir:
    OUTPUT_DIR = Path(_args.output_dir)
else:
    try:
        OUTPUT_DIR = _config.get_output_dir(cfg)
    except Exception:
        OUTPUT_DIR = ROOT / "outputs"
if not (OUTPUT_DIR / "04_modelling").is_dir() and (ROOT / "outputs" / "04_modelling").is_dir():
    OUTPUT_DIR = ROOT / "outputs"

SCRIPT_DIR = OUTPUT_DIR / "09_proxy_diagnostics"
TABLES_DIR = SCRIPT_DIR / "tables"
FIGURES_DIR = SCRIPT_DIR / "figures"
for d in (SCRIPT_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

D_PW: float = cfg["bearing"]["d_pw_mm"]
RPM_MAX: float = cfg["filters"]["rpm_max"]
TEST_SIZE: float = (cfg.get("modelling") or {}).get("test_size", 0.2)
RANDOM_STATE: int = cfg.get("random_state", 42)
SENSOR = "AE"  # the channel whose proxy behaviour we interrogate

# =============================================================================
# Load features, metadata, retained set
# =============================================================================

feat_df = pd.read_parquet(OUTPUT_DIR / "features.parquet")
meta_df = pd.read_parquet(OUTPUT_DIR / "metadata.parquet")
sel = json.loads((OUTPUT_DIR / "feature_selection.json").read_text().rstrip("\x00"))
retained = sel[SENSOR]["retained"]

mask = (feat_df["sensor"] == SENSOR) & (meta_df["rpm"] <= RPM_MAX).values
df = feat_df[mask].reset_index(drop=True)
meta = meta_df[mask.values if hasattr(mask, "values") else mask].reset_index(drop=True)

meta["kappa"] = meta.apply(
    lambda r: calculate_kappa(
        rpm=r["rpm"], temp_c=r["temperature_c"], d_pw=D_PW,
        nu_40=r["viscosity_40c_cst"], nu_100=r["viscosity_100c_cst"],
    ),
    axis=1,
)
print(f"{SENSOR}: {len(df)} sweeps, {len(retained)} retained features")

X = df[retained]
valid = X.notna().all(axis=1)
X, df, meta = X[valid].reset_index(drop=True), df[valid].reset_index(drop=True), meta[valid].reset_index(drop=True)

# =============================================================================
# 1. Per-feature correlations with RPM / T / kappa
# =============================================================================

rows = []
for c in retained:
    r_rpm = spearmanr(X[c], meta["rpm"])[0]
    r_tmp = spearmanr(X[c], meta["temperature_c"])[0]
    r_kap = spearmanr(X[c], meta["kappa"])[0]
    rows.append({
        "feature": c,
        # |rho| columns (unchanged schema for downstream tables / export)
        "rho_rpm": abs(r_rpm),
        "rho_temp": abs(r_tmp),
        "rho_kappa": abs(r_kap),
        # signed values for the proxy-map figure
        "rho_rpm_signed": r_rpm,
        "rho_temp_signed": r_tmp,
        "rho_kappa_signed": r_kap,
    })
corr_df = pd.DataFrame(rows).sort_values("rho_kappa", ascending=False)
corr_df.to_csv(TABLES_DIR / "feature_oc_correlations.csv", index=False)
print("\nTop features -- |rho| with kappa vs operating conditions:")
print(corr_df.head(8).to_string(index=False))

# =============================================================================
# 2. Same 80/20 split as 04_modelling.py
# =============================================================================

from sklearn.model_selection import GroupShuffleSplit, train_test_split

GROUPED_SPLIT: bool = bool((cfg.get("modelling") or {}).get("grouped_split", False))


def _derive_hold_groups(meta_df: pd.DataFrame) -> np.ndarray:
    """Contiguous staircase-hold ids (same definition as 03/04)."""
    sweep_no = meta_df["sweep"].str.split("_").str[1].astype(int).values
    files = meta_df["file"].values
    step = np.round(meta_df["rpm"].values / 100.0)
    order = np.lexsort((sweep_no, files))
    gid = np.empty(len(meta_df), dtype=int)
    g = 0
    prev = None
    for pos in order:
        key = (files[pos], step[pos])
        if prev is None or key[0] != prev[0] or key[1] != prev[1]:
            g += 1
        gid[pos] = g
        prev = key
    return gid


sweep_keys = df[["file", "sweep"]].drop_duplicates().reset_index(drop=True)
if GROUPED_SPLIT:
    hold_groups = _derive_hold_groups(meta)
    _key_first = pd.DataFrame({"file": df["file"], "sweep": df["sweep"],
                               "g": hold_groups}).drop_duplicates(["file", "sweep"])
    _gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(_gss.split(np.arange(len(sweep_keys)),
                                     groups=_key_first["g"].values))
    print(f"grouped 80/20 split over {_key_first['g'].nunique()} hold groups")
else:
    tr_idx, te_idx = train_test_split(
        np.arange(len(sweep_keys)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, shuffle=True,
    )
train_set = set(zip(sweep_keys.iloc[tr_idx]["file"], sweep_keys.iloc[tr_idx]["sweep"]))
in_train = df.apply(lambda r: (r["file"], r["sweep"]) in train_set, axis=1).values

X_tr, X_te = X[in_train], X[~in_train]
meta_tr, meta_te = meta[in_train], meta[~in_train]
print(f"\nsplit: {in_train.sum()} train / {(~in_train).sum()} holdout")

# =============================================================================
# 3. Set-level proxy models and two-stage kappa
# =============================================================================

import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error


def fit_predict(target_tr, target_te, label):
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        random_state=RANDOM_STATE, verbose=-1,
    )
    model.fit(X_tr, target_tr)
    pred = model.predict(X_te)
    r2 = r2_score(target_te, pred)
    rmse = float(np.sqrt(mean_squared_error(target_te, pred)))
    print(f"  {label:32s} R2 = {r2:.4f}  RMSE = {rmse:.4g}")
    return pred, r2, rmse


print("\nSet-level proxy models (retained AE features):")
rpm_pred, rpm_r2, rpm_rmse = fit_predict(meta_tr["rpm"], meta_te["rpm"], "features -> RPM")
tmp_pred, tmp_r2, tmp_rmse = fit_predict(meta_tr["temperature_c"], meta_te["temperature_c"], "features -> temperature")

# Two-stage: predicted operating point -> ISO 281 formula -> kappa
nu40 = float(meta["viscosity_40c_cst"].iloc[0])
nu100 = float(meta["viscosity_100c_cst"].iloc[0])
two_stage = np.array([
    calculate_kappa(rpm=max(float(r), 1.0), temp_c=float(t), d_pw=D_PW, nu_40=nu40, nu_100=nu100)
    for r, t in zip(rpm_pred, tmp_pred)
])
y_te = meta_te["kappa"].values
ts_r2 = r2_score(y_te, two_stage)
ts_rmse = float(np.sqrt(mean_squared_error(y_te, two_stage)))
print(f"  {'two-stage (RPM_hat,T_hat)->kappa':32s} R2 = {ts_r2:.4f}  RMSE = {ts_rmse:.4g}")

# Direct model from 04_modelling for reference (same split by construction)
direct_path = OUTPUT_DIR / "04_modelling" / "predictions" / "model_holdout_lightgbm_ae.csv"
direct = pd.read_csv(direct_path).dropna(subset=["file"])
d_r2 = r2_score(direct["y_true"], direct["y_pred"])
d_rmse = float(np.sqrt(mean_squared_error(direct["y_true"], direct["y_pred"])))
print(f"  {'direct features->kappa (04)':32s} R2 = {d_r2:.4f}  RMSE = {d_rmse:.4g}")

summary = pd.DataFrame([
    {"model": "features->RPM", "R2": rpm_r2, "RMSE": rpm_rmse},
    {"model": "features->temperature", "R2": tmp_r2, "RMSE": tmp_rmse},
    {"model": "two_stage_kappa", "R2": ts_r2, "RMSE": ts_rmse},
    {"model": "direct_kappa_lightgbm_04", "R2": d_r2, "RMSE": d_rmse},
])
pd.DataFrame({
    "file": meta_te["file"].values, "sweep": meta_te["sweep"].values,
    "rpm": meta_te["rpm"].values, "temp": meta_te["temperature_c"].values,
    "y_true": y_te, "y_pred_two_stage": two_stage,
}).to_csv(TABLES_DIR / "two_stage_predictions.csv", index=False)
summary.to_csv(TABLES_DIR / "proxy_summary.csv", index=False)

(SCRIPT_DIR / "proxy_stats.json").write_text(json.dumps({
    "rpm_r2": rpm_r2, "rpm_rmse": rpm_rmse,
    "temp_r2": tmp_r2, "temp_rmse": tmp_rmse,
    "two_stage_r2": ts_r2, "two_stage_rmse": ts_rmse,
    "direct_r2": d_r2, "direct_rmse": d_rmse,
    "n_features": len(retained), "sensor": SENSOR,
}, indent=1))

# =============================================================================
# 4. Figure: two-stage vs direct on holdout
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, (pred, true, ttl, r2, rmse) in zip(axes, [
    (two_stage, y_te, "Two-stage: features → (RPM̂, T̂) → κ", ts_r2, ts_rmse),
    (direct["y_pred"].values, direct["y_true"].values, "Direct: features → κ (LightGBM, 04)", d_r2, d_rmse),
]):
    ax.scatter(true, pred, s=4, alpha=0.3)
    lim = [0, max(true.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("true κ"); ax.set_title(f"{ttl}\nR²={r2:.3f}, RMSE={rmse:.3f}")
axes[0].set_ylabel("predicted κ")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "two_stage_vs_direct.png", dpi=150)

# =============================================================================
# 5. Figure: per-feature operating-point proxy map (one per sensor)
#    x = signed Spearman with temperature, y = signed Spearman with RPM,
#    colour = |Spearman| with kappa. Visualises Stage-1 (operating-point)
#    proxy structure of the retained features -- NOT the within-step channel.
# =============================================================================


def _short(name: str) -> str:
    """Compact feature label for annotation."""
    return (name.replace("AE_", "").replace("UL_", "").replace("US_", "").replace("__", " ")
                .replace("_", " ").replace("kHz", " kHz").strip())


def per_feature_corr(feature_df, meta_df, cols):
    """Signed + |.| Spearman of each feature with RPM / T / kappa."""
    out = []
    for c in cols:
        r_rpm = spearmanr(feature_df[c], meta_df["rpm"])[0]
        r_tmp = spearmanr(feature_df[c], meta_df["temperature_c"])[0]
        r_kap = spearmanr(feature_df[c], meta_df["kappa"])[0]
        out.append({
            "feature": c,
            "rho_rpm": abs(r_rpm), "rho_temp": abs(r_tmp), "rho_kappa": abs(r_kap),
            "rho_rpm_signed": r_rpm, "rho_temp_signed": r_tmp, "rho_kappa_signed": r_kap,
        })
    return pd.DataFrame(out).sort_values("rho_kappa", ascending=False)


def proxy_map_figure(cdf, sensor, fname):
    """Scatter of signed rho(T) vs signed rho(RPM), coloured by |rho(kappa)|."""
    fig2, ax = plt.subplots(figsize=(7.2, 6.4))
    xs = cdf["rho_temp_signed"].values
    ys = cdf["rho_rpm_signed"].values
    cs = cdf["rho_kappa"].values  # |rho| with kappa drives colour
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0, vmax=1,
                    s=70, edgecolor="k", linewidth=0.5, zorder=3)
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvline(0, color="0.6", lw=0.8, zorder=1)
    # stagger label vertical offset to reduce overlap in the dense cluster
    for i, (x, y, name) in enumerate(zip(xs, ys, cdf["feature"].values)):
        dy = 4 if i % 2 == 0 else -9
        ax.annotate(_short(name), (x, y), fontsize=7, xytext=(5, dy),
                    textcoords="offset points", zorder=4)
    ax.set_xlabel(r"signed Spearman $\rho$ with temperature")
    ax.set_ylabel(r"signed Spearman $\rho$ with RPM (speed)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Operating-point proxy map of retained {sensor} features")
    cb = fig2.colorbar(sc, ax=ax)
    cb.set_label(r"$|\rho|$ with $\kappa$")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / fname, dpi=150)
    plt.close(fig2)


# AE map reuses the corr_df already computed and saved above.
proxy_map_figure(corr_df, "AE", "feature_proxy_map_ae.png")

# US map: same retained-feature treatment, US (passive ultrasound = "UL") channel.
us_retained = sel["UL"]["retained"]
us_mask = (feat_df["sensor"] == "UL") & (meta_df["rpm"] <= RPM_MAX).values
us_df = feat_df[us_mask].reset_index(drop=True)
us_meta = meta_df[us_mask.values if hasattr(us_mask, "values") else us_mask].reset_index(drop=True)
us_meta["kappa"] = us_meta.apply(
    lambda r: calculate_kappa(
        rpm=r["rpm"], temp_c=r["temperature_c"], d_pw=D_PW,
        nu_40=r["viscosity_40c_cst"], nu_100=r["viscosity_100c_cst"],
    ),
    axis=1,
)
us_X = us_df[us_retained]
us_valid = us_X.notna().all(axis=1)
us_X, us_meta = us_X[us_valid].reset_index(drop=True), us_meta[us_valid].reset_index(drop=True)
us_corr = per_feature_corr(us_X, us_meta, us_retained)
us_corr.to_csv(TABLES_DIR / "feature_oc_correlations_us.csv", index=False)
proxy_map_figure(us_corr, "US", "feature_proxy_map_us.png")
print(f"US: {len(us_X)} sweeps, {len(us_retained)} retained features")

# =============================================================================
# 6. Marginal vs conditional (partition-based) correlation structure
#    For both sensors, every retained feature:
#      marginal     -- Spearman(feature, T) and (feature, RPM) over all sweeps
#      conditional  -- within-RPM-step Spearman(feature, T)   [condition on speed]
#                      within-temp-block Spearman(feature, RPM)[condition on temp]
#    Temperature blocks are TBLOCK-wide bins on MEASURED temperature (spread
#    reported, not assumed). Rendered by 06_plots.py Plot D4c.
# =============================================================================

TBLOCK = 2.0        # deg C, measured-temperature block width
MIN_N_PART = 30
MIN_LEVELS = 8


def _partition_corr(d, feats, by, target, level_col):
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
            out[c] = (np.median(rhos), np.percentile(rhos, 25), np.percentile(rhos, 75))
    return out


cond_rows = []
for _sensor, _feats_all in (("AE", retained), ("UL", us_retained)):
    _m = (feat_df["sensor"] == _sensor) & (meta_df["rpm"] <= RPM_MAX).values
    _f = feat_df[_m].reset_index(drop=True)
    _md = meta_df[_m.values if hasattr(_m, "values") else _m].reset_index(drop=True)
    _feats = [c for c in _feats_all if c in _f.columns]
    d = _f[_feats].copy()
    d["rpm"] = _md["rpm"].values
    d["temp"] = _md["temperature_c"].values
    d = d[d["rpm"] >= 60].reset_index(drop=True)
    d["rstep"] = (d["rpm"] / 100).round() * 100
    d["tblock"] = (d["temp"] / TBLOCK).round() * TBLOCK
    _tb = d.groupby("tblock").agg(n=("temp", "size"), tstd=("temp", "std"),
                                  rsteps=("rstep", "nunique"))
    _tb = _tb[(_tb["n"] >= MIN_N_PART) & (_tb["rsteps"] >= MIN_LEVELS)]
    print(f"[{_sensor}] {len(_tb)} temp blocks; median within-block temp std "
          f"{_tb['tstd'].median():.2f} C (max {_tb['tstd'].max():.2f} C)")
    cond_t = _partition_corr(d, _feats, "rstep", "temp", "temp")
    cond_r = _partition_corr(d, _feats, "tblock", "rpm", "rstep")
    for c in _feats:
        if c not in cond_t or c not in cond_r:
            continue
        cond_rows.append({
            "sensor": "US" if _sensor == "UL" else _sensor, "feature": c,
            "marg_temp": spearmanr(d[c], d["temp"])[0],
            "marg_rpm": spearmanr(d[c], d["rpm"])[0],
            "cond_temp": cond_t[c][0], "cond_temp_lo": cond_t[c][1], "cond_temp_hi": cond_t[c][2],
            "cond_rpm": cond_r[c][0], "cond_rpm_lo": cond_r[c][1], "cond_rpm_hi": cond_r[c][2],
        })
pd.DataFrame(cond_rows).to_csv(TABLES_DIR / "cond_vs_marginal.csv", index=False)
print(f"Wrote cond_vs_marginal.csv ({len(cond_rows)} features)")

print(f"\nWrote tables, stats, and figures to {SCRIPT_DIR}")
