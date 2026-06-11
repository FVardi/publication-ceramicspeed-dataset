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
    rows.append({
        "feature": c,
        "rho_rpm": abs(spearmanr(X[c], meta["rpm"])[0]),
        "rho_temp": abs(spearmanr(X[c], meta["temperature_c"])[0]),
        "rho_kappa": abs(spearmanr(X[c], meta["kappa"])[0]),
    })
corr_df = pd.DataFrame(rows).sort_values("rho_kappa", ascending=False)
corr_df.to_csv(TABLES_DIR / "feature_oc_correlations.csv", index=False)
print("\nTop features -- |rho| with kappa vs operating conditions:")
print(corr_df.head(8).to_string(index=False))

# =============================================================================
# 2. Same 80/20 split as 04_modelling.py
# =============================================================================

from sklearn.model_selection import train_test_split

sweep_keys = df[["file", "sweep"]].drop_duplicates().reset_index(drop=True)
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
print(f"\nWrote tables, stats, and figure to {SCRIPT_DIR}")
