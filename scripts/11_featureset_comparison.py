"""
11_featureset_comparison.py
===========================
Simplified, leak-free protocol + full-vs-selected feature-set comparison.

Motivation
----------
The main pipeline selects features in 02_feature_analysis.py on the *whole*
dataset (including the 20% holdout), which leaks the test set into selection.
This script implements the simplified protocol discussed for the paper:

    1. Acquisition-grouped 80/20 split (identical to 04_modelling.py).
    2. Feature selection run on the **80% training partition only** (leak-free),
       via ceramicspeed.analysis.select_features.
    3. Hyperparameters tuned, and the CV estimate computed, by **GroupKFold**
       on the 80% train (folds split on acquisition-hold groups, so near-
       duplicate windows never cross the train/validation boundary).
    4. A single evaluation on the acquisition-grouped 20% holdout.
    5. No nested CV.

It runs each model for two feature sets so their performance can be compared:

    * "selected" : Stage 1 (|rho|,|r| >= corr_min) + Stage 2 (iterative VIF),
                   fit on the training rows only.
    * "full"     : all candidate features, no selection (regularisation only;
                   this is leak-free by construction).

Rows are held identical across the two feature sets (validity is computed over
the full candidate set) so any difference reflects the feature set, not sample
count. Per-sweep holdout predictions are saved with acquisition-group ids so
05-style group-paired significance tests can be run downstream.

Usage
-----
    python scripts/11_featureset_comparison.py
    python scripts/11_featureset_comparison.py --lightgbm   # also run LightGBM
    python scripts/11_featureset_comparison.py --config alt.yaml
"""

# %%
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from ceramicspeed.loading import load_parquet_pair
from ceramicspeed.cleaning import filter_by_metadata
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.analysis import select_features


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--lightgbm", action="store_true",
                        help="Also run LightGBM (config hyperparameters, no Optuna).")
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = get_output_dir(cfg)
SCRIPT_DIR = OUTPUT_DIR / "11_featureset_comparison"
PRED_DIR = SCRIPT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

D_PW_MM = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RANDOM_STATE = cfg.get("random_state", 42)

model_cfg = cfg.get("modelling", {})
CV_N_SPLITS = model_cfg.get("cv_n_splits", 5)
TEST_SIZE = model_cfg.get("test_size", 0.2)
GROUPED_SPLIT = bool(model_cfg.get("grouped_split", True))

fs_cfg = cfg.get("feature_selection", {})
CORR_MIN = fs_cfg.get("corr_min", 0.1)
VIF_THRESHOLD = fs_cfg.get("vif_threshold", 5.0)

enet_cfg = model_cfg.get("elastic_net", {})
ENET_ALPHAS = enet_cfg.get("alphas")
ENET_L1_RATIOS = enet_cfg.get("l1_ratios") or [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
ENET_MAX_ITER = enet_cfg.get("max_iter", 10_000)

lgb_cfg = model_cfg.get("lightgbm", {})

_SENSOR_LABEL = {"UL": "US"}  # AE keeps its name


# %%
# =============================================================================
# Model factories — fresh, self-contained estimators (scaling inside the
# pipeline so it is refit per fold; never leaks across the split).
# =============================================================================
def make_enet():
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=ENET_L1_RATIOS, alphas=ENET_ALPHAS,
                     cv=CV_N_SPLITS, max_iter=ENET_MAX_ITER,
                     random_state=RANDOM_STATE),
    )


def make_lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=lgb_cfg.get("n_estimators", 500),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        max_depth=lgb_cfg.get("max_depth", 6),
        num_leaves=lgb_cfg.get("num_leaves", 31),
        min_child_samples=lgb_cfg.get("min_child_samples", 10),
        subsample=lgb_cfg.get("subsample", 0.8),
        colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
        reg_alpha=lgb_cfg.get("reg_alpha", 0.0),
        reg_lambda=lgb_cfg.get("reg_lambda", 1.0),
        random_state=RANDOM_STATE, verbose=-1,
    )


MODELS = [("ElasticNet", make_enet)]
if args.lightgbm:
    MODELS.append(("LightGBM", make_lgbm))


def grouped_cv_oof(make_est, X, y, groups):
    """Leak-free out-of-fold predictions via GroupKFold (clipped at 0)."""
    k = min(CV_N_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=k)
    oof = np.full(len(y), np.nan)
    for tr, va in gkf.split(X, y, groups):
        est = make_est()
        est.fit(X.iloc[tr], y[tr])
        oof[va] = np.clip(est.predict(X.iloc[va]), 0.0, None)
    return oof


# %%
# =============================================================================
# Load + filter + kappa
# =============================================================================
raw_feature_df, raw_metadata_df = load_parquet_pair(OUTPUT_DIR)

with open(OUTPUT_DIR / "feature_selection.json") as fh:
    feature_selection = json.load(fh)

df, metadata = filter_by_metadata(raw_feature_df, raw_metadata_df, rpm_max=RPM_MAX)
df = df.reset_index(drop=True)
metadata = metadata.reset_index(drop=True)

metadata["kappa"] = metadata.apply(
    lambda row: calculate_kappa(
        rpm=row["rpm"], temp_c=row["temperature_c"], d_pw=D_PW_MM,
        nu_40=row["viscosity_40c_cst"], nu_100=row["viscosity_100c_cst"],
    ),
    axis=1,
)
print(f"Rows after RPM filter: {len(df)}")


# %%
# =============================================================================
# Acquisition-hold groups (identical logic to 04_modelling.py) + 80/20 split
# =============================================================================
def _derive_hold_groups(meta_df: pd.DataFrame) -> np.ndarray:
    sweep_no = meta_df["sweep"].str.split("_").str[1].astype(int).values
    files = meta_df["file"].values
    step = np.round(meta_df["rpm"].values / 100.0)
    order = np.lexsort((sweep_no, files))
    gid = np.empty(len(meta_df), dtype=int)
    g, prev = 0, None
    for pos in order:
        key = (files[pos], step[pos])
        if prev is None or key[0] != prev[0] or key[1] != prev[1]:
            g += 1
        gid[pos] = g
        prev = key
    return gid


hold_groups = _derive_hold_groups(metadata)
group_of = {
    (f, s): int(g) for f, s, g in zip(df["file"], df["sweep"], hold_groups)
}

sweep_keys = df[["file", "sweep"]].drop_duplicates().reset_index(drop=True)
if GROUPED_SPLIT:
    _key_first = pd.DataFrame(
        {"file": df["file"], "sweep": df["sweep"], "g": hold_groups}
    ).drop_duplicates(["file", "sweep"])
    _gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_sweep_idx, _ = next(
        _gss.split(np.arange(len(sweep_keys)), groups=_key_first["g"].values))
    print(f"Grouped 80/20 split over {_key_first['g'].nunique()} hold groups")
else:
    from sklearn.model_selection import train_test_split
    train_sweep_idx, _ = train_test_split(
        np.arange(len(sweep_keys)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, shuffle=True)

train_sweeps = set(
    zip(sweep_keys.iloc[train_sweep_idx]["file"], sweep_keys.iloc[train_sweep_idx]["sweep"]))
row_in_train = df.apply(lambda r: (r["file"], r["sweep"]) in train_sweeps, axis=1)

df_train = df[row_in_train].reset_index(drop=True)
df_test = df[~row_in_train].reset_index(drop=True)
meta_train = metadata[row_in_train.values].reset_index(drop=True)
meta_test = metadata[~row_in_train.values].reset_index(drop=True)
print(f"Train rows: {len(df_train)}  Test rows: {len(df_test)}")


# %%
# =============================================================================
# Per-sensor keyed frames (rows validated over the FULL candidate set so they
# are identical across feature sets). Columns renamed exactly as in
# 04_modelling.py so the combined model can concatenate AE + US safely.
# =============================================================================
def _rename(col: str, label: str) -> str:
    if col in ("kappa", "rpm") or col.startswith(f"{label}_"):
        return col
    return f"{label}__{col}"


def _keyed(df_src, meta_src, sensor, all_cols, label):
    mask = df_src["sensor"] == sensor
    X = df_src.loc[mask, ["file", "sweep"] + all_cols].reset_index(drop=True)
    km = meta_src.loc[mask, ["kappa", "rpm"]].reset_index(drop=True)
    X["kappa"] = km["kappa"].values
    X["rpm"] = km["rpm"].values
    X = X[X[all_cols].notna().all(axis=1)]
    X = X.set_index(["file", "sweep"])
    return X.rename(columns=lambda c: _rename(c, label))


keyed_train, keyed_test = {}, {}
all_cols_renamed, selected_renamed = {}, {}

for sensor, info in feature_selection.items():
    label = _SENSOR_LABEL.get(sensor, sensor)
    all_cols = info["all_columns"]
    ren = [_rename(c, label) for c in all_cols]

    ktr = _keyed(df_train, meta_train, sensor, all_cols, label)
    kte = _keyed(df_test, meta_test, sensor, all_cols, label)
    keyed_train[sensor], keyed_test[sensor] = ktr, kte
    all_cols_renamed[sensor] = ren

    sel = select_features(  # leak-free: TRAIN rows only
        ktr[ren], ktr["kappa"].values,
        corr_min=CORR_MIN, vif_threshold=VIF_THRESHOLD,
    )
    selected_renamed[sensor] = sel
    print(f"{label}: {len(ren)} candidates -> {len(sel)} selected on train "
          f"(train n={len(ktr)}, test n={len(kte)})")


def _combined(keyed):
    ae, us = keyed["AE"], keyed["UL"]
    feat_ae = [c for c in ae.columns if c not in ("kappa", "rpm")]
    feat_us = [c for c in us.columns if c not in ("kappa", "rpm")]
    return ae[feat_ae + ["kappa", "rpm"]].join(us[feat_us], how="inner")


comb_train = _combined(keyed_train)
comb_test = _combined(keyed_test)
print(f"Combined train n={len(comb_train)}, test n={len(comb_test)}")


def _groups_for(frame) -> np.ndarray:
    return np.array([group_of[idx] for idx in frame.index])


# %%
# =============================================================================
# Train every (feature-set x model x target): GroupKFold CV + grouped holdout
# =============================================================================
def _frames(target):
    if target == "combined":
        return comb_train, comb_test
    return keyed_train[target], keyed_test[target]


def _cols_for(target, mode):
    src = all_cols_renamed if mode == "full" else selected_renamed
    return (src["AE"] + src["UL"]) if target == "combined" else src[target]


TARGETS = ["AE", "UL", "combined"]
_DISPLAY = {"AE": "AE", "UL": "US", "combined": "Combined"}

rows = []
for mode in ("selected", "full"):
    for target in TARGETS:
        tr, te = _frames(target)
        cols = _cols_for(target, mode)
        X_tr, y_tr = tr[cols], tr["kappa"].values
        X_te, y_te = te[cols], te["kappa"].values
        g_tr = _groups_for(tr)

        for model_name, make_est in MODELS:
            tag = f"{model_name}_{_DISPLAY[target]}_{mode}"
            try:
                # Leak-free GroupKFold CV estimate on the 80% train
                oof = grouped_cv_oof(make_est, X_tr, y_tr, g_tr)
                cv_r2 = r2_score(y_tr, oof)

                # Final fit on full 80% train, single grouped holdout
                est = make_est()
                est.fit(X_tr, y_tr)
                y_pred = np.clip(est.predict(X_te), 0.0, None)
                ho_r2 = r2_score(y_te, y_pred)
                ho_mae = mean_absolute_error(y_te, y_pred)
                ho_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

                # Save per-sweep holdout predictions (group ids + rpm for plots)
                pred_df = te.index.to_frame(index=False)
                pred_df["group"] = _groups_for(te)
                pred_df["rpm"] = te["rpm"].values
                pred_df["y_true"] = y_te
                pred_df["y_pred"] = y_pred
                pred_df.to_csv(PRED_DIR / f"holdout_{tag}.csv", index=False)

                # Save grouped out-of-fold CV predictions (for CV scatter plots)
                cv_df = tr.index.to_frame(index=False)
                cv_df["group"] = g_tr
                cv_df["rpm"] = tr["rpm"].values
                cv_df["y_true"] = y_tr
                cv_df["y_pred"] = oof
                cv_df.to_csv(PRED_DIR / f"cv_{tag}.csv", index=False)

                rows.append({
                    "model": model_name, "target": _DISPLAY[target], "feature_set": mode,
                    "n_features": len(cols), "n_train": len(X_tr), "n_test": len(X_te),
                    "cv_r2_grouped": round(cv_r2, 4),
                    "holdout_r2": round(ho_r2, 4),
                    "holdout_mae": round(ho_mae, 4),
                    "holdout_rmse": round(ho_rmse, 4),
                })
                print(f"  {tag:32s} CV(grouped) R2={cv_r2:.3f}  HO R2={ho_r2:.3f}  "
                      f"({len(cols)} feats)")
            except Exception as exc:
                print(f"  {tag:32s} FAILED: {exc}")


# %%
# =============================================================================
# Comparison tables
# =============================================================================
comp = pd.DataFrame(rows)

wide = comp.pivot_table(
    index=["model", "target"], columns="feature_set", values="holdout_r2"
).reset_index()
if {"full", "selected"}.issubset(wide.columns):
    wide["delta(full-selected)"] = (wide["full"] - wide["selected"]).round(4)

comp.to_csv(SCRIPT_DIR / "featureset_comparison.csv", index=False)
wide.to_csv(SCRIPT_DIR / "featureset_comparison_wide.csv", index=False)

print("\n" + "=" * 78)
print("HOLDOUT R2 - selected (leak-free) vs full feature set")
print("=" * 78)
print(wide.to_string(index=False))
print(f"\nSaved tables -> {SCRIPT_DIR}")
print(f"Saved per-sweep holdout predictions -> {PRED_DIR}")

if __name__ == "__main__":
    print("\n11_featureset_comparison complete.")
