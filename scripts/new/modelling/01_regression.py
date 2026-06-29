"""
11_featureset_comparison.py
===========================
Leak-free kappa-regression on the FULL candidate feature set.

NOTE: the VIF-"selected" feature set was removed (2026-06-29) -- the pipeline
now models the full candidate set only (ElasticNet + LightGBM). Earlier versions
also ran a leak-free "selected" set for comparison; that is gone. Script/file
names keep the "featureset_comparison" prefix for continuity with downstream
scripts (13/14).

Default (recommended) protocol
-------------------------------
    1. Single-pass, disjoint GroupKFold (k=--n-folds) over the WHOLE dataset,
       grouped by operating-point-merged acquisition holds (see "Operating-
       point grouping" below) -- not a single 80/20 split. For each fold in
       turn, that fold is held out completely while feature selection,
       tuning, and fitting are redone fresh on the other folds only, then
       predictions are made on the held-out fold. Every group ends up held
       out exactly once, by a model that never saw it during selection/
       tuning/training. Pooling all folds' held-out predictions gives more
       groups for the downstream significance test
       (13_group_paired_tests.py) without needing a Nadeau-Bengio-style
       correction: the folds are disjoint, so the standard group-level
       Harvey-DM test applies directly to the pooled set.
    2. Operating-point grouping: the RPM protocol sweeps up AND back down
       through the same speeds within each temperature block, so a given
       (rpm, temperature) operating point is often revisited by two or more
       separate acquisition holds ("twins") -- empirically, about 28% of
       holds have at least one twin elsewhere in the dataset. Since kappa is
       a deterministic function of (rpm, temperature) alone here, twins have
       nearly identical kappa targets, so a model with no genuine
       sensitivity to lubrication-film state -- only to the operating point
       -- could still score well on a held-out twin if its non-twin sibling
       was in training. All holds sharing a (rounded) RPM/temperature bin
       are therefore merged into one group before any split/CV, so twins are
       always kept together. Use --allow-twin-split to disable this and
       compare against the more permissive (twin-leakage-prone) grouping.
    3. Hyperparameters tuned (if --tune-lightgbm), and the CV estimate
       computed, by inner GroupKFold on the training partition.
    4. No nested CV: tuning/fitting never see the held-out fold.

It runs ElasticNet and LightGBM on the **full candidate feature set** (all
candidate features, no selection -- regularisation/tree structure handle
redundancy, and this is leak-free by construction). The candidate set is
computed fresh from only target-independent cleaning (NaN/Inf handling,
constant-column removal, RPM filter) via cleaning.true_candidate_columns -- it
deliberately does NOT reuse feature_selection.json's "all_columns", which
02_feature_analysis.py captures *after* a whole-dataset, kappa-correlated filter
has already discarded most candidate columns. True candidate counts: AE 42,
US 56.

Per-sweep holdout predictions are saved with acquisition-group ids so
group-paired significance tests can be run downstream.

Tuned LightGBM (--tune-lightgbm)
---------------------------------
By default, LightGBM uses fixed hyperparameters straight from config.yaml
(no tuning at all). With --tune-lightgbm, a single Optuna study is run per
(feature_set, target) combination, using ONLY the training partition's own
inner GroupKFold splits (never the held-out fold) to score trials. This is
safe because the held-out fold never participates in the inner GroupKFold
used to score trials.

Filename suffixes
------------------
Output files for the default (recommended) protocol use plain names
(e.g. featureset_comparison.csv). Any deviation from the default appends an
explicit suffix: _singlesplit (--single-split) and/or _twinsplit
(--allow-twin-split).

Usage
-----
    python scripts/11_featureset_comparison.py
    python scripts/11_featureset_comparison.py --tune-lightgbm
    python scripts/11_featureset_comparison.py --n-folds 10
    python scripts/11_featureset_comparison.py --single-split   # legacy single 80/20 split
    python scripts/11_featureset_comparison.py --allow-twin-split   # compare against twin-leakage-prone grouping
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
from ceramicspeed.cleaning import filter_by_metadata, true_candidate_columns
from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import load_config, get_output_dir
from ceramicspeed.grouping import derive_hold_groups, merge_twin_groups


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tune-lightgbm", action="store_true",
                        help="Tune LightGBM via Optuna using GroupKFold on the "
                             "training partition only (single-level, held-out "
                             "fold never touched).")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optuna trials per (feature_set, target[, fold]) "
                             "combination (default: modelling.lightgbm.n_trials "
                             "from config, or 20).")
    parser.add_argument("--single-split", action="store_true",
                        help="Use a single 80/20 GroupShuffleSplit instead of the "
                             "default pooled GroupKFold over the whole dataset. "
                             "Kept for comparison; noisier and less powerful than "
                             "the default (see module docstring).")
    parser.add_argument("--n-folds", type=int, default=None,
                        help="Number of folds for the default pooled GroupKFold "
                             "(default: modelling.cv_n_splits from config, or 5). "
                             "Ignored with --single-split.")
    parser.add_argument("--allow-twin-split", action="store_true",
                        help="Do not merge operating-point twin holds before "
                             "splitting/CV, i.e. use the plain acquisition-hold "
                             "grouping. Kept for comparison; allows near-duplicate "
                             "operating points to leak across train/test (see "
                             "module docstring). Default merges twins.")
    parser.add_argument("--rpm-bin-width", type=float, default=100.0,
                        help="RPM bin width for operating-point twin merging "
                             "(default 100). Ignored with --allow-twin-split.")
    parser.add_argument("--temp-bin-width", type=float, default=1.0,
                        help="Temperature bin width in C for operating-point twin "
                             "merging (default 1.0). Ignored with --allow-twin-split.")
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = get_output_dir(cfg)
NEW_DIR = OUTPUT_DIR / "new"
SCRIPT_DIR = NEW_DIR / "regression"
PRED_DIR = SCRIPT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

D_PW_MM = cfg["bearing"]["d_pw_mm"]
RPM_MAX = cfg["filters"]["rpm_max"]
RPM_MIN = cfg["filters"].get("rpm_min", 0.0)  # drop startup/standstill transients
TEMP_MIN = cfg["filters"].get("temp_min", None)  # drop sub-floor cold-start
RANDOM_STATE = cfg.get("random_state", 42)

model_cfg = cfg.get("modelling", {})
CV_N_SPLITS = model_cfg.get("cv_n_splits", 5)
TEST_SIZE = model_cfg.get("test_size", 0.2)
GROUPED_SPLIT = bool(model_cfg.get("grouped_split", True))
N_FOLDS = args.n_folds or CV_N_SPLITS

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


def _tune_lgbm_hparams(X_tr, y_tr, groups_tr, n_trials, n_inner_splits=3,
                        screen_n_estimators=200):
    """Single-level Optuna search for LightGBM, scored via GroupKFold on the
    given training rows only. The holdout (or, in --kfold-pool mode, the
    current outer fold) is never passed to this function, so the resulting
    hyperparameters are blind to it -- safe to freeze and reuse for the final
    fit + held-out evaluation."""
    import optuna
    from lightgbm import LGBMRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base = dict(
        min_child_samples=lgb_cfg.get("min_child_samples", 10),
        subsample=lgb_cfg.get("subsample", 0.8),
        colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
        reg_alpha=lgb_cfg.get("reg_alpha", 0.0),
        reg_lambda=lgb_cfg.get("reg_lambda", 1.0),
        random_state=RANDOM_STATE, verbose=-1,
    )
    param_grid = lgb_cfg.get("param_grid") or {
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, -1],
    }

    k = min(n_inner_splits, len(np.unique(groups_tr)))
    inner_cv = GroupKFold(n_splits=k)
    splits = list(inner_cv.split(X_tr, y_tr, groups_tr))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            **base,
            "n_estimators": screen_n_estimators,
            **{name: trial.suggest_categorical(name, choices)
               for name, choices in param_grid.items()},
        }
        scores = []
        for itr, ival in splits:
            m = LGBMRegressor(**params)
            m.fit(X_tr.iloc[itr], y_tr[itr])
            scores.append(float(np.sqrt(mean_squared_error(
                y_tr[ival], m.predict(X_tr.iloc[ival])))))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    tuned = {**base, "n_estimators": lgb_cfg.get("n_estimators", 500), **study.best_params}
    return tuned, float(study.best_value)


N_TRIALS_LGBM = args.n_trials or lgb_cfg.get("n_trials", 20)

MODELS = [("ElasticNet", make_enet), ("LightGBM", make_lgbm)]


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
raw_feature_df, raw_metadata_df = load_parquet_pair(NEW_DIR)

true_all_columns = {
    sensor: true_candidate_columns(raw_feature_df, raw_metadata_df, sensor, RPM_MAX)
    for sensor in ("AE", "UL")
}
print(f"True candidate columns (target-independent cleaning only): "
      f"AE={len(true_all_columns['AE'])}, UL={len(true_all_columns['UL'])}")

df, metadata = filter_by_metadata(raw_feature_df, raw_metadata_df,
                                  rpm_max=RPM_MAX, rpm_min=RPM_MIN, temp_min=TEMP_MIN)
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
# Acquisition-hold groups (identical logic to 04_modelling.py), merged into
# operating-point groups by default (see ceramicspeed.grouping)
# =============================================================================
hold_groups = derive_hold_groups(metadata)
if not args.allow_twin_split:
    hold_groups = merge_twin_groups(
        metadata, hold_groups,
        rpm_bin_width=args.rpm_bin_width, temp_bin_width=args.temp_bin_width,
    )
group_of = {
    (f, s): int(g) for f, s, g in zip(df["file"], df["sweep"], hold_groups)
}

sweep_keys = df[["file", "sweep"]].drop_duplicates().reset_index(drop=True)
_key_first = pd.DataFrame(
    {"file": df["file"], "sweep": df["sweep"], "g": hold_groups}
).drop_duplicates(["file", "sweep"]).reset_index(drop=True)
assert len(_key_first) == len(sweep_keys)
N_HOLD_GROUPS = _key_first["g"].nunique()


# %%
# =============================================================================
# Per-(mode, target, model) training, given an arbitrary train/test sweep
# split. Used once for the default 80/20 split, or --n-folds times for
# --kfold-pool. Feature selection, tuning, and fitting are all redone fresh
# on whichever rows are passed in as "train" -- nothing here is leak-free
# only by accident, it is leak-free by construction regardless of how the
# split was produced upstream.
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


def _combined(keyed):
    ae, us = keyed["AE"], keyed["UL"]
    feat_ae = [c for c in ae.columns if c not in ("kappa", "rpm")]
    feat_us = [c for c in us.columns if c not in ("kappa", "rpm")]
    return ae[feat_ae + ["kappa", "rpm"]].join(us[feat_us], how="inner")


def _groups_for(frame) -> np.ndarray:
    return np.array([group_of[idx] for idx in frame.index])


TARGETS = ["AE", "UL", "combined"]
_DISPLAY = {"AE": "AE", "UL": "US", "combined": "Combined"}


def run_split(train_sweep_idx, test_sweep_idx, fold_id):
    """Run the full leak-free protocol for one train/test sweep partition.

    Returns (holdout_preds, cv_preds, summary_rows): dicts/list keyed by tag
    (model_target_mode), ready to be saved directly (single-split mode) or
    accumulated across folds and pooled (--kfold-pool mode).
    """
    train_sweeps = set(
        zip(sweep_keys.iloc[train_sweep_idx]["file"], sweep_keys.iloc[train_sweep_idx]["sweep"]))
    row_in_train = df.apply(lambda r: (r["file"], r["sweep"]) in train_sweeps, axis=1)

    df_train = df[row_in_train].reset_index(drop=True)
    df_test = df[~row_in_train].reset_index(drop=True)
    meta_train = metadata[row_in_train.values].reset_index(drop=True)
    meta_test = metadata[~row_in_train.values].reset_index(drop=True)
    print(f"[fold {fold_id}] Train rows: {len(df_train)}  Test rows: {len(df_test)}")

    keyed_train, keyed_test = {}, {}
    all_cols_renamed = {}

    for sensor, all_cols in true_all_columns.items():
        label = _SENSOR_LABEL.get(sensor, sensor)
        ren = [_rename(c, label) for c in all_cols]

        ktr = _keyed(df_train, meta_train, sensor, all_cols, label)
        kte = _keyed(df_test, meta_test, sensor, all_cols, label)
        keyed_train[sensor], keyed_test[sensor] = ktr, kte
        all_cols_renamed[sensor] = ren

        print(f"  [fold {fold_id}] {label}: {len(ren)} candidate features "
              f"(train n={len(ktr)}, test n={len(kte)})")

    comb_train = _combined(keyed_train)
    comb_test = _combined(keyed_test)

    def _frames(target):
        if target == "combined":
            return comb_train, comb_test
        return keyed_train[target], keyed_test[target]

    def _cols_for(target, mode):
        src = all_cols_renamed
        return (src["AE"] + src["UL"]) if target == "combined" else src[target]

    holdout_preds, cv_preds, summary_rows = {}, {}, []

    for mode in ("full",):
        for target in TARGETS:
            tr, te = _frames(target)
            cols = _cols_for(target, mode)
            X_tr, y_tr = tr[cols], tr["kappa"].values
            X_te, y_te = te[cols], te["kappa"].values
            g_tr = _groups_for(tr)

            for model_name, default_make_est in MODELS:
                tag = f"{model_name}_{_DISPLAY[target]}_{mode}"
                make_est = default_make_est
                tuned_params, tuned_rmse = None, None
                try:
                    if model_name == "LightGBM" and args.tune_lightgbm:
                        from lightgbm import LGBMRegressor
                        tuned_params, tuned_rmse = _tune_lgbm_hparams(
                            X_tr, y_tr, g_tr, n_trials=N_TRIALS_LGBM,
                        )
                        print(f"  [fold {fold_id}] {tag:32s} tuned "
                              f"(inner CV RMSE={tuned_rmse:.4f}): {tuned_params}")

                        def make_est(_p=tuned_params):
                            return LGBMRegressor(**_p)

                    # Leak-free GroupKFold CV estimate on this fold's train rows
                    oof = grouped_cv_oof(make_est, X_tr, y_tr, g_tr)
                    cv_r2 = r2_score(y_tr, oof)

                    # Final fit on this fold's full train rows, single held-out eval
                    est = make_est()
                    est.fit(X_tr, y_tr)
                    y_pred = np.clip(est.predict(X_te), 0.0, None)
                    ho_r2 = r2_score(y_te, y_pred)
                    ho_mae = mean_absolute_error(y_te, y_pred)
                    ho_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

                    pred_df = te.index.to_frame(index=False)
                    pred_df["group"] = _groups_for(te)
                    pred_df["fold"] = fold_id
                    pred_df["rpm"] = te["rpm"].values
                    pred_df["y_true"] = y_te
                    pred_df["y_pred"] = y_pred
                    holdout_preds[tag] = pred_df

                    cv_df = tr.index.to_frame(index=False)
                    cv_df["group"] = g_tr
                    cv_df["fold"] = fold_id
                    cv_df["rpm"] = tr["rpm"].values
                    cv_df["y_true"] = y_tr
                    cv_df["y_pred"] = oof
                    cv_preds[tag] = cv_df

                    summary_rows.append({
                        "model": model_name, "target": _DISPLAY[target], "feature_set": mode,
                        "fold": fold_id,
                        "n_features": len(cols), "n_train": len(X_tr), "n_test": len(X_te),
                        "cv_r2_grouped": round(cv_r2, 4),
                        "holdout_r2": round(ho_r2, 4),
                        "holdout_mae": round(ho_mae, 4),
                        "holdout_rmse": round(ho_rmse, 4),
                        "tuned_params": json.dumps(tuned_params) if tuned_params else "",
                        "tuned_inner_cv_rmse": round(tuned_rmse, 4) if tuned_rmse else "",
                    })
                    print(f"  [fold {fold_id}] {tag:32s} CV(grouped) R2={cv_r2:.3f}  "
                          f"HO R2={ho_r2:.3f}  ({len(cols)} feats)")
                except Exception as exc:
                    print(f"  [fold {fold_id}] {tag:32s} FAILED: {exc}")

    return holdout_preds, cv_preds, summary_rows


# %%
# =============================================================================
# Run: by default, pooled GroupKFold over the whole dataset; --single-split
# for the legacy single 80/20 split.
# =============================================================================
all_summary_rows = []
_suffix_parts = []
if args.single_split:
    _suffix_parts.append("singlesplit")
if args.allow_twin_split:
    _suffix_parts.append("twinsplit")
_suffix = "" if not _suffix_parts else "_" + "_".join(_suffix_parts)

if not args.single_split:
    print(f"\nPooled GroupKFold (k={N_FOLDS}) over "
          f"{N_HOLD_GROUPS} hold groups, selection/tuning/fitting redone per fold.\n")
    gkf = GroupKFold(n_splits=N_FOLDS)
    pooled_holdout: dict[str, list] = {}

    for fold_id, (train_sweep_idx, test_sweep_idx) in enumerate(
        gkf.split(np.arange(len(sweep_keys)), groups=_key_first["g"].values)
    ):
        holdout_preds, _cv_preds, summary_rows = run_split(train_sweep_idx, test_sweep_idx, fold_id)
        all_summary_rows.extend(summary_rows)
        for tag, pdf in holdout_preds.items():
            pooled_holdout.setdefault(tag, []).append(pdf)

    for tag, frames in pooled_holdout.items():
        pooled = pd.concat(frames, ignore_index=True)
        pooled.to_csv(PRED_DIR / f"holdout_pooled{_suffix}_{tag}.csv", index=False)
    print(f"\nPooled {N_FOLDS} folds -> "
          f"{sum(len(v) for v in pooled_holdout.values()) // max(len(pooled_holdout), 1)} "
          f"held-out rows per tag (across all folds combined)")

else:
    if GROUPED_SPLIT:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_sweep_idx, test_sweep_idx = next(
            gss.split(np.arange(len(sweep_keys)), groups=_key_first["g"].values))
        print(f"Grouped 80/20 split over {N_HOLD_GROUPS} hold groups")
    else:
        from sklearn.model_selection import train_test_split
        train_sweep_idx, test_sweep_idx = train_test_split(
            np.arange(len(sweep_keys)), test_size=TEST_SIZE,
            random_state=RANDOM_STATE, shuffle=True)

    holdout_preds, cv_preds, summary_rows = run_split(train_sweep_idx, test_sweep_idx, fold_id=0)
    all_summary_rows.extend(summary_rows)
    for tag, pdf in holdout_preds.items():
        pdf.drop(columns=["fold"]).to_csv(PRED_DIR / f"holdout{_suffix}_{tag}.csv", index=False)
    for tag, cdf in cv_preds.items():
        cdf.drop(columns=["fold"]).to_csv(PRED_DIR / f"cv{_suffix}_{tag}.csv", index=False)


# %%
# =============================================================================
# Comparison tables
# =============================================================================
comp = pd.DataFrame(all_summary_rows)

if not args.single_split:
    # Aggregate holdout R2/MAE/RMSE over the POOLED predictions (all folds
    # combined), not the mean of per-fold metrics, so n reflects the full
    # pooled sample.
    pooled_rows = []
    for tag, frames in pooled_holdout.items():
        model_name, disp_target, mode = tag.rsplit("_", 2)
        pooled = pd.concat(frames, ignore_index=True)
        fold_rows = comp[(comp.model == model_name) & (comp.target == disp_target)
                         & (comp.feature_set == mode)]
        pooled_rows.append({
            "model": model_name, "target": disp_target, "feature_set": mode,
            "n_features": int(fold_rows["n_features"].iloc[0]) if len(fold_rows) else np.nan,
            "cv_r2_grouped": round(float(fold_rows["cv_r2_grouped"].mean()), 4) if len(fold_rows) else np.nan,
            "n_pooled": len(pooled),
            "holdout_r2": round(r2_score(pooled["y_true"], pooled["y_pred"]), 4),
            "holdout_mae": round(mean_absolute_error(pooled["y_true"], pooled["y_pred"]), 4),
            "holdout_rmse": round(float(np.sqrt(
                mean_squared_error(pooled["y_true"], pooled["y_pred"]))), 4),
        })
    pooled_comp = pd.DataFrame(pooled_rows)
    pooled_comp.to_csv(SCRIPT_DIR / f"featureset_comparison{_suffix}.csv", index=False)
    print("\n" + "=" * 78)
    print(f"POOLED HOLDOUT R2 (all {N_FOLDS} folds combined)")
    print("=" * 78)
    print(pooled_comp.to_string(index=False))

comp.to_csv(SCRIPT_DIR / f"featureset_comparison{_suffix}_byfold.csv", index=False)

wide = comp.groupby(["model", "target"], as_index=False)["holdout_r2"].mean()
wide.to_csv(SCRIPT_DIR / f"featureset_comparison{_suffix}_wide.csv", index=False)

print("\n" + "=" * 78)
print("HOLDOUT R2 (per-fold mean) - full feature set")
print("=" * 78)
print(wide.to_string(index=False))
print(f"\nSaved tables -> {SCRIPT_DIR}")
print(f"Saved per-sweep holdout predictions -> {PRED_DIR}")

if __name__ == "__main__":
    print("\n11_featureset_comparison complete.")
