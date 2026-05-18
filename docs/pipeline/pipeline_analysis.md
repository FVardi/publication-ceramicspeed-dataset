# Pipeline Analysis — Strengths, Weaknesses & Improvement Recommendations

**Project:** CeramicSpeed — Lubrication condition monitoring via AE and UL  
**Target variable:** κ (kappa) — ISO 281 viscosity ratio  
**Created:** 2026-05-12 | **Last reviewed:** 2026-05-18 | **Fixes applied:** 2026-05-18

---

## Pipeline Overview

The pipeline predicts lubrication regime (κ) from acoustic emission (AE) and ultrasound (UL) bearing sensor data. It runs in four numbered scripts plus optional EDA scripts.

```
HDF5 files
    ↓ 01_feature_generation.py
features.parquet + metadata.parquet
    ↓ 02_feature_analysis.py
feature_selection.json  (+ correlation/VIF/PCA figures)
    ↓ 04_modelling.py
9 fitted models + holdout predictions + SHAP
    ↓ 03_evaluation.py
repeated CV scores + stat tests + performance table
```

---

## Stage-by-Stage Description

### 01 — Feature Generation (`scripts/01_feature_generation.py`)

**Input:** HDF5 measurement files (discovered via glob pattern)  
**Output:** `outputs/features.parquet`, `outputs/metadata.parquet`, `outputs/signal_quality.parquet`

Loads HDF5 files in parallel (joblib loky backend), optionally cleans raw signals, then extracts features per sweep × sensor.

**Signal cleaning** (optional, configured via `config.yaml` — currently disabled by default):
- Fix NaN/Inf via linear interpolation (edge values: nearest-neighbour extrapolation)
- Clipping detection (>1% of samples at signal rails)
- Saturation detection (longest flat run ≥ threshold)
- Z-score spike removal (`remove_outliers_z: null` = disabled)

**Pre-filtering:** UL broadband features are extracted from a 0–20 kHz pre-filtered signal to prevent DAQ noise from biasing spectral features. Band-specific features (AE only) are extracted from the raw cleaned signal.

**Feature extraction — 26 features per sensor × band:**

| Category | Features (12) |
|---|---|
| Time-domain | peak, RMS, std, variance, skewness, kurtosis, crest factor, shape factor, impulse factor, margin factor, Hjorth mobility, Hjorth complexity |

| Category | Features (14) |
|---|---|
| Frequency-domain | dominant frequency, spectral mean/std/skewness/kurtosis, center frequency, RMS frequency, peak frequency, spectral flatness, frequency-weighted std, normalized frequency std, frequency skewness/kurtosis, normalized bandwidth |

Band features (AE only): 20–500 kHz, 500k–1M Hz, 1M–2M Hz — each band gets a full copy of all 26 features prefixed by band label (e.g. `AE_20-500kHz__rms`).

---

### 02 — Feature Analysis & Selection (`scripts/02_feature_analysis.py`)

**Input:** `features.parquet`, `metadata.parquet`  
**Output:** `outputs/feature_selection.json`, ranking CSVs, correlation/VIF/PCA figures

**Kappa calculation:** ASTM D341 Walther equation using viscosity at operating temperature (ISO kinematic viscosity). Falls back silently to hardcoded Keratech 22 constants (22 cSt @ 40°C, 4.1 cSt @ 100°C) when viscosity metadata is missing.

**Sensor splitting before cleaning:** AE and UL are separated before feature-level cleaning so that NaN values in AE-only band columns don't cause UL rows to be dropped.

**Feature selection (two steps):**

1. **Correlation filter:** Retain features with both |Spearman ρ| ≥ 0.1 AND |Pearson r| ≥ 0.1 vs κ (threshold hardcoded at `CORR_MIN = 0.1`)
2. **Redundancy reduction:** Greedy removal using inter-feature Spearman correlation matrix + Variance Inflation Factor (VIF); keeps the subset that maximises target correlation while minimising collinearity

**Visualization outputs:** Feature ranking bar plots (Spearman ρ vs Pearson r), PCA scatter coloured by κ regime, inter-feature correlation heatmap, VIF bar plot

---

### 03 — Evaluation (`scripts/03_evaluation.py`)

**Input:** `features.parquet`, `metadata.parquet`, `feature_selection.json`  
**Output:** Repeated CV scores, performance table, stat test tables, SHAP agreement tables, figures

**Train/test split:** 80/20 stratified at sweep level — both sensors for a given sweep always land in the same partition. Combined models (AE + UL) use an inner join on (file, sweep).

**9 models trained (3 types × 3 feature sets: AE, UL, Combined):**

| Model | HP selection |
|---|---|
| Elastic Net | ElasticNetCV (9 alphas × 6 l1_ratios) |
| Polynomial (degree 2, top-5 features) | RidgeCV |
| LightGBM | Optuna (50 trials, early stopping at 50 rounds) |

**Evaluation strategy:**
- Outer 5-fold CV on 80% training set → out-of-fold predictions
- Final refit on full 80% → evaluated on 20% holdout
- Predictions clipped to [0, max training κ]

**Post-training outputs:** SHAP values (TreeExplainer for LightGBM, LinearExplainer for linear models), sensor contribution grouped by AE\_\_/UL\_\_ prefix for combined models, feature weights/importance CSVs.

---

### 04 — Modelling (`scripts/04_modelling.py`)

**Input:** `features.parquet`, `metadata.parquet`, `feature_selection.json`  
**Output:** Holdout prediction CSVs, SHAP CSVs, figures, `best_params.json`

Reproduces the identical 80/20 split and runs **repeated nested CV (R=10 × k=5 = 50 scores per model)** with full HP re-selection per fold for linear models; fixed params from `best_params.json` for LightGBM (no early stopping in eval folds — intentional, so folds are not used for selection).

**Statistical testing — two levels:**

| Level | Test | Correction |
|---|---|---|
| Within feature set (architecture comparison) | Corrected repeated k-fold t-test (Nadeau & Bengio 2003) | Holm-Bonferroni |
| Cross feature set (best per sensor) | Same t-test + Wilcoxon signed-rank + Diebold-Mariano + bootstrap ΔRMSE CI (10k resamples) | Holm-Bonferroni |

**Cross-model feature agreement:** SHAP top-10 feature overlap across model types per feature set.

---

## General State of the Pipeline

The structure is solid — reproducible YAML config, rigorous nested CV, three-model comparison, statistical testing with multiple-comparison correction, SHAP explainability. For a research pipeline this is well above average in methodological rigour. The main gaps are concentrated in preprocessing assumptions, a biased HP selection step, and publication-readiness of the reported metrics.

---

## Weaknesses

### High severity

**1. Polynomial top-k feature selection bias** ✓ Fixed 2026-05-18

~~In `scripts/04_evaluation.py:262`, Pearson correlation is computed on the **full training set** to select the top-k features before the outer fold split.~~

*Clarification:* `scripts/03_evaluation.py:_poly_fold_score` was already correct — it uses only `X_vals[tr, j]`. The bug was in `src/ceramicspeed/modelling.py:train_polynomial_cv`, where the internal CV for alpha selection used top-k features pre-selected on the full training set (mild leakage into alpha selection). Fixed by computing top-k inside each fold loop independently; the final model still uses global top-k (legitimate).

**2. Viscosity fallback is silent** ✓ Fixed 2026-05-18

~~`src/ceramicspeed/loading.py` applies Keratech 22 viscosity constants whenever the measurement file lacks viscosity metadata with no warning.~~

`_ensure_viscosity` now emits `logger.warning(...)` listing the injected keys whenever fallback values are applied.

**3. Pre-filter / band-filter inconsistency** — documented, by design

Broadband UL features are extracted from the pre-filtered signal (0–20 kHz), but AE band-specific features are extracted from the raw cleaned signal. The rationale is documented in `config.yaml` under `sensor_prefilter`. No code change needed.

**4. Combined model data loss is silent** ✓ Fixed 2026-05-18

~~The inner join on (file, sweep) drops sweeps silently.~~

`_merge_sensors` in `scripts/04_modelling.py` now prints a before → after sweep count with the number of dropped rows. Additionally, holdout CSVs now include `file` and `sweep` columns so `scripts/05_holdout_tests.py` can align cross-model comparisons on the common subset rather than skipping them.

**5. Correlation threshold hardcoded and unjustified** ✓ Fixed 2026-05-18

~~`CORR_MIN = 0.1` hardcoded in `scripts/02_feature_analysis.py`.~~

Moved to `config.yaml` under `feature_selection.corr_min`. The misleading comment (which said "0.5") has been corrected. `VIF_THRESHOLD` also moved to `feature_selection.vif_threshold` and passed explicitly to `identify_redundant_features` and `reduce_redundant_features`.

---

### Medium severity

**6. No κ regime stratification in CV** ✓ Fixed 2026-05-18

~~`KFold` splits are random. Rare high-κ or low-κ samples can concentrate in one fold.~~

`repeated_nested_cv` in `scripts/03_evaluation.py` now uses `StratifiedKFold` with 5 quantile-based κ bins, ensuring rare high/low κ values are spread evenly across folds.

**7. Cross-feature-set holdout tests are incomplete** ✓ Fixed 2026-05-18

~~Wilcoxon / Diebold-Mariano tests silently skipped when combined and single-sensor models have different test set sizes.~~

Holdout CSVs now include `file` and `sweep` columns (added in `scripts/04_modelling.py`). `scripts/05_holdout_tests.py` now uses `_align_predictions()` which merges models on their common sweeps before running tests, replacing the old length-match + allclose check.

**8. SHAP for Polynomial is misleading** ✓ Documented 2026-05-18

`LinearExplainer` is applied to degree-2 expanded features (e.g., `rms²`, `rms × kurtosis`), not to the original features. These interaction-term importances are not directly comparable to ElasticNet or LightGBM SHAP values. A NOTE comment has been added in `scripts/05_holdout_tests.py` before the cross-model agreement computation.

**9. Publication reporting gaps** — partially resolved

- ✓ Fixed 2026-05-18 — Bootstrap CIs on absolute holdout R², MAE, RMSE added to `05_holdout_tests.py` via `bootstrap_metric_ci()` in `src/ceramicspeed/evaluation.py`. Results saved to `outputs/05_holdout_tests/tables/holdout_metrics_with_ci.csv`.
- No regime-level classification accuracy (κ < 0.5 / 0.5–1.0 / ≥ 1.0 confusion matrix or F1) — not pursued; this is a regression task and regime breakdown is not a standard expectation.

---

### Low severity

**10. Signal cleaning disabled by default** — resolved

`signal_cleaning.enabled: true` in `config.yaml` (was previously `false` in an older version; now correctly enabled).

**11. Several tunable parameters are hardcoded throughout the codebase:**

| Parameter | Location | Status |
|---|---|---|
| SHAP top-k agreement | `scripts/05_holdout_tests.py` | ✓ Moved to `evaluation.shap_top_k` in `config.yaml` |
| VIF threshold | `scripts/02_feature_analysis.py` | ✓ Moved to `feature_selection.vif_threshold` in `config.yaml` |
| Saturation flat tolerance | `src/ceramicspeed/cleaning.py` | open — minor, internal constant |
| Z-score spike threshold default | `src/ceramicspeed/loading.py` DEFAULT_SIGNAL_CLEAN_CFG | open — already overridable via config |

**12. No error handling for malformed HDF5 files** ✓ Fixed 2026-05-18

~~A single malformed file crashes the parallel worker.~~

`load_and_process_file` in `src/ceramicspeed/loading.py` now wraps the `load_hdf5_file` call in a try-except, logs the error, and returns empty results so the parallel job continues.

---

## Prioritised Recommendations

| Priority | Fix | Status | Impact |
|---|---|---|---|
| 1 | Per-fold top-k in `train_polynomial_cv` — fold-local feature selection | ✓ Fixed | High |
| 2 | Warn when viscosity fallback is applied | ✓ Fixed | High — data integrity |
| 3 | Log combined model data loss (sweep count before/after inner join) | ✓ Fixed | High |
| 4 | Move `CORR_MIN` and `VIF_THRESHOLD` to `config.yaml` | ✓ Fixed | Medium |
| 5 | Stratified CV by κ quantile bins | ✓ Fixed | Medium |
| 6 | Add regime classification metrics (confusion matrix / F1 per regime) | **Open** | High — publication |
| 7 | Enable holdout-level tests for cross-feature-set pairs | ✓ Fixed | Medium |
| 8 | Document pre-filter vs band-filter design intent | ✓ Documented in config | Medium |
| 9 | HDF5 error handling with per-file logging | ✓ Fixed | Low-Medium |
| 10 | Move SHAP top-k to `config.yaml` | ✓ Fixed | Low |
| 11 | Add confidence intervals on absolute holdout performance metrics | **Open** | High — publication |
