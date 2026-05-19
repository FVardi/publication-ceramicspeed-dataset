# Pipeline Analysis — Strengths, Weaknesses & Improvement Recommendations

**Project:** CeramicSpeed — Lubrication condition monitoring via AE and UL  
**Target variable:** κ (kappa) — ISO 281 viscosity ratio  
**Created:** 2026-05-12 | **Last reviewed:** 2026-05-19 | **Fixes applied:** 2026-05-18, 2026-05-19

---

## Pipeline Overview

The pipeline predicts lubrication regime (κ) from acoustic emission (AE) and ultrasound (UL) bearing sensor data. It runs in four numbered scripts plus optional EDA scripts.

```
HDF5 files
    ↓ 01_feature_generation.py
features.parquet + metadata.parquet
    ↓ 02_feature_analysis.py
feature_selection.json  (+ correlation/VIF/PCA figures)
    ↓ 03_evaluation.py
repeated CV scores + performance table
    ↓ 04_modelling.py
9 fitted models + holdout predictions + SHAP + CV out-of-fold predictions
    ↓ 05_holdout_tests.py
stat tests + bootstrap CIs + SHAP agreement tables
    ↓ 06_plots.py  (standalone — reads saved CSVs, no retraining)
all publication figures
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

### 05 — Holdout Tests (`scripts/05_holdout_tests.py`)

**Input:** `outputs/04_modelling/predictions/`, `outputs/03_evaluation/predictions/repeated_cv_scores.csv`, `feature_selection.json`
**Output:** `outputs/05_holdout_tests/tables/`, `outputs/05_holdout_tests/shap/`

Runs two levels of significance testing on the 9 trained models and produces bootstrap CIs on absolute holdout metrics.

**Level 1 — within feature set (CV scores):** Nadeau-Bengio corrected repeated k-fold t-test on all model-pair combinations within each sensor configuration; Holm-Bonferroni correction.

**Level 2 — cross feature set (holdout residuals):** Wilcoxon signed-rank + Diebold-Mariano + bootstrap ΔRMSE CI (10 k resamples) comparing the best model per feature set. Predictions are aligned on common sweeps via `_align_predictions()` before testing.

**SHAP agreement:** Top-k feature overlap across ElasticNet and LightGBM per feature set (Polynomial excluded — SHAP is on expanded degree-2 features, not comparable to original-feature SHAP).

---

### 06 — Plots (`scripts/06_plots.py`)

**Input:** All saved CSVs from scripts 03–05 plus `features.parquet` and `metadata.parquet`
**Output:** `outputs/06_plots/` — ~40 figures covering all model results

Standalone script that regenerates every publication figure without retraining. Edit the **aesthetics block at the top** (DPI, font sizes, colours, RPM colormap step) to restyle all figures at once.

Figures produced:
- **E1** CV score violin plots (one per feature set)
- **E2** Mean ± std RMSE bar chart (all models)
- **M1a/M1b** Predicted vs true κ scatter — CV out-of-fold and holdout
- **M2** Coefficient / feature importance log-bar charts
- **M3** Per-fold CV R² heatmap
- **M4** Holdout residual plots
- **S1** SHAP mean |value| importance bars
- **S2** SHAP beeswarm plots
- **S3** Sensor contribution (AE vs US share of SHAP importance, combined models only)

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

**8. SHAP for Polynomial is misleading** — open, interpretation caveat

`src/ceramicspeed/modelling.py:compute_shap_values` runs `LinearExplainer` on the **Ridge step** of the Polynomial pipeline. Because `PolynomialFeatures` has already expanded the input at that point, the SHAP values are attributed to degree-2 expanded terms (`rms²`, `rms × kurtosis`, etc.), not to the original features.

Two concrete consequences:

1. **Cross-model SHAP agreement** in `05_holdout_tests.py` compares top-k feature names across all three model types. Polynomial's top features will be things like `rms^2 kurtosis` while ElasticNet and LightGBM return `rms`, `kurtosis`. The overlap will be zero by construction — the agreement table for Polynomial is meaningless as currently computed.

2. **Individual SHAP importance plots** show expanded terms whose magnitude cannot be compared to ElasticNet coefficients or LightGBM SHAP values, even for the same underlying feature.

A proper fix would require collapsing each expanded term's SHAP value back to the original feature it derives from — summing all terms `f_i × f_j` back to `f_i` and `f_j`. This is non-trivial because interaction terms contribute to two features simultaneously. The pragmatic alternative is to **exclude Polynomial from the cross-model SHAP agreement table** and note in the paper that Polynomial SHAP is reported on expanded features.

Current status: a NOTE comment is in `scripts/05_holdout_tests.py`; the computation still runs and the agreement scores will be misleadingly low for Polynomial.

**9. Publication reporting gaps** ✓ Fixed 2026-05-18

Bootstrap CIs on absolute holdout R², MAE, RMSE added to `05_holdout_tests.py` via `bootstrap_metric_ci()` in `src/ceramicspeed/evaluation.py`. Results saved to `outputs/05_holdout_tests/tables/holdout_metrics_with_ci.csv`.

---

### Low severity

**10. Signal cleaning disabled by default** — resolved

`signal_cleaning.enabled: true` in `config.yaml` (was previously `false` in an older version; now correctly enabled).

**11. Several tunable parameters are hardcoded throughout the codebase:**

| Parameter | Location | Status |
|---|---|---|
| SHAP top-k agreement | `scripts/05_holdout_tests.py` | ✓ Moved to `evaluation.shap_top_k` in `config.yaml` |
| VIF threshold | `scripts/02_feature_analysis.py` | ✓ Moved to `feature_selection.vif_threshold` in `config.yaml` |
| Saturation flat tolerance | `src/ceramicspeed/cleaning.py` | open — see note below |
| Z-score spike threshold default | `src/ceramicspeed/loading.py` DEFAULT_SIGNAL_CLEAN_CFG | open — already overridable via config |

*Saturation flat tolerance note:* `detect_saturation` (`src/ceramicspeed/cleaning.py:182`) uses `flat_tolerance = 1e-10` as an absolute threshold on consecutive-sample differences. A sample pair is counted as "flat" if `|x[n+1] - x[n]| < 1e-10`. For a DAQ at 12.5 MHz with a ±10 V range and 16-bit resolution the LSB is ~305 µV — genuine ADC noise between samples will nearly always exceed 1e-10 V, so this threshold is effectively correct for raw voltage signals. However, if signals are ever normalised (e.g. divided by RMS) before saturation checking, the threshold becomes orders of magnitude too strict and saturation would never be detected. The tolerance should be documented as assuming raw-voltage units. Note: `check_saturation` is currently disabled in `config.yaml`, so this has no effect on current runs.

**12. No error handling for malformed HDF5 files** ✓ Fixed 2026-05-18

~~A single malformed file crashes the parallel worker.~~

`load_and_process_file` in `src/ceramicspeed/loading.py` now wraps the `load_hdf5_file` call in a try-except, logs the error, and returns empty results so the parallel job continues.

---

## Prioritised Recommendations

| Priority | Fix | Status | Impact |
|---|---|---|---|
| 1 | Per-fold top-k in `train_polynomial_cv` — fold-local feature selection | ✓ Fixed 2026-05-18 | High |
| 2 | Warn when viscosity fallback is applied | ✓ Fixed 2026-05-18 | High — data integrity |
| 3 | Log combined model data loss (sweep count before/after inner join) | ✓ Fixed 2026-05-18 | High |
| 4 | Move `CORR_MIN` and `VIF_THRESHOLD` to `config.yaml` | ✓ Fixed 2026-05-18 | Medium |
| 5 | Stratified CV by κ quantile bins | ✓ Fixed 2026-05-18 | Medium |
| 6 | Enable holdout-level tests for cross-feature-set pairs | ✓ Fixed 2026-05-18 | Medium |
| 7 | Document pre-filter vs band-filter design intent | ✓ Documented in config 2026-05-18 | Medium |
| 8 | HDF5 error handling with per-file logging | ✓ Fixed 2026-05-18 | Low-Medium |
| 9 | Move SHAP top-k to `config.yaml` | ✓ Fixed 2026-05-18 | Low |
| 10 | Add confidence intervals on absolute holdout performance metrics | ✓ Fixed 2026-05-18 | High — publication |
| 11 | Exclude Polynomial from cross-model SHAP agreement table (expanded-feature SHAP not comparable) | ✓ Fixed 2026-05-18 | Medium — publication |
| 12 | Sensor label mismatch (UL→US) in `feature_sets` in scripts 05 and 06 causing US models to be silently dropped from within-feature-set tests and CV violin plots | ✓ Fixed 2026-05-19 | High — correctness |
| 13 | RPM duplicate rows in metadata causing `c` shape mismatch in holdout scatter (2 sensor rows per sweep) | ✓ Fixed 2026-05-19 | Medium |
| 14 | `plt.cm.get_cmap` deprecation warning in `06_plots.py` | ✓ Fixed 2026-05-19 | Low |
| 15 | Stale `_ul` and `bayesianridge` output files from old runs | ✓ Cleaned 2026-05-19 | Low |
| 16 | Saturation flat tolerance assumes raw-voltage units — document or make relative | not pursued — raw signals are never normalised, threshold is correct as-is | Low |
