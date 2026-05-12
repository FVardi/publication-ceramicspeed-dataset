# Pipeline Analysis — Strengths, Weaknesses & Improvement Recommendations

**Project:** CeramicSpeed — Lubrication condition monitoring via AE and UL  
**Target variable:** κ (kappa) — ISO 281 viscosity ratio  
**Date:** 2026-05-12

---

## Pipeline Overview

The pipeline predicts lubrication regime (κ) from acoustic emission (AE) and ultrasound (UL) bearing sensor data. It runs in four numbered scripts plus optional EDA scripts.

```
HDF5 files
    ↓ 01_feature_generation.py
features.parquet + metadata.parquet
    ↓ 02_feature_analysis.py
feature_selection.json  (+ correlation/VIF/PCA figures)
    ↓ 03_modelling.py
9 fitted models + holdout predictions + SHAP
    ↓ 04_evaluation.py
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

### 03 — Modelling (`scripts/03_modelling.py`)

**Input:** `features.parquet`, `metadata.parquet`, `feature_selection.json`  
**Output:** Holdout prediction CSVs, SHAP CSVs, figures, `best_params.json`

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

### 04 — Evaluation (`scripts/04_evaluation.py`)

**Input:** `features.parquet`, `metadata.parquet`, `feature_selection.json`, `best_params.json`, holdout prediction CSVs  
**Output:** Repeated CV scores, performance table, stat test tables, SHAP agreement tables, figures

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

**1. Polynomial top-k feature selection is biased**

In `scripts/04_evaluation.py:262` and equivalently in script 03, Pearson correlation is computed on the **full training set** to select the top-k features before the outer fold split. This means feature selection can see validation data, invalidating the nested CV for the Polynomial model. The top-k selection must be performed inside the outer fold using only that fold's training indices.

**2. Viscosity fallback is silent**

`src/ceramicspeed/loading.py` applies Keratech 22 viscosity constants whenever the measurement file lacks viscosity metadata. No warning is printed, no flag is set in the output. Since κ is entirely derived from viscosity, systematically wrong values could be present for a subset of experiments without any trace.

**3. Pre-filter / band-filter inconsistency**

Broadband UL features are extracted from the pre-filtered signal (0–20 kHz), but AE band-specific features are extracted from the raw cleaned signal. The two code paths diverge without documentation of the rationale, making the design easy to misread or accidentally break when modifying feature extraction.

**4. Combined model data loss is silent**

The inner join on (file, sweep) used to build the combined AE+UL feature matrix drops any sweep where one sensor is missing. There is no log of how many sweeps were lost, so combined models could be trained on substantially less data than single-sensor models without any visible signal.

**5. Correlation threshold hardcoded and unjustified**

`CORR_MIN = 0.1` in `scripts/02_feature_analysis.py` is the gate for feature retention. No reference or rationale is given. A threshold of 0.1 is permissive (retains weakly correlated features). It should live in `config.yaml` with a comment explaining the choice.

---

### Medium severity

**6. No κ regime stratification in CV**

`KFold` splits are random. Rare high-κ or low-κ samples can concentrate in one fold, making CV estimates unreliable for the tails of the distribution. `StratifiedKFold` on the three κ regime labels (< 0.5, 0.5–1.0, ≥ 1.0) would give more reliable and reproducible estimates.

**7. Cross-feature-set holdout tests are incomplete**

The Wilcoxon / Diebold-Mariano tests in `scripts/04_evaluation.py:536` only run if `len(y_true_a) == len(y_true_b)` and values are `allclose`. Because combined models have fewer samples (inner join), sensor-vs-combined comparisons are silently skipped — leaving the most practically interesting comparison unevaluated at the holdout level.

**8. SHAP for Polynomial is misleading**

`LinearExplainer` is applied to degree-2 expanded features (e.g., `rms²`, `rms × kurtosis`), not to the original features. These interaction-term importances are not directly comparable to the original-feature SHAP values from Elastic Net or LightGBM, and should not appear in the same cross-model agreement tables without a note.

**9. Publication reporting gaps**

- No confidence intervals on absolute model performance (only ΔRMSE between models is CI-bounded)
- No regime-level classification accuracy (κ < 0.5 / 0.5–1.0 / ≥ 1.0 confusion matrix or F1)
- No holdout regime-stratified test — the most practically relevant question (does the model generalise to the right lubrication regime?) is not answered with a dedicated metric

---

### Low severity

**10. Signal cleaning disabled by default**

`signal_cleaning.enabled: true` appears in a comment but `enabled: false` is the config default. Whether this is intentional (data is already clean) or a development shortcut is not documented.

**11. Several tunable parameters are hardcoded throughout the codebase:**

| Parameter | Location | Value | Should be |
|---|---|---|---|
| PCA variance threshold | `src/ceramicspeed/analysis.py` | 95% | `config.yaml` |
| SHAP top-k agreement | `src/ceramicspeed/evaluation.py` | 10 | `config.yaml` |
| Saturation flat tolerance | `src/ceramicspeed/cleaning.py` | 1e-10 | `config.yaml` |
| VIF threshold | `src/ceramicspeed/analysis.py` | hardcoded | `config.yaml` |
| Z-score spike threshold | `src/ceramicspeed/features.py` | 6.0 | `config.yaml` |

**12. No error handling for malformed HDF5 files**

`src/ceramicspeed/loading.py` has no try-except around h5py reads. A single malformed file crashes the parallel worker and loses all data from that job.

---

## Prioritised Recommendations

| Priority | Fix | Effort | Impact |
|---|---|---|---|
| 1 | Fix polynomial top-k selection — use only current fold's train indices | Medium | High — fixes biased CV |
| 2 | Warn (or error) when viscosity fallback is applied | Low | High — data integrity |
| 3 | Log combined model data loss (sweep count before/after inner join) | Low | High — silent data loss |
| 4 | Move `CORR_MIN` to `config.yaml`; justify threshold choice | Low | Medium |
| 5 | Add stratified CV by κ regime | Medium | Medium |
| 6 | Add regime classification metrics (confusion matrix / F1 per regime) | Medium | High — publication |
| 7 | Enable holdout-level tests for cross-feature-set pairs (relax alignment check) | Medium | Medium |
| 8 | Document pre-filter vs band-filter design intent | Low | Medium |
| 9 | Add try-except around HDF5 loads with per-file error logging | Low | Low-Medium |
| 10 | Move all hardcoded thresholds to `config.yaml` | Low | Medium — maintainability |
