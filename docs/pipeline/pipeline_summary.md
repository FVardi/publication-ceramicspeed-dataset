# CeramicSpeed Pipeline — Quick Summary

## Goal
Predict **κ (kappa)** — the lubrication film thickness ratio — from raw bearing sensor signals.

---

## Data
- **Format**: HDF5 files, one file per test run, multiple sweeps per file
- **Sensors**: two channels per sweep
  - `AE` — Acoustic Emission, effective bandwidth 0–190 kHz
  - `US` — Ultrasound (heterodyned probe), effective bandwidth 0–20 kHz (HDF5 channel key: `UL`)
- **Processing**: files loaded in parallel (joblib, all CPU cores)

---

## Features (26 per signal)

### Time-domain (12)
| Feature | Description |
|---|---|
| `peak` | Half peak-to-peak amplitude |
| `rms` | Root mean square |
| `std` | Standard deviation |
| `variance` | Signal variance |
| `skewness` | Third standardised moment |
| `kurtosis` | Fourth standardised moment |
| `crest_factor` | peak / rms |
| `shape_factor` | rms / mean absolute value |
| `impulse_factor` | peak / mean absolute value |
| `margin_factor` | peak / mean sqrt amplitude² |
| `mobility` | Hjorth mobility (frequency spread proxy) |
| `complexity` | Hjorth complexity (waveform complexity proxy) |

### Frequency-domain (14)
Computed from one-sided FFT magnitude spectrum.

| Feature | Description |
|---|---|
| `dominant_frequency` | Frequency of peak power |
| `spectral_mean` / `spectral_std` | Mean / std of magnitude spectrum |
| `spectral_skewness` / `spectral_kurtosis` | Shape of spectral distribution |
| `center_frequency` | Spectral centroid (power-weighted mean frequency) |
| `rms_frequency` | RMS of frequency weighted by power |
| `peak_frequency` | 4th-order spectral moment ratio |
| `spectral_flatness` | Geometric / arithmetic mean of spectrum (tonality proxy) |
| `frequency_weighted_std` | Std of frequency axis, power-weighted |
| `normalized_frequency_std` | frequency_weighted_std / center_frequency |
| `frequency_skewness` / `frequency_kurtosis` | Shape of power-weighted frequency distribution |
| `normalized_bandwidth` | Fractional bandwidth proxy |

### Band features
Physics-motivated bandpass sub-bands (enabled in `config.yaml` under `frequency_bands`):
- AE: 20–500 kHz, 500–1000 kHz, 1000–2000 kHz
- US: 0–10 kHz, 10–20 kHz (applied to the 0–20 kHz pre-filtered signal)

Each band produces a full copy of all 26 features with a label prefix, e.g. `AE_20-500kHz__rms`, `US_0-10kHz__mobility`.

---

## Models (3)

Three model types × three sensor configurations (AE only, US only, AE+US combined) = 9 models total.
Feature selection retains 32 AE features and 12 US features (44 combined) after correlation filtering and redundancy reduction.
Evaluation uses 80/20 hold-out split + repeated nested 5-fold StratifiedKFold CV on κ quantile bins (5 repeats = 25 scores per model). Metrics: R², MAE, RMSE.

| Model | Type | Regularisation | Notes |
|---|---|---|---|
| **Elastic Net** | Linear | L1 + L2 | alpha and l1_ratio tuned via `ElasticNetCV` inside each outer fold |
| **Polynomial (deg 2)** | Linear + interaction terms | Ridge (L2) | top-k features selected per fold; alpha via `RidgeCV` inside each outer fold |
| **LightGBM** | Gradient boosted trees | L1 + L2 on leaves | Optuna HP search (20 trials) with early stopping inside each outer fold |

Feature importances (linear: signed coefficients; LightGBM: SHAP values via TreeExplainer) are saved per model.

---

## Scripts

| Script | Role |
|---|---|
| `01_feature_generation.py` | HDF5 → `features.parquet` + `metadata.parquet` |
| `02_feature_analysis.py` | Correlation filtering, VIF, redundancy reduction → `feature_selection.json` |
| `03_evaluation.py` | Repeated nested CV → CV score distributions, performance table |
| `04_modelling.py` | Holdout evaluation, SHAP, coefficient tables → all model outputs |
| `05_holdout_tests.py` | Significance tests (Nadeau-Bengio, Wilcoxon, Diebold-Mariano, bootstrap CIs) |
| `06_plots.py` | Standalone figure regeneration from saved CSVs — no retraining needed |
