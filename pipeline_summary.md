# CeramicSpeed Pipeline — Quick Summary

## Goal
Predict **κ (kappa)** — the lubrication film thickness ratio — from raw bearing sensor signals.

---

## Data
- **Format**: HDF5 files, one file per test run, multiple sweeps per file
- **Sensors**: two channels per sweep
  - `AE` — Acoustic Emission, effective bandwidth 0–190 kHz
  - `UL` — Ultrasound (heterodyned probe), effective bandwidth 0–10 kHz
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

### Optional band features
Physics-motivated bandpass sub-bands (disabled by default, enabled in `config.yaml`):
- AE: 0–50 kHz, 50–190 kHz
- UL: 0–4 kHz, 6–10 kHz

Each band produces a full copy of all 26 features with a label prefix, e.g. `AE_50-190kHz__mobility`.

---

## Models (4)

All models are trained per sensor (AE and UL separately). Evaluation uses 80/20 hold-out split + 5-fold KFold CV on the training set. Metrics: R², MAE, RMSE.

| Model | Type | Regularisation | Notes |
|---|---|---|---|
| **Elastic Net** | Linear | L1 + L2 | alpha and l1_ratio tuned via `ElasticNetCV`; sparse feature selection |
| **Bayesian Ridge** | Linear | Empirical Bayes | Self-tunes alpha/lambda via EM; no grid search needed |
| **Polynomial (deg 2)** | Linear + interaction terms | Ridge (L2) | `PolynomialFeatures` → `Ridge`; alpha tuned via `RidgeCV` |
| **LightGBM** | Gradient boosted trees | L1 + L2 on leaves | Early stopping per fold; final model uses mean best iteration |

Feature importances (linear: signed coefficients; LightGBM: split importance) are stored in `ModelResult` for inspection.
