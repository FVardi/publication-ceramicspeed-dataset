# Research Summary: AE and Passive Ultrasound for Lubrication Condition Monitoring

## Research Goal

The central goal is to determine whether simultaneously acquired acoustic emission (AE) and passive ultrasound (US) signals can be used to predict the ISO 281 viscosity ratio κ — a physics-grounded, continuous index of lubrication adequacy in rolling element bearings — and whether combining both sensor modalities improves prediction accuracy over either sensor used alone. Because both sensors are passive and non-invasive, the work is motivated by practical industrial retrofit: no bearing modification is required.

## Research Gap

Inadequate lubrication is responsible for an estimated 30–80% of rolling element bearing failures, yet reliable online lubrication condition monitoring (LCM) remains unsolved. Prior work by Jakobsen et al. established κ regression from vibration and passive ultrasound using LASSO and neural networks. The present study addresses two gaps those works leave open: (1) AE has never been applied to κ regression; and (2) AE and passive ultrasound (US) have never been studied simultaneously in a unified experimental and regression framework. It is therefore unknown whether the two modalities provide complementary κ information or largely redundant information.

## Hypotheses

The primary hypothesis is that a multi-sensor model combining AE and US features will outperform either single-sensor model. A secondary hypothesis is that operating conditions (speed and temperature) substantially confound the raw feature–κ correlations, and that partial correlation analysis controlling for these variables will reveal a reduced but physically interpretable set of robust κ indicators.

## Experimental Setup

Experiments are conducted at CeramicSpeed (Holstebro, Denmark) on a custom bearing test stand fitted with a single deep-groove ball bearing (SYJ25, bore 25 mm, pitch diameter 38 mm, 9 rolling elements) lubricated with Keratech 22 oil. A wideband piezoelectric AE sensor and a heterodyned passive ultrasound probe are mounted simultaneously on the bearing housing and sampled at 1.6 MHz. Data are collected across two temperature set-points (40 °C and 75 °C) and five rotational speeds (500–2500 rpm), spanning boundary, mixed, and full-film lubrication regimes. The resulting κ range is 0.16–1.55. After quality filtering, approximately 1 600 sweep–sensor pairs are retained. The κ value for each sweep is derived analytically from bearing geometry, lubricant viscosity–temperature properties, and the measured operating conditions using the ISO 281 / Walther framework — no direct film thickness reference is available.

## Signal Processing and Feature Selection

Each signal is cleaned (NaN/Inf interpolation, spike removal, clipping and saturation detection) before feature extraction. Because the US sensor is a heterodyned probe with effective content below approximately 20 kHz, a sensor-specific pre-filter (0–20 kHz lowpass) is applied to US signals before feature computation, preventing out-of-band DAQ noise from biasing spectral features such as centre frequency and RMS frequency. In addition to broadband processing, physics-motivated bandpass-filtered variants are extracted: for AE, bands at 20–500 kHz, 500–1000 kHz, and 1000–2000 kHz; for US, bands at 0–10 kHz and 10–20 kHz. From each signal, 26 features are computed comprising 12 time-domain statistics (including Hjorth mobility and complexity) and 14 frequency-domain statistics (spectral shape, flatness, kurtosis, normalised bandwidth). Feature selection applies a Spearman rank correlation threshold (|ρ| ≥ 0.1 and |r| ≥ 0.1) followed by a variance inflation factor and pairwise correlation filter, retaining 32 AE features and 12 US features for modelling.

## Modelling

Three model families are benchmarked across three sensor configurations (AE only, US only, AE + US combined): Elastic Net, polynomial regression (degree 2), and LightGBM gradient boosting. Models are evaluated via repeated nested cross-validation (5 outer folds × 5 repeats, κ-stratified) on an 80% training split and evaluated on a held-out 20% test set.

Key holdout results (LightGBM, 95% bootstrap CI):

| Configuration | R² | RMSE |
|---|---|---|
| AE only | 0.951 [0.941, 0.959] | 0.078 [0.071, 0.085] |
| US only | 0.805 [0.781, 0.827] | 0.155 [0.145, 0.165] |
| AE + US combined | 0.960 [0.951, 0.966] | 0.071 [0.065, 0.078] |

AE substantially outperforms US alone (ΔRMSE = 0.077, p < 0.001). The combined model outperforms AE alone by a modest but statistically significant margin (ΔRMSE = 0.007, 95% CI [0.004, 0.010], p < 0.001), indicating that US carries a small amount of κ information complementary to AE. Within each sensor configuration, LightGBM significantly outperforms both linear models (p < 0.001 for all pairs). Statistical testing uses the Nadeau-Bengio corrected repeated k-fold t-test with Holm-Bonferroni correction for within-feature-set comparisons, and Wilcoxon signed-rank, Diebold-Mariano, and bootstrap ΔRMSE CIs for cross-feature-set comparisons.

## Scope and Limitations

The study is intentionally exploratory and is scoped to a single bearing type with a single lubricant under controlled laboratory conditions. Generalisation to other bearings, lubricants, or load conditions is not claimed. The mixed lubrication regime is underrepresented due to the discrete temperature set-points used. The κ reference is model-derived rather than directly measured. Speed-robust modelling and multi-bearing validation are identified as directions for future work.
