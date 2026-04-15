# Research Summary: AE and Passive Ultrasound for Lubrication Condition Monitoring

## Research Goal

The central goal is to determine whether simultaneously acquired acoustic emission (AE) and passive ultrasound (UL) signals can be used to predict the ISO 281 viscosity ratio κ — a physics-grounded, continuous index of lubrication adequacy in rolling element bearings — and whether combining both sensor modalities improves prediction accuracy over either sensor used alone. Because both sensors are passive and non-invasive, the work is motivated by practical industrial retrofit: no bearing modification is required.

## Research Gap

Inadequate lubrication is responsible for an estimated 30–80% of rolling element bearing failures, yet reliable online lubrication condition monitoring (LCM) remains unsolved. Prior work by Jakobsen et al. established κ regression from vibration and passive ultrasound using LASSO and neural networks. The present study addresses two gaps those works leave open: (1) AE has never been applied to κ regression; and (2) AE and passive ultrasound have never been studied simultaneously in a unified experimental and regression framework. It is therefore unknown whether the two modalities provide complementary κ information or largely redundant information.

## Hypotheses

The primary hypothesis is that a multi-sensor model combining AE and UL features will outperform either single-sensor model. A secondary hypothesis is that operating conditions (speed and temperature) substantially confound the raw feature–κ correlations, and that partial correlation analysis controlling for these variables will reveal a reduced but physically interpretable set of robust κ indicators.

## Experimental Setup

Experiments are conducted at CeramicSpeed (Holstebro, Denmark) on a custom bearing test stand fitted with a single deep-groove ball bearing (SYJ25, bore 25 mm, pitch diameter 38 mm, 9 rolling elements) lubricated with Keratech 22 oil. A wideband piezoelectric AE sensor and a heterodyned passive ultrasound probe are mounted simultaneously on the bearing housing and sampled at 1.6 MHz. Data are collected across two temperature set-points (40 °C and 75 °C) and five rotational speeds (500–2500 rpm), spanning boundary, mixed, and full-film lubrication regimes. The resulting κ range is 0.16–1.55. After quality filtering, approximately 1 600 sweep–sensor pairs are retained. The κ value for each sweep is derived analytically from bearing geometry, lubricant viscosity–temperature properties, and the measured operating conditions using the ISO 281 / Walther framework — no direct film thickness reference is available.

## Signal Processing and Feature Selection

Each signal is cleaned (NaN/Inf interpolation, spike removal, clipping and saturation detection) before feature extraction. Because the UL sensor is a heterodyned probe with effective content below approximately 10 kHz, a sensor-specific pre-filter is applied to UL signals before feature computation, preventing the 790 kHz of out-of-band noise from biasing spectral features such as centre frequency and RMS frequency. In addition to broadband processing, physics-motivated bandpass-filtered variants are extracted: for AE, bands at 0–50 kHz, 50–200 kHz, and 200–800 kHz; for UL, bands at 0–4 kHz and 6–10 kHz (bracketing the VFD switching artefact at ~5 kHz). From each signal, 26 features are computed comprising 12 time-domain statistics (including Hjorth mobility and complexity) and 14 frequency-domain statistics (spectral shape, flatness, kurtosis, normalised bandwidth). Feature selection applies a Spearman rank correlation ranking followed by a variance inflation factor and pairwise correlation filter, retaining 10 AE and 11 UL features for modelling.

## Modelling

Five model families are benchmarked across three sensor configurations (AE only, UL only, AE + UL combined): linear regression, Bayesian ridge regression, polynomial regression, LightGBM gradient boosting, and a shallow neural network. Models are tuned via 5-fold cross-validation on an 80% training split and evaluated on a held-out 20% test set. The best hold-out performance is achieved by LightGBM on AE features alone (R² = 0.64, RMSE = 0.19 κ-units). Fusing AE and UL features does not improve upon AE alone (R² = 0.60 combined), a negative fusion result attributed to the limited independent κ information carried by the heterodyned acoustic channel relative to AE.

## Scope and Limitations

The study is intentionally exploratory and is scoped to a single bearing type with a single lubricant under controlled laboratory conditions. Generalisation to other bearings, lubricants, or load conditions is not claimed. The mixed lubrication regime is underrepresented due to the discrete temperature set-points used. The κ reference is model-derived rather than directly measured. Speed-robust modelling and multi-bearing validation are identified as directions for future work.
