# Reframing proposals: title, contributions, abstract, highlights

Proposals only — no tex edited. Macro names (`\res...`) kept so accepted text can drop straight in. The strategy throughout: lead with the two signal-physics/methodology findings (film-gated 1–2 MHz band; operating-point confound decomposition), use model performance as evidence, not headline.

---

## 1. Title

Current (~30 words, application-framed):
> Acoustic Emission and Passive Acoustic Regression to the ISO 281 Viscosity Ratio κ for Online, Non-Invasive Lubrication Condition Monitoring of Ball Bearings: A Controlled Dual-Channel Study

**Option A — leads with the physical finding (recommended):**
> Film-gated high-frequency acoustic emission and operating-point confounding in dual-channel lubrication condition monitoring of a ball bearing

**Option B — leads with the methodological question:**
> Separating lubrication-state information from operating-point proxies in simultaneous acoustic emission and passive ultrasound monitoring of ball bearings

**Option C — conservative, keeps κ-regression identity, trimmed:**
> Sub-band acoustic emission and passive ultrasound regression to the ISO 281 viscosity ratio in a ball bearing: film-gated high-frequency content and sensor complementarity

A/B signal "signal processing + mechanism" to the editor; C is closest to the current paper identity. All drop "online, non-invasive" (shown, not claimed, in the body) and the "A Controlled … Study" tail.

## 2. Contribution list (replaces current §Contributions)

1. **A film-gated high-frequency AE band is identified and validated.** Total-band-power analysis at fixed temperature — robust to VFD EMI by construction, since spectral redistribution conserves band energy — shows that 1–2 MHz AE content vanishes at a common lubrication state (κ ≈ \resKneeKappaCool–\resKneeKappaHot) rather than a common speed. Two film-mediated generation mechanisms (asperity emission vs. drive-induced bearing currents) are delineated with testable implications. SHAP attribution independently ranks 1–2 MHz complexity first across all three model families.
2. **A confound-decomposition methodology for acoustic LCM.** Because κ is determined by the operating point, any feature–κ regression risks being an operating-point soft sensor. A two-stage decomposition (features → RPM/T → ISO 281 mapping, recovering R² = \resTwoStageRsq of the direct \resHOrsqLgbAe) quantifies the proxy share, while within-RPM-step correlation (median ρ = \resWithinStepRhoHC with temperature at fixed drive state) isolates the genuine lubrication-channel sensitivity. This diagnostic is applicable to any condition-monitoring study with an operating-point-derived target.
3. **The AE–US complementarity question is answered quantitatively.** First simultaneous acquisition of calibrated wideband AE and heterodyned passive US on one bearing: AE strongly outperforms US (ΔR² = \resDrsqLgbAeUs), and fusion adds a small but statistically robust gain (ΔRMSE = \resDrmseAeComb, all four tests significant) — complementarity is real but modest, with the heterodyned baseband identified as the bottleneck.
4. **A reproducible dual-channel dataset and evaluation protocol.** \resNsweeps sweeps spanning boundary→full-film regimes (κ ∈ [\resKappaMin, \resKappaMax]) with analytic κ ground truth; repeated nested CV (\resNouterScores outer scores) with Nadeau–Bengio-corrected and Holm–Bonferroni-adjusted inference. Data, features, and pipeline released on Zenodo.

(Current contribution #2, slipring film-thickness measurement, is omitted pending the decision flagged in the gap analysis — it appears nowhere else in the paper.)

## 3. Abstract (~215 words, replaces current ~370)

> Inadequate lubrication causes an estimated 30–80% of rolling-element bearing failures, yet online, non-invasive lubrication condition monitoring (LCM) remains unsolved: lubrication-relevant signal content is weak and strongly confounded with operating conditions. This paper presents a controlled dual-channel study in which wideband acoustic emission (AE) and heterodyned passive ultrasound (US) are acquired simultaneously from a ball bearing across staircase speed sweeps (\resRpmMinMeas–\resRpmMaxMeas rpm) and 13 temperature blocks (\resTempMinMeas–\resTempMaxMeas °C), spanning the boundary, mixed, and full-film regimes, with the ISO 281 viscosity ratio κ as a continuous, physics-grounded regression target. Sub-band decomposition reveals a film-gated high-frequency AE band: 1–2 MHz power vanishes at a common lubrication state rather than a common speed, and its leading feature dominates model attributions across all model families. Gradient boosting on \resNretAe selected AE features attains hold-out R² = \resHOrsqLgbAe; a two-stage decomposition shows most of this performance is recoverable from acoustically inferred operating points alone, while within-speed-step analysis confirms a genuine, temperature-mediated lubrication channel beyond the operating-point proxy. Fusing AE with US yields a small but consistently significant improvement, answering the long-argued but previously untested question of AE–US complementarity: the heterodyned US channel carries modest non-redundant κ information. All results are validated by repeated nested cross-validation with corrected significance tests.

Key cuts vs. current: the feature-count arithmetic, the selection-procedure detail, the full battery of p-values and CI bounds (kept qualitative: "consistently significant"), and the test-name list — all remain in the body.

## 4. Highlights (≤85 chars incl. spaces, counted)

1. `Simultaneous wideband AE and passive US regressed to ISO 281 viscosity ratio` (77)
2. `1–2 MHz AE power vanishes at a common lubrication state, not a common speed` (76)
3. `Two-stage decomposition quantifies operating-point proxy share of kappa models` (79)
4. `Within-speed-step analysis isolates a genuine temperature-mediated film channel` (80)
5. `AE–US fusion gain is small but significant: complementarity confirmed, modest` (79)

## Open choices

- Title: A, B, or C (or a hybrid).
- Whether contribution #2's "diagnostic methodology" claim is pitched as generalisable (stronger for MSSP, slightly bolder) or study-specific (safer).
- Abstract keeps one headline number (R²) — could add ΔRMSE for fusion if you prefer two.
