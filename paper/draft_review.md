# Critical review of the draft (paper/main.tex)

Reviewed 2026-06-10 against the current pipeline outputs in `outputs/`. Results are still changing, so no numbers were edited in the .tex — but every place a number appears is flagged below so nothing is missed when you freeze the pipeline.

---

## 1. Major issues (address before anything else)

### 1.1 The draft's numbers no longer match `outputs/` — and the central narrative may reverse

The draft's headline story is "AE dominates, fusion gain is marginal and unconfirmed on hold-out." The current outputs contradict both halves:

| Quantity | Draft | Current `outputs/` |
|---|---|---|
| LightGBM AE, HO R² / RMSE | 0.951 / 0.078 | 0.951 / 0.078 ✓ |
| LightGBM Combined, HO R² / RMSE | 0.953 / 0.076 | **0.960 / 0.071** |
| LightGBM US, HO R² / RMSE | 0.600 / 0.222 | **0.805 / 0.155** |
| AE–US gap (LightGBM ΔR²) | 0.35 | **0.146** |
| Fusion CV corrected t-test | p = 0.024 | p = 3.9e-5 |
| Fusion hold-out Wilcoxon | p = 0.88 (n.s.) | **p = 3.2e-9 (significant)** |
| Fusion hold-out Diebold–Mariano | p = 0.28 (n.s.) | **p = 1.8e-6 (significant)** |
| Fusion ΔRMSE (bootstrap 95% CI) | 0.0011, CI includes 0 | **0.0070, CI [0.0042, 0.0097] excludes 0** |
| ElasticNet US, HO R² | 0.396 | 0.628 |
| ElasticNet Combined, HO R² | 0.830 | 0.843 |
| Polynomial US, HO R² | 0.477 | 0.701 |
| Polynomial Combined, HO R² | 0.803 | 0.782 |
| US features retained | 6 | **12** |
| Top US feature | broadband spectral mean, ρ=0.663 | **US_10–20kHz peak, ρ=0.723** |

If the current outputs hold, the paper's conclusion flips from "fusion unconfirmed" to "fusion gives a small but statistically confirmed gain." Highlights, abstract, contribution 5, §7.4 results, §8.2 discussion, and conclusion items 2–3 all encode the old story. Do not polish any of these sections until the pipeline is frozen.

Two oddities in the new numbers worth checking before freezing: (a) Polynomial Combined is now *worse* than Polynomial AE in CV (0.166 vs 0.164) — plausibly the top-20 feature cap, but verify; (b) conclusion item 4 claims the ElasticNet–Polynomial difference is not significant on Combined — currently it is highly significant (p ≈ 1.6e-6, ElasticNet better).

**Recommendation:** stop hand-typing numbers. Extend the pipeline to emit `.tex` macros or table fragments (as already done for the appendix tables via `\IfFileExists`) and `\input` them everywhere — abstract, highlights, contributions, results, discussion, conclusion. This eliminates the entire class of stale-number errors.

### 1.2 AE bandwidth contradiction — the paper's headline feature may be out-of-band

`config.yaml` (`sensors: AE: f_max: 190_000`) and `docs/pipeline/pipeline_summary.md` both state the AE channel's **effective bandwidth is 0–190 kHz**. Yet the paper's single most important finding is that **1–2 MHz complexity** is the dominant predictor, framed physically as "high-frequency micro-fracture events" (§5, §7, conclusion item 1).

If the sensor (or its analog chain) rolls off near 190 kHz, content at 1–2 MHz is noise floor, electronic interference, or aliasing — which can still correlate with κ via RPM/temperature-dependent noise, but the physical interpretation collapses, and a reviewer who spots this will reject the framing. Before submission you need one of:

- the AE sensor datasheet showing genuine response to 2 MHz (then fix the config comment), or
- evidence from the measured spectra (e.g. `outputs/eda/eda_spectrum_ae_1000-2000khz.png`) that the 1–2 MHz band contains structured, operating-condition-dependent signal well above the instrument noise floor, discussed explicitly in the paper, or
- re-framing: drop the micro-fracture interpretation and report the band's predictive value with an honest caveat.

Related: §4.3 states a 12.5 MHz shared sample rate; the config comment mentions "~5 MHz DAQ bandwidth." Reconcile.

### 1.3 κ circularity needs a baseline, not just a caveat

κ is computed *deterministically* from measured RPM and temperature (Eqs. 1–2). A model given RPM and temperature directly would achieve R² ≈ 1 by construction. So "R² = 0.95 from AE features" is only meaningful relative to how much of that is the features acting as proxies for (n, T) — which §6.2/§8.3 admit but don't quantify.

Two cheap, high-value additions:

1. **Oracle baseline row:** regression from measured (RPM, T) → κ. This bounds what any operating-condition proxy can achieve and makes the contribution honest: "AE features recover κ *without* tachometer or thermometer."
2. **Proxy quantification:** regress the top AE features → RPM (and T). If 1–2 MHz complexity predicts RPM with R² ≈ 0.95, say so and frame accordingly; partial correlation with κ controlling for (n, T) would be even better. §6.2's todo already points here — this is the single analysis reviewers are most likely to demand.

### 1.4 Sweep-level random split likely leaks — blocked validation needed

Sweeps are consecutive segments of a continuous staircase protocol: adjacent sweeps share nearly identical (RPM, T, κ) and highly correlated signals. A *random* 80/20 split (§7.1) and randomly re-seeded 5-fold CV mean nearly every hold-out sweep has near-duplicate neighbours in training. The hold-out therefore measures interpolation, not generalisation, and R² = 0.95 is likely optimistic. The conclusion's future-work line ("investigating temporal autocorrelation and its effect on hold-out validity") concedes the point — MSSP reviewers will not accept it deferred.

**Recommendation:** add a blocked split as a robustness experiment — e.g. leave-one-temperature-block-out (13 folds) or contiguous-time blocking. If performance holds, the paper gets much stronger; if it drops, you need to know now. This pairs naturally with 1.3.

### 1.5 Single session

`config.yaml` `file_patterns` includes only `scope_20260424_115654`, while `data/` holds four acquisition files (latest 2026-05-29). If the published results use one session, say so explicitly in Limitations; better, use a later session as a true out-of-session hold-out — that would substantially de-fang both 1.3 and 1.4.

---

## 2. Consistency and correctness errors (mechanical, fix when freezing)

1. **Operating-condition ranges disagree.** Abstract: 1–5985 rpm, 33–100 °C. Contribution 1 and §4.2: 100–3000 rpm, 40–100 °C. Presumably measured vs commanded; pick one convention, state the other once. (Also `config.yaml` discards sweeps below 50 rpm — if the abstract's "1 rpm" survives filtering, check it.)
2. **Candidate feature counts.** Paper: 26 × 4 = 104 AE, 26 × 3 = 78 US. `feature_selection.json`: **92 AE, 60 US** candidates (23/band AE, 20/band US) — some features are evidently dropped by a quality filter before ranking. Either document the filter and the post-filter counts, or fix the bookkeeping. The appendix text ("passed the initial data quality filter") hints at this but the body never explains it.
3. **US retained count.** Abstract has `\todo{update}`; §8.2 and conclusion say 6. Current value: 12 (44 combined).
4. **"Relative RMSE ≈ 15% of the target range"** (§7.4) is wrong: 0.078 / 1.64 ≈ **4.8%** of range. 15% is relative to the *mean* (0.078/0.51). Fix the denominator wording.
5. **US band contradiction.** §8.1 says US useful content is "below approximately 10 kHz"; background §3.2 says the heterodyned baseband is 0–20 kHz; the current ranking has the **10–20 kHz** sub-band on top. Reconcile after freezing.
6. **Broken sentence/heading** in §7.4: "*Sensor fusion yields a marginal, statistically significant.*" — incomplete.
7. **§4.2 mangled text:** "This, therefore, results in This is repeated for each of 13 temperature blocks…".
8. **"representable"** → "representative" (§8.4).
9. **Two figure captions are literally "Caption"** (`fig:op_conditions`, `fig:data_partition`).
10. **λ = 0.7** (§3.1, Walther paragraph): ASTM D341 has no λ. If this is the grease-thickener film-reduction factor or similar, name and cite it; otherwise remove.
11. **ISO 281 regime boundaries.** ISO 281 defines κ for the life-modification factor; the 0.5/1.0 boundary-mixed-full mapping is a convention layered on top. Soften "The ISO 281 standard characterises these regimes through…" and cite the actual source for the boundaries. Also note κ = 0.003 is far below ISO 281's stated validity range (κ ≈ 0.1–4) — worth one sentence.
12. **"8783 sweep–sensor pairs per channel"** (abstract) — it's 8 783 sweeps per channel (17 566 pairs total). Rephrase.
13. **US |r| column** in Table 4 is empty (`---`); values now exist in `feature_ranking_us.csv`.
14. **Stratification detail:** pipeline uses StratifiedKFold on κ *quantile bins*; §7.1 says "stratified to preserve the κ regime distribution." Make the text match the code.
15. **Wilcoxon caveat:** signed-rank on paired squared residuals tests the median difference, not RMSE; one sentence acknowledging this (the bootstrap ΔRMSE CI carries the RMSE claim) preempts a pedantic reviewer.

---

## 3. Missing content (the todos, prioritized)

**Blocking — needs information you must obtain:**

- §4.1/§4.3 sensor specs (AE sensor model/bandwidth, US probe model/heterodyne details, temperature and RPM sensing) — flagged "Need help from CeramicSpeed." Critical given issue 1.2.
- §4.1 lubricant quantity, temperature control method, RPM control method, rig photo.
- §4.1 VFD noise: switching frequency value and the filtering actually applied (ties to §5.1).
- §5.1 Signal cleaning is an empty bullet. Per `config.yaml`: NaN/Inf repair enabled; clipping/saturation checks and z-score outlier removal disabled; sweeps below 50 rpm discarded. Write this up honestly (including what was *not* done).
- Slipring: contribution 2 is a placeholder, §4.4 has a conditional todo, and `dev/exploration/kappa_vs_slipring.py` exists. Decide now whether slipring film-thickness data is in this paper. If yes, it upgrades κ from model-derived to partially validated — a big strengthening that touches §4.4, §8.4, and limitations. If no, delete contribution 2.
- §4.5 Dataset availability vs backmatter (Zenodo on acceptance) — duplicated; keep one.
- Funding todo (IFD?); affiliation "Department of …"; consider a non-gmail institutional email.

**Resolvable from existing outputs once results freeze:**

- Abstract US-retained count; Table 4 US |r|; §6.2 partial-correlation analysis; §6.3 PCA figure; op-conditions figure + caption; limitations items 3–4 (data gap, κ reference); all "update with new results" todos.

---

## 4. Structure and writing

- **Title** is ~30 words with two subtitles. Suggestion: "Regression of acoustic emission and passive ultrasound features to the ISO 281 viscosity ratio for lubrication monitoring of ball bearings."
- **Abstract** is methods-heavy: the final sentence (inventory of significance tests) belongs in §7, and the band lists can go. Target ~200 words; MSSP readers want design → headline numbers → fusion answer.
- **Highlights** in main.tex still carry the old numbers and must respect ≤85 chars after updating.
- **Contributions list**: five is too many given that 3–5 overlap. Consider merging 3+4 (AE performance and AE≫US) and making the dataset, the fusion answer, and (if included) the slipring the headline contributions.
- **Background §3.2** is good, but the heterodyne explanation is repeated in §3.2, §5 (US pre-filtering), and §8.1 — say it fully once, reference it thereafter.
- **§7 ordering** is sound. Add a residuals-vs-κ figure (exists as `outputs/06_plots/model_residuals_holdout.png`) to support the "spread increases in the mixed regime" claim in the Fig. 8 caption — currently asserted but not shown.
- **Discussion** lacks a comparison-to-literature paragraph with numbers: §8.4 claims to exceed Jakobsen et al. with a `\todo{True?}` — verify against their reported metrics before claiming it, and discuss *why* (wideband AE, sub-bands, dataset size).
- **Class/bibliography:** MSSP (Elsevier) wants `elsarticle` with `num-names`; you're on `article` + `unsrt`. Fine for drafting; budget time for conversion.
- `Introduction_revised.docx` appears to be an older variant of the intro already superseded by `sections/introduction.tex` — confirm and remove it from `paper/` to avoid editing the wrong version.

---

## 5. Suggested order of work

1. Freeze the pipeline (decide on sessions/data files, re-run scripts 01–06).
2. Auto-generate all result numbers/tables into the .tex (extend `copy_figures.py` or add a `copy_tables.py`).
3. Resolve the AE bandwidth question (1.2) — this decides the paper's framing.
4. Run the two robustness analyses: blocked split (1.4) and RPM+T oracle baseline / proxy quantification (1.3).
5. Rewrite abstract, highlights, contributions, §7.4, §8, conclusion against the frozen numbers.
6. Fill the setup section with CeramicSpeed's input; write §5.1 cleaning; decide slipring in/out.
7. Mechanical fixes from §2 of this review; then language pass; then elsarticle conversion.

Items 3 and 4 are the difference between a paper that survives review and one that gets a major-revision verdict on first pass; everything else is bookkeeping.
