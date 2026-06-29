# Paper writing status — section-by-section walkthrough

Updated 2026-06-12; new-pipeline findings appended 2026-06-29 (see banner below —
walkthrough Gates 1-8 and their dates are otherwise as last left on 2026-06-16
and have NOT been re-verified against the new pipeline). Legacy-pipeline numbers
regenerate via `scripts/07_paper_export.py`; figures via `scripts/06_plots.py` +
`scripts/copy_figures.py`. New-pipeline numbers regenerate via
`scripts/run_new_pipeline.py` (or 11-17 individually); no figure-export script
for the new pipeline beyond `14_new_method_figures.py`/`16_channel_mechanism.py`/
`17_feature_oc_table.py` yet.

## ⚠ 2026-06-29: new leak-free pipeline (scripts 11-17) — frozen results below are SUPERSEDED

> **Restructured 2026-06-30:** the new pipeline moved to `scripts/new/` (stage subfolders `signal_processing/` + `modelling/`; owns its own feature generation; outputs now under `outputs/new/`). Legacy moved to `scripts/legacy/` (root-path finders bumped a level, behaviour preserved). The script numbers below (11-19) are the OLD flat names. Current structure, run command, and results: `docs/pipeline/new_pipeline_summary.md`.

A long investigation (separate from the Gate 1-8 walkthrough below, triggered by
re-checking the LightGBM/Combined complementarity claim) found and fixed several
issues in the modelling protocol, using a new, simplified, leak-free pipeline
(`scripts/11-17`, orchestrated by `scripts/run_new_pipeline.py`) that runs
alongside the legacy pipeline (01-10, untouched). **The "Frozen results" block
immediately below this one reflects the OLD/legacy pipeline and is no longer the
best available evidence** — see "Current best results" further down for the
validated replacement numbers. The legacy pipeline and its frozen numbers are
left as-is (not deleted) for traceability; nothing in scripts 01-10 was modified.

What was found, in the order discovered:
1. **VIF feature selection flips significance for LightGBM/Combined** — initial
   trigger for the whole investigation.
2. **Candidate-pool leak**: `feature_selection.json`'s `"all_columns"` (used as
   the "full" feature set in the legacy pipeline and, until fixed, in the early
   new-pipeline scripts) is captured in `02_feature_analysis.py` *after* a
   whole-dataset, kappa-correlated Stage-1 filter has already run — so "full"
   was never actually the full candidate set. True counts: AE 42 (was 36), US 56
   (was 35) raw candidates. Fixed via `ceramicspeed.cleaning.true_candidate_columns`
   (target-independent cleaning only: NaN/Inf handling, constant-column removal,
   RPM filter — no kappa-correlation step), used by both `11_featureset_comparison.py`
   and `12_fullset_decomposition.py`.
3. **Single 80/20 holdout is underpowered/noisy**: re-evaluating via a single-pass,
   disjoint pooled `GroupKFold` (every acquisition-hold group held out exactly
   once across k=5 folds, selection/tuning/fitting redone fresh per fold) gives
   ~5x more groups for the downstream significance test without needing a
   Nadeau-Bengio correction (folds are disjoint). This reversed the original
   LightGBM/Combined-vs-AE null (non-significant on the single split) to highly
   significant — the single split was a power problem, not evidence of a true null.
4. **Operating-point twin leakage** (the most subtle finding): the RPM protocol
   sweeps up AND back down through the same speeds within each temperature block,
   so a given (rpm, temperature) operating point is often revisited by two or more
   separate acquisition holds ("twins"). Since kappa is a deterministic function of
   (rpm, temperature) alone here, twins have nearly identical kappa targets — the
   plain acquisition-hold grouping does not prevent twins from landing on opposite
   sides of a split, letting a model that has only learned to proxy the operating
   point score well on a held-out twin. Empirically ~24% of hold groups (164/686)
   have a twin elsewhere. Fixed via `ceramicspeed.grouping.merge_twin_groups`
   (merges all holds sharing a rounded RPM/temperature bin into one group before
   any split/CV). This is now the default; `--allow-twin-split` opts out for
   comparison. Effect: the apparent LightGBM/Combined "full beats selected"
   result (highly significant, p≈0.0000, under twin-leakage-prone grouping) was
   entirely a leakage artifact — once twins are merged, full vs selected for that
   one model/target combination is not significant (p=0.15). Every other
   full-vs-selected and complementarity result is essentially unaffected by this
   fix (still p≈0.0000), so this was a narrowly-scoped but important correction.
5. Hyperparameter tuning (`--tune-lightgbm`, single-level Optuna inside the
   training partition only) does not rescue or change any conclusion — fixed
   config hyperparameters are fine to use as the reported default.

**Net effect on the headline claims:**
- Complementarity (Combined significantly beats both AE-alone and US-alone) is
  *more* robust than ever: survives VIF choice, hyperparameter tuning, the
  candidate-pool fix, and the twin-leakage fix, for both models and feature sets
  (8/8 contrasts, p≈0.0000 in nearly all configurations). **Use this as the
  central, most defensible claim.**
- Full feature sets beat VIF-selected ones everywhere except LightGBM/Combined,
  where there is now no established difference (not "selected wins" — genuinely
  inconclusive). Report full as primary; selected as a secondary/interpretability
  check, stating the LightGBM/Combined exception explicitly.
- **Operating-point decomposition (`12_fullset_decomposition.py`) under the new,
  strict protocol shows two-stage (RPM+temperature-only) kappa R² >= direct kappa
  R² for ALL THREE channels** (AE: 0.877 vs 0.876; US: 0.870 vs 0.844; Combined:
  0.921 vs 0.908) — i.e. **no residual kappa-regression performance was detected
  beyond pure operating-point soft-sensing**, for any channel. This is a material
  change from the legacy "Two-stage proxy: 0.826 of 0.834" frozen result (which
  implied a small but present residual for AE) and directly affects how
  Contribution 1 in the introduction should be framed (see "Sections likely
  needing rewriting" below). Plausibly not surprising in hindsight: lubricant
  degradation/contamination was never independently varied in this protocol, so
  there is no genuine extra lubrication signal beyond what RPM/temperature
  already determine for any model to find.
- §4.3 "Sub-band signal validation" removed again (second time — was removed
  then restored on 2026-06-15, see Gate 6 log below). Author's reasoning
  2026-06-29: (a) the kept "VFD harmonics lie below the 20 kHz band edge"
  argument has a real gap — the comb's 5th harmonic alone (~23.5 kHz) already
  exceeds 20 kHz, so it doesn't actually establish exclusion; the genuinely
  decisive argument (the AE coupler's hardware Butterworth band-pass,
  50 kHz-1 MHz, already stated in setup.tex) was never even cited; (b) the
  power-rises-with-RPM mechanical-origin evidence is redundant with the
  marginal/conditional correlation analysis already done elsewhere
  (12_fullset_decomposition.py / 17_feature_oc_table.py). Removed cleanly:
  grep confirms no other `\ref` to `subsec:band_validation` or `fig:band_power`
  anywhere in the paper; compiles clean (40pp, 0 errors, 0 undefined refs).
  Orphaned and NOT yet cleaned up: macros `\resRhoPowRpmLow`/`\resRhoPowRpmMid`
  in `paper/tables/results_macros.tex` (generated by `07_paper_export.py` lines
  ~175, 579-585) are now unused in prose; the "EMI" glossary entry in
  nomenclature.tex is also now unused. Low priority, cosmetic only.

Validated new-pipeline code: `scripts/11_featureset_comparison.py`,
`12_fullset_decomposition.py`, `13_group_paired_tests.py` (significance tests),
`14_new_method_figures.py`, `16_channel_mechanism.py`, `17_feature_oc_table.py`,
shared helpers `ceramicspeed.grouping` + `ceramicspeed.cleaning.true_candidate_columns`.
Default behaviour = the validated/recommended protocol (pooled GroupKFold,
twin-merged groups, true full candidate pool); `--single-split` and
`--allow-twin-split` are explicit opt-outs kept for comparison only.

### Current best results (new pipeline, pooled GroupKFold k=5, twin-merged, n=522 groups)

- LightGBM Combined holdout R² = 0.909 (full) / 0.911 (selected) — not significantly
  different (p=0.15); AE 0.832 (full) / 0.809 (selected); US 0.789 / 0.746
- Complementarity: Combined significantly beats AE and beats US, for both models
  and both feature sets — all 8 contrasts p≈0.0000 (outputs/13_group_paired_tests/complementarity_tests.csv)
- Full vs selected: full wins for ElasticNet (all 3 targets, p≈0.0000) and
  LightGBM AE/US (p≈0.0000); LightGBM Combined inconclusive (p=0.15) —
  outputs/13_group_paired_tests/full_vs_selected_tests.csv
- Operating-point decomposition: two-stage kappa R² >= direct kappa R² for AE,
  US, and Combined — no detectable residual beyond RPM/temperature soft-sensing
  (outputs/12_fullset_decomposition/tables/decomposition_summary.csv)
- True candidate feature counts: AE 42, US 56 (raw, target-independent cleaning only)

### Sections likely needing rewriting given the above (not yet done)

- **Introduction, Contribution 1** ("a decomposition which separates the
  predictive performance ... into an operating-point soft-sensing share and a
  residual variation consistent with changes in lubrication adequacy") — under
  the new protocol the residual is essentially null for all three channels; the
  claim needs to be reframed as "we quantify how much of the apparent
  kappa-regression performance is explainable by operating-point soft-sensing
  alone (essentially all of it under the strictest test), rather than ...".
- **Introduction, Contribution 2** (complementarity) — can be stated more
  confidently than before; this claim got *stronger*, not weaker.
- **Introduction, Contribution 3** ("under acquisition-grouped evaluation that
  prevents closely related measurement windows from crossing the
  training-evaluation boundary") — should be extended to describe the
  operating-point-twin leakage mode specifically (revisited RPM/temperature
  points via the up/down sweep), which is a more subtle and arguably more
  interesting leakage mode than simple near-duplicate windows within one hold.
- **Pipeline §4.3** — removed (done, this session).
- **Features / Modelling sections (current §5-6)** — these describe the LEGACY
  pipeline's protocol (repeated nested CV, ElasticNetCV/RidgeCV/poly families,
  VIF-selected feature counts 11 AE + 12 US). If the new pipeline becomes the
  paper's reported methodology, these sections need substantial rewriting, not
  touch-ups: single-level pooled GroupKFold (no nested CV), full feature sets as
  primary (42 AE / 56 US), twin-merged grouping, ElasticNet + LightGBM only.
- **Discussion** — needs to address the null decomposition residual honestly
  (frame as consistent with the controlled protocol design, not a sensing
  failure — lubricant state was never independently varied) and to discuss
  operating-point-twin leakage as a methodological finding in its own right.
- **Conclusion** — currently mirrors the legacy frozen results; needs
  re-verification against the new-pipeline numbers once the above sections are
  rewritten.
- This list is a flag, not yet executed — no section text below §4.3 has been
  rewritten for the new-pipeline findings yet.

## Frozen results (legacy pipeline; grouped splits, AE ≤1 MHz, US ≤100 kHz, iterative VIF@5) — SUPERSEDED, see above

- LightGBM Combined HO R² = 0.908 / RMSE 0.081; AE 0.834 / 0.110; US 0.653 / 0.158
- Fusion: ΔRMSE = 0.028 (~26%), significant on all four tests — headline finding
- AE vs US: hold-out significant, grouped-CV t-test inconclusive (p = 0.12) — report both
- Two-stage proxy: 0.826 of 0.834 (soft-sensor decomposition robust)
- Within-step lubrication channel: 20–500 kHz mobility, ρ = −0.56
- Retained features: 11 AE + 12 US; >1 MHz excluded (origin undetermined, no mechanism claims)
- Jakobsen comparison: parity under matched random splits (their NN 0.94 vs our old 0.95);
  grouped figures are first-of-kind — frame as protocol contribution, never "exceeds"

## Walkthrough gates

| Gate | Section | Status |
|---|---|---|
| 1 | Title / highlights / contributions | DONE (title T6: "Sensor complementarity and operating-point proxies…") |
| 2 | Abstract | DONE (EMI/noise-floor sentence removed per author) |
| 3 | Introduction | DONE (incl. 4 new MSSP citations: hua_dife_2025, jiang_mechanism_2025, pang_smia_2024, wang_fingerprint_2023) |
| 4 | Background | DONE (2026-06-12: §2.2 heterodyne rewrite applied — mechanism-neutral wording; "0–20 kHz baseband" claim removed; above-baseband energy retained as empirical, no origin attributed; \todo on 38–40 kHz sensing band remains) |
| 5 | Setup | IN PROGRESS (2026-06-15). DONE: load fixed (``2 N'' wrong — telem_mass_g = 61.2 kg mean ⇒ ≈600 N); temp-timing corrected per author (set-point changes at low-rpm staircase bottom ~250 rpm, NOT at 3000 peak — verified: 12 sv increases at 140–365 rpm); standstill reworded (no RPM=0 sweeps; only excluded warm-up segment); sub-40°C softened to controller-instability (tentative, author unsure); AE sensor+coupler bullets written from docs/ datasheets = **Kistler 8152C1 sensor (100–900 kHz, 48 dB) + 5125C coupler Butterworth BP 50 kHz–1 MHz** (author confirmed 8152C1). FIXED 2026-06-15: the ``1 MHz'' is the COUPLER low-pass, not the sensor — corrected in abstract, intro Jakobsen line, and intro contribution 3 (was ``the AE sensor's 1 MHz specification''). Background uses only general ranges (``tens of kHz into MHz'', ``≥50 kHz''), no fix needed. REMAINING (collaborator/author): rig photo, lubricant charge amount, US/temp/RPM/load sensor models, VFD-noise filter description, confirm load conversion |
| 6 | Pipeline + Features + Modelling | IN PROGRESS (2026-06-15). Clarified: the >1 MHz (1–2 MHz) band IS already excluded from current outputs (outputs/02_*, 04_* dated 06-12, 1–2 MHz absent; macros agree: NcandAe=42=14×3, NcandUs=56=14×4, TopAeFeat=500k–1MHz mobility). The stale 1–2 MHz refs in the prose came from OLD pre-restructure leftovers at outputs/ root (05-06) — NO pipeline re-run needed. DONE: pipeline candidate arithmetic fixed (was 14×4/14×3 backwards), Table tab:subbands fixed (dropped AE 1–2 MHz row; US now broadband 0–100 kHz + 0–10/10–20/20–100 kHz), sub-band prose rewritten, truncated §candidates sentence completed; modelling split rewritten to visit-grouped GroupShuffleSplit (+ folds wording + fig caption); SHAP prose+tab:shap_agree rebuilt to current data (dominant = 500k–1MHz mobility 0.205); features within-step feature → \resWithinStepFeat macro. Compiles (38pp). REMAINING: interpretation paragraphs (ranking/redundancy/PCA) still unwritten — need current vif_log/pca figures; regenerate nested_cv_splits.png (grouped) + decide band_power figure (author todo "obsolete"); tone §band_validation mechanism claim ("rotation-generated and gated by film") to satisfy "no mechanism claims"; SHAP numbers+agreement should move into 07_paper_export (hand-typed); ElasticNetCV grid values + "verify ElasticNetCV/RidgeCV" todo; source within-step negative-step fraction as macro (old 91% was the removed feature); NON-FATAL: appendix feature_ranking longtable throws "infinite glue shrinkage" warning. 2026-06-15 (author call): REMOVED obsolete §4.3 "Sub-band signal validation" (band_power_vs_rpm.png, 1–2 MHz) AND its discussion counterpart §disc_hf "Origin of the high-frequency AE content" (wholly about the excluded band). Cascade fixes: pipeline §4.2 ref, setup VFD note ref, intro contribution 3 ("characterised and excluded"→"excluded"), §disc_ae ("2 MHz"→"1 MHz", dropped dead refs). Orphaned macros (resRhoPowRpm*, resKneeRpm*, resKneeKappa*) now unused. LOST a still-valid point (20–500 kHz transfers across drives) — relocate if wanted. Then RESTORED §4.3 as a trimmed figure-light "Sub-band signal validation" (retained bands only: ρ=+0.93/+0.71 rise with speed = mechanical not EMI; 4.7 kHz comb below 50 kHz coupler HP; no mechanism claims). Figure band_power_vs_rpm.png STILL 3-band — author to regenerate: scripts/06_plots.py Plot D7, drop validation_extra_bands append (~line 836) → re-run 06+copy_figures. ρ macros from 07_paper_export. Compiles 38pp. PIPELINE §4 now DONE (2026-06-15): §4.1 spectrogram→chronological only (κ-sort removed); §4.2 sub-band prose + Table; §4.3 validation reworded (freq-separation + speed-dependence) and figure regenerated dense (per_step=10, 143 sweeps/window, 2-band) → ρ refreshed +0.91/+0.69, stale todo removed; §4.4 feature tables rewritten to interpretive descriptions + mobility/RMS-freq clarification; §4.5 candidate sets + combined-config foreshadowing. Compiles 39pp. Gate 6 REMAINING = §5 interpretation paragraphs (ranking/redundancy/PCA) + §6 (nested_cv_splits.png regen, ElasticNetCV grid, verify ElasticNetCV/RidgeCV, pred-vs-actual interp [2026-06-16: §5 interp paragraphs written from outputs (09 feature_oc_correlations.csv, 02 vif csvs, pca png), no hand-typed numbers. CORRECTED selection-method prose: was OLD unused pairwise (corr>=0.90 AND VIF>=10 greedy); actual = reduce_redundant_features_iterative (iterative relevance-aware VIF elimination, recomputed each round, threshold=5 per config). Rewrote Stage2 + redundancy + fig:vif caption. 06_plots.py:813 VIF line fixed 10->cfg(5); vif_log.png REGENERATED 2026-06-16 (standalone faithful repro of Plot D6 -> paper/figures/, dashed line now at 5; full 06_plots/ceramicspeed env NOT installed on this machine so ran a one-off, not the pipeline). inline todo removed. Compiles 39pp.] [2026-06-16 ENV+FIG: .venv now present (ceramicspeed editable + numpy/pandas/scipy/sklearn/matplotlib/lightgbm/shap/yaml); installed optuna 4.9 + added "optuna" to pyproject.toml deps (was undeclared). optuna only needed for a full model re-run (03/04), NOT for paper tasks since results are frozen. nested_cv_splits.png REGENERATED via NEW committed generator scripts/make_nested_cv_diagram.py (figure was hand-made, no generator existed in repo) now depicting the visit-grouped scheme (cells=visits, ticks=sweeps, whole-hold assignment, 3 tiers); modelling.tex inline todo removed. Also fixed §6 AE-US significance wording to "report both" (Wilcoxon sig / Nadeau-Bengio p=0.121 inconclusive), matching conclusion. Compiles 39pp. REMAINING §6: ElasticNetCV grid values, verify ElasticNetCV/RidgeCV (code read only), move SHAP numbers -> 07_paper_export.] [2026-06-16 §6 DONE: (1) ElasticNetCV/RidgeCV grids added to tab:model_families (ENet 9 alpha 1e-4..1e2 x 6 l1-ratios, inner 5-fold; Poly 15 log-spaced lambda 1e-3..1e4 RidgeCV-LOO) from config.yaml; grid todo removed. (2) Verified + rewrote the "HP selection within each outer partition" sentence (confirmed in 03_evaluation.py _enet_fold_score/_poly_fold_score: inner ElasticNetCV/RidgeCV fit on outer-train rows only); "verify" todo removed. (3) SHAP prose top-3 now macro-driven: added resShapTop/Second/Third {Feat,Val} to 07_paper_export.py (reads shap_importance_lightgbm_ae.csv) and surgically appended the 6 macros to committed results_macros.tex. CAUTION: a FULL 07_paper_export run on THIS machine flips 8 macros to ?? (missing inputs: resNcandAe/Us, resRhoRpm/TempKappa, removed 1-2MHz knee/relRmse) -> did NOT adopt full regen, surgical add only; need complete outputs/ for a clean full run. tab:shap_agree TABLE now AUTOMATED 2026-06-16: 07_paper_export.py builds table_shap_agreement_tabular.tex from the three shap_importance_*_ae.csv (base features in top-k of >1 model; ranks = position in each model full importance list, poly incl. interaction/squared terms; sort: count desc, rank-sum asc, LGB rank). Generated content byte-for-byte matches the old hand table (cosmetic: full "500~kHz--1~MHz" + "spectral kurtosis"). modelling.tex now \inputs it via IfFileExists; caption corrected (poly ranks are full-list positions, not base-only); SHAP \todo fully removed. New committed file paper/tables/table_shap_agreement_tabular.tex. Compiles 39pp. GROUPING NOTE: repeated nested CV groups = contiguous staircase holds via _derive_hold_groups (new group when round(rpm/100) step changes); GroupShuffleSplit for hold-out AND outer folds (RANDOM_STATE+r per repeat); inner HP-CV is plain 5-fold (not grouped) but only affects HP choice. Compiles 39pp.]; within-step neg-fraction macro: now 95% for 20–500 kHz mobility) | [2026-06-29: §4.3 "Sub-band signal validation" REMOVED AGAIN — author judged the restored 2026-06-15 version filler given a stronger physical argument was already available but uncited (AE coupler hardware band-pass 50 kHz-1 MHz, setup.tex) and the power-vs-RPM evidence is redundant with the marginal/conditional correlation analysis (12_fullset_decomposition.py / 17_feature_oc_table.py). No cascade refs found this time (grep clean); compiles 40pp, 0 errors. Orphaned: \resRhoPowRpmLow/Mid macros (results_macros.tex / 07_paper_export.py ~L175,579-585), "EMI" nomenclature entry — NOT cleaned up, cosmetic only. See the 2026-06-29 banner near the top of this file for the much larger, separate finding: the new leak-free pipeline (scripts 11-17) supersedes this Gate's "frozen" legacy numbers; §5/§6 prose below still describes the LEGACY protocol and has not been re-verified or rewritten against the new pipeline.] |
| 7 | Discussion | §7.2 RESOLVED (2026-06-15): §disc_hf removed entirely with the 1–2 MHz band (was the "trim to claim-free / bearing-current refs" item — no longer needed). Queued: §7.1/§disc_ae "0–20 kHz baseband" wording (US now to 100 kHz — todo flagged in disc_ae); Jakobsen paragraph rewrite; limitation #2 wording; [2026-06-16 GATE 7 DONE: (a) disc_ae+disc_fusion+limitation#2 reconciled to grounded US framing (probe senses ~38-40 kHz, heterodynes to audio baseband, acquires 0-100 kHz; info concentrated <20 kHz, 20-100 kHz weak) per background subsec:origins + pipeline filter_bands; (b) Jakobsen para rewritten to PARITY under matched random splits + visit-grouped protocol as the methodological contribution, NOT "exceeds"; (c) limitation 3 = "Incomplete operating-condition coverage" (high-speed/high-temp corner sparse per fig:op_conditions_distribution; kappa skewed to boundary/mixed, full-film under-rep) verified from eda_operating_conditions.png; (d) limitation 4 = "Model-derived kappa reference" (computed from RPM+housing temp via ISO281+Walther per setup; housing-temp-as-contact-temp proxy caveat). All discussion.tex todos cleared, compiles 39pp. NB Edit tool fails on CRLF files in this repo -> used .NET literal replace. REMAINING (optional): relocate the lost "20-500 kHz transfers across drives" deployment point in the disc_ae comment block.]; limitations 3–4 todos |
| 8 | Conclusion | DONE 2026-06-16: verified all 5 findings + future-work against macros/frozen results. Fixed 2 genuinely stale items: item 1 (was "1-2 MHz complexity feature, first for all 3 models" -> now \resTopAeFeat = 500kHz-1MHz mobility, first for LightGBM+poly, 3rd ElasticNet, per modelling SHAP); item 3 (was "highly significant by ALL test statistics" -> now reports BOTH: hold-out Wilcoxon p=\resPwxAeUs significant, grouped-CV t-test p=\resPcvAeUs=0.121 inconclusive, per frozen "report both"). Items 2/4/5 + future-work checked CONSISTENT, no change. Compiles 39pp. |

## After all gates: compliance pass

elsarticle conversion, line numbers on, ToC removal, highlights file,
graphical abstract (optional), cover letter, suggested reviewers,
ML pre-approval check against official MSSP guide, affiliation/funding todos.

## Standing cautions

- File sync corrupts files (truncation/null bytes) — check `git diff` for mid-word
  endings before committing; all auto-generated tables should end with `\end{tabular}`
  or `\end{longtable}`.
- After any pipeline re-run: 06_plots → copy_figures → 07_paper_export → compile.
- config.yaml machine profiles must match the current machine's repo path.

## Remaining work (snapshot 2026-06-16)

Body writing through Gates 1-8 is essentially complete: all sections compile (39pp) and are internally consistent against the frozen results. NOTE: the four `\IfFileExists ... \todo[inline]{Run scripts/...}` fallbacks (appendix x2, features:76 table_top_features, modelling:87 table_models, modelling:154 table_shap_agreement) are NOT gaps - they never render because the table files exist. Real remaining items:

### A. Blocked on author/collaborator (experimental facts, not writing)
- setup.tex:8  - test-rig photo
- setup.tex:56 - US probe, temperature, RPM, load sensor models (only Kistler AE sensor/coupler documented)
- setup.tex:20 - how operating conditions are controlled ("How?")
- setup.tex:11 - VFD-noise filtering description ("explain the process and expected outcome")
- background.tex:48 - confirm US probe 38-40 kHz sensing band is correct
- backmatter.tex:21 - funding source (IFD?)
- backmatter.tex:28 + introduction.tex:80 - public dataset-release plan / wording, align with final data availability
- Gate 5 leftovers from earlier: lubricant charge amount, load-conversion confirmation (~600 N from telem_mass_g)

### B. Doable now (env is set up; pipeline/data, not blocked)
- features.tex:178 - source exact fraction of steps with negative within-step rho from 09_proxy_diagnostics, add as a macro (now prose-only; old 91% referred to removed 1-2MHz feature, status note says ~95% for 20-500kHz mobility)
- features.tex:210 - confirm pca_kappa_regimes.png is from final visit-grouped run
- modelling.tex:135 - refresh pred-vs-actual visual-interpretation sentence + confirm figure is final
- setup.tex:34 - cosmetic: consider inverting operating-conditions figure colours

### C. Submission compliance pass (separate phase, not started)
elsarticle conversion; line numbers on; ToC removal; separate highlights file; graphical abstract (optional); cover letter; suggested reviewers; MSSP ML-guideline pre-approval check; affiliation/funding finalisation.

### D. Infrastructure / housekeeping
- Reproducibility gap RESOLVED 2026-06-16: the 8 ?? macros were NOT missing data -- root cause was a config machine-profile mismatch. config.yaml machines: listed au808956 + Bruger but not favn; Path.home()=C:/Users/favn matched neither, so load_config fell back to the inaccessible Bruger path and 07_paper_export dropped to a stripped config missing bearing/frequency_bands -> those macros went ??. (My earlier "run the whole pipeline / need raw HDF5" diagnosis was WRONG: outputs/ + features/metadata.parquet were here all along.) FIX: (1) added a favn machine profile to config.yaml; (2) deleted 5 genuinely-dead 1-2MHz knee macros from 07_paper_export.py (resKneeRpm/KappaCool/Hot, resRhoPowRpmHigh) and decoupled resRhoPowRpmLow/Mid (still used in S4.3, +0.91/+0.69) from the removed knee logic. RESULT: 07_paper_export now runs CLEAN -- 116 macros, 0 missing; SHAP macros de-duplicated (HEAD already contained them, so the earlier surgical +6 had duplicated them). Working results_macros.tex now equals a full regen (vs HEAD: only the 5 dead-knee deletions, no value changes). Compiles 39pp. NOTE: config.yaml standing caution satisfied for the favn machine.
- Two new untracked files to commit: paper/tables/table_shap_agreement_tabular.tex, scripts/make_nested_cv_diagram.py.
- Env set up this session: MiKTeX 25.12 + Strawberry Perl 5.42 (for latexmk); .venv completed with optuna 4.9 (+ added to pyproject.toml). Edit tool fails on this repo's CRLF .tex/.md files -> use PowerShell .NET literal replace.

### E. New-pipeline reconciliation (added 2026-06-29, NOT started)
The Gate 1-8 walkthrough above and its "Bottom line" were written against the
legacy pipeline's frozen results. The 2026-06-29 banner near the top of this
file documents a separate, substantial investigation (new pipeline, scripts
11-17) whose findings are not yet reflected in the paper text below §4.3. This
supersedes the "Bottom line" below for body-text accuracy purposes: real,
unblocked rewriting work remains, namely the "Sections likely needing
rewriting" list in the 2026-06-29 banner (introduction contributions 1-3,
features/modelling sections 5-6, discussion, conclusion). Recommend treating
this as Gate 9 (or re-opening Gates 3/6/7/8) before any further submission-
compliance work, since several headline numbers and at least one contribution
claim (the soft-sensing residual) will change.

Bottom line: what remains is mostly (A) collaborator-supplied experimental details and (C) the journal-submission formatting pass for the legacy-pipeline text. (B) is the only substantive writing/data work that was unblocked under the old plan — but (E) is now the higher-priority item: reconciling the body text with the new pipeline's findings before any of (A)/(C) are worth finishing.