# Paper writing status — section-by-section walkthrough

Updated 2026-06-12. Numbers regenerate via `scripts/07_paper_export.py`;
figures via `scripts/06_plots.py` + `scripts/copy_figures.py`.

## Frozen results (grouped splits, AE ≤1 MHz, US ≤100 kHz, iterative VIF@5)

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
| 6 | Pipeline + Features + Modelling | IN PROGRESS (2026-06-15). Clarified: the >1 MHz (1–2 MHz) band IS already excluded from current outputs (outputs/02_*, 04_* dated 06-12, 1–2 MHz absent; macros agree: NcandAe=42=14×3, NcandUs=56=14×4, TopAeFeat=500k–1MHz mobility). The stale 1–2 MHz refs in the prose came from OLD pre-restructure leftovers at outputs/ root (05-06) — NO pipeline re-run needed. DONE: pipeline candidate arithmetic fixed (was 14×4/14×3 backwards), Table tab:subbands fixed (dropped AE 1–2 MHz row; US now broadband 0–100 kHz + 0–10/10–20/20–100 kHz), sub-band prose rewritten, truncated §candidates sentence completed; modelling split rewritten to visit-grouped GroupShuffleSplit (+ folds wording + fig caption); SHAP prose+tab:shap_agree rebuilt to current data (dominant = 500k–1MHz mobility 0.205); features within-step feature → \resWithinStepFeat macro. Compiles (38pp). REMAINING: interpretation paragraphs (ranking/redundancy/PCA) still unwritten — need current vif_log/pca figures; regenerate nested_cv_splits.png (grouped) + decide band_power figure (author todo "obsolete"); tone §band_validation mechanism claim ("rotation-generated and gated by film") to satisfy "no mechanism claims"; SHAP numbers+agreement should move into 07_paper_export (hand-typed); ElasticNetCV grid values + "verify ElasticNetCV/RidgeCV" todo; source within-step negative-step fraction as macro (old 91% was the removed feature); NON-FATAL: appendix feature_ranking longtable throws "infinite glue shrinkage" warning. 2026-06-15 (author call): REMOVED obsolete §4.3 "Sub-band signal validation" (band_power_vs_rpm.png, 1–2 MHz) AND its discussion counterpart §disc_hf "Origin of the high-frequency AE content" (wholly about the excluded band). Cascade fixes: pipeline §4.2 ref, setup VFD note ref, intro contribution 3 ("characterised and excluded"→"excluded"), §disc_ae ("2 MHz"→"1 MHz", dropped dead refs). Orphaned macros (resRhoPowRpm*, resKneeRpm*, resKneeKappa*) now unused. LOST a still-valid point (20–500 kHz transfers across drives) — relocate if wanted. Then RESTORED §4.3 as a trimmed figure-light "Sub-band signal validation" (retained bands only: ρ=+0.93/+0.71 rise with speed = mechanical not EMI; 4.7 kHz comb below 50 kHz coupler HP; no mechanism claims). Figure band_power_vs_rpm.png STILL 3-band — author to regenerate: scripts/06_plots.py Plot D7, drop validation_extra_bands append (~line 836) → re-run 06+copy_figures. ρ macros from 07_paper_export. Compiles 38pp. PIPELINE §4 now DONE (2026-06-15): §4.1 spectrogram→chronological only (κ-sort removed); §4.2 sub-band prose + Table; §4.3 validation reworded (freq-separation + speed-dependence) and figure regenerated dense (per_step=10, 143 sweeps/window, 2-band) → ρ refreshed +0.91/+0.69, stale todo removed; §4.4 feature tables rewritten to interpretive descriptions + mobility/RMS-freq clarification; §4.5 candidate sets + combined-config foreshadowing. Compiles 39pp. Gate 6 REMAINING = §5 interpretation paragraphs (ranking/redundancy/PCA) + §6 (nested_cv_splits.png regen, ElasticNetCV grid, verify ElasticNetCV/RidgeCV, pred-vs-actual interp; within-step neg-fraction macro: now 95% for 20–500 kHz mobility) |
| 7 | Discussion | §7.2 RESOLVED (2026-06-15): §disc_hf removed entirely with the 1–2 MHz band (was the "trim to claim-free / bearing-current refs" item — no longer needed). Queued: §7.1/§disc_ae "0–20 kHz baseband" wording (US now to 100 kHz — todo flagged in disc_ae); Jakobsen paragraph rewrite; limitation #2 wording; limitations 3–4 todos |
| 8 | Conclusion | queued: full rewrite against frozen results (items 2/3/5 stale) |

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
