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
| 4 | Background | **PENDING — awaiting author verdict** on §2.2 heterodyne rewrite (probe claims 0–20 kHz baseband; measured content reaches ≥40 kHz; proposed mechanism-neutral wording in chat) |
| 5 | Setup | queued: CeramicSpeed todos, standstill sentence decision, 2 N load check (possible unit error: telem key is mass_g) |
| 6 | Pipeline + Features + Modelling | queued: §6.1 still says "random sampling, stratified" — must describe grouped protocol; nested_cv_splits.png diagram still depicts random scheme; §5 interpretation paragraphs (ranking/redundancy/PCA) unwritten; ElasticNetCV grid todo |
| 7 | Discussion | queued: §7.1 "0–20 kHz baseband" fix; §7.2 trim to claim-free under "no mechanism" rule (bearing-current refs Muetze 2011 / RF-EDM pinned, unverified metadata); Jakobsen paragraph rewrite; limitation #2 wording; limitations 3–4 todos |
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
