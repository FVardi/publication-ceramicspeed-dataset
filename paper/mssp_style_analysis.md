# MSSP writing-style analysis and section-by-section edit plan

Date: 2026-07-03. Complements `mssp_gap_analysis.md` (strategic fit, submission checklist) and `mssp_reframing_proposals.md` (title/abstract/contributions text). This document covers *how MSSP papers are written* — structure, degree of detail, and language — and maps it onto the current draft.

**Corpus.** Four MSSP papers from `articles/` (the four domain-adaptation/fingerprint papers you listed are not in the repo; verified by full-text search of all 121 PDFs, not just filenames):

| Paper | Type | MSSP vol. |
|---|---|---|
| Jakobsen et al., *Detecting insufficient lubrication… MEMS microphone* | Research (experimental) | 200 (2023) |
| Pan et al., *Symplectic geometry mode decomposition (SGMD)* | Research (method) | 114 (2019) |
| Wakiru et al., *Lubricant condition monitoring… review* | Review | 118 (2019) |
| Sun et al., *Online oil debris monitoring… three decades* | Review | 149 (2021) |

Jakobsen 2023 is the load-bearing template: same journal, same group (AU + CeramicSpeed), same topic, direct predecessor of this study. Where it and the others disagree, follow Jakobsen.

---

## Part 1 — How MSSP papers are written

### 1.1 Structure

**Introduction = the entire related-work section.** None of the four papers has a separate "Related work" section. The introduction does everything, in a fixed funnel:

1. Industrial importance, 1 paragraph ("Rolling element bearings are among the most widely used components…"; "It is almost impossible in the modern industry to have a machine that does not employ rotating elements").
2. Method landscape, reviewed *critically*, method family by method family, each paragraph ending in that family's limitation. Jakobsen walks ECR → accelerometers → resonant piezo → AE → FFT/envelope → kurtosis/spectral kurtosis → Hjorth, ~35 citations. Pan enumerates three explicit "flaws" up front, then dissects wavelet/EMD/EEMD/ITD/LCD/SSA against them one by one — the gap is *constructed*, not asserted.
3. Explicit motivation paragraph ("Our main motivation for the work presented is twofold…").
4. Numbered contribution list (research papers). Jakobsen's four items are concrete deliverables: "We design and describe a comprehensive experiment…", "A prototype sensor … is built and presented". Deliverable-shaped, not claim-shaped.
5. Mandatory roadmap paragraph ("The paper is organised as follows: Section 2 provides…").

**Body order (experimental papers), from Jakobsen:** background theory on the target quantity (§2 Kappa) → experiment setup (§3, subsections down to 3.4.1, 3.6.1) → analysis techniques (§4: signal origin, features, models) → dataset generation (§5) → results (§6) → discussion (§7, per-topic subsections) → sum-up (§8). Method papers (Pan): theory with full derivation (§2) → **simulation validation** (§3) → experimental application (§4) → conclusions (§5). Deep subsection nesting (x.y.z) is normal and encouraged.

**Conclusions are short and self-contained.** Enumerated findings or 2–3 paragraphs; no statistics beyond one or two headline numbers; future work folded into the last paragraph. Pan's conclusion is ~1/3 page for a 23-page paper.

### 1.2 Degree of detail

- **Hardware: exhaustive.** Exact part numbers, bearing designations (NSK 6208, SKF 2206), lubricant grades, sample rates, filter corner frequencies, mounting method, load values. Jakobsen dedicates subsections to bearing details *per test bench* and to the lubricant. This is the reproducibility contract MSSP expects — the draft's `setup.tex` already writes at this level; the open `\todo`s (mounting, oil quantity, VFD filter) are exactly the details an MSSP reviewer will ask for.
- **Methods: equations for everything used, even textbook material.** Jakobsen derives κ; Pan derives SGMD over ~6 pages. Feature definitions are tabulated with formulas (draft already does this, well).
- **Justification granularity:** every design choice gets one sentence of *why* (e.g. Jakobsen on why time-domain features: "variance is calculated as a simple sum of squares…"; why MEMS: power budget, "years on an AA like battery"). Choices are motivated by application constraints, not just cited.
- **Statistics: light.** None of the four papers runs a significance-testing battery. Results are argued via figures + tables + a few R²/RMSE-type numbers. The draft's inference protocol (Nadeau–Bengio, Holm, DM, bootstrap) *exceeds* the journal norm — that is a strength, but it means the prose must carry the reader: state the claim first, the test second, and keep all p-values out of abstract and conclusion.
- **Figures: many, with interpretive captions.** Captions state what to see, not just what is plotted ("The speed axis is essentially unchanged by conditioning…" — the draft's `marg_cond` caption is already perfectly MSSP-idiomatic). Simulation/validation figures precede application figures.

### 1.3 Abstract conventions

Single paragraph, **200–225 words** measured (Jakobsen 222, Pan 202). Fixed skeleton: problem/limitation of existing approaches (2–3 sentences) → what was done, with the key experimental facts (one sentence can carry "18 day experiment", sensor, bearing type) → main finding stated qualitatively ("show a good correlation", "with good precision", "accurately and effectively") → one sentence of implication/use. **Zero p-values, zero CI bounds, at most one or two numbers.** The ~215-word abstract in `mssp_reframing_proposals.md` §3 fits this template almost exactly — adopt it (it currently carries one R² and several macro-numbers in ranges; that is at the upper limit of what these papers do, still acceptable).

Keywords: 5–6, noun phrases, mix of method + object ("Vibration analysis; Insufficient lubrication; MEMS microphone; Bearings; Hjorth's parameters").

### 1.4 Language and jargon

- **Voice:** mixed. "We" freely used for design decisions and findings ("we are able to predict", "we shoved [sic] coherence"); passive for procedure ("Vibration signals are recorded…, then band-pass and high-pass filtered"). The draft already does this — keep it; do not passive-ify everything.
- **Tense:** present for the paper's own work and for standing facts; past for specific prior experiments ("Lugt et al. [5] investigated…").
- **Citation style:** numeric; author-prominent form when contrasting or building on specific work ("Cocconcelli et al. [35] recently shoved the applicability of Hjorth's Parameters…"), bare-bracket for background. Intro carries the bulk (~30–35 in Jakobsen). The draft's intro has ~27 citation instances — in range.
- **Spelling:** Jakobsen uses British -ise ("utilise", "analysed", "organised"). The draft mostly does too; keep and make consistent (currently mixed: "digitised"/"summarised" vs "synchronized"/"colored" in `setup.tex`/`modelling.tex`).
- **Domain jargon used without definition** (safe to use bare): rolling-element bearing, raceway, asperity contact, EHL, boundary/mixed/full-film regime, oil-film formation, starved/insufficient lubrication, incipient defect, spalling, characteristic bearing frequencies, envelope/enveloped FFT, spectral kurtosis, cyclostationary, stress waves, structure-borne, transmission path, kurtosis/crest factor, condition monitoring / condition-based maintenance.
- **Terms defined at first use, then abbreviated:** AE, LCM, ECR, EMD/EEMD, VFD, and — Jakobsen precedent — "the viscosity ratio Kappa, κ" (spelled out once with the symbol, thereafter κ). Hjorth parameters get provenance ("originate from EEG analysis") — draft already does this.
- **Idiom notes:** Jakobsen writes "insufficient lubrication" (not "poor"/"bad"), "oil-film formation quality", "low cost" as premodifier, "purpose built prototype sensor". "Passive ultrasound" as the draft uses it is *not* standard vocabulary in this corpus — Jakobsen says "ultrasonic MEMS microphone"/"airborne ultrasound" for the analogous channel. The draft's footnote defining passive vs active is therefore justified — but MSSP papers define such terms inline in the body text, not in footnotes (no footnotes in any of the four papers except corresponding-author). Move it inline in §1 or §2.2.

---

## Part 2 — Application to the draft, section by section

Overall verdict first: the draft's structure (Intro → Background → Setup → Pipeline → Features → Modelling → Discussion → Conclusion) maps 1:1 onto Jakobsen's and needs **no reorganisation**. The detail level in setup/features is already journal-typical. The gaps are: an empty abstract, statistics density in the wrong places (abstract/conclusion), ~20 unresolved `\todo`s that are mostly exactly-what-reviewers-ask details, and surface-language issues.

### abstract.tex
Empty (`Insert abstract`). Adopt `mssp_reframing_proposals.md` §3 (~215 words, matches the measured 200–225 norm). Trim to ≤2 retained numbers; replace the test-battery clause with "validated by repeated nested cross-validation with corrected significance tests" (already done there). Keywords: cut to 6 (gap analysis flags 7).

### introduction.tex — closest to done
Already has the funnel, the critical per-method-family review, numbered contributions, and roadmap. Remaining deltas vs the corpus:

1. **Research questions as an enumerated list** (the three questions) has no precedent in the four papers; Pan states his three "flaws" as a numbered list, so a list per se is fine, but consider converting the questions into the gap statement prose and letting the contribution list answer them — the questions and contributions currently overlap ~1:1, which reads as duplication.
2. **Contributions should be deliverable-shaped.** Current items are claim-shaped ("it shows that…"). Reshape per Jakobsen: "We design and describe…", "We perform the first simultaneous…", "We release…". The dataset+protocol release deserves its own numbered item (Jakobsen precedent: the dataset *is* contribution #1) — it currently isn't in the list at all, and `mssp_reframing_proposals.md` §2 already drafts it as item 4.
3. **Move the passive/active footnote inline** (see §1.4).
4. The final scope-limitation sentence ("Because lubricant degradation and contamination are not independently varied…") is good and has Jakobsen precedent (his κ-assumption caveats) — keep.

### background.tex — good match, minor
Mirrors Jakobsen §2 (κ theory) + his §4.1 (signal origin) in one section — fine. Keep both equations. Notes:
- The κ-regime boundary listing (0.1/0.4/1) is the right level of detail. The final sentences on what κ does *not* encode are exactly the honesty MSSP reviewers reward.
- §2.2's commercial-probe paragraph drifts into product-catalogue register ("UE Systems, Sonotec, SDT Ultrasound"). Jakobsen names manufacturers only where the specific device matters (FAG/SKF as *sources of the 40–80% figure*). Name the class, cite, drop the brand list or move it to setup where the actual probe is specified.
- "respond from tens of kHz to several 100s kHz" → give the range as numbers with units (corpus style is numeric: "up to 100 kHz", "20 kHz–50 kHz").
- Nomenclature section: `mssp_gap_analysis.md` says MSSP papers "customarily include one" — **none of the four sampled papers has one**. Treat it as optional; symbol definitions at first use (current practice) match the corpus.

### setup.tex — right detail level, wrong polish
The granularity (part numbers, corner frequencies, load, oil spec) is exactly Jakobsen §3/§3.6. Fixes:
- Resolve the reviewer-bait `\todo`s: sensor mounting method (Jakobsen specifies "adhesive mounted" for the AE comparison — mounting affects coupling and *will* be asked), VFD filter description, oil quantity, the 20-min 3000 rpm hold rationale.
- Typos: "It can bee seen", "THe nominal", "Tmperature", "aquisition time" (caption), "cosistency" (in a todo). Load stated as "60kg" in one sentence and "approximately 61 kg" in the next — reconcile.
- Unit formatting: "0.55kW", "5kHz", "0.5ml", "12.5MHz" bypass `\SI{}`; the corpus (and Elsevier) sets a space: 0.55 kW. Make `siunitx` universal.
- Consider a rig-component table (component / model / role), mirroring Jakobsen's bearing-details subsections — optional, but it would absorb the part-number prose.
- Jakobsen's §3.4 "Test protocol" gives nominal *and achieved* conditions; the draft's §"Acquisition segments and measured operating conditions" already does this well (the ≈33 °C floor honesty is corpus-idiomatic).
- `[H]` float placement throughout the draft fights Elsevier's layout; use `[tb]` and let them float.

### pipeline.tex — fine; sharpen justifications
- The sub-band-choice paragraph is honest ("not … the most informative, physically motivated bands") but the *why-these-bands* sentence rests on "visual inspection of Figure 3". Corpus style backs such choices with one physical sentence each (Jakobsen justifies each analysis band by source mechanism: component resonances 0.1–20 kHz, random ultrasonic 20–50 kHz, plastic deformation >50 kHz). One sentence per band tying it to §2.2's mechanisms would MSSP-proof this.
- "dataming" → "data mining". "sought filtered out" (setup) and "the information … is expected to be more localised" are fine.
- The >1 MHz energy observation ("No physical interpretation is applied") — reviewers will poke this; either one candidate-mechanism sentence (bearing currents? aliasing? sensor resonance) or cite the coupler roll-off spec.
- Feature tables with formula + interpretation columns: better than corpus norm, keep.

### features.tex — strongest section, corpus-idiomatic already
Interpretive captions, marginal-vs-conditional logic, explicit "We therefore condition on each operating variable in turn" signposting — all match. Only notes: define ρ as Spearman once, and the "surprisingly weakly correlated" editorialising is fine (Jakobsen editorialises identically).

### modelling.tex — content exceeds corpus norm; presentation must compensate
- **Bold run-in headings** ("\textbf{Overall performance.}") have no corpus precedent — MSSP papers use numbered subsubsections. Convert to `\subsubsection` (matches the x.y.z nesting the corpus loves).
- The stats battery is beyond journal norm (§1.2). Keep it, but lead every results paragraph with the engineering claim in plain words, then the test. The current text mostly does this; the Table 3 caption carrying the CV-vs-holdout divergence discussion is good.
- Typos/grammar: "three different features sets", "Each of the later described models" → "of the models described below". The `\todo`s on group counts/thresholds are must-fixes (the evaluation protocol is a headline contribution — it cannot contain placeholders).
- The `\itemize` grouping-protocol steps: corpus uses numbered lists for procedures; minor.
- Check consistency: §"Data sets and splits" describes a *single-pass 5-fold* protocol while §"Repeated nested cross-validation" describes *repeated* nested CV with an 80/20 hold-out — per `WRITING_STATUS.md` the new leak-free pipeline supersedes the legacy one; make sure only one protocol is narrated (this is currently the draft's biggest internal inconsistency, bigger than any style issue).

### discussion.tex — structure right
Per-topic subsections mirror Jakobsen §7. The limitations `enumerate` with bold lead-ins is fine (no corpus precedent against). Finish limitation #5 (`\todo` — the interpolation-not-extrapolation point is the one reviewers will care most about, per gap analysis §2). "might reflect a fundamental hardware difference" — corpus hedges the same way; keep.

### conclusion.tex — trim and de-statistic
Enumerated findings match Jakobsen §8, but: 6 items is many (corpus: 3–4); items #2 and #3 carry full p-value batteries and CI bounds — **no corpus conclusion contains a p-value**. State each finding qualitatively + at most one number, point to Table 4 for inference. Merge #2+#3 (fusion + AE-vs-US are one sensor-comparison story) and #5 into #6 (evaluation rigour is one finding). Future work as closing prose paragraph: matches corpus, keep.

### Cross-cutting language pass (one sweep, ~1 h)
1. Typos: bee/THe/Tmperature/aquisition/dataming/features sets/cosistency (list from grep; rerun before submission).
2. -ise/-ize and colour/color consistency (pick British per Jakobsen; "Synchronized", "colored", "artifacts" currently American).
3. Terminology drift: "segment" is defined as the unit of observation in setup.tex, but features/modelling repeatedly say "per sweep", "high-κ sweeps", "computed for each sweep". Pick "segment" everywhere; reserve "sweep" for the RPM staircase.
4. "RPM"/"rpm" mixed; corpus uses lower-case rpm as a unit.
5. Spell out κ once as "the viscosity ratio Kappa (κ)" at first body use (Jakobsen convention), then symbol only.

### Priority order
1. Resolve the modelling-protocol narration inconsistency (new vs legacy pipeline).
2. Abstract in, todos out (especially setup hardware details + limitation #5 + group counts).
3. Reshape contribution list (deliverable-shaped, add dataset item) — text exists in `mssp_reframing_proposals.md`.
4. De-statistic conclusion + abstract; convert bold run-ins to subsubsections.
5. Language sweep (typos, British spelling, segment/sweep, units).

---

## Addendum 2026-07-07 — re-verified against current draft

Independent re-read of the three text-extractable MSSP papers and all current `.tex`. Part 1 findings confirmed throughout; the following items are updated:

1. **Corpus note.** The Wakiru review PDF has no extractable text layer (scanned image), which is why full-text scans for the MSSP header find only 3 of the 4 corpus papers. Its MSSP provenance is confirmed via citation (Mech. Syst. Signal Process. 118 (2019) 108–132).
2. **The protocol inconsistency (Priority 1) has moved, not gone.** `modelling.tex` now cleanly narrates the new pipeline only: single-pass 5-fold grouped CV + 80/20 group holdout, Elastic Net + LightGBM, group-level Diebold–Mariano with Harvey correction. The stale legacy narration now lives in **discussion.tex** (§disc_kappa: "repeated nested cross-validation", `\resNouterScores` outer scores; §disc_fusion: "all three hold-out tests") and **conclusion.tex** (items 2, 3, 5: polynomial model, within-CV corrected t-test, Wilcoxon, bootstrap CI, "repeated nested cross-validation"). These sections must be rewritten to match modelling.tex — still Priority 1.
3. **§abstract.tex recommendation is void as written.** The reframing-proposals §3 abstract can no longer be adopted as-is: it leads with the film-gated 1–2 MHz finding, which has been **dropped from the draft** (pipeline.tex now excludes >1 MHz content from all analysis; grep confirms zero occurrences of film-gated/1–2 MHz/`\resKneeKappa`), and it cites the superseded stats protocol. The abstract needs fresh drafting around the current headline results (complementarity test, operating-point decomposition, leakage-controlled evaluation). Skeleton and length norms in §1.3 still apply.
4. **introduction.tex has been rewritten since 2026-07-03** and now makes **three "first" claims** in the final contribution paragraph (first simultaneous acquisition, first complementarity test, first leakage-controlled figures). No corpus paper stacks firsts; Sun's precedent is to name prior works and their specific deficiency, then claim the one gap. Keep the best-defended first (the complementarity test — reviews explicitly calling for it are already cited) and let the other two stand as facts. Points 1–2 of the original §introduction.tex notes (RQ-list/contribution overlap; claim-shaped contributions; missing dataset-release item) remain valid against the new text.
5. **conclusion.tex** additionally names a "polynomial model" that appears nowhere in modelling.tex — covered by item 2 above but worth naming as the single most visible reviewer-catch.
6. **pipeline.tex >1 MHz note** ("No physical interpretation is applied") — original recommendation stands, now more important: since the 1–2 MHz analysis was cut, this sentence is the only trace of the phenomenon. Either give one candidate-mechanism sentence + exclusion rationale, or cite the coupler roll-off.

### Addendum 2026-07-07 (2) — draft narrative vs new-pipeline *results* (`docs/pipeline/new_pipeline_summary.md`, `outputs/new/`)

The style points above address how the draft is written; these address where the draft text contradicts the new-pipeline numbers it is supposed to report. Style guidance for each rewrite is included.

7. **"AE substantially outperforms US across both model families" (modelling.tex, disc_ae, conclusion #3) is contradicted.** The group-paired DM result splits by model family: ElasticNet favours US (mean dMSE +0.010, p≈0), LightGBM favours AE (−0.005, p≈0); direct R² gap is small (0.91 vs 0.89). Rewrite as a model-family-dependent finding with the mechanism sentence corpus style demands: ElasticNet exploits the simpler temperature-driven US signal, LightGBM the nonlinear speed-dependent AE structure.
8. **The "temperature-mediated lubrication channel beyond the operating-point proxy" claim (disc_confound, conclusion #6) is no longer supported.** Two-stage κ R² ≈ direct κ R² for every channel (AE 0.90 vs 0.91; US 0.89 vs 0.89; combined 0.95 vs 0.95) — no detectable lubrication-specific residual. The within-step temperature correlations are operating-point (temperature) information, not lubrication residual. MSSP precedent for reporting this honestly exists (Jakobsen §7.1 self-critique); the negative result should be *stated as the finding*, not softened with a residual-channel caveat.
9. **Complementarity robustness is under-sold.** All 4 combined-vs-single contrasts p≈0, both families, Combined beats both channels (R² 0.95 vs 0.91/0.89). Keep the effect-size honesty ("modest"), upgrade the robustness language — this is the paper's most defensible claim and should read like it.
10. **Feature-selection references are stale.** `\resNretAe`/`\resNretUs` and "retained features after selection" (disc_fusion, conclusion) — the new pipeline trains on the full candidate set (AE 42, US 56); interpretation is clustered SHAP. Purge selection language.
11. **Intro contribution #1 wording** ("a residual variation consistent with changes in lubrication adequacy") promises what the decomposition then refutes. Reword before results exist in tex — otherwise intro and discussion will contradict each other in review.
