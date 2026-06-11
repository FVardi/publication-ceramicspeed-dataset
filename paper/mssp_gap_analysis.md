# Gap analysis: draft vs. MSSP expectations

Date: 2026-06-11. Based on all `.tex` files in `paper/` and MSSP's public author guidance (the ScienceDirect guide-for-authors page is JS-rendered and could not be fetched directly; items marked **[verify]** should be checked against the official guide before submission).

---

## 1. Strategic fit — the biggest risk

MSSP's bar is a **demonstrable signal-processing advance validated on a mechanical system**, not an application of established methods to a new dataset. Its most common desk-rejection trigger is exactly the shape this draft currently has on the surface: canonical features + off-the-shelf ML (LightGBM, Elastic Net, SHAP, Optuna) + careful statistics.

**The draft contains MSSP-grade material, but it is not what leads.** The elements an MSSP editor would consider novel signal-processing contributions are currently buried mid-paper:

1. **The film-gated 1–2 MHz band discovery** (§ band validation + § disc_hf). The finding that high-frequency AE power vanishes at a constant *lubrication state* (κ) rather than constant speed — and the mechanical-vs-electrical (bearing current) mechanism question — is the most original signal-physics result in the paper. It appears in a validation subsection, not in the title, abstract opening, or contribution #1.
2. **The confound decomposition** (§ confound + § disc_confound). The two-stage "acoustic operating-point soft sensor" analysis and within-RPM-step temperature-channel evidence is a methodological contribution to how LCM papers should handle operating-condition confounding. This framing — a *diagnostic methodology* for separating lubrication information from operating-point proxies — is MSSP material.
3. **The quantified fusion answer** — first simultaneous AE+US κ regression; complementarity "argued but never tested" is a clean gap statement. This is well framed already.

**Recommendation:** reframe the contribution list to lead with (1) and (2) as signal-processing/physics contributions; demote "LightGBM achieves R²=0.95" from headline to supporting evidence. The current contribution #3 ("Features of sub-bands of AE signals predict κ") reads as an application result; the editor question will be "where is the engineering advance?"

## 2. The κ-circularity vulnerability

κ is computed analytically from measured RPM + temperature; the features predict RPM and temperature nearly perfectly; therefore high R² on κ is close to guaranteed by construction. The draft is admirably honest about this (two-stage R² ≈ direct R²), but then claims in the conclusion that "the κ regression framework is validated as a highly effective approach … achieving substantially higher performance than prior comparable studies." A signal-processing reviewer will flag the tension: the headline metric mostly measures operating-point recovery, and the *genuine* lubrication sensitivity rests on the within-step analysis.

- Tone down conclusion #5 and the discussion claim of exceeding Jakobsen et al. (itself marked `\todo{True?}`).
- Consider promoting a within-step / amplitude-invariant metric to a headline result, since that is the part that survives the circularity critique.
- Limitation #4 ("Model-derived κ reference") is an unresolved todo — this must be written, as it is the limitation reviewers will care about most.

## 3. Benchmarking gap

MSSP expects comparison against state-of-the-art method families, not only internal model comparisons. Currently the three model families are all generic regressors on the same feature set.

- No baseline reproducing prior art (e.g., Jakobsen et al.'s LASSO / NN feature approach on this data; envelope or spectral-correlation methods from the cited literature).
- No naive/reference baselines in the results table (e.g., broadband-RMS-only model, RPM-from-tacho upper bound) to anchor what the sub-band features add. The two-stage analysis partially serves this but isn't presented as a benchmark.
- A small "what does the sub-band decomposition buy" ablation (broadband-only features vs. full 56) would directly support contribution #1 and is cheap to run.

## 4. Possible ML pre-approval requirement **[verify]**

Secondary sources state MSSP asks **machine-learning/soft-computing–focused submissions to obtain prior proposal approval from the Editor-in-Chief** and rejects many ML papers without review. Whether this draft counts as "ML-focused" depends on the framing fixed in §1 — another reason to lead with signal physics rather than LightGBM. Check the official guide; if applicable, the proposal email (title, authors, summary, ToC) precedes submission.

## 5. Required submission elements

| Element | Status |
|---|---|
| Highlights (3–5, ≤85 chars) | Drafted as comments in `main.tex` — need char-count check and a separate file; #3 and #5 currently lead with model performance (see §1) **[verify count/limit]** |
| Graphical abstract | Missing (optional but encouraged at Elsevier) **[verify]** |
| Abstract | Present but ~370 words, overloaded with p-values/macros; Elsevier norm ≤ ~250 words, self-contained. Trim statistics to one or two key numbers |
| Title | Very long (~30 words, two clauses + colon). Elsevier prefers concise; also mismatch with the comment header in `main.tex` ("Passive Acoustic Regression to kappa") |
| Keywords | 7 listed; MSSP cap is typically 6 **[verify]** |
| Nomenclature section | Missing — paper is symbol-heavy (κ, ν, ν₁, d_pw, ρ, σ_w …); MSSP papers customarily include one |
| Affiliation | Placeholder: "Department of …, Aarhus University" |
| CRediT statement | Present ✔ |
| Competing interests | Present ✔ |
| Funding | Present but `\todo{IFD?}` unresolved |
| Data availability | Present (Zenodo on acceptance) ✔ — add an explicit **code** availability sentence |
| AI declaration | Present ✔ |
| Cover letter | Not in repo — must state the signal-processing contribution and MSSP fit |
| Suggested reviewers | Not in repo — MSSP asks for ~4 with no recent collaboration **[verify number]** |
| Line numbers | `\linenumbers` commented out — enable for submission |
| Document class | `article` with manual MSSP approximation; consider `elsarticle` (`elsarticle-num` bib style instead of `unsrt`) **[verify acceptance of plain article]** |
| Table of contents | Present — remove for journal submission |

## 6. Engagement with the MSSP literature

Only 2 of 26 references are MSSP papers. For a journal that screens for fit with its own conversation, this is thin. The `articles/` folder likely contains recent MSSP work on AE-based condition monitoring, sub-band/spectral methods, and interpretable ML for machine health — worth weaving 4–6 such citations into the introduction and discussion, particularly recent (2023–2026) MSSP papers the contribution builds on.

## 7. Technical completeness (reviewers will notice)

Unresolved `\todo`s that affect reviewability, roughly in order of importance:

- **Sensor specifications** (§ setup): AE sensor model/bandwidth/calibration, US probe model, temperature/RPM/load sensors — bullet list is empty. An MSSP experimental paper cannot omit these.
- **Contribution #2 "Slipring based film thickness measurement"** — one line, `\todo{Expand if novel}`. Either it's a real contribution (then it needs a section: nothing about a slipring appears anywhere else in the paper) or it must be deleted. Currently it contradicts the backmatter and limitation #4.
- **Radial load = 2 N** on a bearing with 14 kN dynamic rating — effectively unloaded. Reviewers will ask whether boundary/mixed-regime asperity contact claims are meaningful at near-zero load and whether κ-regime boundaries apply. Needs explicit justification or discussion.
- Temperature control method, lubricant quantity, RPM control, VFD switching frequency (5 kHz vs. 4.7 kHz inconsistency between § setup and § band validation), US probe sensing band `\todo{Check}`.
- Limitation #3 "Data gap" — empty todo.
- Figure issues: `nested_cv_splits.png` caption is literally "Caption"; several interpretation todos (PCA figure, pred-vs-actual after new results, top-feature ranking interpretation).
- Typos: "IS  O~281" (intro), "distribution fo" (setup caption), "accordig" (pipeline), "stille" (limitations).
- 3000 rpm over-representation (~⅓ of sweeps) is honestly reported but its effect on metrics isn't analysed — a reviewer may ask for a re-weighted or per-regime error breakdown (the per-regime breakdown would also strengthen §2).
- Sweep-level random train/test split with temporally adjacent sweeps: conclusion already names temporal autocorrelation as future work, but reviewers may demand a block-wise (e.g., leave-one-temperature-block-out) split as a robustness check now, not later. This is the most likely "major revision" request.

## 8. What is already strong (keep)

- Statistical evaluation design (repeated nested CV, Nadeau–Bengio, Holm–Bonferroni, multiple hold-out tests) exceeds the field norm — a genuine differentiator.
- Honest confound analysis and limitation discussion.
- Macro-driven results numbers (`results_macros.tex`) — excellent reproducibility practice; mention the pipeline in the data statement.
- Clear gap statement (AE–US complementarity argued but untested).
- Backmatter sections largely Elsevier-ready.

## Suggested priority order

1. Reframe contributions/title/abstract around the signal-physics findings (§1, §2) — determines desk-screen survival.
2. Resolve setup todos incl. sensor specs and the 2 N load question (§7).
3. Decide the slipring contribution (§7).
4. Add baseline/ablation benchmarking and a block-wise split robustness check (§3, §7).
5. Compliance pass: abstract length, title, keywords, nomenclature, elsarticle, line numbers, TOC removal, highlights file (§5).
6. MSSP citation pass (§6).
7. Cover letter + suggested reviewers + check ML pre-approval requirement (§4, §5).
