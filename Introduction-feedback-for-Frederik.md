# Feedback on the Introduction

**Manuscript:** *Sensor Complementarity and Operating-Point Proxies in AE and Passive
Ultrasound Regression of the ISO 281 Viscosity Ratio*

---

## Paragraph 1 — Industrial need
- **Include:** bearing importance; lubrication-related failures; consequences of inadequate
  lubrication; limits of schedule-based relubrication; value of online LCM.
- **Define LCM here**, at first use — it carries the rest of the paper.
- **Refs:** [1–3, 11]. **Length:** ~120–150 w.
- **Don't:** name sensors, κ, or Jakobsen yet.

## Paragraph 2 — Conventional methods are insufficient for LCM
- **Include:** vibration / characteristic-frequency analysis is mature for *structural* faults
  but poorly matched to the subtle, non-stationary changes of film formation/breakdown;
  temperature and oil analysis are indirect or invasive.
- **Keep one subordinate clause** acknowledging that data-driven *vibration* work can recover
  lubrication-state information from high-frequency broadband energy [13] — otherwise a
  reviewer objects "vibration already does LCM." Keep it subordinate to your point.
- **Close broadly.** One clause max on the operating-point confound — save the full argument
  for Paragraph 5. **Length:** ~150–180 w.

## Paragraph 3 — AE and passive US are promising; complementarity matters
- **Include:** non-invasive sensing; common contact-related origins; differences in acquisition
  bandwidth and signal representation; commercial relevance of passive US (threshold tools);
  the practical need to prove *incremental* value before adding a second sensor.
- **End with one crisp gap sentence**, e.g.: *the two channels have never been
  characterised simultaneously, against the same physically grounded lubrication reference.*
  That sentence is the hook into the rest of the paper. **Length:** ~130–160 w.

## Paragraph 4 — Prior AE/US work (synthesise by role, NOT paper-by-paper)
Replace the Miettinen→Yoshioka→Tandon→Cornel→Wang→Hou&Zhang→Renhart→Marticorena→Liu→Zhang
roll-call with three role-based statements:
- AE is sensitive to lubrication regime and lubricant properties [refs];
- AE detects contamination and early tribological damage, sometimes before vibration [refs];
- lubrication information is frequency-dependent and physically tied to EHL / asperity
  contact [refs].
- **Close on the shared limitation:** discrete labels, fixed conditions, features confounded
  by speed/load/transfer path, and — crucially — no one has quantified whether AE and passive
  US carry *non-redundant* information for continuous κ regression.
- This is the biggest single cut. It makes the section read like an argument, not a list.
  **Length:** ~190–230 w.

## Paragraph 5 — κ, Jakobsen, and the problem 
- Introduce κ as a continuous, physically interpretable reference. **Immediately** state it is
  calculated from speed and temperature — that one fact creates the whole problem.
- Position Jakobsen et al. as the direct predecessor: established acoustic κ-regression
  feasibility using low-cost passive acoustics and Hjorth features.
- **State the problem plainly** (this is the most important sentence in the paper). Model:
  > *Acoustic signals can predict κ accurately — but because κ is fixed by speed and
  > temperature, and acoustic signals also encode speed and temperature, it is unclear whether
  > such a model senses the lubricant film at all, or merely re-derives the operating point.*
- **Close on the unresolved questions** (frame as questions, not "we extend in three respects"):
  (i) how much performance is operating-point inference; (ii) whether a wideband AE channel
  changes the sensing information; (iii) whether passive US adds non-redundant value;
  (iv) whether performance survives acquisition-grouped evaluation. **Length:** ~220–260 w.

## Paragraph 6 — Present study + three contributions (worked example below)
Drop the "deliberately fixes bearing/lubricant/load" rationale from the intro (move
generalisation to Limitations). State what the study does, then the **three** contributions in
prose (the dataset is **not** a contribution yet — the release decision is still open), and
end with the bounded-claims sentence. Fold the dual-channel rig/acquisition into the study
description and the complementarity contribution — do not make "we built a rig" a standalone
contribution (it reads as an applications paper, an MSSP rejection risk). Preview the
AE↔speed / US↔temperature mechanism in *one clause, no numbers*.

**Example paragraph (adapt to your voice; verify wording against final results):**

> *The present study addresses these questions through the first simultaneous,
> time-synchronised acquisition of wideband acoustic emission and heterodyned passive
> ultrasound from a single rolling bearing, under controlled variation of rotational speed and
> temperature, with the ISO 281 viscosity ratio κ as a continuous reference. It makes three
> contributions. First, it introduces a confound decomposition for acoustic κ regression that
> separates the share of predictive performance attributable to acoustically inferring the
> operating point from a residual acoustic variation consistent with changes in lubrication
> adequacy. Second, it provides the first quantitative test of AE–US complementarity, showing
> that the two channels encode partly distinct components of the operating point and therefore
> offer complementary, rather than redundant, access to the lubrication state. Third, it adopts
> an acquisition-grouped, leakage-aware evaluation protocol with corrected significance
> testing, so that closely related measurement windows cannot cross the training–evaluation
> boundary. Because lubricant degradation and contamination are not independently varied, the
> study addresses acoustic prediction of nominal ISO 281 lubrication adequacy rather than
> direct detection of starvation, ageing, or contamination.*

---

## Do NOT include in the Introduction (move where indicated)
- **Headline numbers** — R², RMSE, ρ, ΔRMSE, p-values, the R²=0.826/0.834 decomposition figures
  → Abstract + Results. (At most one optional signature figure per contribution if you want a
  hook.)
- **Model names** — LASSO, elastic net, polynomial, LightGBM → Methods.
- **The 1 MHz AE-coupler (Kistler) limit** and the exclusion of content above it → Methods;
  it is instrumentation hygiene, not a contribution.
- **Exact frequency bands** (20–500 kHz, 20–100 kHz, 10–20 kHz), filter orders, heterodyning
  details, sub-band definitions → Methods.
- **VIF thresholds, feature formulas, sensor part numbers, CV fold counts** → Methods.

## Wording guardrails (non-negotiable)
- Delete **"genuine lubrication-channel sensitivity"** → "residual variation consistent with
  changes in lubrication adequacy." (Align with the Abstract, which already does this.)
- No **"leakage-free"** → "acquisition-grouped evaluation."
- No claims of detecting **starvation / ageing / contamination** — keep the bounded-claims
  sentence.
- Do **not** compare raw R² with Jakobsen unless the split protocol is matched.


