# CeramicSpeed — New (leak-free, soft-sensing) Pipeline

A **parallel** pipeline that supersedes the modelling/evaluation half of the
legacy pipeline. The legacy scripts live in `scripts/legacy/` (01–10), are
**untouched and still runnable** (see [`pipeline_summary.md`](pipeline_summary.md)).
The new pipeline lives in `scripts/new/`, is grouped by stage, owns its own
feature generation, and writes everything under `outputs/new/`.

## Folder layout (restructured 2026-06-30)

```
scripts/
  legacy/                         # FROZEN: 01–10 + 06_plots, 07_paper_export,
                                  #   copy_figures, make_nested_cv_diagram, _apply_*
  new/
    run_pipeline.py               # orchestrator (signal_processing -> modelling)
    signal_processing/            # the "signal-processing package" for the paper
      01_feature_generation.py    #   raw HDF5 -> outputs/new/features.parquet (owns it)
      02_pca.py                   #   PCA of the full feature set, coloured by kappa
      03_correlations.py          #   marginal vs conditional operating-condition corr
      04_channel_mechanism.py     #   sub-band / feature-type coupling + SHAP-on-(RPM,T)
      05_feature_oc_table.py      #   per-feature OC correlation table + signed heatmaps
    modelling/
      01_regression.py            #   kappa regression, full set, ElasticNet + LightGBM
      02_decomposition.py         #   two-stage (RPM,T) vs direct kappa decomposition
      03_group_paired_tests.py    #   group-paired Diebold-Mariano complementarity tests
      04_clustered_shap.py        #   redundancy-aware clustered SHAP importance for kappa
      05_figures.py               #   predicted-vs-true grids, comparison table
      06_inspect_results.py       #   p-value table + grouping/fold visualisation
src/ceramicspeed/                 # SHARED by both pipelines (grouping, cleaning, ...)
outputs/new/                      # new-pipeline outputs (features.parquet + per-stage dirs)
```

`src/ceramicspeed` is shared; new-specific helpers (`grouping.derive_hold_groups`,
`grouping.merge_twin_groups`, `cleaning.true_candidate_columns`) live there. The
new pipeline does **not** use `02`'s `feature_selection.json`.

## Why a new pipeline

1. **Full feature set only — no selection.** Models train on the full candidate
   set (target-independent candidates: **AE 42, US 56**). The VIF-"selected"
   set was removed entirely; it did not improve accuracy. Interpretation uses
   **clustered SHAP** on the full set (`modelling/04_clustered_shap.py`).
2. **Pooled, twin-merged, leak-free protocol — no nested CV.** Single-pass
   `GroupKFold` (k=5) over the whole dataset; each fold held out once while
   selection/fitting are redone on the others; pooled held-out predictions give
   one prediction per operating-point group from a model that never saw it. No
   Nadeau–Bengio correction needed (folds disjoint).
3. **Two-level grouping** (`ceramicspeed.grouping`):
   - *acquisition holds* — contiguous same-(file, RPM-step) sweeps;
   - *operating-point twin merge* — up/down sweeps revisit the same (RPM, temp);
     since κ is deterministic in (RPM, temp), twins are merged so a soft sensor
     can't score on a held-out twin via its trained sibling.
4. **Operating-condition filters** (config `filters`): `rpm_min: 50` (drop
   startup/standstill transients) and `temp_min: 38` (drop the sub-floor
   cold-start before the controller settles). Both are out-of-distribution
   transients, justified on experimental-validity grounds — they removed two
   prediction-vs-true artifacts and slightly improved R².
5. **Two models: ElasticNet + LightGBM** (Polynomial and BayesianRidge dropped).
6. **Soft-sensing framing.** κ is a deterministic function of (RPM, temp); the
   decomposition shows κ-regression is operating-point inference with no
   measurable residual beyond it.

## How to run

```bash
# build features from raw HDF5 (slow; only when raw data changes)
python scripts/new/run_pipeline.py --with-feature-generation
# or, if outputs/new/features.parquet already exists, just the analysis:
python scripts/new/run_pipeline.py
# individual stage, e.g.:
python scripts/new/signal_processing/02_pca.py
```

`--config alt.yaml` is forwarded to every step. Comparison opt-outs (modelling
steps): `--single-split` (single 80/20 instead of pooled) and
`--allow-twin-split` (skip twin merge); both write suffixed files and are for
comparison only.

## Headline results (pooled GroupKFold k=5, twin-merged; ~8.5k sweeps, ~520 groups)

**Operating-point decomposition (full features)** — two-stage (RPM,T-only) κ R²
≈ direct κ R² for every channel → **no detectable lubrication-specific residual**:

| Channel | feat→RPM R² | feat→T R² | two-stage κ R² | direct κ R² | residual |
|---|---|---|---|---|---|
| AE | 0.97 | 0.78 | 0.90 | 0.91 | ≈ +0.01 |
| US | 0.96 | 0.89 | 0.89 | 0.89 | ≈ 0.00 |
| Combined | 0.98 | 0.90 | 0.95 | 0.95 | ≈ 0.00 |

(The small AE positive residual is two-stage temperature-estimate error
propagation, not lubrication sensing.)

**Complementarity (group-paired Diebold–Mariano)** — robust and strong: Combined
beats **both** AE and US, for **both** models, **all 4 contrasts p≈0** (naive
window-level p≈0 too — shown only to demonstrate the inflation grouping prevents).
The most defensible headline claim.

**Channel encoding (descriptive):** marginally both channels are speed-dominated;
**conditionally**, US carries temperature information (10–20 kHz mobility/complexity,
ρ up to ~0.78 at fixed speed) that AE largely lacks, concentrated in US low bands.

**Interpretation (clustered SHAP for κ):** AE dominated by a 500 kHz–1 MHz mobility
cluster; US by a 10–20 kHz amplitude (rms) cluster plus distinct low-band
complexity/mobility singletons.

## Known cosmetic TODOs
- Scripts' `if __name__` completion prints and some docstrings still reference the
  old flat numbering (11/12/...); functional, not yet swept.
- `02_pca.py` output (`outputs/new/pca/`) not yet wired into the paper.
