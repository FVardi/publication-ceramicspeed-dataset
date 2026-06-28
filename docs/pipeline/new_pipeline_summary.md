# CeramicSpeed — New (leak-free, soft-sensing) Pipeline

This is a **parallel** pipeline that supersedes the modelling/evaluation half of
the legacy pipeline. The legacy scripts (`01`–`10`) are **untouched and still
runnable** — if you change your mind, run the old pipeline exactly as before
(see [`pipeline_summary.md`](pipeline_summary.md)).

## Why a new pipeline

Four decisions motivated by the leakage analysis and the operating-point
decomposition:

1. **No feature-selection gate.** Models train on the **full candidate set**;
   selection (correlation + VIF) is kept only as a *descriptive* characterisation
   of the feature space, not as a filter that decides what the models see. This
   removes the holdout leakage caused by selecting features on the whole dataset
   in `02_feature_analysis.py`, and the full set predicts as well or better
   (ElasticNet +~0.13 R²; LightGBM unchanged).
2. **Simplified, leak-free protocol — no nested CV.** Acquisition-grouped 80/20
   split (138 holdout groups), `GroupKFold` CV on the 80% train for the CV
   estimate, single grouped holdout for the reported numbers. Nested CV added
   little given the number of independent groups.
3. **Two models only: ElasticNet + LightGBM.** Polynomial and BayesianRidge are
   dropped (Polynomial is unstable on the full collinear set and does not fit the
   narrative).
4. **Soft-sensing framing.** Because κ is a deterministic function of (RPM, T),
   the decomposition asks how much κ-regression is operating-point inference. It
   is essentially **all** of it (two-stage ≈ direct; residual ≈ 0), and AE vs US
   encode **partly distinct operating-point components** (AE↔speed,
   US↔speed+temperature).

## Relationship to the legacy pipeline

| | Legacy (`01`–`10`) | New (`11`, `12`) |
|---|---|---|
| Feature selection | gate before modelling (leaky: full-data) | descriptive only; models use full set |
| Performance estimate | repeated **nested** CV (`03`) + holdout (`04`) | grouped CV on 80% + single grouped holdout |
| Models | ElasticNet, Polynomial, LightGBM | ElasticNet, LightGBM |
| Leakage control | window-grouped split | window-grouped split **and** train-only selection |
| Core result | κ-regression performance + SHAP | operating-point decomposition + AE/US encoding |

Both share `01` (features) and `02` (which the new pipeline uses **only** for the
candidate `all_columns` schema — never the `retained` selection).

## Scripts

| Script | Role | Key outputs |
|---|---|---|
| `11_featureset_comparison.py` | Full vs selected feature sets; ElasticNet + LightGBM; grouped CV + grouped holdout; saves per-sweep holdout predictions with group ids | `outputs/11_featureset_comparison/featureset_comparison.csv`, `..._wide.csv`, `predictions/holdout_*.csv` |
| `12_fullset_decomposition.py` | Operating-point decomposition (features→RPM, →T, two-stage→κ, direct→κ) and marginal/conditional correlations on full features | `outputs/12_fullset_decomposition/tables/decomposition_summary.csv`, `cond_vs_marginal_full.csv`, `figures/proxy_map_full_{ae,us}.png` |
| `13_group_paired_tests.py` | Group-paired (138-group) modified Diebold-Mariano tests for complementarity (Combined vs single) and full vs selected; reports naive window-level p alongside to show inflation | `outputs/13_group_paired_tests/complementarity_tests.csv`, `full_vs_selected_tests.csv` |
| `run_new_pipeline.py` | Runs `11 --lightgbm` → `12` → `13` in order | — |

Shared library: `ceramicspeed.analysis.select_features` (new) packages the
two-stage selection so it can be called on a training partition only (leak-free).

## How to run

```bash
# prerequisites (shared with legacy pipeline)
python scripts/01_feature_generation.py
python scripts/02_feature_analysis.py

# new pipeline
python scripts/run_new_pipeline.py
# or individually:
python scripts/11_featureset_comparison.py --lightgbm
python scripts/12_fullset_decomposition.py
```

## Headline results (grouped holdout, 138 groups)

**Feature set (holdout R²)**

| Model | AE | US | Combined |
|---|---|---|---|
| ElasticNet (full) | 0.62 | 0.60 | 0.74 |
| ElasticNet (selected) | 0.49 | 0.46 | 0.59 |
| LightGBM (full) | 0.89 | 0.82 | 0.93 |
| LightGBM (selected) | 0.83 | 0.65 | 0.91 |

**Operating-point decomposition (full features, LightGBM)**

| Channel | feat→RPM R² | feat→T R² | two-stage κ R² | direct κ R² | residual |
|---|---|---|---|---|---|
| AE | 0.95 | 0.79 | 0.885 | 0.889 | 0.004 |
| US | 0.94 | 0.89 | 0.809 | 0.810 | 0.001 |
| Combined | 0.97 | 0.90 | 0.921 | 0.923 | 0.002 |

Two-stage ≈ direct everywhere → κ-regression is operating-point soft-sensing with
no measurable lubrication-specific residual. Marginally both channels are
speed-dominated; **conditionally**, US carries substantial temperature
information (median |ρ| 0.09→0.41 once speed is held) that AE does not — the
mechanistic basis for AE/US complementarity.

**Group-paired significance (138 groups; naive window-level p ≈ 0 everywhere,
showing why grouping is required):**

- *Complementarity is model-dependent.* Under ElasticNet, Combined beats both AE
  (p≈1e-4) and US (p≈3e-4). Under LightGBM, Combined beats US (p≈0.002) but is
  **not** significantly better than AE alone (p=0.14 full / 0.33 selected) — a
  strong model extracts from AE alone what US would otherwise add.
- *Full vs selected:* full is significantly better for ElasticNet (all targets)
  and LightGBM single channels (p<1e-4); for LightGBM-combined it is a wash
  (p=0.10) — trees are robust to the redundancy linear models need help with.
