# Feature audit — which of the 26 features to keep

Audit of `src/ceramicspeed/features.py` against the paper's Tables 2–3 and the empirical rankings in `outputs/02_feature_analysis/`. Verdict up front: **of the 26 features, only ~14 are mathematically independent.** The rest are exact transforms or products of others. The paper's Tables 2–3 do faithfully describe the code (including its quirks), so fixing the code and the tables together is one job.

---

## 1. Exact duplicates (provable from the code, confirmed in the data)

Groups below have **bit-identical |Spearman|** with κ in `feature_ranking_ae.csv` / `_us.csv` — the smoking gun for deterministic monotone transforms:

| Group | Why | Identical ρ confirmed in |
|---|---|---|
| `std`, `variance` | variance = std² | every band, both sensors |
| `rms`, `std`, `variance` | bandpass filtering removes the mean, so rms = std exactly in all sub-bands (broadband AE keeps a tiny DC offset, hence rms differs there only) | AE 500–1000 kHz, AE 1–2 MHz, US sub-bands |
| `center_frequency`, `frequency_weighted_std`, `normalized_frequency_std` | `frequency_weighted_std` is computed **unweighted** over the fixed FFT frequency grid: σf = √(mean((f_k − fc)²)) = √(Var(grid) + (f̄ − fc)²), a strictly monotone function of fc for fc < f̄. `normalized_frequency_std` = σf/fc, again a pure function of fc. | all five band groups where the triplet survives the corr filter (ρ = 0.7894 in AE 500–1000 kHz, etc.) |

Exact algebraic identities (not flagged by identical ρ because Spearman of a *product* isn't preserved, but still fully redundant given the pair):

- `peak` = `crest_factor` × `rms` (by definition)
- `impulse_factor` = `crest_factor` × `shape_factor` (peak/|x̄| = (peak/rms)·(rms/|x̄|))
- `rms_frequency`² = `center_frequency`² + (magnitude-weighted bandwidth)² — i.e. once you have fc and a *proper* weighted bandwidth, rms_frequency adds nothing.

## 2. Implementation/definition problems (the root cause)

1. **`frequency_weighted_std` is not weighted.** The name (and intent) say magnitude-weighted spectral spread; the code averages over the bare frequency grid. The paper's Table 3 honestly footnotes "unweighted by magnitude" — but that makes it, and `normalized_frequency_std`, pure functions of fc (see above). The fix is the standard weighted bandwidth: σ_w = √(Σ(f−fc)²·S / ΣS).
2. **`frequency_skewness` / `frequency_kurtosis`** normalise by K·σf^p with the broken σf, and divide by K instead of ΣS — so they are not standardised shape moments; they scale with signal amplitude and mix amplitude into a "shape" feature. Standard form: Σ(f−fc)^p S / (ΣS · σ_w^p).
3. **`spectral_skewness` / `spectral_kurtosis`** normalise a frequency-moment (units Hz³·V) by `spectral_std`³ (units V³) — dimensionally incoherent. Once item 2 is fixed, these two are the same quantity with a worse denominator; they should not coexist with it. (This resolves the existing `\todo` in features.tex about the skewness pair.)
4. **`peak_frequency`** (code name) is the paper's "f²-weighted RMS frequency" — the code name is misleading and will confuse readers cross-referencing the dataset; rename to `f2_rms_frequency` or similar.
5. **`spectral_mean` / `spectral_std`**: for noise-like signals the FFT magnitudes are ~Rayleigh, where std/mean is a constant — near-duplicates of each other, and both are amplitude measures largely duplicating band rms (Parseval). Empirically in the US ranking they sit within 0.001–0.06 ρ of rms/std.
6. **Hjorth `mobility`** is the power-weighted RMS frequency in disguise (Parseval): conceptually overlaps `rms_frequency`/`center_frequency` but with S² rather than S weighting — empirically distinct here (ρ 0.841 vs 0.789 in AE 500–1000 kHz), so defensible to keep, but say so in the paper. `complexity` likewise is a power-weighted bandwidth proxy — and it is your headline feature, so keep it.

## 3. Recommended canonical set (14 per band)

Time domain (8): `rms`, `skewness`, `kurtosis`, `crest_factor`, `shape_factor`, `margin_factor`, `mobility`, `complexity`
Frequency domain (6): `dominant_frequency`, `center_frequency`, `spectral_bandwidth` (the fixed σ_w), `spectral_skewness` (fixed, standardised), `spectral_kurtosis` (fixed, standardised), `spectral_flatness`

Dropped, with reason:

| Dropped | Subsumed by |
|---|---|
| `std`, `variance` | = rms in sub-bands |
| `peak` | = crest_factor · rms |
| `impulse_factor` | = crest_factor · shape_factor |
| `frequency_weighted_std`, `normalized_frequency_std` | deterministic functions of center_frequency; replaced by true σ_w |
| `frequency_skewness`, `frequency_kurtosis` | replaced by the fixed standardised versions (keep one pair, not two) |
| `rms_frequency` | = √(fc² + σ_w²) once σ_w is fixed |
| `peak_frequency` | 4th-moment tail emphasis largely captured by fixed spectral_kurtosis; drop or rename+keep if you want it |
| `spectral_mean`, `spectral_std` | amplitude duplicates of rms (Parseval / Rayleigh) |

Counts become 14 × 4 = **56 AE** and 14 × 3 = **42 US** candidates (vs 104/78). Nothing of predictive value is lost — every dropped feature is recoverable from the kept ones, and VIF selection was already discarding most of them downstream anyway.

**Impact on the paper:** the headline features survive unchanged (`1–2 MHz complexity`, `500 kHz–1 MHz mobility`, sub-band `center_frequency`). Tables 2–3 shrink and lose their two todos; the "two-stage selection" story gets cleaner because stage 2 no longer spends its budget deleting tautologies; and no reviewer can object that the candidate count is padded with `std` *and* `variance` *and* `rms`.

## 4. Two ways to do it

- **Option A — principled (recommended, since results are unfrozen anyway):** implement §3 in `features.py` (fix σ_w, standardise the moments, delete duplicates), re-run 01→06, update Tables 2–3. One re-run, permanently defensible.
- **Option B — minimal:** keep `features.py` as is, drop the exact duplicates only (std, variance, peak, impulse_factor, frequency_weighted_std, normalized_frequency_std, plus one of the two skewness/kurtosis pairs) at the selection stage, and add a sentence to the paper acknowledging the remaining quirks. No new code, but the unweighted-σf oddity stays in print.

One caveat for either option: the dropped features change the VIF/correlation-stage inputs, so the retained sets (currently 32 AE / 12 US) and downstream numbers will shift — another reason to do this **before** freezing results, not after.

## 5. Bookkeeping note (relates to review §2.2)

The 92 AE / 60 US rows in the ranking CSVs are **post**-filter survivors of the 104/78 candidates: `clean_features(drop_constant=True)` removes constant columns, then the |ρ| ≥ 0.1 AND |r| ≥ 0.1 threshold (corr_min = 0.1) removes weakly correlated ones (e.g. `kurtosis` falls below it in most AE bands). The paper currently implies the threshold filter happens *after* the "candidate" count, but never states the post-threshold counts — with the canonical set, report: candidates → post-threshold → retained, with all three numbers auto-generated.
