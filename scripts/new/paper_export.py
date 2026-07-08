"""
paper_export.py
===============
Generate LaTeX macros + the model-performance table for the paper from the NEW
pipeline's result CSVs (outputs/new/). The analogue of the legacy
07_paper_export.py, for the pooled-GroupKFold / group-paired-DM protocol.

Emits the exact macro names referenced by modelling.tex (\\resPooledRsq*,
\\resRpmRsq*, \\resDeltaRmse*, \\resDmGroup*, ...). Each macro is written as
    \\providecommand{\\X}{}\\renewcommand{\\X}{value}
so it is collision-proof and always overrides the \\providecommand fallbacks in
main.tex, regardless of load order or legacy definitions.

Reads:
  outputs/new/regression/featureset_comparison.csv        (pooled R2/MAE/RMSE, cv R2)
  outputs/new/decomposition/tables/decomposition_summary.csv
  outputs/new/group_paired_tests/channel_comparison.csv   (AE vs US)
  outputs/new/group_paired_tests/complementarity_tests.csv (fusion contrasts)

Writes (paper/tables/ and outputs/new/paper_export/):
  results_macros_new.tex            precise \\newcommand-equivalents for prose
  table_models_tabular.tex          model x sensor performance table body

Usage
-----
    python scripts/new/paper_export.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from ceramicspeed.config import load_config, get_output_dir

cfg = load_config()
NEW = get_output_dir(cfg) / "new"
REG = NEW / "regression"
GPT = NEW / "group_paired_tests"
DEC = NEW / "decomposition" / "tables"

PAPER_TABLES = Path(__file__).resolve().parent.parent.parent / "paper" / "tables"
OUT = NEW / "paper_export"
OUT.mkdir(parents=True, exist_ok=True)

_CHAN = {"AE": "Ae", "US": "Us", "Combined": "Comb"}
_MODEL = {"ElasticNet": "Enet", "LightGBM": "Lgb"}
_DISP = {"ElasticNet": "Elastic Net", "LightGBM": "LightGBM"}


def _pfmt(p):
    return r"$<0.001$" if p < 1e-3 else f"${p:.3f}$"


# ---------------------------------------------------------------------------
perf = pd.read_csv(REG / "featureset_comparison.csv")
dec = pd.read_csv(DEC / "decomposition_summary.csv")
chan = pd.read_csv(GPT / "channel_comparison.csv")
comp = pd.read_csv(GPT / "complementarity_tests.csv")

# quick lookups
r2 = {(r.model, r.target): r.holdout_r2 for r in perf.itertuples()}
rmse = {(r.model, r.target): r.holdout_rmse for r in perf.itertuples()}

macros: dict[str, str] = {}

macros["resNgroups"] = str(int(chan["n_groups"].iloc[0]))
macros["resNwindowsPooled"] = str(int(perf["n_pooled"].iloc[0]))
macros["resNcandAe"] = str(int(perf[perf.target == "AE"]["n_features"].iloc[0]))
macros["resNcandUs"] = str(int(perf[perf.target == "US"]["n_features"].iloc[0]))

# pooled R2 / RMSE per model x channel
for r in perf.itertuples():
    tag = _MODEL[r.model] + _CHAN[r.target]
    macros[f"resPooledRsq{tag}"] = f"{r.holdout_r2:.3f}"
    macros[f"resPooledRmse{tag}"] = f"{r.holdout_rmse:.3f}"

# decomposition
for r in dec.itertuples():
    t = _CHAN[r.channel]
    macros[f"resRpmRsq{t}"] = f"{r.r2_rpm:.3f}"
    macros[f"resTempRsq{t}"] = f"{r.r2_temp:.3f}"
    macros[f"resTwoStageRsq{t}"] = f"{r.r2_two_stage_kappa:.3f}"
    macros[f"resDirectRsq{t}"] = f"{r.r2_direct_kappa:.3f}"
    macros[f"resResidRsq{t}"] = f"${r.residual_direct_minus_twostage:+.3f}$"

# DM inference (LightGBM; ΔRMSE with sign = second − first, so + means first better)
_pair = {"AE vs US": ("AE", "US", "AeUs"),
         "Combined vs AE": ("Combined", "AE", "CombAe"),
         "Combined vs US": ("Combined", "US", "CombUs")}
dm = pd.concat([chan, comp], ignore_index=True)
for r in dm.itertuples():  # both models; LightGBM unsuffixed, Elastic Net "Enet"-suffixed
    a, b, tag = _pair[r.contrast]
    suffix = "" if r.model == "LightGBM" else "Enet"
    d_rmse = rmse[(r.model, b)] - rmse[(r.model, a)]
    macros[f"resDeltaRmse{suffix}{tag}"] = f"${d_rmse:+.3f}$"
    macros[f"resDmGroup{suffix}{tag}"] = _pfmt(r.p_group)
    macros[f"resDmNaive{suffix}{tag}"] = _pfmt(r.p_window_naive)

# clustered SHAP: summed mean |SHAP| of the leading two clusters per channel
for ch, fn in (("Ae", "clustered_shap_ae.csv"), ("Us", "clustered_shap_us.csv")):
    s = pd.read_csv(NEW / "clustered_shap" / fn).sort_values("total_shap", ascending=False)
    macros[f"resShap{ch}Top"] = f"{s.iloc[0]['total_shap']:.3f}"
    macros[f"resShap{ch}Second"] = f"{s.iloc[1]['total_shap']:.3f}"

lines = [rf"\providecommand{{\{k}}}{{}}\renewcommand{{\{k}}}{{{v}}}" for k, v in macros.items()]
(PAPER_TABLES / "results_macros_new.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "results_macros_new.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(macros)} macros -> results_macros_new.tex")


# ---------------------------------------------------------------------------
# table_models_tabular.tex : Sensor x Model -> CV R2, pooled R2, RMSE, MAE
# ---------------------------------------------------------------------------
best = perf.loc[perf.groupby("target")["holdout_r2"].idxmax()].set_index("target")["model"].to_dict()
tab = [r"\begin{tabular}{llcccc}", r"\toprule",
       r"Sensor & Model & CV $R^2$ & $R^2$ & RMSE & MAE \\", r"\midrule"]
for ch in ("AE", "US", "Combined"):
    for m in ("ElasticNet", "LightGBM"):
        r = perf[(perf.model == m) & (perf.target == ch)].iloc[0]
        r2c = f"{r['holdout_r2']:.3f}"
        if best[ch] == m:
            r2c = rf"\textbf{{{r2c}}}"
        sensor = ch if m == "ElasticNet" else ""
        tab.append(f"{sensor} & {_DISP[m]} & {r['cv_r2_grouped']:.3f} & {r2c} & "
                   f"{r['holdout_rmse']:.3f} & {r['holdout_mae']:.3f} \\\\")
    if ch != "Combined":
        tab.append(r"\midrule")
tab += [r"\bottomrule", r"\end{tabular}"]
(PAPER_TABLES / "table_models_tabular.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")
print("Wrote table_models_tabular.tex")
print(f"\nArtifacts -> {PAPER_TABLES}")

if __name__ == "__main__":
    print("\npaper_export complete.")
