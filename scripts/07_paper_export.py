"""
07_paper_export.py
==================
Export pipeline results as LaTeX macros and table fragments so that the paper
never contains hand-typed numbers.

Pipeline position: 7th script -- reads tables from 02-05 plus
features/metadata parquet, writes .tex fragments to
``outputs/07_paper_export/`` and copies them to ``paper/tables/``.

Outputs
-------
results_macros.tex
    ``\\newcommand`` definitions for every number quoted in the paper
    (performance metrics, p-values, CIs, feature counts, kappa statistics).
    Missing inputs yield ``\\textbf{??}`` so stale values are visible, never
    silently wrong.
table_models_tabular.tex
    Body of the model-comparison table (Table ``tab:models``).
table_top_features_tabular.tex
    Body of the top-10 feature ranking table (Table ``tab:raw_spearman``).

Usage
-----
    python scripts/07_paper_export.py
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "cs_config", ROOT / "src" / "ceramicspeed" / "config.py")
    _config = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_config)
    load_config, get_output_dir = _config.load_config, _config.get_output_dir
    try:
        cfg = load_config()
    except Exception:
        # tolerate null-padded yaml (file-sync artifact)
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8").rstrip("\x00"))
        if "machines" in cfg:  # resolve machine profile like load_config would
            import socket
            for prof in cfg.get("machines", {}).values():
                pass

    OUTPUT_DIR = get_output_dir(cfg)
except Exception as exc:  # pragma: no cover - fallback for standalone use
    print(f"WARNING: could not load config ({exc}); using outputs/")
    cfg = {}
    OUTPUT_DIR = ROOT / "outputs"

# If the configured output dir is not usable (e.g. foreign machine profile),
# fall back to the repo-local outputs/ directory.
if not (OUTPUT_DIR / "03_evaluation").is_dir() and (ROOT / "outputs" / "03_evaluation").is_dir():
    print(f"NOTE: {OUTPUT_DIR} has no pipeline outputs; falling back to repo outputs/")
    OUTPUT_DIR = ROOT / "outputs"

EXPORT_DIR = OUTPUT_DIR / "07_paper_export"
PAPER_TABLES_DIR = ROOT / "paper" / "tables"
MISSING = r"\textbf{??}"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(x, nd=3):
    return f"{float(x):.{nd}f}"


def fmt_p(p):
    """p-values: plain to 3 dp above 0.001, scientific below."""
    p = float(p)
    if p >= 0.001:
        return f"{p:.3f}"
    if p <= 0:
        return r"<10^{-300}"
    e = math.floor(math.log10(p))
    m = p / 10**e
    return rf"{m:.1f}\times 10^{{{e}}}"


def fmt_int(n):
    return f"{int(n):,}".replace(",", r"\,")


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------

MODEL_KEYS = {"ElasticNet": "Enet", "Polynomial": "Poly", "LightGBM": "Lgb"}
MODEL_DISPLAY = {"ElasticNet": "Elastic Net", "Polynomial": "Polynomial", "LightGBM": "LightGBM"}
SENSOR_KEYS = {"AE": "Ae", "UL": "Us", "US": "Us", "combined": "Comb", "Combined": "Comb"}
SENSOR_DISPLAY = {"Ae": "AE", "Us": "US", "Comb": "Combined"}

BAND_DISPLAY = {
    "AE_20-500kHz": "20--500~kHz",
    "AE_500-1000kHz": "500~kHz--1~MHz",
    "AE_1000-2000kHz": "1--2~MHz",
    "US_0-10kHz": "0--10~kHz",
    "US_10-20kHz": "10--20~kHz",
    "US_20-100kHz": "20--100~kHz",
}

FEAT_DISPLAY = {
    "rms": "RMS", "skewness": "skewness", "kurtosis": "kurtosis",
    "crest_factor": "crest factor", "shape_factor": "shape factor",
    "margin_factor": "margin factor", "mobility": "mobility",
    "complexity": "complexity", "dominant_frequency": "dominant freq.",
    "center_frequency": "centre freq.", "spectral_bandwidth": "spectral bandwidth",
    "spectral_skewness": "spectral skewness", "spectral_kurtosis": "spectral kurtosis",
    "spectral_flatness": "spectral flatness",
    # legacy names (pre-canonical feature set), kept so old CSVs still export
    "std": "std", "variance": "variance", "peak": "peak",
    "impulse_factor": "impulse factor", "spectral_mean": "spectral mean",
    "spectral_std": "spectral std", "rms_frequency": "RMS freq.",
    "peak_frequency": "$f^2$-wt.\\ RMS freq.",
    "frequency_weighted_std": "freq.-wt.\\ std",
    "normalized_frequency_std": "norm.\\ freq.\\ std",
    "frequency_skewness": "freq.\\ skewness", "frequency_kurtosis": "freq.\\ kurtosis",
    "normalized_bandwidth": "norm.\\ bandwidth",
}


def pretty_feature(name: str) -> str:
    if "__" in name:
        band, feat = name.rsplit("__", 1)
        band_disp = BAND_DISPLAY.get(band, band.replace("_", r"\_"))
        return f"{band_disp} {FEAT_DISPLAY.get(feat, feat)}"
    disp = FEAT_DISPLAY.get(name, name)
    return disp[0].upper() + disp[1:] if disp and disp[0].islower() else disp


# ---------------------------------------------------------------------------
# Macro registry -- every macro the paper uses is listed here so the file
# always defines all of them (value ?? when an input is missing).
# ---------------------------------------------------------------------------

macros: dict[str, str] = {}

for mk in MODEL_KEYS.values():
    for sk in ("Ae", "Us", "Comb"):
        for metric in ("CVrmse", "CVrmseStd", "HOrsq", "HOmae", "HOrmse"):
            macros[f"res{metric}{mk}{sk}"] = MISSING

for pair in ("AeUs", "AeComb", "UsComb"):
    for metric in ("Pcv", "Pwx", "Pdm", "Drmse", "DrmseLo", "DrmseHi"):
        macros[f"res{metric}{pair}"] = MISSING

for name in (
    "resNcandAe", "resNcandUs", "resNpostAe", "resNpostUs",
    "resNretAe", "resNretUs", "resNretComb",
    "resNsweeps", "resNtrain", "resNholdout",
    "resKappaMin", "resKappaMax", "resKappaMean",
    "resTopAeFeat", "resTopAeRho", "resTopUsFeat", "resTopUsRho",
    "resDrsqLgbAeUs", "resDrsqLgbCombAe", "resRelRmsePctLgbAe",
    "resRhoRpmKappa", "resRhoTempKappa",
    "resNrepeats", "resNouterScores", "resNfoldsCv",
    "resProxyRpmRsq", "resProxyRpmRmse", "resProxyTempRsq", "resProxyTempRmse",
    "resTwoStageRsq", "resTwoStageRmse",
    "resSnrLowDb", "resSnrMidDb", "resSnrHighDb",
    "resWithinStepRhoHC", "resWithinStepFeat", "resKneeRpmCool", "resKneeRpmHot",
    "resRhoPowRpmLow", "resRhoPowRpmMid", "resRhoPowRpmHigh",
    "resKneeKappaCool", "resKneeKappaHot",
    "resNsweepsRaw", "resNsweepsRemoved", "resWindowMs", "resWindowIntervalS",
    "resSweepsPerHold", "resRpmMinMeas", "resRpmMaxMeas",
    "resTempMinMeas", "resTempMaxMeas",
    "resShapTopFeat", "resShapTopVal", "resShapSecondFeat", "resShapSecondVal",
    "resShapThirdFeat", "resShapThirdVal",
):
    macros[name] = MISSING


def warn(msg):
    print(f"  WARNING: {msg}")


# ---------------------------------------------------------------------------
# 1. Model comparison metrics
# ---------------------------------------------------------------------------

table_models_body = None
mc_path = OUTPUT_DIR / "04_modelling" / "tables" / "model_comparison.csv"
cv_path = OUTPUT_DIR / "03_evaluation" / "tables" / "performance_table_cv.csv"

def _read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return None

mc = _read_csv(mc_path)
if mc is not None:
    mc.columns = [c.replace("²", "2") for c in mc.columns]
    mc = mc.dropna(subset=["model"])
    cv_std = {}
    cv = _read_csv(cv_path)
    if cv is not None:
        cv_std = dict(zip(cv["model"], cv["std_rmse"]))
    else:
        warn(f"{cv_path} missing -- CV std unavailable")

    rows = []
    for _, r in mc.iterrows():
        mname, sname = r["model"].split("_", 1)
        mk, sk = MODEL_KEYS[mname], SENSOR_KEYS[r["sensor"]]
        macros[f"resCVrmse{mk}{sk}"] = fmt(r["CV_RMSE"], 4)
        if r["model"] in cv_std:
            macros[f"resCVrmseStd{mk}{sk}"] = fmt(cv_std[r["model"]], 4)
        macros[f"resHOrsq{mk}{sk}"] = fmt(r["HO_R2"])
        macros[f"resHOmae{mk}{sk}"] = fmt(r["HO_MAE"])
        macros[f"resHOrmse{mk}{sk}"] = fmt(r["HO_RMSE"])
        rows.append({
            "model": MODEL_DISPLAY[mname], "sensor": SENSOR_DISPLAY[sk],
            "cv_rmse": float(r["CV_RMSE"]),
            "cv_std": float(cv_std.get(r["model"], float("nan"))),
            "ho_r2": float(r["HO_R2"]), "ho_mae": float(r["HO_MAE"]),
            "ho_rmse": float(r["HO_RMSE"]),
        })

    # Derived deltas
    ho = {(t["model"], t["sensor"]): t["ho_r2"] for t in rows}
    if ("LightGBM", "AE") in ho and ("LightGBM", "US") in ho:
        macros["resDrsqLgbAeUs"] = fmt(ho[("LightGBM", "AE")] - ho[("LightGBM", "US")], 2)
    if ("LightGBM", "Combined") in ho and ("LightGBM", "AE") in ho:
        macros["resDrsqLgbCombAe"] = fmt(ho[("LightGBM", "Combined")] - ho[("LightGBM", "AE")], 3)

    # Table body — sorted by CV RMSE ascending (best first)
    rows.sort(key=lambda t: t["cv_rmse"])
    best_r2 = max(t["ho_r2"] for t in rows)
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Sensors} &"
             r" \textbf{CV RMSE} & \textbf{HO $R^2$} & \textbf{HO MAE} & \textbf{HO RMSE} \\",
             r"\midrule"]
    for t in rows:
        cvs = "" if math.isnan(t["cv_std"]) else rf" \pm {t['cv_std']:.4f}"
        r2 = rf"\textbf{{{t['ho_r2']:.3f}}}" if t["ho_r2"] == best_r2 else f"{t['ho_r2']:.3f}"
        lines.append(
            f"{t['model']:<11s} & {t['sensor']:<8s} & ${t['cv_rmse']:.4f}{cvs}$ & "
            f"{r2} & {t['ho_mae']:.3f} & {t['ho_rmse']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table_models_body = "\n".join(lines)
else:
    warn(f"{mc_path} missing -- model metrics unavailable")

# ---------------------------------------------------------------------------
# 1b. CV design parameters
# ---------------------------------------------------------------------------

_cv_design = _read_csv(cv_path)
if _cv_design is not None:
    _cv_design = _cv_design.dropna(subset=["model"])
if _cv_design is not None and "n_scores" in _cv_design.columns:
    n_scores = int(_cv_design["n_scores"].iloc[0])
    n_folds = int((cfg.get("modelling") or {}).get("cv_n_splits", 5))
    macros["resNouterScores"] = str(n_scores)
    macros["resNfoldsCv"] = str(n_folds)
    macros["resNrepeats"] = str(n_scores // n_folds)
else:
    warn("performance_table_cv.csv missing n_scores -- CV design macros unavailable")

# ---------------------------------------------------------------------------
# 2. Cross-feature-set significance tests
# ---------------------------------------------------------------------------

ct_path = OUTPUT_DIR / "05_holdout_tests" / "tables" / "stat_tests_cross_featureset.csv"
ct = _read_csv(ct_path)
if ct is not None:
    ct = ct.dropna(subset=["model_a"])
    for _, r in ct.iterrows():
        sa = SENSOR_KEYS[r["model_a"].split("_", 1)[1]]
        sb = SENSOR_KEYS[r["model_b"].split("_", 1)[1]]
        pair = f"{sa}{sb}"
        if pair not in ("AeUs", "AeComb", "UsComb"):
            pair = f"{sb}{sa}"
        d, lo, hi = float(r["delta_rmse"]), float(r["ci_95_lo"]), float(r["ci_95_hi"])
        if d < 0:  # orient so the quoted gap is positive (improvement of better model)
            d, lo, hi = -d, -hi, -lo
        macros[f"resPcv{pair}"] = fmt_p(r["cv_p_value"])
        macros[f"resPwx{pair}"] = fmt_p(r["wilcoxon_p"])
        macros[f"resPdm{pair}"] = fmt_p(r["dm_p"])
        macros[f"resDrmse{pair}"] = fmt(d, 4)
        macros[f"resDrmseLo{pair}"] = fmt(lo, 4)
        macros[f"resDrmseHi{pair}"] = fmt(hi, 4)
        macros["resNholdout"] = fmt_int(r["n_common_sweeps"])
else:
    warn(f"{ct_path} missing -- significance tests unavailable")

# ---------------------------------------------------------------------------
# 3. Feature counts and rankings
# ---------------------------------------------------------------------------

try:
    _spec = _ilu.spec_from_file_location(
        "cs_features", ROOT / "src" / "ceramicspeed" / "features.py")
    _features = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_features)
    n_feat = len(_features.FEATURE_NAMES)
    bands = cfg.get("frequency_bands", {})
    if bands:
        macros["resNcandAe"] = str((len(bands.get("AE", [])) + 1) * n_feat)
        macros["resNcandUs"] = str((len(bands.get("UL", [])) + 1) * n_feat)
except Exception as exc:
    warn(f"could not derive candidate counts ({exc})")

fs_path = OUTPUT_DIR / "feature_selection.json"
try:
    sel = json.loads(fs_path.read_text().rstrip("\x00"))
except (FileNotFoundError, OSError):
    sel = None
if sel is not None:
    n_ae = len(sel.get("AE", {}).get("retained", []))
    n_us = len(sel.get("UL", sel.get("US", {})).get("retained", []))
    macros["resNretAe"], macros["resNretUs"] = str(n_ae), str(n_us)
    macros["resNretComb"] = str(n_ae + n_us)
else:
    warn(f"{fs_path} missing -- retained counts unavailable")

table_top_feats_body = None
rk_ae_path = OUTPUT_DIR / "02_feature_analysis" / "tables" / "feature_ranking_ae.csv"
rk_us_path = OUTPUT_DIR / "02_feature_analysis" / "tables" / "feature_ranking_us.csv"
rk_ae, rk_us = _read_csv(rk_ae_path), _read_csv(rk_us_path)
if rk_ae is not None and rk_us is not None:
    macros["resNpostAe"], macros["resNpostUs"] = str(len(rk_ae)), str(len(rk_us))
    macros["resTopAeFeat"] = pretty_feature(rk_ae.iloc[0]["feature"])
    macros["resTopAeRho"] = fmt(rk_ae.iloc[0]["|rho|"])
    macros["resTopUsFeat"] = pretty_feature(rk_us.iloc[0]["feature"])
    macros["resTopUsRho"] = fmt(rk_us.iloc[0]["|rho|"])

    lines = [r"\begin{tabular}{lccl ccc}", r"\toprule",
             r"\multicolumn{3}{c}{\textbf{AE}} & \phantom{x} &"
             r" \multicolumn{3}{c}{\textbf{US}} \\",
             r"\cmidrule(lr){1-3}\cmidrule(lr){5-7}",
             r"Feature & $|\rho|$ & $|r|$ & & Feature & $|\rho|$ & $|r|$ \\",
             r"\midrule"]
    for i in range(10):
        a = rk_ae.iloc[i] if i < len(rk_ae) else None
        u = rk_us.iloc[i] if i < len(rk_us) else None
        _ret_ae = set(sel.get("AE", {}).get("retained", [])) if sel else set()
        _ret_us = set(sel.get("UL", sel.get("US", {})).get("retained", [])) if sel else set()
        def _mark(row, ret):
            d = r"$^{\dagger}$" if row["feature"] in ret else ""
            return f"{pretty_feature(row['feature'])}{d} & {row['|rho|']:.3f} & {row['|r|']:.3f}"
        left = _mark(a, _ret_ae) if a is not None else " & & "
        right = _mark(u, _ret_us) if u is not None else " & & "
        lines.append(f"{left} & & {right} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table_top_feats_body = "\n".join(lines)
else:
    warn("feature ranking CSVs missing -- top-feature table unavailable")

# ---------------------------------------------------------------------------
# 3b. Acquisition / operating-condition statistics (eda stats json)
# ---------------------------------------------------------------------------

oc_path = OUTPUT_DIR / "eda" / "operating_conditions_stats.json"
try:
    oc = json.loads(oc_path.read_text().rstrip("\x00"))
    ds, acq, opc = oc["dataset"], oc["acquisition"], oc["operating_conditions"]
    macros["resNsweepsRaw"] = fmt_int(ds["n_sweeps_raw"])
    macros["resNsweepsRemoved"] = fmt_int(ds["n_sweeps_removed"])
    macros["resNsweeps"] = fmt_int(ds["n_sweeps_retained"])
    macros["resWindowMs"] = f"{acq['waveform_duration_ms']:.0f}"
    macros["resWindowIntervalS"] = f"{acq['window_interval_s']:.1f}"
    macros["resSweepsPerHold"] = f"{acq['ramp_dwell_windows_median']:.0f}"
    macros["resRpmMinMeas"] = f"{opc['rpm_min']:.0f}"
    macros["resRpmMaxMeas"] = f"{opc['rpm_max']:.0f}"
    macros["resTempMinMeas"] = f"{opc['temperature_min']:.0f}"
    macros["resTempMaxMeas"] = f"{opc['temperature_max']:.0f}"
    macros["resKappaMin"] = fmt(opc["kappa_min"], 2)
    macros["resKappaMax"] = fmt(opc["kappa_max"], 2)
    macros["resKappaMean"] = fmt(opc["kappa_mean"], 2)
    if macros["resNholdout"] != MISSING:
        n_ho = int(macros["resNholdout"].replace("\\,", ""))
        macros["resNtrain"] = fmt_int(ds["n_sweeps_retained"] - n_ho)
except (FileNotFoundError, OSError, KeyError) as exc:
    warn(f"{oc_path} unavailable ({exc}); falling back to parquet for kappa stats")

# ---------------------------------------------------------------------------
# 4. Dataset / kappa statistics (needs parquet pair)
# ---------------------------------------------------------------------------

try:
    from scipy.stats import spearmanr
    _spec = _ilu.spec_from_file_location(
        "cs_kappa", ROOT / "src" / "ceramicspeed" / "calculate_kappa.py")
    _ck = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ck)
    calculate_kappa = _ck.calculate_kappa

    feat_df = pd.read_parquet(OUTPUT_DIR / "features.parquet")
    meta_df = pd.read_parquet(OUTPUT_DIR / "metadata.parquet")
    ae_meta = meta_df[feat_df["sensor"] == "AE"].reset_index(drop=True)
    n_sweeps = int(feat_df.loc[feat_df["sensor"] == "AE", "sweep"].nunique())
    if macros["resNsweeps"] == MISSING:
        macros["resNsweeps"] = fmt_int(n_sweeps)
    if macros["resNtrain"] == MISSING and macros["resNholdout"] != MISSING:
        n_ho = int(macros["resNholdout"].replace("\\,", ""))
        macros["resNtrain"] = fmt_int(n_sweeps - n_ho)

    kappa = ae_meta.apply(
        lambda row: calculate_kappa(
            rpm=row["rpm"], temp_c=row["temperature_c"],
            d_pw=cfg["bearing"]["d_pw_mm"],
            nu_40=row["viscosity_40c_cst"], nu_100=row["viscosity_100c_cst"],
        ), axis=1)
    if macros["resKappaMin"] == MISSING:
        macros["resKappaMin"] = fmt(kappa.min(), 2)
        macros["resKappaMax"] = fmt(kappa.max(), 2)
        macros["resKappaMean"] = fmt(kappa.mean(), 2)
    macros["resRhoRpmKappa"] = f"{spearmanr(ae_meta['rpm'], kappa)[0]:+.2f}"
    macros["resRhoTempKappa"] = f"{spearmanr(ae_meta['temperature_c'], kappa)[0]:+.2f}"

    if macros["resHOrmseLgbAe"] != MISSING:
        rel = 100 * float(macros["resHOrmseLgbAe"]) / kappa.mean()
        macros["resRelRmsePctLgbAe"] = f"{rel:.0f}"
except Exception as exc:
    warn(f"parquet-based statistics unavailable ({exc})")

# ---------------------------------------------------------------------------
# 5. Diagnostics: proxy models (09) and band validation/mechanism (08, 10)
# ---------------------------------------------------------------------------

try:
    px = json.loads((OUTPUT_DIR / "09_proxy_diagnostics" / "proxy_stats.json")
                    .read_text().rstrip("\x00"))
    macros["resProxyRpmRsq"] = fmt(px["rpm_r2"])
    macros["resProxyRpmRmse"] = f"{px['rpm_rmse']:.0f}"
    macros["resProxyTempRsq"] = fmt(px["temp_r2"])
    macros["resProxyTempRmse"] = f"{px['temp_rmse']:.1f}"
    macros["resTwoStageRsq"] = fmt(px["two_stage_r2"])
    macros["resTwoStageRmse"] = fmt(px["two_stage_rmse"])
except (FileNotFoundError, OSError, KeyError) as exc:
    warn(f"proxy_stats.json unavailable ({exc})")

try:
    bv = json.loads((OUTPUT_DIR / "08_band_validation" / "band_validation_stats.json")
                    .read_text().rstrip("\x00"))
    bands = bv["bands"]
    for key, name in [("resSnrLowDb", "AE_20-500kHz"),
                      ("resSnrMidDb", "AE_500-1000kHz"), ("resSnrHighDb", "AE_1000-2000kHz")]:
        macros[key] = f"{bands[name]['snr_db_median']:+.0f}"
except (FileNotFoundError, OSError, KeyError) as exc:
    warn(f"band_validation_stats.json unavailable ({exc})")

try:
    bm = json.loads((OUTPUT_DIR / "10_band_mechanism" / "tw_45-55C" /
                     "band_mechanism_stats.json").read_text().rstrip("\x00"))
    _wsb = bm["test_B_within_step"]
    _best = max(_wsb, key=lambda k: abs(_wsb[k]["median_rho"]))
    macros["resWithinStepRhoHC"] = f"{_wsb[_best]['median_rho']:+.2f}"
    macros["resWithinStepFeat"] = pretty_feature(_best)
except (FileNotFoundError, OSError, KeyError) as exc:
    warn(f"band_mechanism_stats.json unavailable ({exc})")

# ---------------------------------------------------------------------------
# 5b. SHAP importances (LightGBM, AE) -- top contributors quoted in the prose
# ---------------------------------------------------------------------------
try:
    shap_imp = pd.read_csv(
        OUTPUT_DIR / "04_modelling" / "shap" / "shap_importance_lightgbm_ae.csv",
        index_col=0,
    ).sort_values("mean_abs_shap", ascending=False)
    for rank, key in ((0, "Top"), (1, "Second"), (2, "Third")):
        macros[f"resShap{key}Feat"] = pretty_feature(str(shap_imp.index[rank]))
        macros[f"resShap{key}Val"] = fmt(float(shap_imp["mean_abs_shap"].iloc[rank]))
    print(f"SHAP top-3 (LightGBM AE): {shap_imp.index[0]}="
          f"{shap_imp['mean_abs_shap'].iloc[0]:.3f}")
except (FileNotFoundError, OSError, KeyError, IndexError) as exc:
    warn(f"shap importance unavailable ({exc})")

# Cross-model SHAP agreement table (tab:shap_agree): base features that fall in the
# top-k by mean |SHAP| of more than one model. Ranks are positions in each model's
# full importance list -- for the polynomial that list includes interaction and
# squared terms, so a base feature's rank reflects competition with those terms.
table_shap_body = ""
try:
    _topk = int(cfg.get("evaluation", {}).get("shap_top_k", 10))
    _shap_dir = OUTPUT_DIR / "04_modelling" / "shap"
    _models = (("ElasticNet", "elasticnet"), ("Polynomial", "polynomial"), ("LightGBM", "lightgbm"))
    _ranks: dict[str, dict[str, int]] = {}
    for _disp, _key in _models:
        _imp = pd.read_csv(_shap_dir / f"shap_importance_{_key}_ae.csv", index_col=0)
        _imp = _imp.sort_values("mean_abs_shap", ascending=False)
        _ranks[_disp] = {str(f): i + 1 for i, f in enumerate(_imp.index)}

    def _is_base(f: str) -> bool:  # exclude interaction (" ") and squared ("^") terms
        return (" " not in f) and ("^" not in f)

    _agg: dict[str, dict[str, int]] = {}
    for _disp, _ in _models:
        for _feat, _rk in _ranks[_disp].items():
            if _rk <= _topk and _is_base(_feat):
                _agg.setdefault(_feat, {})[_disp] = _rk
    _rows = [(f, mr) for f, mr in _agg.items() if len(mr) >= 2]
    _rows.sort(key=lambda it: (-len(it[1]), sum(it[1].values()),
                               it[1].get("LightGBM", 99), it[0]))

    def _disp_feat(f: str) -> str:
        return pretty_feature(f) if "__" in f else "Broadband " + FEAT_DISPLAY.get(f, f)

    _lines = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Feature} & \textbf{ElasticNet} & \textbf{Polynomial} & "
        rf"\textbf{{LightGBM}} & \textbf{{Models in top {_topk}}} \\",
        r"\midrule",
    ]
    for _feat, _mr in _rows:
        _e, _p, _l = (_mr.get(m, "--") for m in ("ElasticNet", "Polynomial", "LightGBM"))
        _lines.append(rf"{_disp_feat(_feat)} & {_e} & {_p} & {_l} & {len(_mr)} \\")
    _lines += [r"\bottomrule", r"\end{tabular}"]
    table_shap_body = "\n".join(_lines)
    print(f"SHAP agreement table: {len(_rows)} features (top-{_topk})")
except (FileNotFoundError, OSError, KeyError) as exc:
    warn(f"shap agreement table unavailable ({exc})")

# Knee of the 1-2 MHz broadband decay vs RPM, per temperature window
try:
    import importlib.util as _ilu2
    _spec2 = _ilu2.spec_from_file_location("cs_kappa2", ROOT / "src" / "ceramicspeed" / "calculate_kappa.py")
    _ck2 = _ilu2.module_from_spec(_spec2); _spec2.loader.exec_module(_ck2)
    for tw, tmid, suffix in [("tw_45-55C", 50.0, "Cool"), ("tw_85-95C", 90.0, "Hot")]:
        cs = _read_csv(OUTPUT_DIR / "10_band_mechanism" / tw / "tables" / "comb_strip.csv")
        if cs is None:
            continue
        b = cs[(cs["band"] == "AE_1000-2000kHz") & (cs["rpm"] >= 60)].sort_values("rpm")
        floor = b.nlargest(8, "rpm")["p_total"].median()
        above = b[b["p_total"] > 2 * floor]
        if above.empty:
            continue
        knee = float(above["rpm"].max())
        if suffix == "Cool":
            from scipy.stats import spearmanr as _sp
            for band, key in [("AE_20-500kHz", "resRhoPowRpmLow"),
                              ("AE_500-1000kHz", "resRhoPowRpmMid"),
                              ("AE_1000-2000kHz", "resRhoPowRpmHigh")]:
                bb = cs[(cs["band"] == band) & (cs["rpm"] >= 60)]
                macros[key] = f"{_sp(bb['rpm'], bb['p_total'])[0]:+.2f}"
        macros[f"resKneeRpm{suffix}"] = f"{knee:.0f}"
        macros[f"resKneeKappa{suffix}"] = fmt(_ck2.calculate_kappa(
            rpm=knee, temp_c=tmid, d_pw=cfg["bearing"]["d_pw_mm"], nu_40=22.0, nu_100=4.1), 2)
except Exception as exc:
    warn(f"knee analysis unavailable ({exc})")

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

header = ("% Auto-generated by scripts/07_paper_export.py -- do not edit by hand.\n"
          "% Values shown as ?? indicate a missing pipeline output; re-run the\n"
          "% relevant script and then this one.\n")

macro_lines = [header]
for name, val in macros.items():
    macro_lines.append(rf"\newcommand{{\{name}}}{{{val}}}")
macros_tex = "\n".join(macro_lines) + "\n"

files = {"results_macros.tex": macros_tex}
if table_models_body:
    files["table_models_tabular.tex"] = header + table_models_body + "\n"
if table_top_feats_body:
    files["table_top_features_tabular.tex"] = header + table_top_feats_body + "\n"
if table_shap_body:
    files["table_shap_agreement_tabular.tex"] = header + table_shap_body + "\n"

# Refresh the appendix ranking-table copies so the paper compiles without outputs/
for _n in ("ae", "us"):
    _src = OUTPUT_DIR / "02_feature_analysis" / "tables" / f"feature_ranking_{_n}_appendix.tex"
    try:
        _txt = _src.read_text(encoding="utf-8").rstrip("\x00")
        (PAPER_TABLES_DIR / _src.name).write_text(_txt, encoding="utf-8")
        print(f"Copied {_src.name} to paper/tables")
    except (FileNotFoundError, OSError):
        warn(f"{_src} unavailable -- appendix copy not refreshed")

for fname, content in files.items():
    for dest in (EXPORT_DIR / fname, PAPER_TABLES_DIR / fname):
        dest.write_text(content, encoding="utf-8")
    print(f"Wrote {fname} ({len(content)} bytes) to outputs/07_paper_export and paper/tables")

n_missing = sum(1 for v in macros.values() if v == MISSING)
print(f"Done: {len(macros)} macros defined, {n_missing} missing (??).")
