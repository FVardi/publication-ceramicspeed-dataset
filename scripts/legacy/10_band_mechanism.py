"""
10_band_mechanism.py
====================
Mechanism tests for the AE frequency sub-bands: is the band content
bearing-generated or drive (VFD) EMI?

Standalone diagnostic; writes only to its own output folder.

Test A -- comb stripping (raw HDF5)
-----------------------------------
For sweeps spanning the RPM range at roughly constant temperature, the Welch
PSD is split into a line component (the VFD comb) and a broadband residual by
median-filtering the PSD across frequency. If the *residual broadband* power
in a band rises with RPM, the band contains rotation-generated content
beneath the comb; if only the line component changes, the band is EMI.

Test B -- within-RPM-step temperature sensitivity (features parquet)
--------------------------------------------------------------------
At a fixed staircase RPM step the drive state is ~constant while lubricant
viscosity varies strongly across the 13 temperature blocks. Within each step,
the Spearman correlation of each feature with temperature measures whatever
the feature senses *beyond* speed. A pure drive-tracker is flat; a
lubrication-sensitive feature co-varies with temperature.

Outputs (outputs/10_band_mechanism/)
------------------------------------
tables/comb_strip.csv         per-sweep band powers (total / broadband / line)
tables/within_step_rho.csv    per-step Spearman(feature, temperature)
figures/comb_strip_vs_rpm.png
figures/within_step_rho.png
band_mechanism_stats.json

Usage
-----
    python scripts/10_band_mechanism.py
    python scripts/10_band_mechanism.py --temp-window 45 55 --output-dir outputs
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

# numpy 2.x renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
from scipy.ndimage import median_filter
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util as _ilu


def _load_module(name, rel):
    spec = _ilu.spec_from_file_location(name, ROOT / rel)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_config = _load_module("cs_config", "src/ceramicspeed/config.py")
try:
    cfg = _config.load_config()
except Exception:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8").rstrip("\x00"))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data-file", type=str, default=None)
p.add_argument("--features-parquet", type=str, default=None)
p.add_argument("--metadata-parquet", type=str, default=None)
p.add_argument("--output-dir", type=str, default=None)
p.add_argument("--temp-window", type=float, nargs=2, default=[45.0, 55.0],
               help="temperature range for Test A (holds temp ~constant)")
p.add_argument("--nperseg", type=int, default=1 << 16)
p.add_argument("--median-kernel", type=int, default=31,
               help="median filter width in PSD bins (must span > line width, < comb spacing x2)")
p.add_argument("--per-step", type=int, default=10, help="sweeps per RPM step in Test A")
args, _ = p.parse_known_args()

if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    try:
        OUTPUT_DIR = _config.get_output_dir(cfg)
    except Exception:
        OUTPUT_DIR = ROOT / "outputs"
    if not OUTPUT_DIR.exists() and (ROOT / "outputs").exists():
        OUTPUT_DIR = ROOT / "outputs"

_tw = f"tw_{args.temp_window[0]:.0f}-{args.temp_window[1]:.0f}C"
SCRIPT_DIR = OUTPUT_DIR / "10_band_mechanism" / _tw
TABLES_DIR, FIGURES_DIR = SCRIPT_DIR / "tables", SCRIPT_DIR / "figures"
for d in (SCRIPT_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

BANDS = [(b["f_lo"], b["f_hi"], b["label"]) for b in cfg["frequency_bands"]["AE"]]
BANDS += [(b["f_lo"], b["f_hi"], b["label"])
          for b in cfg.get("validation_extra_bands", {}).get("AE", [])]

if args.data_file:
    DATA_FILE = Path(args.data_file)
else:
    input_dir = Path(cfg["paths"]["input_dir"])
    if not input_dir.is_dir():
        input_dir = ROOT / "data"
    pats = cfg["filters"].get("file_patterns") or ["*"]
    DATA_FILE = sorted(pp for pp in input_dir.iterdir()
                       if pp.suffix in (".hdf5", ".h5") and any(s in pp.stem for s in pats))[0]
print(f"Test A data file: {DATA_FILE}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Test A: comb stripping
# =============================================================================

t_lo, t_hi = args.temp_window
with h5py.File(DATA_FILE, "r") as f:
    sw = f["sweeps"]
    names = sorted(sw.keys(), key=lambda n: int(n.split("_")[1]))
    attr_rows = []
    for n in names:
        a = dict(sw[n].attrs)
        attr_rows.append({
            "sweep": n,
            "rpm": float(a.get("rpm", a.get("telem_rpm_meas", np.nan))),
            "temp": float(a.get("temperature_c", a.get("telem_omron_pv_c", np.nan))),
        })
    idx = pd.DataFrame(attr_rows).dropna()

    cand = idx[(idx["temp"] >= t_lo) & (idx["temp"] <= t_hi) & (idx["rpm"] >= 60) & (idx["rpm"] <= 3000)]
    sel = []
    for tgt in np.arange(200, 3000, 200):
        c = cand.copy(); c["d"] = (c["rpm"] - tgt).abs()
        sel.append(c.nsmallest(args.per_step, "d"))
    # stationary warm-up reference (motor off; heater active) within/near window
    still = idx[idx["rpm"] < 60].copy()
    sel.append(still.tail(3))
    sel_df = pd.concat(sel).drop_duplicates("sweep")
    print(f"Test A: {len(sel_df)} sweeps, temp {t_lo}-{t_hi} C")

    rows = []
    for _, r in sel_df.iterrows():
        x = sw[r["sweep"]]["AE"]["voltage"][()].astype(float)
        fs = 12.5e6 if "fs" not in dir() else fs  # noqa
        fxx, pxx = welch(x, fs=12.5e6, nperseg=min(args.nperseg, len(x)))
        broad = median_filter(pxx, size=args.median_kernel)
        line = np.clip(pxx - broad, 0, None)
        for f_lo, f_hi, label in BANDS:
            m = (fxx >= f_lo) & (fxx < f_hi)
            rows.append({
                "sweep": r["sweep"], "rpm": r["rpm"], "temp": r["temp"], "band": label,
                "p_total": float(_trapz(pxx[m], fxx[m])),
                "p_broad": float(_trapz(broad[m], fxx[m])),
                "p_line": float(_trapz(line[m], fxx[m])),
            })
comb = pd.DataFrame(rows)
comb["line_share"] = comb["p_line"] / comb["p_total"]
comb.to_csv(TABLES_DIR / "comb_strip.csv", index=False)

run = comb[comb["rpm"] >= 60]
print("\nTest A -- Spearman(rho) of residual broadband power with RPM (running sweeps):")
statsA = {}
for label in run["band"].unique():
    sub = run[run["band"] == label]
    rho_b = spearmanr(sub["rpm"], sub["p_broad"])[0]
    rho_l = spearmanr(sub["rpm"], sub["p_line"])[0]
    ls = sub["line_share"].median()
    statsA[label] = {"rho_broad_rpm": round(float(rho_b), 3),
                     "rho_line_rpm": round(float(rho_l), 3),
                     "line_share_median": round(float(ls), 3)}
    print(f"  {label:18s} broadband-vs-rpm rho={rho_b:+.2f}   line-vs-rpm rho={rho_l:+.2f}   "
          f"median line share={ls:.0%}")

fig, axes = plt.subplots(1, len(BANDS), figsize=(5 * len(BANDS), 4.2), sharex=True)
for ax, (f_lo, f_hi, label) in zip(np.atleast_1d(axes), BANDS):
    sub = comb[comb["band"] == label]
    r0 = sub[sub["rpm"] < 60]
    rr = sub[sub["rpm"] >= 60]
    ax.semilogy(rr["rpm"], rr["p_broad"], "o", ms=5, label="broadband residual")
    ax.semilogy(rr["rpm"], rr["p_line"], "s", ms=5, alpha=0.7, label="line (comb) component")
    if not r0.empty:
        ax.axhline(r0["p_broad"].median(), color="k", ls="--", lw=1, label="stationary broadband")
    ax.set_title(f"{label}  (temp {t_lo}-{t_hi}°C)")
    ax.set_xlabel("RPM")
ax0 = np.atleast_1d(axes)[0]
ax0.set_ylabel("band power [V²]")
ax0.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "comb_strip_vs_rpm.png", dpi=150)
plt.close(fig)

# =============================================================================
# Test B: within-RPM-step temperature sensitivity
# =============================================================================

fp = Path(args.features_parquet) if args.features_parquet else OUTPUT_DIR / "features.parquet"
mp = Path(args.metadata_parquet) if args.metadata_parquet else OUTPUT_DIR / "metadata.parquet"
feat = pd.read_parquet(fp)
meta = pd.read_parquet(mp)
mask = (feat["sensor"] == "AE").values
feat, meta = feat[mask].reset_index(drop=True), meta[mask].reset_index(drop=True)

TEST_FEATURES = [c for c in [
    "AE_1000-2000kHz__complexity", "AE_1000-2000kHz__mobility", "AE_1000-2000kHz__rms",
    "AE_500-1000kHz__complexity", "AE_500-1000kHz__mobility",
    "AE_20-500kHz__complexity", "AE_20-500kHz__mobility", "AE_20-500kHz__rms",
] if c in feat.columns]

df = feat[TEST_FEATURES].copy()
df["rpm"], df["temp"] = meta["rpm"].values, meta["temperature_c"].values
df = df[(df["rpm"] >= 60) & (df["rpm"] <= 3000)]
df["step"] = (df["rpm"] / 100).round() * 100

rows = []
for step, sub in df.groupby("step"):
    if sub["temp"].nunique() < 8 or len(sub) < 30:
        continue
    for c in TEST_FEATURES:
        rho = spearmanr(sub[c], sub["temp"])[0]
        rows.append({"step": step, "n": len(sub), "feature": c, "rho_temp": rho})
ws = pd.DataFrame(rows)
ws.to_csv(TABLES_DIR / "within_step_rho.csv", index=False)

print("\nTest B -- within-RPM-step Spearman(feature, temperature), median |rho| across steps:")
statsB = {}
med = ws.groupby("feature")["rho_temp"].agg(median_rho="median",
                                            median_abs=lambda s: s.abs().median(),
                                            frac_negative=lambda s: (s < 0).mean())
for c, r in med.sort_values("median_abs", ascending=False).iterrows():
    statsB[c] = {k: round(float(v), 3) for k, v in r.items()}
    print(f"  {c:34s} median rho={r['median_rho']:+.2f}  median |rho|={r['median_abs']:.2f}  "
          f"frac neg={r['frac_negative']:.0%}")

fig, ax = plt.subplots(figsize=(9, 5))
for c in TEST_FEATURES:
    sub = ws[ws["feature"] == c]
    ax.plot(sub["step"], sub["rho_temp"], marker="o", ms=3, lw=1, label=c)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("RPM step"); ax.set_ylabel(r"Spearman $\rho$(feature, temperature) within step")
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(FIGURES_DIR / "within_step_rho.png", dpi=150)
plt.close(fig)

# =============================================================================
# Test B (US): within-RPM-step temperature sensitivity, passive ultrasound
# Same conditioning as the AE case, run over the retained UL features so the
# figure is directly comparable to within_step_rho.png. Uses every available
# sweep per step (no subsampling); the --per-step flag affects only Test A.
# =============================================================================

sel = json.loads((OUTPUT_DIR / "feature_selection.json").read_text().rstrip("\x00"))
US_FEATURES = sel["UL"]["retained"]

feat_us = pd.read_parquet(fp)
meta_us = pd.read_parquet(mp)
mask_us = (feat_us["sensor"] == "UL").values
feat_us = feat_us[mask_us].reset_index(drop=True)
meta_us = meta_us[mask_us].reset_index(drop=True)
US_FEATURES = [c for c in US_FEATURES if c in feat_us.columns]

df_us = feat_us[US_FEATURES].copy()
df_us["rpm"], df_us["temp"] = meta_us["rpm"].values, meta_us["temperature_c"].values
df_us = df_us[(df_us["rpm"] >= 60) & (df_us["rpm"] <= 3000)]
df_us["step"] = (df_us["rpm"] / 100).round() * 100

rows_us = []
for step, sub in df_us.groupby("step"):
    if sub["temp"].nunique() < 8 or len(sub) < 30:
        continue
    for c in US_FEATURES:
        rho = spearmanr(sub[c], sub["temp"])[0]
        rows_us.append({"step": step, "n": len(sub), "feature": c, "rho_temp": rho})
ws_us = pd.DataFrame(rows_us)
ws_us.to_csv(TABLES_DIR / "within_step_rho_us.csv", index=False)

print("\nTest B (US) -- within-RPM-step Spearman(feature, temperature), median |rho| across steps:")
statsB_us = {}
med_us = ws_us.groupby("feature")["rho_temp"].agg(median_rho="median",
                                                  median_abs=lambda s: s.abs().median(),
                                                  frac_negative=lambda s: (s < 0).mean())
for c, r in med_us.sort_values("median_abs", ascending=False).iterrows():
    statsB_us[c] = {k: round(float(v), 3) for k, v in r.items()}
    print(f"  {c:34s} median rho={r['median_rho']:+.2f}  median |rho|={r['median_abs']:.2f}  "
          f"frac neg={r['frac_negative']:.0%}")
n_us = int(ws_us.groupby("step")["n"].first().sum())
print(f"  US within-step total sweeps: {n_us} over {ws_us['step'].nunique()} steps")

fig, ax = plt.subplots(figsize=(9, 5))
for c in US_FEATURES:
    sub = ws_us[ws_us["feature"] == c]
    ax.plot(sub["step"], sub["rho_temp"], marker="o", ms=3, lw=1, label=c)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("RPM step"); ax.set_ylabel(r"Spearman $\rho$(feature, temperature) within step")
ax.legend(fontsize=6, ncol=2); fig.tight_layout()
fig.savefig(FIGURES_DIR / "within_step_rho_us.png", dpi=150)
plt.close(fig)

(SCRIPT_DIR / "band_mechanism_stats.json").write_text(json.dumps(
    {"test_A_comb_strip": statsA, "test_B_within_step": statsB,
     "test_B_within_step_us": statsB_us,
     "temp_window": [t_lo, t_hi], "data_file": DATA_FILE.name}, indent=1))
print(f"\nWrote results to {SCRIPT_DIR}")
