"""
08_band_validation.py
=====================
Validate that the AE frequency sub-bands carry bearing-generated signal
rather than instrument noise or VFD EMI.

Pipeline position: standalone diagnostic -- reads the raw HDF5 file(s)
directly (including the stationary sweeps that the feature pipeline's
rpm_min filter discards). Does not modify any pipeline output.

Rationale
---------
The recording contains stationary sweeps (motor off; telemetry reads its
~14.5 rpm idle floor). These are a noise-floor control: any content present
without rotation cannot be bearing-generated. Three checks:

1. PSD comparison: averaged Welch PSD of stationary vs running sweeps per
   temperature bin; per-band SNR = 10*log10(P_running / P_stationary).
   NOTE: the stationary segment occurs during heater-driven warm-up
   (37-50 C), so its spectrum may contain temperature-controller EMI; treat
   the SNR as indicative, not as a pure noise floor.
2. Feature-at-standstill: the headline feature (1-2 MHz complexity, and the
   per-band feature subset) computed on stationary sweeps. If it varies
   with temperature/time while the shaft is stationary, it tracks
   electronics, not contact events.
3. Line structure: fraction of band power concentrated in the top 1% of
   PSD bins. EMI appears as narrow lines; stress-wave content is broadband.

Outputs (outputs/08_band_validation/)
-------------------------------------
figures/psd_standstill_vs_running_<bin>.png
figures/standstill_feature_vs_temp.png
tables/band_snr.csv
tables/standstill_features.csv
band_validation_stats.json

Usage
-----
    python scripts/08_band_validation.py --standstill-rpm 60 --max-per-group 10
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

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util as _ilu


def _load_module(name, rel):
    spec = _ilu.spec_from_file_location(name, ROOT / rel)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_features = _load_module("cs_features", "src/ceramicspeed/features.py")
bandpass_filter, extract_features = _features.bandpass_filter, _features.extract_features
_config = _load_module("cs_config", "src/ceramicspeed/config.py")

try:
    cfg = _config.load_config()
except Exception:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8").rstrip("\x00"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-group", type=int, default=5,
                   help="sweeps per (temp-bin, rpm-group) cell")
    p.add_argument("--rpm-targets", type=float, nargs="+", default=[300.0, 1500.0, 2900.0])
    p.add_argument("--standstill-rpm", type=float, default=60.0,
                   help="sweeps below this RPM count as stationary")
    p.add_argument("--n-temp-bins", type=int, default=4)
    p.add_argument("--nperseg", type=int, default=1 << 16)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--data-file", type=str, default=None,
                   help="explicit HDF5 path; default: config input_dir + file_patterns")
    args, _ = p.parse_known_args()
    return args


args = parse_args()

if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    try:
        OUTPUT_DIR = _config.get_output_dir(cfg)
    except Exception:
        OUTPUT_DIR = ROOT / "outputs"
    if not OUTPUT_DIR.exists() and (ROOT / "outputs").exists():
        OUTPUT_DIR = ROOT / "outputs"

SCRIPT_DIR = OUTPUT_DIR / "08_band_validation"
FIGURES_DIR = SCRIPT_DIR / "figures"
TABLES_DIR = SCRIPT_DIR / "tables"
for d in (SCRIPT_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

AE_BANDS = [(b["f_lo"], b["f_hi"], b["label"]) for b in cfg["frequency_bands"]["AE"]]
# Bands excluded from modelling but still characterised here:
AE_BANDS += [(b["f_lo"], b["f_hi"], b["label"])
             for b in cfg.get("validation_extra_bands", {}).get("AE", [])]

if args.data_file:
    data_files = [Path(args.data_file)]
else:
    input_dir = Path(cfg["paths"]["input_dir"])
    if not input_dir.is_dir():
        input_dir = ROOT / "data"
    pats = cfg["filters"].get("file_patterns") or ["*"]
    data_files = sorted(p for p in input_dir.iterdir()
                        if p.suffix in (".hdf5", ".h5") and any(s in p.stem for s in pats))
assert data_files, "no HDF5 data file found"
DATA_FILE = data_files[0]
print(f"Data file: {DATA_FILE}")

# =============================================================================
# Pass 1: scan sweep attributes only (cheap)
# =============================================================================


def _rpm_temp(attrs):
    rpm = attrs.get("rpm", attrs.get("telem_rpm_meas", np.nan))
    temp = attrs.get("temperature_c", attrs.get("telem_omron_pv_c", np.nan))
    return float(rpm), float(temp)


with h5py.File(DATA_FILE, "r") as f:
    grp = f["sweeps"]
    names = sorted(grp.keys(), key=lambda n: int(n.split("_")[1]))
    rows = []
    for n in names:
        rpm, temp = _rpm_temp(dict(grp[n].attrs))
        rows.append({"sweep": n, "rpm": rpm, "temp": temp})
idx = pd.DataFrame(rows).dropna()
print(f"Scanned {len(idx)} sweeps; "
      f"stationary (<{args.standstill_rpm} rpm): {(idx['rpm'] < args.standstill_rpm).sum()}")

still = idx[idx["rpm"] < args.standstill_rpm]
assert len(still) > 0, "no stationary sweeps found -- check --standstill-rpm"
t_edges = np.linspace(idx["temp"].min() - 0.1, idx["temp"].max() + 0.1, args.n_temp_bins + 1)
idx["temp_bin"] = pd.cut(idx["temp"], t_edges)

selected = []
for tb, sub in idx.groupby("temp_bin", observed=True):
    s = sub[sub["rpm"] < args.standstill_rpm]
    for _, r in s.head(args.max_per_group).iterrows():
        selected.append((r["sweep"], "standstill", r["rpm"], r["temp"], str(tb)))
    for tgt in args.rpm_targets:
        run = sub[sub["rpm"] >= args.standstill_rpm].copy()
        if run.empty:
            continue
        run["d"] = (run["rpm"] - tgt).abs()
        for _, r in run.nsmallest(args.max_per_group, "d").iterrows():
            selected.append((r["sweep"], f"rpm~{tgt:.0f}", r["rpm"], r["temp"], str(tb)))
sel_df = pd.DataFrame(selected, columns=["sweep", "group", "rpm", "temp", "temp_bin"])
print(f"Selected {len(sel_df)} sweeps across {sel_df['temp_bin'].nunique()} temp bins")

# =============================================================================
# Pass 2: load selected AE signals, PSDs + features
# =============================================================================

psds = {}
feat_rows = []
with h5py.File(DATA_FILE, "r") as f:
    grp = f["sweeps"]
    first = grp[names[0]]["AE"]["time"][()]
    fs = 1.0 / float(np.mean(np.diff(first)))
    print(f"fs = {fs/1e6:.3f} MHz")
    for _, r in sel_df.iterrows():
        x = grp[r["sweep"]]["AE"]["voltage"][()].astype(float)
        fxx, pxx = welch(x, fs=fs, nperseg=min(args.nperseg, len(x)))
        psds[r["sweep"]] = (fxx, pxx)
        row = dict(r)
        for f_lo, f_hi, label in AE_BANDS:
            xb = bandpass_filter(x, fs, f_lo, f_hi)
            fe = extract_features(xb, fs)
            for k in ("rms", "complexity", "mobility", "spectral_flatness"):
                row[f"{label}__{k}"] = fe[k]
        feat_rows.append(row)
feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv(TABLES_DIR / "standstill_features.csv", index=False)

# Persist averaged PSD curves so 06_plots.py can re-render the figures
# without touching the raw HDF5.
_psd_store = {}
for (_tb, _g), _gsub in sel_df.groupby(["temp_bin", "group"]):
    _arr = np.mean([psds[s_][1] for s_ in _gsub["sweep"]], axis=0)
    _psd_store[f"psd__{_tb}__{_g}__n{len(_gsub)}"] = _arr
_psd_store["freq"] = psds[sel_df["sweep"].iloc[0]][0]
np.savez_compressed(TABLES_DIR / "psd_curves.npz", **_psd_store)
print(f"Saved PSD curves: {TABLES_DIR / 'psd_curves.npz'}")

# =============================================================================
# Band SNR and line-structure metrics
# =============================================================================


def band_power(fxx, pxx, f_lo, f_hi):
    m = (fxx >= f_lo) & (fxx < f_hi)
    return float(_trapz(pxx[m], fxx[m])) if m.any() else np.nan


def line_fraction(fxx, pxx, f_lo, f_hi, top=0.01):
    m = (fxx >= f_lo) & (fxx < f_hi)
    p = np.sort(pxx[m])[::-1]
    k = max(1, int(len(p) * top))
    return float(p[:k].sum() / p.sum()) if p.sum() > 0 else np.nan


snr_rows = []
for tb, sub in sel_df.groupby("temp_bin"):
    still_names = sub[sub["group"] == "standstill"]["sweep"]
    if still_names.empty:
        continue
    for f_lo, f_hi, label in AE_BANDS:
        p_still = np.mean([band_power(*psds[s], f_lo, f_hi) for s in still_names])
        lf_still = np.mean([line_fraction(*psds[s], f_lo, f_hi) for s in still_names])
        for g, gsub in sub[sub["group"] != "standstill"].groupby("group"):
            p_run = np.mean([band_power(*psds[s], f_lo, f_hi) for s in gsub["sweep"]])
            lf_run = np.mean([line_fraction(*psds[s], f_lo, f_hi) for s in gsub["sweep"]])
            snr_rows.append({
                "temp_bin": tb, "rpm_group": g, "band": label,
                "snr_db": 10 * np.log10(p_run / p_still) if p_still > 0 else np.nan,
                "line_fraction_run": lf_run, "line_fraction_still": lf_still,
            })
snr_df = pd.DataFrame(snr_rows)
snr_df.to_csv(TABLES_DIR / "band_snr.csv", index=False)
print("\nMedian SNR (dB, running over stationary) per band:")
print(snr_df.groupby("band")["snr_db"].median().round(1).to_string())

# =============================================================================
# Figures
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for tb, sub in sel_df.groupby("temp_bin"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for g, gsub in sub.groupby("group"):
        arr = np.mean([psds[s][1] for s in gsub["sweep"]], axis=0)
        fxx = psds[gsub["sweep"].iloc[0]][0]
        ax.loglog(fxx, arr, lw=1, label=f"{g} (n={len(gsub)})",
                  alpha=0.9 if g == "standstill" else 0.7,
                  color="k" if g == "standstill" else None)
    for f_lo, f_hi, label in AE_BANDS:
        ax.axvspan(max(f_lo, 1), f_hi, alpha=0.06)
    ax.set_xlim(1e3, fs / 2)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("PSD [V$^2$/Hz]")
    ax.set_title(f"AE PSD, temperature bin {tb}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    safe = str(tb).replace(" ", "").replace(",", "_").replace("(", "").replace("]", "")
    fig.savefig(FIGURES_DIR / f"psd_standstill_vs_running_{safe}.png", dpi=150)
    plt.close(fig)

hl = "AE_1000-2000kHz__complexity"
fig, ax = plt.subplots(figsize=(8, 5))
for g, gsub in feat_df.groupby(feat_df["group"] == "standstill"):
    lbl = "stationary" if g else "running"
    ax.scatter(gsub["temp"], gsub[hl], s=18, label=lbl, alpha=0.8)
ax.set_xlabel("temperature [°C]"); ax.set_ylabel(hl)
ax.set_title("Headline feature at standstill vs running")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "standstill_feature_vs_temp.png", dpi=150)
plt.close(fig)

# =============================================================================
# Stats JSON
# =============================================================================

still_mask = feat_df["group"] == "standstill"
stats = {"data_file": DATA_FILE.name, "n_selected": len(sel_df), "fs_mhz": fs / 1e6,
         "bands": {}}
for f_lo, f_hi, label in AE_BANDS:
    band_snr = snr_df[snr_df["band"] == label]["snr_db"]
    entry = {"snr_db_median": round(float(band_snr.median()), 2),
             "snr_db_min": round(float(band_snr.min()), 2)}
    col = f"{label}__complexity"
    if col in feat_df.columns:
        run_vals, still_vals = feat_df.loc[~still_mask, col], feat_df.loc[still_mask, col]
        entry["complexity_running_range"] = round(float(run_vals.max() - run_vals.min()), 4)
        entry["complexity_standstill_range"] = round(float(still_vals.max() - still_vals.min()), 4)
        entry["line_fraction_run_median"] = round(float(
            snr_df[snr_df["band"] == label]["line_fraction_run"].median()), 4)
        entry["line_fraction_still_median"] = round(float(
            snr_df[snr_df["band"] == label]["line_fraction_still"].median()), 4)
    stats["bands"][label] = entry
(SCRIPT_DIR / "band_validation_stats.json").write_text(json.dumps(stats, indent=1))
print(f"\nWrote figures, tables, and stats to {SCRIPT_DIR}")
