"""
00_eda_timeseries.py
====================
Exploratory data analysis of raw AE and Ultrasound time-series signals.

Sections
--------
EDA1 — Operating conditions space (RPM, temperature, κ distribution)
EDA2 — Representative raw waveforms per sensor, ordered by κ
EDA3 — Time-domain statistics per sweep vs κ  (RMS, kurtosis, crest factor)
EDA4 — Amplitude distributions per sensor, coloured by κ
EDA5 — Average Welch PSD with IQR band per sensor
EDA6 — PSD bundle coloured by κ
EDA7 — Spectrograms for lowest and highest κ sweeps
EDA8 — Sub-band energy fractions vs κ
EDA9  — Average PSD per κ regime per sensor

All figures are saved to the configured output directory.

Usage
-----
    python exploration/00_eda_timeseries.py
    python exploration/00_eda_timeseries.py --config alt.yaml
    python exploration/00_eda_timeseries.py --max-files 5   # quick test

Feature caching
---------------
Time-domain and spectral features are cached to parquet for fast reruns.
Load priority:
  1. features.parquet + metadata.parquet  (output of 01_feature_generation.py)
  2. eda_features.parquet + eda_metadata.parquet  (written by this script)
  3. Compute from HDF5 → save as eda_features / eda_metadata parquet
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from scipy.signal import hilbert as _hilbert, welch

sys.path.insert(0, str(Path(__file__).parent.parent))
from own_utils.calculate_kappa import calculate_kappa
from own_utils.config import get_input_dir, get_output_dir, load_config
from own_utils.features import extract_features, bandpass_filter as _bandpass_filter
from own_utils.loading import discover_hdf5_files


# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of HDF5 files processed (for quick testing)",
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

INPUT_DIR = get_input_dir(cfg)
OUTPUT_DIR = get_output_dir(cfg)
D_PW_MM: float = cfg["bearing"]["d_pw_mm"]
RPM_MIN: float = cfg["filters"].get("rpm_min", 0.0)
RPM_MAX: float = cfg["filters"]["rpm_max"]
SENSORS: tuple[str, ...] = ("AE", "UL")

welch_cfg = cfg.get("welch", {})
WELCH_NPERSEG: int = welch_cfg.get("nperseg", 4096)
WELCH_NOVERLAP: int = welch_cfg.get("noverlap", 2048)
WELCH_WINDOW: str = welch_cfg.get("window", "hann")

kappa_cfg = cfg.get("kappa", {})
KAPPA_BOUNDS: list[float] = kappa_cfg.get("boundaries", [0.5, 1.0])

# Waveform/envelope snippet lengths — kept short so full 250k-sample arrays
# are never retained in memory beyond a single sweep's processing.
WAVEFORM_MS: float = 20.0   # milliseconds stored per sweep for EDA2
_ENV_SHOW_MS: float = 5.0   # milliseconds stored per sweep for EDA10 Panel A

# Sub-bands for EDA8 (fixed, independent of the pipeline's band_width_hz)
ANA_BANDS: list[tuple[float, float, str]] = [
    (0,        200_000, "0–200 kHz"),
    (200_000,  400_000, "200–400 kHz"),
    (400_000,  600_000, "400–600 kHz"),
    (600_000,  800_000, "600–800 kHz"),
]

# BP filter used for EDA3b — must match the label used when saving parquet so
# that pre-computed band features can be loaded on subsequent runs.
_BP_LOW_HZ  = 10e3   # 10 kHz
_BP_HIGH_HZ = 200e3  # 200 kHz
_BP_BAND_LABEL = f"{int(_BP_LOW_HZ / 1e3)}-{int(_BP_HIGH_HZ / 1e3)}kHz"  # "10-200kHz"

# %%
# =============================================================================
# Helper functions
# =============================================================================

_VISCOSITY_FALLBACK: dict[str, float] = {
    "viscosity_40c_cst": 22.0,
    "viscosity_100c_cst": 4.1,
}


def _normalize_sweep_params(params: dict) -> dict:
    out = dict(params)
    if "rpm" not in out and "telem_rpm_meas" in out:
        out["rpm"] = out["telem_rpm_meas"]
    if "temperature_c" not in out and "telem_omron_pv_c" in out:
        out["temperature_c"] = out["telem_omron_pv_c"]
    if "load_g" not in out and "telem_mass_g" in out:
        out["load_g"] = out["telem_mass_g"]
    return out


def _psd(v: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(
        v, fs=fs,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        window=WELCH_WINDOW,
    )
    return f, p


# %%
# =============================================================================
# Parquet helpers — fast feature cache
# =============================================================================


def _try_load_parquet() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (feat_df, meta_df) from the first available parquet pair, or None."""
    candidates = [
        (OUTPUT_DIR / "features.parquet",     OUTPUT_DIR / "metadata.parquet"),
        (OUTPUT_DIR / "eda_features.parquet", OUTPUT_DIR / "eda_metadata.parquet"),
    ]
    for feat_path, meta_path in candidates:
        if feat_path.exists() and meta_path.exists():
            print(f"Found cached features: {feat_path.name} + {meta_path.name}")
            return pd.read_parquet(feat_path), pd.read_parquet(meta_path)
    return None


def _build_stats_from_parquet(
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convert parquet DataFrames into stats_df / hp_stats_df.

    Computes kappa per sweep from metadata columns, then merges with features.
    Returns hp_stats_df = None when BP-band prefixed columns are absent.
    """
    sweep_meta = (
        meta_df.drop_duplicates(["file", "sweep"])
        [["file", "sweep", "rpm", "temperature_c",
          "viscosity_40c_cst", "viscosity_100c_cst"]]
        .copy()
    )

    def _kappa(row: pd.Series) -> float:
        try:
            return calculate_kappa(
                rpm=float(row["rpm"]),
                temp_c=float(row["temperature_c"]),
                d_pw=D_PW_MM,
                nu_40=float(row.get("viscosity_40c_cst", 22.0)),
                nu_100=float(row.get("viscosity_100c_cst", 4.1)),
            )
        except Exception:
            return np.nan

    sweep_meta["kappa"] = sweep_meta.apply(_kappa, axis=1)

    # RPM filter (parquet may contain all sweeps)
    sweep_meta = sweep_meta[
        (sweep_meta["rpm"] >= RPM_MIN) & (sweep_meta["rpm"] <= RPM_MAX)
    ]

    stats_df = feat_df.merge(
        sweep_meta[["file", "sweep", "rpm", "temperature_c", "kappa"]],
        on=["file", "sweep"],
        how="inner",
    )

    # BP-filtered stats from band-prefixed columns (written by this script or
    # by 01_feature_generation.py when the matching band is configured)
    bp_prefix = f"{_BP_BAND_LABEL}__"
    bp_cols = [c for c in feat_df.columns if c.startswith(bp_prefix)]
    if bp_cols:
        hp_df = feat_df[["file", "sweep", "sensor"] + bp_cols].rename(
            columns={c: c[len(bp_prefix):] for c in bp_cols}
        )
        hp_stats_df = hp_df.merge(
            sweep_meta[["file", "sweep", "rpm", "temperature_c", "kappa"]],
            on=["file", "sweep"],
            how="inner",
        )
    else:
        hp_stats_df = None
        print(
            f"  Note: no '{bp_prefix}*' columns found — skipping EDA3b. "
            f"Run 01_feature_generation.py with the {_BP_BAND_LABEL} band configured, "
            f"or delete eda_features.parquet to recompute."
        )

    return stats_df, hp_stats_df


# %%
# =============================================================================
# Load raw signals — stream sweep-by-sweep, compute everything in one pass
# =============================================================================


def _load_sweeps(file_paths: list[Path], *, skip_stats: bool = False) -> list[dict]:
    """Stream HDF5 sweeps one at a time, computing derived metrics inline.

    Each 250k-sample voltage array is loaded, processed, and discarded before
    the next sweep is read.  Only short waveform/envelope snippets and scalar
    statistics are retained, keeping peak RAM independent of file size.

    Parameters
    ----------
    skip_stats:
        When True, skip extract_features computation (stats / bp_stats keys
        will be empty dicts).  Use this when stats are already loaded from
        parquet and only visualization data is needed.

    Each returned record contains:
        waveform  — {sensor: array[:n_wave]}   short snippet for EDA2
        envelope  — {sensor: array[:n_env]}    short snippet for EDA10 Panel A
        stats     — {sensor: dict}             broadband features (empty when skip_stats)
        bp_stats  — {sensor: dict}             BP-filtered features (empty when skip_stats)
        psd       — {sensor: (f, p)}           Welch PSD
        env_mean  — {sensor: float}            mean Hilbert envelope amplitude
        env_psd   — {sensor: (f, p)}           Welch PSD of envelope
        env_sf    — {sensor: float}            spectral flatness of envelope PSD
    """
    records: list[dict] = []
    for fp in file_paths:
        try:
            with h5py.File(fp, "r") as f:
                sweeps_grp = f["sweeps"]
                first_key = list(sweeps_grp.keys())[0]
                time_axis = sweeps_grp[first_key][SENSORS[0]]["time"][()]
                fs: float = 1.0 / float(np.mean(np.diff(time_axis)))

                lm = dict(f["metadata"]["lubricant"].attrs)
                for k, v_fb in _VISCOSITY_FALLBACK.items():
                    lm.setdefault(k, v_fb)

                n_wave = int(WAVEFORM_MS * 1e-3 * fs)
                n_env  = int(_ENV_SHOW_MS  * 1e-3 * fs)

                for sweep_name, sweep in sweeps_grp.items():
                    tp  = _normalize_sweep_params(dict(sweep.attrs))
                    rpm = float(tp.get("rpm", np.nan))
                    if rpm < RPM_MIN or rpm > RPM_MAX:
                        continue

                    temp_c = float(tp.get("temperature_c", np.nan))
                    nu_40  = float(lm.get("viscosity_40c_cst", np.nan))
                    nu_100 = float(lm.get("viscosity_100c_cst", np.nan))
                    try:
                        kap = calculate_kappa(
                            rpm=rpm, temp_c=temp_c, d_pw=D_PW_MM,
                            nu_40=nu_40, nu_100=nu_100,
                        )
                    except Exception:
                        kap = np.nan

                    rec: dict = {
                        "file": fp.stem, "sweep": sweep_name,
                        "rpm": rpm, "temperature_c": temp_c,
                        "kappa": kap, "fs": fs,
                        "viscosity_40c_cst": nu_40,
                        "viscosity_100c_cst": nu_100,
                        "waveform": {}, "envelope": {},
                        "stats": {}, "bp_stats": {},
                        "psd": {},
                        "env_mean": {}, "env_psd": {}, "env_sf": {},
                    }

                    for sensor in SENSORS:
                        if sensor not in sweep:
                            continue

                        sig = sweep[sensor]["voltage"][()]  # 250k samples — discarded after this block

                        rec["waveform"][sensor] = sig[:n_wave].copy()
                        rec["psd"][sensor]      = _psd(sig, fs)

                        if not skip_stats:
                            rec["stats"][sensor]    = extract_features(sig, fs)
                            rec["bp_stats"][sensor] = extract_features(
                                _bandpass_filter(sig, fs, _BP_LOW_HZ, _BP_HIGH_HZ), fs
                            )

                        env = np.abs(_hilbert(sig))
                        rec["envelope"][sensor] = env[:n_env].copy()
                        rec["env_mean"][sensor] = float(np.mean(env))
                        _, p_env = welch(env, fs=fs, nperseg=min(256, len(env)))
                        rec["env_psd"][sensor]  = (_, p_env)
                        p_clip = np.clip(p_env, 1e-30, None)
                        rec["env_sf"][sensor]   = float(np.exp(np.mean(np.log(p_clip))) / np.mean(p_clip))

                        del sig, env  # release before next sensor / sweep

                    records.append(rec)

        except Exception as exc:
            print(f"  WARNING: {fp.name}: {exc}")
            continue

    return records


# %%
# =============================================================================
# Discover files and load / compute features
# =============================================================================

FILE_PATTERNS: list[str] | None = cfg.get("filters", {}).get("file_patterns") or None

hdf5_files = discover_hdf5_files(INPUT_DIR, file_patterns=FILE_PATTERNS)
if args.max_files:
    hdf5_files = hdf5_files[: args.max_files]

print(f"Found {len(hdf5_files)} HDF5 file(s) in {INPUT_DIR}")

# ── Try parquet cache ──────────────────────────────────────────────────────
_parquet = _try_load_parquet()

if _parquet is not None:
    print("Loading pre-computed stats from parquet (skip HDF5 feature extraction) …")
    stats_df, hp_stats_df = _build_stats_from_parquet(*_parquet)
    print(f"Loaded {len(stats_df)} feature rows; loading HDF5 for visualizations …")
    sweeps = _load_sweeps(hdf5_files, skip_stats=True)

else:
    print(f"No parquet cache found — computing features from {len(hdf5_files)} HDF5 file(s) …")
    sweeps = _load_sweeps(hdf5_files)
    print(f"Loaded {len(sweeps)} sweeps (after RPM filter {RPM_MIN} – {RPM_MAX})")

    # ── Build stats DataFrames ─────────────────────────────────────────────
    stat_rows: list[dict] = []
    bp_stat_rows: list[dict] = []
    for rec in sweeps:
        for sensor in SENSORS:
            if sensor not in rec["stats"]:
                continue
            base = {
                "file": rec["file"],
                "sweep": rec["sweep"],
                "sensor": sensor,
                "rpm": rec["rpm"],
                "temperature_c": rec["temperature_c"],
                "kappa": rec["kappa"],
            }
            stat_rows.append({**base, **rec["stats"][sensor]})
            bp_stat_rows.append({**base, **rec["bp_stats"][sensor]})

    stats_df = pd.DataFrame(stat_rows)
    hp_stats_df = pd.DataFrame(bp_stat_rows) if bp_stat_rows else None

    # ── Save feature cache to parquet ─────────────────────────────────────
    _feat_rows: list[dict] = []
    _meta_seen: set = set()
    _meta_rows: list[dict] = []
    for rec in sweeps:
        key = (rec["file"], rec["sweep"])
        if key not in _meta_seen:
            _meta_seen.add(key)
            _meta_rows.append({
                "file": rec["file"],
                "sweep": rec["sweep"],
                "rpm": rec["rpm"],
                "temperature_c": rec["temperature_c"],
                "viscosity_40c_cst": rec.get("viscosity_40c_cst", np.nan),
                "viscosity_100c_cst": rec.get("viscosity_100c_cst", np.nan),
            })
        for sensor in SENSORS:
            if sensor not in rec["stats"]:
                continue
            row: dict = {"file": rec["file"], "sweep": rec["sweep"], "sensor": sensor}
            row.update(rec["stats"][sensor])
            # Store BP features under band-prefixed keys so they survive cache reload
            if sensor in rec.get("bp_stats", {}):
                row.update({
                    f"{_BP_BAND_LABEL}__{k}": v
                    for k, v in rec["bp_stats"][sensor].items()
                })
            _feat_rows.append(row)

    pd.DataFrame(_feat_rows).to_parquet(
        OUTPUT_DIR / "eda_features.parquet", engine="pyarrow"
    )
    pd.DataFrame(_meta_rows).to_parquet(
        OUTPUT_DIR / "eda_metadata.parquet", engine="pyarrow"
    )
    print(
        f"Saved feature cache: eda_features.parquet ({len(_feat_rows)} rows), "
        f"eda_metadata.parquet ({len(_meta_rows)} rows)"
    )

print(f"Sweeps for visualization: {len(sweeps)}")

# TODO: THe below code is not modified to the new dataset and much higher memory load.
# %%
# =============================================================================
# Per-sweep time-domain statistics
# =============================================================================

# stats_df / hp_stats_df are already built above (from parquet or from sweeps).
# Nothing to do here — this cell intentionally left as a checkpoint.

# %%
# =============================================================================
# Figure EDA1 — Operating conditions overview
# =============================================================================

meta_df = stats_df.drop_duplicates(["file", "sweep"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# RPM vs temperature, coloured by kappa
valid_meta = meta_df.dropna(subset=["kappa"])
sc = axes[0].scatter(
    valid_meta["rpm"], valid_meta["temperature_c"],
    c=valid_meta["kappa"], cmap="viridis", s=20, alpha=0.7, edgecolors="none",
)
fig.colorbar(sc, ax=axes[0], label="κ")
axes[0].set_xlabel("RPM")
axes[0].set_ylabel("Temperature [°C]")
axes[0].set_title("Operating conditions space")
axes[0].grid(ls=":", alpha=0.4)

# Kappa histogram with regime boundaries
axes[1].hist(
    valid_meta["kappa"], bins=30,
    color="#1f77b4", edgecolor="white", linewidth=0.4,
)
for bound in KAPPA_BOUNDS:
    axes[1].axvline(bound, color="red", ls="--", lw=1, alpha=0.8, label=f"κ = {bound}")
axes[1].set_xlabel("κ (lubrication ratio)")
axes[1].set_ylabel("Count (sweeps)")
axes[1].set_title("κ distribution")
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_operating_conditions.png", dpi=600)
plt.show()
print("Saved: eda_operating_conditions.png")

# %%
# =============================================================================
# Figure EDA2 — Representative raw waveforms ordered by κ
# =============================================================================

N_EXAMPLES = 4

valid_sweeps = [
    r for r in sweeps
    if not np.isnan(r.get("kappa", np.nan)) and all(s in r["waveform"] for s in SENSORS)
]
kappa_sorted = sorted(valid_sweeps, key=lambda r: r["kappa"])
example_idx = np.linspace(0, len(kappa_sorted) - 1, N_EXAMPLES, dtype=int)
example_recs = [kappa_sorted[i] for i in example_idx]

fig, axes = plt.subplots(
    len(SENSORS), N_EXAMPLES,
    figsize=(5 * N_EXAMPLES, 3.5 * len(SENSORS)),
    sharex=False,
)

for row_i, sensor in enumerate(SENSORS):
    for col_i, rec in enumerate(example_recs):
        ax = axes[row_i, col_i]
        v = rec["waveform"][sensor]
        fs = rec["fs"]
        t_ms = np.arange(len(v)) / fs * 1e3
        ax.plot(t_ms, v, lw=0.4, color=f"C{row_i}", alpha=0.9)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Voltage [V]")
        ax.set_title(f"{sensor}\nκ = {rec['kappa']:.2f}   RPM = {rec['rpm']:.0f}")
        ax.grid(ls=":", alpha=0.3)

fig.suptitle(
    f"Representative waveforms — {N_EXAMPLES} sweeps ordered by κ "
    f"(first {WAVEFORM_MS:.0f} ms shown)",
    fontsize=11,
)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_waveforms.png", dpi=600)
plt.show()
print("Saved: eda_waveforms.png")

# %%
# =============================================================================
# Figure EDA3 — Time-domain statistics vs κ
# =============================================================================

stat_cols = [
    ("rms",               "RMS [V]"),
    ("kurtosis",          "Kurtosis"),
    ("crest_factor",      "Crest factor"),
    ("mobility",          "Mobility"),
    ("spectral_flatness", "Spectral flatness"),
]

fig, axes = plt.subplots(
    len(SENSORS), len(stat_cols),
    figsize=(5 * len(stat_cols), 4 * len(SENSORS)),
    squeeze=False,
)

for row_i, sensor in enumerate(SENSORS):
    sub = stats_df[(stats_df["sensor"] == sensor)].dropna(subset=["kappa"])
    for col_i, (col, label) in enumerate(stat_cols):
        ax = axes[row_i, col_i]
        sc = ax.scatter(
            sub["kappa"], sub[col],
            c=sub["rpm"], cmap="plasma",
            s=10, alpha=0.5, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="RPM")

        # Bin-mean trend
        bins = pd.cut(sub["kappa"], bins=12)
        trend = sub.groupby(bins, observed=True)[col].mean()
        centres = [iv.mid for iv in trend.index]
        ax.plot(centres, trend.values, "r-o", ms=4, lw=1.5, label="bin mean")

        ax.set_xlabel("κ")
        ax.set_ylabel(label)
        ax.set_title(f"{sensor} — {label} vs κ")
        ax.legend(fontsize=8)
        ax.grid(ls=":", alpha=0.4)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_time_domain_stats.png", dpi=600)
plt.show()
print("Saved: eda_time_domain_stats.png")

# %%
# =============================================================================
# Figure EDA3b — Time-domain statistics vs κ (BP-filtered)
# =============================================================================

if hp_stats_df is not None:
    fig, axes = plt.subplots(
        len(SENSORS), len(stat_cols),
        figsize=(5 * len(stat_cols), 4 * len(SENSORS)),
        squeeze=False,
    )

    for row_i, sensor in enumerate(SENSORS):
        sub = hp_stats_df[(hp_stats_df["sensor"] == sensor)].dropna(subset=["kappa"])
        for col_i, (col, label) in enumerate(stat_cols):
            ax = axes[row_i, col_i]
            sc = ax.scatter(
                sub["kappa"], sub[col],
                c=sub["rpm"], cmap="plasma",
                s=10, alpha=0.5, edgecolors="none",
            )
            fig.colorbar(sc, ax=ax, label="RPM")

            bins = pd.cut(sub["kappa"], bins=12)
            trend = sub.groupby(bins, observed=True)[col].mean()
            centres = [iv.mid for iv in trend.index]
            ax.plot(centres, trend.values, "r-o", ms=4, lw=1.5, label="bin mean")

            ax.set_xlabel("κ")
            ax.set_ylabel(label)
            ax.set_yscale("log")
            ax.set_title(f"{sensor} — {label} vs κ  [BP 10–200 kHz]")
            ax.legend(fontsize=8)
            ax.grid(ls=":", alpha=0.4)

    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_time_domain_stats_bp10k_500k.png", dpi=600)
    plt.show()
    print("Saved: eda_time_domain_stats_bp10k_500k.png")
else:
    print("Skipping EDA3b (no BP-band features available).")

# %%
# =============================================================================
# Assemble PSD rows (reused in EDA5, EDA6, EDA8, EDA9)
# =============================================================================

psd_rows: list[dict] = []
for rec in sweeps:
    for sensor in SENSORS:
        if sensor not in rec["psd"]:
            continue
        f_arr, p_arr = rec["psd"][sensor]
        psd_rows.append({
            "file": rec["file"],
            "sweep": rec["sweep"],
            "sensor": sensor,
            "kappa": rec["kappa"],
            "rpm": rec["rpm"],
            "f": f_arr,
            "p": p_arr,
        })

# %%
# =============================================================================
# Figure EDA9 — Average PSD per κ regime per sensor
# =============================================================================

# Build kappa interval edges and labels from KAPPA_BOUNDS
_edges = [-np.inf] + list(KAPPA_BOUNDS) + [np.inf]
_kappa_intervals = [
    (lo, hi, (f"κ < {hi}" if np.isinf(lo) else f"κ ≥ {lo}" if np.isinf(hi) else f"{lo} ≤ κ < {hi}"))
    for lo, hi in zip(_edges[:-1], _edges[1:])
]
_interval_colors = [f"C{i}" for i in range(len(_kappa_intervals))]

fig, axes = plt.subplots(1, len(SENSORS), figsize=(7 * len(SENSORS), 5), squeeze=False)

for col_i, sensor in enumerate(SENSORS):
    ax = axes[0, col_i]
    sensor_psds = [r for r in psd_rows if r["sensor"] == sensor and not np.isnan(r.get("kappa", np.nan))]
    if not sensor_psds:
        continue

    f_ref = sensor_psds[0]["f"]

    for color, (lo, hi, label) in zip(_interval_colors, _kappa_intervals):
        group = [r for r in sensor_psds if lo <= r["kappa"] < hi]
        if not group:
            continue
        p_stack = np.stack([r["p"] for r in group])
        p_mean = p_stack.mean(axis=0)
        ax.semilogy(f_ref / 1e3, p_mean, lw=1.5, color=color, alpha=0.5, label=f"{label}  (n={len(group)})")

    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("PSD  [V² / Hz]")
    ax.set_title(f"{sensor} — average PSD per κ regime")
    ax.legend(fontsize=8)
    ax.grid(ls=":", which="both", alpha=0.3)

fig.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_psd_kappa_regimes.png", dpi=600)
plt.show()
print("Saved: eda_psd_kappa_regimes.png")

# %%
# =============================================================================
# Figure EDA10 — Envelope analysis per κ regime per sensor
# =============================================================================
# Panel A (left)  : average Hilbert envelope waveform per κ regime (first 5 ms)
# Panel B (right) : mean envelope value (scalar) vs κ — scatter + bin mean
# Figure EDA11    : PSD of the Hilbert envelope per κ regime per sensor
# =============================================================================

fig10, axes10 = plt.subplots(
    len(SENSORS), 2,
    figsize=(14, 5 * len(SENSORS)),
    squeeze=False,
)

for row_i, sensor in enumerate(SENSORS):
    ax_wave = axes10[row_i, 0]
    ax_scat = axes10[row_i, 1]

    env_mean_rows = []  # [(kappa, mean_env), ...]

    for color, (lo, hi, label) in zip(_interval_colors, _kappa_intervals):
        group = [r for r in sweeps if lo <= r.get("kappa", np.nan) < hi and sensor in r["envelope"]]
        if not group:
            continue

        fs_ex = group[0]["fs"]
        n_show = min(len(r["envelope"][sensor]) for r in group)
        t_ms = np.arange(n_show) / fs_ex * 1e3

        envs = []
        for r in group:
            env_mean_rows.append({"kappa": r["kappa"], "mean_env": r["env_mean"][sensor]})
            env_snip = r["envelope"][sensor]
            if len(env_snip) >= n_show:
                envs.append(env_snip[:n_show])

        if envs:
            ax_wave.plot(t_ms, np.mean(envs, axis=0), lw=1.2, color=color, label=label)

    ax_wave.set_xlabel("Time [ms]")
    ax_wave.set_ylabel("Envelope [V]")
    ax_wave.set_title(f"{sensor} — mean Hilbert envelope per κ regime")
    ax_wave.legend(fontsize=8)
    ax_wave.grid(ls=":", alpha=0.3)

    # Panel B: mean envelope vs κ
    env_df = pd.DataFrame(env_mean_rows).dropna()
    if not env_df.empty:
        ax_scat.scatter(
            env_df["kappa"], env_df["mean_env"],
            s=8, alpha=0.5, edgecolors="none", color="steelblue",
        )
        bins = pd.cut(env_df["kappa"], bins=12)
        trend = env_df.groupby(bins, observed=True)["mean_env"].mean()
        centres = [iv.mid for iv in trend.index]
        ax_scat.plot(centres, trend.values, "r-o", ms=4, lw=1.5, label="bin mean")

    ax_scat.set_xlabel("κ")
    ax_scat.set_ylabel("Mean envelope [V]")
    ax_scat.set_title(f"{sensor} — mean Hilbert envelope vs κ")
    ax_scat.legend(fontsize=8)
    ax_scat.grid(ls=":", alpha=0.3)

fig10.suptitle("EDA10 — Envelope analysis per κ regime", fontsize=11)
fig10.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_envelope.png", dpi=600)
plt.show()
print("Saved: eda_envelope.png")

# %%
# =============================================================================
# Figure EDA11 — PSD of the Hilbert envelope per κ regime per sensor
# =============================================================================

fig11, axes11 = plt.subplots(
    len(SENSORS), 1,
    figsize=(10, 5 * len(SENSORS)),
    squeeze=False,
)

for row_i, sensor in enumerate(SENSORS):
    ax = axes11[row_i, 0]

    for color, (lo, hi, label) in zip(_interval_colors, _kappa_intervals):
        group = [r for r in sweeps if lo <= r.get("kappa", np.nan) < hi and sensor in r["env_psd"]]
        if not group:
            continue

        psds = []
        f_env = None
        for r in group:
            f_env, p_env = r["env_psd"][sensor]
            psds.append(p_env)

        min_len = min(len(p) for p in psds)
        psd_mean = np.mean([p[:min_len] for p in psds], axis=0)
        ax.semilogy(f_env[:min_len], psd_mean, lw=1.2, color=color, alpha=0.7, label=label)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [V²/Hz]")
    ax.set_title(f"{sensor} — envelope PSD per κ regime")
    ax.legend(fontsize=8)
    ax.grid(ls=":", which="both", alpha=0.3)

fig11.suptitle("EDA11 — Hilbert envelope PSD per κ regime", fontsize=11)
fig11.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_envelope_psd.png", dpi=600)
plt.show()
print("Saved: eda_envelope_psd.png")

# %%
# =============================================================================
# Figure EDA12 — Spectral flatness of envelope PSD vs κ per sensor
# =============================================================================
# Spectral flatness = geometric mean(PSD) / arithmetic mean(PSD).
# 1 → white noise, 0 → pure tone.

fig12, axes12 = plt.subplots(
    1, len(SENSORS),
    figsize=(7 * len(SENSORS), 5),
    squeeze=False,
)

for col_i, sensor in enumerate(SENSORS):
    ax = axes12[0, col_i]
    sf_rows = []

    for r in sweeps:
        kappa = r.get("kappa", np.nan)
        if np.isnan(kappa) or sensor not in r["env_sf"]:
            continue
        sf_rows.append({"kappa": kappa, "sf": r["env_sf"][sensor], "rpm": r["rpm"]})

    sf_df = pd.DataFrame(sf_rows).dropna()
    if sf_df.empty:
        continue

    sc = ax.scatter(
        sf_df["kappa"], sf_df["sf"],
        c=sf_df["rpm"], cmap="plasma",
        s=8, alpha=0.6, edgecolors="none",
    )
    fig12.colorbar(sc, ax=ax, label="RPM")
    bins = pd.cut(sf_df["kappa"], bins=12)
    trend = sf_df.groupby(bins, observed=True)["sf"].mean()
    centres = [iv.mid for iv in trend.index]
    ax.plot(centres, trend.values, "r-o", ms=4, lw=1.5, label="bin mean")

    ax.set_yscale("log")
    ax.set_xlabel("κ")
    ax.set_ylabel("Spectral flatness")
    ax.set_title(f"{sensor} — envelope spectral flatness vs κ")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.3)

fig12.suptitle("EDA12 — Envelope spectral flatness vs κ", fontsize=11)
fig12.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_envelope_spectral_flatness.png", dpi=600)
plt.show()
print("Saved: eda_envelope_spectral_flatness.png")

# %%
# =============================================================================
# Console summary
# =============================================================================

print("\n" + "=" * 70)
print("EDA TIME-SERIES SUMMARY")
print("=" * 70)
print(f"\nFiles processed : {len(hdf5_files)}")
print(f"Sweeps loaded   : {len(sweeps)}")

if not stats_df.empty:
    kappa_valid = stats_df.drop_duplicates(["file", "sweep"])["kappa"].dropna()
    print(f"κ range         : [{kappa_valid.min():.3f}, {kappa_valid.max():.3f}]  "
          f"(median = {kappa_valid.median():.3f})")
    print(f"RPM range       : [{meta_df['rpm'].min():.0f}, {meta_df['rpm'].max():.0f}]")
    print(f"Temp range      : [{meta_df['temperature_c'].min():.1f}, "
          f"{meta_df['temperature_c'].max():.1f}] °C")

    print("\nPer-sensor time-domain statistics:")
    for sensor in SENSORS:
        sub = stats_df[stats_df["sensor"] == sensor]
        print(f"\n  {sensor}:")
        for col, label in [("rms", "RMS [V]"), ("kurtosis", "Kurtosis"),
                            ("crest_factor", "Crest factor")]:
            print(f"    {label:20s}  mean={sub[col].mean():.4f}  "
                  f"std={sub[col].std():.4f}  "
                  f"[{sub[col].min():.4f}, {sub[col].max():.4f}]")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# %%
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("\n00_eda_timeseries complete.")
