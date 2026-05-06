"""
eda.py
======
EDA utilities: HDF5 streaming loader, feature-cache helpers, and plot functions.

Imported by dev/exploration/eda.py (the thin orchestrator).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert as _hilbert
from scipy.signal import welch as _scipy_welch

from .calculate_kappa import calculate_kappa
from .features import bandpass_filter as _bandpass_filter
from .features import extract_features

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_VISCOSITY_FALLBACK: dict[str, float] = {
    "viscosity_40c_cst": 22.0,
    "viscosity_100c_cst": 4.1,
}

_STAT_COLS: list[tuple[str, str]] = [
    ("rms",               "RMS [V]"),
    ("kurtosis",          "Kurtosis"),
    ("crest_factor",      "Crest factor"),
    ("mobility",          "Mobility"),
    ("spectral_flatness", "Spectral flatness"),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_sweep_params(params: dict) -> dict:
    out = dict(params)
    if "rpm" not in out and "telem_rpm_meas" in out:
        out["rpm"] = out["telem_rpm_meas"]
    if "temperature_c" not in out and "telem_omron_pv_c" in out:
        out["temperature_c"] = out["telem_omron_pv_c"]
    if "load_g" not in out and "telem_mass_g" in out:
        out["load_g"] = out["telem_mass_g"]
    return out


def _envelope_spectral_flatness(p: np.ndarray) -> float:
    p_clip = np.clip(p, 1e-30, None)
    return float(np.exp(np.mean(np.log(p_clip))) / np.mean(p_clip))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweeps(
    file_paths: list[Path],
    cfg: dict,
    *,
    sensors: tuple[str, ...],
    waveform_ms: float,
    env_show_ms: float,
    bp_low_hz: float | None = None,
    bp_high_hz: float | None = None,
    skip_stats: bool = False,
    max_sweeps: int | None = None,
    sweep_names: list[str] | None = None,
) -> list[dict]:
    """Stream HDF5 files sweep-by-sweep, computing derived metrics inline.

    Only short waveform/envelope snippets and scalars are retained per sweep,
    keeping peak RAM independent of file count.
    """
    rpm_min: float = cfg["filters"].get("rpm_min", 0.0)
    rpm_max: float = cfg["filters"]["rpm_max"]
    d_pw_mm: float = cfg["bearing"]["d_pw_mm"]
    welch_cfg = cfg.get("welch", {})
    nperseg: int = welch_cfg.get("nperseg", 4096)
    noverlap: int = welch_cfg.get("noverlap", 2048)
    window: str = welch_cfg.get("window", "hann")

    records: list[dict] = []
    for fp in file_paths:
        try:
            with h5py.File(fp, "r") as f:
                sweeps_grp = f["sweeps"]
                first_key = list(sweeps_grp.keys())[0]
                time_axis = sweeps_grp[first_key][sensors[0]]["time"][()]
                fs: float = 1.0 / float(np.mean(np.diff(time_axis)))

                lm = dict(f["metadata"]["lubricant"].attrs)
                for k, v_fb in _VISCOSITY_FALLBACK.items():
                    lm.setdefault(k, v_fb)

                n_wave = int(waveform_ms * 1e-3 * fs)
                n_env = int(env_show_ms * 1e-3 * fs)

                for sweep_name, sweep in sweeps_grp.items():
                    if sweep_names is not None and sweep_name not in sweep_names:
                        continue
                    tp = _normalize_sweep_params(dict(sweep.attrs))
                    rpm = float(tp.get("rpm", np.nan))
                    if rpm < rpm_min or rpm > rpm_max:
                        continue

                    temp_c = float(tp.get("temperature_c", np.nan))
                    nu_40 = float(lm.get("viscosity_40c_cst", np.nan))
                    nu_100 = float(lm.get("viscosity_100c_cst", np.nan))
                    try:
                        kap = calculate_kappa(rpm=rpm, temp_c=temp_c, d_pw=d_pw_mm,
                                              nu_40=nu_40, nu_100=nu_100)
                    except Exception:
                        kap = np.nan

                    rec: dict = {
                        "file": fp.stem, "sweep": sweep_name,
                        "rpm": rpm, "temperature_c": temp_c,
                        "kappa": kap, "fs": fs,
                        "viscosity_40c_cst": nu_40, "viscosity_100c_cst": nu_100,
                        "waveform": {}, "envelope": {},
                        "stats": {}, "bp_stats": {},
                        "psd": {},
                        "env_mean": {}, "env_psd": {}, "env_sf": {},
                    }

                    for sensor in sensors:
                        if sensor not in sweep:
                            continue
                        sig = sweep[sensor]["voltage"][()]

                        rec["waveform"][sensor] = sig[:n_wave].copy()
                        rec["psd"][sensor] = _scipy_welch(
                            sig, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
                        )

                        if not skip_stats:
                            rec["stats"][sensor] = extract_features(sig, fs)
                            if bp_low_hz is not None and bp_high_hz is not None:
                                rec["bp_stats"][sensor] = extract_features(
                                    _bandpass_filter(sig, fs, bp_low_hz, bp_high_hz), fs
                                )

                        env = np.abs(_hilbert(sig))
                        rec["envelope"][sensor] = env[:n_env].copy()
                        rec["env_mean"][sensor] = float(np.mean(env))
                        f_env, p_env = _scipy_welch(env, fs=fs, nperseg=min(256, len(env)))
                        rec["env_psd"][sensor] = (f_env, p_env)
                        rec["env_sf"][sensor] = _envelope_spectral_flatness(p_env)

                        del sig, env

                    records.append(rec)
                    if max_sweeps and len(records) >= max_sweeps:
                        return records

        except MemoryError:
            raise
        except Exception as exc:
            print(f"  WARNING: {fp.name}: {exc}")

    return records


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------

def try_load_parquet(
    output_dir: Path,
    eda_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (feat_df, meta_df) from the first available parquet pair, or None."""
    candidates = [
        (output_dir / "features.parquet",    output_dir / "metadata.parquet"),
        (eda_dir    / "eda_features.parquet", eda_dir    / "eda_metadata.parquet"),
    ]
    for feat_path, meta_path in candidates:
        if feat_path.exists() and meta_path.exists():
            print(f"Found cached features: {feat_path.name} + {meta_path.name}")
            return pd.read_parquet(feat_path), pd.read_parquet(meta_path)
    return None


def build_stats_from_parquet(
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    d_pw_mm: float,
    rpm_min: float,
    rpm_max: float,
    bp_band_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convert pipeline-format parquet DataFrames into (stats_df, hp_stats_df)."""
    sweep_meta = (
        meta_df.drop_duplicates(["file", "sweep"])
        [["file", "sweep", "rpm", "temperature_c",
          "viscosity_40c_cst", "viscosity_100c_cst"]]
        .copy()
    )

    def _kappa(row: pd.Series) -> float:
        try:
            return calculate_kappa(
                rpm=float(row["rpm"]), temp_c=float(row["temperature_c"]),
                d_pw=d_pw_mm,
                nu_40=float(row.get("viscosity_40c_cst", 22.0)),
                nu_100=float(row.get("viscosity_100c_cst", 4.1)),
            )
        except Exception:
            return np.nan

    sweep_meta["kappa"] = sweep_meta.apply(_kappa, axis=1)
    sweep_meta = sweep_meta[
        (sweep_meta["rpm"] >= rpm_min) & (sweep_meta["rpm"] <= rpm_max)
    ]
    merge_cols = ["file", "sweep", "rpm", "temperature_c", "kappa"]

    stats_df = feat_df.merge(sweep_meta[merge_cols], on=["file", "sweep"], how="inner")

    bp_prefix = f"{bp_band_label}__"
    bp_cols = [c for c in feat_df.columns if c.startswith(bp_prefix)]
    if bp_cols:
        hp_df = (
            feat_df[["file", "sweep", "sensor"] + bp_cols]
            .rename(columns={c: c[len(bp_prefix):] for c in bp_cols})
        )
        hp_stats_df: pd.DataFrame | None = hp_df.merge(
            sweep_meta[merge_cols], on=["file", "sweep"], how="inner"
        )
    else:
        hp_stats_df = None
        print(f"  Note: no '{bp_prefix}*' columns found — skipping BP stats. "
              f"Delete eda_features.parquet to recompute.")

    return stats_df, hp_stats_df


def build_stats_from_sweeps(
    sweeps: list[dict],
    *,
    sensors: tuple[str, ...],
    bp_band_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build (stats_df, hp_stats_df) directly from raw sweep records."""
    stat_rows: list[dict] = []
    bp_rows: list[dict] = []

    for rec in sweeps:
        for sensor in sensors:
            if sensor not in rec["stats"]:
                continue
            base = {
                "file": rec["file"], "sweep": rec["sweep"], "sensor": sensor,
                "rpm": rec["rpm"], "temperature_c": rec["temperature_c"],
                "kappa": rec["kappa"],
            }
            stat_rows.append({**base, **rec["stats"][sensor]})
            if sensor in rec.get("bp_stats", {}):
                bp_rows.append({**base, **rec["bp_stats"][sensor]})

    return pd.DataFrame(stat_rows), (pd.DataFrame(bp_rows) if bp_rows else None)


def save_parquet_cache(
    sweeps: list[dict],
    *,
    eda_dir: Path,
    sensors: tuple[str, ...],
    bp_band_label: str,
) -> None:
    """Save EDA feature cache in pipeline-compatible parquet format."""
    feat_rows: list[dict] = []
    meta_seen: set = set()
    meta_rows: list[dict] = []

    for rec in sweeps:
        key = (rec["file"], rec["sweep"])
        if key not in meta_seen:
            meta_seen.add(key)
            meta_rows.append({
                "file": rec["file"], "sweep": rec["sweep"],
                "rpm": rec["rpm"], "temperature_c": rec["temperature_c"],
                "viscosity_40c_cst": rec.get("viscosity_40c_cst", np.nan),
                "viscosity_100c_cst": rec.get("viscosity_100c_cst", np.nan),
            })
        for sensor in sensors:
            if sensor not in rec["stats"]:
                continue
            row: dict = {"file": rec["file"], "sweep": rec["sweep"], "sensor": sensor}
            row.update(rec["stats"][sensor])
            if sensor in rec.get("bp_stats", {}):
                row.update({f"{bp_band_label}__{k}": v
                             for k, v in rec["bp_stats"][sensor].items()})
            feat_rows.append(row)

    pd.DataFrame(feat_rows).to_parquet(eda_dir / "eda_features.parquet", engine="pyarrow")
    pd.DataFrame(meta_rows).to_parquet(eda_dir / "eda_metadata.parquet", engine="pyarrow")
    print(f"Saved feature cache: eda_features.parquet ({len(feat_rows)} rows), "
          f"eda_metadata.parquet ({len(meta_rows)} rows)")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def make_kappa_intervals(
    kappa_bounds: list[float],
) -> list[tuple[float, float, str]]:
    """Build [(lo, hi, label), ...] from a boundary list."""
    edges = [-np.inf] + list(kappa_bounds) + [np.inf]
    return [
        (lo, hi, (
            f"κ < {hi}" if np.isinf(lo)
            else f"κ ≥ {lo}" if np.isinf(hi)
            else f"{lo} ≤ κ < {hi}"
        ))
        for lo, hi in zip(edges[:-1], edges[1:])
    ]


def collect_psd_rows(sweeps: list[dict], sensors: tuple[str, ...]) -> list[dict]:
    """Flatten per-sweep PSD arrays into a list of dicts keyed by sensor."""
    return [
        {
            "file": rec["file"], "sweep": rec["sweep"],
            "sensor": sensor, "kappa": rec["kappa"],
            "rpm": rec["rpm"],
            "f": rec["psd"][sensor][0],
            "p": rec["psd"][sensor][1],
        }
        for rec in sweeps
        for sensor in sensors
        if sensor in rec["psd"]
    ]


# ---------------------------------------------------------------------------
# Plot functions  (each returns the Figure; caller handles savefig / show)
# ---------------------------------------------------------------------------

def plot_operating_conditions(
    stats_df: pd.DataFrame,
    kappa_bounds: list[float],
) -> plt.Figure:
    meta = stats_df.drop_duplicates(["file", "sweep"]).dropna(subset=["kappa"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(meta["rpm"], meta["temperature_c"],
                         c=meta["kappa"], cmap="viridis", s=20, alpha=0.7, edgecolors="none")
    fig.colorbar(sc, ax=axes[0], label="κ")
    axes[0].set(xlabel="RPM", ylabel="Temperature [°C]", title="Operating conditions space")
    axes[0].grid(ls=":", alpha=0.4)

    axes[1].hist(meta["kappa"], bins=30, color="#1f77b4", edgecolor="white", linewidth=0.4)
    for b in kappa_bounds:
        axes[1].axvline(b, color="red", ls="--", lw=1, alpha=0.8, label=f"κ = {b}")
    axes[1].set(xlabel="κ (lubrication ratio)", ylabel="Count (sweeps)", title="κ distribution")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_waveforms(
    sweeps: list[dict],
    sensors: tuple[str, ...],
    *,
    n_examples: int = 4,
    waveform_ms: float = 20.0,
) -> plt.Figure:
    valid = sorted(
        [r for r in sweeps if not np.isnan(r.get("kappa", np.nan))
         and all(s in r["waveform"] for s in sensors)],
        key=lambda r: r["kappa"],
    )
    examples = [valid[i] for i in np.linspace(0, len(valid) - 1, n_examples, dtype=int)]

    fig, axes = plt.subplots(len(sensors), n_examples,
                             figsize=(5 * n_examples, 3.5 * len(sensors)), sharex=False)
    for row_i, sensor in enumerate(sensors):
        for col_i, rec in enumerate(examples):
            ax = axes[row_i, col_i]
            v = rec["waveform"][sensor]
            t_ms = np.arange(len(v)) / rec["fs"] * 1e3
            ax.plot(t_ms, v, lw=0.4, color=f"C{row_i}", alpha=0.9)
            ax.set(xlabel="Time [ms]", ylabel="Voltage [V]",
                   title=f"{sensor}\nκ = {rec['kappa']:.2f}   RPM = {rec['rpm']:.0f}")
            ax.grid(ls=":", alpha=0.3)

    fig.suptitle(f"Representative waveforms — {n_examples} sweeps ordered by κ "
                 f"(first {waveform_ms:.0f} ms shown)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_time_domain_stats(
    stats_df: pd.DataFrame,
    sensors: tuple[str, ...],
    *,
    bp: bool = False,
    bp_label: str = "",
) -> plt.Figure:
    fig, axes = plt.subplots(len(sensors), len(_STAT_COLS),
                             figsize=(5 * len(_STAT_COLS), 4 * len(sensors)), squeeze=False)
    for row_i, sensor in enumerate(sensors):
        sub = stats_df[stats_df["sensor"] == sensor].dropna(subset=["kappa"])
        for col_i, (col, label) in enumerate(_STAT_COLS):
            ax = axes[row_i, col_i]
            sc = ax.scatter(sub["kappa"], sub[col], c=sub["rpm"], cmap="plasma",
                            s=10, alpha=0.5, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="RPM")
            bins = pd.cut(sub["kappa"], bins=12)
            trend = sub.groupby(bins, observed=True)[col].mean()
            ax.plot([iv.mid for iv in trend.index], trend.values, "r-o", ms=4, lw=1.5, label="bin mean")
            title = f"{sensor} — {label} vs κ" + (f"  [BP {bp_label}]" if bp else "")
            ax.set(xlabel="κ", ylabel=label, title=title)
            if bp:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_psd_kappa_regimes(
    psd_rows: list[dict],
    sensors: tuple[str, ...],
    kappa_intervals: list[tuple[float, float, str]],
    interval_colors: list[str],
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(sensors), figsize=(7 * len(sensors), 5), squeeze=False)
    for col_i, sensor in enumerate(sensors):
        ax = axes[0, col_i]
        rows = [r for r in psd_rows if r["sensor"] == sensor
                and not np.isnan(r.get("kappa", np.nan))]
        if not rows:
            continue
        f_ref = rows[0]["f"]
        for color, (lo, hi, label) in zip(interval_colors, kappa_intervals):
            group = [r for r in rows if lo <= r["kappa"] < hi]
            if not group:
                continue
            p_mean = np.stack([r["p"] for r in group]).mean(axis=0)
            ax.semilogy(f_ref / 1e3, p_mean, lw=1.5, color=color, alpha=0.5,
                        label=f"{label}  (n={len(group)})")
        ax.set(xlabel="Frequency [kHz]", ylabel="PSD  [V² / Hz]",
               title=f"{sensor} — average PSD per κ regime")
        ax.legend(fontsize=8)
        ax.grid(ls=":", which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_envelope(
    sweeps: list[dict],
    sensors: tuple[str, ...],
    kappa_intervals: list[tuple[float, float, str]],
    interval_colors: list[str],
) -> plt.Figure:
    fig, axes = plt.subplots(len(sensors), 2, figsize=(14, 5 * len(sensors)), squeeze=False)
    for row_i, sensor in enumerate(sensors):
        ax_wave, ax_scat = axes[row_i, 0], axes[row_i, 1]
        env_mean_rows: list[dict] = []

        for color, (lo, hi, label) in zip(interval_colors, kappa_intervals):
            group = [r for r in sweeps if lo <= r.get("kappa", np.nan) < hi
                     and sensor in r["envelope"]]
            if not group:
                continue
            n_show = min(len(r["envelope"][sensor]) for r in group)
            t_ms = np.arange(n_show) / group[0]["fs"] * 1e3
            envs = [r["envelope"][sensor][:n_show] for r in group
                    if len(r["envelope"][sensor]) >= n_show]
            for r in group:
                env_mean_rows.append({"kappa": r["kappa"], "mean_env": r["env_mean"][sensor]})
            if envs:
                ax_wave.plot(t_ms, np.mean(envs, axis=0), lw=1.2, color=color, label=label)

        ax_wave.set(xlabel="Time [ms]", ylabel="Envelope [V]",
                    title=f"{sensor} — mean Hilbert envelope per κ regime")
        ax_wave.legend(fontsize=8)
        ax_wave.grid(ls=":", alpha=0.3)

        env_df = pd.DataFrame(env_mean_rows).dropna()
        if not env_df.empty:
            ax_scat.scatter(env_df["kappa"], env_df["mean_env"],
                            s=8, alpha=0.5, edgecolors="none", color="steelblue")
            bins = pd.cut(env_df["kappa"], bins=12)
            trend = env_df.groupby(bins, observed=True)["mean_env"].mean()
            ax_scat.plot([iv.mid for iv in trend.index], trend.values,
                         "r-o", ms=4, lw=1.5, label="bin mean")
        ax_scat.set(xlabel="κ", ylabel="Mean envelope [V]",
                    title=f"{sensor} — mean Hilbert envelope vs κ")
        ax_scat.legend(fontsize=8)
        ax_scat.grid(ls=":", alpha=0.3)

    fig.suptitle("Envelope analysis per κ regime", fontsize=11)
    fig.tight_layout()
    return fig


def plot_envelope_psd(
    sweeps: list[dict],
    sensors: tuple[str, ...],
    kappa_intervals: list[tuple[float, float, str]],
    interval_colors: list[str],
) -> plt.Figure:
    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 5 * len(sensors)), squeeze=False)
    for row_i, sensor in enumerate(sensors):
        ax = axes[row_i, 0]
        for color, (lo, hi, label) in zip(interval_colors, kappa_intervals):
            group = [r for r in sweeps if lo <= r.get("kappa", np.nan) < hi
                     and sensor in r["env_psd"]]
            if not group:
                continue
            f_env = group[0]["env_psd"][sensor][0]
            psds = [r["env_psd"][sensor][1] for r in group]
            min_len = min(len(p) for p in psds)
            psd_mean = np.mean([p[:min_len] for p in psds], axis=0)
            ax.semilogy(f_env[:min_len], psd_mean, lw=1.2, color=color, alpha=0.7, label=label)
        ax.set(xlabel="Frequency [Hz]", ylabel="PSD [V²/Hz]",
               title=f"{sensor} — envelope PSD per κ regime")
        ax.legend(fontsize=8)
        ax.grid(ls=":", which="both", alpha=0.3)
    fig.suptitle("Hilbert envelope PSD per κ regime", fontsize=11)
    fig.tight_layout()
    return fig


def plot_envelope_spectral_flatness(
    sweeps: list[dict],
    sensors: tuple[str, ...],
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(sensors), figsize=(7 * len(sensors), 5), squeeze=False)
    for col_i, sensor in enumerate(sensors):
        ax = axes[0, col_i]
        sf_df = pd.DataFrame([
            {"kappa": r["kappa"], "sf": r["env_sf"][sensor], "rpm": r["rpm"]}
            for r in sweeps
            if not np.isnan(r.get("kappa", np.nan)) and sensor in r["env_sf"]
        ]).dropna()
        if sf_df.empty:
            continue
        sc = ax.scatter(sf_df["kappa"], sf_df["sf"], c=sf_df["rpm"], cmap="plasma",
                        s=8, alpha=0.6, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="RPM")
        bins = pd.cut(sf_df["kappa"], bins=12)
        trend = sf_df.groupby(bins, observed=True)["sf"].mean()
        ax.plot([iv.mid for iv in trend.index], trend.values, "r-o", ms=4, lw=1.5, label="bin mean")
        ax.set_yscale("log")
        ax.set(xlabel="κ", ylabel="Spectral flatness",
               title=f"{sensor} — envelope spectral flatness vs κ")
        ax.legend(fontsize=8)
        ax.grid(ls=":", alpha=0.3)
    fig.suptitle("Envelope spectral flatness vs κ", fontsize=11)
    fig.tight_layout()
    return fig
