"""
eda_spectrogram.py
==================
Spectral heatmap: frequency (y-axis) vs. sweep index sorted by κ (x-axis).

Each column is the PSD of one sweep, estimated via one of three methods
(--method):

  welch       (default) Sub-segment averaging, Hann-windowed, 50% overlap.
              Matches the main feature-extraction pipeline.
  periodogram Single FFT over the whole sweep, Hann window applied, no
              sub-segment averaging -- noisier than Welch, full resolution.
  rawfft      Single FFT over the whole sweep, NO window (boxcar / rectangular,
              i.e. no tapering), no sub-segmenting, no overlap. The most
              literal "raw FFT" -- equivalent to periodogram with the
              window disabled.

Sweeps are sorted by κ so the plot shows how spectral content shifts across
lubrication regimes.

  AE  : 0 – 2 MHz (display truncated; content above the 1 MHz coupler
        low-pass is excluded from analysis)
  US  : 0 – 100 kHz; dedicated high-resolution nperseg for the narrow band

White dashed lines mark the sub-band boundaries used in feature extraction.

The default (welch) run's chronological figure is also copied to
paper/figures/eda_spectrogram_chronological.png (the paper figure).

Usage
-----
    python dev/exploration/eda_spectrogram.py
    python dev/exploration/eda_spectrogram.py --method periodogram
    python dev/exploration/eda_spectrogram.py --method rawfft
    python dev/exploration/eda_spectrogram.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import pathlib
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import periodogram, welch

from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files, _normalize_sweep_params

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cache and recompute all PSDs from HDF5")
    parser.add_argument("--method", choices=["welch", "periodogram", "rawfft"],
                        default="welch",
                        help="PSD estimation method: welch (default, averaged "
                             "sub-segments), periodogram (single FFT, Hann window), "
                             "or rawfft (single FFT, no window at all)")
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)
METHOD = args.method

OUTPUT_DIR = get_output_dir(cfg)
INPUT_DIR  = get_input_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"

CACHE_PATH = EDA_DIR / f"eda_spectrogram_cache_{METHOD}.npz"

RPM_MIN = cfg["filters"].get("rpm_min", 0.0)
RPM_MAX = cfg["filters"]["rpm_max"]
D_PW_MM = cfg["bearing"]["d_pw_mm"]

_VIS_FALLBACK = {"viscosity_40c_cst": 22.0, "viscosity_100c_cst": 4.1}

# Welch parameters — AE uses config value; US uses larger nperseg for finer
# frequency resolution in the narrow 0–40 kHz band.
AE_NPERSEG   = cfg.get("welch", {}).get("nperseg", 4096)
US_NPERSEG   = 65_536          # ~190 Hz resolution @ 12.5 MHz
NOVERLAP_FRAC = 0.5

# Display limits
AE_F_MAX_HZ = 2_000_000.0      # truncate AE heatmap at this frequency
US_F_MAX_HZ = 100_000.0        # truncate US heatmap at this frequency

# Sub-band boundary overlays (must match config.yaml / tab:subbands in paper)
AE_BAND_EDGES_KHZ = [20.0, 500.0, 1_000.0]
US_BAND_EDGES_KHZ = [10.0, 20.0]

# %%
# =============================================================================
# Stream sweeps — compute PSDs inline, discard raw waveforms
# =============================================================================

_use_cache = (not args.recompute) and CACHE_PATH.exists()
if _use_cache:
    _cache = np.load(CACHE_PATH)
    if "idx_ae" not in _cache.files:
        print("Cache predates chronological support — recomputing from HDF5 ...")
        _use_cache = False

if _use_cache:
    print(f"Loading cached matrices from {CACHE_PATH.name} ...")
    idx_ae     = _cache["idx_ae"]
    idx_us     = _cache["idx_us"]
    kappas_ae  = _cache["kappas_ae"]
    f_ae       = _cache["f_ae"]
    mat_ae_db  = _cache["mat_ae_db"]
    kappas_us  = _cache["kappas_us"]
    f_us       = _cache["f_us"]
    mat_us_db  = _cache["mat_us_db"]
    print(f"AE matrix : {mat_ae_db.shape}  freq range {f_ae[0]/1e6:.3f}–{f_ae[-1]/1e6:.3f} MHz")
    print(f"US matrix : {mat_us_db.shape}  freq range {f_us[0]/1e3:.1f}–{f_us[-1]/1e3:.1f} kHz")

else:
    files = discover_hdf5_files(
        INPUT_DIR,
        file_patterns=cfg.get("filters", {}).get("file_patterns"),
    )
    print(f"Found {len(files)} HDF5 file(s)")

    ae_records: list[dict] = []   # {"kappa": float, "f": ndarray, "p": ndarray}
    us_records: list[dict] = []
    n_skipped = 0

    for fp in files:
        try:
            with h5py.File(fp, "r") as hf:
                sweeps_grp = hf["sweeps"]
                first_key  = list(sweeps_grp.keys())[0]
                time_axis  = sweeps_grp[first_key]["AE"]["time"][()]
                fs: float  = 1.0 / float(np.mean(np.diff(time_axis)))

                lm = dict(hf["metadata"]["lubricant"].attrs)
                for k, v in _VIS_FALLBACK.items():
                    lm.setdefault(k, v)

                for sweep_name, sweep in sweeps_grp.items():
                    sweep_idx = int(sweep_name.split("_")[1])
                    # telem_rpm_meas is unreliable (see dev/exploration/
                    # eda_speed_calibration.py) -- reconstruct rpm from
                    # telem_vfd_cmd_hz via the same normalization the main
                    # pipeline uses, so this figure matches current results.
                    attrs  = _normalize_sweep_params(dict(sweep.attrs))
                    rpm    = float(attrs.get("rpm", np.nan))
                    temp_c = float(attrs.get("temperature_c", np.nan))

                    if not (RPM_MIN <= rpm <= RPM_MAX):
                        n_skipped += 1
                        continue
                    if np.isnan(rpm) or np.isnan(temp_c):
                        n_skipped += 1
                        continue

                    try:
                        kap = calculate_kappa(
                            rpm=rpm, temp_c=temp_c, d_pw=D_PW_MM,
                            nu_40=float(lm["viscosity_40c_cst"]),
                            nu_100=float(lm["viscosity_100c_cst"]),
                        )
                    except Exception:
                        n_skipped += 1
                        continue
                    if np.isnan(kap):
                        n_skipped += 1
                        continue

                    # AE — full bandwidth PSD
                    if "AE" in sweep:
                        sig = sweep["AE"]["voltage"][()]
                        if METHOD == "rawfft":
                            # Most literal "raw FFT": single FFT over the full
                            # sweep, no window (boxcar/rectangular), no
                            # sub-segmenting, no overlap.
                            f_ae, p_ae = periodogram(sig, fs=fs, window="boxcar")
                        elif METHOD == "periodogram":
                            # Single FFT over the full sweep, Hann-windowed,
                            # no sub-segmenting/averaging.
                            f_ae, p_ae = periodogram(sig, fs=fs, window="hann")
                        else:
                            f_ae, p_ae = welch(
                                sig, fs=fs,
                                nperseg=AE_NPERSEG,
                                noverlap=int(AE_NPERSEG * NOVERLAP_FRAC),
                                window="hann",
                            )
                        ae_records.append({"idx": sweep_idx, "kappa": kap, "f": f_ae, "p": p_ae})
                        del sig

                    # US — high-resolution PSD, keep 0–100 kHz slice
                    if "UL" in sweep:
                        sig = sweep["UL"]["voltage"][()]
                        if METHOD == "rawfft":
                            f_us_full, p_us_full = periodogram(sig, fs=fs, window="boxcar")
                        elif METHOD == "periodogram":
                            f_us_full, p_us_full = periodogram(sig, fs=fs, window="hann")
                        else:
                            nperseg_us = min(US_NPERSEG, len(sig))
                            f_us_full, p_us_full = welch(
                                sig, fs=fs,
                                nperseg=nperseg_us,
                                noverlap=int(nperseg_us * NOVERLAP_FRAC),
                                window="hann",
                            )
                        mask_us = f_us_full <= US_F_MAX_HZ
                        us_records.append({
                            "idx": sweep_idx,
                            "kappa": kap,
                            "f": f_us_full[mask_us],
                            "p": p_us_full[mask_us],
                        })
                        del sig

        except Exception as exc:
            print(f"  WARNING: {fp.name}: {exc}")

    print(f"Loaded  {len(ae_records)} AE  /  {len(us_records)} US  sweep PSDs "
          f"({n_skipped} skipped by RPM/κ filter)")

    # Build sorted 2-D matrices  (n_sweeps × n_freq_bins)
    def _build_matrix(
        records: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stack PSDs in chronological (sweep-number) order.

        Returns (sweep_idx_vec, kappa_vec, freq_vec, matrix_dB)."""
        records = sorted(records, key=lambda r: r["idx"])
        idx_vec = np.array([r["idx"] for r in records])
        kappas  = np.array([r["kappa"] for r in records])
        f_vec   = records[0]["f"]
        mat     = np.stack([r["p"] for r in records])
        mat_db  = 10.0 * np.log10(np.maximum(mat, 1e-30))
        return idx_vec, kappas, f_vec, mat_db

    idx_ae, kappas_ae, f_ae, mat_ae_db = _build_matrix(ae_records)
    idx_us, kappas_us, f_us, mat_us_db = _build_matrix(us_records)

    print(f"AE matrix : {mat_ae_db.shape}  freq range {f_ae[0]/1e6:.3f}–{f_ae[-1]/1e6:.3f} MHz")
    print(f"US matrix : {mat_us_db.shape}  freq range {f_us[0]/1e3:.1f}–{f_us[-1]/1e3:.1f} kHz")

    np.savez_compressed(
        CACHE_PATH,
        idx_ae=idx_ae, kappas_ae=kappas_ae, f_ae=f_ae, mat_ae_db=mat_ae_db,
        idx_us=idx_us, kappas_us=kappas_us, f_us=f_us, mat_us_db=mat_us_db,
    )
    print(f"Saved cache → {CACHE_PATH.name}")

# %%
# =============================================================================
# Plot — one figure sorted by κ, one in chronological (sweep-number) order
# =============================================================================

# Truncate the AE heatmap at AE_F_MAX_HZ (display only): content above the
# 1 MHz coupler low-pass is excluded from analysis, so the AE spectrogram is
# shown to 2 MHz rather than to the full Nyquist range.
_ae_mask = f_ae <= AE_F_MAX_HZ
f_ae = f_ae[_ae_mask]
mat_ae_db = mat_ae_db[:, _ae_mask]

# Truncate the US heatmap at US_F_MAX_HZ (display only, applied post-cache so
# a narrower display range doesn't require recomputing from the raw HDF5).
_us_mask = f_us <= US_F_MAX_HZ
f_us = f_us[_us_mask]
mat_us_db = mat_us_db[:, _us_mask]

CMAP = "viridis"


def _render(order_ae, order_us, xlabels_ae, xlabels_us, xlabel, suptitle, outname):
    fig, (ax_ae, ax_us) = plt.subplots(
        2, 1, figsize=(16, 11),
        gridspec_kw={"hspace": 0.38},
    )

    def _xticks(ax, labels, n_total, n_ticks=10):
        idx = np.round(np.linspace(0, n_total - 1, n_ticks)).astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels([labels[i] for i in idx], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)

    # ---------- AE heatmap ----------
    n_ae = len(order_ae)
    f_ae_mhz = f_ae / 1e6
    m_ae = mat_ae_db[order_ae]
    vlo, vhi = np.percentile(m_ae, [2, 98])
    im = ax_ae.imshow(
        m_ae.T, aspect="auto", origin="lower",
        extent=[0, n_ae - 1, f_ae_mhz[0], f_ae_mhz[-1]],
        cmap=CMAP, vmin=vlo, vmax=vhi, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_ae, label="PSD  [dB re V²/Hz]", pad=0.01)
    for edge_khz in AE_BAND_EDGES_KHZ:
        ax_ae.axhline(edge_khz / 1e3, color="white", lw=0.8, ls="--", alpha=0.7,
                      label=f"{edge_khz:.0f} kHz")
    ax_ae.legend(fontsize=7, loc="upper right", framealpha=0.5)
    ax_ae.set_ylabel("Frequency  [MHz]", fontsize=9)
    ax_ae.set_title(
        f"AE — spectral heatmap  ({n_ae} sweeps, 0–{f_ae_mhz[-1]:.2f} MHz)",
        fontsize=10,
    )
    _xticks(ax_ae, xlabels_ae, n_ae)

    # ---------- US heatmap ----------
    n_us = len(order_us)
    f_us_khz = f_us / 1e3
    m_us = mat_us_db[order_us]
    vlo, vhi = np.percentile(m_us, [2, 98])
    im = ax_us.imshow(
        m_us.T, aspect="auto", origin="lower",
        extent=[0, n_us - 1, f_us_khz[0], f_us_khz[-1]],
        cmap=CMAP, vmin=vlo, vmax=vhi, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_us, label="PSD  [dB re V²/Hz]", pad=0.01)
    for edge_khz in US_BAND_EDGES_KHZ:
        ax_us.axhline(edge_khz, color="white", lw=0.8, ls="--", alpha=0.7,
                      label=f"{edge_khz:.0f} kHz")
    if US_BAND_EDGES_KHZ:
        ax_us.legend(fontsize=7, loc="upper right", framealpha=0.5)
    ax_us.set_ylabel("Frequency  [kHz]", fontsize=9)
    ax_us.set_title(
        f"US — spectral heatmap  ({n_us} sweeps, 0–{f_us_khz[-1]:.0f} kHz)",
        fontsize=10,
    )
    _xticks(ax_us, xlabels_us, n_us)

    fig.suptitle(suptitle, fontsize=12, y=0.99)
    out_path = EDA_DIR / outname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


# κ-sorted view (original behaviour)
order_ae = np.argsort(kappas_ae)
order_us = np.argsort(kappas_us)
_method_labels = {"welch": "Welch", "periodogram": "naive periodogram", "rawfft": "raw FFT, no window"}
_method_label = _method_labels[METHOD]
_render(
    order_ae, order_us,
    [f"{k:.2f}" for k in kappas_ae[order_ae]],
    [f"{k:.2f}" for k in kappas_us[order_us]],
    "κ (lubrication ratio) — sweeps sorted left→right by κ",
    f"Spectral heatmaps: PSD vs κ ({_method_label})",
    f"eda_spectrogram_{METHOD}.png",
)

# chronological view (sweep-number order — temperature blocks / staircases visible)
chrono_ae = np.arange(len(idx_ae))
chrono_us = np.arange(len(idx_us))
_render(
    chrono_ae, chrono_us,
    [str(i) for i in idx_ae],
    [str(i) for i in idx_us],
    "sweep number — chronological order",
    f"Spectral heatmaps: PSD vs time (sweep number) ({_method_label})",
    f"eda_spectrogram_chronological_{METHOD}.png",
)

# Copy the chronological view into the paper so it compiles standalone (same
# convention as scripts/new/signal_processing/06_feature_kappa_figure.py).
# Only for the default "welch" method -- that's what the paper figure means
# and what the main feature-extraction pipeline actually uses; a
# --method periodogram/rawfft run shouldn't silently overwrite it.
if METHOD == "welch":
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    _src = EDA_DIR / f"eda_spectrogram_chronological_{METHOD}.png"
    shutil.copy2(_src, PAPER_FIG_DIR / "eda_spectrogram_chronological.png")
    print(f"Copied -> {PAPER_FIG_DIR / 'eda_spectrogram_chronological.png'}")

# %%
if __name__ == "__main__":
    print("\neda_spectrogram complete.")
