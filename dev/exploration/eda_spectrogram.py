"""
eda_spectrogram.py
==================
Spectral heatmap: frequency (y-axis) vs. sweep index sorted by κ (x-axis).

Each column is the Welch PSD of one sweep.  Sweeps are sorted by κ so the
plot shows how spectral content shifts across lubrication regimes.

  AE  : full bandwidth, 0 – Nyquist (~6.25 MHz at 12.5 MHz sample rate)
  US  : 0 – 40 kHz; dedicated high-resolution nperseg for the narrow band

White dashed lines mark the sub-band boundaries used in feature extraction.

Usage
-----
    python dev/exploration/eda_spectrogram.py
    python dev/exploration/eda_spectrogram.py --config alt.yaml
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import welch

from ceramicspeed.calculate_kappa import calculate_kappa
from ceramicspeed.config import get_input_dir, get_output_dir, load_config
from ceramicspeed.loading import discover_hdf5_files

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = get_output_dir(cfg)
INPUT_DIR  = get_input_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

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
US_F_MAX_HZ = 40_000.0         # truncate US heatmap at this frequency

# Sub-band boundary overlays (must match config.yaml / tab:subbands in paper)
AE_BAND_EDGES_KHZ = [20.0, 500.0, 1_000.0, 2_000.0]
US_BAND_EDGES_KHZ = [10.0, 20.0]

# %%
# =============================================================================
# Stream sweeps — compute PSDs inline, discard raw waveforms
# =============================================================================

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
                attrs  = dict(sweep.attrs)
                rpm    = float(attrs.get("telem_rpm_meas",    attrs.get("rpm",           np.nan)))
                temp_c = float(attrs.get("telem_omron_pv_c",  attrs.get("temperature_c", np.nan)))

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

                # AE — full bandwidth Welch
                if "AE" in sweep:
                    sig = sweep["AE"]["voltage"][()]
                    f_ae, p_ae = welch(
                        sig, fs=fs,
                        nperseg=AE_NPERSEG,
                        noverlap=int(AE_NPERSEG * NOVERLAP_FRAC),
                        window="hann",
                    )
                    ae_records.append({"kappa": kap, "f": f_ae, "p": p_ae})
                    del sig

                # US — high-resolution Welch, keep 0–40 kHz slice
                if "UL" in sweep:
                    sig = sweep["UL"]["voltage"][()]
                    nperseg_us = min(US_NPERSEG, len(sig))
                    f_us_full, p_us_full = welch(
                        sig, fs=fs,
                        nperseg=nperseg_us,
                        noverlap=int(nperseg_us * NOVERLAP_FRAC),
                        window="hann",
                    )
                    mask_us = f_us_full <= US_F_MAX_HZ
                    us_records.append({
                        "kappa": kap,
                        "f": f_us_full[mask_us],
                        "p": p_us_full[mask_us],
                    })
                    del sig

    except Exception as exc:
        print(f"  WARNING: {fp.name}: {exc}")

print(f"Loaded  {len(ae_records)} AE  /  {len(us_records)} US  sweep PSDs "
      f"({n_skipped} skipped by RPM/κ filter)")

# %%
# =============================================================================
# Build sorted 2-D matrices  (n_sweeps × n_freq_bins)
# =============================================================================

def _build_matrix(
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by κ, stack PSDs.  Returns (kappa_vec, freq_vec, matrix_dB)."""
    records = sorted(records, key=lambda r: r["kappa"])
    kappas  = np.array([r["kappa"] for r in records])
    f_vec   = records[0]["f"]
    mat     = np.stack([r["p"] for r in records])           # (n_sweeps, n_freq)
    mat_db  = 10.0 * np.log10(np.maximum(mat, 1e-30))
    return kappas, f_vec, mat_db


kappas_ae, f_ae, mat_ae_db = _build_matrix(ae_records)
kappas_us, f_us, mat_us_db = _build_matrix(us_records)

print(f"AE matrix : {mat_ae_db.shape}  freq range {f_ae[0]/1e6:.3f}–{f_ae[-1]/1e6:.3f} MHz")
print(f"US matrix : {mat_us_db.shape}  freq range {f_us[0]/1e3:.1f}–{f_us[-1]/1e3:.1f} kHz")

# %%
# =============================================================================
# Plot
# =============================================================================

CMAP = "inferno"

fig, (ax_ae, ax_us) = plt.subplots(
    2, 1, figsize=(16, 11),
    gridspec_kw={"hspace": 0.38},
)


def _kappa_xticks(
    kappas: np.ndarray,
    ax: plt.Axes,
    n_ticks: int = 10,
) -> None:
    """Set x-ticks at evenly spaced sweep indices with κ value labels."""
    idx = np.round(np.linspace(0, len(kappas) - 1, n_ticks)).astype(int)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{kappas[i]:.2f}" for i in idx], fontsize=8)
    ax.set_xlabel("κ (lubrication ratio) — sweeps sorted left→right by κ", fontsize=9)


# ---------- AE heatmap ----------
n_ae = len(kappas_ae)
f_ae_mhz = f_ae / 1e6
vlo_ae, vhi_ae = np.percentile(mat_ae_db, [2, 98])

im_ae = ax_ae.imshow(
    mat_ae_db.T,           # transpose: y = frequency, x = sweep index
    aspect="auto",
    origin="lower",
    extent=[0, n_ae - 1, f_ae_mhz[0], f_ae_mhz[-1]],
    cmap=CMAP,
    vmin=vlo_ae, vmax=vhi_ae,
    interpolation="nearest",
)
fig.colorbar(im_ae, ax=ax_ae, label="PSD  [dB re V²/Hz]", pad=0.01)

for edge_khz in AE_BAND_EDGES_KHZ:
    ax_ae.axhline(edge_khz / 1e3, color="white", lw=0.8, ls="--", alpha=0.7,
                  label=f"{edge_khz:.0f} kHz")
ax_ae.legend(fontsize=7, loc="upper right", framealpha=0.5)

ax_ae.set_ylabel("Frequency  [MHz]", fontsize=9)
ax_ae.set_title(
    f"AE — spectral heatmap  ({n_ae} sweeps, full bandwidth 0–{f_ae_mhz[-1]:.2f} MHz)",
    fontsize=10,
)
_kappa_xticks(kappas_ae, ax_ae)

# ---------- US heatmap ----------
n_us = len(kappas_us)
f_us_khz = f_us / 1e3
vlo_us, vhi_us = np.percentile(mat_us_db, [2, 98])

im_us = ax_us.imshow(
    mat_us_db.T,
    aspect="auto",
    origin="lower",
    extent=[0, n_us - 1, f_us_khz[0], f_us_khz[-1]],
    cmap=CMAP,
    vmin=vlo_us, vmax=vhi_us,
    interpolation="nearest",
)
fig.colorbar(im_us, ax=ax_us, label="PSD  [dB re V²/Hz]", pad=0.01)

for edge_khz in US_BAND_EDGES_KHZ:
    ax_us.axhline(edge_khz, color="white", lw=0.8, ls="--", alpha=0.7,
                  label=f"{edge_khz:.0f} kHz")
ax_us.legend(fontsize=7, loc="upper right", framealpha=0.5)

ax_us.set_ylabel("Frequency  [kHz]", fontsize=9)
ax_us.set_title(
    f"US — spectral heatmap  ({n_us} sweeps, 0–{f_us_khz[-1]:.0f} kHz)",
    fontsize=10,
)
_kappa_xticks(kappas_us, ax_us)

fig.suptitle("Spectral heatmaps: PSD vs κ", fontsize=12, y=0.99)

out_path = EDA_DIR / "eda_spectrogram.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {out_path}")

# %%
if __name__ == "__main__":
    print("\neda_spectrogram complete.")
