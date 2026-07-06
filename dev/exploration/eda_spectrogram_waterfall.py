"""
eda_spectrogram_waterfall.py
=============================
3D surface view of the same PSD data as eda_spectrogram.py: frequency vs.
sweep order, with PSD shown as height (z) and coloured by that same height
(dB), so amplitude is encoded twice (redundantly, for readability) rather
than via a separate κ colour channel. Rows can be ordered/offset along the
y-axis either by κ (default) or chronologically by sweep number (--chrono).

Reuses the PSD cache produced by eda_spectrogram.py -- run that script
first with the matching --method so the cache file exists.

A subset of sweeps (evenly spaced after sorting) forms the surface rows;
using all ~8500 sweeps would be too slow/dense to render usefully.

Usage
-----
    python dev/exploration/eda_spectrogram.py --method welch        # build the cache first
    python dev/exploration/eda_spectrogram_waterfall.py
    python dev/exploration/eda_spectrogram_waterfall.py --method periodogram
    python dev/exploration/eda_spectrogram_waterfall.py --method rawfft
    python dev/exploration/eda_spectrogram_waterfall.py --chrono
    python dev/exploration/eda_spectrogram_waterfall.py --n-lines 150
"""

# %%
# =============================================================================
# Imports
# =============================================================================

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ceramicspeed.config import get_output_dir, load_config

# %%
# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--method", choices=["welch", "periodogram", "rawfft"],
                        default="welch",
                        help="Which PSD cache to load (must match how "
                             "eda_spectrogram.py was last run with --method)")
    parser.add_argument("--n-lines", type=int, default=150,
                        help="Number of sweep rows in the surface mesh")
    parser.add_argument("--chrono", action="store_true",
                        help="Order/offset rows by sweep number (chronological) "
                             "instead of by κ.")
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
cfg = load_config(args.config)
METHOD = args.method
ORDER_LABEL = "sweep number" if args.chrono else "κ"

OUTPUT_DIR = get_output_dir(cfg)
EDA_DIR    = OUTPUT_DIR / "eda"
CACHE_PATH = EDA_DIR / f"eda_spectrogram_cache_{METHOD}.npz"

AE_F_MAX_HZ = 1_400_000.0
US_F_MAX_HZ = 100_000.0
N_ROWS = args.n_lines

# %%
# =============================================================================
# Load cached PSD matrices (computed by eda_spectrogram.py)
# =============================================================================

if not CACHE_PATH.exists():
    raise FileNotFoundError(
        f"{CACHE_PATH} not found. Run "
        f"`eda_spectrogram.py --method {METHOD}` first to build it."
    )

_cache = np.load(CACHE_PATH)
idx_ae    = _cache["idx_ae"]
kappas_ae = _cache["kappas_ae"]
f_ae      = _cache["f_ae"]
mat_ae_db = _cache["mat_ae_db"]
idx_us    = _cache["idx_us"]
kappas_us = _cache["kappas_us"]
f_us      = _cache["f_us"]
mat_us_db = _cache["mat_us_db"]

print(f"Loaded cache: {CACHE_PATH.name}")
print(f"AE matrix : {mat_ae_db.shape}")
print(f"US matrix : {mat_us_db.shape}")

# %%
# =============================================================================
# Waterfall surface plot
# =============================================================================


def _waterfall(y_values, f, mat_db, f_max_hz, freq_unit_div, freq_label,
                y_label, band_edges, title, outname, max_points_per_line=400):
    order = np.argsort(y_values)
    mask = f <= f_max_hz
    f_disp = f[mask] / freq_unit_div

    # Decimate the frequency axis so the mesh stays a manageable size.
    step = max(1, f_disp.size // max_points_per_line)
    f_disp = f_disp[::step]

    # Evenly spaced subset of sorted sweeps forms the surface rows.
    pick = np.linspace(0, len(order) - 1, min(N_ROWS, len(order))).astype(int)
    sel = order[pick]

    Z = mat_db[sel][:, mask][:, ::step]          # (n_rows, n_freq)
    X, Y = np.meshgrid(f_disp, y_values[sel])    # both (n_rows, n_freq)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((2.2, 1.4, 1.0))

    z_lo, z_hi = np.percentile(Z, [1, 99.5])

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="viridis", vmin=z_lo, vmax=z_hi,
        linewidth=0, antialiased=True, rstride=1, cstride=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="PSD  [dB re V²/Hz]")

    ax.set_xlim(f_disp.min(), f_disp.max())
    ax.set_ylim(y_values[sel].min(), y_values[sel].max())
    ax.set_zlim(z_lo, z_hi)

    for edge in band_edges:
        ax.plot([edge, edge], [y_values[sel].min(), y_values[sel].max()], [z_lo, z_lo],
                color="grey", lw=1.0, ls="--", alpha=0.7)

    ax.set_xlabel(freq_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_zlabel("PSD  [dB re V²/Hz]", labelpad=10)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=25, azim=-55)

    out_path = EDA_DIR / outname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


_y_ae = idx_ae.astype(float) if args.chrono else kappas_ae
_y_us = idx_us.astype(float) if args.chrono else kappas_us
_y_label = "Sweep number" if args.chrono else "κ"
_suffix = f"chronological_{METHOD}" if args.chrono else METHOD

_waterfall(
    _y_ae, f_ae, mat_ae_db, AE_F_MAX_HZ, 1e6, "Frequency  [MHz]",
    _y_label,
    band_edges=[0.02, 0.5, 1.0],
    title=f"AE waterfall — PSD vs frequency, rows ordered by {ORDER_LABEL} ({METHOD})",
    outname=f"eda_spectrogram_waterfall_ae_{_suffix}.png",
)

_waterfall(
    _y_us, f_us, mat_us_db, US_F_MAX_HZ, 1e3, "Frequency  [kHz]",
    _y_label,
    band_edges=[10.0, 20.0],
    title=f"US waterfall — PSD vs frequency, rows ordered by {ORDER_LABEL} ({METHOD})",
    outname=f"eda_spectrogram_waterfall_us_{_suffix}.png",
)

# %%
if __name__ == "__main__":
    print("\neda_spectrogram_waterfall complete.")
