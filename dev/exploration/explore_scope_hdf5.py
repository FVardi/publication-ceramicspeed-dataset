# %%
"""
Exploration script for scope_*.hdf5 files.
Cells can be run interactively in VS Code (Jupyter mode).
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

FILE = Path("C:/Users/au808956/Documents/Repos/publication-ceramicspeed-dataset/data/scope_20260416_142720.hdf5")

CH_COLORS = {"CHAN1": "C0", "CHAN2": "C1", "CHAN3": "C2", "CHAN4": "C3",
             "AE": "C4", "ACC": "C5"}

# %%
# --- 1. Inspect file structure (top-level only) ---

with h5py.File(FILE, "r") as f:
    print(f"Top-level keys: {list(f.keys())}")
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Group):
            print(f"  [{key}/]  subkeys: {list(obj.keys())[:10]}")
            for k, v in obj.attrs.items():
                print(f"    @ {k}: {v}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  [{key}]  shape={obj.shape}  dtype={obj.dtype}")

# %%
# --- 2. Load metadata and list sweeps / channels ---

with h5py.File(FILE, "r") as f:

    # Metadata
    if "metadata" in f:
        print("Metadata attributes:")
        for k, v in f["metadata"].attrs.items():
            print(f"  {k}: {v}")
        for subkey in f["metadata"].keys():
            print(f"  [{subkey}]")
            for k, v in f["metadata"][subkey].attrs.items():
                print(f"    {k}: {v}")
    else:
        print("No metadata group found.")

    # Sweeps
    if "sweeps" in f:
        sweep_names = sorted(f["sweeps"].keys())
        print(f"\nSweeps ({len(sweep_names)} total): {sweep_names[:10]}{'...' if len(sweep_names) > 10 else ''}")

        # Channels in first sweep
        first_sweep = sweep_names[0]
        channels = [k for k in f[f"sweeps/{first_sweep}"].keys() if k != "digital"]
        print(f"\nChannels in '{first_sweep}': {channels}")

        # Sweep-level attributes
        print(f"\nSweep attributes for '{first_sweep}':")
        for k, v in f[f"sweeps/{first_sweep}"].attrs.items():
            print(f"  {k}: {v}")

        # Channel-level attributes
        for ch in channels:
            print(f"\n  Channel '{ch}' attributes:")
            for k, v in f[f"sweeps/{first_sweep}/{ch}"].attrs.items():
                print(f"    {k}: {v}")
    else:
        print("No sweeps group found.")

# %%
# --- 3. Load all sweeps into memory ---

sweeps_data = {}   # sweep_name -> {channel -> (time, voltage)}
sweep_attrs = {}   # sweep_name -> dict of attributes

with h5py.File(FILE, "r") as f:
    sweep_names = sorted(f["sweeps"].keys())
    channels = [k for k in f[f"sweeps/{sweep_names[0]}"].keys() if k != "digital"]

    for sweep_name in sweep_names:
        grp = f[f"sweeps/{sweep_name}"]
        sweep_attrs[sweep_name] = dict(grp.attrs)
        sweeps_data[sweep_name] = {}
        for ch in channels:
            if ch in grp:
                t = grp[f"{ch}/time"][()]
                v = grp[f"{ch}/voltage"][()]
                sweeps_data[sweep_name][ch] = (t, v)

print(f"Loaded {len(sweeps_data)} sweeps, channels: {channels}")

# %%
# --- 4. Plot time-domain waveforms for a chosen sweep ---

SWEEP = sweep_names[0]   # change as needed

fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3 * len(channels)), sharex=True)
if len(channels) == 1:
    axes = [axes]

for ax, ch in zip(axes, channels):
    t, v = sweeps_data[SWEEP][ch]
    color = CH_COLORS.get(ch, None)
    ax.plot(t * 1e3, v, lw=0.5, color=color)   # time in ms
    ax.set_ylabel(f"{ch}\nVoltage (V)")
    ax.set_title(f"{ch}  —  {sweep_attrs[SWEEP]}", fontsize=8)

axes[-1].set_xlabel("Time (ms)")
fig.suptitle(f"Time domain  |  {FILE.name}  |  {SWEEP}", fontsize=10)
plt.tight_layout()
plt.show()

# %%
# --- 5. FFT for all channels in a chosen sweep ---

SWEEP = sweep_names[0]   # change as needed
MAX_FREQ_HZ = None        # set e.g. 500_000 to zoom in; None = full spectrum

fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3 * len(channels)), sharex=False)
if len(channels) == 1:
    axes = [axes]

for ax, ch in zip(axes, channels):
    t, v = sweeps_data[SWEEP][ch]
    fs = 1.0 / (t[1] - t[0])
    n = len(v)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(v))

    mask = freq <= MAX_FREQ_HZ if MAX_FREQ_HZ else np.ones(len(freq), dtype=bool)
    color = CH_COLORS.get(ch, None)
    ax.plot(freq[mask] / 1e3, mag[mask], lw=0.6, color=color)
    ax.set_ylabel(f"{ch}\nAmplitude")
    ax.set_xlabel("Frequency (kHz)")

fig.suptitle(f"FFT  |  {FILE.name}  |  {SWEEP}", fontsize=10)
plt.tight_layout()
plt.show()

# %%
# --- 6. Spectrogram for a chosen channel and sweep ---

SWEEP = sweep_names[0]   # change as needed
CH = channels[0]          # change as needed
MAX_FREQ_HZ = None        # zoom in e.g. 500_000; None = full

t, v = sweeps_data[SWEEP][CH]
fs = 1.0 / (t[1] - t[0])

f_spec, t_spec, Sxx = signal.spectrogram(v, fs=fs, nperseg=512, noverlap=256)

mask = f_spec <= MAX_FREQ_HZ if MAX_FREQ_HZ else np.ones(len(f_spec), dtype=bool)

fig, ax = plt.subplots(figsize=(14, 4))
pcm = ax.pcolormesh(t_spec * 1e3, f_spec[mask] / 1e3,
                    10 * np.log10(Sxx[mask] + 1e-30),
                    shading="gouraud", cmap="inferno")
plt.colorbar(pcm, ax=ax, label="Power (dB)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Frequency (kHz)")
ax.set_title(f"Spectrogram  |  {FILE.name}  |  {SWEEP}  |  {CH}")
plt.tight_layout()
plt.show()

# %%
# --- 7. Overlay a single channel across all sweeps ---

CH = channels[0]   # change as needed

fig, ax = plt.subplots(figsize=(14, 4))
for sweep_name, data in sweeps_data.items():
    if CH in data:
        t, v = data[CH]
        ax.plot(t * 1e3, v, lw=0.3, alpha=0.5, label=sweep_name)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (V)")
ax.set_title(f"All sweeps overlay  |  {CH}  |  {FILE.name}")
if len(sweeps_data) <= 20:
    ax.legend(fontsize=6, ncol=3)
plt.tight_layout()
plt.show()

# %%
# --- 8. Sweep attributes summary table ---

import pandas as pd

attr_df = pd.DataFrame(sweep_attrs).T
attr_df.index.name = "sweep"
print(attr_df.to_string())
attr_df
