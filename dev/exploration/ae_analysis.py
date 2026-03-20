
# %%

import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import antropy as ant
from own_utils import own_utils
import importlib
importlib.reload(own_utils)
from scipy import signal
from tqdm import tqdm
import pandas as pd

# %%
# Get files paths

# Work dirs
# INPUT_DIR = "C:/Users/au808956/Documents/Repos/PhD/data/ceramicspeed/"
# OUTPUT_FILE = "C:/Users/au808956/Documents/Repos/PhD/CeramicSpeed_analysis/features/features.parquet"

# Home dirs
INPUT_DIR = "C:/Users/Bruger/Documents/Repos/PhD/data/ceramicspeed/"
OUTPUT_FILE = "C:/Users/Bruger/Documents/Repos/PhD/CeramicSpeed_analysis/features/features.parquet"

files = []
files.extend(Path(INPUT_DIR).glob("*hdf5"))

# %%
import pandas as pd
import h5py

series = {}  # file_name -> DataFrame

test_params = {}

for file_path in files:

    file_name = file_path.stem
    sweep_data = {}
    test_params[file_name] = {}

    with h5py.File(file_path, "r") as f:
        
        sweeps = f["sweeps"]

        print(dict(f["metadata"]["bearing"].attrs))


        for sweep_name, sweep in sweeps.items():

            test_parameters = dict(sweep.attrs)

            sensor = sweep["AE"]

            time = sensor["time"][()]
            voltage = sensor["voltage"][()]

            sweep_data[sweep_name] = voltage

            test_params[file_name][sweep_name] = test_parameters


    df = pd.DataFrame(sweep_data, index=time)
    df.index.name = "time"

    series[file_name] = df
# %%

rows = [
    {"file": f, "sweep": s, **params}
    for f, sweeps in test_params.items()
    for s, params in sweeps.items()
]

meta_df = pd.DataFrame(rows).set_index(["file", "sweep"])
meta_df = meta_df[meta_df.rpm < 10000]

# %%
plt.scatter(meta_df.rpm, meta_df.temperature_c)



# %% Plot averaged spectra

stems = [s.stem for s in files]

for i in range(len(stems)):

    df = series[stems[i]]

    fs = len(df) / (df.index.max() - df.index.min())
    freq = np.fft.fftfreq(len(df), 1/fs)[:len(df) // 2]
    f = np.fft.fft(df, axis=0)
    mag = abs(f)
    avg = np.mean(mag, axis=1)

    max_freq = np.max(np.where(freq < 800000))

    plt.figure()
    plt.plot(freq[:max_freq], avg[:len(avg) // 2][:max_freq])
    plt.title(stems[i])

# %% Envelope spectral analysis

stems = [s.stem for s in files]

for i in range(len(stems)):

    df = series[stems[i]]
    fs = len(df) / (df.index.max() - df.index.min())
    freq = np.fft.fftfreq(len(df), 1/fs)[:len(df) // 2]

    env, res = signal.envelope(df.to_numpy(), axis=0)
    fft = np.fft.fft(env, axis=0)
    mag = abs(fft)
    avg_mag = np.mean(mag, axis=1)

    idx = np.max(np.where(freq < 500000))
    plt.figure()
    plt.plot(freq[:idx], avg_mag[:idx])


# %% Auto power spectral density
stems = [s.stem for s in files]

for i in range(len(stems)):

    df = series[stems[i]]
    t = df.index
    fs = len(t) / (t.max() - t.min())
    s = df.to_numpy()
    f, p = signal.csd(s, s, fs, axis=0)

    start = np.min(np.where(f > 40000))
    end = np.max(np.where(f < 250000))


    plt.figure()
    plt.plot(f[start:end], np.mean(p[start:end], axis=1))

