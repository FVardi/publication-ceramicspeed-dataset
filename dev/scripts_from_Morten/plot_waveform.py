
#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
from utils.analysis import compute_fft, compute_spectrogram

CHAN_COLORS = {
    "CHAN1": "yellow",
    "CHAN2": "green",
    "CHAN3": "blue",
    "CHAN4": "red",
}

def list_sweeps(h5file):
    with h5py.File(h5file, "r") as f:
        if "sweeps" not in f:
            return []
        return sorted(f["sweeps"].keys())


def list_channels(h5file, sweep_id):
    with h5py.File(h5file, "r") as f:
        grp = f[f"sweeps/{sweep_id}"]
        return [k for k in grp.keys() if k != "digital"]


def load_waveform(h5file, sweep_id, channel_alias):
    with h5py.File(h5file, "r") as f:
        grp = f[f"sweeps/{sweep_id}/{channel_alias}"]
        t = grp["time"][:]
        v = grp["voltage"][:]
        attrs = dict(grp.attrs)
    return t, v, attrs


def load_meta_header(h5file):
    header = {}
    with h5py.File(h5file, "r") as f:
        if "metadata" in f:
            mg = f["metadata"]
            header["created_utc"] = mg.attrs.get("created_utc", "")
            header["scope_idn"] = mg.attrs.get("scope_idn", "")
    return header


def human_fs(sample_rate):
    if sample_rate is None:
        return "fs: n/a"
    units = [("GSa/s", 1e9), ("MSa/s", 1e6), ("kSa/s", 1e3), ("Sa/s", 1)]
    for name, scale in units:
        if sample_rate >= scale:
            return f"fs: {sample_rate/scale:.3g} {name}"
    return f"fs: {sample_rate:.3g} Sa/s"


def channel_legend(alias, attrs):
    src = attrs.get("source", "")
    vdiv = attrs.get("chan_scale_v_per_div", None)
    fs = attrs.get("sample_rate", None)
    vdiv_txt = f"{vdiv:.3g} V/div" if vdiv is not None else "V/div n/a"
    return f"{alias} ({src}, {vdiv_txt}, {human_fs(fs)})"


def plot_time_overlays(traces, title, subtitle):
    fig = go.Figure()
    for alias, (t, v, label, source) in traces.items():
        color = CHAN_COLORS.get(source, None)
        fig.add_trace(go.Scatter(
            x=t, y=v, mode="lines", name=label,
            line=dict(color=color) if color else {}
        ))
    ttl = title if not subtitle else f"{title}<br><sub>{subtitle}</sub>"
    fig.update_layout(title=ttl, xaxis_title="Time (s)", yaxis_title="Voltage (V)")
    fig.show()


def plot_fft_overlays(traces, title, subtitle):
    fig = go.Figure()
    for alias, (f, mag, label, source) in traces.items():
        color = CHAN_COLORS.get(source, None)
        fig.add_trace(go.Scatter(
            x=f, y=mag, mode="lines", name=label,
            line=dict(color=color) if color else {}
        ))
    ttl = title if not subtitle else f"{title}<br><sub>{subtitle}</sub>"
    fig.update_layout(title=ttl, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
    fig.show()


def plot_spectrogram(v, fs, title):
    f_spec, t_spec, Sxx = compute_spectrogram(v, fs)
    fig = go.Figure(data=go.Heatmap(z=10 * np.log10(Sxx), x=t_spec, y=f_spec))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    fig.show()


def main():
    p = argparse.ArgumentParser(description="Plot waveforms from HDF5 (time/FFT/spectrogram).")
    p.add_argument("h5file", help="Path to HDF5 file")
    p.add_argument("--sweep", default=None,
                   help="Sweep id (e.g. 'sweep_000'). If omitted, uses the first sweep in the file.")
    p.add_argument("--channels", default=None,
                   help="Comma-separated channel aliases (e.g. 'AE,ACC'). Default: all analog channels.")
    p.add_argument("--fft", action="store_true", help="Also show FFT overlays for selected channels.")
    p.add_argument("--spectrogram", action="store_true",
                   help="Show spectrogram for the first selected channel.")
    args = p.parse_args()

    sweeps = list_sweeps(args.h5file)
    if not sweeps:
        raise SystemExit("No sweeps found in HDF5.")
    sweep_id = args.sweep or sweeps[0]

    available = list_channels(args.h5file, sweep_id)
    if not available:
        raise SystemExit(f"No analog channels found in {sweep_id}.")
    chosen = [c.strip() for c in args.channels.split(",")] if args.channels else available

    header = load_meta_header(args.h5file)
    subtitle = " • ".join([x for x in [
        header.get("scope_idn", ""),
        header.get("created_utc", "")
    ] if x])

    # --- Time domain overlays
    time_traces = {}
    first_fs = None
    first_alias = None
    for alias in chosen:
        if alias not in available:
            print(f"Warning: channel '{alias}' not in {sweep_id}. Available: {available}")
            continue
        
        t, v, attrs = load_waveform(args.h5file, sweep_id, alias)
        label = channel_legend(alias, attrs)
        source = attrs.get("source", "")
        time_traces[alias] = (t, v, label, source)

        if first_fs is None:
            first_fs = attrs.get("sample_rate", 1.0 / (t[1] - t[0]))
            first_alias = alias

    if not time_traces:
        raise SystemExit("Nothing to plot (no valid channels).")

    plot_time_overlays(time_traces, title=f"{args.h5file} • {sweep_id}", subtitle=subtitle)

    # --- FFT overlays
    if args.fft:
        fft_traces = {}
        for alias, (t, v, label) in time_traces.items():
            fs = 1.0 / (t[1] - t[0])
            f, mag = compute_fft(v, fs)
            fft_traces[alias] = (f, mag, label.replace("(", "FFT ("))
        plot_fft_overlays(fft_traces, title=f"FFT • {args.h5file} • {sweep_id}", subtitle=subtitle)

    # --- Spectrogram (first selected channel only)
    if args.spectrogram:
        t, v, _ = load_waveform(args.h5file, sweep_id, first_alias)
        fs = 1.0 / (t[1] - t[0])
        plot_spectrogram(v, fs, title=f"Spectrogram • {first_alias} • {sweep_id}")


if __name__ == "__main__":
    main()
