import numpy as np
from scipy.signal import spectrogram, windows
from scipy.fft import rfft, rfftfreq

def compute_fft(signal, sample_rate):
    N = len(signal)
    win = windows.hann(N)
    signal_win = signal * win
    fft_vals = rfft(signal_win)
    fft_mag = np.abs(fft_vals) / N
    freqs = rfftfreq(N, 1 / sample_rate)
    return freqs, fft_mag

def compute_spectrogram(signal, sample_rate, nperseg=256, noverlap=128):
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window='hann')
    return f, t, Sxx