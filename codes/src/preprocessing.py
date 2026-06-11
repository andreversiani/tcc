"""
Signal preprocessing utilities for VeHIF records.

Typical pipeline for a raw current signal:
    1. remove_dc()          — remove constant offset
    2. bandpass_filter()    — isolate frequency band of interest
    3. normalize()          — scale to [-1, 1] or unit variance
    4. segment_by_cycles()  — split into analysis windows
"""

import numpy as np
from scipy import signal as sp_signal


def remove_dc(sig: np.ndarray) -> np.ndarray:
    """Subtract the mean to remove the DC component."""
    return sig - np.mean(sig)


def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    sig     : input signal
    fs      : sampling frequency (Hz)
    lowcut  : lower cutoff frequency (Hz)
    highcut : upper cutoff frequency (Hz)
    order   : filter order (default 4)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, sig)


def normalize(sig: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalize signal amplitude.

    Parameters
    ----------
    sig    : input signal
    method : 'minmax' scales to [-1, 1]; 'zscore' gives zero mean, unit std
    """
    if method == "minmax":
        peak = np.max(np.abs(sig))
        return sig / peak if peak > 0 else sig
    elif method == "zscore":
        std = np.std(sig)
        return (sig - np.mean(sig)) / std if std > 0 else sig
    else:
        raise ValueError(f"method must be 'minmax' or 'zscore', got '{method}'")


def segment_by_cycles(
    sig: np.ndarray,
    fs: float,
    n_cycles: int = 5,
    fundamental: float = 60.0,
    overlap: float = 0.0,
) -> list:
    """Split signal into windows of exactly n_cycles of the fundamental.

    Parameters
    ----------
    sig        : input signal
    fs         : sampling frequency (Hz)
    n_cycles   : number of cycles per window
    fundamental: fundamental frequency (Hz) — 60 Hz for Brazil
    overlap    : fractional overlap between consecutive windows [0, 1)

    Returns
    -------
    List of np.ndarray windows.
    """
    samples_per_cycle = int(round(fs / fundamental))
    window_size = n_cycles * samples_per_cycle
    step = int(window_size * (1.0 - overlap))
    windows = []
    start = 0
    while start + window_size <= len(sig):
        windows.append(sig[start : start + window_size])
        start += step
    return windows
