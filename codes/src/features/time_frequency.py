"""
Time-frequency feature extraction for VeHIF signals.

Two transforms are used (as described in revisaobibliografica.tex):

DWT — Discrete Wavelet Transform:
    d_j[k] = sum_n x[n] * h[2k - n]   (detail coefficients, high-frequency)
    a_j[k] = sum_n x[n] * g[2k - n]   (approximation coefficients, low-frequency)
    Captures arc re-ignition transients via per-level energy.

STFT — Short-Time Fourier Transform:
    STFT{x}(m, k) = sum_n x[n] * w[n-m] * exp(-j*2*pi*k*n/N)
    Provides time-varying spectral content with fixed resolution trade-off.
"""

import numpy as np
import pywt
from scipy.signal import stft as scipy_stft


def dwt_features(
    sig: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> np.ndarray:
    """Per-level DWT detail energy as a feature vector.

    The db4 (Daubechies 4) wavelet is widely used for transient fault
    analysis due to its good time-frequency localisation properties.

    Parameters
    ----------
    sig    : input signal window
    wavelet: PyWavelets wavelet name (default 'db4')
    level  : decomposition depth

    Returns
    -------
    np.ndarray of length `level` with relative energy of detail coefficients
    at levels 1 (highest frequency) through `level` (lowest detail).
    Each value is the ratio of detail energy to total signal energy.
    """
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # coeffs[0] = approximation, coeffs[1..level] = details (level 1 = finest)
    detail_energies = np.array([float(np.sum(c ** 2)) for c in coeffs[1:]])
    total = float(np.sum(sig ** 2))
    if total == 0:
        return np.zeros(level)
    return detail_energies / total


def stft_hf_energy(
    sig: np.ndarray,
    fs: float,
    nperseg: int = 256,
    f_low: float = 2000.0,
    f_high: float = 10000.0,
) -> float:
    """Mean high-frequency energy across STFT time frames.

    Measures how much high-frequency content (2–10 kHz) persists over time,
    which is characteristic of arc-sustained VeHIFs vs. transient events.

    Parameters
    ----------
    sig     : input signal window
    fs      : sampling frequency (Hz)
    nperseg : STFT window length in samples
    f_low   : lower bound of high-frequency band (Hz)
    f_high  : upper bound of high-frequency band (Hz)

    Returns
    -------
    Mean fraction of energy in [f_low, f_high] across all STFT frames.
    """
    freqs, _, Zxx = scipy_stft(sig, fs=fs, nperseg=nperseg)
    power = np.abs(Zxx) ** 2
    hf_mask = (freqs >= f_low) & (freqs <= f_high)
    total_per_frame = np.sum(power, axis=0)
    hf_per_frame = np.sum(power[hf_mask, :], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_per_frame = np.where(total_per_frame > 0, hf_per_frame / total_per_frame, 0.0)
    return float(np.mean(ratio_per_frame))


def extract_all(
    sig: np.ndarray,
    fs: float,
    wavelet: str = "db4",
    dwt_level: int = 5,
    nperseg: int = 256,
) -> dict:
    """Compute all time-frequency features for a signal window.

    Returns
    -------
    dict with keys:
        tf_dwt_energy_1 .. tf_dwt_energy_N  — DWT detail energies per level
        tf_stft_hf_energy                   — mean STFT HF energy ratio
    """
    dwt_energies = dwt_features(sig, wavelet=wavelet, level=dwt_level)
    features = {f"tf_dwt_energy_{i+1}": float(e) for i, e in enumerate(dwt_energies)}
    features["tf_stft_hf_energy"] = stft_hf_energy(sig, fs, nperseg=nperseg)
    return features
