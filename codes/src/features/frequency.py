"""
Frequency-domain feature extraction for VeHIF signals.

Based on DFT analysis as described in the literature review (revisaobibliografica.tex):
  X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)

FAI signatures in the frequency domain:
- Elevated harmonic components (2nd, 3rd, 5th) due to arc non-linearity
- Inter-harmonic content from arc length variation
- High-frequency energy in the 2–10 kHz band (HF channel feature)
"""

import numpy as np
from scipy.fft import rfft, rfftfreq


def _compute_spectrum(sig: np.ndarray, fs: float):
    """Return one-sided magnitude spectrum and frequency axis."""
    N = len(sig)
    magnitudes = np.abs(rfft(sig)) * (2.0 / N)
    freqs = rfftfreq(N, d=1.0 / fs)
    return freqs, magnitudes


def harmonic_amplitudes(
    sig: np.ndarray,
    fs: float,
    fundamental: float = 60.0,
    n_harmonics: int = 5,
    tol_hz: float = 2.0,
) -> np.ndarray:
    """Amplitude of the fundamental and its harmonics.

    Parameters
    ----------
    sig        : input signal window
    fs         : sampling frequency (Hz)
    fundamental: fundamental frequency (Hz) — 60 Hz for Brazil
    n_harmonics: number of harmonics to extract (1 = fundamental only)
    tol_hz     : frequency tolerance window (Hz) around each harmonic

    Returns
    -------
    np.ndarray of length n_harmonics with peak amplitudes at
    fundamental, 2*fundamental, ..., n_harmonics*fundamental.
    """
    freqs, mags = _compute_spectrum(sig, fs)
    amplitudes = np.zeros(n_harmonics)
    for i in range(1, n_harmonics + 1):
        target = i * fundamental
        mask = (freqs >= target - tol_hz) & (freqs <= target + tol_hz)
        amplitudes[i - 1] = np.max(mags[mask]) if np.any(mask) else 0.0
    return amplitudes


def interharmonic_content(
    sig: np.ndarray,
    fs: float,
    fundamental: float = 60.0,
    n_harmonics: int = 5,
    tol_hz: float = 5.0,
) -> float:
    """Total spectral energy outside harmonic frequencies (inter-harmonic content).

    Inter-harmonics arise from the time-varying arc length in VeHIFs and are
    absent in normal load currents.

    Returns
    -------
    Ratio of inter-harmonic energy to total signal energy.
    """
    freqs, mags = _compute_spectrum(sig, fs)
    harmonic_mask = np.zeros(len(freqs), dtype=bool)
    for i in range(1, n_harmonics + 1):
        target = i * fundamental
        harmonic_mask |= (freqs >= target - tol_hz) & (freqs <= target + tol_hz)
    total_energy = float(np.sum(mags ** 2))
    if total_energy == 0:
        return 0.0
    interharmonic_energy = float(np.sum(mags[~harmonic_mask] ** 2))
    return interharmonic_energy / total_energy


def hf_energy(
    sig: np.ndarray,
    fs: float,
    f_low: float = 2000.0,
    f_high: float = 10000.0,
) -> float:
    """Fraction of signal energy in the high-frequency band [f_low, f_high] Hz.

    High-frequency content (2–10 kHz) is the primary signature captured by
    the HF channel of the VeHIF dataset and a key discriminator between
    vegetation fault records and normal operation.

    Returns
    -------
    Ratio of HF band energy to total energy.
    """
    freqs, mags = _compute_spectrum(sig, fs)
    total_energy = float(np.sum(mags ** 2))
    if total_energy == 0:
        return 0.0
    hf_mask = (freqs >= f_low) & (freqs <= f_high)
    hf_band_energy = float(np.sum(mags[hf_mask] ** 2))
    return hf_band_energy / total_energy


def extract_all(
    sig: np.ndarray,
    fs: float,
    fundamental: float = 60.0,
    n_harmonics: int = 5,
) -> dict:
    """Compute all frequency-domain features for a signal window.

    Returns
    -------
    dict with keys:
        fd_harmonic_1 .. fd_harmonic_N  — individual harmonic amplitudes
        fd_interharmonic                — inter-harmonic content ratio
        fd_hf_energy                    — high-frequency energy ratio
    """
    amps = harmonic_amplitudes(sig, fs, fundamental, n_harmonics)
    features = {f"fd_harmonic_{i+1}": float(amps[i]) for i in range(n_harmonics)}
    features["fd_interharmonic"] = interharmonic_content(sig, fs, fundamental, n_harmonics)
    features["fd_hf_energy"] = hf_energy(sig, fs)
    return features
