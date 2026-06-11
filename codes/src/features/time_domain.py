"""
Time-domain feature extraction for VeHIF signals.

Features based on the methodology from:
- Gomes et al. (2018), IEEE Trans. Power Delivery
- Yang et al. (2025), Energies

All functions receive a 1-D numpy array (one signal window) and return a
scalar float, making them composable via extract_all() below.
"""

import numpy as np
from scipy.stats import skew, kurtosis as scipy_kurtosis


def energy(sig: np.ndarray) -> float:
    """Sum of squared samples (signal energy)."""
    return float(np.sum(sig ** 2))


def skewness(sig: np.ndarray) -> float:
    """Statistical skewness (asymmetry of amplitude distribution).

    FAI currents are typically asymmetric between positive and negative
    half-cycles due to the arc's rectifying behaviour.
    """
    return float(skew(sig))


def kurtosis(sig: np.ndarray) -> float:
    """Excess kurtosis (peakedness of amplitude distribution).

    High kurtosis indicates impulsive transients characteristic of arc
    re-ignitions in VeHIFs.
    """
    return float(scipy_kurtosis(sig))


def buildup(sig: np.ndarray, fs: float, fundamental: float = 60.0) -> float:
    """Buildup ratio: energy of later cycles relative to the first cycle.

    The buildup effect is a key VeHIF signature — current grows progressively
    as the contact heats and carbonises the vegetation surface.

    Returns the ratio energy(cycles 2..end) / energy(cycle 1).
    Returns NaN if the signal is shorter than two cycles.

    Parameters
    ----------
    sig        : signal window (should span several cycles)
    fs         : sampling frequency (Hz)
    fundamental: fundamental frequency (Hz)
    """
    samples_per_cycle = int(round(fs / fundamental))
    if len(sig) < 2 * samples_per_cycle:
        return float("nan")
    first_cycle = sig[:samples_per_cycle]
    rest = sig[samples_per_cycle:]
    e_first = float(np.sum(first_cycle ** 2))
    e_rest = float(np.sum(rest ** 2))
    if e_first == 0:
        return float("nan")
    return e_rest / e_first


def roughness(sig: np.ndarray) -> float:
    """Roughness: variance of the first-order difference (signal derivative proxy).

    High roughness indicates rapid local fluctuations associated with arc
    instability in high-impedance faults.
    """
    return float(np.var(np.diff(sig)))


def extract_all(sig: np.ndarray, fs: float, fundamental: float = 60.0) -> dict:
    """Compute all time-domain features for a signal window.

    Returns
    -------
    dict with keys: energy, skewness, kurtosis, buildup, roughness
    """
    return {
        "td_energy":    energy(sig),
        "td_skewness":  skewness(sig),
        "td_kurtosis":  kurtosis(sig),
        "td_buildup":   buildup(sig, fs, fundamental),
        "td_roughness": roughness(sig),
    }
