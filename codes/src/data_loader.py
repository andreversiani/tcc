"""
VeHIF dataset loader.

The VeHIF dataset (Vegetation High-Impedance Fault) was produced by the
Powerline Bushfire Safety Program (PBSP) in Victoria, Australia, and is
described in Gomes & Ozansoy (2022), IEEE Transactions on Power Delivery.
Each HDF5 file corresponds to one test record from the real 22 kV network.

HDF5 structure (to be confirmed against actual files):
    /lf_current     — low-frequency current channel (numpy array)
    /hf_current     — high-frequency current channel (numpy array)
    /lf_voltage     — low-frequency voltage channel (numpy array, if present)
    /metadata/      — sampling rate, species, moisture, label, etc.

NOTE: Implement load_record() after inspecting a real file with:
    import h5py; f = h5py.File('record.h5', 'r'); print(list(f.keys()))
"""

import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd


def load_record(path: str) -> dict:
    """Load a single VeHIF HDF5 record.

    Parameters
    ----------
    path : str
        Path to the .h5 / .hdf5 file.

    Returns
    -------
    dict with keys:
        'lf'       : np.ndarray — low-frequency channel samples
        'hf'       : np.ndarray — high-frequency channel samples
        'fs_lf'    : float      — sampling rate of LF channel (Hz)
        'fs_hf'    : float      — sampling rate of HF channel (Hz)
        'label'    : str        — event label (e.g. 'fault', 'normal')
        'species'  : str        — vegetation species
        'moisture' : str        — moisture condition
        'path'     : str        — source file path
    """
    raise NotImplementedError(
        "Implement after inspecting the HDF5 structure of a real VeHIF file. "
        "Run: import h5py; f = h5py.File(path, 'r'); print(list(f.keys()))"
    )


def load_dataset(data_dir: str, pattern: str = "*.h5") -> pd.DataFrame:
    """Load all VeHIF records from a directory into a DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing .h5 / .hdf5 files.
    pattern : str
        Glob pattern for record files.

    Returns
    -------
    pd.DataFrame with one row per record and columns:
        path, label, species, moisture, fs_lf, fs_hf
    The raw signal arrays are NOT stored in the DataFrame to avoid memory
    issues — use load_record(row['path']) to retrieve them on demand.
    """
    data_dir = Path(data_dir)
    records = []
    for fpath in sorted(data_dir.glob(pattern)):
        rec = load_record(str(fpath))
        records.append({
            "path":     rec["path"],
            "label":    rec["label"],
            "species":  rec["species"],
            "moisture": rec["moisture"],
            "fs_lf":    rec["fs_lf"],
            "fs_hf":    rec["fs_hf"],
        })
    return pd.DataFrame(records)


def get_channel(record: dict, channel: str) -> np.ndarray:
    """Extract a signal channel from a loaded record.

    Parameters
    ----------
    record : dict
        Output of load_record().
    channel : str
        'lf' or 'hf'.

    Returns
    -------
    np.ndarray of signal samples.
    """
    channel = channel.lower()
    if channel not in ("lf", "hf"):
        raise ValueError(f"channel must be 'lf' or 'hf', got '{channel}'")
    return record[channel]
