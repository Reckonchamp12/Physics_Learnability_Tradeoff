"""
Dataset loading and preprocessing for the GC benchmark.

Each HDF5 file is assumed to have the following structure:
  - Spectra arrays   : T, R, A  — shape (N, L)
  - Wavelength axis  : wavelengths_um — shape (L,)
  - Geometry params  : period_nm, fill_factor, etch_depth_nm,
                       oxide_thickness_nm, si_thickness_nm — shape (N,)
  - Scalar metrics   : lambda_center_nm, bandwidth_um, n_eff,
                       peak_transmission — shape (N,)
  - Valid mask       : valid — shape (N,)  [optional; default all True]
"""

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

from src.config import (
    PARAM_KEYS, SCALAR_KEYS, SPECTRUM_KEYS,
    TRAIN_FRAC, VAL_FRAC, SEED, BATCH_SIZE, DEVICE,
)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 loader
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5_dataset(path: str) -> dict:
    """Load one HDF5 grating-coupler dataset into numpy arrays."""
    data = {}
    with h5py.File(path, "r") as f:
        for k in SPECTRUM_KEYS:
            if k in f:
                data[k] = f[k][:]
        if "wavelengths_um" in f:
            data["wavelengths_um"] = f["wavelengths_um"][:]
        for k in PARAM_KEYS:
            if k in f:
                data[k] = f[k][:]
            elif "parameters" in f and k in f["parameters"]:
                data[k] = f["parameters"][k][:]
        for k in SCALAR_KEYS:
            if k in f:
                data[k] = f[k][:]
            elif "metrics" in f and k in f["metrics"]:
                data[k] = f["metrics"][k][:]
        if "valid" in f:
            data["valid"] = f["valid"][:].astype(bool)
        else:
            n = len(data.get("T", data.get(PARAM_KEYS[0], [])))
            data["valid"] = np.ones(n, dtype=bool)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(raw: dict) -> dict:
    """
    Apply valid mask, build feature matrices, fit scalers on training split.

    Returns a dict with:
        geo_raw, geo_sc     — geometry (N, 5), raw and z-scored
        sca_raw, sca_sc     — scalar targets (N, 4)
        T_raw,  T_sc        — transmission spectrum (N, L)
        R_raw, A_raw        — reflectance / absorption spectra (N, L)
        wavelengths         — wavelength axis (L,)
        idx_train/val/test  — index arrays
        scaler_geo, scaler_scalars, scaler_spectra
        n_wavelengths       — int
    """
    mask    = raw["valid"]
    geo     = np.column_stack([raw[k][mask] for k in PARAM_KEYS]).astype(np.float32)
    scalars = np.column_stack([raw[k][mask] for k in SCALAR_KEYS]).astype(np.float32)
    T       = raw["T"][mask].astype(np.float32)
    R       = raw.get("R", np.zeros_like(T))[mask].astype(np.float32)
    A       = raw.get("A", np.zeros_like(T))[mask].astype(np.float32)
    wavelengths = raw["wavelengths_um"]

    n_total = mask.sum()
    idx     = np.arange(n_total)
    idx_train, idx_temp = train_test_split(idx, train_size=TRAIN_FRAC, random_state=SEED)
    val_of_temp = VAL_FRAC / (1 - TRAIN_FRAC)
    idx_val, idx_test   = train_test_split(idx_temp, train_size=val_of_temp, random_state=SEED)

    scaler_geo     = StandardScaler().fit(geo[idx_train])
    scaler_scalars = StandardScaler().fit(scalars[idx_train])
    T_mean         = T[idx_train].mean(0, keepdims=True)
    T_std          = T[idx_train].std(0, keepdims=True) + 1e-8
    scaler_spectra = {"mean": T_mean, "std": T_std}

    geo_sc = scaler_geo.transform(geo)
    sca_sc = scaler_scalars.transform(scalars)
    T_sc   = (T - T_mean) / T_std

    return {
        "geo_raw": geo,      "geo_sc": geo_sc,
        "sca_raw": scalars,  "sca_sc": sca_sc,
        "T_raw":   T,        "T_sc":   T_sc,
        "R_raw":   R,        "A_raw":  A,
        "wavelengths":  wavelengths,
        "idx_train":    idx_train,
        "idx_val":      idx_val,
        "idx_test":     idx_test,
        "scaler_geo":     scaler_geo,
        "scaler_scalars": scaler_scalars,
        "scaler_spectra": scaler_spectra,
        "n_wavelengths":  T.shape[1],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_tensor(*arrays, device=DEVICE):
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]


def make_loader(X, Y, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE):
    Xt, Yt = to_tensor(X, Y, device=device)
    return DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_datasets(paths: dict, verbose: bool = True) -> dict:
    datasets = {}
    for name, path in paths.items():
        if verbose:
            print(f"  {name}...", end=" ", flush=True)
        raw  = load_hdf5_dataset(path)
        ds   = preprocess_dataset(raw)
        datasets[name] = ds
        if verbose:
            n = len(ds["idx_train"]) + len(ds["idx_val"]) + len(ds["idx_test"])
            print(f"N={n:,}  L={ds['n_wavelengths']}")
    return datasets
