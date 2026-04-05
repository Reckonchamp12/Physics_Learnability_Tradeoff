"""
Central configuration for the Grating-Coupler ML Benchmark.
All hyper-parameters and dataset paths live here.
"""

import torch

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 2048
EPOCHS      = 30
PATIENCE    = 5
LR          = 1e-3
WD          = 1e-4

# ── Data splits ───────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC = 0.15  (remainder)

# ── Dataset paths ─────────────────────────────────────────────────────────────
# Update these paths for your environment (Kaggle, local, HPC, etc.)
DATASET_PATHS = {
    "Reference":              "/kaggle/input/datasets/drahulray/gc-all/gc_Reference_50k_20251214_235343.h5",
    "A_NoEnergyConservation": "/kaggle/input/datasets/drahulray/gc-all/gc_A_NoEnergyConservation_50k_20251214_235734.h5",
    "B_NoFabryPerot":         "/kaggle/input/datasets/drahulray/gc-all/gc_B_NoFabryPerot_50k_20251215_000120.h5",
    "C_FixedBandwidth":       "/kaggle/input/datasets/drahulray/gc-all/gc_C_FixedBandwidth_50k_20251215_000515.h5",
    "D_NoNoise":              "/kaggle/input/datasets/drahulray/gc-all/gc_D_NoNoise_50k_20251215_000909.h5",
}

# ── Feature keys ──────────────────────────────────────────────────────────────
PARAM_KEYS   = ["period_nm", "fill_factor", "etch_depth_nm", "oxide_thickness_nm", "si_thickness_nm"]
SCALAR_KEYS  = ["lambda_center_nm", "bandwidth_um", "n_eff", "peak_transmission"]
SPECTRUM_KEYS = ["T", "R", "A"]
