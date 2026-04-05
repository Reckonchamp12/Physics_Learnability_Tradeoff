#!/usr/bin/env python
"""
GC-Benchmark — entry point.

Usage
-----
    python run_benchmark.py                   # forward + inverse, all datasets
    python run_benchmark.py --forward-only
    python run_benchmark.py --inverse-only
    python run_benchmark.py --dataset Reference
"""

import argparse
import pickle
import warnings
import torch
import numpy as np

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
from src.config import SEED, DEVICE, USE_AMP, DATASET_PATHS
torch.manual_seed(SEED)
np.random.seed(SEED)

try:
    from dtaidistance import dtw as _dtw  # noqa: F401
    _dtw_avail = True
except ImportError:
    _dtw_avail = False

try:
    from torchdiffeq import odeint as _oi  # noqa: F401
    _ode_avail = True
except ImportError:
    _ode_avail = False

print(f"Device: {DEVICE}  |  AMP: {USE_AMP}  |  DTW: {_dtw_avail}  |  ODE: {_ode_avail}")


def main():
    parser = argparse.ArgumentParser(description="GC-Benchmark")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--inverse-only", action="store_true")
    parser.add_argument("--dataset",      type=str, default=None,
                        help="Run a single dataset by name.")
    parser.add_argument("--output",       type=str, default="results/benchmark_results.pkl",
                        help="Path to save pickled RESULTS dict.")
    args = parser.parse_args()

    # ── Subset datasets if requested ──────────────────────────────────────
    paths = DATASET_PATHS
    if args.dataset:
        if args.dataset not in paths:
            raise ValueError(f"Unknown dataset '{args.dataset}'. "
                             f"Available: {list(paths.keys())}")
        paths = {args.dataset: paths[args.dataset]}

    # ── Load datasets ─────────────────────────────────────────────────────
    from src.data import load_all_datasets
    print("\nLoading datasets...")
    DATASETS = load_all_datasets(paths)
    print("Datasets loaded.\n")

    RESULTS = {ds: {} for ds in DATASETS}

    # ── Forward tasks ─────────────────────────────────────────────────────
    if not args.inverse_only:
        from experiments.forward_tasks import run_all_forward
        print("\n" + "="*60)
        print("FORWARD TASKS")
        print("="*60)
        run_all_forward(DATASETS, RESULTS)

    # ── Inverse tasks ─────────────────────────────────────────────────────
    if not args.forward_only:
        from experiments.inverse_tasks import run_all_inverse
        print("\n" + "="*60)
        print("INVERSE TASKS")
        print("="*60)
        run_all_inverse(DATASETS, RESULTS)

    # ── Save results ──────────────────────────────────────────────────────
    import os
    os.makedirs("results", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(RESULTS, f)
    print(f"\nResults saved to {args.output}")
    return RESULTS


if __name__ == "__main__":
    main()
