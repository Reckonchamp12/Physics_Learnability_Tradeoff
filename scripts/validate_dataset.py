#!/usr/bin/env python
"""
Validate a GC HDF5 dataset (physical + statistical tests).

Usage
-----
    python scripts/validate_dataset.py --file datasets/Reference/gc_Reference_50k.h5
    python scripts/validate_dataset.py --file gc.h5 --n 20000 --json results/validation.json
"""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.validation.dataset_validator import PhysicallyCorrectValidator, StatisticalDatasetValidator


def main():
    parser = argparse.ArgumentParser(description="GC dataset validator")
    parser.add_argument("--file",  required=True, help="Path to HDF5 dataset")
    parser.add_argument("--n",     type=int, default=10_000,
                        help="Number of samples for statistical tests")
    parser.add_argument("--json",  type=str, default=None,
                        help="Optional path to save JSON results")
    parser.add_argument("--skip-physical",   action="store_true")
    parser.add_argument("--skip-statistical", action="store_true")
    args = parser.parse_args()

    if not args.skip_physical:
        print("\n" + "="*60 + "\nPHYSICAL VALIDATION\n" + "="*60)
        pv = PhysicallyCorrectValidator(args.file)
        pv.run_all()

    if not args.skip_statistical:
        print("\n" + "="*60 + "\nSTATISTICAL VALIDATION (10 tests)\n" + "="*60)
        sv = StatisticalDatasetValidator(args.file, n_samples=args.n)
        results = sv.run_all()
        if args.json:
            os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
            sv.to_json(args.json)


if __name__ == "__main__":
    main()
