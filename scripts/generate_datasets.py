#!/usr/bin/env python
"""
Generate one or more GC dataset variants.

Usage
-----
    # Reference only
    python scripts/generate_datasets.py --variants Reference --n 50000

    # All five variants
    python scripts/generate_datasets.py --variants all --n 50000

    # Custom output directory
    python scripts/generate_datasets.py --variants all --n 50000 --out /data/gc
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generator.ablation_variants import ABLATION_REGISTRY
from src.generator.physics_generator import generate_large_dataset


def main():
    parser = argparse.ArgumentParser(description="GC dataset generator")
    parser.add_argument("--variants", nargs="+", default=["Reference"],
                        help="Variant names or 'all'")
    parser.add_argument("--n",      type=int, default=50_000, help="Samples per dataset")
    parser.add_argument("--out",    type=str, default="./datasets", help="Output directory")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--batch",  type=int, default=5_000)
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()

    variants = list(ABLATION_REGISTRY.keys()) if "all" in args.variants else args.variants
    for v in variants:
        if v not in ABLATION_REGISTRY:
            print(f"[ERROR] Unknown variant '{v}'. Available: {list(ABLATION_REGISTRY.keys())}")
            sys.exit(1)

    for name in variants:
        print(f"\n{'='*60}\nGenerating: {name}\n{'='*60}")
        gen = ABLATION_REGISTRY[name](seed=args.seed)
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)

        # Monkey-patch the generator's batch method into generate_large_dataset
        # by using the imported helper with the correct generator class
        import time, h5py, numpy as np
        n_wl    = len(gen.wavelengths_um)
        ts      = time.strftime("%Y%m%d_%H%M%S")
        h5_path = os.path.join(out_dir, f"gc_{name}_{args.n//1000}k_{ts}.h5")
        compr   = None if args.no_compress else "gzip"

        with h5py.File(h5_path, "w") as f:
            for key in ("R", "T", "A"):
                f.create_dataset(key, (args.n, n_wl), dtype=np.float32,
                                 chunks=(1000, n_wl),
                                 compression=compr,
                                 compression_opts=(None if compr is None else 4))
            for key in gen.PARAM_RANGES:
                f.create_dataset(f"parameters/{key}", (args.n,), dtype=np.float32)
            for key in ("n_eff", "energy_error", "peak_transmission",
                        "bandwidth_um", "lambda_center_nm"):
                f.create_dataset(f"metrics/{key}", (args.n,), dtype=np.float32)
            f.create_dataset("valid", (args.n,), dtype=bool)
            f.create_dataset("wavelengths_um",
                             data=gen.wavelengths_um.astype(np.float32))
            f.attrs.update({"n_samples": args.n, "n_wavelengths": n_wl,
                            "variant": name, "seed": args.seed,
                            "creation_timestamp": ts})

            idx = 0
            while idx < args.n:
                end   = min(idx + args.batch, args.n)
                batch = gen.generate_batch(end - idx)
                f["R"][idx:end] = batch["R"]
                f["T"][idx:end] = batch["T"]
                f["A"][idx:end] = batch["A"]
                for k in gen.PARAM_RANGES:
                    f[f"parameters/{k}"][idx:end] = batch[k]
                for k in ("n_eff", "energy_error", "peak_transmission",
                          "bandwidth_um", "lambda_center_nm"):
                    f[f"metrics/{k}"][idx:end] = batch[k]
                f["valid"][idx:end] = batch["valid"]
                idx = end
                print(f"  {idx:>7,} / {args.n:,}", end="\r", flush=True)

        print(f"\n  Saved: {h5_path}")
    print("\nAll datasets generated.")


if __name__ == "__main__":
    main()
