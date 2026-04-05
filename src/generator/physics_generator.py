"""
PhysicallyConsistentGratingGenerator
=====================================
Semi-analytical model for silicon photonic grating coupler spectra with:
  - Strict energy conservation  (R + T + A = 1 within float32 precision)
  - Uniform parameter sampling over realistic silicon photonics ranges
  - Fabry-Pérot resonance fringes
  - Wavelength-dependent silicon absorption
  - Gaussian measurement noise

Parameter ranges
----------------
  period_nm            : 300 – 700 nm   (grating pitch)
  fill_factor          : 0.3 – 0.7     (duty cycle)
  etch_depth_nm        : 50  – 200 nm
  si_thickness_nm      : 200 – 300 nm
  oxide_thickness_nm   : 1000 – 2000 nm

Wavelength grid
---------------
  100 equally-spaced points from 1.2 µm to 1.6 µm (telecom C-band +)
"""

from __future__ import annotations
import time
import os
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import h5py

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

class PhysicallyConsistentGratingGenerator:
    """
    Generates physically consistent grating coupler spectra.

    Physics model
    -------------
    1. Effective index via semi-analytical waveguide model.
    2. Lorentzian coupling resonance centred at λ = Λ × n_eff.
    3. Fabry-Pérot fringes from the waveguide cavity.
    4. Wavelength-dependent silicon absorption.
    5. Per-wavelength energy conservation enforced by construction.
    """

    # Parameter bounds for uniform sampling
    PARAM_RANGES: Dict[str, Tuple[float, float]] = {
        "period_nm":           (300.0, 700.0),
        "fill_factor":         (0.3,   0.7),
        "etch_depth_nm":       (50.0,  200.0),
        "si_thickness_nm":     (200.0, 300.0),
        "oxide_thickness_nm":  (1000.0, 2000.0),
    }

    # Physical constants (Si, SiO₂, air at 1.55 µm)
    N_SI    = 3.48
    N_AIR   = 1.0
    N_OXIDE = 1.44

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng  = np.random.default_rng(seed)

        self.wavelengths_um = np.linspace(1.2, 1.6, 100)
        self.wavelengths_nm = self.wavelengths_um * 1000

        self.model_params = {
            "slab_decay_length":      150.0,   # nm
            "etch_factor_weight":     0.5,
            "oxide_decay_length":     1000.0,  # nm
            "bandwidth_base":         30.0,    # nm
            "bandwidth_ff_factor":    20.0,    # nm
            "bandwidth_etch_factor":  10.0,    # nm
            "fp_amplitudes":          [0.05, 0.02],
            "fresnel_loss_factor":    0.85,
            "transmission_efficiency": 0.9,
            "absorption_base":        0.01,
            "scaling_factor":         0.001,
            "noise_level":            0.01,
        }

    # ── Physics helpers ───────────────────────────────────────────────────────

    def _effective_index(self, p: Dict[str, float]) -> float:
        ff  = p["fill_factor"]
        ed  = p["etch_depth_nm"]
        si  = p["si_thickness_nm"]
        ox  = p["oxide_thickness_nm"]
        mp  = self.model_params

        n_slab     = self.N_SI * (1 - 0.2 * np.exp(-si / mp["slab_decay_length"]))
        n_grating  = self.N_SI * ff + self.N_AIR * (1 - ff)
        etch_f     = 1 - mp["etch_factor_weight"] * (ed / si)
        n_combined = n_slab * etch_f + n_grating * (1 - etch_f)
        oxide_f    = 1 - 0.3 * np.exp(-ox / mp["oxide_decay_length"])
        return float(n_combined * oxide_f)

    def _coupling_efficiency(self, p: Dict[str, float], n_eff: float) -> np.ndarray:
        mp      = self.model_params
        wl      = self.wavelengths_nm
        ff      = p["fill_factor"]
        ed      = p["etch_depth_nm"]
        si      = p["si_thickness_nm"]

        lam_c   = p["period_nm"] * n_eff
        bw      = (mp["bandwidth_base"]
                   + mp["bandwidth_ff_factor"] * (1 - ff)
                   + mp["bandwidth_etch_factor"] * (ed / 100.0))

        T_lor   = bw ** 2 / (bw ** 2 + (wl - lam_c) ** 2)

        L_rt    = 2 * n_eff * si
        T_fp    = np.zeros_like(wl)
        if L_rt > 0:
            for amp, order in zip(mp["fp_amplitudes"], [1, 2]):
                T_fp += amp * np.sin(2 * np.pi * wl / (L_rt / order)) ** 2

        return np.clip(T_lor + T_fp, 0, 0.95)

    def _spectra_from_coupling(self, T_coup: np.ndarray,
                                p: Dict[str, float],
                                n_eff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mp  = self.model_params
        wl_um = self.wavelengths_um

        R0  = ((n_eff - 1) / (n_eff + 1)) ** 2
        R   = np.clip(R0 + (1 - R0) * (1 - T_coup) * mp["fresnel_loss_factor"], 0, 1)
        T   = np.clip((1 - R0) * T_coup * mp["transmission_efficiency"], 0, 1)

        # Absorption (closure — enforces R + T + A = 1)
        alpha   = 2.0 + 10.0 * np.exp(-(wl_um - 1.2) / 0.1)
        A_mat   = alpha * mp["scaling_factor"] * (p["si_thickness_nm"] / 100.0)
        A_scat  = mp["absorption_base"] * (p["etch_depth_nm"] / 50.0)
        A_est   = np.clip(A_mat + A_scat, 0, 1)

        total   = R + T + A_est
        R /= total; T /= total; A_est /= total        # exact conservation
        return R.astype(np.float32), T.astype(np.float32), A_est.astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_sample(self, params: Optional[Dict[str, float]] = None,
                        sample_idx: int = 0,
                        add_noise: bool = True) -> Dict[str, object]:
        """Generate one sample. Random params drawn if none provided."""
        if params is None:
            params = {k: float(self.rng.uniform(*v))
                      for k, v in self.PARAM_RANGES.items()}

        n_eff   = self._effective_index(params)
        T_coup  = self._coupling_efficiency(params, n_eff)
        R, T, A = self._spectra_from_coupling(T_coup, params, n_eff)

        if add_noise:
            noise = (self.rng.standard_normal(len(T)) * self.model_params["noise_level"]).astype(np.float32)
            T = np.clip(T + noise, 0, 1)
            # Re-normalise after noise injection
            s = R + T + A; R /= s; T /= s; A /= s

        energy_error = float(np.max(np.abs(R + T + A - 1.0)))
        lam_c        = float(self.wavelengths_nm[np.argmax(T)])
        half         = T.max() / 2
        above        = self.wavelengths_um[T >= half]
        bw_um        = float(above[-1] - above[0]) if len(above) >= 2 else 0.0

        return {
            "R": R, "T": T, "A": A,
            "n_eff":             n_eff,
            "energy_error":      energy_error,
            "peak_transmission": float(T.max()),
            "bandwidth_um":      bw_um,
            "lambda_center_nm":  lam_c,
            "valid":             energy_error < 1e-4,
            **params,
        }

    def generate_batch(self, n: int, add_noise: bool = True) -> Dict[str, np.ndarray]:
        """Generate n samples; returns dict of numpy arrays."""
        samples = [self.generate_sample(add_noise=add_noise) for _ in range(n)]
        keys = list(samples[0].keys())
        out = {}
        for k in keys:
            v = samples[0][k]
            if isinstance(v, np.ndarray):
                out[k] = np.stack([s[k] for s in samples])
            else:
                out[k] = np.array([s[k] for s in samples], dtype=np.float32)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate & save HDF5
# ─────────────────────────────────────────────────────────────────────────────

def generate_large_dataset(n_samples: int = 50_000,
                           output_dir: str = "./dataset",
                           seed: int = 42,
                           batch_size: int = 5_000,
                           compress: bool = True,
                           add_noise: bool = True) -> str:
    """
    Generate a large-scale physically consistent grating coupler dataset
    and save it as an HDF5 file.

    Args:
        n_samples   : total number of samples to generate
        output_dir  : directory where the .h5 file will be written
        seed        : random seed for reproducibility
        batch_size  : number of samples generated per iteration
        compress    : whether to apply gzip compression in HDF5
        add_noise   : whether to add measurement noise

    Returns:
        Path to the generated HDF5 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    h5_path  = os.path.join(output_dir, f"gc_{n_samples // 1000}k_{ts}.h5")
    gen      = PhysicallyConsistentGratingGenerator(seed=seed)
    n_wl     = len(gen.wavelengths_um)
    compr    = "gzip" if compress else None
    c_opts   = 4      if compress else None

    print(f"Generating {n_samples:,} samples → {h5_path}")

    with h5py.File(h5_path, "w") as f:
        # Pre-allocate datasets
        for key in ("R", "T", "A"):
            f.create_dataset(key, (n_samples, n_wl), dtype=np.float32,
                             chunks=(1000, n_wl),
                             compression=compr, compression_opts=c_opts)
        for key in gen.PARAM_RANGES:
            f.create_dataset(f"parameters/{key}", (n_samples,), dtype=np.float32)
        for key in ("n_eff", "energy_error", "peak_transmission",
                    "bandwidth_um", "lambda_center_nm"):
            f.create_dataset(f"metrics/{key}", (n_samples,), dtype=np.float32)
        f.create_dataset("valid", (n_samples,), dtype=bool)
        f.create_dataset("wavelengths_um", data=gen.wavelengths_um.astype(np.float32))

        f.attrs.update({"n_samples": n_samples, "n_wavelengths": n_wl,
                        "seed": seed, "creation_timestamp": ts})

        idx = 0
        while idx < n_samples:
            end   = min(idx + batch_size, n_samples)
            batch = gen.generate_batch(end - idx, add_noise=add_noise)
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
            print(f"  {idx:>7,} / {n_samples:,}", end="\r", flush=True)

    print(f"\nDataset saved: {h5_path}")
    return h5_path
