"""
Ablation variants of PhysicallyConsistentGratingGenerator.

Each variant modifies exactly one physical component of the reference
generator so that benchmarks can isolate the effect of individual
physics terms on ML learnability.

Variants
--------
A_NoEnergyConservation  — energy normalisation step omitted
B_NoFabryPerot          — Fabry-Pérot fringes removed
C_FixedBandwidth        — bandwidth fixed at its mean value (no geometry dependence)
D_NoNoise               — measurement noise omitted
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple

from src.generator.physics_generator import PhysicallyConsistentGratingGenerator


# ─────────────────────────────────────────────────────────────────────────────
# A — No energy conservation
# ─────────────────────────────────────────────────────────────────────────────

class AblationA_NoEnergyConservation(PhysicallyConsistentGratingGenerator):
    """
    Ablation A: energy-conservation normalisation step is omitted.

    R + T + A may deviate from 1. This tests whether ML models can detect
    and be harmed by physically inconsistent training data.
    """

    def _spectra_from_coupling(self, T_coup: np.ndarray,
                                p: Dict[str, float],
                                n_eff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mp  = self.model_params
        wl_um = self.wavelengths_um

        R0  = ((n_eff - 1) / (n_eff + 1)) ** 2
        R   = np.clip(R0 + (1 - R0) * (1 - T_coup) * mp["fresnel_loss_factor"], 0, 1)
        T   = np.clip((1 - R0) * T_coup * mp["transmission_efficiency"], 0, 1)

        alpha   = 2.0 + 10.0 * np.exp(-(wl_um - 1.2) / 0.1)
        A_mat   = alpha * mp["scaling_factor"] * (p["si_thickness_nm"] / 100.0)
        A_scat  = mp["absorption_base"] * (p["etch_depth_nm"] / 50.0)
        A_est   = np.clip(A_mat + A_scat, 0, 1)

        # ← normalisation deliberately omitted
        return R.astype(np.float32), T.astype(np.float32), A_est.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# B — No Fabry-Pérot
# ─────────────────────────────────────────────────────────────────────────────

class AblationB_NoFabryPerot(PhysicallyConsistentGratingGenerator):
    """
    Ablation B: Fabry-Pérot oscillation terms are zeroed out.

    Spectra are smoother Lorentzians. This tests whether FP fringes
    encode geometry information that is learnable / necessary.
    """

    def _coupling_efficiency(self, p: Dict[str, float], n_eff: float) -> np.ndarray:
        mp  = self.model_params
        wl  = self.wavelengths_nm
        ff  = p["fill_factor"]
        ed  = p["etch_depth_nm"]

        lam_c  = p["period_nm"] * n_eff
        bw     = (mp["bandwidth_base"]
                  + mp["bandwidth_ff_factor"] * (1 - ff)
                  + mp["bandwidth_etch_factor"] * (ed / 100.0))

        T_lor  = bw ** 2 / (bw ** 2 + (wl - lam_c) ** 2)
        # ← Fabry-Pérot terms omitted
        return np.clip(T_lor, 0, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# C — Fixed bandwidth
# ─────────────────────────────────────────────────────────────────────────────

class AblationC_FixedBandwidth(PhysicallyConsistentGratingGenerator):
    """
    Ablation C: resonance bandwidth fixed at the mean value (40 nm),
    independent of fill_factor and etch_depth.

    This reduces spectral diversity and tests whether bandwidth variation
    is necessary for learning geometry–spectrum mappings.
    """

    FIXED_BW_NM: float = 40.0   # mean of reference bandwidth distribution

    def _coupling_efficiency(self, p: Dict[str, float], n_eff: float) -> np.ndarray:
        wl    = self.wavelengths_nm
        lam_c = p["period_nm"] * n_eff
        bw    = self.FIXED_BW_NM   # ← geometry-independent

        T_lor = bw ** 2 / (bw ** 2 + (wl - lam_c) ** 2)

        # Retain FP fringes (only bandwidth dependence is ablated)
        mp   = self.model_params
        si   = p["si_thickness_nm"]
        L_rt = 2 * n_eff * si
        T_fp = np.zeros_like(wl)
        if L_rt > 0:
            for amp, order in zip(mp["fp_amplitudes"], [1, 2]):
                T_fp += amp * np.sin(2 * np.pi * wl / (L_rt / order)) ** 2

        return np.clip(T_lor + T_fp, 0, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# D — No noise
# ─────────────────────────────────────────────────────────────────────────────

class AblationD_NoNoise(PhysicallyConsistentGratingGenerator):
    """
    Ablation D: measurement / simulation noise is omitted (add_noise=False).

    This produces clean spectra and tests whether noise is needed for
    ML regularisation and robustness.
    """

    def generate_sample(self, params: Optional[Dict[str, float]] = None,
                        sample_idx: int = 0,
                        add_noise: bool = True) -> Dict[str, object]:
        # Force add_noise=False regardless of the caller's argument
        return super().generate_sample(params, sample_idx, add_noise=False)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_REGISTRY = {
    "Reference":              PhysicallyConsistentGratingGenerator,
    "A_NoEnergyConservation": AblationA_NoEnergyConservation,
    "B_NoFabryPerot":         AblationB_NoFabryPerot,
    "C_FixedBandwidth":       AblationC_FixedBandwidth,
    "D_NoNoise":              AblationD_NoNoise,
}
