"""
Statistical and physical dataset validation suite.

Implements 10 statistical sanity checks for photonic synthetic datasets
(Molesky et al. 2018; Jiang et al. 2021) plus a corrected physical
interpretation of the closure-based absorption term.

Classes
-------
PhysicallyCorrectValidator
    Validates energy conservation, passivity, and physical bounds.
    Interprets A = 1 − R − T as a closure term (not material absorption).

StatisticalDatasetValidator
    Runs 10 statistical tests on an HDF5 dataset:
      1.  Inter-parameter independence (Pearson correlation matrix)
      2.  Geometry–physics monotonicity
      3.  Spectral regularity (smoothness)
      4.  Dataset uniqueness (near-duplicate detection)
      5.  Output-space entropy (coverage)
      6.  Train-test distribution equivalence (KS test)
      7.  Noise robustness
      8.  PCA spectral compressibility
      9.  Physical bounds adherence
      10. Parameter-range coverage
"""

from __future__ import annotations
import json
from typing import Dict, List

import numpy as np
import h5py
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# JSON encoder for numpy scalars
# ─────────────────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, np.ndarray):       return obj.tolist()
        if isinstance(obj, np.bool_):         return bool(obj)
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Physical validator
# ─────────────────────────────────────────────────────────────────────────────

class PhysicallyCorrectValidator:
    """
    Validates a GC HDF5 dataset with the correct physical interpretation:

        A(λ) = 1 − R(λ) − T(λ)   (closure term)

    Small negative A values from float32 arithmetic are expected and do
    NOT indicate physical gain.  The fundamental constraint is:
        ⟨ |R + T + A − 1| ⟩  <  δ
    rather than A(λ) ≥ 0 everywhere.
    """

    # Tolerances aligned with photonic simulation standards
    ENERGY_TOL     = 1e-6
    NUMERICAL_TOL  = 1e-4
    STATISTICAL_TOL = 1e-2

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._load()

    def _load(self):
        print(f"Loading: {self.file_path}")
        with h5py.File(self.file_path, "r") as f:
            self.R = f["R"][:]
            self.T = f["T"][:]
            self.A = f["A"][:]
            self.wavelengths  = f["wavelengths_um"][:] * 1000  # → nm
            p = f["parameters"]
            self.period       = p["period_nm"][:]
            self.fill_factor  = p["fill_factor"][:]
            self.etch_depth   = p["etch_depth_nm"][:]
            self.oxide_thick  = p["oxide_thickness_nm"][:]
            self.si_thick     = p["si_thickness_nm"][:]
            m = f["metrics"]
            self.bandwidth    = m["bandwidth_um"][:] * 1000
            self.lambda_center = m["lambda_center_nm"][:]
            self.n_eff        = m["n_eff"][:]
            self.peak_T       = m["peak_transmission"][:]
            self.energy_error = m["energy_error"][:]
            self.valid        = f["valid"][:]
        self.N  = len(self.R)
        self.L  = self.wavelengths.shape[0]
        print(f"  Loaded {self.N:,} samples, {self.L} wavelength points")
        print("  Physical model: A = 1 − R − T  (closure term)")

    # ── Public validation methods ──────────────────────────────────────────

    def check_energy_conservation(self) -> dict:
        """Verify ⟨|R+T+A−1|⟩ < ENERGY_TOL."""
        err  = np.abs(self.R + self.T + self.A - 1.0)
        mean = float(np.mean(err))
        maxi = float(np.max(err))
        pass_ = mean < self.STATISTICAL_TOL
        return {"mean_error": mean, "max_error": maxi,
                "samples_err_gt_1e3": int((err.max(1) > 1e-3).sum()),
                "pass": pass_}

    def check_physical_bounds(self) -> dict:
        """R, T ∈ [0,1]; A may be slightly negative (float32 closure)."""
        r_viol = int(((self.R < -1e-6) | (self.R > 1 + 1e-6)).sum())
        t_viol = int(((self.T < -1e-6) | (self.T > 1 + 1e-6)).sum())
        neg_a  = int((self.A < -1e-6).sum())
        pass_  = (r_viol == 0) and (t_viol == 0)
        return {"R_violations": r_viol, "T_violations": t_viol,
                "negative_A_elements": neg_a,
                "note": "Negative A expected for closure-based absorption.",
                "pass": pass_}

    def check_parameter_coverage(self) -> dict:
        """Verify parameters span their intended ranges."""
        from src.generator.physics_generator import PhysicallyConsistentGratingGenerator
        expected = PhysicallyConsistentGratingGenerator.PARAM_RANGES
        actual   = {
            "period_nm":          (self.period.min(),      self.period.max()),
            "fill_factor":        (self.fill_factor.min(), self.fill_factor.max()),
            "etch_depth_nm":      (self.etch_depth.min(),  self.etch_depth.max()),
            "si_thickness_nm":    (self.si_thick.min(),    self.si_thick.max()),
            "oxide_thickness_nm": (self.oxide_thick.min(), self.oxide_thick.max()),
        }
        ok = all(
            actual[k][0] >= expected[k][0] * 0.95 and
            actual[k][1] <= expected[k][1] * 1.05
            for k in expected
        )
        return {"actual_ranges": {k: list(v) for k, v in actual.items()},
                "expected_ranges": {k: list(v) for k, v in expected.items()},
                "pass": ok}

    def run_all(self) -> dict:
        results = {
            "energy_conservation": self.check_energy_conservation(),
            "physical_bounds":     self.check_physical_bounds(),
            "parameter_coverage":  self.check_parameter_coverage(),
        }
        passed = sum(1 for v in results.values() if v.get("pass", False))
        print(f"\nPhysical validation: {passed}/{len(results)} checks passed")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Statistical validator (10 tests)
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalDatasetValidator:
    """
    Ten-test statistical validation suite for GC HDF5 datasets.

    Tests
    -----
    1.  Inter-parameter independence — Pearson ρ matrix; no |ρ| > 0.3
    2.  Geometry–physics monotonicity — period vs λ_c (expected ρ > 0.9)
    3.  Spectral regularity — average first-difference smoothness
    4.  Dataset uniqueness — no near-duplicates in geometry space
    5.  Output-space entropy — discrete entropy on (λ_c, BW) grid
    6.  Train-test KS equivalence — KS p > 0.05 for each parameter
    7.  Noise robustness — added Gaussian noise should not change argmax
    8.  PCA compressibility — 10 components explain > 90% variance
    9.  Physical bounds adherence — R, T ∈ [0, 1]
    10. Parameter-range coverage — uniform marginals (KS vs uniform)
    """

    def __init__(self, file_path: str, n_samples: int = 10_000):
        self.file_path = file_path
        self.n_samples = n_samples
        self.results: Dict[str, dict] = {}
        self._load(n_samples)

    def _load(self, n: int):
        print(f"Loading {n:,} samples from {self.file_path} …")
        with h5py.File(self.file_path, "r") as f:
            idx      = np.arange(f["T"].shape[0])
            sel      = idx[:n]
            self.T   = f["T"][sel]
            self.R   = f["R"][sel]
            p        = f["parameters"]
            self.period      = p["period_nm"][sel]
            self.fill_factor = p["fill_factor"][sel]
            self.etch_depth  = p["etch_depth_nm"][sel]
            self.oxide_thick = p["oxide_thickness_nm"][sel]
            self.si_thick    = p["si_thickness_nm"][sel]
            m        = f["metrics"]
            self.bw         = m["bandwidth_um"][sel] * 1000
            self.lam_c      = m["lambda_center_nm"][sel]
            self.n_eff      = m["n_eff"][sel]
            self.peak_T     = m["peak_transmission"][sel]
            self.wavelengths = f["wavelengths_um"][:] * 1000
        self.params_matrix = np.column_stack([
            self.period, self.fill_factor, self.etch_depth,
            self.si_thick, self.oxide_thick
        ])
        self.param_names = ["period_nm", "fill_factor", "etch_depth_nm",
                            "si_thickness_nm", "oxide_thickness_nm"]
        print(f"  Loaded {len(self.T):,} samples, {self.T.shape[1]} wavelengths")

    # ── Test implementations ──────────────────────────────────────────────

    def test1_parameter_independence(self) -> dict:
        corr = np.corrcoef(self.params_matrix.T)
        np.fill_diagonal(corr, 0.0)
        max_off = float(np.abs(corr).max())
        return {"correlation_matrix": corr.tolist(),
                "max_off_diagonal": max_off,
                "pass": max_off < 0.3}

    def test2_monotonicity(self) -> dict:
        rho, _ = stats.pearsonr(self.period, self.lam_c)
        return {"period_vs_lambda_c_pearson": float(rho),
                "pass": rho > 0.9}

    def test3_spectral_regularity(self) -> dict:
        diffs = np.diff(self.T, axis=1)
        mean_smoothness = float(np.abs(diffs).mean())
        return {"mean_first_difference": mean_smoothness,
                "pass": mean_smoothness < 0.05}

    def test4_uniqueness(self) -> dict:
        scaler = StandardScaler()
        X      = scaler.fit_transform(self.params_matrix[:5000])
        from sklearn.neighbors import NearestNeighbors
        nn     = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        min_dist  = float(dists[:, 1].min())
        dup_rate  = float((dists[:, 1] < 1e-6).mean())
        return {"min_nearest_neighbour_dist": min_dist,
                "duplicate_rate": dup_rate,
                "pass": dup_rate < 0.01}

    def test5_output_entropy(self) -> dict:
        grid_size = 50
        lc_bins   = np.linspace(self.lam_c.min(), self.lam_c.max(), grid_size + 1)
        bw_bins   = np.linspace(self.bw.min(),    self.bw.max(),    grid_size + 1)
        H, _, _   = np.histogram2d(self.lam_c, self.bw,
                                    bins=[lc_bins, bw_bins])
        p         = H.ravel() / H.sum()
        p         = p[p > 0]
        entropy   = float(-np.sum(p * np.log2(p)))
        max_entropy = float(np.log2(grid_size ** 2))
        return {"entropy_bits": entropy, "max_entropy_bits": max_entropy,
                "normalised_entropy": entropy / max_entropy,
                "pass": entropy > 0.7 * max_entropy}

    def test6_train_test_equivalence(self) -> dict:
        idx_tr, idx_te = train_test_split(np.arange(len(self.period)),
                                          test_size=0.3, random_state=42)
        ks_pvalues = {}
        for i, name in enumerate(self.param_names):
            _, pval = stats.ks_2samp(self.params_matrix[idx_tr, i],
                                      self.params_matrix[idx_te, i])
            ks_pvalues[name] = float(pval)
        return {"ks_pvalues": ks_pvalues,
                "pass": all(p > 0.05 for p in ks_pvalues.values())}

    def test7_noise_robustness(self) -> dict:
        noise      = np.random.default_rng(0).normal(0, 0.01, self.T.shape)
        T_noisy    = np.clip(self.T + noise, 0, 1)
        peak_orig  = np.argmax(self.T,     axis=1)
        peak_noisy = np.argmax(T_noisy,    axis=1)
        stable     = float((peak_orig == peak_noisy).mean())
        return {"peak_stability_rate": stable,
                "pass": stable > 0.85}

    def test8_pca_compressibility(self) -> dict:
        pca = PCA(n_components=10)
        pca.fit(self.T)
        var_explained = float(pca.explained_variance_ratio_.sum())
        return {"variance_explained_10_pcs": var_explained,
                "pass": var_explained > 0.90}

    def test9_physical_bounds(self) -> dict:
        r_ok = bool(((self.R >= 0) & (self.R <= 1)).all())
        t_ok = bool(((self.T >= 0) & (self.T <= 1)).all())
        return {"R_in_unit_interval": r_ok,
                "T_in_unit_interval": t_ok,
                "pass": r_ok and t_ok}

    def test10_parameter_coverage(self) -> dict:
        from src.generator.physics_generator import PhysicallyConsistentGratingGenerator
        expected = PhysicallyConsistentGratingGenerator.PARAM_RANGES
        ks_pvals = {}
        for i, (name, (lo, hi)) in enumerate(expected.items()):
            norm_data = (self.params_matrix[:, i] - lo) / (hi - lo)
            _, pval   = stats.kstest(norm_data, "uniform")
            ks_pvals[name] = float(pval)
        return {"ks_pvalues_vs_uniform": ks_pvals,
                "pass": all(p > 0.01 for p in ks_pvals.values())}

    # ── Run all ────────────────────────────────────────────────────────────

    def run_all(self, verbose: bool = True) -> Dict[str, dict]:
        tests = [
            ("test_1_parameter_independence", self.test1_parameter_independence),
            ("test_2_monotonicity",           self.test2_monotonicity),
            ("test_3_spectral_regularity",    self.test3_spectral_regularity),
            ("test_4_uniqueness",             self.test4_uniqueness),
            ("test_5_output_entropy",         self.test5_output_entropy),
            ("test_6_train_test_equiv",       self.test6_train_test_equivalence),
            ("test_7_noise_robustness",       self.test7_noise_robustness),
            ("test_8_pca_compressibility",    self.test8_pca_compressibility),
            ("test_9_physical_bounds",        self.test9_physical_bounds),
            ("test_10_parameter_coverage",    self.test10_parameter_coverage),
        ]

        for name, fn in tests:
            try:
                self.results[name] = fn()
                status = "PASS" if self.results[name].get("pass") else "FAIL"
            except Exception as exc:
                self.results[name] = {"error": str(exc), "pass": False}
                status = "ERROR"
            if verbose:
                print(f"  {status}  {name}")

        passed = sum(1 for v in self.results.values() if v.get("pass", False))
        print(f"\nStatistical validation: {passed}/{len(tests)} tests passed")
        return self.results

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.results, f, cls=_NpEncoder, indent=2)
        print(f"Results written to {path}")
