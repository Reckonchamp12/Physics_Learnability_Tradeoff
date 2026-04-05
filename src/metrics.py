"""
Evaluation metrics for forward (scalar & spectrum) and inverse (geometry) tasks.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.signal import hilbert

try:
    from dtaidistance import dtw as dtaidtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Spectral metrics
# ─────────────────────────────────────────────────────────────────────────────

def spectral_angle_mapper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SAM in degrees, averaged over samples."""
    dot  = (y_true * y_pred).sum(1)
    norm = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + 1e-8
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1))).mean()


def cosine_similarity_spectra(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    dot  = (y_true * y_pred).sum(1)
    norm = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + 1e-8
    return (dot / norm).mean()


def log_spectral_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-6
    lsd = np.sqrt(((10 * np.log10((y_true + eps) / (y_pred + eps))) ** 2).mean(1))
    return lsd.mean()


def peak_wavelength_error(y_true: np.ndarray, y_pred: np.ndarray,
                          wavelengths: np.ndarray) -> float:
    pk_true = wavelengths[np.argmax(y_true, axis=1)]
    pk_pred = wavelengths[np.argmax(y_pred, axis=1)]
    return np.abs(pk_true - pk_pred).mean()


def _fwhm_single(spectrum: np.ndarray, wavelengths: np.ndarray) -> float:
    half  = spectrum.max() / 2.0
    above = wavelengths[spectrum >= half]
    return float(above[-1] - above[0]) if len(above) >= 2 else 0.0


def fwhm_error(y_true: np.ndarray, y_pred: np.ndarray,
               wavelengths: np.ndarray) -> float:
    return np.mean([abs(_fwhm_single(y_true[i], wavelengths)
                        - _fwhm_single(y_pred[i], wavelengths))
                    for i in range(len(y_true))])


def dtw_distance(y_true: np.ndarray, y_pred: np.ndarray, sub: int = 10) -> float:
    """Average DTW distance on sub-sampled spectra (Euclidean fallback if unavailable)."""
    yt = y_true[:, ::sub]; yp = y_pred[:, ::sub]
    if DTW_AVAILABLE:
        yt = yt.astype(np.float64); yp = yp.astype(np.float64)
        return np.mean([dtaidtw.distance_fast(yt[i], yp[i])
                        for i in range(min(500, len(yt)))])
    return np.sqrt(((yt - yp) ** 2).sum(1)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Physics consistency metrics
# ─────────────────────────────────────────────────────────────────────────────

def energy_conservation_violation(T: np.ndarray, R: np.ndarray, A: np.ndarray) -> float:
    """Mean absolute deviation of (T + R + A) from 1."""
    return np.abs(T + R + A - 1.0).mean()


def fabry_perot_ripple(spectra: np.ndarray) -> float:
    """Estimate FP ripple as std of high-pass filtered spectra."""
    kernel  = np.array([-1, 2, -1]) / 4.0
    ripples = [np.abs(np.convolve(s, kernel, mode="same")).std()
               for s in spectra[:500]]
    return float(np.mean(ripples))


def causality_metric(spectra: np.ndarray) -> float:
    """Hilbert-based causality proxy: fraction of negative imaginary component."""
    return float((np.imag(hilbert(spectra, axis=1)) < 0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-design metrics
# ─────────────────────────────────────────────────────────────────────────────

def success_rate(y_pred: np.ndarray, y_true: np.ndarray, tol_frac: float) -> float:
    """Fraction of predictions within tol_frac relative error."""
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    return float((rel_err <= tol_frac).mean())


def cycle_consistency_error(geo_pred: np.ndarray, geo_true: np.ndarray,
                             scaler_geo) -> float:
    g_p = scaler_geo.inverse_transform(geo_pred)
    g_t = scaler_geo.inverse_transform(geo_true)
    return float(np.abs(g_p - g_t).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated metric bundles
# ─────────────────────────────────────────────────────────────────────────────

def scalar_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    r2_per = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    mape   = np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)).mean() * 100
    pearson = np.mean([stats.pearsonr(y_true[:, i], y_pred[:, i])[0]
                       for i in range(y_true.shape[1])])
    return {
        "r2_per":   r2_per,
        "r2_macro": r2_score(y_true, y_pred, multioutput="uniform_average"),
        "mae":      mean_absolute_error(y_true, y_pred),
        "rmse":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape":     float(mape),
        "max_err":  float(np.abs(y_true - y_pred).max()),
        "pearson":  float(pearson),
    }


def spectrum_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                     wavelengths: np.ndarray) -> dict:
    return {
        "mse":         float(mean_squared_error(y_true, y_pred)),
        "mae":         float(mean_absolute_error(y_true, y_pred)),
        "cosine_sim":  float(cosine_similarity_spectra(y_true, y_pred)),
        "sam_deg":     float(spectral_angle_mapper(y_true, y_pred)),
        "dtw":         float(dtw_distance(y_true, y_pred)),
        "lsd_db":      float(log_spectral_distance(y_true, y_pred)),
        "peak_wl_err": float(peak_wavelength_error(y_true, y_pred, wavelengths)),
        "fwhm_err":    float(fwhm_error(y_true, y_pred, wavelengths)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific metrics (GC-benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def resonance_localization_error(y_true: np.ndarray, y_pred: np.ndarray,
                                  wavelengths: np.ndarray,
                                  min_bw: float = 5.0) -> float:
    """
    Resonance Localization Error (RLE).

    RLE = (1/N) Σ |λ_peak_true − λ_peak_pred| / FWHM_true

    Normalised by the true FWHM so the metric is scale-invariant across
    different resonance widths.  A value < 0.5 means the predicted peak
    is within half a bandwidth of the true peak.

    Args:
        y_true      : true transmission spectra (N, L)
        y_pred      : predicted transmission spectra (N, L)
        wavelengths : wavelength axis (L,) in nm
        min_bw      : minimum FWHM assumed when FWHM cannot be computed [nm]
    """
    rle = []
    for i in range(len(y_true)):
        tp = wavelengths[np.argmax(y_true[i])]
        pp = wavelengths[np.argmax(y_pred[i])]
        half  = y_true[i].max() / 2
        above = wavelengths[y_true[i] > half]
        bw    = (above[-1] - above[0]) if len(above) >= 2 else min_bw
        bw    = max(bw, min_bw)
        rle.append(abs(tp - pp) / bw)
    return float(np.mean(rle))


def energy_consistency_error(y_pred: np.ndarray) -> float:
    """
    Energy Consistency Error (ECE-ML).

    Measures physical feasibility of predicted spectra by checking how
    close R + T is to 1 (assuming negligible absorption for GCs).

    ECE-ML = E_λ[ |R_est(λ) + T(λ) − 1| ]
           = E_λ[ |(1 − T) + T − 1| ]   →  0 for perfectly lossless prediction

    In practice, this signals whether the model has learned a globally
    consistent energy budget.  A value near zero indicates lossless predictions.
    """
    R_est = 1.0 - y_pred
    return float(np.mean(np.abs(R_est + y_pred - 1.0)))


def integrated_power_error(y_true: np.ndarray, y_pred: np.ndarray,
                            wavelengths: np.ndarray) -> float:
    """
    Integrated Power Error (IPE).

    IPE = (1/N) Σ ∫ |T_pred(λ) − T_true(λ)| dλ   [nm]

    Trapezoidal integration over the wavelength grid gives an
    energy-like error with physical units (transmission × nm).
    """
    ipe = [float(np.trapz(np.abs(y_true[i] - y_pred[i]), wavelengths))
           for i in range(len(y_true))]
    return float(np.mean(ipe))


def gc_spectrum_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                         wavelengths: np.ndarray) -> dict:
    """
    Extended spectrum metrics including GC-specific measures.

    Includes all standard spectrum_metrics() plus:
      rle      — Resonance Localization Error
      ece_ml   — Energy Consistency Error
      ipe      — Integrated Power Error (nm)
    """
    base = spectrum_metrics(y_pred, y_true, wavelengths)
    base["rle"]    = resonance_localization_error(y_true, y_pred, wavelengths)
    base["ece_ml"] = energy_consistency_error(y_pred)
    base["ipe"]    = integrated_power_error(y_true, y_pred, wavelengths)
    return base


def inverse_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                    scaler_geo=None) -> dict:
    if scaler_geo is not None:
        y_pred_o = scaler_geo.inverse_transform(y_pred)
        y_true_o = scaler_geo.inverse_transform(y_true)
    else:
        y_pred_o, y_true_o = y_pred, y_true
    rel_err = np.abs(y_pred_o - y_true_o) / (np.abs(y_true_o) + 1e-8)
    return {
        "success_1pct":  float(success_rate(y_pred_o, y_true_o, 0.01)),
        "success_2pct":  float(success_rate(y_pred_o, y_true_o, 0.02)),
        "success_5pct":  float(success_rate(y_pred_o, y_true_o, 0.05)),
        "success_10pct": float(success_rate(y_pred_o, y_true_o, 0.10)),
        "mae":           float(mean_absolute_error(y_true_o, y_pred_o)),
        "rmse":          float(np.sqrt(mean_squared_error(y_true_o, y_pred_o))),
        "med_rel_err":   np.median(rel_err, axis=0).tolist(),
    }
