"""
Forward task runners: geometry → scalar metrics, geometry → T spectrum.

Each runner fits a model on the training split of a single dataset dict
and returns a flat metrics dict that can be stored in RESULTS.
"""

import time
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

from src.config import DEVICE, SEED, BATCH_SIZE, EPOCHS, PATIENCE
from src.data import make_loader
from src.metrics import scalar_metrics, spectrum_metrics, energy_conservation_violation, fabry_perot_ripple, causality_metric
from src.trainer import train_pytorch, reset_gpu_mem, gpu_mem_mb, compute_timing
from src.models.forward_models import MLP, ResNet1D, TabularTransformer, DilatedCNN1D, NeuralODE, DeepONet, FNO1D


# ─────────────────────────────────────────────────────────────────────────────
# Tree / sklearn runners
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_scalar_tree(model, ds: dict, tag: str = "") -> dict:
    """Fit a sklearn/XGBoost model for forward scalar prediction."""
    tr, te = ds["idx_train"], ds["idx_test"]
    X_tr, X_te = ds["geo_raw"][tr], ds["geo_raw"][te]
    Y_tr, Y_te = ds["sca_raw"][tr], ds["sca_raw"][te]

    t0      = time.time()
    model.fit(X_tr, Y_tr)
    t_train = time.time() - t0
    Y_pred  = model.predict(X_te)
    mets    = scalar_metrics(Y_pred, Y_te)

    t0 = time.perf_counter()
    for _ in range(50): model.predict(X_te[:1])
    t_b1 = (time.perf_counter() - t0) / 50 * 1000

    x1k  = X_te[:1024] if len(X_te) >= 1024 else X_te
    t0   = time.perf_counter()
    for _ in range(10): model.predict(x1k)
    t_b1k = (time.perf_counter() - t0) / 10 * 1000

    mets.update({"train_time_s": t_train,
                 "infer_batch1_ms": t_b1, "infer_batch1024_ms": t_b1k})
    return mets


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch NN runners
# ─────────────────────────────────────────────────────────────────────────────

def run_forward_scalar_nn(model_cls, model_kwargs: dict,
                           ds_name: str, ds: dict, tag: str = "",
                           use_amp_override=None) -> tuple:
    tr, va, te       = ds["idx_train"], ds["idx_val"], ds["idx_test"]
    X_tr, X_va, X_te = ds["geo_sc"][tr], ds["geo_sc"][va], ds["geo_sc"][te]
    Y_tr, Y_va       = ds["sca_sc"][tr], ds["sca_sc"][va]

    in_d, out_d = X_tr.shape[1], Y_tr.shape[1]
    model       = model_cls(in_d, out_d, **model_kwargs).to(DEVICE)

    reset_gpu_mem()
    t_train = train_pytorch(model, make_loader(X_tr, Y_tr),
                             make_loader(X_va, Y_va, shuffle=False),
                             use_amp_override=use_amp_override)

    model.eval()
    with torch.no_grad():
        X_te_t    = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        Y_pred_sc = model(X_te_t).cpu().numpy()

    Y_pred = ds["scaler_scalars"].inverse_transform(Y_pred_sc)
    Y_true = ds["sca_raw"][te]
    mets   = scalar_metrics(Y_pred, Y_true)
    mets.update({"train_time_s": t_train, "gpu_mem_mb": gpu_mem_mb(),
                 **compute_timing(model, X_te_t)})
    return mets, model


def run_forward_spectrum_nn(model_cls, model_kwargs: dict,
                             ds_name: str, ds: dict, tag: str = "",
                             use_amp_override=None) -> tuple:
    tr, va, te       = ds["idx_train"], ds["idx_val"], ds["idx_test"]
    X_tr, X_va, X_te = ds["geo_sc"][tr], ds["geo_sc"][va], ds["geo_sc"][te]
    Y_tr, Y_va       = ds["T_sc"][tr],   ds["T_sc"][va]

    in_d, out_d = X_tr.shape[1], Y_tr.shape[1]
    model       = model_cls(in_d, out_d, **model_kwargs).to(DEVICE)

    reset_gpu_mem()
    t_train = train_pytorch(model, make_loader(X_tr, Y_tr),
                             make_loader(X_va, Y_va, shuffle=False),
                             use_amp_override=use_amp_override)

    model.eval()
    with torch.no_grad():
        X_te_t    = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        Y_pred_sc = model(X_te_t).float().cpu().numpy()

    sp     = ds["scaler_spectra"]
    Y_pred = Y_pred_sc * sp["std"] + sp["mean"]
    Y_true = ds["T_raw"][te]

    mets = spectrum_metrics(Y_pred, Y_true, ds["wavelengths"])
    mets["energy_viol"] = energy_conservation_violation(Y_pred, ds["R_raw"][te], ds["A_raw"][te])
    mets["fp_ripple"]   = fabry_perot_ripple(Y_pred)
    mets["causality"]   = causality_metric(Y_pred)
    mets.update({"train_time_s": t_train, "gpu_mem_mb": gpu_mem_mb(),
                 **compute_timing(model, X_te_t)})
    return mets, model


def run_forward_spectrum_deeponet(ds: dict, tag: str = "") -> dict:
    tr, va, te       = ds["idx_train"], ds["idx_val"], ds["idx_test"]
    X_tr, X_va, X_te = ds["geo_sc"][tr], ds["geo_sc"][va], ds["geo_sc"][te]
    Y_tr, Y_va       = ds["T_sc"][tr],   ds["T_sc"][va]

    in_d, n_wl = X_tr.shape[1], ds["n_wavelengths"]
    model      = DeepONet(in_d, n_wl).to(DEVICE)

    reset_gpu_mem()
    t_train = train_pytorch(model, make_loader(X_tr, Y_tr),
                             make_loader(X_va, Y_va, shuffle=False))
    model.eval()
    with torch.no_grad():
        X_te_t    = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        Y_pred_sc = model(X_te_t).cpu().numpy()

    sp = ds["scaler_spectra"]
    Y_pred, Y_true = Y_pred_sc * sp["std"] + sp["mean"], ds["T_raw"][te]
    mets = spectrum_metrics(Y_pred, Y_true, ds["wavelengths"])
    mets["energy_viol"] = energy_conservation_violation(Y_pred, ds["R_raw"][te], ds["A_raw"][te])
    mets["fp_ripple"]   = fabry_perot_ripple(Y_pred)
    mets["causality"]   = causality_metric(Y_pred)
    mets.update({"train_time_s": t_train, "gpu_mem_mb": gpu_mem_mb(),
                 **compute_timing(model, X_te_t)})
    return mets


def run_forward_spectrum_fno(ds: dict, tag: str = "") -> dict:
    """
    FNO1D runner — AMP disabled to keep FFT in fp32.
    cuFFT fp16 requires power-of-two signal lengths; wavelength grids
    (e.g., L=100) violate this constraint.
    """
    tr, va, te       = ds["idx_train"], ds["idx_val"], ds["idx_test"]
    X_tr, X_va, X_te = ds["geo_sc"][tr], ds["geo_sc"][va], ds["geo_sc"][te]
    Y_tr, Y_va       = ds["T_sc"][tr],   ds["T_sc"][va]

    in_d, n_wl = X_tr.shape[1], ds["n_wavelengths"]
    model      = FNO1D(in_d, n_wl).to(DEVICE)

    reset_gpu_mem()
    t_train = train_pytorch(model, make_loader(X_tr, Y_tr),
                             make_loader(X_va, Y_va, shuffle=False),
                             use_amp_override=False)
    model.eval()
    with torch.no_grad():
        X_te_t    = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        Y_pred_sc = model(X_te_t).cpu().numpy()

    sp = ds["scaler_spectra"]
    Y_pred, Y_true = Y_pred_sc * sp["std"] + sp["mean"], ds["T_raw"][te]
    mets = spectrum_metrics(Y_pred, Y_true, ds["wavelengths"])
    mets["energy_viol"] = energy_conservation_violation(Y_pred, ds["R_raw"][te], ds["A_raw"][te])
    mets["fp_ripple"]   = fabry_perot_ripple(Y_pred)
    mets["causality"]   = causality_metric(Y_pred)
    mets.update({"train_time_s": t_train, "gpu_mem_mb": gpu_mem_mb(),
                 **compute_timing(model, X_te_t)})
    return mets


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all_forward(DATASETS: dict, RESULTS: dict) -> None:
    xgb_base = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                    n_jobs=-1, random_state=SEED, tree_method="hist",
                    device="cuda" if torch.cuda.is_available() else "cpu")

    for ds_name, ds in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
        RESULTS[ds_name] = {}

        print("  [1] XGBoost — forward scalar")
        mets = run_forward_scalar_tree(
            MultiOutputRegressor(xgb.XGBRegressor(**xgb_base)), ds)
        RESULTS[ds_name]["XGBoost"] = {"forward_scalar": mets}
        print(f"    R²={mets['r2_macro']:.4f}  MAE={mets['mae']:.4f}")

        print("  [2] Random Forest — forward scalar")
        mets = run_forward_scalar_tree(
            MultiOutputRegressor(RandomForestRegressor(
                n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED)), ds)
        RESULTS[ds_name]["RandomForest"] = {"forward_scalar": mets}
        print(f"    R²={mets['r2_macro']:.4f}  MAE={mets['mae']:.4f}")

        print("  [3] GP Regressor — forward scalar (subset=2000)")
        gp_idx = ds["idx_train"][:2000]; gp_te = ds["idx_test"]
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        gp = MultiOutputRegressor(GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=0, random_state=SEED))
        t0 = time.time(); gp.fit(ds["geo_sc"][gp_idx], ds["sca_sc"][gp_idx]); t_gp = time.time() - t0
        Y_gp_pred = ds["scaler_scalars"].inverse_transform(gp.predict(ds["geo_sc"][gp_te]))
        mets_gp   = scalar_metrics(Y_gp_pred, ds["sca_raw"][gp_te])
        mets_gp.update({"train_time_s": t_gp})
        RESULTS[ds_name]["GP"] = {"forward_scalar": mets_gp}
        print(f"    R²={mets_gp['r2_macro']:.4f}  MAE={mets_gp['mae']:.4f}")

        print("  [4] MLP — forward scalar")
        mets, _ = run_forward_scalar_nn(MLP, {"hidden": (256,256,256), "dropout": 0.1}, ds_name, ds)
        RESULTS[ds_name]["MLP"] = {"forward_scalar": mets}
        print(f"    R²={mets['r2_macro']:.4f}  MAE={mets['mae']:.4f}")

        print("  [4b] MLP — forward spectrum")
        mets, _ = run_forward_spectrum_nn(MLP, {"hidden": (256,256,256), "dropout": 0.1}, ds_name, ds)
        RESULTS[ds_name]["MLP"]["forward_spectrum"] = mets
        print(f"    MSE={mets['mse']:.6f}  cosine={mets['cosine_sim']:.4f}")

        print("  [5] ResNet1D — forward spectrum")
        mets, _ = run_forward_spectrum_nn(ResNet1D, {"width": 256, "n_blocks": 4}, ds_name, ds)
        RESULTS[ds_name]["ResNet1D"] = {"forward_spectrum": mets}
        print(f"    MSE={mets['mse']:.6f}  cosine={mets['cosine_sim']:.4f}")

        print("  [6] TabTransformer — forward scalar")
        mets, _ = run_forward_scalar_nn(TabularTransformer,
                                         {"d_model": 64, "nhead": 4, "n_layers": 3}, ds_name, ds)
        RESULTS[ds_name]["TabTransformer"] = {"forward_scalar": mets}
        print(f"    R²={mets['r2_macro']:.4f}  MAE={mets['mae']:.4f}")

        print("  [7] DilatedCNN1D — forward spectrum")
        mets, _ = run_forward_spectrum_nn(DilatedCNN1D, {"channels": 128, "seq_len": 8}, ds_name, ds)
        RESULTS[ds_name]["DilatedCNN1D"] = {"forward_spectrum": mets}
        print(f"    MSE={mets['mse']:.6f}  cosine={mets['cosine_sim']:.4f}")

        print("  [8] NeuralODE — forward spectrum")
        mets, _ = run_forward_spectrum_nn(NeuralODE, {"hidden": 128, "n_steps": 5}, ds_name, ds)
        RESULTS[ds_name]["NeuralODE"] = {"forward_spectrum": mets}
        print(f"    MSE={mets['mse']:.6f}  cosine={mets['cosine_sim']:.4f}")

        print("  [10] DeepONet — forward spectrum")
        mets = run_forward_spectrum_deeponet(ds)
        RESULTS[ds_name]["DeepONet"] = {"forward_spectrum": mets}
        print(f"    MSE={mets['mse']:.6f}  cosine={mets['cosine_sim']:.4f}")

        print("  [11] FNO1D — forward spectrum (AMP=False, fp32 FFT)")
        mets = run_forward_spectrum_fno(ds)
        RESULTS[ds_name]["FNO1D"] = {"forward_spectrum": mets}
        print(f"    MSE={mets['mse']:.6f}  cosine={mets['cosine_sim']:.4f}")
