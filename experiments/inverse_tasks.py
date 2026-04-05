"""
Inverse task runners: scalars → geometry, spectrum → geometry.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

from src.config import DEVICE, SEED, BATCH_SIZE, EPOCHS, PATIENCE, LR, WD
from src.data import make_loader
from src.metrics import inverse_metrics, cycle_consistency_error
from src.trainer import (train_pytorch, reset_gpu_mem, gpu_mem_mb,
                          compute_timing, count_params)
from src.models.forward_models import MLP, ResNet1D, TabularTransformer, DilatedCNN1D
from src.models.inverse_models import (PINN, train_pinn, SpectrumVAE, vae_loss,
                                        BayesianNN, SiameseTriplet, train_siamese,
                                        cINN, train_cinn)
import copy


# ─────────────────────────────────────────────────────────────────────────────
# Shared runners
# ─────────────────────────────────────────────────────────────────────────────

def run_inverse_scalar_tree(model, ds: dict) -> dict:
    tr, te = ds["idx_train"], ds["idx_test"]
    X_tr, X_te = ds["sca_raw"][tr], ds["sca_raw"][te]
    Y_tr, Y_te = ds["geo_raw"][tr], ds["geo_raw"][te]
    t0 = time.time(); model.fit(X_tr, Y_tr); t_train = time.time() - t0
    Y_pred = model.predict(X_te)
    mets   = inverse_metrics(Y_pred, Y_te)

    t0 = time.perf_counter()
    for _ in range(50): model.predict(X_te[:1])
    tb1 = (time.perf_counter() - t0) / 50 * 1000

    x1k = X_te[:1024] if len(X_te) >= 1024 else X_te
    t0 = time.perf_counter()
    for _ in range(10): model.predict(x1k)
    tb1k = (time.perf_counter() - t0) / 10 * 1000

    mets.update({"train_time_s": t_train,
                 "infer_batch1_ms": tb1, "infer_batch1024_ms": tb1k})
    return mets


def _run_inverse_nn(model, X_tr, X_va, X_te, Y_tr, Y_va, Y_te,
                    ds, device=DEVICE) -> tuple:
    reset_gpu_mem()
    t_train = train_pytorch(model, make_loader(X_tr, Y_tr),
                              make_loader(X_va, Y_va, shuffle=False))
    model.eval()
    with torch.no_grad():
        X_te_t    = torch.tensor(X_te, dtype=torch.float32, device=device)
        Y_pred_sc = model(X_te_t).cpu().numpy()
    mets = inverse_metrics(Y_pred_sc, Y_te, ds["scaler_geo"])
    mets.update({"train_time_s": t_train, "gpu_mem_mb": gpu_mem_mb(),
                 **compute_timing(model, X_te_t)})
    return mets, model, X_te_t, Y_pred_sc


def run_inverse_scalar_nn(model_cls, model_kwargs: dict, ds: dict) -> tuple:
    tr, va, te = ds["idx_train"], ds["idx_val"], ds["idx_test"]
    X_tr, X_va, X_te = ds["sca_sc"][tr], ds["sca_sc"][va], ds["sca_sc"][te]
    Y_tr, Y_va, Y_te = ds["geo_sc"][tr], ds["geo_sc"][va], ds["geo_sc"][te]
    in_d, out_d = X_tr.shape[1], Y_tr.shape[1]
    model = model_cls(in_d, out_d, **model_kwargs).to(DEVICE)
    return _run_inverse_nn(model, X_tr, X_va, X_te, Y_tr, Y_va, Y_te, ds)


def run_inverse_spectrum_nn(model_cls, model_kwargs: dict, ds: dict) -> tuple:
    tr, va, te = ds["idx_train"], ds["idx_val"], ds["idx_test"]
    X_tr, X_va, X_te = ds["T_sc"][tr], ds["T_sc"][va], ds["T_sc"][te]
    Y_tr, Y_va, Y_te = ds["geo_sc"][tr], ds["geo_sc"][va], ds["geo_sc"][te]
    in_d, out_d = X_tr.shape[1], Y_tr.shape[1]
    model = model_cls(in_d, out_d, **model_kwargs).to(DEVICE)
    return _run_inverse_nn(model, X_tr, X_va, X_te, Y_tr, Y_va, Y_te, ds)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_all_inverse(DATASETS: dict, RESULTS: dict) -> None:
    xgb_base = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                    n_jobs=-1, random_state=SEED, tree_method="hist",
                    device="cuda" if torch.cuda.is_available() else "cpu")

    for ds_name, ds in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {ds_name} — Inverse Tasks\n{'='*60}")
        n_geo = ds["geo_sc"].shape[1]
        n_wl  = ds["n_wavelengths"]
        tr, va, te = ds["idx_train"], ds["idx_val"], ds["idx_test"]

        print("  [1] XGBoost — inverse scalar")
        mets = run_inverse_scalar_tree(
            MultiOutputRegressor(xgb.XGBRegressor(**xgb_base)), ds)
        RESULTS[ds_name]["XGBoost"]["inverse_scalar"] = mets
        print(f"    success@5%={mets['success_5pct']:.4f}  MAE={mets['mae']:.4f}")

        print("  [2] RandomForest — inverse scalar")
        mets = run_inverse_scalar_tree(
            MultiOutputRegressor(RandomForestRegressor(
                n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED)), ds)
        RESULTS[ds_name]["RandomForest"]["inverse_scalar"] = mets
        print(f"    success@5%={mets['success_5pct']:.4f}  MAE={mets['mae']:.4f}")

        print("  [4] MLP — inverse scalar")
        mets, _ = run_inverse_scalar_nn(MLP, {"hidden": (256,256,256), "dropout": 0.1}, ds)
        RESULTS[ds_name]["MLP"]["inverse_scalar"] = mets
        print(f"    success@5%={mets['success_5pct']:.4f}  MAE={mets['mae']:.4f}")

        print("  [6] TabTransformer — inverse scalar")
        mets, _ = run_inverse_scalar_nn(TabularTransformer,
                                         {"d_model": 64, "nhead": 4, "n_layers": 3}, ds)
        RESULTS[ds_name]["TabTransformer"]["inverse_scalar"] = mets
        print(f"    success@5%={mets['success_5pct']:.4f}  MAE={mets['mae']:.4f}")

        print("  [5] ResNet1D — inverse spectrum")
        mets, _ = run_inverse_spectrum_nn(ResNet1D, {"width": 256, "n_blocks": 4}, ds)
        RESULTS[ds_name]["ResNet1D"]["inverse_spectrum"] = mets
        print(f"    success@5%={mets['success_5pct']:.4f}  MAE={mets['mae']:.4f}")

        print("  [7] DilatedCNN1D — inverse spectrum")
        mets, _ = run_inverse_spectrum_nn(DilatedCNN1D, {"channels": 128, "seq_len": 8}, ds)
        RESULTS[ds_name]["DilatedCNN1D"]["inverse_spectrum"] = mets
        print(f"    success@5%={mets['success_5pct']:.4f}  MAE={mets['mae']:.4f}")

        # ── PINN ──────────────────────────────────────────────────────────
        print("  [9] PINN — inverse geometry")
        pinn_model  = PINN(n_wl, n_geo).to(DEVICE)
        train_ldr_p = make_loader(ds["T_sc"][tr], ds["geo_sc"][tr])
        val_ldr_p   = make_loader(ds["T_sc"][va], ds["geo_sc"][va], shuffle=False)
        reset_gpu_mem()
        t_pinn = train_pinn(pinn_model, train_ldr_p, val_ldr_p)
        pinn_model.eval()
        X_pinn_te = torch.tensor(ds["T_sc"][te], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pinn_pred = pinn_model(X_pinn_te).cpu().numpy()
        mets_pinn = inverse_metrics(pinn_pred, ds["geo_sc"][te], ds["scaler_geo"])
        mets_pinn.update({"train_time_s": t_pinn, "gpu_mem_mb": gpu_mem_mb(),
                          **compute_timing(pinn_model, X_pinn_te)})
        RESULTS[ds_name]["PINN"] = {"inverse_geometry": mets_pinn}
        print(f"    success@5%={mets_pinn['success_5pct']:.4f}  MAE={mets_pinn['mae']:.4f}")

        # ── VAE ───────────────────────────────────────────────────────────
        print("  [12] VAE+Regressor — inverse geometry")
        import torch.optim as optim
        vae_model = SpectrumVAE(n_wl, n_geo, latent_dim=32).to(DEVICE)
        opt_vae   = optim.AdamW(vae_model.parameters(), lr=LR, weight_decay=WD)
        sched_vae = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_vae, patience=2, factor=0.5)
        Xv_tr = torch.tensor(ds["T_sc"][tr],   dtype=torch.float32, device=DEVICE)
        Yv_tr = torch.tensor(ds["geo_sc"][tr], dtype=torch.float32, device=DEVICE)
        Xv_va = torch.tensor(ds["T_sc"][va],   dtype=torch.float32, device=DEVICE)
        Yv_va = torch.tensor(ds["geo_sc"][va], dtype=torch.float32, device=DEVICE)
        best_v, best_st_v, wait_v = np.inf, None, 0
        reset_gpu_mem(); t0_v = time.time()
        for epoch in range(1, EPOCHS + 1):
            vae_model.train()
            perm = torch.randperm(len(Xv_tr), device=DEVICE)
            for i in range(0, len(Xv_tr), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                opt_vae.zero_grad()
                gp, mu, lv = vae_model(Xv_tr[idx])
                vae_loss(gp, Yv_tr[idx], mu, lv).backward()
                opt_vae.step()
            vae_model.eval()
            with torch.no_grad():
                gp_v, mu_v, lv_v = vae_model(Xv_va)
                vl = vae_loss(gp_v, Yv_va, mu_v, lv_v).item()
            sched_vae.step(vl)
            if vl < best_v:
                best_v, best_st_v, wait_v = vl, copy.deepcopy(vae_model.state_dict()), 0
            else:
                wait_v += 1
                if wait_v >= PATIENCE: break
        vae_model.load_state_dict(best_st_v)
        t_vae = time.time() - t0_v
        vae_model.eval()
        X_vae_te = torch.tensor(ds["T_sc"][te], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            vae_pred = vae_model.predict(X_vae_te).cpu().numpy()
        mets_vae = inverse_metrics(vae_pred, ds["geo_sc"][te], ds["scaler_geo"])
        mets_vae.update({"train_time_s": t_vae, "gpu_mem_mb": gpu_mem_mb(),
                         **compute_timing(vae_model, X_vae_te)})
        RESULTS[ds_name]["VAE"] = {"inverse_geometry": mets_vae}
        print(f"    success@5%={mets_vae['success_5pct']:.4f}  MAE={mets_vae['mae']:.4f}")

        # ── BayesianNN ────────────────────────────────────────────────────
        print("  [13] BayesianNN — inverse geometry")
        bnn = BayesianNN(n_wl, n_geo).to(DEVICE)
        reset_gpu_mem()
        t_bnn = train_pytorch(bnn, make_loader(ds["T_sc"][tr], ds["geo_sc"][tr]),
                               make_loader(ds["T_sc"][va], ds["geo_sc"][va], shuffle=False))
        X_bnn_te = torch.tensor(ds["T_sc"][te], dtype=torch.float32, device=DEVICE)
        bnn_mean, bnn_std = bnn.mc_predict(X_bnn_te, n_samples=100)
        bnn_pred  = bnn_mean.cpu().numpy()
        bnn_unc   = bnn_std.cpu().numpy()
        mets_bnn  = inverse_metrics(bnn_pred, ds["geo_sc"][te], ds["scaler_geo"])
        geo_te_o  = ds["scaler_geo"].inverse_transform(ds["geo_sc"][te])
        geo_pr_o  = ds["scaler_geo"].inverse_transform(bnn_pred)
        std_o     = bnn_unc * ds["scaler_geo"].scale_
        lo = geo_pr_o - 1.645 * std_o; hi = geo_pr_o + 1.645 * std_o
        mets_bnn["coverage_90"]    = float(((geo_te_o >= lo) & (geo_te_o <= hi)).mean())
        mets_bnn["mean_calib_err"] = float(np.abs(geo_te_o - geo_pr_o).mean() / (std_o.mean() + 1e-8))
        mets_bnn.update({"train_time_s": t_bnn, "gpu_mem_mb": gpu_mem_mb(),
                         "n_params": count_params(bnn)})
        RESULTS[ds_name]["BayesianNN"] = {"inverse_geometry": mets_bnn}
        print(f"    success@5%={mets_bnn['success_5pct']:.4f}  coverage90={mets_bnn['coverage_90']:.4f}")

        # ── Siamese ───────────────────────────────────────────────────────
        print("  [14] Siamese+Triplet — inverse geometry")
        siam = SiameseTriplet(n_geo, n_wl, embed_dim=128).to(DEVICE)
        reset_gpu_mem()
        t_siam = train_siamese(siam,
                                ds["geo_sc"][tr], ds["T_sc"][tr],
                                ds["geo_sc"][va], ds["T_sc"][va])
        siam.eval()
        X_siam_te = torch.tensor(ds["T_sc"][te], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            siam_pred = siam.predict_geo(X_siam_te).cpu().numpy()
        mets_siam = inverse_metrics(siam_pred, ds["geo_sc"][te], ds["scaler_geo"])
        mets_siam["cycle_consistency_err"] = float(
            cycle_consistency_error(siam_pred, ds["geo_sc"][te], ds["scaler_geo"]))
        mets_siam.update({"train_time_s": t_siam, "gpu_mem_mb": gpu_mem_mb(),
                           "n_params": count_params(siam),
                           **compute_timing(siam, X_siam_te)})
        RESULTS[ds_name]["Siamese"] = {"inverse_geometry": mets_siam}
        print(f"    success@5%={mets_siam['success_5pct']:.4f}  cycle={mets_siam['cycle_consistency_err']:.4f}")

        # ── cINN ──────────────────────────────────────────────────────────
        print("  [15] cINN — inverse geometry")
        cinn = cINN(n_geo, n_wl, n_layers=6).to(DEVICE)
        reset_gpu_mem()
        t_cinn = train_cinn(cinn,
                             ds["geo_sc"][tr], ds["T_sc"][tr],
                             ds["geo_sc"][va], ds["T_sc"][va])
        cinn.eval()
        X_cinn_te = torch.tensor(ds["T_sc"][te], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            z_zero    = torch.zeros(len(X_cinn_te), n_geo, device=DEVICE)
            cinn_pred = cinn.inverse(z_zero, X_cinn_te).cpu().numpy()
        mets_cinn = inverse_metrics(cinn_pred, ds["geo_sc"][te], ds["scaler_geo"])

        samples   = cinn.sample(X_cinn_te[:500], n_samples=10)
        geo_std   = samples.std(0).cpu().numpy()
        mets_cinn["multimodal_rate"] = float((geo_std.mean(1) > 0.1).mean())

        samp_all   = cinn.sample(X_cinn_te, n_samples=100)
        lo_c       = samp_all.quantile(0.05, dim=0).cpu().numpy()
        hi_c       = samp_all.quantile(0.95, dim=0).cpu().numpy()
        mets_cinn["coverage_90"] = float(
            ((ds["geo_sc"][te] >= lo_c) & (ds["geo_sc"][te] <= hi_c)).mean())
        mets_cinn.update({"train_time_s": t_cinn, "gpu_mem_mb": gpu_mem_mb(),
                           "n_params": count_params(cinn)})
        RESULTS[ds_name]["cINN"] = {"inverse_geometry": mets_cinn}
        print(f"    success@5%={mets_cinn['success_5pct']:.4f}  "
              f"modal_rate={mets_cinn['multimodal_rate']:.4f}  coverage90={mets_cinn['coverage_90']:.4f}")
