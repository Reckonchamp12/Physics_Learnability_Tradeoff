"""
Inverse-task neural network models (spectrum / scalars → geometry).

Models
------
PINN             — Physics-Informed NN with energy-conservation residual
SpectrumVAE      — Variational Autoencoder + regression head
BayesianNN       — MC-Dropout Bayesian NN with uncertainty quantification
SiameseTriplet   — Siamese network trained with triplet + cycle-consistency loss
cINN             — Conditional Invertible NN (Real NVP) for multi-modal design
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEVICE, EPOCHS, PATIENCE, LR, WD, BATCH_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# PINN
# ─────────────────────────────────────────────────────────────────────────────

class PINN(nn.Module):
    """
    Physics-Informed NN for inverse design (spectrum → geometry).

    Adds an auxiliary energy-conservation residual via a small MLP head
    that estimates T + R + A ≈ 1 from the predicted geometry.

    Args:
        in_dim  : spectrum length
        out_dim : number of geometry parameters
        hidden  : hidden layer widths
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: tuple = (256, 256, 256)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net          = nn.Sequential(*layers)
        self.energy_head  = nn.Sequential(
            nn.Linear(out_dim, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physics_loss(self, geo_pred: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        residual = (self.energy_head(geo_pred) - 1.0) ** 2
        return residual.mean() * alpha


def train_pinn(model: PINN, train_loader, val_loader,
               epochs: int = EPOCHS, patience: int = PATIENCE):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    mse   = nn.MSELoss()
    best_val, best_state, wait = np.inf, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, Yb in train_loader:
            opt.zero_grad()
            geo_pred = model(Xb)
            (mse(geo_pred, Yb) + model.physics_loss(geo_pred)).backward()
            opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                val_losses.append(mse(model(Xb), Yb).item())
        val_loss = float(np.mean(val_losses))
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# VAE + Regressor
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumVAE(nn.Module):
    """
    Variational Autoencoder that maps a spectrum to a latent space and
    then regresses geometry from the posterior mean.

    Args:
        spec_dim   : spectrum length
        geo_dim    : number of geometry parameters
        latent_dim : VAE latent dimension
    """
    def __init__(self, spec_dim: int, geo_dim: int, latent_dim: int = 32):
        super().__init__()
        self.enc     = nn.Sequential(nn.Linear(spec_dim, 256), nn.GELU(),
                                     nn.Linear(256, 128),      nn.GELU())
        self.mu_head = nn.Linear(128, latent_dim)
        self.lv_head = nn.Linear(128, latent_dim)
        self.reg     = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(),
                                     nn.Linear(128, 128),         nn.GELU(),
                                     nn.Linear(128, geo_dim))

    def encode(self, x):
        h = self.enc(x)
        return self.mu_head(h), self.lv_head(h)

    def reparameterize(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.reg(self.reparameterize(mu, lv)), mu, lv

    def predict(self, x):
        mu, _ = self.encode(x)
        return self.reg(mu)


def vae_loss(geo_pred, geo_true, mu, lv, beta: float = 1e-3):
    recon = F.mse_loss(geo_pred, geo_true)
    kld   = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
    return recon + beta * kld


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian NN (MC Dropout)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianNN(nn.Module):
    """
    MC-Dropout Bayesian NN for uncertainty-aware inverse design.

    Args:
        in_dim  : spectrum length
        out_dim : number of geometry parameters
        hidden  : hidden layer widths
        dropout : dropout probability
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: tuple = (256, 256, 256), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 100):
        """Return (mean, std) over n_samples stochastic forward passes."""
        self.train()   # keep dropout active
        with torch.no_grad():
            preds = torch.stack([self.net(x) for _ in range(n_samples)])
        return preds.mean(0), preds.std(0)


# ─────────────────────────────────────────────────────────────────────────────
# Siamese + Triplet Loss
# ─────────────────────────────────────────────────────────────────────────────

class SiameseTriplet(nn.Module):
    """
    Siamese network with a shared embedding space for geometry and spectra,
    trained with triplet loss + cycle-consistency.

    The default forward() path is spectrum → geometry (required by compute_timing).

    Args:
        geo_dim   : geometry feature dimension
        spec_dim  : spectrum length
        embed_dim : shared embedding dimension
    """
    def __init__(self, geo_dim: int, spec_dim: int, embed_dim: int = 128):
        super().__init__()
        self.geo_enc  = nn.Sequential(nn.Linear(geo_dim,  256), nn.GELU(),
                                      nn.Linear(256, embed_dim))
        self.spec_enc = nn.Sequential(nn.Linear(spec_dim, 256), nn.GELU(),
                                      nn.Linear(256, embed_dim))
        self.geo_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(),
                                      nn.Linear(128, geo_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default: spectrum → geometry."""
        return self.geo_head(F.normalize(self.spec_enc(x), dim=-1))

    def forward_geo(self,  g): return F.normalize(self.geo_enc(g),  dim=-1)
    def forward_spec(self, s): return F.normalize(self.spec_enc(s), dim=-1)
    def predict_geo(self,  s): return self.geo_head(self.forward_spec(s))

    def triplet_loss(self, anchor_s, pos_g, neg_g, margin: float = 0.5):
        a  = self.forward_spec(anchor_s)
        p  = self.forward_geo(pos_g)
        n  = self.forward_geo(neg_g)
        return F.relu((a - p).pow(2).sum(1) - (a - n).pow(2).sum(1) + margin).mean()

    def cycle_loss(self, geo, spec):
        return F.mse_loss(self.predict_geo(spec), geo)


def train_siamese(model: SiameseTriplet, Xg_tr, Xs_tr, Xg_va, Xs_va,
                  epochs: int = EPOCHS, patience: int = PATIENCE):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    best_val, best_state, wait = np.inf, None, 0
    t0 = time.time()

    Xg_tr_t = torch.tensor(Xg_tr, dtype=torch.float32, device=DEVICE)
    Xs_tr_t = torch.tensor(Xs_tr, dtype=torch.float32, device=DEVICE)
    Xg_va_t = torch.tensor(Xg_va, dtype=torch.float32, device=DEVICE)
    Xs_va_t = torch.tensor(Xs_va, dtype=torch.float32, device=DEVICE)
    N = len(Xg_tr_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            neg = torch.randperm(len(idx), device=DEVICE)
            g, s = Xg_tr_t[idx], Xs_tr_t[idx]
            opt.zero_grad()
            (model.triplet_loss(s, g, g[neg]) + model.cycle_loss(g, s)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model.predict_geo(Xs_va_t), Xg_va_t).item()
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# cINN (Real NVP)
# ─────────────────────────────────────────────────────────────────────────────

class _AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer that correctly handles odd-dimensional inputs.
    x1 = x[:half1], x2 = x[half1:]  (half2 = dim - half1 handles odd dim).
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.half1 = dim // 2
        self.half2 = dim - self.half1
        self.net_s = nn.Sequential(
            nn.Linear(self.half1 + cond_dim, 128), nn.GELU(),
            nn.Linear(128, self.half2),
        )
        self.net_t = nn.Sequential(
            nn.Linear(self.half1 + cond_dim, 128), nn.GELU(),
            nn.Linear(128, self.half2),
        )

    def forward(self, x, cond, reverse: bool = False):
        x1 = x[:, :self.half1]; x2 = x[:, self.half1:]
        h  = torch.cat([x1, cond], dim=-1)
        s  = torch.tanh(self.net_s(h)); t = self.net_t(h)
        if not reverse:
            y2, log_det = x2 * torch.exp(s) + t, s.sum(1)
        else:
            y2, log_det = (x2 - t) * torch.exp(-s), -s.sum(1)
        return torch.cat([x1, y2], dim=-1), log_det


class cINN(nn.Module):
    """
    Conditional Invertible Neural Network (Real NVP) for multi-modal
    inverse design: spectrum (condition) → geometry (latent).

    Args:
        geo_dim  : number of geometry parameters
        spec_dim : spectrum length (condition)
        n_layers : number of affine coupling layers
    """
    def __init__(self, geo_dim: int, spec_dim: int, n_layers: int = 6):
        super().__init__()
        self.cond_enc = nn.Sequential(
            nn.Linear(spec_dim, 128), nn.GELU(), nn.Linear(128, 64)
        )
        self.layers  = nn.ModuleList(
            [_AffineCouplingLayer(geo_dim, 64) for _ in range(n_layers)]
        )
        self.geo_dim = geo_dim

    def _flip(self, x, layer_idx):
        if layer_idx % 2 == 1:
            return torch.cat([x[:, self.geo_dim // 2:],
                              x[:, :self.geo_dim // 2]], dim=-1)
        return x

    def forward(self, geo, cond_spec):
        cond = self.cond_enc(cond_spec)
        z, log_det_sum = geo, 0.0
        for i, layer in enumerate(self.layers):
            z = self._flip(z, i)
            z, ld = layer(z, cond)
            log_det_sum = log_det_sum + ld
        return z, log_det_sum

    def inverse(self, z, cond_spec):
        cond = self.cond_enc(cond_spec)
        x    = z
        for i, layer in enumerate(reversed(self.layers)):
            ri   = len(self.layers) - 1 - i
            x, _ = layer(x, cond, reverse=True)
            x    = self._flip(x, ri)
        return x

    def nll_loss(self, geo, cond_spec):
        z, log_det = self.forward(geo, cond_spec)
        return (0.5 * z.pow(2).sum(1) - log_det).mean()

    def sample(self, cond_spec, n_samples: int = 10):
        B = cond_spec.shape[0]
        with torch.no_grad():
            samples = [self.inverse(torch.randn(B, self.geo_dim, device=cond_spec.device),
                                    cond_spec)
                       for _ in range(n_samples)]
        return torch.stack(samples, dim=0)   # (n_samples, B, geo_dim)


def train_cinn(model: cINN, Xg_tr, Xs_tr, Xg_va, Xs_va,
               epochs: int = EPOCHS, patience: int = PATIENCE):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    best_val, best_state, wait = np.inf, None, 0
    t0 = time.time()

    Xg_tr_t = torch.tensor(Xg_tr, dtype=torch.float32, device=DEVICE)
    Xs_tr_t = torch.tensor(Xs_tr, dtype=torch.float32, device=DEVICE)
    Xg_va_t = torch.tensor(Xg_va, dtype=torch.float32, device=DEVICE)
    Xs_va_t = torch.tensor(Xs_va, dtype=torch.float32, device=DEVICE)
    N = len(Xg_tr_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            model.nll_loss(Xg_tr_t[idx], Xs_tr_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = model.nll_loss(Xg_va_t, Xs_va_t).item()
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return time.time() - t0
