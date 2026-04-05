"""
Forward-task neural network models.

All models follow the (in_dim, out_dim, **kwargs) signature to enable
unified training loops.

Models
------
MLP              — Multi-layer perceptron with LayerNorm + GELU + Dropout
ResNet1D         — Residual MLP with skip connections
TabularTransformer — Feature-tokenised Transformer
DilatedCNN1D     — Geometry → short sequence, dilated convolutions, global pool
NeuralODE        — Neural ODE with Euler fallback when torchdiffeq is absent
DeepONet         — Deep operator network (branch + trunk)
FNO1D            — Fourier Neural Operator (1-D, fp32-only FFT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Feed-forward MLP with LayerNorm, GELU activation, and Dropout.

    Args:
        in_dim  : number of input features
        out_dim : number of output features
        hidden  : tuple of hidden layer widths
        dropout : dropout probability (applied after each activation)
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: tuple = (256, 256, 256), dropout: float = 0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet1D
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + x)


class ResNet1D(nn.Module):
    """
    Residual MLP with additive skip connections.

    Args:
        in_dim   : input feature dimension
        out_dim  : output dimension
        width    : hidden layer width (constant across all blocks)
        n_blocks : number of residual blocks
    """
    def __init__(self, in_dim: int, out_dim: int,
                 width: int = 256, n_blocks: int = 4):
        super().__init__()
        self.stem   = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([_ResBlock(width) for _ in range(n_blocks)])
        self.head   = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Tabular Transformer
# ─────────────────────────────────────────────────────────────────────────────

class TabularTransformer(nn.Module):
    """
    Transformer that treats each scalar input feature as a separate token.

    Args:
        in_dim   : number of input scalar features
        out_dim  : output dimension
        d_model  : token embedding dimension
        nhead    : number of attention heads
        n_layers : number of Transformer encoder layers
        dropout  : dropout probability
    """
    def __init__(self, in_dim: int, out_dim: int,
                 d_model: int = 64, nhead: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed   = nn.Linear(1, d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool    = nn.Linear(d_model, 1)
        self.head    = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x.unsqueeze(-1))   # (B, F, d_model)
        tokens = self.encoder(tokens)
        pooled = self.pool(tokens).squeeze(-1)  # (B, F)
        return self.head(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Dilated CNN (1-D)
# ─────────────────────────────────────────────────────────────────────────────

class DilatedCNN1D(nn.Module):
    """
    Projects input geometry to a short sequence, applies 1-D dilated
    convolutions at rates {1, 2, 4}, then global-average-pools.

    Args:
        in_dim   : input feature dimension
        out_dim  : output dimension (spectrum length)
        channels : convolutional channel width
        seq_len  : length of the synthesised intermediate sequence
    """
    def __init__(self, in_dim: int, out_dim: int,
                 channels: int = 128, seq_len: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.proj  = nn.Linear(in_dim, channels * seq_len)
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(channels, channels, 3, dilation=4, padding=4)
        self.norm  = nn.BatchNorm1d(channels)
        self.head  = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = F.gelu(self.proj(x)).view(B, -1, self.seq_len)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        h = self.norm(h).mean(-1)   # global average pool
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Neural ODE
# ─────────────────────────────────────────────────────────────────────────────

class _ODEFunc(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)
        )

    def forward(self, t, y):          # noqa: ARG002  (t unused for autonomous ODE)
        return self.net(y)


class NeuralODE(nn.Module):
    """
    Latent Neural ODE: geometry → hidden state → integrate ODE → spectrum.

    Falls back to a fixed-step Euler integrator when torchdiffeq is not installed.

    Args:
        in_dim  : input feature dimension
        out_dim : output dimension
        hidden  : ODE hidden-state dimension
        n_steps : number of integration steps
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: int = 128, n_steps: int = 5):
        super().__init__()
        self.encoder  = nn.Linear(in_dim, hidden)
        self.ode_func = _ODEFunc(hidden)
        self.decoder  = nn.Linear(hidden, out_dim)
        self.n_steps  = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.encoder(x))
        if TORCHDIFFEQ_AVAILABLE:
            t_span = torch.linspace(0, 1, self.n_steps, device=x.device)
            h = odeint(self.ode_func, h, t_span, method="euler")[-1]
        else:
            dt = 1.0 / self.n_steps
            for _ in range(self.n_steps):
                h = h + dt * self.ode_func(None, h)
        return self.decoder(h)


# ─────────────────────────────────────────────────────────────────────────────
# DeepONet
# ─────────────────────────────────────────────────────────────────────────────

class DeepONet(nn.Module):
    """
    Deep Operator Network (Lu et al., 2021).

    Branch net : geometry  → p-dimensional coefficient vector.
    Trunk  net : normalised wavelength index → p-dimensional basis functions.
    Output     : branch · trunk^T + bias  →  (B, L).

    Args:
        geo_dim       : geometry feature dimension
        n_wavelengths : spectrum length (number of output points)
        p             : latent operator dimension
    """
    def __init__(self, geo_dim: int, n_wavelengths: int, p: int = 128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(geo_dim, 256), nn.GELU(),
            nn.Linear(256, 256),     nn.GELU(),
            nn.Linear(256, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, p),
        )
        self.bias = nn.Parameter(torch.zeros(n_wavelengths))
        self.n_wl = n_wavelengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(x)                                               # (B, p)
        wl_idx     = torch.linspace(0, 1, self.n_wl, device=x.device).unsqueeze(1)
        trunk_out  = self.trunk(wl_idx)                                            # (L, p)
        return torch.mm(branch_out, trunk_out.T) + self.bias                      # (B, L)


# ─────────────────────────────────────────────────────────────────────────────
# Fourier Neural Operator (1-D)
# ─────────────────────────────────────────────────────────────────────────────

class _FNOLayer(nn.Module):
    """
    Real-valued 1-D FNO spectral mixing layer (kept in fp32).

    cuFFT half-precision requires power-of-two signal lengths, which
    grating-coupler wavelength grids typically violate. All tensors are
    explicitly cast to fp32 inside this layer and in FNO1D.forward().
    """
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        self.R = nn.Parameter(
            torch.randn(width, width, modes, dtype=torch.cfloat) * 0.02
        )
        self.W = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x      = x.float()
        B, L, C = x.shape
        x_ft   = torch.fft.rfft(x, dim=1)
        modes  = min(self.modes, x_ft.shape[1])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :modes, :] = torch.einsum(
            "bmi,iom->bmo",
            x_ft[:, :modes, :],
            self.R[:, :, :modes],
        )
        x_sp = torch.fft.irfft(out_ft, n=L, dim=1)
        return F.gelu(x_sp + self.W(x))


class FNO1D(nn.Module):
    """
    1-D Fourier Neural Operator for geometry → spectrum prediction.

    AMP must be disabled for this model (use use_amp_override=False in the
    trainer); the FNOLayer.forward() also casts to fp32 as a safety net.

    Args:
        geo_dim       : geometry feature dimension
        n_wavelengths : spectrum length
        width         : channel width inside FNO layers
        modes         : number of retained Fourier modes
        n_layers      : number of FNO layers
    """
    def __init__(self, geo_dim: int, n_wavelengths: int,
                 width: int = 64, modes: int = 16, n_layers: int = 4):
        super().__init__()
        self.n_wl   = n_wavelengths
        self.lift   = nn.Linear(geo_dim + 1, width)
        self.layers = nn.ModuleList(
            [_FNOLayer(width, modes) for _ in range(n_layers)]
        )
        self.proj   = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.float()
        B   = x.size(0)
        pos = torch.linspace(0, 1, self.n_wl, device=x.device) \
                   .unsqueeze(0).expand(B, -1)
        x_ex = x.unsqueeze(1).expand(-1, self.n_wl, -1)
        h    = torch.cat([x_ex, pos.unsqueeze(-1)], dim=-1)
        h    = self.lift(h)
        for layer in self.layers:
            h = layer(h)
        return self.proj(h).squeeze(-1)
