"""
Generic training loop, inference timing, and parameter counting utilities.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.config import DEVICE, EPOCHS, PATIENCE, LR, WD, USE_AMP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_mem_mb() -> float:
    """Peak GPU memory in MB since last reset_gpu_mem() call."""
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def reset_gpu_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def infer_time_ms(model: nn.Module, X_test_tensor: torch.Tensor,
                  batch1: bool = True, batch1024: bool = True) -> dict:
    """Measure forward-pass latency at batch sizes 1 and 1024."""
    model.eval()
    times = {}
    _sync = lambda: torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    with torch.no_grad():
        if batch1:
            x1 = X_test_tensor[:1]
            _sync(); t0 = time.perf_counter()
            for _ in range(50): model(x1)
            _sync(); times["batch1"] = (time.perf_counter() - t0) / 50 * 1000
        if batch1024:
            x1k = X_test_tensor[:1024] if len(X_test_tensor) >= 1024 else X_test_tensor
            _sync(); t0 = time.perf_counter()
            for _ in range(10): model(x1k)
            _sync(); times["batch1024"] = (time.perf_counter() - t0) / 10 * 1000
    return times


def compute_timing(model: nn.Module, X_test_t: torch.Tensor) -> dict:
    itimes = infer_time_ms(model, X_test_t)
    return {
        "infer_batch1_ms":   itimes.get("batch1"),
        "infer_batch1024_ms": itimes.get("batch1024"),
        "n_params":          count_params(model),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_pytorch(model: nn.Module, train_loader, val_loader,
                  epochs: int = EPOCHS, patience: int = PATIENCE,
                  loss_fn=None, label: str = "",
                  use_amp_override=None) -> float:
    """
    Generic PyTorch training loop with early stopping and LR scheduling.

    Args:
        model             : PyTorch model (already on device).
        train_loader      : training DataLoader.
        val_loader        : validation DataLoader.
        epochs            : maximum training epochs.
        patience          : early-stopping patience (epochs without improvement).
        loss_fn           : callable loss function; defaults to nn.MSELoss().
        label             : string tag for logging (unused internally).
        use_amp_override  : if False, disables mixed precision for this call
                            (required for FNO whose FFT must stay in fp32).

    Returns:
        Elapsed training time in seconds.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    use_amp    = USE_AMP if use_amp_override is None else use_amp_override
    opt        = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched      = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    amp_scaler = GradScaler() if use_amp else None
    best_val, best_state, wait = np.inf, None, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, Yb in train_loader:
            opt.zero_grad()
            if use_amp:
                with autocast():
                    loss = loss_fn(model(Xb), Yb)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(opt); amp_scaler.update()
            else:
                loss = loss_fn(model(Xb), Yb)
                loss.backward(); opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                val_losses.append(loss_fn(model(Xb), Yb).item())
        val_loss = float(np.mean(val_losses))
        sched.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return time.time() - t0
