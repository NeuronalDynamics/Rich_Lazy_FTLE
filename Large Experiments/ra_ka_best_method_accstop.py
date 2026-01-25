# ra_ka_best_method_accstop.py
# --------------------------------------------------------------
# Lean + fast + Phase2/Phase3-compatible training/checkpoint module
#
# Exports (used by phase2/phase3):
#   FC, make_circle, verify_or_train_checkpoint, dataset_to_loader,
#   DEVICE, TRAIN_ACC_TARGET, MAX_EPOCHS, BATCH_SIZE_TRAIN,
#   fmt_float, ckpt_path
# --------------------------------------------------------------

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import math
import random
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler


# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE_TRAIN = 8192

# Accuracy-based early stopping
TRAIN_ACC_TARGET = 0.95
MAX_EPOCHS       = 4000

# Evaluate train accuracy only every N epochs (big speed win)
EVAL_EVERY_EPOCHS = 10
LOG_EVERY_EPOCHS  = 100

# Deterministic init seed base (used across scripts)
SEED_BASE = 0

# Checkpoint dir (must match phase2/phase3)
CHECKPOINT_DIR = "rk_ckpts_v4"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------- GPU knobs --------------------
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# -------------------- Optional AMP training --------------------
# BF16 is fastest + stable on modern GPUs (Blackwell/Ada/Hopper), no GradScaler needed.
USE_AMP_TRAIN = (DEVICE.type == "cuda")
AMP_DTYPE = (
    torch.bfloat16
    if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)

USE_SCALER = USE_AMP_TRAIN and (DEVICE.type == "cuda") and (AMP_DTYPE == torch.float16)

SCALER = None
if USE_SCALER:
    # Avoid the deprecation warning by preferring torch.amp.GradScaler when available.
    try:
        from torch.amp import GradScaler  # PyTorch newer API
        SCALER = GradScaler("cuda")
    except Exception:
        SCALER = torch.cuda.amp.GradScaler()


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _amp_ctx():
    if USE_AMP_TRAIN and DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


def fmt_float(x: float) -> str:
    """Compact filename-safe float: 0.1->0p1, -0.5->m0p5"""
    s = f"{x:.3g}".replace(".", "p")
    return ("m" + s[1:]) if s.startswith("-") else s


def ckpt_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    return os.path.join(
        CHECKPOINT_DIR,
        f"model_N{N}_L{L}_g{fmt_float(gain)}_lr{fmt_float(base_lr)}_seed{seed}.pt",
    )


# ------------------------- DATA -----------------------------------
TOTAL_PTS, TRAIN_SPLIT = 40_000, 0.9


def make_circle(
    total_pts: int = TOTAL_PTS,
    train_split: float = TRAIN_SPLIT,
    seed: Optional[int] = None,
):
    """
    Returns:
      (x_train, y_train), (x_test, y_test)
    where y in {-1,+1} and shape [N,1].
    """
    if seed is not None:
        np.random.seed(seed)

    xy = np.random.uniform(-1.0, 1.0, (total_pts, 2)).astype(np.float32)
    r2 = (xy * xy).sum(axis=1)
    thr = np.median(r2)
    y = ((r2 < thr).astype(np.float32) * 2.0 - 1.0).astype(np.float32)

    idx = np.random.permutation(total_pts)
    tr = int(train_split * total_pts)

    xtr = torch.from_numpy(xy[idx[:tr]])
    ytr = torch.from_numpy(y[idx[:tr]])[:, None]
    xte = torch.from_numpy(xy[idx[tr:]])
    yte = torch.from_numpy(y[idx[tr:]])[:, None]
    return (xtr, ytr), (xte, yte)


# ------------------------- MODEL ----------------------------------
class FC(nn.Module):
    def __init__(self, width: int, depth: int, gain: float = 1.0):
        super().__init__()
        self.depth = depth
        self.gain = gain

        layers = []
        prev = 2
        for _ in range(depth):
            l = nn.Linear(prev, width)
            nn.init.normal_(l.weight, 0.0, gain / math.sqrt(prev))
            nn.init.zeros_(l.bias)
            layers.append(l)
            prev = width
        self.hid = nn.ModuleList(layers)

        self.out = nn.Linear(prev, 1)
        nn.init.normal_(self.out.weight, 0.0, gain / math.sqrt(prev))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, *, hid: bool = False, grad: bool = False):
        for l in self.hid:
            x = torch.tanh(l(x))
        if hid:
            return x
        if grad:
            return self.out(x)          # logits (NTK / jacobians)
        return torch.tanh(self.out(x))  # tanh output (training/classification)


# -------------------- Optimizer (fan-in scaling) --------------------
def per_layer_lr(layer: nn.Linear, base_lr: float) -> float:
    return base_lr / float(layer.weight.size(1))


def make_optim(net: nn.Module, base_lr: float) -> torch.optim.Optimizer:
    # Fewer param groups than splitting weight/bias separately
    groups = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            lr = per_layer_lr(m, base_lr)
            params = [m.weight]
            if m.bias is not None:
                params.append(m.bias)
            groups.append({"params": params, "lr": lr})
    return torch.optim.SGD(groups, momentum=0.0)


# -------------------- DataLoader helper --------------------
def dataset_to_loader(pair, batch_size: int, shuffle: bool, device=DEVICE) -> DataLoader:
    """
    Keeps tensors on DEVICE (works well since inputs are tiny: 2D points).
    We avoid DataLoader overhead in training by slicing dataset.tensors directly.
    """
    x, y = pair
    x = x.to(device, non_blocking=False)
    y = y.to(device, non_blocking=False)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


# -------------------- Fast eval --------------------
@torch.inference_mode()
def _acc_loss_on_tensor(net: FC, X: torch.Tensor, y: torch.Tensor, max_batch: int = 65536) -> Tuple[float, float]:
    net.eval()
    n = X.shape[0]
    correct = torch.zeros((), device=X.device, dtype=torch.int64)
    loss_sum = torch.zeros((), device=X.device, dtype=torch.float32)

    for s in range(0, n, max_batch):
        out = net(X[s:s + max_batch])
        correct += (torch.sign(out) == y[s:s + max_batch]).sum()
        loss_sum += F.mse_loss(out, y[s:s + max_batch], reduction="sum")

    return correct.item() / float(n), loss_sum.item() / float(n)


@torch.inference_mode()
def loader_acc_and_loss(net: FC, loader: DataLoader, _unused_mse: Optional[nn.Module] = None) -> Tuple[float, float]:
    """
    Backward-compatible signature: accepts an optional 3rd arg (ignored).
    This prevents crashes if some older code still calls loader_acc_and_loss(net, loader, mse).
    """
    if isinstance(loader.dataset, TensorDataset):
        X, y = loader.dataset.tensors
        return _acc_loss_on_tensor(net, X, y, max_batch=65536)

    # Generic fallback
    net.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for xb, yb in loader:
        out = net(xb)
        correct += (torch.sign(out) == yb).sum().item()
        total += yb.size(0)
        loss_sum += F.mse_loss(out, yb, reduction="sum").item()
    return correct / float(total), loss_sum / float(total)


# -------------------- Training --------------------
def train_until_acc(
    net: FC,
    train_loader: DataLoader,
    acc_target: float,
    max_epochs: int,
    base_lr: float,
) -> float:
    """
    Trains until train accuracy >= acc_target or max_epochs hit.
    Returns final measured train accuracy.
    """
    opt = make_optim(net, base_lr)

    # Fast path: train directly on TensorDataset tensors (no DataLoader iteration overhead)
    X, y = train_loader.dataset.tensors if isinstance(train_loader.dataset, TensorDataset) else (None, None)
    bs = int(train_loader.batch_size or X.shape[0] if X is not None else 8192)
    do_shuffle = isinstance(getattr(train_loader, "sampler", None), RandomSampler)

    acc = 0.0
    loss_eval = float("nan")

    for ep in range(1, max_epochs + 1):
        net.train()

        if X is not None:
            n = X.shape[0]
            perm = torch.randperm(n, device=X.device) if do_shuffle else None

            for s in range(0, n, bs):
                idx = perm[s:s + bs] if perm is not None else slice(s, s + bs)
                xb = X[idx]
                yb = y[idx]

                opt.zero_grad(set_to_none=True)

                with _amp_ctx():
                    out = net(xb)
                    # compute loss in fp32 for stability (still fast)
                    loss = F.mse_loss(out.float(), yb.float())

                if SCALER is not None:
                    SCALER.scale(loss).backward()
                    SCALER.step(opt)
                    SCALER.update()
                else:
                    loss.backward()
                    opt.step()

        else:
            # Fallback: iterate loader
            for xb, yb in train_loader:
                opt.zero_grad(set_to_none=True)
                with _amp_ctx():
                    out = net(xb)
                    loss = F.mse_loss(out.float(), yb.float())
                if SCALER is not None:
                    SCALER.scale(loss).backward()
                    SCALER.step(opt)
                    SCALER.update()
                else:
                    loss.backward()
                    opt.step()

        # Evaluate periodically
        if ep == 1 or ep % EVAL_EVERY_EPOCHS == 0 or ep == max_epochs:
            acc, loss_eval = loader_acc_and_loss(net, train_loader)
            if ep == 1 or ep % LOG_EVERY_EPOCHS == 0 or ep == max_epochs:
                print(f"[train] epoch={ep:4d}  acc={acc:.3f}  loss={loss_eval:.4f}")
            if acc >= acc_target:
                print(f"[early-stop] hit acc_target={acc_target:.3f} at epoch {ep} (acc={acc:.3f})")
                break

    return float(acc)


def verify_or_train_checkpoint(
    N: int, L: int,
    gain: float, base_lr: float,
    seed: int,
    train_loader: DataLoader,
    acc_target: float,
    max_epochs: int,
    fail_policy: str = "return",   # "return" | "none" | "raise"
) -> Optional[nn.Module]:
    """
    Load-or-train with a HARD gate.
    - Writes checkpoint dict with: state_dict, train_acc, acc_target, max_epochs, failed
    - If fail_policy == "none", returns None when model fails to hit acc_target.
    """
    if fail_policy not in ("return", "none", "raise"):
        raise ValueError("fail_policy must be one of: 'return', 'none', 'raise'")

    path = ckpt_path(N, L, gain, base_lr, seed)
    net = FC(N, L, gain=gain).to(DEVICE)

    # ---------- Try load ----------
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)

        if isinstance(state, dict) and "state_dict" in state:
            net.load_state_dict(state["state_dict"])

            # Fast skip if it was previously marked failed (under >= requested target)
            if bool(state.get("failed", False)) and float(state.get("acc_target", acc_target)) >= acc_target:
                print(f"[ckpt] {os.path.basename(path)} previously marked FAILED (train_acc={state.get('train_acc', 'NA')})")
                net.eval()
                if fail_policy == "none":
                    return None
                if fail_policy == "raise":
                    raise RuntimeError(f"Checkpoint previously failed: {path}")
                return net

            # Fast accept if metadata says good
            if (not bool(state.get("failed", False))) and ("train_acc" in state) and float(state["train_acc"]) >= acc_target:
                print(f"[ckpt] loaded {os.path.basename(path)} (meta train_acc={float(state['train_acc']):.3f} ≥ target)")
                net.eval()
                return net

        else:
            # Old raw state_dict checkpoint
            net.load_state_dict(state)

        # Verify accuracy if metadata missing/insufficient
        acc, _ = loader_acc_and_loss(net, train_loader)
        if acc >= acc_target:
            torch.save({
                "state_dict": net.state_dict(),
                "train_acc": float(acc),
                "acc_target": float(acc_target),
                "max_epochs": int(max_epochs),
                "failed": False,
            }, path)
            print(f"[ckpt] loaded {os.path.basename(path)} (verified acc={acc:.3f} ≥ target)")
            net.eval()
            return net

        print(f"[ckpt] {os.path.basename(path)} acc={acc:.3f} < target → continuing training")

    # ---------- Train ----------
    _seed_all(SEED_BASE + seed)

    _ = train_until_acc(net, train_loader, acc_target, max_epochs, base_lr)
    acc, _ = loader_acc_and_loss(net, train_loader)

    failed = bool(acc < acc_target)
    torch.save({
        "state_dict": net.state_dict(),
        "train_acc": float(acc),
        "acc_target": float(acc_target),
        "max_epochs": int(max_epochs),
        "failed": failed,
    }, path)

    if failed:
        print(f"[ckpt-failed] {os.path.basename(path)} train_acc={acc:.3f} < target={acc_target:.3f}  → SKIPPING")
        net.eval()
        if fail_policy == "none":
            return None
        if fail_policy == "raise":
            raise RuntimeError(f"Training failed to reach acc_target for: {path}")
        return net

    print(f"[ckpt] saved {os.path.basename(path)} (train_acc={acc:.3f} ≥ target)")
    net.eval()
    return net
