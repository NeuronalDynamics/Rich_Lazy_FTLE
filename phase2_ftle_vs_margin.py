# phase2_ftle_vs_margin.py
"""
Phase 2:  FTLE  ↔  Adversarial Margin  (+ Jacobian variance)
RESUME-SAFE + GPU-FAST + CACHE-REPAIR VERSION
------------------------------------------------------------

Key change vs your current script (requested):
  ✅ Spearman aggregation matches OLD code:
     concatenate all seeds’ samples -> one big (lambda, margin) -> spearmanr

Why this matters:
  - Per-seed rho averaging (Fisher / mean) is NOT equivalent and can flip sign near 0.

Extra (no speed loss):
  - Computes and prints an additional diagnostic rho computed on UNSATURATED points only
    (margins < EPS_HI), which helps explain sign changes due to censoring at EPS_HI.

Other small correctness/variance fixes:
  - Decision rule for success uses torch.sign(net(...)) exactly (do NOT coerce 0 -> +1).
  - Optional: compute loss in FP32 even under AMP (negligible overhead, slightly stabilizes PGD).

Files written:
  - FTLE grids: ftle/ftle_N{N}_L{L}_g{g}_lr{lr}_seed{seed}_g{grid}.npy
  - Per-seed cache: phase2_cache/seedstats_*.npz
  - Grid state: phase2_grid_state.npz
  - Plots: plots/*.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import math
import random
import time
import contextlib
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

# ────────────────────────────────────────────────────────────────────
# Import Phase-1 objects
# ────────────────────────────────────────────────────────────────────
from ra_ka_best_method_accstop import (
    FC, make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    DEVICE, TRAIN_ACC_TARGET, MAX_EPOCHS, BATCH_SIZE_TRAIN,
    fmt_float,
)

device = DEVICE

# ────────────────────────────────────────────────────────────────────
# GPU performance knobs (safe)
# ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# ────────────────────────────────────────────────────────────────────
# USER CONFIG
# ────────────────────────────────────────────────────────────────────
WIDTHS   = [10, 50, 100, 200]
DEPTHS   = [2, 6, 8, 12]
GAINS    = [0.8, 0.9, 1.0, 1.1]
BASE_LRS = [0.025, 0.05, 0.075, 0.10]
SEEDS    = [0, 1, 2]

# Attack / evaluation
EPS_HI          = 0.30
PGD_STEPS       = 20
BISECTION_ITERS = 10

# FTLE caching
FTLE_GRID = 161
BBOX      = (-1.2, 1.2)

# Resume / caching
CKPT_DIR    = "rk_ckpts_v4"
FTLE_DIR    = "ftle"
CACHE_DIR   = "phase2_cache"
GRID_STATE  = "phase2_grid_state.npz"
PLOT_DIR    = "plots"

# Data caching (IMPORTANT for resume consistency)
DATA_SEED       = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"

# Save partial seed progress every N points (power-outage safe)
SAVE_EVERY_POINTS = 200

# Control behavior
DO_COMPUTE = True
DO_PLOT    = True

# ────────────────────────────────────────────────────────────────────
# SPEED KNOBS (tuned for a big GPU like RTX PRO 6000)
# ────────────────────────────────────────────────────────────────────
MARGIN_BATCH = 8192   # test set ~4000 => one batch
USE_AMP      = True   # BF16 autocast for speed
AMP_DTYPE    = torch.bfloat16
USE_COMPILE  = False  # optional torch.compile
LOSS_FP32    = True   # compute PGD loss in FP32 even under AMP (tiny overhead, stabilizes)

# ────────────────────────────────────────────────────────────────────
# CACHE VERSIONING
# ────────────────────────────────────────────────────────────────────
CACHE_VERSION = 2
GRID_VERSION  = 3   # bumped so you re-aggregate rho with the new pooled method

# ────────────────────────────────────────────────────────────────────
# NUMERICAL SAFETY
# ────────────────────────────────────────────────────────────────────
FTLE_ABS_MAX_OK = 100.0
LOG_EXP_CLIP    = 700.0  # prevents exp overflow in float64

def autocast_ctx():
    if USE_AMP and (device.type == "cuda"):
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return contextlib.nullcontext()

def sanitize_lambda(arr: np.ndarray) -> np.ndarray:
    lam = arr.astype(np.float64, copy=False)
    lam = np.array(lam, dtype=np.float64, copy=True)
    lam[~np.isfinite(lam)] = np.nan
    return lam

# OLD behavior: pooled spearman across all samples (no per-seed averaging)
def spearman_rho_only(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    # rank with tie handling = average ranks (matches scipy spearmanr’s ranking behavior)
    rx = rankdata(x[m], method="average")
    ry = rankdata(y[m], method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)

# ────────────────────────────────────────────────────────────────────
# Directories
# ────────────────────────────────────────────────────────────────────
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FTLE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# Utilities: atomic save/load (power-outage safe)
# ────────────────────────────────────────────────────────────────────
def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)

def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)

def safe_load_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None

def safe_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None

# ────────────────────────────────────────────────────────────────────
# Dataset caching (so resume uses the same test points)
# ────────────────────────────────────────────────────────────────────
def load_or_make_circle_data(cache_path: str, seed: int):
    if os.path.exists(cache_path):
        d = safe_load_npz(cache_path)
        if d is not None:
            xt = torch.tensor(d["xt"], dtype=torch.float32)
            yt = torch.tensor(d["yt"], dtype=torch.float32)
            xe = torch.tensor(d["xe"], dtype=torch.float32)
            ye = torch.tensor(d["ye"], dtype=torch.float32)
            return (xt, yt), (xe, ye)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    (xt, yt), (xe, ye) = make_circle()

    atomic_save_npz(
        cache_path,
        xt=xt.cpu().numpy().astype(np.float32),
        yt=yt.cpu().numpy().astype(np.float32),
        xe=xe.cpu().numpy().astype(np.float32),
        ye=ye.cpu().numpy().astype(np.float32),
    )
    return (xt, yt), (xe, ye)

# ────────────────────────────────────────────────────────────────────
# Checkpoint paths + net loading
# ────────────────────────────────────────────────────────────────────
def model_ckpt_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(CKPT_DIR, f"model_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}.pt")

def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader):
    path = model_ckpt_path(N, L, gain, base_lr, seed)
    net = FC(N, L, gain=gain).to(device)

    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        net.eval()
        return net

    print(f"[train-missing] N={N} L={L} gain={gain} lr={base_lr} seed={seed}")
    net = verify_or_train_checkpoint(
        N, L, gain, base_lr, seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS
    )
    net.eval()
    return net

# ────────────────────────────────────────────────────────────────────
# FTLE grid (cached)
# ────────────────────────────────────────────────────────────────────
def ftle_grid_path(N: int, L: int, gain: float, base_lr: float, seed: int, grid: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(FTLE_DIR, f"ftle_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_g{grid}.npy")

def ftle_grid_looks_corrupt(arr: np.ndarray, grid: int) -> bool:
    if arr is None:
        return True
    if arr.shape != (grid, grid):
        return True
    if np.isinf(arr).any():
        return True
    m = np.nanmax(np.abs(arr))
    if np.isfinite(m) and m > 1e3:
        return True
    return False

# Optional fast FTLE via torch.func.jvp
try:
    from torch.func import jvp
    _HAS_JVP = True
except Exception:
    _HAS_JVP = False

def ftle_field_jvp(net: FC, depth: int, grid: int = 161, bbox=(-1.2, 1.2)) -> np.ndarray:
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)

    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def hidden(z):
        return net(z, hid=True)

    lam_out = torch.empty((pts.shape[0],), device=device, dtype=torch.float32)

    bs = pts.shape[0]  # 161^2 = 25921 -> fine
    start = 0
    while start < pts.shape[0]:
        xb = pts[start:start + bs]

        v1 = torch.zeros_like(xb); v1[:, 0] = 1.0
        v2 = torch.zeros_like(xb); v2[:, 1] = 1.0

        _, j1 = jvp(hidden, (xb,), (v1,))
        _, j2 = jvp(hidden, (xb,), (v2,))

        a = (j1 * j1).sum(dim=1)
        b = (j2 * j2).sum(dim=1)
        c = (j1 * j2).sum(dim=1)

        disc = torch.sqrt(torch.clamp((a - b) * (a - b) + 4.0 * c * c, min=0.0))
        eigmax = 0.5 * ((a + b) + disc)
        sigmax = torch.sqrt(torch.clamp(eigmax, min=0.0))

        lam = (1.0 / depth) * torch.log(sigmax + 1e-12)
        lam = torch.where(torch.isfinite(lam), lam, torch.full_like(lam, float("nan")))
        lam_out[start:start + xb.shape[0]] = lam.to(torch.float32)

        start += xb.shape[0]

    return lam_out.reshape(grid, grid).detach().cpu().numpy().astype(np.float32)

def ftle_field_slow_jacobian(net: FC, depth: int, grid: int = 161, bbox=(-1.2, 1.2)) -> np.ndarray:
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)
    field = np.full((grid, grid), np.nan, dtype=np.float32)
    net.eval()

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            x = torch.tensor([xv, yv], requires_grad=True, device=device)
            with torch.enable_grad():
                J = torch.autograd.functional.jacobian(
                    lambda z: net(z.unsqueeze(0), hid=True).squeeze(0), x,
                    create_graph=False
                )
            if not torch.isfinite(J).all():
                continue
            sigmax = torch.linalg.svdvals(J).max()
            if (not torch.isfinite(sigmax)) or (sigmax <= 0):
                continue
            lam = (1.0 / depth) * torch.log(sigmax + 1e-12)
            if not torch.isfinite(lam):
                continue
            field[j, i] = float(lam)
    return field

def ftle_field(net: FC, depth: int, grid: int = 161, bbox=(-1.2, 1.2)) -> np.ndarray:
    if _HAS_JVP:
        return ftle_field_jvp(net, depth, grid, bbox)
    return ftle_field_slow_jacobian(net, depth, grid, bbox)

def load_ftle_grid(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader, grid: int = 161) -> np.ndarray:
    path = ftle_grid_path(N, L, gain, base_lr, seed, grid)

    if os.path.exists(path):
        arr = safe_load_npy(path)
        if arr is not None and (not ftle_grid_looks_corrupt(arr, grid)):
            return arr
        print(f"[recompute-ftle] {os.path.basename(path)} looks corrupt -> recomputing")
        try:
            os.remove(path)
        except OSError:
            pass

    net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    fld = ftle_field(net, L, grid, bbox=BBOX)
    atomic_save_npy(path, fld)
    return fld

def ftle_vals_from_grid(fld: np.ndarray, X_test: torch.Tensor, bbox=(-1.2, 1.2)) -> np.ndarray:
    Xcpu = X_test.detach().cpu().numpy()
    gx = fld.shape[1] - 1
    gy = fld.shape[0] - 1
    xmin, xmax = bbox

    ii = ((Xcpu[:, 0] - xmin) / (xmax - xmin) * gx).astype(np.int64)
    jj = ((Xcpu[:, 1] - xmin) / (xmax - xmin) * gy).astype(np.int64)

    np.clip(ii, 0, gx, out=ii)
    np.clip(jj, 0, gy, out=jj)

    return fld[jj, ii].astype(np.float32, copy=False)

# ────────────────────────────────────────────────────────────────────
# Batched PGD + batched margin (GPU-fast)
# ────────────────────────────────────────────────────────────────────
def pgd_batch(net: FC, X: torch.Tensor, y: torch.Tensor, eps: torch.Tensor, k: int = 20) -> torch.Tensor:
    B = X.shape[0]
    eps2 = eps.view(B, 1)
    delta = torch.zeros_like(X)

    for _ in range(k):
        delta.requires_grad_(True)

        with autocast_ctx():
            out = net(X + delta)  # tanh output (matches old code)
            if LOSS_FP32:
                loss = - (y.float() * out.float()).sum()
            else:
                loss = - (y * out).sum()

        grad = torch.autograd.grad(loss, delta, create_graph=False, retain_graph=False)[0]
        step = eps2 / 10.0
        delta = (delta + step * grad.sign()).detach()
        delta = torch.max(torch.min(delta, eps2), -eps2)

    return (X + delta).detach()

def margin_batch(net: FC, X: torch.Tensor, y: torch.Tensor,
                 eps_hi: float = 0.30, bisection_iters: int = 10, pgd_steps: int = 20) -> torch.Tensor:
    """
    Returns eps*: [B]
    IMPORTANT: decision rule matches old code:
      success if torch.sign(net(adv)) != y
    (do NOT coerce sign==0 to +1)
    """
    B = X.shape[0]
    lo = torch.zeros((B,), device=X.device, dtype=X.dtype)
    hi = torch.full((B,), eps_hi, device=X.device, dtype=X.dtype)

    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        adv = pgd_batch(net, X, y, eps=mid, k=pgd_steps)

        with torch.no_grad():
            with autocast_ctx():
                pred = torch.sign(net(adv))  # keep 0 as 0 (old behavior)
            success = (pred != y).view(-1)

        hi = torch.where(success, mid, hi)
        lo = torch.where(success, lo, mid)

    return hi

# ────────────────────────────────────────────────────────────────────
# Per-seed caching
# ────────────────────────────────────────────────────────────────────
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedstats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_grid{FTLE_GRID}_dseed{DATA_SEED}.npz"
    )

def cache_version_of(d: Dict[str, np.ndarray]) -> int:
    if d is None or "cache_version" not in d:
        return 0
    return int(np.array(d["cache_version"]).item())

def is_finished_seed_cache(d: Dict[str, np.ndarray]) -> bool:
    if d is None or "finished" not in d:
        return False
    finished = bool(np.array(d["finished"]).item())
    return finished and (cache_version_of(d) == CACHE_VERSION)

def compute_or_resume_seed_stats(N: int, L: int, gain: float, base_lr: float, seed: int,
                                train_loader, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    - Reuses cached margins if present (big time saver).
    - Recomputes ftle_vals from FTLE grid (cheap, avoids old np.empty bugs).
    """
    path = seed_cache_path(N, L, gain, base_lr, seed)
    n_test = X_test.shape[0]

    cached = safe_load_npz(path) if os.path.exists(path) else None

    # Load FTLE grid and compute ftle_vals always (vectorized)
    fld = load_ftle_grid(N, L, gain, base_lr, seed, train_loader=train_loader, grid=FTLE_GRID)
    ftle_vals = ftle_vals_from_grid(fld, X_test, bbox=BBOX)

    absmax = np.nanmax(np.abs(ftle_vals))
    if np.isfinite(absmax) and absmax > FTLE_ABS_MAX_OK:
        print(f"[warn] absurd FTLE magnitude (|λ|max={absmax:.2e}) for N={N} L={L} g={gain} lr={base_lr} seed={seed}.")
        print("       Deleting FTLE grid and recomputing once.")
        try:
            os.remove(ftle_grid_path(N, L, gain, base_lr, seed, FTLE_GRID))
        except OSError:
            pass
        fld = load_ftle_grid(N, L, gain, base_lr, seed, train_loader=train_loader, grid=FTLE_GRID)
        ftle_vals = ftle_vals_from_grid(fld, X_test, bbox=BBOX)

    # Load margins if present; else NaNs
    if cached is not None and "margins" in cached and cached["margins"].shape[0] == n_test:
        margins = cached["margins"].astype(np.float32, copy=True)
    else:
        margins = np.full(n_test, np.nan, dtype=np.float32)

    todo = np.where(~np.isfinite(margins))[0]

    # Compute missing margins (GPU-fast)
    if todo.size > 0:
        net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
        net.eval()

        if USE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
            net = torch.compile(net, mode="reduce-overhead")

        for p in net.parameters():
            p.requires_grad_(False)

        t0 = time.time()
        processed = 0
        last_save = 0

        for start in range(0, todo.size, MARGIN_BATCH):
            idx_np = todo[start:start + MARGIN_BATCH]
            idx_t  = torch.as_tensor(idx_np, device=X_test.device, dtype=torch.long)

            Xb = X_test[idx_t]
            yb = y_test[idx_t]

            eps_star = margin_batch(
                net, Xb, yb,
                eps_hi=EPS_HI,
                bisection_iters=BISECTION_ITERS,
                pgd_steps=PGD_STEPS
            )

            margins[idx_np] = eps_star.detach().float().cpu().numpy()

            processed += idx_np.size
            if (processed - last_save) >= SAVE_EVERY_POINTS or (processed == todo.size):
                last_save = processed
                atomic_save_npz(
                    path,
                    cache_version=np.array(CACHE_VERSION, dtype=np.int32),
                    finished=np.array(False),
                    N=np.array(N), L=np.array(L),
                    gain=np.array(gain, dtype=np.float32),
                    base_lr=np.array(base_lr, dtype=np.float32),
                    seed=np.array(seed, dtype=np.int32),
                    n_test=np.array(n_test, dtype=np.int32),
                    margins=margins,
                    ftle_vals=ftle_vals,
                )
                done = int(np.isfinite(margins).sum())
                dt = time.time() - t0
                print(f"[save-partial] {os.path.basename(path)}  done={done}/{n_test}  dt={dt/60:.1f} min")

        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Debug print
    finite_m = np.isfinite(margins)
    sat = float(np.mean(margins[finite_m] >= (EPS_HI - 1e-6))) if finite_m.any() else np.nan
    mmin = float(np.min(margins[finite_m])) if finite_m.any() else np.nan
    mmed = float(np.median(margins[finite_m])) if finite_m.any() else np.nan
    mmax = float(np.max(margins[finite_m])) if finite_m.any() else np.nan

    finite_l = np.isfinite(ftle_vals)
    lmin = float(np.min(ftle_vals[finite_l])) if finite_l.any() else np.nan
    lmed = float(np.median(ftle_vals[finite_l])) if finite_l.any() else np.nan
    lmax = float(np.max(ftle_vals[finite_l])) if finite_l.any() else np.nan
    lstd = float(np.std(ftle_vals[finite_l])) if finite_l.any() else np.nan

    print(
        f"[dbg] N={N} L={L} g={gain} lr={base_lr} seed={seed} "
        f"margin(min/med/max)={mmin:.4f}/{mmed:.4f}/{mmax:.4f}  sat@EPS_HI={sat:.3f}  "
        f"lambda(min/med/max/std)={lmin:.4f}/{lmed:.4f}/{lmax:.4f}/{lstd:.4f}"
    )

    # Per-seed stats (kept for debugging; aggregation uses pooled Spearman)
    lam = sanitize_lambda(ftle_vals)
    G_lambda = float(np.nanvar(lam))
    log_j = np.clip(L * lam, -LOG_EXP_CLIP, LOG_EXP_CLIP)
    jac_norms = np.exp(log_j)
    G_J = float(np.nanvar(jac_norms))

    # Spearman per seed (not used for grid rho anymore)
    rho_lambda_seed = spearman_rho_only(lam, margins.astype(np.float64))
    rho_J_seed      = spearman_rho_only(jac_norms, margins.astype(np.float64))

    out = dict(
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        finished=np.array(True),
        N=np.array(N), L=np.array(L),
        gain=np.array(gain, dtype=np.float32),
        base_lr=np.array(base_lr, dtype=np.float32),
        seed=np.array(seed, dtype=np.int32),
        n_test=np.array(n_test, dtype=np.int32),
        margins=margins.astype(np.float32),
        ftle_vals=ftle_vals.astype(np.float32),
        G_lambda=np.array(G_lambda, dtype=np.float64),
        G_J=np.array(G_J, dtype=np.float64),
        rho_lambda_margin=np.array(rho_lambda_seed, dtype=np.float64),
        rho_J_margin=np.array(rho_J_seed, dtype=np.float64),
    )
    atomic_save_npz(path, **out)
    return out

# ────────────────────────────────────────────────────────────────────
# Aggregation over seeds (OLD behavior: pooled samples)
# ────────────────────────────────────────────────────────────────────
def aggregate_config_pooled(seed_dicts: List[Dict[str, np.ndarray]], L: int) -> Dict[str, float]:
    """
    Old code behavior:
      concatenate all seeds’ (lambda, margin) samples, then compute ONE Spearman rho.
    """
    # Means of variances (keep your previous grid stats semantics)
    G_lams = np.array([float(d["G_lambda"]) for d in seed_dicts], dtype=np.float64)
    G_Js   = np.array([float(d["G_J"]) for d in seed_dicts], dtype=np.float64)
    G_lambda_mean = float(np.nanmean(G_lams))
    G_J_mean      = float(np.nanmean(G_Js))

    # Pool samples across seeds
    lam_all = np.concatenate([sanitize_lambda(d["ftle_vals"]) for d in seed_dicts], axis=0)
    m_all   = np.concatenate([d["margins"].astype(np.float64, copy=False) for d in seed_dicts], axis=0)

    rho_lambda = spearman_rho_only(lam_all, m_all)

    log_j_all = np.clip(L * lam_all, -LOG_EXP_CLIP, LOG_EXP_CLIP)
    jac_all   = np.exp(log_j_all)
    rho_J     = spearman_rho_only(jac_all, m_all)

    # Diagnostics to help interpret sign flips near 0:
    # (1) how many margins are saturated (censored) at EPS_HI
    sat_mask = np.isfinite(m_all) & (m_all >= (EPS_HI - 1e-6))
    sat_frac = float(sat_mask.mean()) if np.isfinite(m_all).any() else float("nan")

    # (2) rho using only UNSATURATED points (often more stable in lazy regime)
    unsat_mask = np.isfinite(m_all) & (m_all < (EPS_HI - 1e-6))
    rho_lambda_unsat = spearman_rho_only(lam_all[unsat_mask], m_all[unsat_mask]) if unsat_mask.sum() >= 3 else float("nan")

    return dict(
        G_lambda_mean=G_lambda_mean,
        G_J_mean=G_J_mean,
        rho_lambda_mean=rho_lambda,
        rho_J_mean=rho_J,
        sat_frac=sat_frac,
        rho_lambda_unsat=rho_lambda_unsat,
    )

# ────────────────────────────────────────────────────────────────────
# Grid state save/load (incremental)
# ────────────────────────────────────────────────────────────────────
def save_grid_state(path: str,
                    widths, depths, gains, base_lrs, seeds,
                    G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map):
    atomic_save_npz(
        path,
        grid_version=np.array(GRID_VERSION, dtype=np.int32),
        data_seed=np.array(DATA_SEED, dtype=np.int32),
        widths=np.array(widths, dtype=np.int32),
        depths=np.array(depths, dtype=np.int32),
        gains=np.array(gains, dtype=np.float32),
        base_lrs=np.array(base_lrs, dtype=np.float32),
        seeds=np.array(seeds, dtype=np.int32),
        G_lambda_map=G_lambda_map.astype(np.float64),
        G_J_map=G_J_map.astype(np.float64),
        rho_lambda_map=rho_lambda_map.astype(np.float64),
        rho_J_map=rho_J_map.astype(np.float64),
        done_map=done_map.astype(np.bool_),
    )

def try_load_grid_state(path: str, widths, depths, gains, base_lrs, seeds):
    if not os.path.exists(path):
        return None
    d = safe_load_npz(path)
    if d is None:
        return None

    gv = int(np.array(d.get("grid_version", 0)).item()) if "grid_version" in d else 0
    ds = int(np.array(d.get("data_seed", -1)).item()) if "data_seed" in d else -1
    if gv != GRID_VERSION or ds != DATA_SEED:
        print("[grid-state] version/data_seed mismatch; ignoring existing phase2_grid_state.npz")
        return None

    if (not np.array_equal(np.array(widths), d.get("widths")) or
        not np.array_equal(np.array(depths), d.get("depths")) or
        not np.allclose(np.array(gains, dtype=np.float32), d.get("gains").astype(np.float32)) or
        not np.allclose(np.array(base_lrs, dtype=np.float32), d.get("base_lrs").astype(np.float32)) or
        not np.array_equal(np.array(seeds), d.get("seeds"))):
        print("[grid-state] axes changed; ignoring existing phase2_grid_state.npz")
        return None

    return d

# ────────────────────────────────────────────────────────────────────
# Main grid runner (resume-safe)
# ────────────────────────────────────────────────────────────────────
def run_grid_resume(widths, depths, gains, base_lrs, seeds, train_loader, X_test, y_test):
    shape = (len(gains), len(base_lrs), len(depths), len(widths))

    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, base_lrs, seeds)
    if loaded is not None:
        G_lambda_map   = loaded["G_lambda_map"]
        G_J_map        = loaded["G_J_map"]
        rho_lambda_map = loaded["rho_lambda_map"]
        rho_J_map      = loaded["rho_J_map"]
        done_map       = loaded["done_map"].astype(bool)
        print(f"[grid-state] loaded existing grid with {done_map.sum()}/{done_map.size} cells complete")
    else:
        G_lambda_map   = np.full(shape, np.nan, dtype=np.float64)
        G_J_map        = np.full(shape, np.nan, dtype=np.float64)
        rho_lambda_map = np.full(shape, np.nan, dtype=np.float64)
        rho_J_map      = np.full(shape, np.nan, dtype=np.float64)
        done_map       = np.zeros(shape, dtype=bool)

    total = done_map.size

    for gi, gain in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue

                    print(f"\n[cell] N={N} L={L} gain={gain} lr={lr}")

                    seed_stats = []
                    for sd in seeds:
                        stats_sd = compute_or_resume_seed_stats(
                            N, L, gain, lr, sd,
                            train_loader=train_loader,
                            X_test=X_test, y_test=y_test
                        )
                        seed_stats.append(stats_sd)

                    # ✅ OLD behavior: pooled Spearman across all seeds’ samples
                    agg = aggregate_config_pooled(seed_stats, L=L)

                    G_lambda_map[gi, li, di, wi]   = agg["G_lambda_mean"]
                    G_J_map[gi, li, di, wi]        = agg["G_J_mean"]
                    rho_lambda_map[gi, li, di, wi] = agg["rho_lambda_mean"]
                    rho_J_map[gi, li, di, wi]      = agg["rho_J_mean"]
                    done_map[gi, li, di, wi]       = True

                    done = int(done_map.sum())
                    print(
                        f"[cell-done] ({done}/{total})  "
                        f"Gλ={agg['G_lambda_mean']:.3e}  "
                        f"ρλ(pooled)={agg['rho_lambda_mean']:.3f}  "
                        f"ρλ(unsat)={agg['rho_lambda_unsat']:.3f}  "
                        f"sat@EPS_HI={agg['sat_frac']:.3f}"
                    )

                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map,
                        done_map
                    )

    save_grid_state(
        GRID_STATE,
        widths, depths, gains, base_lrs, seeds,
        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map,
        done_map
    )
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs, seeds=seeds,
        G_lambda_map=G_lambda_map, G_J_map=G_J_map,
        rho_lambda_map=rho_lambda_map, rho_J_map=rho_J_map,
        done_map=done_map
    )

# ────────────────────────────────────────────────────────────────────
# Plotting (unchanged)
# ────────────────────────────────────────────────────────────────────
def plot_heatmap(mat2d: np.ndarray,
                 widths: List[int], depths: List[int],
                 title: str, out_path: str,
                 vmin=None, vmax=None,
                 log10: bool = False):
    plt.figure(figsize=(6, 4))
    M = mat2d.copy()
    if log10:
        M = np.log10(M + 1e-12)
    M_masked = np.ma.masked_invalid(M)
    im = plt.imshow(M_masked, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Width N")
    plt.ylabel("Depth L")
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_all_slices(grid: Dict):
    widths = grid["widths"]
    depths = grid["depths"]
    gains = grid["gains"]
    base_lrs = grid["base_lrs"]

    G_lambda_map = grid["G_lambda_map"]
    rho_lambda_map = grid["rho_lambda_map"]
    G_J_map = grid["G_J_map"]
    rho_J_map = grid["rho_J_map"]

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            G_slice  = G_lambda_map[gi, li]
            R_slice  = rho_lambda_map[gi, li]
            GJ_slice = G_J_map[gi, li]
            RJ_slice = rho_J_map[gi, li]

            gstr  = fmt_float(float(g))
            lrstr = fmt_float(float(lr))

            plot_heatmap(
                G_slice, widths, depths,
                title=f"log10 G_lambda (Var[λ])   gain={g}  lr={lr}",
                out_path=os.path.join(PLOT_DIR, f"heatmap_log10_Glambda_g{gstr}_lr{lrstr}.png"),
                log10=True
            )
            plot_heatmap(
                R_slice, widths, depths,
                title=f"rho(λ, margin) [POOLED]   gain={g}  lr={lr}",
                out_path=os.path.join(PLOT_DIR, f"heatmap_rho_lambda_g{gstr}_lr{lrstr}.png"),
                vmin=-1, vmax=1
            )
            plot_heatmap(
                GJ_slice, widths, depths,
                title=f"log10 G_J (Var[||J||_2])   gain={g}  lr={lr}",
                out_path=os.path.join(PLOT_DIR, f"heatmap_log10_GJ_g{gstr}_lr{lrstr}.png"),
                log10=True
            )
            plot_heatmap(
                RJ_slice, widths, depths,
                title=f"rho(||J||_2, margin) [POOLED]   gain={g}  lr={lr}",
                out_path=os.path.join(PLOT_DIR, f"heatmap_rho_J_g{gstr}_lr{lrstr}.png"),
                vmin=-1, vmax=1
            )

    Gl_avg  = np.nanmean(G_lambda_map, axis=(0, 1))
    rho_avg = np.nanmean(rho_lambda_map, axis=(0, 1))

    plot_heatmap(
        Gl_avg, widths, depths,
        title="log10 G_lambda averaged over gain/lr",
        out_path=os.path.join(PLOT_DIR, "heatmap_log10_Glambda_avg.png"),
        log10=True
    )
    plot_heatmap(
        rho_avg, widths, depths,
        title="rho(λ, margin) [POOLED] averaged over gain/lr",
        out_path=os.path.join(PLOT_DIR, "heatmap_rho_lambda_avg.png"),
        vmin=-1, vmax=1
    )

    Gl_flat = G_lambda_map.reshape(-1)
    rho_flat = rho_lambda_map.reshape(-1)
    mask = np.isfinite(Gl_flat) & np.isfinite(rho_flat)
    if mask.sum() > 0:
        plt.figure(figsize=(5, 4))
        plt.scatter(np.log10(Gl_flat[mask] + 1e-12), rho_flat[mask], s=12)
        plt.xlabel("log10 G_lambda")
        plt.ylabel("rho(λ, margin) [POOLED]")
        plt.title("All configs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "scatter_rho_vs_log10_Glambda.png"), dpi=220)
        plt.close()

# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.set_grad_enabled(True)

    print("[device]", device)
    if torch.cuda.is_available():
        print("[gpu]", torch.cuda.get_device_name(0))

    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)

    X_test = xe.to(device)
    y_test = ye.to(device)

    if DO_COMPUTE:
        grid = run_grid_resume(
            WIDTHS, DEPTHS, GAINS, BASE_LRS, SEEDS,
            train_loader=train_loader,
            X_test=X_test, y_test=y_test
        )
    else:
        d = safe_load_npz(GRID_STATE)
        if d is None:
            raise RuntimeError(f"No grid state found at {GRID_STATE}. Run with DO_COMPUTE=True first.")
        grid = dict(
            widths=d["widths"].tolist(),
            depths=d["depths"].tolist(),
            gains=d["gains"].tolist(),
            base_lrs=d["base_lrs"].tolist(),
            seeds=d["seeds"].tolist(),
            G_lambda_map=d["G_lambda_map"],
            G_J_map=d["G_J_map"],
            rho_lambda_map=d["rho_lambda_map"],
            rho_J_map=d["rho_J_map"],
            done_map=d["done_map"],
        )

    if DO_PLOT:
        print("[plot] generating figures from grid_state ...")
        plot_all_slices(grid)
        print(f"[plot] saved to: {PLOT_DIR}/")
