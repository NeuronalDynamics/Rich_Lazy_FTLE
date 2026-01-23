# phase3_ra_ka_grid.py
"""
Phase 3: RA/KA on the SAME (N,L,g,lr) grid as Phase-2 (G_lambda, rho_lambda)
-------------------------------------------------------------------------

Key points:
  - Loads the SAME checkpoints (rk_ckpts_v4/model_N{N}_L{L}_g{g}_lr{lr}_seed{seed}.pt)
    that Phase-2 used. Trains only if checkpoint is missing.
  - Uses the SAME cached dataset mechanism (circle_data_seed{DATA_SEED}.npz) to avoid
    dataset mismatch & accidental retraining.
  - Computes:
      RA = linear CKA between init vs trained last-hidden features
      KA = Frobenius cosine between init vs trained sample-NTK (subset)
  - Resume-safe:
      per-seed cache files + an aggregated grid_state file saved after every completed cell.

Outputs:
  - ra_ka_cache/seedrakastats_*.npz
  - ra_ka_grid_state.npz
  - optional plots in plots_ra_ka/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import math
import random
import time
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- GPU speed knobs ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# ---------------- Import Phase-1 primitives ----------------
from ra_ka_best_method_accstop import (
    FC,
    make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    fmt_float,
    ckpt_path as phase1_ckpt_path,   # uses same naming convention
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
)

# ---------------- Paths ----------------
CKPT_DIR = "rk_ckpts_v4"               # must match phase2
PHASE2_GRID_STATE = "phase2_grid_state.npz"  # to auto-load axes
DATA_SEED = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"

CACHE_DIR  = "ra_ka_cache"
GRID_STATE = "ra_ka_grid_state.npz"
PLOT_DIR   = "plots_ra_ka"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- RA/KA config ----------------
SEED_BASE = 0

# Probe set (use phase2 test split as probe)
BATCH_SIZE_PROBE = 8192

# KA subset size and deterministic subset selection
KA_SUBSET = 64
KA_SUBSET_SEED = 12345

# Resume/caching versions
CACHE_VERSION = 1
GRID_VERSION  = 1

# ---------------- I/O utils (power-outage safe) ----------------
def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)

def safe_load_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None

# ---------------- Dataset caching (match phase2 style) ----------------
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

# ---------------- Model loading: load ckpt fast; train only if missing ----------------
def model_ckpt_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    # Use the exact same naming as phase1_ckpt_path
    return phase1_ckpt_path(N, L, gain, base_lr, seed)

def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader):
    path = model_ckpt_path(N, L, gain, base_lr, seed)
    net = FC(N, L, gain=gain).to(DEVICE)

    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        net.eval()
        return net

    # missing -> train using phase1 helper (same as phase2 behavior)
    print(f"[train-missing] N={N} L={L} g={gain} lr={base_lr} seed={seed}")
    net = verify_or_train_checkpoint(
        N, L,
        gain=gain, base_lr=base_lr,
        seed=seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS
    )
    net.eval()
    return net

# ---------------- Alignment math (fast) ----------------
def frob_norm(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(A, ord="fro")

def frob_cosine(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> float:
    # Frobenius cosine similarity <A,B> / (||A|| ||B||)
    num = (A * B).sum()
    den = frob_norm(A) * frob_norm(B) + eps
    return float((num / den).detach().cpu())

@torch.no_grad()
def linear_cka_from_features(H0: torch.Tensor, HT: torch.Tensor, eps: float = 1e-12) -> float:
    """
    H0, HT: [n_samples, n_features]
    linear CKA between centered features:
      CKA = || H0c^T HTc ||_F^2 / ( ||H0c^T H0c||_F * ||HTc^T HTc||_F )
    This is equivalent to Frobenius alignment of centered Gram matrices
    but avoids building n_samples x n_samples matrices (HUGE speed win).
    """
    H0c = H0 - H0.mean(dim=0, keepdim=True)
    HTc = HT - HT.mean(dim=0, keepdim=True)

    A = H0c.T @ HTc
    B = H0c.T @ H0c
    C = HTc.T @ HTc

    num = (A * A).sum()  # ||A||_F^2
    den = frob_norm(B) * frob_norm(C) + eps
    return float((num / den).detach().cpu())

# ---------------- NTK alignment (KA) ----------------
# Optional: faster per-sample grads via torch.func
try:
    from torch.func import functional_call, vmap, grad
    _HAS_TORCHFUNC = True
except Exception:
    _HAS_TORCHFUNC = False

def grad_matrix_loop(net: FC, xs: torch.Tensor) -> torch.Tensor:
    """
    Slow fallback: builds [Ns, P] gradient matrix by looping samples.
    """
    net.eval()
    params = [p for p in net.parameters() if p.requires_grad]
    rows = []
    for i in range(xs.shape[0]):
        net.zero_grad(set_to_none=True)
        y = net(xs[i:i+1], grad=True)  # logit scalar
        y.sum().backward()
        flat = []
        for p in params:
            g = p.grad
            flat.append(g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
        rows.append(torch.cat(flat))
    return torch.stack(rows, 0)

def grad_matrix_torchfunc(net: FC, xs: torch.Tensor) -> torch.Tensor:
    """
    Faster: vmap over per-sample grad w.r.t. parameters.
    Returns [Ns, P].
    """
    net.eval()
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    names = list(params.keys())

    def f(params, buffers, x):
        # x: [2]
        y = functional_call(net, (params, buffers), (x.unsqueeze(0),), kwargs={"grad": True})
        return y.squeeze()  # scalar

    g = vmap(grad(f), in_dims=(None, None, 0))(params, buffers, xs)  # pytree of [Ns, ...]
    # flatten each parameter block to [Ns, -1] and concatenate -> [Ns, P]
    flats = []
    for name in names:
        flats.append(g[name].reshape(xs.shape[0], -1))
    return torch.cat(flats, dim=1)

def grad_matrix(net: FC, xs: torch.Tensor) -> torch.Tensor:
    if _HAS_TORCHFUNC:
        return grad_matrix_torchfunc(net, xs)
    return grad_matrix_loop(net, xs)

@torch.no_grad()
def ntk_align(net_init: FC, net_trained: FC, X_ka: torch.Tensor) -> float:
    """
    KA = cosine_frob( K_init, K_trained ) where K = G G^T and
    G is per-sample gradient of logit wrt parameters.
    """
    # Need grads -> enable grad temporarily
    with torch.enable_grad():
        G0 = grad_matrix(net_init, X_ka)
        GT = grad_matrix(net_trained, X_ka)

    K0 = G0 @ G0.T
    KT = GT @ GT.T
    return frob_cosine(KT, K0)

# ---------------- Per-seed RA/KA caching ----------------
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedrakastats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_kasub{KA_SUBSET}_dseed{DATA_SEED}.npz"
    )

def cache_ok(d: Optional[Dict[str, np.ndarray]]) -> bool:
    if d is None:
        return False
    if "finished" not in d or "cache_version" not in d:
        return False
    if int(np.array(d["cache_version"]).item()) != CACHE_VERSION:
        return False
    return bool(np.array(d["finished"]).item())

def compute_or_load_seed_ra_ka(
    N: int, L: int, gain: float, base_lr: float, seed: int,
    train_loader,
    X_probe: torch.Tensor,
    X_ka: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Computes RA/KA for one seed+config or loads from cache if done.
    """
    path = seed_cache_path(N, L, gain, base_lr, seed)
    cached = safe_load_npz(path) if os.path.exists(path) else None
    if cache_ok(cached):
        return cached

    # init net (seeded)
    torch.manual_seed(SEED_BASE + seed)
    np.random.seed(SEED_BASE + seed)
    random.seed(SEED_BASE + seed)

    net_init = FC(N, L, gain=gain).to(DEVICE)
    net_init.eval()

    # trained net (load ckpt or train if missing)
    net_tr = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    net_tr.eval()

    # RA (linear CKA on last-hidden)
    with torch.no_grad():
        H0 = net_init(X_probe, hid=True)
        HT = net_tr(X_probe, hid=True)
        ra = linear_cka_from_features(H0, HT)

    # KA (NTK alignment on subset)
    ka = ntk_align(net_init, net_tr, X_ka)

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        finished=np.array(True),
        N=np.array(N), L=np.array(L),
        gain=np.array(gain, np.float32),
        base_lr=np.array(base_lr, np.float32),
        seed=np.array(seed, np.int32),
        RA=np.array(ra, np.float64),
        KA=np.array(ka, np.float64),
    )
    atomic_save_npz(path, **out)

    # cleanup
    del net_init, net_tr
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return out

# ---------------- Grid state save/load ----------------
def save_grid_state(path: str,
                    widths, depths, gains, base_lrs, seeds,
                    RA_map, KA_map, RA_std_map, KA_std_map, done_map):
    atomic_save_npz(
        path,
        grid_version=np.array(GRID_VERSION, np.int32),
        data_seed=np.array(DATA_SEED, np.int32),
        widths=np.array(widths, np.int32),
        depths=np.array(depths, np.int32),
        gains=np.array(gains, np.float32),
        base_lrs=np.array(base_lrs, np.float32),
        seeds=np.array(seeds, np.int32),
        KA_SUBSET=np.array(KA_SUBSET, np.int32),
        RA_map=RA_map.astype(np.float64),
        KA_map=KA_map.astype(np.float64),
        RA_std_map=RA_std_map.astype(np.float64),
        KA_std_map=KA_std_map.astype(np.float64),
        done_map=done_map.astype(np.bool_),
    )

def try_load_grid_state(path: str, widths, depths, gains, base_lrs, seeds):
    if not os.path.exists(path):
        return None
    d = safe_load_npz(path)
    if d is None:
        return None
    gv = int(np.array(d.get("grid_version", 0)).item())
    ds = int(np.array(d.get("data_seed", -1)).item())
    ks = int(np.array(d.get("KA_SUBSET", -999)).item())

    if gv != GRID_VERSION or ds != DATA_SEED or ks != KA_SUBSET:
        print("[ra/ka grid] version/data_seed/KA_SUBSET mismatch -> ignoring old ra_ka_grid_state.npz")
        return None

    if (not np.array_equal(np.array(widths), d.get("widths")) or
        not np.array_equal(np.array(depths), d.get("depths")) or
        not np.allclose(np.array(gains, dtype=np.float32), d.get("gains").astype(np.float32)) or
        not np.allclose(np.array(base_lrs, dtype=np.float32), d.get("base_lrs").astype(np.float32)) or
        not np.array_equal(np.array(seeds), d.get("seeds"))):
        print("[ra/ka grid] axes changed -> ignoring old ra_ka_grid_state.npz")
        return None
    return d

# ---------------- Main runner ----------------
def load_axes_from_phase2_or_defaults():
    # Defaults if phase2_grid_state.npz not present
    widths  = [10, 50, 100, 200]
    depths  = [2, 6, 8, 12]
    gains   = [0.8, 0.9, 1.0, 1.1]
    base_lrs = [0.025, 0.05, 0.075, 0.10]
    seeds   = [0, 1, 2]

    if os.path.exists(PHASE2_GRID_STATE):
        d = safe_load_npz(PHASE2_GRID_STATE)
        if d is not None:
            widths  = d["widths"].astype(int).tolist()
            depths  = d["depths"].astype(int).tolist()
            gains   = d["gains"].astype(float).tolist()
            base_lrs = d["base_lrs"].astype(float).tolist()
            seeds   = d["seeds"].astype(int).tolist()
            print("[axes] loaded axes from phase2_grid_state.npz")
    return widths, depths, gains, base_lrs, seeds

def run_ra_ka_grid(widths, depths, gains, base_lrs, seeds, train_loader, X_probe, X_ka):
    shape = (len(gains), len(base_lrs), len(depths), len(widths))

    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, base_lrs, seeds)
    if loaded is not None:
        RA_map = loaded["RA_map"]
        KA_map = loaded["KA_map"]
        RA_std_map = loaded["RA_std_map"]
        KA_std_map = loaded["KA_std_map"]
        done_map = loaded["done_map"].astype(bool)
        print(f"[ra/ka grid] loaded grid: {done_map.sum()}/{done_map.size} cells done")
    else:
        RA_map = np.full(shape, np.nan, dtype=np.float64)
        KA_map = np.full(shape, np.nan, dtype=np.float64)
        RA_std_map = np.full(shape, np.nan, dtype=np.float64)
        KA_std_map = np.full(shape, np.nan, dtype=np.float64)
        done_map = np.zeros(shape, dtype=bool)

    total = done_map.size

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue

                    print(f"\n[cell] N={N} L={L} g={g} lr={lr}")

                    ra_list = []
                    ka_list = []
                    for sd in seeds:
                        sdat = compute_or_load_seed_ra_ka(
                            N, L, g, lr, sd,
                            train_loader=train_loader,
                            X_probe=X_probe,
                            X_ka=X_ka,
                        )
                        ra_list.append(float(sdat["RA"]))
                        ka_list.append(float(sdat["KA"]))

                    RA_map[gi, li, di, wi] = float(np.mean(ra_list))
                    KA_map[gi, li, di, wi] = float(np.mean(ka_list))
                    RA_std_map[gi, li, di, wi] = float(np.std(ra_list, ddof=0))
                    KA_std_map[gi, li, di, wi] = float(np.std(ka_list, ddof=0))
                    done_map[gi, li, di, wi] = True

                    done = int(done_map.sum())
                    print(f"[cell-done] ({done}/{total})  RA={RA_map[gi,li,di,wi]:.3f}±{RA_std_map[gi,li,di,wi]:.3f}  "
                          f"KA={KA_map[gi,li,di,wi]:.3f}±{KA_std_map[gi,li,di,wi]:.3f}")

                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        RA_map, KA_map, RA_std_map, KA_std_map, done_map
                    )

    save_grid_state(
        GRID_STATE,
        widths, depths, gains, base_lrs, seeds,
        RA_map, KA_map, RA_std_map, KA_std_map, done_map
    )
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs, seeds=seeds,
        RA_map=RA_map, KA_map=KA_map,
        RA_std_map=RA_std_map, KA_std_map=KA_std_map,
        done_map=done_map
    )

# Optional plotting (simple heatmaps per (g,lr))
def plot_heatmap(mat2d, widths, depths, title, out_path, vmin=0, vmax=1):
    plt.figure(figsize=(6, 4))
    M = np.ma.masked_invalid(mat2d)
    im = plt.imshow(M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Width N")
    plt.ylabel("Depth L")
    plt.xticks(range(len(widths)), widths)
    plt.yticks(range(len(depths)), depths)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_ra_ka_slices(grid):
    widths = grid["widths"]
    depths = grid["depths"]
    gains  = grid["gains"]
    lrs    = grid["base_lrs"]
    RA = grid["RA_map"]
    KA = grid["KA_map"]

    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            ra2d = RA[gi, li]  # [depth, width]
            ka2d = KA[gi, li]
            gstr  = fmt_float(float(g))
            lrstr = fmt_float(float(lr))
            plot_heatmap(ra2d, widths, depths,
                         title=f"RA (linear CKA)  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_RA_g{gstr}_lr{lrstr}.png"))
            plot_heatmap(ka2d, widths, depths,
                         title=f"KA (NTK align)  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_KA_g{gstr}_lr{lrstr}.png"))

# ---------------- Entry point ----------------
if __name__ == "__main__":
    torch.set_grad_enabled(True)

    print("[device]", DEVICE)
    if torch.cuda.is_available():
        print("[gpu]", torch.cuda.get_device_name(0))
        print("[torch.func]", _HAS_TORCHFUNC)

    widths, depths, gains, base_lrs, seeds = load_axes_from_phase2_or_defaults()
    print("[grid]", "widths", widths)
    print("[grid]", "depths", depths)
    print("[grid]", "gains", gains)
    print("[grid]", "lrs", base_lrs)
    print("[grid]", "seeds", seeds)

    # Fixed dataset (matches phase2 pattern)
    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)

    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=DEVICE)

    # Probe set (use test split)
    X_probe = xe.to(DEVICE)
    # Deterministic KA subset selection (same subset for all configs/seeds => lower variance)
    gen = torch.Generator(device="cpu").manual_seed(KA_SUBSET_SEED)
    idx = torch.randperm(X_probe.shape[0], generator=gen)[:KA_SUBSET].to(torch.long)
    X_ka = X_probe[idx.to(DEVICE)]

    grid = run_ra_ka_grid(widths, depths, gains, base_lrs, seeds,
                          train_loader=train_loader,
                          X_probe=X_probe,
                          X_ka=X_ka)

    print("[saved]", GRID_STATE)
    plot_ra_ka_slices(grid)
    print("[plots] saved to", PLOT_DIR)
