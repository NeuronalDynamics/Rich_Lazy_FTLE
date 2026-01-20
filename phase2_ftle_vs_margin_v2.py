# phase2_ftle_vs_margin.py
"""
Phase 2:  FTLE  ↔  Adversarial Margin  (+ Jacobian variance)
RESUME-SAFE VERSION
------------------------------------------------------------

What this version adds vs your current code:
  1) Per-seed caching of margins/ftle/stats to survive power outages.
     - saves partial progress every SAVE_EVERY_POINTS points.
     - skips any (N,L,gain,lr,seed) already finished.
  2) Incremental saving of the full 4D grid after EACH completed cell.
     - allows restarting without losing completed cells.
  3) Final plotting AFTER everything finishes (and plot-only mode).

Files written:
  - FTLE grids: ftle/ftle_N{N}_L{L}_g{g}_lr{lr}_seed{seed}_g{grid}.npy  (already in your pipeline)
  - Per-seed cache: phase2_cache/seedstats_*.npz
  - Grid state: phase2_grid_state.npz
  - Plots: plots/*.png
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import math
import random
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
# USER CONFIG
# ────────────────────────────────────────────────────────────────────
WIDTHS  = [10, 50, 100, 200]
DEPTHS  = [2, 6, 8, 12]
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
CKPT_DIR    = "rk_ckpts_v4"               # must match Phase-1
FTLE_DIR    = "ftle"                      # already used in your setup
CACHE_DIR   = "phase2_cache"              # NEW: per-seed margins/ftle/stats
GRID_STATE  = "phase2_grid_state.npz"     # NEW: aggregated grid arrays (incremental)
PLOT_DIR    = "plots"

# Data caching (IMPORTANT for resume consistency)
# If you change DATA_SEED later, use a new cache filename or delete old caches.
DATA_SEED       = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"

# Save partial seed progress every N points (power-outage safe)
SAVE_EVERY_POINTS = 200

# Control behavior
DO_COMPUTE = True     # set False if you only want to load cached results and plot
DO_PLOT    = True     # set False if you only want to compute and not plot

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
    """
    Returns (train_x, train_y), (test_x, test_y). Cached to disk.
    """
    if os.path.exists(cache_path):
        d = safe_load_npz(cache_path)
        if d is not None:
            xt = torch.tensor(d["xt"], dtype=torch.float32)
            yt = torch.tensor(d["yt"], dtype=torch.float32)
            xe = torch.tensor(d["xe"], dtype=torch.float32)
            ye = torch.tensor(d["ye"], dtype=torch.float32)
            return (xt, yt), (xe, ye)

    # create deterministically
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
# Checkpoint paths + net loading (avoid retraining existing checkpoints)
# ────────────────────────────────────────────────────────────────────
def model_ckpt_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(CKPT_DIR, f"model_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}.pt")

def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int,
                      train_loader):
    """
    Loads checkpoint if present; otherwise trains using Phase-1 helper.
    Avoids re-verifying accuracy on a potentially different dataset.
    """
    path = model_ckpt_path(N, L, gain, base_lr, seed)
    net = FC(N, L, gain=gain).to(device)

    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
        net.eval()
        return net

    # train if missing
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

@torch.no_grad()
def ftle_field(net: FC, depth: int, grid: int = 161, bbox=(-1.2, 1.2)) -> np.ndarray:
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)
    field = np.empty((grid, grid), np.float32)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            x = torch.tensor([xv, yv], requires_grad=True, device=device)
            J = torch.autograd.functional.jacobian(
                lambda z: net(z.unsqueeze(0), hid=True).squeeze(0), x
            )
            sigmax = torch.linalg.svdvals(J).max()
            field[j, i] = (1.0 / depth) * math.log(float(sigmax) + 1e-12)
    return field

def load_ftle_grid(N: int, L: int, gain: float, base_lr: float, seed: int,
                   train_loader,
                   grid: int = 161) -> np.ndarray:
    path = ftle_grid_path(N, L, gain, base_lr, seed, grid)
    if os.path.exists(path):
        arr = safe_load_npy(path)
        if arr is not None:
            return arr
        # corrupted -> recompute
        print(f"[recompute] corrupted FTLE grid: {path}")

    net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
    fld = ftle_field(net, L, grid, bbox=BBOX)
    atomic_save_npy(path, fld)
    return fld

def ftle_lookup(fld: np.ndarray, x: np.ndarray, bbox=(-1.2, 1.2)) -> float:
    gx, gy = fld.shape[1] - 1, fld.shape[0] - 1
    xmin, xmax = bbox
    i = int((x[0] - xmin) / (xmax - xmin) * gx)
    j = int((x[1] - xmin) / (xmax - xmin) * gy)
    i = np.clip(i, 0, gx); j = np.clip(j, 0, gy)
    return float(fld[j, i])

# ────────────────────────────────────────────────────────────────────
# PGD + margin
# ────────────────────────────────────────────────────────────────────
def pgd(net: FC, x: torch.Tensor, y: torch.Tensor, eps: float, step: float, k: int = 20):
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(k):
        out = net(x + delta)
        loss = - (y * out).mean()
        loss.backward()
        delta.data = (delta + step * delta.grad.sign()).clamp(-eps, eps)
        delta.grad.zero_()
    return (x + delta).detach()

def margin(net: FC, x: torch.Tensor, y: torch.Tensor,
           eps_hi: float = 0.30,
           bisection_iters: int = 10,
           pgd_steps: int = 20) -> float:
    lo, hi = 0.0, eps_hi
    for _ in range(bisection_iters):
        mid = 0.5 * (lo + hi)
        adv = pgd(net, x, y, mid, step=mid/10, k=pgd_steps)
        if torch.sign(net(adv)) != y:
            hi = mid
        else:
            lo = mid
    return hi

# ────────────────────────────────────────────────────────────────────
# Per-seed caching (resume + skip finished)
# ────────────────────────────────────────────────────────────────────
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr  = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedstats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_grid{FTLE_GRID}_dseed{DATA_SEED}.npz"
    )

def is_finished_seed_cache(d: Dict[str, np.ndarray]) -> bool:
    if "finished" not in d:
        return False
    return bool(np.array(d["finished"]).item())

def compute_or_resume_seed_stats(N: int, L: int, gain: float, base_lr: float, seed: int,
                                train_loader,
                                X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Computes (or resumes) margins and ftle_vals for one seed and config.
    Saves partial progress every SAVE_EVERY_POINTS points.
    """
    path = seed_cache_path(N, L, gain, base_lr, seed)
    n_test = X_test.shape[0]

    cached = safe_load_npz(path) if os.path.exists(path) else None
    if cached is not None and is_finished_seed_cache(cached):
        return cached

    fld = load_ftle_grid(N, L, gain, base_lr, seed, train_loader=train_loader, grid=FTLE_GRID)

    # initialize/resume arrays
    if cached is not None and "margins" in cached and "ftle_vals" in cached:
        margins   = cached["margins"].astype(np.float32, copy=True)
        ftle_vals = cached["ftle_vals"].astype(np.float32, copy=True)
        if margins.shape[0] != n_test:
            print(f"[warn] cache test-size mismatch; restarting: {os.path.basename(path)}")
            margins   = np.full(n_test, np.nan, np.float32)
            ftle_vals = np.empty(n_test, np.float32)
    else:
        margins   = np.full(n_test, np.nan, np.float32)
        ftle_vals = np.empty(n_test, np.float32)

    # compute ftle_vals once (fast lookup)
    if not np.all(np.isfinite(ftle_vals)):
        for i in range(n_test):
            ftle_vals[i] = ftle_lookup(fld, X_test[i].detach().cpu().numpy(), bbox=BBOX)

    todo = np.where(~np.isfinite(margins))[0]

    if todo.size > 0:
        net = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)
        net.eval()

        t0 = time.time()
        for k, idx in enumerate(todo, start=1):
            xi = X_test[idx:idx+1]
            yi = y_test[idx:idx+1]
            eps_star = margin(
                net, xi, yi,
                eps_hi=EPS_HI,
                bisection_iters=BISECTION_ITERS,
                pgd_steps=PGD_STEPS
            )
            margins[idx] = float(eps_star)

            if (k % SAVE_EVERY_POINTS) == 0 or (k == todo.size):
                atomic_save_npz(
                    path,
                    finished=np.array(False),
                    N=np.array(N), L=np.array(L),
                    gain=np.array(gain, dtype=np.float32),
                    base_lr=np.array(base_lr, dtype=np.float32),
                    seed=np.array(seed),
                    n_test=np.array(n_test),
                    margins=margins,
                    ftle_vals=ftle_vals,
                )
                done = int(np.isfinite(margins).sum())
                dt = time.time() - t0
                print(f"[save-partial] {os.path.basename(path)}  done={done}/{n_test}  dt={dt/60:.1f} min")

        del net
        torch.cuda.empty_cache()

    # final stats
    jac_norms = np.exp(L * ftle_vals).astype(np.float64)     # ||J||_2
    G_lambda = float(np.var(ftle_vals))
    G_J      = float(np.var(jac_norms))

    rho_lambda, p_lambda = spearmanr(ftle_vals, margins)
    rho_J,      p_J      = spearmanr(jac_norms, margins)

    out = dict(
        finished=np.array(True),
        N=np.array(N), L=np.array(L),
        gain=np.array(gain, dtype=np.float32),
        base_lr=np.array(base_lr, dtype=np.float32),
        seed=np.array(seed),
        n_test=np.array(n_test),
        margins=margins,
        ftle_vals=ftle_vals,
        G_lambda=np.array(G_lambda, dtype=np.float64),
        G_J=np.array(G_J, dtype=np.float64),
        rho_lambda_margin=np.array(rho_lambda, dtype=np.float64),
        p_lambda_margin=np.array(p_lambda, dtype=np.float64),
        rho_J_margin=np.array(rho_J, dtype=np.float64),
        p_J_margin=np.array(p_J, dtype=np.float64),
    )
    atomic_save_npz(path, **out)
    return out

# ────────────────────────────────────────────────────────────────────
# Aggregation over seeds
# ────────────────────────────────────────────────────────────────────
def fisher_mean(rhos: List[float]) -> float:
    r = np.array(rhos, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    return float(np.tanh(z.mean()))

def aggregate_config(seed_dicts: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    G_lams = [float(d["G_lambda"]) for d in seed_dicts]
    G_Js   = [float(d["G_J"]) for d in seed_dicts]
    r_lam  = [float(d["rho_lambda_margin"]) for d in seed_dicts]
    r_J    = [float(d["rho_J_margin"]) for d in seed_dicts]
    return dict(
        G_lambda_mean=float(np.mean(G_lams)),
        G_J_mean=float(np.mean(G_Js)),
        rho_lambda_mean=fisher_mean(r_lam),
        rho_J_mean=fisher_mean(r_J),
    )

# ────────────────────────────────────────────────────────────────────
# Grid state save/load (incremental)
# ────────────────────────────────────────────────────────────────────
def save_grid_state(path: str,
                    widths, depths, gains, base_lrs, seeds,
                    G_lambda_map, G_J_map, rho_lambda_map, rho_J_map, done_map):
    atomic_save_npz(
        path,
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

def try_load_grid_state(path: str,
                        widths, depths, gains, base_lrs, seeds):
    if not os.path.exists(path):
        return None
    d = safe_load_npz(path)
    if d is None:
        return None

    # only reuse if axes match (prevents subtle “wrong indexing” bugs)
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
def run_grid_resume(widths, depths, gains, base_lrs, seeds,
                    train_loader, X_test, y_test):
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

                    agg = aggregate_config(seed_stats)

                    G_lambda_map[gi, li, di, wi]   = agg["G_lambda_mean"]
                    G_J_map[gi, li, di, wi]        = agg["G_J_mean"]
                    rho_lambda_map[gi, li, di, wi] = agg["rho_lambda_mean"]
                    rho_J_map[gi, li, di, wi]      = agg["rho_J_mean"]
                    done_map[gi, li, di, wi]       = True

                    done = int(done_map.sum())
                    print(f"[cell-done] ({done}/{total})  Gλ={agg['G_lambda_mean']:.3e}  ρλ={agg['rho_lambda_mean']:.3f}")

                    # IMPORTANT: save after each completed cell
                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        G_lambda_map, G_J_map, rho_lambda_map, rho_J_map,
                        done_map
                    )

    # final save
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
# Plotting
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

    # per (gain, lr) slice
    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            G_slice  = G_lambda_map[gi, li]     # [depth, width]
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
                title=f"rho(λ, margin)   gain={g}  lr={lr}",
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
                title=f"rho(||J||_2, margin)   gain={g}  lr={lr}",
                out_path=os.path.join(PLOT_DIR, f"heatmap_rho_J_g{gstr}_lr{lrstr}.png"),
                vmin=-1, vmax=1
            )

    # average over gain/lr
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
        title="rho(λ, margin) averaged over gain/lr",
        out_path=os.path.join(PLOT_DIR, "heatmap_rho_lambda_avg.png"),
        vmin=-1, vmax=1
    )

    # scatter: rho vs log10 G_lambda for all cells
    Gl_flat = G_lambda_map.reshape(-1)
    rho_flat = rho_lambda_map.reshape(-1)
    mask = np.isfinite(Gl_flat) & np.isfinite(rho_flat)
    if mask.sum() > 0:
        plt.figure(figsize=(5, 4))
        plt.scatter(np.log10(Gl_flat[mask] + 1e-12), rho_flat[mask], s=12)
        plt.xlabel("log10 G_lambda")
        plt.ylabel("rho(λ, margin)")
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

    # 1) Fixed dataset (so cache is consistent across restarts)
    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)

    # 2) Train loader once
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=device)

    # 3) Test tensors once
    X_test = xe.to(device)
    y_test = ye.to(device)

    # 4) Compute/resume grid OR plot-only
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

    # 5) Plot after finished
    if DO_PLOT:
        print("[plot] generating figures from grid_state ...")
        plot_all_slices(grid)
        print(f"[plot] saved to: {PLOT_DIR}/")
