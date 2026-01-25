import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import random
from typing import Dict, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from ra_ka_best_method_accstop import (
    FC,
    make_circle,
    verify_or_train_checkpoint,
    dataset_to_loader,
    ckpt_path,
    fmt_float,
    TRAIN_ACC_TARGET,
    MAX_EPOCHS,
    BATCH_SIZE_TRAIN,
)

# ---------------- GPU knobs ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

# ---------------- Paths / cache ----------------
PHASE2_GRID_STATE = "phase2_grid_state.npz"   # to auto-load axes
DATA_SEED = 0
DATA_CACHE_FILE = f"circle_data_seed{DATA_SEED}.npz"

CACHE_DIR  = "ra_ka_cache"
GRID_STATE = "ra_ka_grid_state.npz"
PLOT_DIR   = "plots_ra_ka"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- RA/KA config ----------------
SEED_BASE = 0

# Probe set: set PROBE_SUBSET (e.g. 2048) if you want faster RA
PROBE_SUBSET = None

# KA subset
KA_SUBSET = 64
KA_SUBSET_SEED = 12345

# Keep your existing values so you don't invalidate old caches/state:
CACHE_VERSION = 1
GRID_VERSION  = 1

# Optional: only compute cells that phase2 finished
PHASE2_DONE_ONLY = False


# ---------------- I/O utils ----------------
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


# ---------------- Dataset caching ----------------
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


# ---------------- Model loading ----------------
def load_or_train_net(N: int, L: int, gain: float, base_lr: float, seed: int, train_loader) -> Optional[FC]:
    net = verify_or_train_checkpoint(
        N, L, gain, base_lr, seed,
        train_loader=train_loader,
        acc_target=TRAIN_ACC_TARGET,
        max_epochs=MAX_EPOCHS,
        fail_policy="none",   # <---- IMPORTANT
    )
    if net is None:
        print(f"[skip-model] N={N} L={L} g={gain} lr={base_lr} seed={seed} failed to reach acc_target={TRAIN_ACC_TARGET:.3f}")
        return None
    net.eval()
    return net


# ---------------- Alignment math ----------------
def frob_cosine(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> float:
    num = (A * B).sum()
    den = torch.linalg.norm(A) * torch.linalg.norm(B) + eps
    return float((num / den).detach().cpu())


@torch.no_grad()
def linear_cka_features(H0: torch.Tensor, HT: torch.Tensor, eps: float = 1e-12) -> float:
    H0c = H0 - H0.mean(dim=0, keepdim=True)
    HTc = HT - HT.mean(dim=0, keepdim=True)

    A = H0c.T @ HTc
    B = H0c.T @ H0c
    C = HTc.T @ HTc

    num = (A * A).sum()
    den = torch.linalg.norm(B) * torch.linalg.norm(C) + eps
    return float((num / den).detach().cpu())


# ---------------- NTK alignment (KA) ----------------
try:
    from torch.func import functional_call, vmap, grad
    _HAS_TORCHFUNC = True
except Exception:
    _HAS_TORCHFUNC = False


def grad_matrix(net: FC, xs: torch.Tensor) -> torch.Tensor:
    net.eval()

    if _HAS_TORCHFUNC:
        params = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        names = list(params.keys())

        def f(p, b, x):
            y = functional_call(net, (p, b), (x.unsqueeze(0),), kwargs={"grad": True})
            return y.squeeze()

        g = vmap(grad(f), in_dims=(None, None, 0))(params, buffers, xs)
        flats = [g[name].reshape(xs.shape[0], -1) for name in names]
        return torch.cat(flats, dim=1)

    # fallback loop
    rows = []
    params_list = [p for p in net.parameters() if p.requires_grad]
    for i in range(xs.shape[0]):
        net.zero_grad(set_to_none=True)
        y = net(xs[i:i + 1], grad=True)
        y.sum().backward()
        flat = [p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for p in params_list]
        rows.append(torch.cat(flat))
    return torch.stack(rows, 0)


@torch.no_grad()
def ntk_align(net_init: FC, net_trained: FC, X_ka: torch.Tensor) -> float:
    with torch.enable_grad():
        G0 = grad_matrix(net_init, X_ka)
        GT = grad_matrix(net_trained, X_ka)
    K0 = G0 @ G0.T
    KT = GT @ GT.T
    return frob_cosine(KT, K0)


# ---------------- Per-seed caching ----------------
def seed_cache_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    gstr = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    return os.path.join(
        CACHE_DIR,
        f"seedrakastats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_kasub{KA_SUBSET}_dseed{DATA_SEED}.npz",
    )


def cache_ok(d: Optional[Dict[str, np.ndarray]]) -> bool:
    if d is None:
        return False
    if not bool(np.array(d.get("finished", False)).item()):
        return False
    if int(np.array(d.get("cache_version", 0)).item()) != CACHE_VERSION:
        return False
    return True


def compute_or_load_seed_ra_ka(
    N: int, L: int, gain: float, base_lr: float, seed: int,
    train_loader,
    X_probe: torch.Tensor,
    X_ka: torch.Tensor,
) -> Dict[str, np.ndarray]:
    path = seed_cache_path(N, L, gain, base_lr, seed)
    cached = safe_load_npz(path) if os.path.exists(path) else None
    if cache_ok(cached):
        return cached

    # deterministic init
    torch.manual_seed(SEED_BASE + seed)
    np.random.seed(SEED_BASE + seed)
    random.seed(SEED_BASE + seed)

    # init net (always exists)
    net_init = FC(N, L, gain=gain).to(DEVICE)
    net_init.eval()

    # trained net (may be None if training fails)
    net_tr = load_or_train_net(N, L, gain, base_lr, seed, train_loader=train_loader)

    if net_tr is None:
        out = dict(
            cache_version=np.array(CACHE_VERSION, np.int32),
            finished=np.array(True),
            train_ok=np.array(False),
            N=np.array(N, np.int32), L=np.array(L, np.int32),
            gain=np.array(gain, np.float32),
            base_lr=np.array(base_lr, np.float32),
            seed=np.array(seed, np.int32),
            RA=np.array(np.nan, np.float64),
            KA=np.array(np.nan, np.float64),
        )
        atomic_save_npz(path, **out)
        return out

    net_tr.eval()

    # RA (linear CKA on hidden features)
    with torch.inference_mode():
        H0 = net_init(X_probe, hid=True)
        HT = net_tr(X_probe, hid=True)
    ra = linear_cka_features(H0, HT)

    # KA (NTK alignment on subset)
    ka = ntk_align(net_init, net_tr, X_ka)

    out = dict(
        cache_version=np.array(CACHE_VERSION, np.int32),
        finished=np.array(True),
        train_ok=np.array(True),
        N=np.array(N, np.int32), L=np.array(L, np.int32),
        gain=np.array(gain, np.float32),
        base_lr=np.array(base_lr, np.float32),
        seed=np.array(seed, np.int32),
        RA=np.array(ra, np.float64),
        KA=np.array(ka, np.float64),
    )
    atomic_save_npz(path, **out)
    return out


# ---------------- Grid state ----------------
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
    if int(np.array(d.get("grid_version", 0)).item()) != GRID_VERSION:
        return None
    if int(np.array(d.get("data_seed", -1)).item()) != DATA_SEED:
        return None
    if int(np.array(d.get("KA_SUBSET", -1)).item()) != KA_SUBSET:
        return None
    if (not np.array_equal(np.array(widths), d.get("widths")) or
        not np.array_equal(np.array(depths), d.get("depths")) or
        not np.allclose(np.array(gains, np.float32), d.get("gains").astype(np.float32)) or
        not np.allclose(np.array(base_lrs, np.float32), d.get("base_lrs").astype(np.float32)) or
        not np.array_equal(np.array(seeds), d.get("seeds"))):
        return None
    return d


# ---------------- Axes ----------------
def load_axes_from_phase2_or_defaults():
    widths  = [10, 50, 100, 200]
    depths  = [2, 6, 8, 12]
    gains   = [0.8, 0.9, 1.0, 1.1]
    base_lrs = [0.025, 0.05, 0.075, 0.10]
    seeds   = [0, 1, 2]
    done_mask = None

    if os.path.exists(PHASE2_GRID_STATE):
        d = safe_load_npz(PHASE2_GRID_STATE)
        if d is not None:
            widths = d["widths"].astype(int).tolist()
            depths = d["depths"].astype(int).tolist()
            gains = d["gains"].astype(float).tolist()
            base_lrs = d["base_lrs"].astype(float).tolist()
            seeds = d["seeds"].astype(int).tolist()
            if PHASE2_DONE_ONLY and "done_map" in d:
                done_mask = d["done_map"].astype(bool)
            print("[axes] loaded axes from phase2_grid_state.npz")
    return widths, depths, gains, base_lrs, seeds, done_mask


# ---------------- Main runner ----------------
def run_ra_ka_grid(widths, depths, gains, base_lrs, seeds, train_loader, X_probe, X_ka, phase2_done_mask):
    shape = (len(gains), len(base_lrs), len(depths), len(widths))

    loaded = try_load_grid_state(GRID_STATE, widths, depths, gains, base_lrs, seeds)
    if loaded is not None:
        RA_map = loaded["RA_map"]
        KA_map = loaded["KA_map"]
        RA_std_map = loaded["RA_std_map"]
        KA_std_map = loaded["KA_std_map"]
        done_map = loaded["done_map"].astype(bool)
        print(f"[ra/ka grid] loaded {done_map.sum()}/{done_map.size} cells")
    else:
        RA_map = np.full(shape, np.nan, np.float64)
        KA_map = np.full(shape, np.nan, np.float64)
        RA_std_map = np.full(shape, np.nan, np.float64)
        KA_std_map = np.full(shape, np.nan, np.float64)
        done_map = np.zeros(shape, bool)

    total = done_map.size

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if done_map[gi, li, di, wi]:
                        continue
                    if phase2_done_mask is not None and not phase2_done_mask[gi, li, di, wi]:
                        continue

                    print(f"\n[cell] N={N} L={L} g={g} lr={lr}")

                    ra_vals, ka_vals = [], []
                    for sd in seeds:
                        sdat = compute_or_load_seed_ra_ka(
                            N, L, g, lr, sd,
                            train_loader=train_loader,
                            X_probe=X_probe,
                            X_ka=X_ka,
                        )
                        ok = bool(np.array(sdat.get("train_ok", True)).item())
                        if not ok:
                            continue
                        ra_vals.append(float(sdat["RA"]))
                        ka_vals.append(float(sdat["KA"]))

                    if len(ra_vals) == 0:
                        RA_map[gi, li, di, wi] = np.nan
                        KA_map[gi, li, di, wi] = np.nan
                        RA_std_map[gi, li, di, wi] = np.nan
                        KA_std_map[gi, li, di, wi] = np.nan
                    else:
                        RA_map[gi, li, di, wi] = float(np.mean(ra_vals))
                        KA_map[gi, li, di, wi] = float(np.mean(ka_vals))
                        RA_std_map[gi, li, di, wi] = float(np.std(ra_vals, ddof=0))
                        KA_std_map[gi, li, di, wi] = float(np.std(ka_vals, ddof=0))
                    done_map[gi, li, di, wi] = True

                    done = int(done_map.sum())
                    print(f"[cell-done] ({done}/{total})  RA={RA_map[gi,li,di,wi]:.3f}±{RA_std_map[gi,li,di,wi]:.3f}  "
                          f"KA={KA_map[gi,li,di,wi]:.3f}±{KA_std_map[gi,li,di,wi]:.3f}")

                    save_grid_state(
                        GRID_STATE,
                        widths, depths, gains, base_lrs, seeds,
                        RA_map, KA_map, RA_std_map, KA_std_map, done_map,
                    )

    save_grid_state(
        GRID_STATE,
        widths, depths, gains, base_lrs, seeds,
        RA_map, KA_map, RA_std_map, KA_std_map, done_map,
    )
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs, seeds=seeds,
        RA_map=RA_map, KA_map=KA_map,
        RA_std_map=RA_std_map, KA_std_map=KA_std_map,
        done_map=done_map,
    )


# ---------------- Plotting ----------------
def plot_heatmap(mat2d: np.ndarray, widths: List[int], depths: List[int],
                 title: str, out_path: str, vmin=0, vmax=1):
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


def plot_ra_ka_slices(grid: Dict):
    widths = grid["widths"]
    depths = grid["depths"]
    gains = grid["gains"]
    lrs = grid["base_lrs"]

    RA = grid["RA_map"]
    KA = grid["KA_map"]

    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            gstr = fmt_float(float(g))
            lrstr = fmt_float(float(lr))

            plot_heatmap(RA[gi, li], widths, depths,
                         title=f"RA (linear CKA)  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_RA_g{gstr}_lr{lrstr}.png"))
            plot_heatmap(KA[gi, li], widths, depths,
                         title=f"KA (NTK align)  g={g} lr={lr}",
                         out_path=os.path.join(PLOT_DIR, f"heatmap_KA_g{gstr}_lr{lrstr}.png"))


# ---------------- Entry ----------------
if __name__ == "__main__":
    print("[device]", DEVICE)
    if DEVICE.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))
        print("[torch.func]", _HAS_TORCHFUNC)

    widths, depths, gains, base_lrs, seeds, phase2_done_mask = load_axes_from_phase2_or_defaults()
    print("[grid] widths", widths)
    print("[grid] depths", depths)
    print("[grid] gains", gains)
    print("[grid] lrs", base_lrs)
    print("[grid] seeds", seeds)

    (xt, yt), (xe, ye) = load_or_make_circle_data(DATA_CACHE_FILE, DATA_SEED)
    train_loader = dataset_to_loader((xt, yt), BATCH_SIZE_TRAIN, shuffle=True, device=DEVICE)

    X_probe_full = xe.to(DEVICE)
    if PROBE_SUBSET is None or PROBE_SUBSET >= X_probe_full.shape[0]:
        X_probe = X_probe_full
    else:
        gen = torch.Generator(device="cpu").manual_seed(999)
        idxp = torch.randperm(X_probe_full.shape[0], generator=gen)[:PROBE_SUBSET].to(torch.long)
        X_probe = X_probe_full[idxp.to(DEVICE)]

    gen = torch.Generator(device="cpu").manual_seed(KA_SUBSET_SEED)
    idx = torch.randperm(X_probe_full.shape[0], generator=gen)[:KA_SUBSET].to(torch.long)
    X_ka = X_probe_full[idx.to(DEVICE)]

    grid = run_ra_ka_grid(
        widths, depths, gains, base_lrs, seeds,
        train_loader=train_loader,
        X_probe=X_probe,
        X_ka=X_ka,
        phase2_done_mask=phase2_done_mask,
    )

    print("[saved]", GRID_STATE)
    plot_ra_ka_slices(grid)
    print("[plots] saved to", PLOT_DIR)
