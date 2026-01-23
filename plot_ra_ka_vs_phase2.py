import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PHASE2_PATH = "phase2_grid_state.npz"
PHASE3_PATH = "ra_ka_grid_state.npz"   # produced by phase3_ra_ka_grid.py
OUT_DIR     = "plots_ra_ka_vs_phase2"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def load_npz(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def find_index(arr: np.ndarray, value, *, rtol=1e-6, atol=1e-8):
    """
    Robust float matching for axes like gains/base_lrs stored as float32.
    Returns index or None.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind in ("f", "c"):
        idx = np.where(np.isclose(arr, value, rtol=rtol, atol=atol))[0]
    else:
        idx = np.where(arr == value)[0]
    if idx.size == 0:
        return None
    return int(idx[0])

def safe_log10(x: np.ndarray, floor=1e-300) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log10(np.maximum(x, floor))

def finite_pairs(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def spearman_txt(x, y):
    x2, y2 = finite_pairs(x, y)
    if x2.size < 3:
        return "Spearman ρ = NaN (too few points)"
    rho, p = spearmanr(x2, y2)
    return f"Spearman ρ = {rho:.3f} (p={p:.2e}, n={x2.size})"

def scatter_plot(x, y, xlabel, ylabel, title, out_path, *, xlim=None, ylim=None):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=18, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

# ---------- load grids ----------
p2 = load_npz(PHASE2_PATH)
p3 = load_npz(PHASE3_PATH)

# Phase-2 axes and arrays
w2   = p2["widths"].astype(int)
d2   = p2["depths"].astype(int)
g2   = p2["gains"].astype(np.float64)
lr2  = p2["base_lrs"].astype(np.float64)

Glam2 = p2["G_lambda_map"]      # shape: [g, lr, L, N]
rhol2 = p2["rho_lambda_map"]
done2 = p2["done_map"].astype(bool)

# Phase-3 axes and arrays
w3   = p3["widths"].astype(int)
d3   = p3["depths"].astype(int)
g3   = p3["gains"].astype(np.float64)
lr3  = p3["base_lrs"].astype(np.float64)

RA3   = p3["RA_map"]            # shape: [g, lr, L, N]
KA3   = p3["KA_map"]
done3 = p3["done_map"].astype(bool)

# ---------- match configs by (N, L, g, lr) ----------
records = []
miss = 0

for gi2, g in enumerate(g2):
    gi3 = find_index(g3, g, rtol=1e-5, atol=1e-7)
    if gi3 is None:
        miss += len(lr2) * len(d2) * len(w2)
        continue

    for li2, lr in enumerate(lr2):
        li3 = find_index(lr3, lr, rtol=1e-5, atol=1e-7)
        if li3 is None:
            miss += len(d2) * len(w2)
            continue

        for di2, L in enumerate(d2):
            di3 = find_index(d3, L)
            if di3 is None:
                miss += len(w2)
                continue

            for wi2, N in enumerate(w2):
                wi3 = find_index(w3, N)
                if wi3 is None:
                    miss += 1
                    continue

                if (not done2[gi2, li2, di2, wi2]) or (not done3[gi3, li3, di3, wi3]):
                    continue

                Glam = float(Glam2[gi2, li2, di2, wi2])
                rhol = float(rhol2[gi2, li2, di2, wi2])
                ra   = float(RA3[gi3, li3, di3, wi3])
                ka   = float(KA3[gi3, li3, di3, wi3])

                records.append((N, L, g, lr, ra, ka, Glam, rhol))

print(f"[match] matched cells: {len(records)}")
if miss > 0:
    print(f"[match] cells skipped due to missing axis matches: ~{miss}")

if len(records) == 0:
    raise RuntimeError("No matched cells found. Check that both grids finished and share the same axes.")

# unpack
N_arr   = np.array([r[0] for r in records], dtype=int)
L_arr   = np.array([r[1] for r in records], dtype=int)
g_arr   = np.array([r[2] for r in records], dtype=np.float64)
lr_arr  = np.array([r[3] for r in records], dtype=np.float64)
RA_arr  = np.array([r[4] for r in records], dtype=np.float64)
KA_arr  = np.array([r[5] for r in records], dtype=np.float64)
G_arr   = np.array([r[6] for r in records], dtype=np.float64)
rho_arr = np.array([r[7] for r in records], dtype=np.float64)

# Commonly, G_lambda spans orders of magnitude => log-scale is much more readable.
logG = safe_log10(G_arr)

# ---------- plots ----------
# 1) RA vs G_lambda
title = "RA vs log10(G_lambda)\n" + spearman_txt(RA_arr, logG)
scatter_plot(
    logG, RA_arr,
    xlabel="log10(G_lambda)",
    ylabel="RA (linear CKA)",
    title=title,
    out_path=os.path.join(OUT_DIR, "RA_vs_log10_Glambda.png"),
    ylim=(0, 1),
)

# 2) KA vs G_lambda
title = "KA vs log10(G_lambda)\n" + spearman_txt(KA_arr, logG)
scatter_plot(
    logG, KA_arr,
    xlabel="log10(G_lambda)",
    ylabel="KA (NTK alignment)",
    title=title,
    out_path=os.path.join(OUT_DIR, "KA_vs_log10_Glambda.png"),
    ylim=(0, 1),
)

# 3) RA vs rho_lambda
title = "RA vs rho_lambda\n" + spearman_txt(RA_arr, rho_arr)
scatter_plot(
    rho_arr, RA_arr,
    xlabel="rho_lambda (Spearman(λ, margin))",
    ylabel="RA (linear CKA)",
    title=title,
    out_path=os.path.join(OUT_DIR, "RA_vs_rho_lambda.png"),
    xlim=(-1, 1),
    ylim=(0, 1),
)

# 4) KA vs rho_lambda
title = "KA vs rho_lambda\n" + spearman_txt(KA_arr, rho_arr)
scatter_plot(
    rho_arr, KA_arr,
    xlabel="rho_lambda (Spearman(λ, margin))",
    ylabel="KA (NTK alignment)",
    title=title,
    out_path=os.path.join(OUT_DIR, "KA_vs_rho_lambda.png"),
    xlim=(-1, 1),
    ylim=(0, 1),
)

print(f"[saved] plots -> {OUT_DIR}/")

# ---------- optional: “colored by something” versions (no extra dependencies) ----------
# If you want color-coded views, uncomment ONE block below.

# # Color by depth L
# plt.figure(figsize=(6,5))
# sc = plt.scatter(logG, RA_arr, c=L_arr, s=18, alpha=0.8)
# plt.xlabel("log10(G_lambda)"); plt.ylabel("RA"); plt.title("RA vs log10(G_lambda) colored by depth L")
# plt.colorbar(sc, label="Depth L"); plt.grid(True, alpha=0.3); plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "RA_vs_log10_Glambda_colored_by_L.png"), dpi=240); plt.close()

# # Color by width N
# plt.figure(figsize=(6,5))
# sc = plt.scatter(logG, rho_arr, c=N_arr, s=18, alpha=0.8)
# plt.xlabel("log10(G_lambda)"); plt.ylabel("rho_lambda"); plt.title("rho_lambda vs log10(G_lambda) colored by width N")
# plt.colorbar(sc, label="Width N"); plt.grid(True, alpha=0.3); plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "rho_vs_log10_Glambda_colored_by_N.png"), dpi=240); plt.close()
