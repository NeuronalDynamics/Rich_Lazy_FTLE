# phase4_transition_scaling.py
"""
Phase 4: Transition analysis (finite-size scaling + susceptibility)
------------------------------------------------------------------

This script uses:
  - phase2_grid_state.npz      (G_lambda_map, rho_lambda_map, axes)
  - ra_ka_grid_state.npz       (RA_map, KA_map, axes)
  - phase2_cache/*.npz         (per-seed margins + ftle_vals)

Key feature:
  * Recomputes pooled Spearman rho_lambda per config by concatenating
    all seeds' samples (old behavior), which reduces rho variance and
    avoids sign flips near 0.

Outputs (created under plots_transition/):
  - scatter and fit overlays
  - k(N), x0(N), chi(N) scaling plots
  - susceptibility curves and peak scaling
  - data collapse plot (best nu)
  - CSV table with all paired metrics
  - NPZ cache with pooled rho/G_lambda for quick re-runs
"""

import os
import math
import csv
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.optimize import curve_fit


# ---------------------------- USER CONFIG ----------------------------
PHASE2_GRID = "phase2_grid_state.npz"
RAKA_GRID   = "ra_ka_grid_state.npz"
PHASE2_CACHE_DIR = "phase2_cache"

OUT_DIR = "plots_transition"
os.makedirs(OUT_DIR, exist_ok=True)

# If True: recompute rho_lambda and G_lambda by concatenating per-seed samples from phase2_cache.
# This matches your old method and reduces rho variance without re-running PGD/training.
RECOMPUTE_POOLED_FROM_SEED_CACHES = True

# Cache pooled results here to avoid re-reading all seedstats every time you run the script.
POOLED_NPZ = os.path.join(OUT_DIR, "pooled_from_seed_caches.npz")

# Margin saturation diagnostic (from Phase2)
EPS_HI = 0.30
SAT_TOL = 1e-6
COMPUTE_RHO_UNSAT = True   # also compute rho excluding saturated margins (diagnostic)

# Susceptibility binning (per width)
SUSC_BINS = 20  # equal-count bins
MIN_POINTS_PER_BIN = 5

# Data collapse search
NU_GRID = np.linspace(0.0, 2.0, 41)  # try nu in [0,2] step 0.05
COLLAPSE_BINS = 60

# Sigmoid fit bounds
# y = a + (b-a)/(1 + exp(-k*(x-x0))), x=RA in [0,1], y=rho in [-1,1]
BOUNDS = (
    [-1.2, -1.2, 1e-3, 0.0],   # a, b, k, x0
    [ 1.2,  1.2, 200.0, 1.0]
)
MAXFEV = 200000


# ---------------------------- HELPERS ----------------------------
def fmt_float(x: float) -> str:
    """Match your Phase2 fmt_float: f'{x:.3g}' then '.'->'p', '-'->'m' prefix."""
    s = f"{x:.3g}"
    s = s.replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s

def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def pick_key(d: Dict[str, np.ndarray], candidates: List[str]) -> Optional[str]:
    """Pick a key from dict by exact, case-insensitive, then substring match."""
    for c in candidates:
        if c in d:
            return c
    lower = {k.lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for k in d.keys():
        kl = k.lower()
        for c in candidates:
            if c.lower() in kl:
                return k
    return None

def logistic4(x, a, b, k, x0):
    return a + (b - a) / (1.0 + np.exp(-k * (x - x0)))

def fit_sigmoid(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """Fit logistic4(y vs x). Returns params + metrics or None if fails."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 10:
        return None

    # Initial guess from quantiles
    a0 = float(np.nanpercentile(y, 5))
    b0 = float(np.nanpercentile(y, 95))
    if not np.isfinite(a0): a0 = float(np.nanmin(y))
    if not np.isfinite(b0): b0 = float(np.nanmax(y))

    # Ensure b0 != a0
    if abs(b0 - a0) < 1e-6:
        b0 = a0 + 1e-3

    mid = 0.5 * (a0 + b0)
    i0 = int(np.argmin(np.abs(y - mid)))
    x00 = float(np.clip(x[i0], 0.0, 1.0))

    # Rough slope estimate near mid
    # Use a small neighborhood around x00
    idx = np.argsort(x)
    xs = x[idx]; ys = y[idx]
    j = np.searchsorted(xs, x00)
    j0 = max(0, j - 3); j1 = min(xs.size, j + 3)
    if j1 - j0 >= 2:
        # local linear slope
        px = xs[j0:j1]; py = ys[j0:j1]
        denom = float(np.var(px)) + 1e-12
        slope = float(np.cov(px, py, bias=True)[0, 1] / denom)
    else:
        slope = float(np.cov(xs, ys, bias=True)[0, 1] / (float(np.var(xs)) + 1e-12))

    k0 = float(np.clip(4.0 * slope / (b0 - a0 + 1e-12), 0.1, 10.0))

    p0 = [a0, b0, k0, x00]

    try:
        popt, pcov = curve_fit(
            logistic4, x, y, p0=p0,
            bounds=BOUNDS, maxfev=MAXFEV
        )
    except Exception as e:
        return None

    a, b, k, x0 = [float(v) for v in popt]
    yhat = logistic4(x, *popt)
    resid = y - yhat
    sse = float(np.sum(resid * resid))
    rmse = float(np.sqrt(sse / max(1, x.size)))
    mae = float(np.mean(np.abs(resid)))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2)) + 1e-12
    r2 = float(1.0 - sse / sst)

    # AIC/BIC (Gaussian errors; constants dropped)
    n = x.size
    p = 4
    aic = float(n * np.log(sse / max(1, n) + 1e-12) + 2 * p)
    bic = float(n * np.log(sse / max(1, n) + 1e-12) + p * np.log(max(2, n)))

    chi = float((b - a) * k / 4.0)  # slope at midpoint

    return dict(a=a, b=b, k=k, x0=x0, chi=chi, rmse=rmse, mae=mae, r2=r2, aic=aic, bic=bic, n=n)

def seed_cache_path(N: int, L: int, g: float, lr: float, seed: int, grid: int, dseed: int) -> str:
    gstr = fmt_float(float(g))
    lrstr = fmt_float(float(lr))
    fn = f"seedstats_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_grid{grid}_dseed{dseed}.npz"
    return os.path.join(PHASE2_CACHE_DIR, fn)

def pooled_metrics_from_seed_caches(widths, depths, gains, lrs, seeds, grid: int, dseed: int) -> Dict[str, np.ndarray]:
    """
    Recompute per-config:
      - rho_pool: Spearman(lambda, margin) after concatenating all seeds' samples.
      - G_pool: Var(lambda) on concatenated samples.
      - sat_frac: fraction of samples saturated at EPS_HI
      - rho_unsat (optional): Spearman on unsaturated only

    Returns maps of shape [g, lr, L, N] matching Phase2 ordering.
    """
    shape = (len(gains), len(lrs), len(depths), len(widths))
    rho_pool = np.full(shape, np.nan, dtype=np.float64)
    G_pool   = np.full(shape, np.nan, dtype=np.float64)
    sat_frac = np.full(shape, np.nan, dtype=np.float64)
    rho_uns  = np.full(shape, np.nan, dtype=np.float64)

    missing = 0
    total_cfg = np.prod(shape)

    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    lam_all = []
                    m_all   = []
                    for sd in seeds:
                        p = seed_cache_path(int(N), int(L), float(g), float(lr), int(sd), grid, dseed)
                        if not os.path.exists(p):
                            missing += 1
                            continue
                        d = load_npz(p)
                        if "ftle_vals" not in d or "margins" not in d:
                            missing += 1
                            continue
                        lam_all.append(d["ftle_vals"].astype(np.float64, copy=False).reshape(-1))
                        m_all.append(d["margins"].astype(np.float64, copy=False).reshape(-1))

                    if len(lam_all) == 0:
                        continue

                    lam = np.concatenate(lam_all, axis=0)
                    mar = np.concatenate(m_all, axis=0)

                    mask = np.isfinite(lam) & np.isfinite(mar)
                    lam = lam[mask]
                    mar = mar[mask]
                    if lam.size < 10:
                        continue

                    # pooled variance of lambda
                    G_pool[gi, li, di, wi] = float(np.var(lam))

                    # saturation fraction
                    sat = float(np.mean(mar >= (EPS_HI - SAT_TOL)))
                    sat_frac[gi, li, di, wi] = sat

                    # pooled spearman
                    rho, _ = spearmanr(lam, mar)
                    rho_pool[gi, li, di, wi] = float(rho)

                    if COMPUTE_RHO_UNSAT:
                        mu = mar < (EPS_HI - SAT_TOL)
                        if np.sum(mu) >= 10:
                            rho_u, _ = spearmanr(lam[mu], mar[mu])
                            rho_uns[gi, li, di, wi] = float(rho_u)

    print(f"[pooled] built pooled metrics for {total_cfg} configs. missing seed files: {missing}")
    return dict(rho_pool=rho_pool, G_pool=G_pool, sat_frac=sat_frac, rho_unsat=rho_uns)


def collapse_error_for_nu(rows_by_N: Dict[int, Dict[str, np.ndarray]], nu: float) -> float:
    """
    Compute a simple collapse error for y_norm vs x_scaled across widths.
    rows_by_N[N] contains arrays: ra, rho, a,b,x0
    """
    # Gather all x_scaled to determine global bin edges
    xs_all = []
    ys_all = []
    Ns = sorted(rows_by_N.keys())
    for N in Ns:
        ra = rows_by_N[N]["ra"]
        rho = rows_by_N[N]["rho"]
        a = rows_by_N[N]["a"]; b = rows_by_N[N]["b"]; x0 = rows_by_N[N]["x0"]
        if not np.isfinite(a) or not np.isfinite(b) or abs(b-a) < 1e-9 or not np.isfinite(x0):
            continue
        y = (rho - a) / (b - a)
        x = (ra - x0) * (float(N) ** nu)
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) < 10:
            continue
        xs_all.append(x[m])
        ys_all.append(y[m])

    if len(xs_all) < 2:
        return float("inf")

    X = np.concatenate(xs_all)
    xmin, xmax = float(np.nanmin(X)), float(np.nanmax(X))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return float("inf")

    edges = np.linspace(xmin, xmax, COLLAPSE_BINS + 1)

    # For each bin, compute per-N mean, then variance across N
    err = 0.0
    for bi in range(COLLAPSE_BINS):
        lo, hi = edges[bi], edges[bi+1]
        means = []
        for N in Ns:
            ra = rows_by_N[N]["ra"]
            rho = rows_by_N[N]["rho"]
            a = rows_by_N[N]["a"]; b = rows_by_N[N]["b"]; x0 = rows_by_N[N]["x0"]
            if not np.isfinite(a) or not np.isfinite(b) or abs(b-a) < 1e-9 or not np.isfinite(x0):
                continue
            y = (rho - a) / (b - a)
            x = (ra - x0) * (float(N) ** nu)
            m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x < hi)
            if np.sum(m) >= MIN_POINTS_PER_BIN:
                means.append(float(np.mean(y[m])))
        if len(means) >= 2:
            v = float(np.var(np.array(means, dtype=np.float64)))
            err += v
    return err


# ---------------------------- MAIN ANALYSIS ----------------------------
def main():
    # ----- Load phase2 grid -----
    if not os.path.exists(PHASE2_GRID):
        raise FileNotFoundError(f"Missing {PHASE2_GRID} in current folder.")

    p2 = load_npz(PHASE2_GRID)
    widths = p2["widths"].astype(int).tolist()
    depths = p2["depths"].astype(int).tolist()
    gains  = p2["gains"].astype(float).tolist()
    lrs    = p2["base_lrs"].astype(float).tolist()
    seeds  = p2["seeds"].astype(int).tolist()

    data_seed = int(p2["data_seed"].item()) if "data_seed" in p2 else 0
    # grid size is in filenames (161) not stored in p2; assume 161 unless you used a different one
    grid_size = 161

    # Get phase2 maps
    k_G = pick_key(p2, ["G_lambda_map", "G_lambda", "G_lambda_grid"])
    k_r = pick_key(p2, ["rho_lambda_map", "rho_lambda", "rho_lam_map"])
    if k_G is None or k_r is None:
        print("[phase2 keys]", list(p2.keys()))
        raise KeyError("Couldn't find G_lambda_map or rho_lambda_map in phase2_grid_state.npz")

    G_lambda_map_phase2 = p2[k_G].astype(np.float64)
    rho_lambda_map_phase2 = p2[k_r].astype(np.float64)

    done_map = p2["done_map"].astype(bool) if "done_map" in p2 else np.isfinite(G_lambda_map_phase2)

    print(f"[phase2] axes: widths={widths}, depths={depths}, gains={gains}, lrs={lrs}, seeds={seeds}")
    print(f"[phase2] map shapes: G={G_lambda_map_phase2.shape}, rho={rho_lambda_map_phase2.shape}, done={done_map.shape}")

    # ----- Load RA/KA grid -----
    if not os.path.exists(RAKA_GRID):
        raise FileNotFoundError(f"Missing {RAKA_GRID} in current folder.")

    rk = load_npz(RAKA_GRID)
    # Try to find RA/KA map keys
    k_RA = pick_key(rk, ["RA_map", "ra_map", "RA"])
    k_KA = pick_key(rk, ["KA_map", "ka_map", "KA"])
    if k_RA is None or k_KA is None:
        print("[ra_ka keys]", list(rk.keys()))
        raise KeyError("Couldn't find RA_map / KA_map in ra_ka_grid_state.npz")

    RA_map = rk[k_RA].astype(np.float64)
    KA_map = rk[k_KA].astype(np.float64)

    print(f"[ra_ka] map shapes: RA={RA_map.shape}, KA={KA_map.shape}")

    # Sanity: axes match?
    for name, arr, ref in [
        ("widths", rk.get("widths"), np.array(widths, dtype=np.int32)),
        ("depths", rk.get("depths"), np.array(depths, dtype=np.int32)),
        ("gains",  rk.get("gains"),  np.array(gains, dtype=np.float32)),
        ("base_lrs", rk.get("base_lrs"), np.array(lrs, dtype=np.float32)),
    ]:
        if arr is None:
            continue
        if name in ["gains", "base_lrs"]:
            ok = np.allclose(arr.astype(np.float64), ref.astype(np.float64))
        else:
            ok = np.array_equal(arr.astype(ref.dtype), ref)
        if not ok:
            print(f"[warn] axis mismatch on {name}: ra_ka_grid_state != phase2_grid_state")
            print("       I'll proceed, but pairing could be wrong if the axis ordering differs.")

    # ----- Optional: pooled recompute from seed caches -----
    if RECOMPUTE_POOLED_FROM_SEED_CACHES:
        # Try load cached pooled results
        pooled_ok = False
        if os.path.exists(POOLED_NPZ):
            pd = load_npz(POOLED_NPZ)
            try:
                # check axes match
                ok = (
                    np.array_equal(pd["widths"].astype(int), np.array(widths, dtype=int)) and
                    np.array_equal(pd["depths"].astype(int), np.array(depths, dtype=int)) and
                    np.allclose(pd["gains"].astype(float), np.array(gains, dtype=float)) and
                    np.allclose(pd["lrs"].astype(float), np.array(lrs, dtype=float)) and
                    np.array_equal(pd["seeds"].astype(int), np.array(seeds, dtype=int)) and
                    int(pd["data_seed"].item()) == data_seed and
                    int(pd["grid_size"].item()) == grid_size
                )
                if ok:
                    rho_pool = pd["rho_pool"]
                    G_pool = pd["G_pool"]
                    sat_frac = pd["sat_frac"]
                    rho_uns = pd["rho_unsat"] if "rho_unsat" in pd else np.full_like(rho_pool, np.nan)
                    pooled_ok = True
                    print("[pooled] loaded cached pooled metrics:", POOLED_NPZ)
            except Exception:
                pooled_ok = False

        if not pooled_ok:
            print("[pooled] recomputing pooled rho/G_lambda from phase2_cache (one-time) ...")
            pm = pooled_metrics_from_seed_caches(widths, depths, gains, lrs, seeds, grid=grid_size, dseed=data_seed)
            rho_pool = pm["rho_pool"]
            G_pool = pm["G_pool"]
            sat_frac = pm["sat_frac"]
            rho_uns = pm["rho_unsat"]

            np.savez(
                POOLED_NPZ,
                widths=np.array(widths, dtype=np.int32),
                depths=np.array(depths, dtype=np.int32),
                gains=np.array(gains, dtype=np.float32),
                lrs=np.array(lrs, dtype=np.float32),
                seeds=np.array(seeds, dtype=np.int32),
                data_seed=np.array(data_seed, dtype=np.int32),
                grid_size=np.array(grid_size, dtype=np.int32),
                rho_pool=rho_pool,
                G_pool=G_pool,
                sat_frac=sat_frac,
                rho_unsat=rho_uns,
            )
            print("[pooled] saved:", POOLED_NPZ)

        # Use pooled values as "rho_lambda" and "G_lambda" for analysis
        rho_used = rho_pool
        G_used   = G_pool
        rho_used_name = "rho_lambda_pooled"
        G_used_name = "G_lambda_pooled"

    else:
        rho_used = rho_lambda_map_phase2
        G_used   = G_lambda_map_phase2
        rho_used_name = "rho_lambda_phase2"
        G_used_name = "G_lambda_phase2"
        sat_frac = np.full_like(rho_used, np.nan)
        rho_uns = np.full_like(rho_used, np.nan)

    # ----- Build a flat paired table -----
    rows = []
    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    if not done_map[gi, li, di, wi]:
                        continue
                    ra = float(RA_map[gi, li, di, wi])
                    ka = float(KA_map[gi, li, di, wi])
                    rho = float(rho_used[gi, li, di, wi])
                    Gl  = float(G_used[gi, li, di, wi])
                    sf  = float(sat_frac[gi, li, di, wi]) if np.isfinite(sat_frac[gi, li, di, wi]) else np.nan
                    ru  = float(rho_uns[gi, li, di, wi]) if np.isfinite(rho_uns[gi, li, di, wi]) else np.nan
                    if not (np.isfinite(ra) and np.isfinite(rho) and np.isfinite(Gl)):
                        continue
                    rows.append(dict(N=N, L=L, g=g, lr=lr, RA=ra, KA=ka,
                                     rho_lambda=rho, G_lambda=Gl, sat_frac=sf, rho_unsat=ru))

    print(f"[data] paired rows: n={len(rows)} (expected ~{len(widths)*len(depths)*len(gains)*len(lrs)})")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "paired_phase2_phase3_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[saved]", csv_path)

    # Convert to arrays for global plots
    RA_all = np.array([r["RA"] for r in rows], dtype=np.float64)
    KA_all = np.array([r["KA"] for r in rows], dtype=np.float64)
    rho_all = np.array([r["rho_lambda"] for r in rows], dtype=np.float64)
    Gl_all  = np.array([r["G_lambda"] for r in rows], dtype=np.float64)

    # ----- Global scatter plots -----
    def scatter(x, y, xlabel, ylabel, title, outname, xlim=None, ylim=None):
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, s=18, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if xlim: plt.xlim(*xlim)
        if ylim: plt.ylim(*ylim)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, outname), dpi=220)
        plt.close()

    rho_s, p_s = spearmanr(RA_all, rho_all)
    scatter(
        RA_all, rho_all,
        xlabel="RA (linear CKA)",
        ylabel=f"{rho_used_name} = Spearman(λ, margin)",
        title=f"rho_lambda vs RA  (Spearman(RA,rho)={rho_s:.3f}, p={p_s:.1e}, n={len(RA_all)})",
        outname="scatter_rho_vs_RA.png",
        xlim=(0,1), ylim=(-1,1)
    )

    rho_s, p_s = spearmanr(KA_all, rho_all)
    scatter(
        KA_all, rho_all,
        xlabel="KA (NTK alignment)",
        ylabel=f"{rho_used_name} = Spearman(λ, margin)",
        title=f"rho_lambda vs KA  (Spearman(KA,rho)={rho_s:.3f}, p={p_s:.1e}, n={len(KA_all)})",
        outname="scatter_rho_vs_KA.png",
        xlim=(0,1), ylim=(-1,1)
    )

    scatter(
        RA_all, np.log10(Gl_all + 1e-12),
        xlabel="RA (linear CKA)",
        ylabel=f"log10({G_used_name})",
        title="G_lambda vs RA",
        outname="scatter_log10G_vs_RA.png",
        xlim=(0,1)
    )

    # ----- Per-width sigmoid fits (finite-size scaling) -----
    rows_by_N = {}
    for N in widths:
        maskN = np.array([r["N"] == N for r in rows], dtype=bool)
        rows_by_N[N] = dict(
            ra=RA_all[maskN],
            rho=rho_all[maskN]
        )

    fit_params = {}
    for N in widths:
        x = rows_by_N[N]["ra"]
        y = rows_by_N[N]["rho"]
        fit = fit_sigmoid(x, y)
        if fit is None:
            print(f"[fit] N={N}: sigmoid fit FAILED")
            continue
        fit_params[N] = fit
        rows_by_N[N].update(fit)
        print(f"[fit] N={N}: a={fit['a']:.3f} b={fit['b']:.3f} k={fit['k']:.3f} x0={fit['x0']:.3f} chi={fit['chi']:.3f} R2={fit['r2']:.3f}")

    # Overlay plot of rho vs RA by width + fitted curves
    plt.figure(figsize=(7, 6))
    for N in widths:
        x = rows_by_N[N]["ra"]
        y = rows_by_N[N]["rho"]
        plt.scatter(x, y, s=16, alpha=0.55, label=f"N={N}")
        if N in fit_params:
            fp = fit_params[N]
            xs = np.linspace(0, 1, 400)
            ys = logistic4(xs, fp["a"], fp["b"], fp["k"], fp["x0"])
            plt.plot(xs, ys, linewidth=2.0)
    plt.xlabel("RA (linear CKA)")
    plt.ylabel(f"{rho_used_name} = Spearman(λ, margin)")
    plt.title("Finite-size sigmoid fits: rho_lambda vs RA (colored by width)")
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rho_vs_RA_sigmoid_fits_by_width.png"), dpi=240)
    plt.close()

    # Scaling plots: k(N), x0(N), chi(N)
    Ns = np.array([N for N in widths if N in fit_params], dtype=np.float64)
    kN  = np.array([fit_params[int(N)]["k"] for N in Ns], dtype=np.float64)
    x0N = np.array([fit_params[int(N)]["x0"] for N in Ns], dtype=np.float64)
    chiN = np.array([fit_params[int(N)]["chi"] for N in Ns], dtype=np.float64)

    def plot_xy(x, y, xlabel, ylabel, title, fname, logx=False, logy=False):
        plt.figure(figsize=(6, 4.5))
        xx, yy = x.copy(), y.copy()
        if logx: xx = np.log10(xx)
        if logy: yy = np.log10(yy)
        plt.plot(xx, yy, "o-")
        plt.xlabel(("log10 " if logx else "") + xlabel)
        plt.ylabel(("log10 " if logy else "") + ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=240)
        plt.close()

    plot_xy(Ns, kN, "N", "k", "Sigmoid sharpness k vs width N", "k_vs_N.png")
    plot_xy(Ns, x0N, "N", "x0", "Critical RA x0 vs width N", "x0_vs_N.png")
    plot_xy(Ns, chiN, "N", "chi", "Midpoint slope chi=(b-a)k/4 vs width N", "chi_vs_N.png")

    # Optional: power-law fit for k ~ N^alpha (very few points; still informative)
    if Ns.size >= 3 and np.all(kN > 0):
        alpha, c = np.polyfit(np.log(Ns), np.log(kN), deg=1)
        print(f"[scaling] fit k ~ N^alpha: alpha={alpha:.3f}")
        # show fitted line in log-log
        plt.figure(figsize=(6, 4.5))
        plt.plot(np.log10(Ns), np.log10(kN), "o", label="data")
        xx = np.linspace(np.min(np.log10(Ns)), np.max(np.log10(Ns)), 200)
        yy = (alpha / np.log(10)) * (xx * np.log(10)) + (c / np.log(10))
        plt.plot(xx, yy, "-", label=f"fit alpha={alpha:.2f}")
        plt.xlabel("log10 N")
        plt.ylabel("log10 k")
        plt.title("Log-log scaling: k vs N")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "k_vs_N_loglog.png"), dpi=240)
        plt.close()

    # ----- Susceptibility (variance peak) -----
    peak_rows = []
    plt.figure(figsize=(7, 6))
    for N in widths:
        x = rows_by_N[N]["ra"]
        y = rows_by_N[N]["rho"]
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        if x.size < SUSC_BINS * MIN_POINTS_PER_BIN:
            continue

        # equal-count bins via sorting
        idx = np.argsort(x)
        bins = np.array_split(idx, SUSC_BINS)

        centers = []
        means = []
        vars_ = []
        for b in bins:
            if b.size < MIN_POINTS_PER_BIN:
                continue
            xb = x[b]; yb = y[b]
            centers.append(float(np.mean(xb)))
            means.append(float(np.mean(yb)))
            vars_.append(float(np.var(yb)))

        centers = np.array(centers, dtype=np.float64)
        vars_ = np.array(vars_, dtype=np.float64)
        if centers.size < 5:
            continue

        # plot variance curve
        plt.plot(centers, vars_, "o-", label=f"N={N}")

        # peak
        jpk = int(np.nanargmax(vars_))
        peak_rows.append(dict(N=int(N), peak_RA=float(centers[jpk]), peak_var=float(vars_[jpk])))

    plt.xlabel("RA (bin centers)")
    plt.ylabel("Var[rho_lambda | RA bin]")
    plt.title("Susceptibility proxy: conditional variance vs RA (by width N)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "susceptibility_var_curves_by_width.png"), dpi=240)
    plt.close()

    if len(peak_rows) > 0:
        # peak scaling
        peakN = np.array([r["N"] for r in peak_rows], dtype=np.float64)
        peakRA = np.array([r["peak_RA"] for r in peak_rows], dtype=np.float64)
        peakV  = np.array([r["peak_var"] for r in peak_rows], dtype=np.float64)

        plot_xy(peakN, peakV, "N", "peak Var", "Peak susceptibility height vs width N", "susceptibility_peak_height_vs_N.png")
        plot_xy(peakN, peakRA, "N", "peak RA", "Peak susceptibility location vs width N", "susceptibility_peak_location_vs_N.png")

        # save peaks CSV
        peaks_csv = os.path.join(OUT_DIR, "susceptibility_peaks_by_width.csv")
        with open(peaks_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N", "peak_RA", "peak_var"])
            w.writeheader()
            w.writerows(peak_rows)
        print("[saved]", peaks_csv)

    # ----- Data collapse (best nu) -----
    if len(fit_params) >= 2:
        errs = []
        for nu in NU_GRID:
            err = collapse_error_for_nu(rows_by_N, float(nu))
            errs.append(err)
        errs = np.array(errs, dtype=np.float64)
        best_i = int(np.nanargmin(errs))
        best_nu = float(NU_GRID[best_i])
        print(f"[collapse] best nu={best_nu:.3f}  error={errs[best_i]:.4e}")

        # plot error vs nu
        plt.figure(figsize=(6, 4.5))
        plt.plot(NU_GRID, errs, "o-")
        plt.axvline(best_nu, linestyle="--")
        plt.xlabel("nu")
        plt.ylabel("collapse error (sum bin var across N)")
        plt.title("Data collapse search")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "collapse_error_vs_nu.png"), dpi=240)
        plt.close()

        # plot collapsed curves
        plt.figure(figsize=(7, 6))
        for N in widths:
            if N not in fit_params:
                continue
            ra = rows_by_N[N]["ra"]
            rho = rows_by_N[N]["rho"]
            a = rows_by_N[N]["a"]; b = rows_by_N[N]["b"]; x0 = rows_by_N[N]["x0"]
            if not np.isfinite(a) or not np.isfinite(b) or abs(b-a) < 1e-9:
                continue
            y = (rho - a) / (b - a)
            x = (ra - x0) * (float(N) ** best_nu)
            m = np.isfinite(x) & np.isfinite(y)
            plt.scatter(x[m], y[m], s=16, alpha=0.55, label=f"N={N}")
        plt.xlabel(r"$(RA - x0)\,N^{\nu}$")
        plt.ylabel(r"$(\rho-a)/(b-a)$")
        plt.title(f"Data collapse (best nu={best_nu:.3f})")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "data_collapse_best_nu.png"), dpi=240)
        plt.close()

    print(f"\nDone. All plots saved in: {OUT_DIR}\n")


if __name__ == "__main__":
    main()
