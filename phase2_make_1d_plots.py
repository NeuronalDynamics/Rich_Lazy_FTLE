# phase2_make_1d_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt

GRID_STATE = "phase2_grid_state.npz"
PLOT_DIR   = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------
# Helpers: robust aggregation
# ----------------------------

def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]

def geom_mean_and_logstd(vals: np.ndarray):
    """
    Geometric mean + 1-sigma spread computed in log10 space.
    Returns (center, lo, hi) in linear space.
    """
    v = _finite(vals)
    v = v[v > 0]  # variance should be >0; ignore non-positive just in case
    if v.size == 0:
        return np.nan, np.nan, np.nan
    logv = np.log10(v)
    mu = logv.mean()
    sd = logv.std(ddof=0)
    center = 10 ** mu
    lo     = 10 ** (mu - sd)
    hi     = 10 ** (mu + sd)
    return float(center), float(lo), float(hi)

def fisher_mean_and_std(rhos: np.ndarray):
    """
    Fisher-z mean + 1-sigma in Fisher space.
    Returns (center, lo, hi) in rho space.
    """
    r = _finite(rhos)
    if r.size == 0:
        return np.nan, np.nan, np.nan
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    mu = z.mean()
    sd = z.std(ddof=0)
    center = np.tanh(mu)
    lo     = np.tanh(mu - sd)
    hi     = np.tanh(mu + sd)
    return float(center), float(lo), float(hi)

def lineplot_with_band(x, center, lo, hi, xlabel, ylabel, title, out_path, yscale=None, ylim=None):
    x = np.asarray(x)
    center = np.asarray(center, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    # Mask non-finite points
    m = np.isfinite(x) & np.isfinite(center) & np.isfinite(lo) & np.isfinite(hi)
    x = x[m]; center = center[m]; lo = lo[m]; hi = hi[m]
    if x.size == 0:
        print(f"[warn] nothing to plot for {out_path}")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(x, center, "o-")
    plt.fill_between(x, lo, hi, alpha=0.2)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if yscale is not None:
        plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(*ylim)

    # Better ticks for categorical-ish floats
    if x.dtype.kind in ("f",):
        plt.xticks(x)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[saved] {out_path}")

# ----------------------------
# Aggregation along each axis
# grid arrays are shaped:
#   [gain, lr, depth, width]
# ----------------------------

def agg_over_axis_G(G_map: np.ndarray, axis: int, axis_vals):
    centers, los, his = [], [], []
    for i in range(len(axis_vals)):
        sl = np.take(G_map, i, axis=axis).ravel()
        c, lo, hi = geom_mean_and_logstd(sl)
        centers.append(c); los.append(lo); his.append(hi)
    return centers, los, his

def agg_over_axis_rho(rho_map: np.ndarray, axis: int, axis_vals):
    centers, los, his = [], [], []
    for i in range(len(axis_vals)):
        sl = np.take(rho_map, i, axis=axis).ravel()
        c, lo, hi = fisher_mean_and_std(sl)
        centers.append(c); los.append(lo); his.append(hi)
    return centers, los, his

# ----------------------------
# Main
# ----------------------------

def main():
    if not os.path.exists(GRID_STATE):
        raise FileNotFoundError(f"Could not find {GRID_STATE}. Put this script next to it.")

    d = np.load(GRID_STATE, allow_pickle=False)

    widths   = d["widths"].astype(int).tolist()
    depths   = d["depths"].astype(int).tolist()
    gains    = d["gains"].astype(float).tolist()
    base_lrs = d["base_lrs"].astype(float).tolist()

    G_lambda_map   = d["G_lambda_map"].astype(float)     # [g, lr, L, N]
    rho_lambda_map = d["rho_lambda_map"].astype(float)   # [g, lr, L, N]

    # Optional completeness warning
    if "done_map" in d:
        done = d["done_map"].astype(bool)
        if not done.all():
            print(f"[warn] grid not fully complete: {done.sum()}/{done.size} cells done. Plots will ignore NaNs.")

    # ---- G_lambda vs N ---- (axis=3)
    c, lo, hi = agg_over_axis_G(G_lambda_map, axis=3, axis_vals=widths)
    lineplot_with_band(
        widths, c, lo, hi,
        xlabel="Width N",
        ylabel="G_lambda (Var[λ])",
        title="G_lambda vs N (geometric mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_Glambda_vs_N.png"),
        yscale="log"
    )

    # ---- G_lambda vs L ---- (axis=2)
    c, lo, hi = agg_over_axis_G(G_lambda_map, axis=2, axis_vals=depths)
    lineplot_with_band(
        depths, c, lo, hi,
        xlabel="Depth L",
        ylabel="G_lambda (Var[λ])",
        title="G_lambda vs L (geometric mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_Glambda_vs_L.png"),
        yscale="log"
    )

    # ---- G_lambda vs g ---- (axis=0)
    c, lo, hi = agg_over_axis_G(G_lambda_map, axis=0, axis_vals=gains)
    lineplot_with_band(
        gains, c, lo, hi,
        xlabel="gain g",
        ylabel="G_lambda (Var[λ])",
        title="G_lambda vs gain g (geometric mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_Glambda_vs_gain.png"),
        yscale="log"
    )

    # ---- G_lambda vs lr ---- (axis=1)
    c, lo, hi = agg_over_axis_G(G_lambda_map, axis=1, axis_vals=base_lrs)
    lineplot_with_band(
        base_lrs, c, lo, hi,
        xlabel="base_lr",
        ylabel="G_lambda (Var[λ])",
        title="G_lambda vs base_lr (geometric mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_Glambda_vs_lr.png"),
        yscale="log"
    )

    # ---- rho_lambda vs N ---- (axis=3)
    c, lo, hi = agg_over_axis_rho(rho_lambda_map, axis=3, axis_vals=widths)
    lineplot_with_band(
        widths, c, lo, hi,
        xlabel="Width N",
        ylabel="rho_lambda (Spearman)",
        title="rho_lambda vs N (Fisher mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_rho_lambda_vs_N.png"),
        ylim=(-1, 1)
    )

    # ---- rho_lambda vs L ---- (axis=2)
    c, lo, hi = agg_over_axis_rho(rho_lambda_map, axis=2, axis_vals=depths)
    lineplot_with_band(
        depths, c, lo, hi,
        xlabel="Depth L",
        ylabel="rho_lambda (Spearman)",
        title="rho_lambda vs L (Fisher mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_rho_lambda_vs_L.png"),
        ylim=(-1, 1)
    )

    # ---- rho_lambda vs g ---- (axis=0)
    c, lo, hi = agg_over_axis_rho(rho_lambda_map, axis=0, axis_vals=gains)
    lineplot_with_band(
        gains, c, lo, hi,
        xlabel="gain g",
        ylabel="rho_lambda (Spearman)",
        title="rho_lambda vs gain g (Fisher mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_rho_lambda_vs_gain.png"),
        ylim=(-1, 1)
    )

    # ---- rho_lambda vs lr ---- (axis=1)
    c, lo, hi = agg_over_axis_rho(rho_lambda_map, axis=1, axis_vals=base_lrs)
    lineplot_with_band(
        base_lrs, c, lo, hi,
        xlabel="base_lr",
        ylabel="rho_lambda (Spearman)",
        title="rho_lambda vs base_lr (Fisher mean ± 1σ)",
        out_path=os.path.join(PLOT_DIR, "line_rho_lambda_vs_lr.png"),
        ylim=(-1, 1)
    )

    print(f"\nDone. Plots saved in: {PLOT_DIR}/")

if __name__ == "__main__":
    main()
