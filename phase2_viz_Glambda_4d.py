# phase2_viz_Glambda_4d.py
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

# 3D plotting support
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

GRID_STATE = "phase2_grid_state.npz"
PLOT_DIR   = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def safe_log10_pos(x):
    x = np.asarray(x)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x) & (x > 0)
    out[m] = np.log10(x[m])
    return out

def mean_std(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=0))

# -------------------------
# Load and flatten grid
# -------------------------
def load_flat_points(npz_path):
    d = np.load(npz_path, allow_pickle=False)

    widths   = d["widths"].astype(int)
    depths   = d["depths"].astype(int)
    gains    = d["gains"].astype(float)
    base_lrs = d["base_lrs"].astype(float)

    G = d["G_lambda_map"].astype(float)  # shape [g, lr, L, N]
    done = d["done_map"].astype(bool) if "done_map" in d else None

    # indices for each element
    gi, li, di, wi = np.indices(G.shape)

    N  = widths[wi].astype(float).ravel()
    L  = depths[di].astype(float).ravel()
    g  = gains[gi].astype(float).ravel()
    lr = base_lrs[li].astype(float).ravel()
    Gl = G.ravel()

    if done is not None:
        done_flat = done.ravel()
    else:
        done_flat = np.ones_like(Gl, dtype=bool)

    # log transforms
    logN  = np.log10(N)
    loglr = np.log10(lr)
    logGl = safe_log10_pos(Gl)

    # keep only completed and finite
    m = done_flat & finite_mask(logN, L, g, loglr, logGl)
    return dict(
        widths=widths, depths=depths, gains=gains, base_lrs=base_lrs,
        N=N[m], L=L[m], g=g[m], lr=lr[m],
        logN=logN[m], loglr=loglr[m], logGl=logGl[m],
        Gl=Gl[m]
    )

# -------------------------
# Plot 1: 3D scatter + marginal bars
# -------------------------
def plot_3d_scatter_with_marginals(P):
    gains = np.unique(P["g"])
    lrs   = np.unique(P["lr"])

    # marker list for gains
    markers = ["o", "^", "s", "D", "P", "X", "v", "<", ">"]
    if gains.size > len(markers):
        raise RuntimeError("Too many gain values for marker list; add more markers.")

    # color mapping by log10(lr)
    loglr = P["loglr"]
    norm = Normalize(vmin=np.min(loglr), vmax=np.max(loglr))
    cmap = plt.get_cmap("viridis")

    # sizes by gain (optional but useful)
    gmin, gmax = gains.min(), gains.max()
    def size_for_gain(g):
        if gmax == gmin:
            return 60.0
        t = (g - gmin) / (gmax - gmin)
        return 40.0 + 140.0 * t

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[3.2, 1.0, 1.0])

    ax3d   = fig.add_subplot(gs[0, 0], projection="3d")
    ax_g   = fig.add_subplot(gs[0, 1])
    ax_lr  = fig.add_subplot(gs[0, 2])

    # 3D scatter: x=log10 N, y=L, z=log10 G_lambda
    for k, gv in enumerate(sorted(gains)):
        mk = markers[k]
        mg = (P["g"] == gv)
        ax3d.scatter(
            P["logN"][mg], P["L"][mg], P["logGl"][mg],
            c=P["loglr"][mg], cmap=cmap, norm=norm,
            marker=mk, s=size_for_gain(gv), alpha=0.85, depthshade=True
        )

    ax3d.set_xlabel("log10(width N)")
    ax3d.set_ylabel("depth L")
    ax3d.set_zlabel("log10(G_lambda)")
    ax3d.set_title("G_lambda across (N,L,g,lr)\ncolor=log10(lr), marker/size=gain")

    # colorbar for lr
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, fraction=0.04, pad=0.08)
    cbar.set_label("log10(base_lr)")
    # show ticks as actual lr values
    tick_vals = np.log10(np.sort(lrs))
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{v:g}" for v in np.sort(lrs)])

    # legend for gains (marker mapping)
    legend_handles = []
    for k, gv in enumerate(sorted(gains)):
        legend_handles.append(
            Line2D([0], [0], marker=markers[k], color="w",
                   label=f"g={gv:g}",
                   markerfacecolor="gray", markeredgecolor="k",
                   markersize=9)
        )
    ax3d.legend(handles=legend_handles, title="gain (marker)", loc="upper left")

    # Marginal bar: mean±std of log10(Gλ) vs gain
    means, stds = [], []
    for gv in sorted(gains):
        m = (P["g"] == gv)
        mu, sd = mean_std(P["logGl"][m])
        means.append(mu); stds.append(sd)

    x = np.arange(len(gains))
    ax_g.bar(x, means)
    ax_g.errorbar(x, means, yerr=stds, fmt="none", capsize=4)
    ax_g.set_xticks(x)
    ax_g.set_xticklabels([f"{gv:g}" for gv in sorted(gains)], rotation=0)
    ax_g.set_title("mean±std log10(Gλ)\nvs gain")
    ax_g.set_xlabel("gain g")
    ax_g.set_ylabel("log10(Gλ)")
    ax_g.grid(True, alpha=0.3)

    # Marginal bar: mean±std of log10(Gλ) vs lr
    means, stds = [], []
    for lv in sorted(lrs):
        m = (P["lr"] == lv)
        mu, sd = mean_std(P["logGl"][m])
        means.append(mu); stds.append(sd)

    x = np.arange(len(lrs))
    ax_lr.bar(x, means)
    ax_lr.errorbar(x, means, yerr=stds, fmt="none", capsize=4)
    ax_lr.set_xticks(x)
    ax_lr.set_xticklabels([f"{lv:g}" for lv in sorted(lrs)], rotation=45, ha="right")
    ax_lr.set_title("mean±std log10(Gλ)\nvs lr")
    ax_lr.set_xlabel("base_lr")
    ax_lr.set_ylabel("log10(Gλ)")
    ax_lr.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "viz_3d_scatter_marginals_Glambda.png")
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"[saved] {out}")

# -------------------------
# Plot 2: Parallel coordinates
# -------------------------
def plot_parallel_coordinates(P):
    # Build matrix: [logN, L, g, loglr, logGl]
    X = np.stack([P["logN"], P["L"], P["g"], P["loglr"], P["logGl"]], axis=1)

    # Normalize each column to [0,1] for plotting
    Xn = X.copy()
    mins = np.nanmin(Xn, axis=0)
    maxs = np.nanmax(Xn, axis=0)
    denom = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
    Xn = (Xn - mins) / denom

    # color by logGl
    c = P["logGl"]
    norm = Normalize(vmin=np.nanpercentile(c, 5), vmax=np.nanpercentile(c, 95))
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    xs = np.arange(Xn.shape[1])
    for i in range(Xn.shape[0]):
        ax.plot(xs, Xn[i], color=cmap(norm(c[i])), alpha=0.35, linewidth=1.0)

    ax.set_xticks(xs)
    ax.set_xticklabels(["log10(N)", "L", "g", "log10(lr)", "log10(Gλ)"])
    ax.set_yticks([])
    ax.set_title("Parallel coordinates (colored by log10(Gλ))")

    # annotate min/max per axis (helps interpretation)
    for j, name in enumerate(["log10(N)", "L", "g", "log10(lr)", "log10(Gλ)"]):
        ax.text(j, -0.06, f"min={mins[j]:.2g}", ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.text(j,  1.02, f"max={maxs[j]:.2g}", ha="center", va="bottom", transform=ax.get_xaxis_transform())

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(Gλ)")

    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "viz_parallel_coords_Glambda.png")
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"[saved] {out}")

# -------------------------
# Plot 3: Super-heatmap (g,lr) x (N,L)
# -------------------------
def plot_super_heatmap(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    widths   = d["widths"].astype(int)
    depths   = d["depths"].astype(int)
    gains    = d["gains"].astype(float)
    base_lrs = d["base_lrs"].astype(float)
    G = d["G_lambda_map"].astype(float)  # [g, lr, L, N]

    # Build row labels (g, lr) and col labels (N, L)
    row_pairs = [(g, lr) for g in gains for lr in base_lrs]          # len = G*LR
    col_pairs = [(N, L) for L in depths for N in widths]            # len = D*W

    M = np.full((len(row_pairs), len(col_pairs)), np.nan, dtype=float)

    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            r = gi * len(base_lrs) + li
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    c = di * len(widths) + wi
                    val = G[gi, li, di, wi]
                    M[r, c] = np.log10(val) if (np.isfinite(val) and val > 0) else np.nan

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", origin="lower")

    ax.set_title("Super-heatmap: rows=(g,lr), cols=(N,L), color=log10(Gλ)")
    ax.set_xlabel("(N, L) combinations")
    ax.set_ylabel("(g, lr) combinations")

    ax.set_xticks(np.arange(len(col_pairs)))
    ax.set_xticklabels([f"N{N}\nL{L}" for (N, L) in col_pairs], rotation=0)

    ax.set_yticks(np.arange(len(row_pairs)))
    ax.set_yticklabels([f"g={g:g}, lr={lr:g}" for (g, lr) in row_pairs])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log10(G_lambda)")

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "viz_super_heatmap_Glambda.png")
    plt.savefig(out, dpi=240)
    plt.close()
    print(f"[saved] {out}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if not os.path.exists(GRID_STATE):
        raise FileNotFoundError(f"Missing {GRID_STATE}. Put this script next to it.")

    P = load_flat_points(GRID_STATE)

    plot_3d_scatter_with_marginals(P)
    plot_parallel_coordinates(P)
    plot_super_heatmap(GRID_STATE)

    print(f"\nDone. Saved plots to: {PLOT_DIR}/")
