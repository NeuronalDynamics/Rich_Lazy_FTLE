# phase2_viz_Glambda_4d_3Dv.py
"""
Simple 3D viewer for the "super heatmap" generated from the phase2 grid state.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # activates 3D projection


def safe_log10_pos(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & (x > 0)
    out[mask] = np.log10(x[mask])
    return out


def load_super_matrix(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}.")
    d = np.load(npz_path, allow_pickle=False)
    widths = d["widths"].astype(int)
    depths = d["depths"].astype(int)
    gains = d["gains"].astype(float)
    base_lrs = d["base_lrs"].astype(float)
    G = d["G_lambda_map"].astype(float)

    row_pairs = [(float(g), float(lr)) for g in gains for lr in base_lrs]
    col_pairs = [(int(N), int(L)) for L in depths for N in widths]

    surface = np.full((len(row_pairs), len(col_pairs)), np.nan, dtype=float)
    for gi, g in enumerate(gains):
        for li, lr in enumerate(base_lrs):
            row = gi * len(base_lrs) + li
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    col = di * len(widths) + wi
                    val = G[gi, li, di, wi]
                    surface[row, col] = np.log10(val) if (np.isfinite(val) and val > 0) else np.nan

    return row_pairs, col_pairs, surface


def plot_3d_super_heatmap(row_pairs, col_pairs, surface):
    if surface.size == 0:
        raise ValueError("Surface data is empty.")

    rows = np.arange(surface.shape[0])
    cols = np.arange(surface.shape[1])
    X, Y = np.meshgrid(cols, rows)  # X -> columns (N,L), Y -> rows (g, lr)

    Z = np.ma.masked_invalid(surface)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
    )

    ax.set_title("Super heatmap surface: (g,lr) × (N,L)")
    ax.set_xlabel("(N, L) index")
    ax.set_ylabel("(g, lr) index")
    ax.set_zlabel("log10(G_lambda)")
    ax.view_init(elev=35, azim=35)

    formatter = FuncFormatter(lambda val, _: f"{int(val)}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # add label text for clarity (avoid overcrowding if many combos)
    ax.set_xticks(cols)
    ax.set_xticklabels([f"N{N}\nL{L}" for (N, L) in col_pairs], rotation=0, fontsize=8)

    ax.set_yticks(rows)
    ax.set_yticklabels([f"g={g:g}\nlr={lr:g}" for (g, lr) in row_pairs], fontsize=8)

    cbar = fig.colorbar(surf, shrink=0.6, pad=0.1)
    cbar.set_label("log10(G_lambda)")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="3D viewer for the phase2 super heatmap.")
    parser.add_argument("--grid-state", default="phase2_grid_state.npz",
                        help="Path to phase2_grid_state.npz")
    args = parser.parse_args()

    row_pairs, col_pairs, surface = load_super_matrix(args.grid_state)
    plot_3d_super_heatmap(row_pairs, col_pairs, surface)


if __name__ == "__main__":
    main()
