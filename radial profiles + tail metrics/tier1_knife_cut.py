#!/usr/bin/env python3
"""
Tier I (Intuition): "Knife cut" FTLE profiles along a canonical slice.

For selected (N, L) configurations, load FTLE grids, extract the horizontal
diameter (y = 0), convert to a radial profile r ↦ λ_T, and draw per-seed
curves plus their average. The ground-truth decision boundary radius from
circle_data_seed0.npz is overlaid for context.
"""

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "tier1_knife_cut_plots")
os.makedirs(OUT_DIR, exist_ok=True)

FTLE_DIR = os.path.join(SCRIPT_DIR, "..", "ftle")
FTLE_DIR = os.path.abspath(FTLE_DIR)
FTLE_GRID = 161
BBOX = (-1.2, 1.2)

CIRCLE_DATA = os.path.join(SCRIPT_DIR, "..", "circle_data_seed0.npz")

# Four "corners" in (width, depth) space; adjust gains/lrs as needed.
CONFIGS: Sequence[Dict] = [
    {"name": "N10_L2", "N": 10, "L": 2, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
    {"name": "N10_L12", "N": 10, "L": 12, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
    {"name": "N200_L2", "N": 200, "L": 2, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
    {"name": "N200_L12", "N": 200, "L": 12, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
]


def fmt_float(x: float) -> str:
    """Match phase2 filename formatting (e.g., 0.1 -> '0p1')."""
    s = f"{x:.3g}"
    s = s.replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


def ftle_path(N: int, L: int, gain: float, base_lr: float, seed: int, grid: int = FTLE_GRID) -> str:
    gstr = fmt_float(gain)
    lrstr = fmt_float(base_lr)
    fname = f"ftle_N{N}_L{L}_g{gstr}_lr{lrstr}_seed{seed}_g{grid}.npy"
    return os.path.join(FTLE_DIR, fname)


def load_ftle_grid(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing FTLE grid: {path}")
    arr = np.load(path)
    if arr.shape != (FTLE_GRID, FTLE_GRID):
        raise ValueError(f"Unexpected FTLE grid shape {arr.shape}, expected {(FTLE_GRID, FTLE_GRID)}")
    return arr


def axes_from_bbox(grid: int, bbox: Tuple[float, float]):
    xs = np.linspace(bbox[0], bbox[1], grid)
    ys = np.linspace(bbox[0], bbox[1], grid)
    return xs, ys


def extract_horizontal_profile(ftle: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    y_idx = int(np.argmin(np.abs(ys - 0.0)))
    row = ftle[y_idx, :]

    center = xs.size // 2
    pos_x = xs[center:]
    pos_vals = row[center:]
    neg_vals = row[:center + 1][::-1]

    count = min(pos_vals.size, neg_vals.size)
    pos_vals = pos_vals[:count]
    neg_vals = neg_vals[:count]
    pos_x = pos_x[:count]

    stack = np.stack([pos_vals, neg_vals], axis=0)
    sym_vals = np.nanmean(stack, axis=0)
    r = np.abs(pos_x)
    return r, sym_vals


def load_ground_truth_boundary(path: str) -> float:
    with np.load(path, allow_pickle=False) as d:
        xt = d["xt"]
    radii = np.linalg.norm(xt, axis=1)
    return float(np.median(radii))


def plot_profiles(config: Dict, radii: np.ndarray, profiles: List[np.ndarray], boundary_radius: float):
    if not profiles:
        print(f"[warn] No FTLE profiles found for {config['name']}")
        return
    plt.figure(figsize=(6.5, 4.5))

    colors = plt.cm.Greys(np.linspace(0.4, 0.8, len(profiles)))
    for col, seed, prof in zip(colors, config["seeds"], profiles):
        plt.plot(radii, prof, color=col, linewidth=1.3, alpha=0.8, label=f"seed {seed}")

    avg = np.nanmean(np.stack(profiles, axis=0), axis=0)
    plt.plot(radii, avg, color="tab:blue", linewidth=2.2, label="seed mean")

    plt.axvline(boundary_radius, color="red", linestyle="--", linewidth=1.5, label="GT boundary")

    plt.xlabel("radius r (along y=0)")
    plt.ylabel("FTLE λ_T")
    plt.title(f"Knife-cut FTLE profile | {config['name']} (g={config['gain']}, lr={config['base_lr']})")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, BBOX[1])
    plt.legend(frameon=True, fontsize=9)

    out_path = os.path.join(OUT_DIR, f"knife_cut_{config['name']}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    boundary = load_ground_truth_boundary(CIRCLE_DATA)
    xs, ys = axes_from_bbox(FTLE_GRID, BBOX)

    for cfg in CONFIGS:
        profiles = []
        radii = None
        for seed in cfg["seeds"]:
            path = ftle_path(cfg["N"], cfg["L"], cfg["gain"], cfg["base_lr"], seed)
            if not os.path.exists(path):
                print(f"[skip] Missing FTLE grid: {os.path.basename(path)}")
                continue
            ftle = load_ftle_grid(path)
            r, sym_vals = extract_horizontal_profile(ftle, xs, ys)
            if radii is None:
                radii = r
            profiles.append(sym_vals)

        if radii is None:
            print(f"[warn] Skipping {cfg['name']} (no data).")
            continue
        plot_profiles(cfg, radii, profiles, boundary)


if __name__ == "__main__":
    main()
