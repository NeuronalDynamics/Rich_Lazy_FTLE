#!/usr/bin/env python3
"""
Tier II (Diagnosis): radial FTLE statistics.

For each selected (N, L) configuration:
  • load every FTLE grid (per seed),
  • convert Cartesian samples into polar bins,
  • compute the angular mean m(r) and variance v(r),
  • overlay curves to compare how width/depth redistribute sensitivity.

Outputs two figures (mean + variance) saved under tier2_radial_profiles/.
"""

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FTLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ftle"))
CIRCLE_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "circle_data_seed0.npz"))

OUT_DIR = os.path.join(SCRIPT_DIR, "tier2_radial_profiles")
os.makedirs(OUT_DIR, exist_ok=True)

FTLE_GRID = 161
BBOX = (-1.2, 1.2)
NUM_BINS = 200

CONFIGS: Sequence[Dict] = [
    {"label": "N10_L2", "N": 10, "L": 2, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
    {"label": "N10_L12", "N": 10, "L": 12, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
    {"label": "N200_L2", "N": 200, "L": 2, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
    {"label": "N200_L12", "N": 200, "L": 12, "gain": 1.0, "base_lr": 0.05, "seeds": [0, 1, 2]},
]


def fmt_float(x: float) -> str:
    s = f"{x:.3g}".replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


def ftle_path(N: int, L: int, gain: float, base_lr: float, seed: int) -> str:
    fname = f"ftle_N{N}_L{L}_g{fmt_float(gain)}_lr{fmt_float(base_lr)}_seed{seed}_g{FTLE_GRID}.npy"
    return os.path.join(FTLE_DIR, fname)


def load_ftle(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing FTLE grid: {path}")
    arr = np.load(path)
    if arr.shape != (FTLE_GRID, FTLE_GRID):
        raise ValueError(f"Unexpected grid shape {arr.shape}")
    return arr


def polar_grid(grid: int, bbox: Tuple[float, float]):
    xs = np.linspace(bbox[0], bbox[1], grid)
    ys = np.linspace(bbox[0], bbox[1], grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    radii = np.sqrt(X ** 2 + Y ** 2)
    return radii


def radial_profiles(values: np.ndarray, radii: np.ndarray, bin_edges: np.ndarray):
    flat_vals = values.reshape(-1)
    flat_r = radii.reshape(-1)
    mask = np.isfinite(flat_vals)
    flat_vals = flat_vals[mask]
    flat_r = flat_r[mask]

    bin_ids = np.digitize(flat_r, bin_edges) - 1
    num_bins = len(bin_edges) - 1

    means = np.full(num_bins, np.nan)
    vars_ = np.full(num_bins, np.nan)

    for b in range(num_bins):
        idx = bin_ids == b
        if not np.any(idx):
            continue
        vals = flat_vals[idx]
        means[b] = float(np.mean(vals))
        if vals.size > 1:
            vars_[b] = float(np.var(vals, ddof=0))
        else:
            vars_[b] = 0.0
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, means, vars_


def load_boundary_radius(path: str) -> float:
    with np.load(path, allow_pickle=False) as d:
        xt = d["xt"]
    radii = np.linalg.norm(xt, axis=1)
    return float(np.median(radii))


def aggregate_config(cfg: Dict, radii_grid: np.ndarray, bin_edges: np.ndarray):
    centers = None
    mean_list: List[np.ndarray] = []
    var_list: List[np.ndarray] = []

    for seed in cfg["seeds"]:
        p = ftle_path(cfg["N"], cfg["L"], cfg["gain"], cfg["base_lr"], seed)
        if not os.path.exists(p):
            print(f"[skip] {cfg['label']} seed{seed}: missing {os.path.basename(p)}")
            continue
        ftle = load_ftle(p)
        r, m, v = radial_profiles(ftle, radii_grid, bin_edges)
        if centers is None:
            centers = r
        mean_list.append(m)
        var_list.append(v)

    if not mean_list:
        print(f"[warn] No FTLE grids available for {cfg['label']}")
        return None

    mean_arr = np.nanmean(np.stack(mean_list, axis=0), axis=0)
    var_arr = np.nanmean(np.stack(var_list, axis=0), axis=0)
    return centers, mean_arr, var_arr


def plot_profiles(profiles: Dict[str, Tuple[np.ndarray, np.ndarray]], boundary_radius: float, metric: str, ylabel: str):
    plt.figure(figsize=(7.5, 4.6))
    for label, (centers, vals) in profiles.items():
        plt.plot(centers, vals, linewidth=2.0, label=label)
    plt.axvline(boundary_radius, color="black", linestyle="--", linewidth=1.5, label="GT boundary")
    plt.xlabel("radius r")
    plt.ylabel(ylabel)
    plt.title(f"Radial {metric} profile")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=9)
    plt.xlim(0.0, BBOX[1])
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{metric}_profiles.png")
    plt.savefig(out_path, dpi=240)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    boundary = load_boundary_radius(CIRCLE_DATA)
    radii_grid = polar_grid(FTLE_GRID, BBOX)
    bin_edges = np.linspace(0.0, BBOX[1], NUM_BINS + 1)

    mean_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    var_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for cfg in CONFIGS:
        stats = aggregate_config(cfg, radii_grid, bin_edges)
        if stats is None:
            continue
        centers, mean_arr, var_arr = stats
        mean_profiles[cfg["label"]] = (centers, mean_arr)
        var_profiles[cfg["label"]] = (centers, var_arr)

    if mean_profiles:
        plot_profiles(mean_profiles, boundary, "mean", "m(r) = E[λ_T]")
    if var_profiles:
        plot_profiles(var_profiles, boundary, "variance", "v(r) = Var[λ_T]")


if __name__ == "__main__":
    main()
