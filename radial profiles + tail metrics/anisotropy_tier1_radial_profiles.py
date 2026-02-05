#!/usr/bin/env python3
"""
Tier 1 (directional): radial alignment profiles from cached anisotropy fields.

For each configuration, load align_N*_*.npz files, bin the radial alignment
a_r(r, θ) by radius, and compute:
  • mean( a_r ) vs r
  • var( a_r ) vs r

Outputs plots under anisotropy_tier1_radial_profiles/ for cross-config comparison.
"""

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGN_DIR = os.path.join(SCRIPT_DIR, "anisotropy_fields")
OUT_DIR = os.path.join(SCRIPT_DIR, "anisotropy_tier1_radial_profiles")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_BINS = 200
BBOX_MAX = 1.2

# same four corner configs
CONFIGS: Sequence[Dict] = [
    {"label": "N10_L2", "N": 10, "L": 2, "gain": 1.0, "base_lr": 0.05},
    {"label": "N10_L12", "N": 10, "L": 12, "gain": 1.0, "base_lr": 0.05},
    {"label": "N200_L2", "N": 200, "L": 2, "gain": 1.0, "base_lr": 0.05},
    {"label": "N200_L12", "N": 200, "L": 12, "gain": 1.0, "base_lr": 0.05},
]


def fmt_float(x: float) -> str:
    s = f"{x:.3g}".replace(".", "p")
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


def load_alignment_npz(path: str):
    with np.load(path, allow_pickle=False) as d:
        r = d["r"]
        a_r = d["a_r"]
    return r, a_r


def matching_files(config: Dict) -> List[str]:
    gstr = fmt_float(config["gain"])
    lrstr = fmt_float(config["base_lr"])
    prefix = f"align_N{config['N']}_L{config['L']}_g{gstr}_lr{lrstr}_seed"
    files = []
    for fname in os.listdir(ALIGN_DIR):
        if fname.startswith(prefix) and fname.endswith(".npz"):
            files.append(os.path.join(ALIGN_DIR, fname))
    return sorted(files)


def radial_stats(r: np.ndarray, a_r: np.ndarray, bin_edges: np.ndarray):
    flat_r = r.reshape(-1)
    flat_a = a_r.reshape(-1)
    mask = np.isfinite(flat_r) & np.isfinite(flat_a)
    flat_r = flat_r[mask]
    flat_a = flat_a[mask]

    bins = np.digitize(flat_r, bin_edges) - 1
    num_bins = len(bin_edges) - 1
    means = np.full(num_bins, np.nan)
    vars_ = np.full(num_bins, np.nan)

    for b in range(num_bins):
        idx = bins == b
        if not np.any(idx):
            continue
        vals = flat_a[idx]
        means[b] = float(np.mean(vals))
        vars_[b] = float(np.var(vals, ddof=0)) if vals.size > 1 else 0.0
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, means, vars_


def aggregate_config(config: Dict, bin_edges: np.ndarray):
    files = matching_files(config)
    if not files:
        print(f"[warn] No alignment fields for {config['label']}")
        return None
    centers = None
    mean_list = []
    var_list = []

    for path in files:
        r, a_r = load_alignment_npz(path)
        c, m, v = radial_stats(r, a_r, bin_edges)
        if centers is None:
            centers = c
        mean_list.append(m)
        var_list.append(v)

    mean_arr = np.nanmean(np.stack(mean_list, axis=0), axis=0)
    var_arr = np.nanmean(np.stack(var_list, axis=0), axis=0)
    return centers, mean_arr, var_arr


def plot_profiles(profiles: Dict[str, Tuple[np.ndarray, np.ndarray]], ylabel: str, metric: str):
    plt.figure(figsize=(7.5, 4.6))
    for label, (centers, vals) in profiles.items():
        plt.plot(centers, vals, linewidth=2.0, label=label)
    plt.xlabel("radius r")
    plt.ylabel(ylabel)
    plt.title(f"Directional {metric} profile")
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, BBOX_MAX)
    plt.legend(frameon=True, fontsize=9)
    out_path = os.path.join(OUT_DIR, f"alignment_{metric}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    bin_edges = np.linspace(0.0, BBOX_MAX, NUM_BINS + 1)
    mean_profiles = {}
    var_profiles = {}

    for cfg in CONFIGS:
        stats = aggregate_config(cfg, bin_edges)
        if stats is None:
            continue
        centers, mean_arr, var_arr = stats
        mean_profiles[cfg["label"]] = (centers, mean_arr)
        var_profiles[cfg["label"]] = (centers, var_arr)

    if mean_profiles:
        plot_profiles(mean_profiles, r"$\bar A_r(r)$", "mean")
    if var_profiles:
        plot_profiles(var_profiles, r"$\mathrm{Var}[A_r(r)]$", "variance")


if __name__ == "__main__":
    main()
