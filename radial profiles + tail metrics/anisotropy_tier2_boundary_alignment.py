#!/usr/bin/env python3
"""
Tier 2: boundary-conditioned alignment statistics.

Within a thin band around the ground-truth decision radius r*, compute:
  • mean radial alignment  A_r^{bdry}
  • variance of radial alignment  Var^{bdry}
  • optional histogram export for future tiers
"""

import os
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGN_DIR = os.path.join(SCRIPT_DIR, "anisotropy_fields")
CIRCLE_DATA = os.path.join(SCRIPT_DIR, "..", "circle_data_seed0.npz")
OUT_CSV = os.path.join(SCRIPT_DIR, "anisotropy_tier2_boundary_stats.csv")

EPSILON = 0.02  # boundary band half-width

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


def load_boundary_radius(path: str) -> float:
    with np.load(path, allow_pickle=False) as d:
        xt = d["xt"]
    radii = np.linalg.norm(xt, axis=1)
    return float(np.median(radii))


def matching_files(config: Dict) -> List[str]:
    gstr = fmt_float(config["gain"])
    lrstr = fmt_float(config["base_lr"])
    prefix = f"align_N{config['N']}_L{config['L']}_g{gstr}_lr{lrstr}_seed"
    files = []
    for fname in os.listdir(ALIGN_DIR):
        if fname.startswith(prefix) and fname.endswith(".npz"):
            files.append(os.path.join(ALIGN_DIR, fname))
    return sorted(files)


def boundary_stats(r: np.ndarray, a_r: np.ndarray, r_star: float, eps: float):
    mask = np.isfinite(r) & np.isfinite(a_r) & (np.abs(r - r_star) < eps)
    vals = a_r[mask]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.var(vals, ddof=0))


def main():
    r_star = load_boundary_radius(CIRCLE_DATA)
    rows = []

    for cfg in CONFIGS:
        files = matching_files(cfg)
        if not files:
            print(f"[warn] No alignment fields for {cfg['label']}")
            continue

        means = []
        vars_ = []
        for path in files:
            with np.load(path, allow_pickle=False) as d:
                r = d["r"]
                a_r = d["a_r"]
            m, v = boundary_stats(r, a_r, r_star, EPSILON)
            if np.isfinite(m):
                means.append(m)
            if np.isfinite(v):
                vars_.append(v)

        if not means:
            continue

        rows.append(dict(
            label=cfg["label"],
            width=cfg["N"],
            depth=cfg["L"],
            gain=cfg["gain"],
            base_lr=cfg["base_lr"],
            mean=np.mean(means),
            var=np.mean(vars_) if vars_ else float("nan"),
            eps=EPSILON,
        ))

    if not rows:
        print("[warn] No boundary stats computed.")
        return

    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
