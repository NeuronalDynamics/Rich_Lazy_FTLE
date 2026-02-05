#!/usr/bin/env python3
"""
Tier III (Measurement): scalar summaries derived from FTLE radial profiles.

For each selected (N, L):
  1. Boundary spike height  H = m(r*) - m(r_center)
  2. Boundary sharpness     (FWHM around r*)
  3. Tail statistic         T = Q_q(λ_T) - Q_0.5(λ_T)

Results are written to a CSV inside this directory for downstream analysis.
"""

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FTLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "ftle"))
CIRCLE_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "circle_data_seed0.npz"))

OUT_CSV = os.path.join(SCRIPT_DIR, "tier3_scalar_metrics.csv")

FTLE_GRID = 161
BBOX = (-1.2, 1.2)
NUM_BINS = 200
TAIL_QUANTILE = 0.95  # can swap to 0.99 for thinner ridges

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
        raise ValueError(f"Unexpected FTLE grid shape {arr.shape}")
    return arr


def polar_radii(grid: int, bbox: Tuple[float, float]):
    xs = np.linspace(bbox[0], bbox[1], grid)
    ys = np.linspace(bbox[0], bbox[1], grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.sqrt(X ** 2 + Y ** 2)


def radial_profile(values: np.ndarray, radii: np.ndarray, bin_edges: np.ndarray):
    flat_vals = values.reshape(-1)
    flat_r = radii.reshape(-1)
    mask = np.isfinite(flat_vals)
    flat_vals = flat_vals[mask]
    flat_r = flat_r[mask]

    bin_ids = np.digitize(flat_r, bin_edges) - 1
    num_bins = len(bin_edges) - 1
    means = np.full(num_bins, np.nan)

    for b in range(num_bins):
        idx = bin_ids == b
        if not np.any(idx):
            continue
        means[b] = float(np.mean(flat_vals[idx]))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, means


def combine_profiles(config: Dict, radii_grid: np.ndarray, bin_edges: np.ndarray):
    profiles: List[np.ndarray] = []
    centers = None
    ftle_values: List[np.ndarray] = []

    for seed in config["seeds"]:
        path = ftle_path(config["N"], config["L"], config["gain"], config["base_lr"], seed)
        if not os.path.exists(path):
            print(f"[skip] {config['label']} seed{seed}: missing {os.path.basename(path)}")
            continue
        ftle = load_ftle(path)
        c, mean_vals = radial_profile(ftle, radii_grid, bin_edges)
        if centers is None:
            centers = c
        profiles.append(mean_vals)
        ftle_values.append(ftle)

    if not profiles:
        return None, None, None

    mean_profile = np.nanmean(np.stack(profiles, axis=0), axis=0)
    flat_vals = np.concatenate([arr.reshape(-1) for arr in ftle_values])
    flat_vals = flat_vals[np.isfinite(flat_vals)]
    return centers, mean_profile, flat_vals


def load_boundary_radius(path: str) -> float:
    with np.load(path, allow_pickle=False) as d:
        xt = d["xt"]
    radii = np.linalg.norm(xt, axis=1)
    return float(np.median(radii))


def first_finite(values: np.ndarray) -> int:
    for i, val in enumerate(values):
        if np.isfinite(val):
            return i
    return -1


def nearest_index(array: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(array - value)))


def compute_fwhm(centers: np.ndarray, values: np.ndarray, base_val: float, peak_val: float, peak_idx: int) -> float:
    if not np.isfinite(base_val) or not np.isfinite(peak_val):
        return float("nan")
    if peak_val <= base_val:
        return 0.0
    half = base_val + 0.5 * (peak_val - base_val)

    def interpolate(i_left: int, i_right: int) -> float:
        v_left = values[i_left]
        v_right = values[i_right]
        r_left = centers[i_left]
        r_right = centers[i_right]
        if not np.isfinite(v_left):
            return r_right
        if not np.isfinite(v_right) or v_right == v_left:
            return r_right
        t = (half - v_left) / (v_right - v_left)
        t = np.clip(t, 0.0, 1.0)
        return r_left + t * (r_right - r_left)

    left = peak_idx
    while left > 0 and np.isfinite(values[left]) and values[left] >= half:
        left -= 1
    left_cross = centers[0] if left == 0 and values[left] >= half else interpolate(left, min(left + 1, len(centers) - 1))

    right = peak_idx
    while right < len(values) - 1 and np.isfinite(values[right]) and values[right] >= half:
        right += 1
    right_cross = centers[-1] if right == len(values) - 1 and values[right] >= half else interpolate(max(right - 1, 0), right)

    width = max(0.0, right_cross - left_cross)
    return float(width)


def compute_metrics(config: Dict, centers: np.ndarray, mean_profile: np.ndarray, flat_vals: np.ndarray, boundary_r: float):
    if centers is None or mean_profile is None:
        return None

    center_idx = first_finite(mean_profile)
    if center_idx < 0:
        return None
    boundary_idx = nearest_index(centers, boundary_r)

    m_center = mean_profile[center_idx]
    m_boundary = mean_profile[boundary_idx]
    spike_height = float(m_boundary - m_center) if np.isfinite(m_boundary) and np.isfinite(m_center) else float("nan")
    sharpness = compute_fwhm(centers, mean_profile, m_center, m_boundary, boundary_idx)

    if flat_vals.size == 0:
        tail_stat = float("nan")
    else:
        q50 = float(np.percentile(flat_vals, 50))
        q_tail = float(np.percentile(flat_vals, TAIL_QUANTILE * 100))
        tail_stat = q_tail - q50

    return dict(
        label=config["label"],
        width=config["N"],
        depth=config["L"],
        gain=config["gain"],
        base_lr=config["base_lr"],
        spike_height=spike_height,
        fwhm=sharpness,
        tail_stat=tail_stat,
    )


def save_csv(rows: List[Dict]):
    if not rows:
        print("[warn] No metrics computed; CSV not written.")
        return
    fieldnames = ["label", "width", "depth", "gain", "base_lr", "spike_height", "fwhm", "tail_stat"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {OUT_CSV}")


def main():
    global TAIL_QUANTILE
    parser = argparse.ArgumentParser(description="Compute Tier III FTLE scalar metrics.")
    parser.add_argument("--quantile", type=float, default=TAIL_QUANTILE,
                        help="Tail quantile (default 0.95). Use 0.99 for thinner ridges.")
    args = parser.parse_args()
    TAIL_QUANTILE = args.quantile

    boundary = load_boundary_radius(CIRCLE_DATA)
    radii_grid = polar_radii(FTLE_GRID, BBOX)
    bin_edges = np.linspace(0.0, BBOX[1], NUM_BINS + 1)

    rows: List[Dict] = []
    for cfg in CONFIGS:
        centers, mean_profile, flat_vals = combine_profiles(cfg, radii_grid, bin_edges)
        if centers is None:
            print(f"[warn] Skipping {cfg['label']} (no FTLE data).")
            continue
        metrics = compute_metrics(cfg, centers, mean_profile, flat_vals, boundary)
        if metrics is not None:
            rows.append(metrics)

    save_csv(rows)


if __name__ == "__main__":
    main()
