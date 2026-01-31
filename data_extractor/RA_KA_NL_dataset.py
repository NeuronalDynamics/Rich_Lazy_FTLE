#!/usr/bin/env python3
"""
Export averaged RA/KA values per (width N, depth L) combination from ra_ka_grid_state.npz.

Matches the aggregation used by plot_ra_ka_vs_depth_by_width.py when SELECT_GAIN/SELECT_LR are None:
we average over all (gain, base_lr) slices while ignoring NaNs.
"""

import argparse
import csv
import os
from typing import List, Dict

import numpy as np


def load_ra_ka_grid(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing grid file: {path}")
    with np.load(path, allow_pickle=False) as d:
        widths = d["widths"].astype(int)
        depths = d["depths"].astype(int)
        RA_map = d["RA_map"].astype(np.float64)
        KA_map = d["KA_map"].astype(np.float64)
    return widths, depths, RA_map, KA_map


def average_over_gain_lr(tensor: np.ndarray) -> np.ndarray:
    # tensor shape [gain, lr, depth, width] -> average across first two axes, ignoring NaN
    return np.nanmean(tensor, axis=(0, 1))


def build_rows(widths, depths, RA_avg, KA_avg) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for di, depth in enumerate(depths):
        for wi, width in enumerate(widths):
            ra = float(RA_avg[di, wi])
            ka = float(KA_avg[di, wi])
            if not (np.isfinite(ra) and np.isfinite(ka)):
                continue
            rows.append(
                dict(
                    N_L_comb=f"N{int(width)}-L{int(depth)}",
                    RA=ra,
                    KA=ka,
                )
            )
    return rows


def save_csv(rows: List[Dict[str, float]], out_path: str):
    if not rows:
        raise RuntimeError("No finite RA/KA entries found to export.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N_L_comb", "RA", "KA"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Export RA/KA averages per (N,L) combination.")
    parser.add_argument("--grid-state", default="ra_ka_grid_state.npz",
                        help="Path to ra_ka_grid_state.npz")
    parser.add_argument("--out", default=os.path.join("data_extractor", "RA_KA_NL_dataset.csv"),
                        help="Output CSV path.")
    args = parser.parse_args()

    widths, depths, RA_map, KA_map = load_ra_ka_grid(args.grid_state)
    RA_avg = average_over_gain_lr(RA_map)
    KA_avg = average_over_gain_lr(KA_map)
    rows = build_rows(widths, depths, RA_avg, KA_avg)
    save_csv(rows, args.out)


if __name__ == "__main__":
    main()
