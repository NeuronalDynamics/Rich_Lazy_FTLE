#!/usr/bin/env python3
"""
Export (gain, lr, width, depth, G_lambda, log10(G_lambda)) tuples from
phase2_grid_state.npz into a CSV for downstream numerical analysis.
"""

import argparse
import csv
import os
from typing import List, Dict, Any

import numpy as np


def safe_log10_pos(x: float) -> float:
    if np.isfinite(x) and x > 0:
        return float(np.log10(x))
    return float("nan")


def flatten_grid(npz_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Grid file {npz_path!r} not found.")

    with np.load(npz_path, allow_pickle=False) as d:
        widths = d["widths"].astype(int)
        depths = d["depths"].astype(int)
        gains = d["gains"].astype(float)
        base_lrs = d["base_lrs"].astype(float)
        G = d["G_lambda_map"].astype(float)
        done = d["done_map"].astype(bool) if "done_map" in d else None

    rows: List[Dict[str, Any]] = []
    for gi, gain in enumerate(gains):
        for li, base_lr in enumerate(base_lrs):
            for di, depth in enumerate(depths):
                for wi, width in enumerate(widths):
                    if done is not None and not done[gi, li, di, wi]:
                        continue
                    value = float(G[gi, li, di, wi])
                    rows.append(
                        dict(
                            gain=float(gain),
                            base_lr=float(base_lr),
                            N_L_comb=f"N{int(width)}-L{int(depth)}",
                            G_lambda=value,
                            G_lambda_log10=safe_log10_pos(value),
                        )
                    )
    return rows


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        raise RuntimeError("No rows extracted from grid; nothing to write.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["gain", "base_lr", "N_L_comb", "G_lambda", "G_lambda_log10"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_path}  ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Extract N/L vs G_lambda entries into CSV.")
    parser.add_argument("--grid-state", default="phase2_grid_state.npz",
                        help="Path to phase2_grid_state.npz")
    parser.add_argument("--out", default=os.path.join("data_extractor", "NL_Glambda_trend.csv"),
                        help="Output CSV path (defaults beside this script).")
    args = parser.parse_args()

    rows = flatten_grid(args.grid_state)
    save_csv(rows, args.out)


if __name__ == "__main__":
    main()
