#!/usr/bin/env python3
"""
Tier 3: directional coherence via the nematic order tensor.

For each anisotropy field, compute the order parameter
  Q = E[u u^T] - (1/2) I
within either radial bins or a boundary band, and report
  C = ||Q||_F

Outputs per-config CSVs (radial profile + boundary band summary).
"""

import csv
import os
from typing import Dict, List, Sequence

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGN_DIR = os.path.join(SCRIPT_DIR, "anisotropy_fields")
CIRCLE_DATA = os.path.join(SCRIPT_DIR, "..", "circle_data_seed0.npz")

OUT_RADIAL = os.path.join(SCRIPT_DIR, "anisotropy_tier3_coherence_radial.csv")
OUT_BDRY = os.path.join(SCRIPT_DIR, "anisotropy_tier3_coherence_boundary.csv")

NUM_BINS = 100
BBOX_MAX = 1.2
EPSILON = 0.02

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


def nematic_order(u_x: np.ndarray, u_y: np.ndarray):
    if u_x.size == 0:
        return float("nan")
    uu_t = np.stack([u_x, u_y], axis=1)
    cov = uu_t.T @ uu_t / uu_t.shape[0]
    cov -= 0.5 * np.eye(2)
    coherence = float(np.linalg.norm(cov, ord="fro"))
    return coherence


def radial_coherence(r: np.ndarray, ux: np.ndarray, uy: np.ndarray, bin_edges: np.ndarray):
    flat_r = r.reshape(-1)
    flat_mask = np.isfinite(flat_r)
    flat_r = flat_r[flat_mask]
    ux_flat = ux.reshape(-1)[flat_mask]
    uy_flat = uy.reshape(-1)[flat_mask]

    bins = np.digitize(flat_r, bin_edges) - 1
    num_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    coherence = np.full(num_bins, np.nan)

    for b in range(num_bins):
        idx = bins == b
        if not np.any(idx):
            continue
        coherence[b] = nematic_order(ux_flat[idx], uy_flat[idx])
    return centers, coherence


def boundary_coherence(r: np.ndarray, ux: np.ndarray, uy: np.ndarray, r_star: float, eps: float):
    mask = np.isfinite(r) & (np.abs(r - r_star) < eps)
    return nematic_order(ux[mask], uy[mask])


def aggregate_radial(config: Dict, bin_edges: np.ndarray):
    files = matching_files(config)
    if not files:
        print(f"[warn] No alignment fields for {config['label']}")
        return None

    centers = None
    coh_list = []

    for path in files:
        with np.load(path, allow_pickle=False) as d:
            r = d["r"]
            ux = d["u_x"]
            uy = d["u_y"]
        c, coh = radial_coherence(r, ux, uy, bin_edges)
        if centers is None:
            centers = c
        coh_list.append(coh)

    coh_arr = np.nanmean(np.stack(coh_list, axis=0), axis=0)
    return centers, coh_arr


def aggregate_boundary(config: Dict, r_star: float):
    files = matching_files(config)
    if not files:
        return float("nan")
    vals = []
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            r = d["r"]
            ux = d["u_x"]
            uy = d["u_y"]
        coh = boundary_coherence(r, ux, uy, r_star, EPSILON)
        if np.isfinite(coh):
            vals.append(coh)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def main():
    r_star = load_boundary_radius(CIRCLE_DATA)
    bin_edges = np.linspace(0.0, BBOX_MAX, NUM_BINS + 1)

    radial_rows = []
    boundary_rows = []

    for cfg in CONFIGS:
        stats = aggregate_radial(cfg, bin_edges)
        if stats is not None:
            centers, coh_arr = stats
            for r, val in zip(centers, coh_arr):
                radial_rows.append(dict(
                    label=cfg["label"],
                    width=cfg["N"],
                    depth=cfg["L"],
                    radius=r,
                    coherence=val,
                ))

        coh_bdry = aggregate_boundary(cfg, r_star)
        boundary_rows.append(dict(
            label=cfg["label"],
            width=cfg["N"],
            depth=cfg["L"],
            coherence=coh_bdry,
            eps=EPSILON,
        ))

    if radial_rows:
        with open(OUT_RADIAL, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=radial_rows[0].keys())
            writer.writeheader()
            writer.writerows(radial_rows)
        print(f"[saved] {OUT_RADIAL}")

    if boundary_rows:
        with open(OUT_BDRY, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=boundary_rows[0].keys())
            writer.writeheader()
            writer.writerows(boundary_rows)
        print(f"[saved] {OUT_BDRY}")


if __name__ == "__main__":
    main()
