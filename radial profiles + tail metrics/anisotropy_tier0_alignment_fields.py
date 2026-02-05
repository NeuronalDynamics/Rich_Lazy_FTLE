#!/usr/bin/env python3
"""
Tier 0: build and cache radial/tangential alignment fields for every FTLE grid.

For each FTLE configuration (N, L, gain, lr, seed) we:
  • load the trained checkpoint (rk_ckpts_v4),
  • evaluate the hidden Jacobian on the 2-D grid,
  • extract the top right-singular vector u(x),
  • project u(x) onto the radial frame (e_r, e_theta),
  • save radius + alignment maps to anisotropy_fields/.

This data is sign-invariant and decouples direction geometry from scalar FTLE.
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ra_ka_best_method_accstop import FC, ckpt_path, fmt_float

try:
    from torch.func import jvp
    _HAS_JVP = True
except Exception:
    _HAS_JVP = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BBOX = (-1.2, 1.2)
FTLE_GRID = 161
DEFAULT_BATCH = 4096

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FTLE_DIR = os.path.join(ROOT_DIR, "ftle")
CKPT_DIR = os.path.join(ROOT_DIR, "rk_ckpts_v4")
OUT_DIR = os.path.join(ROOT_DIR, "anisotropy_fields")
os.makedirs(OUT_DIR, exist_ok=True)

FNAME_RE = re.compile(
    r"ftle_N(?P<N>\d+)_L(?P<L>\d+)_g(?P<gain>[A-Za-z0-9p]+)_lr(?P<lr>[A-Za-z0-9p]+)_seed(?P<seed>\d+)_g(?P<grid>\d+)\.npy"
)


def parse_fmt_float(token: str) -> float:
    token = token.replace("p", ".")
    if token.startswith("m"):
        token = "-" + token[1:]
    return float(token)


def list_ftle_configs(ftle_dir: str, grid: int) -> List[Dict]:
    cfgs: List[Dict] = []
    for fname in os.listdir(ftle_dir):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        if int(m.group("grid")) != grid:
            continue
        cfgs.append(
            dict(
                N=int(m.group("N")),
                L=int(m.group("L")),
                gain=parse_fmt_float(m.group("gain")),
                base_lr=parse_fmt_float(m.group("lr")),
                seed=int(m.group("seed")),
                ftle_path=os.path.join(ftle_dir, fname),
            )
        )
    cfgs.sort(key=lambda d: (d["N"], d["L"], d["gain"], d["base_lr"], d["seed"]))
    return cfgs


def load_checkpoint(N: int, L: int, gain: float, base_lr: float, seed: int) -> FC:
    path = ckpt_path(N, L, gain, base_lr, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    net = FC(N, L, gain=gain).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        net.load_state_dict(state["state_dict"])
    else:
        net.load_state_dict(state)
    net.eval()
    return net


def grid_points(grid: int, bbox: Tuple[float, float], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(*bbox, grid, device=device)
    ys = torch.linspace(*bbox, grid, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)
    return Xg, Yg, pts


def compute_alignment_fields_jvp(net: FC, depth: int, grid: int, bbox, batch_size: int):
    Xg, Yg, pts = grid_points(grid, bbox, DEVICE)
    total = pts.shape[0]

    vec_x = torch.empty(total, device=DEVICE, dtype=torch.float32)
    vec_y = torch.empty(total, device=DEVICE, dtype=torch.float32)

    def hidden(z):
        return net(z, hid=True)

    start = 0
    bs = batch_size
    while start < total:
        xb = pts[start:start + bs]
        v1 = torch.zeros_like(xb); v1[:, 0] = 1.0
        v2 = torch.zeros_like(xb); v2[:, 1] = 1.0

        _, j1 = jvp(hidden, (xb,), (v1,))
        _, j2 = jvp(hidden, (xb,), (v2,))

        a = (j1 * j1).sum(dim=1)
        b = (j2 * j2).sum(dim=1)
        c = (j1 * j2).sum(dim=1)

        disc = torch.sqrt(torch.clamp((a - b) * (a - b) + 4.0 * c * c, min=0.0))
        eigmax = 0.5 * ((a + b) + disc)

        vx = c.clone()
        vy = eigmax - a

        tiny = (vx.abs() + vy.abs()) < 1e-10
        choose_x = a >= b
        vx = torch.where(tiny, torch.where(choose_x, torch.ones_like(vx), torch.zeros_like(vx)), vx)
        vy = torch.where(tiny, torch.where(choose_x, torch.zeros_like(vy), torch.ones_like(vy)), vy)

        norm = torch.sqrt(vx * vx + vy * vy).clamp_min(1e-12)
        vx = vx / norm
        vy = vy / norm

        vec_x[start:start + xb.shape[0]] = vx.to(torch.float32)
        vec_y[start:start + xb.shape[0]] = vy.to(torch.float32)

        start += xb.shape[0]

    return (
        Xg.detach().cpu().numpy().astype(np.float32),
        Yg.detach().cpu().numpy().astype(np.float32),
        vec_x.reshape(grid, grid).detach().cpu().numpy().astype(np.float32),
        vec_y.reshape(grid, grid).detach().cpu().numpy().astype(np.float32),
    )


def compute_alignment_fields_autograd(net: FC, depth: int, grid: int, bbox):
    xs = torch.linspace(*bbox, grid, device=DEVICE)
    ys = torch.linspace(*bbox, grid, device=DEVICE)
    vec_x = np.full((grid, grid), np.nan, dtype=np.float32)
    vec_y = np.full((grid, grid), np.nan, dtype=np.float32)

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            point = torch.tensor([xv, yv], dtype=torch.float32, device=DEVICE, requires_grad=True)
            with torch.enable_grad():
                J = torch.autograd.functional.jacobian(
                    lambda z: net(z.unsqueeze(0), hid=True).squeeze(0), point,
                    create_graph=False
                )
            if not torch.isfinite(J).all():
                continue
            JTJ = J.T @ J
            try:
                eigvals, eigvecs = torch.linalg.eigh(JTJ.cpu())
            except RuntimeError:
                continue
            v = eigvecs[:, -1]
            vec_x[j, i] = float(v[0])
            vec_y[j, i] = float(v[1])

    Xg, Yg = torch.meshgrid(xs.cpu().numpy(), ys.cpu().numpy(), indexing="xy")
    return Xg.astype(np.float32), Yg.astype(np.float32), vec_x, vec_y


def compute_alignment(net: FC, depth: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if _HAS_JVP:
        try:
            return compute_alignment_fields_jvp(net, depth, FTLE_GRID, BBOX, batch_size)
        except Exception as e:
            print(f"[warn] torch.func.jvp failed ({e}); falling back to autograd.")
    return compute_alignment_fields_autograd(net, depth, FTLE_GRID, BBOX)


def radial_alignment(Xg: np.ndarray, Yg: np.ndarray, ux: np.ndarray, uy: np.ndarray):
    r = np.sqrt(Xg ** 2 + Yg ** 2).astype(np.float32)
    eps = 1e-8
    er_x = np.divide(Xg, r, out=np.zeros_like(Xg), where=r > eps)
    er_y = np.divide(Yg, r, out=np.zeros_like(Yg), where=r > eps)
    dot = ux * er_x + uy * er_y
    a_r = np.clip(dot * dot, 0.0, 1.0).astype(np.float32)
    a_r[r <= eps] = np.nan
    a_theta = (1.0 - a_r).astype(np.float32)
    return r, a_r, a_theta


def save_alignment(config: Dict, r: np.ndarray, a_r: np.ndarray, a_theta: np.ndarray, ux: np.ndarray, uy: np.ndarray):
    gstr = fmt_float(config["gain"])
    lrstr = fmt_float(config["base_lr"])
    out_path = os.path.join(
        OUT_DIR,
        f"align_N{config['N']}_L{config['L']}_g{gstr}_lr{lrstr}_seed{config['seed']}.npz"
    )
    np.savez_compressed(
        out_path,
        N=np.array(config["N"], dtype=np.int32),
        L=np.array(config["L"], dtype=np.int32),
        gain=np.array(config["gain"], dtype=np.float32),
        base_lr=np.array(config["base_lr"], dtype=np.float32),
        seed=np.array(config["seed"], dtype=np.int32),
        r=r.astype(np.float32),
        a_r=a_r.astype(np.float32),
        a_theta=a_theta.astype(np.float32),
        u_x=ux.astype(np.float32),
        u_y=uy.astype(np.float32),
    )
    print(f"[saved] {out_path}")


def already_done(config: Dict) -> bool:
    gstr = fmt_float(config["gain"])
    lrstr = fmt_float(config["base_lr"])
    path = os.path.join(
        OUT_DIR,
        f"align_N{config['N']}_L{config['L']}_g{gstr}_lr{lrstr}_seed{config['seed']}.npz"
    )
    return os.path.exists(path)


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Tier 0: cache anisotropy alignment fields.")
    parser.add_argument("--ftle-dir", default=FTLE_DIR, help="Directory containing ftle_*.npy grids.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory to store alignment npz files.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for jvp evaluation.")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many configs.")
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    cfgs = list_ftle_configs(args.ftle_dir, FTLE_GRID)
    if args.limit is not None:
        cfgs = cfgs[:args.limit]
    if not cfgs:
        raise RuntimeError(f"No FTLE grids found under {args.ftle_dir}")

    print(f"[info] device: {DEVICE}, torch.func.jvp available: {_HAS_JVP}")
    print(f"[info] processing {len(cfgs)} ftle grids (grid={FTLE_GRID})")

    for cfg in cfgs:
        if not args.force and already_done(cfg):
            continue
        try:
            net = load_checkpoint(cfg["N"], cfg["L"], cfg["gain"], cfg["base_lr"], cfg["seed"])
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue

        Xg, Yg, ux, uy = compute_alignment(net, cfg["L"], args.batch)
        r, a_r, a_theta = radial_alignment(Xg, Yg, ux, uy)
        save_alignment(cfg, r, a_r, a_theta, ux, uy)


if __name__ == "__main__":
    main()
