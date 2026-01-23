import os
import numpy as np
import matplotlib.pyplot as plt

RAKA_STATE = "ra_ka_grid_state.npz"
OUT_DIR    = "plots_ra_ka_depth"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Choose what to plot:
#   - If SELECT_GAIN and SELECT_LR are None -> average over all (g, lr)
#   - Otherwise -> plot the slice at that (g, lr)
# ─────────────────────────────────────────────────────────────
SELECT_GAIN = None   # e.g. 1.0
SELECT_LR   = None   # e.g. 0.10

# float matching tolerance for axes stored as float32
RTOL = 1e-5
ATOL = 1e-7

def load_npz(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def find_index_float_axis(axis: np.ndarray, value: float, rtol=RTOL, atol=ATOL):
    axis = np.asarray(axis, dtype=np.float64)
    idx = np.where(np.isclose(axis, value, rtol=rtol, atol=atol))[0]
    return None if idx.size == 0 else int(idx[0])

def make_title_suffix(gains, lrs):
    if SELECT_GAIN is None or SELECT_LR is None:
        return " (avg over g, lr)"
    return f" (g={SELECT_GAIN}, lr={SELECT_LR})"

def plot_lines_vs_depth(Y_depth_by_width: np.ndarray,
                        widths: np.ndarray,
                        depths: np.ndarray,
                        ylabel: str,
                        title: str,
                        out_path: str):
    """
    Y_depth_by_width: shape [n_depth, n_width]
    """
    plt.figure(figsize=(7.2, 5.0))
    for wi, N in enumerate(widths):
        y = Y_depth_by_width[:, wi]
        if not np.any(np.isfinite(y)):
            continue
        plt.plot(depths, y, marker="o", linewidth=2, markersize=4, label=f"N={int(N)}")

    plt.xlabel("Depth L")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()

def main():
    d = load_npz(RAKA_STATE)

    widths = d["widths"].astype(int)         # [n_width]
    depths = d["depths"].astype(int)         # [n_depth]
    gains  = d["gains"].astype(np.float64)   # [n_gain]
    lrs    = d["base_lrs"].astype(np.float64)# [n_lr]

    RA_map = d["RA_map"].astype(np.float64)  # [g, lr, depth, width]
    KA_map = d["KA_map"].astype(np.float64)  # [g, lr, depth, width]
    done   = d.get("done_map", None)
    if done is not None:
        done = done.astype(bool)

    # pick slice or average
    if SELECT_GAIN is None or SELECT_LR is None:
        # average over gain/lr (ignores NaNs)
        RA_dw = np.nanmean(RA_map, axis=(0, 1))  # [depth, width]
        KA_dw = np.nanmean(KA_map, axis=(0, 1))
        tag = "avg_glr"
    else:
        gi = find_index_float_axis(gains, float(SELECT_GAIN))
        li = find_index_float_axis(lrs, float(SELECT_LR))
        if gi is None:
            raise ValueError(f"SELECT_GAIN={SELECT_GAIN} not found in gains axis: {gains}")
        if li is None:
            raise ValueError(f"SELECT_LR={SELECT_LR} not found in base_lrs axis: {lrs}")

        RA_dw = RA_map[gi, li]   # [depth, width]
        KA_dw = KA_map[gi, li]
        tag = f"g{SELECT_GAIN}_lr{SELECT_LR}".replace(".", "p")

    suffix = make_title_suffix(gains, lrs)

    # Plot RA vs depth, lines by width
    plot_lines_vs_depth(
        RA_dw, widths, depths,
        ylabel="RA (linear CKA)",
        title="RA vs Depth (colored by width N)" + suffix,
        out_path=os.path.join(OUT_DIR, f"RA_vs_depth_by_width_{tag}.png")
    )

    # Plot KA vs depth, lines by width
    plot_lines_vs_depth(
        KA_dw, widths, depths,
        ylabel="KA (NTK alignment)",
        title="KA vs Depth (colored by width N)" + suffix,
        out_path=os.path.join(OUT_DIR, f"KA_vs_depth_by_width_{tag}.png")
    )

    print(f"[saved] {OUT_DIR}/RA_vs_depth_by_width_{tag}.png")
    print(f"[saved] {OUT_DIR}/KA_vs_depth_by_width_{tag}.png")

if __name__ == "__main__":
    main()
