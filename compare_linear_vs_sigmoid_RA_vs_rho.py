import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# ----------------------------
# INPUT FILES
# ----------------------------
RAKA_STATE  = "ra_ka_grid_state.npz"
PHASE2_STATE = "phase2_grid_state.npz"

OUT_DIR = "plots_fit_compare"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "RA_vs_rho_lambda_linear_vs_sigmoid.png")

# ----------------------------
# OPTIONS
# ----------------------------
DO_KFOLD_CV = True
KFOLD = 5
RANDOM_SEED = 0

# If you want to ignore near-zero regimes (sometimes dominated by noise), set e.g. 0.05
# Otherwise None keeps everything.
ABS_RHO_MIN_FILTER = None  # e.g. 0.05 or None


def load_npz(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def isclose_match(a: float, arr: np.ndarray, rtol=1e-5, atol=1e-7):
    """Return index in arr where arr[idx] ≈ a, else None."""
    idx = np.where(np.isclose(arr.astype(np.float64), float(a), rtol=rtol, atol=atol))[0]
    return None if idx.size == 0 else int(idx[0])


def four_param_logistic(x, a, b, k, x0):
    """
    4-parameter logistic:
      y = a + (b-a) / (1 + exp(-k*(x-x0)))
    a,b: lower/upper asymptotes (we will bound to [-1,1])
    x0: midpoint (bound to [0,1] since RA ∈ [0,1])
    k : slope (can be +/-)
    """
    z = -k * (x - x0)
    z = np.clip(z, -60.0, 60.0)  # avoid overflow
    return a + (b - a) / (1.0 + np.exp(z))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, k_params: int):
    """Return dict of R2, adjR2, RMSE, MAE, SSE, AIC, BIC."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    resid = y_true - y_pred
    sse = float(np.sum(resid * resid))
    n = int(y_true.size)

    ybar = float(np.mean(y_true))
    sst = float(np.sum((y_true - ybar) ** 2))

    r2 = np.nan
    if sst > 0:
        r2 = 1.0 - (sse / sst)

    # adjusted R^2
    # adjR2 = 1 - (1-R2)*(n-1)/(n-k-1)
    adjr2 = np.nan
    if np.isfinite(r2) and (n - k_params - 1) > 0:
        adjr2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k_params - 1)

    rmse = float(np.sqrt(np.mean(resid * resid)))
    mae  = float(np.mean(np.abs(resid)))

    # AIC / BIC (Gaussian errors) using SSE
    # AIC = n ln(SSE/n) + 2k
    # BIC = n ln(SSE/n) + k ln(n)
    eps = 1e-12
    aic = float(n * np.log((sse + eps) / n) + 2.0 * k_params)
    bic = float(n * np.log((sse + eps) / n) + k_params * np.log(n))

    return dict(
        n=n, k=k_params,
        SSE=sse, RMSE=rmse, MAE=mae,
        R2=r2, adjR2=adjr2,
        AIC=aic, BIC=bic
    )


def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds


def fit_linear(x: np.ndarray, y: np.ndarray):
    # y = m x + c
    m, c = np.polyfit(x, y, 1)
    return float(m), float(c)


def predict_linear(x: np.ndarray, m: float, c: float):
    return m * x + c


def fit_sigmoid(x: np.ndarray, y: np.ndarray):
    """
    Fit 4-parameter logistic with bounded parameters:
      a,b in [-1,1]
      k in [-200,200]
      x0 in [0,1]
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # initial guesses
    a0 = float(np.nanpercentile(y, 5))
    b0 = float(np.nanpercentile(y, 95))

    # keep within [-1,1]
    a0 = float(np.clip(a0, -1.0, 1.0))
    b0 = float(np.clip(b0, -1.0, 1.0))

    # slope guess from linear fit
    m, c = np.polyfit(x, y, 1)
    k0 = float(np.clip(10.0 * np.sign(m if m != 0 else -1.0), -50.0, 50.0))

    x0_0 = float(np.median(x))

    p0 = np.array([a0, b0, k0, x0_0], dtype=np.float64)

    bounds_lo = np.array([-1.0, -1.0, -200.0, 0.0], dtype=np.float64)
    bounds_hi = np.array([ 1.0,  1.0,  200.0, 1.0], dtype=np.float64)

    popt, pcov = curve_fit(
        four_param_logistic,
        x, y,
        p0=p0,
        bounds=(bounds_lo, bounds_hi),
        maxfev=20000
    )
    return popt, pcov


def build_ra_rho_pairs():
    ra = load_npz(RAKA_STATE)
    ph = load_npz(PHASE2_STATE)

    # axes
    W_ra = ra["widths"].astype(int)
    D_ra = ra["depths"].astype(int)
    G_ra = ra["gains"].astype(np.float64)
    LR_ra = ra["base_lrs"].astype(np.float64)

    W_ph = ph["widths"].astype(int)
    D_ph = ph["depths"].astype(int)
    G_ph = ph["gains"].astype(np.float64)
    LR_ph = ph["base_lrs"].astype(np.float64)

    # maps
    RA_map = ra["RA_map"].astype(np.float64)  # [g, lr, depth, width]
    done_ra = ra["done_map"].astype(bool)

    rho_map = ph["rho_lambda_map"].astype(np.float64)  # [g, lr, depth, width]
    done_ph = ph["done_map"].astype(bool)

    # intersection over axes (robust matching)
    common_W = sorted(set(W_ra.tolist()).intersection(set(W_ph.tolist())))
    common_D = sorted(set(D_ra.tolist()).intersection(set(D_ph.tolist())))

    # float axes: match by isclose
    common_G = []
    for g in G_ra:
        if isclose_match(g, G_ph) is not None:
            common_G.append(float(g))

    common_LR = []
    for lr in LR_ra:
        if isclose_match(lr, LR_ph) is not None:
            common_LR.append(float(lr))

    if len(common_W) == 0 or len(common_D) == 0 or len(common_G) == 0 or len(common_LR) == 0:
        raise RuntimeError("No overlapping axes between RA/KA state and phase2 state.")

    xs = []
    ys = []
    meta = []  # optional: store (N,L,g,lr)

    for g in common_G:
        gi_ra = isclose_match(g, G_ra)
        gi_ph = isclose_match(g, G_ph)
        for lr in common_LR:
            li_ra = isclose_match(lr, LR_ra)
            li_ph = isclose_match(lr, LR_ph)
            for L in common_D:
                di_ra = int(np.where(D_ra == L)[0][0])
                di_ph = int(np.where(D_ph == L)[0][0])
                for N in common_W:
                    wi_ra = int(np.where(W_ra == N)[0][0])
                    wi_ph = int(np.where(W_ph == N)[0][0])

                    if not (done_ra[gi_ra, li_ra, di_ra, wi_ra] and done_ph[gi_ph, li_ph, di_ph, wi_ph]):
                        continue

                    ra_val = RA_map[gi_ra, li_ra, di_ra, wi_ra]
                    rho_val = rho_map[gi_ph, li_ph, di_ph, wi_ph]

                    if not (np.isfinite(ra_val) and np.isfinite(rho_val)):
                        continue

                    xs.append(float(ra_val))
                    ys.append(float(rho_val))
                    meta.append((int(N), int(L), float(g), float(lr)))

    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)

    # optional filter
    if ABS_RHO_MIN_FILTER is not None:
        m = np.abs(y) >= float(ABS_RHO_MIN_FILTER)
        x = x[m]
        y = y[m]
        meta = [meta[i] for i in np.where(m)[0].tolist()]

    return x, y, meta


def main():
    x, y, meta = build_ra_rho_pairs()

    if x.size < 10:
        raise RuntimeError(f"Too few points to fit reliably: n={x.size}")

    print(f"[data] paired samples: n={x.size}")
    print(f"[data] RA range:  min={x.min():.4f}, med={np.median(x):.4f}, max={x.max():.4f}")
    print(f"[data] rho range: min={y.min():.4f}, med={np.median(y):.4f}, max={y.max():.4f}")

    # ----------------------------
    # Fit linear
    # ----------------------------
    m, c = fit_linear(x, y)
    yhat_lin = predict_linear(x, m, c)
    met_lin = regression_metrics(y, yhat_lin, k_params=2)

    # ----------------------------
    # Fit sigmoid
    # ----------------------------
    sigmoid_ok = True
    try:
        popt, pcov = fit_sigmoid(x, y)
        yhat_sig = four_param_logistic(x, *popt)
        met_sig = regression_metrics(y, yhat_sig, k_params=4)
    except Exception as e:
        sigmoid_ok = False
        popt = None
        met_sig = None
        print("[warn] sigmoid fit failed:", repr(e))

    # ----------------------------
    # K-fold CV RMSE (recommended)
    # ----------------------------
    cv_lin = None
    cv_sig = None
    if DO_KFOLD_CV:
        folds = kfold_indices(x.size, KFOLD, RANDOM_SEED)

        rmses_lin = []
        rmses_sig = []

        for fi in range(KFOLD):
            test_idx = folds[fi]
            train_idx = np.concatenate([folds[j] for j in range(KFOLD) if j != fi])

            xtr, ytr = x[train_idx], y[train_idx]
            xte, yte = x[test_idx], y[test_idx]

            # linear
            m_i, c_i = fit_linear(xtr, ytr)
            ypred = predict_linear(xte, m_i, c_i)
            rmses_lin.append(np.sqrt(np.mean((yte - ypred) ** 2)))

            # sigmoid
            if sigmoid_ok:
                try:
                    popt_i, _ = fit_sigmoid(xtr, ytr)
                    ypred_s = four_param_logistic(xte, *popt_i)
                    rmses_sig.append(np.sqrt(np.mean((yte - ypred_s) ** 2)))
                except Exception:
                    # if a fold fails, mark it as NaN (don’t crash)
                    rmses_sig.append(np.nan)

        cv_lin = float(np.mean(rmses_lin))
        cv_sig = float(np.nanmean(rmses_sig)) if sigmoid_ok else None

    # ----------------------------
    # Print comparison
    # ----------------------------
    print("\n================ FIT COMPARISON ================")
    print("Linear:   y = m*RA + c")
    print(f"  m={m:.6f}, c={c:.6f}")
    print(f"  R2={met_lin['R2']:.4f}  adjR2={met_lin['adjR2']:.4f}  RMSE={met_lin['RMSE']:.4f}  MAE={met_lin['MAE']:.4f}")
    print(f"  AIC={met_lin['AIC']:.2f}  BIC={met_lin['BIC']:.2f}")

    if sigmoid_ok:
        a, b, k, x0 = popt
        print("\nSigmoid: y = a + (b-a)/(1 + exp(-k*(RA-x0)))")
        print(f"  a={a:.6f}, b={b:.6f}, k={k:.6f}, x0={x0:.6f}")
        print(f"  R2={met_sig['R2']:.4f}  adjR2={met_sig['adjR2']:.4f}  RMSE={met_sig['RMSE']:.4f}  MAE={met_sig['MAE']:.4f}")
        print(f"  AIC={met_sig['AIC']:.2f}  BIC={met_sig['BIC']:.2f}")

    if DO_KFOLD_CV:
        print("\nK-fold CV (RMSE):")
        print(f"  Linear  CV_RMSE = {cv_lin:.4f}")
        if sigmoid_ok:
            print(f"  Sigmoid CV_RMSE = {cv_sig:.4f}")
        else:
            print("  Sigmoid CV_RMSE = (fit failed)")

    # “winner” hints (multiple criteria)
    print("\n---------------- Decision Hints ----------------")
    # by AIC/BIC (lower is better)
    if sigmoid_ok:
        print(f"AIC winner: {'Sigmoid' if met_sig['AIC'] < met_lin['AIC'] else 'Linear'}")
        print(f"BIC winner: {'Sigmoid' if met_sig['BIC'] < met_lin['BIC'] else 'Linear'}")
    if DO_KFOLD_CV and sigmoid_ok and (cv_sig is not None):
        print(f"CV_RMSE winner: {'Sigmoid' if cv_sig < cv_lin else 'Linear'}")

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(x, y, s=18, alpha=0.7)

    xg = np.linspace(float(np.min(x)), float(np.max(x)), 400)

    yg_lin = predict_linear(xg, m, c)
    plt.plot(xg, yg_lin, linewidth=2,
             label=f"Linear (R2={met_lin['R2']:.3f}, RMSE={met_lin['RMSE']:.3f})")

    if sigmoid_ok:
        yg_sig = four_param_logistic(xg, *popt)
        plt.plot(xg, yg_sig, linewidth=2,
                 label=f"Sigmoid (R2={met_sig['R2']:.3f}, RMSE={met_sig['RMSE']:.3f})")

    plt.xlabel("RA(N, L, g, lr)")
    plt.ylabel("rho_lambda(N, L, g, lr)")
    plt.title("RA vs rho_lambda: Linear vs Sigmoid Fit")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.05, 1.05)
    plt.xlim(0.0, 1.0)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=240)
    plt.close()

    print("\n[saved plot]", OUT_PNG)


if __name__ == "__main__":
    main()
