# phase4_transition_diagnostics.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PHASE2 = "phase2_grid_state.npz"
PHASE3 = "ra_ka_grid_state.npz"
OUTDIR = "plots_phase_transition"
os.makedirs(OUTDIR, exist_ok=True)

def sigmoid(x, a, b, k, x0):
    # a = low asymptote, b = high asymptote
    return a + (b - a) / (1.0 + np.exp(-k * (x - x0)))

def fit_sigmoid(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 10:
        return None

    # sensible init guesses
    a0 = np.nanpercentile(y, 5)
    b0 = np.nanpercentile(y, 95)
    x00 = np.nanmedian(x)
    k0 = 5.0

    # constrain asymptotes to observed range (helps stability)
    bounds = ([-1.5, -1.5,  1e-3, 0.0],
              [ 1.5,  1.5,  1e3,  1.0])

    popt, pcov = curve_fit(
        sigmoid, x, y, p0=[a0, b0, k0, x00],
        bounds=bounds, maxfev=20000
    )
    a, b, k, x0 = popt
    yhat = sigmoid(x, *popt)
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    # R^2
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # transition width in x between 10% and 90%
    y10 = a + 0.1 * (b - a)
    y90 = a + 0.9 * (b - a)

    # invert sigmoid safely
    def inv_sig(yv):
        # y = a + (b-a)/(1+exp(-k(x-x0)))
        # => (b-a)/(y-a) - 1 = exp(-k(x-x0))
        denom = np.clip(yv - a, 1e-12, None)
        t = (b - a) / denom - 1.0
        t = np.clip(t, 1e-12, None)
        return x0 - (1.0 / k) * np.log(t)

    x10 = float(inv_sig(y10))
    x90 = float(inv_sig(y90))
    width = abs(x90 - x10)

    slope_mid = float((b - a) * k / 4.0)  # derivative at x0

    return dict(a=a, b=b, k=k, x0=x0, rmse=rmse, r2=r2, width=width, slope_mid=slope_mid)

def load_phase2():
    d = np.load(PHASE2)
    widths = d["widths"].astype(int).tolist()
    depths = d["depths"].astype(int).tolist()
    gains  = d["gains"].astype(float).tolist()
    lrs    = d["base_lrs"].astype(float).tolist()
    Glam   = d["G_lambda_map"].astype(float)      # [g,lr,L,N]
    rho    = d["rho_lambda_map"].astype(float)    # [g,lr,L,N]
    return widths, depths, gains, lrs, Glam, rho

def load_phase3():
    d = np.load(PHASE3)
    # Try common key names robustly
    key_RA = None
    key_KA = None
    for k in d.files:
        lk = k.lower()
        if lk in ("ra_map", "ra", "ra_grid", "ra_values"):
            key_RA = k
        if lk in ("ka_map", "ka", "ka_grid", "ka_values"):
            key_KA = k
    if key_RA is None or key_KA is None:
        raise KeyError(f"Could not find RA/KA arrays in {PHASE3}. Keys present: {d.files}")
    RA = d[key_RA].astype(float)
    KA = d[key_KA].astype(float)

    # Also try to read axes; fall back to phase2 axes if missing
    widths = d["widths"].astype(int).tolist() if "widths" in d.files else None
    depths = d["depths"].astype(int).tolist() if "depths" in d.files else None
    gains  = d["gains"].astype(float).tolist() if "gains" in d.files else None
    lrs    = d["base_lrs"].astype(float).tolist() if "base_lrs" in d.files else None

    return widths, depths, gains, lrs, RA, KA

def flatten_grid(widths, depths, gains, lrs, Glam, rho, RA, KA):
    # Expect 4D arrays [g, lr, L, N]
    ng, nlr, nL, nN = Glam.shape
    rows = []
    for gi, g in enumerate(gains):
        for li, lr in enumerate(lrs):
            for di, L in enumerate(depths):
                for wi, N in enumerate(widths):
                    rows.append(dict(
                        N=int(N), L=int(L), g=float(g), lr=float(lr),
                        G_lambda=float(Glam[gi, li, di, wi]),
                        rho_lambda=float(rho[gi, li, di, wi]),
                        RA=float(RA[gi, li, di, wi]),
                        KA=float(KA[gi, li, di, wi]),
                    ))
    return rows

def to_arrays(rows):
    def col(name):
        return np.array([r[name] for r in rows], float)
    return {k: col(k) for k in rows[0].keys()}

def binned_stats(x, y, bins=20):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(bins, np.nan)
    std  = np.full(bins, np.nan)
    count = np.zeros(bins, int)
    for i in range(bins):
        sel = (x >= edges[i]) & (x < edges[i+1])
        if np.any(sel):
            mean[i] = float(np.mean(y[sel]))
            std[i]  = float(np.std(y[sel]))
            count[i] = int(np.sum(sel))
    return centers, mean, std, count

def bootstrap_fit_params(x, y, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 20:
        return None
    ks = []; x0s = []; widths = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_sigmoid(x[idx], y[idx])
        if fit is None:
            continue
        ks.append(fit["k"])
        x0s.append(fit["x0"])
        widths.append(fit["width"])
    if len(ks) < 30:
        return None
    def ci(a):
        a = np.array(a, float)
        return float(np.nanpercentile(a, 16)), float(np.nanpercentile(a, 84))
    return dict(k_ci=ci(ks), x0_ci=ci(x0s), width_ci=ci(widths),
                k_med=float(np.nanmedian(ks)),
                x0_med=float(np.nanmedian(x0s)),
                width_med=float(np.nanmedian(widths)))

def main():
    widths2, depths2, gains2, lrs2, Glam, rho = load_phase2()
    widths3, depths3, gains3, lrs3, RA, KA = load_phase3()

    # sanity: prefer phase2 axes if phase3 missing
    widths = widths2 if widths3 is None else widths3
    depths = depths2 if depths3 is None else depths3
    gains  = gains2  if gains3  is None else gains3
    lrs    = lrs2    if lrs3    is None else lrs3

    # basic shape checks
    if Glam.shape != rho.shape:
        raise ValueError("phase2 shapes mismatch")
    if RA.shape != Glam.shape or KA.shape != Glam.shape:
        raise ValueError(f"phase3 RA/KA shape mismatch vs phase2: "
                         f"RA{RA.shape} KA{KA.shape} vs phase2{Glam.shape}")

    rows = flatten_grid(widths, depths, gains, lrs, Glam, rho, RA, KA)
    A = to_arrays(rows)

    RA_all  = A["RA"]
    rho_all = A["rho_lambda"]
    Glam_all = A["G_lambda"]

    # ---------------- Experiment 2: susceptibility-like peak ----------------
    plt.figure(figsize=(6.2, 4.6))
    plt.scatter(RA_all, np.log10(Glam_all + 1e-12), s=18, alpha=0.65)
    c, m, s, n = binned_stats(RA_all, np.log10(Glam_all + 1e-12), bins=20)
    plt.plot(c, m, linewidth=2)
    plt.xlabel("RA")
    plt.ylabel("log10(G_lambda)")
    plt.title("Fluctuations vs order parameter: log10(G_lambda) vs RA")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(OUTDIR, "susceptibility_Glambda_vs_RA.png")
    plt.savefig(out1, dpi=220)
    plt.close()

    # ---------------- Experiment 1: finite-size scaling of sigmoid ----------------
    # Fit rho(RA) per width N
    Ns = sorted(set(int(x) for x in A["N"]))
    k_list = []; k_lo = []; k_hi = []
    w_list = []; w_lo = []; w_hi = []
    x0_list = []; x0_lo = []; x0_hi = []

    for N in Ns:
        sel = (A["N"] == float(N))
        fit = fit_sigmoid(RA_all[sel], rho_all[sel])
        boot = bootstrap_fit_params(RA_all[sel], rho_all[sel], n_boot=200, seed=123+N)
        if fit is None or boot is None:
            continue
        k_list.append(fit["k"]); x0_list.append(fit["x0"]); w_list.append(fit["width"])
        klo,khi = boot["k_ci"]; x0l,x0h = boot["x0_ci"]; wl,wh = boot["width_ci"]
        k_lo.append(klo); k_hi.append(khi)
        x0_lo.append(x0l); x0_hi.append(x0h)
        w_lo.append(wl); w_hi.append(wh)

    Ns_plot = np.array(Ns[:len(k_list)], float)

    def errbar(x, y, lo, hi, xlabel, ylabel, title, fname):
        plt.figure(figsize=(6.0, 4.2))
        y = np.array(y, float); lo = np.array(lo, float); hi = np.array(hi, float)
        yerr = np.vstack([y - lo, hi - y])
        plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, fname), dpi=220)
        plt.close()

    errbar(Ns_plot, k_list, k_lo, k_hi,
           "Width N", "Sigmoid steepness k",
           "Finite-size scaling: steepness of rho(RA) vs N",
           "finite_size_k_vs_N.png")

    errbar(Ns_plot, w_list, w_lo, w_hi,
           "Width N", "Transition width ΔRA (10%→90%)",
           "Finite-size scaling: transition width vs N",
           "finite_size_width_vs_N.png")

    errbar(Ns_plot, x0_list, x0_lo, x0_hi,
           "Width N", "Midpoint x0 (RA at half-transition)",
           "Finite-size scaling: midpoint x0 vs N",
           "finite_size_x0_vs_N.png")

    # ---------------- Experiment 3: data collapse with kappa ----------------
    kappa = (A["lr"] * (A["g"]**2) * A["L"]) / np.maximum(A["N"], 1.0)
    logk = np.log10(kappa + 1e-12)

    plt.figure(figsize=(6.2, 4.6))
    for N in Ns:
        sel = (A["N"] == float(N))
        plt.scatter(logk[sel], rho_all[sel], s=18, alpha=0.65, label=f"N={N}")
    plt.xlabel("log10( kappa = lr*g^2*L/N )")
    plt.ylabel("rho_lambda")
    plt.title("Data collapse attempt: rho_lambda vs kappa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "collapse_rho_vs_kappa.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(6.2, 4.6))
    for N in Ns:
        sel = (A["N"] == float(N))
        plt.scatter(logk[sel], RA_all[sel], s=18, alpha=0.65, label=f"N={N}")
    plt.xlabel("log10( kappa = lr*g^2*L/N )")
    plt.ylabel("RA")
    plt.title("Data collapse attempt: RA vs kappa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "collapse_RA_vs_kappa.png"), dpi=220)
    plt.close()

    print("[done] wrote plots to:", OUTDIR)

if __name__ == "__main__":
    main()
