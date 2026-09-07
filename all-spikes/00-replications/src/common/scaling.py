"""Fit the scaling form  L(N) = L_inf + A * N^(-alpha)  with bootstrap confidence intervals.

This is the machinery from Bahri et al., PNAS 2024 ("Explaining neural scaling laws").
Two numbers matter and they answer different questions:

  alpha  - how fast error falls as you scale. Resolution-limited theory predicts
           alpha ~ 1/d with d the intrinsic dimension of the data manifold. For a chaotic
           system that manifold is the attractor, so d is its fractal dimension.
  L_inf  - the floor the curve is heading towards. If L_inf is reliably above zero and
           differs between systems by more than seed noise, that is evidence for an
           intrinsic per-system ceiling. If it is indistinguishable from zero, error is
           still falling and the limit is us, not the system.

The bootstrap matters more than the point estimate: L_inf is a extrapolated quantity and
its uncertainty is large whenever the data have not visibly bent over.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _model(N, L_inf, A, alpha):
    return L_inf + A * np.power(N, -alpha)


def fit_scaling(
    N: np.ndarray,
    L: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
) -> dict:
    """Fit L = L_inf + A N^-alpha. Returns estimates plus 95% bootstrap intervals.

    Also reports `pure_power_alpha`, the slope of a plain log-log fit with no floor, and
    `floor_supported`: whether the 95% interval for L_inf excludes zero. That flag is the
    thing R4 actually turns on.
    """
    N = np.asarray(N, float)
    L = np.asarray(L, float)
    ok = np.isfinite(N) & np.isfinite(L) & (N > 0) & (L > 0)
    N, L = N[ok], L[ok]
    if len(N) < 4:
        return {"ok": False, "reason": f"only {len(N)} usable points"}

    # plain power law, no floor - also the initial guess for alpha
    slope, intercept = np.polyfit(np.log(N), np.log(L), 1)
    pure_alpha = float(-slope)

    p0 = [max(L.min() * 0.5, 1e-8), float(np.exp(intercept)), max(pure_alpha, 1e-3)]
    bounds = ([0.0, 0.0, 0.0], [L.max(), np.inf, 5.0])

    def _try(Nx, Lx):
        try:
            popt, _ = curve_fit(_model, Nx, Lx, p0=p0, bounds=bounds, maxfev=20000)
            return popt
        except Exception:
            return None

    popt = _try(N, L)
    if popt is None:
        return {"ok": False, "reason": "curve_fit failed", "pure_power_alpha": pure_alpha}

    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(N))
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(N[take])) < 4:
            continue
        b = _try(N[take], L[take])
        if b is not None:
            boots.append(b)

    out = {
        "ok": True,
        "L_inf": float(popt[0]),
        "A": float(popt[1]),
        "alpha": float(popt[2]),
        "pure_power_alpha": pure_alpha,
        "n_points": int(len(N)),
        "n_boot_ok": len(boots),
    }
    if len(boots) >= 50:
        B = np.array(boots)
        for i, key in enumerate(("L_inf", "A", "alpha")):
            lo, hi = np.percentile(B[:, i], [2.5, 97.5])
            out[f"{key}_lo"], out[f"{key}_hi"] = float(lo), float(hi)
        out["floor_supported"] = bool(out["L_inf_lo"] > 1e-6)
    else:
        out["floor_supported"] = None

    # residual quality of the floor model vs the pure power law
    pred = _model(N, *popt)
    out["r2"] = float(1 - np.sum((L - pred) ** 2) / max(np.sum((L - L.mean()) ** 2), 1e-30))
    return out


def demo() -> None:
    """Self-check against data generated from a known law, with and without a floor."""
    N = np.array([1e3, 2.2e3, 4.6e3, 1e4, 2.2e4, 4.6e4, 1e5])
    rng = np.random.default_rng(0)

    # case 1: a real floor at 0.05, alpha = 0.5
    L = 0.05 + 3.0 * N**-0.5
    L_noisy = L * (1 + 0.02 * rng.standard_normal(len(L)))
    f1 = fit_scaling(N, L_noisy, n_boot=200)
    assert f1["ok"]
    assert abs(f1["L_inf"] - 0.05) < 0.02, f1["L_inf"]
    assert abs(f1["alpha"] - 0.5) < 0.15, f1["alpha"]
    assert f1["floor_supported"] is True, f1
    # a pure power-law fit is fooled: it reports a much shallower slope than the truth
    assert f1["pure_power_alpha"] < 0.5, f1["pure_power_alpha"]

    # case 2: no floor, alpha = 0.4 - the fit must not invent one
    L2 = 2.0 * N**-0.4
    L2n = L2 * (1 + 0.02 * rng.standard_normal(len(L2)))
    f2 = fit_scaling(N, L2n, n_boot=200)
    assert f2["ok"]
    assert f2["L_inf"] < 0.02, f2["L_inf"]
    assert abs(f2["pure_power_alpha"] - 0.4) < 0.05, f2["pure_power_alpha"]

    print(
        f"scaling ok - with floor: L_inf={f1['L_inf']:.4f} "
        f"[{f1['L_inf_lo']:.4f},{f1['L_inf_hi']:.4f}] alpha={f1['alpha']:.3f} | "
        f"no floor: L_inf={f2['L_inf']:.5f} supported={f2['floor_supported']}"
    )


if __name__ == "__main__":
    demo()
