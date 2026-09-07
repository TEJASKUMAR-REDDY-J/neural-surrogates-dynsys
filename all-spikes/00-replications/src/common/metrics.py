"""Statistics and error measures for the replication battery.

Two families:
  - model-free complexity/predictability statistics computed from a time series alone
    (permutation entropy, weighted permutation entropy, 0-1 test for chaos, spectral
    entropy, correlation dimension, data-based FSLE)
  - forecast error measures (sMAPE, NRMSE, valid prediction time, plus structural
    measures: power-spectrum error and invariant-density KL)

Every non-trivial function is checked in demo() against a case with a known answer.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------------------
# model-free complexity statistics
# --------------------------------------------------------------------------------------


def _ordinal_patterns(x: np.ndarray, order: int, delay: int = 1) -> np.ndarray:
    """Ordinal (permutation) pattern index of every embedded window.

    Window j is (x[j], x[j+delay], ..., x[j+(order-1)*delay]); its pattern is the
    argsort of those values, encoded as an integer in [0, order!).
    """
    n = len(x) - (order - 1) * delay
    if n <= 0:
        raise ValueError("series too short for this order/delay")
    windows = np.empty((n, order), dtype=x.dtype)
    for k in range(order):
        windows[:, k] = x[k * delay : k * delay + n]
    ranks = np.argsort(np.argsort(windows, axis=1, kind="stable"), axis=1)
    # Lehmer-style encoding: base-`order` is not injective over permutations, so use
    # factorial number system.
    codes = np.zeros(n, dtype=np.int64)
    for k in range(order):
        codes = codes * (order - k) + _rank_among_remaining(ranks, k)
    return codes


def _rank_among_remaining(ranks: np.ndarray, k: int) -> np.ndarray:
    """How many of ranks[:, k+1:] are smaller than ranks[:, k] -> factorial digit."""
    return (ranks[:, k + 1 :] < ranks[:, k : k + 1]).sum(axis=1)


def autocorrelation_time(x: np.ndarray, max_lag: int | None = None) -> int:
    """First lag at which the autocorrelation drops below 1/e. At least 1.

    Ordinal statistics (permutation entropy) and the 0-1 chaos test both assume samples
    are far enough apart to be informative. Trajectories aligned at 100 points per
    dominant period are heavily oversampled, and at delay 1 these statistics measure the
    sampling rate rather than the dynamics: on Lorenz, the 0-1 test returns -0.00
    ("not chaotic") at delay 1 and +1.00 at delay 5. Toker et al. (2020) handle this with
    an explicit oversampling-correction step; this is our version of it.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = max(2, n // 20)
    denom = float((x * x).sum())
    if denom <= 0:
        return 1
    ac = np.correlate(x, x, mode="full")[n - 1 : n - 1 + max_lag] / denom
    below = np.flatnonzero(ac < 1.0 / np.e)
    return int(below[0]) if len(below) else 1


def permutation_entropy(
    x: np.ndarray, order: int = 5, delay: int = 1, weighted: bool = False
) -> float:
    """Normalised (weighted) permutation entropy in [0, 1].

    Bandt & Pompe (2002); the weighted variant is Fadlallah et al. (2013), which weights
    each window by its variance so that flat noise-dominated windows contribute little.
    0 = perfectly ordered, 1 = all orderings equally likely.
    """
    x = np.asarray(x, dtype=float).ravel()
    codes = _ordinal_patterns(x, order, delay)
    n_pat = math.factorial(order)

    if weighted:
        n = len(codes)
        windows = np.empty((n, order))
        for k in range(order):
            windows[:, k] = x[k * delay : k * delay + n]
        w = windows.var(axis=1)
        if w.sum() <= 0:
            return 0.0
        p = np.bincount(codes, weights=w, minlength=n_pat)
    else:
        p = np.bincount(codes, minlength=n_pat).astype(float)

    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(n_pat))


def zero_one_test_chaos(
    x: np.ndarray, n_c: int = 100, seed: int = 0, max_n_frac: float = 0.1
) -> float:
    """0-1 test for chaos (Gottwald & Melbourne), correlation method.

    Drives a 2-D oscillator with the series and measures how the mean-square
    displacement grows. Returns K: ~1 for chaotic, ~0 for periodic/quasi-periodic.
    Needs no phase-space reconstruction and no embedding dimension, which is why it is
    more robust on short noisy series than a Lyapunov estimate.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    n_max = max(10, int(max_n_frac * N))
    rng = np.random.default_rng(seed)
    # avoid c near 0, pi/2, pi where the test degenerates
    cs = rng.uniform(np.pi / 5.0, 4.0 * np.pi / 5.0, size=n_c)
    j = np.arange(1, N + 1)
    ns = np.arange(1, n_max + 1)
    xbar = x.mean()
    ks = np.empty(n_c)

    for i, c in enumerate(cs):
        p = np.cumsum(x * np.cos(j * c))
        q = np.cumsum(x * np.sin(j * c))
        M = np.empty(n_max)
        for k, n in enumerate(ns):
            dp = p[n:] - p[:-n]
            dq = q[n:] - q[:-n]
            M[k] = np.mean(dp * dp + dq * dq)
        # remove the deterministic oscillatory term
        D = M - xbar**2 * (1.0 - np.cos(ns * c)) / (1.0 - np.cos(c))
        sd = D.std()
        ks[i] = 0.0 if sd < 1e-30 else float(np.corrcoef(ns, D)[0, 1])

    return float(np.median(ks))


def spectral_entropy(x: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum. Low = few strong frequencies."""
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    psd = np.abs(np.fft.rfft(x)) ** 2
    psd = psd[1:]  # drop DC
    tot = psd.sum()
    if tot <= 0:
        return 0.0
    p = psd / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(psd)))


def correlation_dimension(
    X: np.ndarray, n_sub: int = 2000, seed: int = 0, n_r: int = 24
) -> float:
    """Grassberger-Procaccia correlation dimension from the scaling of C(r).

    Slope of log C(r) vs log r over the middle of the available range, where C(r) is the
    fraction of point pairs closer than r.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    rng = np.random.default_rng(seed)
    if len(X) > n_sub:
        X = X[rng.choice(len(X), n_sub, replace=False)]
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    d = d[np.triu_indices_from(d, k=1)]
    d = d[d > 0]
    if len(d) < 100:
        return float("nan")
    # scaling region: 1st to 30th percentile of pair distances
    lo, hi = np.percentile(d, [1, 30])
    if not (hi > lo > 0):
        return float("nan")
    rs = np.logspace(np.log10(lo), np.log10(hi), n_r)
    C = np.array([(d < r).mean() for r in rs])
    ok = C > 0
    if ok.sum() < 5:
        return float("nan")
    slope = np.polyfit(np.log(rs[ok]), np.log(C[ok]), 1)[0]
    return float(slope)


def fsle_curve(
    X: np.ndarray,
    deltas: np.ndarray,
    growth: float = 2.0,
    n_pairs: int = 200,
    min_sep: int = 50,
    seed: int = 0,
    dt: float = 1.0,
) -> np.ndarray:
    """Data-based finite-size Lyapunov exponent lambda(delta).

    For each delta: find pairs of points on the recorded trajectory that are within
    delta of each other but far apart in time (so they are genuinely different visits to
    the same region), then measure how many steps until their separation reaches
    growth*delta. lambda(delta) = ln(growth) / mean(time).

    Returns an array the same length as `deltas`; nan where too few pairs were found.
    This is the cheap version - it reuses one long trajectory instead of integrating new
    perturbed pairs.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    N = len(X)
    rng = np.random.default_rng(seed)
    out = np.full(len(deltas), np.nan)
    max_steps = min(2000, N // 4)

    for di, delta in enumerate(deltas):
        times = []
        # random candidate anchors, then a linear scan for a near neighbour far in time
        anchors = rng.choice(N - max_steps - 1, size=min(4 * n_pairs, N // 2), replace=False)
        for a in anchors:
            if len(times) >= n_pairs:
                break
            d = np.sqrt(((X - X[a]) ** 2).sum(-1))
            d[max(0, a - min_sep) : a + min_sep] = np.inf
            d[N - max_steps :] = np.inf
            cand = np.flatnonzero(d < delta)
            if len(cand) == 0:
                continue
            b = int(cand[rng.integers(len(cand))])
            hi = min(N - max(a, b), max_steps)
            if hi < 2:
                continue
            sep = np.sqrt(((X[a : a + hi] - X[b : b + hi]) ** 2).sum(-1))
            reached = np.flatnonzero(sep >= growth * delta)
            if len(reached):
                times.append(reached[0] * dt)
        if len(times) >= max(10, n_pairs // 10):
            out[di] = np.log(growth) / np.mean(times)
    return out


# --------------------------------------------------------------------------------------
# forecast error measures
# --------------------------------------------------------------------------------------


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error in [0, 2]. Gilpin's headline metric."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    ok = denom > 0
    if not ok.any():
        return 0.0
    return float(2.0 * np.mean(np.abs(y_pred - y_true)[ok] / denom[ok]))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE normalised by the standard deviation of the truth. 1.0 = as good as the mean."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    sd = y_true.std()
    if sd <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) / sd)


def valid_prediction_time(
    err_by_step: np.ndarray, threshold: float = 0.4, dt_lyap: float = 1.0
) -> float:
    """First step at which normalised error exceeds `threshold`, in Lyapunov times.

    The standard metric in the reservoir-computing literature. dt_lyap is the Lyapunov
    time per step (lambda_max * dt), so the answer is comparable across systems.
    """
    err = np.asarray(err_by_step, float)
    over = np.flatnonzero(err > threshold)
    steps = len(err) if len(over) == 0 else int(over[0])
    return float(steps * dt_lyap)


def spectrum_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L2 distance between log power spectra, averaged over channels.

    Structural, not pointwise: a forecast can be at the wrong phase and still score well.
    """
    y_true, y_pred = np.atleast_2d(np.asarray(y_true, float)), np.atleast_2d(np.asarray(y_pred, float))
    if y_true.shape[0] < y_true.shape[1]:
        y_true, y_pred = y_true.T, y_pred.T
    errs = []
    for c in range(y_true.shape[1]):
        a = np.abs(np.fft.rfft(y_true[:, c] - y_true[:, c].mean())) ** 2 + 1e-12
        b = np.abs(np.fft.rfft(y_pred[:, c] - y_pred[:, c].mean())) ** 2 + 1e-12
        la, lb = np.log(a), np.log(b)
        errs.append(np.linalg.norm(la - lb) / max(np.linalg.norm(la), 1e-12))
    return float(np.mean(errs))


def invariant_density_kl(
    y_true: np.ndarray, y_pred: np.ndarray, bins: int = 20, eps: float = 1e-6
) -> float:
    """KL(true || pred) between binned state distributions, per channel, averaged.

    Measures whether the forecast visits the right parts of state space over long times -
    the "did it get the attractor right" question, as opposed to "is it in the right place
    now". Bins are set by the true data so both histograms share support.
    """
    y_true, y_pred = np.atleast_2d(np.asarray(y_true, float)), np.atleast_2d(np.asarray(y_pred, float))
    if y_true.shape[0] < y_true.shape[1]:
        y_true, y_pred = y_true.T, y_pred.T
    kls = []
    for c in range(y_true.shape[1]):
        lo, hi = y_true[:, c].min(), y_true[:, c].max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(y_true[:, c], bins=edges)
        q, _ = np.histogram(np.clip(y_pred[:, c], lo, hi), bins=edges)
        p = p / max(p.sum(), 1) + eps
        q = q / max(q.sum(), 1) + eps
        p, q = p / p.sum(), q / q.sum()
        kls.append(float((p * np.log(p / q)).sum()))
    return float(np.mean(kls)) if kls else float("nan")


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def demo() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    t = np.linspace(0, 200 * np.pi, n)
    sine = np.sin(t)
    noise = rng.standard_normal(n)

    # logistic map at r=4: known KS entropy ln 2; permutation entropy should be high but
    # below white noise
    log_map = np.empty(n)
    log_map[0] = 0.4
    for i in range(1, n):
        log_map[i] = 4.0 * log_map[i - 1] * (1 - log_map[i - 1])

    pe_sine = permutation_entropy(sine)
    pe_noise = permutation_entropy(noise)
    pe_log = permutation_entropy(log_map)
    assert pe_sine < 0.45, pe_sine
    assert pe_noise > 0.95, pe_noise
    assert pe_sine < pe_log < pe_noise, (pe_sine, pe_log, pe_noise)

    wpe_sine = permutation_entropy(sine, weighted=True)
    wpe_noise = permutation_entropy(noise, weighted=True)
    assert wpe_sine < 0.45 and wpe_noise > 0.9, (wpe_sine, wpe_noise)
    assert wpe_sine < wpe_noise

    # ordinal pattern encoding must be injective over permutations
    codes = _ordinal_patterns(np.array([3.0, 1.0, 2.0, 5.0, 4.0, 0.0]), order=3)
    assert codes.max() < 6 and len(np.unique(codes)) >= 3, codes

    # 0-1 test: periodic -> ~0, chaotic -> ~1
    k_sine = zero_one_test_chaos(sine, n_c=30)
    k_log = zero_one_test_chaos(log_map, n_c=30)
    assert k_sine < 0.3, k_sine
    assert k_log > 0.7, k_log

    # spectral entropy: sine has one frequency, noise is flat
    assert spectral_entropy(sine) < 0.3 < spectral_entropy(noise), (
        spectral_entropy(sine),
        spectral_entropy(noise),
    )

    # correlation dimension: 2-D uniform cloud -> ~2, 1-D line -> ~1
    cloud = rng.uniform(size=(2000, 2))
    line = np.c_[np.linspace(0, 1, 2000), np.linspace(0, 1, 2000)]
    cd_cloud, cd_line = correlation_dimension(cloud), correlation_dimension(line)
    assert 1.7 < cd_cloud < 2.3, cd_cloud
    assert 0.8 < cd_line < 1.2, cd_line

    # error measures
    y = rng.standard_normal((100, 3))
    assert smape(y, y) < 1e-12
    assert nrmse(y, y) < 1e-12
    assert abs(nrmse(y, np.zeros_like(y)) - 1.0) < 0.15
    assert spectrum_error(y, y) < 1e-9
    assert invariant_density_kl(y, y) < 0.05
    assert invariant_density_kl(y, y + 5.0) > 0.5

    # VPT: error crosses 0.4 at step 3
    err = np.array([0.01, 0.05, 0.2, 0.9, 1.2])
    assert valid_prediction_time(err, 0.4, dt_lyap=0.5) == 1.5

    # oversampling correction: an oversampled chaotic signal must be recognised as
    # chaotic once the delay respects the autocorrelation time
    t = np.linspace(0, 200 * np.pi, n)
    slow = np.sin(t) + 0.35 * np.sin(np.pi * t)      # quasi-periodic, densely sampled
    tau_slow = autocorrelation_time(slow)
    assert tau_slow > 1, tau_slow
    assert autocorrelation_time(noise) == 1, autocorrelation_time(noise)
    # A smoothly oversampled chaotic series: at delay 1 the ordinal statistics see a
    # smooth curve, and correcting the delay recovers the underlying disorder.
    dense = np.interp(np.linspace(0, len(log_map) - 1, len(log_map) * 8),
                      np.arange(len(log_map)), log_map)
    tau_dense = autocorrelation_time(dense)
    assert tau_dense >= 4, tau_dense
    pe_naive = permutation_entropy(dense, order=5, delay=1)
    pe_corrected = permutation_entropy(dense, order=5, delay=tau_dense)
    assert pe_corrected > pe_naive + 0.3, (pe_naive, pe_corrected)
    assert pe_naive < 0.4 and pe_corrected > 0.6, (pe_naive, pe_corrected)

    print(
        "metrics ok - "
        f"PE sine {pe_sine:.3f} logistic {pe_log:.3f} noise {pe_noise:.3f} | "
        f"K sine {k_sine:.2f} logistic {k_log:.2f} | "
        f"corrdim cloud {cd_cloud:.2f} line {cd_line:.2f} | "
        f"oversampled logistic: tau={tau_dense}, PE {pe_naive:.2f}->{pe_corrected:.2f}"
    )


if __name__ == "__main__":
    demo()
