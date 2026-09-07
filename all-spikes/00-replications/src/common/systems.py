"""Dynamical systems: trajectories, timescale alignment, and cached invariants.

We take the equations and the published invariants from `dysts` (Gilpin 2021), but we do
our own integration. dysts' default is Radau at rtol=atol=1e-12, which costs ~41 s for a
2000-point Lorenz trajectory on this machine; RK45 at 1e-9 gives 20000 points in 7 s and
is far more accurate than the surrogate we are about to fit. The invariants
(largest Lyapunov exponent, Kaplan-Yorke dimension, correlation dimension, multiscale
entropy, dominant period) are metadata and come straight from dysts.

Timescale alignment follows Gilpin (2023): every system is sampled at a fixed number of
points per dominant Fourier period, so a fast system and a slow system are compared on
equal footing, and horizons can be reported in Lyapunov times.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "trajectories"

# Discrete maps we implement locally: dysts is ODE-oriented, and R5 needs a clean
# chaos knob (the standard map is literally Chirikov's, which is the mechanism NNPT
# claims its transition matches).
DISCRETE_MAPS = ("StandardMap", "LogisticMap", "HenonMap")


@dataclass
class SystemSpec:
    """Everything an experiment needs about one system."""

    name: str
    dim: int
    period: float
    lyap_max: float
    invariants: dict = field(default_factory=dict)
    is_map: bool = False
    pts_per_period: int = 100

    @property
    def dt(self) -> float:
        """Time between consecutive samples, after alignment."""
        return 1.0 if self.is_map else self.period / self.pts_per_period

    @property
    def lyap_time_per_step(self) -> float:
        """lambda_max * dt: how many Lyapunov times one sample step is worth."""
        return float(self.lyap_max * self.dt)


# --------------------------------------------------------------------------------------
# discrete maps
# --------------------------------------------------------------------------------------


def _standard_map_step(state: np.ndarray, K: float) -> np.ndarray:
    """Chirikov standard map. K is the chaos knob; K~1 is the transition to global chaos."""
    theta, p = state[..., 0], state[..., 1]
    p_new = (p + K * np.sin(theta)) % (2 * np.pi)
    th_new = (theta + p_new) % (2 * np.pi)
    return np.stack([th_new, p_new], axis=-1)


def _logistic_step(state: np.ndarray, r: float) -> np.ndarray:
    return r * state * (1.0 - state)


def _henon_step(state: np.ndarray, a: float, b: float = 0.3) -> np.ndarray:
    x, y = state[..., 0], state[..., 1]
    return np.stack([1.0 - a * x * x + y, b * x], axis=-1)


def map_trajectory(
    name: str, n: int, param: float, seed: int = 0, burn_in: int = 1000
) -> np.ndarray:
    """Iterate a discrete map, discarding a burn-in so we start on the attractor."""
    rng = np.random.default_rng(seed)
    if name == "StandardMap":
        x = rng.uniform(0, 2 * np.pi, size=2)
        step = lambda s: _standard_map_step(s, param)
    elif name == "LogisticMap":
        x = np.array([rng.uniform(0.1, 0.9)])
        step = lambda s: _logistic_step(s, param)
    elif name == "HenonMap":
        x = rng.uniform(-0.1, 0.1, size=2)
        step = lambda s: _henon_step(s, param)
    else:
        raise ValueError(f"unknown map {name}")

    for _ in range(burn_in):
        x = step(x)
    out = np.empty((n, len(np.atleast_1d(x))))
    for i in range(n):
        x = step(x)
        out[i] = x
    return out


def standard_map_lyapunov(K: float, n: int = 200_000, seed: int = 0) -> float:
    """Largest Lyapunov exponent of the standard map by tangent-space iteration."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 2 * np.pi, size=2)
    v = np.array([1.0, 0.0])
    total = 0.0
    for i in range(n):
        theta = s[0]
        # Jacobian of (theta, p) -> (theta + p + K sin theta, p + K sin theta)
        J = np.array([[1.0 + K * np.cos(theta), 1.0], [K * np.cos(theta), 1.0]])
        v = J @ v
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
            if i > n // 10:
                total += np.log(nrm)
        s = _standard_map_step(s, K)
    return float(total / (n - n // 10))


# --------------------------------------------------------------------------------------
# continuous systems via dysts
# --------------------------------------------------------------------------------------


def list_dysts_systems() -> list[str]:
    import dysts.flows as flows

    from dysts.base import DynSys

    names = []
    for nm in dir(flows):
        obj = getattr(flows, nm)
        if isinstance(obj, type) and issubclass(obj, DynSys) and obj is not DynSys:
            names.append(nm)
    return sorted(names)


def load_spec(name: str, pts_per_period: int = 100) -> SystemSpec:
    """Metadata only - cheap, no integration."""
    if name in DISCRETE_MAPS:
        raise ValueError("use map_spec() for discrete maps")
    import dysts.flows as flows

    m = getattr(flows, name)()
    lam = float(np.atleast_1d(m.maximum_lyapunov_estimated).ravel()[0])
    return SystemSpec(
        name=name,
        dim=len(np.atleast_1d(m.ic)),
        period=float(m.period),
        lyap_max=lam,
        pts_per_period=pts_per_period,
        invariants={
            "lyap_max": lam,
            "kaplan_yorke_dim": float(m.kaplan_yorke_dimension),
            "correlation_dim_pub": float(m.correlation_dimension),
            "multiscale_entropy": float(m.multiscale_entropy),
            "period": float(m.period),
            "dim": len(np.atleast_1d(m.ic)),
        },
    )


def _cache_path(name: str, n: int, pts_per_period: int, ic_seed: int) -> Path:
    key = f"{name}|{n}|{pts_per_period}|{ic_seed}"
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return DATA_DIR / f"{name}__{n}__{h}.npy"


class IntegrationTimeout(RuntimeError):
    """Raised when a system exceeds its wall-clock budget mid-integration."""


def trajectory(
    name: str,
    n: int = 20_000,
    pts_per_period: int = 100,
    ic_seed: int = 0,
    rtol: float = 1e-9,
    use_cache: bool = True,
    timeout_s: float | None = 60.0,
) -> np.ndarray:
    """One aligned trajectory of `n` samples at `pts_per_period` per dominant period.

    ic_seed != 0 perturbs the published initial condition, giving an independent
    trajectory on the same attractor (used for held-out test sets).
    """
    path = _cache_path(name, n, pts_per_period, ic_seed)
    if use_cache and path.exists():
        return np.load(path)

    import dysts.flows as flows

    m = getattr(flows, name)()
    ic = np.array(m.ic, dtype=float).ravel()
    if ic_seed != 0:
        rng = np.random.default_rng(ic_seed)
        ic = ic + 0.01 * np.abs(ic).mean() * rng.standard_normal(ic.shape)

    period = float(m.period)
    T = n * period / pts_per_period

    # Stiff systems can integrate for many minutes with no way to interrupt solve_ivp.
    # The RHS is called constantly, so a deadline check there is cheap and reliable.
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def rhs(t, y):
        if deadline is not None and time.monotonic() > deadline:
            raise IntegrationTimeout(f"{name}: exceeded {timeout_s}s")
        return m.rhs(y, t)

    # integrate a burn-in of 20 periods first so we start on the attractor
    burn_T = 20 * period
    sol0 = solve_ivp(rhs, (0, burn_T), ic, rtol=rtol, atol=rtol, method="RK45")
    ic = sol0.y[:, -1]

    sol = solve_ivp(
        rhs,
        (0, T),
        ic,
        t_eval=np.linspace(0, T, n),
        rtol=rtol,
        atol=rtol,
        method="RK45",
    )
    X = sol.y.T
    if X.shape[0] != n or not np.all(np.isfinite(X)):
        raise RuntimeError(f"{name}: integration failed ({X.shape}, finite={np.all(np.isfinite(X))})")

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, X)
    return X


# --------------------------------------------------------------------------------------
# supervised datasets
# --------------------------------------------------------------------------------------


def one_step_dataset(
    X: np.ndarray, n_train: int, n_test: int = 2000, seed: int = 0
) -> dict:
    """Turn a trajectory into a one-step prediction task, with train-only normalisation.

    Predicts the *increment* x(t+1) - x(t) rather than the raw next state: standard
    practice for surrogates, and it keeps the target scale comparable across systems.
    Train and test come from disjoint contiguous blocks so there is no leakage through
    overlapping windows.
    """
    X = np.asarray(X, dtype=float)
    need = n_train + n_test + 2
    if len(X) < need:
        raise ValueError(f"need {need} points, have {len(X)}")

    train = X[:n_train + 1]
    test = X[n_train + 1 : n_train + 1 + n_test + 1]

    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 0, sd, 1.0)

    def prep(block):
        Z = (block - mu) / sd
        return Z[:-1], Z[1:] - Z[:-1]

    Xtr, Ytr = prep(train)
    Xte, Yte = prep(test)
    return {
        "X_train": Xtr.astype(np.float32),
        "Y_train": Ytr.astype(np.float32),
        "X_test": Xte.astype(np.float32),
        "Y_test": Yte.astype(np.float32),
        "mu": mu,
        "sd": sd,
        "test_traj_norm": ((test - mu) / sd).astype(np.float32),
    }


def demo() -> None:
    """Self-check: alignment, caching, dataset shapes, and the standard map's chaos knob."""
    # standard map: below K~1 mostly regular, well above it strongly chaotic
    lam_low = standard_map_lyapunov(0.3, n=20_000)
    lam_high = standard_map_lyapunov(6.0, n=20_000)
    assert lam_high > 0.8, lam_high
    assert lam_high > lam_low + 0.5, (lam_low, lam_high)

    traj = map_trajectory("StandardMap", 500, param=6.0)
    assert traj.shape == (500, 2) and np.all(np.isfinite(traj))
    assert traj.max() <= 2 * np.pi + 1e-9

    # logistic at r=4 has Lyapunov exponent ln 2; check the trajectory covers [0,1]
    lg = map_trajectory("LogisticMap", 5000, param=4.0)
    assert lg.min() < 0.02 and lg.max() > 0.98, (lg.min(), lg.max())

    # a continuous system, integrated by us
    spec = load_spec("Lorenz")
    assert spec.dim == 3 and spec.lyap_max > 0
    X = trajectory("Lorenz", n=3000, use_cache=False)
    assert X.shape == (3000, 3) and np.all(np.isfinite(X))
    # 100 points per period means the dominant Fourier peak sits near period 100 samples
    xf = np.abs(np.fft.rfft(X[:, 0] - X[:, 0].mean()))
    peak_period = len(X) / max(np.argmax(xf), 1)
    assert 40 < peak_period < 250, peak_period

    ds = one_step_dataset(X, n_train=1500, n_test=500)
    assert ds["X_train"].shape == (1500, 3) and ds["Y_train"].shape == (1500, 3)
    assert ds["X_test"].shape == (500, 3)
    assert abs(ds["X_train"].mean()) < 0.05

    print(
        f"systems ok - standard map lambda(K=0.3)={lam_low:.3f} lambda(K=6)={lam_high:.3f} | "
        f"Lorenz dim={spec.dim} lambda={spec.lyap_max:.3f} dt={spec.dt:.4f} "
        f"lyap/step={spec.lyap_time_per_step:.4f} peak~{peak_period:.0f} samples"
    )


if __name__ == "__main__":
    demo()
