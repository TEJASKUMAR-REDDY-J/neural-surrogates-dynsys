"""Seeded training and rollout evaluation.

Nothing here writes to disk - callers own logging. Every function is deterministic given
its seed, so a rerun of the same config reproduces the same numbers.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from .metrics import (
    invariant_density_kl,
    nrmse,
    smape,
    spectrum_error,
    valid_prediction_time,
)
from .models import HorizonMLP, count_params, mlp_for_budget

# Measured on this machine: 1 thread is as fast as 2 for these model sizes (62.4s vs
# 63.4s for the same three fits), because the work is Python/dispatch bound rather than
# FLOP bound. Running two single-threaded worker processes instead gives ~1.7x. Override
# with TORCH_THREADS if a future experiment is actually FLOP bound.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


def fit(
    X: np.ndarray,
    Y: np.ndarray,
    budget: int,
    depth: int = 2,
    epochs: int = 200,
    batch: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    horizon: bool = False,
    H: np.ndarray | None = None,
    h_max: int = 64,
    verbose: bool = False,
) -> tuple[nn.Module, dict]:
    """Fit a surrogate. Returns (model, info) with info["n_params"] and the loss curve."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xt = torch.from_numpy(np.asarray(X, np.float32))
    Yt = torch.from_numpy(np.asarray(Y, np.float32))
    Ht = None if H is None else torch.from_numpy(np.asarray(H, np.float32))

    model = mlp_for_budget(
        Xt.shape[1], Yt.shape[1], budget, depth=depth, seed=seed, horizon=horizon, h_max=h_max
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = nn.MSELoss()

    n = len(Xt)
    losses = []
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            pred = model(Xt[idx], Ht[idx]) if horizon else model(Xt[idx])
            loss = lossfn(pred, Yt[idx])
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        sched.step()
        losses.append(tot / n)
        if verbose and ep % 50 == 0:
            print(f"  ep {ep} loss {losses[-1]:.5f}")

    return model, {
        "n_params": count_params(model),
        "final_train_loss": losses[-1],
        "loss_curve": losses,
    }


@torch.no_grad()
def rollout(model: nn.Module, x0: np.ndarray, steps: int) -> np.ndarray:
    """Autoregressive rollout from initial states. x0 is (n_starts, dim)."""
    model.eval()
    x = torch.from_numpy(np.asarray(x0, np.float32))
    out = np.empty((len(x), steps, x.shape[1]), dtype=np.float32)
    for s in range(steps):
        x = x + model(x)
        out[:, s] = x.numpy()
    return out


@torch.no_grad()
def direct_predict(model: HorizonMLP, x0: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """One forward pass per horizon - no compounding. Returns (n_starts, n_horizons, dim)."""
    model.eval()
    x = torch.from_numpy(np.asarray(x0, np.float32))
    out = np.empty((len(x), len(horizons), x.shape[1]), dtype=np.float32)
    for j, h in enumerate(horizons):
        hh = torch.full((len(x),), float(h))
        out[:, j] = (x + model(x, hh)).numpy()
    return out


def rollout_truth(traj: np.ndarray, starts: np.ndarray, steps: int) -> np.ndarray:
    """Ground-truth continuations matching what rollout() produces."""
    return np.stack([traj[s + 1 : s + 1 + steps] for s in starts])


def error_by_step(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Normalised error at each rollout step, averaged over starts.

    Normalisation is by the standard deviation of the true trajectory, so 1.0 means "no
    better than predicting the mean" and the number is comparable across systems.
    """
    sd = true.reshape(-1, true.shape[-1]).std(0)
    sd = np.where(sd > 0, sd, 1.0)
    d = ((pred - true) / sd) ** 2
    return np.sqrt(d.mean(axis=(0, 2)))


def evaluate_rollout(
    model: nn.Module,
    test_traj: np.ndarray,
    spec_lyap_per_step: float,
    steps: int = 200,
    n_starts: int = 64,
    seed: int = 0,
    threshold: float = 0.4,
    report_horizons: tuple = (1, 10, 50, 100, 200, 500, 1000),
) -> dict:
    """Full rollout evaluation: pointwise error, VPT, and the structural measures.

    The structural measures (spectrum error, invariant-density KL) are computed on a long
    single rollout, because they are about the shape of the trajectory over time rather
    than about being in the right place at a given step.
    """
    rng = np.random.default_rng(seed)
    usable = len(test_traj) - steps - 2
    if usable <= 1:
        raise ValueError("test trajectory too short for this rollout length")
    starts = rng.choice(usable, size=min(n_starts, usable), replace=False)

    x0 = test_traj[starts]
    pred = rollout(model, x0, steps)
    true = rollout_truth(test_traj, starts, steps)

    err = error_by_step(true, pred)
    vpt = valid_prediction_time(err, threshold, spec_lyap_per_step)

    # long free-running rollout for the structural metrics
    long_steps = min(len(test_traj) - 2, 2000)
    long_pred = rollout(model, test_traj[:1], long_steps)[0]
    long_true = test_traj[1 : 1 + long_steps]
    finite = np.all(np.isfinite(long_pred))

    # Error at fixed horizons. VPT is censored whenever skill outlives the rollout, which
    # happened on a third of R4's fits, so a fixed-horizon error is the honest measure.
    fixed = {
        f"err_h{h}": (float(err[h - 1]) if h <= len(err) else float("nan"))
        for h in report_horizons
    }

    return {
        **fixed,
        "vpt_censored": bool(len(np.flatnonzero(err > threshold)) == 0),
        "rollout_steps": int(steps),
        "smape_1": float(smape(true[:, 0], pred[:, 0])),
        "smape_all": float(smape(true, pred)),
        "nrmse_1": float(nrmse(true[:, 0], pred[:, 0])),
        "err_by_step": err.tolist(),
        "vpt_lyap": vpt,
        "vpt_steps": float(vpt / spec_lyap_per_step) if spec_lyap_per_step > 0 else float("nan"),
        "diverged": not finite,
        "spectrum_error": float(spectrum_error(long_true, long_pred)) if finite else float("nan"),
        "density_kl": float(invariant_density_kl(long_true, long_pred)) if finite else float("nan"),
    }


def demo() -> None:
    """Self-check on a task with a known answer: a stable linear rotation.

    A linear map is exactly representable, so the surrogate must roll out accurately for
    many steps. If this fails, the training loop or the rollout is wrong.
    """
    rng = np.random.default_rng(0)
    n = 4000
    th = 0.05
    A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * 0.999
    traj = np.empty((n, 2))
    traj[0] = [1.0, 0.0]
    for i in range(1, n):
        traj[i] = A @ traj[i - 1]
    traj = (traj - traj.mean(0)) / traj.std(0)

    X, Y = traj[:-1], np.diff(traj, axis=0)
    model, info = fit(X[:2000], Y[:2000], budget=2000, epochs=120, seed=0)
    assert 1500 < info["n_params"] < 2500, info["n_params"]

    ev = evaluate_rollout(model, traj[2000:].astype(np.float32), 0.01, steps=50, n_starts=16)
    assert ev["err_by_step"][0] < 0.1, ev["err_by_step"][:3]
    assert not ev["diverged"]
    assert ev["vpt_steps"] > 20, ev["vpt_steps"]

    # rollout must compound: step 20 error >= step 0 error on this contracting system
    assert ev["err_by_step"][-1] >= ev["err_by_step"][0]

    # same seed reproduces exactly
    m2, i2 = fit(X[:2000], Y[:2000], budget=2000, epochs=120, seed=0)
    assert abs(i2["final_train_loss"] - info["final_train_loss"]) < 1e-12

    # horizon-conditioned model trains on (state, horizon) pairs
    hs = rng.integers(1, 33, size=1500)
    xs = traj[:1500]
    ys = np.stack([traj[min(i + int(h), n - 1)] - traj[i] for i, h in enumerate(hs)])
    hm, hinfo = fit(xs, ys, budget=4000, epochs=80, seed=0, horizon=True, H=hs, h_max=32)
    assert hinfo["n_params"] > 3000

    print(
        f"train ok - linear rotation: params {info['n_params']}, "
        f"step-1 err {ev['err_by_step'][0]:.4f}, VPT {ev['vpt_steps']:.0f} steps, "
        f"spectrum err {ev['spectrum_error']:.3f}"
    )


if __name__ == "__main__":
    demo()
