"""Training-free forecast baselines.

These exist because of Zhang & Gilpin, "Context parroting" (arXiv:2505.11349): a
parameter-free method that copies a matching segment out of the context window beats
leading time-series foundation models on chaotic systems. A chaotic trajectory revisits
similar states, so copying what happened last time you were here is a strong strategy -
and it involves no learning at all.

If a trained surrogate cannot beat these, it is doing recall, not rule-learning, and any
claim about it "having learned the dynamics" is unsupported.
"""

from __future__ import annotations

import numpy as np


def persistence_rollout(x0: np.ndarray, steps: int) -> np.ndarray:
    """Predict that nothing changes. The floor under every forecast."""
    x0 = np.asarray(x0, float)
    return np.repeat(x0[:, None, :], steps, axis=1)


def mean_rollout(x0: np.ndarray, steps: int, mean: np.ndarray) -> np.ndarray:
    """Predict the climatological mean. What a collapsed model converges to."""
    out = np.empty((len(x0), steps, len(mean)))
    out[:] = np.asarray(mean, float)
    return out


def parrot_rollout(
    context: np.ndarray,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Context parroting / nearest-neighbour analogue forecasting.

    For each query state, find the point in `context` whose preceding `lookback` states
    most closely match the query's, then copy the `steps` states that followed it.

    `context` is the history the method is allowed to look at. `x0` is (n_starts, dim) or
    (n_starts, lookback, dim) when lookback > 1.
    """
    context = np.asarray(context, float)
    n, dim = context.shape
    x0 = np.asarray(x0, float)
    if x0.ndim == 2:
        x0 = x0[:, None, :]
    lookback = x0.shape[1]

    # candidate anchors: index j means the window context[j-lookback+1 : j+1]
    lo, hi = lookback - 1, n - steps - 1
    if hi <= lo:
        raise ValueError("context too short for this rollout length")
    anchors = np.arange(lo, hi)

    # (n_anchor, lookback, dim) windows, built by striding
    win = np.stack([context[anchors - (lookback - 1) + t] for t in range(lookback)], axis=1)
    flatwin = win.reshape(len(anchors), -1)
    flatq = x0.reshape(len(x0), -1)

    out = np.empty((len(x0), steps, dim))
    for i in range(len(flatq)):
        d = np.sqrt(((flatwin - flatq[i]) ** 2).sum(1))
        j = anchors[int(np.argmin(d))]
        out[i] = context[j + 1 : j + 1 + steps]
    return out


def demo() -> None:
    rng = np.random.default_rng(0)

    # a periodic signal: parroting should be nearly exact, persistence should not
    t = np.linspace(0, 40 * np.pi, 4000)
    traj = np.c_[np.sin(t), np.cos(t)]
    ctx, test = traj[:3000], traj[3000:]

    starts = np.array([10, 200, 400])
    x0 = test[starts]
    steps = 50
    true = np.stack([test[s + 1 : s + 1 + steps] for s in starts])

    par = parrot_rollout(ctx, x0, steps)
    per = persistence_rollout(x0, steps)

    e_par = np.sqrt(((par - true) ** 2).mean())
    e_per = np.sqrt(((per - true) ** 2).mean())
    assert e_par < 0.35 * e_per, (e_par, e_per)

    # lookback > 1 disambiguates states that a single sample cannot distinguish
    x0m = np.stack([test[s - 3 : s + 1] for s in starts])
    par_lb = parrot_rollout(ctx, x0m, steps)
    e_lb = np.sqrt(((par_lb - true) ** 2).mean())
    assert e_lb <= e_par + 1e-9, (e_lb, e_par)

    # on white noise parroting must be no better than persistence-ish, i.e. bad
    noise = rng.standard_normal((4000, 2))
    par_n = parrot_rollout(noise[:3000], noise[3000:][starts], steps)
    true_n = np.stack([noise[3000:][s + 1 : s + 1 + steps] for s in starts])
    e_noise = np.sqrt(((par_n - true_n) ** 2).mean())
    assert e_noise > 1.0, e_noise

    m = mean_rollout(x0, steps, ctx.mean(0))
    assert m.shape == (3, steps, 2)

    print(
        f"baselines ok - periodic: parrot rmse {e_par:.4f} vs persistence {e_per:.4f}; "
        f"lookback-4 {e_lb:.4f}; white noise parrot {e_noise:.2f}"
    )


if __name__ == "__main__":
    demo()
