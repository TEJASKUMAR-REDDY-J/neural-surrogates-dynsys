"""The task, the training loop, and the methods that do not learn anything.

**The task, in one sentence:** part of the lattice is hidden, and the job is to work out what
was there.

That one task covers every modality we have. Hide random cells of an image and it is
inpainting. Hide random characters and it is text repair. Hide the *end* of a time window and
it is forecasting. So a single pipeline compares an architecture across data that normally
requires completely different setups - which is the whole point of the spike.

Three ways of hiding, because they ask different questions:

    random   scattered single cells       - a neighbour usually knows the answer
    block    one contiguous chunk         - the neighbours are missing too, so a cell has to
                                            rely on something further away
    tail     the last quarter of the line - this is forecasting

`block` is where a global branch should start to earn its keep, and `random` is where it
should not matter much. That contrast is the experiment.

The five non-learning baselines exist because of what the replication battery found: a
fifteen-line lookup ties with a trained network on the standard chaotic-systems benchmark.
No claim about an architecture means anything until it has cleared them.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

CORRUPTIONS = ("random", "block", "tail")


# ---------------------------------------------------------------------------------------
# hiding things
# ---------------------------------------------------------------------------------------

def make_mask(shape: tuple, spatial: tuple, frac: float, mode: str, rng) -> np.ndarray:
    """1.0 = the cell is visible, 0.0 = hidden. Shape (n, 1, *spatial)."""
    n = shape[0]
    m = np.ones((n, 1, *spatial), dtype=np.float32)
    if mode == "random":
        m *= (rng.random((n, 1, *spatial)) > frac).astype(np.float32)
        return m
    if mode == "tail":
        k = max(1, int(round(frac * spatial[-1])))
        m[..., -k:] = 0.0
        return m
    if mode == "block":
        if len(spatial) == 1:
            k = max(1, int(round(frac * spatial[0])))
            for i in range(n):
                s = rng.integers(0, max(1, spatial[0] - k + 1))
                m[i, 0, s:s + k] = 0.0
        else:
            h, w = spatial
            kh = max(1, int(round(np.sqrt(frac) * h)))
            kw = max(1, int(round(np.sqrt(frac) * w)))
            for i in range(n):
                y = rng.integers(0, max(1, h - kh + 1))
                x = rng.integers(0, max(1, w - kw + 1))
                m[i, 0, y:y + kh, x:x + kw] = 0.0
        return m
    raise ValueError(f"unknown corruption mode {mode}")


def corrupt(X: np.ndarray, mask: np.ndarray, noise: float, rng) -> np.ndarray:
    """Hidden cells go to 0.5 (a neutral grey), and visible ones may be measured imperfectly."""
    obs = X * mask + 0.5 * (1.0 - mask)
    if noise > 0:
        obs = obs + noise * rng.standard_normal(obs.shape).astype(np.float32) * mask
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------

def nrmse(pred: np.ndarray, true: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Root mean squared error over the data's own spread. 1.0 = as bad as guessing the mean."""
    sd = true.std() + 1e-12
    d = (pred - true) ** 2
    if weight is not None:
        w = weight.sum()
        if w < 1:
            return float("nan")
        return float(np.sqrt((d * weight).sum() / w) / sd)
    return float(np.sqrt(d.mean()) / sd)


def _masked(a: np.ndarray, w: np.ndarray) -> float:
    tot = w.sum()
    return float((a * w).sum() / tot) if tot >= 1 else float("nan")


def score(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, binary: bool) -> dict:
    """Every metric asked for, on the hidden cells and on the lattice as a whole.

    They are not redundant. MSE punishes a few large misses; MAE does not. nRMSE divides by
    the data's own spread so different datasets are comparable and 1.0 always means "no
    better than guessing the mean". Relative L2 is the norm ratio the PDE-surrogate
    literature reports, so our numbers can be read next to theirs.
    """
    hidden = np.broadcast_to(1.0 - mask, true.shape).astype(float)
    err = pred - true
    out = {
        "nrmse_all": nrmse(pred, true),
        "nrmse_hidden": nrmse(pred, true, hidden),
        "mse_all": float((err ** 2).mean()),
        "mae_all": float(np.abs(err).mean()),
        "mse_hidden": _masked(err ** 2, hidden),
        "mae_hidden": _masked(np.abs(err), hidden),
        # relative L2: ||pred - true|| / ||true||, the neural-operator convention
        "rel_l2_all": float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12)),
        "rel_l2_hidden": float(
            np.sqrt((err ** 2 * hidden).sum()) / (np.sqrt((true ** 2 * hidden).sum()) + 1e-12)
        ),
    }
    if binary:
        hit = ((pred > 0.5) == (true > 0.5)).astype(float)
        h = np.broadcast_to(hidden, true.shape)
        out["acc_hidden"] = float((hit * h).sum() / max(h.sum(), 1))
    return out


# ---------------------------------------------------------------------------------------
# the methods that do not learn
# ---------------------------------------------------------------------------------------

def baseline_predict(name: str, obs, mask, train_X, spatial) -> np.ndarray:
    """Five ways to fill in the blanks without training anything."""
    n = obs.shape[0]
    flat_obs = obs.reshape(n, -1)
    flat_mask = np.broadcast_to(mask, obs.shape).reshape(n, -1)
    flat_tr = train_X.reshape(len(train_X), -1)

    if name == "identity":
        # leave it exactly as it came. The floor under everything.
        return obs.copy()

    if name == "global_mean":
        # every hidden cell gets the training average for that position
        mu = flat_tr.mean(0)
        out = flat_obs * flat_mask + mu[None, :] * (1 - flat_mask)
        return out.reshape(obs.shape)

    if name == "local_mean":
        # diffuse the visible values into the holes. Classical inpainting, no training.
        x = torch.tensor(obs * mask)
        m = torch.tensor(np.broadcast_to(mask, obs.shape).copy())
        nd = len(spatial)
        pad = (1, 1) * nd
        conv = F.conv1d if nd == 1 else F.conv2d
        k = torch.ones((1, 1, *([3] * nd)))
        for _ in range(32):
            num = conv(F.pad(x, pad, mode="circular"), k)
            den = conv(F.pad(m, pad, mode="circular"), k).clamp(min=1e-6)
            fill = num / den
            x = torch.where(m > 0, x, fill)
            m = torch.clamp(m + (den > 1e-6).float(), max=1.0)
        return x.numpy()

    if name == "nearest_neighbour":
        # analogue lookup: whose training example looks most like what we can see?
        # Copy its values into the holes. This is context parroting, generalised.
        out = obs.copy().reshape(n, -1)
        for i in range(n):
            vis = flat_mask[i] > 0
            if vis.sum() == 0:
                continue
            d = ((flat_tr[:, vis] - flat_obs[i, vis]) ** 2).sum(1)
            out[i, ~vis] = flat_tr[int(d.argmin()), ~vis]
        return out.reshape(obs.shape)

    if name == "pca_impute":
        # the data lies near a low-dimensional plane; find the point on it that agrees with
        # what we can see, and read off the rest.
        mu = flat_tr.mean(0)
        Z = flat_tr - mu
        q = min(24, Z.shape[0] - 1, Z.shape[1])
        V = np.linalg.svd(Z, full_matrices=False)[2][:q]                 # (q, d)
        out = flat_obs.copy()
        for i in range(n):
            vis = flat_mask[i] > 0
            A = V[:, vis].T
            b = flat_obs[i, vis] - mu[vis]
            c = np.linalg.lstsq(A, b, rcond=None)[0] if A.shape[0] >= 1 else np.zeros(q)
            rec = mu + c @ V
            out[i, ~vis] = rec[~vis]
        return out.reshape(obs.shape)

    raise ValueError(f"unknown baseline {name}")


BASELINES = ("identity", "global_mean", "local_mean", "nearest_neighbour", "pca_impute")


# ---------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------

def make_optimizer(name: str, params, lr: float, epochs: int):
    opts = {
        "adamw": lambda: torch.optim.AdamW(params, lr=lr, weight_decay=1e-4),
        "nadam": lambda: torch.optim.NAdam(params, lr=lr, weight_decay=1e-4),
        "radam": lambda: torch.optim.RAdam(params, lr=lr, weight_decay=1e-4),
        "adamax": lambda: torch.optim.Adamax(params, lr=lr, weight_decay=1e-4),
        "sgdm": lambda: torch.optim.SGD(params, lr=lr * 10, momentum=0.9, nesterov=True),
    }
    opt = opts[name]()
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=opt.param_groups[0]["lr"], total_steps=max(epochs, 1), pct_start=0.25
    )
    return opt, sched


def train(
    model, train_X, spatial, *, epochs: int = 40, batch: int = 32, lr: float = 2e-3,
    steps_lo: int = 4, steps_hi: int = 10, frac: float = 0.3, mode: str = "random",
    noise: float = 0.0, optimizer: str = "adamw", seed: int = 0, log_every: int = 10,
) -> dict:
    """Teach the automaton to fill in what has been hidden.

    Two details that matter, both borrowed from the NCA literature:

    * **the number of iterations is random each batch** (between `steps_lo` and `steps_hi`).
      Train at a fixed count and the rule learns to be right only at that count, then falls
      apart when run longer - which is exactly the failure the earlier refinement experiment
      found. Randomising it asks for a rule that is stable, not one that is punctual.
    * **the loss is on every cell**, not only the hidden ones, so the rule has to hold what
      it can already see steady instead of scribbling over it.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt, sched = make_optimizer(optimizer, model.parameters(), lr, epochs)
    Xt = torch.tensor(train_X)
    history = []
    t0 = time.time()

    model.train()
    n = len(train_X)
    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss, nb = 0.0, 0
        for b0 in range(0, n, batch):
            idx = order[b0:b0 + batch]
            if len(idx) < 2:
                continue
            xb = Xt[idx]
            m = torch.tensor(make_mask((len(idx),), spatial, frac, mode, rng))
            obs = torch.tensor(corrupt(train_X[idx], m.numpy(), noise, rng))
            k = int(rng.integers(steps_lo, steps_hi + 1))

            pred = model(obs, m, steps=k)
            loss = F.mse_loss(pred, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # NCA training is
            opt.step()                                                   # unstable without it
            ep_loss += float(loss.item()); nb += 1
        sched.step()
        if ep % log_every == 0 or ep == epochs - 1:
            history.append({"epoch": ep, "loss": ep_loss / max(nb, 1)})

    return {
        "train_loss": history[-1]["loss"] if history else float("nan"),
        "train_wall_s": round(time.time() - t0, 2),
        "history": history,
    }


@torch.no_grad()
def evaluate(model, X, spatial, *, steps_grid=(1, 2, 4, 8, 16, 32), frac=0.3, mode="random",
             noise=0.0, binary=False, seed=0) -> dict:
    """Score at several iteration counts, including well past what it trained on.

    Reporting only the trained count would hide the failure mode we care about: a rule that
    looks good at 8 steps and destroys the answer by 32 is not a stable rule.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    m = make_mask((len(X),), spatial, frac, mode, rng)
    obs = corrupt(X, m, noise, rng)
    ot, mt = torch.tensor(obs), torch.tensor(m)

    per_step, s = {}, model.seed(ot, mt)
    for k in range(1, max(steps_grid) + 1):
        s = model.step(s)
        if k in steps_grid:
            per_step[k] = score(s[:, : model.c_data].numpy(), X, m, binary)

    best_k = min(per_step, key=lambda k: per_step[k]["nrmse_hidden"])
    last_k = max(steps_grid)
    return {
        "per_step": per_step,
        "best_steps": best_k,
        "best": per_step[best_k],
        "final": per_step[last_k],
        "overrun_ratio": per_step[last_k]["nrmse_hidden"] / max(per_step[best_k]["nrmse_hidden"], 1e-9),
        "mask": m,
        "obs": obs,
    }


def demo() -> None:
    """Self-check: masks hide what they claim to, and the baselines behave as expected."""
    rng = np.random.default_rng(0)
    spatial = (16,)
    X = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (60, 1))[:, None, :]
    X = X + 0.01 * rng.standard_normal(X.shape).astype(np.float32)

    for mode in CORRUPTIONS:
        m = make_mask((60,), spatial, 0.25, mode, rng)
        got = 1.0 - m.mean()
        assert 0.1 < got < 0.45, f"{mode}: hid {got:.2f} of cells, expected about 0.25"
    assert make_mask((5,), spatial, 0.25, "tail", rng)[0, 0, -1] == 0.0, "tail must hide the end"

    m = make_mask((60,), spatial, 0.3, "random", rng)
    obs = corrupt(X, m, 0.0, rng)
    tr, te = X[:40], X[40:]
    errs = {}
    for b in BASELINES:
        p = baseline_predict(b, obs[40:], m[40:], tr, spatial)
        assert p.shape == te.shape, f"{b}: shape {p.shape} != {te.shape}"
        assert np.isfinite(p).all(), f"{b}: non-finite output"
        # a baseline must not touch what it was allowed to see
        vis = np.broadcast_to(m[40:], te.shape) > 0
        if b != "local_mean":
            assert np.allclose(p[vis], obs[40:][vis], atol=1e-5), f"{b} overwrote visible cells"
        errs[b] = score(p, te, m[40:], binary=False)["nrmse_hidden"]

    assert errs["identity"] > errs["local_mean"], "smooth data: diffusion must beat doing nothing"
    assert errs["identity"] > errs["nearest_neighbour"], "lookup must beat doing nothing"
    print("self-check passed - baseline nrmse on hidden cells:")
    for k, v in sorted(errs.items(), key=lambda kv: kv[1]):
        print(f"   {k:20s} {v:.4f}")


if __name__ == "__main__":
    demo()
