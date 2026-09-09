"""Signal delivery: what happens when the information arrives late, or intermittently.

Dilution asked what happens when most of what you see is irrelevant. This asks a different
and equally common question: what happens when the RELEVANT information is there but does
not arrive on time.

That is the normal condition for anything real. A sensor reports on a delay. A market feed
lags the event that moved it. A patient's bloodwork describes last week. A logging pipeline
batches. The signal is present and fully informative - it is just not aligned with the
moment you have to make the prediction.

Four delivery modes, all carrying the SAME underlying dynamics:

    sync          x(t) -> x(t+1). The situation every experiment in this project assumed.
    lagged        the model sees x(t-L) and must still predict x(t+1). Information is
                  complete but stale by L steps.
    intermittent  the observation refreshes only every k steps and is held constant in
                  between, so its staleness varies from 0 to k-1. Asynchronous sampling.
    jittered      each variable has its OWN independent lag. Nothing is aligned with
                  anything else, which is what a multi-source feed actually looks like.

Two things are measured at each setting: how far a surrogate degrades, and whether a delay
embedding recovers it. If handing the model a short window of history fixes lag, then lag
is a representation problem and cheap to solve. If it does not, lag is destroying
information and no amount of architecture will help.

    python lag.py --demo
    python lag.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "00-replications"))

from src.common import systems as S  # noqa: E402
from src.common.baselines import knn_rollout, tune_knn  # noqa: E402

RESULTS = HERE / "results"


def deliver(real: np.ndarray, mode: str, L: int, rng) -> np.ndarray:
    """Return what the model is allowed to SEE at each time index."""
    n, d = real.shape
    if mode == "sync" or L == 0:
        return real.copy()
    if mode == "lagged":
        out = np.roll(real, L, axis=0)
        out[:L] = real[0]
        return out
    if mode == "intermittent":
        # refresh every L steps, hold in between: staleness cycles 0..L-1
        idx = (np.arange(n) // L) * L
        return real[idx]
    if mode == "jittered":
        lags = rng.integers(0, L + 1, size=d)
        out = np.empty_like(real)
        for j in range(d):
            out[:, j] = np.roll(real[:, j], lags[j])
            out[: lags[j], j] = real[0, j]
        return out
    raise ValueError(mode)


class Net(nn.Module):
    def __init__(self, d_in: int, d_out: int, width: int = 96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, width), nn.GELU(),
                               nn.Linear(width, width), nn.GELU(),
                               nn.Linear(width, d_out))

    def forward(self, x):
        return self.f(x)


def windowed(X: np.ndarray, lookback: int) -> np.ndarray:
    """Stack the last `lookback` observations into one vector per time index."""
    if lookback <= 1:
        return X
    pad = np.repeat(X[:1], lookback - 1, axis=0)
    P = np.concatenate([pad, X], axis=0)
    return np.concatenate([P[i : i + len(X)] for i in range(lookback)], axis=1)


def one_cell(mode: str, L: int, lookback: int, seed: int, args) -> dict:
    rng = np.random.default_rng(seed)
    need = args.n_train + args.gap + args.n_test + 100
    real = S.trajectory("Lorenz", need)[:need]
    real = ((real - real.mean(0)) / (real.std(0) + 1e-9)).astype(np.float32)

    seen = deliver(real, mode, L, rng).astype(np.float32)
    Xall = windowed(seen, lookback)
    # target is always the TRUE next state, whatever the model was allowed to see
    Yall = (np.roll(real, -1, axis=0) - real).astype(np.float32)

    a, b = args.n_train, args.n_train + args.gap
    Xtr, Ytr = Xall[:a - 1], Yall[:a - 1]
    Xte, Yte = Xall[b : b + args.n_test], Yall[b : b + args.n_test]

    torch.manual_seed(seed)
    model = Net(Xtr.shape[1], Ytr.shape[1], args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    t0 = time.time()
    for _ in range(args.epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch):
            idx = perm[i : i + args.batch]
            loss = ((model(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).numpy()
    sd = Yte.std(0) + 1e-9
    err_model = float(np.sqrt((((pred - Yte) / sd) ** 2).mean()))
    # predicting "no change" is the floor any useful model must beat
    err_mean = float(np.sqrt((((Yte.mean(0) - Yte) / sd) ** 2).mean()))

    # lookup on exactly the same delivered observations
    cut = int(len(Xtr) * 0.8)
    ctx, val = Xtr[:cut], Xtr[cut:]
    try:
        cfg = tune_knn(ctx, np.arange(8, min(len(val) - 12, 300), 8), val, 4,
                       ks=(1, 4, 16), lookbacks=(1,))
        pk = knn_rollout(ctx, Xte[:, None, :], 1, k=cfg["k"], weighted=cfg["weighted"])
        # the lookup returns a next OBSERVATION; compare its implied increment
        err_knn = float(np.sqrt(
            (((pk[:, 0, : Yte.shape[1]] - Xte[:, : Yte.shape[1]] - Yte) / sd) ** 2).mean()))
    except Exception:  # noqa: BLE001
        err_knn = float("nan")

    return {"mode": mode, "lag": L, "lookback": lookback, "seed": seed,
            "n_features": Xtr.shape[1], "train_s": round(train_s, 1),
            "err_model": err_model, "err_knn": err_knn, "err_mean_baseline": err_mean,
            "skill": 1.0 - err_model / err_mean,
            "beats_mean": bool(err_model < err_mean * 0.98)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+",
                    default=["sync", "lagged", "intermittent", "jittered"])
    ap.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--gap", type=int, default=1000)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    jobs = [(m, L, lb, s) for m in args.modes for L in args.lags
            for lb in args.lookbacks for s in args.seeds
            if not (m == "sync" and L != 0) and not (m != "sync" and L == 0)]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "lag.csv"
    rows = []
    for i, (m, L, lb, s) in enumerate(jobs, 1):
        r = one_cell(m, L, lb, s, args)
        rows.append(r)
        print(f"[{i}/{len(jobs)}] {m:13s} lag={L:>3d} lookback={lb:>2d} seed{s}  "
              f"model {r['err_model']:.3f}  mean-baseline {r['err_mean_baseline']:.3f}  "
              f"skill {r['skill']:+.3f}  {r['train_s']}s", flush=True)
        keys = sorted({k for rr in rows for k in rr})
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


def demo():
    """Self-check: each delivery mode must do what it claims to the data."""
    rng = np.random.default_rng(0)
    real = np.stack([np.arange(100.0), np.arange(100.0) * 2], axis=1)

    s = deliver(real, "sync", 0, rng)
    assert np.array_equal(s, real), "sync must not alter anything"

    lg = deliver(real, "lagged", 5, rng)
    assert np.allclose(lg[20], real[15]), f"lagged by 5 should show t-5, got {lg[20]}"

    it = deliver(real, "intermittent", 10, rng)
    assert np.allclose(it[23], real[20]), "intermittent should hold the last refresh"
    assert np.allclose(it[20], real[20]), "refresh steps should be exact"

    jt = deliver(real, "jittered", 8, rng)
    diffs = [np.argmax(np.isclose(real[:, j], jt[50, j])) for j in range(2)]
    assert 42 <= min(diffs) <= 50, f"jitter should lag each column independently: {diffs}"

    w = windowed(real, 3)
    assert w.shape == (100, 6), w.shape
    assert np.allclose(w[10, :2], real[8]) and np.allclose(w[10, 4:], real[10]), (
        "the window must stack oldest-to-newest")

    class A:
        n_train, n_test, gap, width = 2000, 1000, 500, 64
        epochs, batch, lr = 25, 128, 2e-3

    r0 = one_cell("sync", 0, 1, 0, A)
    r1 = one_cell("lagged", 32, 1, 0, A)
    assert r0["err_model"] < r1["err_model"], (
        f"a 32-step lag must hurt: sync {r0['err_model']:.3f} vs lagged {r1['err_model']:.3f}")
    print(f"   sync skill {r0['skill']:+.3f}   lag-32 skill {r1['skill']:+.3f}")
    print("self-check passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
