"""Relevance dilution: what happens when most of what a model sees does not matter.

Everything in this project so far studied systems where every measured variable was part of
the dynamics. That is not the situation a world model is in. A world model sees an enormous
number of signals, most of them irrelevant to any particular prediction, arriving on
different clocks, and its hard problem is not "fit the dynamics" but "work out which of
these thousands of things matter here".

So this holds the true dynamics completely fixed and only adds junk alongside it.

    real signal        a 3-variable chaotic system (Lorenz), unchanged throughout
    + D distractors    added columns that carry no information about the real dynamics

Four kinds of distractor, because "irrelevant" is not one thing:

    white       independent noise. Unstructured and unpredictable.
    drift       slow smooth ramps. Irrelevant but HIGHLY predictable, so a model that is
                rewarded for low average error is tempted to spend capacity here.
    foreign     a second, independent chaotic system. Structured, predictable-ish, and on
                its own timescale - the "asynchronous information" case.
    echo        lagged copies of the real variables. Redundant rather than irrelevant, so
                this is the control: it adds dimensions without adding confusion.

The prediction being tested, stated before running:

    A nearest-neighbour lookup should collapse quickly, because distance is computed over
    every column and irrelevant columns dominate it. A trained network should degrade
    gracefully, because it can learn to put near-zero weight on the junk. If so, the thing
    that decides "train a network or just retrieve" is not how much data you have - it is
    what FRACTION of what you measured is relevant.

That would be a more useful rule than anything coverage gave us, and it matches how real
world-model data actually looks.

Metrics logged per fit, because the point is to look for patterns rather than one number:
per-epoch train and held-out loss (so delayed generalisation is visible if it happens),
error at seven horizons, where the model's input sensitivity actually goes, first-layer
effective rank, weight norm, and the lookup baseline under identical conditions.

    python -m run --demo
    python -m run --shard 0 --n-shards 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPL = HERE.parents[0] / "00-replications"
sys.path.insert(0, str(REPL))

from src.common import systems as S  # noqa: E402
from src.common.baselines import knn_rollout, tune_knn  # noqa: E402

RESULTS = HERE / "results"
LOGS = HERE / "logs"
HORIZONS = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------------------
# building a diluted dataset
# ---------------------------------------------------------------------------------------

def make_distractors(kind: str, n: int, d: int, real: np.ndarray, rng) -> np.ndarray:
    """d columns that carry nothing about the real dynamics."""
    if d == 0:
        return np.zeros((n, 0))
    if kind == "white":
        return rng.standard_normal((n, d))
    if kind == "drift":
        t = np.linspace(0, 1, n)[:, None]
        freq = rng.uniform(0.5, 3.0, (1, d))
        phase = rng.uniform(0, 2 * np.pi, (1, d))
        return np.sin(2 * np.pi * freq * t + phase) + 0.02 * rng.standard_normal((n, d))
    if kind == "foreign":
        # an independent chaotic system, tiled to the requested width. Structured and
        # predictable in itself, but unrelated to the target - the asynchronous case.
        other = S.trajectory("Rossler", n + 10)[:n]
        other = (other - other.mean(0)) / (other.std(0) + 1e-9)
        reps = int(np.ceil(d / other.shape[1]))
        tiled = np.tile(other, (1, reps))[:, :d]
        # decorrelate the copies so they are not exact duplicates of each other
        shifts = rng.integers(1, max(2, n // 4), size=d)
        return np.stack([np.roll(tiled[:, j], shifts[j]) for j in range(d)], axis=1)
    if kind == "echo":
        reps = int(np.ceil(d / real.shape[1]))
        tiled = np.tile(real, (1, reps))[:, :d]
        lags = rng.integers(1, 40, size=d)
        return np.stack([np.roll(tiled[:, j], lags[j]) for j in range(d)], axis=1)
    raise ValueError(kind)


def build(kind: str, d: int, n_train: int, n_test: int, gap: int, seed: int):
    """Real dynamics plus d junk columns. Only the real columns are ever scored."""
    rng = np.random.default_rng(seed)
    need = n_train + gap + n_test + 200
    real = S.trajectory("Lorenz", need)[:need]
    real = (real - real.mean(0)) / (real.std(0) + 1e-9)
    junk = make_distractors(kind, len(real), d, real, rng)
    junk = (junk - junk.mean(0)) / (junk.std(0) + 1e-9) if d else junk
    X = np.concatenate([real, junk], axis=1).astype(np.float32)
    tr = X[:n_train]
    te = X[n_train + gap : n_train + gap + n_test]
    return tr, te, real.shape[1]


# ---------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, dim: int, width: int = 96):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim, width), nn.GELU(),
                               nn.Linear(width, width), nn.GELU(),
                               nn.Linear(width, dim))

    def forward(self, x):
        return self.f(x)


def sensitivity(model, X: torch.Tensor, n_real: int) -> dict:
    """Where does the model's input sensitivity actually go?

    Mean absolute gradient of the real-coordinate outputs with respect to each input. If the
    model has learned to ignore the junk, almost all of it sits on the first n_real columns.
    """
    X = X.clone().requires_grad_(True)
    out = model(X)[:, :n_real].abs().sum()
    g = torch.autograd.grad(out, X)[0].abs().mean(0)
    g = g / (g.sum() + 1e-12)
    return {"attn_on_real": float(g[:n_real].sum()),
            "attn_per_real": float(g[:n_real].mean()),
            "attn_per_junk": float(g[n_real:].mean()) if len(g) > n_real else 0.0}


def eff_rank(W: torch.Tensor) -> float:
    """Effective rank: how many directions the first layer actually uses."""
    s = torch.linalg.svdvals(W)
    p = s / (s.sum() + 1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def rollout_err(step_fn, te: np.ndarray, starts, n_real: int, hmax: int) -> dict:
    """Free-running error on the REAL coordinates only, at each horizon."""
    x = te[starts].copy()
    errs = {}
    sd = te[:, :n_real].std(0) + 1e-9
    for h in range(1, hmax + 1):
        x = step_fn(x)
        if h in HORIZONS:
            true = te[starts + h][:, :n_real]
            errs[h] = float(np.sqrt((((x[:, :n_real] - true) / sd) ** 2).mean()))
    return errs


def one_cell(kind: str, d: int, seed: int, args) -> dict:
    tr, te, n_real = build(kind, d, args.n_train, args.n_test, args.gap, seed)
    dim = tr.shape[1]
    torch.manual_seed(seed)

    Xtr = torch.tensor(tr[:-1]); Ytr = torch.tensor(tr[1:] - tr[:-1])
    Xte = torch.tensor(te[:-1]); Yte = torch.tensor(te[1:] - te[:-1])

    model = Net(dim, args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n = len(Xtr)
    curve = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            loss = ((model(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # scored on the REAL coordinates only, so junk cannot flatter the number
                trl = float(((model(Xtr)[:, :n_real] - Ytr[:, :n_real]) ** 2).mean())
                tel = float(((model(Xte)[:, :n_real] - Yte[:, :n_real]) ** 2).mean())
            curve.append((ep, trl, tel))
    train_s = time.time() - t0

    model.eval()

    # no_grad has to be INSIDE the closure: wrapping the def only silences gradients while
    # the function is being defined, not while it runs.
    def step(x):
        with torch.no_grad():
            return x + model(torch.tensor(x, dtype=torch.float32)).numpy()
    starts = np.arange(20, len(te) - max(HORIZONS) - 2, 37)[: args.n_starts]
    e_model = rollout_err(step, te, starts, n_real, max(HORIZONS))

    # ---- lookup on the same data, tuned on a validation slice of train -----------------
    cut = int(len(tr) * 0.8)
    ctx, val = tr[:cut], tr[cut:]
    vs = np.arange(10, min(len(val) - max(HORIZONS) - 2, 400), 8)
    cfg = tune_knn(ctx, vs, val, max(HORIZONS), ks=(1, 4, 16), lookbacks=(1, 2))
    lb = cfg["lookback"]
    q = (np.stack([te[s - lb + 1 : s + 1] for s in starts]) if lb > 1
         else te[starts][:, None, :])
    pk = knn_rollout(ctx, q, max(HORIZONS), k=cfg["k"], weighted=cfg["weighted"])
    sd = te[:, :n_real].std(0) + 1e-9
    e_knn = {h: float(np.sqrt(((((pk[:, h - 1, :n_real]) -
                                te[starts + h][:, :n_real]) / sd) ** 2).mean()))
             for h in HORIZONS}

    sens = sensitivity(model, Xte[:400], n_real)
    W = model.f[0].weight.detach()
    tr_final, te_final = curve[-1][1], curve[-1][2]
    best_te = min(c[2] for c in curve)
    ep_best_te = [c[0] for c in curve if c[2] == best_te][0]
    ep_train_90 = next((c[0] for c in curve if c[1] <= 1.1 * curve[-1][1]), curve[-1][0])

    return {
        "kind": kind, "n_distractors": d, "dim": dim, "n_real": n_real,
        "frac_relevant": n_real / dim, "seed": seed,
        "train_loss": tr_final, "test_loss": te_final,
        "gen_gap": te_final / max(tr_final, 1e-12),
        # delayed generalisation: how long after the training loss settles does the
        # held-out loss keep improving?
        "epoch_train_settled": ep_train_90, "epoch_best_test": ep_best_te,
        "grok_lag": ep_best_te - ep_train_90,
        "weight_norm": float(sum(p.norm() ** 2 for p in model.parameters()) ** 0.5),
        "eff_rank_layer1": eff_rank(W), **sens,
        "knn_k": cfg["k"], "knn_lookback": cfg["lookback"],
        "train_s": round(train_s, 1),
        **{f"model_h{h}": e_model[h] for h in HORIZONS},
        **{f"knn_h{h}": e_knn[h] for h in HORIZONS},
        **{f"ratio_h{h}": e_model[h] / max(e_knn[h], 1e-9) for h in HORIZONS},
        "curve": json.dumps(curve),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["white", "drift", "foreign", "echo"])
    ap.add_argument("--distractors", type=int, nargs="+",
                    default=[0, 3, 6, 12, 24, 48, 96])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--gap", type=int, default=1000)
    ap.add_argument("--n-starts", type=int, default=40)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    jobs = [(k, d, s) for k in args.kinds for d in args.distractors for s in args.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"dilution__shard{args.shard}.csv"
    rows = []
    for i, (k, d, s) in enumerate(jobs, 1):
        try:
            r = one_cell(k, d, s, args)
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {k:8s} d={d:>3d} seed{s}  "
                  f"h1 {r['model_h1']:.3f}/{r['knn_h1']:.3f}  "
                  f"h64 {r['model_h64']:.3f}/{r['knn_h64']:.3f}  "
                  f"attn_real {r['attn_on_real']:.2f}  groklag {r['grok_lag']:>3d}  "
                  f"{r['train_s']}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(jobs)}] FAILED {k} d={d} seed{s}: {exc}", flush=True)
        if rows:
            keys = sorted({kk for r in rows for kk in r})
            with out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


def demo() -> None:
    """Self-check: distractors must be genuinely uninformative, and scoring must ignore them."""
    rng = np.random.default_rng(0)
    tr, te, n_real = build("white", 12, 2000, 1000, 500, 0)
    assert tr.shape[1] == n_real + 12 and n_real == 3

    # a distractor column must not predict the next real state better than chance
    from numpy.linalg import lstsq
    A = tr[:-1, n_real:]
    b = tr[1:, :n_real] - tr[:-1, :n_real]
    coef = lstsq(A, b, rcond=None)[0]
    r2 = 1 - ((b - A @ coef) ** 2).sum() / ((b - b.mean(0)) ** 2).sum()
    assert r2 < 0.05, f"white distractors leak signal: R^2 = {r2:.3f}"

    # echo columns SHOULD be predictive - they are lagged copies. That is the control.
    tr2, _, _ = build("echo", 12, 2000, 1000, 500, 0)
    A2 = tr2[:-1, 3:]; b2 = tr2[1:, :3] - tr2[:-1, :3]
    c2 = lstsq(A2, b2, rcond=None)[0]
    r2e = 1 - ((b2 - A2 @ c2) ** 2).sum() / ((b2 - b2.mean(0)) ** 2).sum()
    assert r2e > r2, f"echo should carry more signal than white, got {r2e:.3f} vs {r2:.3f}"

    class A:
        n_train, n_test, gap, n_starts = 2000, 1000, 500, 20
        width, epochs, batch, lr, wd, log_every = 64, 40, 128, 2e-3, 1e-2, 5

    r = one_cell("white", 6, 0, A)
    assert 0 <= r["attn_on_real"] <= 1
    assert np.isfinite(r["model_h1"]) and np.isfinite(r["knn_h1"])
    print(f"   white d=12 distractor leakage R^2 = {r2:.4f} (echo: {r2e:.4f})")
    print(f"   quick fit d=6: model h1 {r['model_h1']:.3f}, lookup h1 {r['knn_h1']:.3f}, "
          f"sensitivity on real inputs {r['attn_on_real']:.2f} of 1.0")
    print("self-check passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
