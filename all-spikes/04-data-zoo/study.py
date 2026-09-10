"""Many architectures, many datasets: is the outcome a property of the model or of the data?

Every result in this project so far has pointed the same way - architecture matters much
less than expected, and something about the data decides almost everything. That was found
on chaotic simulations and on one masking task. This tests it across 47 real and simulated
datasets spanning weather, sales, macroeconomics, sensors, audio, biomedical panels,
chemistry, images, simulated chaos, and three controls with known answers.

ONE task everywhere, so nothing is confounded by framing: standardise the data, take a
window of the last W observations, predict the next one. Blocked split with a gap so
nothing adjacent in time leaks across it.

EIGHT methods spanning the space of what people actually reach for:

    mean          predict the training mean. The floor - anything not beating this is
                  measuring nothing.
    persistence   predict no change.
    ridge         linear, closed form.
    knn           analogue lookup, no training at all.
    trees         gradient boosting - the thing that usually wins on tabular data.
    mlp           a plain feed-forward net.
    cnn           1-D convolution over the window, so time is treated as structure.
    gru           a recurrent net over the window.

And for each dataset, properties measurable BEFORE any of them are trained: coverage, an
estimated noise floor, how far the distribution moves between train and test, how fast the
signal decorrelates, its spectral concentration, dimension and length.

The three questions:
  1. how big is the spread ACROSS architectures compared to the spread ACROSS datasets?
  2. which data properties predict the best achievable error?
  3. do any data properties predict WHICH architecture wins?

    python study.py --demo
    python study.py --shard 0 --n-shards 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"

W = 8            # window length, fixed everywhere
GAP_FRAC = 0.05  # blocked split gap, as a fraction of the series


# ---------------------------------------------------------------------------------------
# data properties, none of which require training
# ---------------------------------------------------------------------------------------

def properties(Xtr: np.ndarray, Xte: np.ndarray, raw: np.ndarray,
               Ytr: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(0)
    q = Xte[rng.choice(len(Xte), min(300, len(Xte)), replace=False)]
    sub = rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False)
    ctx = Xtr[sub]
    nxt = (Ytr[sub] if Ytr is not None else Xtr[sub])
    nn = np.sqrt(((q[:, None, :] - ctx[None, :, :]) ** 2).sum(-1)).min(1)
    a = ctx[rng.choice(len(ctx), min(300, len(ctx)))]
    b = ctx[rng.choice(len(ctx), min(300, len(ctx)))]
    spread = float(np.sqrt(((a - b) ** 2).sum(-1)).mean()) + 1e-12

    # Noise floor: find near-identical INPUTS, then measure how much their NEXT VALUES
    # disagree. Two inputs that are the same cannot produce different futures because of
    # the input, so whatever difference remains is noise.
    #
    # This was wrong in the first run. `tgt` was set to the input array rather than the
    # targets, so the quantity measured was the distance to your own nearest neighbours -
    # a local density measure, not a noise measure, despite the name. Corrected here.
    # The two correlate at rho 0.95 and the corrected version predicts the achievable
    # ceiling slightly better (-0.871 against -0.820), so every conclusion drawn from the
    # first run stands; only the interpretation of the number needed fixing.
    k = min(5, len(ctx) - 1)
    d2 = ((ctx[:, None, :] - ctx[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    tgt = nxt
    disagree = ((tgt[idx] - tgt[:, None, :]) ** 2).mean(axis=(1, 2))
    noise = float(np.median(disagree) / 2.0 / (tgt.var() + 1e-12))

    # how far does the distribution move between train and test?
    shift = float(np.abs(Xtr.mean(0) - Xte.mean(0)).mean() / (Xtr.std(0).mean() + 1e-12))

    # decorrelation time on the raw series
    z = (raw - raw.mean(0)) / (raw.std(0) + 1e-12)
    ac = []
    for lag in range(1, min(60, len(z) // 4)):
        ac.append(float((z[:-lag] * z[lag:]).mean()))
    tau = next((i + 1 for i, v in enumerate(ac) if v < 1 / np.e), len(ac) or 1)

    # spectral concentration of the first coordinate
    f = np.abs(np.fft.rfft(z[:, 0] - z[:, 0].mean()))
    p = f / (f.sum() + 1e-12)
    sent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))

    return {"coverage": float(np.median(nn) / spread), "noise_floor": min(noise, 2.0),
            "dist_shift": shift, "decorr_time": int(tau), "spectral_entropy": sent,
            "n_dims": int(raw.shape[1]), "n_rows": int(len(raw))}


# ---------------------------------------------------------------------------------------

def windows(X: np.ndarray, w: int):
    n = len(X) - w
    Xw = np.stack([X[i : i + w].ravel() for i in range(n)])
    Y = X[w : w + n]
    return Xw.astype(np.float32), Y.astype(np.float32)


def fit_torch(kind, Xtr, Ytr, Xte, dims, seed, epochs=120):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    w, d = W, dims
    if kind == "mlp":
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 96), nn.GELU(),
                            nn.Linear(96, 96), nn.GELU(), nn.Linear(96, d))
        shape = None
    elif kind == "cnn":
        class C(nn.Module):
            def __init__(s):
                super().__init__()
                s.c = nn.Sequential(nn.Conv1d(d, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv1d(32, 32, 3, padding=1), nn.GELU())
                s.o = nn.Linear(32 * w, d)

            def forward(s, x):
                h = s.c(x.reshape(-1, w, d).transpose(1, 2))
                return s.o(h.flatten(1))
        net, shape = C(), "seq"
    elif kind == "gru":
        class G(nn.Module):
            def __init__(s):
                super().__init__()
                s.r = nn.GRU(d, 48, batch_first=True)
                s.o = nn.Linear(48, d)

            def forward(s, x):
                h, _ = s.r(x.reshape(-1, w, d))
                return s.o(h[:, -1])
        net, shape = G(), "seq"
    else:
        raise ValueError(kind)

    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
    bs = min(128, max(16, len(Xt) // 8))
    for _ in range(epochs):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = p[i : i + bs]
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte)).numpy()


def run_dataset(ds: dict, seed: int, methods: list[str]) -> list[dict]:
    raw = np.asarray(ds["X"], float)
    mu, sd = raw.mean(0), raw.std(0)
    Z = ((raw - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)

    n = len(Z)
    gap = max(5, int(n * GAP_FRAC))
    cut = int(n * 0.7)
    tr_raw, te_raw = Z[:cut], Z[cut + gap :]
    if len(te_raw) < W + 25:
        return []

    Xtr, Ytr = windows(tr_raw, W)
    Xte, Yte = windows(te_raw, W)
    if len(Xtr) < 40 or len(Xte) < 15:
        return []

    d = Z.shape[1]
    props = properties(Xtr, Xte, raw, Ytr)
    var = Yte.var() + 1e-12
    base = {"dataset": ds["name"], "kind": ds["kind"], "source": ds["source"],
            "seed": seed, **props}
    rows = []

    def record(name, pred, secs):
        pred = np.asarray(pred, float).reshape(Yte.shape)
        mse = float(((pred - Yte) ** 2).mean())
        rows.append({**base, "method": name, "mse": mse,
                     "nrmse": float(np.sqrt(mse / var)),
                     "r2": float(1 - mse / var), "fit_s": round(secs, 2)})

    for m in methods:
        t0 = time.time()
        try:
            if m == "mean":
                record(m, np.repeat(Ytr.mean(0)[None], len(Yte), 0), time.time() - t0)
            elif m == "persistence":
                record(m, Xte[:, -d:], time.time() - t0)
            elif m == "ridge":
                from sklearn.linear_model import Ridge
                p = Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte)
                record(m, p, time.time() - t0)
            elif m == "knn":
                from sklearn.neighbors import KNeighborsRegressor
                k = min(8, max(1, len(Xtr) // 20))
                p = KNeighborsRegressor(k, weights="distance").fit(Xtr, Ytr).predict(Xte)
                record(m, p, time.time() - t0)
            elif m == "trees":
                from sklearn.ensemble import HistGradientBoostingRegressor
                P = np.column_stack([
                    HistGradientBoostingRegressor(max_iter=120, random_state=seed)
                    .fit(Xtr, Ytr[:, j]).predict(Xte) for j in range(d)])
                record(m, P, time.time() - t0)
            elif m in ("mlp", "cnn", "gru"):
                p = fit_torch(m, Xtr, Ytr, Xte, d, seed)
                record(m, p, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            rows.append({**base, "method": m, "mse": float("nan"), "nrmse": float("nan"),
                         "r2": float("nan"), "fit_s": 0.0, "error": repr(exc)[:120]})
    return rows


def main():
    from fetch import load_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["mean", "persistence", "ridge", "knn", "trees",
                             "mlp", "cnn", "gru"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    sets = load_all(verbose=False)
    jobs = [(ds, s) for ds in sets for s in args.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"zoo__shard{args.shard}.csv"
    rows = []
    for i, (ds, s) in enumerate(jobs, 1):
        t0 = time.time()
        r = run_dataset(ds, s, args.methods)
        rows += r
        if r:
            ok = [x for x in r if np.isfinite(x["r2"])]
            best = max(ok, key=lambda x: x["r2"]) if ok else None
            print(f"[{i}/{len(jobs)}] {ds['name']:22s} seed{s}  "
                  f"best {best['method'] if best else '-':11s} "
                  f"R2 {best['r2'] if best else float('nan'):+.3f}  "
                  f"cov {r[0]['coverage']:.3f}  shift {r[0]['dist_shift']:.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        else:
            print(f"[{i}/{len(jobs)}] {ds['name']:22s} seed{s}  SKIPPED (too short)",
                  flush=True)
        if rows:
            keys = sorted({k for x in rows for k in x})
            with out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


def demo():
    from fetch import load_all
    sets = load_all(verbose=False)
    print(f"{len(sets)} datasets available")
    for nm in ("sim_Lorenz", "ctrl_white_noise", "ctrl_periodic"):
        ds = next(d for d in sets if d["name"] == nm)
        r = run_dataset(ds, 0, ["mean", "ridge", "knn", "mlp"])
        best = max((x for x in r if np.isfinite(x["r2"])), key=lambda x: x["r2"])
        print(f"  {nm:18s} best {best['method']:8s} R2 {best['r2']:+.3f}  "
              f"coverage {r[0]['coverage']:.3f}  noise {r[0]['noise_floor']:.3f}")
        if nm == "ctrl_white_noise":
            assert best["r2"] < 0.15, f"white noise must be unpredictable, got {best['r2']}"
        if nm == "ctrl_periodic":
            assert best["r2"] > 0.8, f"a periodic signal must be easy, got {best['r2']}"
    print("self-check passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
