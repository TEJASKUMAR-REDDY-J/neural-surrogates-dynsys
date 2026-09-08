"""R9 - if the first attempt is wrong, can the model fix it by thinking again?

The user's question, and a good one. Everything so far has been single-shot: the model looks
at the state and produces an answer. What if instead it produces a draft, looks at the draft,
and revises - repeatedly?

This is the mechanism behind recursive-reasoning models like TRM, and it is the one thing our
battery had not touched. It also connects directly to R8: direct prediction lost to rollout,
but direct prediction only got *one* pass. Rollout gets 64. Maybe the fair comparison is a
direct predictor allowed to think several times.

Three questions, in order of interest:

  1. Does a second pass beat the first? (If not, the idea is dead here.)
  2. Does it keep improving, or settle?
  3. **Can it get worse?** Refinement is a fixed-point iteration, and fixed-point iterations
     can diverge. We test beyond the number of passes the model was trained for, which is
     exactly where a well-behaved method and a lucky one come apart.

Design. One network takes (current state, horizon, current guess) and proposes a correction.
The first guess is just "nothing changes". Training uses deep supervision - every pass is
penalised, not only the last - which is what makes the sequence monotone rather than merely
correct at the end. We train with a fixed number of passes and evaluate with up to four times
as many.

Fair comparison: identical parameter budget and identical number of training pairs as the
direct model from R8, so any difference is the refinement mechanism, not extra capacity.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import systems as S  # noqa: E402
from src.common.baselines import parrot_rollout  # noqa: E402
from src.common.models import count_params, mlp_for_budget, refine_for_budget  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402
from src.common.train import fit, rollout  # noqa: E402

torch.set_num_threads(1)

RESULTS = ROOT / "results" / "r9"
LOGS = ROOT / "logs"

DEFAULT_SYSTEMS = ["Lorenz", "Rossler", "Thomas", "HyperCai", "HenonHeiles", "Bouali2"]


def horizon_pairs(Z: np.ndarray, n_pairs: int, h_max: int, seed: int):
    rng = np.random.default_rng(seed)
    hi = len(Z) - h_max - 1
    t = rng.integers(0, hi, size=n_pairs)
    h = np.clip(np.exp(rng.uniform(0.0, np.log(h_max), size=n_pairs)).round().astype(int), 1, h_max)
    return Z[t].astype(np.float32), Z[t + h].astype(np.float32), h.astype(np.float32)


def train_refiner(X, Y, H, dim, budget, passes, epochs, seed, h_max, batch=256, lr=3e-3):
    """Deep supervision: every pass is penalised, with later passes weighted more heavily.

    Without this the network is free to make passes 1..k-1 meaningless and only get the last
    one right, which would answer a different question than the one being asked.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt, Yt, Ht = (torch.from_numpy(a) for a in (X, Y, H))
    model = refine_for_budget(dim, budget, seed=seed, h_max=h_max)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = nn.MSELoss()
    w = torch.linspace(0.5, 1.5, passes)
    w = w / w.sum()

    n = len(Xt)
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            outs = model(Xt[idx], Ht[idx], passes=passes, return_all=True)
            loss = sum(w[k] * lossfn(o, Yt[idx]) for k, o in enumerate(outs))
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def refine_errors(model, Z, starts, horizons, max_passes):
    """Error after each pass, at each horizon. Shape (passes, horizons)."""
    model.eval()
    sd = Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = np.full((max_passes, len(horizons)), np.nan)
    for j, h in enumerate(horizons):
        x = torch.from_numpy(Z[starts].astype(np.float32))
        hh = torch.full((len(starts),), float(h))
        true = Z[starts + h]
        guess = x
        for k in range(max_passes):
            guess = model.step(x, guess, hh)
            g = guess.numpy()
            if not np.all(np.isfinite(g)):
                break
            out[k, j] = float(np.sqrt((((g - true) / sd) ** 2).mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--n-train", type=int, default=16_000)
    ap.add_argument("--h-max", type=int, default=256)
    ap.add_argument("--horizons", type=int, nargs="+", default=[8, 32, 128, 256])
    ap.add_argument("--train-passes", type=int, default=4)
    ap.add_argument("--eval-passes", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--pts-per-period", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=45_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    args.systems = [s for i, s in enumerate(args.systems) if i % args.n_shards == args.shard]

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    with RunLog("r9_iterative_refinement", vars(args), "cli", LOGS) as log:
        for name in args.systems:
            try:
                spec = S.load_spec(name, pts_per_period=args.pts_per_period)
                X = S.trajectory(name, n=args.n_points, pts_per_period=args.pts_per_period,
                                 use_cache=True, timeout_s=300)
            except Exception as e:  # noqa: BLE001
                log.failure(system=name, error=str(e)[:200])
                print(f"{name}: SETUP FAILED {e}", flush=True)
                continue

            split = args.n_train + 1
            mu, sd = X[:split].mean(0), X[:split].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            Ztr = ((X[:split] - mu) / sd).astype(np.float32)
            Zte = ((X[split : split + 10_000] - mu) / sd).astype(np.float32)
            dim = Ztr.shape[1]

            rng = np.random.default_rng(0)
            starts = rng.choice(len(Zte) - args.h_max - 1, size=256, replace=False)
            sdz = np.where(Zte.std(0) > 0, Zte.std(0), 1.0)

            # training-free reference on the same starts
            par = parrot_rollout(Ztr, Zte[starts], max(args.horizons))
            par_err = {h: float(np.sqrt((((par[:, h - 1] - Zte[starts + h]) / sdz) ** 2).mean()))
                       for h in args.horizons}

            print(f"\n=== {name}  lam={spec.lyap_max:.3f}  dim={dim} ===", flush=True)

            for seed in args.seeds:
                t0 = time.time()
                Xp, Yp, Hp = horizon_pairs(Ztr, len(Ztr) - 1, args.h_max, seed)

                # baseline 1: one-step model rolled out, same budget
                m_roll, i_roll = fit(Ztr[:-1], np.diff(Ztr, axis=0), budget=args.budget,
                                     epochs=args.epochs, seed=seed)
                pr = rollout(m_roll, Zte[starts], max(args.horizons))
                roll_err = {h: float(np.sqrt((((pr[:, h - 1] - Zte[starts + h]) / sdz) ** 2).mean()))
                            for h in args.horizons}

                # baseline 2: single-pass direct predictor, same budget and pairs
                m_dir, i_dir = fit(Xp, Yp - Xp, budget=args.budget, epochs=args.epochs,
                                   seed=seed, horizon=True, H=Hp, h_max=args.h_max)
                with torch.no_grad():
                    m_dir.eval()
                    dir_err = {}
                    for h in args.horizons:
                        xx = torch.from_numpy(Zte[starts].astype(np.float32))
                        p = (xx + m_dir(xx, torch.full((len(starts),), float(h)))).numpy()
                        dir_err[h] = float(np.sqrt((((p - Zte[starts + h]) / sdz) ** 2).mean()))

                # the refiner
                m_ref = train_refiner(Xp, Yp, Hp, dim, args.budget, args.train_passes,
                                      args.epochs, seed, args.h_max)
                errs = refine_errors(m_ref, Zte, starts, args.horizons, args.eval_passes)

                for j, h in enumerate(args.horizons):
                    for k in range(args.eval_passes):
                        rows.append({
                            "system": name, "seed": seed, "horizon": h, "passes": k + 1,
                            "err_refine": errs[k, j],
                            "err_rollout": roll_err[h], "err_direct": dir_err[h],
                            "err_parrot": par_err[h],
                            "train_passes": args.train_passes,
                            "params_refine": count_params(m_ref),
                            "params_direct": i_dir["n_params"],
                            "lyap_max": spec.lyap_max,
                            "lyap_time": spec.lyap_time_per_step * h,
                            "kaplan_yorke_dim": spec.invariants["kaplan_yorke_dim"],
                        })
                log.result(system=name, seed=seed, errs=errs.tolist(),
                           horizons=args.horizons, wall_s=round(time.time() - t0, 1))
                for j, h in enumerate(args.horizons):
                    seq = " ".join(f"{errs[k, j]:.3f}" for k in range(min(8, args.eval_passes)))
                    print(f"  seed {seed} h={h:>4}: passes 1-8 -> {seq}   "
                          f"(direct {dir_err[h]:.3f}, rollout {roll_err[h]:.3f})", flush=True)
                print(f"    [{time.time() - t0:.0f}s]", flush=True)

    keys = sorted({k for r in rows for k in r})
    out = RESULTS / (f"refinement" + (f"__shard{args.shard}" if args.n_shards > 1 else "") + ".csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
