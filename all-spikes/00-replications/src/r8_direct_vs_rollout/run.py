"""R8 - one big jump or many small steps, and what that says about why rollout fails.

Two ways to predict H steps ahead:
  rollout - predict one step, feed the answer back, repeat H times. Errors compound.
  direct  - one network conditioned on the horizon, jumping straight there in a single
            forward pass. No compounding, but it must learn a different effective map for
            every H.

Lakshmanan & Chopra (arXiv:2605.28317) showed the direct route works, so the yes/no
question is settled. The open question is quantitative: for which systems, and how far.

But the sharper reason to run this is Shikhman (arXiv:2601.11428). Across 750 models, all
three architectures failed badly at long rollout on Navier-Stokes and Kuramoto-Sivashinsky,
and the paper offers a bound rather than an explanation. Two candidates:

  - the failure is error *compounding*, a property of the rollout procedure
  - the failure is a *system ceiling*, because the information is not there

A direct model does not compound. So if direct prediction escapes the failure, the cause
was compounding; if it fails just as badly at the same horizon, the cause is the system.
That is the Gilpin-versus-Duraisamy dispute, separated cheaply.

Both modes get matched parameter counts and matched numbers of training pairs.
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import systems as S  # noqa: E402
from src.common.baselines import parrot_rollout  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402
from src.common.train import fit, rollout  # noqa: E402

RESULTS = ROOT / "results" / "r8"
LOGS = ROOT / "logs"

DEFAULT_SYSTEMS = ["Lorenz", "Rossler", "Halvorsen", "Chua", "Aizawa", "Thomas"]


def horizon_dataset(Z: np.ndarray, n_pairs: int, h_max: int, seed: int) -> dict:
    """(state, horizon) -> displacement over that horizon. Horizons sampled log-uniformly
    so short and long horizons get comparable coverage."""
    rng = np.random.default_rng(seed)
    hi = len(Z) - h_max - 1
    t = rng.integers(0, hi, size=n_pairs)
    h = np.exp(rng.uniform(0.0, np.log(h_max), size=n_pairs)).round().astype(int)
    h = np.clip(h, 1, h_max)
    X = Z[t]
    Y = Z[t + h] - Z[t]
    return {"X": X.astype(np.float32), "Y": Y.astype(np.float32), "H": h.astype(np.float32)}


@torch.no_grad()
def direct_errors(model, Z: np.ndarray, starts: np.ndarray, horizons: list[int]) -> list[float]:
    model.eval()
    x0 = torch.from_numpy(Z[starts].astype(np.float32))
    sd = Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = []
    for h in horizons:
        pred = (x0 + model(x0, torch.full((len(x0),), float(h)))).numpy()
        true = Z[starts + h]
        out.append(float(np.sqrt((((pred - true) / sd) ** 2).mean())))
    return out


def rollout_errors(model, Z: np.ndarray, starts: np.ndarray, horizons: list[int]) -> list[float]:
    steps = max(horizons)
    pred = rollout(model, Z[starts].astype(np.float32), steps)
    sd = Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = []
    for h in horizons:
        p = pred[:, h - 1]
        true = Z[starts + h]
        out.append(float(np.sqrt((((p - true) / sd) ** 2).mean())))
    return out


def parrot_errors(Z_ctx: np.ndarray, Z: np.ndarray, starts: np.ndarray, horizons: list[int]) -> list[float]:
    steps = max(horizons)
    pred = parrot_rollout(Z_ctx, Z[starts], steps)
    sd = Z.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [
        float(np.sqrt((((pred[:, h - 1] - Z[starts + h]) / sd) ** 2).mean()))
        for h in horizons
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--n-train", type=int, default=16_000)
    ap.add_argument("--h-max", type=int, default=64)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument(
        "--direct-pair-multipliers", type=int, nargs="+", default=[1],
        help="training-pair multipliers for the direct model. Matching the pair count to "
             "the rollout model (multiplier 1) matches the data budget, but the direct "
             "model must represent a whole family of maps from it, so each horizon gets "
             "roughly 1/h_max of the density the one-step model enjoys. A multiplier > 1 "
             "asks the different question - can direct prediction work at all - rather "
             "than whether it is more sample-efficient.",
    )
    ap.add_argument("--n-points", type=int, default=45_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    args.systems = [s for i, s in enumerate(args.systems) if i % args.n_shards == args.shard]

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    with RunLog("r8_direct_vs_rollout", vars(args), "cli", LOGS) as log:
        for name in args.systems:
            try:
                spec = S.load_spec(name)
                X = S.trajectory(name, n=args.n_points, use_cache=True)
            except Exception as e:  # noqa: BLE001
                log.failure(system=name, error=str(e)[:200])
                print(f"{name}: SETUP FAILED {e}", flush=True)
                continue

            split = args.n_train + 1
            mu, sd = X[:split].mean(0), X[:split].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            Ztr = ((X[:split] - mu) / sd).astype(np.float32)
            Zte = ((X[split : split + 8_000] - mu) / sd).astype(np.float32)

            rng = np.random.default_rng(0)
            starts = rng.choice(len(Zte) - args.h_max - 1, size=256, replace=False)

            par = parrot_errors(Ztr, Zte, starts, args.horizons)
            print(f"\n=== {name}  lam={spec.lyap_max:.3f} ===", flush=True)

            for seed in args.seeds:
                t0 = time.time()
                # (a) one-step model, rolled out
                Xs, Ys = Ztr[:-1], np.diff(Ztr, axis=0)
                m_roll, i_roll = fit(
                    Xs, Ys, budget=args.budget, epochs=args.epochs, seed=seed
                )
                e_roll = rollout_errors(m_roll, Zte, starts, args.horizons)

                # (b) horizon-conditioned, at one or more training-pair budgets
                for mult in args.direct_pair_multipliers:
                    hd = horizon_dataset(
                        Ztr, n_pairs=len(Xs) * mult, h_max=args.h_max, seed=seed
                    )
                    m_dir, i_dir = fit(
                        hd["X"], hd["Y"], budget=args.budget, epochs=args.epochs, seed=seed,
                        horizon=True, H=hd["H"], h_max=args.h_max,
                    )
                    e_dir = direct_errors(m_dir, Zte, starts, args.horizons)

                    for j, h in enumerate(args.horizons):
                        row = {
                            "system": name, "seed": seed, "horizon": h,
                            "direct_pair_multiplier": mult,
                            "n_pairs_direct": len(hd["X"]),
                            "n_pairs_rollout": len(Xs),
                            "lyap_time": spec.lyap_time_per_step * h,
                            "err_rollout": e_roll[j], "err_direct": e_dir[j],
                            "err_parrot": par[j],
                            "params_rollout": i_roll["n_params"],
                            "params_direct": i_dir["n_params"],
                            "lyap_max": spec.lyap_max,
                            "kaplan_yorke_dim": spec.invariants["kaplan_yorke_dim"],
                        }
                        rows.append(row)
                        log.result(**row)
                    print(
                        f"  seed {seed} x{mult}: "
                        + " ".join(
                            f"h{h}:{e_roll[j]:.3f}/{e_dir[j]:.3f}"
                            for j, h in enumerate(args.horizons)
                        )
                        + f"  (rollout/direct, {time.time()-t0:.0f}s)",
                        flush=True,
                    )

    keys = sorted({k for r in rows for k in r})
    with (RESULTS / (f"direct_vs_rollout" + (f"__shard{args.shard}" if args.n_shards > 1 else "") + ".csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {RESULTS/'direct_vs_rollout.csv'}")


if __name__ == "__main__":
    main()
