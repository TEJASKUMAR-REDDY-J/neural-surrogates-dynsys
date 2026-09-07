"""R3 - does a cheap model-free statistic already predict how well our surrogate will do?

Weighted permutation entropy takes a few lines of code and no training. Garland, James &
Bradley (PRE 2014) showed it tracks achievable forecast error for classical forecasters;
Pennekamp et al. (2019) replicated across 461 ecological series. Neither tested it on a
neural surrogate of a dynamical system.

This is the honest bar the project has to clear. If WPE plus the Lyapunov exponent already
explain most of the variation in our surrogate's skill across systems, a learned
"learnability predictor" has almost nothing left to add, and we should say so.

Pre-registered kill criterion: if WPE + lambda explain >= 85% of the variance in surrogate
skill across systems, the learned-predictor direction is not worth pursuing.

Every run also records the training-free baselines (context parroting, persistence) on the
identical starts and horizons, because a surrogate that cannot beat copying is not
evidence of rule-learning.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import systems as S  # noqa: E402
from src.common.baselines import parrot_rollout, persistence_rollout  # noqa: E402
from src.common.metrics import valid_prediction_time  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402
from src.common.train import (  # noqa: E402
    error_by_step,
    evaluate_rollout,
    fit,
    rollout_truth,
)

RESULTS = ROOT / "results" / "r3"
LOGS = ROOT / "logs"


def r2_systems(limit: int, bounded_only: bool = False) -> list[str]:
    path = ROOT / "results" / "r2" / "instruments.csv"
    if not path.exists():
        raise FileNotFoundError("run R2 first")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if bounded_only:
        import src.common.systems as _S
        keep = []
        for r in rows:
            try:
                if not _S.load_spec(r["system"]).unbounded_indices:
                    keep.append(r)
            except Exception:  # noqa: BLE001
                pass
        rows = keep
    # spread across the chaoticity range rather than taking the first N
    rows.sort(key=lambda r: float(r["lyap_max"]))
    if limit >= len(rows):
        return [r["system"] for r in rows]
    idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[i]["system"] for i in idx]


def baseline_scores(ds: dict, spec, steps: int, n_starts: int, seed: int) -> dict:
    """Parroting and persistence on the same starts the model is evaluated on."""
    test = ds["test_traj_norm"]
    rng = np.random.default_rng(seed)
    usable = len(test) - steps - 2
    starts = rng.choice(usable, size=min(n_starts, usable), replace=False)
    x0 = test[starts]
    true = rollout_truth(test, starts, steps)

    context = ds["X_train"]  # normalised training trajectory: what the model saw
    out = {}
    for label, pred in (
        ("parrot", parrot_rollout(context, x0, steps)),
        ("persistence", persistence_rollout(x0, steps)),
    ):
        err = error_by_step(true, pred)
        out[f"{label}_vpt_lyap"] = valid_prediction_time(err, 0.4, spec.lyap_time_per_step)
        out[f"{label}_vpt_steps"] = float(
            out[f"{label}_vpt_lyap"] / spec.lyap_time_per_step
        )
        out[f"{label}_err_10"] = float(err[min(9, len(err) - 1)])
        out[f"{label}_err_final"] = float(err[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-systems", type=int, default=24)
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--n-train", type=int, default=8_000)
    ap.add_argument("--n-test", type=int, default=4_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-points", type=int, default=20_000)
    ap.add_argument(
        "--bounded-only", action="store_true",
        help="drop systems carrying an unbounded clock coordinate; see the note in "
             "systems.load_spec for why they corrupt cross-system regressions",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    names = r2_systems(args.n_systems, bounded_only=args.bounded_only)
    names = [n for i, n in enumerate(names) if i % args.n_shards == args.shard]
    rows = []

    with RunLog("r3_wpe_vs_error", vars(args), "cli", LOGS) as log:
        log.note("systems", n=len(names), systems=names)
        for i, name in enumerate(names):
            try:
                spec = S.load_spec(name)
                X = S.trajectory(name, n=args.n_points, use_cache=True)
                ds = S.one_step_dataset(X, n_train=args.n_train, n_test=args.n_test)
                base = baseline_scores(ds, spec, args.steps, 64, seed=0)
            except Exception as e:  # noqa: BLE001
                log.failure(system=name, stage="setup", error=f"{type(e).__name__}: {e}"[:200])
                print(f"[{i+1}/{len(names)}] {name} SETUP FAILED: {e}", flush=True)
                continue

            for seed in args.seeds:
                t0 = time.time()
                try:
                    model, info = fit(
                        ds["X_train"], ds["Y_train"],
                        budget=args.budget, epochs=args.epochs, seed=seed,
                    )
                    ev = evaluate_rollout(
                        model, ds["test_traj_norm"], spec.lyap_time_per_step,
                        steps=args.steps, n_starts=64, seed=0,
                    )
                    row = {
                        "system": name, "seed": seed,
                        "n_params": info["n_params"], "n_train": args.n_train,
                        "final_train_loss": info["final_train_loss"],
                        "lyap_max": spec.lyap_max, "dim": spec.dim,
                        "lyap_time_per_step": spec.lyap_time_per_step,
                        **{k: v for k, v in ev.items() if k != "err_by_step"},
                        **base,
                        "wall_s": round(time.time() - t0, 2),
                    }
                    rows.append(row)
                    log.result(err_by_step=ev["err_by_step"], **row)
                except Exception as e:  # noqa: BLE001
                    log.failure(system=name, seed=seed, error=f"{type(e).__name__}: {e}"[:200])
                    print(f"    {name} seed {seed} FAILED: {e}", flush=True)

            mine = [r for r in rows if r["system"] == name]
            if mine:
                v = np.mean([r["vpt_steps"] for r in mine])
                print(
                    f"[{i+1:2d}/{len(names)}] {name:<26} "
                    f"VPT model {v:6.1f} steps | parrot {base['parrot_vpt_steps']:6.1f} | "
                    f"persist {base['persistence_vpt_steps']:5.1f} | "
                    f"lam={spec.lyap_max:.3f}",
                    flush=True,
                )

    keys = sorted({k for r in rows for k in r})
    with (RESULTS / (f"surrogate_vs_statistics" + (f"__shard{args.shard}" if args.n_shards > 1 else "") + ".csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {RESULTS/'surrogate_vs_statistics.csv'}")


if __name__ == "__main__":
    main()
