"""R4 - does surrogate skill saturate, and is the saturation level system-dependent?

The crux of the project. Three published positions disagree:

  Gilpin (2023)      - forecast skill decorrelates from the Lyapunov exponent, and what
                       limits us is model scale and data availability.
  Duraisamy (2026)   - there are intrinsic ceilings set by spectral bias and information
                       lost under coarse-graining, which no architecture recovers.
  Chen et al. (2025) - neither: required capacity is non-monotone in chaoticity.

None of them ran a controlled per-system sweep over both capacity and data. Gilpin's
stand-in for capacity was training wall-clock time (rho = -0.31) and his data sweep was
aggregated across systems rather than per system.

We sweep both axes per system over ~2 orders of magnitude, then fit the scaling form of
Bahri et al. (PNAS 2024), L = L_inf + A N^-alpha, and ask two questions:

  1. Is L_inf reliably above zero, and does it differ between systems by more than seed
     noise?  (an intrinsic ceiling)
  2. Does alpha match the resolution-limited prediction alpha ~ 1/d, where d is the
     attractor's fractal dimension?  Gilpin checked invariants against *accuracy*; nobody
     checked them against the *exponent*, and they are different objects.

Caveat that must accompany every conclusion: saturation observed up to 1e5 parameters does
not prove saturation at 1e9. We report the trend and the scale at which we saw it.
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
from src.common.runlog import RunLog  # noqa: E402
from src.common.train import evaluate_rollout, fit  # noqa: E402

RESULTS = ROOT / "results" / "r4"
LOGS = ROOT / "logs"

# Chosen before any R4 result is seen, to span the R2 feature space: two weakly chaotic,
# two moderate, two strongly chaotic, with a spread of attractor dimension.
DEFAULT_SYSTEMS = [
    "Lorenz", "Rossler", "Halvorsen", "Chua", "Aizawa", "Thomas",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument(
        "--budgets", type=int, nargs="+",
        default=[1_000, 2_200, 4_600, 10_000, 22_000, 46_000, 100_000],
    )
    ap.add_argument("--n-trains", type=int, nargs="+", default=[500, 2_000, 8_000, 32_000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=4_000)
    ap.add_argument("--n-points", type=int, default=45_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    args.systems = [s for i, s in enumerate(args.systems) if i % args.n_shards == args.shard]

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(args.systems) * len(args.budgets) * len(args.n_trains) * len(args.seeds)
    done = 0
    t_start = time.time()

    with RunLog("r4_capacity_data", vars(args), "cli", LOGS) as log:
        for name in args.systems:
            try:
                spec = S.load_spec(name)
                X = S.trajectory(name, n=args.n_points, use_cache=True)
            except Exception as e:  # noqa: BLE001
                log.failure(system=name, stage="setup", error=f"{type(e).__name__}: {e}"[:200])
                print(f"{name}: SETUP FAILED {e}", flush=True)
                continue
            print(f"\n=== {name}  lam={spec.lyap_max:.3f}  Dky={spec.invariants['kaplan_yorke_dim']:.2f} ===", flush=True)

            for n_train in args.n_trains:
                ds = S.one_step_dataset(X, n_train=n_train, n_test=args.n_test)
                for budget in args.budgets:
                    for seed in args.seeds:
                        t0 = time.time()
                        try:
                            model, info = fit(
                                ds["X_train"], ds["Y_train"],
                                budget=budget, epochs=args.epochs, seed=seed,
                            )
                            ev = evaluate_rollout(
                                model, ds["test_traj_norm"], spec.lyap_time_per_step,
                                steps=args.steps, n_starts=64, seed=0,
                            )
                            row = {
                                "system": name, "budget": budget,
                                "n_params": info["n_params"], "n_train": n_train,
                                "seed": seed, "epochs": args.epochs,
                                "final_train_loss": info["final_train_loss"],
                                "lyap_max": spec.lyap_max,
                                "kaplan_yorke_dim": spec.invariants["kaplan_yorke_dim"],
                                "corr_dim_pub": spec.invariants["correlation_dim_pub"],
                                "dim": spec.dim,
                                "lyap_time_per_step": spec.lyap_time_per_step,
                                **{k: v for k, v in ev.items() if k != "err_by_step"},
                                "wall_s": round(time.time() - t0, 2),
                            }
                            rows.append(row)
                            log.result(err_by_step=ev["err_by_step"], **row)
                        except Exception as e:  # noqa: BLE001
                            log.failure(
                                system=name, budget=budget, n_train=n_train, seed=seed,
                                error=f"{type(e).__name__}: {e}"[:200],
                            )
                        done += 1
                    last = [r for r in rows if r["system"] == name and r["budget"] == budget
                            and r["n_train"] == n_train]
                    if last:
                        el = time.time() - t_start
                        eta = el / max(done, 1) * (total - done)
                        print(
                            f"  n_train={n_train:>6} params={last[0]['n_params']:>6}: "
                            f"loss {np.mean([r['final_train_loss'] for r in last]):.2e} "
                            f"VPT {np.mean([r['vpt_steps'] for r in last]):6.1f} steps "
                            f"specErr {np.mean([r['spectrum_error'] for r in last]):.3f} "
                            f"[{done}/{total}, eta {eta/60:.0f}m]",
                            flush=True,
                        )

    keys = sorted({k for r in rows for k in r})
    with (RESULTS / (f"capacity_data_sweep" + (f"__shard{args.shard}" if args.n_shards > 1 else "") + ".csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {RESULTS/'capacity_data_sweep.csv'}")


if __name__ == "__main__":
    main()
