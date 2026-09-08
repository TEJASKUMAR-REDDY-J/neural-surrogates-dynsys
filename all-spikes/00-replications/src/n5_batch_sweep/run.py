"""N5 - is batch size 256 actually the right choice?

It was picked by convention and only ever tested upward. A single-seed sweep afterwards
suggested 128 might be better, but the gap sat inside the seed-to-seed variation measured
elsewhere (about 13%), so it could not be called. This runs the sweep with seeds so the
question can actually be answered.

Learning rate is scaled with batch size in the usual way, so this compares recipes rather
than penalising the large batches for an unadjusted step size.
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

RESULTS = ROOT / "results" / "n5"
LOGS = ROOT / "logs"

RECIPES = [(64, 1e-3), (128, 2e-3), (256, 3e-3), (512, 4e-3), (1024, 1e-2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["Lorenz", "Rossler", "Thomas"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--n-train", type=int, default=8_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--pts-per-period", type=int, default=20)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    with RunLog("n5_batch_sweep", vars(args), "cli", LOGS) as log:
        for name in args.systems:
            spec = S.load_spec(name, pts_per_period=args.pts_per_period)
            X = S.trajectory(name, n=45_000, pts_per_period=args.pts_per_period,
                             use_cache=True, timeout_s=300)
            ds = S.one_step_dataset(X, n_train=args.n_train, n_test=6_000)
            print(f"\n=== {name} ===", flush=True)
            for batch, lr in RECIPES:
                accum = []
                for seed in args.seeds:
                    t0 = time.time()
                    m, i = fit(ds["X_train"], ds["Y_train"], budget=args.budget,
                               epochs=args.epochs, batch=batch, lr=lr, seed=seed)
                    ev = evaluate_rollout(m, ds["test_traj_norm"], spec.lyap_time_per_step,
                                          steps=500, n_starts=64, seed=0)
                    row = {"system": name, "batch": batch, "lr": lr, "seed": seed,
                           "train_loss": i["final_train_loss"],
                           "err_h1": ev["err_h1"], "err_h50": ev["err_h50"],
                           "err_h200": ev["err_h200"], "vpt_steps": ev["vpt_steps"],
                           "wall_s": round(time.time() - t0, 1)}
                    rows.append(row); log.result(**row); accum.append(row)
                print(f"  batch {batch:>5} lr {lr:.0e}: "
                      f"loss {np.mean([r['train_loss'] for r in accum]):.2e}  "
                      f"h50 {np.mean([r['err_h50'] for r in accum]):.4f} "
                      f"+/- {np.std([r['err_h50'] for r in accum]):.4f}  "
                      f"({np.mean([r['wall_s'] for r in accum]):.0f}s)", flush=True)

    keys = sorted({k for r in rows for k in r})
    with (RESULTS / "batch_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows")


if __name__ == "__main__":
    main()
