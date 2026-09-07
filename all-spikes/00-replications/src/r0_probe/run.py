"""R0 - environment probe: which systems are usable here, and what do things actually cost.

Two jobs:
  1. Scan the dysts catalogue. For each system, try to integrate an aligned trajectory and
     record the wall time, whether it stayed bounded, and its published invariants. This
     produces the system shortlist every later experiment draws from.
  2. Time a fit at each corner of the R4 capacity x data grid, so the cost estimates in
     PLAN.md get replaced by measurements before the long runs commit.
"""

from __future__ import annotations

import argparse
import json
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
from src.common.train import fit  # noqa: E402

RESULTS = ROOT / "results" / "r0"
LOGS = ROOT / "logs"


def scan_systems(n_points: int, time_budget_s: float, log: RunLog) -> list[dict]:
    names = S.list_dysts_systems()
    log.note(f"dysts catalogue has {len(names)} systems")
    rows = []
    for i, name in enumerate(names):
        rec = {"system": name}
        try:
            t0 = time.time()
            spec = S.load_spec(name)
            rec.update(spec.invariants)
            rec["lyap_time_per_step"] = spec.lyap_time_per_step
            if not np.isfinite(spec.period) or spec.period <= 0:
                raise ValueError(f"bad period {spec.period}")
            X = S.trajectory(name, n=n_points, use_cache=True, timeout_s=time_budget_s)
            rec["gen_s"] = round(time.time() - t0, 2)
            rec["range"] = float(np.ptp(X, axis=0).max())
            rec["ok"] = bool(
                np.all(np.isfinite(X))
                and rec["range"] < 1e6
                and X.std(0).min() > 1e-8
            )
            rec["reason"] = "" if rec["ok"] else "slow_or_degenerate"
        except Exception as e:  # noqa: BLE001 - a failed system is data, not a crash
            rec["ok"] = False
            rec["reason"] = f"{type(e).__name__}: {e}"[:160]
            rec["gen_s"] = round(time.time() - t0, 2) if "t0" in dir() else None
        rows.append(rec)
        log.result(**rec)
        flag = "ok " if rec.get("ok") else "SKIP"
        print(
            f"[{i+1:3d}/{len(names)}] {flag} {name:<28} "
            f"{rec.get('gen_s', '?')}s  lam={rec.get('lyap_max', float('nan')):.3f} "
            f"{rec.get('reason','')}",
            flush=True,
        )
    return rows


def timing_probe(log: RunLog) -> list[dict]:
    """Measure a fit at each corner of the R4 grid. Replaces the PLAN.md estimates."""
    X = S.trajectory("Lorenz", n=40_000, use_cache=True, timeout_s=300)
    rows = []
    for budget in (1_000, 10_000, 100_000):
        for n_train in (500, 8_000, 32_000):
            ds = S.one_step_dataset(X, n_train=n_train, n_test=2000)
            t0 = time.time()
            _, info = fit(ds["X_train"], ds["Y_train"], budget=budget, epochs=200, seed=0)
            dt = time.time() - t0
            row = {
                "budget": budget,
                "n_params": info["n_params"],
                "n_train": n_train,
                "epochs": 200,
                "wall_s": round(dt, 2),
                "final_train_loss": info["final_train_loss"],
            }
            rows.append(row)
            log.result(**row)
            print(f"  fit {info['n_params']:>7} params x {n_train:>6} samples: {dt:6.1f} s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=20_000)
    ap.add_argument("--time-budget", type=float, default=25.0, help="max seconds per trajectory")
    ap.add_argument("--skip-scan", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)

    with RunLog("r0_probe", cfg, "cli", LOGS) as log:
        if not args.skip_scan:
            print("=== scanning dysts catalogue ===", flush=True)
            rows = scan_systems(args.n_points, args.time_budget, log)
            import csv

            keys = sorted({k for r in rows for k in r})
            with (RESULTS / "system_scan.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            n_ok = sum(bool(r.get("ok")) for r in rows)
            print(f"\n{n_ok}/{len(rows)} systems usable", flush=True)
            log.note("scan complete", n_ok=n_ok, n_total=len(rows))

        print("\n=== timing probe ===", flush=True)
        probe = timing_probe(log)
        (RESULTS / "timing_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")

    print("\nR0 done")


if __name__ == "__main__":
    main()
