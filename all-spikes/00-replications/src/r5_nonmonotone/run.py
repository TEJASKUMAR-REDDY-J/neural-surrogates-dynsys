"""R5 - is the "more chaos needs less capacity" result real, or a threshold artefact?

Chen et al. (arXiv:2512.01558, NNPT) found that the network size needed to reach 1%
accuracy rises with chaoticity, peaks, then falls 47% in the fully chaotic regime, and
noted the transition matched Chirikov's resonance-overlap criterion. Their explanation is
"ergodic smoothing": once fully chaotic, trajectory-specific detail becomes effectively
noise, and noise costs no capacity to represent.

Their evidence is thinner than the claim. The 47% drop is a single architectural step
(3x32 -> 2x32), no seed counts are reported, and it is read at one tolerance (1%).
"Smallest model that crosses a threshold" is a high-variance quantity: one unlucky
initialisation moves it a whole step.

So we test it properly, on the Chirikov standard map itself - which tests the claimed
*mechanism* directly instead of by analogy through a three-body mass parameter:

  1. a near-continuous capacity axis (9 sizes over ~2 orders of magnitude) instead of a
     two-point depth step
  2. seeds at every grid point, reporting the distribution of the crossing capacity
  3. four tolerances, read off the same already-computed curves. If the non-monotonicity
     appears only at 1%, it is an artefact. This is the cheapest high-value test here.
  4. a second map family (Henon) as a cross-check, because n=1 on system families is what
     made the original claim fragile

Torus handling: the standard map lives on [0, 2pi)^2. Feeding raw angles to a network
makes the 2pi wrap look like a huge jump, so states enter as (cos, sin) pairs and targets
are the wrapped increments in (-pi, pi].
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
from src.common.runlog import RunLog  # noqa: E402
from src.common.train import fit  # noqa: E402

RESULTS = ROOT / "results" / "r5"
LOGS = ROOT / "logs"

# spans two decades so that even the hardest knob settings produce a crossing;
# NNPT read its result at a single 1% tolerance, which is the fragility we are testing
TOLERANCES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def wrap(a: np.ndarray) -> np.ndarray:
    """Map angles into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def standard_map_dataset(K: float, n_train: int, n_test: int, seed: int) -> dict:
    traj = S.map_trajectory("StandardMap", n_train + n_test + 1, param=K, seed=seed)
    th, p = traj[:, 0], traj[:, 1]
    feats = np.c_[np.cos(th), np.sin(th), np.cos(p), np.sin(p)].astype(np.float32)
    tgt = np.c_[wrap(np.diff(th)), wrap(np.diff(p))].astype(np.float32)
    X, Y = feats[:-1], tgt
    return {
        "X_train": X[:n_train], "Y_train": Y[:n_train],
        "X_test": X[n_train : n_train + n_test], "Y_test": Y[n_train : n_train + n_test],
    }


def henon_dataset(a: float, n_train: int, n_test: int, seed: int) -> dict:
    traj = S.map_trajectory("HenonMap", n_train + n_test + 1, param=a, seed=seed)
    if not np.all(np.isfinite(traj)) or np.abs(traj).max() > 1e3:
        raise RuntimeError(f"Henon a={a} escaped")
    mu, sd = traj[:n_train].mean(0), traj[:n_train].std(0)
    Z = ((traj - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    X, Y = Z[:-1], np.diff(Z, axis=0).astype(np.float32)
    return {
        "X_train": X[:n_train], "Y_train": Y[:n_train],
        "X_test": X[n_train : n_train + n_test], "Y_test": Y[n_train : n_train + n_test],
    }


@torch.no_grad()
def relative_error(model, X: np.ndarray, Y: np.ndarray) -> float:
    """One-step prediction error relative to the spread of the target.

    This is the analogue of NNPT's "equalised accuracy at 1% tolerance": how small can the
    model be and still predict the step to within a given fraction of the step's own size.
    """
    model.eval()
    pred = model(torch.from_numpy(np.asarray(X, np.float32))).numpy()
    denom = np.sqrt((np.asarray(Y, float) ** 2).mean())
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(((pred - Y) ** 2).mean()) / denom)


def crossing_capacity(params: list[int], errs: list[float], tol: float) -> float:
    """Smallest capacity whose error is <= tol. nan if the tolerance is never reached."""
    for p, e in zip(params, errs):
        if np.isfinite(e) and e <= tol:
            return float(p)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--budgets", type=int, nargs="+",
        default=[200, 400, 800, 1_600, 3_200, 6_400, 12_800, 25_600, 51_200],
    )
    ap.add_argument(
        "--k-values", type=float, nargs="+",
        default=[0.4, 0.7, 0.9716, 1.3, 2.0, 3.0, 5.0, 8.0],
    )
    ap.add_argument(
        "--henon-values", type=float, nargs="+",
        default=[1.0, 1.06, 1.12, 1.18, 1.24, 1.30, 1.36, 1.40],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=8_000)
    ap.add_argument("--n-test", type=int, default=4_000)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--skip-henon", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []

    shard = lambda v: [x for i, x in enumerate(v) if i % args.n_shards == args.shard]
    families = [("StandardMap", shard(args.k_values), standard_map_dataset)]
    if not args.skip_henon:
        families.append(("HenonMap", shard(args.henon_values), henon_dataset))

    total = sum(len(v) for _, v, _ in families) * len(args.budgets) * len(args.seeds)
    done, t_start = 0, time.time()

    with RunLog("r5_nonmonotone", vars(args), "cli", LOGS) as log:
        for family, knobs, make_ds in families:
            print(f"\n=== {family} ===", flush=True)
            for knob in knobs:
                lam = (
                    S.standard_map_lyapunov(knob, n=60_000)
                    if family == "StandardMap"
                    else float("nan")
                )
                try:
                    ds = make_ds(knob, args.n_train, args.n_test, seed=0)
                except Exception as e:  # noqa: BLE001
                    log.failure(family=family, knob=knob, error=str(e)[:200])
                    print(f"  knob={knob}: SKIP ({e})", flush=True)
                    done += len(args.budgets) * len(args.seeds)
                    continue

                per_seed_err: dict[int, list[float]] = {s: [] for s in args.seeds}
                params_seen: list[int] = []
                for budget in args.budgets:
                    errs = []
                    for seed in args.seeds:
                        t0 = time.time()
                        model, info = fit(
                            ds["X_train"], ds["Y_train"],
                            budget=budget, epochs=args.epochs, seed=seed,
                        )
                        err = relative_error(model, ds["X_test"], ds["Y_test"])
                        per_seed_err[seed].append(err)
                        errs.append(err)
                        row = {
                            "family": family, "knob": knob, "lyap": lam,
                            "budget": budget, "n_params": info["n_params"],
                            "seed": seed, "rel_error": err,
                            "final_train_loss": info["final_train_loss"],
                            "wall_s": round(time.time() - t0, 2),
                        }
                        rows.append(row)
                        log.result(**row)
                        done += 1
                    if not params_seen or params_seen[-1] != info["n_params"]:
                        params_seen.append(info["n_params"])
                    print(
                        f"  knob={knob:<7} params={info['n_params']:>6}  "
                        f"rel_err {np.mean(errs):.4f} +/- {np.std(errs):.4f}  "
                        f"[{done}/{total}, eta {(time.time()-t_start)/max(done,1)*(total-done)/60:.0f}m]",
                        flush=True,
                    )

                # required capacity per seed per tolerance -> distribution, not one number
                for tol in TOLERANCES:
                    caps = [crossing_capacity(params_seen, per_seed_err[s], tol) for s in args.seeds]
                    finite = [c for c in caps if np.isfinite(c)]
                    srec = {
                        "family": family, "knob": knob, "lyap": lam, "tolerance": tol,
                        "n_seeds_reaching": len(finite),
                        "cap_median": float(np.median(finite)) if finite else float("nan"),
                        "cap_min": float(np.min(finite)) if finite else float("nan"),
                        "cap_max": float(np.max(finite)) if finite else float("nan"),
                        "cap_all": caps,
                    }
                    summary.append(srec)
                    log.result(record_kind="required_capacity", **srec)

    keys = sorted({k for r in rows for k in r})
    with (RESULTS / (f"capacity_curves" + (f"__shard{args.shard}" if args.n_shards > 1 else "") + ".csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    skeys = sorted({k for r in summary for k in r})
    with (RESULTS / (f"required_capacity" + (f"__shard{args.shard}" if args.n_shards > 1 else "") + ".csv")).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=skeys)
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {len(rows)} curve rows and {len(summary)} capacity rows to {RESULTS}")


if __name__ == "__main__":
    main()
