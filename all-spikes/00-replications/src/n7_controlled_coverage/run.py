"""N7 - the controlled coverage experiment. Does the surrogate's advantage track coverage?

N6 found that a trained surrogate's advantage over analogue lookup rises with how sparsely
the training data covers the attractor: rho = +0.558 across 107 systems, surviving a shuffle
control, leave-one-out, and partialling out state dimension.

That was **observational**. Coverage varied because the systems differed, and systems differ
in a hundred other ways at the same time. This experiment holds the system fixed and moves
coverage directly, by changing the training budget. If the advantage tracks coverage along
that axis too, the mechanism is established rather than inferred. If it does not, N6 was a
cross-system confound and the coverage law is dead.

Four things here exist because an external review named them, and each is a real risk:

  1. TEMPORAL EXCLUSION. Train and test were previously adjacent contiguous blocks. Chaotic
     trajectories are heavily autocorrelated, so the last training state sits a few steps
     from the first test state and lookup can retrieve a near-copy for reasons that have
     nothing to do with attractor coverage. A gap of `--gap` steps is now cut out between
     them, and a second gap separates the validation slice.

  2. A TUNED ANALOGUE BASELINE. The previous baseline took the single nearest neighbour at
     lookback 1. That is the weakest analogue forecaster available, and a claim of the form
     "a trained network only ties with lookup" means nothing if the lookup was a strawman.
     The baseline now tunes neighbour count, delay embedding and distance weighting on a
     VALIDATION split. The untuned setting is inside the grid, so tuning cannot lose.

  3. MATCHED CONTEXT. The lookup gets exactly the training block the network was fitted on -
     no more history, no less. Otherwise the comparison is about who saw more data.

  4. A FROZEN COVERAGE DEFINITION. Median distance from a test state to its nearest training
     state, over the mean pairwise distance within the training block. Same definition as
     N6, fixed before this run, so the thresholds it produces can be tested rather than
     fitted.

Usage:
    python -m src.n7_controlled_coverage.run --demo
    python -m src.n7_controlled_coverage.run --shard 0 --n-shards 2
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
from src.common.baselines import knn_rollout, parrot_rollout, tune_knn  # noqa: E402
from src.common.metrics import valid_prediction_time  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402
from src.common.train import error_by_step, fit, rollout, rollout_truth  # noqa: E402

RESULTS = ROOT / "results" / "n7"
LOGS = ROOT / "logs"

# Frozen before the run. Coverage is the median nearest-neighbour distance from a test state
# into the training block, divided by the mean pairwise distance inside that block.
COVERAGE = "median_nn_over_mean_pairwise"


def blocked_split(traj: np.ndarray, n_train: int, gap: int, n_val: int, n_test: int):
    """Train | gap | validation | gap | test, in time order, with nothing overlapping.

    The gaps are the point. Without them the first test state is a few steps after the last
    training state, and on an autocorrelated trajectory that is nearly the same point.
    """
    need = n_train + gap + n_val + gap + n_test + 2
    if len(traj) < need:
        raise ValueError(f"need {need} points, have {len(traj)}")
    a = 0
    b = a + n_train
    c = b + gap
    d = c + n_val
    e = d + gap
    return traj[a:b], traj[c:d], traj[e:e + n_test]


def coverage_of(train: np.ndarray, test: np.ndarray, n_query: int, rng) -> dict:
    """The frozen coverage statistic, plus the raw pieces so it can be re-derived."""
    q = test[rng.choice(len(test), size=min(n_query, len(test)), replace=False)]
    nn = np.sqrt(((q[:, None, :] - train[None, :, :]) ** 2).sum(-1).min(1))
    a = train[rng.choice(len(train), size=min(400, len(train)))]
    b = train[rng.choice(len(train), size=min(400, len(train)))]
    spread = float(np.sqrt(((a - b) ** 2).sum(-1)).mean())
    return {"nn_dist": float(np.median(nn)), "train_spread": spread,
            "coverage": float(np.median(nn) / max(spread, 1e-12))}


def one_cell(name: str, n_train: int, seed: int, args) -> dict:
    spec = S.load_spec(name)
    # One trajectory per system, shared by every budget and seed: the ONLY thing that
    # changes across a sweep is how much of it the model is allowed to train on.
    traj = S.trajectory(name, args.n_points)
    tr_raw, va_raw, te_raw = blocked_split(
        traj, n_train, args.gap, args.n_val, args.n_test)

    mu, sd = tr_raw.mean(0), tr_raw.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    tr, va, te = (tr_raw - mu) / sd, (va_raw - mu) / sd, (te_raw - mu) / sd

    rng = np.random.default_rng(1000 + seed)
    cov = coverage_of(tr, te, args.n_query, rng)

    steps = args.steps
    starts = rng.choice(len(te) - steps - 2, size=min(args.n_starts, len(te) - steps - 2),
                        replace=False)
    true = rollout_truth(te, starts, steps)

    # ---- the trained surrogate --------------------------------------------------------
    X, Y = tr[:-1], tr[1:] - tr[:-1]
    t0 = time.time()
    model, hist = fit(X.astype(np.float32), Y.astype(np.float32), budget=args.budget,
                      epochs=args.epochs, batch=args.batch, lr=args.lr, seed=seed)
    train_s = time.time() - t0
    pred_model = rollout(model, te[starts], steps)
    err_model = error_by_step(true, pred_model)

    # ---- the analogue baselines, tuned on validation ONLY ------------------------------
    val_starts = np.arange(12, min(len(va) - steps - 2, args.n_starts * 6), 8)
    cfg = tune_knn(tr, val_starts, va, steps, ks=args.ks, lookbacks=args.lookbacks)
    lb = cfg["lookback"]
    q = np.stack([te[s - lb + 1 : s + 1] for s in starts]) if lb > 1 else te[starts][:, None, :]
    pred_knn = knn_rollout(tr, q, steps, k=cfg["k"], weighted=cfg["weighted"])
    err_knn = error_by_step(true, pred_knn)

    pred_plain = parrot_rollout(tr, te[starts][:, None, :], steps)
    err_plain = error_by_step(true, pred_plain)

    lt = spec.lyap_time_per_step
    out = {
        "system": name, "dim": int(tr.shape[1]), "n_train": n_train, "seed": seed,
        "gap": args.gap, "coverage_def": COVERAGE, **cov,
        "lyap_max": spec.lyap_max, "n_params": hist["n_params"],
        "train_loss": hist["final_train_loss"], "train_s": round(train_s, 1),
        "knn_k": cfg["k"], "knn_lookback": cfg["lookback"], "knn_weighted": cfg["weighted"],
        "knn_val_rmse": round(cfg["val_rmse"], 6),
    }
    for tag, err in (("model", err_model), ("knn", err_knn), ("plain", err_plain)):
        out[f"vpt_{tag}"] = float(
            valid_prediction_time(err, 0.4, lt) / lt) if lt > 0 else float("nan")
        out[f"err10_{tag}"] = float(err[min(9, len(err) - 1)])
        out[f"err_final_{tag}"] = float(err[-1])
    out["ratio_vs_tuned"] = out["vpt_model"] / max(out["vpt_knn"], 1e-9)
    out["ratio_vs_plain"] = out["vpt_model"] / max(out["vpt_plain"], 1e-9)
    out["tuning_gain"] = out["vpt_knn"] / max(out["vpt_plain"], 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=[
        "Lorenz", "Rossler", "Chua", "Halvorsen", "Thomas", "Aizawa",
        "HyperCai", "DequanLi"])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[250, 500, 1000, 2000, 4000, 8000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gap", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--n-points", type=int, default=20000)
    ap.add_argument("--n-query", type=int, default=400)
    ap.add_argument("--n-starts", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)     # N5 said 128, not 256
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    jobs = [(s, n, sd) for s in args.systems for n in args.budgets for sd in args.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"coverage_sweep__shard{args.shard}.csv"
    rows: list[dict] = []

    with RunLog("n7_controlled_coverage", vars(args), "cli", LOGS) as log:
        for i, (name, n_train, seed) in enumerate(jobs, 1):
            try:
                r = one_cell(name, n_train, seed, args)
                rows.append(r)
                log.result(**r)
                print(f"[{i}/{len(jobs)}] {name:12s} n={n_train:>5d} seed{seed}  "
                      f"cov {r['coverage']:.4f}  model/tuned {r['ratio_vs_tuned']:.2f}  "
                      f"tuning gain {r['tuning_gain']:.2f}  {r['train_s']}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                log.failure(system=name, n_train=n_train, seed=seed, error=repr(exc))
                print(f"[{i}/{len(jobs)}] FAILED {name} n={n_train} seed{seed}: {exc}",
                      flush=True)
            if rows:
                keys = sorted({k for r in rows for k in r})
                with out_path.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")


def demo() -> None:
    """Self-check: the manipulation must move coverage, and the splits must not touch."""
    traj = S.trajectory("Lorenz", 20000)

    class A:
        gap, n_val, n_test, n_query = 2000, 2000, 3000, 300
        n_points, n_starts, steps = 20000, 20, 100
        budget, epochs, batch, lr = 4000, 30, 128, 2e-3
        ks, lookbacks = (1, 4), (1, 2)

    # the gap must genuinely separate the blocks in time
    tr, va, te = blocked_split(traj, 1000, A.gap, A.n_val, A.n_test)
    assert len(tr) == 1000 and len(va) == A.n_val and len(te) == A.n_test
    # the last training row must not reappear anywhere in test
    assert not any(np.allclose(tr[-1], row) for row in te[:50]), "gap failed to separate"

    # more data must mean denser coverage - that is the whole manipulation
    rng = np.random.default_rng(0)
    covs = []
    for n in (250, 1000, 4000):
        t, _, s = blocked_split(traj, n, A.gap, A.n_val, A.n_test)
        mu, sd = t.mean(0), t.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        covs.append(coverage_of((t - mu) / sd, (s - mu) / sd, A.n_query, rng)["coverage"])
    assert covs[0] > covs[1] > covs[2], f"coverage must fall as data grows, got {covs}"

    r = one_cell("Lorenz", 1000, 0, A)
    for key in ("coverage", "vpt_model", "vpt_knn", "vpt_plain", "ratio_vs_tuned"):
        assert np.isfinite(r[key]), f"{key} is not finite: {r[key]}"
    assert r["tuning_gain"] >= 0.95, (
        f"the tuned baseline lost to the untuned one ({r['tuning_gain']:.2f}); the untuned "
        f"setting is inside the tuning grid, so this means validation is unrepresentative")

    print(f"   coverage at n=250/1000/4000: {covs[0]:.4f} / {covs[1]:.4f} / {covs[2]:.4f}")
    print(f"   Lorenz n=1000: model VPT {r['vpt_model']:.0f}, tuned lookup "
          f"{r['vpt_knn']:.0f} (cfg k={r['knn_k']} lb={r['knn_lookback']}), "
          f"untuned {r['vpt_plain']:.0f}")
    print("self-check passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
