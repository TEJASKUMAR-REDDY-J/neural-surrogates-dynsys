"""N6 - why does a trained network only tie with copying?

The most repeated finding in this battery is that a trained surrogate is, on median,
exactly as good as copying the most similar past moment (46/108 systems, ratio 1.00).
Every reading of that so far has been about the *network*: maybe it is not learning
anything.

This asks whether it is about the *data* instead.

Copying works by finding the nearest past state and replaying what followed it. So its
skill should be governed by one thing: how close that nearest neighbour actually is. If a
benchmark's trajectories cover the attractor densely, every query has a near-identical
precedent and copying is unbeatable. If coverage is sparse, copying has nothing to copy.

So we measure coverage density directly - median distance from a test state to its nearest
neighbour in the training set, in normalised coordinates, divided by the attractor's own
spread - and ask whether it explains the model-versus-copying ratio across 107 systems.

No training. This reuses the stored trajectories and the already-logged N3 results, so it
costs a couple of minutes and cannot disturb any existing result.

Checks run here, because a correlation across systems is exactly the shape of thing that
has fooled us before (R3's 0.59, R4b's -0.71):
  - a shuffle control, to price in how much correlation 107 points buy for free
  - leave-one-out, to check no single system carries it
  - partial correlations against state dimension, the obvious confound
  - the ratio decomposed into its two halves, because a ratio can move for two reasons
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.runlog import RunLog  # noqa: E402
from src.common.systems import one_step_dataset  # noqa: E402

RESULTS = ROOT / "results" / "n6"
LOGS = ROOT / "logs"

# These must match what R3/N3 actually trained on, or the coverage number describes a
# different dataset from the one the models and the copying baseline saw.
N_TRAIN, N_TEST = 8_000, 4_000


def coverage_density(traj: np.ndarray, n_query: int, seed: int) -> dict:
    """How far is a typical test state from its nearest precedent in the training set?

    Reported relative to the attractor's own spread, so it is comparable across systems
    with wildly different scales. Small = densely covered = copying has an easy job.
    """
    ds = one_step_dataset(traj, N_TRAIN, N_TEST)
    ctx = ds["X_train"].astype(np.float64)            # exactly what parrot_rollout sees
    test = ds["test_traj_norm"].astype(np.float64)
    rng = np.random.default_rng(seed)

    usable = len(test) - 210                          # leave room for a 200-step rollout
    q = test[rng.choice(usable, size=min(n_query, usable), replace=False)]
    nn = np.sqrt(((q[:, None, :] - ctx[None, :, :]) ** 2).sum(-1).min(1))

    a = ctx[rng.choice(len(ctx), 400)]
    b = ctx[rng.choice(len(ctx), 400)]
    spread = float(np.sqrt(((a - b) ** 2).sum(-1)).mean())

    return {
        "dim": int(ctx.shape[1]),
        "nn_dist": float(np.median(nn)),
        "attractor_spread": spread,
        "coverage": float(np.median(nn) / spread),
    }


def load_n3() -> pd.DataFrame:
    """Median skill per system from the clean-data N3 run."""
    files = sorted(glob.glob(str(ROOT / "results/r3/surrogate_vs_statistics__shard*.csv")))
    if not files:
        raise FileNotFoundError("run N3 first")
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df.noise == 0.0]
    return (
        df.groupby("system")
        .agg(
            lyap_max=("lyap_max", "first"),
            model_vpt=("vpt_steps", "median"),
            parrot_vpt=("parrot_vpt_steps", "median"),
            censored=("vpt_censored", "mean"),
        )
        .reset_index()
    )


def partial_spearman(m: pd.DataFrame, x: str, y: str, given: str) -> float:
    """Rank partial correlation: does x still track y once `given` is regressed out?"""
    X, Y = rankdata(m[x]), rankdata(m[y])
    Z = np.c_[np.ones(len(m)), rankdata(m[given])]
    rx = X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    ry = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=500)
    ap.add_argument("--n-shuffle", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = RunLog("n6_recurrence", vars(args), "cli", LOGS)

    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/trajectories/*__20000__*.npy"))):
        name = Path(path).name.split("__")[0]
        try:
            cov = coverage_density(np.load(path), args.n_query, args.seed)
        except Exception as exc:  # noqa: BLE001
            log.failure(system=name, error=str(exc))
            continue
        rows.append({"system": name, **cov})
        log.result(**rows[-1])

    cov = pd.DataFrame(rows)
    m = cov.merge(load_n3(), on="system", how="inner")
    m = m[m.parrot_vpt > 0].copy()
    m["ratio"] = m.model_vpt / m.parrot_vpt
    m["log_ratio"] = np.log(m.ratio.clip(1e-3))

    RESULTS.mkdir(parents=True, exist_ok=True)
    m.to_csv(RESULTS / "coverage_vs_advantage.csv", index=False)

    rho, p = spearmanr(m.coverage, m.log_ratio)
    rng = np.random.default_rng(args.seed)
    null = np.array([
        abs(spearmanr(rng.permutation(m.coverage.values), m.log_ratio.values)[0])
        for _ in range(args.n_shuffle)
    ])
    loo = [spearmanr(m.coverage.drop(i), m.log_ratio.drop(i))[0] for i in m.index]

    summary = {
        "n_systems": int(len(m)),
        "coverage_median": float(m.coverage.median()),
        "coverage_range": [float(m.coverage.min()), float(m.coverage.max())],
        "rho_coverage_vs_log_ratio": float(rho),
        "p_value": float(p),
        "shuffle_null_p95": float(np.percentile(null, 95)),
        "shuffle_empirical_p": float((null >= abs(rho)).mean()),
        "loo_rho_min": float(min(loo)),
        "loo_rho_max": float(max(loo)),
        "rho_dim_vs_log_ratio": float(spearmanr(m.dim, m.log_ratio)[0]),
        "rho_lyap_vs_log_ratio": float(spearmanr(m.lyap_max, m.log_ratio)[0]),
        "partial_coverage_given_dim": partial_spearman(m, "coverage", "log_ratio", "dim"),
        "partial_dim_given_coverage": partial_spearman(m, "dim", "log_ratio", "coverage"),
        # the ratio has two halves; which one does coverage actually move?
        "rho_coverage_vs_parrot_vpt": float(spearmanr(m.coverage, m.parrot_vpt)[0]),
        "rho_coverage_vs_model_vpt": float(spearmanr(m.coverage, m.model_vpt)[0]),
        "censored_fraction": float(m.censored.mean()),
    }
    m["band"] = pd.qcut(m.coverage, 4, labels=["densest", "dense", "sparse", "sparsest"])
    summary["quartiles"] = (
        m.groupby("band", observed=True)
        .agg(n=("ratio", "size"), coverage_median=("coverage", "median"),
             copying_vpt=("parrot_vpt", "median"), model_vpt=("model_vpt", "median"),
             median_ratio=("ratio", "median"), model_wins=("ratio", lambda s: int((s > 1).sum())))
        .round(4).reset_index().to_dict("records")
    )

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.note("summary", **{k: v for k, v in summary.items() if k != "quartiles"})
    log.close()
    print(json.dumps(summary, indent=2))


def demo() -> None:
    """Self-check: coverage must fall as the training set grows, on a fixed system."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 400, 30_000)
    traj = np.c_[np.sin(t), np.cos(1.7 * t), np.sin(0.3 * t)] + 0.01 * rng.standard_normal((30_000, 3))

    global N_TRAIN
    saved = N_TRAIN
    try:
        N_TRAIN = 1_000
        sparse = coverage_density(traj, 200, 0)["coverage"]
        N_TRAIN = 16_000
        dense = coverage_density(traj, 200, 0)["coverage"]
    finally:
        N_TRAIN = saved

    assert dense < sparse, f"more data must mean denser coverage, got {dense} vs {sparse}"

    m = pd.DataFrame({"a": [1, 2, 3, 4, 5.0], "b": [1, 2, 3, 4, 5.0], "c": [5, 4, 3, 2, 1.0]})
    assert abs(partial_spearman(m, "a", "b", "c") - 1.0) < 1e-9
    print(f"self-check passed (coverage {sparse:.4f} at 1k -> {dense:.4f} at 16k)")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
