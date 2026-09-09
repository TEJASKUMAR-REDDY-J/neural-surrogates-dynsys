"""The local x global grid: every local block against every global block, error attributed
separately to each path.

Twenty combinations, all sized to the same parameter budget, all on identical data, masks
and seeds. Each trained model is then evaluated three ways from the SAME weights:

    both        as trained
    local only  the global path silenced
    global only the local path silenced

which gives the two numbers that matter:

    global_lift = err(local only) / err(both)     > 1 means the global path earned its place
    local_lift  = err(global only) / err(both)    how much of the work is local

A combination where global_lift is 1.0 has a global path that could be deleted without
changing anything. That is a real and common outcome, and worth catching before anyone
concludes the mechanism helped.

    python -m src.run_hybrid --probe
    python -m src.run_hybrid --shard 0 --n-shards 2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import task as T  # noqa: E402
from src.common.data import build_all  # noqa: E402
from src.common.hybrid import GLOBALS, LOCALS, build_hybrid, scale_for_budget  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402

RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def split(ds, frac=0.75):
    k = int(len(ds.X) * frac)
    return ds.X[:k], ds.X[k:]


def score_paths(model, obs, mask, true, steps, binary, seed):
    """One model, three ways to run it. Same weights throughout."""
    out = {}
    for tag, (lg, gg) in (("both", (1.0, 1.0)), ("local", (1.0, 0.0)),
                          ("global", (0.0, 1.0))):
        if tag == "global" and not model.has_global():
            out[tag] = float("nan")
            continue
        torch.manual_seed(9000 + seed)          # identical firing pattern for all three
        model.local_gain, model.global_gain = lg, gg
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(obs), torch.tensor(mask), steps=steps).numpy()
        out[tag] = T.score(pred, true, mask, binary)["nrmse_hidden"]
    model.local_gain, model.global_gain = 1.0, 1.0
    return out


def one_cell(ds, local, glob, seed, args):
    tr, te = split(ds)
    sc = scale_for_budget(local, glob, ds.spatial, ds.channels, args.budget)
    model = build_hybrid(local, glob, ds.spatial, ds.channels, scale=sc)
    t0 = time.time()
    hist = T.train(model, tr, ds.spatial, epochs=args.epochs, batch=args.batch,
                   lr=args.lr, optimizer="adamw", seed=seed)
    train_s = time.time() - t0

    binary = ds.name.startswith("eca")
    rows = []
    for mode, frac in (("random", 0.3), ("block", 0.3), ("random", 0.6)):
        rng = np.random.default_rng(4000 + seed)
        mask = T.make_mask((len(te),), ds.spatial, frac, mode, rng)
        obs = T.corrupt(te, mask, 0.0, rng)
        best = None
        for steps in args.steps_grid:
            p = score_paths(model, obs, mask, te, steps, binary, seed)
            if best is None or p["both"] < best[1]["both"]:
                best = (steps, p)
        steps, p = best
        rows.append({
            "dataset": ds.name, "modality": ds.modality, "ordered": ds.ordered,
            "local": local, "global": glob, "combo": f"{local}+{glob}",
            "seed": seed, "scale": sc, "n_params": model.n_params,
            "cond_mode": mode, "cond_frac": frac, "best_steps": steps,
            "train_loss": hist["train_loss"], "train_s": round(train_s, 1),
            "err_both": p["both"], "err_local_only": p["local"],
            "err_global_only": p["global"],
            "global_lift": p["local"] / max(p["both"], 1e-9),
            "local_lift": (p["global"] / max(p["both"], 1e-9)
                           if np.isfinite(p["global"]) else float("nan")),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cml_lattice", "lorenz_state", "digits_8x8",
                             "cml_lattice__shuffled"])
    ap.add_argument("--locals", nargs="+", default=list(LOCALS))
    ap.add_argument("--globals", nargs="+", default=list(GLOBALS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--budget", type=int, default=13000)
    ap.add_argument("--n-per-set", type=int, default=900)
    ap.add_argument("--steps-grid", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    sets = [d for d in build_all(args.n_per_set) if d.name in args.datasets]
    jobs = [(d, lo, gl, s) for d in sets for lo in args.locals
            for gl in args.globals for s in args.seeds]

    if args.probe:
        tot = 0.0
        for d in sets:
            for lo in args.locals[:1]:
                for gl in args.globals:
                    t0 = time.time()
                    one_cell(d, lo, gl, 0, argparse.Namespace(**{**vars(args), "epochs": 3}))
                    per = (time.time() - t0) / 3 * args.epochs
                    tot += per * len(args.locals) * len(args.seeds)
                    print(f"{d.name:24s} {lo}+{gl:10s} ~{per:6.1f}s/fit", flush=True)
        print(f"\nestimated {tot/3600:.2f} h serial, ~{tot/3600/1.7:.2f} h on two shards")
        return

    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"hybrid__shard{args.shard}.csv"
    rows = []
    with RunLog("nca_hybrid_grid", vars(args), "cli", LOGS) as log:
        for i, (d, lo, gl, s) in enumerate(jobs, 1):
            try:
                rs = one_cell(d, lo, gl, s, args)
                rows += rs
                for r in rs:
                    log.result(**r)
                r0 = rs[0]
                print(f"[{i}/{len(jobs)}] {d.name:22s} {lo}+{gl:10s} seed{s}  "
                      f"both {r0['err_both']:.3f}  local-only {r0['err_local_only']:.3f}  "
                      f"global_lift {r0['global_lift']:.3f}  {r0['train_s']}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                log.failure(dataset=d.name, local=lo, glob=gl, seed=s, error=repr(exc))
                print(f"[{i}/{len(jobs)}] FAILED {d.name} {lo}+{gl} seed{s}: {exc}",
                      flush=True)
            if rows:
                keys = sorted({k for r in rows for k in r})
                with out.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                    w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
