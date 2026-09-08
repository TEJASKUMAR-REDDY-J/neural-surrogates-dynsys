"""Smoke test: three neural cellular automata against five non-learning methods, on nine
datasets spanning five kinds of data.

The question is narrow and worth stating plainly:

    Does adding a "look at the whole lattice" branch to a cellular automaton help, and does
    it help in the places we would predict - on data where the answer is not in the
    neighbours?

Every method sees exactly the same hidden cells, so the comparison is like-for-like. The
five non-learning methods are there because of what the replication battery kept finding:
a lookup table ties with a trained network more often than anyone reports.

    python -m src.run_smoke --probe                    time one fit, estimate the run
    python -m src.run_smoke --shard 0 --n-shards 2     the real thing, one of two workers
    python -m src.run_smoke --optimizer-sweep          which optimiser, measured not assumed
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

from src.common import models as M  # noqa: E402
from src.common import probes as P  # noqa: E402
from src.common import task as T  # noqa: E402
from src.common.data import build_all  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402

RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

# Trained on random 30% holes and clean input. Everything else is a held-out condition -
# a different amount hidden, a contiguous block instead of scatter, noisy measurements, or
# forecasting. Generalising off the training condition is most of what we want to see.
def conditions(ds) -> list[tuple]:
    conds = [
        ("random", 0.1, 0.0), ("random", 0.3, 0.0),
        ("random", 0.5, 0.0), ("random", 0.7, 0.0),
        ("block", 0.3, 0.0),
        ("random", 0.3, 0.10), ("random", 0.3, 0.25),
    ]
    if ds.temporal:
        conds.append(("tail", 0.25, 0.0))
    return conds


def split(ds, n_train_frac: float = 0.75):
    n = len(ds.X)
    k = int(n * n_train_frac)
    return ds.X[:k], ds.X[k:]


def eval_rows(ds, tr, te, seed, model=None, baseline=None, steps_grid=(1, 2, 4, 8, 16, 32)):
    """Score one method on every condition, using masks identical across methods."""
    rows = []
    binary = ds.name.startswith("eca")
    for ci, (mode, frac, noise) in enumerate(conditions(ds)):
        rng = np.random.default_rng(10_000 + 97 * ci + seed)     # same mask for every method
        mask = T.make_mask((len(te),), ds.spatial, frac, mode, rng)
        obs = T.corrupt(te, mask, noise, rng)
        base = dict(dataset=ds.name, modality=ds.modality, ordered=ds.ordered,
                    n_cells=int(np.prod(ds.spatial)), seed=seed,
                    cond_mode=mode, cond_frac=frac, cond_noise=noise,
                    hidden_frac=round(float(1 - mask.mean()), 4))
        if baseline is not None:
            t0 = time.time()
            pred = T.baseline_predict(baseline, obs, mask, tr, ds.spatial)
            wall = time.time() - t0
            sc = T.score(pred, te, mask, binary)
            rows.append({**base, "kind": "baseline", "method": baseline, "arch": "",
                         "n_params": 0, "steps": 0, "best_steps": 0, "overrun_ratio": 1.0,
                         "vpt_steps": np.nan, "scale": np.nan,
                         "infer_wall_s": round(wall, 3), **sc})
        else:
            t0 = time.time()
            # the update is stochastic (half the cells fire each step), so fix the stream:
            # otherwise the same model scores differently on the same data
            torch.manual_seed(50_000 + 97 * ci + seed)
            model.eval()
            with torch.no_grad():
                s = model.seed(torch.tensor(obs), torch.tensor(mask))
                per = {}
                for k in range(1, max(steps_grid) + 1):
                    s = model.step(s)
                    if k in steps_grid:
                        per[k] = T.score(s[:, : model.c_data].numpy(), te, mask, binary)
            wall = time.time() - t0
            bk = min(per, key=lambda k: per[k]["nrmse_hidden"])
            lk = max(steps_grid)
            # Valid prediction time, in iterations: how far the automaton can be run before
            # its answer stops being usable. 0.4 is the threshold used throughout
            # 00-replications, so the numbers are directly comparable. This is the number
            # that catches the R9 failure - a rule that is good at step 4 and garbage at 32.
            vpt = 0
            for k in sorted(per):
                if per[k]["nrmse_hidden"] > 0.4:
                    break
                vpt = k
            for k, sc in per.items():
                rows.append({**base, "kind": "learned", "method": model.arch, "arch": model.arch,
                             "n_params": model.n_params, "steps": k, "best_steps": bk,
                             "vpt_steps": vpt, "scale": round(model.scale, 4),
                             "overrun_ratio": round(
                                 per[lk]["nrmse_hidden"] / max(per[bk]["nrmse_hidden"], 1e-9), 4),
                             "infer_wall_s": round(wall / len(per), 3), **sc})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-per-set", type=int, default=900)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--archs", nargs="+", default=list(M.ARCHITECTURES))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--optimizer-sweep", action="store_true")
    ap.add_argument("--tag", default="")
    # Parameter budget per architecture. The global branches cost a fixed ~5k on top of the
    # shared body, so a single width for all four would leave the local-only one smaller and
    # confound shape with capacity. 0 means "use the literal 32-64-128-64-32 specification".
    ap.add_argument("--budget", type=int, default=13_000)
    ap.add_argument("--no-probes", action="store_true")
    args = ap.parse_args()

    def make(arch, ds):
        """Build one automaton, sized to the budget so the comparison is about shape."""
        if not args.budget:
            return M.build(arch, ds.spatial, ds.channels)
        sc = M.scale_for_budget(arch, ds.spatial, ds.channels, args.budget)
        return M.build(arch, ds.spatial, ds.channels, scale=sc)

    sets = build_all(n_per_set=args.n_per_set)
    if args.datasets:
        sets = [d for d in sets if d.name in args.datasets]

    # ---------------- timing probe: measure before committing to the whole grid ----------
    if args.probe:
        print(f"{'dataset':24s} {'arch':14s} {'params':>8s} {'train_s':>8s} {'eval_s':>7s}",
              flush=True)
        total = 0.0
        for ds in sets:
            tr, te = split(ds)
            for arch in args.archs:
                m = make(arch, ds)
                t0 = time.time()
                T.train(m, tr, ds.spatial, epochs=3, batch=args.batch, lr=args.lr,
                        optimizer=args.optimizer, seed=0)
                ts = (time.time() - t0) / 3 * args.epochs
                t1 = time.time()
                eval_rows(ds, tr, te, 0, model=m)
                es = time.time() - t1
                total += ts + es
                print(f"{ds.name:24s} {arch:14s} {m.n_params:>8,} {ts:>8.1f} {es:>7.1f}",
                      flush=True)
        n_fits = len(args.seeds)
        print(f"\nestimated total: {total * n_fits / 3600:.2f} h serial, "
              f"~{total * n_fits / 3600 / 1.7:.2f} h on two shards")
        return

    # ---------------- optimiser sweep: pick the recipe by measuring it ------------------
    if args.optimizer_sweep:
        rows = []
        with RunLog("nca_optimizer_sweep", vars(args), "cli", LOGS) as log:
            for ds in sets:
                tr, te = split(ds)
                for opt in ("adamw", "nadam", "radam", "adamax", "sgdm"):
                    for seed in args.seeds[:2]:
                        m = make("A2_pooled", ds)
                        h = T.train(m, tr, ds.spatial, epochs=args.epochs, batch=args.batch,
                                    lr=args.lr, optimizer=opt, seed=seed)
                        ev = T.evaluate(m, te, ds.spatial, seed=seed)
                        r = dict(dataset=ds.name, optimizer=opt, seed=seed,
                                 train_loss=h["train_loss"], train_wall_s=h["train_wall_s"],
                                 nrmse_hidden=ev["best"]["nrmse_hidden"],
                                 best_steps=ev["best_steps"])
                        rows.append(r); log.result(**r)
                        print(f"  {ds.name:22s} {opt:7s} seed{seed}  "
                              f"loss {r['train_loss']:.5f}  nrmse {r['nrmse_hidden']:.4f}  "
                              f"{r['train_wall_s']}s", flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"optimizer_sweep{('__' + args.tag) if args.tag else ''}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("wrote", out)
        return

    # ---------------- the smoke test ------------------------------------------------------
    jobs = [(ds, arch, seed) for ds in sets for arch in args.archs for seed in args.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.tag}__shard{args.shard}" if args.tag else f"shard{args.shard}"
    out = RESULTS / f"smoke__{suffix}.csv"
    probe_out = RESULTS / f"probes__{suffix}.csv"
    rows: list[dict] = []
    probe_rows: list[dict] = []

    with RunLog("nca_smoke", vars(args), "cli", LOGS) as log:
        # baselines once per dataset x seed - they do not depend on the architecture
        if args.shard == 0:
            for ds in sets:
                tr, te = split(ds)
                for seed in args.seeds:
                    for b in T.BASELINES:
                        rs = eval_rows(ds, tr, te, seed, baseline=b)
                        rows += rs
                        for r in rs:
                            log.result(**r)
                    print(f"[baselines] {ds.name:24s} seed{seed} done", flush=True)

        for n, (ds, arch, seed) in enumerate(jobs, 1):
            tr, te = split(ds)
            t0 = time.time()
            try:
                m = make(arch, ds)
                h = T.train(m, tr, ds.spatial, epochs=args.epochs, batch=args.batch,
                            lr=args.lr, optimizer=args.optimizer, seed=seed)
                rs = eval_rows(ds, tr, te, seed, model=m)
                for r in rs:
                    r.update(train_loss=h["train_loss"], train_wall_s=h["train_wall_s"],
                             epochs=args.epochs, optimizer=args.optimizer)
                rows += rs
                for r in rs:
                    log.result(**r)

                # The mechanism probes. These are the point of the spike: the error numbers
                # say which architecture won, these say why - and whether the global branch
                # was doing anything at all.
                if not args.no_probes:
                    rngp = np.random.default_rng(777 + seed)
                    mk = T.make_mask((len(te),), ds.spatial, 0.3, "random", rngp)
                    ob = T.corrupt(te, mk, 0.0, rngp)
                    binary = ds.name.startswith("eca")
                    pr = P.propagation(m, ds.spatial, steps=min(32, 2 * max(ds.spatial)))
                    prow = dict(dataset=ds.name, modality=ds.modality,
                                ordered=ds.ordered, n_cells=int(np.prod(ds.spatial)),
                                arch=arch, seed=seed, n_params=m.n_params,
                                scale=round(m.scale, 4), has_global=m.has_global(),
                                **P.propagation_summary(pr),
                                **P.knockout(m, ob, mk, te, steps=8, binary=binary, seed=seed),
                                **P.contribution(m, ob, mk, steps=8, seed=seed),
                                **P.memory(m, ob, mk, te, steps=16, wipe_at=8,
                                           binary=binary, seed=seed))
                    prow["radius_curve"] = ";".join(f"{v:g}" for v in pr["radius"])
                    prow["frac_curve"] = ";".join(f"{v:.4f}" for v in pr["frac_affected"])
                    prow.pop("global_share", None)
                    probe_rows.append(prow)
                    log.result(record_kind="probe",
                               **{k: v for k, v in prow.items()
                                  if k not in ("radius_curve", "frac_curve")})

                best = min(r["nrmse_hidden"] for r in rs
                           if r["cond_mode"] == "random" and r["cond_frac"] == 0.3)
                print(f"[{n}/{len(jobs)}] {ds.name:24s} {arch:14s} seed{seed}  "
                      f"loss {h['train_loss']:.5f}  nrmse@30% {best:.4f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                log.failure(dataset=ds.name, arch=arch, seed=seed, error=repr(exc))
                print(f"[{n}/{len(jobs)}] FAILED {ds.name} {arch} seed{seed}: {exc}", flush=True)

            # write after every fit - a crash must not cost the run
            for dest, data in ((out, rows), (probe_out, probe_rows)):
                if data:
                    keys = sorted({k for r in data for k in r})
                    with dest.open("w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                        w.writeheader()
                        w.writerows(data)

    print(f"\nwrote {out} ({len(rows)} rows)")
    if probe_rows:
        print(f"wrote {probe_out} ({len(probe_rows)} rows)")


if __name__ == "__main__":
    main()
