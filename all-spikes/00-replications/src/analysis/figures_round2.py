"""Figures for the experiments added after the first battery: R4c, R6, R9, N2, N3, N5.

Kept separate from figures.py because these read archived passes and shard files directly
rather than the single canonical CSV per experiment.
"""

from __future__ import annotations

import csv
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "analysis"
RES = ROOT / "results"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_GOOD, C_BAD, C_MID = "#2980b9", "#c0392b", "#7f8c8d"


def rows_from(*globs):
    out = []
    for g in globs:
        for f in sorted(RES.glob(g)):
            out += list(csv.DictReader(f.open(encoding="utf-8")))
    return out


def fl(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote {name}")


def loo(X, y):
    pr = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        A = np.column_stack([np.ones(m.sum()), X[m]])
        b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pr[i] = np.r_[1.0, X[i]] @ b
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# --------------------------------------------------------------------------------------


def fig_r4c():
    """The surviving headline: capacity helps at short horizons and fades with distance."""
    rows = rows_from("r4/pass3_r4c/*.csv", "r4/pass2_r4b/*.csv")
    HS = [1, 10, 50, 100, 200, 500]
    per = {h: [] for h in HS}
    kept = {h: [] for h in HS}
    for s in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == s and r["diverged"] == "False"
               and fl(r, "n_train") == 8000]
        clean = [r for r in sub if not (np.isfinite(fl(r, "err_h500")) and fl(r, "err_h500") > 3.0)]
        if len(sub) < 10:
            continue
        for tag, use in (("all", sub), ("clean", clean)):
            if len(use) < 10:
                continue
            p = np.array([fl(r, "n_params") for r in use])
            for h in HS:
                y = np.array([fl(r, f"err_h{h}") for r in use])
                ok = np.isfinite(p) & np.isfinite(y)
                if ok.sum() < 8:
                    continue
                ra = np.argsort(np.argsort(p[ok])).astype(float)
                rb = np.argsort(np.argsort(-y[ok])).astype(float)
                v = float(np.corrcoef(ra, rb)[0, 1])
                (per if tag == "all" else kept)[h].append(v)

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axs[0]
    med = [np.median(per[h]) for h in HS]
    q1 = [np.percentile(per[h], 25) for h in HS]
    q3 = [np.percentile(per[h], 75) for h in HS]
    ax.fill_between(range(len(HS)), q1, q3, alpha=0.2, color=C_GOOD, label="middle half of systems")
    ax.plot(range(len(HS)), med, "o-", color=C_GOOD, lw=1.6, ms=5, label="median (all fits)")
    ax.plot(range(len(HS)), [np.median(kept[h]) for h in HS], "s--", color=C_BAD, lw=1.2,
            ms=4, label="median (exploded fits removed)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(HS)), [str(h) for h in HS])
    ax.set_xlabel("horizon (steps ahead)")
    ax.set_ylabel("benefit of extra capacity")
    ax.set_title("R4c: a bigger model helps a lot at short range\nand barely at all far out",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axs[1]
    n = [sum(v > 0.3 for v in per[h]) for h in HS]
    ax.bar(range(len(HS)), n, color=C_GOOD, alpha=0.8)
    for i, v in enumerate(n):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(len(HS)), [str(h) for h in HS])
    ax.set_xlabel("horizon (steps ahead)")
    ax.set_ylabel("systems where capacity clearly helps")
    ax.set_ylim(0, len(per[1]) + 2)
    ax.set_title(f"out of {len(per[1])} systems", fontsize=9)
    fig.suptitle("Extra capacity buys short-horizon accuracy, and little else", fontsize=10)
    save(fig, "r4c_horizon_effect.png")


def fig_r6():
    """Capacity improves pointwise accuracy and not structure."""
    rows = rows_from("r4/pass2_r4b/*.csv")
    keys = [("err_h1", "one-step\nerror"), ("err_h50", "error at\nh=50"),
            ("err_h1000", "error at\nh=1000"), ("spectrum_error", "frequency\nspectrum"),
            ("density_kl", "attractor\nshape")]
    vals = {k: [] for k, _ in keys}
    for s in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == s and r["diverged"] == "False"]
        nmax = max(fl(r, "n_train") for r in sub)
        sub = [r for r in sub if fl(r, "n_train") == nmax]
        if len(sub) < 10:
            continue
        p = np.array([fl(r, "n_params") for r in sub])
        for k, _ in keys:
            y = np.array([fl(r, k) for r in sub])
            ok = np.isfinite(p) & np.isfinite(y)
            if ok.sum() < 8:
                continue
            ra = np.argsort(np.argsort(p[ok])).astype(float)
            rb = np.argsort(np.argsort(-y[ok])).astype(float)
            vals[k].append(float(np.corrcoef(ra, rb)[0, 1]))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    med = [np.median(vals[k]) for k, _ in keys]
    cols = [C_GOOD if m > 0.4 else (C_MID if m > 0.2 else C_BAD) for m in med]
    ax.bar(range(len(keys)), med, color=cols, alpha=0.85)
    for i, (m, k) in enumerate(zip(med, [k for k, _ in keys])):
        ax.scatter([i] * len(vals[k]), vals[k], s=12, color="k", alpha=0.35, zorder=3)
        ax.text(i, m + 0.04, f"{m:+.2f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys], fontsize=8)
    ax.set_ylabel("benefit of extra capacity")
    ax.set_ylim(-0.5, 1.05)
    ax.set_title("R6: a bigger model predicts the next step much better,\n"
                 "and gets the system's shape no better at all", fontsize=9.5)
    save(fig, "r6_pointwise_vs_structural.png")


def fig_r9():
    """Refinement improves for a few passes, then turns around."""
    rows = rows_from("r9/refinement__shard*.csv")
    if not rows:
        return
    sysn = sorted({r["system"] for r in rows})
    hs = sorted({int(fl(r, "horizon")) for r in rows})
    tp = int(fl(rows[0], "train_passes"))
    maxp = int(max(fl(r, "passes") for r in rows))

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axs[0]
    shown = 0
    for s in sysn:
        for h in hs:
            sub = [r for r in rows if r["system"] == s and int(fl(r, "horizon")) == h]
            e = {}
            for r in sub:
                e.setdefault(int(fl(r, "passes")), []).append(fl(r, "err_refine"))
            m = np.array([np.nanmean(e[k]) for k in sorted(e)])
            if len(m) < maxp or not np.isfinite(m).all() or m.min() > 0.9:
                continue  # nothing to see where the forecast was useless anyway
            ax.plot(range(1, len(m) + 1), m / m.min(), lw=1.0, alpha=0.75)
            shown += 1
    ax.axvline(tp + 0.5, color="k", ls=":", lw=1.2)
    ax.text(tp + 0.7, ax.get_ylim()[1] * 0.92, "trained\nthis far", fontsize=7)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("number of passes")
    ax.set_ylabel("error, relative to its own best")
    ax.set_title(f"R9: each line is one system and horizon ({shown} shown)\n"
                 "every one turns back up", fontsize=9)

    ax = axs[1]
    best, final = [], []
    for s in sysn:
        for h in hs:
            sub = [r for r in rows if r["system"] == s and int(fl(r, "horizon")) == h]
            e = {}
            for r in sub:
                e.setdefault(int(fl(r, "passes")), []).append(fl(r, "err_refine"))
            m = np.array([np.nanmean(e[k]) for k in sorted(e)])
            if len(m) < maxp or not np.isfinite(m).all():
                continue
            best.append(int(np.argmin(m)) + 1)
            final.append(m[-1] / m.min())
    ax.hist(best, bins=np.arange(0.5, maxp + 1.5), color=C_GOOD, alpha=0.85)
    ax.axvline(tp + 0.5, color="k", ls=":", lw=1.2)
    ax.set_xlabel("pass with the lowest error")
    ax.set_ylabel("cases")
    ax.set_title(f"best pass is always within the {tp} it was trained for\n"
                 f"(median {int(np.median(best))}; "
                 f"{sum(f > 1.1 for f in final)}/{len(final)} end >10% worse)", fontsize=9)
    fig.suptitle("Thinking again helps for about three passes, then hurts", fontsize=10)
    save(fig, "r9_refinement.png")


def fig_n2():
    """Skill survives noise; predictability of skill does not."""
    inst = {r["system"]: r for r in csv.DictReader(
        (RES / "r2" / "instruments.csv").open(encoding="utf-8"))}
    levels = [("0", "surrogate_vs_statistics__shard*.csv"),
              ("1", "surrogate_vs_statistics__noise0.01__shard*.csv"),
              ("5", "surrogate_vs_statistics__noise0.05__shard*.csv"),
              ("20", "surrogate_vs_statistics__noise0.2__shard*.csv")]
    sets = {}
    for lab, pat in levels:
        rr = rows_from(f"r3/{pat}")
        if not rr:
            return
        sets[lab] = {s: (float(np.mean([fl(r, "vpt_steps") for r in rr if r["system"] == s])),
                         fl([r for r in rr if r["system"] == s][0], "parrot_vpt_steps"))
                     for s in sorted({r["system"] for r in rr})}
    common = sorted(set.intersection(*[set(v) for v in sets.values()]) & set(inst))
    feats = ("spectral_entropy", "k01", "lyap_max", "corr_dim_ours", "wpe", "kaplan_yorke_dim")
    raw = np.column_stack([[fl(inst[s], f) for s in common] for f in feats])
    ok = np.isfinite(raw).all(1)
    raw = raw[ok]
    common = [c for c, o in zip(common, ok) if o]
    R = np.argsort(np.argsort(raw, axis=0), axis=0) / (len(raw) - 1)

    labs = [l for l, _ in levels]
    vpt, pred, beat = [], [], []
    for lab in labs:
        v = np.array([sets[lab][s][0] for s in common])
        pa = np.array([sets[lab][s][1] for s in common])
        y = np.log(np.maximum(v, 1e-3))
        pred.append(max(loo(R[:, list(c)], y) for c in itertools.combinations(range(len(feats)), 2)))
        vpt.append(np.median(v))
        beat.append(100.0 * (v > pa).sum() / len(v))

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = range(len(labs))
    axs[0].plot(x, np.array(vpt) / vpt[0], "o-", color=C_GOOD, lw=1.6, ms=5)
    axs[0].set_ylabel("forecast horizon, relative to clean")
    axs[0].set_title("the surrogates cope", fontsize=9)
    axs[0].set_ylim(0, 1.1)

    axs[1].bar(x, pred, color=[C_GOOD if p > 0 else C_BAD for p in pred], alpha=0.85)
    for i, p in enumerate(pred):
        axs[1].text(i, p + (0.015 if p > 0 else -0.03), f"{p:+.2f}", ha="center", fontsize=8)
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_ylabel("predictability of that skill")
    axs[1].set_title("the predictor does not\n(1% noise is enough)", fontsize=9)

    axs[2].plot(x, beat, "o-", color=C_MID, lw=1.6, ms=5)
    axs[2].axhline(50, color="k", ls=":", lw=1.0)
    axs[2].text(0.05, 51, "coin flip", fontsize=7)
    axs[2].set_ylabel("% of systems where the model beats copying")
    axs[2].set_title("and only when noisy does the\nnetwork beat plain copying", fontsize=9)
    for ax in axs:
        ax.set_xticks(list(x), [f"{l}%" for l in labs])
        ax.set_xlabel("observational noise")
    fig.suptitle(f"N2: same {len(common)} systems at every noise level", fontsize=10)
    save(fig, "n2_noise.png")


def fig_n3():
    """Which cheap statistics predict surrogate skill, at 30 systems and at 108."""
    inst = {r["system"]: r for r in csv.DictReader(
        (RES / "r2" / "instruments.csv").open(encoding="utf-8"))}
    rows = [r for r in rows_from("r3/surrogate_vs_statistics__shard*.csv")
            if not r.get("noise") or fl(r, "noise") == 0]
    sysn = sorted({r["system"] for r in rows})
    per = {s: float(np.mean([fl(r, "vpt_steps") for r in rows if r["system"] == s]))
           for s in sysn}
    feats = [("spectral_entropy", "spectral\nentropy"), ("k01", "0-1 chaos\ntest"),
             ("corr_dim_ours", "correlation\ndimension"), ("lyap_max", "Lyapunov\nexponent"),
             ("kaplan_yorke_dim", "Kaplan-Yorke\ndimension"), ("wpe", "weighted perm.\nentropy")]
    common = [s for s in sysn if s in inst]
    raw = np.column_stack([[fl(inst[s], f) for s in common] for f, _ in feats])
    y = np.log(np.maximum([per[s] for s in common], 1e-3))
    ok = np.isfinite(raw).all(1) & np.isfinite(y)
    raw, y = raw[ok], y[ok]
    R = np.argsort(np.argsort(raw, axis=0), axis=0) / (len(raw) - 1)
    single = [loo(R[:, [i]], y) for i in range(len(feats))]
    bestpair = max(((loo(R[:, list(c)], y), c) for c in itertools.combinations(range(len(feats)), 2)))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    order = np.argsort(single)[::-1]
    cols = [C_GOOD if single[i] > 0.1 else (C_MID if single[i] > 0 else C_BAD) for i in order]
    ax.bar(range(len(feats)), [single[i] for i in order], color=cols, alpha=0.85)
    for j, i in enumerate(order):
        ax.text(j, single[i] + (0.008 if single[i] > 0 else -0.02), f"{single[i]:+.2f}",
                ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(bestpair[0], color=C_GOOD, ls="--", lw=1.2)
    ax.text(len(feats) - 0.4, bestpair[0] + 0.008,
            f"best pair: {feats[bestpair[1][0]][1]} + {feats[bestpair[1][1]][1]}"
            .replace("\n", " ") + f" = {bestpair[0]:.2f}", ha="right", fontsize=7.5, color=C_GOOD)
    ax.set_xticks(range(len(feats)), [feats[i][1] for i in order], fontsize=7.5)
    ax.set_ylabel("predicts an unseen system?  (leave-one-out R2)")
    ax.set_title(f"N3: what predicts how well a surrogate will do, across {len(y)} systems\n"
                 "negative means worse than guessing the same value every time", fontsize=9.5)
    save(fig, "n3_predictors.png")


def fig_n5():
    """Batch size against error, with seeds."""
    rows = rows_from("n5/batch_sweep.csv")
    if not rows:
        return
    batches = sorted({int(fl(r, "batch")) for r in rows})
    sysn = sorted({r["system"] for r in rows})
    rel1, rel50, tm = [], [], []
    for b in batches:
        a1, a50, t = [], [], []
        for s in sysn:
            best1 = min(np.mean([fl(r, "err_h1") for r in rows
                                 if r["system"] == s and int(fl(r, "batch")) == bb])
                        for bb in batches)
            best50 = min(np.mean([fl(r, "err_h50") for r in rows
                                  if r["system"] == s and int(fl(r, "batch")) == bb])
                         for bb in batches)
            sub = [r for r in rows if r["system"] == s and int(fl(r, "batch")) == b]
            a1.append(np.mean([fl(r, "err_h1") for r in sub]) / best1)
            a50.append(np.mean([fl(r, "err_h50") for r in sub]) / best50)
            t.append(np.mean([fl(r, "wall_s") for r in sub]))
        rel1.append(np.mean(a1)); rel50.append(np.mean(a50)); tm.append(np.mean(t))

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = range(len(batches))
    ax.plot(x, rel1, "o-", color=C_GOOD, lw=1.5, ms=5, label="one-step error")
    ax.plot(x, rel50, "s-", color=C_BAD, lw=1.5, ms=5, label="error 50 steps out")
    for i, b in enumerate(batches):
        if b == 256:
            ax.axvline(i, color="k", ls=":", lw=1.2)
            ax.text(i + 0.06, max(rel1) * 0.95, "what we used", fontsize=7.5)
    ax.set_xticks(list(x), [str(b) for b in batches])
    ax.set_xlabel("batch size")
    ax.set_ylabel("error, relative to the best setting")
    ax.legend(fontsize=7.5)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, tm, "^--", color=C_MID, lw=1.0, ms=4, alpha=0.7)
    ax2.set_ylabel("seconds per fit", color=C_MID, fontsize=8)
    ax.set_title("N5: smaller batches fit better and cost more\n"
                 "256 was convention, not evidence", fontsize=9.5)
    save(fig, "n5_batch_size.png")


def main():
    print("round-2 figures:")
    for f in (fig_r4c, fig_r6, fig_r9, fig_n2, fig_n3, fig_n5):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"  {f.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
