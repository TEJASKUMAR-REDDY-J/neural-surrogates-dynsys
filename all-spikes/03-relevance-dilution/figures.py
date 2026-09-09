"""Figures for relevance dilution and lag - including what the model actually predicts.

Aggregate error curves say which condition is worse. They do not show what "worse" looks
like, and the two failures here fail in completely different ways: one drifts off the
attractor, the other freezes. So the last figure retrains a handful of models and overlays
their free-running predictions on the truth.

    python figures.py
"""

from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "00-replications"))

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
KIND_C = {"white": "#2980b9", "drift": "#c0392b", "foreign": "#8e44ad", "echo": "#16a085"}
MODE_C = {"sync": "#333333", "lagged": "#2980b9", "intermittent": "#c0392b",
          "jittered": "#16a085"}


def load_dilution():
    return pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / "results/dilution__shard*.csv"))])


# ---------------------------------------------------------------------------------------

def fig_dilution(d):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))

    for k, g in d.groupby("kind"):
        m = g.groupby("n_distractors")[["model_h1", "knn_h1"]].median()
        ax[0].plot(m.index, m.model_h1, "o-", color=KIND_C[k], label=f"{k}", ms=4)
        ax[0].plot(m.index, m.knn_h1, "o--", color=KIND_C[k], ms=3, alpha=0.55)
    ax[0].set_xscale("symlog", linthresh=3); ax[0].set_yscale("log")
    ax[0].set_xlabel("irrelevant columns added")
    ax[0].set_ylabel("one-step error on the REAL variables")
    ax[0].set_title("solid = trained network,  dashed = lookup\n"
                    "lookup collapses; the network mostly does not", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    for k, g in d.groupby("kind"):
        m = g.groupby("n_distractors").ratio_h1.median()
        ax[1].plot(m.index, m.values, "o-", color=KIND_C[k], label=k, ms=4)
    ax[1].axhline(1.0, color="#c0392b", ls="--", lw=1.2)
    ax[1].text(0.02, 0.92, "above the line = the network LOSES to lookup", fontsize=7,
               color="#c0392b", transform=ax[1].transAxes)
    ax[1].set_xscale("symlog", linthresh=3); ax[1].set_yscale("log")
    ax[1].set_xlabel("irrelevant columns added")
    ax[1].set_ylabel("network error / lookup error")
    ax[1].set_title("Only the NON-STATIONARY junk\nflips the verdict", fontsize=9)

    for k, g in d.groupby("kind"):
        m = g.groupby("n_distractors").attn_on_real.median()
        sel = m / np.array([3 / (3 + n) for n in m.index])
        ax[2].plot(m.index, sel.values, "o-", color=KIND_C[k], label=k, ms=4)
    ax[2].axhline(1.0, color="#888", ls=":", lw=1.2)
    ax[2].text(0.02, 0.06, "below the line = worse than ignoring the inputs", fontsize=7,
               color="#666", transform=ax[2].transAxes)
    ax[2].set_xscale("symlog", linthresh=3)
    ax[2].set_xlabel("irrelevant columns added")
    ax[2].set_ylabel("selectivity for the real signal")
    ax[2].set_title("Can it tell what matters?\n(sensitivity on real / blind guess)",
                    fontsize=9)

    fig.suptitle("Relevance dilution: the dynamics never change, only the junk around them",
                 fontsize=10.5, y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "1_dilution.png", bbox_inches="tight")
    plt.close(fig)


def fig_why_drift(d):
    """The generalisation gap, and the distribution shift that explains it."""
    from run import build

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    for k, g in d.groupby("kind"):
        m = g.groupby("n_distractors").gen_gap.median()
        ax[0].plot(m.index, m.values, "o-", color=KIND_C[k], label=k, ms=4)
    ax[0].set_xscale("symlog", linthresh=3); ax[0].set_yscale("log")
    ax[0].set_xlabel("irrelevant columns added")
    ax[0].set_ylabel("test loss / train loss  (real variables)")
    ax[0].set_title("Drift memorises: a gap of 10,000x\nis not misallocated capacity",
                    fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    shifts = {}
    for k in ("white", "drift", "foreign", "echo"):
        tr, te, nr = build(k, 12, 6000, 3000, 1000, 0)
        j_tr, j_te = tr[:, nr:], te[:, nr:]
        shifts[k] = float(np.abs(j_tr.mean(0) - j_te.mean(0)).mean() /
                          (j_tr.std(0).mean() + 1e-9))
    ax[1].bar(range(len(shifts)), list(shifts.values()),
              color=[KIND_C[k] for k in shifts])
    ax[1].set_xticks(range(len(shifts))); ax[1].set_xticklabels(shifts, fontsize=8)
    ax[1].set_ylabel("train -> test shift in the JUNK columns\n(fraction of their own SD)")
    ax[1].set_title("Only drift moves between\ntrain and test", fontsize=9)

    tr, te, nr = build("drift", 6, 6000, 3000, 1000, 0)
    n = 1400
    ax[2].plot(np.arange(n), tr[:n, nr], color="#c0392b", lw=1.1, label="train block")
    ax[2].plot(np.arange(len(te[:n])) + 7000, te[:n, nr], color="#e08e8e", lw=1.1,
               label="test block")
    ax[2].set_xlabel("time index")
    ax[2].set_ylabel("one drift column")
    ax[2].set_title("and here is why: the test block sits\nat a phase training never saw",
                    fontsize=9)
    ax[2].legend(fontsize=7, frameon=False)

    fig.suptitle("Why non-stationary junk is the dangerous kind", fontsize=10.5, y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "2_why_drift.png", bbox_inches="tight")
    plt.close(fig)


def fig_lag():
    d = pd.read_csv(HERE / "results/lag.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    for m, g in d[d.lookback == 1].groupby("mode"):
        if m == "sync":
            continue
        s = g.groupby("lag").skill.median()
        ax[0].plot(s.index, s.values, "o-", color=MODE_C[m], label=m, ms=4)
    sync = d[(d["mode"] == "sync")].skill.median()
    ax[0].axhline(sync, color="#333", ls=":", lw=1.2)
    ax[0].text(1.2, sync - 0.06, "no lag at all", fontsize=7, color="#333")
    ax[0].axhline(0.5, color="#c0392b", ls="--", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("lag (steps)"); ax[0].set_ylabel("skill  (1 = perfect, 0 = useless)")
    ax[0].set_title("A shifted signal survives.\nA frozen one does not.", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    piv1 = d[d.lookback == 1].pivot_table(index="lag", columns="mode", values="skill")
    piv8 = d[d.lookback == 8].pivot_table(index="lag", columns="mode", values="skill")
    rec = (piv8 - piv1).drop(columns=[c for c in ("sync",) if c in piv1])
    for m in rec.columns:
        ax[1].plot(rec.index, rec[m].values, "o-", color=MODE_C[m], label=m, ms=4)
    ax[1].axhline(0, color="#888", ls=":", lw=1)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("lag (steps)")
    ax[1].set_ylabel("skill gained by adding an 8-step history")
    ax[1].set_title("History rescues variable staleness,\nnot uniform delay", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)

    tol = {}
    for m in piv1.columns:
        if m == "sync":
            continue
        ok = piv1.index[piv1[m] > 0.5]
        tol[m] = int(ok.max()) if len(ok) else 0
    ax[2].bar(range(len(tol)), list(tol.values()), color=[MODE_C[k] for k in tol])
    ax[2].set_xticks(range(len(tol))); ax[2].set_xticklabels(tol, fontsize=8)
    ax[2].set_ylabel("largest lag still usable (skill > 0.5)")
    ax[2].set_title("Lag tolerance", fontsize=9)

    fig.suptitle("Delivery: the same information, arriving differently",
                 fontsize=10.5, y=1.03)
    fig.tight_layout(); fig.savefig(FIG / "3_lag.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------

def fig_actuals():
    """What the failures actually look like. Retrains a few small models on purpose."""
    import torch
    from run import Net, build

    conds = [("clean", "white", 0), ("96 white-noise columns", "white", 96),
             ("96 drift columns", "drift", 96)]
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.4))

    for c, (title, kind, d) in enumerate(conds):
        tr, te, nr = build(kind, d, 6000, 3000, 1000, 0)
        torch.manual_seed(0)
        m = Net(tr.shape[1], 96)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        X = torch.tensor(tr[:-1]); Y = torch.tensor(tr[1:] - tr[:-1])
        for _ in range(120):
            p = torch.randperm(len(X))
            for i in range(0, len(X), 128):
                j = p[i : i + 128]
                loss = ((m(X[j]) - Y[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        steps, s0 = 300, 40
        x = te[s0 : s0 + 1].copy()
        traj = [x[0, :nr].copy()]
        with torch.no_grad():
            for _ in range(steps):
                x = x + m(torch.tensor(x, dtype=torch.float32)).numpy()
                traj.append(x[0, :nr].copy())
        traj = np.array(traj)
        true = te[s0 : s0 + steps + 1, :nr]

        ax[0, c].plot(true[:, 0], color="#333", lw=1.4, label="truth")
        ax[0, c].plot(traj[:, 0], color="#c0392b", lw=1.2, label="model, free-running")
        ax[0, c].set_ylim(-3.5, 3.5)
        ax[0, c].set_title(title, fontsize=9.5)
        ax[0, c].set_xlabel("steps ahead")
        if c == 0:
            ax[0, c].set_ylabel("first real variable")
            ax[0, c].legend(fontsize=7, frameon=False)

        ax[1, c].plot(true[:, 0], true[:, 1], color="#333", lw=0.9, label="truth")
        ax[1, c].plot(traj[:, 0], traj[:, 1], color="#c0392b", lw=0.9, alpha=0.85,
                      label="model")
        lim = max(3.5, np.abs(traj[:, :2]).max() * 1.05)
        ax[1, c].set_xlim(-lim, lim); ax[1, c].set_ylim(-lim, lim)
        ax[1, c].set_xlabel("variable 1")
        if c == 0:
            ax[1, c].set_ylabel("variable 2")

    fig.suptitle("What the failures look like: the dynamics are identical in all three, "
                 "only the surrounding junk differs", fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "4_actuals.png", bbox_inches="tight")
    plt.close(fig)


def fig_lag_actuals():
    """Free-running predictions under each delivery mode."""
    import torch
    from lag import Net as LNet, deliver, windowed
    from src.common import systems as S

    rng = np.random.default_rng(0)
    real = S.trajectory("Lorenz", 10100)[:10100]
    real = ((real - real.mean(0)) / (real.std(0) + 1e-9)).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for c, (title, mode, L) in enumerate([("no lag", "sync", 0),
                                          ("uniform lag 32", "lagged", 32),
                                          ("refreshes every 32", "intermittent", 32)]):
        seen = deliver(real, mode, L, rng).astype(np.float32)
        X = windowed(seen, 1)
        Y = (np.roll(real, -1, axis=0) - real).astype(np.float32)
        Xtr, Ytr = X[:6000], Y[:6000]
        torch.manual_seed(0)
        m = LNet(X.shape[1], Y.shape[1], 96)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
        Xt, Yt = torch.tensor(Xtr), torch.tensor(Ytr)
        for _ in range(100):
            p = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 128):
                j = p[i : i + 128]
                loss = ((m(Xt[j]) - Yt[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        sl = slice(7200, 7600)
        with torch.no_grad():
            pred = m(torch.tensor(X[sl])).numpy()
        truth = Y[sl]
        ax[c].plot(truth[:, 0], color="#333", lw=1.2, label="true change")
        ax[c].plot(pred[:, 0], color=MODE_C[mode], lw=1.1, label="predicted")
        ax[c].set_title(title, fontsize=9.5)
        ax[c].set_xlabel("time index")
        if c == 0:
            ax[c].set_ylabel("one-step change in variable 1")
            ax[c].legend(fontsize=7, frameon=False)
    fig.suptitle("Same dynamics, same model, three delivery schemes",
                 fontsize=10.5, y=1.04)
    fig.tight_layout(); fig.savefig(FIG / "5_lag_actuals.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = load_dilution()
    fig_dilution(d); print("wrote 1_dilution.png")
    fig_why_drift(d); print("wrote 2_why_drift.png")
    fig_lag(); print("wrote 3_lag.png")
    fig_actuals(); print("wrote 4_actuals.png")
    fig_lag_actuals(); print("wrote 5_lag_actuals.png")


if __name__ == "__main__":
    main()
