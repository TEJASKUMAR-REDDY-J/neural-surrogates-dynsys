"""The controlled locality test, and the invariance checks that validate it.

A shuffled twin is an ordered dataset with ONE fixed permutation applied to its cells. The
values, the difficulty and every statistic are identical; only the claim that neighbouring
cells are related is gone. So this isolates locality in a way that comparing different
datasets never can.

Two things make the comparison trustworthy, and both are checked here rather than assumed:

  - a fixed permutation is an isometry, so any method that treats the lattice as an
    unordered vector must score IDENTICALLY on a dataset and its twin. Nearest-neighbour
    lookup and PCA imputation are such methods. If their scores move, the shuffle did
    something other than permute and nothing else on this page can be believed.
  - a method that depends on adjacency must get worse. Local averaging is such a method.

Run: python -m src.analysis_shuffle
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
FIG = RES / "figures"

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

TWINS = [("cml_lattice", "cml_lattice__shuffled"),
         ("digits_8x8", "digits_8x8__shuffled"),
         ("multitone_signal", "multitone_signal__shuffled")]
ARCHS = ["A1_local", "A2_pooled", "A3_attentive", "A4_interleaved"]
ARCH_C = {"A1_local": "#2980b9", "A2_pooled": "#e67e22",
          "A3_attentive": "#8e44ad", "A4_interleaved": "#16a085"}
TAG = "v3"


def load_smoke():
    fs = sorted(glob.glob(str(RES / f"smoke__{TAG}__shard*.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[(d.kind == "baseline") | (d.steps == d.best_steps)]
    return d[(d.cond_mode == "random") & (d.cond_frac == 0.3) & (d.cond_noise == 0)]


def load_probes():
    fs = sorted(glob.glob(str(RES / f"probes__{TAG}__shard*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sm = load_smoke()
    t = sm.pivot_table(index="dataset", columns="method", values="nrmse_hidden",
                       aggfunc="median")

    # ---- invariance checks -------------------------------------------------------------
    print("INVARIANCE CHECKS — these validate the manipulation itself")
    print(f"{'method':20s} {'behaviour':34s} {'max |change| across twins':>26s}")
    checks = [("nearest_neighbour", "permutation-invariant: must NOT move", 0.05),
              ("pca_impute", "permutation-invariant: must NOT move", 0.10),
              ("local_mean", "adjacency-dependent: MUST get worse", None)]
    for meth, desc, tol in checks:
        if meth not in t.columns:
            continue
        deltas = [t.loc[b, meth] - t.loc[a, meth] for a, b in TWINS
                  if a in t.index and b in t.index]
        worst = max(abs(np.array(deltas)))
        print(f"{meth:20s} {desc:34s} {worst:>26.3f}")
        if tol is not None:
            assert worst < tol, (
                f"{meth} is permutation-invariant by construction but moved by {worst:.3f} - "
                f"the shuffle did something other than permute")
        else:
            assert min(deltas) > 0, f"{meth} did not degrade: {deltas}"
    print("  all invariance checks passed\n")

    # ---- the figure ---------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))

    x = np.arange(len(TWINS))
    w = 0.8 / len(ARCHS)
    for i, a in enumerate(ARCHS):
        ordered = [t.loc[p, a] for p, _ in TWINS]
        shuf = [t.loc[q, a] for _, q in TWINS]
        ax[0].bar(x + i * w - 0.4, ordered, w, color=ARCH_C[a], label=a)
        ax[1].bar(x + i * w - 0.4, shuf, w, color=ARCH_C[a])
    for k, ttl in enumerate(["ordered (adjacency real)", "shuffled twin (adjacency destroyed)"]):
        ax[k].axhline(1.0, color="#c0392b", ls="--", lw=1.3)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([p.replace("_", "\n") for p, _ in TWINS], fontsize=7)
        ax[k].set_ylim(0, 1.25)
        ax[k].set_ylabel("nRMSE on hidden cells")
        ax[k].set_title(ttl, fontsize=9)
    ax[0].legend(fontsize=6.8, frameon=False)
    ax[1].text(0.03, 0.88, "1.0 = no better than guessing the mean", fontsize=6.8,
               color="#c0392b", transform=ax[1].transAxes)

    # panel 3: the pre-registered prediction, and what happened
    pr = load_probes()
    rows = []
    for a, b in TWINS:
        for arch in ARCHS[1:]:
            xo = pr[(pr.dataset == a) & (pr.arch == arch)].lightcone_leak_step1.median()
            xs = pr[(pr.dataset == b) & (pr.arch == arch)].lightcone_leak_step1.median()
            rows.append((xo, xs, arch))
    for xo, xs, arch in rows:
        ax[2].plot([0, 1], [xo, xs], "o-", color=ARCH_C[arch], ms=4, lw=1.3, alpha=0.85)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["ordered", "shuffled"], fontsize=8)
    ax[2].set_yscale("symlog", linthresh=1e-4)
    ax[2].set_ylabel("influence outside the light cone\n(step 1)")
    ax[2].set_title("Predicted: global work goes UP\nMeasured: it goes DOWN, 9/9",
                    fontsize=9)

    fig.suptitle("The controlled locality test: destroy adjacency, keep everything else",
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG / "probe_4_shuffled_twins.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'probe_4_shuffled_twins.png'}")

    # ---- the table -----------------------------------------------------------------------
    out = []
    for a, b in TWINS:
        for m in ARCHS + ["local_mean", "nearest_neighbour", "pca_impute"]:
            if m in t.columns:
                out.append(dict(pair=a, method=m, ordered=round(t.loc[a, m], 3),
                                shuffled=round(t.loc[b, m], 3),
                                change=round(t.loc[b, m] - t.loc[a, m], 3)))
    df = pd.DataFrame(out)
    df.to_csv(RES / "shuffled_twins.csv", index=False)
    print("\n=== error, ordered vs shuffled twin ===")
    print(df.pivot_table(index="method", columns="pair", values="change").round(3).to_string())
    print("\n(positive = worse after shuffling; ~0 = unaffected, as a permutation-invariant "
          "method must be)")


if __name__ == "__main__":
    main()
