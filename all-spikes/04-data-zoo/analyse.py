"""Three questions of the zoo, and the figures for them.

  1. How big is the spread ACROSS architectures compared to the spread ACROSS datasets?
     If the second dwarfs the first, outcome is a property of the data, not the model.
  2. Which measurable data properties predict the best achievable error?
  3. Do any data properties predict WHICH method wins - i.e. is architecture selection
     obtainable from the data alone?

    python analyse.py
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
PROPS = ["coverage", "noise_floor", "dist_shift", "decorr_time", "spectral_entropy",
         "n_dims", "n_rows"]
NEURAL = ["mlp", "cnn", "gru"]
ORDER = ["mean", "persistence", "ridge", "knn", "trees", "mlp", "cnn", "gru"]

plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def load():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / "results/zoo__shard*.csv"))],
                  ignore_index=True)
    return d[np.isfinite(d.r2)]


def per_dataset(d):
    """Median over seeds, one row per dataset x method."""
    return (d.groupby(["dataset", "kind", "method"] + PROPS, as_index=False)
            .r2.median())


def q1_spread(d, pd_):
    """Architecture spread against dataset spread."""
    piv = pd_.pivot_table(index="dataset", columns="method", values="r2")
    piv = piv[[c for c in ORDER if c in piv.columns]]
    # clip: an R2 of -50 on a hopeless dataset would dominate any spread measure
    P = piv.clip(lower=-1.0)
    within = (P.max(axis=1) - P.min(axis=1))
    trained = P[[c for c in NEURAL + ["ridge", "knn", "trees"] if c in P.columns]]
    within_trained = trained.max(axis=1) - trained.min(axis=1)
    across = P.max(axis=1)

    print("=== Q1: how much does the ARCHITECTURE change the answer? ===")
    print(f"  best-achievable R2 across datasets : sd = {across.std():.3f}  "
          f"range {across.min():+.3f} to {across.max():+.3f}")
    print(f"  spread across ALL methods, per dataset  : median {within.median():.3f}")
    print(f"  spread across TRAINED methods only      : median {within_trained.median():.3f}")
    print(f"  ratio  dataset spread / architecture spread = "
          f"{across.std() / max(within_trained.median(), 1e-9):.1f}x")
    print()
    print("  how often is each method the best?")
    wins = piv.idxmax(axis=1).value_counts()
    for m in ORDER:
        if m in wins.index:
            print(f"    {m:12s} {wins[m]:>3d} / {len(piv)}")
    neural_best = piv.idxmax(axis=1).isin(NEURAL).sum()
    print(f"    -> a neural net wins on {neural_best}/{len(piv)} datasets")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    order = P.max(axis=1).sort_values().index
    for m in P.columns:
        ax[0].plot(range(len(order)), P.loc[order, m], "o", ms=3.5, alpha=0.75, label=m)
    ax[0].set_xlabel("datasets, ordered by how predictable they are")
    ax[0].set_ylabel("R2 on held-out data (clipped at -1)")
    ax[0].set_title("Every method on every dataset.\n"
                    "Vertical spread = architecture. Horizontal = data.", fontsize=9)
    ax[0].legend(fontsize=6.5, frameon=False, ncol=2)

    ax[1].hist([within_trained.dropna(), (P.max(axis=1) - P.min(axis=1)).dropna()],
               bins=14, label=["trained methods only", "all methods incl. baselines"],
               color=["#2980b9", "#bbbbbb"])
    ax[1].axvline(across.std(), color="#c0392b", lw=1.8)
    ax[1].text(across.std() * 1.05, ax[1].get_ylim()[1] * 0.8,
               f"spread ACROSS\ndatasets = {across.std():.2f}", color="#c0392b", fontsize=7.5)
    ax[1].set_xlabel("R2 spread within one dataset")
    ax[1].set_ylabel("datasets")
    ax[1].set_title("Choosing the architecture moves you this much.\n"
                    "Choosing the dataset moves you that much.", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "zoo_1_spread.png", bbox_inches="tight")
    plt.close(fig)
    return piv


def q2_predict(pd_, piv):
    """Do data properties predict the best achievable R2?"""
    prop = pd_.groupby("dataset")[PROPS].first()
    best = piv.clip(lower=-1.0).max(axis=1).rename("best_r2")
    m = prop.join(best).dropna()
    print("\n=== Q2: which data properties predict how well ANYTHING can do? ===")
    rows = []
    for p in PROPS:
        rho, pv = spearmanr(m[p], m.best_r2)
        rows.append((p, rho, pv))
        print(f"  {p:18s} rho {rho:+.3f}   p {pv:.4f}")
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for i, p in enumerate(["coverage", "noise_floor", "dist_shift"]):
        ax[i].scatter(m[p], m.best_r2, s=26, c="#2980b9", edgecolor="white", linewidth=0.4)
        rho, _ = spearmanr(m[p], m.best_r2)
        ax[i].set_xlabel(p); ax[i].set_ylabel("best R2 any method achieved")
        ax[i].set_title(f"{p}   rho = {rho:+.2f}", fontsize=9)
        if p != "dist_shift":
            ax[i].set_xscale("symlog", linthresh=1e-3)
    fig.suptitle("Predicting the ceiling from the data, before training anything",
                 fontsize=10.5, y=1.04)
    fig.tight_layout(); fig.savefig(FIG / "zoo_2_properties.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows, columns=["property", "rho", "p"]), m


def q3_selection(pd_, piv, m):
    """Do data properties predict WHICH method wins?"""
    print("\n=== Q3: can the data tell you WHICH method to use? ===")
    P = piv.clip(lower=-1.0)
    winner = P.idxmax(axis=1)
    fam = winner.map(lambda w: "neural" if w in NEURAL else
                     ("trained-classical" if w in ("ridge", "knn", "trees") else "trivial"))
    j = m.join(fam.rename("family")).dropna()
    print(j.groupby("family").size().to_string())
    print()
    for p in PROPS:
        g = j.groupby("family")[p].median()
        print(f"  {p:18s} " + "  ".join(f"{k}:{v:.3f}" for k, v in g.items()))

    # margin of the winner over the best non-neural alternative
    nn = [c for c in NEURAL if c in P.columns]
    cl = [c for c in ("ridge", "knn", "trees") if c in P.columns]
    margin = (P[nn].max(axis=1) - P[cl].max(axis=1)).rename("neural_edge")
    mm = m.join(margin).dropna()
    print("\n  does any property predict when a NEURAL net beats classical methods?")
    for p in PROPS:
        rho, pv = spearmanr(mm[p], mm.neural_edge)
        flag = "  <--" if pv < 0.05 else ""
        print(f"    {p:18s} rho {rho:+.3f}  p {pv:.4f}{flag}")

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    fams = ["trivial", "trained-classical", "neural"]
    cols = {"trivial": "#bbbbbb", "trained-classical": "#e67e22", "neural": "#2980b9"}
    for i, p in enumerate(["coverage", "noise_floor", "n_dims"]):
        data = [j[j.family == f][p].values for f in fams if (j.family == f).any()]
        labs = [f for f in fams if (j.family == f).any()]
        bp = ax[i].boxplot(data, labels=labs, patch_artist=True, widths=0.6)
        for patch, f in zip(bp["boxes"], labs):
            patch.set_facecolor(cols[f])
        ax[i].set_ylabel(p)
        ax[i].set_title(f"{p} by winning family", fontsize=9)
        if p != "n_dims":
            ax[i].set_yscale("symlog", linthresh=1e-3)
        ax[i].tick_params(axis="x", rotation=15)
    fig.suptitle("Is architecture choice obtainable from the data?", fontsize=10.5, y=1.05)
    fig.tight_layout(); fig.savefig(FIG / "zoo_3_selection.png", bbox_inches="tight")
    plt.close(fig)


def by_kind(pd_, piv):
    P = piv.clip(lower=-1.0)
    kinds = pd_.groupby("dataset").kind.first()
    t = pd.DataFrame({"best_r2": P.max(axis=1), "winner": P.idxmax(axis=1)}).join(kinds)
    print("\n=== by data kind ===")
    print(t.groupby("kind").agg(n=("best_r2", "size"), median_best_r2=("best_r2", "median"))
          .round(3).sort_values("median_best_r2", ascending=False).to_string())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    pd_ = per_dataset(d)
    print(f"{d.dataset.nunique()} datasets, {d.method.nunique()} methods, "
          f"{len(d)} finite results\n")
    piv = q1_spread(d, pd_)
    corr, m = q2_predict(pd_, piv)
    q3_selection(pd_, piv, m)
    by_kind(pd_, piv)
    piv.to_csv(HERE / "results" / "r2_by_dataset_method.csv")
    corr.to_csv(HERE / "results" / "property_correlations.csv", index=False)
    print(f"\nwrote 3 figures to {FIG}")


if __name__ == "__main__":
    main()
