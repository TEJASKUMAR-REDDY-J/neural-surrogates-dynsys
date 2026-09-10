"""Can data properties tell you WHICH model to use? Built and validated as a selector.

This is the actual question: given a dataset you have never seen, and only properties you
can measure without training anything, which family of model should you reach for?

Correlations are not an answer to that. A selector is. So this fits one, and scores it the
only way that matters - on datasets it was not fitted on, against the alternatives a
practitioner actually has.

The metric is REGRET, not accuracy. Accuracy punishes recommending a model that was very
nearly as good as the winner, which is not a real cost. Regret is what you actually lose:

    regret = R2(the best method available) - R2(the method the selector told you to use)

Regret of 0 means the selector matched the oracle. It is scored against four alternatives:

    oracle          always picks the true best. The unreachable ceiling.
    always-X        pick one fixed method for everything. What most people do.
    majority        pick whichever family wins most often overall.
    random          pick uniformly at random. The floor.

Validation is leave-one-out over datasets, and then the much harsher leave-one-DOMAIN-out,
where every dataset of a kind is held out together. The second is the real test: if the
selector only works because it saw four other chaotic systems, it has learned the zoo
rather than a property of data.

    python selector.py
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
PROPS = ["coverage", "noise_floor", "dist_shift", "decorr_time", "spectral_entropy",
         "n_dims", "n_rows"]
METHODS = ["mean", "persistence", "ridge", "knn", "trees", "mlp", "cnn", "gru"]
FAMILY = {"mean": "trivial", "persistence": "trivial", "ridge": "classical",
          "knn": "classical", "trees": "classical",
          "mlp": "neural", "cnn": "neural", "gru": "neural"}


def load():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / "results/zoo__shard*.csv"))],
                  ignore_index=True)
    d = d[np.isfinite(d.r2)].copy()
    d["r2c"] = d.r2.clip(lower=-1.0)
    R = d.groupby(["dataset", "method"]).r2c.median().unstack()
    R = R[[m for m in METHODS if m in R.columns]]
    X = d.groupby("dataset")[PROPS].first()
    kind = d.groupby("dataset").kind.first()
    keep = R.dropna().index.intersection(X.dropna().index)
    return X.loc[keep], R.loc[keep], kind.loc[keep]


def family_r2(R: pd.DataFrame) -> pd.DataFrame:
    """Best R2 achievable within each family."""
    out = {}
    for fam in ("trivial", "classical", "neural"):
        cols = [m for m in R.columns if FAMILY[m] == fam]
        out[fam] = R[cols].max(axis=1)
    return pd.DataFrame(out)


def evaluate(X, F, groups=None, seed=0):
    """Leave-one-out (or leave-one-group-out) regret for a selector over FAMILIES."""
    y = F.idxmax(axis=1)
    oracle = F.max(axis=1)
    idx = list(X.index)
    picks = {}

    splits = ([(g, [i for i in idx if groups[i] == g]) for g in sorted(set(groups))]
              if groups is not None else [(i, [i]) for i in idx])

    for _, test_ids in splits:
        train_ids = [i for i in idx if i not in test_ids]
        if len(set(y.loc[train_ids])) < 2:
            for t in test_ids:
                picks[t] = y.loc[train_ids].mode().iloc[0]
            continue
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, class_weight="balanced")
        clf.fit(X.loc[train_ids], y.loc[train_ids])
        for t, p in zip(test_ids, clf.predict(X.loc[test_ids])):
            picks[t] = p

    picks = pd.Series(picks).loc[idx]
    got = pd.Series([F.loc[i, picks[i]] for i in idx], index=idx)
    regret = oracle - got

    rows = [{"strategy": "oracle (unreachable)", "regret": 0.0, "acc": 1.0}]
    rows.append({"strategy": "SELECTOR (data properties)",
                 "regret": float(regret.mean()),
                 "acc": float((picks == y).mean())})
    maj = y.mode().iloc[0]
    rows.append({"strategy": f"always {maj} (majority)",
                 "regret": float((oracle - F[maj]).mean()),
                 "acc": float((y == maj).mean())})
    for fam in ("trivial", "classical", "neural"):
        if fam != maj:
            rows.append({"strategy": f"always {fam}",
                         "regret": float((oracle - F[fam]).mean()),
                         "acc": float((y == fam).mean())})
    rng = np.random.default_rng(seed)
    rr = [float((oracle - pd.Series(
        [F.loc[i, rng.choice(F.columns)] for i in idx], index=idx)).mean())
        for _ in range(200)]
    rows.append({"strategy": "random pick", "regret": float(np.mean(rr)),
                 "acc": 1 / 3})
    return pd.DataFrame(rows), picks, y, regret


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    X, R, kind = load()
    F = family_r2(R)
    print(f"{len(X)} datasets, {R.shape[1]} methods, 3 families\n")
    print("which family actually wins:")
    print(F.idxmax(axis=1).value_counts().to_string(), "\n")

    print("=== leave-one-DATASET-out ===")
    tab, picks, y, regret = evaluate(X, F)
    print(tab.sort_values("regret").to_string(index=False))

    print("\n=== leave-one-DOMAIN-out (every dataset of a kind held out together) ===")
    tab2, picks2, y2, regret2 = evaluate(X, F, groups=kind)
    print(tab2.sort_values("regret").to_string(index=False))

    print("\n=== what the selector is actually using ===")
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0,
                               class_weight="balanced").fit(X, F.idxmax(axis=1))
    print(export_text(t, feature_names=list(X.columns), max_depth=3))
    rf = RandomForestClassifier(n_estimators=400, random_state=0,
                                class_weight="balanced").fit(X, F.idxmax(axis=1))
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("feature importance:")
    print(imp.round(3).to_string())

    # ---- figure ------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    t1 = tab.set_index("strategy").regret.sort_values()
    cols = ["#16a085" if "SELECTOR" in s else
            ("#cccccc" if "oracle" in s else "#c0392b") for s in t1.index]
    ax[0].barh(range(len(t1)), t1.values, color=cols)
    ax[0].set_yticks(range(len(t1))); ax[0].set_yticklabels(t1.index, fontsize=7.5)
    ax[0].set_xlabel("mean regret  (R2 lost against the best available model)")
    ax[0].set_title("Leave-one-dataset-out", fontsize=9.5)
    ax[0].invert_yaxis()

    t2 = tab2.set_index("strategy").regret.sort_values()
    cols2 = ["#16a085" if "SELECTOR" in s else
             ("#cccccc" if "oracle" in s else "#c0392b") for s in t2.index]
    ax[1].barh(range(len(t2)), t2.values, color=cols2)
    ax[1].set_yticks(range(len(t2))); ax[1].set_yticklabels(t2.index, fontsize=7.5)
    ax[1].set_xlabel("mean regret")
    ax[1].set_title("Leave-one-DOMAIN-out (the harsh test)", fontsize=9.5)
    ax[1].invert_yaxis()

    fam_c = {"trivial": "#bbbbbb", "classical": "#e67e22", "neural": "#2980b9"}
    for fam in ("trivial", "classical", "neural"):
        s = X[F.idxmax(axis=1) == fam]
        ax[2].scatter(s.decorr_time, s.noise_floor, s=42, c=fam_c[fam], label=fam,
                      edgecolor="white", linewidth=0.5)
    ax[2].set_xscale("log"); ax[2].set_yscale("symlog", linthresh=1e-2)
    ax[2].set_xlabel("decorrelation time (steps)")
    ax[2].set_ylabel("estimated noise floor")
    ax[2].set_title("The two properties that separate the regimes", fontsize=9.5)
    ax[2].legend(fontsize=7.5, frameon=False)

    fig.suptitle("Choosing the model family from data properties alone",
                 fontsize=11, y=1.04)
    fig.tight_layout(); fig.savefig(FIG / "zoo_4_selector.png", bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame({"true_best_family": F.idxmax(axis=1), "selector_pick": picks,
                        "pick_domain_holdout": picks2, "regret": regret,
                        "regret_domain": regret2, "kind": kind})
    out.to_csv(HERE / "results" / "selector_per_dataset.csv")
    print(f"\nwrote {FIG / 'zoo_4_selector.png'} and selector_per_dataset.csv")


if __name__ == "__main__":
    main()
