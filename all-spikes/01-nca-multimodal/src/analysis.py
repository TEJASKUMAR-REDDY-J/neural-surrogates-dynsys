"""Turn the smoke-test rows into numbers and pictures.

Separate from the experiment on purpose: this reads CSVs and never trains anything, so any
figure can be regenerated without touching a model.

    python -m src.analysis
"""

from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
FIG = RES / "figures"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCH_C = {"A1_local": "#c0392b", "A2_pooled": "#2980b9", "A3_attentive": "#27ae60",
          "A4_interleaved": "#8e44ad"}
BASE_C = "#7f8c8d"
STD = dict(cond_mode="random", cond_frac=0.3, cond_noise=0.0)     # the training condition


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(RES / "smoke__*.csv")))
    if not files:
        raise FileNotFoundError("no smoke__*.csv - run src.run_smoke first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # for learned methods keep the best iteration count per (fit, condition): reporting a
    # model at an arbitrary iteration count would be scoring the schedule, not the rule
    learned = df[df.kind == "learned"]
    idx = learned.groupby(
        ["dataset", "method", "seed", "cond_mode", "cond_frac", "cond_noise"]
    )["nrmse_hidden"].idxmin()
    return pd.concat([df[df.kind == "baseline"], learned.loc[idx]], ignore_index=True)


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m]


# ---------------------------------------------------------------------------------------

def fig_main(df):
    """Every method on every dataset, at the standard condition."""
    d = sel(df, **STD)
    piv = d.pivot_table(index="dataset", columns="method", values="nrmse_hidden", aggfunc="median")
    order = [c for c in ("identity", "global_mean", "local_mean", "nearest_neighbour",
                         "pca_impute", "A1_local", "A2_pooled", "A3_attentive") if c in piv]
    piv = piv[order]
    dsorder = sorted(piv.index, key=lambda s: piv.loc[s].min())
    piv = piv.loc[dsorder]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=min(1.5, np.nanmax(piv.values)),
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        best = np.nanargmin(piv.values[i])
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        weight="bold" if j == best else "normal",
                        color="white" if v > 0.9 else "black")
    ax.axvline(4.5, color="black", lw=1.5)
    ax.text(2.0, -0.85, "no training", ha="center", fontsize=8, style="italic")
    ax.text(6.0, -0.85, "the three NCAs", ha="center", fontsize=8, style="italic")
    ax.set_title("Error on the hidden cells (lower is better; 1.0 = guessing the mean)\n"
                 "bold = best on that row, 30% of cells hidden at random", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="nRMSE on hidden cells", pad=0.02)
    fig.tight_layout(); fig.savefig(FIG / "01_main_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return piv


def fig_local_vs_global(df):
    """The experiment's actual question: when does the global branch pay?"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

    # (a) random holes vs one contiguous block
    for k, (mode, frac, title) in enumerate([("random", 0.3, "scattered holes"),
                                             ("block", 0.3, "one contiguous block")]):
        d = sel(df, cond_mode=mode, cond_frac=frac, cond_noise=0.0)
        piv = d.pivot_table(index="dataset", columns="method", values="nrmse_hidden",
                            aggfunc="median")
        archs = [a for a in ARCH_C if a in piv]
        x = np.arange(len(piv.index))
        w = 0.8 / len(archs)
        for i, a in enumerate(archs):
            ax[k].bar(x + i * w, piv[a], w, label=a, color=ARCH_C[a])
        if "nearest_neighbour" in piv:
            ax[k].plot(x + 0.4, piv["nearest_neighbour"], "k_", ms=14, mew=2,
                       label="best no-training" if k == 0 else None)
        ax[k].set_xticks(x + 0.4)
        ax[k].set_xticklabels(piv.index, rotation=35, ha="right", fontsize=7)
        ax[k].set_ylabel("nRMSE on hidden cells")
        ax[k].set_title(title, fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Does a global branch help? Only when the neighbours are gone too",
                 fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "02_local_vs_global.png", bbox_inches="tight")
    plt.close(fig)


def fig_ordered(df):
    """Data whose neighbours mean nothing - the control."""
    d = sel(df, **STD)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for grp, lbl in [(True, "neighbours meaningful"), (False, "neighbours meaningless")]:
        sub = d[d.ordered == grp]
        piv = sub.pivot_table(index="dataset", columns="method", values="nrmse_hidden",
                              aggfunc="median")
        archs = [a for a in ARCH_C if a in piv]
        vals = [piv[a].median() for a in archs]
        ax.plot(archs, vals, "o-", lw=2, label=f"{lbl} (n={len(piv)})",
                ls="-" if grp else "--")
    ax.set_ylabel("median nRMSE on hidden cells")
    ax.set_title("The local-only automaton is the one that\nneeds its neighbours to mean something",
                 fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "03_ordered_vs_not.png", bbox_inches="tight")
    plt.close(fig)


def fig_sweeps(df):
    """How error moves with how much is hidden, and with measurement noise."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    d = sel(df, cond_mode="random", cond_noise=0.0)
    for a in [x for x in ARCH_C if x in set(d.method)]:
        g = d[d.method == a].groupby("cond_frac").nrmse_hidden.median()
        ax[0].plot(g.index, g.values, "o-", color=ARCH_C[a], label=a)
    for b in ("nearest_neighbour", "local_mean", "pca_impute"):
        g = d[d.method == b].groupby("cond_frac").nrmse_hidden.median()
        if len(g):
            ax[0].plot(g.index, g.values, "--", color=BASE_C, alpha=0.8, lw=1)
            ax[0].annotate(b, (g.index[-1], g.values[-1]), fontsize=6.5, color=BASE_C,
                           xytext=(3, 0), textcoords="offset points", va="center")
    ax[0].set_xlabel("fraction of cells hidden"); ax[0].set_ylabel("nRMSE on hidden cells")
    ax[0].set_title("More hidden, harder", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    d = sel(df, cond_mode="random", cond_frac=0.3)
    for a in [x for x in ARCH_C if x in set(d.method)]:
        g = d[d.method == a].groupby("cond_noise").nrmse_hidden.median()
        ax[1].plot(g.index, g.values, "o-", color=ARCH_C[a], label=a)
    for b in ("nearest_neighbour", "pca_impute"):
        g = d[d.method == b].groupby("cond_noise").nrmse_hidden.median()
        if len(g):
            ax[1].plot(g.index, g.values, "--", color=BASE_C, alpha=0.8, lw=1)
            ax[1].annotate(b, (g.index[-1], g.values[-1]), fontsize=6.5, color=BASE_C,
                           xytext=(3, 0), textcoords="offset points", va="center")
    ax[1].set_xlabel("measurement noise on visible cells")
    ax[1].set_ylabel("nRMSE on hidden cells")
    ax[1].set_title("Noise: where a learned rule should pull ahead", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "04_sweeps.png", bbox_inches="tight")
    plt.close(fig)


def fig_stability(raw):
    """Run the rule longer than it was trained for. Does it hold, or fall apart?"""
    d = raw[(raw.kind == "learned") & (raw.cond_mode == "random")
            & (raw.cond_frac == 0.3) & (raw.cond_noise == 0.0)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a in [x for x in ARCH_C if x in set(d.method)]:
        g = d[d.method == a].groupby("steps").nrmse_hidden.median()
        ax[0].plot(g.index, g.values, "o-", color=ARCH_C[a], label=a)
    ax[0].axvspan(4, 10, color="gold", alpha=0.22)
    ax[0].annotate("trained in here", (7, ax[0].get_ylim()[1]), ha="center", va="top",
                   fontsize=7.5, color="#8a6d00")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("iterations of the rule"); ax[0].set_ylabel("nRMSE on hidden cells")
    ax[0].set_title("Stability past the training range", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    over = d.groupby(["method", "dataset"]).overrun_ratio.median().reset_index()
    archs = [x for x in ARCH_C if x in set(over.method)]
    ax[1].boxplot([over[over.method == a].overrun_ratio.values for a in archs],
                  labels=archs, showfliers=False)
    ax[1].axhline(1.0, color="black", lw=1, ls="--")
    ax[1].set_ylabel("error at 32 steps / error at best")
    ax[1].set_title("1.0 = no damage from over-iterating", fontsize=9)
    ax[1].tick_params(axis="x", labelrotation=12)
    fig.tight_layout(); fig.savefig(FIG / "05_stability.png", bbox_inches="tight")
    plt.close(fig)


def fig_correlations(df):
    """Which methods behave alike? A method that mirrors lookup is doing lookup."""
    piv = df.pivot_table(index=["dataset", "cond_mode", "cond_frac", "cond_noise", "seed"],
                         columns="method", values="nrmse_hidden")
    piv = piv.dropna(axis=1, how="all").dropna()
    if piv.shape[1] < 2 or len(piv) < 5:
        return None
    rho = spearmanr(piv.values).statistic
    rho = np.atleast_2d(rho)
    cols = list(piv.columns)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(rho[i, j]) > 0.6 else "black")
    ax.set_title(f"How alike do the methods behave?\n"
                 f"Spearman across {len(piv)} dataset x condition x seed cells", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02)
    fig.tight_layout(); fig.savefig(FIG / "06_method_correlations.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rho, index=cols, columns=cols)


def fig_cost(df):
    """What each method costs, against what it buys."""
    d = sel(df, **STD)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for meth, sub in d.groupby("method"):
        c = ARCH_C.get(meth, BASE_C)
        ax[0].scatter(max(sub.n_params.median(), 0.7), sub.nrmse_hidden.median(),
                      s=48, color=c, zorder=3)
        ax[0].annotate(meth, (max(sub.n_params.median(), 0.7), sub.nrmse_hidden.median()),
                       fontsize=6.5, xytext=(5, 3), textcoords="offset points")
        ax[1].scatter(max(sub.infer_wall_s.median(), 1e-4), sub.nrmse_hidden.median(),
                      s=48, color=c, zorder=3)
        ax[1].annotate(meth, (max(sub.infer_wall_s.median(), 1e-4), sub.nrmse_hidden.median()),
                       fontsize=6.5, xytext=(5, 3), textcoords="offset points")
    ax[0].set_xscale("log"); ax[0].set_xlabel("parameters (0-parameter methods at left)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("inference seconds per condition")
    for a in ax:
        a.set_ylabel("median nRMSE on hidden cells")
    ax[0].set_title("Error against size", fontsize=9)
    ax[1].set_title("Error against inference cost", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "07_cost.png", bbox_inches="tight")
    plt.close(fig)


def fig_optimizers():
    import glob as _g
    fs = sorted(_g.glob(str(RES / "optimizer_sweep*.csv")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = d.groupby("optimizer").agg(nrmse=("nrmse_hidden", "median"),
                                   loss=("train_loss", "median"),
                                   wall=("train_wall_s", "median")).sort_values("nrmse")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax[0].bar(g.index, g.nrmse, color="#2980b9"); ax[0].set_ylabel("nRMSE on hidden cells")
    ax[0].set_title("Which optimiser, measured", fontsize=9)
    ax[1].bar(g.index, g.wall, color="#7f8c8d"); ax[1].set_ylabel("training seconds")
    ax[1].set_title("What it costs", fontsize=9)
    for a in ax:
        a.tick_params(axis="x", labelrotation=20)
    fig.tight_layout(); fig.savefig(FIG / "08_optimizers.png", bbox_inches="tight")
    plt.close(fig)
    return g


def fig_metrics(df):
    """Do the metrics agree? If they rank methods differently, saying "the error" is wrong.

    Six ways of measuring the same predictions. MSE punishes a few large misses, MAE does
    not; nRMSE divides by the data's own spread so datasets are comparable; relative L2 is
    the norm ratio the neural-operator papers report. They are not interchangeable, and the
    only way to know whether that matters here is to rank the methods by each and compare.
    """
    cols = [c for c in ("mse_hidden", "mae_hidden", "nrmse_hidden", "rel_l2_hidden",
                        "mse_all", "rel_l2_all") if c in df.columns]
    if len(cols) < 2:
        return None
    d = sel(df, cond_mode="random", cond_frac=0.3, cond_noise=0.0)
    d = d[(d.kind == "baseline") | (d.steps == d.best_steps)]
    if d.empty:
        return None

    piv = d.pivot_table(index="method", columns=[], values=cols, aggfunc="median")
    ranks = piv.rank()

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.8))
    methods = list(piv.index)
    x = np.arange(len(methods))
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        v = piv[c] / piv[c].max()                      # each metric on its own 0-1 scale
        ax[0].bar(x + i * w, v, w, label=c)
    ax[0].set_xticks(x + 0.4)
    ax[0].set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("value, scaled to its own worst case")
    ax[0].set_title("Six metrics, same predictions", fontsize=9)
    ax[0].legend(fontsize=6.5, frameon=False, ncol=2)

    im = ax[1].imshow(ranks.T.values, cmap="RdYlGn_r", aspect="auto")
    ax[1].set_xticks(x); ax[1].set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
    ax[1].set_yticks(range(len(cols))); ax[1].set_yticklabels(cols, fontsize=7)
    for i in range(len(methods)):
        for j in range(len(cols)):
            ax[1].text(i, j, int(ranks.T.values[j, i]), ha="center", va="center", fontsize=6.5)
    ax[1].grid(False)
    spread = int((ranks.max(axis=1) - ranks.min(axis=1)).max())
    ax[1].set_title(f"Rank by each metric (1 = best)\nworst disagreement: "
                    f"{spread} places", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "07_metric_agreement.png", bbox_inches="tight")
    plt.close(fig)
    return ranks


def fig_horizon(df):
    """Error against rollout horizon, and how long each automaton stays usable.

    An automaton that is excellent at step 4 and garbage at step 32 has not learned a rule,
    it has learned a schedule. R9 in 00-replications found exactly that failure in 24 of 24
    cases, so this is the figure that says whether the fix - randomising the iteration count
    during training - actually worked.
    """
    d = sel(df, kind="learned", cond_mode="random", cond_frac=0.3, cond_noise=0.0)
    if d.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6))

    for arch, sub_ in d.groupby("arch"):
        g = sub_.groupby("steps").nrmse_hidden.median()
        ax[0].plot(g.index, g.values, "o-", ms=3.5, color=ARCH_C.get(arch), label=arch)
    ax[0].axhline(1.0, color="#c0392b", ls=":", lw=1)
    ax[0].text(1.05, 1.02, "no better than guessing the mean", fontsize=6.5, color="#c0392b")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("iterations (trained with 4-10)")
    ax[0].set_ylabel("nRMSE on hidden cells")
    ax[0].set_title("Error against rollout horizon", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].axvspan(4, 10, color="#888", alpha=0.10)

    if "vpt_steps" in d.columns:
        piv = d.pivot_table(index="dataset", columns="arch", values="vpt_steps",
                            aggfunc="median")
        archs = [a for a in ARCH_C if a in piv]
        x = np.arange(len(piv.index))
        w = 0.8 / max(len(archs), 1)
        for i, a in enumerate(archs):
            ax[1].bar(x + i * w, piv[a], w, color=ARCH_C[a], label=a)
        ax[1].set_xticks(x + 0.4)
        ax[1].set_xticklabels(piv.index, rotation=35, ha="right", fontsize=6.5)
        ax[1].set_ylabel("valid prediction time (iterations)")
        ax[1].set_title("How long it stays usable\n(nRMSE under 0.4, the 00-replications "
                        "threshold)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "08_horizon_and_vpt.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "smoke__*.csv")))],
                    ignore_index=True)
    df = load()

    piv = fig_main(df)
    fig_local_vs_global(df)
    fig_ordered(df)
    fig_sweeps(df)
    fig_stability(raw)
    corr = fig_correlations(df)
    fig_cost(df)
    fig_metrics(df)
    fig_horizon(df)
    opt = fig_optimizers()

    std = sel(df, **STD)
    best_base = (std[std.kind == "baseline"].groupby("dataset").nrmse_hidden.min())
    summary = {
        "n_rows": int(len(raw)),
        "datasets": sorted(df.dataset.unique().tolist()),
        "median_nrmse_by_method": df[df.cond_mode == "random"].groupby("method")
            .nrmse_hidden.median().round(4).to_dict(),
        "standard_condition_table": piv.round(4).to_dict(),
        "beats_every_baseline": {
            a: int(sum(
                std[(std.method == a) & (std.dataset == d)].nrmse_hidden.median() < best_base[d]
                for d in best_base.index))
            for a in ARCH_C if a in set(std.method)
        },
        "n_datasets": int(len(best_base)),
        "block_vs_random": {
            a: {
                "random": float(sel(df, cond_mode="random", cond_frac=0.3, cond_noise=0.0)
                                .query("method == @a").nrmse_hidden.median()),
                "block": float(sel(df, cond_mode="block", cond_frac=0.3, cond_noise=0.0)
                               .query("method == @a").nrmse_hidden.median()),
            } for a in list(ARCH_C) + ["nearest_neighbour", "local_mean", "pca_impute"]
            if a in set(df.method)
        },
        "overrun_ratio_median": raw[raw.kind == "learned"].groupby("method")
            .overrun_ratio.median().round(4).to_dict(),
        "ordered_vs_unordered": {
            a: {
                "ordered": float(std[(std.method == a) & (std.ordered)].nrmse_hidden.median()),
                "unordered": float(std[(std.method == a) & (~std.ordered)].nrmse_hidden.median()),
            } for a in ARCH_C if a in set(std.method)
        },
        "method_correlations": None if corr is None else corr.round(3).to_dict(),
        "optimizer_sweep": None if opt is None else opt.round(5).to_dict(),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("standard_condition_table", "method_correlations")}, indent=2))
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
