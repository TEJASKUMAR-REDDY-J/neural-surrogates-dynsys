"""Figures for the mechanism probes: what each automaton is doing, not how well it scored.

Four questions, one figure each:

    1. how fast does information travel?          propagation radius against step
    2. does the global branch get used at all?    its share of each update
    3. does silencing it hurt?                    knockout ratio
    4. does the rule rely on what it carried?     memory ratio

and one that ties the spike back to the rest of the project: does the share of work the
global branch does depend on the kind of data, in the direction predicted?

    python -m src.analysis_probes
"""

from __future__ import annotations

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

ARCH_C = {
    "A1_local": "#2980b9",
    "A2_pooled": "#e67e22",
    "A3_attentive": "#8e44ad",
    "A4_interleaved": "#16a085",
}


def load() -> pd.DataFrame:
    files = sorted(RES.glob("probes__*.csv"))
    if not files:
        raise FileNotFoundError("no probes__*.csv yet - run src.run_smoke first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def _curve(s: str) -> np.ndarray:
    return np.array([float(v) for v in str(s).split(";") if v not in ("", "nan")])


# ---------------------------------------------------------------------------------------

def fig_propagation(df):
    """The structural claim, measured: a local rule moves information one cell per step."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))

    d = df[df.dataset == df.dataset.iloc[0]] if "cml_lattice" not in set(df.dataset) \
        else df[df.dataset == "cml_lattice"]
    for arch, sub in d.groupby("arch"):
        curves = [_curve(c) for c in sub.radius_curve.dropna()]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        m = np.median(np.stack([c[:n] for c in curves]), axis=0)
        ax[0].plot(np.arange(1, n + 1), m, "o-", ms=3, color=ARCH_C.get(arch),
                   label=arch, lw=1.6)
    k = np.arange(1, 17)
    ax[0].plot(k, k, "k--", lw=1, alpha=0.6, label="one cell per step")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("how far the ripple reached (cells)")
    ax[0].set_title("Poke one cell. How far does it get?\n(64-cell lattice)", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    piv = df.pivot_table(index="dataset", columns="arch", values="frac_affected_step1",
                         aggfunc="median")
    archs = [a for a in ARCH_C if a in piv]
    x = np.arange(len(piv.index))
    w = 0.8 / max(len(archs), 1)
    for i, a in enumerate(archs):
        ax[1].bar(x + i * w, piv[a], w, color=ARCH_C[a], label=a)
    ax[1].set_xticks(x + 0.4)
    ax[1].set_xticklabels(piv.index, rotation=35, ha="right", fontsize=6.5)
    ax[1].set_ylabel("fraction of lattice reached\nafter ONE step")
    ax[1].set_title("One step of influence", fontsize=9)
    ax[1].axhline(1.0, color="#888", ls=":", lw=1)

    fig.suptitle("Information travel: the limit no amount of capacity removes",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "probe_1_propagation.png", bbox_inches="tight")
    plt.close(fig)


def fig_global_use(df):
    """Is the global branch used, and where? A branch that is never used still costs."""
    g = df[df.has_global.astype(str).str.lower() == "true"]
    if g.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6))

    piv = g.pivot_table(index="dataset", columns="arch", values="global_share_mean",
                        aggfunc="median").sort_values(
        by=list(g.arch.unique())[0], ascending=False)
    archs = [a for a in ARCH_C if a in piv]
    x = np.arange(len(piv.index))
    w = 0.8 / max(len(archs), 1)
    for i, a in enumerate(archs):
        ax[0].bar(x + i * w, piv[a], w, color=ARCH_C[a], label=a)
    ax[0].set_xticks(x + 0.4)
    ax[0].set_xticklabels(piv.index, rotation=35, ha="right", fontsize=6.5)
    ax[0].set_ylabel("share of each update\nsupplied by the global branch")
    ax[0].set_title("How much work does the global branch do?", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    # ordered vs unordered: the pre-registered prediction
    for i, a in enumerate(archs):
        sub = g[g.arch == a]
        for j, (lab, val) in enumerate([("neighbours mean something", True),
                                        ("neighbours mean nothing", False)]):
            s = sub[sub.ordered.astype(str).str.lower() == str(val).lower()]
            if len(s):
                ax[1].bar(j + i * w, s.global_share_mean.median(), w, color=ARCH_C[a])
    ax[1].set_xticks([0.4, 1.4])
    ax[1].set_xticklabels(["neighbours mean\nsomething", "neighbours mean\nnothing"],
                          fontsize=8)
    ax[1].set_ylabel("global share")
    ax[1].set_title("The prediction: the global branch should\nwork harder when locality is "
                    "meaningless", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "probe_2_global_use.png", bbox_inches="tight")
    plt.close(fig)


def fig_knockout_memory(df):
    """Silencing the global branch, and wiping what the rule remembered."""
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.6))

    for k, (col, title, note) in enumerate([
        ("knockout_ratio", "Silence the global branch",
         "> 1 means the branch was helping"),
        ("memory_ratio", "Wipe the scratch channels halfway",
         "> 1 means the rule was carrying something"),
    ]):
        d = df.dropna(subset=[col])
        if col == "knockout_ratio":
            d = d[d.has_global.astype(str).str.lower() == "true"]
        if d.empty:
            continue
        piv = d.pivot_table(index="dataset", columns="arch", values=col, aggfunc="median")
        archs = [a for a in ARCH_C if a in piv]
        x = np.arange(len(piv.index))
        w = 0.8 / max(len(archs), 1)
        for i, a in enumerate(archs):
            ax[k].bar(x + i * w, piv[a], w, color=ARCH_C[a], label=a)
        ax[k].axhline(1.0, color="#c0392b", ls="--", lw=1.2)
        ax[k].set_xticks(x + 0.4)
        ax[k].set_xticklabels(piv.index, rotation=35, ha="right", fontsize=6.5)
        ax[k].set_ylabel(f"{col.replace('_', ' ')}")
        ax[k].set_title(f"{title}\n{note}", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "probe_3_knockout_memory.png", bbox_inches="tight")
    plt.close(fig)


def table(df) -> pd.DataFrame:
    """One row per architecture: the mechanism summary."""
    t = df.groupby("arch").agg(
        n_params=("n_params", "median"),
        reach_step1=("frac_affected_step1", "median"),
        steps_to_cover=("steps_to_cover", "median"),
        global_share=("global_share_mean", "median"),
        knockout_ratio=("knockout_ratio", "median"),
        memory_ratio=("memory_ratio", "median"),
    ).round(4)
    return t


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df)} probe rows, {df.dataset.nunique()} datasets, "
          f"{df.arch.nunique()} architectures\n")
    fig_propagation(df)
    fig_global_use(df)
    fig_knockout_memory(df)
    t = table(df)
    print(t.to_string())
    t.to_csv(RES / "probe_summary.csv")
    print(f"\nwrote 3 figures to {FIG} and {RES / 'probe_summary.csv'}")


if __name__ == "__main__":
    main()
