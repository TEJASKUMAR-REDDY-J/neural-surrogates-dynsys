"""Figures for the local x global grid, with error attributed to each path.

The point of the grid is not which combination wins. It is which PART of a combination is
doing the work - and specifically, how often a global path is present, costs parameters, and
contributes nothing.

    global_lift = err(local path only) / err(both)

    1.0  the global path could be deleted with no effect
    >1   silencing it hurts, so it was doing something
    <1   silencing it HELPS, so it was actively harmful

    python -m src.analysis_hybrid
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
LOCALS = ["fixed", "graph", "depthwise", "gru"]
GLOBALS = ["none", "pooled", "attention", "spectral", "tokenmix"]


def load():
    fs = sorted(glob.glob(str(RES / "hybrid__shard*.csv")))
    if not fs:
        raise FileNotFoundError("no hybrid results yet")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.rename(columns={"global": "glob"})


def _grid(d, value, mode="random", frac=0.3, agg="median"):
    s = d[(d.cond_mode == mode) & (d.cond_frac == frac)]
    p = s.pivot_table(index="local", columns="glob", values=value, aggfunc=agg)
    return p.reindex(index=[l for l in LOCALS if l in p.index],
                     columns=[g for g in GLOBALS if g in p.columns])


def heat(ax, P, title, cmap, center=None, fmt="{:.2f}"):
    v = P.values.astype(float)
    if center is not None:
        lim = max(abs(np.nanmin(v) - center), abs(np.nanmax(v) - center))
        im = ax.imshow(v, cmap=cmap, vmin=center - lim, vmax=center + lim)
    else:
        im = ax.imshow(v, cmap=cmap)
    ax.set_xticks(range(P.shape[1])); ax.set_xticklabels(P.columns, rotation=25, ha="right")
    ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(P.index)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    return im


def fig_attribution(d):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    P = _grid(d, "err_both")
    im = heat(ax[0], P, "Error, both paths running\n(lower is better)", "viridis_r",
              fmt="{:.3f}")
    fig.colorbar(im, ax=ax[0], pad=0.02)

    L = _grid(d, "global_lift")
    im = heat(ax[1], L, "global_lift = err(local only) / err(both)\n"
                        "1.00 means the global path does NOTHING", "RdBu_r", center=1.0)
    fig.colorbar(im, ax=ax[1], pad=0.02)

    G = _grid(d, "local_lift")
    im = heat(ax[2], G, "local_lift = err(global only) / err(both)\n"
                        "large means the local path carries it", "Purples")
    fig.colorbar(im, ax=ax[2], pad=0.02)

    fig.suptitle("Twenty combinations, one budget, error attributed to each path",
                 fontsize=10.5, y=1.04)
    fig.tight_layout(); fig.savefig(FIG / "hybrid_1_attribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_by_dataset(d):
    ds = sorted(d.dataset.unique())
    fig, ax = plt.subplots(1, len(ds), figsize=(3.4 * len(ds), 3.5), squeeze=False)
    for i, name in enumerate(ds):
        P = _grid(d[d.dataset == name], "global_lift")
        im = heat(ax[0][i], P, name, "RdBu_r", center=1.0)
        fig.colorbar(im, ax=ax[0][i], pad=0.02)
    fig.suptitle("Does the global path earn its keep? Per dataset. "
                 "Blue = silencing it helps, red = it was needed", fontsize=10.5, y=1.05)
    fig.tight_layout(); fig.savefig(FIG / "hybrid_2_by_dataset.png", bbox_inches="tight")
    plt.close(fig)


def fig_marginals(d):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    s = d[(d.cond_mode == "random") & (d.cond_frac == 0.3)]

    m = s.groupby("local").err_both.median().reindex(LOCALS).dropna()
    ax[0].bar(range(len(m)), m.values, color="#2980b9")
    ax[0].set_xticks(range(len(m))); ax[0].set_xticklabels(m.index, rotation=20)
    ax[0].set_ylabel("error (median over all global blocks)")
    ax[0].set_title("Which LOCAL block?", fontsize=9)

    g = s.groupby("glob").err_both.median().reindex(GLOBALS).dropna()
    ax[1].bar(range(len(g)), g.values, color="#e67e22")
    ax[1].axhline(g.get("none", np.nan), color="#c0392b", ls="--", lw=1.2)
    ax[1].text(0.02, 0.93, "dashed = no global path at all", fontsize=7, color="#c0392b",
               transform=ax[1].transAxes)
    ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels(g.index, rotation=20)
    ax[1].set_ylabel("error (median over all local blocks)")
    ax[1].set_title("Which GLOBAL block?", fontsize=9)

    gl = s[s.glob != "none"].groupby("glob").global_lift.median().dropna()
    ax[2].bar(range(len(gl)), gl.values,
              color=["#16a085" if v > 1.02 else "#95a5a6" for v in gl.values])
    ax[2].axhline(1.0, color="#c0392b", ls="--", lw=1.2)
    ax[2].set_xticks(range(len(gl))); ax[2].set_xticklabels(gl.index, rotation=20)
    ax[2].set_ylabel("global_lift")
    ax[2].set_title("Grey bars are global paths that\ncould be deleted", fontsize=9)

    fig.tight_layout(); fig.savefig(FIG / "hybrid_3_marginals.png", bbox_inches="tight")
    plt.close(fig)


def report(d):
    s = d[(d.cond_mode == "random") & (d.cond_frac == 0.3)]
    print(f"{len(d)} rows, {d.dataset.nunique()} datasets, "
          f"{d.combo.nunique()} combinations\n")
    print("=== error with both paths (median over datasets and seeds) ===")
    print(_grid(d, "err_both").round(3).to_string())
    print("\n=== global_lift: err(local only) / err(both). 1.00 = the global path is inert ===")
    print(_grid(d, "global_lift").round(3).to_string())
    dead = s[(s.glob != "none") & (s.global_lift.between(0.98, 1.02))]
    tot = s[s.glob != "none"]
    print(f"\ninert global paths (lift within 2% of 1.0): "
          f"{len(dead)}/{len(tot)} cells = {100*len(dead)/max(len(tot),1):.0f}%")
    print("\n=== best combination per dataset ===")
    for name, g in s.groupby("dataset"):
        b = g.groupby("combo").err_both.median().sort_values()
        print(f"  {name:24s} {b.index[0]:22s} {b.iloc[0]:.3f}   "
              f"(worst {b.index[-1]}: {b.iloc[-1]:.3f})")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = load()
    report(d)
    fig_attribution(d); fig_by_dataset(d); fig_marginals(d)
    print(f"\nwrote 3 figures to {FIG}")


if __name__ == "__main__":
    main()
