"""Figure for N6: coverage density explains the copying tie.

Three panels, left to right:
  1. where this benchmark lives on the coverage axis - almost all of it in the dense corner
  2. the correlation itself, with the tie line marked
  3. the decomposition, which is the part that carries the mechanism: sparse coverage
     destroys copying and leaves the trained model roughly untouched
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "analysis"
SRC = ROOT / "results" / "n6" / "coverage_vs_advantage.csv"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_GOOD, C_BAD, C_MID = "#2980b9", "#c0392b", "#7f8c8d"


def main() -> None:
    m = pd.read_csv(SRC)
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # --- 1. where the benchmark sits -------------------------------------------------
    ax[0].hist(m.coverage, bins=np.logspace(np.log10(m.coverage.min()),
                                            np.log10(m.coverage.max()), 24),
               color=C_MID, alpha=0.85)
    ax[0].axvline(m.coverage.median(), color=C_BAD, lw=1.4)
    ax[0].annotate(f"median\n{m.coverage.median():.3f}", (m.coverage.median(), 0.94),
                   xycoords=("data", "axes fraction"), color=C_BAD, fontsize=7.5,
                   ha="left", va="top", xytext=(4, 0), textcoords="offset points")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("coverage: nearest-neighbour distance / attractor spread")
    ax[0].set_ylabel("systems")
    ax[0].set_title("Where this benchmark lives\n(left = every query has a near-twin)",
                    fontsize=9)

    # --- 2. the correlation ------------------------------------------------------------
    ratio = m.model_vpt / m.parrot_vpt
    sc = ax[1].scatter(m.coverage, ratio, c=m.dim, cmap="viridis", s=26,
                       edgecolor="white", linewidth=0.4, zorder=3)
    ax[1].axhline(1.0, color=C_BAD, lw=1.2, ls="--", zorder=2)
    ax[1].annotate("tie", (m.coverage.min(), 1.0), color=C_BAD, fontsize=7.5,
                   va="bottom", xytext=(0, 3), textcoords="offset points")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("coverage (sparser $\\rightarrow$)")
    ax[1].set_ylabel("model VPT / copying VPT")
    ax[1].set_title("Sparser coverage, better network\n"
                    r"$\rho=+0.56$, n=107, shuffle 95th pct 0.19", fontsize=9)
    fig.colorbar(sc, ax=ax[1], label="state dimension", pad=0.02)

    # --- 3. the decomposition ----------------------------------------------------------
    m = m.copy()
    m["band"] = pd.qcut(m.coverage, 4, labels=["densest", "dense", "sparse", "sparsest"])
    q = m.groupby("band", observed=True).agg(copy=("parrot_vpt", "median"),
                                             model=("model_vpt", "median"))
    x = np.arange(len(q))
    ax[2].plot(x, q["copy"], "o-", color=C_BAD, lw=1.8, label="copying (no training)")
    ax[2].plot(x, q["model"], "s-", color=C_GOOD, lw=1.8, label="trained network")
    ax[2].axhline(200, color=C_MID, lw=1, ls=":")
    ax[2].annotate("ruler ends here (200 steps)", (0, 200), color=C_MID, fontsize=7,
                   va="bottom", xytext=(2, 3), textcoords="offset points")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(q.index, fontsize=8)
    ax[2].set_ylabel("forecast horizon (steps)")
    ax[2].set_xlabel("coverage quartile")
    ax[2].set_title("Coverage destroys copying,\nnot the network", fontsize=9)
    ax[2].legend(fontsize=7.5, frameon=False)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "n6_coverage.png", bbox_inches="tight")
    print("wrote", OUT / "n6_coverage.png")


if __name__ == "__main__":
    main()
