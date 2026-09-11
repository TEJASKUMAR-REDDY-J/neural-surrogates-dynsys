"""Two slide figures, built for projection: large type, no overlapping text, two panels each.

Deliberately two panels rather than three. A slide is read in about ten seconds, and three
panels forces type down to a size that does not survive a projector.

Every annotation is anchored in axes coordinates against a fixed corner rather than beside a
data point, which is what causes labels to collide when the numbers move.

    python make_slide_figures.py
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
SPIKES = ROOT / "all-spikes"
OUT = ROOT / "RESULTS" / "figures"

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.labelsize": 11,
})
LOCAL_C, GLOBAL_C = "#2f6fb0", "#d1603d"
GOOD_C, MUTED = "#1a8a70", "#6a6a6a"


# ---------------------------------------------------------------------------------------

def fig_locality():
    pr = pd.concat([pd.read_csv(f) for f in glob.glob(
        str(SPIKES / "01-nca-multimodal/results/probes__v[34]__shard*.csv"))])
    hy = pd.concat([pd.read_csv(f) for f in glob.glob(
        str(SPIKES / "01-nca-multimodal/results/hybrid__shard*.csv"))])
    hy = hy.rename(columns={"global": "glob"})

    label = {"A1_local": "local only", "A6_graph": "graph", "A7_recurrent": "recurrent",
             "A3_attentive": "pooled attention", "A8_spectral": "spectral",
             "A5_transformer": "full attention", "A4_interleaved": "interleaved",
             "A2_pooled": "pooled summary"}
    leak = (pr.groupby("arch").lightcone_leak_step1.median()
            .reindex(["A1_local", "A6_graph", "A7_recurrent", "A3_attentive", "A8_spectral",
                      "A5_transformer", "A4_interleaved", "A2_pooled"]).dropna())

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---- panel A: how far does influence travel in ONE step ----------------------------
    FLOOR = 1e-5                                   # a log axis cannot draw a true zero
    y = np.arange(len(leak))
    vals = leak.values.astype(float)
    is_local = vals <= 0
    ax[0].barh(y, np.where(is_local, FLOOR, vals), height=0.62, zorder=3,
               color=[LOCAL_C if l else GLOBAL_C for l in is_local])
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([label[i] for i in leak.index])
    ax[0].invert_yaxis()
    ax[0].set_xscale("log")
    ax[0].set_xlim(FLOOR * 0.7, 4.0)
    ax[0].set_xlabel("share of influence that escapes one cell, after ONE step")
    ax[0].set_title("Locality is a wall, not a tendency", pad=12)

    for yi, v, loc in zip(y, vals, is_local):
        if loc:
            ax[0].text(FLOOR * 1.6, yi, "exactly 0", va="center", ha="left",
                       color=LOCAL_C, fontweight="bold", fontsize=10.5, zorder=4)
        else:
            ax[0].text(v * 1.4, yi, f"{v:.3f}", va="center", ha="left",
                       color=GLOBAL_C, fontsize=10.5, zorder=4)

    # Upper right: the three local bars at the top are microscopic on a log axis, so that
    # corner is empty. Lower right sat on top of the largest bar's value label.
    ax[0].legend(handles=[Patch(facecolor=LOCAL_C, label="purely local"),
                          Patch(facecolor=GLOBAL_C, label="has a global pathway")],
                 loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

    # ---- panel B: does that reach actually pay? ----------------------------------------
    LOCALITY_REAL = {"cml_lattice", "digits_8x8", "gray_scott_16x16",
                     "game_of_life_16x16", "natural_patches_16x16", "multitone_signal",
                     "cml_timewindow", "eca_rule110", "text_chars"}
    h = hy[hy.glob != "none"].copy()
    h["locality_real"] = h.dataset.isin(LOCALITY_REAL)
    piv = (h.pivot_table(index="glob", columns="locality_real", values="global_lift",
                         aggfunc="median")
           .reindex(["spectral", "attention", "pooled", "tokenmix"]).dropna(how="all"))

    x = np.arange(len(piv))
    w = 0.36
    b1 = ax[1].bar(x - w / 2, piv[True], w, color=LOCAL_C, zorder=3,
                   label="neighbours ARE the signal")
    b2 = ax[1].bar(x + w / 2, piv[False], w, color=GLOBAL_C, zorder=3,
                   label="neighbours mean nothing")
    ax[1].axhline(1.0, color="#8c2d18", ls="--", lw=1.8, zorder=2)

    top = float(np.nanmax(piv.values)) * 1.20
    ax[1].set_ylim(0.97, top)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(piv.index)
    ax[1].set_ylabel("error WITHOUT the global pathway\ndivided by error with it")
    ax[1].set_xlabel("kind of global pathway")
    ax[1].set_title("But that reach only pays where locality fails", pad=12)

    for bars in (b1, b2):
        for r in bars:
            hgt = r.get_height()
            if np.isfinite(hgt):
                ax[1].text(r.get_x() + r.get_width() / 2, hgt + (top - 0.97) * 0.02,
                           f"{hgt:.2f}", ha="center", va="bottom", fontsize=9.5)

    # The dashed line's meaning goes into the legend. As free text at the foot of the panel
    # it ran straight through the spectral and attention bars, which sit just above 1.00.
    from matplotlib.lines import Line2D as _L2D
    hh, ll = ax[1].get_legend_handles_labels()
    hh.append(_L2D([0], [0], color="#8c2d18", ls="--", lw=1.8))
    ll.append("1.00  =  switching it off changes nothing")
    ax[1].legend(hh, ll, loc="upper left", fontsize=10, frameon=True, framealpha=0.95)

    fig.suptitle("Local versus global pathways  -  20 combinations, all matched to "
                 "13k parameters", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "slide_locality_vs_global.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote slide_locality_vs_global.png")
    return piv


# ---------------------------------------------------------------------------------------

def fig_stepwise():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(
        str(SPIKES / "00-replications/results/r8/direct_vs_rollout__shard*.csv"))])
    d = d[d.direct_pair_multiplier == 4]           # the fair comparison: 4x training pairs
    med = d.groupby("horizon")[["err_rollout", "err_direct"]].median()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---- panel A: the two curves --------------------------------------------------------
    ax[0].axhspan(1.0, 2.4, color="#dcdcdc", alpha=0.6, zorder=1)
    ax[0].plot(med.index, med.err_rollout, "o-", color=GOOD_C, lw=2.6, ms=7, zorder=4,
               label="64 small steps  (feed the output back in)")
    ax[0].plot(med.index, med.err_direct, "s-", color=GLOBAL_C, lw=2.6, ms=7, zorder=4,
               label="one big jump  (t straight to t+h)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xlim(0.85, 700)
    ax[0].set_ylim(2e-4, 2.4)
    ax[0].set_xlabel("how far ahead the forecast reaches (steps)")
    ax[0].set_ylabel("error   (1.0 = no better than guessing the average)")
    ax[0].set_title("Small steps start 100x better and stay usable twice as long", pad=12)
    ax[0].text(0.985, 0.95, "USELESS ZONE", transform=ax[0].transAxes, ha="right",
               va="top", color="#555555", fontsize=10.5, fontweight="bold")

    # Where does each stop being usable? 0.4 is the threshold used throughout the project.
    # Crossings are interpolated on the log axis and drawn as vertical rules; the labels sit
    # low in the panel so they never touch a curve or a marker.
    #
    # Note on the far right: the jump's curve ends LOWER than stepping's (0.974 against
    # 1.073) but neither is forecasting there. The jump settles onto the average while
    # stepping overshoots past it. That is graceful failure, not a win, and the figure
    # must not imply otherwise - hence the shaded zone and the note beneath it.
    def cross(series, thr=0.4):
        hs_ = np.asarray(series.index, float)
        vs_ = np.asarray(series.values, float)
        for i in range(1, len(vs_)):
            if vs_[i - 1] < thr <= vs_[i]:
                f = ((np.log(thr) - np.log(vs_[i - 1]))
                     / (np.log(vs_[i]) - np.log(vs_[i - 1])))
                return float(np.exp(np.log(hs_[i - 1])
                                    + f * (np.log(hs_[i]) - np.log(hs_[i - 1]))))
        return np.nan

    hj, hs = cross(med.err_direct), cross(med.err_rollout)
    ax[0].axhline(0.4, color=MUTED, ls="--", lw=1.4, zorder=2)

    # The meaning of the dashed line goes INTO the legend rather than floating in the panel.
    # Every placement tried collided with something: on the left it hid under the legend,
    # below the line it ran through the orange curve, on the right it met the curves as they
    # converge. A legend entry cannot overlap anything by construction.
    from matplotlib.lines import Line2D
    handles, labels_ = ax[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=MUTED, ls="--", lw=1.4))
    labels_.append("0.4  =  forecast no longer usable")
    ax[0].legend(handles, labels_, loc="upper left", fontsize=10, frameon=True,
                 framealpha=0.95)

    # The two crossings are close together on a log axis, so their labels are separated
    # HORIZONTALLY - one hanging left of its rule, one right - rather than stacked. Arrows
    # up to the 0.4 line crossed both curves and are gone.
    for hv, col, txt, side in ((hj, GLOBAL_C, "jump stops\n~%d steps" % round(hj), "right"),
                               (hs, GOOD_C, "stepping stops\n~%d steps" % round(hs), "left")):
        if np.isfinite(hv):
            ax[0].axvline(hv, color=col, ls=":", lw=1.7, zorder=2)
            ax[0].text(hv * (0.93 if side == "right" else 1.08), 2.6e-4, txt, color=col,
                       fontsize=9.5, ha=side, va="bottom", zorder=5,
                       bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                 edgecolor="none", alpha=0.9))

    # ---- panel B: per system, how far each stays usable ---------------------------------
    rows = []
    for name, g in d.groupby("system"):
        m = g.groupby("horizon")[["err_rollout", "err_direct"]].median()

        def furthest(series):
            ok = series[series < 0.4]
            return float(ok.index[-1]) if len(ok) else 0.0

        rows.append({"system": name, "steps": furthest(m.err_rollout),
                     "jump": furthest(m.err_direct)})
    t = pd.DataFrame(rows).set_index("system").sort_values("steps")

    y = np.arange(len(t))
    bh = 0.36
    ax[1].barh(y - bh / 2, t.steps, bh, color=GOOD_C, zorder=3, label="64 small steps")
    ax[1].barh(y + bh / 2, t.jump, bh, color=GLOBAL_C, zorder=3, label="one big jump")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(t.index)
    ax[1].set_xlim(0, max(t.steps.max(), t.jump.max()) * 1.28)
    ax[1].set_xlabel("furthest horizon still usable   (error below 0.4)")
    ax[1].set_title("Stepping reaches further on 5 of 6 systems", pad=12)
    ax[1].legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95)

    span = ax[1].get_xlim()[1]
    for yi, (s_, j_) in enumerate(zip(t.steps, t.jump)):
        ax[1].text(s_ + span * 0.013, yi - bh / 2, "%d" % s_, va="center",
                   fontsize=9.5, color=GOOD_C)
        ax[1].text(j_ + span * 0.013, yi + bh / 2, "%d" % j_, va="center",
                   fontsize=9.5, color=GLOBAL_C)

    fig.suptitle("Stepping versus jumping straight to t+h  -  6 systems, 360 runs, the "
                 "jump trained on 4x the examples", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "slide_stepwise_vs_direct.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote slide_stepwise_vs_direct.png")
    print("  usable to h=%.0f (jump) and h=%.0f (stepping)" % (hj, hs))
    print("  past 256 neither forecasts: jump ends at %.3f, stepping at %.3f"
          % (med.err_direct.iloc[-1], med.err_rollout.iloc[-1]))
    return med, t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(fig_locality().round(3).to_string())
    print()
    _, tbl = fig_stepwise()
    print(tbl.to_string())
