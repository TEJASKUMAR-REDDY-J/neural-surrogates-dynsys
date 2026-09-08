"""Block diagrams of the four automata, drawn from the code rather than from memory.

Every box label is filled in from the actual module - channel counts, widths and parameter
counts are read off a built model, so a diagram cannot drift away from what runs. If the
architecture changes and the diagram is not regenerated, the numbers on it will be wrong in
an obvious way rather than a quiet one.

    python -m src.diagrams
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import models as M  # noqa: E402

OUT = ROOT / "results" / "figures"

plt.rcParams.update({"figure.dpi": 150, "font.size": 8})

# one colour per role, used identically in all four diagrams
C = {
    "state": "#e8e8e8",
    "perceive": "#cfe3f7",
    "mlp": "#e2d5f0",
    "head": "#fbe0c3",
    "global": "#f7cfd6",
    "out": "#d3ecd6",
}


def box(ax, x, y, w, h, text, kind, fontsize=7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=C[kind], edgecolor="#5a5a5a", linewidth=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, linespacing=1.35)
    return (x + w, y + h / 2), (x, y + h / 2)


def arrow(ax, a, b, style="-|>", rad=0.0, color="#4a4a4a"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=9, linewidth=0.9,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=1, shrinkB=1))


def frame(title, subtitle, w=12.4, h=2.5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left", pad=14, fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=7.5,
            color="#555", va="bottom")
    return fig, ax


def facts(arch: str, spatial=(64,), budget=13_000):
    """Read the real numbers off a built model so the labels cannot go stale."""
    sc = M.scale_for_budget(arch, spatial, 1, budget)
    m = M.build(arch, spatial, 1, scale=sc)
    return m, sc, M.widths(sc)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------------------

def diagram_a1():
    m, sc, w = facts("A1_local")
    fig, ax = frame(
        "A1  local  —  a cell sees its neighbours and nothing else",
        f"{m.n_params:,} parameters · width scale {sc} · the classic neural cellular "
        f"automaton · one step moves information exactly one cell")
    y, h = 0.32, 0.36
    r, _ = box(ax, 0.01, y, 0.10, h, f"state\n{m.c_state} channels\non the lattice", "state")
    r2, l2 = box(ax, 0.15, y, 0.14, h,
                 f"perceive\nfixed 3x3 filters\nidentity, grad, laplacian\n"
                 f"circular padding", "perceive")
    arrow(ax, r, l2)
    r3, l3 = box(ax, 0.33, y, 0.28, h,
                 f"per-cell MLP  (1x1 convolutions, shared by every cell)\n"
                 f"{m.c_state * M._perception_mult(1)} -> " + " -> ".join(str(v) for v in w),
                 "mlp")
    arrow(ax, r2, l3)
    r4, l4 = box(ax, 0.65, y, 0.13, h, f"update head\n{w[-1]} -> {m.c_state}\nzero-initialised",
                 "head")
    arrow(ax, r3, l4)
    r5, l5 = box(ax, 0.82, y, 0.07, h, "fire\nmask\n50%", "head")
    arrow(ax, r4, l5)
    r6, l6 = box(ax, 0.92, y, 0.07, h, "+", "out")
    arrow(ax, r5, l6)
    ax.text(0.995, y + h / 2, "next\nstate", fontsize=7, va="center")
    # residual: the state is added back, so an untrained rule is exactly the identity
    arrow(ax, (0.06, y), (0.955, y - 0.14), rad=-0.06, color="#888")
    arrow(ax, (0.955, y - 0.14), (0.955, y), color="#888")
    ax.text(0.5, y - 0.185, "residual — the head starts at zero, so an untrained automaton "
                            "is exactly the identity and has to earn every change",
            fontsize=6.8, ha="center", color="#666")
    ax.text(0.5, 0.03, "iterated 4–10 times during training (the count is random each batch), "
                       "evaluated out to 32", fontsize=7, ha="center", style="italic",
            color="#444")
    save(fig, "arch_A1_local.png")


def diagram_a2():
    m, sc, w = facts("A2_pooled")
    fig, ax = frame(
        "A2  pooled global  —  one summary of the whole lattice, broadcast to everybody",
        f"{m.n_params:,} parameters · width scale {sc} · the global branch runs in PARALLEL "
        f"and every cell receives the same context vector")
    y, h = 0.42, 0.30
    yg = 0.06
    r, _ = box(ax, 0.01, y, 0.09, h, f"state\n{m.c_state} ch", "state")
    r2, l2 = box(ax, 0.14, y, 0.13, h, "perceive\nfixed 3x3\ncircular", "perceive")
    arrow(ax, r, l2)
    # global branch
    rg, lg = box(ax, 0.14, yg, 0.13, h, "pool whole lattice\nmean · max · spread", "global")
    arrow(ax, (0.055, y), (0.14, yg + h / 2), rad=-0.15)
    rg2, lg2 = box(ax, 0.31, yg, 0.15, h,
                   f"context net\n{m.c_state * 3} -> {m.global_net[0].out_features} "
                   f"-> {m.c_ctx}", "global")
    arrow(ax, rg, lg2)
    rg3, lg3 = box(ax, 0.50, yg, 0.13, h, f"broadcast\nsame {m.c_ctx} numbers\nto every cell",
                   "global")
    arrow(ax, rg2, lg3)
    rc, lc = box(ax, 0.50, y, 0.13, h, "concatenate\nneighbourhood\n+ context", "perceive")
    arrow(ax, r2, lc)
    arrow(ax, rg3, (0.565, y), rad=0.0)
    r3, l3 = box(ax, 0.67, y, 0.16, h,
                 "per-cell MLP\n" + "-".join(str(v) for v in w), "mlp")
    arrow(ax, rc, l3)
    r4, l4 = box(ax, 0.87, y, 0.11, h, "head + residual\nzero-init", "head")
    arrow(ax, r3, l4)
    ax.text(0.5, 0.005, "global_gain multiplies the broadcast context. Setting it to 0 at "
                        "evaluation silences the branch without touching a weight — that is "
                        "the knockout probe.", fontsize=7, ha="center", style="italic",
            color="#444")
    save(fig, "arch_A2_pooled.png")


def diagram_a3():
    m, sc, w = facts("A3_attentive")
    fig, ax = frame(
        "A3  attentive global  —  different cells hear different parts of the summary",
        f"{m.n_params:,} parameters · width scale {sc} · parallel branch again, but the "
        f"context is position-dependent · attention capped at {M.MAX_TOKENS} tokens so cost "
        f"never grows with the lattice")
    y, h = 0.42, 0.30
    yg = 0.06
    r, _ = box(ax, 0.01, y, 0.09, h, f"state\n{m.c_state} ch", "state")
    r2, l2 = box(ax, 0.14, y, 0.12, h, "perceive\nfixed 3x3\ncircular", "perceive")
    arrow(ax, r, l2)
    rg, lg = box(ax, 0.13, yg, 0.13, h, f"squeeze to\n{m.tgrid} tokens\n(adaptive pool)",
                 "global")
    arrow(ax, (0.055, y), (0.13, yg + h / 2), rad=-0.15)
    rg2, lg2 = box(ax, 0.30, yg, 0.15, h,
                   f"self-attention\n{m.c_ctx} ch, 4 heads\ntokens talk to tokens", "global")
    arrow(ax, rg, lg2)
    rg3, lg3 = box(ax, 0.49, yg, 0.14, h, "scatter back\neach cell reads\nits own token",
                   "global")
    arrow(ax, rg2, lg3)
    rc, lc = box(ax, 0.49, y, 0.14, h, "concatenate\nneighbourhood\n+ own context",
                 "perceive")
    arrow(ax, r2, lc)
    arrow(ax, rg3, (0.56, y))
    r3, l3 = box(ax, 0.67, y, 0.16, h, "per-cell MLP\n" + "-".join(str(v) for v in w), "mlp")
    arrow(ax, rc, l3)
    r4, l4 = box(ax, 0.87, y, 0.11, h, "head + residual\nzero-init", "head")
    arrow(ax, r3, l4)
    ax.text(0.5, 0.005, "A2 gives every cell the same global sentence; A3 lets each cell hear "
                        "the part of it that covers its own region.",
            fontsize=7, ha="center", style="italic", color="#444")
    save(fig, "arch_A3_attentive.png")


def diagram_a4():
    m, sc, w = facts("A4_interleaved")
    fig, ax = frame(
        "A4  interleaved  —  local, global, local, global, in series",
        f"{m.n_params:,} parameters · width scale {sc} · the global step sits BETWEEN local "
        f"stages rather than beside them, the arrangement a diffusion UNet uses")
    y, h = 0.34, 0.34
    r, _ = box(ax, 0.005, y, 0.075, h, f"state\n{m.c_state} ch", "state")
    r2, l2 = box(ax, 0.10, y, 0.10, h, "perceive\nfixed 3x3\ncircular", "perceive")
    arrow(ax, r, l2)
    r3, l3 = box(ax, 0.22, y, 0.12, h, f"local block 1\n1x1: {w[0]} -> {w[1]}", "mlp")
    arrow(ax, r2, l3)
    r4, l4 = box(ax, 0.36, y, 0.13, h,
                 f"GLOBAL MIX 1\npool -> {m.tokens} tokens\nlearned token mixing\nzero-init out",
                 "global")
    arrow(ax, r3, l4)
    r5, l5 = box(ax, 0.51, y, 0.13, h,
                 f"local block 2\ndepthwise 3x3\n+ 1x1 -> {w[2]}\n(one more hop)", "mlp")
    arrow(ax, r4, l5)
    r6, l6 = box(ax, 0.66, y, 0.12, h, "GLOBAL MIX 2\nsame, again", "global")
    arrow(ax, r5, l6)
    r7, l7 = box(ax, 0.80, y, 0.10, h, f"local block 3\n{w[3]} -> {w[4]}", "mlp")
    arrow(ax, r6, l7)
    r8, l8 = box(ax, 0.92, y, 0.075, h, "head\n+ residual", "head")
    arrow(ax, r7, l8)
    for cx in (0.425, 0.72):
        ax.text(cx, y + h + 0.04, "+", fontsize=11, ha="center", color="#b03050")
    ax.text(0.5, 0.10, "each global mix is added as a residual, so the local path still works "
                       "if the global path contributes nothing —\nwhich is exactly what the "
                       "contribution probe measures, per step",
            fontsize=7, ha="center", style="italic", color="#444")
    save(fig, "arch_A4_interleaved.png")


def diagram_propagation():
    """Why the global branch exists at all, in one picture."""
    fig, ax = frame(
        "Why any of this matters  —  a purely local rule moves information one cell per step",
        "the structural limit no amount of capacity removes", w=9.6, h=2.6)
    ax.text(0.02, 0.72, "local only", fontsize=8.5, fontweight="bold", color="#2b6cb0")
    ax.text(0.02, 0.30, "with a global branch", fontsize=8.5, fontweight="bold",
            color="#b03050")
    for i, k in enumerate([1, 2, 4, 8]):
        x = 0.27 + i * 0.175
        box(ax, x, 0.60, 0.145, 0.26, f"step {k}\nreach: {k} cells", "perceive", fontsize=7)
        box(ax, x, 0.18, 0.145, 0.26, f"step {k}\nreach: whole lattice", "global", fontsize=7)
    ax.text(0.5, 0.04,
            "On a 100x100 grid a local rule needs ~100 steps before one corner can influence "
            "the opposite one.\nThat is arithmetic, not training. probes.propagation measures "
            "it directly for every architecture.",
            fontsize=7.5, ha="center", color="#333")
    save(fig, "arch_00_why_global.png")


def main():
    diagram_propagation()
    diagram_a1()
    diagram_a2()
    diagram_a3()
    diagram_a4()


if __name__ == "__main__":
    main()
