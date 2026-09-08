"""Show what the surrogate actually produces, next to the truth.

Every other figure in this battery is a summary statistic. This one shows the thing itself:
the attractor the real system traces, the attractor the trained model traces when left to run
free, and how the two come apart step by step.

Four systems chosen to span the range we measured - from one the model handles well to one
where it fails at the first step.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import systems as S  # noqa: E402
from src.common.baselines import parrot_rollout  # noqa: E402
from src.common.train import fit, rollout  # noqa: E402

OUT = ROOT / "results" / "analysis"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.2,
    "axes.spines.top": False, "axes.spines.right": False,
})

# picked to span what R4c measured: capacity pays off for a long time on Lorenz, and not at
# all on Bouali2 (attractor dimension 16.5)
SYSTEMS = [
    ("Rossler", "weakly chaotic - copying beats the network here"),
    ("Lorenz", "the standard example"),
    ("HyperCai", "strongly chaotic, lambda = 1.68"),
    ("Bouali2", "attractor dimension 16.5 - capacity buys nothing"),
]

PTS_PER_PERIOD = 20
N_TRAIN = 8000
BUDGET = 20_000
FREE_STEPS = 3000


def prepare(name: str):
    spec = S.load_spec(name, pts_per_period=PTS_PER_PERIOD)
    X = S.trajectory(name, n=45_000, pts_per_period=PTS_PER_PERIOD, use_cache=True, timeout_s=300)
    ds = S.one_step_dataset(X, n_train=N_TRAIN, n_test=12_000)
    model, info = fit(ds["X_train"], ds["Y_train"], budget=BUDGET, epochs=200, seed=0)
    test = ds["test_traj_norm"]
    pred = rollout(model, test[:1], min(FREE_STEPS, len(test) - 2))[0]
    truth = test[1 : 1 + len(pred)]
    par = parrot_rollout(ds["X_train"], test[:1], min(FREE_STEPS, len(test) - 2))[0]
    sd = truth.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    err = np.sqrt((((pred - truth) / sd) ** 2).mean(1))
    return spec, truth, pred, par, err, info


def fig_attractors() -> None:
    """Left: the real attractor. Middle: what the model draws on its own. Right: error growth."""
    n = len(SYSTEMS)
    fig, axs = plt.subplots(n, 3, figsize=(10.5, 2.7 * n))
    for r, (name, note) in enumerate(SYSTEMS):
        try:
            spec, truth, pred, par, err, info = prepare(name)
        except Exception as e:  # noqa: BLE001
            axs[r, 0].text(0.5, 0.5, f"{name}: {e}", ha="center", fontsize=7)
            continue

        lim = np.percentile(np.abs(truth[:, :2]), 99.5) * 1.25

        ax = axs[r, 0]
        ax.plot(truth[:, 0], truth[:, 1], lw=0.25, color="#222")
        ax.set_title(f"{name} - the real system\n{note}", fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

        ax = axs[r, 1]
        ok = np.all(np.abs(pred[:, :2]) < 50, axis=1)
        ax.plot(pred[ok, 0], pred[ok, 1], lw=0.25, color="#c0392b")
        blew = (~ok).sum()
        ax.set_title(
            "what the model draws, running free"
            + (f"\n({blew} of {len(pred)} steps flew off the chart)" if blew else
               f"\n{info['n_params']:,} parameters"), fontsize=8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

        ax = axs[r, 2]
        steps = np.arange(1, len(err) + 1)
        ax.plot(steps, err, lw=1.0, color="#c0392b", label="model")
        sd = truth.std(0); sd = np.where(sd > 0, sd, 1.0)
        perr = np.sqrt((((par[: len(truth)] - truth) / sd) ** 2).mean(1))
        ax.plot(steps, perr, lw=0.9, color="#2980b9", ls="--", label="copying")
        ax.axhline(1.0, color="k", lw=0.7, ls=":", label="no better than the average")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-4, 5)
        ax.set_xlabel("steps ahead"); ax.set_ylabel("error")
        secx = ax.secondary_xaxis(
            "top", functions=(lambda s: s * spec.lyap_time_per_step,
                              lambda l: l / max(spec.lyap_time_per_step, 1e-12)))
        secx.set_xlabel("Lyapunov times", fontsize=7)
        if r == 0:
            ax.legend(fontsize=6.5, loc="lower right")
        ax.set_title("how the forecast comes apart", fontsize=8)

    fig.suptitle(
        "What the surrogate sees. Left: truth. Middle: the model left to run on its own for "
        f"{FREE_STEPS} steps. Right: error growth.", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories_attractors.png")
    plt.close(fig)
    print("  wrote trajectories_attractors.png")


def fig_timeseries() -> None:
    """The first coordinate over time: truth against forecast, so the divergence is visible."""
    n = len(SYSTEMS)
    fig, axs = plt.subplots(n, 1, figsize=(10, 1.9 * n), sharex=True)
    for r, (name, note) in enumerate(SYSTEMS):
        try:
            spec, truth, pred, par, err, info = prepare(name)
        except Exception as e:  # noqa: BLE001
            axs[r].text(0.5, 0.5, f"{name}: {e}", ha="center", fontsize=7)
            continue
        m = 600
        t = np.arange(m)
        axs[r].plot(t, truth[:m, 0], lw=1.1, color="#222", label="real system")
        axs[r].plot(t, np.clip(pred[:m, 0], -8, 8), lw=1.0, color="#c0392b",
                    label="model, running free")
        # mark where the forecast stops being useful
        bad = np.flatnonzero(err[:m] > 0.4)
        if len(bad):
            axs[r].axvline(bad[0], color="#c0392b", ls=":", lw=1.0)
            axs[r].text(bad[0], axs[r].get_ylim()[1] * 0.75,
                        f"  useful to here ({bad[0]} steps, "
                        f"{bad[0]*spec.lyap_time_per_step:.1f} Lyapunov times)",
                        fontsize=6.5, color="#c0392b")
        axs[r].set_ylabel(name, fontsize=8)
        if r == 0:
            axs[r].legend(fontsize=7, ncol=2, loc="upper right")
    axs[-1].set_xlabel("steps")
    fig.suptitle("The first coordinate over time: truth in black, the model's free run in red",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories_timeseries.png")
    plt.close(fig)
    print("  wrote trajectories_timeseries.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("trajectory figures (trains one model per system, a few minutes):")
    fig_attractors()
    fig_timeseries()


if __name__ == "__main__":
    main()
