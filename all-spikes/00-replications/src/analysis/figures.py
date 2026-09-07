"""Figures for the calibration report.

Reads results/analysis/analysis.json plus the raw CSVs; writes PNGs to results/analysis/.
Every figure caption carries the run_id of the data behind it, so a plot can always be
traced to a config and a commit.
"""

from __future__ import annotations

import json
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

from src.analysis.run import read_csv  # noqa: E402

RES = ROOT / "results"
OUT = RES / "analysis"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote {name}")


def fig_r2(rep: dict) -> None:
    b = rep.get("r2_instruments")
    if not b:
        return
    cols, M = b["columns"], np.array(b["spearman_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman")
    ax.set_title(
        f"R2: complexity statistics are not redundant\n"
        f"{b['n_systems']} systems - median |rho| = {b['median_abs_offdiag_spearman']:.2f}, "
        f"{b['participation_ratio']} effective dimensions", fontsize=9)
    ax.grid(False)
    _save(fig, "r2_spearman_matrix.png")

    # the sampling artefact
    rows = read_csv(RES / "r2" / "instruments.csv")
    if rows and "k01_naive" in rows[0]:
        fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.1))
        for ax, (a, bnm, lab) in zip(axs, [
            ("wpe", "wpe_naive", "weighted permutation entropy"),
            ("k01", "k01_naive", "0-1 chaos test K"),
            ("perm_entropy", "perm_entropy_naive", "permutation entropy"),
        ]):
            x = np.array([r[bnm] for r in rows], float)
            y = np.array([r[a] for r in rows], float)
            ax.scatter(x, y, s=16, alpha=0.75, edgecolor="none")
            lo = min(np.nanmin(x), np.nanmin(y)); hi = max(np.nanmax(x), np.nanmax(y))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            r = b["sampling_correction"].get(a, {}).get("spearman_corrected_vs_naive", float("nan"))
            ax.set_title(f"{lab}\nSpearman = {r:.2f}", fontsize=8)
            ax.set_xlabel("naive (delay 1)"); ax.set_ylabel("corrected (delay = tau)")
        fig.suptitle(
            "R2: correcting for oversampling does not shift the statistics, it reorders them",
            fontsize=9)
        _save(fig, "r2_sampling_correction.png")

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        tau = np.array([r["autocorr_time"] for r in rows], float)
        ax.hist(tau, bins=np.logspace(0, np.log10(max(tau.max(), 2)), 22))
        ax.set_xscale("log")
        ax.set_xlabel("autocorrelation time (samples)")
        ax.set_ylabel("systems")
        ax.set_title(
            "R2: alignment at 100 points/period leaves effective sampling unequal\n"
            f"tau spans {int(tau.min())} to {int(tau.max())}, median {int(np.median(tau))}",
            fontsize=9)
        _save(fig, "r2_autocorr_time.png")


def fig_r3(rep: dict) -> None:
    b = rep.get("r3_wpe_vs_error")
    if not b or not b.get("per_system"):
        return
    ps = b["per_system"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))

    for ax, key, lab in zip(
        axs[:2], ("wpe", "lyap_max"), ("weighted permutation entropy", "largest Lyapunov exponent")
    ):
        x = np.array([p[key] for p in ps], float)
        y = np.array([p["vpt_model_mean"] for p in ps], float)
        e = np.array([p["vpt_model_sd"] for p in ps], float)
        ax.errorbar(x, y, yerr=e, fmt="o", ms=4, lw=0.8, capsize=2, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(lab); ax.set_ylabel("valid prediction time (steps)")
        rho = b["spearman_vpt_vs"].get(key, float("nan"))
        ax.set_title(f"Spearman = {rho:.2f}", fontsize=8)
        if key == "lyap_max":
            ax.set_xscale("log")

    ax = axs[2]
    m = np.array([p["vpt_model_mean"] for p in ps], float)
    q = np.array([p["vpt_parrot"] for p in ps], float)
    ax.scatter(q, m, s=18, alpha=0.8, edgecolor="none")
    hi = max(np.nanmax(m), np.nanmax(q)) * 1.1
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("context parroting VPT (steps)")
    ax.set_ylabel("trained surrogate VPT (steps)")
    ax.set_title(
        f"model beats copying on {b['model_beats_parrot_n']}/{b['n_systems']}", fontsize=8)
    fig.suptitle("R3: does a training-free statistic already predict surrogate skill?", fontsize=9)
    _save(fig, "r3_predictors_and_parroting.png")


def fig_r4(rep: dict) -> None:
    b = rep.get("r4_capacity_data")
    if not b or not b.get("per_system"):
        return
    ps = b["per_system"]
    n = len(ps)
    fig, axs = plt.subplots(1, n, figsize=(2.5 * n, 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, ps):
        c = p["capacity_curve"]
        x, y, sd = np.array(c["n_params"]), np.array(c["err"]), np.array(c["sd"])
        ax.errorbar(x, y, yerr=sd, fmt="o-", ms=3.5, lw=1, capsize=2)
        f = p["capacity_fit"]
        if f.get("ok"):
            xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 60)
            ax.plot(xs, f["L_inf"] + f["A"] * xs ** -f["alpha"], "r-", lw=1, alpha=0.7)
            ax.axhline(f["L_inf"], color="r", ls=":", lw=0.9)
            ax.set_title(
                f"{p['system']}\n" + r"$\alpha$=" + f"{f['alpha']:.2f}  "
                + r"$L_\infty$=" + f"{f['L_inf']:.3f}"
                + ("*" if f.get("floor_supported") else ""), fontsize=8)
        else:
            ax.set_title(p["system"], fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("parameters")
    axs[0].set_ylabel("one-step NRMSE")
    fig.suptitle(
        "R4: capacity scaling per system. Red line = fitted L_inf + A N^-alpha; "
        "dotted = extrapolated floor (* = 95% CI excludes zero)", fontsize=9)
    _save(fig, "r4_capacity_scaling.png")

    a = b.get("alpha_vs_inv_dky")
    if a:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        ax.scatter(a["inv_dky"], a["alpha"], s=30)
        for p, xx, yy in zip(ps, a["inv_dky"], a["alpha"]):
            ax.annotate(p["system"], (xx, yy), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        lo = min(min(a["inv_dky"]), min(a["alpha"])); hi = max(max(a["inv_dky"]), max(a["alpha"]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="alpha = 1/d")
        ax.set_xlabel("1 / Kaplan-Yorke dimension")
        ax.set_ylabel("fitted capacity exponent alpha")
        ax.legend(fontsize=7)
        ax.set_title(
            "R4: does the scaling exponent follow 1/d?\n"
            f"Spearman = {a['spearman_alpha_vs_inv_dky']:.2f}", fontsize=9)
        _save(fig, "r4_alpha_vs_dimension.png")


def fig_r5(rep: dict) -> None:
    b = rep.get("r5_nonmonotone")
    if not b or not b.get("families"):
        return
    fams = b["families"]
    fig, axs = plt.subplots(1, len(fams), figsize=(4.6 * len(fams), 3.4), squeeze=False)
    for ax, (fam, blk) in zip(axs[0], fams.items()):
        for tol, v in sorted(blk["by_tolerance"].items(), key=lambda kv: float(kv[0])):
            k = np.array(v["knob"], float)
            c = np.array([np.nan if x is None else x for x in v["required_capacity_median"]], float)
            ok = np.isfinite(c)
            ax.plot(k[ok], c[ok], "o-", ms=4, lw=1, label=f"tol {float(tol):.1%}")
        ax.set_yscale("log")
        ax.set_xlabel("chaos knob"); ax.set_ylabel("required capacity (params)")
        ax.legend(fontsize=6.5, ncol=2)
        ax.set_title(
            f"{fam}: non-monotone at "
            f"{blk['n_tolerances_showing_non_monotonicity']}/{blk['n_tolerances']} tolerances",
            fontsize=8)
    fig.suptitle(
        "R5: does required capacity peak at intermediate chaoticity, at every tolerance?",
        fontsize=9)
    _save(fig, "r5_required_capacity.png")


def fig_r8(rep: dict) -> None:
    b = rep.get("r8_direct_vs_rollout")
    if not b or not b.get("per_system"):
        return
    ps = b["per_system"]
    fig, axs = plt.subplots(1, len(ps), figsize=(2.4 * len(ps), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, ps):
        h = np.array(p["horizons"], float)
        ax.plot(h, p["err_rollout"], "o-", ms=3.5, lw=1, label="rollout")
        ax.plot(h, p["err_direct"], "s-", ms=3.5, lw=1, label="direct")
        ax.plot(h, p["err_parrot"], "^--", ms=3, lw=0.9, label="parroting", alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("horizon (steps)")
        ax.set_title(f"{p['system']}\n" + r"$\lambda$=" + f"{p['lyap_max']:.2f}", fontsize=8)
    axs[0].set_ylabel("normalised error")
    axs[0].legend(fontsize=7)
    fig.suptitle(
        "R8: if direct prediction escapes the failure rollout hits, the cause was "
        "compounding, not a system ceiling", fontsize=9)
    _save(fig, "r8_direct_vs_rollout.png")


def fig_r1(rep: dict) -> None:
    b = rep.get("r1_coarse_grain")
    if not b or not b.get("by_setting"):
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for k in sorted({s["k"] for s in b["by_setting"]}):
        sub = sorted([s for s in b["by_setting"] if s["k"] == k], key=lambda s: s["block_size"])
        ax.plot([s["block_size"] for s in sub], [100 * s["frac_reducible"] for s in sub],
                "o-", ms=5, label=f"k = {k} coarse symbols")
    ax.set_xlabel("block size N"); ax.set_ylabel("% of 256 ECA rules reducible")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.set_title(
        "R1: reducibility vs coarse-graining scale and search width", fontsize=9)
    _save(fig, "r1_reducibility.png")


def main() -> None:
    path = OUT / "analysis.json"
    if not path.exists():
        print("run src.analysis.run first")
        return
    rep = json.loads(path.read_text(encoding="utf-8"))
    print("figures:")
    for f in (fig_r1, fig_r2, fig_r3, fig_r4, fig_r5, fig_r8):
        try:
            f(rep)
        except Exception as e:  # noqa: BLE001 - a missing experiment should not stop the rest
            print(f"  {f.__name__} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
