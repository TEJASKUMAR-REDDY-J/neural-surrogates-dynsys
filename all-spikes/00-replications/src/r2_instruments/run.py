"""R2 - how many genuinely different things are we measuring when we measure "complexity"?

The whole project assumes that "learnability" and "chaoticity" are separate axes, which
presupposes that the statistics people use to describe systems are not all the same number
in different clothes. If eight statistics are mutually 0.95-correlated, the system-property
feature space is effectively one-dimensional and any predictor built on it has almost
nothing to work with.

No training here. For each system we compute the published invariants (from dysts) and a
set of model-free statistics from the trajectory, then measure how redundant they are.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import metrics as M  # noqa: E402
from src.common import systems as S  # noqa: E402
from src.common.runlog import RunLog  # noqa: E402

RESULTS = ROOT / "results" / "r2"
LOGS = ROOT / "logs"


def usable_systems(limit: int | None = None, max_gen_s: float = 25.0) -> list[str]:
    """Systems that R0 found integrate cleanly and quickly."""
    path = ROOT / "results" / "r0" / "system_scan.csv"
    if not path.exists():
        raise FileNotFoundError("run R0 first")
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["ok"] == "True"]
    rows = [r for r in rows if float(r["gen_s"]) <= max_gen_s]
    rows.sort(key=lambda r: float(r["gen_s"]))
    names = [r["system"] for r in rows]
    return names[:limit] if limit else names


def instruments_for(name: str, n_points: int, fsle_deltas: np.ndarray) -> dict:
    spec = S.load_spec(name)
    X = S.trajectory(name, n=n_points, use_cache=True)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)

    rec: dict = {"system": name}
    rec.update(spec.invariants)
    rec["lyap_time_per_step"] = spec.lyap_time_per_step

    # Model-free statistics: per channel then averaged, so they do not depend on which
    # component someone happened to record.
    #
    # Both a naive and a sampling-corrected version are recorded. Trajectories are aligned
    # at 100 points per dominant period (Gilpin's protocol), which is heavily oversampled
    # for ordinal statistics: on Lorenz the 0-1 chaos test reads -0.00 at delay 1 and
    # +1.00 at delay 5, on identical data. The corrected version uses each channel's
    # autocorrelation time as the delay, following the oversampling-correction step in
    # Toker et al. (2020). The naive values are kept because the gap between them is a
    # result in its own right: these statistics are not properties of a system until the
    # sampling convention is fixed.
    pes, wpes, ks, ses = [], [], [], []
    pes_n, wpes_n, ks_n, taus = [], [], [], []
    for c in range(Z.shape[1]):
        x = Z[:, c]
        tau = M.autocorrelation_time(x)
        taus.append(tau)
        pes_n.append(M.permutation_entropy(x, order=5, delay=1))
        wpes_n.append(M.permutation_entropy(x, order=5, delay=1, weighted=True))
        ks_n.append(M.zero_one_test_chaos(x, n_c=40, max_n_frac=0.05))
        pes.append(M.permutation_entropy(x, order=5, delay=tau))
        wpes.append(M.permutation_entropy(x, order=5, delay=tau, weighted=True))
        ks.append(M.zero_one_test_chaos(x[::tau], n_c=40, max_n_frac=0.05))
        ses.append(M.spectral_entropy(x))
    rec["autocorr_time"] = float(np.median(taus))
    rec["perm_entropy"] = float(np.mean(pes))
    rec["wpe"] = float(np.mean(wpes))
    rec["wpe_x"] = float(wpes[0])
    rec["k01"] = float(np.median(ks))
    rec["perm_entropy_naive"] = float(np.mean(pes_n))
    rec["wpe_naive"] = float(np.mean(wpes_n))
    rec["k01_naive"] = float(np.median(ks_n))
    rec["spectral_entropy"] = float(np.mean(ses))
    rec["corr_dim_ours"] = M.correlation_dimension(Z, n_sub=2000)

    # FSLE: the scale-resolved growth rate. A plateau structure means an emergent coarse
    # level exists (Cecconi/Falcioni/Vulpiani). We record the curve and two summaries.
    lam = M.fsle_curve(Z, fsle_deltas, n_pairs=120, dt=spec.dt)
    rec["fsle"] = [None if not np.isfinite(v) else round(float(v), 6) for v in lam]
    ok = np.isfinite(lam)
    if ok.sum() >= 2:
        rec["fsle_small"] = float(lam[ok][0])
        rec["fsle_large"] = float(lam[ok][-1])
        rec["fsle_ratio"] = float(lam[ok][0] / lam[ok][-1]) if lam[ok][-1] > 0 else None
        # flatness: how constant is log lambda across scales? near 0 means one regime
        rec["fsle_log_range"] = float(np.ptp(np.log(lam[ok][lam[ok] > 0]))) if (lam[ok] > 0).any() else None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-systems", type=int, default=45)
    ap.add_argument("--n-points", type=int, default=20_000)
    ap.add_argument("--max-gen-s", type=float, default=25.0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    names = usable_systems(args.n_systems, args.max_gen_s)
    deltas = np.logspace(-2.2, -0.2, 9)

    rows = []
    with RunLog("r2_instruments", vars(args), "cli", LOGS) as log:
        log.note("system shortlist", n=len(names), systems=names)
        for i, name in enumerate(names):
            t0 = time.time()
            try:
                rec = instruments_for(name, args.n_points, deltas)
                rec["wall_s"] = round(time.time() - t0, 2)
                rows.append(rec)
                log.result(**rec)
                print(
                    f"[{i+1:3d}/{len(names)}] {name:<26} "
                    f"lam={rec['lyap_max']:6.3f} tau={rec['autocorr_time']:.0f} "
                    f"WPE={rec['wpe']:.3f}(naive {rec['wpe_naive']:.3f}) "
                    f"K={rec['k01']:5.2f}(naive {rec['k01_naive']:5.2f}) "
                    f"Dky={rec['kaplan_yorke_dim']:.2f} Dcorr={rec['corr_dim_ours']:.2f} "
                    f"({rec['wall_s']}s)",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                log.failure(system=name, error=f"{type(e).__name__}: {e}"[:200])
                print(f"[{i+1:3d}/{len(names)}] {name:<26} FAILED {e}", flush=True)

    scalar_keys = [
        k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k != "wall_s"
    ]
    with (RESULTS / "instruments.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["system"] + scalar_keys + ["wall_s"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    import json

    (RESULTS / "fsle_curves.json").write_text(
        json.dumps({r["system"]: r.get("fsle") for r in rows}, indent=1)
        , encoding="utf-8"
    )
    (RESULTS / "fsle_deltas.json").write_text(json.dumps(deltas.tolist()), encoding="utf-8")
    print(f"\nwrote {len(rows)} systems to {RESULTS/'instruments.csv'}")


if __name__ == "__main__":
    main()
