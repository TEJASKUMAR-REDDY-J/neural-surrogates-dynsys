"""Analysis for the replication battery: turns the raw result CSVs into the numbers that
answer each experiment's question, plus figures.

Deliberately separate from the experiment scripts so analysis can be rerun and corrected
without retraining anything. Reads only from results/, writes to results/analysis/.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.scaling import fit_scaling  # noqa: E402

RES = ROOT / "results"
OUT = RES / "analysis"


def read_csv(path: Path) -> list[dict]:
    """Read a results CSV, transparently merging any shard files beside it.

    Experiments can be split across worker processes; each shard writes
    <stem>__shardN.csv. Analysis should not care how the work was divided.
    """
    paths = [path] if path.exists() else []
    paths += sorted(path.parent.glob(path.stem + "__shard*.csv"))
    if not paths:
        return []
    rows = []
    for r in [x for pth in paths for x in csv.DictReader(pth.open(encoding="utf-8"))]:
        out = {}
        for k, v in r.items():
            if v is None or v == "":
                out[k] = None
                continue
            if v in ("True", "False"):
                out[k] = v == "True"
                continue
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        rows.append(out)
    return rows


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def ols_r2(Xcols: list[np.ndarray], y: np.ndarray) -> float:
    """R^2 of an ordinary least squares fit with an intercept."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(y)] + [np.asarray(c, float) for c in Xcols])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 2:
        return float("nan")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --------------------------------------------------------------------------------------


def analyse_r1() -> dict:
    rows = read_csv(RES / "r1" / "eca_coarse_grain.csv")
    if not rows:
        return {}
    out: dict = {"by_setting": [], "union_by_k": {}}
    settings = sorted({(r["block_size"], r["k"]) for r in rows})
    per_rule_any: dict[float, set] = {}
    for N, k in settings:
        sub = [r for r in rows if r["block_size"] == N and r["k"] == k]
        red = [r for r in sub if r["reducible"]]
        irr = sorted(int(r["rule"]) for r in sub if not r["reducible"])
        out["by_setting"].append(
            {
                "block_size": int(N), "k": int(k),
                "n_rules": len(sub),
                "n_projections_tried": int(sub[0]["n_projections_tried"]),
                "exhaustive": bool(sub[0]["exhaustive"]),
                "n_reducible": len(red),
                "frac_reducible": len(red) / len(sub),
                "irreducible": irr,
            }
        )
        per_rule_any.setdefault(k, set()).update(int(r["rule"]) for r in red)

    all_rules = {int(r["rule"]) for r in rows}
    for k, red in per_rule_any.items():
        out["union_by_k"][int(k)] = {
            "n_reducible_at_some_block_size": len(red),
            "irreducible_everywhere": sorted(all_rules - red),
        }

    # the measurement the papers do not make: does widening the search change the count?
    sens = []
    for N in sorted({s[0] for s in settings}):
        ks = sorted({s[1] for s in settings if s[0] == N})
        if len(ks) < 2:
            continue
        fr = {
            int(k): next(
                b["frac_reducible"] for b in out["by_setting"]
                if b["block_size"] == N and b["k"] == k
            )
            for k in ks
        }
        sens.append({"block_size": int(N), "frac_by_k": fr,
                     "delta": max(fr.values()) - min(fr.values())})
    out["search_width_sensitivity"] = sens
    return out


def analyse_r2() -> dict:
    rows = read_csv(RES / "r2" / "instruments.csv")
    if not rows:
        return {}
    cols = [
        "lyap_max", "kaplan_yorke_dim", "correlation_dim_pub", "multiscale_entropy",
        "perm_entropy", "wpe", "k01", "spectral_entropy", "corr_dim_ours", "dim",
        "autocorr_time",
    ]
    cols = [c for c in cols if c in rows[0]]
    M = np.array([[r.get(c, np.nan) for c in cols] for r in rows], float)
    keep = np.isfinite(M).all(1)
    M = M[keep]

    corr = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            corr[i, j] = spearman(M[:, i], M[:, j])

    Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1.0)
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    ev = ev / ev.sum()
    cum = np.cumsum(ev)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    # participation ratio: a continuous "effective number of dimensions"
    part = float((ev.sum() ** 2) / (ev**2).sum())

    off = corr[~np.eye(len(cols), dtype=bool)]
    naive = {}
    for a, b in (("wpe", "wpe_naive"), ("k01", "k01_naive"), ("perm_entropy", "perm_entropy_naive")):
        if b in rows[0]:
            x = np.array([r[a] for r in rows], float)
            y = np.array([r[b] for r in rows], float)
            naive[a] = {
                "spearman_corrected_vs_naive": spearman(x, y),
                "mean_corrected": float(np.nanmean(x)),
                "mean_naive": float(np.nanmean(y)),
            }
    return {
        "sampling_correction": naive,
        "n_systems": int(keep.sum()),
        "columns": cols,
        "spearman_matrix": corr.round(3).tolist(),
        "median_abs_offdiag_spearman": float(np.median(np.abs(off))),
        "max_abs_offdiag_spearman": float(np.max(np.abs(off))),
        "explained_variance_ratio": ev.round(4).tolist(),
        "n_components_for_90pct": n90,
        "participation_ratio": round(part, 2),
    }


def analyse_r3() -> dict:
    rows = read_csv(RES / "r3" / "surrogate_vs_statistics.csv")
    inst = {r["system"]: r for r in read_csv(RES / "r2" / "instruments.csv")}
    if not rows or not inst:
        return {}

    systems = sorted({r["system"] for r in rows})
    per_sys = []
    for s in systems:
        sub = [r for r in rows if r["system"] == s]
        i = inst.get(s)
        if not i:
            continue
        per_sys.append(
            {
                "system": s,
                "vpt_model_mean": float(np.mean([r["vpt_steps"] for r in sub])),
                "vpt_model_sd": float(np.std([r["vpt_steps"] for r in sub])),
                "vpt_parrot": float(sub[0]["parrot_vpt_steps"]),
                "vpt_persistence": float(sub[0]["persistence_vpt_steps"]),
                "spectrum_error": float(np.mean([r["spectrum_error"] for r in sub])),
                "density_kl": float(np.mean([r["density_kl"] for r in sub])),
                "diverged_frac": float(np.mean([bool(r["diverged"]) for r in sub])),
                "wpe": i["wpe"], "lyap_max": i["lyap_max"], "k01": i["k01"],
                "perm_entropy": i["perm_entropy"],
                "spectral_entropy": i["spectral_entropy"],
                "kaplan_yorke_dim": i["kaplan_yorke_dim"],
                "corr_dim_ours": i["corr_dim_ours"],
            }
        )

    y = np.array([np.log(max(p["vpt_model_mean"], 1e-3)) for p in per_sys])
    g = lambda k: np.array([p[k] for p in per_sys], float)  # noqa: E731

    r2 = {
        "wpe_only": ols_r2([g("wpe")], y),
        "lyap_only": ols_r2([g("lyap_max")], y),
        "wpe_plus_lyap": ols_r2([g("wpe"), g("lyap_max")], y),
        "k01_only": ols_r2([g("k01")], y),
        "full_feature_set": ols_r2(
            [g("wpe"), g("lyap_max"), g("k01"), g("spectral_entropy"),
             g("kaplan_yorke_dim"), g("corr_dim_ours")], y
        ),
    }
    beats = [p for p in per_sys if p["vpt_model_mean"] > p["vpt_parrot"]]
    return {
        "n_systems": len(per_sys),
        "per_system": per_sys,
        "spearman_vpt_vs": {
            k: spearman(g(k), np.array([p["vpt_model_mean"] for p in per_sys]))
            for k in ("wpe", "lyap_max", "k01", "spectral_entropy",
                      "kaplan_yorke_dim", "perm_entropy")
        },
        "r2_log_vpt": {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in r2.items()},
        "kill_criterion_wpe_plus_lyap_ge_0.85": bool(
            np.isfinite(r2["wpe_plus_lyap"]) and r2["wpe_plus_lyap"] >= 0.85
        ),
        "model_beats_parrot_n": len(beats),
        "model_beats_parrot_frac": len(beats) / max(len(per_sys), 1),
        "median_vpt_ratio_model_over_parrot": float(
            np.median([p["vpt_model_mean"] / max(p["vpt_parrot"], 1e-9) for p in per_sys])
        ),
    }


def analyse_r4() -> dict:
    rows = read_csv(RES / "r4" / "capacity_data_sweep.csv")
    if not rows:
        return {}
    out: dict = {"per_system": []}
    for s in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == s]
        n_max = max(r["n_train"] for r in sub)
        p_max = max(r["n_params"] for r in sub)

        # capacity scaling at the largest data budget, and vice versa
        cap = [r for r in sub if r["n_train"] == n_max]
        dat = [r for r in sub if r["n_params"] == p_max]

        def agg(rs, key):
            xs = sorted({r[key] for r in rs})
            ys = [
                float(np.mean([r["nrmse_1"] for r in rs if r[key] == x and np.isfinite(r["nrmse_1"])]))
                for x in xs
            ]
            sd = [
                float(np.std([r["nrmse_1"] for r in rs if r[key] == x and np.isfinite(r["nrmse_1"])]))
                for x in xs
            ]
            return np.array(xs, float), np.array(ys, float), np.array(sd, float)

        pc, pe, psd = agg(cap, "n_params")
        dc, de, dsd = agg(dat, "n_train")
        fit_p = fit_scaling(pc, pe) if len(pc) >= 4 else {"ok": False}
        fit_d = fit_scaling(dc, de) if len(dc) >= 4 else {"ok": False}

        rec = {
            "system": s,
            "lyap_max": sub[0]["lyap_max"],
            "kaplan_yorke_dim": sub[0]["kaplan_yorke_dim"],
            "corr_dim_pub": sub[0]["corr_dim_pub"],
            "capacity_curve": {"n_params": pc.tolist(), "err": pe.tolist(), "sd": psd.tolist()},
            "data_curve": {"n_train": dc.tolist(), "err": de.tolist(), "sd": dsd.tolist()},
            "capacity_fit": fit_p,
            "data_fit": fit_d,
            "seed_sd_at_largest": float(psd[-1]) if len(psd) else float("nan"),
            "best_vpt_steps": float(np.max([r["vpt_steps"] for r in sub])),
            "diverged_frac": float(np.mean([bool(r["diverged"]) for r in sub])),
        }
        out["per_system"].append(rec)

    ok = [p for p in out["per_system"] if p["capacity_fit"].get("ok")]
    if ok:
        floors = np.array([p["capacity_fit"]["L_inf"] for p in ok])
        seedsd = np.array([p["seed_sd_at_largest"] for p in ok])
        out["floors"] = {
            p["system"]: {
                "L_inf": round(p["capacity_fit"]["L_inf"], 5),
                "lo": round(p["capacity_fit"].get("L_inf_lo", float("nan")), 5),
                "hi": round(p["capacity_fit"].get("L_inf_hi", float("nan")), 5),
                "supported": p["capacity_fit"].get("floor_supported"),
            }
            for p in ok
        }
        out["floors_differ_by_more_than_seed_noise"] = bool(
            np.ptp(floors) > 2 * np.nanmedian(seedsd)
        )
        out["n_systems_with_supported_floor"] = int(
            sum(bool(p["capacity_fit"].get("floor_supported")) for p in ok)
        )
        # P5's prediction: alpha ~ 1/d
        alphas = np.array([p["capacity_fit"]["alpha"] for p in ok])
        dky = np.array([p["kaplan_yorke_dim"] for p in ok])
        out["alpha_vs_inv_dky"] = {
            "alpha": alphas.round(4).tolist(),
            "kaplan_yorke_dim": dky.round(3).tolist(),
            "inv_dky": (1.0 / dky).round(4).tolist(),
            "spearman_alpha_vs_inv_dky": spearman(alphas, 1.0 / dky),
            "spearman_alpha_vs_dky": spearman(alphas, dky),
            "spearman_alpha_vs_lyap": spearman(
                alphas, np.array([p["lyap_max"] for p in ok])
            ),
        }
    return out


def analyse_r5() -> dict:
    curves = read_csv(RES / "r5" / "capacity_curves.csv")
    req = read_csv(RES / "r5" / "required_capacity.csv")
    if not req:
        return {}
    out: dict = {"families": {}}
    for fam in sorted({r["family"] for r in req}):
        tolerances = sorted({r["tolerance"] for r in req if r["family"] == fam})
        per_tol = {}
        for tol in tolerances:
            sub = sorted(
                [r for r in req if r["family"] == fam and r["tolerance"] == tol],
                key=lambda r: r["knob"],
            )
            knobs = [r["knob"] for r in sub]
            caps = [r["cap_median"] for r in sub]
            finite = [c for c in caps if np.isfinite(c)]
            peak_idx = int(np.nanargmax(caps)) if finite else None
            non_monotone = (
                peak_idx is not None
                and 0 < peak_idx < len(caps) - 1
                and np.isfinite(caps[-1])
                and caps[-1] < caps[peak_idx]
            )
            per_tol[str(tol)] = {
                "knob": knobs,
                "required_capacity_median": caps,
                "cap_min": [r["cap_min"] for r in sub],
                "cap_max": [r["cap_max"] for r in sub],
                "n_seeds_reaching": [r["n_seeds_reaching"] for r in sub],
                "peak_at_knob": knobs[peak_idx] if peak_idx is not None else None,
                "non_monotone_peak_interior": bool(non_monotone),
                "drop_from_peak_pct": (
                    float(100 * (1 - caps[-1] / caps[peak_idx]))
                    if non_monotone and caps[peak_idx] > 0 else None
                ),
            }
        n_tol_nonmono = sum(v["non_monotone_peak_interior"] for v in per_tol.values())
        out["families"][fam] = {
            "by_tolerance": per_tol,
            "n_tolerances": len(tolerances),
            "n_tolerances_showing_non_monotonicity": n_tol_nonmono,
            "robust_across_tolerances": bool(n_tol_nonmono >= max(2, len(tolerances) - 1)),
        }
    # seed spread at fixed setting, the thing NNPT never reported
    if curves:
        spreads = []
        for fam in sorted({r["family"] for r in curves}):
            for knob in sorted({r["knob"] for r in curves if r["family"] == fam}):
                for b in sorted({r["budget"] for r in curves
                                 if r["family"] == fam and r["knob"] == knob}):
                    e = [r["rel_error"] for r in curves
                         if r["family"] == fam and r["knob"] == knob and r["budget"] == b]
                    if len(e) > 1 and np.mean(e) > 0:
                        spreads.append(float(np.std(e) / np.mean(e)))
        out["median_seed_cv_of_error"] = float(np.median(spreads)) if spreads else None
    return out


def analyse_r6() -> dict:
    """Pointwise vs structural skill, computed off R4's grid. No new training."""
    rows = read_csv(RES / "r4" / "capacity_data_sweep.csv")
    if not rows:
        return {}
    out: dict = {"per_system": []}
    for s in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == s and not r["diverged"]]
        if len(sub) < 8:
            continue
        point = np.array([r["nrmse_1"] for r in sub], float)
        vpt = np.array([r["vpt_steps"] for r in sub], float)
        spec = np.array([r["spectrum_error"] for r in sub], float)
        kl = np.array([r["density_kl"] for r in sub], float)
        params = np.array([r["n_params"] for r in sub], float)
        out["per_system"].append(
            {
                "system": s,
                "n_models": len(sub),
                "spearman_pointwise_vs_spectrum": spearman(point, spec),
                "spearman_pointwise_vs_density_kl": spearman(point, kl),
                "spearman_vpt_vs_spectrum": spearman(-vpt, spec),
                "spearman_capacity_vs_pointwise": spearman(params, -point),
                "spearman_capacity_vs_spectrum": spearman(params, -spec),
                "spearman_capacity_vs_density_kl": spearman(params, -kl),
            }
        )
    if out["per_system"]:
        for key in (
            "spearman_pointwise_vs_spectrum", "spearman_pointwise_vs_density_kl",
            "spearman_capacity_vs_pointwise", "spearman_capacity_vs_spectrum",
            "spearman_capacity_vs_density_kl",
        ):
            vals = [p[key] for p in out["per_system"] if np.isfinite(p[key])]
            out[f"median_{key}"] = float(np.median(vals)) if vals else None
    return out


def analyse_r8() -> dict:
    rows = read_csv(RES / "r8" / "direct_vs_rollout.csv")
    if not rows:
        return {}
    out: dict = {"per_system": []}
    for s in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == s]
        hs = sorted({r["horizon"] for r in sub})
        roll = [float(np.mean([r["err_rollout"] for r in sub if r["horizon"] == h])) for h in hs]
        dirc = [float(np.mean([r["err_direct"] for r in sub if r["horizon"] == h])) for h in hs]
        par = [float(np.mean([r["err_parrot"] for r in sub if r["horizon"] == h])) for h in hs]
        cross = next((h for h, a, b in zip(hs, roll, dirc) if b < a), None)
        out["per_system"].append(
            {
                "system": s,
                "lyap_max": sub[0]["lyap_max"],
                "kaplan_yorke_dim": sub[0]["kaplan_yorke_dim"],
                "horizons": hs,
                "err_rollout": [round(v, 4) for v in roll],
                "err_direct": [round(v, 4) for v in dirc],
                "err_parrot": [round(v, 4) for v in par],
                "crossover_horizon": cross,
                "ratio_at_max_horizon": float(roll[-1] / dirc[-1]) if dirc[-1] > 0 else None,
                "direct_beats_parrot_at_max": bool(dirc[-1] < par[-1]),
                "rollout_beats_parrot_at_max": bool(roll[-1] < par[-1]),
            }
        )
    ratios = [p["ratio_at_max_horizon"] for p in out["per_system"] if p["ratio_at_max_horizon"]]
    out["median_rollout_over_direct_at_max_horizon"] = (
        float(np.median(ratios)) if ratios else None
    )
    out["n_systems_direct_wins_at_max"] = int(sum(r > 1 for r in ratios))
    # the discriminating question: does direct prediction escape the failure rollout hits?
    out["interpretation_note"] = (
        "If direct error stays low where rollout error saturates, the long-horizon failure "
        "is error compounding. If both saturate at the same level, it is a system ceiling."
    )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "r1_coarse_grain": analyse_r1(),
        "r2_instruments": analyse_r2(),
        "r3_wpe_vs_error": analyse_r3(),
        "r4_capacity_data": analyse_r4(),
        "r5_nonmonotone": analyse_r5(),
        "r6_pointwise_vs_structural": analyse_r6(),
        "r8_direct_vs_rollout": analyse_r8(),
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for name, block in report.items():
        print(f"\n===== {name} =====")
        if not block:
            print("  (no data yet)")
            continue
        if name == "r1_coarse_grain":
            for b in block["by_setting"]:
                print(
                    f"  N={b['block_size']} k={b['k']}: {b['n_reducible']}/{b['n_rules']} "
                    f"reducible ({b['frac_reducible']:.1%}) with {b['n_projections_tried']:,} "
                    f"projections {'exhaustive' if b['exhaustive'] else 'sampled'}; "
                    f"irreducible={b['irreducible'] if len(b['irreducible'])<=12 else len(b['irreducible'])}"
                )
            for k, u in block["union_by_k"].items():
                print(f"  union over block sizes, k={k}: irreducible everywhere = {u['irreducible_everywhere']}")
            for sset in block["search_width_sensitivity"]:
                print(f"  search-width sensitivity N={sset['block_size']}: {sset['frac_by_k']} delta={sset['delta']:.3f}")
        elif name == "r2_instruments":
            print(f"  {block['n_systems']} systems, {len(block['columns'])} statistics")
            print(f"  median |Spearman| off-diagonal: {block['median_abs_offdiag_spearman']:.3f}")
            print(f"  components for 90% variance: {block['n_components_for_90pct']}")
            print(f"  participation ratio (effective dims): {block['participation_ratio']}")
        elif name == "r3_wpe_vs_error":
            print(f"  {block['n_systems']} systems")
            print(f"  R2 of log VPT: {block['r2_log_vpt']}")
            print(f"  KILL CRITERION (WPE+lambda >= 0.85): {block['kill_criterion_wpe_plus_lyap_ge_0.85']}")
            print(f"  model beats parroting on {block['model_beats_parrot_n']}/{block['n_systems']} systems")
            print(f"  median VPT ratio model/parrot: {block['median_vpt_ratio_model_over_parrot']:.2f}")
            print(f"  Spearman VPT vs: {json.dumps({k: round(v,3) for k,v in block['spearman_vpt_vs'].items()})}")
        elif name == "r4_capacity_data":
            for p in block["per_system"]:
                f = p["capacity_fit"]
                if f.get("ok"):
                    print(
                        f"  {p['system']:<12} alpha={f['alpha']:.3f} L_inf={f['L_inf']:.4f} "
                        f"[{f.get('L_inf_lo', float('nan')):.4f},{f.get('L_inf_hi', float('nan')):.4f}] "
                        f"floor={f.get('floor_supported')} Dky={p['kaplan_yorke_dim']:.2f}"
                    )
            if "alpha_vs_inv_dky" in block:
                a = block["alpha_vs_inv_dky"]
                print(f"  alpha vs 1/D_KY Spearman: {a['spearman_alpha_vs_inv_dky']:.3f}")
                print(f"  floors differ > 2x seed noise: {block.get('floors_differ_by_more_than_seed_noise')}")
        elif name == "r5_nonmonotone":
            for fam, b in block["families"].items():
                print(
                    f"  {fam}: non-monotone at {b['n_tolerances_showing_non_monotonicity']}"
                    f"/{b['n_tolerances']} tolerances -> robust={b['robust_across_tolerances']}"
                )
                for tol, v in b["by_tolerance"].items():
                    print(
                        f"    tol={tol}: caps={v['required_capacity_median']} "
                        f"peak@{v['peak_at_knob']} nonmono={v['non_monotone_peak_interior']}"
                    )
            print(f"  median seed CV of error: {block.get('median_seed_cv_of_error')}")
        elif name == "r6_pointwise_vs_structural":
            for k, v in block.items():
                if k.startswith("median_"):
                    print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: n/a")
        elif name == "r8_direct_vs_rollout":
            for p in block["per_system"]:
                print(
                    f"  {p['system']:<12} h={p['horizons']}\n"
                    f"      rollout {p['err_rollout']}\n"
                    f"      direct  {p['err_direct']}\n"
                    f"      parrot  {p['err_parrot']}  crossover@{p['crossover_horizon']}"
                )
            print(f"  median rollout/direct at max horizon: {block['median_rollout_over_direct_at_max_horizon']}")

    print(f"\nwrote {OUT/'analysis.json'}")


if __name__ == "__main__":
    main()
