"""R1 - is "this rule has a simple coarse description" a property, or an artefact of search?

Israeli & Goldenfeld (PRL 2004) found 240 of 256 elementary cellular automata admit a
coarse-graining at block size N <= 4. Dzwinel & Magiera (2015) pushed to N = 7 and found
the irreducible set converges to {30, 45, 106, 154}. The 2006 extension of the original
paper reports that the probability of finding a coarse description approaches 1 as the
scale gets coarser.

Neither paper measured the thing we actually need to know: how much of "reducible" is a
fact about the rule and how much is a fact about how many candidate projections you were
willing to try. So we sweep the *width of the search space* at fixed block size and watch
what the reducible count does.

Method. For block size N, the block automaton maps three consecutive N-blocks to the
middle block N steps later (a radius-1 CA needs exactly 3N cells to determine N cells N
steps on). A projection P maps the 2^N block symbols onto k coarse symbols. The
coarse-graining is valid iff P(f(x)) is determined by (P(x1), P(x2), P(x3)) - i.e. no two
triples that look the same after projection have outputs that look different.
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

from src.common.runlog import RunLog  # noqa: E402

RESULTS = ROOT / "results" / "r1"
LOGS = ROOT / "logs"


def eca_step(row: np.ndarray, rule: int) -> np.ndarray:
    """One step of an elementary CA with periodic boundaries."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    return ((rule >> idx) & 1).astype(np.uint8)


def block_automaton_table(rule: int, N: int) -> np.ndarray:
    """Lookup table for the N-block automaton.

    Index is b1*S^2 + b2*S + b3 with S = 2^N; value is the middle block after N steps.
    Built by taking each triple as a 3N-cell strip, running N steps with enough context,
    and reading off the middle N cells.
    """
    S = 1 << N
    # decode all triples into 3N-cell strips at once
    tri = np.arange(S**3, dtype=np.int64)
    b1, b2, b3 = tri // (S * S), (tri // S) % S, tri % S
    bits = np.empty((len(tri), 3 * N), dtype=np.uint8)
    for j in range(N):
        shift = N - 1 - j
        bits[:, j] = (b1 >> shift) & 1
        bits[:, N + j] = (b2 >> shift) & 1
        bits[:, 2 * N + j] = (b3 >> shift) & 1

    # N steps on a periodic strip of 3N cells: the middle N cells are unaffected by
    # wrap-around because the light cone from the middle block reaches at most N cells
    # each side, which is exactly the strip.
    state = bits
    for _ in range(N):
        left = np.roll(state, 1, axis=1)
        right = np.roll(state, -1, axis=1)
        idx = (left.astype(np.int64) << 2) | (state.astype(np.int64) << 1) | right
        state = ((rule >> idx) & 1).astype(np.uint8)

    mid = state[:, N : 2 * N]
    out = np.zeros(len(tri), dtype=np.int64)
    for j in range(N):
        out = (out << 1) | mid[:, j]
    return out


def enumerate_projections(S: int, k: int, max_count: int | None, seed: int = 0) -> np.ndarray:
    """Projections of S block symbols onto k coarse symbols, as a (B, S) integer array.

    Only surjective projections are kept: a projection that never uses one of its symbols
    is really a projection onto fewer symbols, and the constant projection trivially
    satisfies the consistency condition for every rule, which would make the whole
    measurement meaningless.
    """
    total = k**S
    rng = np.random.default_rng(seed)
    if max_count is None or total <= max_count:
        codes = np.arange(total, dtype=np.int64)
    else:
        codes = rng.choice(total, size=max_count, replace=False)
    P = np.empty((len(codes), S), dtype=np.int64)
    c = codes.copy()
    for j in range(S):
        P[:, j] = c % k
        c //= k
    keep = np.ones(len(P), dtype=bool)
    for v in range(k):
        keep &= (P == v).any(axis=1)
    return P[keep]


def check_projections(
    out_table: np.ndarray, P: np.ndarray, S: int, k: int, chunk: int = 2048
) -> np.ndarray:
    """Boolean per projection: does it induce a single-valued coarse rule?

    A projection is valid iff no coarse class ever produces two different coarse outputs.

    Implemented as a bitmask reduction rather than a scatter. For each output symbol v we
    OR together a bit for every class in which v occurs, giving one k^3-bit word per
    projection; the projection is inconsistent exactly when two of those words share a
    bit. This needs k passes of `bitwise_or.reduce` over a uint8-indexed array instead of
    a bincount over B*k^4 bins, and it is ~15x faster at block size 4, which is where the
    whole experiment's cost lives. Requires k**3 <= 64, true for every k we use.
    """
    if k**3 > 64:
        raise ValueError(f"k={k} needs more than 64 classes; bitmask packing does not fit")

    tri_all = np.arange(S**3, dtype=np.int64)
    Pu = np.ascontiguousarray(P.astype(np.uint8))
    one = np.uint64(1)

    def _consistent(Pc: np.ndarray, tri: np.ndarray) -> np.ndarray:
        b1, b2, b3 = tri // (S * S), (tri // S) % S, tri % S
        cls = (Pc[:, b1] * k + Pc[:, b2]) * k + Pc[:, b3]          # (B, T) uint8
        o = Pc[:, out_table[tri]]                                   # (B, T) uint8
        bit = one << cls.astype(np.uint64)
        masks = [
            np.bitwise_or.reduce(np.where(o == v, bit, np.uint64(0)), axis=1)
            for v in range(k)
        ]
        bad = np.zeros(len(Pc), dtype=np.uint64)
        for a in range(k):
            for b in range(a + 1, k):
                bad |= masks[a] & masks[b]
        return bad == 0

    # Two-stage filter. A projection that is inconsistent on a subset of triples is
    # inconsistent, full stop - so a cheap pass over a stratified subsample rejects the
    # overwhelming majority at a fraction of the cost, and only survivors get the full
    # check. Exact, not approximate: stage 1 only ever rejects definite failures.
    n_probe = max(64, min(len(tri_all), len(tri_all) // 16))
    probe = tri_all[:: max(1, len(tri_all) // n_probe)]

    valid = np.zeros(len(P), dtype=bool)
    for s in range(0, len(P), chunk):
        Pc = Pu[s : s + chunk]
        survivors = _consistent(Pc, probe)
        if not survivors.any():
            continue
        idx = np.flatnonzero(survivors)
        valid[s + idx] = _consistent(Pc[idx], tri_all)
    return valid


def coarse_rule_id(out_table: np.ndarray, P: np.ndarray, S: int, k: int) -> int | None:
    """For a valid binary projection, the induced coarse rule as an ECA rule number."""
    if k != 2:
        return None
    table = np.full(8, -1, dtype=np.int64)
    tri = np.arange(S**3, dtype=np.int64)
    b1, b2, b3 = tri // (S * S), (tri // S) % S, tri % S
    cls = (P[b1] * 2 + P[b2]) * 2 + P[b3]
    o = P[out_table]
    for c in range(8):
        m = cls == c
        if m.any():
            table[c] = o[m][0]
    if (table < 0).any():
        return None
    return int(sum(int(table[c]) << c for c in range(8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--k-values", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--max-projections", type=int, default=200_000)
    ap.add_argument("--rules", type=int, nargs="*", default=None)
    ap.add_argument(
        "--escalate-sizes", type=int, nargs="*", default=[],
        help="after the main sweep, retry only the still-irreducible rules at these larger "
             "block sizes. Dzwinel & Magiera report the set converging to {30,45,106,154} "
             "at N=7, and the survivor set shrinks fast, so escalation is cheap.",
    )
    ap.add_argument(
        "--escalate-projections", type=int, nargs="+",
        default=[40_000, 400_000, 4_000_000],
        help="search budgets tried in order. The budget at which a rule first "
             "becomes reducible IS the measurement: past block size 4 the "
             "projection space cannot be enumerated, so every count is a lower "
             "bound whose value depends on how hard we looked.",
    )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rules = args.rules if args.rules else list(range(256))
    rows = []

    with RunLog("r1_coarse_grain", vars(args), "cli", LOGS) as log:
        for N in args.block_sizes:
            S = 1 << N
            for k in args.k_values:
                if k >= S:
                    continue
                t0 = time.time()
                P = enumerate_projections(S, k, args.max_projections)
                total_possible = k**S
                exhaustive = total_possible <= args.max_projections
                log.note(
                    "projection set built",
                    N=N,
                    k=k,
                    n_surjective=int(len(P)),
                    total_possible=int(total_possible),
                    exhaustive=bool(exhaustive),
                    build_s=round(time.time() - t0, 2),
                )
                print(
                    f"\n=== N={N} (S={S}) k={k}: {len(P):,} surjective projections "
                    f"of {total_possible:,} {'(exhaustive)' if exhaustive else '(sampled)'} ===",
                    flush=True,
                )

                n_red = 0
                for r_i, rule in enumerate(rules):
                    t1 = time.time()
                    tab = block_automaton_table(rule, N)
                    valid = check_projections(tab, P, S, k)
                    n_valid = int(valid.sum())
                    reducible = n_valid > 0
                    n_red += reducible
                    target = None
                    if reducible and k == 2:
                        target = coarse_rule_id(tab, P[np.argmax(valid)], S, k)
                    row = {
                        "block_size": N,
                        "k": k,
                        "rule": rule,
                        "n_projections_tried": int(len(P)),
                        "exhaustive": bool(exhaustive),
                        "n_valid_projections": n_valid,
                        "reducible": bool(reducible),
                        "example_coarse_rule": target,
                        "wall_s": round(time.time() - t1, 3),
                    }
                    rows.append(row)
                    log.result(**row)
                    if (r_i + 1) % 32 == 0:
                        print(
                            f"  rules {r_i+1:3d}/{len(rules)}  reducible so far {n_red}"
                            f"  ({time.time()-t0:.0f}s)",
                            flush=True,
                        )

                irreducible = sorted(
                    r["rule"] for r in rows
                    if r["block_size"] == N and r["k"] == k and not r["reducible"]
                )
                print(
                    f"  N={N} k={k}: {n_red}/{len(rules)} reducible; "
                    f"irreducible = {irreducible if len(irreducible) <= 24 else str(len(irreducible)) + ' rules'}",
                    flush=True,
                )
                log.note("block size done", N=N, k=k, n_reducible=n_red, irreducible=irreducible)

        # Escalation: larger block sizes are only interesting for rules that have not
        # already been reduced, and the survivor set collapses quickly, so this costs
        # little and reaches the scale where the published answer lives.
        for N in args.escalate_sizes:
            S = 1 << N
            reduced = {r["rule"] for r in rows if r["reducible"]}
            survivors = [r for r in rules if r not in reduced]
            if not survivors:
                print(f"\n=== N={N}: nothing left to test ===", flush=True)
                break
            print(
                f"\n=== N={N} (S={S}) k=2: escalating {len(survivors)} still-irreducible "
                f"rules through budgets {args.escalate_projections} ===", flush=True)
            log.note("escalation", N=N, n_survivors=len(survivors), survivors=survivors,
                     budgets=args.escalate_projections)
            pending = list(survivors)
            for budget in args.escalate_projections:
                if not pending:
                    break
                P = enumerate_projections(S, 2, budget)
                nxt = []
                for rule in pending:
                    t1 = time.time()
                    tab = block_automaton_table(rule, N)
                    valid = check_projections(tab, P, S, 2, chunk=512)
                    n_valid = int(valid.sum())
                    row = {
                        "block_size": N, "k": 2, "rule": rule,
                        "n_projections_tried": int(len(P)),
                        "exhaustive": bool(2**S <= budget),
                        "n_valid_projections": n_valid,
                        "reducible": bool(n_valid > 0),
                        "example_coarse_rule": (
                            coarse_rule_id(tab, P[np.argmax(valid)], S, 2) if n_valid else None
                        ),
                        "wall_s": round(time.time() - t1, 3),
                    }
                    rows.append(row)
                    log.result(**row)
                    if n_valid:
                        print(
                            f"  N={N} budget {budget:>9,}: rule {rule:>3} REDUCIBLE -> "
                            f"{row['example_coarse_rule']} ({row['wall_s']:.1f}s)", flush=True)
                    else:
                        nxt.append(rule)
                print(f"  N={N} budget {budget:>9,}: {len(pending)-len(nxt)} newly reducible, "
                      f"{len(nxt)} still irreducible", flush=True)
                pending = nxt
            print(f"  N={N}: still irreducible after all budgets = {pending}", flush=True)
            log.note("escalation done", N=N, still_irreducible=pending)

    keys = sorted({k for r in rows for k in r})
    with (RESULTS / "eca_coarse_grain.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {RESULTS/'eca_coarse_grain.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
