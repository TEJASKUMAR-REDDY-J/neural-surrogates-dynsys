"""Rebuild R1's results CSV from its streaming run log.

Needed once because an integer overflow at block size 6 killed the sweep before the CSV was
written, and again because a small smoke test then overwrote the recovered file. The JSONL
run log records every row as it is produced, so nothing was actually lost - this is the
reason the log format exists.

Picks the run with the most result rows, so a later smoke test never clobbers a full sweep.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "r1" / "eca_coarse_grain.csv"


def main() -> None:
    runs = {}
    for f in sorted(glob.glob(str(ROOT / "logs" / "r1_coarse_grain__*.jsonl"))):
        rows = []
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("record") == "result" and "block_size" in r:
                rows.append({k: v for k, v in r.items() if k != "record"})
        if rows:
            runs[f] = rows
    if not runs:
        raise SystemExit("no R1 result rows found in logs/")

    path, rows = max(runs.items(), key=lambda kv: len(kv[1]))
    keys = sorted({k for r in rows for k in r})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"recovered {len(rows)} rows from {Path(path).name} -> {OUT}")

    from collections import defaultdict

    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["block_size"], r["k"], r["n_projections_tried"])
        agg[key][0] += 1
        agg[key][1] += bool(r["reducible"])
    for (N, k, npj), (tot, red) in sorted(agg.items()):
        print(f"  N={N} k={k} projections={npj:>9,}: {red:>3}/{tot} reducible ({100*red/tot:5.1f}%)")


if __name__ == "__main__":
    main()
