"""Run a queue of experiments one after another, with stall detection and a hard cap.

Written because watchers were being left to spin on jobs that had already died, and because
an experiment that hangs is worse than one that fails - it burns the machine and reports
nothing.

Each stage declares how long it is expected to take. The supervisor then:

  - starts the stage and checks progress on a fixed interval
  - counts the lines the stage has produced; if that count has not moved for
    `stall_checks` consecutive checks, the stage is presumed hung and killed
  - kills any stage that exceeds twice its own estimate, whether or not it is producing
  - moves to the next stage either way, and writes a one-line verdict per stage

so nothing is left running unattended and a hang costs one stall window rather than a night.

    python supervise.py --plan plan.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def count_progress(log: Path, marker: str) -> int:
    if not log.exists():
        return 0
    try:
        return sum(1 for line in log.read_text(encoding="utf-8", errors="ignore").splitlines()
                   if line.startswith(marker))
    except OSError:
        return 0


def run_stage(stage: dict, args) -> dict:
    name = stage["name"]
    logs = [ROOT / p for p in stage["logs"]]
    est_s = stage["est_minutes"] * 60
    hard_cap = est_s * stage.get("cap_multiple", 2.0)
    marker = stage.get("marker", "[")

    for lg in logs:
        lg.parent.mkdir(parents=True, exist_ok=True)

    procs = []
    for cmd, lg in zip(stage["cmds"], logs):
        fh = lg.open("w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=ROOT / stage["cwd"], stdout=fh,
                                       stderr=subprocess.STDOUT, shell=False), fh))
    print(f"[supervise] {name}: started {len(procs)} process(es), "
          f"estimate {stage['est_minutes']} min, hard cap "
          f"{hard_cap/60:.0f} min", flush=True)

    t0 = time.time()
    last, stalls = -1, 0
    verdict = "completed"
    while True:
        time.sleep(args.interval)
        alive = [p for p, _ in procs if p.poll() is None]
        done = sum(count_progress(lg, marker) for lg in logs)
        elapsed = time.time() - t0

        if not alive:
            verdict = "completed"
            break
        if elapsed > hard_cap:
            verdict = f"KILLED at hard cap ({hard_cap/60:.0f} min)"
            break
        if done == last:
            stalls += 1
            if stalls >= args.stall_checks:
                verdict = (f"KILLED as stalled: no new output for "
                           f"{stalls * args.interval / 60:.0f} min at {done} units")
                break
        else:
            stalls = 0
            print(f"[supervise] {name}: {done} units, {elapsed/60:.0f} min elapsed",
                  flush=True)
        last = done

    if verdict != "completed":
        for p, _ in procs:
            if p.poll() is None:
                p.kill()
    for _, fh in procs:
        fh.close()

    done = sum(count_progress(lg, marker) for lg in logs)
    out = {"stage": name, "verdict": verdict, "units": done,
           "minutes": round((time.time() - t0) / 60, 1)}
    print(f"[supervise] {name}: {verdict} - {done} units in {out['minutes']} min",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--stall-checks", type=int, default=5)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    results = []
    for stage in plan["stages"]:
        results.append(run_stage(stage, args))
        Path(ROOT / "supervise_status.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== supervisor summary ===")
    for r in results:
        print(f"  {r['stage']:22s} {r['verdict']:48s} {r['units']:>4d} units "
              f"{r['minutes']:>6.1f} min")


if __name__ == "__main__":
    main()
