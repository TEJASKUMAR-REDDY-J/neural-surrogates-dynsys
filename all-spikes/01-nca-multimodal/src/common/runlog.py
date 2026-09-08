"""Run identity and structured logging for the replication battery.

Every run writes exactly one logs/<run_id>.jsonl: a manifest line, one line per unit of
work, and a summary line. See PLAN.md for the schema and the rerun policy.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def config_hash(config: dict) -> str:
    """Stable hash of a resolved config. sort_keys so key order never changes it."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _env() -> dict:
    env = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch

        env["threads"] = torch.get_num_threads()
        env["cuda"] = torch.cuda.is_available()
    except Exception:
        pass
    return env


class RunLog:
    """Append-only JSONL log for one run.

    with RunLog("r4_capacity_data", cfg, cfg_path, log_dir) as log:
        for unit in units:
            log.result(system=..., width=..., metrics={...}, wall_s=...)
    """

    def __init__(
        self,
        experiment: str,
        config: dict,
        config_path: str | Path,
        log_dir: str | Path,
        seed: int | None = None,
    ):
        chash = config_hash(config)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
        self.run_id = f"{experiment}__{stamp}__{chash[:6]}"
        self.experiment = experiment
        self.path = Path(log_dir) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._failed = 0
        self._t0 = time.time()
        self._write(
            {
                "record": "manifest",
                "run_id": self.run_id,
                "experiment": experiment,
                "started_utc": _utc(),
                "git_sha": _git("rev-parse", "--short", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "config_path": str(config_path),
                "config_hash": f"sha256:{chash}",
                "config": config,
                "seed": seed,
                "env": _env(),
            }
        )

    def _write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def result(self, **fields) -> None:
        """One completed unit of work. Free-form fields plus a `metrics` dict."""
        self._n += 1
        self._write({"record": "result", **fields})

    def failure(self, **fields) -> None:
        """A unit that errored. Recorded, not raised - a partial run is still evidence."""
        self._failed += 1
        self._write({"record": "failure", **fields})

    def note(self, message: str, **fields) -> None:
        self._write({"record": "note", "message": message, **fields})

    def close(self) -> None:
        self._write(
            {
                "record": "summary",
                "n_units": self._n,
                "failed": self._failed,
                "wall_s": round(time.time() - self._t0, 1),
                "finished_utc": _utc(),
            }
        )

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.note("run aborted", error=f"{exc_type.__name__}: {exc}")
        self.close()
        return False


def demo() -> None:
    """Self-check: a log round-trips and the summary counts what was written."""
    import tempfile

    cfg = {"widths": [8, 16], "seeds": [0, 1]}
    assert config_hash(cfg) == config_hash({"seeds": [0, 1], "widths": [8, 16]}), (
        "config_hash must not depend on key order"
    )

    with tempfile.TemporaryDirectory() as tmp:
        with RunLog("demo", cfg, "configs/demo.yaml", tmp) as log:
            run_id = log.run_id
            log.result(system="Lorenz", width=8, metrics={"smape": 0.04}, wall_s=1.0)
            log.result(system="Lorenz", width=16, metrics={"smape": 0.03}, wall_s=1.1)
            log.failure(system="Rossler", width=8, error="diverged")

        records = [
            json.loads(line)
            for line in (Path(tmp) / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    kinds = [r["record"] for r in records]
    assert kinds == ["manifest", "result", "result", "failure", "summary"], kinds
    assert records[0]["config"] == cfg
    assert records[-1]["n_units"] == 2 and records[-1]["failed"] == 1
    print(f"ok - {len(records)} records, run_id={run_id}")


if __name__ == "__main__":
    demo()
    sys.exit(0)
