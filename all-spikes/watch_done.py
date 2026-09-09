"""Block until the supervisor finishes, then print its summary. Bounded by a hard cap."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAP_MIN = 420


def alive() -> int:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'run_hybrid|supervise.py' } | "
          "Measure-Object | Select-Object -ExpandProperty Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return 0


t0 = time.time()
while True:
    n = alive()
    mins = (time.time() - t0) / 60
    if n == 0:
        print(f"SUPERVISOR DONE after {mins:.0f} min watching")
        break
    if mins > CAP_MIN:
        print(f"WATCH CAP REACHED at {mins:.0f} min with {n} process(es) still alive")
        break
    time.sleep(180)

st = ROOT / "supervise_status.json"
if st.exists():
    print(json.dumps(json.loads(st.read_text(encoding="utf-8")), indent=2))
