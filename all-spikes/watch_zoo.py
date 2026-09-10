import json, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CAP = 320
def alive():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'study.py|supervise.py' } | "
          "Measure-Object | Select-Object -ExpandProperty Count")
    try:
        r = subprocess.run(["powershell","-NoProfile","-Command",ps],
                           capture_output=True, text=True, timeout=60)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return 0
t0=time.time()
while True:
    n=alive(); mins=(time.time()-t0)/60
    if n==0: print(f"ZOO DONE after {mins:.0f} min"); break
    if mins>CAP: print(f"WATCH CAP at {mins:.0f} min, {n} alive"); break
    time.sleep(180)
p=ROOT/"supervise_status.json"
if p.exists(): print(p.read_text(encoding="utf-8"))
