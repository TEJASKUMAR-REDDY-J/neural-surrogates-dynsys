# Environment audit

Audited 2026-09-06. This is the feasibility budget: any proposed experiment must fit here.

## Hardware
| | |
|---|---|
| CPU | Intel Core i3-10110U @ 2.10 GHz — **2 cores / 4 threads** |
| RAM | 23.9 GB total, ~6 GB free at audit |
| GPU | Intel UHD Graphics (integrated, 1 GB shared) — **no CUDA, no usable accelerator** |
| Disk | 369 GB free on C: |
| OS | Windows 11, PowerShell + Git Bash |

## Software
- Python 3.12.7 (Anaconda, `C:\ProgramData\anaconda3\python.exe`)
- torch 2.8.0 (CPU build), torchvision 0.23.0, torch-geometric 2.8.0
- jax 0.7.1 + jaxlib 0.7.1 (CPU), numpy 1.26.4, scipy 1.12.0
- scikit-learn 1.8.0, pandas 2.2.2, matplotlib 3.9.2
- git 2.54.0

## Missing / needs decision
- **No LaTeX** (`pdflatex` not found). Required for the final CAISc PDF.
  Options: install MiKTeX (~250 MB) or Tectonic (single binary), or compile on Overleaf.
- **No `codex` CLI.** `voila.md` and `research-philosophy.md` both call for an
  *independent* critic at decision points. A subagent of the same model is not
  independent and must not be presented as such.
- No `torchdiffeq` / `diffrax` (installable if an ODE-solver-in-the-loop design is chosen;
  `scipy.integrate.solve_ivp` covers most needs without a new dependency).

## What this constrains
Two slow cores, no GPU. Budget accordingly:
- Train runs must be **minutes, not hours**. Assume ~1-5 min per model fit.
- Model scale: MLPs / small RNNs / small operator nets up to ~10^5-10^6 params.
  Anything transformer-scale or image-scale is out.
- Systems: low-dimensional ODEs (Lorenz, Van der Pol, Duffing, double pendulum,
  two-body) and small-grid 1D PDEs (KS, Burgers on <=256 points). Not 3D CFD.
- Datasets must be **generated**, not downloaded — cheap trajectory integration is free,
  large public benchmarks are not.
- Seeds are cheap and parallelisable across 4 threads; **prefer many small runs over one
  big run**. This is a real advantage for rigor (multi-seed, ablations) and should shape
  which research questions are chosen.
