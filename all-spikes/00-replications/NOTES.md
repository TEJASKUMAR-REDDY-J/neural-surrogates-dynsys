# 00-replications — notes

Running log for the replication battery. Newest at the bottom.
Plan: [PLAN.md](PLAN.md). Results as they land: `RESULTS.md` (not yet created).

This spike produces **no novel claim**. Its output is a calibration report: for each
measurement the literature offers, what it costs on this hardware, what it actually measures,
how much it moves across seeds, and whether it is worth building a research question on.

---

## 2026-09-07 — spike created

**What.** Scaffolded the directory layout, logging schema and shared run-identity utility.
Wrote `PLAN.md`: nine experiments (R0–R8), each with a claim under test, a design, dependent
variables, a cost estimate on this machine, and — where applicable — a pre-registered kill
criterion.

**Why these nine.** Each targets a measurement the project would otherwise depend on
unexamined:

| | Tests | Kills if |
|---|---|---|
| R1 | is "admits a coarse description" a property or a search artefact | reducible fraction moves with search-space width |
| R2 | how many independent axes the system-property feature space really has | instruments are redundant (1–2 effective dims) |
| R3 | does weighted permutation entropy already explain surrogate error | WPE + λ explain ≥ 85% of variance |
| R4 | does skill saturate per system; is α ≈ 1/d_KY | no saturation ⇒ Gilpin is right, leap-horizon framing dies |
| R5 | does NNPT's non-monotone capacity survive seeds and a finer grid | effect appears only at one tolerance |
| R6 | when pointwise and distributional skill diverge | (free off R4; no kill, pure measurement) |
| R7 | how much of "learning" is lookup | parroting matches trained models |
| R8 | direct vs autoregressive, and where the crossover is | — |

**Key design decision.** R7 (context parroting baseline) runs *before* the spike is chosen,
not after. If a parameter-free copy baseline matches our trained surrogates, every
leap-horizon claim downstream is unfounded, and it is cheaper to learn that now.

**Logging.** One `logs/<run_id>.jsonl` per run: manifest (config, config hash, git sha,
environment) → one line per unit of work → summary. `run_id` carries into every results row
and figure caption, so any number in the write-up traces to a config and a commit. Reruns
never overwrite; a changed config changes the hash and therefore the run_id.
`src/common/runlog.py` has an `assert`-based self-check under `__main__` — passes.

**Not yet done.** Nothing has been run. `src/common/{systems,metrics,models,train,scaling}.py`
are unwritten; R0 is the first task and is blocking for everything except R1.

**Next.** R0 (dependency check + metrics library validated against published values on
Lorenz / logistic / white noise), then R2 and R7, which are near-free.

## 2026-09-07 — cost estimates re-derived; plan rewritten in plain language

**Why.** User asked why some experiments were estimated at 6 hours. Fair challenge: those
numbers were padded guesses, not arithmetic.

**What was done.** Built a cost model calibrated on the two measurements in `ENVIRONMENT.md`:
- 133k params, 50k samples, 20 epochs = 17 s  =>  `1.28e-10 s per (param x sample x epoch)`,
  which implies ~47 GFLOP/s on these two cores. Believable for batched GEMM on this CPU.
- Below ~1e4 params the model is not FLOP-bound. Per-batch overhead (~150 us) dominates, so
  small models have a floor of `(samples/256) x epochs x 150us`.

**Result: most estimates were 4-6x too high. Total battery 15-20 h -> ~5-6 h.**

| | old | derived |
|---|---|---|
| R1 | 2 h | 1-2 h |
| R2 | 1 h | 45 min |
| R3 | 2 h | 15 min |
| R4 | 6 h | 1.5 h |
| R5 | 4 h | 1.2 h |
| R8 | 3 h | 20 min |

**The substantive finding, which changed how the plan is written.** Training is not the
bottleneck at our scale. A single fit is seconds: the largest cell in R4's grid (1e5 params,
32k samples, 200 epochs) is 82 s and the median cell is under 5 s. The real costs are:
1. **R1's combinatorics** — 69e9 elementwise ops at N=4 (65,536 projections x 256 rules x
   4,096 triples). Nothing to do with ML, and now the longest single experiment in the battery.
   This was the surprise.
2. **FSLE in R2** — many perturbed trajectory pairs per system, ~37 min of R2's 45.
3. **Trajectory generation** for stiff systems, which is the one number the model cannot
   predict: 0.28 s per 4k-point Lorenz trajectory says nothing about a stiff system needing
   1000x smaller steps.

**Action taken.** R0 now includes a **timing probe**: run one fit and one trajectory at each
scale in the grid and replace every estimate with a measurement before committing to the big
runs. If the probe disagrees with the model, the grid gets rescaled rather than the schedule
slipping.

**Also.** PLAN.md rewritten so every experiment states, in plain language: what it does, the
specifics, the hypothesis (H1/H0 and any competing explanation), why we care, and what the
opposite outcome would tell us. Added a glossary for the recurring terms (rollout, capacity,
saturation, Lyapunov time, WPE, attractor, coarse-graining).

**One thing the rewrite made clearer than it was before.** R8 is the sharpest experiment in the
battery and it is 20 minutes. P8 reports that all three architectures fail at long rollout and
offers a bound, not an explanation. A direct horizon-conditioned model does not compound errors.
So if the direct model escapes the failure, the cause was compounding; if it fails equally, the
cause is a system ceiling. That is the Gilpin-vs-Duraisamy dispute, separated for 20 minutes of
compute. It should probably be promoted ahead of R5.

## 2026-09-08 — R9 done; Kuramoto-Sivashinsky attempted and blocked

**R9 (iterative refinement) complete.** 1,152 rows, 6 systems, 4 horizons, 16 evaluated passes,
3 seeds. Answers in RESULTS.md. Short version: pass 2 beats pass 1 in 21/24 cases, the best
pass is always inside the 4 it was trained with (24/24), it never kept improving to pass 16
(0/24), and on 7/24 cases pass 16 is more than 10% worse than the best - one case 291% worse.
Refinement improves on single-pass direct prediction but still loses to plain rollout by 20x at
short horizons, while failing more gracefully than rollout at long ones.

**Kuramoto-Sivashinsky: attempted, not working, deliberately not shipped.**

N1 was the highest-value planned addition - a spatially extended system to bridge to the
neural-operator literature. Implemented ETDRK4 (Kassam & Trefethen) pseudospectrally. It runs
and produces correct-looking chaotic dynamics over short integrations, then destabilises.

What was tried, in order:
- initial condition scaled to the domain (the textbook one is written for length 32*pi and is
  nearly constant on a length-22 domain) - fixed a genuine bug, did not fix the instability
- 2/3 dealiasing on the nonlinear term - standard, applied, did not fix it
- step sizes 0.25, 0.1, 0.05, 0.025 - only 0.025 survives 11k steps, and even that diverges by
  ~42k
- zeroing the Nyquist mode in the derivative operator, as the published code does - no effect
- pinning the mean mode to zero (it is undamped, so roundoff accumulates there) - no effect
- the canonical parameters themselves (L=32*pi, N=128, dt=0.25) - also diverges on long runs,
  which is the tell: this is a bug in our implementation, not a parameter choice, because those
  settings are published as stable

Conclusion: the scheme is right on paper and something in the implementation is wrong in a way
that only shows over long integrations. The right fix is to check against a validated reference
implementation rather than keep guessing, so KS is parked rather than shipped half-working.
Nothing that depends on it has been claimed.

**What ran instead.** The other four planned items, all of which use already-validated code:
R2 extended to every usable system (the predictor features N3 needs), N3 at full scale, N5 the
batch-size sweep with seeds, N2 observational noise at three levels.
