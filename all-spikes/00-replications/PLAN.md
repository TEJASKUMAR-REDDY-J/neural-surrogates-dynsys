# Replication battery — end-to-end tests of what the literature actually gives us

Created 2026-09-07. Status: **plan only, nothing run yet.**

## Why this exists

Reading a paper tells you what the authors claim. Running it tells you what the measurement
actually gives you — how noisy it is, how sensitive to a threshold, how much of the effect
survives a seed change, and whether the dependent variable means what the abstract implies.
Before committing to a research question we run a small end-to-end test of each measurement
we might depend on.

This is exploration, not the spike. It is deliberately cheap, it produces no novel claim, and
its output is a **calibration report**: for each instrument, what it costs, what it measures,
how much it varies, and whether it is worth building on.

Every experiment below is chosen so that **both outcomes are informative**. If a measurement
turns out to be unstable, that kills a candidate direction early, which is the whole point.

---

## Budget

From `ENVIRONMENT.md`: 2 cores, no GPU. Measured — 100 Lorenz trajectories of 4k points in
28 s; a 133k-param MLP, 50k samples, 20 epochs in 17 s. So a small fit is tens of seconds and
five seeds of one configuration is ~15 minutes.

Total estimated compute for the full battery: **~15–20 hours wall clock**, run in background
batches. No single experiment blocks another except where noted.

---

## The battery

Ordering is by information-per-hour, not by paper number. R2, R7, R0 are near-free and should
run first.

---

### R0 — Environment and dependency check
**Cost:** minutes.
**Does:** install and smoke-test `dysts`; confirm trajectory generation, the precomputed
invariant annotations, and timescale alignment work on this machine. Implement and unit-test
the shared metrics (WPE, permutation entropy, 0-1 test K, sMAPE, VPT, correlation dimension)
against published values on Lorenz / logistic / white noise.
**Gives us:** whether the substrate in A2 is usable here at all, and a metrics library whose
numbers we trust. **Blocking for R2–R8.**
**Kill criterion:** if `dysts` cannot generate aligned trajectories on this machine within a
few minutes per system, fall back to a hand-rolled system list and say so.

---

### R1 — Coarse-graining scan of elementary cellular automata
**Source:** P1 (Israeli & Goldenfeld 2004), A12 (PRE 2006), A13 (Dzwinel & Magiera 2015).
**Claim under test:** 240 of 256 ECA admit a coarse-graining at N ≤ 4; the irreducible set
converges to {30, 45, 106, 154}; and — A12 — the probability of finding a coarse-graining
tends to 1 as the scale gets coarser.

**Design.** For each of the 256 rules and each block size N ∈ {2, 3, 4}: build the block
automaton `A^N`, enumerate candidate projections P of the `2^N`-symbol block alphabet onto a
smaller alphabet, and test the single-valuedness condition. Vectorise the consistency check
over projections. N ≤ 3 is trivial; N = 4 is `2^16` binary projections × `4096` neighbourhood
triples per rule and needs chunked numpy — estimate 1–2 h for the full 256.

**The extra thing we do that the papers do not.** Report reducibility **as a function of the
declared search space**: binary projections only, vs projections onto 3 symbols, vs all
projections. This directly measures the *manufactured closure* hazard — how much of "this
system is reducible" is a fact about the system and how much is a fact about how hard we
looked.

**Dependent variables:** count of reducible rules per (N, search space); the fail set; the
transition graph between rules.
**Gives us:** (i) whether "admits a closed coarse description" is a stable label or a search
artefact; (ii) the actual base rate of the positive class, which decides whether any
CA-substrate learnability study is even well-posed as a classification problem.
**Kill criterion for the CA direction:** if the reducible fraction moves by more than a few
percent when the projection search space is widened at fixed N, the label is not a property of
the system and CA-based learnability classification is dead. Report and move on.
**Cost:** ~2 h. No training, pure combinatorics.

---

### R2 — Do the predictability instruments agree with each other?
**Source:** P2 (FSLE, ε-entropy), A14 (CDTA, 0-1 test), A15 (WPE), A1 (dysts invariants).
**Claim under test:** implicit in the whole project — that "chaoticity" and "complexity" are
several distinct measurable axes rather than one axis wearing different hats.

**Design.** For ~40 `dysts` systems, generate aligned trajectories and compute, with no model
training: largest Lyapunov exponent (from the annotation and re-estimated), Kaplan–Yorke
dimension, correlation dimension, multiscale entropy, **weighted permutation entropy**,
**modified 0-1 test K**, spectral entropy, and the FSLE curve λ(δ) on a log grid of δ.

**Dependent variables:** the full pairwise rank-correlation matrix between instruments; the
number of principal components needed to explain 90% of their variance; the δ-location of any
FSLE plateau.
**Gives us:** the honest dimensionality of the "system properties" feature space. This is the
cheapest possible check on a load-bearing assumption.
**Why both outcomes matter:** if the instruments are highly redundant (say 1–2 effective
dimensions), then "predict learnability from system properties" has very little signal to work
with and the ceiling on any such predictor is low — worth knowing before spending four days on
it. If they are genuinely multi-dimensional, that supports the residual-after-λ framing.
**Cost:** ~1 h. No training.

---

### R3 — Does a model-free predictability statistic predict *our* surrogate's error?
**Source:** A15 (Garland 2014), A16 (Pennekamp 2019), A18, A1.
**Claim under test:** weighted permutation entropy correlates with achievable forecast error —
and specifically, does it do so for a *neural surrogate of a dynamical system*, which is the
setting none of the three papers covers.

**Design.** ~20 `dysts` systems. One fixed small MLP one-step predictor, fixed capacity, fixed
data budget, 5 seeds each. Measure forecast error in Lyapunov times. Regress error on: WPE
alone, λ alone, WPE + λ, and the full R2 feature set. Include the Garland diagnostic reading —
distance from the fitted curve as a *method–data mismatch* signal.

**Dependent variables:** R² and rank correlation for each predictor set; the fitted
WPE–error curve; per-system residuals.
**Gives us:** the actual bar our project has to clear. If WPE alone explains most of the
variance in surrogate error, then a learned meta-model has almost nothing left to explain, and
we should say so rather than build one. If λ is weak but WPE is strong, that is a clean,
publishable refinement of Gilpin's decorrelation result — *the right statistic was never λ*.
**Kill criterion (pre-registered):** if WPE + λ explain ≥ 85% of the variance in surrogate
error across systems, the "learned learnability predictor" direction is not worth pursuing.
Report the negative.
**Cost:** ~2 h (≈100 fits).
**Depends on:** R0, R2.

---

### R4 — Per-system capacity × data sweep: does surrogate skill saturate?
**Source:** the crux. A1 (Gilpin: scale and data are the limit) vs A5 (Duraisamy: intrinsic
ceilings) vs P3 (non-monotone), with P5 supplying the machinery.
**Claim under test:** for a fixed system, does forecast skill keep climbing with capacity and
data, or does it saturate at a system-dependent floor?

**Design.** 6 systems chosen in advance to span the R2 feature space. Grid: 7 model widths on
a geometric ladder (~2 orders of magnitude in parameter count) × 4 training-set sizes (also
~2 orders) × 3 seeds = 504 fits. Fit the P5 scaling form per system — exponent α and
asymptotic floor `L∞` — and report confidence intervals on both.

**Dependent variables:** fitted α and `L∞` per system; whether `L∞` confidence intervals
separate across systems by more than seed variance; whether α correlates with the system's
Kaplan–Yorke dimension (P5 predicts `α ≈ 1/d`).
**Gives us:** the single most important number in the project. Also a direct test of a
quantitative prediction (`α ≈ 1/d_KY`) that nobody in either literature has checked, because
Gilpin regressed invariants against *accuracy* and not against the *exponent*.
**Why both outcomes matter:** saturation at system-dependent levels supports the intrinsic-
ceiling reading; no saturation supports Gilpin and kills the leap-horizon framing — and either
way we have measured something that three published papers disagree about.
**Honest caveat to state up front:** saturation observed up to 1e5 params does not prove
saturation at 1e9. We report the trend and the scale at which it was observed, never more.
**Cost:** ~6 h.
**Depends on:** R0, R2.

---

### R5 — Is the non-monotone capacity requirement real?
**Source:** P3 (NNPT). Directly addresses the user's question about parameter counts.
**Claim under test:** required capacity to reach a fixed tolerance peaks at intermediate
chaoticity and *falls* in the fully chaotic regime.

**Design.** The Chirikov standard map, sweeping K across the resonance-overlap transition —
this tests NNPT's claimed *mechanism* directly rather than by analogy through the three-body
mass parameter. 8 values of K × 9 widths on a fine geometric grid × 5 seeds = 360 fits.
Read required capacity off the same runs at **four tolerances (0.5%, 1%, 2%, 5%)** — free,
since it is a threshold on an already-computed curve.

Four things we do that NNPT did not:
1. Near-continuous **width** axis instead of a two-point depth step.
2. Seeds at every grid point; report the distribution of the threshold-crossing capacity.
3. **Threshold sensitivity** across four tolerances. If the non-monotonicity appears only at
   1%, it is an artefact. This is the highest-value cheap test in the battery.
4. Both dependent variables — required-capacity-at-tolerance **and** the fitted floor from R4's
   machinery — because they are different quantities and may disagree.

Plus a **no-subtraction control**: NNPT learns a residual after subtracting an exact solution.
Whether the effect belongs to the system or to the residual is untested. Run both.

**Gives us:** whether the most interesting recent result in this space survives contact with
seeds and a finer grid. A clean negative here is genuinely useful and cheap.
**Cost:** ~4 h.
**Depends on:** R0, R4 (shares the fitting code).

---

### R6 — When do pointwise and distributional skill come apart?
**Source:** Cluster 3 — A1, A3, A4, P3, P7. Five papers, four groups, nobody studying it
directly.
**Claim under test:** pointwise forecast error and attractor-statistics error diverge as the
horizon grows.

**Design.** Reuses R4's trained models — **no new training**. For each model and each horizon,
compute a pointwise metric (sMAPE / NRMSE in Lyapunov times) and structural metrics
(correlation dimension of the predicted trajectory, power-spectrum error, KL divergence of the
invariant density). Plot both against horizon and against capacity.

**Dependent variables:** the horizon at which the two rankings decorrelate; whether a model
that is worse pointwise is better distributionally; whether increasing capacity improves them
at different rates.
**Gives us:** a measurement of the corpus's most convergent unexploited pattern, essentially
for free, on models we already had to train. If the decoupling is large and systematic, that
is a candidate research direction in its own right.
**Cost:** ~30 min of metric computation.
**Depends on:** R4.

---

### R7 — How much of "learning the dynamics" is lookup?
**Source:** A4 (context parroting). Non-negotiable per the design rules in
`03-data-representations.md`.
**Claim under test:** a parameter-free copy baseline is competitive with trained models on
chaotic systems.

**Design.** Implement context parroting / nearest-neighbour analogue prediction: find the
closest matching segment in the context window and copy its continuation. No training. Run on
the same systems and horizons as R3 and R4 and compare directly.

**Dependent variables:** parroting error vs trained-model error per system and per horizon; the
fraction of systems where parroting wins; the context length at which it starts to win.
**Gives us:** the floor under every other claim in the project. If parroting matches our
trained surrogates, then our models are doing recall, not rule-learning, and any leap-horizon
claim we make is unfounded. This must be known **before**, not after, we design the spike.
**Cost:** ~30 min. No training.
**Depends on:** R0.

---

### R8 — Direct horizon-conditioned prediction vs autoregressive rollout
**Source:** A6 (Hybrid Neural World Models), P8 (shared architecture-invariant rollout
failure).
**Claim under test:** direct prediction of `x(t+H)` avoids the error compounding that all
architectures suffer under rollout — and the advantage is system-dependent.

**Design.** Same 6 systems. Two training modes at matched parameter count and matched data:
(a) one-step model rolled out H times; (b) a single model conditioned on continuous horizon H.
3 seeds each. Evaluate both at a ladder of horizons.

**Dependent variables:** error vs horizon curves for both modes; the crossover horizon; whether
the crossover location correlates with any R2 instrument.
**Gives us:** the quantitative form of the question A6 settled qualitatively — *for which
systems, and how far*. Also a direct probe of P8's unexplained shared rollout failure: if the
direct model does not suffer it, the failure is compounding; if it does, it is a system
ceiling. **That distinction is the whole Gilpin-vs-Duraisamy dispute, tested cheaply.**
**Cost:** ~3 h.
**Depends on:** R0, R4.

---

## Execution order

```
R0  →  R2  →  R7                (near-free, run first, ~2 h total)
    →  R1                       (independent, background, ~2 h)
    →  R3                       (~2 h)
    →  R4  →  R6, R5, R8        (R4 is the long pole, ~6 h; R6 free off it)
```

Run R4 in the background while R1/R3 analysis is written up.

---

## Directory layout

```
all-spikes/00-replications/
  PLAN.md              # this file
  NOTES.md             # running log for this spike: decisions, surprises, dead ends
  RESULTS.md           # calibration report — one section per experiment, written as each lands
  src/
    common/
      __init__.py
      systems.py       # dysts wrappers + local fallback integrators; timescale alignment
      metrics.py       # WPE, permutation entropy, 0-1 test K, sMAPE, VPT, corr. dimension,
                       # spectral entropy, FSLE, invariant-density KL
      models.py        # small MLP / horizon-conditioned MLP factories, param-count helpers
      train.py         # seeded fit loop; returns a metrics dict, writes nothing itself
      runlog.py        # run manifest + structured logging (see below)
      scaling.py       # P5 scaling-form fit: exponent, floor, confidence intervals
    r1_coarse_grain/
    r2_instruments/
    r3_wpe_vs_error/
    r4_capacity_data/
    r5_nonmonotone/
    r6_pointwise_vs_dist/
    r7_parroting/
    r8_direct_vs_rollout/
  configs/
    r1_eca_scan.yaml
    r2_instruments.yaml
    ...                # one config per experiment; a config fully determines a run
  results/
    r1/ r2/ ... r8/    # metrics as CSV + figures as PNG/PDF. Committed.
  logs/
    <run_id>.jsonl     # one structured log per run. Committed.
```

**Rules.**
- `src/common/` holds everything shared. An experiment directory holds only what is specific
  to it. If two experiments need the same function it moves to `common/`.
- Configs are data, not code. A run is `python -m src.r4_capacity_data.run --config
  configs/r4_capacity_data.yaml`. No hard-coded paths, no notebook-only results.
- `results/` holds derived artefacts small enough to commit — metrics tables and figures.
  Raw trajectories and checkpoints are regenerable and are gitignored.
- Nothing in `results/` is edited by hand. If a number is wrong, the config or the code is
  wrong.

## Run identity and logging

Every run writes exactly one `logs/<run_id>.jsonl`. First line is the manifest:

```json
{
  "record": "manifest",
  "run_id": "r4_capacity_data__20260907T1412__a3f9c1",
  "experiment": "r4_capacity_data",
  "started_utc": "2026-09-07T14:12:03Z",
  "git_sha": "6436f31",
  "git_dirty": false,
  "config_path": "configs/r4_capacity_data.yaml",
  "config_hash": "sha256:...",
  "config": { "...": "the full resolved config, inlined" },
  "seed": 0,
  "env": {"python": "3.12.7", "torch": "2.8.0", "numpy": "1.26.4",
          "threads": 2, "cuda": false}
}
```

Then one line per unit of work:

```json
{"record": "result", "system": "Lorenz", "width": 32, "n_train": 5000,
 "seed": 0, "metrics": {"smape": 0.041, "vpt_lyap": 3.7}, "wall_s": 38.2}
```

and a final line:

```json
{"record": "summary", "n_units": 504, "failed": 0, "wall_s": 21140,
 "finished_utc": "2026-09-07T20:04:23Z"}
```

`run_id` = `<experiment>__<UTC timestamp>__<first 6 of config hash>`. It appears in the log
filename, in every results row, and in every figure caption, so any number in `RESULTS.md`
traces back to a config and a commit.

**Rerun policy:** reruns get a new `run_id`; logs are never overwritten. If a config changes,
the hash changes, and the old results stay valid for the old config.

## What "done" looks like

`RESULTS.md` with one section per experiment, each stating: what was run, the numbers, whether
the source paper's claim reproduced, how much the result moved across seeds, and **one line on
what it means for our direction**. Plus an updated threat register in `NOTES.md`.

Then — and only then — the three candidate directions go to the user for selection.
