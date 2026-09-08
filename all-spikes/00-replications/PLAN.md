# Replication battery — end-to-end tests of what the literature actually gives us

Created 2026-09-07. Revised 2026-09-07 (costs re-derived from measurements; plain-language
explanations added). **Closed out 2026-09-08 — everything here has been run.**

> **This document is the plan as it was written, kept for the record.** What actually happened
> is in [RESULTS.md](RESULTS.md); how the machinery works is in [METHODS.md](METHODS.md). The
> section below tracks what each planned experiment turned into, including the three that had
> to be rerun and the one that could not be delivered. The plan itself is left unedited so the
> difference between what we intended and what we found stays visible.

---

## What actually happened to each of these

| planned | outcome |
|---|---|
| **R0** environment probe | Ran. 124/129 systems usable. Found our cost model **14× optimistic** and the library's own solver unusable; both fixed before anything else ran. |
| **R1** coarse-graining | Ran, 1,698 checks. Reproduced two published transitions exactly. Found reducibility is **scale-indexed, not binary** (36% → 59% → 79% by block size) and stable under a 100× search increase. We **overstated** the search-effort conclusion at first and corrected it: past block size 4 brute force covers 1 part in 10¹⁴ and says nothing. |
| **R2** instrument redundancy | Ran, 45 systems, later extended to all 124. Answer: **~5–6 independent axes**, so the project's assumption survives. Also produced the unplanned **R2b** — the statistics were reading our sampling rate, and the fix *reorders* systems rather than shifting them. |
| **R3** cheap predictor | Ran — and the result was **a mirage** (R²=0.59 → 0.0001 once seven clock-carrying systems were removed). Rerun as **R3b** (30 clean systems) and again as **N3** (all 108). |
| **R4** capacity × data | Ran and **could not answer its own question** — task too easy, ruler too short. Rerun as **R4b** (fixed) and **R4c** (18 systems), which killed our own best explanation. |
| **R5** non-monotone capacity | Ran. The published claim **does not reproduce** at any of 6 tolerances on either of 2 map families. |
| **R6** pointwise vs structural | Ran, free off R4b. Capacity buys short-horizon accuracy (+0.84) and **nothing structural** (+0.06 to +0.16). |
| **R7** copying baseline | Folded into R3/R3b/N3 as planned. The single most-repeated finding in the battery: **the trained network ties with copying**, 46/108, median ratio exactly 1.00. |
| **R8** direct vs rollout | Ran, then rerun as **R8b** with a long enough ruler and a fairness fix. Direct prediction fails **earlier**, so the long-horizon wall is an **information limit**, not error accumulation. |

### Added after the plan was written

| | why | outcome |
|---|---|---|
| **R9** iterative refinement | User question: can more passes fix a wrong answer? | Helps for ~3 passes, then **reliably degrades**; never improves beyond its training budget. |
| **N3** predictor at full scale | 27% explained on 30 systems needed more power | Holds and strengthens to **~32%**; corrected our claim that the Lyapunov exponent is anti-predictive. |
| **N5** batch-size sweep | 256 was convention, tested only upward | **256 was suboptimal** (1.66 vs 1.07 relative error); our earlier single-seed check was underpowered. |
| **N2** observational noise | Everything assumed perfect measurement | Surrogates cope; **the predictor dies at 1% noise**. And at 20% noise the network finally beats copying — its real advantage is **denoising**. |
| **N6** coverage density | Asked what the copying tie is a property *of* | **It is a property of the benchmark.** Coverage predicts the network's advantage (rho +0.56, n=107, controls passed); it moves copying's skill (-0.44) and not the network's (-0.02). Costs no training. |
| **N1** Kuramoto–Sivashinsky | The largest coverage gap: no spatially extended system | **Not delivered.** Integrator destabilises; canonical published parameters fail the same way, so the fault is ours. Parked with a full record. |

### Kill criteria, and whether they fired

Written down before the runs, which is what stops the goalposts moving.

| criterion | fired? |
|---|---|
| R1: reducible fraction moves with search width → CA substrate is dead | **No** — moved by zero under a 100× search increase |
| R2: statistics collapse to 1–2 dimensions → nothing to predict | **No** — 5.62 effective dimensions |
| R3: cheap statistics explain ≥85% → learned predictor is pointless | **No** — 32% at best, so two thirds remains open |

All three passed. The direction was not killed on its own terms — it was **reshaped by N2**,
which was not a pre-registered criterion because we had not thought to ask about noise.

---

## Why this exists

Reading a paper tells you what the authors claim. Running it tells you what the measurement
is actually like to work with — how noisy it is, whether it changes when you change a
threshold, whether it survives a different random seed, and whether the number in the abstract
means what it sounds like.

We are about to pick a research question that will depend on several of these measurements.
Before that, we run a cheap version of each one and find out which are solid and which are not.

This produces no new scientific claim. It produces a **calibration report**: for each tool,
what it costs, what it measures, how much it wobbles, and whether it is safe to build on.

Every experiment is designed so that **both possible outcomes are useful**. If a measurement
turns out to be unreliable, that removes a candidate direction early — which is the point.

---

## Glossary (plain meanings, used consistently below)

| Term | Plain meaning |
|---|---|
| **System** | A set of equations that produces a trajectory over time. Lorenz, a pendulum, a fluid. We generate our own data from these, so we have unlimited clean data and know the ground truth. |
| **Surrogate** | A neural network trained to imitate the equations. Input: the state now. Output: the state later. Much faster than solving the equations, if it works. |
| **Rollout** | Using a one-step surrogate repeatedly: feed its own output back in to get step 2, then step 3, and so on. Errors compound. |
| **Autoregressive** | The same thing as rollout. Predicting one step at a time, feeding output back in. |
| **Direct prediction** | Skipping the loop: one forward pass that jumps straight to step 63. |
| **Lyapunov time** | The natural clock of a chaotic system — roughly how long it takes a small error to grow by a factor of e. Measuring horizons in Lyapunov times lets us compare a fast system against a slow one fairly. |
| **Capacity** | How big the model is. Here, parameter count. |
| **Saturation / error floor** | When making the model bigger and giving it more data stops helping, error levels off at some value. That value is the floor. Whether a floor exists, and whether it differs per system, is the central question of this project. |
| **Scaling exponent** | The slope on a log-log plot of error against model size or data size. Theory (P5) predicts the slope is set by the geometry of the data, not by the architecture. |
| **Seed** | The random number that decides the initial weights and data shuffling. Same config, different seed, different result. If an effect vanishes when you change the seed, it was noise. |
| **Weighted permutation entropy (WPE)** | A cheap statistic computed directly from a time series, no model needed. It looks at the *ordering* of nearby values (is this triple going up-down-up?) and measures how varied those orderings are. Low = repetitive and predictable. High = disordered. Robust to noise because it only uses ordering, not magnitudes. |
| **Coarse-graining** | Describing a system at a blurrier level — grouping cells into blocks, then asking whether the blurry description follows its own self-contained rule. If it does, you can predict the blurry version without simulating the detail. |
| **Attractor** | The shape a chaotic trajectory traces out in state space over long times. Two forecasts can both be wrong about *where* the system is right now while one gets the attractor's shape right and the other does not. |
| **Reducible (of a CA rule)** | A coarse-grained description exists that follows its own consistent rule. |

---

## How to read the cost numbers

The earlier version of this plan carried padded guesses. They are now derived from the two
measurements in `ENVIRONMENT.md`, and most were **4–6× too high**.

The calibration:

- Measured: a 133,000-parameter MLP, 50,000 samples, 20 epochs = **17 s**. That works out to
  about **47 GFLOP/s** on these two cores, which is a believable number for well-batched matrix
  multiplication on this CPU.
- So training time ≈ `1.28e-10 × parameters × samples × epochs` seconds, **when the model is
  big enough to keep the cores busy**.
- Below roughly 10,000 parameters that formula stops applying. The cores are no longer the
  bottleneck; the per-batch Python and PyTorch overhead is (~150 microseconds per batch). So
  small models have a **time floor** of about `(samples / 256) × epochs × 150µs`, no matter how
  small you make them.

What that yields per fit, at 200 epochs:

| parameters | 500 samples | 2,000 | 8,000 | 32,000 |
|---|---|---|---|---|
| 1,000 | 0.0 s | 0.2 s | 0.9 s | 3.7 s |
| 10,000 | 0.1 s | 0.5 s | 2.0 s | 8.2 s |
| 100,000 | 1.3 s | 5.1 s | 20.5 s | 81.8 s |

**The important consequence: training is not the bottleneck at our scale.** A single fit is
seconds. The real costs are elsewhere:

1. **Generating trajectories.** 0.28 s per 4,000-point Lorenz trajectory at tight tolerance.
   Fine in bulk, but stiff systems need much smaller steps and could be 10–100× worse. Unknown
   until measured.
2. **FSLE**, which needs ensembles of perturbed trajectory pairs — many integrations per system.
   This is the single most expensive thing in R2.
3. **R1's combinatorics**, which has nothing to do with machine learning. At block size N=4
   there are 65,536 candidate projections × 256 rules × 4,096 neighbourhood triples to check
   — about **69 billion elementwise operations**. That is why R1 is the longest experiment in
   the battery, and it is the one that surprised me.

So the corrected totals:

| | Old guess | Derived | Where the time goes |
|---|---|---|---|
| R0 | min | **~30 min** | installs + validating metrics + a timing probe |
| R1 | 2 h | **1–2 h** | 69e9 elementwise ops, chunked through numpy |
| R2 | 1 h | **~45 min** | FSLE ensembles ≈ 37 min of it |
| R3 | 2 h | **~15 min** | training is only 3.4 min |
| R4 | 6 h | **~1.5 h** | training 1.0 h + data + rollout evaluation |
| R5 | 4 h | **~1.2 h** | training 0.9 h |
| R6 | 30 min | **~15 min** | metrics only, no training |
| R7 | 30 min | **~10 min** | no training |
| R8 | 3 h | **~20 min** | 72 fits |
| **Total** | 15–20 h | **~5–6 h** | |

**R0 includes a timing probe** — run one fit and one trajectory at each scale in the grid and
replace every estimate above with a measurement before committing to the big runs. If the probe
disagrees with the model, the plan gets rescaled, not the schedule.

---

## The battery

Order is by information gained per hour, not by paper number. R0, R2, R7 go first because they
are nearly free.

---

### R0 — Get the tools working and check they give correct answers

**In plain terms.** Before measuring anything, make sure our measuring instruments are correct.
Install the system library, generate some trajectories, and check every statistic we plan to use
against cases where the right answer is already known.

**What we actually do.** Install `dysts` (a public collection of 135 chaotic systems, each with
its properties already computed by the authors). Confirm it generates trajectories on this
laptop. Then write the shared statistics library and validate it: WPE and permutation entropy
should be near 0 for a sine wave and near 1 for white noise; the Lyapunov exponent of the Lorenz
system should come out near the published 0.906; the logistic map at r=4 should give ln 2; the
0-1 chaos test should give K near 1 for Lorenz and near 0 for a periodic orbit. Then run the
timing probe described above.

**Hypothesis being tested.** None — this is instrument calibration. But it does test one
practical assumption: that a 2-core laptop with no GPU can run this substrate at all.

**Why we care.** Every number in every later experiment flows through this code. A sign error in
the entropy calculation would silently invalidate R2, R3 and R6.

**If it goes wrong.** If `dysts` is too slow here, fall back to a hand-written list of ten
low-dimensional systems and record that the system selection is now ours, not a public list —
which weakens the "we did not cherry-pick systems" defence and must be stated.

**Cost: ~30 min.** Blocks everything except R1.

---

### R1 — Is "this system has a simple blurry description" a real property, or an artefact of how hard we looked?

**Source:** P1 (Israeli & Goldenfeld 2004), A12 (their 2006 extension), A13 (Dzwinel & Magiera
2015).

**In plain terms.** Cellular automata are the simplest possible "systems": a row of cells, each
0 or 1, updated by a fixed rule. There are exactly 256 such rules, so we can check all of them.

Israeli and Goldenfeld asked: if I squint — group cells into blocks of N and only look at a
summary of each block — does the blurry version follow its own self-contained rule? They found
240 of the 256 rules do. That is the origin of the idea that even an unpredictable system can
have a predictable coarse description.

But here is the catch. Finding a blurry description means **searching** over ways to summarise a
block. The more ways you try, the more likely you find one that happens to work. So "this system
has a simple coarse description" might be a fact about the system, or it might be a fact about
how long you searched. Nobody has measured which.

**What we actually do.** For all 256 rules and block sizes N = 2, 3, 4: build the blocked
automaton, enumerate ways of summarising a block, and test whether the resulting blurry rule is
consistent (does the same blurry input always give the same blurry output?). Then — the part the
papers did not do — **repeat with search spaces of different widths**: only two-symbol summaries,
then three-symbol, then everything we can afford. Plot the fraction of "reducible" rules against
how wide the search was.

**What we measure.** Count of reducible rules per (block size, search width). The set that
resists. The graph of which rule turns into which.

**Hypothesis being tested.**
- H1 (the papers' implied position): reducibility is a property of the rule. The count should be
  stable when the search space is widened at fixed block size.
- H0 (our worry): reducibility is mostly a property of the search. The count climbs steadily as
  you allow more ways of summarising.
- A12 already tells us something uncomfortable here: the 2006 paper reports that the probability
  of finding a coarse description **approaches 1 as you go to coarser scales**. If that is right,
  H0 is at least partly true and "is it reducible?" is not a yes/no property — it is "at what
  scale does it become reducible?", which is a different question.

**Why we care.** If we ever want to label a system "has a self-contained simple description",
this experiment tells us whether that label means anything. It also gives the base rate: A13
found that at block size 7 only **four** rules stay irreducible — {30, 45, 106, 154}. Four out of
256 is 1.6%. Trying to train a predictor to spot a 1.6% class is a very different, much harder
experiment than people imagine when they propose it.

**If it comes out the other way.** If the count is stable, the label is real and the CA substrate
becomes attractive: it is cheap, exhaustive, and has a provable ground truth, which almost nothing
else in this field does.

**Pre-registered kill criterion.** If the reducible fraction moves by more than a few percent
when the search space is widened at fixed block size, CA-based learnability classification is
dead. We report that and stop.

**Cost: 1–2 h.** No machine learning at all — pure combinatorics. This is the longest experiment
in the battery, because at N=4 it is 65,536 candidate summaries × 256 rules × 4,096 cases to
check, roughly 69 billion elementwise operations, done in chunks through numpy.

---

### R2 — How many genuinely different things are we measuring when we measure "complexity"?

**Source:** P2 (FSLE, ε-entropy), A14 (chaos decision tree, 0-1 test), A15 (WPE), A1 (the dysts
annotations).

**In plain terms.** There is a long list of numbers people compute to say how chaotic or complex
a system is: Lyapunov exponent, entropy, fractal dimension, permutation entropy, and more. The
whole project assumes these are **different** measurements capturing **different** things — that
is what makes "learnability is a separate axis from chaoticity" a meaningful sentence.

But they might mostly be the same number in different clothes. If eight statistics are all 0.95
correlated with each other, then "system properties" is effectively a one-dimensional feature
space, and any predictor built on it has very little to work with.

**What we actually do.** Take about 40 systems. Generate aligned trajectories (aligned means
resampled so each system gets the same number of points per natural oscillation, so a fast system
and a slow system are compared fairly). Compute, with **no model training at all**: Lyapunov
exponent (both from the published annotation and re-estimated ourselves), Kaplan-Yorke dimension,
correlation dimension, multiscale entropy, weighted permutation entropy, the modified 0-1 chaos
statistic K, spectral entropy, and the finite-size Lyapunov curve λ(δ) across a range of
perturbation sizes.

**What we measure.** The full table of rank correlations between every pair of statistics. How
many principal components are needed to explain 90% of the variation across systems. And whether
the FSLE curve shows a plateau, and at what perturbation size.

**Hypothesis being tested.**
- H1: these statistics are genuinely multi-dimensional (say 3+ effective dimensions). Then
  "learnability is not the same as chaoticity" has room to be true and measurable.
- H0: they collapse to 1–2 dimensions. Then there is almost no residual for a learned predictor
  to explain, and the ceiling on the whole meta-model idea is low.

Second, smaller hypothesis: P2's FSLE plateau is supposed to mark the scale where an emergent
simpler description takes over. We are checking whether that plateau is even visible in these
systems, because if it is, it is a ready-made way to measure "does a self-contained coarse
description exist" for continuous systems — the R1 question, but for ODEs.

**Why we care.** This is the cheapest possible check on a load-bearing assumption, and it costs
under an hour. If it fails, we have saved four days.

**Cost: ~45 min.** No training. Most of the time — about 37 min — is the FSLE, which needs many
perturbed trajectory pairs per system.

---

### R3 — Does a 20-line statistic from 2002 already predict how well our neural network will do?

**Source:** A15 (Garland, James & Bradley 2014), A16 (Pennekamp 2019), A18, A1.

**In plain terms.** Weighted permutation entropy is computed directly from a time series in a few
lines of code, with no model and no training. Three separate papers have shown it tracks how
accurately *some* forecasting method can predict that series: Garland used 120 series and 4
classical forecasters, Pennekamp used 461 ecological series, and a 2025/26 paper used 4,700
trained deep models with a related complexity measure.

None of them tested it on a neural surrogate of a dynamical system, which is our setting. So we
test it. This is the honest bar our whole project has to clear: if a free statistic from 2002
already explains most of the variation in our model's error, there is nothing left for a learned
predictor to add.

**What we actually do.** About 20 systems. One fixed small MLP that predicts the next state from
the current one — same size, same data budget, same training recipe for every system, 5 seeds
each. Measure how far ahead it stays accurate, in Lyapunov times. Then check how well that is
predicted by: WPE alone, Lyapunov exponent alone, both together, and the full R2 feature set.

We also reproduce Garland's actual proposed use, which is subtler than a correlation. He fits a
curve of WPE against error and reads **distance from the curve** as a signal of
*method-data mismatch*: a point well off the curve means the series contains more exploitable
structure than this particular model is exploiting. His own caution, which we adopt: WPE tells you
*some* method could do better, not *which* method.

**What we measure.** R² and rank correlation for each predictor set. The fitted curve. Per-system
residuals.

**Hypothesis being tested.**
- H1 (ours, hopeful): cheap statistics explain some but not most of the variation, leaving a
  substantial residual that a learned model could capture.
- H0 (the threat): WPE plus Lyapunov exponent explain nearly all of it. Then a learned predictor
  is unnecessary.
- H2 (the interesting middle, and the most likely): Lyapunov exponent is weak but WPE is strong.
  That would be a clean and publishable refinement of Gilpin's result — his finding that
  forecast skill decorrelates from the Lyapunov exponent would become *he was using the wrong
  statistic*, not *no statistic works*.

**Pre-registered kill criterion.** If WPE + λ explain 85% or more of the variance in surrogate
error across systems, the learned-predictor direction is not worth pursuing. We report the
negative and pick a different direction.

**Cost: ~15 min.** Training is only 3.4 min of that; the rest is data generation and evaluation.

---

### R4 — Does a surrogate stop improving, and does the stopping point depend on the system?

**Source:** the central dispute. A1 (Gilpin: what limits us is model size and data) vs A5
(Duraisamy: there are hard ceilings no amount of training removes) vs P3 (Chen: neither — the
relationship is not even monotone). P5 supplies the statistical machinery.

**In plain terms.** This is the crux of the whole project.

Take one system. Train a tiny network, then a bigger one, then a bigger one — over about two
orders of magnitude. Do the same with the amount of training data. Two things can happen:

- **Error keeps falling.** Then the limit is us, not the system. Buy a bigger computer. Gilpin is
  right, and the idea of a per-system predictable "ceiling" is dead.
- **Error levels off at some value, and that value is different for different systems.** Then
  there is something intrinsic about each system that caps how well any surrogate can do, and
  measuring it in advance becomes a sensible goal.

Nobody has run this properly. Gilpin's stand-in for model size was **training wall-clock time**,
which correlated with error at only −0.31, and his data sweep was averaged **across** systems
rather than run per system. So his conclusion is a conclusion, not a controlled measurement.

**What we actually do.** 6 systems, chosen in advance to spread across the R2 feature space (not
chosen after seeing results). Grid: 7 model sizes on a geometric ladder from about 1,000 to
100,000 parameters × 4 training-set sizes from 500 to 32,000 samples × 3 seeds = **504 fits**.
For each system, fit the P5 scaling form — a power law with an offset — and extract two numbers
with confidence intervals: the **slope** (how fast error falls as you scale) and the **floor**
(what it is heading towards).

**What we measure.** Slope and floor per system. Whether the floors separate across systems by
more than seed-to-seed noise. And whether the slope matches P5's prediction that it should be
about `1/d`, where `d` is the dimension of the data's underlying shape — for a chaotic system that
is the attractor's fractal dimension, which `dysts` already gives us.

**Hypothesis being tested.**
- H1: floors exist and differ between systems by more than noise. Supports an intrinsic ceiling.
- H0: no floor within our range — error still falling at the largest size we can run. Supports
  Gilpin.
- H2 (the extra one, and the sharpest thing in this plan): **slope ≈ 1 / fractal dimension.**
  P5 derives this for general data manifolds. For a dynamical system the manifold is the
  attractor. Gilpin checked whether classical invariants predict *accuracy* and found they
  barely do. **Nobody checked whether they predict the slope.** Accuracy and slope are different
  objects, and the theory makes a prediction about the second one.

**Why we care.** Three published papers disagree about this and none of them ran the experiment
that separates them. Whichever way it comes out, we have measured something contested.

**The caveat we state up front, every time.** Seeing a floor up to 100,000 parameters does not
prove there is a floor at a billion. We report the trend and the scale at which we observed it,
and never more than that. This is the single biggest threat to the credibility of anything we
conclude, and it is worse on a 2-core laptop than it would be on a cluster.

**Cost: ~1.5 h.** Training is 1.0 h of that; the rest is data generation and rollout evaluation.
This is the long pole, so it runs in the background while R1 and R3 get written up.

---

### R5 — Is the "chaos makes it easier" result real, or is it a threshold artefact?

**Source:** P3 (NNPT). This is the one the user flagged as interesting.

**In plain terms.** NNPT found something counterintuitive. As they made a three-body system more
chaotic, the network size needed to hit 1% accuracy went *up*, peaked in the middle, and then went
**down by 47%** in the fully chaotic regime. Their explanation: once fully chaotic, the fine
detail becomes effectively noise, and noise does not need capacity to represent — only the smooth
statistical part is left to learn.

That is an interesting claim, and it is testable on our hardware, because their models were tiny
(1,000–2,000 parameters). The user's point is right: small models are an advantage for us, not a
limitation.

The problem is not model size. It is that the 47% drop is measured across a **single architectural
step** — 3 layers of 32 units down to 2 layers of 32 units — with no reported seed counts, at a
**single** accuracy threshold of 1%. "The smallest model that reaches a threshold" is a
high-variance quantity: one unlucky random initialisation moves it a whole step. Two points and
one threshold is not a curve.

**What we actually do.** Use the **Chirikov standard map** and sweep its chaos parameter K. This
matters: NNPT observed that their transition matched Chirikov's resonance-overlap criterion, so
using the standard map tests their claimed *mechanism* directly instead of by analogy. 8 values of
K × 9 model widths on a fine geometric grid × 5 seeds = 360 fits.

Four things they did not do:
1. **A near-continuous size axis** — vary width, not depth, so the capacity axis has 9 points
   instead of 2.
2. **Seeds at every point**, reporting the spread of the threshold-crossing size rather than the
   single luckiest run.
3. **Four thresholds instead of one** — 0.5%, 2%, 1%, 5%. This costs nothing, because they are
   all read off the same already-computed error curves. **If the non-monotonicity only appears at
   1%, it is an artefact.** This is the highest-value cheap test in the whole battery.
4. **Both dependent variables** — "smallest model that reaches a threshold" and "the error floor
   from R4's fitting". These are different quantities and can move in opposite directions: a
   system can need more capacity to hit 1% while having a *lower* eventual floor. Treating them
   as one thing is a confound waiting to happen.

Plus a **control**: NNPT trains on the residual left after subtracting a known exact solution.
Whether the non-monotonicity belongs to the system or to that residual is untested. We run both.

**Hypothesis being tested.**
- H1: the non-monotone capacity requirement is real and survives seeds, a finer grid, and all
  four thresholds.
- H0: it is a threshold artefact, or it disappears inside seed variance.
- H2: it is real but belongs to the residual-subtraction setup rather than to the system.

**Why we care.** If H1 holds, "how chaotic is it" predicts model difficulty in a shape nobody
expects, and that is genuinely surprising to experts — which is the bar the research philosophy
sets. If H0 holds, we have cheaply retired a distracting result.

**Honest assessment.** Even done well, this is replication plus extension. On its own it is
workshop-grade, not headline-grade. It becomes headline-grade only if the effect generalises to
other system families **and** the location of the capacity peak can be predicted in advance from
a cheap statistic — which is the same shape as R3, and worth noticing.

**Cost: ~1.2 h.**

---

### R6 — When does "wrong about where it is" stop meaning "wrong about what it is"?

**Source:** five papers converging — A1, A3, A4, P3, P7 — and nobody studying it directly.

**In plain terms.** There are two different ways to be right about a chaotic system.

You can be right about **where it is now**: predict the exact state at step 100. For a chaotic
system this becomes impossible fairly quickly no matter what you do.

Or you can be right about **what it is**: reproduce the shape of the attractor, the spectrum, the
distribution of states over long times. This can survive long after point-by-point prediction has
completely failed.

Five separate papers noticed this in passing. Zhang and Gilpin found foundation models keep the
attractor's geometry *after* their point forecasts fail. Gilpin found NBEATS ranks best on fractal
dimension while ranking differently on pointwise error. Context parroting works because copying
preserves the statistics. NNPT's ergodic-smoothing story predicts exactly this. REALM found models
with correlation above 0.8 that still get the detonation cell size wrong. Nobody has made the
split itself the object of study.

**What we actually do. No new training** — this reuses R4's already-trained models. For each model
and each horizon, compute a pointwise error (sMAPE, in Lyapunov times) and structural errors
(correlation dimension of the predicted trajectory, power-spectrum error, KL divergence between
the predicted and true distribution of states).

**What we measure.** The horizon at which the two rankings stop agreeing. Whether a model that is
worse pointwise is better structurally. Whether increasing model size improves the two at
different rates.

**Hypothesis being tested.** H1: the two measures diverge systematically, and the horizon at
which they diverge is system-dependent. H0: they track each other and the distinction does not
matter in practice.

**Why we care.** If H1 holds, then **which metric you choose determines your conclusion** — and
since Gilpin, Duraisamy and Chen use different metrics, some of their disagreement may be a metric
artefact rather than a real dispute. That would be a genuinely useful finding, obtained for free
from models we had to train anyway.

**Cost: ~15 min.** Metric computation only.

---

### R7 — How much of "the model learned the dynamics" is actually just copying?

**Source:** A4 (context parroting). Non-negotiable.

**In plain terms.** A chaotic trajectory revisits similar states over and over. So here is a
forecasting method with no parameters and no training: look back through the history for the
segment that most resembles the recent past, then copy what happened next.

Zhang and Gilpin showed this beats leading time-series foundation models on chaotic systems, at a
tiny fraction of the cost. Which means a lot of the "models can forecast chaos for 20 Lyapunov
times" literature may be measuring **how repetitive the attractor is**, not what the model
learned.

**What we actually do.** Implement it — find the nearest matching segment in the context window,
copy its continuation. No training. Run it on the same systems and horizons as R3 and R4 and
compare directly against the trained networks.

**What we measure.** Copying error vs trained-model error, per system and per horizon. The
fraction of systems where copying wins. How long the context has to be before copying starts
winning.

**Hypothesis being tested.** H1: our trained models clearly beat copying, so they are learning
something. H0: copying matches them, so they are doing recall and any claim about "learning the
underlying rule" is unfounded.

**Why we care, and why it runs first.** This is the floor under every other claim in the project.
If copying matches our surrogates, every leap-horizon statement downstream is empty. It costs ten
minutes and it must be known **before** we design the spike, not after.

**Cost: ~10 min.** No training.

---

### R8 — Is it better to take 63 small steps or one big jump?

**Source:** A6 (Hybrid Neural World Models — the Lossfunk lab paper), P8 (all three architectures
fail at long rollout and nobody explains why).

**In plain terms.** This is the user's `t → t+63` question, made quantitative.

Two ways to predict 63 steps ahead. **Rollout:** predict one step, feed the answer back in, repeat
63 times. Each prediction inherits the previous error, so errors compound. **Direct:** one network
that takes "how far ahead" as an input and jumps straight there in a single forward pass. No
compounding, but it has to learn a separate effective map for every horizon.

A6 already showed direct prediction works, with 26–72× speedups over a classical solver. So the
yes/no question is settled. The open question is *for which systems, and how far*.

**What we actually do.** Same 6 systems as R4. Two training modes at matched parameter count and
matched data: (a) a one-step model rolled out, (b) a single model conditioned on a continuous
horizon input. 3 seeds each. Evaluate both at a ladder of horizons.

**What we measure.** Error-vs-horizon curves for both modes. The horizon where they cross. Whether
the crossing point correlates with any statistic from R2.

**Hypothesis being tested.** H1: direct prediction wins at long horizons, and the crossover
horizon is system-dependent and predictable from cheap statistics.

**But the reason this is the sharpest experiment in the battery is a different hypothesis.** P8
found that all three architectures fail badly at long rollout on Navier-Stokes and
Kuramoto-Sivashinsky, and offered only a bound, not an explanation. Two candidate explanations:

- The failure is **error compounding** — a property of the rollout procedure.
- The failure is a **system ceiling** — the information simply is not there.

A direct model does not compound errors. So: **if the direct model does not suffer the same
failure, the cause was compounding. If it fails just as badly, the cause is the system.** That
distinction is exactly the Gilpin-versus-Duraisamy dispute, and this experiment separates them in
20 minutes.

**Cost: ~20 min.** 72 fits.

---

## Execution order

```
R0                    ~30 min   installs, metric validation, timing probe  [blocks all but R1]
  ├─ R2               ~45 min   are the complexity statistics redundant?
  ├─ R7               ~10 min   does copying beat our models?
  └─ R1               1-2 h     is reducibility real or a search artefact?  [independent, background]
       R3             ~15 min   does WPE already predict our error?        [needs R0, R2]
       R4             ~1.5 h    does skill saturate?                      [needs R0, R2]
         ├─ R6        ~15 min   pointwise vs structural                   [free off R4]
         ├─ R5        ~1.2 h    is the non-monotonicity real?             [shares R4 fitting code]
         └─ R8        ~20 min   direct vs rollout                         [needs R4]
```

Total **~5–6 h**. R1 runs in the background while the R2/R3/R7 results get written up. R4 runs in
the background while R1 gets written up.

---

## Directory layout

```
all-spikes/00-replications/
  PLAN.md              # this file
  NOTES.md             # running log: decisions, surprises, dead ends
  RESULTS.md           # the calibration report, one section per experiment as each lands
  src/
    common/
      __init__.py
      systems.py       # dysts wrappers + local fallback integrators; timescale alignment
      metrics.py       # WPE, permutation entropy, 0-1 test K, sMAPE, VPT, correlation
                       # dimension, spectral entropy, FSLE, invariant-density KL
      models.py        # small MLP / horizon-conditioned MLP factories, param-count helpers
      train.py         # seeded fit loop; returns a metrics dict, writes nothing itself
      runlog.py        # run manifest + structured logging  [written, self-check passes]
      scaling.py       # P5 scaling-form fit: slope, floor, confidence intervals
    r1_coarse_grain/  r2_instruments/  r3_wpe_vs_error/  r4_capacity_data/
    r5_nonmonotone/   r6_pointwise_vs_dist/  r7_parroting/  r8_direct_vs_rollout/
  configs/             # one YAML per experiment; a config fully determines a run
  results/r1..r8/      # metrics as CSV + figures. Committed.
  logs/<run_id>.jsonl  # one structured log per run. Committed.
```

**Rules.**
- Anything shared lives in `src/common/`. If two experiments need the same function it moves
  there.
- Configs are data, not code. A run is
  `python -m src.r4_capacity_data.run --config configs/r4_capacity_data.yaml`.
  No hard-coded paths, no results that exist only inside a notebook.
- `results/` holds only artefacts small enough to commit — metric tables and figures. Raw
  trajectories and checkpoints are regenerable and are gitignored.
- Nothing in `results/` is edited by hand. If a number is wrong, the config or the code is wrong.

## Run identity and logging

Every run writes exactly one `logs/<run_id>.jsonl`. First line is the manifest:

```json
{
  "record": "manifest",
  "run_id": "r4_capacity_data__20260907T1412__a3f9c1",
  "experiment": "r4_capacity_data",
  "started_utc": "2026-09-07T14:12:03Z",
  "git_sha": "a05a69b",
  "git_dirty": false,
  "config_path": "configs/r4_capacity_data.yaml",
  "config_hash": "sha256:...",
  "config": { "...": "the full resolved config, inlined" },
  "seed": 0,
  "env": {"python": "3.12.7", "torch": "2.8.0", "numpy": "1.26.4",
          "threads": 2, "cuda": false}
}
```

Then one line per unit of work, and a summary line at the end:

```json
{"record": "result", "system": "Lorenz", "width": 32, "n_train": 5000,
 "seed": 0, "metrics": {"smape": 0.041, "vpt_lyap": 3.7}, "wall_s": 38.2}
{"record": "summary", "n_units": 504, "failed": 0, "wall_s": 5400,
 "finished_utc": "2026-09-07T15:42:11Z"}
```

`run_id` = `<experiment>__<UTC timestamp>__<first 6 of the config hash>`. It goes in the log
filename, in every results row, and in every figure caption, so any number in the write-up traces
back to a specific config and a specific commit.

**Rerun policy.** A rerun gets a new `run_id`; logs are never overwritten. Change a config and the
hash changes, so old results stay valid for the old config.

## What "done" looks like

`RESULTS.md`, one section per experiment, each stating: what was run, the numbers, whether the
source paper's claim reproduced, how much the result moved across seeds, and **one line on what it
means for our direction**. Plus an updated threat register in `NOTES.md`.

Then, and only then, the three candidate research directions go to the user for selection.
