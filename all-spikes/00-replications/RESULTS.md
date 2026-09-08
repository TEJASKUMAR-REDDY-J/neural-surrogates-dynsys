# What we found

This is the plain-language record of the replication battery: what we asked, what we did,
what happened, and what it means. Technical detail sits underneath each story, not instead
of it.

Plan and hypotheses: [PLAN.md](PLAN.md). Raw numbers: `results/`. Figures:
`results/analysis/`. Run logs: `logs/`.

**Status:** R0, R2, R3 and R8 done. R4 running. R5, R1 and two follow-up runs queued.

---

## The short version

| | What we asked | What we found |
|---|---|---|
| **R0** | Can this laptop run any of this? | Yes. 124 of 129 systems work. Our cost estimates were 14× too optimistic and got fixed. |
| **R2** | Are the ways of measuring "complexity" all secretly the same number? | **No** — about 5–6 genuinely independent ones. Good news: there is something to predict. |
| **R2b** | *(unplanned)* Are those measurements even trustworthy? | **Not as normally computed.** A standard setup made 40 of 45 chaotic systems look non-chaotic. |
| **R3** | Does a cheap statistic already predict how well a network will do? | Looked like a yes (R²=0.59). **It was a mirage.** Fix the data flaw and it drops to 0.002. |
| **R8** | Is it better to take 63 small steps or one big jump? | **Small steps win**, on 5 of 6 systems. But our test was too short to be the real test — rerunning. |
| **R4** | Does a model stop improving no matter how big you make it? | *Running.* This is the crux. |
| **R5** | Does more chaos really need a *smaller* model? | *Queued.* |
| **R1** | Is "this system has a simple summary" a real property? | *Queued.* Code already reproduces two published results exactly. |

Two of these — R2b and R3 — are results we did not plan for. Both came from noticing that a
number looked wrong and chasing it. That is what this battery is for.

---

## R0 — Can this laptop do the work at all?

**In one sentence:** yes, but two things we assumed turned out to be false, and finding
that early saved the whole schedule.

### What we did

We took a public library of 129 chaotic systems (`dysts`) and tried to generate a
trajectory from each one. Then we timed how long it takes to train a small network at
different sizes.

### What happened

**124 of 129 systems worked.** Five didn't: three were too slow, one was degenerate, one
was missing a data field in the library itself.

**Surprise 1: the library's own solver was unusably slow.** It uses a very cautious
integration method by default. On this laptop:

```
dysts' default solver:   2,000 points in 41 seconds
our own solver:         20,000 points in  7 seconds
```

That is roughly **57× faster per point** — and still far more accurate than the neural
network we are about to fit, so nothing is lost. We now take the *equations* and the
*published system properties* from the library, but do the integration ourselves.

**Surprise 2: one system could hang forever.** The integration routine cannot be
interrupted from outside. One stiff system just sat there. We added a stopwatch *inside*
the equations themselves — every time the solver asks "what is the rate of change here?",
it also checks whether time is up. Simple, and it cannot be evaded.

**Surprise 3: our cost model was badly wrong.**

| model size | 500 samples | 8,000 | 32,000 |
|---|---|---|---|
| 1,011 params | 3.8 s | 11.1 s | 51.2 s |
| 9,987 params | 1.0 s | 13.5 s | 55.2 s |
| 99,843 params | 1.9 s | 28.3 s | **156.7 s** |

We had predicted 3.7 seconds for that 1,011 × 32,000 cell. It took 51.2 — **14× over**.

Look at the first two rows: a model **ten times bigger** takes *the same time*. That is the
tell. The computer is not busy doing arithmetic; it is busy with bookkeeping between steps.
Cost is driven by **how many times you update the model**, not how big the model is.

### What we did about it

Two speed decisions followed, and one of them we deliberately refused:

- **Threading buys nothing.** Same three fits: 63.4 s using two CPU threads, 62.4 s using
  one. So instead we run **two separate single-threaded workers side by side**, which gives
  about **1.7×**. Every training phase now runs as two shards.
- **Bigger batches were rejected.** Using batches of 1024 instead of 256 cut a fit from
  30.9 s to 7.3 s — over 4× faster. But the final training error went from 0.0000026 to
  0.000011, four times worse. R4's entire job is to measure *the error a model cannot get
  below*. If we make our own optimiser sloppier, we manufacture exactly the "there is a hard
  floor here" conclusion we are supposed to be testing. **We took the speed elsewhere.**

---

## R2 — Are all the "complexity" numbers secretly the same number?

**In one sentence:** no, there are about 5–6 genuinely different things being measured, so
the project's central assumption survives.

### Why this matters

The whole proposal rests on a sentence like *"learnability is a different thing from
chaoticity."* That sentence only means something if the numbers people use to describe
systems are actually different from each other. If Lyapunov exponent, entropy, fractal
dimension and the rest are all 95% correlated, then "system properties" is really just one
number wearing different hats, and there is nothing left for a clever predictor to find.

So before doing anything expensive, we checked.

### What we did

Took 45 systems. Computed 11 different descriptive numbers for each. **Trained nothing.**
Then asked: how much do these 11 numbers actually disagree with each other?

### What happened

| | |
|---|---|
| typical correlation between any two statistics | **0.23** (low — they disagree a lot) |
| how many independent directions to explain 90% of the variation | **7 out of 11** |
| effective number of independent dimensions | **5.62** |

So the answer is clearly **no, they are not the same number**. There are roughly 5–6
independent things here. The pre-registered failure case did not happen.

### Three things worth noticing

**1. The Lyapunov exponent is almost unrelated to everything else.** Its strongest
correlation with any other statistic is 0.43. Against attractor dimension it's 0.30.

This is Gilpin's famous "forecast skill decorrelates from the Lyapunov exponent" result —
except we can see it *before training a single model*. It is not that models behave
strangely; it is that the Lyapunov exponent simply does not carry the same information as
the other descriptions of a system.

**2. The "0-1 chaos test" is mostly a spectral measurement (correlation 0.89 with spectral
entropy).** That test is presented in the literature as a *dynamical* test — a way of
asking whether trajectories diverge. On these 45 systems it turns out to be largely
reporting what the frequency spectrum looks like. Useful to know before leaning on it.

**3. Our own correlation-dimension estimate agrees with the library's published value at
0.71.** That is a check on our code, not a finding — but it is the kind of check that
should be reported.

![R2 correlation matrix](results/analysis/r2_spearman_matrix.png)

---

## R2b — The measurements were lying, and here is the story

**In one sentence:** a standard way of preparing the data made 40 of 45 obviously-chaotic
systems look calm, and fixing it changed not just the values but the whole ranking.

**We did not plan this experiment.** It came from looking at a line of output and thinking
"that cannot be right."

### The moment

R2's output scrolled past with this:

```
HyperCai   lambda = 1.678   K = -0.01
```

The Lyapunov exponent of 1.678 says: *this system is strongly chaotic.* The 0-1 chaos test
score of −0.01 says: *this system is not chaotic at all.* Both cannot be true.

### The cause, with a picture

Some of these statistics work by looking at the **order** of nearby values — is this run of
five numbers going up, down, up-down-up? Then they ask how varied those patterns are. Lots
of different patterns means disorder. Few patterns means order.

Now: we sample each system at **100 points per oscillation**, so that fast systems and slow
systems can be compared fairly. That is standard practice. But it means consecutive samples
are *extremely* close together. Zoom in far enough on any wiggly curve and it looks like a
straight line:

```
sampled every point   ->   . . . . . . . . . .     always going up. Looks boring.
                                                    "order" -> low disorder score

sampled every 20th    ->   .     .     .          up, down, up, down. Looks chaotic.
                                 .     .           "disorder" -> high score
```

At delay 1 we were not measuring the system. **We were measuring our own sampling rate.**

### The evidence

Same Lorenz data. Only the sampling gap changes:

| gap between samples | disorder score | chaos test |
|---|---|---|
| 1 | 0.248 | **−0.00** ("not chaotic") |
| 2 | 0.325 | −0.01 |
| **5** | 0.500 | **+1.00** ("chaotic") |
| 10 | 0.679 | +1.00 |
| 20 | 0.865 | +1.00 |
| 40 | 0.905 | +1.00 |

The verdict flips from "not chaotic" to "definitely chaotic" between gap 2 and gap 5, on
**identical data**.

### The fix, and why we kept the broken version too

We now choose the gap per system: wait until the signal has genuinely changed (technically,
until its autocorrelation drops below 1/e), then measure. Toker et al. (2020) include
exactly this step in their published chaos-detection pipeline; we had skipped it.

We record **both** versions, because the difference is itself the result:

| statistic | before fix | after fix | do they even rank systems the same way? |
|---|---|---|---|
| weighted disorder score | 0.170 | 0.641 | **0.14** — barely |
| 0-1 chaos test | 0.097 | 0.774 | **0.24** — barely |
| plain disorder score | 0.220 | 0.659 | 0.59 |

**Before the fix, the chaos test called 5 of 45 systems chaotic. After, it calls 35.**

And the deeper point: a correlation of 0.14 means the before and after versions are
essentially **different variables**. This is not a bias you can subtract out. Any study that
computed these statistics this way was working with something close to noise.

### The part that reaches beyond our code

We measured how long each system takes to "genuinely change." Across the 45 systems it
ranges from **1 step to 229 steps**, median 17. Only 4 systems were not oversampled.

So: sampling at 100 points per oscillation successfully makes systems comparable **in
time**. It does *not* make them comparable **in information** — some systems give you a
fresh piece of information every step, others every 229 steps. That is a criticism of the
alignment convention that the whole chaos-forecasting benchmark is built on, and it fell out
of a debugging session.

![R2 sampling correction](results/analysis/r2_sampling_correction.png)
![R2 autocorrelation times](results/analysis/r2_autocorr_time.png)

---

## R3 — Does a cheap statistic already predict how well our network will do?

**In one sentence:** it looked like a spectacular yes, and then it turned out to be an
artefact of seven systems that carry a clock.

**This is the most valuable result in the battery, and it is a negative one.**

### Why we ran it

If a 20-line statistic you can compute in a second already tells you how well a neural
network will do, then the proposal's "learned predictor" has nothing left to add, and we
should find that out now rather than in four days. We wrote down the kill criterion in
advance: **if cheap statistics explain 85% or more of the variation, stop.**

### What we did

24 systems. One fixed network (~10,000 parameters), same size everywhere. Same amount of
data. 5 different random starts each. Then measure how far ahead each trained model stays
accurate, and see which cheap statistic predicts that.

### What happened first: a very exciting number

| what we used to predict skill | how much of the variation it explained |
|---|---|
| **disorder score (weighted permutation entropy)** | **59%** |
| **largest Lyapunov exponent** | **0.15%** |
| both together | 59% — *the Lyapunov exponent added nothing at all* |

Read at face value, that is a headline. Gilpin's result says "the Lyapunov exponent doesn't
predict forecast skill." Ours would have said: **"right, because it was the wrong statistic
— here is one that works, and yours adds literally zero on top of it."**

### Then we looked closer

Seven of the 24 systems are *driven* — pushed by an outside rhythm, like a swing being
pushed. To handle these, the library adds an extra coordinate that just counts time.

Here is what that coordinate actually looks like on `ForcedVanDerPol`:

```
coordinate 0:   -2.05  ...  +2.05     wiggles      (real physics)
coordinate 1:  -12.98  ... +12.94     wiggles      (real physics)
coordinate 2:  869.80  ... 2693.89    only ever goes UP   <-- a clock
```

That clock breaks two things at once:

1. **It has no disorder.** It counts 869, 870, 871, forever. Its "order pattern" is always
   the same. So it drags the system's average disorder score down toward zero.
2. **It makes prediction impossible by construction.** We train on the early part of the
   trajectory and test on the later part. So every clock value in the test set is a number
   the network has *never seen* — like training a model on years 1990–2000 and testing on
   3000. The network isn't failing to learn dynamics; it's being asked to extrapolate.

So those seven systems get **low disorder scores** and **terrible prediction scores** for
the same silly reason. That alone creates the correlation.

Here is the smoking gun — the seven systems with a clock are *exactly* the seven where even
dumb copying fails immediately:

| group | how many | typical skill | typical disorder | copying fails instantly |
|---|---|---|---|---|
| has a clock coordinate | 7 | 18 steps | 0.42 | **7 out of 7** |
| no clock | 17 | 179 steps | 0.68 | **0 out of 17** |

### Remove the seven, and the result evaporates

| | all 24 systems | the 17 clean ones |
|---|---|---|
| disorder score explains | 59% | **0.2%** |
| Lyapunov exponent explains | 0.2% | 11.8% |

We checked this a second way too, in case it was an artefact of our scoring cap (7 systems
maxed out the 200-step measurement). Using a different, uncapped score gives the same
collapse: 57.5% → 0.4%.

**The disorder score was not measuring learnability. It was measuring "does this system have
a clock attached."**

### Why this matters beyond us

This failure is *silent*. You get a strong R², a clean mechanism story, and a result that is
entirely about your data pipeline. Anyone doing a cross-system study on this benchmark needs
to handle these systems explicitly — and the benchmark's full 135-system set contains them.

**What's queued:** R3b runs on clean systems only, 30 of them, to properly answer the
original question now that the mirage is gone.

---

## R8 — 63 small steps, or one big jump?

**In one sentence:** small steps won on 5 of 6 systems — but our test was too short to be
the real test, so it is being rerun.

### The question in plain terms

You want to know where a system will be 63 steps from now. Two ways:

- **Small steps ("rollout"):** predict one step. Feed that answer back in. Predict again.
  Repeat 63 times. Simple, but every mistake gets carried forward and amplified.
- **One big jump ("direct"):** a single network that takes "how far ahead?" as an input and
  jumps straight there. No accumulated mistakes — but it has to learn a different effective
  rule for every distance.

This is your `t → t+63` question. The lab's own paper showed the big jump *works*. The open
question was *when it wins*.

### What happened

Error at each horizon (lower is better), averaged over 3 runs:

| Lorenz | h=1 | h=8 | h=16 | h=32 | h=64 |
|---|---|---|---|---|---|
| small steps | 0.0004 | 0.0029 | 0.0063 | 0.0186 | 0.1945 |
| one big jump | 0.0358 | 0.0449 | 0.0675 | 0.1214 | 0.6673 |

| Chua | h=1 | h=8 | h=16 | h=32 | h=64 |
|---|---|---|---|---|---|
| small steps | 0.0009 | 0.0075 | 0.0169 | 0.0429 | 0.1548 |
| one big jump | 0.0108 | 0.0107 | **0.0120** | **0.0208** | **0.0907** |

Small steps win at the longest horizon on **5 of 6** systems. Typically about **2.6× better**.

### But look at the *shape* — that's the interesting part

Watch the "one big jump" row for Chua: `0.0108, 0.0095, 0.0097, 0.0107, 0.0120`.
It is **flat**. Going 16 times further ahead barely costs it anything.

Now the "small steps" row: `0.0009 → 0.0169`. It grew **19×** over the same span.

That is exactly the two mechanisms, made visible:

```
small steps:  start very low  ────╱  climb fast     (mistakes compound)
one big jump: start higher    ─────  stay flat      (no compounding, but a fixed cost)
                                  ↑
                            they cross here
```

Small steps start 10–90× better because predicting one step is easy. Then compounding eats
the lead. On Chua and Rossler, they crossed at 16 steps.

### And a humbling one

Plain copying — find the most similar moment in the past and copy what happened next, no
training at all — **beat the big-jump network on 3 of 6 systems**, and beat the small-steps
network on Rossler at h=64.

### Why we are rerunning this

Two flaws in our own design, both ours:

1. **The test was far too short.** 64 steps sounds like a lot. On Lorenz it is **0.86
   Lyapunov times** — less than one "natural unit" of chaos. We never let small steps fail
   the way the literature says they do, which was the entire point.
2. **We handicapped the big jump.** Both models got the same number of training examples.
   But the big-jump model has to learn 64 different rules from those examples, so it gets
   about 1/64th of the practice per rule. Fair on budget; unfair on the question "can this
   work at all?"

**R8b** runs to **512 steps** (~7 Lyapunov times) and gives the big-jump model **both 1× and
4×** the training examples, so we measure that handicap instead of assuming it away.

---

## R4 — Does a model stop improving no matter how big you make it?

**In one sentence:** the experiment ran perfectly and **could not answer its own question**,
because we made the task too easy and the ruler too short. Both are fixed and it is rerunning.

This is the crux of the project. Three published papers disagree about it and none of them
ran this experiment. So it is worth being precise about why our first attempt failed.

### What we did

Six systems. For each one, train the same network at **7 sizes** (1,011 up to 99,843
parameters — a hundredfold range) and with **4 amounts of data** (500 up to 32,000 samples),
**3 random starts** each. **504 trained models.** Then fit a curve to the errors and ask
where it is heading.

If the curve flattens out at some error level, that level is a property of the *system* —
evidence for a real ceiling. If it keeps falling, the limit is us.

### What happened: two ceilings, and both were ours

**Problem 1: the task was too easy.** Here is Aizawa's one-step error as the model grows a
hundredfold:

```
params:  1,011   2,196   4,611   9,987  21,891  46,212  99,843
error:  0.0004  0.0003  0.0003  0.0002  0.0002  0.0003  0.0004
```

That is not a curve. It is noise. Across all 504 models the smallest error we ever saw was
**0.00019** — and it was reached by the *smallest* models. The network was not the
bottleneck; we had hit the floating-point and data noise floor.

The reason is that we sampled each system 100 times per oscillation. Predicting one step
ahead when the next point is that close is almost trivial — barely harder than drawing a
straight line. A 1,000-parameter network already does it perfectly.

**Problem 2: the ruler was too short.** Our other measure was "how many steps until the
forecast goes wrong." We rolled forward 200 steps and stopped.

**33% of the 504 models never went wrong within 200 steps.** They hit the end of the ruler.
And on Lorenz, 200 steps is only **2.7 Lyapunov times** — less than three natural units of
chaos. We were asking "when does this fail?" and stopping before it had a chance to.

**Problem 3 (our own fault, differently): no leverage on the key prediction.** Scaling
theory predicts that the improvement rate should go as 1 divided by the attractor's
dimension. We picked six systems by name before seeing the data — and they all turned out to
have almost the *same* dimension:

```
Lorenz 2.08   Rossler 2.01   Chua 2.11   Halvorsen 2.12   Thomas 2.13   Aizawa 2.35
```

Testing "does the rate depend on dimension?" when dimension barely varies is testing
nothing. The correlation we computed (−0.486) is noise, and we are not reporting it as a
result.

### What we could still read off it

One system, Thomas, *was* hard enough to show a real trend: one-step error fell cleanly from
0.0025 to 0.0007 as the model grew. So the machinery works — the other five systems were
just too easy.

And a hint worth carrying forward: **bigger models did not improve the *structural* error at
all.** Averaged across systems, the correlation between model size and getting the frequency
spectrum right was **−0.085** — very slightly *negative*. Pointwise accuracy and "did it
capture the shape of the attractor" appear to come apart. That is the pattern five separate
papers have noticed in passing. But we cannot lean on it here, because the pointwise numbers
it is being compared against are noise. It rides along on the rerun.

### The fix

Three changes, all measured before committing:

| | before | after | effect |
|---|---|---|---|
| samples per oscillation | 100 | **20** | one-step error 0.0002 → **0.0057**, so capacity matters again |
| rollout length | 200 steps | **1,000 steps** | 2.7 → **67 Lyapunov times** on Lorenz |
| attractor dimensions covered | 2.01–2.35 | **2.0–16.5** | the rate-versus-dimension test finally has something to measure |

We verified the first two on a single test fit before queuing 378 more:

```
one-step 0.0057  ->  h50 0.38  ->  h200 1.33  ->  h1000 1.60      not censored
```

The error now grows all the way out, saturating above 1.0 (worse than guessing the average) —
exactly the regime where "does it stop improving?" is a real question.

**This is what the battery is for.** Had we skipped the calibration pass and run this as the
main experiment, we would have concluded "error keeps falling, so there is no intrinsic
ceiling" — and that conclusion would have been about our sampling rate, not about the
systems.

---

## R4b — the crux, properly configured

**In one sentence:** making the model bigger helps enormously at short horizons and **not at
all** at long ones — and the horizon where it stops paying varies **450×** between systems,
which the Lyapunov exponent does not explain.

378 models across 6 systems spanning attractor dimension 2.0 to 16.5. Zero censoring this
time: the 1,000-step ruler was long enough for every single fit.

### The main result, as one table

How much does a hundredfold increase in model size buy you, at each horizon? (The number is
the improvement rate — 0 means extra capacity buys nothing.)

| system | dimension | λ | h=1 | h=10 | h=50 | h=100 | h=200 | h=500 |
|---|---|---|---|---|---|---|---|---|
| Rossler | 2.01 | 0.15 | 0.64 | 0.40 | 0.31 | 0.26 | **0.09** | 0.01 |
| Lorenz | 2.08 | 0.89 | 1.13 | 0.20 | 0.22 | 0.61 | **0.00** | 0.00 |
| Thomas | 2.13 | 0.63 | 0.30 | **0.04** | 0.00 | 0.00 | 0.00 | 0.00 |
| HyperCai | 3.14 | 1.68 | 0.98 | 0.24 | 0.45 | **0.09** | 0.02 | 0.02 |
| HenonHeiles | 6.81 | 0.04 | 1.01 | 0.89 | 0.76 | 0.46 | 0.50 | **0.00** |
| Bouali2 | 16.47 | 0.08 | **0.00** | 0.00 | 0.03 | 0.08 | 0.04 | — |

Read left to right: every system starts with capacity helping and ends with it not helping.
On Lorenz, going from 1,000 to 100,000 parameters improves one-step prediction a lot
(rate 1.13) and improves 200-step prediction **by nothing at all** (rate 0.00, with a fitted
floor at 1.19 whose confidence interval excludes zero — and 1.0 means "no better than
guessing the average").

**Bouali2 is the extreme case.** At attractor dimension 16.5, a hundredfold capacity increase
buys nothing at *any* horizon, including the very first step:

```
params:  1,011   2,196   4,611   9,987  21,891  46,212  99,843
h=1:    0.0088  0.0121  0.0149  0.0158  0.0142  0.0133  0.0093
```

### So is the ceiling just "chaos runs out"?

That is the obvious explanation, and it is the one we had to rule out. Chaotic systems become
unpredictable after a few Lyapunov times no matter what. If that were the whole story, then
the point where capacity stops paying should land at **the same number of Lyapunov times on
every system.**

It doesn't:

| system | dimension | λ | stops paying at | in Lyapunov times |
|---|---|---|---|---|
| Bouali2 | 16.47 | 0.08 | step 1 | **0.0** |
| Thomas | 2.13 | 0.63 | step 10 | **1.6** |
| HyperCai | 3.14 | 1.68 | step 100 | **5.1** |
| HenonHeiles | 6.81 | 0.04 | step 500 | **5.7** |
| Rossler | 2.01 | 0.15 | step 200 | **8.9** |
| Lorenz | 2.08 | 0.89 | step 200 | **13.4** |

**A 450× spread.** Lorenz keeps rewarding a bigger model for over thirteen Lyapunov times;
Bouali2 stops rewarding it before the first step. These are not the same phenomenon with
different clocks.

And the Lyapunov exponent does not predict which is which:

| predictor | correlation with where capacity stops paying |
|---|---|
| 1 / Lyapunov exponent | **0.14** — no relationship |
| attractor dimension | **−0.71** — higher dimension, stops sooner |

### What this means

This looks like it **reconciles the three papers that disagree**:

- **Gilpin** says scale and data are what limit us. True — at short horizons. Capacity
  clearly helps there, on every system.
- **Duraisamy** says there are intrinsic ceilings nothing recovers. Also true — at long
  horizons. Capacity buys literally nothing, and the floor sits at "no better than guessing".
- They disagree because **they were measuring at different horizons.**

And the residual Gilpin found — the part not explained by the Lyapunov exponent — has a
candidate name here: **attractor dimension**. That is also what scaling theory predicts,
since the improvement rate should be governed by the intrinsic dimension of the data, not by
how fast trajectories diverge.

### What we are *not* claiming

Honesty about the limits, because they are real:

- **Six systems.** The dimension correlation of −0.71 with n=6 is **not statistically
  significant** (p ≈ 0.11). It is suggestive, not established.
- **The improvement-rate fits are noisy** where the curve is flat. Two entries in the table
  above are obvious fit artefacts (Thomas 1.88 at h=1000, Bouali2 4.75 at h=500) and are
  excluded from the reading, not explained away.
- **We tested the exponent, not the ceiling's height.** "Capacity stops helping" is not the
  same claim as "no architecture could ever do better."
- **Up to 100,000 parameters only.** Saturation observed here does not prove saturation at a
  billion.

**Follow-up queued (R4c).** The same measurement on **14 systems** spanning a 200× range in
Lyapunov exponent (0.011 to 2.325), which makes the "does λ explain it?" test properly
powered. The dimension test will stay underpowered, because the catalogue simply does not
contain many low-dimensional attractors above dimension 2.5 — and that limitation is worth
reporting rather than hiding.

## R5 — Does more chaos really need a *smaller* model?

*Queued.* Smoke test hint (not conclusive): at low chaos the map needed ~400 parameters to
hit 1% error; at high chaos, 3,235 parameters still only reached 3.5%. That is the *opposite*
of the published claim. Also: run-to-run variation was already 4.9%, enough to move the
answer by a full model size — which is the fragility we are testing for.

## R1 — Is "this system has a simple summary" a real property?

**In one sentence:** yes at a fixed zoom level — searching 100× harder found nothing new —
but the answer changes enormously with the zoom level, so "is it reducible?" is the wrong
question and "at what scale?" is the right one.

### The setup, in plain terms

Cellular automata are the simplest systems there are: a row of cells, each 0 or 1, updated by
a fixed rule. There are exactly **256** such rules, so you can check every single one — no
sampling, no selection bias.

The question Israeli and Goldenfeld asked in 2004: if you **squint** — group cells into
blocks of N, and only record a summary of each block — does the blurry version follow a rule
of its own? If it does, you can predict the blurry picture without simulating the detail.
They found 240 of 256 rules can be squinted at this way.

Our worry: finding that summary means **searching** through ways to summarise a block. The
harder you search, the more likely you stumble on one that happens to work. So is
"reducible" a fact about the rule, or a fact about your patience?

### First, a validity check

Before pushing further than the original authors, we checked we could reproduce them exactly.
Two published examples:

- rule 105 simplifies to rule 150 at block size 2 — **reproduced**
- rule 146 simplifies to rule 128 at block size 3, using their own specific recipe
  (summarise as 1 only if the block is `111`, else 0) — **reproduced**

And a stronger check. A 2015 follow-up (Dzwinel & Magiera) built a faster algorithm, pushed
to block size 7, and found the truly irreducible set narrows to exactly four rules:
**{30, 45, 106, 154}**. All four are in our surviving set. **Our method never wrongly
simplifies a rule that is known to be irreducible.**

### What happened: zoom level matters enormously

1,698 checks across all 256 rules. Reducibility, cumulatively:

```
block size 2  ->  92/256 reducible   (35.9%)   ██████████
block size 3  -> 151/256              (59.0%)  ████████████████
block size 4  -> 202/256              (78.9%)  ██████████████████████
```

Squint a bit harder and a third more rules become simple. That is a clean replication of the
extended 2006 paper's claim: **the chance of finding a simple summary heads toward certainty
as you look more coarsely.**

(Our 78.9% is below the published 93.75% because we sampled 60,000 of the 65,536 possible
summaries at block size 4, and the original authors also allowed summaries with more than two
or three categories. Ours is a lower bound. The confirmatory rerun makes block size 4
exhaustive.)

### And searching harder did *not* matter

This was the real test of our worry. At block size 5, we took the 54 rules that had survived
so far and searched with three budgets in turn:

| summaries tried | newly reducible |
|---|---|
| 40,000 | **0** |
| 400,000 | **0** |
| 4,000,000 | **0** |

**A hundredfold increase in search effort found nothing.** If valid summaries were lying
around waiting to be stumbled upon, more searching would have found some. It didn't. So at a
fixed zoom level, "reducible" looks like a genuine property, not an artefact of patience.

Our pre-registered kill criterion for this direction was "if the count moves by more than a
few percent when the search widens." It moved by **zero**. The criterion did not fire.

### A surprise about "wider" searches

We also tried summarising into **three** categories instead of two, expecting that a richer
summary would make more rules reducible. The opposite happened:

| block size | 2 categories | 3 categories |
|---|---|---|
| 2 | 35.9% | 25.0% |
| 3 | 55.1% | 35.2% |
| 4 | 78.9% | 32.0% |

And more pointedly: **not one rule was reducible with three categories that was not already
reducible with two.**

The reason, once you see it, is obvious. With two categories you must keep 8 combinations
consistent. With three you must keep 27 consistent. More categories is a **stricter**
demand, not a wider search. So "number of categories" is not a search-effort dial at all —
it is a different question. Worth knowing, because we had it backwards going in.

### What this means for the project

Good news and bad news.

**Good:** the label is trustworthy at a fixed scale. If we ever want to use "does a simple
summary exist?" as ground truth, it is not going to move under us.

**Bad, and decisive:** the answer is overwhelmingly **yes**. 79% of rules are already
reducible at block size 4, and the published work reaches 252 of 256 by block size 7. Trying
to train a predictor to spot the rare exception means predicting a class that occurs about
**1.6%** of the time. That is a very different, much harder experiment than the one people
imagine when they propose it — and it is a strong argument against building the project on
this substrate.

The honest framing, which the 2006 paper already reached: reducibility is not a yes/no
property of a system. It is **scale-indexed**. The interesting quantity is *at what scale*
simplicity appears, which is a regression, not a classification.

### A process note worth recording

The full sweep crashed at block size 6 with an integer overflow: 2⁶⁴ possible summaries
cannot be counted in a 64-bit integer. Because the results CSV was only written at the very
end, a two-hour sweep appeared to be lost.

It wasn't. Every row had already been streamed to the run log as it was produced, so all
1,698 rows were recovered (`src/r1_coarse_grain/recover.py`). That is precisely why the
logging format writes one line per unit of work instead of one file per run — and it is the
first time it has actually paid for itself.
