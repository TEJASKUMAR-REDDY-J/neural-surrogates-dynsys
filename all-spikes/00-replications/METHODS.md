# What we actually did, explained simply

[RESULTS.md](RESULTS.md) says what we *found*. This says what we *did* — the machinery, in
plain language, assuming nothing.

Read it if you want to know what a number in the results actually came from.

---

## Part 1 — What we are even studying

### Dynamical systems

A **dynamical system** is a set of rules that says how something changes from one moment to
the next. A swinging pendulum. Weather. Chemicals reacting in a beaker. Written down, it looks
like "the rate of change of x depends on x, y and z in this particular way."

If you know the rules and the starting point, you can compute the future by taking tiny steps
forward, over and over. That is called **integrating** the system, and it is what a physics
simulator does. It is accurate but slow — thousands of tiny steps for every second of
simulated time.

Some of these systems are **chaotic**. That means two starting points that differ by a
millionth end up completely different later. Chaos is not randomness — the rules are exact and
fixed — but it puts a hard limit on how far ahead anyone can predict, because you never know
the starting point perfectly.

The natural clock for a chaotic system is its **Lyapunov time**: roughly how long it takes a
small error to grow by a factor of ~2.7. One Lyapunov time later, your uncertainty has grown
noticeably. Ten Lyapunov times later, you know nothing. This matters because different systems
run at wildly different speeds, and comparing "50 steps on system A" to "50 steps on system B"
is meaningless unless you convert to their own clocks.

### Surrogates

A **surrogate** is a neural network trained to imitate the simulator. You show it lots of
(state now → state shortly after) pairs, and it learns the mapping. Then you use the network
instead of the simulator, because it is much faster.

That is the whole commercial promise, and it is why the field exists.

### The question underneath everything

Surrogates work well for a while and then fail. The disagreement in the literature is about
**why**:

- Is it that our models are too small and our datasets too limited? (Then buy more compute.)
- Or is there a hard ceiling built into the system that no model can pass? (Then stop buying.)

Everything below is machinery built to answer that.

---

## Part 2 — Where the data comes from

### The system library

We use **`dysts`**, a public collection of 129 chaotic systems, assembled by William Gilpin.
Each one comes with its equations *and* with published measurements of its properties —
Lyapunov exponent, attractor dimension, and so on. That matters: it means we are not grading
our own homework on what these systems are like.

We tried to generate a trajectory from all 129. **124 worked.** Of the five that didn't: three
took too long, one was degenerate, one was missing a data field in the library itself.

### We do our own integration

The library's built-in solver is set up very cautiously — it uses a method designed for hard
problems, with extremely tight tolerances. On this laptop it took **41 seconds to produce 2,000
points** of a Lorenz trajectory.

We take the *equations* and the *published properties* from the library, but integrate
ourselves with a standard method (Runge–Kutta 4/5) at a tolerance of one part in a billion.
That produces **20,000 points in 7 seconds** — about 57× faster per point, and still vastly
more precise than the neural network we are about to fit, so nothing is lost.

One trap: the integration routine cannot be interrupted from outside. A stiff system can sit
there forever. We fixed this by putting a stopwatch *inside* the equations — every time the
solver asks "what is the rate of change here?", the answer function also checks whether time
is up. It cannot be evaded, because the solver has to call it.

### Making systems comparable: timescale alignment

Different systems oscillate at wildly different speeds. To compare them, we sample each one at
a **fixed number of points per oscillation** — so one "step" means the same fraction of a cycle
everywhere. This is the standard convention in this field.

We used 100 points per oscillation at first, and later **20**, for reasons explained in Part 7.

Trajectories are cached to disk, so generating them is a one-time cost.

---

## Part 3 — What we train

### The task

We give the network the current state and ask for the **change** to the next state, not the
next state itself. So if the system is at (1.2, −0.4, 3.1), the network outputs something like
(0.02, −0.01, 0.005), which we add on.

Predicting the *change* rather than the *value* is standard practice — it keeps the numbers
the network has to produce at a similar scale across very different systems.

### Normalisation, and the leakage trap

Before training we rescale every coordinate so it has average 0 and spread 1. The statistics
for that rescaling are computed **only from the training portion**. If you compute them from
all the data, information about the test set leaks into training and your results are
optimistic for no good reason.

Training and test come from **separate, non-overlapping stretches** of the trajectory.

### The network

Deliberately simple: a plain feed-forward network, 2 hidden layers, `tanh` activations. Nothing
clever. We are not trying to build a good surrogate; we are trying to measure how surrogate
quality *responds* to things, and a simple model makes that measurement clean.

### Sizing by budget, not by width

Experiments sweep model **size**, so we specify a parameter count directly — "give me a model
with about 10,000 parameters" — and a helper solves for the layer width that hits it. Across
three orders of magnitude the result lands within 1% of the request:

```
asked for   1,000  ->  got  1,011
asked for  10,000  ->  got   9,987
asked for 100,000  ->  got  99,843
```

This matters because "width 64" means different parameter counts on a 3-dimensional system
versus a 16-dimensional one, and we compare across systems.

### Training details

Adam optimiser, learning rate 0.003 decaying on a cosine schedule, batches of 256, **200
epochs**, mean-squared error. Every run takes a **seed** that fixes the random initialisation
and the shuffling, so the same configuration reproduces exactly. We verified this: two runs
with the same seed agree to fifteen decimal places.

We ran **multiple seeds for everything**. Single-run results are how people fool themselves.

---

## Part 4 — How we measure whether it worked

This turned out to be the most consequential part of the whole design, so it is worth being
precise.

### Rollout — the thing that actually matters

A one-step model is not useful on its own. You use it by **rolling out**: predict one step,
feed that prediction back in as the new input, predict again, and so on. Errors from step 1
are baked into the input for step 2.

We start from many different points on a held-out trajectory (64 of them), roll each forward,
and compare against what the simulator actually did.

### Five different measures, because they disagree

**1. One-step error.** How wrong is a single step? Easiest to satisfy, least informative.

**2. Error at a fixed horizon.** How wrong are you exactly 50, or 200, or 1,000 steps out? We
normalise so that **1.0 means "no better than just guessing the system's average value"**. That
gives every number an intuitive meaning: below 1.0 you are forecasting, above 1.0 you would
have done better saying nothing.

**3. Valid prediction time.** How many steps until the error crosses a threshold? Reported in
**Lyapunov times** so systems are comparable. This is the standard metric in the field — and it
has a nasty flaw we walked straight into (Part 7).

**4. Spectrum error.** Does the forecast *oscillate* like the real thing? Compare the frequency
content of predicted and true trajectories. A forecast can be in completely the wrong place and
still get this right.

**5. Attractor-shape error.** Over a long run, does the forecast *visit the same regions* of
space, in the same proportions, as the real system? Formally a comparison of two distributions.

Measures 1–3 are **pointwise**: where is it *right now*. Measures 4–5 are **structural**: is it
still behaving like the right kind of thing. A model can be excellent at one and hopeless at
the other — and R6 found that model size improves one and not the other.

---

## Part 5 — What we compare against

A result means nothing without something to beat.

**Persistence.** Predict that nothing changes. The absolute floor.

**Context parroting.** This one matters. A chaotic trajectory keeps coming back near places it
has been before. So: look through the history, find the moment that most resembles right now,
and **copy whatever happened next**. No training. No parameters. About fifteen lines of code.

We included this because a 2025 paper showed it beats leading time-series foundation models on
chaotic systems. If your trained network cannot beat copying, then your network is doing
lookup, not learning — and any claim about it "understanding the dynamics" is unfounded.

**It was the right call.** Across 30 systems our trained networks beat copying on 12, with a
median skill ratio of exactly 1.00. A coin flip.

---

## Part 6 — The experiments, one at a time

Nine experiments, **~5,700 individual runs**, all logged.

### R0 — Does any of this work on this laptop?

Tried all 129 systems, timed everything, and timed a training run at each corner of the size ×
data grid.

Found: 124 systems usable; the library's solver unusable; and our cost predictions **14× too
optimistic**. The reason is instructive — a model ten times bigger took the same time to train,
because the computer was spending its time on bookkeeping between steps, not arithmetic. Cost
tracks *how many times you update the model*, not how big it is.

That led to two decisions: run two single-threaded workers side by side (1.7× speedup, since
threading bought nothing), and **refuse** the obvious 4× speedup of using bigger batches,
because it made the final training error four times worse — and R4's whole job is measuring
error floors, so a sloppier optimiser would have manufactured the answer.

### R1 — Is "this system has a simple summary" a real property?

**No neural networks at all.** Pure combinatorics.

Cellular automata are the simplest possible systems: a row of cells, each 0 or 1, updated by a
fixed rule. There are exactly **256** such rules, so you can check every one — no sampling, no
selection bias.

The question, from a famous 2004 paper: if you squint — group cells into blocks and keep only a
summary of each block — does the blurry version follow a rule of its own? Their answer was yes
for 240 of 256.

Our worry: finding that summary means **searching** through possible ways to summarise. Search
harder, find more. So is "simple" a property of the rule, or of your patience?

*What we did:* for each rule and each block size, build the blurred version and test every
candidate summary for self-consistency. Then repeat with search budgets differing by 100×.

*One engineering note:* the naive check took 19 seconds per rule. We rewrote it as a bit-packed
operation with an exact early-rejection pass — a summary that fails on a handful of cases fails,
full stop, so you can throw most candidates out cheaply. **19 seconds → 0.45 seconds, 42×**,
with the published results still reproducing exactly.

### R2 — Are the "complexity" numbers all the same number in disguise?

**No training.** 45 systems, 11 different descriptive statistics each, then measure how much
they disagree.

This is a check on a load-bearing assumption. If all the ways of describing a system are 95%
correlated, then "learnability is different from chaoticity" has nothing to measure.

### R3 / R3b — Does a cheap statistic already predict how well a network does?

Train one fixed-size network on each of many systems, 5 seeds each. Measure how far ahead it
stays accurate. Then check which cheap, training-free statistic predicts that.

**The honest way to check:** not just "does it correlate", but **leave-one-out** — drop each
system, fit on the rest, predict the one you dropped. That asks whether the relationship would
work on a system you have never seen, which is the only version that is useful.

We also ran a **shuffle control**: scramble the predictors 2,000 times and see how much apparent
signal appears by chance. With 6 predictors and 30 systems, pure noise produces R² of 0.19 on
average. Without knowing that, 0.46 looks impressive.

### R4 / R4b / R4c — Does a model stop improving no matter how big you make it?

The crux. For each system, train at **7 different sizes** spanning a hundredfold range and
**several data amounts**, 3 seeds each, then look at how error responds.

If error keeps falling, the limit is us. If it flattens at some level, that level is a property
of the system.

Ran three times, because the first two taught us things (Part 7). Final version: 18 systems,
**573 trained models**.

### R5 — Does more chaos need a *smaller* model?

Testing a specific published claim: that required model size rises with chaos, peaks, then
drops 47%.

We used the **Chirikov standard map** — a system with a single dial that turns chaos up
smoothly, and the exact mechanism the original paper appeals to. For each dial setting, train at
9 sizes × 5 seeds, and find the smallest model reaching a given accuracy.

The key design improvement: the original read its result at **one** accuracy threshold. We read
ours at **six**, which costs nothing because they all come off the same curves. If an effect
only appears at one threshold, it is an artefact.

### R6 — Does a bigger model get the *shape* right, or just the next step?

**No new training.** Reuses R4's models and asks a different question: how much does extra
capacity improve *pointwise* accuracy versus *structural* accuracy?

### R8 / R8b — 63 small steps, or one big jump?

Two ways to predict far ahead:

- **Small steps:** the rollout described above. Errors compound.
- **One big jump:** a network that takes "how far ahead?" as an extra input and jumps straight
  there in a single shot. No compounding, but it must learn a different effective rule for every
  distance.

Both get the same parameter count and the same data. We also ran the big-jump model with **4×
the training examples**, because it has to learn a whole family of rules from the same data and
matching example counts quietly handicaps it. Measuring that handicap is better than assuming
it away.

**Why this is the sharpest experiment:** a big-jump model *cannot* accumulate errors. So if the
long-horizon wall is caused by accumulation, the big jump should sail past it. If it fails in
the same place, the wall is about information running out. That distinguishes two published
positions with one experiment.

---

## Part 7 — The mistakes, and how we caught them

Five things went wrong. Every one was caught by a number looking odd.

### 1. The statistics were measuring our sampling rate

Some statistics work by looking at the *order* of nearby values — is this run going up, down,
up? But we sampled 100 points per oscillation, so consecutive points sit right on top of each
other. Zoom in far enough on any curve and it looks like a straight line.

```
every point   . . . . . . . . . .   always rising  -> looks calm
every 20th    .     .     .         up, down, up   -> looks chaotic
```

A system with a Lyapunov exponent of 1.678 — unmistakably chaotic — scored −0.01 on a chaos
test. **The naive version called 5 of 45 systems chaotic; corrected, 35.**

*Fix:* space the samples out by each system's own "time until something new happens." We record
both versions, because the gap is itself a finding: the correction **reorders** systems rather
than shifting them (correlation 0.14 between before and after), so they are effectively
different variables.

### 2. Seven systems were carrying a clock

Our most exciting early result — a cheap statistic predicting network skill with R² = 0.59 —
was fake.

Seven of 24 systems are *driven*, like a pushed swing, and the library handles them by adding a
coordinate that just counts time:

```
coord 0:   -2.05 ... +2.05    wiggles       (real physics)
coord 2:  869.80 ... 2693.89  only goes UP  <- a clock
```

That breaks two things at once. A clock has no disorder, so it drags the statistic down. And we
train on early data and test on later data, so every clock value at test time is a number the
network has never seen — like training on 1990–2000 and testing on the year 3000.

Two unrelated failures, same seven systems, one spurious correlation. Removing them: **0.59 →
0.0001**.

*How we knew:* those seven are exactly the seven where even *copying* fails instantly — 7 out of
7, versus 0 out of 17 for the others.

### 3. The crux experiment was measuring nothing

R4's first run was clean and useless. Both measures were pinned:

- **Too easy.** Sampling 100 points per oscillation makes a single step nearly trivial. Across
  504 models the best error was 0.00019 — reached by the *smallest* models. We had hit the
  floating-point floor, not a model limit.
- **Ruler too short.** We rolled out 200 steps. **33% of models never failed** within that — and
  200 steps on Lorenz is only 2.7 Lyapunov times.

*Fix:* sample 5× more coarsely (one-step error 0.0002 → 0.0057, so size matters again), roll out
1,000 steps (2.7 → 67 Lyapunov times), and pick systems spanning a real range of attractor
dimension instead of six that all happened to sit near 2.1.

### 4. Our own best finding did not replicate

R4b found that attractor dimension predicted which systems stop rewarding extra capacity,
correlation **−0.71**. Clean mechanism, quantitative, novel.

On 18 systems: **−0.22**. Noise. It was six systems getting lucky.

We had queued that replication *before* knowing the answer, which is the only reason we caught
it. A correlation from six points is a hypothesis, not a finding.

### 5. Two software failures worth recording

**A crash destroyed two hours of work — and it didn't.** The cellular-automata sweep died at
block size 6 on an integer overflow (2⁶⁴ possibilities cannot be counted in a 64-bit number),
and the results file was only written at the very end. But every row had already been streamed
to the run log as it was produced, so all **1,698** were recovered. That is exactly why the
logging writes one line per unit of work rather than one file per run.

**A divergence check that didn't check enough.** We flagged blown-up rollouts by testing whether
the numbers were *finite*. One system produced an error of 224 — finite, and obviously exploded.
The check should test magnitude too. One system in eighteen; no conclusions affected.

---

## Part 8 — How the runs are kept honest

**Every run writes a log.** One line per unit of work, plus a header recording the exact
configuration, a hash of it, the git commit, and the software versions. Any number in the
results traces back to a specific configuration and a specific commit.

**Reruns never overwrite.** Change a setting and the hash changes, so old results stay valid for
the old settings. Each pass is archived (`pass1`, `pass2_r4b`, `pass3_r4c`) so superseded
results remain inspectable.

**Analysis is separate from experiments.** The analysis code only reads result files, never
trains anything. So we can fix an analysis mistake without re-running days of compute — which we
did, twice.

**Every shared component self-tests.** Each module has a check that runs against cases with
known answers: permutation entropy must be near 0 for a sine wave and near 1 for noise; the
chaos test must say 0 for a periodic orbit and 1 for the logistic map; the correlation dimension
must return 2 for a 2-D cloud and 1 for a line; the scaling fit must recover a planted floor and
must *refuse to invent one* when there isn't any.

**Kill criteria were written down in advance.** Before running R3 we wrote: *if cheap statistics
explain 85% or more of the variance, stop — a learned predictor has nothing to add.* Writing
that down first is what stops you moving the goalposts when the number arrives.

---

## Part 9 — The scale of it

| | |
|---|---|
| systems examined | 129 tried, 124 usable, 45 measured in depth, 18 in the final capacity study |
| neural networks trained | **~2,300** |
| cellular-automata checks | 1,698 |
| total logged results | **~5,700** |
| run logs | 30 |
| experiments | 9, three of them rerun after finding a flaw |
| wall-clock time | about 13 hours on a 2-core laptop, no GPU |

The single most important design choice was spending the first day testing the *instruments*
instead of running the experiment. Five things were wrong. Four of them would have produced
confident, wrong, publishable-looking claims.
