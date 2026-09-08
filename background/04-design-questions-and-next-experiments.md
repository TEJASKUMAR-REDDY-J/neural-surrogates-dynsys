# Design questions, honest answers, and what to run next

Written 2026-09-08, in response to direct questions about whether the experimental setup is
good enough. Several of these are things we should have checked and hadn't.

---

## 1. Is `dysts` the only library we need? Is it enough to prove anything?

**Short answer: no, and it was never going to be. It is the right tool for one specific
question and the wrong tool for the question most people care about.**

### What `dysts` gives us that almost nothing else does

- **Known equations.** We can generate unlimited perfectly clean data, and we know the true
  answer exactly. Nothing is estimated.
- **Published properties.** Lyapunov exponent, attractor dimension and so on come from the
  library, measured by someone else. We are not grading our own homework when we ask "does
  attractor dimension predict X?"
- **A fixed public list.** We cannot cherry-pick systems that flatter a result, and someone
  else can rerun exactly what we ran.

Those three things are what make the *mechanism* questions answerable at all. "Is this ceiling
intrinsic to the system, or is it our model being too small?" is only a well-formed question
when you have ground truth and can turn model size up and down freely.

### What it does not give us

- **Everything is synthetic and noiseless.** Real measurements have noise, gaps, and drift.
- **Everything is low-dimensional.** Our systems have 2 to 16 variables. The surrogate
  literature we are arguing with — REALM, Shikhman, the whole neural-operator field — is about
  **spatially extended** systems: fluid on a grid, flames on a mesh, thousands to millions of
  variables. **We have tested none of that.** This is the single largest gap.
- **Everything is stationary.** No regime changes, no seasonality, no non-stationarity.

So: `dysts` results can support claims of the form *"here is a mechanism, and here is evidence
it is real"*. They cannot support *"and therefore this holds for practical surrogate
modelling"*. Any writeup has to be explicit about that boundary.

### On the specific alternatives suggested

**TSB-AD** — this is an **anomaly detection** benchmark: find the unusual points in a series.
That is a different task from forecasting, with different metrics and different failure modes.
It would not help here. (Worth saying plainly rather than nodding along.)

**Monash Time Series Forecasting Archive** — ~30 collections of real forecasting data
(electricity, traffic, tourism, weather). **Genuinely useful, for a different question.** It
would tell us whether the predictor story survives contact with real measurements. But it has
no ground-truth dynamics, so there is no Lyapunov exponent, no attractor dimension, and no way
to distinguish "the model is too small" from "the information is not there". It can measure
*achieved* error and nothing about ceilings.

The right way to use Monash is as a **generalisation check on the R3b result**, not as a
substrate for the mechanism work.

### What would actually strengthen the case, in order

1. **One spatially extended system.** Kuramoto–Sivashinsky on a 1-D grid of 64–128 points.
   Chaotic, well studied, the standard testbed in the reservoir-computing and neural-operator
   literature, and **cheap enough for a CPU**. This closes the biggest gap between what we
   tested and what the field argues about, at a cost of a few hours.
2. **Observational noise.** Add measurement noise at several levels to systems we already have.
   This is the cheapest way to move toward realism, and it directly tests something we
   currently assume away.
3. **Real chaotic data with a known story.** The Santa Fe laser series is the classic: real
   measurements from a system understood well enough that the dynamics are not a mystery.
4. **Monash**, for external validity of the predictor result only.

**Bottom line:** `dysts` alone is enough to establish a mechanism and not enough to claim
generality. Adding Kuramoto–Sivashinsky is the highest-value single addition, and it is
affordable here.

---

## 2. Is batch size 256 actually good?

**We had only ever tested bigger. Here is the test we should have run.** Lorenz, 20,000-parameter
model, 8,000 samples, 200 epochs, learning rate scaled with batch size as usual.

| batch | learning rate | time | training loss | error at h=50 | error at h=200 |
|---|---|---|---|---|---|
| 64 | 1e-3 | 69 s | **1.02e-06** | 0.414 | 1.336 |
| 128 | 2e-3 | 37 s | 1.56e-06 | **0.242** | 1.452 |
| **256** | 3e-3 | 22 s | 4.09e-06 | 0.315 | 1.397 |
| 512 | 4e-3 | 15 s | 1.07e-05 | 0.622 | 1.419 |
| 1024 | 1e-2 | 12 s | 1.48e-05 | 0.709 | 1.399 |

Three readings:

- **Training loss improves monotonically as batches get smaller.** Batch 64 is 4× better than
  256. So 256 is *not* the optimum for fitting.
- **Rollout error does not follow.** 128 looks best, 256 next, 64 worse than both. That
  non-monotonicity is a warning sign: this is a **single seed**, and elsewhere we measured
  seed-to-seed variation of about 13%. The gap between 0.24 and 0.31 is inside that. **We
  should not claim 128 is better.**
- **The decision to reject 1024 was correct**, and now for a stronger reason: error at h=50
  more than doubles (0.31 → 0.71).

**Does this undermine any conclusion?** No — and it is worth being clear why. Every claim we
made is a *relative* one: bigger model versus smaller model, this horizon versus that horizon,
model versus copying, all under one fixed recipe held constant everywhere. A recipe that is
slightly suboptimal shifts all conditions together and does not change their ordering.

It would matter if we had claimed an *absolute* number — "the error floor of Lorenz is X" — and
we deliberately did not.

**What to do:** keep 256 for continuity with everything already run; add a batch size sweep at
3 seeds to the next round so the choice is evidence-based rather than conventional.

---

## 3. Why did five systems fail in R0?

Now diagnosed precisely rather than lumped together:

| system | what actually happened |
|---|---|
| **HyperLu** | Stiff. Exceeded the 25-second integration budget. |
| **InteriorSquirmer** | Same. |
| **MacArthur** | Same. |
| **SwingingAtwood** | **A gap in `dysts` itself** — the system is missing its `multiscale_entropy` field, so our metadata read failed. Nothing wrong with the system; we could include it by treating that field as optional. |
| **BickleyJet** | **Mislabelled by us.** Our report said "slow or degenerate". It is neither. It has a clock coordinate whose period is 97,827, so that coordinate runs to ~850,000 and trips our "is this bounded?" check. It is one of the driven systems — correctly excluded, wrong reason given. |

Two lessons. The three timeouts are genuine and expected — stiff systems need implicit solvers
we chose not to use. The other two are **our bookkeeping**, not the systems: one is a
recoverable metadata gap, one is a correct exclusion with a wrong label. Both are fixable and
would bring us to 126 of 129.

---

## 4. If the first answer is wrong, can more passes fix it?

**This is running now as experiment R9.** Design and reasoning:

The idea is that instead of producing an answer in one shot, the model produces a draft, looks
at the draft, and revises — repeatedly. This is the mechanism behind recursive-reasoning models
like TRM, and it is the one thing the battery had not touched.

It also gives R8 a fair rematch. Direct prediction lost badly to step-by-step rollout — but
direct prediction got **one** pass while rollout got 64. Maybe the right comparison is a direct
predictor allowed to think more than once.

**How it works.** One network takes (where the system is now, how far ahead, current guess) and
proposes a correction to the guess. The first guess is "nothing changes". Then it runs again on
its own output, and again. Training penalises *every* pass, not just the last — otherwise the
network is free to make the intermediate passes meaningless.

**Three questions, in increasing order of interest:**

1. Does pass 2 beat pass 1? If not, the idea is dead in this setting.
2. Does it keep improving, or does it settle?
3. **Can it make things worse?** This is the real question. Repeated refinement is a fixed-point
   iteration, and fixed-point iterations can diverge. We train with 4 passes and evaluate with
   **16** — four times more than it ever saw. That is exactly where a genuinely stable method
   and a merely lucky one come apart.

**Early signal from the smoke test** (undertrained, so the absolute numbers are meaningless —
only the shape matters):

```
pass:   1      2      3      4      5      6      7      8
error:  0.994  0.977  0.975  0.977  0.976  0.977  0.977  0.977
```

It improved on pass 2, then **settled and stayed settled** — it did not run away. That is the
well-behaved outcome. Whether it holds when the model is properly trained, at long horizons,
and on harder systems is what the full run answers.

**What each outcome would mean:**

- *Improves then settles* — refinement is a real mechanism here, and the interesting question
  becomes how many passes are worth their compute.
- *No improvement after pass 1* — the network is not using the extra computation, which would be
  evidence against the "internal iteration" story in this setting.
- *Improves then degrades* — the most interesting result, because it means the number of passes
  is a hyperparameter that must be tuned per system, and "think longer" is not free.

---

## 5. What to run next, ranked

Existing results and replications stay exactly as they are. These are additions.

### N1 — Add one spatially extended system (highest value)

**Kuramoto–Sivashinsky** on a 64–128 point grid. Chaotic, standard, and the bridge between what
we tested and what the surrogate literature actually argues about.

Rerun the three findings that matter on it: does capacity still help at short horizons and not
long ones? Does step-by-step still beat one big jump? Does copying still tie with the network?

**Why first:** it is the difference between "we found a mechanism in toy systems" and "we found
a mechanism that shows up where the field works". Cost: a few hours on this laptop.

### N2 — Add observational noise

Take systems we already have and add measurement noise at several levels. Everything so far
assumes perfect observation, which no real application has. This also connects to a result we
already read: above the noise level, chaos and randomness are indistinguishable — so there
should be a noise level at which our predictors stop working, and finding it is informative.

### N3 — Push the predictor properly

R3b explained 27% of surrogate skill with two cheap statistics, leaving 73% open, and the kill
criterion did not fire. The obvious next step:

- all ~100 usable bounded systems instead of 30
- more candidate statistics
- and critically, **hold out whole families of systems**, not random systems — because that is
  the split that the neighbouring literature says kills this class of method

### N4 — Fix the driven systems instead of excluding them

We threw out about 20% of the library because of clock coordinates. The proper fix is to wrap
the clock into a bounded phase (its sine and cosine) and treat it as a *known input* rather than
something to predict — which is what it is. That recovers those systems and removes a caveat.

### N5 — Batch size and training recipe sweep

Small, cheap, and closes the gap identified in question 2 above. 3 seeds × 5 batch sizes on 3
systems.

---

## The honest summary

The setup is sound for what it was built to do — establish mechanisms with ground truth
available — and it has real limits: synthetic data, low dimension, no spatial extent, no noise.
Two of the five R0 exclusions were our bookkeeping rather than the systems. The batch size is
defensible but was chosen by convention and only ever tested in one direction.

None of that changes the conclusions already drawn, because those are all relative comparisons
under a constant recipe. It does bound how far they can be claimed to generalise, and N1 is the
cheapest way to widen that bound meaningfully.
