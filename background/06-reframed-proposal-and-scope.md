# The reframed proposal: when is training a surrogate worth it at all?

Written 2026-09-08, after the devil's-advocate pass in
[05-devils-advocate-and-reframing.md](05-devils-advocate-and-reframing.md) and after one new
result (N6) that arrived while answering these questions and changed the answer.

This document answers four questions put directly:

1. What can bring value now — and how would **neural cellular automata** fit?
2. If a surrogate has to *win*, what does that take?
3. Is the proposed direction actually **feasible**, robustly, on this hardware?
4. If we rewrote the **problem statement and scope** from scratch, what would they be?

Nothing already recorded is altered. This sits on top.

---

## Part 0 — The short answer

The old proposal asked: *can we predict which architecture will work?* That question is dead
for a reason we can now state precisely — **it was asked about the model, and the answer lives
in the data.**

The reframed question is:

> **When is training a neural surrogate worth it at all, and can you tell before you train?**

And the answer has a shape: the advantage of a trained surrogate over a fifteen-line lookup is
governed by **two measurable properties of the dataset** — how densely it covers the state
space, and how noisy it is. Current benchmarks sit in the corner where that advantage is zero.

The single most important consequence: **this question is fully answerable at 10⁵ parameters
on two CPU cores.** The old one needed 10⁹ and never could be. That is not a smaller ambition
dressed up — it is a different question, about the data regime rather than the capacity limit,
and our hardware is sufficient for it rather than embarrassing.

---

## Part 1 — What changed today

Before answering anything about direction, one thing was checkable for free, and it decided
the rest. Full write-up and figure: [N6 in RESULTS.md](../all-spikes/00-replications/RESULTS.md).

We had a finding we kept repeating without understanding: **a trained network ties with
copying**, 46 of 108 systems, median ratio exactly 1.00. Every reading of it had been about the
network — maybe it is not learning anything.

Wrong object. Copying works by finding the nearest past state and replaying what followed. So
its skill is set by **how close that nearest state is**. We measured that directly across 107
systems, and:

| | |
|---|---|
| coverage vs. the network's advantage over copying | **rho = +0.558**, p = 4e-10, n = 107 |
| shuffle control (5,000 permutations), 95th pct of noise | **0.187** |
| leave-one-out range | **+0.546 to +0.583**, no sign flips |
| controlling for state dimension | coverage **holds at +0.488**; dimension collapses to +0.056 |

And the decomposition, which is the whole story:

| | correlation with coverage |
|---|---|
| **copying's** forecast horizon | **-0.444** — sparse coverage wrecks it |
| **the network's** forecast horizon | **-0.020** — it barely notices |

The tie is not a fact about neural networks. **It is a fact about the benchmark.** The median
system's "held-out" test state has a near-twin already in the training set, one percent of the
attractor away.

That converts a complaint into a law with a free variable, and everything below follows from it.

---

## Part 2 — What does it take for a surrogate to win?

Five levers. We can now rank them, because we have measured four.

| lever | what it says | evidence | strength |
|---|---|---|---|
| **1. Sparse coverage** | the network wins when the data does *not* densely cover the state space | N6: rho +0.56; sparsest quartile network wins **22/27**, densest **6/27** | **strongest, measured** |
| **2. Observational noise** | the network denoises; copying replays noise along with signal | N2: **19/40** clean, **29/40** at 20% noise | **measured** |
| **3. Matching inductive bias** | a weight-shared local rule beats a dense net on a system that *is* a local rule | **untested — this is what NCA is for** | predicted |
| **4. Data budget** | the same axis as (1), from the other side: less data means sparser coverage | implied by N6; controlled version unrun (our self-check moves coverage 0.30 → 0.007 from 1k → 16k points) | cheap to test |
| **5. Capacity** | bigger models | R4c: **0.94** at h=1 falling to **0.13** at h=500; R6: structure **+0.06** | **measured, weakest** |

Read the last row against the first two. **Capacity — the thing the entire field scales — is
the weakest lever we found, and the two strongest are properties of the data, not the model.**

Which gives a decision rule you can run in about a minute, before training anything:

```
measure  d = median nearest-neighbour distance / spread of your training data
measure  s = your observational noise level

d < 0.01,  s = 0   ->  a lookup table will match whatever you build.  Do not build it.
                       (this is where every chaotic-systems benchmark sits)
d > 0.03           ->  the network wins about 80% of the time
s > 0.2            ->  the network wins about 73% of the time, whatever d is
```

That is the contribution in one box: **not a predictor of which architecture, but a predictor
of whether to bother.** It is what the original proposal was reaching for, aimed at a target
that actually exists.

---

## Part 3 — Where neural cellular automata come in

An NCA is a single small network applied identically at every cell, updating that cell's state
from its immediate neighbours, iterated many times (Mordvintsev et al., *Growing Neural
Cellular Automata*, Distill 2020). Typically a few thousand parameters, regardless of how big
the grid is.

It fits this project unusually well, for five reasons — and one of them removes our blocker.

### 3.1 It is the architecture whose bias matches the physics

A PDE **is** a local update rule applied identically everywhere. An MLP on a flattened
64-variable state has to *learn* translation-equivariance from data; an NCA has it for free.

That makes NCA the natural instrument for lever 3 — the one we have never measured. Put three
methods on identical data and you get a clean ladder of built-in structure:

| method | locality | weight sharing | parameters at 64 sites |
|---|---|---|---|
| lookup / copying | none | n/a | 0 |
| MLP on the flat state | no | no | ~10⁵ |
| **NCA / 1-D CNN** | **yes** | **yes** | **~5×10³** |

Nobody in our corpus varies this axis while holding the system and the data budget fixed. And
it is exactly the *task × architecture interaction* that P4's critique said Task2Vec and
Model2Vec structurally cannot express.

### 3.2 Parameter count decouples from problem size — this is the feasibility unlock

Our binding constraint has been 10⁵ parameters on two cores. An NCA with 5,000 parameters
works on a 64-site lattice or a 512-site one at the **same parameter cost**; only the forward
pass grows, and it is a `conv1d`. So we can finally go spatially extended **without** the
compute wall that made the original question unanswerable.

### 3.3 It closes the loop back to R1

R1 is elementary cellular automata: 256 rules, checked exhaustively, 202 of 256 reducible at
block size 4. An NCA trained to imitate an ECA is a neural surrogate of a system whose
**structural label we know exactly rather than estimating it**.

That is the original proposal's dream — *does a measurable structural property of the system
predict how well a surrogate does?* — with ground truth that is exact, on a substrate that
cannot numerically diverge. It is also the honest test of R1's own bad news: with 202/256
reducible, the interesting quantity is not the binary label but **at what block size** the
simple description appears. That is a regression with a real range.

### 3.4 It has a published fix for the failure R9 found

R9 found refinement helps for about three passes then reliably degrades, and **never** improves
beyond the number of passes it was trained with (0 of 24 cases). The NCA literature hit exactly
this and solved it: train from a **pool of previous states** and **damage** them, so the rule
must stay stable far past its training horizon.

Whether that trick transfers to autoregressive surrogates of chaotic systems is a concrete,
cheap and genuinely open question that our own negative result motivates. It is the natural
follow-up to R9 rather than a new direction.

### 3.5 Robustness is structural, not incidental

NCAs are trained to be perturbation-tolerant — they regrow after damage. Our denoising claim
currently rests on an inference ("a network averages over examples and cannot reproduce one
noise realisation"). An NCA trained with the damage protocol makes that mechanism **explicit
and adjustable**, so lever 2 becomes something we can dial rather than observe.

### 3.6 And the part that unblocks us: stop trying to fix Kuramoto–Sivashinsky first

KS has blocked this project for two sessions. But look at what we actually need from it:
*spatially extended, many variables, local coupling, genuinely chaotic, known ground truth.*

KS satisfies that **and** demands a stiff spectral PDE integrator we cannot get stable. A
**coupled map lattice** (Kaneko, 1989) satisfies the same requirement with **no integrator at
all**:

```
x     (i) = (1 - eps) * f(x (i))  +  (eps/2) * [ f(x (i-1)) + f(x (i+1)) ]
 n+1                       n                       n              n

with f(x) = a*x*(1-x),  periodic boundaries
```

One line. Explicit. Cannot diverge. Any lattice size. Genuine spatiotemporal chaos. Two knobs —
coupling `eps` and nonlinearity `a` — that sweep from spatially coherent to fully developed
chaos. And the Jacobian is analytic and tridiagonal, so we can compute the **full Lyapunov
spectrum and Kaplan–Yorke dimension exactly** by the standard QR method, rather than estimating
them the way we must for `dysts`.

That is *better* ground truth than we have had all project, on the axis we were missing, for
about an afternoon of work.

**So the substrate ladder becomes:**

| rung | what it adds | integration risk |
|---|---|---|
| `dysts` low-dimensional ODEs | done — 108 systems, published invariants | none (done) |
| **ECA** | exact structural label from R1, discrete | **none — no integrator** |
| **CML** | spatially extended, continuous, exact Lyapunov spectrum | **none — explicit map** |
| KS | a real PDE, direct bridge to the neural-operator literature | **the known blocker** |

KS stops being a prerequisite and becomes a credibility upgrade to add if it validates. That is
the single most useful structural change in this document.

---

## Part 4 — Is it feasible? Honestly

### Costs, on two cores with no GPU

| item | cost | basis |
|---|---|---|
| CML trajectories, 64 sites × 100k steps | **seconds** | vectorised numpy, explicit map |
| Exact Lyapunov spectrum, 64 sites | **~minutes** | QR on a tridiagonal Jacobian |
| ECA data | **free** | already generated in R1 |
| NCA / CNN fit, 64 sites, ~5k params | **~1–3 min** | comparable per-step to the MLPs we ran; conv1d on 2 cores |
| Nearest-neighbour lookup, 64-D over 8k context | **milliseconds/query** | brute force, already measured in N6 |
| **The full grid** — 3 substrates × 5 coverage levels × 4 noise levels × 3 methods × 3 seeds | **~3–4 h in two shards** | ~540 fits at ~40 s, at the measured 1.7x sharding speedup |

Feasible. Comfortably, for the first time in this project.

### What could go wrong, named in advance

| risk | severity | mitigation |
|---|---|---|
| **NCA training is unstable on chaotic targets** | medium | fall back to a plain 1-D CNN — same inductive bias, none of the recursion machinery. The NCA-specific claims (3.4, 3.5) then drop; the architecture-bias claim (3.1) survives |
| **Lorenz 1969 already said it** — analogues are rare in high dimensions | **high, and real** | our claim is *not* "recurrence matters". It is "**today's benchmarks sit in the corner where the 1969 method already suffices**", plus the noise interaction. Must do a targeted literature check **before** committing — this is the first task, not an afterthought |
| **Zhang & Gilpin already published the parroting comparison** | medium | they showed the tie exists; we explain *when and why* it exists and give the threshold. Must state the difference in one sentence in any writeup |
| **Someone has done NCA-as-PDE-surrogate** | medium | likely partly true. NCA is our *instrument*, not our claim — the claim is about coverage and noise. Check anyway |
| **A reviewer says CML and ECA are not PDEs** | medium | true. That is what the KS rung is for, and why the ladder is stated as a ladder with honest rungs |
| **Time** | **high** | three days to the stated 2026-09-11 deadline. See Part 7 |

### Is it robust enough to satisfy the four criteria?

| criterion | assessment |
|---|---|
| **Surprising to experts** | **Moderate–high.** "Your benchmark is sampled so densely that a lookup table matches every architecture you are comparing" is not a claim anyone in this area is making, and it is checkable in a minute |
| **Fruitful** | **High.** It is upstream of every result in the field, it gives a pre-training go/no-go rule, and it prescribes a benchmark fix |
| **Rigorous** | **Good, and improvable.** Two axes we can *manipulate* rather than merely observe — which is what N6 currently lacks. Kill conditions in Part 8 |
| **Feasible** | **Yes.** The claim is about the data regime, not model scale, so 10⁵ parameters is *sufficient* rather than embarrassing |

The honest weak spot is that N6 is observational. The whole design below exists to make it
interventional.

---

## Part 5 — The reframed problem statement

### Before

> Can we characterise a task's learnability in advance, use that to predict which
> architectures will succeed, and train according to problem structure rather than
> brute-force search?

Died because: the predictor is a noiseless-simulation artefact (N2, 32% → 0 at 1% noise); the
structural claims were 20-year-old results; and the model-scale question needs 10⁹ parameters.

### After

> **A trained neural surrogate only beats not training at all in a specific and measurable
> regime — sparse coverage of the state space, or non-trivial observational noise. Standard
> benchmarks sit outside that regime, which is why architecture comparisons on them are
> uninformative. We map the regime, give the threshold, and show what changes when you cross
> it.**

Three claims, each with a pre-registered test:

- **C1 (the law).** A surrogate's advantage over analogue lookup rises with coverage sparsity
  and with noise, and the relationship is quantitative enough to predict.
  *Observational support already: rho = +0.56, n = 107, with controls.*
- **C2 (the diagnosis).** Standard benchmarks — including the 129-system library the chaos
  literature runs on — sit at coverage ~0.01 and noise 0, which is precisely where the
  advantage vanishes.
  *Already measured.*
- **C3 (the mechanism).** The advantage is denoising plus interpolation, not recall. Testable
  directly: an explicitly denoised lookup baseline should close the noise half of the gap; a
  matched-inductive-bias network should widen the sparse half.
  *Unrun. This is the experiment that makes it a paper rather than an observation.*

### Why this scores better on all four criteria

The comparison is in Part 4. The one that matters most: **the old question could not be settled
on this hardware and this one is settled by it**, because the free variables are properties of
the dataset, which cost nothing to move.

---

## Part 6 — Scope

The philosophy this project follows warns about exactly one tightrope: narrow claims are
technically correct and inconsequential, broad ones are consequential and unsupportable. Here
is where the line goes.

### In scope

- **Substrates:** `dysts` low-dimensional ODEs (done), CML (spatially extended, exact
  invariants), ECA (exact structural label). KS if and only if a validated implementation drops
  in cleanly.
- **Free variables:** coverage density (moved by training-set size and by lattice size), noise
  level and noise *model*, architecture class.
- **Methods compared:** analogue lookup, denoised lookup, MLP, weight-shared local network
  (CNN / NCA).
- **One headline figure:** the regime map — coverage on one axis, noise on the other, the
  win/lose boundary drawn, and published benchmarks plotted as points on it.

### Out of scope — stated, not quietly dropped

- **Anything about 10⁸–10⁹-parameter models.** We cannot speak to it and will not.
- **Real-world data** (Monash and similar). No ground-truth dynamics, so no way to separate
  "the model is too small" from "the information is not there". Named as future work.
- **The learned meta-predictor of architecture ranking.** Dead, killed by N2, and the writeup
  should say so with the evidence rather than omitting it.
- **Claims about "neural surrogates" in general.** The supportable claim level is
  **autoregressive surrogates of chaotic dynamics trained on synthetic trajectories** — and
  every sentence must stay inside that.

### The claim level, written out so it can be checked

> *For autoregressive neural surrogates of chaotic dynamical systems trained on synthetic
> trajectories, the advantage over parameter-free analogue forecasting is governed by training-
> set coverage density and observational noise; standard benchmarks are constructed in the
> regime where that advantage is absent.*

Not "neural networks do not learn dynamics." Not "surrogates are useless." That sentence, and
we can carry it.

---

## Part 7 — The plan, against three days

Assuming the 2026-09-11 deadline in `RESEARCH-LOG.md` still stands. Ordered so that stopping
at any point still leaves something coherent.

| # | task | cost | if it fails |
|---|---|---|---|
| **0** | **Literature check on the Lorenz-1969 / analogue-forecasting threat**, and on NCA-as-PDE-surrogate | 1 h | If the coverage law is already published, the spine moves to the *noise interaction*, which is still ours. **Do this first** |
| **1** | **Controlled coverage experiment.** Fix ~8 systems, vary training-set size across 5 levels, measure lookup vs MLP. Turns N6 from observational into interventional | ~1.5 h | If the advantage does *not* track coverage under direct manipulation, **C1 is dead** and we report that. This is the highest-information hour available |
| **2** | **CML substrate + exact Lyapunov spectrum**, with a self-check against published values | ~2 h | Low risk — explicit map, no integrator |
| **3** | **The regime grid** on CML: coverage × noise × {lookup, MLP, CNN} | ~3 h | Partial results still map part of the plane |
| **4** | **Denoised-lookup baseline** — the direct mechanism test for C3 | ~1 h | A null here weakens C3 to a correlation, and we say so |
| **5** | **Writeup**, CAISc format, plus the regime-map figure | ~1 day | — |
| *opt* | ECA + NCA structural-label loop (3.3); KS if a reference implementation validates | — | Cut first if time runs short |

Batch 128 reruns (from N5) only for numbers actually quoted as absolutes.

---

## Part 8 — Pre-registered kill conditions

Written before the runs, so they can be checked rather than argued about.

- **If, holding the system fixed, the surrogate's advantage does not rise as the training set
  shrinks** — C1 is wrong, N6 was a cross-system confound, and the direction dies. *(Task 1,
  and it is deliberately first.)*
- **If a denoised lookup baseline closes the whole noise gap** — the mechanism is denoising
  alone and the surrogate contributes nothing beyond a smoother, which is a *stronger* negative
  result and should be reported as one.
- **If the coverage law does not transfer to a spatially extended system** — the claim is
  low-dimension-specific, and the honest paper is narrower.
- **If the coverage-versus-analogue relationship is already published quantitatively** — this
  becomes a replication plus the noise interaction, which is a workshop paper rather than a
  conference one. *Check first, at task 0.*

Note what has changed from the earlier kill list in
[05](05-devils-advocate-and-reframing.md): *"if the copying tie disappears on a spatially
extended system, the spine is gone"* is now the **wrong** test. N6 predicts that it will
disappear, and says why. The sharp version is quantitative — does it disappear **at the rate
coverage predicts**?

---

## Part 9 — The one-paragraph version

We set out to predict which architectures suit a problem. That failed, for a reason worth more
than the original question: **we were studying the model when the answer was in the data.** A
neural surrogate ties with a fifteen-line lookup table on the standard benchmark not because it
learned nothing, but because the benchmark samples its attractors so densely that every test
state has a near-twin in training — median nearest-neighbour distance, one percent of the
attractor. Move off that corner, by thinning the data or adding one percent of noise, and the
network pulls ahead; and it is the lookup baseline that moves, not the network. That gives a
law with two measurable axes, a decision rule you can run before training anything, and a
concrete criticism of how the field builds benchmarks. It needs one spatially extended
system — and a coupled map lattice provides one exactly, with no integrator to destabilise,
which is what has been blocking us. Neural cellular automata are the right instrument for the
one lever we have never measured: how much of a surrogate's advantage is built-in structure
rather than capacity.
