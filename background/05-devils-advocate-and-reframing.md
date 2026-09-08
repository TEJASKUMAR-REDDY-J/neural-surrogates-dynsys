# Devil's advocate: what is this research actually worth now?

Written 2026-09-08, after the full replication battery. The job here is to argue *against* the
project as hard as it can be argued, see what survives, and then say honestly what the problem
statement is worth and what would have to change to give it a purpose.

Nothing already recorded is altered. This is an assessment laid on top.

---

## Part 1 — The case for the prosecution

If a hostile reviewer read everything in this repository, this is the case they would make. It
is a strong one and every point is true.

### 1.1 Every original claim was already known, or died

The proposal's four numbered claims:

| claim | verdict |
|---|---|
| **C1** The leap horizon is a property of the system, measurable before training | Partially pre-empted (REALM already ties rollout error to system properties), and our own attempt to identify *which* property **failed to replicate** |
| **C2** Learnability is a separate axis from chaoticity | **Known since 2003** in nonlinear dynamics; measured at scale on 135 systems by Gilpin in 2023 |
| **C3** Computational irreducibility can be made a continuous measured quantity | **Known since 2004/2006.** The 2006 paper already showed the probability of a coarse description tends to 1 at coarser scales, which makes it scale-indexed rather than binary — the exact reframing we arrived at independently |
| **C4** A predictor will generalise to held-out rule families | Never tested at family level. And the neighbouring literature says this class of method usually fails exactly there |

And the broader hypotheses fare no better. "Short-horizon accuracy does not imply long-horizon
stability" was published in TMLR with 750 models. "Architecture-free learnability" is
ill-posed — learnability is defined relative to a model family. "Direct multi-step prediction"
was already demonstrated by the same lab.

**A reviewer would say: you spent a week rediscovering the state of the art.**

### 1.2 Our one distinctive positive finding did not replicate

R4b found attractor dimension predicted where capacity stops paying off, correlation −0.71.
Clean mechanism, quantitative, novel. On 18 systems: **−0.22**. Noise.

That was the single most paper-shaped thing the project produced, and it was six systems
getting lucky.

### 1.3 The most useful-sounding result is a simulation artefact

The predictor — "32% of surrogate skill is predictable from cheap statistics" — **collapses to
zero under 1% observational noise**, and that was in the generous setting where the predictor
got clean-data statistics and only the model's training data was corrupted.

No real measurement is cleaner than 1%. So the headline result exists only in a regime nobody
deploys in.

### 1.4 The neural networks are not obviously doing anything

Across 108 systems, the trained surrogate beats a fifteen-line copy-the-most-similar-past-
moment baseline on **46 of them**, median skill ratio **exactly 1.00**.

A reviewer would ask, reasonably: if a lookup table matches your neural network on the
benchmark, what is the neural network for?

### 1.5 We never tested the thing the field actually argues about

REALM, Shikhman, the entire neural-operator literature we position against — all of it is about
**spatially extended systems**: fluid on a grid, flames on a mesh, thousands to millions of
variables. Every system we tested has **3 to 16 variables**.

The one experiment that would have bridged this (Kuramoto–Sivashinsky) **failed to run**. So
the strongest claim we can make is about low-dimensional ODEs, and there is no evidence it
transfers.

### 1.6 Someone already did the ambitious version, at 1.6× the scale

`arXiv:2510.02729` relates minimum attainable forecasting error to a pre-training complexity
statistic across **4,700 trained models**. We trained ~2,900 on a laptop. It is univariate and
does not cover rollout or architecture choice — but it occupies the framing.

### 1.7 The compute ceiling makes the central question unanswerable here

The crux is "is there an intrinsic ceiling, or are our models too small?" We can only sweep to
**10⁵ parameters**. The field operates at 10⁸–10⁹. Observing saturation at 10⁵ says nothing
about 10⁹, and we said so ourselves.

**Prosecution rests.** On the original problem statement as written, this is close to a
complete loss.

---

## Part 2 — What survives cross-examination

Four things. They are not what the proposal set out to find.

### 2.1 The horizon reconciliation — solid, modest

Capacity helps enormously at short horizons and progressively less further out: median benefit
0.94 at one step, 0.13 by five hundred, across 18 systems and 573 fits.

And the discriminating experiment is clean: a model that **cannot** accumulate error (direct
horizon-conditioned prediction) fails **earlier** than one that does. So the long-horizon wall
is an **information limit, not error accumulation.**

This reconciles a genuine published three-way disagreement — Gilpin (scale and data limit us),
Duraisamy (intrinsic ceilings), Chen (non-monotone) — by observing they measured at different
horizons.

*Weakness:* it is a reconciliation, not a discovery. And it is low-dimensional ODEs only.

### 2.2 The denoising explanation — new, mechanistic, and the best thing here

On clean data the network ties with copying (46/108). At 20% observational noise it wins on
**29 of 40**.

The mechanism is clean: **copying reproduces a past segment including its noise; a network
averages over thousands of examples and cannot.** So the network's real advantage over lookup
is *denoising* — and on clean data there is nothing to denoise.

This explains a pattern that at least five separate papers noticed without explaining, and it
is a direct criticism of how the field evaluates: **every benchmark in this area is clean
synthetic data, which is precisely the regime where a surrogate's main advantage is switched
off.**

Nobody appears to have said this. It is the one genuinely novel claim the project produced.

### 2.3 The measurement critique — cheap, checkable, useful

Three concrete ways the standard setup misleads, each with a mechanism and a fix:

- **Ordinal statistics read the sampling rate, not the system.** At the field-standard 100
  points per oscillation, the 0-1 chaos test called 5 of 45 chaotic systems chaotic. Corrected:
  35. And the correction *reorders* systems (rank correlation 0.14 before-to-after), so it is
  not a bias you can subtract.
- **About 20% of the standard benchmark carries a monotone clock coordinate** that makes
  one-step prediction an extrapolation problem and simultaneously destroys ordinal statistics.
  This manufactured our own R²=0.59 result out of nothing.
- **Timescale alignment does not equalise information.** Time-to-decorrelation ranges from 1 to
  229 steps across systems that are all "aligned".

### 2.4 The negative results — clean, and negatives are cheap to trust

- The published "more chaos needs a smaller model" claim **does not reproduce** at any of six
  tolerances on either of two map families.
- Iterative refinement helps for ~3 passes then **reliably degrades**; it never improves beyond
  its training budget.
- The Lyapunov exponent carries essentially no information about surrogate skill (+0.03).

---

## Part 3 — How much has the value changed?

Blunt assessment, using the project's own four criteria.

### The original problem statement

> *Can we characterise a task's learnability in advance, use that to predict which
> architectures will succeed, and train according to problem structure rather than brute-force
> search?*

| criterion | before | after | why |
|---|---|---|---|
| **Surprising to experts** | Believed high | **Low** | The core claims are 20-year-old results in nonlinear dynamics |
| **Fruitful** | Believed high | **Low–moderate** | The predictor does not survive 1% noise, so it cannot be built on |
| **Rigorous / falsifiable** | Unknown | **Good** — and it failed | We ran the discriminating tests. That is a success of process and a failure of hypothesis |
| **Feasible** | Assumed | **No** at the ambitious version | 10⁵ parameters cannot settle a question about 10⁹ |

**Value of the original statement: down sharply.** Not to zero — the process produced real
findings — but the specific thing proposed cannot be delivered from here, and would not be
novel if it could.

### What replaced it

| criterion | assessment |
|---|---|
| **Surprising** | **Moderate–high.** "Your benchmark switches off the thing your model is good at" is not a claim anyone in this area is making |
| **Fruitful** | **High.** Evaluation methodology is upstream of every result in the field |
| **Rigorous** | **Good but incomplete.** Strong evidence on low-dimensional ODEs, none on spatially extended systems |
| **Feasible** | **Yes** — mostly already done, on a laptop |

**Net:** the *proposal* lost most of its value. The *project* gained a different and arguably
better one. That is not a consolation prize — it is the expected outcome of a well-run
exploration phase, and it is exactly what the research philosophy this project follows says
should happen.

---

## Part 4 — What would give it a proper purpose

Three candidate reframings, argued honestly, then a recommendation.

### Option A — "Neural surrogate benchmarks measure the wrong thing"

**The claim.** The standard evaluation setup in this field systematically hides what matters,
in at least five specific and demonstrable ways, and the corrections change conclusions.

**Evidence already in hand:**

1. Ordinal statistics read the sampling rate (5/45 → 35/45)
2. 20% of the benchmark carries clock coordinates that make prediction extrapolation
3. Timescale alignment does not equalise information (decorrelation 1 to 229 steps)
4. Capacity improvements are short-horizon only, and papers report short-horizon metrics
5. Networks tie with parameter-free copying on clean data — and **every benchmark is clean data**

**The constructive half**, which turns a critique into a contribution:

- Report **context parroting** as a mandatory baseline. If your surrogate does not beat copying,
  you have measured attractor recurrence, not learning.
- Report **at a realistic noise level**, because that is where surrogates actually beat lookup —
  and we can show the crossover.
- Report **pointwise and structural error separately**, because capacity improves one and not
  the other.
- State the **sampling convention** explicitly, because the statistics are not properties of the
  system without it.

**Against:** it is a critique paper. Some reviewers discount those. And it needs at least one
spatially extended system to be credible.

**For:** it is the only framing where the evidence we have is *sufficient* rather than
suggestive, every claim is falsifiable and already tested, and the recommendations are cheap to
adopt.

### Option B — "Why surrogates fail at long horizons, adjudicated"

**The claim.** Three published positions disagree; they are reconcilable; the wall is
informational rather than procedural, and here is the experiment that separates them.

**For:** it answers a live, named disagreement, and the discriminating experiment (direct
prediction fails *earlier*) is genuinely clean.

**Against:** it is a reconciliation, not a discovery. Reviewers may see "they were measuring
different horizons" as bookkeeping. And the mechanism question — *what* runs out — is left
open, which is the part anyone would actually want.

### Option C — "Learnability prediction is a simulation artefact"

**The claim.** Predicting surrogate skill from cheap statistics works in simulation (32%) and
collapses entirely under 1% observational noise. Here is where the cliff is.

**For:** it is a *useful negative* — it tells a whole line of work not to bother, with a
mechanism. Directly relevant to the "accuracy law" line of research.

**Against:** thin on its own. One noise model, 40 systems, and no positive contribution. Better
as a section of A than a paper.

### Recommendation

**Option A as the spine, with B and C as sections.** The reasons:

1. It is the only framing where the evidence is **sufficient** rather than suggestive.
2. It contains the one **genuinely novel** claim — the denoising explanation for the copying tie.
3. It is **actionable**: four concrete changes to how people evaluate.
4. It is **honest** about what happened. We set out to build a predictor, found the predictor
   is a simulation artefact, and the reason why turned out to be more interesting than the
   predictor.

---

## Part 5 — What has to be true for this to be worth writing

Three things, in order. Without the first, this is a laptop study of toy systems.

### 5.1 One spatially extended system — non-negotiable

The whole critique is aimed at the neural-operator literature, and we have tested nothing
resembling it. **Kuramoto–Sivashinsky on a 64-point grid** is the standard, cheap bridge.

Our attempt failed and the honest reading is that our integrator is wrong, since the canonical
published parameters diverge the same way. **The fix is to take a validated implementation
rather than debug ours further.**

Then re-run four checks on it: does the copying tie hold on a 64-dimensional state? does the
capacity-versus-horizon pattern hold? does the noise crossover hold? does direct prediction
still fail earlier?

**If they hold, the critique is well-founded. If they do not, that is also publishable** — the
effects are dimension-specific, which nobody has shown either.

### 5.2 The denoising claim needs a proper test

Right now it rests on one comparison at one noise level with one noise model. To carry a paper
it needs:

- a finer noise ladder to locate the crossover precisely
- at least one realistic noise model (time-correlated, not independent per coordinate)
- and the mechanism tested directly — e.g. does an explicitly denoised copying baseline close
  the gap? If it does, that confirms the explanation; if not, something else is going on

### 5.3 Rerun the headline numbers at batch 128

N5 showed our recipe sits ~1.7× above achievable error. Relative comparisons are unaffected,
but anything quoted as an absolute number needs the better recipe.

### What would make it *not* worth writing

Stated in advance, so it can be checked rather than argued about later:

- If the copying tie **disappears** on a spatially extended system, the central novel claim is
  low-dimension-specific and the paper loses its spine.
- If the noise crossover **does not reproduce** under a realistic noise model, the denoising
  explanation is an artefact of independent Gaussian noise.
- If someone has already published the parroting-versus-surrogate comparison at scale, this
  becomes a replication.

---

## Part 6 — The one-paragraph version

The original proposal asked whether we could predict, before training, which architectures suit
a problem. The answer is: **partly, in simulation, and not at all once measurements are 1%
noisy** — so the thing it proposed to build cannot be built. Along the way the project found
something the proposal was not looking for: **on clean synthetic data, a neural surrogate is
statistically indistinguishable from copying the most similar moment in the past, and its real
advantage — averaging away noise — is exactly what clean benchmarks switch off.** That, plus
four demonstrable ways the standard evaluation setup misleads, is a smaller, sharper, and
better-evidenced contribution than the one originally proposed. It needs one spatially extended
system to be credible, and that is the next thing to build.
