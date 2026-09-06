# Literature review: what is already known

Compiled 2026-09-06, day 1. Purpose: establish what the proposal's claims would be
competing against, *before* committing to a research question. Per
`research-philosophy.md`, the most common and most justified failure is a reviewer
showing the claim was already known.

Every entry below was retrieved this session. Where only the abstract was retrieved,
that is stated.

---

## 0. What the proposal actually claims

`proposal/lossfunk_proposal.pdf` ("Predicting Predictability: When can a neural network
skip the simulation?"). Core claims, in the proposal's own terms:

- **C1.** How far a surrogate can leap ahead before it breaks ("leap horizon") is
  largely a property of the **system**, not the model, and is measurable before training.
- **C2.** **Learnability is a separate axis from chaoticity.** The largest Lyapunov
  exponent does not tell you what you need. Two equally chaotic systems can differ in
  whether a self-contained coarse description exists.
- **C3.** Computational irreducibility can be turned from a binary philosophical claim
  into a **continuous, measured** quantity across rule space.
- **C4.** A predictor fit on part of rule space will **generalise to held-out rule
  families**, beating a Lyapunov baseline by more than seed variance.

Crux experiment the proposal itself names: sweep model capacity and training-set size
over ~2 orders of magnitude; if leap horizon keeps climbing, the ceiling is the model and
the project dies; if it saturates at system-dependent levels, the ceiling is the system.
Preregistered kill criterion: if Lyapunov exponent + entropy rate explain >= 90% of
variance in leap horizon, report and stop.

---

## 1. The eight sources provided

### 1.1 Israeli & Goldenfeld 2003/2006 — coarse-graining beats irreducibility
`arXiv:nlin/0309047`, PRL 92, 074105 (2004); extended as `nlin/0508033`, Phys. Rev. E 73,
026203 (2006).

Computationally irreducible cellular automata can be **predictable and computationally
reducible at a coarse-grained level**. They construct exact local coarse-grainings for CA
in all Wolfram classes; at least one CA that is both irreducible and a universal Turing
machine becomes predictable when observed coarsely.

*Bearing on the proposal.* This is the intellectual foundation of the proposal's most
interesting predictor ("closure of a coarse description"). It also **weakens C3**: the
existence of a coarse closed description is already the accepted answer to "when is an
irreducible system predictable". The contribution cannot be the idea; it can only be the
measurement at scale, and its connection to neural surrogates specifically.

*Also a warning.* Israeli-Goldenfeld coarse-grainings are found by **search over supercell
maps**, and only a subset of rules admit one. "What counts as a fair coarse-graining" is
exactly the judgment call the proposal flags as the human core. That degree of freedom is
also a rigor hazard: a sufficiently flexible coarse-graining search can manufacture
closure.

### 1.2 Cecconi, Falcioni & Vulpiani 2003 — predictability != Lyapunov, already
`arXiv:nlin/0307013`.

Review connecting Lyapunov exponents, Kolmogorov-Sinai entropy, Shannon entropy and
algorithmic complexity. Emphasises **finite-resolution effects**: at finite observational
resolution the standard indicators are the wrong ones and need generalising
(epsilon-entropy, finite-size Lyapunov exponent). Explicitly treats distinguishing chaos
from noise, and the modelling problem.

*Bearing on the proposal.* **This is the strongest hit against C2's novelty.** In the
nonlinear-dynamics community, "the largest Lyapunov exponent is not the relevant
predictability measure at finite resolution" has been standard for over twenty years.
See also Boffetta, Cencini, Falcioni & Vulpiani, *Predictability: a way to characterize
complexity*, Phys. Rep. 356 (2002), the long version.

C2 is therefore **not surprising to nonlinear-dynamics experts**. It may still be
surprising to the ML surrogate community, which does lean on Lyapunov time as the unit of
account. That is an audience-relative novelty, and the philosophy warns that "novel for
you but not for the field" is the classic failure. It would need to be framed honestly as
*importing a known result into a field that ignores it*, which is a weaker contribution.

### 1.3 Chen, Shen, Fain & Nussinov 2025 — capacity is non-monotone in chaos
`arXiv:2512.01558` (v2, June 2026), "Neural Network Perturbation Theory".

Learn residual corrections after analytically subtracting a known exact solution.
Three-body problem, mass parameter swept 0.05x-30x Jupiter. Finding: **required network
capacity peaks at intermediate complexity and then falls by ~47% in the fully chaotic
regime** ("ergodic smoothing": trajectory-specific detail becomes noise, so only
statistically smooth corrections need capturing).

*Bearing on the proposal.* Directly relevant and **useful**: independent evidence that the
map from chaoticity to model difficulty is non-monotone, which supports C2's spirit. It
also supplies a mechanism the proposal does not currently have. But it undercuts the
framing that this is unexplored territory.

### 1.4 Achille et al. 2019 — Task2Vec
`arXiv:1902.03545`.

Fisher-information-based fixed-dimension embedding of a task via a probe network; a
meta-model on those embeddings **predicts which feature extractor will perform well**,
near-optimally, at a fraction of the cost of exhaustive evaluation.

*Bearing on the broader proposal.* Canonical prior art for "a meta-model predicts
architecture performance from task characteristics". The idea is not novel. What is
unestablished is whether it transfers to **dynamical-system surrogates and to long-horizon
failure modes** rather than i.i.d. classification accuracy.

### 1.5 Bahri, Dyer, Kaplan, Lee & Sharma 2024 — explaining neural scaling laws
PNAS 121(27) e2311878121; `PMC11228526`.

Four scaling regimes (variance-limited / resolution-limited, in data and in parameters).
Resolution-limited exponents derive from the model discretising a smooth data manifold,
with exponent set by **intrinsic manifold dimension**. Performance converges to a finite
floor when the limits set by data structure are reached.

*Bearing on the proposal.* This is the **theoretical machinery the crux experiment needs
and currently lacks**. "Does the leap horizon saturate under capacity and data scaling" is
precisely a question about which scaling regime the surrogate is in. Framing the crux as a
scaling-law measurement — fit exponent and floor per system, then ask which system
properties predict the floor — is far more rigorous than eyeballing a saturation curve,
and connects to an audience that already cares.

### 1.6 Mahmood, Lucas, Alvarez, Fidler & Law 2025 — optimal data collection
JMLR 26 (2025) 1-52.

Formalises data collection as an optimisation problem with performance targets, costs and
penalties; Learn-Optimize-Collect beats extrapolating neural scaling laws for deciding how
much data to gather.

*Bearing on the proposal.* Adjacent rather than central. The relevant transfer is the
decision-theoretic framing: the practical value of a pre-training predictor is not its R^2
but the **cost of acting on it**. If a learnability predictor is sold as a compute-saving
device, it should be evaluated as a decision rule with an explicit cost model, not as a
regression.

### 1.7 Mao et al. 2025 — REALM benchmark, "the illusion of mastery"
`arXiv:2512.18595` (v2, Feb 2026).

11 high-fidelity multiphysics datasets, a dozen-plus architectures. Three findings:
(1) a **scaling barrier tied to dimensionality, stiffness and mesh properties** causes
escalating rollout error; (2) **architectural design pattern matters more than model
size**; (3) standard accuracy metrics are **disconnected from physical reliability**
(high correlation, missed transients and integral quantities).

*Bearing on the proposal.* Finding (1) is a partial pre-emption of C1: system properties
are already known to gate rollout error. Finding (3) independently confirms the proposal's
own "your error metric hides things" alternative explanation — good, the concern is real
and there is precedent for how to answer it.

### 1.8 Shikhman 2026 — failure modes of neural operators
`arXiv:2601.11428` (v7, May 2026).

3 architectures x 5 PDE families, **750 trained models**, baseline-normalised degradation
factors plus spectral and rollout diagnostics. Headline: **"strong in-distribution
accuracy does not reliably predict robustness"**, and failure mechanisms depend jointly on
architecture and PDE type.

*Bearing on the proposal.* Directly answers one of the user's listed questions — "is
short-horizon predictive accuracy a reliable proxy for true rule learning?" — with **no,
and this is already published**. Any project proposing to establish that would be
re-deriving a known result. Cite it, do not claim it.

---

## 2. The decisive paper, which was not in the provided list

### Gilpin 2023 — *Model scale versus domain knowledge in statistical forecasting of chaotic systems*
`arXiv:2303.08011`, Phys. Rev. Research 5, 043252 (2023). Abstract retrieved; full text
not yet read.

24 forecasting methods x 135 low-dimensional chaotic systems x 17 metrics, on the `dysts`
database (Gilpin, NeurIPS 2021 Datasets & Benchmarks, `arXiv:2110.05266`), where every
system is annotated with largest Lyapunov exponent, entropy, fractal dimension and other
invariants, with timescales aligned across systems.

Findings, in substance:
- large domain-agnostic models stay accurate to **~2 dozen Lyapunov times**;
- **"invariant properties of the underlying dynamical systems only weakly correlate with
  the ability of the best-performing forecast models to forecast them"**;
- **"accuracy decorrelates with classical invariant measures of predictability like the
  Lyapunov exponent"**;
- stated conclusion: **scale and dataset availability, rather than intrinsic dynamical
  properties, limit current ability to forecast chaos.**

**This is the most consequential paper for this project, and it cuts both ways.**

*Against the proposal.* C2 ("learnability comes apart from chaoticity") has been measured,
at scale, on exactly the kind of substrate the proposal proposes, and published in a
physics journal. Presenting it as a new finding would be the textbook novelty failure.

*For the proposal.* Gilpin's **explanation** of the residual is the opposite of the
proposal's, and he did not run the experiment that discriminates them. He observes
decorrelation and attributes it to model scale and data availability. The proposal
attributes it to an intrinsic per-system ceiling. **These are competing explanations of
the same measured residual, and the crux experiment the proposal already describes — a
controlled capacity x data sweep, per system — is exactly the experiment that separates
them.** Nobody appears to have run it.

That reframing turns the project from "discover that learnability != chaoticity" (known)
into "adjudicate a published disagreement about *why* learnability != chaoticity" (open,
named, falsifiable in both directions).

*Practical asset.* `dysts` is pip-installable, gives 135 low-dimensional ODEs with aligned
timescales and precomputed invariants, and is cheap enough to run on a CPU. It removes
weeks of substrate-building and makes results directly comparable to a published baseline.

Follow-ups to read: *Zero-shot forecasting of chaotic systems* (`arXiv:2409.15771`);
*Context parroting: a simple but tough-to-beat baseline for foundation models in
scientific ML* (`arXiv:2505.11349`) — the latter matters because a trivial baseline
beating sophisticated models would also explain Gilpin's decorrelation.

---

## 3. The rest of the landscape, by claim

### 3.1 "Learnability" as a formal, model-relative quantity — solved, and the answer is uncomfortable
**Xu, Zhao, Song, Ermon & Ermon, ICLR 2020, *A Theory of Usable Information Under
Computational Constraints*** (`arXiv:2002.10689`). Predictive **V-information**: how much
information X carries about Y *when the observer is restricted to model family V*. Reduces
to Shannon mutual information when V is unrestricted, to R^2 under other restrictions,
violates the data-processing inequality (computation can create usable information), and is
estimable with PAC-style guarantees.

**This is the formalisation of the user's Learnable / Memorizable / Not-learnable taxonomy,
and it already exists.** It also delivers a verdict on the user's question "can learnability
be measured independently of architecture": **no, not coherently.** Learnability is
inherently indexed to a predictive family. An architecture-free learnability number is
either (a) the Shannon/Bayes quantity, which ignores computation and is not what anyone
means, or (b) implicitly indexed to some reference family that must be declared. Any project
proposing an "architecture-independent learnability measure" must answer this or it is dead
on arrival.

Memorisation vs generalisation is likewise formalised: Zhang et al. 2017 (rethinking
generalization), Feldman 2020 (long-tail memorization is *necessary*), Jiang et al.
C-scores, Swayamdipta et al. Dataset Cartography, Toneva et al. forgetting events. The
three-way taxonomy maps onto existing machinery rather than needing new machinery.

### 3.2 Complexity measures vs neural predictability — partially tested
- **Marzen, Riechers & Crutchfield, `arXiv:2303.14553`**, *Complexity-calibrated
  benchmarks...*: generate processes from large epsilon-machines with **known** statistical
  complexity, use Fano's inequality as the optimal-error bound, and show next-generation
  reservoir computers sit **>= 60% above the minimal attainable error** on highly
  non-Markovian processes. This is the computational-mechanics-vs-neural-learnability test
  the proposal lists as its most theoretically interesting predictor, already run for RC on
  stochastic processes. Not run for neural surrogates of spatiotemporal systems.
- **Garrido 2026**, *Evaluating predictability in deterministic cellular automata through
  machine learning*, Array 30:100751 (open access). LSTM vs Transformer on Elementary CA.
  LSTM >99% on short/mid-range rules, fails on Rule 30; Transformer perfect on Rule 90 and
  62 at higher cost. Conclusion: **different architectures suit different forms of
  structural complexity.** Small in scope (ECA only, two architectures, no rule-space map,
  no capacity sweep) but it occupies part of the proposal's substrate-1 territory.
- Wolfram-adjacent and APL ML 2024 work on CA-generated images showing intrinsic CNN limits
  exists but is weaker.
- `arXiv:2606.30512` (*Informational Frustration...*) claims a sharp learnable-to-unlearnable
  phase transition. **Treat with suspicion**: theorem-heavy, evidence-light, single-group
  preprint. Not a citable anchor; flagged only because it would collide with a "phase
  boundary in rule space" framing.

### 3.3 Rollout error, Lyapunov exponents, and intrinsic ceilings — actively contested
- **Duraisamy 2026**, *Predictivity and Utility of Neural Surrogates of Multiscale PDEs*
  (`arXiv:2604.20061`). Argues surrogate successes occur on low-dimensional solution
  manifolds where any competent reduced model interpolates; identifies **spectral bias** and
  **irreversible information loss under coarse-graining** as mechanisms; states that in many
  multiscale problems **no architecture or training procedure can recover what the coarse
  representation discards**. Also argues that with positive Lyapunov exponent and a broadband
  spectrum there is a finite T* beyond which trajectory error exceeds any threshold
  *regardless of training*.

  This is a **direct competitor claim to C1** — and note it is the *pro*-Lyapunov position,
  contradicting Gilpin. The field genuinely disagrees here. Good news for an
  adjudication-shaped project, bad news for a discovery-shaped one.
- Standard mitigations exist and must be baselines, not novelties: pushforward trick and
  temporal bundling (Brandstetter et al., *Message Passing Neural PDE Solvers*), PDE-Refiner
  (NeurIPS 2023), predicting the increment rather than the state (`arXiv:2412.13074`),
  noise-inspired regularisation.
- Correlation time / valid prediction time measured **in Lyapunov times** is the established
  metric in this literature (reservoir computing reaches ~5-6 Lyapunov times on KS, hybrid
  schemes ~12, parallel + dimensionality reduction ~9.2). The proposal's "leap horizon" is a
  cousin of VPT. **It must be positioned against VPT explicitly**, and normalising by
  Lyapunov time is exactly what makes "residual after Lyapunov" meaningful.

### 3.4 Direct multi-step vs autoregressive rollout — a live area, not a gap
The user's "why iterate t -> t+1 -> ... -> t+63 instead of predicting t -> t+63" is a
recognised design axis with existing answers on both sides:
- Direct horizon-conditioned prediction: **Lakshmanan & Chopra 2026, *Hybrid Neural World
  Models*** (`arXiv:2605.28317`) — one network with **continuous horizon conditioning**
  predicts any future state at horizon T in one forward pass, across reaction-diffusion,
  compressible Euler and rigid-body collisions; multi-horizon training is shown to matter
  (single-horizon retraining is 1.7x-6.0x worse at h=64). This is the Lossfunk lab paper the
  proposal builds on, and it has already established the direct-leap capability.
- Autoregression-free neural operators (`arXiv:2605.25413`), latent structured spectral
  propagators (`arXiv:2605.10154`), multi-stepsize mixture-of-experts operators
  (`arXiv:2604.12794`), temporal bundling, DYffusion-style approaches.
- Known trade-off: direct prediction avoids error accumulation but loses the semigroup
  structure and needs a separate output per horizon (or conditioning); autoregression reuses
  one map but compounds error and suffers spectral bias.

So "can direct multi-step replace iterative rollout" is **not an open question in the binary
form**. The open form is quantitative: *for which systems* does the direct route win, by how
much, and is that predictable in advance. That is the same system-property question again —
convergent evidence about where the real gap is.

### 3.5 Predicting architecture performance without training — established and known fragile
- Zero-cost NAS proxies (snip, grasp, synflow, zen-score, NASWOT, NEAR, ...).
- **NAS-Bench-Suite-Zero, NeurIPS 2022** (`arXiv:2210.03230`): 13 proxies x 28 tasks.
  Verdict: **simple baselines like parameter count and FLOPs are competitive with all
  existing zero-cost proxies in most settings, and most proxies do not generalise across
  benchmarks.**

  The single most important cautionary result for the meta-model hypothesis. Any "predict the
  architecture's performance in advance" project **must** include the dumb baselines
  (parameter count, one short training run, a linear or nearest-neighbour model on trivial
  features) and beat them on **held-out families**, not held-out samples. The base rate for
  this class of result failing to replicate is high.
- Learning-curve extrapolation and short-run probes (TuneAhead `arXiv:2606.17660` and
  predecessors) are the "cheat" baseline: a short training run is often a better predictor
  than any training-free statistic, and is genuinely cheap. It must be in the comparison, and
  it sets a hard bar.

### 3.6 Small specialised models vs large general models — real, but a different project
- **TRM** (Tiny Recursive Model): ~7M params, 45% on ARC-AGI-1, 8% on ARC-AGI-2, beating much
  larger "thinking" LLMs; follow-up analysis `arXiv:2512.11847` (inductive biases, identity
  conditioning, test-time compute) and a Mamba-2 hybrid (`arXiv:2602.12078`).
- **Important caveat found in the follow-up literature**: test-time augmentation and
  1000-sample majority voting account for roughly **+11 points of Pass@1** over single-pass
  canonical inference. The headline "7M beats 100x larger" is therefore partly a
  test-time-compute result, not purely a parameter-efficiency result. Anyone citing TRM as
  evidence that parameter count is the wrong axis must handle this.
- This branch (specialisation, MoE, parameter count vs capability) is a coherent research area
  but shares almost no infrastructure with the dynamical-systems branch. Under a 5-day budget
  the two cannot both be done. Listed here so the choice is explicit.

---

## 4. Verdict on each hypothesis

| # | Hypothesis (from the proposal or the user's brief) | Status | Note |
|---|---|---|---|
| 1 | Learnability is a separate axis from chaoticity / Lyapunov | **Known** | Cecconi 2003, Boffetta 2002 (theory); Gilpin 2023 (measured, 135 systems) |
| 2 | Short-horizon accuracy does not imply long-horizon stability | **Known** | Shikhman 2026 (750 models), REALM 2025, the whole rollout literature |
| 3 | Coarse-grained descriptions rescue irreducible systems | **Known** | Israeli & Goldenfeld 2004/2006 |
| 4 | A meta-model can predict architecture performance pre-training | **Known idea, fragile in practice** | Task2Vec; NAS-Bench-Suite-Zero says params/FLOPs baselines are competitive |
| 5 | Learnable / memorizable / unlearnable taxonomy is formalisable | **Already formalised** | V-information; Feldman; C-scores. The taxonomy is a re-description |
| 6 | Learnability can be measured **independently of architecture** | **Probably ill-posed** | V-information shows learnability is indexed to a predictive family. Needs reformulation, not measurement |
| 7 | Direct multi-step prediction can replace autoregressive rollout | **Known to be possible** | Hybrid Neural World Models; AFNO. The open part is *when* it wins |
| 8 | Rollout collapse is an intrinsic system ceiling, not underfitting | **OPEN and contested** | Duraisamy 2026 says yes-ish; Gilpin 2023 says no, it is scale and data. **No controlled capacity x data sweep exists** |
| 9 | Which system properties predict *residual* error after Lyapunov is regressed out | **OPEN** | Gilpin reports weak correlation but does not model the residual |
| 10 | Whether a learnability predictor generalises to **held-out system families** | **OPEN** | This is the part that would beat the NAS-proxy failure mode |
| 11 | Small specialised models beat large general ones on structured tasks | **Partly known, confounded** | TRM real, but the test-time-compute confound is large |
| 12 | Statistical complexity / excess entropy predicts *neural* learnability | **Partly open** | Done for reservoir computers on epsilon-machine processes (Marzen 2023); not for surrogates of spatiotemporal systems |

---

## 5. Where the real gap is

Three survive as genuinely open, and they are the same gap seen from three angles:

**The field has a published disagreement about why neural surrogates fail at long horizons,
and the discriminating experiment has not been run.**

- Gilpin 2023: forecast skill decorrelates from Lyapunov exponent; **cause is model scale and
  data availability**.
- Duraisamy 2026: there are **intrinsic ceilings** set by spectral bias and coarse-graining
  information loss that no architecture or training procedure recovers.
- Chen et al. 2025: required capacity is **non-monotone** in chaoticity, which neither of the
  above predicts.

These cannot all be right. Separating them needs a per-system controlled sweep over model
capacity and training-set size, with the horizon at which error crosses threshold as the
dependent variable, and the resulting saturation level (or absence of one) regressed on
system invariants. That is small-model, low-dimensional, many-short-runs work — the exact
shape that fits a 2-core CPU — and `dysts` supplies the substrate with invariants already
computed.

The claim that would be surprising, fruitful and defensible is **not** "learnability is not
chaoticity" (known). It is closer to:

> *For a given system there is a measurable saturation level of surrogate skill that further
> capacity and data do not move; this level varies across systems by more than seed noise;
> and it is not explained by the Lyapunov exponent.*

with the honest negative outcome — that skill keeps climbing with capacity and Gilpin is
simply right — being an equally publishable result, and one that would matter to the people
currently buying GPUs on the opposite assumption.

## 6. Threats to carry forward into design

1. **Underfitting masquerading as irreducibility.** The proposal names this. It is the whole
   ballgame, and on a 2-core CPU the risk is *severe*, because "our biggest model was small"
   is exactly the criticism. Mitigation: sweep capacity across orders of magnitude and report
   the **trend**, never a single point; state at what capacity saturation is claimed; say
   plainly that saturation observed up to 10^5 params does not prove saturation at 10^9.
2. **Metric artefacts.** MSE on pattern-forming systems conflates phase error with structural
   failure (REALM finding 3 confirms). Need at least one structure-aware metric alongside a
   pointwise one, and claims must hold under both.
3. **Baseline inflation.** NAS-Bench-Suite-Zero's lesson. Parameter count, a short training
   run, and one trivial statistic must appear as baselines.
4. **Selection over systems.** `dysts` is a fixed public list, which helps; any additional
   systems must be fixed before results are seen.
5. **Family-level generalisation.** Held-out *families*, not held-out samples, or the result
   is not about what it claims to be about.
