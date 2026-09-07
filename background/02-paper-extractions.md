# Paper extractions

Compiled day 1. Every paper below was retrieved and read this session (full text where an
HTML or PDF full text was reachable; abstract-plus-metadata where it was not, and that is
stated per entry). Extraction template is the user's:

1. Problem  2. Central hypothesis  3. Assumptions  4. Method  5. What the experiment
actually shows  6. Limitations  7. What is genuinely new  8. How it helps us
9. Gaps / trends visible here that we may not have

`Verification` line on each entry records what was actually retrieved, so nothing here is
resting on memory.

---
---

# Part A — the eight sources provided

---

## P1. Israeli & Goldenfeld (2003) — *On computational irreducibility and the predictability of complex physical systems*

`arXiv:nlin/0309047v2` · Phys. Rev. Lett. **92**, 074105 (2004) · 4 pages
Verification: full HTML text read.

**1. Problem.** Wolfram's computational irreducibility (CIR) says that for some systems no
shortcut exists — you must simulate step by step. If physical systems are CIR, prediction
is hopeless. Is that actually true for the questions physicists ask?

**2. Central hypothesis.** CIR is a statement about *microscopic* description. Physics
usually only needs coarse information. So a CIR system can be predictable, and even
computationally *reducible*, at a coarse-grained level.

**3. Assumptions.**
- 1D nearest-neighbour binary elementary CA (ECA) are an adequate model system.
- A legitimate coarse-graining is *local* and *stroboscopic*: block N cells, run N time
  steps, project the block alphabet down.
- Wolfram's four-class taxonomy is the right frame (they note it may need refining).

**4. Method.** Build the N-th block automaton `A^N` — each cell of `A^N` is a block of N
cells of `A`, one step of `A^N` is N steps of `A`. Apply a projection `P` from the block
alphabet `{0..S_A^N - 1}` onto a smaller alphabet. The coarse-graining is *valid* only if
the induced rule is single-valued, i.e.

> `f_A^N[x1,x2,x3]‾ = f_A^N[y1,y2,y3]‾` for all x,y with `x̄_i = ȳ_i`

If that fails the coarse rule is multi-valued and the attempt fails. They then **scan**
(N, P) for all 256 ECA rules. Note their own characterisation: the procedure is **"not
constructive, but instead a self-consistency condition"** — there is no algorithm that
finds a coarse-graining, only a test that checks a candidate.

**5. What the experiment actually shows.**
- **240 of 256 ECA rules (93.75%)** admit a valid coarse-graining, spanning all four
  Wolfram classes.
- **Rule 110** — class 4, universal Turing machine, provably CIR — coarse-grains to
  **Rule 0** at N=5. Its coarse-grained dynamics is trivially predictable.
- Rule 146 (class 3) → Rule 128 (class 1) at N=3, P(7)=1 else 0.
- Rule 105 (class 3) → Rule 150 at N=2.
- **16 rules resisted**: 12 class-3 (30, 45, 106 and their symmetries), 4 class-2
  (incl. 154).
- Search was capped at N ≤ 4 (larger N for a few specific cases) for compute reasons.

**6. Limitations (their words and ours).**
- *Theirs:* "We don't know if our inability [to coarse-grain the 16] comes from limited
  computing power or from something deeper." CIR status of rules 18, 54, 126 is unproven.
- *Ours:* the coarse-graining is found by **search over projections**. The space of
  candidate projections is a free parameter. A wide enough search manufactures closure —
  this is a rigour hazard, not just a caveat.

**7. Genuinely new.** First demonstration that a **provably irreducible** system has a
reducible coarse description; first systematic survey across a complete rule family; the
induced partial order on rule space via coarse-graining transitions.

**8. How it helps us.** This is the intellectual foundation for the proposal's most
interesting predictor — "does a closed coarse description exist?" It also supplies the
honest framing: the *existence* of coarse closure as the answer to irreducible-but-
predictable is 2004 knowledge. Our contribution cannot be that idea. It can only be
(a) measuring it continuously rather than binary, and (b) tying it to *neural* surrogate
behaviour, which they never touch.

**9. Gaps / trends we may not have.** Their fail-set is tiny and structured (Rule 30, 45,
106 — the class-3 chaotic core). If a "learnability" axis exists at all, this is direct
evidence it is **very coarse**: 94% of rule space collapses. A predictor whose target is
94%-one-class is a badly imbalanced regression problem and will look good for the wrong
reason. That is a concrete design trap for any CA-substrate spike.

---

## P2. Cecconi, Falcioni & Vulpiani (2003) — *Complexity Characterization of Dynamical Systems Through Predictability*

`arXiv:nlin/0307013` · Acta Phys. Polonica **34**, 3851 (2003) · 26 pp review
Verification: full HTML text read.

**1. Problem.** Lyapunov exponents and Kolmogorov–Sinai entropy are asymptotic objects
(t→∞, ε→0). Real observation is finite-time and finite-resolution. What are the right
predictability indicators then?

**2. Central thesis.** *Complex = incompressible = unpredictable* — unpredictability is a
measure of complexity — **but the standard indicators must be generalised to finite
resolution or they give the wrong answer.**

**3. Assumptions.** Deterministic bounded systems, ergodic invariant measure, smooth
trajectories; the analysis explicitly separates asymptotic from finite-resolution regimes.

**4. Method / tools.**
- **Lyapunov exponent:** `|δx(t)| ~ |δx(0)| e^{λ1 t}`; predictability time
  `T_p ~ (1/λ) ln(Δ/δ0)` — note the *logarithmic* dependence on tolerance, which is why
  λ alone is a weak lever.
- **KS entropy:** `h_KS = sup_A h(A) = lim_{ε→0} h(A_ε)`.
- **ε-entropy:** `h(ε,τ) = (1/τ) lim_{m→∞} (1/m) H_m(ε,τ)` — KS entropy at finite
  resolution ε. Recovers `h_KS` as ε→0.
- **Finite-Size Lyapunov Exponent (FSLE):** time `τ(δ,r)` for a separation δ to grow to
  rδ; `λ(δ) = ln(r)/⟨τ(δ,r)⟩_e`. `lim_{δ→0} λ(δ) = λ1`. For finite δ the value is set by
  the *nonlinear* dynamics, not the tangent map.
- **Brudno–White:** algorithmic complexity per symbol `C(x) = h_KS / ln 2`.

**5. What the worked examples actually show.**
- *Weakly coupled regular+chaotic system:* global λ ≈ λ_chaotic, so naive theory says
  `T_p ~ 1/λ_y` independent of coupling ε. **Actually** the regular component's error
  diffuses, `|δx(t)| ~ ε t^{1/2}`, giving `T_p ~ ε^{-2}`. The Lyapunov exponent is
  simply the wrong number.
- *Globally coupled maps:* FSLE has **two plateaus** — a microscopic one below
  δ_c ~ O(1/√N) and a much smaller macroscopic one above. An effective low-dimensional
  description emerges at coarse scale. (This is coarse-graining, in FSLE language.)
- *Chaotic diffusive map vs. stable map + noise σ:* for ε > σ the two are
  **indistinguishable**. At a realistic σ = 1e-4 you cannot tell deterministic chaos from
  noise from data.
- *PRNG:* chaotic at ε ≤ 1/N1, apparently random above. Multi-scale character.

**6. Limitations.** Asymptotic definitions vs finite data; `h(ε) ~ O(N)` in high
dimension makes resolution demands prohibitive; power-law ε-entropy `h(ε) ~ ε^{-α}` is
consistent with *at least four* mechanisms (chaotic diffusion, noise, high-d linear,
non-chaotic complicated), so it under-determines the model; the Gaspard "microscopic
chaos" claim is criticised for assuming determinism and having too little data
(~1e34 points needed for a clean answer in a real liquid).

**7. Genuinely new (in 2003).** FSLE and ε-entropy as an operational, scale-resolved
predictability toolkit; the explicit demonstration that chaos and noise are
observationally identical above a resolution scale; the "complexity helps modelling"
point — a stochastic model can be the *right* model at coarse scale.

**8. How it helps us.** This is the strongest single hit against the novelty of "learn-
ability is not chaoticity". In nonlinear dynamics that has been standard for 20+ years.
It also hands us better instruments than λ: **FSLE λ(δ) and ε-entropy h(ε) are exactly
"predictability as a function of scale"**, which is what a leap-horizon study is
implicitly measuring. If we use λ_max as our only baseline we are attacking a straw man
that this community abandoned in 2002.

**9. Gaps / trends we may not have.** Two.
(a) The **FSLE plateau structure is a ready-made, physics-blessed operationalisation of
"does a closed coarse description exist"** — a second plateau at δ_c *is* the emergent
macroscopic level. Nobody appears to have asked whether the scale at which a neural
surrogate's error saturates coincides with the FSLE plateau. That is a sharp, cheap,
testable link and it is ours to take.
(b) Their chaos/noise indistinguishability result implies a hard ceiling on any
"estimate learnability from data before training" programme: **above the noise scale the
data does not contain the answer.** Any pre-training learnability estimator must state
the resolution at which its claim holds.

---

## P3. Chen, Shen, Fain & Nussinov (2025) — *Neural Network Perturbation Theory (NNPT)*

`arXiv:2512.01558v2` (v1 Dec 2025, v2 Jun 2026) · physics.comp-ph · "about to submit to PRE"
Verification: abstract + metadata retrieved verbatim.

**1. Problem.** Many physical systems = exactly solvable part + perturbation. Using a
network on the whole thing wastes capacity re-learning the solvable part.

**2. Central hypothesis.** Learn only the **residual** after analytically subtracting the
known exact solution. And: the capacity required to do so is a diagnostic of the physics.

**3. Assumptions.** An exact solution exists and is subtractable; symplectic integration
is accurate enough that capacity effects are physical not numerical (they verify energy
conservation < 2e-4); an "equalised-accuracy protocol" at 1% tolerance is a fair way to
define required capacity; architecture family held fixed while width/depth vary.

**4. Method.** Three-body problem; Jovian mass swept `f = 0.05 → 30 ×` physical.
Architecture fixed in family, capacity varied; find the smallest network reaching 1%
tolerance at each f. Sequential/hierarchical correction stages tested.

**5. What the experiment actually shows.**
- **Non-monotonic capacity profile.** Required capacity **peaks at f=5** (late integrable
  regime; 3×32, 2242 params), stays elevated through the transition (f ≈ 15–17), then
  **falls 47% in the fully chaotic regime** (f ≥ 17; 2×32, 1186 params).
- Capacity transition at **f_c = 16.6 ± 2.8**, matching **Chirikov resonance overlap**.
- Second-stage corrections are negligible (`||y2||/||y1|| ≈ 0.997`) — one stage captures
  the dominant perturbative structure.
- Mechanism offered: **ergodic smoothing** — in the fully chaotic regime trajectory-
  specific detail becomes irreducible noise, so only statistically smooth corrections
  remain to be learned, and those are cheap.

**6. Limitations.** One system family (three-body); one architecture family; tiny networks
(~1e3 params) so "capacity" is measured in a regime far from anything deployed; "required
capacity" is defined by a 1% threshold, and threshold choice can move a non-monotonicity;
no seed counts reported in the abstract.

**7. Genuinely new.** A **measured non-monotonic** relation between chaoticity and required
model capacity, with a named mechanism (ergodic smoothing) and a matching physical
criterion (Chirikov). Both arms of the curve are surprising: harder-then-easier.

**8. How it helps us.** Independent evidence that difficulty-vs-chaoticity is **not
monotone**, which is stronger and more interesting than "not correlated". It also gives a
*mechanism* the proposal currently lacks: past a threshold, the unlearnable part becomes
*noise-like*, which is a different failure from "too complex to fit". That distinction —
**capacity-limited vs noise-limited** — is exactly the axis our crux experiment should
resolve, and NNPT shows it is measurable.

**9. Gaps / trends we may not have.** The ergodic-smoothing story predicts something
sharp that we should test: in the fully chaotic regime, **pointwise error should saturate
while distributional/statistical error keeps improving**. If true, the choice of metric
alone flips the sign of the conclusion. That is a confound we must design against, and it
connects directly to A3 (foundation models preserve attractor statistics after point
forecasts fail).

---

## P4. Achille et al. (2019) — *Task2Vec: Task Embedding for Meta-Learning*

`arXiv:1902.03545` · ICCV 2019
Verification: full text (ar5iv) read.

**1. Problem.** Choosing which pre-trained expert to use for a new task requires training
against all candidates. Expensive and O(N).

**2. Central hypothesis.** The Fisher Information Matrix of a *fixed probe network*,
evaluated on a task, is a fixed-dimension signature of that task; a meta-model over those
signatures can pick the right expert without training them all.

**3. Assumptions.** A single probe network (ResNet-34/ImageNet) is meaningful across
domains; diagonal FIM suffices; per-filter averaging preserves what matters; embeddings
are comparable across different label spaces; the tasks the experts were trained on are
known.

**4. Method.**
- Retrain **only the classifier head** on the task; compute the **diagonal FIM of the
  feature-extractor parameters** (not the head). Fixed dimension regardless of #classes.
- **Robust Fisher** via a variational objective:
  `L(ŵ;Λ) = E_{w~N(ŵ,Λ)}[H(y|x)] + β·KL(N(0,Λ)||N(0,λ²I))` — stabilises FIM in low data.
- **Model2Vec:** `m_i = F_i + b_i`, expert = its task embedding plus a learned bias.
- **Asymmetric distance:** `d_asym(t_a→t_b) = d_sym(t_a,t_b) − α·d_sym(t_a,t_0)` with
  α = 0.15, `t_0` the trivial embedding. Source-task *complexity* enters explicitly.

**5. What the experiment actually shows.** 1,460 tasks (207 iNaturalist, 25 CUB-200,
228 iMaterialist, 1,000 DeepFashion). Relative error increase over brute-force optimal:

| meta-task | optimal | chance | ImageNet | task2vec | asym task2vec | model2vec |
|---|---|---|---|---|---|---|
| iNat+CUB | 31.24% | +59.52% | +30.18% | +42.54% | **+9.97%** | **+6.81%** |
| Mixed | 22.90% | +112.49% | +75.73% | +40.30% | +29.23% | **+27.81%** |

Note the **symmetric** embedding (+42.5%) is *worse than just using ImageNet* (+30.2%).
Only the asymmetric variant wins. Embedding **norm correlates with task difficulty**;
domain-only embeddings collapse on iMaterialist where task embeddings do not.

**6. Limitations (theirs).** Task collection does not capture real variety; probe-network
choice matters a lot (ResNet/DenseNet ≫ VGG); poor with too few samples; needs to know the
experts' training tasks — black-box experts unusable; ignores task–model interaction
beyond the task itself.

**7. Genuinely new.** Fixed-dimension, label-space-independent task embedding; variational
robust Fisher; asymmetry as a transfer predictor; scale (1,460 tasks vs Taskonomy's 26).

**8. How it helps us.** This is the canonical prior art for the user's "meta-model predicts
which architecture works". **The idea is not novel and we must not claim it.** What is
genuinely unestablished is the transfer to *dynamical-system surrogates* and to
*long-horizon failure modes* rather than i.i.d. accuracy.

**9. Gaps / trends we may not have.** The table above is the warning. The naive version of
the idea **lost to a trivial baseline**; it only worked once an asymmetric correction for
source-task complexity was bolted on. Any surrogate-selection predictor we build must be
reported against (i) chance, (ii) "always pick the single best architecture overall",
(iii) one short training run. Expect the naive version to lose. Also: their embedding is
*model-relative by construction* (a FIM of a probe net) — quiet corroboration of A8's
verdict that architecture-free learnability is ill-posed.

---

## P5. Bahri, Dyer, Kaplan, Lee & Sharma (2024) — *Explaining neural scaling laws*

PNAS **121**(27) e2311878121 · PMC11228526
Verification: full text read.

**1. Problem.** Why are neural scaling laws power laws, what sets the exponent, and do the
data-scaling and parameter-scaling exponents relate?

**2. Central hypothesis.** Four distinct regimes with distinct mechanisms; in the
*resolution-limited* regimes the exponent is set by the **intrinsic dimension d of the
data manifold**, via nearest-neighbour spacing.

**3. Assumptions.** Data on a compact d-dimensional manifold with smooth target
`y = F(x)`; kernel / large-width (NTK, random-feature) regime; teacher–student with MSE;
Lipschitz/smoothness conditions on loss and target.

**4. Method — the four regimes.**

| Regime | Condition | Exponent | Mechanism |
|---|---|---|---|
| Variance-limited (data) | D large, P fixed | α_D = **1** | concentration to infinite-data limit |
| Variance-limited (width) | w large, D fixed | α_W = **1** | output fluctuations O(1/w) |
| Resolution-limited (data) | P ≫ D ≫ 1 | α_D ≈ **n/d** | NN distance ~ D^{-1/d} on the manifold |
| Resolution-limited (params) | D ≫ P ≫ 1 | α_P ≈ **1/d** | P points used to approximate smooth F |

Kernel-spectrum route: `λ_i ~ i^{-(1+α_K)}` ⇒ `α_D = α_P = α_K`, with `α_K ∝ 1/d` for
smooth kernels on a d-manifold.

**5. What the experiment actually shows.**
- Variance-limited α ≈ 1 **universally** across WRN / 4-layer CNN, MNIST / CIFAR-10 / -100,
  MSE and cross-entropy, ReLU and Erf. Very robust.
- Teacher–student with controlled input dimension: measured α_D matches the predicted
  4/d closely.
- On **real** data the manifold-dimension prediction is **weaker than theory**: the plot of
  4/α_D vs measured intrinsic dimension correlates less well for standard networks.
- α_D is insensitive to the number of classes but **sensitive to input noise** ⇒ the
  network's scaling is driven by input-manifold structure more than by the task.
- Both regime families converge to a **finite non-zero loss floor** `L(∞,P)` or `L(D,∞)`.

**6. Limitations (theirs).** Results asymptotic, experiments finite — the hierarchy
(P ≫ D or D ≫ P) breaks at large scale; "a precise definition of the data manifold is
lacking", proxies used; feature learning not captured (finite-width kernels evolve during
training); theory rigorous only for random features at large width.

**7. Genuinely new.** The four-regime taxonomy itself; the geometric derivation of
resolution-limited exponents from nearest-neighbour distance; the duality between
parameter- and data-scaling exponents; the explicit kernel-spectrum ↔ manifold-dimension
link; empirical demonstration of all four regimes.

**8. How it helps us.** **This is the missing statistical machinery for the crux
experiment.** The proposal's crux — "sweep capacity and data, does the leap horizon
saturate?" — is exactly the question of which scaling regime the surrogate is in. Framing
it as *fit an exponent and a floor per system, then ask which system invariants predict
the floor* is far more rigorous than eyeballing a curve, and it gives a principled
definition of "intrinsic ceiling": **the finite loss floor the fitted law extrapolates to.**

**9. Gaps / trends we may not have.** Two important ones.
(a) The finding that α_D is **noise-sensitive but class-count-insensitive** implies that
much of what we would call "task learnability" is really *input-manifold* geometry. For
dynamical systems the input manifold is the attractor, and its **fractal / Kaplan–Yorke
dimension is exactly a d** — so scaling theory predicts `α ≈ 1/d_KY`. **That is a concrete,
quantitative, falsifiable prediction linking a classical dynamical invariant to a surrogate
scaling exponent, and it is a much sharper hypothesis than anything currently in the
proposal.** Gilpin (A1) reports invariants correlate weakly with *accuracy*; nobody has
checked whether they predict the *exponent*. Accuracy and exponent are different objects.
(b) Their honest admission that real networks fit the manifold-dimension theory *worse*
than teacher–student ones is a warning about how much of our result would be architecture
artefact.

---

## P6. Mahmood, Lucas, Alvarez, Fidler & Law (2025) — *Optimizing Data Collection for Machine Learning*

JMLR **26** (2025) 1–52
Verification: full PDF parsed (52 pp).

**1. Problem.** How much data to collect, from which source, to hit a performance target
within budget and horizon — over-collect and you waste money, under-collect and you miss
the target and pay again.

**2. Central hypothesis.** Treat it as a formal stochastic optimisation problem over the
*distribution* of the data requirement, not as a point estimate from an extrapolated
scaling law. Doing so cuts failure rate dramatically at comparable cost.

**3. Assumptions.** The score function is a monotonically non-decreasing stochastic
process in dataset size; the minimum data requirement D* is an absolutely continuous
random variable with differentiable CDF; the model and training algorithm are fixed
across rounds; the target is achievable with finite data.

**4. Method.** Minimise expected total collection cost plus a failure penalty P over T
rounds with non-decreasing q_t:

`min_{q1≤...≤qT} Σ_t cᵀ(q_t − q_{t−1})(1 − F(q_{t−1})) + P(1 − F(q_T))`

where F is the CDF of D*. **Learn–Optimize–Collect (LOC):** (1) *Learn* F by bootstrap
resampling the observed performance statistics and fitting a KDE/GMM; (2) *Optimize* the
above by gradient descent; (3) *Collect*. Difference from scaling-law practice: the
baseline fits a curve, extrapolates a point estimate of D*, and collects that much. LOC
optimises **over the uncertainty in D***, which is asymmetric because under-collection
incurs a penalty and over-collection only costs money.

**5. What the experiment actually shows.** CIFAR-10/100, ImageNet, BDD100K, nuScenes,
PASCAL VOC; classification, semantic segmentation, 2D detection. Metrics: *failure rate*
and *cost ratio* vs an oracle that knows D*. LOC drives failure rate to near 0% where
regression baselines fail 30–100% of the time; for K=1 sources, failure rates fall
**40–90%** relative to regression baselines. Cost ratios often ≤ 1, sometimes higher for
K=2 but with vastly lower failure. Robust to order-of-magnitude changes in cost/penalty.

**6. Limitations (theirs).** Monotonicity assumed; target assumed achievable; fixed model
across rounds; **numerical experiments are simulations over pre-constructed ground-truth
learning curves, not live retraining**; extreme outlier decisions possible with K=2;
validation-set bias not handled.

**7. Genuinely new.** Formalising optimal data collection; the LOC framework; multi-round,
multi-source treatment with heterogeneous costs; principled handling of estimator
uncertainty rather than extrapolation.

**8. How it helps us.** Adjacent but load-bearing for framing. If a learnability /
architecture predictor is sold as a **compute-saving device**, the right evaluation is not
R² or Spearman ρ — it is a **decision rule with an explicit asymmetric cost model**: what
does it cost when the predictor is wrong in each direction? A predictor with ρ = 0.6 that
never mis-ranks the top architecture can be more valuable than one with ρ = 0.8 that
occasionally does. We should adopt this framing for whatever we build; it is cheap to add
and it is the kind of rigour reviewers reward.

**9. Gaps / trends we may not have.** Their headline result is that **point estimates from
scaling laws are systematically unsafe** because the loss is asymmetric. That directly
threatens the proposal's implicit product ("predict, then skip the training run"): a
predictor used to *skip* work has exactly this asymmetry, and none of the NAS/proxy
literature (A9) evaluates it that way. Adopting the decision-theoretic evaluation is a
small, concrete way our project can be more rigorous than its neighbours.

---

## P7. Mao et al. (2025) — *REALM: Benchmarking neural surrogates on realistic spatiotemporal multiphysics flows*

`arXiv:2512.18595v2` · 52 pp, 20 figs · code `github.com/deepflame-ai/REALM`
Verification: full HTML text read.

**1. Problem.** Neural-surrogate evaluation runs on simplified low-dimensional proxies, so
the field has an "illusion of mastery". What happens on realistic, stiff, multiphysics,
irregular-mesh reactive flows?

**2. Central hypothesis.** Current benchmarks systematically flatter surrogates; on
realistic regimes their fragility is exposed, and the controlling variables are
architectural inductive bias plus problem geometry — not scale.

**3. Assumptions.** High-fidelity CFD is ground truth; a common preprocessing/rollout
protocol makes heterogeneous architectures comparable; MSE-on-preprocessed-variables plus
correlation is an adequate primary metric (they then argue against this themselves).

**4. Method.** 11 datasets, four geometry classes:
- *2D regular:* IgnitHIT (1024², 36 traj), EvolveJet (800×550, 30 traj, 36 species),
  PlanarDet (840×400, 9 traj)
- *3D regular:* ReactTGV (256³ ≈ 1.6e7 cells, 16 traj), PoolFire (15 traj),
  PropHIT (1536×128×128, 8 traj)
- *2D irregular:* SupCavityFlame (~3e6 cells), ObstacleDet (~3.1e5), SymmCoaxFlame (~2.95e5)
- *3D irregular:* MultiCoaxFlame (~1.35e7 cells), FacadeFire (~1.8e5)

6–40 channels per case; 20–50 timesteps per trajectory. Models: FNO, FFNO, CROP, DPOT,
UNO, LSM (spectral); CNext (conv); FactFormer, Transolver, ONO, GNOT (transformer);
DeepONet, PointNet (pointwise); GraphSAGE, GraphUNet, MeshGraphNet (graph). Three capacity
tiers ≈ 1e6 / 1e7 / 1e8 params. Preprocessing: **Box–Cox (λ=0.1)** on species mass
fractions to compress a 1e-k dynamic range to O(1), then per-channel z-score. Training:
autoregressive multistep with backprop through the final step only; Adam + one-cycle.
Metrics: normalised prediction error, Pearson correlation, params, inference time, peak
memory.

**5. What the experiment actually shows.**
- *(i) Scaling barrier.* 2D regular → slow error growth; 3D regular → "markedly faster";
  irregular → "sharp increase in difficulty". On 3D irregular MultiCoaxFlame most models
  hit **unit relative L2 error within a few steps**. Shown as error curves; **no tabulated
  growth rates** — the trend is real, the quantification is loose.
- *(ii) Architecture ≫ parameter count.* DeepONet is poor on regular grids but achieves
  the **lowest loss across all irregular cases**. Lightweight CNext matches or beats much
  larger transformers. Spectral/conv win on regular meshes (resolution/translation
  invariance); graph nets over-smooth on irregular ones.
- *(iii) Accuracy ≠ trustworthiness.* On PlanarDet several models exceed **correlation
  0.8** yet fail to reproduce the temporal evolution of mean detonation cell size.
  Systematic train/test correlation gaps reveal real overfitting.

**6. Limitations.** Reactive multiphysics only; trajectory counts are *tiny* (6–36 per
case) so "overfitting" is partly a small-data artefact; loss on the final rollout step
only, no pushforward or noise injection — a known-weak rollout protocol that may understate
what careful training achieves; error-growth claims not tabulated.

**7. Genuinely new.** Realistic, stiff, irregular-mesh multiphysics at a scale where
classical solvers cost hundreds–thousands of CPU/GPU-hours per trajectory; unified
preprocessing across 16+ architectures; the explicit regular-vs-irregular geometry axis,
which PDEBench / The Well / PDEArena do not have.

**8. How it helps us.** Finding (iii) independently confirms the metric-artefact threat we
had already flagged, with a concrete example (corr > 0.8, wrong cell size). Finding (ii) is
a large-scale, independent statement that **parameter count is the wrong axis** — directly
relevant to the proposal's specialisation hypothesis, and evidence we can cite rather than
re-derive. Finding (i) partially pre-empts C1: system properties (dimensionality, stiffness,
mesh irregularity) are already known to gate rollout error.

**9. Gaps / trends we may not have.** Their difficulty axis is **geometric and numerical**
— dimensionality, stiffness, mesh irregularity — not dynamical (no Lyapunov exponent, no
entropy, no attractor dimension anywhere in the benchmark). Meanwhile the chaos-forecasting
literature (A1) uses only dynamical invariants and ignores stiffness and discretisation
entirely. **Neither community measures the other's variables.** A study that puts both
families of predictors into the same regression is genuinely unoccupied ground, and it is
the kind of thing that becomes obvious only after reading both literatures.

---

## P8. Shikhman (2026) — *Diagnosing Failure Modes of Neural Operators Across Diverse PDE Families*

`arXiv:2601.11428v7` · **Transactions on Machine Learning Research** · 17 pp
Verification: full HTML text read.

**1. Problem.** Neural PDE solvers are evaluated on in-distribution test error. Deployment
involves shifted coefficients, boundary conditions, discretisation and longer rollouts.
Does in-distribution accuracy predict robustness?

**2. Central hypothesis.** No — and failure patterns are jointly determined by architecture
and PDE family, so function-space generalisation under structured shift must be a
first-class evaluation target.

**3. Assumptions.** Three architectures are "deliberately different" enough to be
representative; baseline-normalised ratios isolate robustness from absolute accuracy;
worst-case-over-shifts is the right summary.

**4. Method.** Five shifts: (1) coefficient/parameter (larger κ in NLS, lower ν in
Navier–Stokes, higher σ in Black–Scholes, rougher Poisson coefficients); (2) boundary /
terminal-condition family; (3) resolution extrapolation with frequency-binned spectral
error; (4) long-horizon rollout beyond the training horizon; (5) input perturbation.
Five families: NLS (dispersive), Poisson (elliptic), Navier–Stokes (multiscale fluid),
Black–Scholes (financial parabolic), Kuramoto–Sivashinsky (chaotic).
Three architectures: FNO (width 64, depth 4, 16 modes 1D / 12×12 2D), CNO (width 64,
depth 5), DeepONet (width 128, depth 2), all with coordinate channels, Adam @ 1e-3.
**750 models = 50 seeds × 5 families × 3 architectures.**
Degradation factor `D = E_stress / E_base`, worst case over shifts.
Theory: Prop. 1 bounds rollout error growth as `ε Σ_j L^j` under composition; Prop. 2
splits resolution error into statistical and spectral-tail terms.

**5. What the experiment actually shows.**
- **FNO has the lowest baseline error on all five families and ranks poorly under stress.**
- **Poisson reversal:** FNO baseline best → **18.4×** degradation under resolution shift;
  CNO worst baseline → **1.05×**.
- FNO also degrades 11.2× (Poisson perturbation), 6.3× (Black–Scholes payoff shift),
  1.6× (KS rollout).
- DeepONet most consistently stable; 2.3× on Black–Scholes payoff vs FNO's 6.3×.
- CNO high variance: 1.05× on Poisson, **12.2×** on Black–Scholes payoff.
- **All three** degrade badly under long-horizon rollout in Navier–Stokes (~3.3×) and KS ⇒
  autoregressive composition is a **general** bottleneck, not an architecture defect.
- 50 seeds per condition with confidence intervals.

**6. Limitations.** Three architectures only; shift magnitudes largely qualitative, no
sensitivity analysis of magnitude vs degradation; the propositions are motivating, not
causal proofs; **training sets of only 128–512 samples**.

*Our added caveat:* with 128–512 training samples, "in-distribution accuracy does not
predict robustness" is partly confounded with a small-data regime. It does not follow that
the same holds at 1e5 samples. This is precisely the underfitting-masquerading-as-
intrinsic-limit confound we have to design against — and here it is, live, in a published
paper we intend to cite.

**7. Genuinely new.** A unified stress protocol across five qualitatively different PDE
families with 50 seeds each; the systematic **ranking-reversal** result; failure-mode
decomposition tied to inductive bias (spectral vs localised vs branch–trunk); the two
formal mechanisms.

**8. How it helps us.** Answers one of the user's listed questions — *"is short-horizon
accuracy a reliable proxy for rule learning?"* — with **no, already published, 750 models,
in TMLR.** We cite this; we do not re-derive it. It also hands us the right dependent
variable shape (a normalised degradation factor) and demonstrates the seed count a claim
of this kind needs.

**9. Gaps / trends we may not have.** The shared long-horizon failure across all three
architectures on NS and KS **is the one result nobody explains**. Prop. 1 says error grows
like `ε Σ L^j` — that is a bound, not a measurement, and it does not distinguish "the model
is imperfect and error compounds" from "the system has an intrinsic ceiling". The published
literature therefore contains a *shared, unexplained, architecture-invariant* failure —
a better target than an architecture-specific one, because architecture-invariance is prima
facie evidence for a system-level cause. **This is arguably the sharpest opening any of
these eight papers leaves.**

---
---

# Part B — additional sources located this session

---

## A1. Gilpin (2023) — *Model scale versus domain knowledge in statistical forecasting of chaotic systems*

`arXiv:2303.08011v3` · Phys. Rev. Research **5**, 043252
Verification: full PDF parsed (20 pp).

**Problem / hypothesis.** Prior chaos-forecasting comparisons use a handful of hand-picked
systems, so they cannot separate the effect of modelling choices from the effect of the
system. Do specialised dynamical methods beat general large-scale ones, and what actually
determines empirical predictability?

**Method.** 24 forecasting methods × **135** low-dimensional chaotic systems (`dysts`) × 17
metrics. Systems aligned by dominant timescale: integration step set by `1/t_max`,
trajectories resampled to **100 points per dominant Fourier period `t_peak`**. Training
history `t* = 10 t_peak`; forecast to `+2 t_peak`; test trajectories from different initial
conditions. Lookback window tuned per method. Errors reported in **Lyapunov time λ_max·t**.
Metrics include sMAPE (headline), NRMSE, MASE, r², Spearman, mutual information, Granger
causality, plus invariant-measure errors (correlation dimension via Grassberger–Procaccia,
multivariate multiscale entropy).

**What it actually shows.**
- Large domain-agnostic models stay accurate to **~2 dozen Lyapunov times** — a regime
  classical methods never reached.
- At **short** horizons (< 1/λ_max) forecast error correlates weakly with λ_max; **that
  correlation degrades at long horizons**. This is the "decorrelation" claim, and it is
  **horizon-dependent**, which the one-line version of it loses.
- Training walltime (a proxy for capacity, since parameter counts are not comparable across
  these architectures) correlates with error at **ρ = −0.31 ± 0.04**. That is weak.
- Data titration (Fig. 3C): all models improve with more data; inductive-bias models
  (nODE, nVAR) drop fastest in the low-data regime but plateau higher; NBEATS is good in
  both regimes.
- NBEATS/NHiTS are much better than everything else at reproducing **fractal dimension** —
  attractor structure, not pointwise error.
- Conclusion as stated: in the long-horizon regime, **scale and dataset availability rather
  than intrinsic dynamical properties** limit current forecasting of chaos; but in
  data-limited settings physics-based hybrids retain an advantage.

**Limitations (theirs).** Results may be specific to `dysts`; the hyperparameter and
architecture space is effectively infinite; λ-based rescaling is ill-defined for degenerate
constant forecasts; MASE/WAPE diverge at long horizon; mutual information too noisy to be
useful.

**How it helps / hurts us.** **The most consequential single paper for this project.** It
cuts both ways:
- *Against:* C2 ("learnability ≠ chaoticity") has been measured at scale on exactly our
  intended substrate and published in a physics journal. Claiming it as new is the textbook
  novelty failure.
- *For:* his **explanation** of the residual is the opposite of the proposal's. He
  attributes decorrelation to *model scale and data availability*; the proposal attributes
  it to an *intrinsic per-system ceiling*. **These are competing explanations of the same
  measured residual, and he did not run the experiment that separates them.** His capacity
  proxy is walltime at ρ = −0.31; his data sweep is aggregated **across** systems, not
  per-system. A controlled, **per-system** capacity × data sweep with a fitted exponent and
  floor is unoccupied.

**Gaps / trends we may not have.** (a) The decorrelation is horizon-dependent — so the
right dependent variable is not "does λ predict error" but "does λ predict error *as a
function of horizon*", i.e. a curve, not a scalar. (b) He reports **invariant-measure
metrics separately from pointwise metrics and they rank models differently**. Combined with
A3 and P3's ergodic smoothing, a consistent picture is forming across four independent
papers: **pointwise skill and distributional skill decouple**, and which one you measure
decides your conclusion. Nobody has made that decoupling the object of study.

---

## A2. Gilpin (2021) — *Chaos as an interpretable benchmark for forecasting and data-driven modelling* (`dysts`)

`arXiv:2110.05266` · NeurIPS 2021 Datasets & Benchmarks
Verification: abstract + metadata.

131 known chaotic systems (135 in the 2023 study) from astrophysics, climatology,
biochemistry and elsewhere, each annotated with known mathematical properties, with
precomputed multivariate and univariate time series and **re-integration to arbitrary
length and granularity**. Proof-of-concept tasks: surrogate transfer learning for
time-series classification, importance sampling, symbolic-regression benchmarking.

**How it helps us.** Directly: `pip install dysts`, 135 low-dimensional ODEs, aligned
timescales, precomputed invariants, cheap on CPU, results comparable to a published
baseline. It removes a week of substrate building and it pre-registers the system list,
which kills the "you picked your systems after seeing the results" objection. Under a
2-core-CPU budget this is the single highest-leverage asset found.

**Caveat.** A fixed public list is still a *selection* — 135 textbook attractors are not a
uniform sample of dynamical systems, they are a sample of systems interesting enough for
someone to have published. Any "across systems" claim inherits that bias.

---

## A3. Zhang & Gilpin (2025) — *Zero-shot forecasting of chaotic systems*

ICLR 2025 · `arXiv:2409.15771`
Verification: abstract verbatim + metadata.

135 chaotic systems, 1e8 timepoints. Time-series **foundation models forecast competitively
with custom-trained models** (NBEATS, TiDE), *especially when training data is limited*.
The headline finding for us: **"even after point forecasts fail, large foundation models
are able to preserve the geometric and statistical properties of the chaotic attractors."**
Attributed to in-context learning, with **context parroting** identified as the mechanism.

**How it helps us.** A third independent observation of the pointwise/distributional
decoupling. It also weakens the "more data and scale is the answer" reading of A1: a model
that was never trained on the system does fine, which points at *in-context* structure
rather than learned system-specific structure.

---

## A4. Zhang & Gilpin (2025/26) — *Context parroting: a simple but tough-to-beat baseline for foundation models in scientific machine learning*

`arXiv:2505.11349v3`
Verification: abstract + metadata.

A parameter-free baseline that **copies a matching segment from the context window**
outperforms leading time-series foundation models on chaotic, turbulent and oscillatory
systems at a tiny fraction of the cost. Foundation models show failure modes such as
converging to the mean. Explicitly linked to induction heads.

**How it helps us — a warning, not a gift.** If a naive copy baseline beats foundation
models on chaotic systems, then a large part of the "long-horizon forecasting of chaos"
literature is measuring **recurrence structure of the attractor**, not learned dynamics.
Any leap-horizon or learnability measurement we make **must include a parroting /
nearest-neighbour-analogue baseline**, or a "the model learned the rule" claim is
unfounded. This single baseline could invalidate a large fraction of naive designs, and it
is cheap to implement. **Non-negotiable in our design.**

---

## A5. Duraisamy (2026) — *Predictivity and Utility of Neural Surrogates of Multiscale PDEs*

`arXiv:2604.20061` · math-ph, nlin.CD
Verification: abstract + metadata.

Argues that apparent surrogate successes occur on **low-dimensional solution manifolds
where any competent reduced model will interpolate well**; identifies **spectral bias** and
**irreversible information loss under coarse-graining** as the mechanisms; argues that
medium-range weather on reanalysis data is a favourable special case that will not extend
to genuinely chaotic multiscale problems; advocates neural–classical hybrids and better
reporting standards.

**How it helps us.** This is the **direct competitor claim to C1 — and it is the
pro-intrinsic-ceiling position.** Note the field structure this creates:

- Duraisamy: intrinsic ceilings exist, set by spectral bias + coarse-graining loss.
- Gilpin: no, it is scale and data.
- Chen et al. (P3): neither — required capacity is *non-monotone* in chaoticity.

**These three cannot all be right.** That is the shape of a real, live, named disagreement,
and adjudicating it is a legitimate contribution in a way that "discovering" either
position is not.

**Gap.** Duraisamy's argument is largely analytical and positional (two illustrative
examples). Gilpin's is empirical but uncontrolled for capacity per system. Neither ran the
sweep.

---

## A6. Lakshmanan & Chopra (2026) — *Hybrid Neural World Models*

`arXiv:2605.28317` · preprint under review · cs.LG, cs.AI, math.NA, physics.comp-ph
Verification: abstract verbatim + metadata. **This is the Lossfunk lab paper the proposal
builds on.**

A single network with **continuous horizon conditioning**, trained with direct supervision
against reference solvers, predicts any future state at horizon T **in one forward pass**.
Although nothing in the data, loss or architecture supervises discontinuity location, the
trained surrogate **encodes it implicitly**, recoverable from forward passes alone as a
per-trajectory error map that concentrates on shocks, fronts and contacts and stays small
elsewhere. That map is competitive with or better than deep ensembles, learned error heads,
gradient-magnitude indicators and locally-adaptive conformal prediction, using one network,
no calibration set and no governing-equation knowledge. Two modes: surrogate alone
(**26×–72× CPU speedup** vs textbook solvers) or error-map-gated fallback to the reference
solver (roughly halves residual error). Works unmodified across reaction–diffusion,
compressible Euler and rigid-body collision.

**How it helps us.** It settles the user's `t→t+63` question in the affirmative *as a
capability*: direct horizon-conditioned leaping works, is published, and is in-house. So
"can we skip the rollout" is **not** an open question in binary form. The open form is
quantitative — *for which systems* does direct leaping hold up, how far, and is that
predictable in advance. That is the leap-horizon question, and it sits downstream of this
paper rather than competing with it.

**Gap / trend.** The implicit-error-map result is a striking instance of a general theme:
**a trained surrogate contains more information about its own failure than we currently
extract.** That is a different and possibly better framing of "predict failure modes" than
the proposal's pre-training meta-model — post-hoc, one network, no meta-dataset, already
demonstrated to work. Worth weighing as a rival direction.

---

## A7. Marzen, Riechers & Crutchfield (2023) — *Complexity-calibrated benchmarks for machine learning…*

`arXiv:2303.14553`
Verification: abstract verbatim + metadata.

Generate processes from **large ε-machines with known statistical complexity**, so the
optimal next-symbol error is computable via **Fano's inequality**. Result: next-generation
reservoir computers with reasonably long memory traces have an error probability
**≥ ~60% higher than the minimal attainable error** on highly non-Markovian processes;
popular RNNs also fall far short. Plus concentration-of-measure results for randomly
generated complex processes.

**How it helps us.** The cleanest existing template for the thing we actually want: a
substrate where **the theoretical optimum is known**, so "the model failed" can be
separated from "the task is hard". That separation is the single hardest methodological
problem in this project, and computational mechanics has solved it — for stochastic
processes and reservoir computers, **not** for neural surrogates of spatiotemporal systems.

**Gap.** ε-machine processes are discrete, stochastic and finite-memory. Extending the
known-optimum trick to continuous dynamical systems is not free. But even a *partial*
import — a substrate where the Bayes-optimal predictor is computable — would let us make
claims nobody in the surrogate literature can currently make.

---

## A8. Xu, Zhao, Song, Stewart & Ermon (2020) — *A Theory of Usable Information Under Computational Constraints* (V-information)

ICLR 2020 (talk) · `arXiv:2002.10689`
Verification: abstract + metadata.

Predictive **V-information**: how much information X carries about Y *when the observer is
restricted to a predictive family V*. Recovers Shannon mutual information when V is
unrestricted and the coefficient of determination under other restrictions. **Violates the
data processing inequality** — computation can *create* usable information, which is why
representation learning helps at all. Estimable from data in high dimension with PAC-style
guarantees. Demonstrated on structure learning and fair representation learning.

**How it helps us — a verdict, not a resource.** This is the existing formalisation of the
user's Learnable / Memorizable / Not-learnable taxonomy. And it answers the user's question
*"can learnability be measured independently of architecture?"*: **no, not coherently.**
Learnability is *indexed to a predictive family* by construction. An "architecture-free
learnability number" is either (a) the Shannon/Bayes quantity, which ignores computation
and is not what anyone means, or (b) implicitly indexed to an unstated reference family.
Any spike proposing architecture-independent learnability must confront this in its first
paragraph or it is dead on arrival at review.

The constructive reading: **stop trying to remove the architecture; make it an explicit
axis.** A "learnability profile" is then a *vector* over a declared set of families V, not
a scalar. That is both defensible and closer to what would actually be useful for model
selection.

---

## A9. Krishnakumar, White, Zela, Tu, Safari & Hutter (2022) — *NAS-Bench-Suite-Zero*

NeurIPS 2022 Datasets & Benchmarks · `arXiv:2210.03230`
Verification: full PDF parsed (37 pp).

**13 zero-cost proxies × 28 tasks**, unified implementation. Proxies: epe-nas, fisher,
flops, grad-norm, grasp, l2-norm, jacov, nwot, **params**, plain, snip, synflow, zen-score.
Tasks span NAS-Bench-101/201/301, TransNAS-Bench-101 Micro and Macro (7 tasks each),
Spherical-CIFAR-100, NinaPro, SVHN.

Two statements matter to us, verbatim from the paper:
- *"several recent works have shown that simple baselines such as 'number of parameters'
  and 'FLOPS' are competitive with all existing ZC proxies across most settings"*
- *"most ZC proxies do not generalize well across different benchmarks"*

Their positive result is about **combination**, not any single proxy: adding all 13 to a
NAS surrogate improves Spearman ρ on NAS-Bench-201 CIFAR-100 from 0.640 ± 0.042 to
**0.908 ± 0.012** (+41.7%), averaged over 100 trials.

**Limitations (theirs).** Purely empirical; NAS-Bench-301 evaluated on only 11,000 of ~1e18
architectures.

**How it helps us.** The single most important cautionary result for the meta-model
hypothesis. The base rate for "training-free predictor of architecture performance"
failing to generalise is high. Concretely it forces three design rules:
1. **Dumb baselines are mandatory** — parameter count, FLOPs, and one short training run.
2. **Held-out *families*, not held-out samples.** Cross-benchmark generalisation is where
   this class of method dies.
3. **Expect the win to come from combination, not from one clever statistic** — and report
   it that way.

**Gap / trend.** Note the shape of their success: proxies are individually weak but
*complementary*, and value appears when fused into a surrogate. That is a different product
from "one number that tells you the answer", and a more honest target.

---

## A10 / A11. TRM and its audit

**A10.** Jolicoeur-Martineau (2025), *Less is More: Recursive Reasoning with Tiny Networks*,
`arXiv:2510.04871`. TRM: a **single tiny 2-layer network**, recursing on a latent reasoning
feature with deep supervision, simpler than HRM. **7M params → 45% on ARC-AGI-1, 8% on
ARC-AGI-2**, beating most LLMs (Deepseek R1, o3-mini, Gemini 2.5 Pro) at <0.01% of the
parameters. Verification: abstract verbatim; ablation tables visible (two-feature TRM 87.4%
vs single-z 71.9% on the Sudoku-style ablation; recursion-count sweep 46.4 → 63.2).

**A11.** Roye-Azar, Vargas-Naranjo, Ghai, Balamurugan & Amir (2025/26), *Tiny Recursive
Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute*,
`arXiv:2512.11847`. Audits the ARC Prize TRM checkpoint. Four findings:
1. **The 1000-sample test-time-augmentation + majority-vote pipeline is worth ~11
   percentage points of Pass@1** over single-pass canonical inference.
2. **Strict dependence on the puzzle identifier** — a blank or random puzzle ID gives
   **zero** accuracy.
3. **Recursion is shallow in effect**: most final accuracy is reached at the *first*
   recursion step and performance saturates after a few latent updates.
4. Heavy augmentation broadens the candidate distribution and improves multi-sample success.
Conclusion: TRM's performance "appears to arise from an interaction between efficiency,
task-specific conditioning, and aggressive test-time compute rather than deep internal
reasoning."

**How they help us.** The user's specialisation-vs-scale hypothesis leans on TRM. A10 is
real and impressive; **A11 substantially reframes what it demonstrates.** Anyone citing TRM
as proof that "parameter count is the wrong axis" must handle the 11-point
test-time-compute contribution and the puzzle-ID conditioning. The honest statement is
*"a tiny model plus task-specific conditioning plus 1000× test-time compute beats a large
model at one inference pass"* — a claim about **where you spend compute**, not about
parameter efficiency as such.

**Gap / trend.** Finding 3 is quietly important and cuts against the proposal's "latent
iterative computation replaces output iteration" intuition: the recursion **is not doing
deep iterative work** — it saturates almost immediately. That is evidence *against* the
mechanism the user hoped TRM demonstrated, from a direct audit of the released checkpoint.

---
---

# Part C — how these papers relate

**Cluster 1 — predictability is scale-dependent, and λ is the wrong scalar.**
P2 (2003, theory: FSLE, ε-entropy) → P1 (2004, coarse-graining rescues irreducible CA) →
A1 (2023, measured at scale: forecast skill decorrelates from λ at long horizon) →
P3 (2025, required capacity is *non-monotone* in chaoticity).
Direction of travel: from "λ is insufficient" (theory, 20 yrs old) to "λ is actively
misleading" (measured) to "the relationship has structure nobody predicted" (non-monotone).
**P3 is the newest and least absorbed; that is where the surprise still lives.**

**Cluster 2 — in-distribution accuracy does not predict deployment behaviour.**
P8 (750 models, ranking reversals, FNO 18.4× on Poisson) ‖ P7 (corr > 0.8 yet wrong
detonation cell size) ‖ A5 (mechanism: spectral bias + coarse-graining loss).
Three independent 2025–26 results, different substrates, same conclusion. **Settled. Do not
re-derive.** P8 and P7 agree that the *shared* failure is **autoregressive rollout**, and
neither explains it.

**Cluster 3 — pointwise skill and distributional skill decouple.**
A1 (NBEATS/NHiTS best at fractal dimension, ranked differently than by sMAPE) →
A3 (attractor geometry preserved *after* point forecasts fail) →
A4 (context parroting: copying beats learning, because attractor recurrence carries the
statistics) → P3 (ergodic smoothing: in the chaotic regime only *statistically smooth*
corrections remain) → P7 finding (iii) (correlation high, integral quantities wrong).
**Five papers, four groups, converging.** Nobody has made the decoupling itself the object
of study. The most under-exploited pattern in the corpus.

**Cluster 4 — predicting model performance before training.**
P4 (Task2Vec: works, but only with the asymmetric correction; the naive version loses to
ImageNet) → A9 (13 proxies × 28 tasks: params/FLOPs competitive, proxies do not generalise,
value is in *combination*) → P6 (and the right evaluation is a decision rule under
asymmetric cost, not a correlation).
Direction of travel: the field has moved from "find the magic statistic" to "fuse many weak
signals and evaluate as a decision". A project proposing the former in 2026 is behind.

**Cluster 5 — a substrate with a known optimum.**
A7 (ε-machines + Fano: known optimal error, RC sits ~60% above it) is methodologically
upstream of everything else here, and unexploited outside computational mechanics.

**Cluster 6 — direct multi-horizon prediction.**
A6 (horizon-conditioned single-pass, 26–72× speedup, implicit discontinuity map) is the
capability; A11 (recursion saturates immediately) is a caution about the "latent iteration"
story; P8's shared rollout failure is the *motivation* for direct prediction.

**Cross-cluster tension that matters most.**
A1 says *scale and data* limit chaos forecasting. A5 says *intrinsic ceilings* do. P3 says
*neither, and the curve is non-monotone*. P5 supplies the machinery (fit exponent + floor
per system) that would settle it, and nobody has applied it here. That is the gap.

---
---

# Part D — what challenges our theory, what aids it

## Challenges (evidence against the proposal's claims)

| # | Threat | Source | Severity |
|---|---|---|---|
| 1 | "Learnability ≠ chaoticity" is 20-year-old theory and has been measured at scale | P2, A1 | **Fatal to novelty as stated** |
| 2 | "Short-horizon accuracy doesn't imply long-horizon stability" is published: 750 models, TMLR | P8, P7 | **Fatal to novelty as stated** |
| 3 | Coarse-graining rescues irreducible systems — known since 2004 | P1 | Fatal to C3-as-idea |
| 4 | Architecture-free learnability is probably ill-posed | A8 | **Conceptual; must be answered up front** |
| 5 | Training-free performance predictors mostly fail to generalise; params/FLOPs competitive | A9 | High — sets a hard baseline bar |
| 6 | A parameter-free copy baseline beats foundation models on chaotic systems | A4 | **High — could invalidate a naive design** |
| 7 | Underfitting masquerading as an intrinsic ceiling — and P8 shows the trap live (128–512 training samples) | P8 + our note | **Severe on a 2-core CPU** |
| 8 | Metric choice can flip the sign of the conclusion (pointwise vs distributional) | P7, A1, A3, P3 | High |
| 9 | Direct multi-horizon leaping already demonstrated in-house | A6 | Removes "can we?" as a question |
| 10 | TRM's headline is substantially test-time compute + task-ID conditioning | A11 | Undercuts the specialisation argument as stated |

## Aids (evidence that helps, or infrastructure we can use)

| # | Asset | Source |
|---|---|---|
| 1 | `dysts`: 135 systems, aligned timescales, precomputed invariants, pip-installable, CPU-cheap, publicly fixed list | A2, A1 |
| 2 | Scaling-law machinery: fit exponent + finite floor per system; the `α ≈ 1/d` prediction from manifold dimension | P5 |
| 3 | A live, named, three-way disagreement about *why* surrogates fail long-horizon | A1 vs A5 vs P3 |
| 4 | Non-monotone capacity result + a mechanism (ergodic smoothing) + a physical criterion (Chirikov) | P3 |
| 5 | FSLE / ε-entropy: scale-resolved predictability instruments, far better than λ alone; the FSLE plateau as an operational "closed coarse description" | P2 |
| 6 | Baseline-normalised degradation factor + 50-seed protocol as a rigour template | P8 |
| 7 | Known-optimum substrates via ε-machines + Fano | A7 |
| 8 | Decision-theoretic evaluation under asymmetric cost | P6 |
| 9 | Cluster-3 decoupling: five convergent papers, nobody has studied it directly | A1, A3, A4, P3, P7 |
| 10 | Neither community measures the other's variables: dynamical invariants (chaos lit) vs stiffness/dimensionality/mesh (PDE-surrogate lit) | A1 vs P7 |

---
---

# Part E — revisions to `01-literature-review.md`

Verification this session **confirms** the earlier review's sourcing — every identifier
checked resolves to a real paper with the claimed content. Three refinements:

1. **Gilpin's decorrelation is horizon-dependent, not global.** At horizons shorter than
   `1/λ_max` a weak correlation with λ does hold; it degrades at long horizons. The earlier
   one-line summary lost this. It matters: the right dependent variable is a *curve over
   horizon*, not a scalar.

2. **Gilpin's capacity evidence is weaker than his conclusion.** His capacity proxy is
   training walltime at **ρ = −0.31 ± 0.04**, and his data sweep is aggregated across
   systems rather than run per-system. So "scale and data are the limit" is a *conclusion*,
   not a controlled measurement. This strengthens the case that the discriminating
   experiment is genuinely unrun.

3. **Shikhman trains on 128–512 samples.** The "in-distribution accuracy does not predict
   robustness" result is therefore partly confounded with a small-data regime. Cite the
   finding and *name the confound* — it is the same confound that threatens us, and naming
   it in a published neighbour is good for our credibility.

New material not in `01`: A6's abstract retrieved in full (horizon conditioning, 26–72×
speedup, implicit discontinuity error map); A11's numbers (≈11 pp from test-time compute;
zero accuracy without the puzzle ID; recursion saturates at step 1); A9's verbatim
statements; P3's exact capacity numbers and Chirikov threshold; P1's 240/256 count and the
16-rule fail-set; P8's per-architecture degradation factors.

---
---

# Part F — follow-ups requested on P1–P4

Added 2026-09-07 in response to user questions. **This document is append-only**: nothing
above this line has been or will be removed or renumbered.

---

## F.1 — P1: the limitations explained, and what the field has done about "solving system complexity"

### F.1.1 The three limitations, unpacked

**(a) The search was capped at N ≤ 4, and the procedure is a test, not a constructor.**
Israeli & Goldenfeld do not have an algorithm that *finds* a coarse-graining. They have a
consistency condition that *checks* a candidate `(N, P)` pair. Finding one means brute-force
enumeration: for block size N the block alphabet has `2^N` symbols, and the number of
projections onto a smaller alphabet grows super-exponentially. At N=4 the alphabet is 16 and
a binary projection is one of `2^16`; at N=7 it is one of `2^128`. So "240 of 256 rules are
reducible" is a statement about **what a brute-force search up to N=4 could reach**, not
about rule space. Their own words: *"We don't know if our inability comes from limited
computing power or from something deeper."*

**(b) The free parameter is the projection space, and a wide enough search manufactures
closure.** This is the rigour hazard. If we ever use "admits a closed coarse description" as
a learnability label, the label is only as meaningful as the pre-declared search space. Widen
the search and everything becomes reducible.

**(c) Their fail set was not characterised.** 16 rules resisted, but they could say nothing
about *why*, and could not even establish the CIR status of rules 18, 54, 126.

### F.1.2 What has been done since — and it substantially resolves (a) and (c)

**A12 — Israeli & Goldenfeld (2006), the extended version.** *Coarse-graining of cellular
automata, emergence, and the predictability of complex systems*, Phys. Rev. E **73**, 026203.
Verification: abstract retrieved verbatim. Two results that are not in the PRL and that
matter to us more than the PRL does:

- The large-scale dynamics of CA is **very simple as measured by the Kolmogorov complexity of
  the large-scale update rule**, and this obeys a **novel scaling law**.
- Consequently *"the probability of finding a coarse-grained description of CA approaches
  unity as one goes to increasingly coarser scales."*

**Read that second point carefully — it is close to fatal for the naive framing.** If almost
everything becomes coarse-grainable at a coarse enough scale, then "does a closed coarse
description exist?" is not a binary property of a system. It is a **scale-indexed** property,
and the only meaningful question is *at what scale does closure appear*. That converts a
proposed classification problem into a regression on a scale variable — which is a better
problem, but a different one, and it is the version already staked out in 2006.

**A13 — Dzwinel & Magiera (2015).** *Irreducible elementary cellular automata found*,
J. Comput. Sci. **11**, 300–308. Verification: abstract retrieved verbatim. This is the
direct answer to limitation (a) and (c):

- They give **a new coarse-graining algorithm with substantially lower computational load**,
  explicitly because Israeli–Goldenfeld's brute force gave *"very fragmentary"* results that
  *"do not allow to draw viable conclusions about reducibility of ECA for larger grain sizes
  than N = 4."* Our limitation (a), stated by the people who fixed it.
- Pushing to larger N, the count of irreducible ECA **decreases with grain size** and at
  **N = 7 converges to a stable set of exactly four inequivalent rules: {30, 45, 106, 154}.**
- And the characterisation: this is *"the complete set of strong chain-rules representing
  maximally chaotic automata"* in Wuensche's taxonomy, and simultaneously *"the complete set
  of strong surjective automata, i.e. highly irreversible automata."*

**Why this is the most useful thing found for P1.** The boundary of coarse-grainability in
ECA space is not fuzzy and not an artefact — it is four rules, and they are picked out by an
**algebraic** property (surjectivity ⇒ irreversibility ⇒ information destruction), not by a
dynamical one. That is a real, sharp, non-obvious link: *what makes a system resist reduction
is that its update map destroys information*. It also means the ECA substrate has a
**4-out-of-256 positive class**. Any "predict learnability from rule structure" study on ECA
is a 1.6% base-rate problem, which is a very different experiment from the one people imagine
they are running.

### F.1.3 The wider "solving system complexity" landscape

The coarse-graining question is one corner of a large, mostly disconnected literature on
reducing complex systems to closed descriptions. The families, and what each would give us:

| Family | Core idea | Bearing on us |
|---|---|---|
| **CA coarse-graining** (Israeli–Goldenfeld; Dzwinel–Magiera) | block-and-project; exact closure as a consistency condition | exact, discrete, decidable — the only family where "closed description exists" is a *provable* statement rather than an approximation quality |
| **Renormalisation group** | integrate out short scales; fixed points and relevant operators | the physics ancestor of all of this; gives the language of *relevant vs irrelevant* degrees of freedom |
| **Mori–Zwanzig / optimal prediction** | exact projection of full dynamics onto resolved variables, producing a Markovian term + a memory kernel + noise | **the continuous-system analogue of the CA result, and the honest one**: closure is never exact, the price of coarse-graining is an explicit memory term. Directly relevant to "why does rollout fail" — the memory kernel is what a one-step surrogate discards |
| **Computational mechanics / ε-machines** (see A7) | causal states = the minimal sufficient statistic for prediction; statistical complexity | the only framework that gives a *computable optimum*, so "model failed" separates from "task is hard" |
| **Koopman / DMD lifting** | represent nonlinear dynamics as a linear operator on lifted observables | the "make it reducible by changing representation" move, in continuous form |
| **Model order reduction** (POD/PCA, balanced truncation, reduced-basis) | project onto a low-dimensional empirical subspace | supplies the empirical intrinsic dimension `d` that P5's `α ≈ 1/d` needs |
| **Closure modelling in turbulence / LES** | model the unresolved stresses | the engineering version; decades of evidence that closure is systematically hard and system-specific |
| **Causal emergence / information closure** | quantify when a macro description is more informative or more causally complete than the micro one | gives *graded* rather than binary closure — the thing A12's result says we actually need |

**The synthesis, and it is worth stating plainly.** Israeli–Goldenfeld say closure often
exists in discrete systems. Mori–Zwanzig says that in continuous systems closure is *never*
exact and the residue is a memory kernel. Those two are not in conflict — they are the same
statement about different alphabets — but **the neural-surrogate literature cites neither**,
and instead rediscovers the memory kernel empirically as "autoregressive rollout error". P8's
unexplained architecture-invariant rollout failure and Mori–Zwanzig's memory term are
plausibly the same object. That connection is not made anywhere in the corpus we have read,
and it is cheap to test.

---

## F.2 — P2: what is better than the Lyapunov exponent, and does any of it translate to learnability?

First a correction of framing, because it matters for how we cite this. **The Lyapunov
exponent is not "proven wrong because it assumes determinism."** Two separate problems, and
conflating them will get us caught:

1. **It is asymptotic and infinitesimal.** λ describes the growth of an *infinitesimal*
   perturbation as `t → ∞`. Real errors are finite and real horizons are finite. P2's coupled
   regular+chaotic example shows the failure concretely: global λ predicts `T_p ~ 1/λ`
   independent of coupling, but the true answer is `T_p ~ ε^{-2}`. λ is not wrong, it is
   answering a different question.
2. **Estimating λ from data requires assuming determinism**, and above the noise scale chaos
   and noise are observationally identical (P2). So the *estimator* inherits an assumption
   the *definition* does not need.

### F.2.1 The better instruments

| Instrument | What it measures | Why better than λ | Cost |
|---|---|---|---|
| **FSLE** `λ(δ)` (P2) | growth rate of a *finite* perturbation δ | scale-resolved; reveals plateaus ⇒ emergent macroscopic levels; correct in the diffusive regime where λ is not | cheap; needs trajectory ensembles |
| **ε-entropy** `h(ε)` (P2) | information rate at resolution ε | KS entropy without the `ε→0` fiction; the natural home for "predictability at a stated resolution" | moderate; needs symbolisation |
| **Permutation entropy / weighted PE** (Bandt–Pompe 2002; Fadlallah 2013) | ordinal-pattern complexity | ordinal ⇒ **robust to noise and monotone transforms**, no embedding parameters, converges to KS entropy rate for ergodic sources (Amigó) | very cheap, O(n) |
| **0-1 test for chaos, modified** (Gottwald–Melbourne; Eyébé Fouda mod.) | K-statistic → 1 chaotic, → 0 periodic, from mean-squared displacement of a driven 2D system | needs no phase-space reconstruction, no embedding dimension, no λ estimation; noise-robust | very cheap |
| **Chaos Decision Tree Algorithm** (A14) | full pipeline: stochastic / periodic / chaotic | the practical answer to "is λ even applicable here" | cheap, code published |
| **Statistical complexity / ε-machines** (A7) | minimal sufficient statistic; computable optimal error | the only one that gives an *optimum* to compare a model against | expensive |
| **Covariant Lyapunov vectors, finite-time LEs** | directional, time-local instability | resolves *where* on the attractor instability lives, not just the average | moderate |

**A14 — Toker, Sommer & D'Esposito (2020).** *A simple method for detecting chaos in nature*,
Communications Biology **3**, 11. Verification: full metadata + detailed method description
retrieved. The **Chaos Decision Tree Algorithm**, four steps:
1. **Stochasticity test** — permutation entropy as the test statistic against **Amplitude
   Adjusted Fourier Transform** and **Cyclic Phase Permutation** surrogates. If the series' PE
   sits inside the surrogate distribution, call it stochastic and stop.
2. **De-noising** — Schreiber's nonlinear noise-reduction algorithm by default.
3. **Oversampling correction** — detect and downsample.
4. **Chaos test** — the **modified 0-1 test**, giving `K → 1` chaotic, `K → 0` periodic.
Permutation entropy is reused at the end as a **proxy for the degree of chaos**
(order 5, lag 1).
Validation: a large battery — logistic, cubic, Lorenz, Rössler, Hénon, Ikeda, generalised
Hénon (hyperchaotic), Poincaré oscillator, ARMA, random walks, coloured noise, cyclostationary
processes, plus biological simulations and empirical data (laser, star flux, NAO index,
tremors, HRV). White noise added **up to 40% of the data's standard deviation**. Reported
near-perfect classification even at high noise, and good behaviour at 1,000–5,000 points. One
honest failure: the noise-driven sine map is consistently misclassified as chaotic. Their
applied result: **heart rate variability is stochastic, not chaotic**, contradicting a
long-standing claim.

*Why we want this.* It is a pre-flight check. Before we assert anything about a system's
learnability we should know whether the classical chaos machinery even applies to it. And it
gives a defensible **degree-of-chaos scalar (K, or PE) that is noise-robust**, which λ is not.

### F.2.2 Has anyone connected these to learnability? Yes — and this is the most important
### thing found today

The user asked exactly the right question. The answer is that a small, mostly-ignored
literature has been drawing precisely this correlation since 2014, and **none of it appears in
the ML surrogate literature**.

**A15 — Garland, James & Bradley (2014).** *Model-free quantification of time-series
predictability*, Phys. Rev. E **90**, 052910 (`arXiv:1404.6823`). Verification: full HTML text
read.
- Claim: **weighted permutation entropy (WPE) correlates with achievable forecast accuracy**,
  across forecasting methods, with no model of the system.
- WPE: ordinal patterns over windows of length ℓ, each pattern weighted by the squared
  deviation of its window from the window mean (amplifies real structure, suppresses
  noise-driven patterns), Shannon entropy over pattern frequencies, normalised by `log2(ℓ!)`
  to land in [0,1].
- Data: 120 real time series of processor instructions-per-cycle on an Intel i7-2600 —
  `col_major` (simple, near-periodic), `403.gcc` (near-random), `dgesdd` split into 6 regimes.
- Forecasters: random walk, naive mean, `auto.arima`, and **LMA (Lorenz Method of Analogues)**
  — nearest-neighbour prediction in delay-coordinate space. One-step iterative, fit on 90%.
- Result: error (MASE) rises with WPE along a fitted **logarithmic** trend
  `WPE ≈ a·log(b·MASE + 1)`, `a ≈ 0.0797`, `b ≈ 1520`. Examples: `col_major` WPE 0.513 → LMA
  MASE 0.050 (20× better than random walk); `403.gcc` WPE 0.943 → best MASE 1.138 (i.e. no
  better than random walk).
- **It is a trend with scatter, explicitly not a hard bound.** Their proposed use is a
  *diagnostic*: a point far off the curve means **method–data mismatch** — the series has more
  exploitable structure than the model is exploiting. `col_major` with `auto.arima` sits off
  the curve; the same data with LMA sits on it.
- Their own caution, which we should adopt verbatim: **WPE tells you that *some* method could
  do better; it does not tell you *which*.**
- Limitations: three programs, four forecasters, one error metric; MASE pathologies on
  oscillatory/non-stationary data; ℓ chosen heuristically; regime segmentation was *"visual
  and subjective"*.

**A16 — Pennekamp et al. (2019).** *The intrinsic predictability of ecological time series and
its potential to guide forecasting*, Ecological Monographs, `10.1002/ecm.1359`. Verification:
**partial** — journal page returned 403; content below is from the indexed article text
retrieved via search, not from a full read. Flag before citing.
- **461 ecological time series**; weighted permutation entropy vs square-root-transformed
  nRMSE; forecast error increases with PE.
- Introduces the distinction we need: **intrinsic predictability** (a property of the process,
  estimated model-free) vs **realised predictability** (what a given model actually achieved).
  Points *above* the PE–error curve are model failures; points *below* are… suspicious.
- They argue the model-free nature of PE is what makes **cross-system comparison** possible,
  which is exactly our use case.

**A17 — Wang et al. (2025/26).** *Exploring Accuracy Law for Deep Time Series Forecasters: An
Empirical Study*, `arXiv:2510.02729`. Verification: abstract verbatim + metadata.
**This is the closest thing to the proposal's core hypothesis that exists, and it must be
read in full before we commit to a direction.**
- They ask explicitly: *"how to estimate the performance upper bound of deep time series
  forecasters?"*, starting from the community consensus that forecasting *"inherently faces a
  non-zero error lower bound."*
- Their move: classical **series-wise** predictability metrics are the wrong granularity,
  because deep models are sequence-to-sequence over windows. They introduce a quantitative
  measure of **window-wise pattern complexity**.
- Evidence: **over 4,700 newly trained deep forecasting models**. They report a consistent
  empirical relationship between the **minimum attainable error** and window-wise complexity
  — the **"accuracy law"**.
- Downstream use: identifying **saturated benchmark tasks** (where no further progress is
  available) and deriving a training strategy for time-series foundation models.

*Bearing on us.* This is a "predict the achievable error before training, from a complexity
statistic" result at a scale we cannot match, published in the last year. It does **not**
cover dynamical-systems surrogates, rollout stability, or architecture *selection*, and it is
univariate. But it means the framing "estimate the error floor from a data statistic" is now
occupied territory for time series. We must position against it explicitly.

**A18 — Wang, Klee & Roos (2025).** *Time Series Forecastability Measures*,
`arXiv:2507.13556`. Verification: abstract verbatim. Proposes **spectral predictability score**
plus **largest Lyapunov exponent** as pre-training forecastability metrics; validated on
synthetic data and M5; reports *"strong correlation with the actual forecast performance of
various models"*. Practical framing: decide which products are worth forecasting at all.
Note the tension with A1 — this paper is happy to use λ as a forecastability proxy on
real-world retail data, while Gilpin finds λ decorrelates from skill on chaotic systems.
Both cannot be generally true; the resolution is probably that they are in different regimes
(short-horizon and weakly-chaotic vs long-horizon and strongly-chaotic), which is itself a
testable statement.

### F.2.3 The verdict for us

- **Better chaoticity instruments exist and are cheap**: FSLE, ε-entropy, weighted permutation
  entropy, the modified 0-1 test, and the CDTA pipeline that decides whether chaos machinery
  applies at all.
- **The translation to learnability has been drawn, three times, and never for our setting**:
  Garland 2014 (120 series, 4 classical forecasters), Pennekamp 2019 (461 ecological series),
  Wang 2025/26 (4,700 deep models, univariate, window-wise complexity).
- **What is genuinely unoccupied**: none of these look at *rollout stability*, *architecture
  selection*, or *multivariate dynamical-system surrogates*, and none of them regress the
  residual after controlling for capacity and data. The gap is narrower than the proposal
  assumes, but it is still there — and it is now a *much better specified* gap, because we can
  state exactly which published curve we are extending and in which direction.
- **Immediate practical consequence**: WPE and the 0-1 K-statistic go into our feature set as
  **baselines to beat**, alongside λ. If a learned surrogate-predictor cannot beat weighted
  permutation entropy — a 20-line, parameter-free statistic from 2002 — we have no result.

---

## F.3 — P3: the user's point about parameter count, and what to do with it

**The user's position:** the small parameter counts in NNPT (1186–2242 params) are an
advantage for us, not a limitation, and the result should be testable at slightly larger
scale — order 1e3–1e5.

**Agreed on the strategic point, with one correction on what the actual weakness is.**

**Where the user is right.**
1. NNPT's regime is *precisely* our hardware regime. A 2-core CPU cannot do transformer-scale
   anything, but it can do thousands of 1e3–1e5-parameter fits. A paper whose central result
   lives at 1e3 params is a paper we can replicate, extend, and beat on rigour without a GPU.
   That is rare and it is worth a lot.
2. The result is *about capacity*, and capacity sweeps are embarrassingly parallel across
   seeds and configurations. Our environment audit says ~1–3 min per fit, 5 seeds ≈ 15 min per
   configuration. A capacity × chaoticity grid is affordable.
3. Small models make the *equalised-accuracy* protocol meaningful. At large scale everything
   hits the tolerance and "required capacity" degenerates.

**Where the real weakness is — and it is not that the models are small.**
The weakness is **resolution and support of the capacity axis, not its magnitude**:
- The headline non-monotonicity is carried by a **depth change: 3×32 (2242 params) → 2×32
  (1186 params)**. That is a **two-point grid on a discrete axis**. A 47% drop measured across
  one architectural step is not a curve; it is two points with a line drawn through them.
- No seed count is reported in the abstract. "Required capacity" is a threshold-crossing
  statistic, and threshold-crossing statistics are **high-variance**: one unlucky
  initialisation moves the smallest passing model by a whole architecture step.
- "Required capacity at 1% tolerance" is **not the same quantity** as "achievable error
  floor". They can move in opposite directions: a system can require more capacity to reach 1%
  *and* have a lower floor. NNPT measures the first; P5's scaling machinery and the
  Gilpin/Duraisamy dispute are about the second. Treating them as one quantity is a
  confound waiting to happen.
- One system family (three-body with a mass knob) and one architecture family. The Chirikov
  agreement is elegant but it is `n = 1` as a cross-system claim.

**So the upgrade is not "bigger models" — it is a better-measured capacity axis.** Concretely,
what we can do on this hardware that they did not:
1. **Continuous capacity axis.** Sweep *width* on a fine geometric grid (e.g. 8, 11, 16, 23,
   32, 45, 64, 90, 128 hidden units) at fixed depth, so capacity is near-continuous over
   ~2 orders of magnitude, then repeat at 2–3 depths. This turns a two-point step into a curve.
2. **Seeds at every grid point.** 5+ seeds, report the *distribution* of the threshold-crossing
   capacity, not the argmin over one run.
3. **Threshold sensitivity.** Report required capacity at 0.5%, 1%, 2%, 5% tolerance. If the
   non-monotonicity survives all four, it is real; if it appears only at 1%, it is a threshold
   artefact. **This single check is the highest-value cheap test in the whole replication.**
4. **Both dependent variables.** Required-capacity-at-tolerance *and* fitted error floor from
   a scaling-law fit (P5). If they disagree, that disagreement is itself a finding, and it
   speaks directly to the Gilpin-vs-Duraisamy dispute.
5. **More than one chaos knob.** Three-body mass is one path through parameter space. Cheap
   alternatives with a clean chaoticity knob: logistic map `r`, Hénon `a`, Lorenz `ρ`, Duffing
   forcing amplitude, standard map `K` (which *is* Chirikov, so it tests the claimed mechanism
   directly rather than by analogy).
6. **A no-subtraction control.** NNPT's whole premise is learning the residual after
   subtracting an exact solution. Whether the non-monotonicity is a property of *the system*
   or *of the residual* is unresolved and untested. Run both.

**Honest counterweight.** Even done well, this is a replication-plus-extension. On its own it
is a workshop-grade contribution, not a headline one. It becomes headline-grade only if the
non-monotonicity **generalises across system families** and **predicts something** — e.g. if
the capacity peak location can be predicted in advance from a cheap statistic. That is the
version worth aiming at, and note it is the same shape as the F.2 finding: a pre-training
statistic predicting a post-training quantity, with the honest baselines attached.

---

## F.4 — P4: the user's critique of Task2Vec is correct, and it generalises

**User's position:** Task2Vec is not impactful or novel for our purposes; the missing
task–model interaction is a major limitation, not a footnote; "what architecture for what
task" need not be true.

**Agreed, and the critique is sharper than the version in the P4 entry above. Here it is
stated formally, because the formal version is usable in a paper.**

Task2Vec's model-selection score is, structurally:

```
score(task t, expert m)  =  −d_sym(F_t, F_m)  +  α·d_sym(F_t, t_0)
Model2Vec:   m_i = F_i + b_i        (a learned per-model bias vector)
```

Both terms are **additive**: one term depends on the task, one on the model. There is no
term that depends on *the pair*. In factorisation language, Model2Vec's per-model bias is a
**rank-one correction** — it can say "this expert is generally good" or "this task is
generally hard", but it **cannot express** "architecture A is good on task type X and bad on
task type Y while architecture B is the reverse".

That crossing pattern is exactly an **interaction effect**, and an additive model is
provably unable to represent it.

**Three pieces of evidence that this is not a nitpick:**

1. **Their own numbers say so.** Symmetric task2vec gives +42.54% relative error increase on
   iNat+CUB — *worse than the trivial "always use the ImageNet expert" baseline* at +30.18%.
   The method only works after α-weighting a term for source-task *complexity* (+9.97%). The
   fix is a hand-designed asymmetry, not a learned interaction, and it is doing all the work.
2. **Their own limitations section admits it**, in one line: the method *"ignores interactions
   with the model beyond the task itself"*. The user identified the same gap independently;
   the authors named it and did not solve it.
3. **In our domain the interaction is the entire phenomenon.** P7: DeepONet is poor on regular
   grids and **best on every irregular-mesh case**. CNO in P8: 1.05× degradation on Poisson
   (best) and 12.2× on Black–Scholes payoff shift (worst). These are textbook crossings. No
   additive score can rank them correctly. A method that assigns one number per architecture
   and one per task will get the sign wrong on exactly the cases we care about.

**What this rules out and what it leaves.**
- **Ruled out:** any "embed the task, look up the best architecture" design. It is
  mis-specified for this problem, and it will be beaten by "always pick the architecture that
  is best on average" — which is the baseline it needs to beat, and which A9 confirms is
  brutally hard to beat in the neighbouring NAS literature.
- **What remains defensible:** predicting the **interaction term directly**. i.e. the target
  is not "which architecture is best for task t" but a matrix `E[t, m]` of task × architecture
  outcomes, and the question is whether that matrix has low rank, and whether its factors
  correlate with anything measurable in advance. That is a strictly harder and strictly more
  honest question, and it has a clean negative result available: if `E` is essentially rank-1,
  then architecture choice does not depend on the task and the whole enterprise is
  unnecessary — a genuinely useful thing to know and a publishable negative.
- **Note the connection to A8.** V-information already says learnability is indexed to a
  predictor family. "Learnability is a property of the (task, family) pair, not of the task"
  and "the score must contain an interaction term, not a sum" are the **same statement** in two
  vocabularies. The user's critique of P4 and the V-information verdict are one objection.

**Concrete consequence for design:** any spike in this direction must produce a
**task × architecture outcome matrix** with enough cells to estimate its rank, and must report
(i) the rank-1 baseline (best-on-average architecture), (ii) the rank of the residual, and
(iii) whether pre-training statistics predict the residual factors. Anything less is a
Task2Vec re-run with the same flaw.

---
---

# Part G — sources added in this addendum

| ID | Reference | Verification |
|---|---|---|
| A12 | Israeli & Goldenfeld (2006), *Coarse-graining of cellular automata, emergence, and the predictability of complex systems*, Phys. Rev. E **73**, 026203 | abstract verbatim |
| A13 | Dzwinel & Magiera (2015), *Irreducible elementary cellular automata found*, J. Comput. Sci. **11**, 300–308, doi:10.1016/j.jocs.2015.07.001 | abstract verbatim |
| A14 | Toker, Sommer & D'Esposito (2020), *A simple method for detecting chaos in nature*, Communications Biology **3**, 11 | full metadata + detailed method extraction |
| A15 | Garland, James & Bradley (2014), *Model-free quantification of time-series predictability*, Phys. Rev. E **90**, 052910 / `arXiv:1404.6823` | full HTML text read |
| A16 | Pennekamp et al. (2019), *The intrinsic predictability of ecological time series and its potential to guide forecasting*, Ecological Monographs, doi:10.1002/ecm.1359 | **PARTIAL — indexed text only, journal page 403. Verify before citing.** |
| A17 | Wang et al. (2025/26), *Exploring Accuracy Law for Deep Time Series Forecasters: An Empirical Study*, `arXiv:2510.02729` | abstract verbatim + metadata. **Read in full before choosing a direction.** |
| A18 | Wang, Klee & Roos (2025), *Time Series Forecastability Measures*, `arXiv:2507.13556` | abstract verbatim |

**Standing threat register update.** A17 is a new entry at the top of the challenges table:
a 4,700-model empirical law relating minimum attainable forecasting error to a pre-training
complexity statistic. It does not cover our setting (dynamical-system surrogates, rollout,
architecture selection, multivariate), but it occupies the framing. Any direction we pick must
say in one sentence how it differs from the accuracy law.
