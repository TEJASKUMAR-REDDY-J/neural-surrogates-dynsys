# Data forms, representations, and transformations

Reference document. Purpose: the proposal wants to estimate "learnability of a task" and
"which architecture will fit". **Neither quantity is well defined until the representation
is fixed** — the same physical process, encoded differently, is a different learning
problem. This document maps the space of choices so that our experimental design states
them explicitly rather than inheriting them by accident.

Provenance: synthesis of standard practice plus the specific choices made in the papers
read this session (see `02-paper-extractions.md`). Where a claim comes from a specific
paper it is cited; the rest is established practice and is marked as such.

---

## 0. Why this is a research question and not just plumbing

Three results from the corpus make representation a first-class variable, not preprocessing:

- **A8 (V-information).** Usable information is defined *relative to a predictive family*.
  Computation can create usable information — V-information violates the data processing
  inequality. A transformation that is information-preserving in the Shannon sense can
  still make a task learnable or unlearnable for a given model class. **So "how much is
  learnable" is a function of (data, representation, model family), never of data alone.**
- **P5 (scaling laws).** Resolution-limited exponents go as `1/d` where d is the intrinsic
  dimension of the *input manifold*. Change the embedding and you change d, therefore the
  scaling exponent, therefore the measured "difficulty". Their own finding that α is
  sensitive to input noise but insensitive to class count says the input representation
  dominates.
- **A4 (context parroting).** A model given a long context window on a recurrent attractor
  can score well by copying. The *representation of the input window* — how much history,
  at what sampling rate — determines whether the task rewards rule-learning or lookup.

Consequence for us: **any learnability number we report must be tagged with the
representation and the model family, and a representation ablation is not optional.**

---

## 1. The pipeline: how anything becomes model input

Every scenario factors into the same five stages. Naming them separately is useful because
different literatures conflate different pairs.

| Stage | Question | Typical free choices |
|---|---|---|
| 1. Source form | What does the phenomenon natively look like? | continuous field, event stream, relational records, text |
| 2. Sampling / discretisation | Where and when is it measured? | grid vs mesh vs scattered; Δt, Δx; resolution; irregular sampling |
| 3. Value encoding | How is each measurement turned into a number? | units, scaling, log/Box–Cox, quantisation, one-hot, embedding |
| 4. Structural encoding | How is the arrangement conveyed to the model? | tensor axes, adjacency, coordinates, positional encoding, tokens |
| 5. Task framing | What is the input→output contract? | one-step vs multi-step, horizon conditioning, autoregressive vs direct |

Stages 2 and 5 are where most of the "learnability" of a dynamical-systems task is actually
decided, and both are usually reported in one line of a methods section.

---

## 2. Data forms and how each reaches a model

### 2.1 Tabular / fixed-length feature vectors
Rows are i.i.d. samples, columns are heterogeneous features. Reaches the model as a flat
vector. No spatial or temporal structure assumed. Standard consumers: GBDTs, MLPs.
*Notes:* mixed types force per-column encoding (numeric scaling, categorical one-hot /
target encoding / learned embedding). GBDTs remain hard to beat here; the absence of useful
inductive bias in the representation is exactly why.

### 2.2 Time series (regular)
Shape `(T, C)` — T timesteps, C channels. Reaches the model as a window `(L, C)` slid over
the series. Consumers: RNN/LSTM/GRU, TCN, N-BEATS/N-HiTS, transformers, state-space models
(S4/Mamba), reservoir computers.
*Key free choices:* lookback length L, stride, normalisation scope (global vs per-window),
whether to predict the value or the **increment**.

### 2.3 Time series (irregular / event-based)
Timestamps are not uniform. Options: interpolate to a grid (destroys information about when
observation happened), feed `(Δt, value)` pairs, or use continuous-time models (neural ODE,
neural CDE, latent ODE, Hawkes processes).
*Note:* the choice of interpolation is a strong prior. Cubic interpolation of a chaotic
signal manufactures smoothness that was not measured.

### 2.4 Scalar/vector fields on regular grids
Shape `(C, H, W)` or `(C, D, H, W)`, optionally with a leading time axis. This is the
easiest case and the one benchmarks over-represent (P7's whole complaint). Consumers: CNNs,
U-Nets, FNO and spectral operators, vision transformers with patchification.
*Why it is easy:* translation invariance and a well-defined Fourier basis both come free.

### 2.5 Fields on unstructured meshes
Node/cell values plus connectivity. No global Fourier basis, no translation invariance.
Reaches the model as (a) a graph — nodes, edges, edge features (relative position) — for
GNNs/MeshGraphNet; (b) a point cloud with coordinates for pointwise operators
(DeepONet, PointNet); or (c) interpolated onto a background grid, trading geometry fidelity
for architectural convenience.
*Empirical anchor:* P7 found DeepONet worst on regular grids and **best on every irregular
case**, while graph nets over-smoothed. The representation, not the parameter count,
decided the ranking.

### 2.6 Point clouds and particle systems
Unordered sets of positions with attributes. Requires permutation invariance. Consumers:
PointNet-style set encoders, GNNs on k-NN graphs, continuous convolutions.
*Note:* a k-NN graph is a *constructed* representation — the value of k is a modelling
choice that changes what is learnable.

### 2.7 Graphs and relational data
Nodes, edges, optional global features. Reaches the model as message passing over an
adjacency. Distinct failure mode: over-smoothing with depth; distinct capability: encoding
known interaction structure as a hard prior.

### 2.8 Sequences of discrete tokens
Text, code, symbolic states, quantised continuous values. Reaches the model as integer ids →
embedding table → transformer. Tokenisation is a lossy, non-obvious transform: for numeric
time series, digit-level tokenisation, binning, and value-scaled quantisation all give
different learnability. (This is how time-series foundation models — A3, A4 — ingest
trajectories.)

### 2.9 Images and video
`(C, H, W)` / `(T, C, H, W)`. Patchification into tokens, or convolutional stacks.
Relevant to us only as an analogy: CA rasters and 2D field snapshots are images, and CA
"learnability" studies often reduce to image prediction.

### 2.10 Cellular automata and discrete-state lattices
Symbolic lattice `(T, N)` over a small alphabet. Reaches the model as one-hot channels, as
binary values, or as tokens. Note that **the coarse-graining of P1 is itself a
representation change** — a block-and-project transform — and it converts 240/256 rules from
"irreducible" to "reducible". That is the cleanest existing demonstration that
representation determines measured learnability.

### 2.11 Multiphysics / multi-channel stiff systems
Many channels with wildly different dynamic ranges (species mass fractions spanning
`1e-8` to `1`). Reaches the model as a many-channel tensor, but only after range
compression. P7 uses **Box–Cox with λ = 0.1** on mass fractions, then per-channel z-score,
precisely to stop the small-magnitude channels being invisible to an MSE loss.
*This is a case where a transformation choice silently reweights the objective.*

---

## 3. Discretisation choices specific to dynamical systems (our case)

This is where our design decisions live, so it is worth separating out.

| Choice | Options | Consequence |
|---|---|---|
| Observable | full state; partial state; scalar observable | Partial observation makes the process non-Markovian and forces delay embedding |
| Sampling rate Δt | fixed; per-system aligned to a dynamical timescale | A1 resamples to **100 points per dominant Fourier period**, so systems are comparable in *dynamical* rather than wall-clock time. Without this, "which system is harder" is confounded with sampling |
| Time normalisation | raw t; Lyapunov time λ·t | The chaos literature reports everything in Lyapunov times. Not doing so makes results incomparable to A1 and to the reservoir-computing VPT literature |
| Trajectory length / count | few long vs many short | Changes whether the model sees the attractor's recurrence structure — directly determines whether parroting (A4) is available |
| State vs increment | predict `x(t+1)` or `x(t+1) − x(t)` | Increment prediction is standard for stability; it changes the target's scale and spectrum |
| Horizon framing | autoregressive one-step; multi-step unrolled; **direct horizon-conditioned** | A6 conditions continuously on horizon T. This is a representation choice at stage 5, and it is the difference between compounding error and not |
| Noise | clean integration vs observational noise σ | P2: above the noise scale, chaos and noise are indistinguishable. σ sets a hard resolution floor on any learnability claim |
| Precision / tolerance | integrator rtol/atol | For chaotic systems the "ground truth" is itself resolution-limited; a loose tolerance manufactures irreducible error |

---

## 4. Transformation methods, by what they change

### 4.1 Value-level (per-feature, no structure change)
- **Affine scaling:** z-score (per-channel, per-window, or global), min-max, robust
  (median/IQR). *Scope matters:* per-window normalisation removes level information and can
  make a non-stationary problem look stationary.
- **Monotone range compression:** log, log1p, **Box–Cox** (needs positivity),
  **Yeo–Johnson** (handles zeros/negatives). Used to stop small-magnitude channels being
  ignored by MSE (P7).
- **Rank / quantile transforms:** rank-Gauss, quantile normalisation. Destroys magnitude,
  keeps order. Very robust, and *removes* exactly the information a physics loss needs.
- **Whitening / decorrelation:** PCA-whitening, ZCA. Changes the effective conditioning of
  the problem and hence optimisation, not just the data.
- **Clipping / winsorising:** trades tail fidelity for stability. On stiff systems the tails
  *are* the physics.

### 4.2 Basis / domain transforms (linear, structure-preserving)
- **Fourier (DFT/FFT):** the natural basis for periodic domains and the substrate of FNO.
  Makes smooth-field problems low-dimensional; **introduces spectral bias** (A5's named
  mechanism) because truncating modes discards exactly the high-frequency content that
  drives shocks and chaos.
- **Wavelets:** localised in both time and frequency; appropriate when the signal has
  transient structure that Fourier smears.
- **POD / PCA / SVD:** empirical orthogonal basis from data; gives a data-driven intrinsic
  dimension estimate, which is directly the `d` in P5's `α ≈ 1/d`.
- **DMD / Koopman lifting:** represent nonlinear dynamics as a linear operator on a lifted
  observable space. The lift is the representation choice, and a good lift can make a
  nonlinear system linearly predictable — the continuous analogue of P1's coarse-graining.
- **Spherical harmonics:** for data on a sphere (weather). Same trade-offs as Fourier.

### 4.3 Temporal-structure transforms
- **Delay embedding (Takens):** `x(t) → [x(t), x(t−τ), …, x(t−(m−1)τ)]`. Reconstructs an
  attractor diffeomorphic to the true one from a scalar observable, given m > 2d. **This is
  the single most important transform for partially observed dynamical systems**, and the
  choice of (m, τ) changes the geometry of the reconstructed manifold and therefore the
  measured difficulty.
- **Differencing / increments:** removes drift, changes spectral content.
- **Detrending / deseasonalising:** classical; usually inappropriate for autonomous
  dynamical systems.
- **Windowing / lookback selection:** determines the availability of the parroting shortcut
  (A4). A long window on a recurrent attractor makes copying viable.
- **Resampling to a dynamical timescale:** as in A1. Makes cross-system comparison
  meaningful.

### 4.4 Discretisation and symbolisation
- **Uniform / quantile binning:** continuous → discrete alphabet; enables information-
  theoretic quantities (ε-entropy, statistical complexity) to be estimated. The bin width
  *is* the ε in P2's `h(ε)`.
- **SAX, ordinal (permutation) patterns:** rank-based symbolisations; permutation entropy is
  a cheap, robust complexity statistic and a plausible pre-training feature.
- **Coarse-graining (block-and-project):** P1's supercell construction. Formally a
  representation change; the finding is that it converts irreducible to reducible for 94% of
  ECA rule space.
- **ε-machine / causal-state reconstruction:** the maximally predictive minimal
  representation of a stochastic process; gives statistical complexity and a computable
  optimal error (A7).

### 4.5 Geometric / structural transforms
- **Mesh → graph:** nodes + edges + relative-position edge features.
- **Mesh → point cloud + coordinates:** drops connectivity, keeps geometry.
- **Scattered → grid interpolation:** enables CNN/FNO at the cost of geometric fidelity;
  the interpolation kernel is a prior.
- **Coordinate channels / positional encoding:** appending `(x, y, t)` or Fourier features
  of position. Used by all three architectures in P8. Cheap, and it changes what the network
  can express about location.
- **Signed distance functions / level sets:** encode geometry implicitly for complex
  boundaries.
- **Patchification:** grid → sequence of patch tokens. Trades locality for sequence-model
  machinery.

### 4.6 Learned representations
- **Autoencoders / VAEs:** learned low-dimensional latent; latent dimension is a tunable `d`.
- **Latent ODE / latent dynamics models:** learn an encoder plus dynamics in latent space.
- **VQ / discrete tokenisation:** continuous → learned codebook → sequence model.
- *Caution:* a learned representation is fitted on data, so any "pre-training learnability
  estimate" built on it is no longer training-free. This is a real trap for the proposal's
  "estimate before training" framing.

### 4.7 Augmentation and invariance
- **Symmetry augmentation:** translation, rotation, reflection, Galilean boost, time
  reversal for reversible systems.
- **Equivariant architectures:** bake the symmetry into the model instead of the data.
- **Noise injection / pushforward trick / temporal bundling:** rollout-stability training
  techniques. These are *task-framing* transforms — they change the training distribution to
  match the rollout distribution. **Note that P7 does not use them** (loss on the final
  rollout step only), which weakens its rollout conclusions.

---

## 5. Representation ↔ architecture inductive bias

Which architecture is even applicable is largely determined by stage 4, not by the physics.

| Representation | Natural architectures | Bias it grants | Where it fails |
|---|---|---|---|
| Regular grid | CNN, U-Net, FNO/spectral | translation invariance; global multiscale coupling (spectral) | irregular geometry; sharp fronts (spectral bias) |
| Unstructured mesh graph | GNN, MeshGraphNet | locality on arbitrary topology | over-smoothing with depth (P7) |
| Point cloud + coords | DeepONet, PointNet | permutation invariance; geometry-agnostic | no explicit locality; needs many points |
| Sequence of tokens | transformer, SSM | long-range dependence; in-context learning | quantisation loss; parroting shortcut (A4) |
| Delay-embedded vector | MLP, reservoir computer | Takens guarantee on attractor reconstruction | (m, τ) sensitivity; scalar-observable limits |
| Spectral coefficients | FNO, spectral methods | resolution independence | truncation discards shock-relevant modes |
| Flat feature vector | MLP, GBDT | none | everything structural |

**The practical reading:** P7's finding that "architecture matters more than parameter
count" is, restated, *the match between representation and inductive bias matters more than
scale*. That reframing is more useful to us than the headline, and it makes the proposal's
architecture-selection question a **representation-matching** question.

---

## 6. How transformations corrupt learnability measurements

Concrete, checkable failure modes to design against.

1. **Normalisation scope leakage.** Computing normalisation statistics over train+test
   leaks. Per-window normalisation additionally removes level information and can make a
   drifting system look stationary.
2. **Objective reweighting by scaling.** MSE after per-channel z-scoring weights all
   channels equally; MSE on raw values weights the largest-magnitude channel. Box–Cox
   changes the weighting again. **The "same" loss is three different objectives.** (P7 makes
   this choice explicitly; most papers do not.)
3. **Metric/representation interaction.** Pointwise error in a spectral basis is not
   pointwise error in state space. Cluster 3 of the corpus (pointwise vs distributional
   decoupling) is partly a representation effect.
4. **Manufactured smoothness.** Interpolating irregular samples, or integrating with loose
   tolerance, creates structure the model then "learns".
5. **Manufactured closure.** P1's coarse-graining is found by *search over projections*. A
   wide enough search over representations will find one in which almost anything looks
   predictable. Any "does a closed coarse description exist" measurement must fix the search
   space in advance.
6. **Shortcut availability.** Long context windows on recurrent attractors make copying
   viable (A4). Whether the task measures rule-learning or lookup is set by window length.
7. **Resolution floor.** Above the observational noise scale, chaos and noise are
   indistinguishable (P2). Any learnability estimate must state its ε.
8. **Intrinsic dimension is representation-dependent.** P5's exponent goes as `1/d`. Change
   the embedding, change d, change the measured exponent — with no change to the physics.

---

## 7. Decisions this forces on our experiments

Not conclusions, but the list of things that must be **fixed and stated before results are
seen**, and ablated afterwards.

- [ ] **Substrate and system list fixed in advance** — `dysts` (A2) gives a public, fixed
      135-system list with precomputed invariants; use it rather than choosing systems.
- [ ] **Timescale alignment** — resample per system to a fixed number of points per dominant
      period (A1's protocol), and report horizons in Lyapunov times.
- [ ] **Observable** — full state vs scalar-plus-delay-embedding. If delay embedding is used,
      (m, τ) selection rule is stated in advance.
- [ ] **Normalisation** — per-channel z-score with train-only statistics; state the scope.
      If any range-compressing transform is used, justify it and ablate it.
- [ ] **Target** — state vs increment; fixed across all conditions.
- [ ] **Task framing** — autoregressive vs direct horizon-conditioned; if both, treat as a
      factor, not a confound.
- [ ] **Baselines that must exist:** (i) persistence, (ii) **context parroting /
      nearest-neighbour analogue (A4)**, (iii) parameter count and one short training run
      (A9), (iv) a Lyapunov-only regression. Without (ii) our claims are unfounded.
- [ ] **Metrics reported in pairs** — one pointwise (sMAPE/NRMSE, in Lyapunov time) and one
      distributional/structural (correlation dimension, invariant-measure error, spectrum).
      Claims must hold under both or the divergence itself is reported.
- [ ] **Representation ablation** — at minimum one alternative encoding, to show the result
      is about the system and not the encoding. Given A8, a result that is not stable across
      representations is a result about the representation.
- [ ] **ε stated** — the resolution/noise level at which every learnability claim holds.
- [ ] **Seeds** — multiple seeds before any claim; P8 uses 50 per condition, our environment
      audit says 5 seeds of one configuration is ~15 min, so this is affordable.

---

## 8. One-line summary

Representation is not preprocessing here — it is a free variable that moves the quantity we
are trying to measure. The corpus contains three independent demonstrations of this
(P1: coarse-graining converts irreducible to reducible; A8: usable information is
family-relative; P5: the scaling exponent is set by input-manifold dimension). Any
"learnability profile" we produce is a function of (system, representation, model family),
and the honest version of the proposal's contribution states all three.
