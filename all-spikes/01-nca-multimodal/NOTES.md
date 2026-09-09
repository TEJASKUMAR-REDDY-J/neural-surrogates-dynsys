# 01-nca-multimodal — notes

Running log for the neural-cellular-automata spike. Newest at the bottom.

**What this spike is for.** Every experiment in `00-replications` used one architecture — a
plain MLP that sees the whole state at once — on one kind of data. That leaves the most
obvious question unasked: **does the shape of the architecture matter, and does it matter
differently on different kinds of data?**

This spike builds three cellular automata that differ in exactly one way — how much of the
lattice a single cell is allowed to see — and runs them across five kinds of data against
five methods that do no learning at all.

**What it is not.** It is not a claim about state-of-the-art anything. It is a smoke test:
does the machinery work, do the architectures behave differently, and is the difference in the
direction we predicted? More architectures come after.

---

## 2026-09-08 — spike created

### The one design decision everything else follows from

To compare an architecture across chaotic systems, images, text and tabular data, all four
have to become the same kind of object. So everything is a **lattice of cells**:

| data | lattice | do neighbours mean anything? |
|---|---|---|
| coupled map lattice, 64 sites | a line of 64 | yes — they are physically coupled |
| a 64-step time window | a line of 64 | yes — they are consecutive in time |
| a Lorenz state | a line of 3 | **no** |
| rule 110 | a line of 64 | yes |
| digits | an 8×8 grid | yes |
| natural image patches | a 16×16 grid | yes |
| characters | a line of 64 | yes |
| clinical features | a line of 30 | **no** |
| sum of three sinusoids | a line of 64 | yes |

The two "no" rows are the control. A local-only automaton should struggle there and a
global-context one should not. If that contrast does not appear, the global branch is not
doing what it was added for and the whole design is wrong.

### The task, and why it is the same task everywhere

Hide some cells, work out what was there.

- hide scattered cells of an image → inpainting
- hide scattered characters → text repair
- hide the **end** of a time window → **forecasting**

One pipeline, one metric, every modality. Three ways of hiding: `random` (a neighbour
probably knows), `block` (the neighbours are missing too, so you need something further
away), `tail` (forecasting). **`block` is where a global branch should earn its keep.**

### The three architectures

All share the specified per-cell body — **32-64-128-64-32** — applied identically at every
cell, iterated. They differ only in what feeds it:

| | what a cell sees | extra parameters |
|---|---|---|
| **A1 local** | itself and its immediate neighbours | — |
| **A2 pooled** | + one summary of the whole lattice (mean, max, spread), same for everyone | ~6k |
| **A3 attentive** | + a summary it queries for itself: the lattice is squeezed to ≤64 tokens, those attend to each other, each cell reads back the token covering its region | ~6k |

22,864 / 29,104 / 28,720 parameters on a 64-cell line, so the comparison is close to
size-matched and any difference is about *shape*, not capacity.

Two details borrowed from the NCA literature, both load-bearing:

- **the output layer is zero-initialised**, so an untrained automaton is exactly the identity
  and has to earn every change it makes. The self-check asserts this.
- **the number of iterations is random each batch** (4 to 10). Train at a fixed count and the
  rule learns to be punctual rather than stable — which is precisely the failure R9 found in
  `00-replications`, where refinement degraded past its training budget in 24/24 cases. This
  is the cheap fix for it, and evaluating out to 32 steps tests whether it worked.

A3's attention is capped at 64 tokens by adaptive pooling, so its cost does not grow with the
lattice. A 16×16 grid and a 512-cell line both attend over ≤64 tokens.

### The five methods that do not learn

Included because of the single most-repeated finding in `00-replications`: a trained network
ties with a lookup table on the standard benchmark, and N6 showed why. No architecture claim
means anything until it has cleared these.

`identity` (leave it alone) · `global_mean` · `local_mean` (diffuse the visible values in) ·
`nearest_neighbour` (**the analogue-lookup baseline, generalised**) · `pca_impute`.

### Substrate note: the Kuramoto–Sivashinsky blocker is gone

The spatially extended substrate here is a **coupled map lattice**, not KS. It is an explicit
map — no integrator, so it cannot destabilise — and its Jacobian is analytic and tridiagonal,
so the Lyapunov spectrum is computed exactly rather than estimated. Self-check measures
λ_max = 0.311 at a=3.9, ε=0.3, confirming genuine spatiotemporal chaos.

This is what `background/06` argued for, and it took an afternoon rather than two sessions.

### Honest limitations, recorded before any results

- **Text is encoded crudely** — a character is its byte value over 255, so 'a' and 'b' are
  near each other for no good reason. Deliberate: it keeps every modality in one identical
  pipeline. It means the text row measures *geometry handling*, not language modelling, and it
  must be described that way.
- **One channel everywhere.** Colour images and token embeddings are the obvious next step.
- **Small.** ~29k parameters, ~675 training examples per dataset, 40 epochs. A smoke test.
- **`local_mean` is allowed to adjust visible cells**, unlike the other baselines; the
  self-check exempts it explicitly rather than silently.

### Self-checks, all passing

- `data.py` — 9 datasets, correct rank/shape/dtype/range; CML confirmed chaotic; ECA binary.
- `models.py` — all 3 architectures run on 1-D, 2-D and degenerate lattices; untrained output
  is exactly the input; gradients reach parameters.
- `task.py` — each masking mode hides roughly what it claims; baselines do not overwrite what
  they were allowed to see; diffusion and lookup both beat doing nothing on smooth data.
- `runlog.py` — copied from `00-replications` so this spike stands alone; self-check passes.

---

## 2026-09-08 — a fourth architecture, a parameter budget, and probes that measure mechanism

The spike as first built answered "which architecture scores better". That is the less
interesting half. This round adds the machinery to ask **what each automaton is actually
doing**, plus the architecture that was missing.

### A4 — interleaved, the "global in between local, like a diffusion model" idea

A2 and A3 hand a cell its global context **once, at the input**, and then let a stack of
per-cell layers work on it. A4 alternates instead: local block, global mix, a genuinely
local depthwise 3x3 hop, another global mix, then the rest. That is how a diffusion UNet is
arranged — convolution blocks with global attention inserted at intervals — and it is a
different hypothesis about *where* global information is useful, not just how much of it
there is.

Its global mix is deliberately **not** attention. The lattice is pooled to tokens and those
tokens are mixed by one learned matrix over token positions (the MLP-Mixer token-mixing
idea). Cheap, no softmax to saturate, and it makes the global path a plain linear operator
whose contribution is straightforward to measure. Both global mixes are zero-initialised and
added as residuals, so at the start of training the global path contributes exactly nothing
and has to earn its way in.

### Parameter budget: size-matched, not width-matched

The specified body 32-64-128-64-32 costs ~23k for A1 and ~29-30k once a global branch is
attached. The budget asked for is 10-15k.

Applying one width to all four would be the obvious move and it would be **wrong**: the
global branches cost a fixed ~5k on top of the shared body, so a common width leaves the
local-only automaton materially smaller, and then any difference between architectures is
partly capacity — which is the exact confound this comparison exists to avoid.

So the shape (1:2:4:2:1) is kept and each architecture gets its own width scale, chosen by
`models.scale_for_budget` to land on the same parameter count:

| | width scale | parameters |
|---|---|---|
| A1 local | 0.742 | 13,215 |
| A2 pooled | 0.613 | 13,046 |
| A3 attentive | 0.555 | 13,163 |
| A4 interleaved | 0.586 | 12,795 |

Within 3% of each other, all inside the budget. `--budget 0` runs the literal specification
instead, so the effect of the width itself can be measured rather than assumed.

### Four probes, in `src/common/probes.py`

The point of this spike is not to fit data. It is to read a signal off an architecture.

- **propagation** — poke one cell, watch how far the ripple has travelled after each step.
  This is the measurement that motivates the whole design: a purely local rule moves
  information one cell per step, so on a 100x100 grid one corner cannot influence the
  opposite corner in under ~100 steps regardless of parameter count. That is arithmetic, not
  a training problem.
- **knockout** — silence the global branch (`global_gain = 0`) and re-score. Same weights,
  so the comparison is one model against itself rather than against a separately trained one.
- **contribution** — at every step, the share of the update's magnitude the global branch is
  responsible for. A branch that is present but idle is a real outcome and worth catching.
- **memory** — halfway through a rollout, erase the scratch channels and carry on. The
  visible data and the mask are left intact, so only what the rule *worked out along the way*
  is destroyed. This is the "how dependent is it on historical information" question, asked
  of a system that has no explicit history buffer.

Every probe fixes the fire rate to 1.0 and restores it afterwards, because otherwise two
rollouts that should differ only in the thing being probed also differ in which cells fired.

**Instrument calibration, because "greater than zero" is a weak test.** On a randomly
perturbed model the measured global share is ~3e-04 — correctly near zero, since an
untrained branch is not used. So the self-check also amplifies the global branch 1x, 10x and
100x and asserts the reading follows: **2.2e-03 -> 3.4e-02 -> 9.8e-01**. If it had not, every
number the probe produces later would be worthless.

**First trained model checked, and the signature is exactly as predicted.** A1 on the coupled
map lattice: radius by step `1;2;3;4;5;6;7;8;9;9;9;...`, one cell per step until the ripple
decays below tolerance; `frac_affected_step1` = 3/64 cells, which is the 3-cell stencil
exactly; `knockout_ratio` = 1.000000, since A1 has no branch to silence; and
`memory_ratio` = 1.139, so wiping the scratch channels costs 14% — the trained rule really is
carrying something across steps.

### Metrics

`task.score` now reports everything asked for, on the hidden cells and on the whole lattice:
**MSE, MAE, nRMSE, relative L2**, plus per-step curves that give **one-step error, multi-step
error and error against rollout horizon**, plus **valid prediction time** in iterations at the
0.4 threshold used throughout `00-replications`, so the two spikes' numbers can be read side
by side. VPT here catches the R9 failure directly: a rule that is good at step 4 and garbage
at step 32.

### On CAX — evaluated, deliberately not adopted

`cax` 0.4.4 (Cellular Automata Accelerated in JAX) downloads fine, but it requires
**jax >= 0.10.0 and flax >= 0.12.6**; this machine has **jax 0.7.1 and flax 0.8.4**. Adopting
it means a major upgrade of a working environment, and its speedup comes from JIT-compiled
`lax.scan` on accelerators — on two CPU cores with no GPU the payoff is small and the
breakage risk is not.

What was taken from it instead is the design, which costs nothing: the strict
**perceive -> update** split, a state carrying hidden channels beyond the observable ones,
and treating the step function as the unit that gets composed. All of that is already how
`models.py` is written. If this spike ever moves to a machine with a GPU, porting is the
first thing to do, and the architectures are written so that it is a rewrite of the step
function only.

### Architecture diagrams

`src/diagrams.py` draws all four, plus one explaining why a global branch exists at all.
Every label — channel counts, layer widths, parameter counts, token-grid shapes — is read
off a **built model** rather than typed in, so a diagram cannot quietly drift away from the
code. If the architecture changes and the diagram is not regenerated, the numbers on it are
wrong in an obvious way instead of a silent one.

---

## 2026-09-09 — v4: eight architectures, and the shuffled-twin prediction confirmed

180 more fits, zero failures. Eight architectures now run on the same fifteen datasets,
same masks, same seeds, same protocol, all between 12,795 and 14,479 parameters.

### The prediction I put on record, and what happened

Before the run: *"A5 is the test of the structural explanation. Full self-attention with
per-cell tokens is the one architecture here that could in principle learn arbitrary
cell-to-cell relations and so recover a permuted lattice. If it also collapses, the limit is
the whole weight-shared family. If it survives, my explanation was wrong and it was pooling,
not weight sharing."*

**A5 collapses with the rest.** On the shuffled twins: cml 0.614 -> 1.000, digits 0.515 ->
0.979, multitone 0.432 -> 0.995. All eight architectures land at roughly 1.0, which is no
better than guessing the mean.

### But the mechanism is sharper than the explanation I gave, and it corrects me

I said the limit was *weight sharing*. Looking at which architecture survives says something
more specific.

**A5 has no positional encoding.** Its tokens are produced by a shared 1x1 convolution from
each cell's neighbourhood, so the attention layer sees an unordered bag of tokens and is
*permutation-equivariant by construction*. It cannot distinguish cell 17 from cell 42 even in
principle. So A5 does not test "can attention recover a permutation" - it tests a transformer
without position information, and those are different claims. My phrasing was too strong.

The one partial survivor points at the real discriminator. **A4 is the only architecture
containing a learned map between specific positions** - its global mix is
`nn.Linear(T, T)` over token *positions*, the MLP-Mixer token-mixing matrix. And A4 is the
only architecture that survives any shuffled twin (digits 0.505 -> 0.707, against roughly
1.0 for the other seven).

So the honest statement is narrower and better: **an automaton recovers a permuted lattice
only to the extent it contains a learned position-to-position map.** Weight sharing is the
reason most of them lack one, not the mechanism itself. The clean follow-up is A5 plus a
learned positional encoding; if that survives the twin, position information is the whole
story and weight sharing is irrelevant.

### Two design questions answered cleanly

**Fixed perception filters were a real limitation.** A6 uses learned messages from
(cell, neighbour, difference) with *identical* locality to A1 - both leak exactly 0.0000
outside the light cone, so the light cone is not a confound - and beats A1 on **10 of 15**
datasets: gray_scott 0.136 -> 0.100, natural_patches 0.317 -> 0.243, multitone 0.518 ->
0.473. The identity/gradient/Laplacian stencil was leaving something on the table.

**Pooling was hurting A3.** A5 is the same attention without the squeeze to 64 tokens, and
beats A3 on **12 of 15** datasets. So A3's earlier underperformance was the pooling, not
attention itself.

### A prediction of mine that failed outright

A7 was built as the direct test of the memory question - a GRU at every site, gates designed
for carrying state, against every other architecture that carries memory only implicitly. I
wrote that if explicit gating matters, A7 is the architecture whose score should suffer most
when the scratch channels are wiped.

**A7's memory ratio is 1.0001.** Wiping its scratch channels halfway through a rollout costs
essentially nothing. The highest memory reliance belongs to **A4 at 1.1010**, which has no
gating at all.

Giving an automaton explicit memory machinery did not make it use memory. The likely reason
is that the update gate learns to stay open, leaving the rule effectively feedforward, but
that is a hypothesis and the gate statistics were not recorded. Worth logging next time.

### The headline did not move

| | 4 architectures (v3) | **8 architectures (v3+v4)** |
|---|---|---|
| trained beats training-free | 9 / 15 | **9 / 15** |

Doubling the architecture families - adding transformer, graph, recurrent and spectral layers
to the four context variants - **changed nothing**. Gray-Scott is still won by PCA imputation
at 0.016 against the best of eight at 0.083, a factor of five, on the dataset added
specifically as the positive control for locality.

### Mechanism summary, all eight

| arch | params | light-cone leak | global share | knockout | memory |
|---|---|---|---|---|---|
| A1_local | 13,215 | **0.0000** | 0.0000 | 1.0000 | 1.0016 |
| A2_pooled | 13,046 | 0.2632 | 0.2489 | 1.0122 | 1.0118 |
| A3_attentive | 13,163 | 0.0061 | 0.8353 | 1.1506 | 1.0056 |
| A4_interleaved | 12,795 | 0.0776 | 0.2478 | 1.0320 | **1.1010** |
| A5_transformer | 14,479 | 0.0353 | 0.6746 | 1.0489 | 1.0159 |
| A6_graph | 13,108 | **0.0000** | 0.0000 | 1.0000 | 0.9975 |
| A7_recurrent | 13,038 | **0.0000** | 0.0000 | 1.0000 | 1.0001 |
| A8_spectral | 13,835 | 0.0310 | 0.0639 | 1.0041 | 0.9986 |

Three architectures leak *exactly* zero outside the light cone - A1 by construction, A6 as
asserted in its self-check, A7 as a consequence of reading only its neighbourhood. That is the
structural result restated: locality is a hard constraint, not a tendency.

A8 is the surprise in this table. Its spectral path is global in one step, yet it supplies
only 6% of each update and silencing it costs 0.4%. An FNO-style global path is present and
almost unused.
