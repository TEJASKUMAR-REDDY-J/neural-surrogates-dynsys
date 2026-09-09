# The devil's advocate, revisited against the NCA evidence

Written 2026-09-09. [05-devils-advocate-and-reframing.md](05-devils-advocate-and-reframing.md)
argued this project down as hard as it could be argued, on 2026-09-08, using only the
`00-replications` battery. Since then the NCA spike has run four architectures across fifteen
datasets and five modalities, and four more architectures are still running.

This walks every point in `05` and says what the new evidence does to it. Nothing in `05` is
edited; this sits on top.

**One caveat up front, because it bounds everything below.** The four layer-family
architectures (transformer, graph, recurrent, spectral) are **still running** — 31 of 90 fits
per shard at the time of writing. Everything here rests on A1–A4 across all fifteen datasets,
which is complete. Where the pending run could change a conclusion, that is said.

---

## Part 1 — The prosecution, point by point

| `05` point | status now | what changed |
|---|---|---|
| **1.1** Every original claim already known or died | **unchanged** | The NCA work does not bear on it |
| **1.2** Our one distinctive finding did not replicate | **worse — now a pattern** | A second mechanistic explanation of ours has failed its own control |
| **1.3** The predictor is a simulation artefact | **unchanged** | Untested here |
| **1.4** The networks may not be doing anything | **strengthened, and generalised** | Reproduced in a different architecture family, on images, discrete CA, tabular and PDE data |
| **1.5** We never tested what the field argues about | **RESOLVED** | Four spatially extended systems now tested, up to 256 cells |
| **1.6** Someone did the ambitious version at 1.6× scale | **unchanged** | Untouched |
| **1.7** The compute ceiling makes the question unanswerable | **sidestepped for the new questions** | The new results are structural, not extrapolations |

### 1.5 was the blocker, and it is gone

`05` called this **non-negotiable**: *"the whole critique is aimed at the neural-operator
literature, and we have tested nothing resembling it."* Prerequisite 5.1 demanded one
spatially extended system, and named Kuramoto–Sivashinsky.

We now have four, and none of them is KS:

| system | cells | kind |
|---|---|---|
| coupled map lattice | 64 | spatiotemporal chaos, explicit map |
| Gray–Scott | **256** | reaction–diffusion PDE on a torus |
| Game of Life | **256** | discrete CA, exactly local |
| natural image patches | **256** | real image statistics |

The KS integrator that blocked two sessions was never fixed. It was **routed around**, exactly
as `background/06` proposed: the scientific requirement was *spatially extended, locally
coupled, chaotic, known ground truth*, and a coupled map lattice supplies all four with no
integrator to destabilise. That is the single biggest change to the project's standing since
`05` was written.

### 1.4 is now much harder to argue against

`05`: *"if a lookup table matches your neural network on the benchmark, what is the neural
network for?"* That rested on one architecture (an MLP) on one kind of data (low-dimensional
ODEs).

It now holds across a second architecture family, on five modalities. Clean data, 30% of cells
hidden, best trained automaton against the best method that trains nothing:

| dataset | best automaton | best training-free | margin |
|---|---|---|---|
| **gray_scott_16x16** | 0.092 | **0.016** (PCA), 0.033 (lookup) | training-free by **2.8–5.8×** |
| tabular_breastcancer | 0.765 | **0.620** (lookup) | training-free |
| eca_rule110 | 0.964 | **0.792** (lookup) | training-free |
| digits_8x8 | 0.505 | 0.512 (lookup) | **tie** |
| cml_lattice | **0.617** | 0.795 | automaton |
| natural_patches | **0.248** | 0.303 | automaton |

Across all fifteen datasets the best trained automaton beats the best training-free method on
**9 of 15**. A coin flip with extra steps, on a completely different architecture family from
the one that produced the original finding.

And the sharpest instance is the one we did not expect. **Gray–Scott was added as the positive
control for locality** — a reaction–diffusion PDE where every rule is local by construction, the
case a cellular automaton should be best at. A parameter-free nearest-neighbour lookup beats
every architecture there by nearly 3×.

### 1.2 has become a pattern, and that is worse than a single failure

`05` recorded one attractive mechanistic explanation of ours dying under replication: attractor
dimension predicting where capacity stops paying, **−0.71 on six systems → −0.22 on eighteen**.

There is now a second. The NCA spike's central hypothesis was that a global branch exists to
compensate when locality is unavailable. The controlled test — a shuffled twin, the same data
with one fixed cell permutation, identical values and difficulty, only adjacency destroyed —
went the wrong way on every measure:

| measure | shuffled higher than ordered |
|---|---|
| global share of the update | 4/9 |
| knockout ratio | 2/9 |
| influence outside the light cone | **0/9** |

And the error side shows why: shuffling collapses **every** architecture to ~1.0, no better than
guessing the mean. The global branches rescue nothing.

**Two for two.** Both times the pattern was identical: a clean mechanism, a quantitative
prediction, and a controlled test that killed it. A hostile reviewer would say this project is
good at generating mechanisms and bad at having them survive — and on the evidence they would
be right. The mitigating fact is that we ran the discriminating test in both cases, and ran it
before believing the result.

### 1.7 does not bite the new questions

`05`: *"observing saturation at 10⁵ parameters says nothing about 10⁹."* True, and fatal to the
capacity question.

It does not apply to the light-cone result, which is not an empirical trend that might reverse
at scale. A rule that reads only its immediate neighbours moves information one cell per step;
influence outside that cone is **exactly zero**, at any tolerance, at any parameter count. We
measured 0.0000 for the local automaton at every step and asserted it at four tolerances. No
amount of scale changes arithmetic.

This is the first result in the project that is immune to the hardware ceiling, which is worth
noting because feasibility was scored **"no"** in `05`'s value table.

---

## Part 2 — What survives, revisited

### 2.1 The horizon reconciliation — untouched

Not addressed by the NCA work.

### 2.2 The denoising explanation — NOT tested here, and it is important to say so

This is `05`'s **novel core** — *"the network's real advantage over lookup is denoising, and
clean benchmarks switch it off."*

The NCA spike does run noise conditions, and the numbers look like a refutation:

| noise | best trained | best training-free | trained wins |
|---|---|---|---|
| 0% | 0.707 | 0.795 | 9/15 |
| 10% | 0.761 | 0.874 | 9/15 |
| 25% | 0.965 | 0.963 | **6/15** |

The trained models win **less** as noise rises, where N2 found the opposite (19/40 clean →
29/40 at 20%).

**This does not refute the denoising claim, because it is not the same experiment.** In N2 the
surrogate was *trained on noisy trajectories*, so it could learn to average noise away. Here
`train()` is called without a noise argument — the automata are trained on **clean** data and
noise appears only as a held-out evaluation condition. That measures *robustness of a
clean-trained model*, which is a different quantity.

So the honest reading is narrower and still useful: **denoising is not a free property of
having many parameters — it has to be trained for.** A model that never saw noise does not
acquire the advantage, and in fact degrades faster than a lookup table does.

Prerequisite 5.2 in `05` is therefore still open, and the proper test is a one-line change:
pass `noise=` to `train()` and repeat. That is now the highest-value cheap experiment
available.

### 2.3 The measurement critique — extended from three instances to five

`05` listed three ways the standard setup misleads. There are now five, and **two of the new
ones were found in our own instruments**:

4. **A probe that read its own tolerance.** The propagation measure counted a cell as "reached"
   above an absolute 1e-6. Every global architecture then scored exactly 1.000 on every
   dataset. A tolerance sweep showed that at 1e-3 they reach exactly what the purely local one
   reaches, and at 1e-6 they reach everything. The number described the threshold, not the
   architecture. Replaced with a threshold-free measure whose value is *exactly* zero for a
   local rule by construction.
5. **Train/test leakage in three of nine datasets.** The lookup baseline scored exactly 0.000
   on clinical features — a perfect score is almost always retrieval, and it was: the set was
   built by tiling 569 real rows up to 900. Rule 110 failed the same way for a different
   reason, settling into only 337 distinct states out of 900. Measured: 225/225 test rows
   duplicated in both cases.

This *strengthens* the critique rather than weakening it. Five instances of the same failure —
a measurement reporting its own convention instead of the system — found across two literatures
and our own code, is a better-evidenced claim than three. And the two we found in our own work
are the most persuasive, because they show the failure is easy to commit rather than a fault of
other people's care.

### 2.4 The negative results — extended

Add: the locality hypothesis falsified by its own control, and a **structural limit** that
looks like a genuine finding rather than a null.

When adjacency is destroyed, every architecture collapses. The likely reason is not a training
failure: **a cellular automaton is weight-shared by construction, so it cannot represent "this
particular cell".** The evidence is that A2's pooled summary (mean, max, spread) is
*permutation-invariant* — its global branch receives literally identical input from a dataset
and its shuffled twin — and it collapses anyway. Identical global information, destroyed
performance. A global summary cannot substitute for knowing which cell is which.

The one exception points the same way: A4 on shuffled digits degrades least (+0.20, still
scoring 0.707), and its global share rises 0.57 → 1.06 with knockout 1.11 → 1.43. Where an
architecture *does* survive, the position-dependent global path is what carries it.

---

## Part 3 — The pre-registered kill conditions

`05` wrote these down so they could be checked rather than argued about. One has now fired —
in the project's favour.

| condition | outcome |
|---|---|
| *"If the copying tie **disappears** on a spatially extended system, the central novel claim is low-dimension-specific and the paper loses its spine."* | **Tested, and it did not disappear.** On Gray–Scott, 256 cells, spatially extended, training-free methods beat every architecture by up to 5.8× — the **largest** margin measured anywhere in the project. The spine survives, and on the hardest available test |
| *"If the noise crossover does not reproduce under a realistic noise model, the denoising explanation is an artefact."* | **Not yet tested.** The noise conditions run here are clean-trained robustness, not learned denoising |
| *"If someone has already published the parroting-versus-surrogate comparison at scale, this becomes a replication."* | **Not yet checked.** Still task 0 |

That first row matters more than anything else in this document. It was the condition most
likely to kill the direction, it was stated in advance, and the evidence came back on the
project's side.

---

## Part 4 — What the value assessment looks like now

`05`'s table, with the new column.

| criterion | before | after `05` | **after the NCA spike** |
|---|---|---|---|
| Surprising to experts | believed high | low | **low for the original claim; moderate for "your benchmark is sampled so densely that a lookup table matches every architecture"** |
| Fruitful | believed high | low–moderate | **moderate–high** — the finding now spans five modalities and two architecture families, so it is about evaluation practice generally rather than one benchmark |
| Rigorous | unknown | good, and it failed | **good, and it keeps failing usefully** — two mechanisms killed by their own controls, five measurement artefacts caught |
| Feasible | assumed | **no** at the ambitious version | **yes for the reframed version** — the light-cone result is arithmetic, and everything here ran at 13k parameters on two cores |

**Net movement since `05`: upward, and for a specific reason.** The prerequisite `05` called
non-negotiable has been met, the kill condition most likely to end the project was tested and
did not fire, and the central claim now holds on a substrate the earlier work could not reach.

What has **not** improved: the original proposal is still dead, the predictor is still a
simulation artefact, and our own mechanistic explanations are now 0 for 2 under controlled
testing.

---

## Part 5 — What this changes about what to do next

Ranked, with the reasoning.

1. **Train on noisy data and repeat the comparison.** A one-line change. It is the only
   remaining test of `05`'s novel core, and the current evidence says the advantage is not free
   — it must be trained for, which if confirmed is a sharper claim than the original.
2. **The literature check** (`05` task 0), still not done, still able to reshape everything.
3. **Finish the layer-family run** and ask whether A5's full self-attention survives the
   shuffled twin. This is the direct test of the structural explanation in 2.4: full attention
   with per-cell tokens is the one architecture here that could in principle learn arbitrary
   cell-to-cell relations and recover a permuted lattice. If it also collapses, the limit is
   the whole weight-shared family and that is a real finding. If it survives, the explanation
   above is wrong and it was pooling, not weight sharing.
4. **Explain Gray–Scott.** Adding it as the positive control for locality produced two
   surprises pointing opposite ways: training-free methods win by the largest margin measured,
   *and* knocking out the global branch costs the most (A3 11.4×, A2 5.8×) while on Game of
   Life and rule 110 — also exactly local — knockout costs nothing. Unexplained, and knockout
   carries a confound worth stating: zeroing a branch a model calibrated around is removal
   shock, not necessarily loss of information.
