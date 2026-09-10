# Results summary

Written 2026-09-10. Everything below is measured in this repository and traceable to a run
log. Scope limits are stated with each claim rather than collected at the end.

---

## Executive summary

We set out to predict, before training, which neural architecture would suit a given
dynamical-systems problem. Across 47 datasets spanning 13 domains and eight method families,
the architecture turned out to be a second-order effect: choosing the best method over the
second-best buys a median of 0.038 R², while the achievable ceiling varies by 1.232 R² across
datasets — a factor of 33. What *is* predictable, and was the original goal in a different
form, is the ceiling itself: an estimated noise floor computed with no training predicts how
well any method will do at rho = -0.87. A model-family selector built on seven such
properties beats every fixed strategy and, critically, holds up when entire domains are held
out (regret 0.038 against 0.063 for the best fixed choice). The most consequential finding
was not hypothesised: neural networks do not degrade gracefully. On 18 of 47 datasets they
land within 0.05 of the best available method; on 16 they fall below R² = -0.5, worse than
predicting the mean. Which side you land on is predictable from rows-per-dimension and
coverage before a single gradient step.

---

## Hypotheses against results

| Initial hypothesis | What the results say | Scale it was tested at |
|---|---|---|
| The horizon a surrogate can reach is a property of the system, measurable before training | **Partly.** The ceiling is predictable (rho -0.87). *Which* property governs the horizon did not replicate: -0.71 on 6 systems became -0.22 on 18 | 18 systems, 573 fits; 47 datasets |
| Learnability is a separate axis from chaoticity | **Yes**, and already known since 2003. Lyapunov exponent carries no information about surrogate skill (+0.03) | 108 systems, 3 seeds |
| Irreducibility can be measured as a continuous quantity | **Yes**, and already known since 2004/2006. Reducibility is scale-indexed: 36% → 59% → 79% across block sizes | 256 rules, 1,698 exhaustive checks |
| A predictor will generalise to unseen families | **Yes**, in the model-selection form. Domain-holdout regret 0.038 vs dataset-holdout 0.040 | 47 datasets, 13 domains held out whole |
| Short-horizon accuracy does not imply long-horizon stability | **Yes.** Capacity buys +0.94 at one step, +0.13 at 500 | 18 systems, 573 fits |
| Coarse-graining rescues irreducible systems | **Yes**, and already known. 202/256 rules reducible at block size 4 | 256 rules, exhaustive |
| Learnability is architecture-independent | **No.** Methods tie where data is dense, diverge sharply where it is sparse | 47 datasets x 8 methods; 15 datasets x 8 architectures |
| A model can jump t to t+64 directly instead of stepping | **Yes, with a limit.** Usable on 5 of 6 systems, one forward pass. But it fails *earlier* than stepping at long range, so the wall is informational | 6 systems, 360 runs, horizons to 512 |
| A meta-model can pick the right architecture | **Partly.** Beats every fixed strategy (p = 0.013) but 60% family accuracy against 33% chance, and the winner flips across seeds on 21 of 47 | 47 datasets, 8 methods, 3 seeds |

### Findings that were not hypothesised

| Finding | Evidence | Scale |
|---|---|---|
| Neural networks tie or fail, rarely in between | 18/47 within 0.05 of best; 16/47 below -0.5; only 13 in between | 47 datasets |
| Non-stationarity breaks the trained model without lowering the ceiling | rho +0.07 (p=0.63) with the ceiling, yet 896x error and a generalisation gap of 38,415 in controlled dilution | 84 controlled cells |
| Irrelevant columns damage retrieval far more than training | Lookup 0.019 → 0.640; network 0.001 → 0.112 | 84 cells, 4 distractor types |
| A frozen signal is worse than a stale one | Uniform lag 32 keeps skill 0.812; refresh-and-hold at 32 collapses to 0.060 | 132 cells |
| Locality is a hard structural limit | Influence outside the light cone measured at exactly 0.0000, at any parameter count | 8 architectures, 15 datasets |
| A third of global mechanisms are inert | 43 of 128 cells within 2% of doing nothing when silenced | 20 combinations, 480 rows |
| Simulated and real data differ sharply | Median best R²: simulated chaos 1.000, sensor 0.316, biomedical ~0.00 | 47 datasets |

---

## Where this will not hold: denser, more complex data

Everything above was measured in a regime that a world model does not live in. The honest
list of what would have to be re-tested, and why each one threatens a specific claim:

| Our regime | What a world model actually sees | Which claim is at risk |
|---|---|---|
| 1 to 60 dimensions | thousands to millions (pixels, tokens, sensor fleets) | The **noise-floor estimator** degrades with dimension, measured: accurate to 4 dims, reads 0.68 at 8, 0.42 at 32 against a true 0.80. Above ~16 dims it is a lower bound, not an estimate. The ceiling prediction is the single most useful result and it is the one most exposed |
| 150 to 42,000 rows | continuous streams, billions of frames | The regime map's boundaries are in *rows per dimension*. At 274 vs 81 the outcome flipped. Those thresholds are almost certainly not the right ones at scale |
| 10^4 to 10^5 parameters | 10^8 to 10^12 | "Architecture is second-order" is measured only where every method is small. A transformer at scale may not sit in the same regime as a 96-unit GRU |
| One prediction task, window 8 | multi-task, multi-horizon, multi-modal, with actions | Nothing here tests transfer, instruction, or interaction. The selector is fitted to one task shape |
| Stationary or mildly drifting | continual shift, regime change, feedback from the agent's own actions | Non-stationarity was our largest measured failure (896x). A world model has *more* of it, and also causes it |
| Fully observed | partial, occluded, asynchronous, missing | The lag experiment touches this; occlusion and missingness are untested |

**Will the current hypotheses survive?** Split them:

- **Likely to survive:** locality as a structural limit (it is arithmetic, not an empirical
  trend); non-stationarity as the dominant failure mode; the tie-or-explode pattern, since
  its cause is coverage and coverage worsens with dimension.
- **Likely to weaken:** the specific thresholds in the regime map; the noise-floor estimator's
  accuracy; "architecture is second-order", which is the claim most likely to invert at scale
  where architecture is the whole game.
- **Untested and unsafe to extrapolate:** anything about multi-task, action-conditioned, or
  representation-learning settings.

---

## Final position

**On the original question.** Predicting which architecture suits a problem is possible only
at the level of *family* and *regime*, not the specific model. That is a weaker result than
proposed, and it is genuinely useful, because the dominant practical decision is not "which
network" but "should this be a network at all", and that one is answerable in seconds.

**On neural surrogates as a field.** The evidence here supports three positions:

1. **Report the training-free baseline.** Across 47 datasets a trivial or classical method won
   38 times. On simulated chaos, the setting most surrogate papers use, a lookup table ties a
   trained network. A surrogate result without an analogue baseline is not interpretable.
2. **Report the data regime alongside the error.** Coverage, noise floor, rows-per-dimension
   and distribution shift cost nothing and determine the outcome more than the architecture
   does. Two papers reporting the same R² on different regimes have not done the same thing.
3. **Stop treating simulated benchmarks as evidence about real deployment.** Median best R²
   was 1.000 on simulated chaos and near zero on real biomedical panels. The gap is larger
   than any architectural difference measured anywhere in this project.

**On scope with world models.** A world model is the adversarial case for everything measured
here: high dimension, non-stationary, partially observed, multi-task, and it influences its
own data distribution. The two results most likely to transfer are the two that are structural
rather than empirical — that locality bounds information flow per step regardless of scale,
and that distribution shift damages a learned model while leaving retrieval intact. The
practical implication is that a world model should be expected to need *retrieval alongside
learning*, not instead of it, and that the interesting engineering question is when to route
a query to which.

---

## What to present

**Visuals, in order.**

| # | Figure | What it shows |
|---|---|---|
| 1 | `04-data-zoo/figures/zoo_1_spread.png` | Every method on every dataset. Vertical spread is architecture, horizontal is data. The 33x |
| 2 | `04-data-zoo/figures/zoo_2_properties.png` | The ceiling predicted from properties measured before training |
| 3 | `04-data-zoo/figures/zoo_4_selector.png` | The selector against fixed strategies, including the domain-holdout test |
| 4 | `03-relevance-dilution/figures/4_actuals.png` | Free-running predictions against truth. Same dynamics, three junk conditions. What failure looks like |
| 5 | `03-relevance-dilution/figures/2_why_drift.png` | Why non-stationary junk is the dangerous kind |
| 6 | `03-relevance-dilution/figures/3_lag.png` | Delivery: a shifted signal survives, a frozen one does not |
| 7 | `01-nca-multimodal/figures/probe_1_propagation.png` | Locality as a hard limit, measured |
| 8 | `01-nca-multimodal/figures/hybrid_1_attribution.png` | A third of global mechanisms do nothing |

**The four numbers to lead with:** 33x, rho -0.87, 18 versus 16, and 896x.

---

## Future directions

Ranked by what would change the conclusions most per unit of effort.

1. **Fix or bound the noise-floor estimator above 16 dimensions.** It is the most useful single
   result and its known failure is measured but unaddressed. Candidate approaches: local
   linear residuals instead of nearest-neighbour disagreement, or a learned density ratio.
2. **Re-run the zoo at one order of magnitude more capacity.** "Architecture is second-order"
   is the claim most likely to invert at scale, and it is the claim a reviewer will attack.
3. **Method-level rather than family-level selection.** The current selector recommends
   "something classical". Recommending "ridge specifically" is the useful output if it survives
   validation.
4. **A non-stationarity benchmark.** We have the sharpest measured result in the project on
   distribution shift and no standard way to report it. A protocol that reports error under
   controlled shift would be adoptable independently of everything else here.
5. **Hold out modalities, not just domains.** The selector survived domain holdout; it has
   never been asked to generalise from tabular to image, or from simulated to real.
6. **The action-conditioned case.** Nothing here tests a model whose predictions influence the
   data it later sees. That is the defining property of a world model and the largest gap
   between this work and that setting.
