# RESULTS

A story-ordered copy of everything worth presenting. **These are copies** - every
original stays where the code that produced it expects it, so each spike remains
reproducible on its own. Rebuild this directory with `python build_results.py`.

Start with [SUMMARY.md](SUMMARY.md): executive summary, hypotheses against evidence,
scope limits, and future directions.

## The four numbers

| number | meaning |
|---|---|
| **33x** | the data matters 33x more than the choice of method (0.038 R2 against 1.232) |
| **rho -0.87** | how well the achievable ceiling is predicted before any training |
| **18 vs 16** | datasets where a neural net ties the best method, against where it falls below -0.5 |
| **896x** | error increase from non-stationary irrelevant columns, which do not lower the ceiling |

## Figures, in the order the story is told

| # | file | what it shows |
|---|---|---|
| 01 | [01_data_vs_architecture.png](figures/01_data_vs_architecture.png) | Every method on every dataset. Vertical spread is architecture, horizontal is data. Method choice buys 0.038 R2; the data spans 1.232. A factor of 33. |
| 02 | [02_predicting_the_ceiling.png](figures/02_predicting_the_ceiling.png) | How well ANY method can do, predicted from properties measured before training. Noise floor rho -0.87. |
| 03 | [03_model_selector.png](figures/03_model_selector.png) | A model-family selector against fixed strategies, including the leave-one-DOMAIN-out test. Regret 0.038 against 0.063 for the best fixed choice. |
| 04 | [04_which_family_wins_where.png](figures/04_which_family_wins_where.png) | The data properties that separate the three winning regimes. |
| 05 | [05_what_failure_looks_like.png](figures/05_what_failure_looks_like.png) | Free-running predictions against truth. Identical dynamics in all three panels; only the surrounding junk differs. Clean tracks 300 steps, both diluted models collapse. |
| 06 | [06_irrelevant_columns.png](figures/06_irrelevant_columns.png) | Adding columns that carry no information. Lookup collapses 34x, the network mostly does not - except under non-stationary junk, where it loses. |
| 07 | [07_why_nonstationarity_is_worst.png](figures/07_why_nonstationarity_is_worst.png) | Generalisation gap of 38,415, and the train/test distribution shift that causes it. |
| 08 | [08_lag_and_delivery.png](figures/08_lag_and_delivery.png) | A time-shifted signal survives; a frozen one does not. History rescues variable staleness, not uniform delay. |
| 09 | [09_lag_actuals.png](figures/09_lag_actuals.png) | Predictions under three delivery schemes, same dynamics and same model. |
| 10 | [10_locality_is_a_hard_limit.png](figures/10_locality_is_a_hard_limit.png) | Poke one cell and watch the ripple. A local rule moves information one cell per step; influence outside that cone is exactly zero at any parameter count. |
| 11 | [11_global_paths_often_inert.png](figures/11_global_paths_often_inert.png) | 20 local x global combinations with error attributed to each path. 34% of global paths change nothing when silenced. |
| 12 | [12_shuffled_twin_control.png](figures/12_shuffled_twin_control.png) | Destroy adjacency, keep everything else. All eight architectures collapse; the pre-registered prediction failed 9 times out of 9. |
| 13 | [13_hybrid_by_dataset.png](figures/13_hybrid_by_dataset.png) | Where the global path earns its keep, per dataset. |
| 14 | [14_capacity_vs_horizon.png](figures/14_capacity_vs_horizon.png) | Extra capacity buys short-horizon accuracy (+0.94) and almost nothing at long range (+0.13), across 18 systems. |
| 15 | [15_direct_jump_vs_stepping.png](figures/15_direct_jump_vs_stepping.png) | t to t+64 in one shot against 64 small steps. The one-shot model fails EARLIER, so the long-horizon wall is informational, not error accumulation. |
| 16 | [16_coverage_explains_the_tie.png](figures/16_coverage_explains_the_tie.png) | Why a trained network only ties with lookup: the benchmark is densely sampled. Coverage moves the baseline's skill, not the network's. |
| 17 | [17_noise_kills_predictability.png](figures/17_noise_kills_predictability.png) | Surrogate skill degrades gracefully under measurement noise; the ability to PREDICT that skill collapses at 1%. |
| 18 | [18_pointwise_vs_structural.png](figures/18_pointwise_vs_structural.png) | Capacity improves next-step accuracy (+0.84) and attractor structure not at all (+0.06). |
| 19 | [19_example_trajectories.png](figures/19_example_trajectories.png) | What a surrogate actually draws, against the real attractor. |

### Purpose-built slide figures

| file | what it shows |
|---|---|
| [slide_locality_vs_global.png](figures/slide_locality_vs_global.png) | Local versus global pathways. Left: a local rule leaks EXACTLY zero influence beyond one cell, at any parameter count. Right: that reach only pays when neighbours are meaningless - 1.33x for the position-map pathway, and 1.01 to 1.07 everywhere else. |
| [slide_stepwise_vs_direct.png](figures/slide_stepwise_vs_direct.png) | Stepping versus jumping straight to t+h. Small steps start 100x better and stay usable to ~230 steps against ~113 for the jump. Past 256 neither forecasts: the jump settles onto the average while stepping overshoots past it. |

Regenerate both with `python make_slide_figures.py`.

### Architecture diagrams

`figures/architectures/` - all eight cellular-automaton architectures plus one
explaining why a global pathway exists at all. Every label is read off a built model,
so a diagram cannot drift from the code.

`arch_00_why_global.png`, `arch_A1_local.png`, `arch_A2_pooled.png`, `arch_A3_attentive.png`, `arch_A4_interleaved.png`, `arch_A5_transformer.png`, `arch_A6_graph.png`, `arch_A7_recurrent.png`, `arch_A8_spectral.png`

## Tables

| file | contents |
|---|---|
| [zoo_r2_by_dataset_and_method.csv](tables/zoo_r2_by_dataset_and_method.csv) | R2 for all 8 methods on all 47 datasets. The source of the headline number. |
| [zoo_property_correlations.csv](tables/zoo_property_correlations.csv) | Each data property against the achievable ceiling. |
| [zoo_selector_per_dataset.csv](tables/zoo_selector_per_dataset.csv) | What the selector recommended per dataset, and its regret - including the six cases where it loses. |
| [zoo_raw_shard0.csv](tables/zoo_raw_shard0.csv) | Raw zoo results, shard 0: every dataset x method x seed with data properties. |
| [zoo_raw_shard1.csv](tables/zoo_raw_shard1.csv) | Raw zoo results, shard 1. |
| [dilution_shard0.csv](tables/dilution_shard0.csv) | Relevance dilution: adding irrelevant columns of four kinds. |
| [dilution_shard1.csv](tables/dilution_shard1.csv) | Relevance dilution, shard 1. |
| [lag_delivery.csv](tables/lag_delivery.csv) | Four delivery schemes across eight lags, with and without a history window. |
| [hybrid_layer_grid_shard0.csv](tables/hybrid_layer_grid_shard0.csv) | 20 local x global combinations with per-path error attribution. |
| [hybrid_layer_grid_shard1.csv](tables/hybrid_layer_grid_shard1.csv) | Hybrid grid, shard 1. |
| [nca_probe_summary.csv](tables/nca_probe_summary.csv) | Mechanism probes per architecture: light-cone leak, global share, knockout, memory. |
| [controlled_coverage_shard0.csv](tables/controlled_coverage_shard0.csv) | Controlled coverage: system fixed, training budget varied. |
| [controlled_coverage_shard1.csv](tables/controlled_coverage_shard1.csv) | Controlled coverage, shard 1. |
| [shuffled_twins.csv](tables/shuffled_twins.csv) | The locality control: same data, one fixed permutation. |

## Where these came from

| directory | what it did |
|---|---|
| `all-spikes/00-replications` | 12-experiment replication battery on 108 chaotic systems |
| `all-spikes/01-nca-multimodal` | 8 cellular-automaton architectures on 15 datasets, 5 modalities |
| `all-spikes/02-dataset-diagnostic` | the standalone no-training diagnostic |
| `all-spikes/03-relevance-dilution` | irrelevant columns, and lag/delivery schemes |
| `all-spikes/04-data-zoo` | 47 datasets, 13 domains, 8 methods - the headline study |

Background reading, verdicts per hypothesis and the paper-extraction corpus are in
`background/`. The running decision log is `RESEARCH-LOG.md`.
