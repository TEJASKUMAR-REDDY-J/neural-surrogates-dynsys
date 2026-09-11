"""Assemble RESULTS/ - a clean, story-ordered copy of everything worth presenting.

Copies rather than moves. The originals stay where the code that produced them expects
them, so nothing in any spike breaks and every result stays reproducible from its own
directory. RESULTS/ is a presentation layer, not a relocation.

Figures are renumbered into the order the story is told, so the directory can be walked
top to bottom when building slides.

    python build_results.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "RESULTS"
SPIKES = ROOT / "all-spikes"

# (destination name, source path, one-line description)
FIGURES = [
    # --- the headline -------------------------------------------------------------------
    ("01_data_vs_architecture.png",
     "04-data-zoo/figures/zoo_1_spread.png",
     "Every method on every dataset. Vertical spread is architecture, horizontal is data. "
     "Method choice buys 0.038 R2; the data spans 1.232. A factor of 33."),
    ("02_predicting_the_ceiling.png",
     "04-data-zoo/figures/zoo_2_properties.png",
     "How well ANY method can do, predicted from properties measured before training. "
     "Noise floor rho -0.87."),
    ("03_model_selector.png",
     "04-data-zoo/figures/zoo_4_selector.png",
     "A model-family selector against fixed strategies, including the leave-one-DOMAIN-out "
     "test. Regret 0.038 against 0.063 for the best fixed choice."),
    ("04_which_family_wins_where.png",
     "04-data-zoo/figures/zoo_3_selection.png",
     "The data properties that separate the three winning regimes."),

    # --- what failure looks like ----------------------------------------------------------
    ("05_what_failure_looks_like.png",
     "03-relevance-dilution/figures/4_actuals.png",
     "Free-running predictions against truth. Identical dynamics in all three panels; only "
     "the surrounding junk differs. Clean tracks 300 steps, both diluted models collapse."),
    ("06_irrelevant_columns.png",
     "03-relevance-dilution/figures/1_dilution.png",
     "Adding columns that carry no information. Lookup collapses 34x, the network mostly "
     "does not - except under non-stationary junk, where it loses."),
    ("07_why_nonstationarity_is_worst.png",
     "03-relevance-dilution/figures/2_why_drift.png",
     "Generalisation gap of 38,415, and the train/test distribution shift that causes it."),
    ("08_lag_and_delivery.png",
     "03-relevance-dilution/figures/3_lag.png",
     "A time-shifted signal survives; a frozen one does not. History rescues variable "
     "staleness, not uniform delay."),
    ("09_lag_actuals.png",
     "03-relevance-dilution/figures/5_lag_actuals.png",
     "Predictions under three delivery schemes, same dynamics and same model."),

    # --- architecture mechanism -----------------------------------------------------------
    ("10_locality_is_a_hard_limit.png",
     "01-nca-multimodal/results/figures/probe_1_propagation.png",
     "Poke one cell and watch the ripple. A local rule moves information one cell per step; "
     "influence outside that cone is exactly zero at any parameter count."),
    ("11_global_paths_often_inert.png",
     "01-nca-multimodal/results/figures/hybrid_1_attribution.png",
     "20 local x global combinations with error attributed to each path. 34% of global "
     "paths change nothing when silenced."),
    ("12_shuffled_twin_control.png",
     "01-nca-multimodal/results/figures/probe_4_shuffled_twins.png",
     "Destroy adjacency, keep everything else. All eight architectures collapse; the "
     "pre-registered prediction failed 9 times out of 9."),
    ("13_hybrid_by_dataset.png",
     "01-nca-multimodal/results/figures/hybrid_2_by_dataset.png",
     "Where the global path earns its keep, per dataset."),

    # --- the earlier replication battery ---------------------------------------------------
    ("14_capacity_vs_horizon.png",
     "00-replications/results/analysis/r4c_horizon_effect.png",
     "Extra capacity buys short-horizon accuracy (+0.94) and almost nothing at long range "
     "(+0.13), across 18 systems."),
    ("15_direct_jump_vs_stepping.png",
     "00-replications/results/analysis/r8_direct_vs_rollout.png",
     "t to t+64 in one shot against 64 small steps. The one-shot model fails EARLIER, so "
     "the long-horizon wall is informational, not error accumulation."),
    ("16_coverage_explains_the_tie.png",
     "00-replications/results/analysis/n6_coverage.png",
     "Why a trained network only ties with lookup: the benchmark is densely sampled. "
     "Coverage moves the baseline's skill, not the network's."),
    ("17_noise_kills_predictability.png",
     "00-replications/results/analysis/n2_noise.png",
     "Surrogate skill degrades gracefully under measurement noise; the ability to PREDICT "
     "that skill collapses at 1%."),
    ("18_pointwise_vs_structural.png",
     "00-replications/results/analysis/r6_pointwise_vs_structural.png",
     "Capacity improves next-step accuracy (+0.84) and attractor structure not at all (+0.06)."),
    ("19_example_trajectories.png",
     "00-replications/results/analysis/trajectories_attractors.png",
     "What a surrogate actually draws, against the real attractor."),
]

ARCH_DIAGRAMS = [
    (f"arch_{n}.png", f"01-nca-multimodal/results/figures/arch_{n}.png")
    for n in ("00_why_global", "A1_local", "A2_pooled", "A3_attentive", "A4_interleaved",
              "A5_transformer", "A6_graph", "A7_recurrent", "A8_spectral")
]

TABLES = [
    ("zoo_r2_by_dataset_and_method.csv",
     "04-data-zoo/results/r2_by_dataset_method.csv",
     "R2 for all 8 methods on all 47 datasets. The source of the headline number."),
    ("zoo_property_correlations.csv",
     "04-data-zoo/results/property_correlations.csv",
     "Each data property against the achievable ceiling."),
    ("zoo_selector_per_dataset.csv",
     "04-data-zoo/results/selector_per_dataset.csv",
     "What the selector recommended per dataset, and its regret - including the six cases "
     "where it loses."),
    ("zoo_raw_shard0.csv", "04-data-zoo/results/zoo__shard0.csv",
     "Raw zoo results, shard 0: every dataset x method x seed with data properties."),
    ("zoo_raw_shard1.csv", "04-data-zoo/results/zoo__shard1.csv",
     "Raw zoo results, shard 1."),
    ("dilution_shard0.csv", "03-relevance-dilution/results/dilution__shard0.csv",
     "Relevance dilution: adding irrelevant columns of four kinds."),
    ("dilution_shard1.csv", "03-relevance-dilution/results/dilution__shard1.csv",
     "Relevance dilution, shard 1."),
    ("lag_delivery.csv", "03-relevance-dilution/results/lag.csv",
     "Four delivery schemes across eight lags, with and without a history window."),
    ("hybrid_layer_grid_shard0.csv", "01-nca-multimodal/results/hybrid__shard0.csv",
     "20 local x global combinations with per-path error attribution."),
    ("hybrid_layer_grid_shard1.csv", "01-nca-multimodal/results/hybrid__shard1.csv",
     "Hybrid grid, shard 1."),
    ("nca_probe_summary.csv", "01-nca-multimodal/results/probe_summary.csv",
     "Mechanism probes per architecture: light-cone leak, global share, knockout, memory."),
    ("controlled_coverage_shard0.csv",
     "00-replications/results/n7/coverage_sweep__shard0.csv",
     "Controlled coverage: system fixed, training budget varied."),
    ("controlled_coverage_shard1.csv",
     "00-replications/results/n7/coverage_sweep__shard1.csv",
     "Controlled coverage, shard 1."),
    ("shuffled_twins.csv", "01-nca-multimodal/results/shuffled_twins.csv",
     "The locality control: same data, one fixed permutation."),
]


def copy(src_rel: str, dest: Path) -> bool:
    src = SPIKES / src_rel
    if not src.exists():
        print(f"  MISSING  {src_rel}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "figures" / "architectures").mkdir(parents=True)
    (OUT / "tables").mkdir(parents=True)

    ok_f = [(n, s, d) for n, s, d in FIGURES if copy(s, OUT / "figures" / n)]
    ok_a = [n for n, s in ARCH_DIAGRAMS if copy(s, OUT / "figures" / "architectures" / n)]
    ok_t = [(n, s, d) for n, s, d in TABLES if copy(s, OUT / "tables" / n)]

    shutil.copy2(ROOT / "RESULTS-SUMMARY.md", OUT / "SUMMARY.md")

    # The two slide figures are generated straight into RESULTS/figures/ rather than copied
    # from a spike, so they must be regenerated AFTER the rmtree above or a rebuild would
    # silently delete them.
    import make_slide_figures
    make_slide_figures.OUT.mkdir(parents=True, exist_ok=True)
    make_slide_figures.fig_locality()
    make_slide_figures.fig_stepwise()

    lines = [
        "# RESULTS",
        "",
        "A story-ordered copy of everything worth presenting. **These are copies** - every",
        "original stays where the code that produced it expects it, so each spike remains",
        "reproducible on its own. Rebuild this directory with `python build_results.py`.",
        "",
        "Start with [SUMMARY.md](SUMMARY.md): executive summary, hypotheses against evidence,",
        "scope limits, and future directions.",
        "",
        "## The four numbers",
        "",
        "| number | meaning |",
        "|---|---|",
        "| **33x** | the data matters 33x more than the choice of method (0.038 R2 against 1.232) |",
        "| **rho -0.87** | how well the achievable ceiling is predicted before any training |",
        "| **18 vs 16** | datasets where a neural net ties the best method, against where it falls below -0.5 |",
        "| **896x** | error increase from non-stationary irrelevant columns, which do not lower the ceiling |",
        "",
        "## Figures, in the order the story is told",
        "",
        "| # | file | what it shows |",
        "|---|---|---|",
    ]
    for n, _, d in ok_f:
        lines.append(f"| {n.split('_')[0]} | [{n}](figures/{n}) | {d} |")

    lines += [
        "",
        "### Purpose-built slide figures",
        "",
        "| file | what it shows |",
        "|---|---|",
        "| [slide_locality_vs_global.png](figures/slide_locality_vs_global.png) | Local versus "
        "global pathways. Left: a local rule leaks EXACTLY zero influence beyond one cell, at "
        "any parameter count. Right: that reach only pays when neighbours are meaningless - "
        "1.33x for the position-map pathway, and 1.01 to 1.07 everywhere else. |",
        "| [slide_stepwise_vs_direct.png](figures/slide_stepwise_vs_direct.png) | Stepping "
        "versus jumping straight to t+h. Small steps start 100x better and stay usable to "
        "~230 steps against ~113 for the jump. Past 256 neither forecasts: the jump settles "
        "onto the average while stepping overshoots past it. |",
        "",
        "Regenerate both with `python make_slide_figures.py`.",
        "",
        "### Architecture diagrams",
        "",
        "`figures/architectures/` - all eight cellular-automaton architectures plus one",
        "explaining why a global pathway exists at all. Every label is read off a built model,",
        "so a diagram cannot drift from the code.",
        "",
        f"{', '.join('`' + n + '`' for n in ok_a)}",
        "",
        "## Tables",
        "",
        "| file | contents |",
        "|---|---|",
    ]
    for n, _, d in ok_t:
        lines.append(f"| [{n}](tables/{n}) | {d} |")

    lines += [
        "",
        "## Where these came from",
        "",
        "| directory | what it did |",
        "|---|---|",
        "| `all-spikes/00-replications` | 12-experiment replication battery on 108 chaotic systems |",
        "| `all-spikes/01-nca-multimodal` | 8 cellular-automaton architectures on 15 datasets, 5 modalities |",
        "| `all-spikes/02-dataset-diagnostic` | the standalone no-training diagnostic |",
        "| `all-spikes/03-relevance-dilution` | irrelevant columns, and lag/delivery schemes |",
        "| `all-spikes/04-data-zoo` | 47 datasets, 13 domains, 8 methods - the headline study |",
        "",
        "Background reading, verdicts per hypothesis and the paper-extraction corpus are in",
        "`background/`. The running decision log is `RESEARCH-LOG.md`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    n_png = len(list((OUT / "figures").rglob("*.png")))
    n_csv = len(list((OUT / "tables").glob("*.csv")))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\nRESULTS/  {n_png} figures, {n_csv} tables, {size:.1f} MB")


if __name__ == "__main__":
    main()
