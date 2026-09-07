# 00-replications — notes

Running log for the replication battery. Newest at the bottom.
Plan: [PLAN.md](PLAN.md). Results as they land: `RESULTS.md` (not yet created).

This spike produces **no novel claim**. Its output is a calibration report: for each
measurement the literature offers, what it costs on this hardware, what it actually measures,
how much it moves across seeds, and whether it is worth building a research question on.

---

## 2026-09-07 — spike created

**What.** Scaffolded the directory layout, logging schema and shared run-identity utility.
Wrote `PLAN.md`: nine experiments (R0–R8), each with a claim under test, a design, dependent
variables, a cost estimate on this machine, and — where applicable — a pre-registered kill
criterion.

**Why these nine.** Each targets a measurement the project would otherwise depend on
unexamined:

| | Tests | Kills if |
|---|---|---|
| R1 | is "admits a coarse description" a property or a search artefact | reducible fraction moves with search-space width |
| R2 | how many independent axes the system-property feature space really has | instruments are redundant (1–2 effective dims) |
| R3 | does weighted permutation entropy already explain surrogate error | WPE + λ explain ≥ 85% of variance |
| R4 | does skill saturate per system; is α ≈ 1/d_KY | no saturation ⇒ Gilpin is right, leap-horizon framing dies |
| R5 | does NNPT's non-monotone capacity survive seeds and a finer grid | effect appears only at one tolerance |
| R6 | when pointwise and distributional skill diverge | (free off R4; no kill, pure measurement) |
| R7 | how much of "learning" is lookup | parroting matches trained models |
| R8 | direct vs autoregressive, and where the crossover is | — |

**Key design decision.** R7 (context parroting baseline) runs *before* the spike is chosen,
not after. If a parameter-free copy baseline matches our trained surrogates, every
leap-horizon claim downstream is unfounded, and it is cheaper to learn that now.

**Logging.** One `logs/<run_id>.jsonl` per run: manifest (config, config hash, git sha,
environment) → one line per unit of work → summary. `run_id` carries into every results row
and figure caption, so any number in the write-up traces to a config and a commit. Reruns
never overwrite; a changed config changes the hash and therefore the run_id.
`src/common/runlog.py` has an `assert`-based self-check under `__main__` — passes.

**Not yet done.** Nothing has been run. `src/common/{systems,metrics,models,train,scaling}.py`
are unwritten; R0 is the first task and is blocking for everything except R1.

**Next.** R0 (dependency check + metrics library validated against published values on
Lorenz / logistic / white noise), then R2 and R7, which are near-free.
