# Handoff — read this first

Paste the block in **Part 0** as your first message after a compact. Everything below it is the
context that prompt refers to.

---

## Part 0 — The prompt to paste

```
Read HANDOFF.md in the project root, then CLAUDE.md, then
all-spikes/00-replications/RESULTS.md. Those three bring you up to date on a research
project that is mid-flight.

Work exactly the way the previous session did. In priority order:

1. EXPLAIN IN PLAIN LANGUAGE, AS A STORY. What we wanted to know, what we did, what
   happened, what it means. Concrete example before the general claim. Use tables and
   figures wherever they help. This applies to chat replies AND to written documents like
   RESULTS.md, METHODS.md and PLAN.md. Simplify the language, never the content - keep
   every number and every caveat.

2. NEVER DELETE OR RENUMBER background/02-paper-extractions.md. It is append-only. New
   material goes in a new Part at the end. Corrections are added as new text that names
   what they correct. I cite its P1-P8 / A1-A18 numbering outside this repo.

3. CHECK YOUR OWN WORK BEFORE REPORTING IT. Seven times in this project a striking result
   turned out to be an artefact, and four of those were ours. Before reporting any
   correlation or effect: how many samples, would it survive leave-one-out, is there a
   confound that produces it for free, does a shuffle control beat it. Report the check,
   not just the number.

4. WHEN A RESULT IS CORRECTED, LEAVE THE ORIGINAL VISIBLE and attach the correction. Do
   not silently edit history. The gap between what we expected and what we found is
   information.

5. NEVER PUT AI ATTRIBUTION in commits, branches, or PRs. No Co-Authored-By, no
   "Generated with". Disclosure belongs in the paper's AI Involvement Checklist.

6. Commit at every meaningful step with factual messages. Push to
   https://github.com/TEJASKUMAR-REDDY-J/neural-surrogates-dynsys.git

Do not re-run anything already done - results are in all-spikes/00-replications/results/
and must be preserved. Tell me what you plan to do before starting long compute.
```

---

## Part 1 — What this project is

A research project following the **autovoila** contract in `autovoila-main/` (read-only:
`voila.md` the workflow, `research-philosophy.md` what counts as a good question,
`draft-format/` the CAISc 2026 LaTeX template).

**Original question, from `proposal/lossfunk_proposal.pdf`:** can we tell in advance, from the
data, which model architectures will work on a problem — and specifically, how far a neural
surrogate can "leap" ahead before it breaks?

**Hardware:** 2-core i3 laptop, no GPU. This is a hard constraint and it shaped everything.
`ENVIRONMENT.md` has the audit.

**Phase:** exploration is finished. A twelve-experiment replication battery has been run and
written up. The next decision is which direction to commit to.

---

## Part 2 — Where things stand, in one screen

| | |
|---|---|
| experiments run | **12** (three rerun after finding flaws in our own design) |
| neural networks trained | ~2,900 |
| logged results | ~6,550 across 46 run logs |
| figures | 17 |
| compute | ~20 hours, 2 cores, no GPU |
| everything committed and pushed | yes |

**The findings, shortest form:**

- Complexity statistics are **~5–6 independent axes**, not one number in disguise.
- Those statistics, computed the standard way, **read the sampling rate rather than the
  system** — the correction reorders systems, it is not a bias you can subtract.
- Our most exciting result (a cheap statistic predicting surrogate skill, R²=0.59) was **an
  artefact of seven systems carrying a clock coordinate**. Clean: 0.0001.
- Redone properly on 108 systems: **~32% of surrogate skill is predictable**, best from spectral
  entropy plus correlation dimension. The Lyapunov exponent contributes nothing.
- **That predictability collapses to zero under 1% observational noise.**
- Capacity helps a lot at short horizons and **barely at all far out** (0.94 → 0.13 across 18
  systems). Our explanation for *which* systems stop early did **not replicate** (−0.71 on 6
  systems, −0.22 on 18).
- A model that **cannot** accumulate error fails **earlier** than one that does, so the
  long-horizon wall is an **information limit**, not error accumulation.
- Bigger models improve next-step accuracy (+0.84) and **structure not at all** (+0.06).
- **A trained network ties with fifteen lines of copy-the-most-similar-past-moment**: 46 of 108
  systems, median ratio exactly 1.00. At 20% noise it wins 29/40 — because its real advantage
  is **denoising**, which clean benchmarks switch off. *This is the most novel thing here.*
- Iterative refinement helps for ~3 passes then **reliably degrades**, never beyond its
  training budget.
- The published "more chaos needs a smaller model" claim **does not reproduce**.

---

## Part 3 — Where to find things

```
autovoila-main/            the research contract. READ-ONLY, never edit
proposal/                  the original problem statement
ENVIRONMENT.md             hardware audit = the feasibility budget
RESEARCH-LOG.md            decisions, results, surprises, in order
CLAUDE.md                  project rules (append-only doc, no AI attribution, layout)
HANDOFF.md                 this file

background/
  01-literature-review.md              the landscape, and a verdict per hypothesis
  02-paper-extractions.md              19 papers on a 9-question template. APPEND-ONLY
  03-data-representations.md           how data reaches models; why representation is a
                                       free variable that moves what we measure
  04-design-questions-and-next-experiments.md   is dysts enough, batch size, R0 failures
  05-devils-advocate-and-reframing.md  ** READ THIS ** the honest assessment of what the
                                       project is now worth and what to do about it

all-spikes/00-replications/
  PLAN.md      the plan as written, plus an outcome tracker for what each became
  RESULTS.md   ** the main document ** — every experiment as a story, 16 figures
  METHODS.md   how the machinery works, in plain language, assuming no background
  NOTES.md     running log, including the full record of the failed KS attempt
  src/common/  shared library, every module with a self-check that passes
  src/r*/ src/n*/   one directory per experiment
  results/     raw CSVs, archived passes (pass1, pass2_r4b, pass3_r4c), and analysis/
  logs/        one JSONL per run: config, config hash, git sha, one line per unit of work
```

**Reproducing any number:** every result row carries a `run_id` that traces to a config hash
and a git commit in `logs/`. Analysis is separate from experiments — `python -m
src.analysis.run` and `python -m src.analysis.figures` regenerate everything from the CSVs
without retraining.

---

## Part 4 — What to do next

`background/05-devils-advocate-and-reframing.md` argues this at length. The short version:

**The original proposal cannot be delivered.** Its core claims are 20-year-old results, and the
one deliverable-shaped outcome — a predictor of surrogate performance — is a simulation
artefact that dies at 1% noise.

**The recommended reframe** is a critique-plus-remedy of how this field evaluates surrogates,
built on five demonstrated failures of the standard setup, with the denoising explanation as
the novel core.

**Three things must happen before that is worth writing:**

1. **One spatially extended system.** Everything we tested has 3–16 variables; the literature we
   criticise is about grids with thousands. Kuramoto–Sivashinsky on 64 points is the standard
   cheap bridge. **Our attempt failed** — see NOTES.md for the full list of what was tried. The
   canonical published parameters diverge the same way, so the fault is ours; take a validated
   implementation rather than debugging further. Then re-run four checks on it: the copying tie,
   the capacity-versus-horizon pattern, the noise crossover, and direct-versus-rollout.
2. **A proper test of the denoising claim** — finer noise ladder, a realistic (time-correlated)
   noise model, and a directly denoised copying baseline to test the mechanism.
3. **Rerun headline numbers at batch 128**, since N5 showed our recipe sits ~1.7× above
   achievable error. Relative comparisons are unaffected; absolute numbers are not.

**Pre-registered kill conditions for the reframe**, so they can be checked rather than argued:

- If the copying tie disappears on a spatially extended system, the central claim is
  low-dimension-specific and the spine is gone.
- If the noise crossover does not reproduce under realistic noise, the denoising explanation is
  an artefact of independent Gaussian noise.

---

## Part 5 — Things that will bite you

Learned the hard way. Each one cost time.

- **`dysts`' own integrator is unusably slow** (41 s for 2,000 Lorenz points). We take its
  equations and published invariants but integrate ourselves. Do not undo this.
- **`solve_ivp` cannot be interrupted.** There is a deadline check inside the RHS. Keep it.
- **About 20% of `dysts` systems carry an unbounded clock coordinate** (`unbounded_indices`).
  They break ordinal statistics *and* make prediction an extrapolation problem. Always filter
  with `--bounded-only` for cross-system work, or handle the clock explicitly.
- **Ordinal statistics need a delay correction.** At 100 points per oscillation they measure
  the sampling rate. `metrics.autocorrelation_time` exists for this.
- **R3 draws its features from R2.** Scaling R3 to more systems means running R2 on them first.
  We silently ran on 33 systems instead of 108 by forgetting this.
- **A rollout can explode while staying finite.** The divergence check now tests boundedness
  against the system's own scale. 42 of 573 fits were affected before it was fixed.
- **Two single-threaded workers beat one two-threaded process** by about 1.7×. Threading buys
  nothing at these model sizes. Every training script takes `--shard`/`--n-shards`.
- **`tail -f` monitors do not die when the task stops.** Kill them explicitly or they
  accumulate.
- **Verify that a patch applied.** A `str.replace` that does not match is a silent no-op. Assert
  the anchor exists. We shipped a broken run because of this.
- **Log-then-write.** One experiment crashed before writing its CSV and two hours looked lost;
  the JSONL log had every row. `src/r1_coarse_grain/recover.py` rebuilds from logs.

---

## Part 6 — What was promised and not delivered

Say this plainly if asked, rather than letting it slide:

- **Kuramoto–Sivashinsky (N1).** The highest-value planned addition. Attempted, integrator
  unstable, parked with a full record. Nothing depending on it has been claimed. **This is the
  largest gap in the work.**
- **A written research proposal** for the reframed direction. Offered, not yet requested.
