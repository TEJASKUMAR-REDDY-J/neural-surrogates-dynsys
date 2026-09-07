# Research log

Append-only. Newest entries at the bottom. One entry per meaningful decision, result,
surprise or dead end — this is what makes the work resumable and what feeds the
process appendix of the paper.

Entry format: `## YYYY-MM-DD — <what happened>` then What / Why / Result / Next.

---

## 2026-09-06 — Session 0: setup

**What.** Read the full autovoila contract (`voila.md`, `research-philosophy.md`,
`draft-format/caisc_2026.tex`). Audited hardware and software. Created repo skeleton
and version-control conventions.

**Contract, in brief.**
- Phases: exploration -> question sharpening -> experiment execution -> paper writing.
  Do not start coding at exploration time.
- A good question is surprising to experts, fruitful, rigorous (forecloses alternative
  explanations), and feasible. Narrow claims are safe but inconsequential; the skill is
  walking that line.
- Exploration must produce **several** candidate directions; top 3 get presented to the
  user with background, and the user picks. Do not attach to the first plausible one.
- Rigor means the evidence must be able to carry the claim: multi-seed, ablations,
  baselines, confound analysis. Do not claim "reasoning" from one benchmark.
- Negative or inconclusive outcomes are acceptable results, not failures.
- Paper: CAISc 2026 template, **8 content pages max**; references, checklists and
  appendices are free. Two mandatory checklists (AI Involvement; Reproducibility and
  Responsibility) — omitting either is desk rejection. Human authors only.
- Appendix must carry the process log: user inputs, directions considered, what was
  selected, feedback, major decisions, prompts, verification, critic feedback.

**Result.** `ENVIRONMENT.md` written — CPU-only i3, 2 cores, no GPU. This is the binding
constraint on the whole project and must be used as a filter during exploration, not
discovered later during execution.

**Open questions for the user.** (1) problem statement / topic, (2) time budget,
(3) how to satisfy the independent-critic role given no `codex` CLI, (4) LaTeX toolchain.

**Next.** Await problem statements, then run exploration.

## 2026-09-06 — Session 0: user calibration

**Time budget: 5 days.** Not the 2-hour voila.md default. This changes the plan
materially — it buys a real literature review, several candidate spikes explored before
committing, multi-seed everything, and a genuine ablation grid rather than a single
run per condition. Deadline 2026-09-11.

**Independent critic: user relays to ChatGPT/Gemini and pastes back.** So critique
points must be packaged as self-contained blocks the user can copy out — full context,
the claim, the plan, and the specific question being put to the critic. The critic's
*disagreement* is the useful signal; agreement is not evidence.

**LaTeX:** deferred. Install Tectonic (single binary, no TeX Live) when paper writing
starts; not on the critical path until then.

**Phase plan under a 5-day budget**
| Day | Phase |
|---|---|
| 1 | Problem statement intake, literature review, exploration - generate many candidate claims |
| 2 | Narrow to 3 directions, present to user, sharpen the chosen one, critic pass |
| 3-4 | Experiment execution: baselines first, then the discriminating experiments, multi-seed |
| 5 | Interpretation, reviewer self-check, draft, critic pass, compile PDF |

Slack is deliberate: the philosophy expects evidence to redirect the question, and that
costs time.

**Next.** Awaiting problem statements from the user.

## 2026-09-06 — Session 1: problem statement received; literature review and background

**What.** User supplied `lossfunk_proposal.pdf` ("Predicting Predictability: when can a
neural network skip the simulation?"), a long brief of hypotheses to test rather than
assume, eight starting references, and two explicit asks: (1) a per-paper extraction using
a fixed nine-question template plus relations/challenges/aids, (2) a survey of what forms
data takes and how it reaches models.

**Why.** Per `research-philosophy.md`, the most common and most justified failure is a
reviewer showing the claim was already known. Exploration therefore starts with a
verification pass, not with code.

**Result.**
- `background/01-literature-review.md` (written earlier in the session) — landscape and a
  first verdict on each hypothesis.
- `background/02-paper-extractions.md` — 19 papers extracted against the user's template:
  the 8 provided plus 11 located this session. Full text read for 7 of them; the rest are
  abstract-plus-metadata and each entry says which. Includes a relation map (6 clusters), a
  challenges table (10 threats, severity-ranked) and an aids table (10 assets).
- `background/03-data-representations.md` — data forms, the five-stage pipeline, per-form
  model interfaces, transformation methods by category, representation-to-inductive-bias
  mapping, eight ways transformations corrupt a learnability measurement, and a
  pre-registration checklist.

**Verification.** Every arXiv identifier in `01` was checked this session and resolves to a
real paper with the claimed content. No fabricated citations found.

**Surprises / things that changed the picture.**
1. **Gilpin's decorrelation is horizon-dependent.** A weak correlation with λ_max does hold
   below one Lyapunov time and degrades above it. His capacity proxy is training walltime at
   ρ = −0.31 ± 0.04, and his data sweep is aggregated across systems, not per-system. So
   "scale and data are the limit" is his *conclusion*, not a controlled measurement — which
   strengthens the case that the discriminating experiment is unrun.
2. **Shikhman (750 models, TMLR) trains on 128–512 samples.** The published "in-distribution
   accuracy does not predict robustness" result is partly confounded with a small-data
   regime — the same confound that threatens us.
3. **Cluster 3 is the most under-exploited pattern found.** Five papers from four groups
   (Gilpin 2023; Zhang & Gilpin ICLR 2025; context parroting; NNPT ergodic smoothing; REALM
   finding iii) all report that pointwise skill and distributional skill decouple. Nobody
   has made the decoupling itself the object of study.
4. **Context parroting is a design-level threat.** A parameter-free copy baseline beats
   foundation models on chaotic systems. Any leap-horizon claim without a parroting baseline
   is unfounded. This is now non-negotiable in the design.
5. **The TRM audit (2512.11847) reframes the specialisation argument.** ~11 pp of Pass@1
   comes from 1000-sample test-time voting; zero accuracy without the puzzle ID; recursion
   saturates at the first step. The honest claim is about where compute is spent, not about
   parameter efficiency.
6. **Neither community measures the other's variables.** The chaos-forecasting literature
   uses only dynamical invariants; the PDE-surrogate benchmarks (REALM) use only
   dimensionality, stiffness and mesh irregularity. Putting both in one regression is
   unoccupied ground.
7. **P5 gives a sharper hypothesis than the proposal has.** Resolution-limited scaling
   predicts exponent α ≈ 1/d with d the input-manifold dimension. For a dynamical system
   that d is the attractor's fractal dimension — a classical invariant. Gilpin checked
   invariants against *accuracy*; nobody has checked them against the *exponent*.

**Standing verdict on the proposal's claims.** Three of the user's listed hypotheses are
already established in the literature (learnability ≠ chaoticity; short-horizon accuracy
does not imply long-horizon stability; coarse-graining rescues irreducible systems), one is
probably ill-posed as stated (architecture-independent learnability — V-information says
learnability is family-relative), and the meta-model idea has a high failure base rate
(NAS-Bench-Suite-Zero). What survives is an adjudication-shaped question about *why*
surrogates fail at long horizons, where Gilpin, Duraisamy and Chen et al. give three
mutually incompatible answers and none ran the controlled per-system capacity × data sweep.

**Next.** Convert this into candidate research directions, evaluate against the four
criteria, and present the top three to the user for selection. Do not start experiments
before that choice is made.

## 2026-09-07 — Session 2: P1–P4 follow-ups; replication battery planned

**What.** User asked four targeted questions on P1–P4, set an append-only constraint on
`background/02-paper-extractions.md`, and asked for an end-to-end test of the experiments in
the corpus, delivered as a separate document with a clean structure for runs and logs.

**Append-only rule.** Recorded in `CLAUDE.md`. `02-paper-extractions.md` grew from 993 to
1392 lines by appending Parts F and G; all eight P entries verified intact after the append.

**New sources (A12–A18).** Seven, all verified this session except A16 (partial — journal
page 403; flagged in the doc).

**Findings that change the picture.**
1. **A13, Dzwinel & Magiera 2015.** Fixed exactly the limitation flagged in P1: a cheaper
   coarse-graining algorithm reaches N=7, where the irreducible ECA set converges to four
   rules — {30, 45, 106, 154} — which is precisely Wuensche's set of strong chain-rules *and*
   the complete set of strong surjective (highly irreversible) automata. So the boundary of
   reducibility is picked out by an **algebraic** property, information destruction, not a
   dynamical one. Also: the positive class in any ECA learnability study is 4/256.
2. **A12, the 2006 PRE extension.** The probability of finding a coarse-grained description
   **approaches unity at coarser scales**, and large-scale update rules have low Kolmogorov
   complexity obeying a scaling law. Closure is therefore *scale-indexed*, not binary — which
   reframes the question as "at what scale" and stakes out that framing in 2006.
3. **A15/A16 — the learnability translation already exists.** Garland, James & Bradley (PRE
   2014) show weighted permutation entropy tracks achievable forecast error across methods
   (logarithmic trend, explicitly not a bound, proposed as a *method–data mismatch* diagnostic:
   WPE says some method could do better, not which). Pennekamp 2019 replicates across 461
   ecological series and gives the intrinsic-vs-realised predictability distinction. Neither
   covers neural surrogates, rollout, or architecture selection.
4. **A17 is a serious novelty threat.** *Exploring Accuracy Law for Deep Time Series
   Forecasters* (2510.02729): over **4,700 trained models**, an empirical law relating the
   **minimum attainable error** to **window-wise pattern complexity**, used to identify
   saturated benchmarks. That is "predict the error floor from a pre-training statistic", at a
   scale we cannot match, published within the year. It is univariate and does not touch
   rollout or architecture choice — but any direction we pick must state in one sentence how
   it differs. **Read in full before the direction is chosen.**
5. **A14, the Chaos Decision Tree Algorithm** gives a noise-robust, published pipeline that
   decides stochastic / periodic / chaotic before any chaos machinery is applied, and a
   degree-of-chaos scalar that does not require estimating λ.

**Positions taken on the user's four questions.**
- *P1:* limitations were compute-bound search and an undeclared projection space; both now
  addressed in the literature, and the fix makes the CA substrate look worse for us, not
  better (4/256 base rate, scale-indexed closure).
- *P2:* λ is not "wrong for assuming determinism" — it is asymptotic and infinitesimal, and
  separately its *estimator* needs determinism. Better instruments exist and are cheap: FSLE,
  ε-entropy, weighted permutation entropy, the modified 0-1 test, CDTA. WPE now enters our
  design as a **baseline to beat**, not a feature.
- *P3:* agreed with the user that small parameter counts are an advantage here — NNPT's regime
  is our hardware's regime. But the real weakness is not model size, it is that the
  non-monotonicity rests on a **two-point depth grid** with no reported seeds at a single 1%
  tolerance. The upgrade is a near-continuous width axis, seeds at every point, and a
  **threshold-sensitivity check across four tolerances** — the cheapest high-value test in the
  battery. Also: "required capacity at tolerance" and "achievable error floor" are different
  quantities and may disagree.
- *P4:* user's critique is right and generalises. Task2Vec's score is **additive** in task and
  model; Model2Vec's per-model bias is a rank-one correction. Neither can express an
  interaction — and the interaction is the entire phenomenon in our domain (DeepONet worst on
  regular meshes, best on every irregular one; CNO 1.05× on Poisson, 12.2× on Black–Scholes).
  Their own limitations section concedes it. Consequence: any spike here must produce a
  task × architecture outcome **matrix**, report the rank-1 baseline, and test whether
  pre-training statistics predict the residual. This is the same objection as V-information's
  family-relativity, in different vocabulary.

**Result.** `all-spikes/00-replications/` created: `PLAN.md` (R0–R8, each with claim under
test, design, dependent variables, cost on this machine, and pre-registered kill criteria),
`NOTES.md`, directory scaffold, and `src/common/runlog.py` (run identity + JSONL manifest
logging, self-check passes). Total estimated compute ~15–20 h.

**Pre-registered kill criteria now on record.** R1: if the reducible fraction moves with the
width of the projection search space, CA-based learnability classification is dead. R3: if
WPE + λ explain ≥85% of surrogate-error variance across systems, the learned-predictor
direction is not worth pursuing. R7 runs *before* direction selection, because if context
parroting matches our trained models then every leap-horizon claim downstream is unfounded.

**Nothing has been run yet.** R0 is the first task and blocks everything except R1.

**Next.** Await user go-ahead to execute the battery, starting with R0 → R2 → R7.
