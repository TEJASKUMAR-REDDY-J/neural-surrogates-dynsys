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
