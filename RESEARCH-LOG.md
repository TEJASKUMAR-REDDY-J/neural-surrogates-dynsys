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
