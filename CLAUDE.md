# Project rules

## Immutable
`autovoila-main/` is the research contract: `voila.md` (workflow), `research-philosophy.md`
(what counts as a good research question), `draft-format/` (CAISc 2026 LaTeX template).
**Never edit any file under `autovoila-main/`.** Read it, follow it, leave it alone.

## Layout
- `autovoila-main/` — instructions (read-only)
- `all-spikes/<spike-name>/` — one directory per research idea; all artifacts for that
  idea live inside it (code, data gen, results, figures, paper, notes)
- `ENVIRONMENT.md` — hardware/software audit; the feasibility budget for any experiment
- `RESEARCH-LOG.md` — running log of decisions, results, surprises, dead ends

## Spike layout
```
all-spikes/<spike-name>/
  NOTES.md      # hypothesis, decisions, log for this spike
  src/          # experiment code
  configs/      # run configs (yaml/json), one per experiment
  results/      # metrics (csv/json) + figures, committed
  paper/        # tex + compiled pdf, uses draft-format/caisc_2026.sty
```

## Versioning
- Remote: https://github.com/TEJASKUMAR-REDDY-J/neural-surrogates-dynsys.git
- `main` holds setup, environment audit, log, and merged spikes.
- One branch per spike: `spike/<short-name>`. Merge to `main` when the spike concludes
  (positive, negative, or inconclusive — all three get merged and recorded).
- Commit at each meaningful step: data generator works, baseline trains, result lands,
  hypothesis revised. Small commits, factual messages.
- **No AI attribution anywhere**: no `Co-Authored-By: Claude`, no "Generated with Claude
  Code" footers, no `claude/` branch prefixes, in any commit, branch, tag, or PR.
  (AI involvement is disclosed in the paper's AI Involvement Checklist instead — that is
  the required place for it.)

## Experiment discipline
- Every run: fixed seed, config recorded, metrics written to `results/` as a file.
- Multiple seeds before any claim. Baseline before any "our method beats X".
- Report negative results. The philosophy explicitly permits concluding the hypothesis
  is unsupported.
