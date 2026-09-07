#!/usr/bin/env bash
# Follow-up runs, ordered by value, driven by what the first pass found.
#
# Each one exists because the first pass revealed a flaw in our own design, not because the
# result was inconvenient. Ordered so that if time runs out, the least important is dropped.
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo ""; echo "############ $* ############"; date; echo ""; }

# --------------------------------------------------------------------------------------
# R4b - the crux, rerun so it can actually answer its question.
#
# The first pass was saturated at both ends. Valid prediction time hit the 200-step
# measurement ceiling on 33% of fits, and one-step error bottomed out at 2e-4 regardless of
# capacity, because a 3-D ODE sampled 100 times per oscillation is nearly trivial to step
# forward. Separately, all six systems happened to sit at attractor dimension 2.01-2.35, so
# the alpha ~ 1/d prediction from scaling theory had no leverage.
#
# Three fixes, each verified on a single fit first: sample 5x more coarsely (one-step error
# 0.0002 -> 0.0057, and each step is worth 5x more Lyapunov time), roll out 1000 steps
# instead of 200 (67 Lyapunov times on Lorenz versus 2.7), and pick systems spanning
# Kaplan-Yorke dimension 2.0 to 16.5.
# --------------------------------------------------------------------------------------
log "R4b crux rerun: coarser sampling, long rollout, real spread of attractor dimension"
R4B="Lorenz Rossler Thomas HyperCai HenonHeiles Bouali2"
for sh in 0 1; do
  python -u -m src.r4_capacity_data.run --systems $R4B \
      --pts-per-period 20 --steps 1000 --n-trains 2000 8000 32000 \
      --n-points 45000 --shard $sh --n-shards 2 &
done
wait

# --------------------------------------------------------------------------------------
# R8b - long horizons, and the budget-matching confound measured rather than assumed.
#
# The first pass used h_max=64, which on Lorenz is 0.86 Lyapunov times: rollout was never
# given the chance to fail the way the literature reports. And matching training-pair counts
# handicaps the direct model by construction, since it must represent a whole family of maps
# from the same data. Running it at 1x and 4x pairs separates "more sample-efficient" from
# "works at all".
# --------------------------------------------------------------------------------------
log "R8b long horizons, direct model at 1x and 4x training pairs"
for sh in 0 1; do
  python -u -m src.r8_direct_vs_rollout.run --h-max 512 \
      --horizons 1 2 4 8 16 32 64 128 256 512 \
      --direct-pair-multipliers 1 4 --shard $sh --n-shards 2 &
done
wait

# --------------------------------------------------------------------------------------
# R3b - the predictor question, asked again without the artefact.
#
# The first pass found weighted permutation entropy explaining 59% of the variance in
# surrogate skill. That was entirely an artefact of 7 driven systems whose appended clock
# coordinate is both monotone (killing ordinal entropy) and out of range at test time
# (making prediction extrapolation). Excluding them, R2 collapses to 0.002.
# --------------------------------------------------------------------------------------
log "R3b bounded systems only, 30 of them"
for sh in 0 1; do
  python -u -m src.r3_wpe_vs_error.run --n-systems 30 --bounded-only \
      --seeds 0 1 2 3 4 --shard $sh --n-shards 2 &
done
wait

log "ANALYSIS + FIGURES"
python -u -m src.analysis.run
python -u -m src.analysis.figures

# --------------------------------------------------------------------------------------
# R1 block size 6 - lowest priority, and the reason it is last.
#
# The reducibility trend is already clear from block sizes 2, 3 and 4 (35.9% -> 59.0% ->
# 78.9%). Block size 6 would add one more point at 12 s per rule per 40k projections, so the
# large search budgets are expensive for a confirmatory datum. Capped accordingly.
# --------------------------------------------------------------------------------------
log "R1b block size 6 on the 54 surviving rules"
python -u -m src.r1_coarse_grain.run --block-sizes 2 3 4 --k-values 2 \
    --max-projections 70000 --escalate-sizes 5 6 \
    --escalate-projections 40000 200000

log "ANALYSIS + FIGURES (final)"
python -u -m src.analysis.run
python -u -m src.analysis.figures

log "FOLLOWUPS COMPLETE"
