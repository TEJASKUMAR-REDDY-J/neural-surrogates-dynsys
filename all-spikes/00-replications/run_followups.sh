#!/usr/bin/env bash
# Follow-up runs, driven by what the first pass found.
#
# 1. R3 rerun restricted to systems with no unbounded clock coordinate, with more systems
#    for statistical power. The first pass found WPE's apparent R2 of 0.59 was entirely an
#    artefact of 7 driven systems whose lifted clock coordinate is both monotone (killing
#    ordinal entropy) and out-of-range at test time (making prediction extrapolation).
#    Excluding them, WPE's R2 collapses to 0.002.
# 2. R8 rerun with longer horizons. The first pass used h_max=64, which on Lorenz is only
#    0.86 Lyapunov times - far too short to stress rollout, whose failure is the thing the
#    experiment exists to diagnose.
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo ""; echo "############ $* ############"; date; echo ""; }

log "R3b bounded-only, 30 systems"
python -u -m src.r3_wpe_vs_error.run --n-systems 30 --bounded-only --seeds 0 1 2 3 4 --shard 0 --n-shards 2 &
python -u -m src.r3_wpe_vs_error.run --n-systems 30 --bounded-only --seeds 0 1 2 3 4 --shard 1 --n-shards 2 &
wait

log "R8b long horizons"
python -u -m src.r8_direct_vs_rollout.run --h-max 512 \
    --horizons 1 2 4 8 16 32 64 128 256 512 --direct-pair-multipliers 1 4 --shard 0 --n-shards 2 &
python -u -m src.r8_direct_vs_rollout.run --h-max 512 \
    --horizons 1 2 4 8 16 32 64 128 256 512 --direct-pair-multipliers 1 4 --shard 1 --n-shards 2 &
wait

log "R4b crux rerun: coarser sampling, long rollout, real spread of attractor dimension"
# The first pass could not answer its own question. Both dependent variables were saturated:
# valid prediction time hit the 200-step measurement ceiling on 33% of fits, and one-step
# error bottomed out at 2e-4 regardless of capacity because a 3-D ODE sampled 100 times per
# oscillation is nearly trivial to step forward. And all six systems happened to have almost
# the same attractor dimension (2.01 to 2.35), so the alpha ~ 1/d prediction had no leverage.
#
# Three fixes: sample 5x more coarsely (one-step error 0.0002 -> 0.0057, and each step is
# worth 5x more Lyapunov time), roll out 1000 steps instead of 200 (67 Lyapunov times on
# Lorenz, versus 2.7), and choose systems spanning Kaplan-Yorke dimension 2.0 to 16.5.
R4B_SYSTEMS="Lorenz Rossler Thomas HyperCai HenonHeiles Bouali2"
python -u -m src.r4_capacity_data.run --systems $R4B_SYSTEMS     --pts-per-period 20 --steps 1000 --n-trains 2000 8000 32000     --n-points 45000 --shard 0 --n-shards 2 &
python -u -m src.r4_capacity_data.run --systems $R4B_SYSTEMS     --pts-per-period 20 --steps 1000 --n-trains 2000 8000 32000     --n-points 45000 --shard 1 --n-shards 2 &
wait

log "ANALYSIS + FIGURES"
python -u -m src.analysis.run
python -u -m src.analysis.figures

log "FOLLOWUPS COMPLETE"
