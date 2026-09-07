#!/usr/bin/env bash
# Sequential run of the replication battery. Two cores, so no parallelism: overlapping
# CPU-bound jobs would only make each slower and muddy the timing numbers.
set -u
cd "$(dirname "$0")"
log() { echo ""; echo "############ $* ############"; date; echo ""; }

log "R0 environment probe"
python -u -m src.r0_probe.run --n-points 20000 --time-budget 25

log "R2 instruments"
python -u -m src.r2_instruments.run --n-systems 45 --n-points 20000

log "R7/R3 surrogate vs statistics, with training-free baselines"
python -u -m src.r3_wpe_vs_error.run --n-systems 24 --seeds 0 1 2 3 4

log "R8 direct vs rollout"
python -u -m src.r8_direct_vs_rollout.run

log "R4 capacity x data sweep"
python -u -m src.r4_capacity_data.run

log "R5 non-monotone capacity"
python -u -m src.r5_nonmonotone.run

log "R1 ECA coarse-graining scan"
python -u -m src.r1_coarse_grain.run --block-sizes 2 3 4 --k-values 2 3 --max-projections 50000

log "BATTERY COMPLETE"
