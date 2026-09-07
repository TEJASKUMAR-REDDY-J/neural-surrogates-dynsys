#!/usr/bin/env bash
# Replication battery.
#
# Measured on this machine: torch's 2 threads give nothing for models this small (62.4s
# vs 63.4s for the same three fits) because the work is dispatch-bound, not FLOP-bound.
# Two single-threaded worker processes give ~1.7x instead. So every training phase runs
# as two shards; phases stay sequential to keep the timing numbers interpretable.
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo ""; echo "############ $* ############"; date; echo ""; }

log "R2 instruments (sampling-corrected + naive)"
python -u -m src.r2_instruments.run --n-systems 45 --n-points 20000

log "R3 surrogate vs statistics, with training-free baselines (R7 folded in)"
python -u -m src.r3_wpe_vs_error.run --n-systems 24 --seeds 0 1 2 3 4 --shard 0 --n-shards 2 &
python -u -m src.r3_wpe_vs_error.run --n-systems 24 --seeds 0 1 2 3 4 --shard 1 --n-shards 2 &
wait

log "R8 direct vs rollout"
python -u -m src.r8_direct_vs_rollout.run --shard 0 --n-shards 2 &
python -u -m src.r8_direct_vs_rollout.run --shard 1 --n-shards 2 &
wait

log "R4 capacity x data sweep"
python -u -m src.r4_capacity_data.run --shard 0 --n-shards 2 &
python -u -m src.r4_capacity_data.run --shard 1 --n-shards 2 &
wait

log "R5 non-monotone capacity"
python -u -m src.r5_nonmonotone.run --shard 0 --n-shards 2 &
python -u -m src.r5_nonmonotone.run --shard 1 --n-shards 2 &
wait

log "R1 ECA coarse-graining scan, with escalation to larger block sizes"
python -u -m src.r1_coarse_grain.run --block-sizes 2 3 4 --k-values 2 3 \
    --max-projections 60000 --escalate-sizes 5 6 \
    --escalate-projections 40000 400000 4000000

log "ANALYSIS"
python -u -m src.analysis.run

log "BATTERY COMPLETE"
