#!/usr/bin/env bash
# Insurance: the remaining phases, launchable standalone if the main chain dies.
# (run_battery.sh was edited while bash was executing it, which can corrupt its read offset.)
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo ""; echo "############ $* ############"; date; echo ""; }

if [ ! -s results/r4/capacity_data_sweep__shard0.csv ]; then
  log "R4 capacity x data sweep"
  python -u -m src.r4_capacity_data.run --shard 0 --n-shards 2 &
  python -u -m src.r4_capacity_data.run --shard 1 --n-shards 2 &
  wait
fi

if [ ! -s results/r5/capacity_curves__shard0.csv ]; then
  log "R5 non-monotone capacity"
  python -u -m src.r5_nonmonotone.run --shard 0 --n-shards 2 &
  python -u -m src.r5_nonmonotone.run --shard 1 --n-shards 2 &
  wait
fi

log "R1 ECA coarse-graining with escalation"
python -u -m src.r1_coarse_grain.run --block-sizes 2 3 4 --k-values 2 3 \
    --max-projections 60000 --escalate-sizes 5 6 \
    --escalate-projections 40000 400000 4000000

log "ANALYSIS + FIGURES"
python -u -m src.analysis.run
python -u -m src.analysis.figures
log "REMAINING COMPLETE"
