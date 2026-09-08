#!/usr/bin/env bash
# The smoke test, as two single-threaded workers.
#
# Two workers beat one two-threaded process by about 1.7x at this model size - measured in
# 00-replications and it holds here. Each shard writes its own CSV so a crash in one costs
# nothing in the other.
set -u
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-40}
NPER=${NPER:-900}
SEEDS=${SEEDS:-"0 1 2"}

export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== optimiser sweep (which recipe, measured) ==="
python -u -m src.run_smoke --optimizer-sweep --epochs "$EPOCHS" --n-per-set "$NPER" \
    --datasets cml_lattice digits_8x8 --seeds 0 1 > logs/sweep.out 2>&1

echo "=== smoke test, two shards ==="
python -u -m src.run_smoke --shard 0 --n-shards 2 --epochs "$EPOCHS" \
    --n-per-set "$NPER" --seeds $SEEDS > logs/shard0.out 2>&1 &
P0=$!
python -u -m src.run_smoke --shard 1 --n-shards 2 --epochs "$EPOCHS" \
    --n-per-set "$NPER" --seeds $SEEDS > logs/shard1.out 2>&1 &
P1=$!
wait $P0 $P1

echo "=== analysis ==="
python -u -m src.analysis
