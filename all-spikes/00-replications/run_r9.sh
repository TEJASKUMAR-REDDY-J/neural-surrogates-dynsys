#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "############ R9 iterative refinement ############"; date
for sh in 0 1; do
  python -u -m src.r9_iterative_refinement.run --shard $sh --n-shards 2 &
done
wait
echo "############ R9 COMPLETE ############"; date
