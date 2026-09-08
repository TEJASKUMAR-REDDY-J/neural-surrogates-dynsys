#!/usr/bin/env bash
# The planned follow-ups that are ready to run. Kuramoto-Sivashinsky (N1) is deliberately
# absent - see NOTES.md: our ETDRK4 integration destabilises past roughly 11k steps and needs
# a validated reference implementation rather than more guessing.
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo ""; echo "############ $* ############"; date; echo ""; }

# R2 supplies the predictor features that R3 regresses against, so scaling R3 up to the whole
# library means profiling the whole library first. The earlier run covered 45 systems.
log "R2-wide: statistics for all usable systems"
python -u -m src.r2_instruments.run --n-systems 130 --n-points 20000

log "N3 predictor at scale: every bounded system"
for sh in 0 1; do
  python -u -m src.r3_wpe_vs_error.run --n-systems 130 --bounded-only       --seeds 0 1 2 --shard $sh --n-shards 2 &
done
wait

log "N5 batch size sweep with seeds"
python -u -m src.n5_batch_sweep.run

log "N2 observational noise, three levels"
for nz in 0.01 0.05 0.20; do
  for sh in 0 1; do
    python -u -m src.r3_wpe_vs_error.run --n-systems 40 --bounded-only --noise $nz         --seeds 0 1 2 --shard $sh --n-shards 2 &
  done
  wait
done

log "ANALYSIS + FIGURES"
python -u -m src.analysis.run
python -u -m src.analysis.figures
log "NEXT-ROUND COMPLETE"
