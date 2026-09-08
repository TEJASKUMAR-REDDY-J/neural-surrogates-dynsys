#!/usr/bin/env bash
# R4c - firm up the sharpest finding in the battery.
#
# R4b found that the horizon at which extra model capacity stops paying varies by ~450x
# across systems when measured in Lyapunov times (0.0 to 13.4). If the ceiling were simply
# "chaos runs out", those numbers would all match. They do not, and the Lyapunov exponent
# does not explain the spread (Spearman 0.14). Attractor dimension looked better (-0.71) but
# with n=6 that is suggestive at best.
#
# This run takes the measurement to 14 systems spanning a 200x range in Lyapunov exponent
# (0.011 to 2.325). That makes the "does lambda predict it?" test genuinely well powered.
# The attractor-dimension test stays underpowered, because the catalogue simply does not
# contain many low-dimensional attractors above dimension 2.5 - which is itself worth saying.
#
# Cheaper than R4b per system: one data budget instead of three, and 5 capacities instead of
# 7, since only the exponent is needed.
set -u
cd "$(dirname "$0")"
export TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo ""; echo "############ $* ############"; date; echo ""; }

R4C="GlycolyticOscillation Blasius ChenLee Dadras Chua ExcitableCell Chen \
AtmosphericRegime BurkeShaw Hopfield DequanLi HyperBao CaTwoPlusQuasiperiodic Bouali2"

log "R4c horizon-where-capacity-stops-paying, 14 systems"
for sh in 0 1; do
  python -u -m src.r4_capacity_data.run --systems $R4C \
      --pts-per-period 20 --steps 1000 --n-trains 8000 \
      --budgets 1000 3000 10000 32000 100000 \
      --n-points 30000 --seeds 0 1 2 --shard $sh --n-shards 2 &
done
wait

log "ANALYSIS + FIGURES"
python -u -m src.analysis.run
python -u -m src.analysis.figures
log "R4C COMPLETE"
