#!/bin/bash
#SBATCH --job-name=lce-phase0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase0_%j.out

# VRAM note: one LM run peaks near 1.2 GB, so a single 11 GB 2080 Ti or a 12 GB
# 4070 Super holds every worker this job will start. --cpus-per-task is the knob
# that actually governs throughput, because the PMO oracles are CPU-bound.

set -euo pipefail
module load cuda/11.8 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lce

mkdir -p logs
python scripts/run_sweep.py --workers "${SLURM_CPUS_PER_TASK:-8}"
