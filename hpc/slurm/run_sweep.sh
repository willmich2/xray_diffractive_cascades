#!/bin/bash

#SBATCH --gres=gpu:volta:2
#SBATCH -n 40
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch run_sweep.sh <study_name> [extra run_sweep.py args...]"
  echo
  echo "Examples:"
  echo "  sbatch run_sweep.sh n_sweeps"
  echo "  sbatch run_sweep.sh bandwidth_energy --workers-per-gpu 2"
  echo "  sbatch run_sweep.sh thickness_energy_fig2a --save-dir paper_data"
  echo
  echo "List studies with:"
  echo "  python paper/sweeps/run_sweep.py --list-studies"
  exit 1
fi

STUDY_NAME="$1"
shift

if command -v module >/dev/null 2>&1; then
  module load conda/Python-ML-2025b-pytorch
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "${REPO_ROOT}"

echo "Running paper sweep study: ${STUDY_NAME}"
python paper/sweeps/run_sweep.py --study "${STUDY_NAME}" "$@"
