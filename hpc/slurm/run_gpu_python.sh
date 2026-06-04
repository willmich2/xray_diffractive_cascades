#!/bin/bash

#SBATCH --job-name=gpu-python
#SBATCH --gres=gpu:volta:2
#SBATCH -n 40
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Run an arbitrary repository Python script on GPU cluster resources.
#
# Submit from repository root:
#   sbatch hpc/slurm/run_gpu_python.sh paper/postprocess/placement_robustness.py [args...]
#
# For convenience, basenames are also resolved under common directories:
#   sbatch hpc/slurm/run_gpu_python.sh placement_robustness.py [args...]

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -f "${SLURM_SUBMIT_DIR}/hpc/slurm/_gpu_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/hpc/slurm/_gpu_env.sh"
  elif [[ -f "${SLURM_SUBMIT_DIR}/_gpu_env.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/_gpu_env.sh"
  else
    echo "Could not find _gpu_env.sh under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
  fi
else
  source "$(cd "$(dirname "$0")" && pwd)/_gpu_env.sh"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch hpc/slurm/run_gpu_python.sh <script.py> [args...]"
  echo
  echo "Examples:"
  echo "  sbatch hpc/slurm/run_gpu_python.sh paper/postprocess/placement_robustness.py --base-id <ID> --run-id 0 --data-dir paper_data"
  echo "  sbatch hpc/slurm/run_gpu_python.sh placement_robustness.py --base-id <ID> --run-id 0 --data-dir paper_data"
  exit 1
fi

INPUT_SCRIPT="$1"
shift

RESOLVED_SCRIPT=""
CANDIDATES=(
  "${INPUT_SCRIPT}"
  "paper/postprocess/${INPUT_SCRIPT}"
  "examples/${INPUT_SCRIPT}"
  "paper/experiments/${INPUT_SCRIPT}"
  "paper/sweeps/${INPUT_SCRIPT}"
)

for candidate in "${CANDIDATES[@]}"; do
  if [[ -f "${candidate}" ]]; then
    RESOLVED_SCRIPT="${candidate}"
    break
  fi
done

if [[ -z "${RESOLVED_SCRIPT}" ]]; then
  echo "Python script not found: ${INPUT_SCRIPT}" >&2
  echo "Tried:" >&2
  for candidate in "${CANDIDATES[@]}"; do
    echo "  - ${candidate}" >&2
  done
  exit 2
fi

echo "Running ${RESOLVED_SCRIPT}"
python "${RESOLVED_SCRIPT}" "$@"
