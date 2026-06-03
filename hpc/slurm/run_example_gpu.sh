#!/bin/bash

#SBATCH --job-name=example-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH -n 8
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Run an examples/ script on GPU with default SLURM resources.
#
# Submit from the repository root:
#   sbatch hpc/slurm/run_example_gpu.sh examples/<script>.py [extra args...]
#
# Extra args are forwarded to the Python script. --device cuda is added
# automatically unless the command already includes --device.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/hpc/slurm/_gpu_env.sh"
else
  source "$(cd "$(dirname "$0")" && pwd)/_gpu_env.sh"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch hpc/slurm/run_example_gpu.sh <example_script.py> [args...]"
  echo "Example: sbatch hpc/slurm/run_example_gpu.sh examples/xray_focusing_testing.py"
  exit 1
fi

EXAMPLE_SCRIPT="$1"
shift

if [[ ! -f "${EXAMPLE_SCRIPT}" ]]; then
  echo "Example script not found: ${EXAMPLE_SCRIPT}"
  exit 2
fi

DEVICE_ARGS=(--device cuda)
for arg in "$@"; do
  if [[ "${arg}" == "--device" ]]; then
    DEVICE_ARGS=()
    break
  fi
done

echo "Running ${EXAMPLE_SCRIPT} on GPU"
python "${EXAMPLE_SCRIPT}" "${DEVICE_ARGS[@]}" "$@"
