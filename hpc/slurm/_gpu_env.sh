# Shared GPU job setup for example and sweep launchers.
# Source this file from the repo copy, not via $0 (SLURM copies job scripts
# into /var/spool/slurmd/... where $0 no longer lives beside this file).
#
# Callers should source as:
#   source "${SLURM_SUBMIT_DIR}/hpc/slurm/_gpu_env.sh"
# or, when running locally:
#   source "$(dirname "$0")/_gpu_env.sh"

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  _gpu_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_gpu_env_dir}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load conda/Python-ML-2025b-pytorch
fi

mkdir -p logs/slurm
