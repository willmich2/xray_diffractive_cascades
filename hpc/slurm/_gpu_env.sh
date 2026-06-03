# Shared GPU job setup for example and sweep launchers.
# Source this file from the repo copy, not via $0 (SLURM copies job scripts
# into /var/spool/slurmd/... where $0 no longer lives beside this file).
#
# Callers must source this file by absolute path (see run_example_gpu.sh).
# BASH_SOURCE resolves the real repo path; SLURM copies job scripts to /var/spool.

set -euo pipefail

_gpu_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_gpu_env_dir}/../.." && pwd)"
cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load conda/Python-ML-2025b-pytorch
fi

mkdir -p logs/slurm
