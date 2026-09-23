#!/bin/bash
#SBATCH --job-name=exp_familiar
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Gradient-cost diagnostic (see exp_familiar.py). Independent of the job chain.
set -uo pipefail
# ── environment (identical in every *.sbatch) ──
ENV_FILE="${BIRL_ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh}"
[ -f "${ENV_FILE}" ] || ENV_FILE="/scratch/${USER}/birl_v2/08_BIRL_v2/slurm/env.sh"
source "${ENV_FILE}"
module load ${MODULES}
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1
export BIRL_OUT_DIR="${REMOTE_PKG}/outputs/exp_familiar_tmp"
cd "${REMOTE_PKG}"
echo "== exp_familiar job ${SLURM_JOB_ID} on $(hostname) $(date)"
python3 -u slurm/exp_familiar.py 6000 2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
echo "== exit ${PIPESTATUS[0]} $(date)"
