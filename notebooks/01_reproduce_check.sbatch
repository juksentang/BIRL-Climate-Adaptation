#!/bin/bash
#SBATCH --job-name=nb01_check
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Runs notebooks/01_reproduce.py end to end as plain python (tier-1 check).
set -uo pipefail
ENV_FILE="${BIRL_ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh}"
[ -f "${ENV_FILE}" ] || ENV_FILE="/scratch/${USER}/birl_v2/08_BIRL_v2/slurm/env.sh"
source "${ENV_FILE}"
module load ${MODULES}
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1
cd "${REMOTE_ROOT}"
python3 -c "import cropchoice" 2>/dev/null || pip install -e "${REMOTE_ROOT}" --no-deps -q
T0=$SECONDS
echo "== nb01_check job ${SLURM_JOB_ID} on $(hostname) $(date)"
cd "${REMOTE_ROOT}/notebooks"
python3 -u 01_reproduce.py 2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
RC=${PIPESTATUS[0]}
echo "== exit ${RC:-?} $(date)"
echo "== wall $((SECONDS-T0)) s"
