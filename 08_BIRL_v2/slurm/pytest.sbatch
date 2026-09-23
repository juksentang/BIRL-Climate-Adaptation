#!/bin/bash
#SBATCH --job-name=pytest
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
# Full test suite (toy + realdata) of the cropchoice package on one MIG slice.
# Submit from 08_BIRL_v2/ with: bash slurm/sb.sh slurm/pytest.sbatch
set -uo pipefail
ENV_FILE="${BIRL_ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh}"
[ -f "${ENV_FILE}" ] || ENV_FILE="/scratch/${USER}/birl_v2/08_BIRL_v2/slurm/env.sh"
source "${ENV_FILE}"
module load ${MODULES}
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1
cd "${REMOTE_ROOT}"
echo "== pytest job ${SLURM_JOB_ID} on $(hostname) $(date)"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"
python3 -m pytest -q -p no:cacheprovider tests/ 2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
echo "== exit ${PIPESTATUS[0]} $(date)"
