#!/bin/bash
#SBATCH --job-name=choice_cf
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# Stage 3 (rewritten): choice-model counterfactual on the Step 08 posterior.
# Submit from the 08_BIRL_v2 directory on the cluster so sb.sh supplies
# --account/--output from env.sh:
#   cd $REMOTE_PKG && bash slurm/sb.sh ../07_2050_Counter_Fact/scripts/03_choice_counterfactual.sbatch
set -euo pipefail
ENV_FILE="${BIRL_ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh}"
[ -f "${ENV_FILE}" ] || ENV_FILE="/scratch/${USER}/birl_v2/08_BIRL_v2/slurm/env.sh"
source "${ENV_FILE}"
module load ${MODULES}
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1
export BIRL_DATA_DIR="${REMOTE_DATA}"                 # 06 data (birl_sample, env_model_output, action config)
export BIRL_OUT_DIR="${REMOTE_PKG}/outputs"           # 08 config: jax cache lives here
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${REMOTE_ROOT}/07_2050_Counter_Fact"
echo "== choice_cf job ${SLURM_JOB_ID} on $(hostname) $(date '+%F %T')  cwd=$(pwd)"
python3 -c "import jax; d=jax.devices(); print('jax', jax.__version__, d); assert d[0].platform=='gpu'"
python3 -u scripts/03_choice_counterfactual.py --posterior "${REMOTE_PKG}/outputs/semipar/posterior.npz" "$@" \
  2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
echo "== exit ${PIPESTATUS[0]} $(date '+%F %T')"
