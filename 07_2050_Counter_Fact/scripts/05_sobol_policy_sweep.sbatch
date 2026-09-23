#!/bin/bash
#SBATCH --job-name=cf_sobol
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Sobol sweep of the policy parameters for the choice counterfactual (stage 3).
# Submit from the 08_BIRL_v2 directory on the cluster so sb.sh supplies
# --account/--output from env.sh:
#   cd $REMOTE_PKG && bash slurm/sb.sh ../07_2050_Counter_Fact/scripts/05_sobol_policy_sweep.sbatch [--n-base 128]
set -euo pipefail
ENV_FILE="${BIRL_ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh}"
[ -f "${ENV_FILE}" ] || ENV_FILE="/scratch/${USER}/birl_v2/08_BIRL_v2/slurm/env.sh"
source "${ENV_FILE}"
module load ${MODULES}
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1
export BIRL_DATA_DIR="${REMOTE_DATA}"
export BIRL_OUT_DIR="${REMOTE_PKG}/outputs"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${REMOTE_ROOT}/07_2050_Counter_Fact"
echo "== cf_sobol job ${SLURM_JOB_ID} on $(hostname) $(date '+%F %T')  cwd=$(pwd)"
python3 -c "import jax; d=jax.devices(); print('jax', jax.__version__, d); assert d[0].platform=='gpu'"
python3 -u scripts/05_sobol_policy_sweep.py --posterior "${REMOTE_PKG}/outputs/semipar/posterior.npz" "$@" \
  2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
echo "== exit ${PIPESTATUS[0]} $(date '+%F %T')"
