#!/bin/bash
#SBATCH --job-name=exp_familiar
#SBATCH --account=def-zhiming_gpu
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jsentang/birl_v2/08_BIRL_v2/outputs/slurm_logs/%x-%j.out
# Gradient-cost diagnostic (see exp_familiar.py). Independent of the job chain.
set -uo pipefail
source /scratch/jsentang/birl_v2/08_BIRL_v2/slurm/env.sh
module load ${MODULES}
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1
export BIRL_OUT_DIR=/scratch/jsentang/birl_v2/08_BIRL_v2/outputs/exp_familiar_tmp
cd /scratch/jsentang/birl_v2/08_BIRL_v2
echo "== exp_familiar job ${SLURM_JOB_ID} on $(hostname) $(date)"
python3 -u slurm/exp_familiar.py 6000 2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
echo "== exit ${PIPESTATUS[0]} $(date)"
