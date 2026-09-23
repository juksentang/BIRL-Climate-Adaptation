#!/bin/bash
#SBATCH --job-name=semipar_assets
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# --account and --output are added at submission by slurm/sb.sh (from env.sh);
# #SBATCH directives cannot expand variables.
# ---------------------------------------------------------------------------
# 02_main: main run, variant v2_country (rho_c, s_c, beta_c sampled; s_max 0.6)
# One full H100 80GB, 24 h (partition tier b3; charged on use).  4 chains x
# (1000 warmup + 1000 samples) on one device -> chain_method=vectorized (chosen
# by run_v2.py; pass --chain-method to override).  The 1000-step warmup runs in
# one shot and its post-warmup state is saved to warmup_state.pkl; sampling is
# checkpointed every 250 draws.  --resume makes a re-submission of this same
# script (SAME --gres: the checkpoint stores chain_method) continue from the
# last checkpoint, skip the warmup if only warmup_state.pkl exists, or only
# recompute diagnostics when posterior.npz already exists.
# Output: outputs/v2_country/{posterior.npz, summary.csv, convergence.txt,
#         ppc_action_freq.csv, ppc_crop_x_intensity.csv, model_diagnostics.json,
#         run.log, run_info.json}
# ---------------------------------------------------------------------------
set -euo pipefail
# ── environment (identical in every *.sbatch) ──
ENV_FILE="${BIRL_ENV_FILE:-${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/env.sh}"
[ -f "${ENV_FILE}" ] || ENV_FILE="/scratch/${USER}/birl_v2/08_BIRL_v2/slurm/env.sh"
source "${ENV_FILE}"
module load ${MODULES}                       # StdEnv/2023 python/3.11 cuda/12.9 cudnn/9.13.1.26 arrow/25.0.0 (cuda/12.9: ptxas 12.6 clamp miscompile warning)
source "${VENV}/bin/activate"
export BIRL_HOST_DEVICES=1                   # GPU node: no CPU pseudo-devices for numpyro
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${REMOTE_ROOT}/08_BIRL_v2"
echo "== ${SLURM_JOB_NAME} job ${SLURM_JOB_ID} on $(hostname) $(date '+%F %T')"
echo "== gres=${SLURM_JOB_GPUS:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}M  cwd=$(pwd)"
nvidia-smi -L 2>/dev/null || true
# sanity: JAX must see the GPU (a silent CPU fallback would waste the whole allocation)
python3 -c "import jax; d = jax.devices(); print('jax', jax.__version__, 'devices:', d); assert d[0].platform == 'gpu', f'JAX sees no GPU: {d}'"

python3 -u slurm/run_semipar_assets.py --chains 4 --warmup 1000 --samples 1000 2>&1 | grep -v "slow_operation_alarm\|constant-folding\|xla_dump\|reduce_sum\|necessarily a bug\|file a bug\|^$"
echo "== done $(date '+%F %T')"; tail -n 3 outputs/v2_country/convergence.txt 2>/dev/null || true
