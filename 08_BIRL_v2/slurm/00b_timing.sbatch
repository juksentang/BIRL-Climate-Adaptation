#!/bin/bash
#SBATCH --job-name=00b_timing
#SBATCH --account=def-zhiming_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jsentang/birl_v2/08_BIRL_v2/outputs/slurm_logs/%x-%j.out
# (--output must be literal: keep in sync with REMOTE_ROOT in env.sh; the
#  directory is created by sync_up.sh / submit_chain.sh)
# ---------------------------------------------------------------------------
# 00b_timing: the walltime gate for 02/03/04, measured in THEIR layout.
# One full H100 80GB, 2 h (tier b1).  run_v2.py --timing --chains 4 ->
# chain_method=vectorized on one device, exactly like the main jobs:
# 50 warmup draws (collect_warmup), a 20-draw sampling pass (includes the
# sampling-loop compile) and a second 20-draw pass from the same state
# (compile cached -> steady-state s/draw and s/leapfrog).  Runs concurrently
# with 00_smoke; 01-04 depend on both (afterok) and are submitted held.
# A 1-chain timing on a MIG 1g slice was tried first and did not finish 70
# draws in 70 min, and its number does not transfer (7x fewer SMs, 1 chain).
# Output: outputs/v2_country_timing/{timing.json, run.log, posterior.npz, ...}
#   timing.json: projected_full_run_h = s_per_leapfrog_steady x (warmup
#   leapfrog/draw x 1000 + steady leapfrog/draw x 1000) / 3600 for THIS layout
#   (the warmup term is an upper bound: mean depth of 50 early draws).
#   If it exceeds ~20 h, the main jobs still finish over two 24 h submissions
#   (warmup_state.pkl + 250-draw checkpoints; re-submit with --resume).
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

python3 run_v2.py --variant v2_country --timing --chains 4
echo "== done $(date '+%F %T')"; grep -h "^TIMING" outputs/v2_country_timing/run.log | tail -1 || true
