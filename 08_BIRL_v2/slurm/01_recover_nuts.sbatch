#!/bin/bash
#SBATCH --job-name=01_recover_nuts
#SBATCH --gres=gpu:h100_2g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# --account and --output are added at submission by slurm/sb.sh (from env.sh);
# #SBATCH directives cannot expand variables.
# ---------------------------------------------------------------------------
# 01_recover_nuts: the PASS verdict of the simulation-recovery test.
# One MIG 2g.20gb slice, 12 h (partition tier b2).  Flat priors, truth sets A
# and B on the real q arrays, NUTS 2 chains x (500 + 500) each (2 chains on one
# device -> chain_method=vectorized, chosen by simulate_recover.py).
# The smoke's 1-chain 50+20 timing on a 1g slice ran > 45 min (deep warmup
# trees), so 2 sets x 2 lock-stepped chains x 1000 draws can take many hours:
# each set is checkpointed (nuts_posterior_<set>.npz + report.json after every
# set) and --resume loads finished sets instead of re-fitting them, so after a
# TIMEOUT simply re-submit this same script (same --gres).
# Output: outputs/recovery/nuts/{report.md, report.json, nuts_posterior_{A,B}.npz}
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

python3 simulate_recover.py --mode nuts --sets A B --nuts-chains 2 --nuts-warmup 500 --nuts-samples 500 --resume
echo "== done $(date '+%F %T')"; grep -h "verdict" outputs/recovery/nuts/run.log | tail -1 || true
