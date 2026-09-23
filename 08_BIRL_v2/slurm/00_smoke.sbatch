#!/bin/bash
#SBATCH --job-name=00_smoke
#SBATCH --gres=gpu:h100_1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# --account and --output are added at submission by slurm/sb.sh (from env.sh);
# #SBATCH directives cannot expand variables.
# ---------------------------------------------------------------------------
# 00_smoke: the gate for 01-04 (they are submitted with --dependency=afterok).
# One MIG 1g.10gb slice, 3 h (tier b1 max).  Three steps, each run even if an
# earlier one fails so the log shows everything; the job exits non-zero if ANY
# step failed, which cancels the dependants (kill_invalid_depend on Rorqual).
# The walltime timing run lives in 00b_timing.sbatch (full H100, 4 chains):
# on this 1g slice a 1-chain 50+20 run did not finish in 70 min.
#   1. pytest -q tests/                 toy + realdata (all 222K x 27 cells; T1-T7 + guards)
#   2. checkpoint / resume round-trip   run_v2.py 2 chains vectorized, 5+4 draws, chunk 2:
#                                       run 1 stops after chunk 1 (warmup_state.pkl + mcmc_state.pkl +
#                                       samples_partial.npz written), run 2 --resume continues from the
#                                       pickled HMC state and finishes, run 3 --resume only reloads
#                                       posterior.npz -> exactly the code path the 24 h jobs rely on
#   3. simulate_recover.py --mode svi   SVI point-estimate recovery smoke -> outputs/recovery/svi/
# 01-04 are submitted HELD by submit_chain.sh: read 00b's TIMING line, then
# `scontrol release` them (the command is printed at submission).
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

rc=0
step() { echo; echo "==== [$1] $(date '+%F %T') ===="; }

step "1/3 pytest (toy + realdata)"
python3 -m pytest -q -p no:cacheprovider tests/ || { echo "!! pytest FAILED"; rc=1; }

step "2/3 checkpoint / resume round-trip (2 chains vectorized, 5+4 draws, chunk 2)"
CK="run_v2.py --variant v2_country --chains 2 --warmup 5 --samples 4 --chunk 2 --n-ppc 4 --tag ckpt"
rm -rf outputs/v2_country_ckpt
python3 ${CK} --stop-after-chunk 1 || { echo "!! ckpt run 1 (stop after chunk 1) FAILED"; rc=1; }
ls -la outputs/v2_country_ckpt/ 2>/dev/null | grep -E "warmup_state.pkl|mcmc_state.pkl|samples_partial.npz" || { echo "!! ckpt run 1 left no checkpoint files"; rc=1; }
python3 ${CK} --resume || { echo "!! ckpt run 2 (--resume from mcmc_state.pkl) FAILED"; rc=1; }
python3 ${CK} --resume || { echo "!! ckpt run 3 (--resume with posterior.npz) FAILED"; rc=1; }
CKLOG=outputs/v2_country_ckpt/run.log
grep -q "Resumed from checkpoint: 2/4" "${CKLOG}" || { echo "!! ckpt: resume from mcmc_state.pkl not exercised"; rc=1; }
grep -q "Chunk 2: " "${CKLOG}"                   || { echo "!! ckpt: chunk 2 never ran after resume"; rc=1; }
grep -q "\[SKIP\] MCMC: loaded existing" "${CKLOG}" || { echo "!! ckpt: existing posterior.npz branch not exercised"; rc=1; }
[ -f outputs/v2_country_ckpt/posterior.npz ] && ! ls outputs/v2_country_ckpt/*.pkl >/dev/null 2>&1 \
    || { echo "!! ckpt: posterior.npz missing or checkpoint files not cleaned up"; rc=1; }
grep -h "Resumed\|Chunk\|\[SKIP\]\|\[STOP\]\|Saved: .*warmup_state" "${CKLOG}" | tail -n 8 || true

step "3/3 SVI recovery smoke (sets A and B)"
python3 simulate_recover.py --mode svi || { echo "!! svi recovery FAILED"; rc=1; }

echo
echo "==== summary $(date '+%F %T') ===="
grep -h "verdict" outputs/recovery/svi/run.log 2>/dev/null | tail -1 || true
grep -c "ptxas with version 12.6" "${SLURM_SUBMIT_DIR:-.}/outputs/slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null | sed "s/^/ptxas-12.6 clamp warnings in this log: /" || true
echo "SMOKE exit=${rc}  (0 = 01-04 may be released once 00b_timing is in)"
exit ${rc}
