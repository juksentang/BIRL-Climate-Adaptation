# Copy to env.sh (untracked) and fill in the two values under 'Who / where'.
# BIRL v2 cluster settings -- sourced by every slurm/ script, on the laptop
# (sync_up.sh / sync_down.sh / submit_chain.sh / status.sh re-exec over ssh)
# and on Rorqual (setup_venv.sh, every *.sbatch).  Pure variable definitions:
# no side effects, safe under `set -euo pipefail`.
#
# Values verified live on rorqual3 on 2026-09-19 (see README.md, "How the
# versions were chosen").  If you change REMOTE_ROOT, also change the
# nothing else: slurm/sb.sh passes --account/--output at submission.

# ── Who / where ──
RORQUAL_USER=<alliance-username>            # e.g. from `whoami` on the login node
ACCOUNT=def-<pi>_gpu                          # your GPU allocation; passed by slurm/sb.sh as --account
SSH_HOST=rorqual                              # ~/.ssh/config alias (ControlMaster, MFA done)
REMOTE_ROOT=/scratch/${RORQUAL_USER}/birl_v2  # $SCRATCH/birl_v2
REMOTE_PKG=${REMOTE_ROOT}/08_BIRL_v2          # the code (this package)
REMOTE_DATA=${REMOTE_ROOT}/06_BIRL_MCMC/data  # so DATA_DIR = ../06_BIRL_MCMC/data resolves unchanged
VENV=${REMOTE_ROOT}/venv                      # built once on the login node by setup_venv.sh

# ── Laptop side (derived from the location of this file) ──
_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_PKG=$(cd "${_ENV_DIR}/.." && pwd)                       # .../08_BIRL_v2
LOCAL_DATA=$(cd "${LOCAL_PKG}/../06_BIRL_MCMC/data" 2>/dev/null && pwd || echo "${LOCAL_PKG}/../06_BIRL_MCMC/data")
DATA_FILES="birl_sample.parquet env_model_output.npz action_space_config.json"

# ── Software stack (Alliance StdEnv/2023; compute nodes have no internet) ──
MODULES="StdEnv/2023 python/3.11 cuda/12.9 cudnn/9.13.1.26 arrow/25.0.0"
#   python/3.11  -> 3.11.5 (default python/3.11)
#   cuda/12.9    -> cudacore 12.9.1, ptxas 12.9. NOT the default cuda/12.6 (= cudacore 12.6.2,
#                   ptxas 12.6.77): XLA warns that CUDA <= 12.6.2 miscompiles certain clamping
#                   edge cases, and the gradient path clips (models.py T_CLIP / .clip(min=1) /
#                   log-beta clip), so a miscompiled NUTS kernel would silently bias the posterior.
#                   jax_cuda12_plugin 0.10.2 dlopens the CUDA 12 / cuDNN 9 (>=9.8) libs from the
#                   loaded modules, so no venv rebuild is needed when switching cuda modules.
#   cudnn/9.13.1.26 -> the only cudnn that loads with cuda/12.9 (cudnn/9.10.0.56 does not)
#   arrow/25.0.0 -> provides pyarrow (the wheelhouse only has a dummy pyarrow-9999 stub)
# One consistent cp311 set from the Alliance wheelhouse (pip install --no-index):
JAX_VERSION=0.10.2          # jax == jaxlib == jax_cuda12_plugin == jax_cuda12_pjrt
NUMPYRO_VERSION=0.21.0      # requires jax >= 0.7
PIP_PINS="jax==${JAX_VERSION} jaxlib==${JAX_VERSION} jax_cuda12_plugin==${JAX_VERSION} jax_cuda12_pjrt==${JAX_VERSION} numpyro==${NUMPYRO_VERSION} numpy==2.4.2 scipy==1.17.1 pandas==2.3.3 pytest==9.1.1"

# ── GPU requests (documentation only: #SBATCH lines cannot expand variables) ──
GRES_SMOKE=gpu:h100_1g.10gb:1     # 00_smoke        MIG 1/7 slice, 3 h, billing 1743 / GPU-h
GRES_TIMING=gpu:h100:1            # 00b_timing      one full H100, 2 h, --timing --chains 4 (the main jobs' layout)
GRES_RECOVER=gpu:h100_2g.20gb:1   # 01_recover_nuts MIG 2/7 slice, 12 h, billing 3486 / GPU-h (per-set --resume)
GRES_MAIN=gpu:h100:1              # 02/03/04        one full H100 80GB, 24 h, 4 chains vectorized, billing 12200 / GPU-h
# Re-submit a timed-out job with the SAME --gres: the checkpoint stores chain_method
# (1 device -> vectorized); a different device count changes it and restarts from scratch.
