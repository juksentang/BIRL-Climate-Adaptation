#!/bin/bash
# Build the Python environment ONCE on the Rorqual LOGIN node (compute nodes
# have no internet; the wheelhouse is on /cvmfs so --no-index needs none either).
#
#   laptop:  bash slurm/setup_venv.sh            (re-execs itself over ssh)
#   rorqual: bash $REMOTE_PKG/slurm/setup_venv.sh [--force]   (--force rebuilds)
#
# Ends with the installed versions and `jax.devices()` (CPU on the login node:
# the CUDA plugin only finds a GPU inside a job) and the toy pytest (40 rows).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/env.sh"

if ! command -v module >/dev/null 2>&1 && ! [ -d /cvmfs/soft.computecanada.ca ]; then
    echo ">> not on the cluster: running on ${SSH_HOST} (${REMOTE_PKG}/slurm/setup_venv.sh $*)"
    exec ssh -o BatchMode=yes "${SSH_HOST}" "bash ${REMOTE_PKG}/slurm/setup_venv.sh $*"
fi

FORCE=0
for a in "$@"; do case "$a" in --force) FORCE=1 ;; *) echo "unknown arg $a"; exit 2 ;; esac; done

echo ">> modules: ${MODULES}"
module load ${MODULES}
echo ">> python: $(which python) ($(python --version 2>&1))"

if [ -x "${VENV}/bin/python" ] && [ "${FORCE}" = 0 ]; then
    echo ">> venv exists: ${VENV} (use --force to rebuild)"
else
    [ "${FORCE}" = 1 ] && rm -rf "${VENV}"
    mkdir -p "${REMOTE_ROOT}"
    echo ">> creating venv ${VENV}"
    python -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --no-index --quiet --upgrade pip
echo ">> pip install --no-index ${PIP_PINS}"
pip install --no-index ${PIP_PINS}

echo
echo ">> installed:"
pip list --format=columns 2>/dev/null | grep -iE "^(jax|jaxlib|jax-cuda12-plugin|jax-cuda12-pjrt|numpyro|numpy|scipy|pandas|pytest|ml-dtypes) "
# JAX_PLATFORMS=cpu: the CUDA plugin would otherwise print a CUDA_ERROR_NO_DEVICE traceback (login node has no GPU)
JAX_PLATFORMS=cpu python - <<'PY'
import jax, jaxlib, numpyro, numpy, scipy, pandas, pyarrow, pytest
print(f"jax {jax.__version__}  jaxlib {jaxlib.__version__}  numpyro {numpyro.__version__}  "
      f"numpy {numpy.__version__}  scipy {scipy.__version__}  pandas {pandas.__version__}  "
      f"pyarrow {pyarrow.__version__} (module)  pytest {pytest.__version__}")
print("jax.devices() on the login node (CPU expected here; GPU only inside a job):", jax.devices())
PY

echo
echo ">> toy pytest (40 synthetic rows, CPU, no data) -- validates the jax/numpyro combination"
cd "${REMOTE_PKG}"
# Login-node guard: XLA sizes its CPU thread pool from the visible cores (192 here) and
# pthread_create() then fails (EAGAIN) under the login-node process cap -> "Fatal Python
# error: Aborted" inside backend_compile.  A 4-core affinity + single-threaded Eigen
# keeps it small.  Compute nodes are cgroup-confined to --cpus-per-task, so no guard needed there.
BIRL_HOST_DEVICES=1 BIRL_JAX_CACHE=0 JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
    taskset -c 0-3 python -m pytest -q -x -m toy -p no:cacheprovider tests/
echo ">> setup_venv.sh done"
