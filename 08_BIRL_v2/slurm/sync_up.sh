#!/bin/bash
# Laptop -> Rorqual: rsync 08_BIRL_v2 (code, tests, slurm; NOT outputs) and
# the three 06 data files to $REMOTE_ROOT.  Re-runnable; `--dry-run` / `-n`
# shows what would change.  Remote outputs/ is never touched.
# Refuses to sync while this user has jobs queued or running on Rorqual (the
# queued sbatch scripts import run_v2.py/src at start, so an rsync --delete of
# the code tree silently changes what they execute); `--force` overrides.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/env.sh"

DRY=""; FORCE=0
for a in "$@"; do case "$a" in -n|--dry-run) DRY="-n" ;; --force) FORCE=1 ;; *) echo "unknown arg $a"; exit 2 ;; esac; done

if [ -z "${DRY}" ]; then
    Q=$(ssh -o BatchMode=yes "${SSH_HOST}" 'squeue -h -u $USER -o "%.10i %.16j %.4t %R"' 2>/dev/null || true)
    if [ -n "${Q}" ]; then
        echo ">> ${SSH_HOST} has jobs queued/running for ${RORQUAL_USER}:"; echo "${Q}"
        if [ "${FORCE}" = 1 ]; then
            echo ">> --force: syncing anyway (queued jobs will run the NEW code; running jobs keep their imported modules)"
        else
            echo ">> refusing to rsync --delete the code tree under them; wait, scancel, or pass --force"; exit 1
        fi
    fi
fi

for f in ${DATA_FILES}; do
    [ -f "${LOCAL_DATA}/${f}" ] || { echo "missing ${LOCAL_DATA}/${f}"; exit 1; }
done

echo ">> mkdir -p ${REMOTE_PKG} ${REMOTE_DATA} on ${SSH_HOST}"
ssh -o BatchMode=yes "${SSH_HOST}" "mkdir -p '${REMOTE_PKG}/outputs/slurm_logs' '${REMOTE_DATA}'"

echo ">> code: ${LOCAL_PKG}/ -> ${SSH_HOST}:${REMOTE_PKG}/"
rsync -av ${DRY} --delete -e "ssh -o BatchMode=yes" \
      --exclude 'outputs/' --exclude '__pycache__/' --exclude '*.pyc' \
      --exclude '.pytest_cache/' --exclude '.jax_cache/' --exclude '.DS_Store' \
      "${LOCAL_PKG}/" "${SSH_HOST}:${REMOTE_PKG}/"

echo ">> data: ${LOCAL_DATA}/{${DATA_FILES// /,}} -> ${SSH_HOST}:${REMOTE_DATA}/"
FILES=()
for f in ${DATA_FILES}; do FILES+=("${LOCAL_DATA}/${f}"); done
rsync -av ${DRY} -e "ssh -o BatchMode=yes" "${FILES[@]}" "${SSH_HOST}:${REMOTE_DATA}/"

echo ">> chmod +x slurm/*.sh"
ssh -o BatchMode=yes "${SSH_HOST}" "chmod +x ${REMOTE_PKG}/slurm/*.sh; ls -la ${REMOTE_DATA}"
echo ">> sync_up done"
