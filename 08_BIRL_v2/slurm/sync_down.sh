#!/bin/bash
# Rorqual -> laptop: rsync 08_BIRL_v2/outputs (posteriors, summaries,
# diagnostics, recovery reports, slurm logs) back into the local package.
# Skips the JAX compilation cache and in-flight checkpoints.  `-n` = dry run.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/env.sh"

DRY=""
for a in "$@"; do case "$a" in -n|--dry-run) DRY="-n" ;; *) echo "unknown arg $a"; exit 2 ;; esac; done

mkdir -p "${LOCAL_PKG}/outputs"
echo ">> ${SSH_HOST}:${REMOTE_PKG}/outputs/ -> ${LOCAL_PKG}/outputs/"
rsync -av ${DRY} -e "ssh -o BatchMode=yes" \
      --exclude '.jax_cache/' --exclude 'mcmc_state.pkl' --exclude 'samples_partial.npz' \
      --exclude '*.tmp' \
      "${SSH_HOST}:${REMOTE_PKG}/outputs/" "${LOCAL_PKG}/outputs/"

# ── Step 07 choice-counterfactual results ──
LOCAL_07=$(cd "${LOCAL_PKG}/../07_2050_Counter_Fact" && pwd)
REMOTE_07="${REMOTE_ROOT}/07_2050_Counter_Fact"
mkdir -p "${LOCAL_07}/results/choice_cf"
echo ">> ${SSH_HOST}:${REMOTE_07}/results/choice_cf/ -> ${LOCAL_07}/results/choice_cf/"
rsync -av ${DRY} -e "ssh -o BatchMode=yes" "${SSH_HOST}:${REMOTE_07}/results/choice_cf/" "${LOCAL_07}/results/choice_cf/" || echo ">> (no choice_cf results on the cluster yet)"
echo ">> sync_down done: $(du -sh "${LOCAL_PKG}/outputs" | cut -f1) in ${LOCAL_PKG}/outputs"
