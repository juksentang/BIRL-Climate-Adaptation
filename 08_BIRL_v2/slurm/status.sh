#!/bin/bash
# Queue state + tail of every slurm log and run.log, plus the one-line results
# (TIMING / RECOVERY verdict / convergence) as they appear.
#   laptop:  bash slurm/status.sh [N]      (N = lines per log, default 12; re-execs over ssh)
#   rorqual: bash $REMOTE_PKG/slurm/status.sh [N]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/env.sh"
N="${1:-12}"

if ! command -v squeue >/dev/null 2>&1; then
    exec ssh -o BatchMode=yes "${SSH_HOST}" "bash ${REMOTE_PKG}/slurm/status.sh ${N}"
fi

cd "${REMOTE_PKG}"
echo "===== squeue -u ${USER}  ($(date '+%F %T'), $(hostname)) ====="
squeue -u "${USER}" -o "%.10i %.16j %.4t %.10M %.11l %.6D %.20b %R" || true
# gres/gpumem from sstat is the GPU check for RUNNING jobs: gres/gpuutil reads 0 on MIG
# slices (NVML has no per-instance utilization) and is NOT evidence of a CPU fallback;
# the jax.devices() assertion at the top of every sbatch log is the authoritative check.
for j in $(squeue -h -u "${USER}" -t R -o "%i"); do
    echo "-- sstat ${j}.batch: $(sstat -n -j "${j}.batch" --format=TRESUsageInTot%120 2>/dev/null | tr -s ' ' | grep -oE 'gres/gpumem=[^,]*|gres/gpuutil=[^,]*|cpu=[^,]*' | tr '\n' ' ')"
done
echo
echo "===== sacct (last 2 days) ====="
sacct -u "${USER}" -X -S "$(date -d '2 days ago' +%F)" \
      --format=JobID%10,JobName%16,State%12,Elapsed%10,Timelimit%10,ExitCode,ReqTRES%38 2>/dev/null || true
echo
echo "===== results so far ====="
grep -h "^TIMING" outputs/*_timing/run.log 2>/dev/null | tail -n 2 || true
grep -h "verdict:" outputs/recovery/*/run.log 2>/dev/null | tail -n 4 || true
for d in outputs/v2_country outputs/v2_country_gfix outputs/v2_country_smax08; do
    if [ -f "$d/posterior.npz" ]; then
        echo "-- $d: posterior.npz done; $(grep -m1 -iE 'divergen' "$d/convergence.txt" 2>/dev/null || echo 'see convergence.txt')"
    elif [ -f "$d/samples_partial.npz" ]; then
        echo "-- $d: in progress (checkpoint present)"
    fi
done
echo
shopt -s nullglob
for f in $(ls -t outputs/slurm_logs/*.out 2>/dev/null | head -n 8); do
    echo "===== ${f}  ($(stat -c %y "$f" | cut -d. -f1)) ====="
    tail -n "${N}" "$f"; echo
done
for f in outputs/*/run.log outputs/recovery/*/run.log; do
    echo "----- ${f} (last 3) -----"; tail -n 3 "$f"
done
