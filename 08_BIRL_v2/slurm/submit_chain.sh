#!/bin/bash
# Submit the Phase 1 chain on Rorqual:
#   00_smoke (gate, MIG 1g) + 00b_timing (full H100, 4-chain timing)  ->  01_recover_nuts, 02_main, 03_gfix, 04_smax08
# 01-04 each carry --dependency=afterok:<00>:<00b> and are independent of
# each other.  Rorqual runs kill_invalid_depend, so a failed smoke cancels
# them automatically.  01-04 are submitted HELD (sbatch -H) unless
# --skip-smoke or --no-hold: afterok alone would start three full-H100 jobs
# the instant the smoke exits 0, before anyone has read the TIMING line the
# smoke exists to produce.  Read outputs/v2_country_timing/timing.json, then
# run the printed `scontrol release ...`.  Prints every job id and appends
# them to outputs/slurm_logs/jobids.txt.
#
#   laptop:  bash slurm/submit_chain.sh [opts]      (re-execs over ssh)
#   rorqual: bash $REMOTE_PKG/slurm/submit_chain.sh [opts]
# opts:
#   --smoke-only        submit 00 (+ 00b) only
#   --no-timing         do not submit 00b_timing
#   --skip-smoke        submit 01-04 with no dependency (smoke already passed)
#   --after ID[,ID]     use afterok:ID[:ID] instead of submitting a new 00/00b
#   --only NAME[,NAME]  restrict 01-04 to these (e.g. --only 02_main,03_gfix)
#   --no-hold           do not hold 01-04 (they start as soon as the smoke passes)
#   --release           release every held job of this user and exit
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/env.sh"

if ! command -v sbatch >/dev/null 2>&1; then
    echo ">> not on the cluster: running on ${SSH_HOST}"
    exec ssh -o BatchMode=yes "${SSH_HOST}" "bash ${REMOTE_PKG}/slurm/submit_chain.sh $*"
fi

SMOKE_ONLY=0; SKIP_SMOKE=0; AFTER=""; ONLY="01_recover_nuts,02_main,03_gfix,04_smax08"; HOLD=1; RELEASE=0; TIMING=1
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1; HOLD=0 ;;
        --after) AFTER="$2"; shift ;;
        --only) ONLY="$2"; shift ;;
        --no-hold) HOLD=0 ;;
        --no-timing) TIMING=0 ;;
        --release) RELEASE=1 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac; shift
done

if [ "${RELEASE}" = 1 ]; then
    HELD=$(squeue -h -u "${USER}" -t PD -o "%i %r" | awk '$2 == "JobHeldUser" {print $1}' | tr '\n' ' ')
    [ -n "${HELD}" ] || { echo "no held jobs for ${USER}"; exit 0; }
    echo "scontrol release ${HELD}"; scontrol release ${HELD}
    squeue -u "${USER}" -o "%.10i %.16j %.4t %.10M %.11l %E %R"; exit 0
fi

cd "${REMOTE_PKG}"
mkdir -p outputs/slurm_logs
[ -x "${VENV}/bin/python" ] || { echo "venv missing at ${VENV}: run slurm/setup_venv.sh first"; exit 1; }
for f in ${DATA_FILES}; do
    [ -f "${REMOTE_DATA}/${f}" ] || { echo "data missing: ${REMOTE_DATA}/${f} (run slurm/sync_up.sh)"; exit 1; }
done

DEP=""
IDS=()
HELD_IDS=()
[ "${HOLD}" = 1 ] || HOLD=""            # empty -> ${HOLD:+-H} expands to nothing
if [ -n "${AFTER}" ]; then
    DEP="--dependency=afterok:${AFTER//,/:}"; echo "00_smoke/00b: reusing job(s) ${AFTER}"
elif [ "${SKIP_SMOKE}" = 1 ]; then
    echo "00_smoke: skipped (no dependency)"
else
    J0=$(sbatch --parsable slurm/00_smoke.sbatch)
    IDS+=("00_smoke ${J0}"); DEP="--dependency=afterok:${J0}"
    echo "00_smoke        -> job ${J0}"
    if [ "${TIMING}" = 1 ]; then
        J0B=$(sbatch --parsable slurm/00b_timing.sbatch)
        IDS+=("00b_timing ${J0B}"); DEP="${DEP}:${J0B}"
        echo "00b_timing      -> job ${J0B}   (full H100, --timing --chains 4; concurrent with 00)"
    fi
fi

if [ "${SMOKE_ONLY}" = 0 ]; then
    for name in ${ONLY//,/ }; do
        [ -f "slurm/${name}.sbatch" ] || { echo "no slurm/${name}.sbatch"; exit 2; }
        J=$(sbatch --parsable ${HOLD:+-H} ${DEP} "slurm/${name}.sbatch")
        IDS+=("${name} ${J}"); HELD_IDS+=("${J}")
        printf '%-15s -> job %s %s%s\n' "${name}" "${J}" "${DEP:+(${DEP#--dependency=})}" "${HOLD:+ HELD}"
    done
fi

{ echo "# $(date '+%F %T') submit_chain.sh $*"; printf '%s\n' "${IDS[@]}"; } >> outputs/slurm_logs/jobids.txt
echo
echo "JOBIDS: $(printf '%s ' "${IDS[@]}" | sed 's/ \+/ /g')"
echo "logs:   ${REMOTE_PKG}/outputs/slurm_logs/<name>-<jobid>.out    (bash slurm/status.sh)"
if [ -n "${HOLD}" ] && [ ${#HELD_IDS[@]} -gt 0 ]; then
    echo
    echo "01-04 are HELD (JobHeldUser). When 00_smoke and 00b_timing have finished, read the"
    echo "TIMING line (outputs/v2_country_timing/timing.json: projected_full_run_h is for 4 chains"
    echo "vectorized on one full H100 = the main jobs' layout) and release them with:"
    echo "    scontrol release ${HELD_IDS[*]}        (on rorqual)"
    echo "    bash slurm/submit_chain.sh --release   (from the laptop: releases all held jobs)"
fi
squeue -u "${USER}" -o "%.10i %.16j %.4t %.10M %.11l %.6D %R" 2>/dev/null || true
