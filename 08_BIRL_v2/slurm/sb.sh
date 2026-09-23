#!/bin/bash
# sbatch wrapper: adds --account and --output from env.sh, because #SBATCH
# directives cannot expand variables and env.sh (cluster user / allocation)
# is not tracked in git.  Usage, on the cluster, from 08_BIRL_v2/:
#   bash slurm/sb.sh [sbatch options] slurm/<job>.sbatch
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "${HERE}/env.sh" ] || { echo "missing ${HERE}/env.sh: copy env.example.sh to env.sh and fill in user/account" >&2; exit 1; }
source "${HERE}/env.sh"
mkdir -p "${REMOTE_PKG}/outputs/slurm_logs"
exec sbatch --account="${ACCOUNT}" --output="${REMOTE_PKG}/outputs/slurm_logs/%x-%j.out" "$@"
