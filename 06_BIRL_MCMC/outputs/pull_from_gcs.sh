#!/usr/bin/env bash
# Pull BIRL MCMC outputs from GCS via rclone
# Usage:
#   ./pull_from_gcs.sh                  # pull all variants
#   ./pull_from_gcs.sh hier_noalpha     # pull specific variant
#   ./pull_from_gcs.sh --list           # list remote files without downloading
#   ./pull_from_gcs.sh --no-pkl         # skip large .pkl files

set -euo pipefail

REMOTE="gcs:subsahra/birl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VARIANTS=("hier_noalpha" "R3")

# ── Parse flags ──
LIST_ONLY=false
SKIP_PKL=false
TARGETS=()

for arg in "$@"; do
    case "$arg" in
        --list)    LIST_ONLY=true ;;
        --no-pkl)  SKIP_PKL=true ;;
        -h|--help)
            echo "Usage: $0 [--list] [--no-pkl] [variant ...]"
            echo ""
            echo "Options:"
            echo "  --list     List remote files without downloading"
            echo "  --no-pkl   Skip large .pkl files (posterior.pkl, mcmc_state.pkl)"
            echo ""
            echo "Variants: ${VARIANTS[*]}"
            exit 0
            ;;
        *)         TARGETS+=("$arg") ;;
    esac
done

if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("${VARIANTS[@]}")
fi

# ── Check rclone ──
if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone not found. Install: curl https://rclone.org/install.sh | sudo bash" >&2
    exit 1
fi

# ── Build exclude filter ──
FILTER=()
if $SKIP_PKL; then
    FILTER+=(--exclude "*.pkl")
fi

# ── List mode ──
if $LIST_ONLY; then
    for variant in "${TARGETS[@]}"; do
        echo "=== ${variant} ==="
        rclone ls "${REMOTE}/${variant}/" "${FILTER[@]}" --human-readable 2>/dev/null || echo "  (empty or not found)"
        echo ""
    done
    exit 0
fi

# ── Download ──
for variant in "${TARGETS[@]}"; do
    dest="${SCRIPT_DIR}/${variant}"
    mkdir -p "$dest"
    echo "========================================"
    echo "Pulling ${variant} → ${dest}/"
    echo "========================================"

    rclone copy "${REMOTE}/${variant}/" "$dest/" \
        "${FILTER[@]}" \
        --progress \
        --transfers 4 \
        --multi-thread-streams 8 \
        --stats 1s

    echo ""
    echo "Done: ${variant}"
    echo ""
done

echo "All done. Files in: ${SCRIPT_DIR}/"
