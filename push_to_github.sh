#!/bin/bash
# Push Formal Analysis to GitHub
# Run: bash push_to_github.sh

set -e

REPO_NAME="BIRL-Climate-Adaptation"
REPO_OWNER="juksentang"
VISIBILITY="public"  # change to "private" if needed

cd "$(dirname "$0")"

echo "=== Pre-flight checks ==="
echo "Files to commit:"
git status --short | head -20
echo "..."
echo "Total: $(git status --short | wc -l) files"
echo ""

read -p "Push to ${REPO_OWNER}/${REPO_NAME}? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
if gh repo view "${REPO_OWNER}/${REPO_NAME}" &>/dev/null; then
    echo "=== Repo already exists, skipping creation ==="
else
    echo "=== Creating ${VISIBILITY} repo ==="
    gh repo create "${REPO_OWNER}/${REPO_NAME}" \
        --${VISIBILITY} \
        --description "Bayesian Inverse Reinforcement Learning for Smallholder Climate Adaptation — 6 Sub-Saharan African Countries"
fi

echo ""
echo "=== Setting remote and pushing ==="
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
git branch -M main
git push -u origin main --force

echo ""
echo "=== Done ==="
echo "https://github.com/${REPO_OWNER}/${REPO_NAME}"
