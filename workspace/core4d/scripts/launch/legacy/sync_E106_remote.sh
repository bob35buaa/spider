#!/usr/bin/env bash
# Sync E106 code/data to spider-remote without relying on git pull.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
TASK_ROOT="example_datasets/processed/core4d/unitree_g1/humanoid_object"

echo "[$(date '+%H:%M:%S')] === sync E106 to ${REMOTE}:${REMOTE_REPO} ==="

rsync -av \
  workspace/core4d/scripts/E106/ \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/E106/"

rsync -av \
  workspace/core4d/scripts/run_E106_remote.sh \
  workspace/core4d/scripts/pull_E106_remote_results.sh \
  workspace/core4d/scripts/sync_E106_remote.sh \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/"

rsync -av \
  workspace/core4d/scripts/train/train_E106_box026_candidate_batch.sh \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/train/"

rsync -av \
  workspace/core4d/scripts/convert/snapshot_scenes.sh \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/convert/"

rsync -av \
  workspace/core4d/scripts/eval/eval_E106_box026_candidate_batch.py \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/eval/"

rsync -av \
  examples/config/override/ \
  "${REMOTE}:${REMOTE_REPO}/examples/config/override/"

rsync -av \
  example_datasets/processed/core4d/assets/objects/box026/ \
  "${REMOTE}:${REMOTE_REPO}/example_datasets/processed/core4d/assets/objects/box026/"

mapfile -t tasks < <(
  awk -F '\t' 'NF && $1 !~ /^#/ {print $4}' workspace/core4d/scripts/E106/variants.tsv | sort -u
)
if [ "${#tasks[@]}" -eq 0 ]; then
  echo "No E106 tasks found in variants.tsv" >&2
  exit 2
fi

for task in "${tasks[@]}"; do
  rsync -a --delete \
    "${TASK_ROOT}/${task}/" \
    "${REMOTE}:${REMOTE_REPO}/${TASK_ROOT}/${task}/"
done

mkdir -p workspace/core4d/results/E106
ssh "$REMOTE" "mkdir -p '${REMOTE_REPO}/workspace/core4d/results/E106'"

rsync -av \
  workspace/core4d/results/E106/preprocess_failures.tsv \
  workspace/core4d/results/E106/manifest_summary.md \
  workspace/core4d/results/E106/data_readiness.tsv \
  workspace/core4d/results/E106/clean_task_preflight.tsv \
  workspace/core4d/results/E106/pre_cem_visual_gate.tsv \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E106/"

rsync -av \
  workspace/core4d/results/E106/pre_cem_visual_review/ \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E106/pre_cem_visual_review/"

echo "[$(date '+%H:%M:%S')] === sync E106 done: tasks=${#tasks[@]} ==="
