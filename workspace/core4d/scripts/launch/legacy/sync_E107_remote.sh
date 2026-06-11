#!/usr/bin/env bash
# Sync E107 selected-4 code/data to spider-remote without relying on git pull.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

REMOTE="${REMOTE:-spider-remote}"
REMOTE_REPO="${REMOTE_REPO:-/home/xiayb/pHRI_workspace/spider}"
TASK_ROOT="example_datasets/processed/core4d/unitree_g1/humanoid_object"

echo "[$(date '+%H:%M:%S')] === sync E107 to ${REMOTE}:${REMOTE_REPO} ==="

rsync -av \
  workspace/core4d/scripts/E107/ \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/E107/"

rsync -av \
  workspace/core4d/scripts/run_E107_remote.sh \
  workspace/core4d/scripts/pull_E107_remote_results.sh \
  workspace/core4d/scripts/sync_E107_remote.sh \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/"

rsync -av \
  workspace/core4d/scripts/train/train_E107_box021_selected4_full.sh \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/train/"

rsync -av \
  workspace/core4d/scripts/convert/snapshot_scenes.sh \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/convert/"

rsync -av \
  workspace/core4d/scripts/eval/eval_E107_box021_selected4.py \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/scripts/eval/"

rsync -av \
  examples/config/override/core4d_E107C*.yaml \
  "${REMOTE}:${REMOTE_REPO}/examples/config/override/"

rsync -av \
  example_datasets/processed/core4d/assets/objects/box021/ \
  "${REMOTE}:${REMOTE_REPO}/example_datasets/processed/core4d/assets/objects/box021/"

mapfile -t tasks < <(
  awk -F '\t' 'NF && $1 !~ /^#/ {print $4}' workspace/core4d/scripts/E107/selected4_variants.tsv | sort -u
)
if [ "${#tasks[@]}" -eq 0 ]; then
  echo "No E107 tasks found in selected4_variants.tsv" >&2
  exit 2
fi

for task in "${tasks[@]}"; do
  rsync -a --delete \
    "${TASK_ROOT}/${task}/" \
    "${REMOTE}:${REMOTE_REPO}/${TASK_ROOT}/${task}/"
done

ssh "$REMOTE" "mkdir -p '${REMOTE_REPO}/workspace/core4d/results/E107'"

rsync -av \
  workspace/core4d/results/E107/selected_case_to_cem.json \
  workspace/core4d/results/E107/box021_clean_gate_summary.tsv \
  workspace/core4d/results/E107/selected4_manifest_summary.md \
  workspace/core4d/results/E107/selected4_clean_task_preflight.tsv \
  workspace/core4d/results/E107/pre_cem_visual_gate.tsv \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E107/"

rsync -av \
  workspace/core4d/results/E107/pre_cem_visual_review/ \
  "${REMOTE}:${REMOTE_REPO}/workspace/core4d/results/E107/pre_cem_visual_review/"

echo "[$(date '+%H:%M:%S')] === sync E107 done: tasks=${#tasks[@]} ==="
