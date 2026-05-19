#!/usr/bin/env bash
# Snapshot generated scene XMLs and minimal task metadata for an experiment.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP_ID="${1:-}"
if [ -z "$EXP_ID" ]; then
  echo "Usage: $0 E0NN <case1> [case2 ...]" >&2
  exit 2
fi
shift
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 E0NN <case1> [case2 ...]" >&2
  exit 2
fi

BASE="example_datasets/processed/core4d/unitree_g1/humanoid_object"
OUT_ROOT="workspace/core4d_collab_retarget/results/${EXP_ID}/scene_snapshot"
mkdir -p "$OUT_ROOT"

manifest="$OUT_ROOT/manifest.txt"
{
  echo "experiment=${EXP_ID}"
  echo "timestamp_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_status_short_begin"
  git status --short || true
  echo "git_status_short_end"
  echo
} > "$manifest"

for case_name in "$@"; do
  src="${BASE}/${case_name}"
  if [ ! -d "$src" ]; then
    echo "Missing case directory: $src" >&2
    exit 1
  fi
  dst="${OUT_ROOT}/${case_name}"
  mkdir -p "$dst/0"
  find "$src" -maxdepth 1 -type f \( -name 'scene*.xml' -o -name '*_meta.json' -o -name 'task_info.json' \) -print0 |
    while IFS= read -r -d '' file; do
      cp "$file" "$dst/$(basename "$file")"
    done
  if [ -f "$src/0/trajectory_kinematic.npz" ]; then
    cp "$src/0/trajectory_kinematic.npz" "$dst/0/trajectory_kinematic.npz"
  fi
done

{
  echo "files_begin"
  find "$OUT_ROOT" -type f ! -name 'manifest.txt' -print0 |
    sort -z |
    xargs -0 sha256sum
  echo "files_end"
} >> "$manifest"

echo "Wrote $manifest"
