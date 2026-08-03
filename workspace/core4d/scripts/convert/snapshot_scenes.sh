#!/usr/bin/env bash
# Snapshot scene XML files used by an experiment into the experiment's results dir.
# This creates a frozen copy that gets committed alongside the experiment log,
# so the exact scene state used for E0NN can always be recovered later.
#
# Usage:
#   bash workspace/core4d/scripts/convert/snapshot_scenes.sh E060 box023_person1 bucket005_s2_person1
#
# Output:
#   workspace/core4d/results/E060/scene_snapshot/{case_name}/scene*.xml
#   workspace/core4d/results/E060/scene_snapshot/manifest.txt  (case + sha256 + size)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <EXP_ID> <case1> [case2 ...]" >&2
  echo "Example: $0 E060 box023_person1 bucket005_s2_person1" >&2
  exit 1
fi

EXP_ID="$1"; shift
DST="workspace/core4d/results/$EXP_ID/scene_snapshot"
SRC_BASE="example_datasets/processed/core4d/unitree_g1/humanoid_object"
mkdir -p "$DST"
MANIFEST="$DST/manifest.txt"
{
  echo "# Scene snapshot for $EXP_ID"
  echo "# Captured: $(date -Iseconds)"
  echo "# Git HEAD: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo
  printf "%-32s %-20s %10s  %s\n" "case" "file" "size" "sha256"
} > "$MANIFEST"

for case_name in "$@"; do
  src_dir="$SRC_BASE/$case_name"
  if [ ! -d "$src_dir" ]; then
    echo "ERROR: case dir not found: $src_dir" >&2
    exit 1
  fi
  dst_dir="$DST/$case_name"
  mkdir -p "$dst_dir"
  shopt -s nullglob
  scene_files=("$src_dir"/scene*.xml)
  shopt -u nullglob
  if [ "${#scene_files[@]}" -eq 0 ]; then
    echo "ERROR: no scene XML found: $src_dir" >&2
    exit 1
  fi
  files=("${scene_files[@]}")
  for optional in scene_act_meta.json task_info.json; do
    [ -f "$src_dir/$optional" ] && files+=("$src_dir/$optional")
  done
  for src in "${files[@]}"; do
    f="$(basename "$src")"
    cp "$src" "$dst_dir/$f"
    size=$(stat -c%s "$src")
    sha=$(sha256sum "$src" | awk '{print $1}')
    printf "%-32s %-20s %10d  %s\n" "$case_name" "$f" "$size" "$sha" >> "$MANIFEST"
  done
done

echo "Snapshot written to $DST"
echo "Files:"
ls -la "$DST"
echo
echo "Manifest:"
cat "$MANIFEST"
