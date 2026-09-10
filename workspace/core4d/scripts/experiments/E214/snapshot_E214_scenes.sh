#!/usr/bin/env bash
# E214 -- rules/experiment.md §7 Safeguard 2.
#
# E214 re-uses each paper case's baseline scene_act (read-only) via
# load_config_path; no scene XML is created or edited.  This snapshots the exact
# scene XML (the config's own model_path, resolved locally) + adjacent metadata
# for all 50 cases into results/E214/scene_snapshot/<case>/, with a manifest
# recording git HEAD + sha256, so the training state stays recoverable.
#
# Run BEFORE the CEM batch (train script step 1).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DST="workspace/core4d/results/E214/scene_snapshot"
mkdir -p "$DST"
MANIFEST="$DST/manifest.txt"
{
  echo "# Scene snapshot for E214 (four-ablation study)"
  echo "# Captured: $(date -Iseconds)"
  echo "# Git HEAD: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "# Source: each case's baseline scene_act (model_path in its config_act.yaml), read-only reuse."
  echo
  printf "%-34s %-52s %10s  %s\n" "case" "file" "size" "sha256"
} > "$MANIFEST"

# Enumerate (case_id, resolved scene xml) from the manifest's run_model_path.
mapfile -t PAIRS < <(
  MUJOCO_GL=disable .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "workspace/core4d/scripts/experiments/E214")
import e214_common as C
seen = set()
for case in C.load_cases():
    bp = C.baseline_paths(case)
    ri = bp.get("run_inputs", {})
    mp = ri.get("model_path")
    if mp is None:
        continue
    p = mp.resolve()
    if case in seen:
        continue
    seen.add(case)
    print(f"{case}\t{p}")
PY
)

if [ "${#PAIRS[@]}" -eq 0 ]; then
  echo "ERROR: no scenes enumerated" >&2
  exit 1
fi

n=0
for pair in "${PAIRS[@]}"; do
  case_id="${pair%%$'\t'*}"
  scene="${pair#*$'\t'}"
  if [ ! -f "$scene" ]; then
    echo "ERROR: scene not found for $case_id: $scene" >&2
    exit 1
  fi
  dst_dir="$DST/$case_id"
  mkdir -p "$dst_dir"
  src_dir="$(dirname "$scene")"
  # scene xml + any adjacent scene metadata (mesh/robot includes stay referenced by path)
  files=("$scene")
  for opt in "$src_dir/scene_act_meta.json" "$src_dir/task_info.json"; do
    [ -f "$opt" ] && files+=("$opt")
  done
  for src in "${files[@]}"; do
    f="$(basename "$src")"
    cp -f "$src" "$dst_dir/$f"
    size=$(stat -c%s "$src")
    sha=$(sha256sum "$src" | awk '{print $1}')
    printf "%-34s %-52s %10d  %s\n" "$case_id" "$f" "$size" "$sha" >> "$MANIFEST"
  done
  n=$((n + 1))
done

echo "[E214] snapshotted $n case scenes -> $DST"
echo "[E214] manifest: $MANIFEST"
