#!/usr/bin/env bash
# E152 remote launcher: run hand-gate split on remote GPU0/GPU1.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xiayb/pHRI_workspace/spider}"
SESSION="${SESSION:-E152_${STAGE}_$(date +%H%M%S)}"
HAND_GATE_MIN_SDF_M="${HAND_GATE_MIN_SDF_M:-${E152_HAND_GATE_MIN_SDF_M:--0.010}}"
HAND_GATE_MAX_VIOLATION_PCT="${HAND_GATE_MAX_VIOLATION_PCT:-${E152_HAND_GATE_MAX_VIOLATION_PCT:-0.05}}"
VARIANTS_FILE="${VARIANTS_FILE:-workspace/core4d/scripts/E152/variants.tsv}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

retry() {
  local attempt=1
  local max_attempts="${E152_REMOTE_RETRY_MAX:-4}"
  while true; do
    if "$@"; then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    echo "Retry ${attempt}/${max_attempts}: $*" >&2
    sleep "$((attempt * 3))"
    attempt=$((attempt + 1))
  done
}

.venv/bin/python workspace/core4d/scripts/E152/build_axis1_hand_gate_manifest.py \
  --hand-gate-min-sdf-m "$HAND_GATE_MIN_SDF_M" \
  --hand-gate-max-violation-pct "$HAND_GATE_MAX_VIOLATION_PCT" \
  >/tmp/e152_manifest_build.log
.venv/bin/python workspace/core4d/scripts/E152/check_hand_gate_preflight.py >/tmp/e152_preflight.log

echo "Syncing E152 artifacts to ${REMOTE}:${REMOTE_ROOT}"
retry ssh "$REMOTE" "mkdir -p \
  '$REMOTE_ROOT/spider/optimizers' \
  '$REMOTE_ROOT/spider/simulators' \
  '$REMOTE_ROOT/workspace/core4d/scripts/E152' \
  '$REMOTE_ROOT/workspace/core4d/scripts/train' \
  '$REMOTE_ROOT/workspace/core4d/scripts/eval' \
  '$REMOTE_ROOT/workspace/core4d/results/E152/axis1_hand_object_physics_gate/cem/${STAGE}' \
  '$REMOTE_ROOT/workspace/core4d/results/E143/contact_masks' \
  '$REMOTE_ROOT/logs/E152/cem/${STAGE}' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/assets/objects' \
  '$REMOTE_ROOT/example_datasets/processed/core4d/unitree_g1/humanoid_object' \
  '$REMOTE_ROOT/examples/config/override'"

retry rsync -az spider/config.py "${REMOTE}:${REMOTE_ROOT}/spider/config.py"
retry rsync -az spider/simulators/mjwp.py "${REMOTE}:${REMOTE_ROOT}/spider/simulators/mjwp.py"
retry rsync -az spider/optimizers/sampling.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling.py"
retry rsync -az spider/optimizers/sampling_fast.py "${REMOTE}:${REMOTE_ROOT}/spider/optimizers/sampling_fast.py"
retry rsync -az workspace/core4d/scripts/E152/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/E152/"
retry rsync -az workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/scripts/train/"
retry rsync -az workspace/core4d/results/E152/axis1_hand_object_physics_gate/ "${REMOTE}:${REMOTE_ROOT}/workspace/core4d/results/E152/axis1_hand_object_physics_gate/"
retry rsync -az examples/config/override/ "${REMOTE}:${REMOTE_ROOT}/examples/config/override/"

mapfile -t task_dirs < <(.venv/bin/python - <<'PY' "$VARIANTS_FILE"
import csv
import sys
from pathlib import Path
with open(sys.argv[1], encoding="utf-8", newline="") as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["run_status"] == "to_run"]
for row in sorted({str(Path(row["target_scene"]).parent) for row in rows}):
    print(row)
PY
)
for dir in "${task_dirs[@]}"; do
  [ -d "$dir" ] || continue
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

mapfile -t object_keys < <(.venv/bin/python - <<'PY' "$VARIANTS_FILE"
import csv
import sys
with open(sys.argv[1], encoding="utf-8", newline="") as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["run_status"] == "to_run"]
for key in sorted({r["object_key"] for r in rows}):
    print(key)
PY
)
for object_key in "${object_keys[@]}"; do
  retry rsync -az "example_datasets/processed/core4d/assets/objects/${object_key}/" \
    "${REMOTE}:${REMOTE_ROOT}/example_datasets/processed/core4d/assets/objects/${object_key}/"
done

mapfile -t mask_dirs < <(.venv/bin/python - <<'PY' "$VARIANTS_FILE"
import csv
import sys
from pathlib import Path
with open(sys.argv[1], encoding="utf-8", newline="") as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["run_status"] == "to_run"]
for path in sorted({str(Path(r["mask_path"]).parent) for r in rows if r.get("mask_path")}):
    print(path)
PY
)
for dir in "${mask_dirs[@]}"; do
  [ -d "$dir" ] || continue
  retry ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/$dir'"
  retry rsync -az "$dir/" "${REMOTE}:${REMOTE_ROOT}/${dir}/"
done

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  .venv/bin/python -m py_compile \
    spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/optimizers/sampling_fast.py \
    workspace/core4d/scripts/E152/build_axis1_hand_gate_manifest.py workspace/core4d/scripts/E152/check_hand_gate_preflight.py && \
  bash -n workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh && \
  E152_ALLOW_MISSING_REUSE=1 bash workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh list '$STAGE' 0 remote-gpu0 >/tmp/e152_gpu0.list && \
  E152_ALLOW_MISSING_REUSE=1 bash workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh list '$STAGE' 0 remote-gpu1 >/tmp/e152_gpu1.list && \
  echo remote-gpu0=\$(wc -l </tmp/e152_gpu0.list) remote-gpu1=\$(wc -l </tmp/e152_gpu1.list)"

retry ssh "$REMOTE" "cd '$REMOTE_ROOT' && \
  tmux new-session -d -s '$SESSION' \"bash -lc 'set -euo pipefail; \
    echo E152 remote session $SESSION stage=$STAGE hand_gate=$HAND_GATE_MIN_SDF_M/$HAND_GATE_MAX_VIOLATION_PCT; \
    (E152_ALLOW_MISSING_REUSE=1 bash workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh remote-gpu0 $STAGE 0) & PID0=\\\$!; \
    (E152_ALLOW_MISSING_REUSE=1 bash workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh remote-gpu1 $STAGE 1) & PID1=\\\$!; \
    echo launched PID0=\\\$PID0 PID1=\\\$PID1; \
    wait \\\$PID0; echo remote-gpu0 done; \
    wait \\\$PID1; echo remote-gpu1 done; \
    echo E152 remote complete'\""

echo "Started remote tmux session: ${SESSION}"
echo "Monitor: ssh ${REMOTE} \"tmux capture-pane -t ${SESSION} -p | tail -80\""
