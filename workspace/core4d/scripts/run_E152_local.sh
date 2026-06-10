#!/usr/bin/env bash
# Launch E152 local split in tmux.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
HAND_GATE_MIN_SDF_M="${HAND_GATE_MIN_SDF_M:-${E152_HAND_GATE_MIN_SDF_M:--0.010}}"
HAND_GATE_MAX_VIOLATION_PCT="${HAND_GATE_MAX_VIOLATION_PCT:-${E152_HAND_GATE_MAX_VIOLATION_PCT:-0.05}}"
SESSION="${SESSION:-E152_local_${STAGE}_$(date +%H%M%S)}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

.venv/bin/python workspace/core4d/scripts/E152/build_axis1_hand_gate_manifest.py \
  --hand-gate-min-sdf-m "$HAND_GATE_MIN_SDF_M" \
  --hand-gate-max-violation-pct "$HAND_GATE_MAX_VIOLATION_PCT" \
  >/tmp/e152_manifest_build_local.log
.venv/bin/python workspace/core4d/scripts/E152/check_hand_gate_preflight.py >/tmp/e152_preflight_local.log
bash -n workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh
bash workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh list "$STAGE" 0 local-gpu0 >/tmp/e152_local_gpu0.list
echo "local-gpu0=$(wc -l </tmp/e152_local_gpu0.list)"

tmux new-session -d -s "$SESSION" "bash -lc 'set -euo pipefail; \
  cd \"$(pwd)\"; \
  echo E152 local session $SESSION stage=$STAGE hand_gate=$HAND_GATE_MIN_SDF_M/$HAND_GATE_MAX_VIOLATION_PCT; \
  bash workspace/core4d/scripts/train/train_E152_axis1_hand_object_physics_gate.sh local-gpu0 $STAGE 0; \
  echo E152 local complete'"

echo "Started local tmux session: ${SESSION}"
echo "Monitor: tmux capture-pane -t ${SESSION} -p | tail -80"
