#!/usr/bin/env bash
# Launch E151 local split in tmux.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
SESSION="${SESSION:-E151_local_${STAGE}_$(date +%H%M%S)}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

.venv/bin/python workspace/core4d/scripts/E151/build_route_b_manifest.py >/tmp/e151_manifest_build_local.log
.venv/bin/python workspace/core4d/scripts/E151/check_mesh_sdf_and_targets.py >/tmp/e151_preflight_local.log
bash -n workspace/core4d/scripts/train/train_E151_route_b_hand_surface_contact.sh
bash workspace/core4d/scripts/train/train_E151_route_b_hand_surface_contact.sh list "$STAGE" 0 local-gpu0 >/tmp/e151_local_gpu0.list
echo "local-gpu0=$(wc -l </tmp/e151_local_gpu0.list)"

tmux new-session -d -s "$SESSION" "bash -lc 'set -euo pipefail; \
  cd \"$(pwd)\"; \
  echo E151 local session $SESSION stage=$STAGE; \
  bash workspace/core4d/scripts/train/train_E151_route_b_hand_surface_contact.sh local-gpu0 $STAGE 0; \
  echo E151 local complete'"

echo "Started local tmux session: ${SESSION}"
echo "Monitor: tmux capture-pane -t ${SESSION} -p | tail -80"
