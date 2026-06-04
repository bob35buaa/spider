#!/usr/bin/env bash
# E124 remote runner: SBTO carry-horizon smoke/full on spider-remote GPUs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-smoke}"
LOG_ROOT="logs/E124/remote"
mkdir -p "$LOG_ROOT"

bash workspace/core4d/scripts/convert/snapshot_scenes.sh E124_remote_smoke \
  d003_box021_20231011_035_p2_e107_clean \
  d003_box021_20231011_035_p1_e107_clean \
  > "$LOG_ROOT/remote_scene_snapshot_${STAGE}.log" 2>&1 || true

(
  SMOKE_SBTO_MAX_ITER_PER_KNOT="${SMOKE_SBTO_MAX_ITER_PER_KNOT:-1}" \
  SMOKE_NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-512}" \
  bash workspace/core4d/scripts/train/train_E124_sbto_carry_horizon.sh remote-gpu0 "$STAGE" 0
) > "$LOG_ROOT/remote_gpu0_${STAGE}.log" 2>&1 &
PID0=$!

(
  SMOKE_SBTO_MAX_ITER_PER_KNOT="${SMOKE_SBTO_MAX_ITER_PER_KNOT:-1}" \
  SMOKE_NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-512}" \
  bash workspace/core4d/scripts/train/train_E124_sbto_carry_horizon.sh remote-gpu1 "$STAGE" 1
) > "$LOG_ROOT/remote_gpu1_${STAGE}.log" 2>&1 &
PID1=$!

echo "E124 remote ${STAGE}: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
wait "$PID0"
echo "E124 remote ${STAGE}: GPU0 done"
wait "$PID1"
echo "E124 remote ${STAGE}: GPU1 done"
