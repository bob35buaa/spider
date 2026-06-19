#!/usr/bin/env bash
# Queue E166 remaining4 A/A_B2 after current three-case A/A_B2 SUGAR completes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

POLL_INTERVAL="${POLL_INTERVAL:-1200}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"
REMOTE="${REMOTE:-spider-remote}"
REMOTE_SPIDER_ROOT="${REMOTE_SPIDER_ROOT:-/home/xiayb/pHRI_workspace/spider}"
REMOTE_SUGAR_ROOT="${REMOTE_SUGAR_ROOT:-/home/xiayb/pHRI_workspace/Loco-Manipulation-projects/SUGAR-private}"
LOCAL_SUGAR_ROOT="${LOCAL_SUGAR_ROOT:-/home/ubuntu/Workspace/Loco-Manipulation/SUGAR}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CURRENT_OUT="${LOCAL_SUGAR_ROOT}/outputs/core4d/e166_foot_smooth_refiner_rl"
REM4_OUT="${LOCAL_SUGAR_ROOT}/outputs/core4d/e166_remaining4_A_B2_refiner_rl"
STARTED="$(date +%s)"

log() {
  echo "[$(date '+%F %T')] $*"
}

pull_current() {
  bash "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/pull_core4d_e166_refiner_remote.sh" || true
}

pull_remaining4_cem_stage() {
  local stage="${1:-full}"
  bash workspace/core4d/scripts/launch/active/pull_E166_remaining4_remote_results.sh "$stage" || true
}

pull_remaining4() {
  pull_remaining4_cem_stage full
  bash "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/pull_core4d_e166_remaining4_refiner_remote.sh" || true
}

current_done_counts() {
  python - "$CURRENT_OUT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = [
    ("box021_035_p2", "A", "box021_035_p2_E166_A_6000"),
    ("box021_035_p2", "A_B2_postSmooth", "box021_035_p2_E166_A_B2_postSmooth_6000"),
    ("box004_082_p1", "A", "box004_082_p1_E166_A_6000"),
    ("box004_082_p1", "A_B2_postSmooth", "box004_082_p1_E166_A_B2_postSmooth_6000"),
    ("box004_083_p2", "A", "box004_083_p2_E166_A_6000"),
    ("box004_083_p2", "A_B2_postSmooth", "box004_083_p2_E166_A_B2_postSmooth_6000"),
]
ckpt = 0
csv = 0
for case, arm, run in rows:
    if (root / case / arm / "train/logs" / run / "model_5999.pt").is_file():
        ckpt += 1
    if (root / case / arm / "eval_staggered_phase_mw30/analysis/success_vs_phase.csv").is_file():
        csv += 1
print(f"{ckpt} {csv}")
PY
}

wait_current_done() {
  while true; do
    local now elapsed ckpt_count csv_count
    now="$(date +%s)"
    elapsed="$((now - STARTED))"
    if [ "$elapsed" -gt "$MAX_WAIT_SECONDS" ]; then
      log "timeout waiting for current E166 A/A_B2 after ${elapsed}s"
      exit 3
    fi
    pull_current
    read -r ckpt_count csv_count < <(current_done_counts)
    log "current A/A_B2 final_ckpt=${ckpt_count}/6 success_csv=${csv_count}/6 elapsed=${elapsed}s"
    if [ "$ckpt_count" = "6" ] && [ "$csv_count" = "6" ]; then
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
}

cem_done_counts() {
  local stage="${1:-full}"
  python - "$stage" <<'PY'
from pathlib import Path
import sys
stage = sys.argv[1]
root = Path("workspace/core4d/results/E166/foot_smooth_retarget")
cases = ["box023_person2", "box021_029_p2", "box021_035_p1", "box004_083_p1"]
cem = 0
post = 0
for case in cases:
    v = f"E166_{case}_A"
    if (root / f"cem/{stage}" / f"{v}.npz").is_file() and (root / f"cem/{stage}" / f"{v}_{stage}.mp4").is_file():
        cem += 1
    if (root / "postprocess/full" / f"E166_{case}_A_B2_postSmooth.npz").is_file():
        post += 1
print(f"{cem} {post}")
PY
}

start_remaining4_cem_stage() {
  local stage="${1:-full}"
  local local_session="E166_remaining4_local_${stage}"
  local remote_session="E166_remaining4_remote_${stage}"
  if tmux has-session -t "$local_session" 2>/dev/null; then
    log "local CEM session already exists: ${local_session}"
  else
    tmux new-session -d -s "$local_session" \
      "cd $(pwd) && E166_SPLIT=local-gpu0 LOCAL_GPU=0 bash workspace/core4d/scripts/launch/active/run_E166_remaining4_local.sh ${stage}"
    log "started local CEM session ${local_session}"
  fi
  if ssh "$REMOTE" "tmux has-session -t '$remote_session' 2>/dev/null"; then
    log "remote CEM session already exists: ${remote_session}"
  else
    SESSION="$remote_session" bash workspace/core4d/scripts/launch/active/run_E166_remaining4_remote.sh "$stage"
  fi
}

wait_remaining4_cem_stage() {
  local stage="${1:-full}"
  start_remaining4_cem_stage "$stage"
  while true; do
    local cem_count post_count
    pull_remaining4_cem_stage "$stage"
    read -r cem_count post_count < <(cem_done_counts "$stage")
    log "remaining4 ${stage} CEM=${cem_count}/4 postprocess=${post_count}/4"
    if [ "$cem_count" = "4" ]; then
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
}

wait_remaining4_cem_and_post() {
  wait_remaining4_cem_stage full
  while true; do
    local cem_count post_count
    read -r cem_count post_count < <(cem_done_counts full)
    log "remaining4 full CEM=${cem_count}/4 postprocess=${post_count}/4"
    if [ "$cem_count" = "4" ]; then
      bash workspace/core4d/scripts/launch/active/run_E166_remaining4_postprocess.sh full
      read -r cem_count post_count < <(cem_done_counts full)
      log "remaining4 postprocess after run=${post_count}/4"
      if [ "$post_count" = "4" ]; then
        return 0
      fi
    fi
    sleep "$POLL_INTERVAL"
  done
}

eval_and_export_remaining4() {
  "$PYTHON_BIN" workspace/core4d/scripts/eval/runners/eval_E166_foot_smooth_retarget.py full \
    --variants workspace/core4d/scripts/experiments/E166/remaining4_variants.tsv \
    --eval-dir workspace/core4d/results/E166/foot_smooth_retarget/eval/remaining4
  bash workspace/core4d/scripts/launch/active/run_E166_remaining4_sugar_export.sh
}

sync_remaining4_sugar_remote() {
  rsync -az "${LOCAL_SUGAR_ROOT}/scripts/data_preprocess/convert_core4d_e166_remaining4_manifest_to_sugar.py" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/scripts/data_preprocess/convert_core4d_e166_remaining4_manifest_to_sugar.py"
  rsync -az "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh"
  rsync -az "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/eval_staggered_phase_e166_remaining4.sh" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/scripts/sugar_rl/eval_staggered_phase_e166_remaining4.sh"
  rsync -az "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/pull_core4d_e166_remaining4_refiner_remote.sh" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/scripts/sugar_rl/pull_core4d_e166_remaining4_refiner_remote.sh"
  rsync -az "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/summarize_core4d_e166_remaining4_failed_windows.py" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/scripts/sugar_rl/summarize_core4d_e166_remaining4_failed_windows.py"
  rsync -az "${LOCAL_SUGAR_ROOT}/data/Core4D_E166_A_B2_postSmooth_Box021_029_p2" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/data/"
  rsync -az "${LOCAL_SUGAR_ROOT}/data/Core4D_E166_A_B2_postSmooth_Box004_083_p1" \
    "${REMOTE}:${REMOTE_SUGAR_ROOT}/data/"
  ssh "$REMOTE" "cd '$REMOTE_SUGAR_ROOT' && bash -n scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh scripts/sugar_rl/eval_staggered_phase_e166_remaining4.sh && python3 -m py_compile scripts/data_preprocess/convert_core4d_e166_remaining4_manifest_to_sugar.py scripts/sugar_rl/summarize_core4d_e166_remaining4_failed_windows.py"
}

start_remaining4_sugar() {
  sync_remaining4_sugar_remote
  if tmux has-session -t E166_remaining4_sugar_local 2>/dev/null; then
    log "local SUGAR session already exists"
  else
    tmux new-session -d -s E166_remaining4_sugar_local \
      "cd '$LOCAL_SUGAR_ROOT' && bash scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh local"
    log "started local SUGAR session E166_remaining4_sugar_local"
  fi
  if ssh "$REMOTE" "tmux has-session -t E166_remaining4_sugar_remote 2>/dev/null"; then
    log "remote SUGAR session already exists"
  else
    ssh "$REMOTE" "cd '$REMOTE_SUGAR_ROOT' && tmux new-session -d -s E166_remaining4_sugar_remote \"bash -lc 'set -euo pipefail; (bash scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh remote_gpu0) & PID0=\\\$!; (bash scripts/sugar_rl/launch_core4d_e166_remaining4_refiner.sh remote_gpu1) & PID1=\\\$!; echo launched remaining4 sugar PID0=\\\$PID0 PID1=\\\$PID1; wait \\\$PID0; echo remote_gpu0 done; wait \\\$PID1; echo remote_gpu1 done'\""
    log "started remote SUGAR session E166_remaining4_sugar_remote"
  fi
}

remaining4_sugar_done_counts() {
  python - "$REM4_OUT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
rows = [
    ("box023_person2", "box023_person2_E166_A_B2_postSmooth_6000"),
    ("box021_035_p1", "box021_035_p1_E166_A_B2_postSmooth_6000"),
    ("box021_029_p2", "box021_029_p2_E166_A_B2_postSmooth_6000"),
    ("box004_083_p1", "box004_083_p1_E166_A_B2_postSmooth_6000"),
]
ckpt = 0
csv = 0
for case, run in rows:
    arm = "A_B2_postSmooth"
    if (root / case / arm / "train/logs" / run / "model_5999.pt").is_file():
        ckpt += 1
    if (root / case / arm / "eval_staggered_phase_mw30/analysis/success_vs_phase.csv").is_file():
        csv += 1
print(f"{ckpt} {csv}")
PY
}

wait_remaining4_sugar() {
  start_remaining4_sugar
  while true; do
    local ckpt_count csv_count
    pull_remaining4
    read -r ckpt_count csv_count < <(remaining4_sugar_done_counts)
    log "remaining4 SUGAR final_ckpt=${ckpt_count}/4 success_csv=${csv_count}/4"
    if [ "$ckpt_count" = "4" ] && [ "$csv_count" = "4" ]; then
      python "${LOCAL_SUGAR_ROOT}/scripts/sugar_rl/summarize_core4d_e166_remaining4_failed_windows.py"
      log "remaining4 complete: ${REM4_OUT}/comparison"
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
}

log "E166 remaining4 queue started poll=${POLL_INTERVAL}s"
"$PYTHON_BIN" workspace/core4d/scripts/experiments/E166/build_remaining4_A_A_B2_manifest.py
wait_current_done
wait_remaining4_cem_stage smoke
wait_remaining4_cem_and_post
eval_and_export_remaining4
wait_remaining4_sugar
