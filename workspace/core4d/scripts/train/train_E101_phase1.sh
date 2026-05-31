#!/usr/bin/env bash
# E101 Phase 1: typical 3 case full CEM 双卡并行验证。
#
# H1 控制实验 (face_changed=False, fingertip target ≈ palm target):
#   GPU0 seed0: 18029_p2 (E082-E088 反复 FAIL, H1 主战场)
#   GPU0 seed1: 035_p2   (E090 S1 FAIL, pelvis 0.134m)
#
# Face-changed 测试 (face_changed_R: -z→-x):
#   GPU1 seed0/1: box004_083_p2 (E091 SMOKE FAIL / E094 P1 FAIL)
#
# Full CEM default (max_num_iterations=32, num_samples=1024)；预计 ~1-2.5h per run
# 双卡并行 → ~3-5h total
#
# 必须用 +use_torch_compile=false (Phase 0 已验证)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=.venv/bin/python
RESULTS=workspace/core4d/results/E101/phase1
LOGS=logs/core4d/E101/phase1
mkdir -p "$RESULTS" "$LOGS"

run_one() {
  local variant=$1
  local override=$2
  local task=$3
  local seed=$4
  local gpu=$5
  local out_dir="$RESULTS/${variant}_seed${seed}_outdir"
  local log="$LOGS/${variant}_seed${seed}.log"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === GPU${gpu} seed${seed} ${variant} task=${task} ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 $PY -u examples/run_mjwp.py \
    +override="$override" \
    task="$task" \
    seed=$seed \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${variant}_seed${seed}.mp4" \
    > "$log" 2>&1
  if [ -f "$out_dir/trajectory_mjwp_act.npz" ]; then
    cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}_seed${seed}.npz"
  fi
  echo "[$(date '+%H:%M:%S')] === GPU${gpu} seed${seed} ${variant} done ==="
}

# GPU 0: 18029_p2 seed0 + 035_p2 seed0 (H1 控制实验, 各 1 seed)
(
  run_one "E101P1_box021_18029_p2_fingertip" \
          "core4d_E100_d003_box021_20231018_029_p2_fingertip" \
          "d003_box021_20231018_029_p2_upperobj_e083" \
          0 0
  run_one "E101P1_box021_11035_p2_fingertip" \
          "core4d_E101_d003_box021_20231011_035_p2_fingertip" \
          "d003_box021_20231011_035_p2_upperobj_e083" \
          0 0
) &
GPU0_PID=$!

# GPU 1: box004_083_p2 2 seed (face-changed 主要测试)
(
  for seed in 0 1; do
    run_one "E101P1_box004_083_p2_fingertip" \
            "core4d_E101_e091_box004_20231003_2_083_p2_fingertip" \
            "e091_box004_20231003_2_083_p2_e092_dyn" \
            "$seed" 1
  done
) &
GPU1_PID=$!

wait $GPU0_PID
wait $GPU1_PID
echo ""
echo "[$(date '+%H:%M:%S')] === E101 Phase 1 ALL DONE ==="
ls -la "$RESULTS"/*.npz "$RESULTS"/*.mp4 2>/dev/null
