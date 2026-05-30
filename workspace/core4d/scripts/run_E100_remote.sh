#!/usr/bin/env bash
# E100 干净 A/B 远程双卡执行脚本 — 24-step mini CEM
#
# 2 case × 2 target source × 3 seed = 12 run
#   case1: d003_box021_20231018_029_p2 (H1 主战场)
#     A: palm-based target (E085 raw target NPZ, 已存在)
#     B: fingertip-based target (E100 new NPZ)
#   case2: box023_person2 (守门反向)
#     A: ref_fk baseline (无 NPZ)
#     B: fingertip-based target (E100 new NPZ)
#
# 双 GPU 并行：GPU0 跑 case1, GPU1 跑 case2；每个 case 3 seed 串行
#
# 使用前提：GPU 至少有 12 GiB 空闲；当前 spider 训练通常占用 16-22 GiB
#
# 输出：logs/core4d/E100/{variant}/seed{N}/

set -euo pipefail
cd "$(dirname "$0")/../../../"

PY=.venv/bin/python
LOG_ROOT=logs/core4d/E100
mkdir -p "$LOG_ROOT"

# 关键参数（与 E085 / E094 等同口径）
NUM_ITER="${NUM_ITER:-24}"      # mini CEM 24 步
NUM_SEED="${NUM_SEED:-3}"
NUM_ENVS="${NUM_ENVS:-256}"

run_one() {
  local variant=$1
  local override=$2
  local gpu=$3
  for seed in $(seq 0 $((NUM_SEED-1))); do
    local out_dir="$LOG_ROOT/${variant}/seed${seed}"
    mkdir -p "$out_dir"
    echo "[GPU${gpu} seed${seed}] $variant -> $out_dir"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl $PY examples/run_mjwp.py \
      +override="$override" \
      seed=$seed \
      num_iters=$NUM_ITER \
      num_envs=$NUM_ENVS \
      output_dir="$out_dir" \
      viewer=null \
      2>&1 | tee "$out_dir/train.log"
  done
}

# Case 1 (18029_p2) 在 GPU 0
(
  run_one E100_18029_palmbase   core4d_E100_d003_box021_20231018_029_p2_palmbase   0
  run_one E100_18029_fingertip  core4d_E100_d003_box021_20231018_029_p2_fingertip  0
) &
GPU0_PID=$!

# Case 2 (box023_p2) 在 GPU 1
(
  run_one E100_box023p2_reffk      core4d_E100_box023_person2_reffk       1
  run_one E100_box023p2_fingertip  core4d_E100_box023_person2_fingertip   1
) &
GPU1_PID=$!

wait $GPU0_PID
wait $GPU1_PID
echo "ALL E100 runs done. Results under $LOG_ROOT/"
