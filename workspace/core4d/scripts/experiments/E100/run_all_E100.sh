#!/usr/bin/env bash
# E100 一键运行 (offline 部分)：target 生成 + 单测 + Tier 1 视觉对比
set -euo pipefail
cd "$(dirname "$0")/../../../../"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
PY=.venv/bin/python

echo "[1/3] build fingertip-aware target (全 17 case)"
$PY workspace/core4d/scripts/E100/build_fingertip_aware_target.py

echo "[2/3] unit tests (3/3)"
$PY workspace/core4d/scripts/E100/test_build_fingertip_aware_target.py

echo "[3/3] Tier 1 case 对比 PNG (9 case)"
$PY workspace/core4d/scripts/E100/render_target_compare.py

echo ""
echo "DONE. CEM A/B 实际运行见 workspace/core4d/scripts/run_E100_remote.sh (推迟到 E101 联合)"
ls workspace/core4d/results/E100/
