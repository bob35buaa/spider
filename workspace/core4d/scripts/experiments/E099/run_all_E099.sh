#!/usr/bin/env bash
# E099 一键运行：自检 → fingertip vote → palm vote → quat audit → 可视化
set -euo pipefail
cd "$(dirname "$0")/../../../../"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
PY=.venv/bin/python

echo "[1/6] case_to_raw self-test"
$PY workspace/core4d/scripts/E099/case_to_raw.py

echo "[2/6] fingertip_face_vote unit tests"
$PY workspace/core4d/scripts/E099/test_fingertip_face_vote.py

echo "[3/6] fingertip vote (全 20 case)"
$PY workspace/core4d/scripts/E099/fingertip_face_vote.py

echo "[4/6] palm vote (全 20 case, 用于对比)"
$PY workspace/core4d/scripts/E099/palm_face_vote_full.py

echo "[5/6] quat 普查"
$PY workspace/core4d/scripts/E099/quat_identity_audit.py

echo "[6/6] raw_contact 3D 可视化 (PNG + MP4，~10 分钟)"
$PY workspace/core4d/scripts/E099/render_raw_contact_3d.py

echo ""
echo "DONE. 产物："
ls workspace/core4d/results/E099/
