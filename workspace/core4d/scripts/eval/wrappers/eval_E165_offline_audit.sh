#!/bin/bash
# E165 Phase0 离线审计 (训练-free): A box004标签 / C box023溯源 / E1 on-rails探针 + 可视化。
# 用法: bash workspace/core4d/scripts/eval/wrappers/eval_E165_offline_audit.sh
set -euo pipefail
ROOT="/mnt/public/usr/yancilin/work_dir/embodied/spider"
PY="${E165_PY:-/mnt/public/usr/yancilin/work_dir/.holosoma_deps/miniconda3/envs/hssim/bin/python}"
R="$ROOT/workspace/core4d/scripts/eval/runners"
cd "$ROOT"

echo "== A: box004 contact label audit (+box021/box023 对照) =="
$PY "$R/eval_E165_box004_contact_audit.py" --case box004
$PY "$R/eval_E165_box004_contact_audit.py" --case box021 >/dev/null
$PY "$R/eval_E165_box004_contact_audit.py" --case box023 >/dev/null

echo "== C: box023 penetration trace (spider vs omni) =="
$PY "$R/eval_E165_box023_penetration_trace.py"

echo "== E1: on-rails probe scalars =="
$PY "$R/eval_E165_isaac_onrails_probe.py"

echo "== plots =="
$PY "$ROOT/workspace/core4d/scripts/eval/reports/gen_E165_offline_plots.py"
echo "Done. results: workspace/core4d/results/E165/"
