#!/usr/bin/env bash
# E200 · G1A2 立即打分（不等剩余 CEM）：eval 现有完成条 -> funnel 分层 -> 刷新三 arm master xlsx。
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY=.venv/bin/python
DONE=workspace/core4d/results/E200/s6_downstream/eval/.g1a2_master_DONE
rm -f "$DONE"
echo "[eval-now $(date '+%H:%M:%S')] eval prg_g1a2 (现有完成条)"
"$PY" workspace/core4d/scripts/eval/runners/eval_E200_arm_augmentation.py --arms prg_g1a2
echo "[eval-now $(date '+%H:%M:%S')] classify"
"$PY" workspace/core4d/scripts/experiments/E201/classify_funnel.py --exp E200_prg_g1a2
echo "[eval-now $(date '+%H:%M:%S')] gen master xlsx"
"$PY" workspace/core4d/scripts/eval/reports/gen_E200_three_arm_master_xlsx.py
touch "$DONE"
echo "[eval-now $(date '+%H:%M:%S')] DONE"
