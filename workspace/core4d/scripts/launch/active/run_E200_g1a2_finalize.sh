#!/usr/bin/env bash
# E200 · G1A2 收尾：等 PRG+G1+A2 CEM 全部完成 -> 打分(249) -> funnel 分层 -> 刷新三 arm master xlsx。
# 后台 nohup 跑；完成后 touch .g1a2_master_DONE。
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY=.venv/bin/python
M=workspace/core4d/results/E200/s6_downstream/manifests/e200_prg_g1a2_priority_manifest.tsv
DONE=workspace/core4d/results/E200/s6_downstream/eval/.g1a2_master_DONE
rm -f "$DONE"

echo "[finalize $(date '+%H:%M:%S')] 等 G1A2 CEM 全 done ..."
while :; do
  st=$("$PY" -c "import csv;from collections import Counter;r=list(csv.DictReader(open('$M'),delimiter='\t'));c=Counter(x['status'] for x in r);print(c.get('',0)+c.get('running',0))")
  [ "$st" = "0" ] && break
  echo "[finalize $(date '+%H:%M:%S')] CEM 未完成待跑=$st"; sleep 120
done
echo "[finalize $(date '+%H:%M:%S')] CEM 全完成，开始打分(249) ..."
"$PY" workspace/core4d/scripts/eval/runners/eval_E200_arm_augmentation.py --arms prg_g1a2
echo "[finalize $(date '+%H:%M:%S')] funnel 分层 ..."
"$PY" workspace/core4d/scripts/experiments/E201/classify_funnel.py --exp E200_prg_g1a2
echo "[finalize $(date '+%H:%M:%S')] 刷新三 arm master xlsx ..."
"$PY" workspace/core4d/scripts/eval/reports/gen_E200_three_arm_master_xlsx.py
echo "[finalize $(date '+%H:%M:%S')] DONE"
touch "$DONE"
