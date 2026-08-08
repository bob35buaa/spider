#!/usr/bin/env bash
# Detached watcher: wait until the E194 Full CEM launcher finishes, then run the
# four-cell eval + C1-C9 report automatically. Survives the launching session
# (start with: setsid nohup ... &). Polls the Full launch log for its completion
# marker; falls back to a 24h timeout.
#
#   setsid nohup bash workspace/core4d/scripts/launch/active/watch_and_eval_E194_full.sh \
#     > logs/E194_auto_eval.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="${PYTHON_BIN:-.venv/bin/python}"
LAUNCH_LOG=logs/E194_full_launch.log
unset MUJOCO_GL

echo "[watch] started $(date -Iseconds); waiting for Full CEM to finish ..."
for _ in $(seq 1 288); do          # 288 * 5min = 24h cap
  if grep -q "full CEM done" "$LAUNCH_LOG" 2>/dev/null; then
    echo "[watch] Full CEM finished at $(date -Iseconds); running eval + report"
    done_ct=$(find workspace/core4d/results/E194/s6_downstream/cem/full -name 'trajectory_mjwp_act.npz' 2>/dev/null | wc -l)
    echo "[watch] completed Full rollouts on disk: ${done_ct}/45"
    bash workspace/core4d/scripts/eval/wrappers/eval_E194_gravcomp_arms.sh full
    "$PY" workspace/core4d/scripts/eval/reports/gen_E194_arm_comparison.py full
    echo "[watch] eval+report done at $(date -Iseconds); starting render (osmesa CPU)"
    MUJOCO_GL=osmesa bash workspace/core4d/scripts/launch/active/run_E194_render_all.sh
    echo "[watch] DONE eval+report+render at $(date -Iseconds)"
    echo "[watch] report:    workspace/core4d/results/E194/s6_downstream/eval/full/E194_arm_comparison.md"
    echo "[watch] montages:  workspace/core4d/results/E194/s6_downstream/render/full/keyframes/"
    exit 0
  fi
  sleep 300
done
echo "[watch] TIMED OUT after 24h waiting for Full CEM done marker" >&2
exit 1
