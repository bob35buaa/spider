#!/usr/bin/env bash
# Wait for strict 28/28 recovery, then render and evaluate E169 locally.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

POLL_SECONDS="${E169_POSTPROCESS_POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${E169_POSTPROCESS_MAX_WAIT_SECONDS:-86400}"
SUMMARY="workspace/core4d/results/E169/artifacts/full/artifact_summary.json"
EVAL_DIR="workspace/core4d/results/E169/eval/full"
XLSX="$EVAL_DIR/E169_lowerbody_object_factorial_metrics.xlsx"
START_SECONDS="$(date +%s)"

strict_recovery_pass() {
  [ -f "$SUMMARY" ] || return 1
  .venv/bin/python - "$SUMMARY" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(
    0
    if payload.get("status") == "pass"
    and payload.get("expected_rows") == 28
    and payload.get("complete_rows") == 28
    and not payload.get("incomplete")
    else 1
)
PY
}

while ! strict_recovery_pass; do
  now="$(date +%s)"
  if [ "$((now - START_SECONDS))" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "E169 postprocess timed out waiting for strict 28/28 recovery" >&2
    exit 3
  fi
  echo "[$(date -Is)] waiting for E169 strict 28/28 recovery"
  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] E169 strict recovery passed; starting local render"
MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python \
  workspace/core4d/scripts/experiments/E169/render_factorial_results.py \
  --stage full --montage

echo "[$(date -Is)] starting E169 full evaluation"
MUJOCO_GL="${MUJOCO_GL:-egl}" bash \
  workspace/core4d/scripts/eval/wrappers/eval_E169_lowerbody_factorial.sh full

echo "[$(date -Is)] generating E169 workbook"
python workspace/core4d/scripts/eval/reports/gen_E169_lowerbody_factorial_xlsx.py \
  --eval-dir "$EVAL_DIR" --output "$XLSX"
RECALC_JSON="$(python "$HOME/.codex/skills/xlsx/scripts/recalc.py" "$XLSX" 120)"
printf '%s\n' "$RECALC_JSON"
RECALC_JSON="$RECALC_JSON" .venv/bin/python - <<'PY'
import json, os
payload = json.loads(os.environ["RECALC_JSON"])
if payload.get("status") != "success" or payload.get("total_errors") != 0:
    raise SystemExit(f"E169 workbook formula validation failed: {payload}")
PY

printf '%s\n' "completed_at=$(date -Is)" > \
  workspace/core4d/results/E169/POSTPROCESS_COMPLETE.txt
echo "[$(date -Is)] E169 automatic postprocess complete"
