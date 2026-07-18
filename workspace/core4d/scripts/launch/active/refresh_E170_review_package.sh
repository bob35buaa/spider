#!/usr/bin/env bash
# Refresh the E170 workbook after Codex or user review without touching remote jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-}"
case "$STAGE" in
  pre_user) MARKER="PRE_USER_REVIEW_PACKAGE_READY.txt" ;;
  final) MARKER="FINAL_USER_LABELS_VALIDATED.txt" ;;
  *) echo "usage: $0 {pre_user|final}" >&2; exit 2 ;;
esac

RESULT_ROOT="workspace/core4d/results/E170"
EVAL_DIR="$RESULT_ROOT/s6_downstream/eval/full"
XLSX="$EVAL_DIR/E170_box021_prg_full_validation.xlsx"
RECALC_JSON_PATH="$EVAL_DIR/xlsx_recalc_validation.json"

[ -f "$RESULT_ROOT/POSTPROCESS_COMPLETE.txt" ] || {
  echo "E170 automated postprocess is not complete" >&2
  exit 3
}
if [ "$STAGE" = "final" ] && [ ! -f "$RESULT_ROOT/PRE_USER_REVIEW_PACKAGE_READY.txt" ]; then
  echo "E170 Codex pre-user package is not ready" >&2
  exit 4
fi

MUJOCO_GL="${MUJOCO_GL:-egl}" bash workspace/core4d/scripts/eval/wrappers/eval_E170_box021_prg.sh full
python3 workspace/core4d/scripts/eval/reports/gen_E170_box021_prg_xlsx.py \
  --eval-dir "$EVAL_DIR" --output "$XLSX" --require-keyframes
RECALC_JSON="$(python3 "$HOME/.codex/skills/xlsx/scripts/recalc.py" "$XLSX" 120)"
printf '%s\n' "$RECALC_JSON"
printf '%s\n' "$RECALC_JSON" > "$RECALC_JSON_PATH"
RECALC_JSON="$RECALC_JSON" python3 - <<'PY'
import json, os
payload = json.loads(os.environ["RECALC_JSON"])
if payload.get("status") != "success" or payload.get("total_errors") != 0:
    raise SystemExit(f"E170 workbook formula validation failed: {payload}")
PY
python3 workspace/core4d/scripts/experiments/E170/audit_review_package.py --stage "$STAGE"
printf 'validated_at=%s\nstage=%s\n' "$(date -Is)" "$STAGE" > "$RESULT_ROOT/$MARKER"
printf 'E170 %s review package validated: %s\n' "$STAGE" "$RESULT_ROOT/$MARKER"
