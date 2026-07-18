#!/usr/bin/env bash
# Run E170 render/eval/workbook only after strict 24-new + 4-reuse completion.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SUMMARY="workspace/core4d/results/E170/s6_downstream/artifacts/full/artifact_summary.json"
EVAL_DIR="workspace/core4d/results/E170/s6_downstream/eval/full"
XLSX="$EVAL_DIR/E170_box021_prg_full_validation.xlsx"

[ -f "$SUMMARY" ] || { echo "missing artifact summary: $SUMMARY" >&2; exit 2; }
.venv/bin/python - "$SUMMARY" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
if not (payload.get("status")=="pass" and payload.get("required_rows")==24 and payload.get("manifest_rows")==24 and payload.get("complete_rows")==24 and not payload.get("incomplete")):
    raise SystemExit(f"E170 strict 24-row artifact gate not satisfied: {payload}")
PY

MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python workspace/core4d/scripts/experiments/E170/render_box021_prg_results.py --stage full
MUJOCO_GL="${MUJOCO_GL:-egl}" bash workspace/core4d/scripts/eval/wrappers/eval_E170_box021_prg.sh full
python3 workspace/core4d/scripts/eval/reports/gen_E170_box021_prg_xlsx.py --eval-dir "$EVAL_DIR" --output "$XLSX"
RECALC_JSON="$(python3 "$HOME/.codex/skills/xlsx/scripts/recalc.py" "$XLSX" 120)"
printf '%s\n' "$RECALC_JSON"
RECALC_JSON="$RECALC_JSON" .venv/bin/python - <<'PY'
import json,os
payload=json.loads(os.environ["RECALC_JSON"])
if payload.get("status")!="success" or payload.get("total_errors")!=0:
    raise SystemExit(f"E170 workbook formula validation failed: {payload}")
PY
printf 'completed_at=%s\n' "$(date -Is)" > workspace/core4d/results/E170/POSTPROCESS_COMPLETE.txt
