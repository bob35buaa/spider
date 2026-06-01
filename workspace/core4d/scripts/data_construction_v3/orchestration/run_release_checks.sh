#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPIDER_REPO="${SPIDER_REPO:-$(git -C "$SCRIPT_ROOT/../../../.." rev-parse --show-toplevel)}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-${HOME}/Workspace/holosoma}"
RUN_ROOT="${RUN_ROOT:-/tmp/core4d_dcv3_release_checks}"
CORE4D_RAW_ROOT="${CORE4D_RAW_ROOT:-}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-}"
SMOKE_MODE="auto"
OBJECT_KEYS="box004"
SAMPLE_COUNT="300"
MAX_SEQUENCES="1"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --spider-repo)
      SPIDER_REPO="$2"
      shift 2
      ;;
    --holosoma-repo)
      HOLOSOMA_REPO="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --core4d-raw-root)
      CORE4D_RAW_ROOT="$2"
      shift 2
      ;;
    --smplx-model-dir)
      SMPLX_MODEL_DIR="$2"
      shift 2
      ;;
    --with-smoke)
      SMOKE_MODE="with"
      shift
      ;;
    --no-smoke)
      SMOKE_MODE="none"
      shift
      ;;
    --object-keys)
      OBJECT_KEYS="$2"
      shift 2
      ;;
    --sample-count)
      SAMPLE_COUNT="$2"
      shift 2
      ;;
    --max-sequences)
      MAX_SEQUENCES="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RUN_ROOT"
cd "$SPIDER_REPO"

echo "[release-checks] py_compile"
mapfile -d '' PY_FILES < <(find "$SCRIPT_ROOT" -path '*/__pycache__' -prune -o -name '*.py' -print0)
python3 -m py_compile "${PY_FILES[@]}"

echo "[release-checks] bash -n legacy Stage2b wrapper"
bash -n "$SPIDER_REPO/workspace/core4d/data_preprocess/pipeline.sh"

echo "[release-checks] codebase release audit"
AUDIT_STDOUT="$RUN_ROOT/release_audit_stdout.json"
AUDIT_REPORT="$RUN_ROOT/release_audit/pipeline_release_audit.json"
python3 "$SCRIPT_ROOT/qa/audit_pipeline_release.py" \
  --spider-repo "$SPIDER_REPO" \
  --out-dir "$RUN_ROOT/release_audit" \
  > "$AUDIT_STDOUT"
python3 - "$AUDIT_REPORT" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)
print(
    f"[release-checks] release audit {report['status']} "
    f"({report['passed']} passed, {report['failed']} failed): {path}"
)
PY

run_smoke=false
if [[ "$SMOKE_MODE" == "with" ]]; then
  run_smoke=true
elif [[ "$SMOKE_MODE" == "auto" && -n "$CORE4D_RAW_ROOT" && -n "$SMPLX_MODEL_DIR" && -d "$CORE4D_RAW_ROOT" && -d "$SMPLX_MODEL_DIR" ]]; then
  run_smoke=true
fi

if [[ "$run_smoke" == true ]]; then
  if [[ -z "$CORE4D_RAW_ROOT" || ! -d "$CORE4D_RAW_ROOT" ]]; then
    echo "--with-smoke requires --core4d-raw-root or CORE4D_RAW_ROOT pointing to an existing directory" >&2
    exit 2
  fi
  if [[ -z "$SMPLX_MODEL_DIR" || ! -d "$SMPLX_MODEL_DIR" ]]; then
    echo "--with-smoke requires --smplx-model-dir or SMPLX_MODEL_DIR pointing to an existing directory" >&2
    exit 2
  fi
  echo "[release-checks] compact smoke suite"
  python3 "$SCRIPT_ROOT/qa/run_smoke_suite.py" \
    --run-root "$RUN_ROOT/smoke_suite" \
    --spider-repo "$SPIDER_REPO" \
    --holosoma-repo "$HOLOSOMA_REPO" \
    --core4d-raw-root "$CORE4D_RAW_ROOT" \
    --smplx-model-dir "$SMPLX_MODEL_DIR" \
    --object-keys "$OBJECT_KEYS" \
    --sample-count "$SAMPLE_COUNT" \
    --max-sequences "$MAX_SEQUENCES"
else
  echo "[release-checks] compact smoke suite skipped; provide CORE4D_RAW_ROOT and SMPLX_MODEL_DIR or pass --with-smoke to run it"
fi

echo "[release-checks] PASS"
echo "[release-checks] outputs: $RUN_ROOT"
