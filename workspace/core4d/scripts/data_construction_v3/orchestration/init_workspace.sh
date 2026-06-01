#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPIDER_REPO="${SPIDER_REPO:-$(git -C "$SCRIPT_ROOT/../../../.." rev-parse --show-toplevel)}"
HOLOSOMA_REPO="${HOLOSOMA_REPO:-${HOME}/Workspace/holosoma}"
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  RUN_ID="$1"
  shift
else
  RUN_ID="$(date +dcv3_%Y%m%d_%H%M%S)"
fi
DATA_CONSTRUCTION_RUN_ROOT="${DATA_CONSTRUCTION_RUN_ROOT:-${HOLOSOMA_REPO}/workspace/v3/data_construction_v3_runs}"
CHECK_ARGS=()
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
      DATA_CONSTRUCTION_RUN_ROOT="$2"
      shift 2
      ;;
    --core4d-raw-root)
      CORE4D_RAW_ROOT="$2"
      CHECK_ARGS+=("--core4d-raw-root" "$2")
      shift 2
      ;;
    --out-dir)
      echo "init_workspace.sh owns --out-dir; environment check will be written inside the run directory." >&2
      shift 2
      ;;
    *)
      CHECK_ARGS+=("$1")
      shift
      ;;
  esac
done
RUN_DIR="${DATA_CONSTRUCTION_RUN_ROOT}/${RUN_ID}"

mkdir -p \
  "$RUN_DIR/config" \
  "$RUN_DIR/registries" \
  "$RUN_DIR/inputs" \
  "$RUN_DIR/stage_s0_environment" \
  "$RUN_DIR/stage_s1_raw_contact" \
  "$RUN_DIR/stage_s2_templates" \
  "$RUN_DIR/stage_s3_retarget" \
  "$RUN_DIR/stage_s4_gate_visual_qc" \
  "$RUN_DIR/stage_s5_handoff" \
  "$RUN_DIR/stage_s6_downstream" \
  "$RUN_DIR/logs" \
  "$RUN_DIR/imported_snapshots"

python3 "$SCRIPT_ROOT/stages/s0_environment/check_environment.py" \
  --spider-repo "$SPIDER_REPO" \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --run-root "$DATA_CONSTRUCTION_RUN_ROOT" \
  --out-dir "$RUN_DIR/stage_s0_environment" \
  "${CHECK_ARGS[@]}"

python3 "$SCRIPT_ROOT/stages/s3_retarget/register_retarget_variant.py" \
  --registry-dir "$RUN_DIR/registries" \
  --spider-repo "$SPIDER_REPO" \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --init-defaults

python3 "$SCRIPT_ROOT/state/update_case_state_registry.py" \
  --registry-dir "$RUN_DIR/registries" \
  --init-empty

cat > "$RUN_DIR/run_manifest.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "mode": "uninitialized",
  "created_at": "$(date -Iseconds)",
  "spider_repo": "${SPIDER_REPO}",
  "holosoma_repo": "${HOLOSOMA_REPO}",
  "core4d_raw_root": "${CORE4D_RAW_ROOT:-}",
  "output_root": "${RUN_DIR}",
  "schema_version": "core4d_data_construction_v3.0"
}
EOF

echo "$RUN_DIR"
