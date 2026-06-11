#!/usr/bin/env bash
# E156 local runner for clean8 +gateA / E155_decay CEM rows.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
LOCAL_GPU="${LOCAL_GPU:-0}"
E156_SPLIT="${E156_SPLIT:-local-gpu0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
VARIANTS="workspace/core4d/scripts/experiments/E156/variants.tsv"
RESULTS="workspace/core4d/results/E156/clean8_gate_decay/cem/${STAGE}"
LOGS="logs/E156/cem/${STAGE}"
CASE_METHODS="${CASE_METHODS:-}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

if [ ! -f "$VARIANTS" ]; then
  "$PYTHON_BIN" workspace/core4d/scripts/experiments/E156/build_clean8_gate_decay_manifest.py --legacy-copy >/dev/null
fi
mkdir -p "$RESULTS" "$LOGS"

is_complete() {
  local variant=$1
  [ -f "$RESULTS/${variant}.npz" ] \
    && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] \
    && [ -f "$RESULTS/${variant}_outdir_${STAGE}/trajectory_mjwp_act.npz" ]
}

selected_rows() {
  "$PYTHON_BIN" - "$VARIANTS" "$E156_SPLIT" "$CASE_METHODS" <<'PY'
import csv
import sys
from pathlib import Path

variants = Path(sys.argv[1])
split = sys.argv[2]
spec_text = sys.argv[3].strip()

def norm_method(text: str) -> str:
    text = text.strip()
    if text in {"gateA", "+gateA"}:
        return "+gateA"
    if text in {"decay", "E155_decay"}:
        return "E155_decay"
    return text

specs: set[tuple[str, str]] = set()
if spec_text:
    for item in spec_text.split():
        if ":" not in item:
            raise SystemExit(f"CASE_METHODS item must be case:method, got {item}")
        case, method = item.split(":", 1)
        specs.add((case, norm_method(method)))

fields = [
    "variant",
    "short_case_id",
    "method",
    "derived_task",
    "override",
    "cem_hand_gate_min_sdf_m",
    "cem_hand_gate_max_violation_pct",
    "cem_hand_gate_hard_floor_m",
    "contact_hdmi_mask_carry_union",
    "hand_support_decay_frac",
]
with variants.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["run_status"] != "to_run":
            continue
        if specs:
            if (row["short_case_id"], row["method"]) not in specs:
                continue
        elif row["split"] != split:
            continue
        print("\t".join(row[field] for field in fields))
PY
}

run_one() {
  local variant=$1
  local case_id=$2
  local method=$3
  local task=$4
  local override_path=$5
  local min_sdf=$6
  local max_viol=$7
  local hard_floor=$8
  local carry_union=$9
  local decay_frac=${10}

  local override
  override="$(basename "$override_path" .yaml)"
  local out_dir="$RESULTS/${variant}_outdir_${STAGE}"

  if is_complete "$variant"; then
    echo "[$(date '+%H:%M:%S')] === ${variant} complete; skip ==="
    return 0
  fi

  local smoke_args=()
  if [ "$STAGE" = "smoke" ]; then
    smoke_args=(num_samples=64 max_num_iterations=4)
  fi

  local extra_args=(
    "cem_hand_gate_min_sdf_m=${min_sdf}"
    "cem_hand_gate_max_violation_pct=${max_viol}"
    "+cem_hand_gate_hard_floor_m=${hard_floor}"
  )
  if [ "$method" = "E155_decay" ]; then
    extra_args+=(
      "+contact_hdmi_mask_carry_union=${carry_union}"
      "+hand_support_decay_frac=${decay_frac}"
    )
  fi

  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === START ${variant} case=${case_id} method=${method} GPU=${LOCAL_GPU} stage=${STAGE} ==="

  CUDA_VISIBLE_DEVICES="$LOCAL_GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override" task="$task" \
      +use_torch_compile=false video_camera=auto \
      "${extra_args[@]}" \
      "${smoke_args[@]}" \
      output_dir="$out_dir" \
      video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  echo "[$(date '+%H:%M:%S')] === DONE ${variant} ==="
}

echo "=== E156 ${STAGE} local runner split=${E156_SPLIT} GPU=${LOCAL_GPU} ==="
if [ -n "$CASE_METHODS" ]; then
  echo "CASE_METHODS=${CASE_METHODS}"
fi

row_count=0
while IFS=$'\t' read -r variant case_id method task override min_sdf max_viol hard_floor carry_union decay_frac; do
  row_count=$((row_count + 1))
  run_one "$variant" "$case_id" "$method" "$task" "$override" "$min_sdf" "$max_viol" "$hard_floor" "$carry_union" "$decay_frac"
done < <(selected_rows)

echo "=== E156 ${STAGE} local runner complete; selected_rows=${row_count} ==="
