#!/usr/bin/env bash
# E165-D local runner for peak-margin CEM rerank.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-full}"
LOCAL_GPU="${LOCAL_GPU:-0}"
E165D_SPLIT="${E165D_SPLIT:-local-gpu0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
VARIANTS="workspace/core4d/scripts/experiments/E165/variants.tsv"
RESULTS="workspace/core4d/results/E165/peak_margin_rerank/cem/${STAGE}"
LOGS="logs/E165D/cem/${STAGE}"
CASE_METHODS="${CASE_METHODS:-}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-0}"
GPU_IDLE_MAX_MEM_MB="${E165D_GPU_IDLE_MAX_MEM_MB:-3000}"
GPU_IDLE_MAX_UTIL_PCT="${E165D_GPU_IDLE_MAX_UTIL_PCT:-20}"
GPU_IDLE_POLL_SEC="${E165D_GPU_IDLE_POLL_SEC:-120}"
GPU_IDLE_STABLE_POLLS="${E165D_GPU_IDLE_STABLE_POLLS:-2}"

if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "full" ]; then
  echo "Invalid STAGE=$STAGE (use smoke|full)" >&2
  exit 2
fi

"$PYTHON_BIN" workspace/core4d/scripts/experiments/E165/build_peak_margin_manifest.py >/dev/null
mkdir -p "$RESULTS" "$LOGS"

is_complete() {
  local variant=$1
  [ -f "$RESULTS/${variant}.npz" ] \
    && [ -f "$RESULTS/${variant}_${STAGE}.mp4" ] \
    && [ -f "$RESULTS/${variant}_outdir_${STAGE}/trajectory_mjwp_act.npz" ] \
    && [ -f "$RESULTS/${variant}_outdir_${STAGE}/config_act.yaml" ]
}

selected_rows() {
  "$PYTHON_BIN" - "$VARIANTS" "$E165D_SPLIT" "$CASE_METHODS" <<'PY'
import csv
import sys
from pathlib import Path

variants = Path(sys.argv[1])
split = sys.argv[2]
spec_text = sys.argv[3].strip()
specs = set()
if spec_text:
    for item in spec_text.split():
        specs.add(tuple(item.split(":", 1)) if ":" in item else (item, ""))

fields = [
    "variant",
    "short_case_id",
    "method_group",
    "derived_task",
    "override",
    "cem_peak_margin_ee_threshold_m",
    "cem_peak_margin_anchor_threshold_m",
    "cem_peak_margin_buffer_m",
]
with variants.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["run_status"] != "to_run":
            continue
        if specs:
            if (row["short_case_id"], row["method_group"]) not in specs and (row["short_case_id"], "") not in specs:
                continue
        elif row["split"] != split:
            continue
        print("\t".join(row[field] for field in fields))
PY
}

wait_for_gpu_idle() {
  if [ "$WAIT_FOR_GPU_IDLE" != "1" ]; then
    return 0
  fi
  local stable=0
  echo "[$(date '+%H:%M:%S')] waiting for GPU ${LOCAL_GPU} idle: mem<=${GPU_IDLE_MAX_MEM_MB}MiB util<=${GPU_IDLE_MAX_UTIL_PCT}% stable=${GPU_IDLE_STABLE_POLLS}"
  while true; do
    local stat mem util
    stat="$(nvidia-smi --id="$LOCAL_GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ' || true)"
    mem="${stat%%,*}"
    util="${stat##*,}"
    if [ -n "$mem" ] && [ -n "$util" ] && [ "$mem" -le "$GPU_IDLE_MAX_MEM_MB" ] && [ "$util" -le "$GPU_IDLE_MAX_UTIL_PCT" ]; then
      stable=$((stable + 1))
      echo "[$(date '+%H:%M:%S')] GPU ${LOCAL_GPU} idle poll ${stable}/${GPU_IDLE_STABLE_POLLS}: mem=${mem}MiB util=${util}%"
      if [ "$stable" -ge "$GPU_IDLE_STABLE_POLLS" ]; then
        return 0
      fi
    else
      stable=0
      echo "[$(date '+%H:%M:%S')] GPU ${LOCAL_GPU} busy: mem=${mem:-NA}MiB util=${util:-NA}%"
    fi
    sleep "$GPU_IDLE_POLL_SEC"
  done
}

run_one() {
  local variant=$1
  local case_id=$2
  local group=$3
  local task=$4
  local override_path=$5
  local ee_threshold=$6
  local anchor_threshold=$7
  local buffer_m=$8
  local override
  override="$(basename "$override_path" .yaml)"
  local out_dir="$RESULTS/${variant}_outdir_${STAGE}"

  if is_complete "$variant"; then
    echo "[$(date '+%H:%M:%S')] === ${variant} complete; skip ==="
    return 0
  fi

  wait_for_gpu_idle

  local smoke_args=()
  if [ "$STAGE" = "smoke" ]; then
    smoke_args=(num_samples=64 max_num_iterations=4)
  fi

  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === START ${variant} case=${case_id} group=${group} GPU=${LOCAL_GPU} stage=${STAGE} ==="

  CUDA_VISIBLE_DEVICES="$LOCAL_GPU" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u examples/run_mjwp.py \
      +override="$override" task="$task" \
      +use_torch_compile=false video_camera=auto \
      "cem_peak_margin_ee_threshold_m=${ee_threshold}" \
      "cem_peak_margin_anchor_threshold_m=${anchor_threshold}" \
      "cem_peak_margin_buffer_m=${buffer_m}" \
      "${smoke_args[@]}" \
      output_dir="$out_dir" \
      video_output_path="$RESULTS/${variant}_${STAGE}.mp4" \
      > "$LOGS/${variant}.log" 2>&1

  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${variant}.npz"
  echo "[$(date '+%H:%M:%S')] === DONE ${variant} ==="
}

echo "=== E165D ${STAGE} local runner split=${E165D_SPLIT} GPU=${LOCAL_GPU} ==="
if [ -n "$CASE_METHODS" ]; then
  echo "CASE_METHODS=${CASE_METHODS}"
fi

row_count=0
while IFS=$'\t' read -r variant case_id group task override ee_threshold anchor_threshold buffer_m; do
  row_count=$((row_count + 1))
  run_one "$variant" "$case_id" "$group" "$task" "$override" "$ee_threshold" "$anchor_threshold" "$buffer_m"
done < <(selected_rows)

echo "=== E165D ${STAGE} local runner complete; selected_rows=${row_count} ==="
