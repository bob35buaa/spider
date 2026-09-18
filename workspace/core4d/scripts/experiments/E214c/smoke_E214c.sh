#!/usr/bin/env bash
# E214c fast GPU smoke: validate config-load + per-case float toggles + gate logic
# (esp. safety gate with non-NaN hard_floor for A5, and the over-strict hand gate
# for A6 that may reject most candidates -> least_violation fallback) WITHOUT running
# the full 32-iteration CEM.  Runs on a single free GPU; does NOT touch the keepalive.
#
# Usage: bash workspace/core4d/scripts/experiments/E214c/smoke_E214c.sh [GPU]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
GPU="${1:-5}"
PY=.venv/bin/python
SMOKE_DIR="workspace/core4d/results/E214c/_smoke"
MAN="workspace/core4d/results/E214c/manifests/e214c_partwise_cem.tsv"

# Pull row 0 (A5) and row 1 (A6) paths from the manifest.
read_field() { awk -F'\t' -v r="$1" -v f="$2" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next} NR==r+1{print $h[f]}' "$MAN"; }

for ROW in 1 2; do
  CASE=$(read_field "$ROW" case_id)
  ABL=$(read_field "$ROW" ablation)
  CFG=$(read_field "$ROW" baseline_config_act)
  MODEL=$(read_field "$ROW" run_model_path)
  DATA=$(read_field "$ROW" run_data_path)
  MASK=$(read_field "$ROW" run_contact_mask_path)
  TOGGLES=$(read_field "$ROW" toggles)
  OUT="$SMOKE_DIR/$ABL"
  rm -rf "$OUT"; mkdir -p "$OUT"

  # build the "+gate" override tokens (gate keys are absent from default.yaml)
  GTOK=""
  for t in $TOGGLES; do GTOK="$GTOK +$t"; done

  echo "=== [smoke] $ABL  case=$CASE  gpu=$GPU  iters=3 ==="
  echo "    toggles: $TOGGLES"
  CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=disable TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1 \
    $PY -u examples/run_mjwp.py \
      +load_config_path="$CFG" \
      output_dir="$OUT" \
      save_video=false save_config=true +use_torch_compile=false \
      model_path="$MODEL" data_path="$DATA" +contact_hdmi_mask_path="$MASK" \
      max_num_iterations=3 \
      $GTOK \
      > "logs/E214c/smoke_${ABL}.log" 2>&1
  RC=$?
  echo "    exit=$RC"
  if [ -f "$OUT/config_act.yaml" ]; then
    echo "    saved config gate values:"
    grep -nE "cem_safety_gate_min_sdf_m|cem_safety_gate_hard_floor_m|cem_safety_gate_max_violation_pct|cem_hand_gate_min_sdf_m|cem_hand_gate_hard_floor_m|cem_hand_gate_max_violation_pct|cem_leg_gate_min_sdf_m|cem_leg_gate_hard_floor_m|cem_leg_gate_max_violation_pct" "$OUT/config_act.yaml" | sed 's/^/      /'
  else
    echo "    NO config_act.yaml saved (see logs/E214c/smoke_${ABL}.log)"
  fi
  if [ -f "$OUT/trajectory_mjwp_act.npz" ]; then
    echo "    npz: OK ($(stat -c%s "$OUT/trajectory_mjwp_act.npz") bytes)"
  else
    echo "    npz: MISSING"
  fi
done
echo "=== smoke done ==="
