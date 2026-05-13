#!/usr/bin/env bash
# Regression test: box025 + (E041c | E041 | E039) under 3-box hand collision.
#
# Purpose
# -------
# E060.0 commit (fa2e181) replaced G1 hand_collision from a single 5cm sphere
# to 3 boxes per side (wrist cuff + palm slab + angled finger). This changes
# physics: contact pairs went from 4 to 12 per case, and palm vs back of hand
# is now distinguishable (was rotationally symmetric).
#
# Historical box025 results (sphere hand) for reference (from EXPERIMENT_TRACKER):
#   E041c (additive w=0.3): Contact 66%, Stable 100%, MPKPE 1.4cm
#                           [video: 趴在箱上, no real carry — but stable + contact metrics OK]
#   E041  (multiply w=1.0): Contact 62%, Stable 100%, MPKPE 1.6cm
#   E039  (no ori reward):  ~Contact 60%+, Stable 100%
#
# This script reruns the same 3 reward stacks on box025 under 3-box hand to
# check if the historical contact/stability metrics are preserved or regressed.
#
# Decision interpretation:
#   - 3 metrics within ±5pp of historical → 3-box hand is benign on box025
#   - significant regression on E039 (no-ori control) → 3-box hand alone breaks it
#   - regression on E041/E041c only → ori reward × 3-box hand interaction issue
#
# Usage
# -----
#   bash workspace/core4d/scripts/run_box025_3box_regression.sh [parallel|serial] [GPU_A] [GPU_B] [GPU_C]
#
# Default: serial on GPU 0 (3 runs back-to-back, ~1.5h total)
# Parallel on 2 GPUs: bash ... parallel 0 1   (runs E041c+E039 in parallel, then E041 alone)
# Parallel on 3 GPUs: bash ... parallel 0 1 2 (all 3 simultaneously)
#
# Prerequisites
# -------------
#   1. git pull to have commit fa2e181+ (3-box hand)
#   2. .venv with torch 2.8.0 + nccl working (see workspace/hdmi_reproduce/env.md)
#   3. GPU(s) available; nvidia-smi to confirm
#   4. Tested working: ~30 min wall per run
#
# Output
# ------
#   workspace/core4d/results/E060_box025_regression/
#     box025_E041c.{npz,mp4,log}     # reward = E041c (current baseline)
#     box025_E041.{npz,mp4,log}      # reward = E041 (multiply mode)
#     box025_E039.{npz,mp4,log}      # reward = E039 (no ori reward, pure data-layer test)
#     scene_snapshot/box025_person1/{scene.xml,scene_act.xml,scene_act_meta.json,task_info.json}
#     scene_snapshot/manifest.txt    # git HEAD + sha256 for reproducibility

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${1:-serial}"
GPU_A="${2:-0}"
GPU_B="${3:-${GPU_A}}"
GPU_C="${4:-${GPU_A}}"
RESULTS=workspace/core4d/results/E060_box025_regression
LOGS=logs/E060_box025_regression
mkdir -p "$RESULTS" "$LOGS"

# Snapshot the box025 scene state (sha256 + git HEAD) for reproducibility.
echo "[$(date '+%H:%M:%S')] === scene snapshot ==="
EXP_ID=E060_box025_regression
mkdir -p "$RESULTS/scene_snapshot/box025_person1"
SRC_BASE="example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1"
for f in scene.xml scene_act.xml scene_act_meta.json task_info.json; do
  if [ -f "$SRC_BASE/$f" ]; then cp "$SRC_BASE/$f" "$RESULTS/scene_snapshot/box025_person1/$f"; fi
done
{
  echo "# Scene snapshot for $EXP_ID"
  echo "# Captured: $(date -Iseconds)"
  echo "# Git HEAD: $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "# Hand collision: 3-box per side (verify in scene.xml geom names lh/lh2/lh3)"
  echo
  for f in scene.xml scene_act.xml; do
    sha=$(sha256sum "$SRC_BASE/$f" | awk '{print $1}')
    echo "box025_person1/$f  sha256=$sha"
  done
} > "$RESULTS/scene_snapshot/manifest.txt"
cat "$RESULTS/scene_snapshot/manifest.txt"

# Sanity: confirm 3-box hand is actually present
echo
echo "=== Sanity check: hand geoms in box025 scene_act.xml ==="
.venv/bin/python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('$SRC_BASE/scene_act.xml')
hand = sorted([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               for i in range(m.ngeom)
               if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
               and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith(('lh','rh'))])
print(f'  hand geoms: {hand}')
print(f'  npair: {m.npair}')
assert len(hand) == 6, f'Expected 6 hand boxes, got {len(hand)} ({hand}). Did you git pull commit fa2e181+?'
print('  ✅ 3-box hand confirmed')
"

run_one() {
  local override=$1 gpu=$2
  local name="box025_${override#core4d_}"   # e.g. box025_e041c
  name="${name//core4d_/}"
  local out_dir="$RESULTS/${name}_outdir"
  mkdir -p "$out_dir"
  echo "[$(date '+%H:%M:%S')] === $name (override=$override, GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python -u examples/run_mjwp.py \
    +override=$override \
    task=box025_person1 \
    +use_torch_compile=false \
    output_dir="$out_dir" \
    video_output_path="$RESULTS/${name}.mp4" \
    > "$LOGS/${name}.log" 2>&1
  cp "$out_dir/trajectory_mjwp_act.npz" "$RESULTS/${name}.npz"
  echo "[$(date '+%H:%M:%S')] $name done."
}

case "$MODE" in
  parallel)
    if [ "$GPU_C" != "$GPU_A" ] && [ "$GPU_C" != "$GPU_B" ]; then
      echo "Mode: parallel-3 — E041c on GPU $GPU_A, E041 on GPU $GPU_B, E039 on GPU $GPU_C"
      run_one core4d_e041c "$GPU_A" &
      P0=$!
      run_one core4d_e041  "$GPU_B" &
      P1=$!
      run_one core4d_e039  "$GPU_C" &
      P2=$!
      wait $P0 $P1 $P2
    else
      echo "Mode: parallel-2 + 1 — E041c on GPU $GPU_A + E039 on GPU $GPU_B, then E041 on GPU $GPU_A"
      run_one core4d_e041c "$GPU_A" &
      P0=$!
      run_one core4d_e039  "$GPU_B" &
      P1=$!
      wait $P0 $P1
      run_one core4d_e041  "$GPU_A"
    fi
    ;;
  serial)
    echo "Mode: serial on GPU $GPU_A"
    run_one core4d_e041c "$GPU_A"
    run_one core4d_e041  "$GPU_A"
    run_one core4d_e039  "$GPU_A"
    ;;
  *)
    echo "Unknown mode: $MODE (use parallel|serial)"; exit 1 ;;
esac

echo
echo "=== All runs done. Outputs: ==="
ls -lh "$RESULTS"/*.{npz,mp4} 2>/dev/null

echo
echo "=== Quick metrics (pelvis_min + first/last pelvis_z) ==="
.venv/bin/python -c "
import numpy as np, glob, os
for npz in sorted(glob.glob('$RESULTS/*.npz')):
    d = np.load(npz, allow_pickle=True)
    qpos = d['qpos']
    if qpos.ndim == 3: qpos = qpos[:,0,:]
    pz = qpos[:,2]
    name = os.path.basename(npz).replace('.npz','')
    print(f'  {name:<24} T={qpos.shape[0]:3d}  pz_first={pz[0]:.3f}  pz_last={pz[-1]:.3f}  pz_min={pz.min():.3f}  pz_mean={pz.mean():.3f}')
"

echo
echo "=== Comparison vs historical (sphere hand) box025 baselines from EXPERIMENT_TRACKER ==="
echo "  Historical Stable% (full episode): E041c=100%  E041=100%  E039=>=99%"
echo "  Historical pelvis_min (cm) for E048+sphere baseline: 57.5 (E041c), similar for E041/E039"
echo "  → If new pz_min < 0.30m: 3-box hand caused regression on this reward stack"
echo "  → If pz_min >= 0.50m and similar to historical: 3-box hand is benign"
echo "  → Mixed results: interpret per-reward (E039 is the cleanest data-layer-only test)"
echo
echo "Send back: $RESULTS/ (npz + mp4 + log + scene_snapshot)"
