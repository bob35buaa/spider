#!/bin/bash
# E010: Connect constraint retargeting
# Verifies G1 kinematic feasibility by welding hands to object
set -euo pipefail

TASK=${1:-box025_person1}
DATA_ROOT="example_datasets/processed/core4d/unitree_g1/humanoid_object"

echo "=== E010: Connect Constraint Retargeting ==="
echo "Task: $TASK"

# Step 1: Generate scene_connect.xml (if not exists or --regen)
if [[ ! -f "${DATA_ROOT}/${TASK}/scene_connect.xml" ]] || [[ "${2:-}" == "--regen" ]]; then
    echo ">>> Generating scene_connect.xml..."
    python3 workspace/core4d/scripts/generate_scene_connect.py --task "$TASK"
fi

# Step 2: Run retargeting
echo ">>> Running MJWP retargeting with connect constraints..."
uv run examples/run_mjwp.py +override=core4d_box025_connect \
    task="$TASK" data_id=0 viewer=none

# Step 3: Objective verification (qpos, NOT reward)
echo ">>> Objective verification..."
python3 -c "
import numpy as np

task = '${TASK}'
base = '${DATA_ROOT}/${TASK}/0'

sim = np.load(f'{base}/trajectory_mjwp.npz')
ref = np.load(f'{base}/trajectory_kinematic.npz')

sq = sim['qpos']
rq = ref['qpos']
T = min(len(sq), len(rq))

# Object z (world frame: obj initial z + qpos z)
sim_obj_z = sq[:, 38]
ref_obj_z = rq[:T, 38]

# Pelvis error
sim_pelvis = sq[:, :3]
ref_pelvis = rq[:T, :3]
pelvis_err = np.linalg.norm(sim_pelvis[:T] - ref_pelvis, axis=-1)

# Joint error
sim_joints = sq[:, 7:36]
ref_joints = rq[:T, 7:36]
joint_err = np.abs(sim_joints[:T] - ref_joints).mean(axis=-1)

# Object pos error (world)
sim_obj_pos = sq[:, 36:39]
ref_obj_pos = rq[:T, 36:39]
obj_pos_err = np.linalg.norm(sim_obj_pos[:T] - ref_obj_pos, axis=-1)

print('=== E010 RESULTS (qpos-based, NOT reward) ===')
print(f'  obj z:       min={sim_obj_z.min():.3f}, max={sim_obj_z.max():.3f}, mean={sim_obj_z.mean():.3f}')
print(f'  ref obj z:   min={ref_obj_z.min():.3f}, max={ref_obj_z.max():.3f}, mean={ref_obj_z.mean():.3f}')
print(f'  pelvis err:  mean={pelvis_err.mean():.3f}, max={pelvis_err.max():.3f}')
print(f'  joint err:   mean={joint_err.mean():.4f} rad')
print(f'  obj pos err: mean={obj_pos_err.mean():.3f}, max={obj_pos_err.max():.3f}')
print()
# Check claims
z_max = sim_obj_z.max()
p_err = pelvis_err.mean()
j_err = joint_err.mean()
print(f'  C1 (obj z_max >= 0.40): {z_max:.3f} -> {\"PASS\" if z_max >= 0.40 else \"FAIL\"}'  )
print(f'  C2 (pelvis_err <= 0.15): {p_err:.3f} -> {\"PASS\" if p_err <= 0.15 else \"FAIL\"}')
print(f'  C4 (joint_err <= 0.10): {j_err:.4f} -> {\"PASS\" if j_err <= 0.10 else \"FAIL\"}')

# Frames where obj z > 0.40
high_frames = (sim_obj_z > 0.40).sum()
print(f'  Frames with obj z > 0.40: {high_frames}/{len(sim_obj_z)} ({100*high_frames/len(sim_obj_z):.1f}%)')
"

echo "=== Done ==="
