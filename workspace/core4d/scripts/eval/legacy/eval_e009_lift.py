"""Evaluate E009 variant lift performance objectively (qpos-based)."""
import sys
import numpy as np

variant = sys.argv[1] if len(sys.argv) > 1 else 'a'
suffix = '_act' if variant in ('a',) else ''
base = 'example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0'
sim_path = f'{base}/trajectory_mjwp{suffix}.npz'
ref_path = f'{base}/trajectory_kinematic.npz'

ds = np.load(sim_path)
dr = np.load(ref_path)

q_sim = ds['qpos']
flat = q_sim.reshape(-1, q_sim.shape[-1]) if q_sim.ndim == 3 else q_sim

# act scene: nq=42, obj joints are slide+hinge relative to body initial pos (0.155, -0.124, 0.310)
# non-act scene: nq=43, obj is freejoint with absolute world pos
if flat.shape[-1] == 42:
    init_z = 0.310
    init_x = 0.155
    init_y = -0.124
    obj_z = flat[:, 38] + init_z
    obj_x = flat[:, 36] + init_x
    obj_y = flat[:, 37] + init_y
    print(f'[act scene]')
else:
    obj_z = flat[:, 38]
    obj_x = flat[:, 36]
    obj_y = flat[:, 37]
    print(f'[freejoint scene]')

ref_z = dr['qpos'][:, 38]
ref_x = dr['qpos'][:, 36]
ref_y = dr['qpos'][:, 37]

print(f'\n=== E009{variant} Evaluation ===')
print(f'sim obj z (world): max={obj_z.max():.3f}, mean={obj_z.mean():.3f}, min={obj_z.min():.3f}')
print(f'ref obj z (world): max={ref_z.max():.3f}, mean={ref_z.mean():.3f}')
print(f'sim obj x range: [{obj_x.min():.3f}, {obj_x.max():.3f}]')
print(f'sim obj y range: [{obj_y.min():.3f}, {obj_y.max():.3f}]')
print(f'ref obj x range: [{ref_x.min():.3f}, {ref_x.max():.3f}]')
print(f'ref obj y range: [{ref_y.min():.3f}, {ref_y.max():.3f}]')

# pelvis
pel_sim = flat[:, :3]
T = min(len(flat), len(dr['qpos']))
ratio = len(flat) / len(dr['qpos'])
# resample ref to sim length
idx = (np.arange(len(flat)) / ratio).astype(int).clip(0, len(dr['qpos']) - 1)
pel_ref = dr['qpos'][idx, :3]
pelv_err = np.linalg.norm(pel_sim - pel_ref, axis=-1)
print(f'pelvis err: mean={pelv_err.mean():.3f}, max={pelv_err.max():.3f}')

# Pass / fail
LIFT_THRESH = 0.45
PELV_THRESH = 0.20
lift_pass = obj_z.max() >= LIFT_THRESH
pelv_pass = pelv_err.mean() < PELV_THRESH
print(f'\nC-lift (obj_z_max >= {LIFT_THRESH}): {"PASS" if lift_pass else "FAIL"} ({obj_z.max():.3f})')
print(f'C-pelvis (pelv_err mean < {PELV_THRESH}): {"PASS" if pelv_pass else "FAIL"} ({pelv_err.mean():.3f})')
