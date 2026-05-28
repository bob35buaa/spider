"""Deep comparison of box021 vs box023 vs box025 trajectories.

Extracts:
- pelvis xyz trajectory + height profile
- object pose (assumed last 7 dims of qpos: xyz + quat)
- contact mask per hand
- contact_pos (where each hand says it contacts)
- robot hand (wrist) world positions via simple FK approximation: use ctrl-only diff
  (we won't do real FK here; we use contact_pos as the "intended contact" signal)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

CASES = {
    "box021_D003_18029_p2": "d003_box021_20231018_029_p2_upperobj_e083_m10_e087",
    "box021_D003_11035_p2": "d003_box021_20231011_035_p2_upperobj_e083",
    "box021_D003_20019_p1": "d003_box021_20231020_019_p1_upperobj_e083",
    "box023_p2_GUARD_OK": "box023_person2",
    "box025_p2_partial": "box025_person2",
    "box023_p2_e083": "box023_person2_upperobj_e083",
}

def analyze(label, qpos, ctrl, contact, contact_pos):
    """qpos: (T, 43). last 7 dims are object freejoint xyz+quat."""
    T = qpos.shape[0]
    nq = qpos.shape[1]
    pelvis_xyz = qpos[:, :3]
    obj_xyz = qpos[:, -7:-4]
    obj_quat = qpos[:, -4:]

    z0 = obj_xyz[0, 2]
    z_max = obj_xyz[:, 2].max()
    z_min = obj_xyz[:, 2].min()
    dz_first_to_max = z_max - z0
    # Lift quantification
    lift_thresholds = [0.02, 0.05, 0.10, 0.20]
    lift_pcts = {f"lift>{int(t*100)}cm": float((obj_xyz[:, 2] > (z0 + t)).mean() * 100) for t in lift_thresholds}

    # Horizontal travel
    horiz_path = np.linalg.norm(np.diff(obj_xyz[:, :2], axis=0), axis=1).sum()
    horiz_end_to_start = float(np.linalg.norm(obj_xyz[-1, :2] - obj_xyz[0, :2]))
    # Object rotation: convert quats to yaw angle change
    def quat_yaw(q):
        # q = (w, x, y, z); pretend yaw around z = atan2(2(wz+xy), 1-2(yy+zz))
        w, x, y, z = q
        siny_cosp = 2.0 * (w*z + x*y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return np.arctan2(siny_cosp, cosy_cosp)
    yaws = np.array([quat_yaw(q) for q in obj_quat])
    yaw_swept_deg = float(np.degrees(np.abs(yaws).max() - yaws[0]))
    yaw_change_deg = float(np.degrees(yaws[-1] - yaws[0]))

    # Pelvis change
    pelvis_z_min = float(pelvis_xyz[:, 2].min())
    pelvis_dxy_path = float(np.linalg.norm(np.diff(pelvis_xyz[:, :2], axis=0), axis=1).sum())

    # Contact stats: contact (T, 2) is per-hand binary-ish
    contact_per_hand = contact.mean(axis=0)  # length 2
    # contact_pos: (T, 2, 3) hand world position when in contact
    # When contact==1 in mask, what's the hand height? compute relative to obj_xyz
    rel_to_obj = contact_pos - obj_xyz[:, None, :]  # (T,2,3)

    # Approximate "contact face" by which axis has the largest abs value
    # If hand is on top: rel z > 0 and dominant; bottom: rel z < 0 dominant; side: x or y dominant
    def classify_face(rel_vec):
        # rel_vec is (3,)
        ax = np.argmax(np.abs(rel_vec))
        sgn = np.sign(rel_vec[ax])
        names = ["x", "y", "z"]
        return f"{'+' if sgn >= 0 else '-'}{names[ax]}"

    # Per hand, count face classification across frames where contact==1
    face_counts = []
    for h in range(2):
        active = contact[:, h] > 0.5
        if active.sum() == 0:
            face_counts.append({})
            continue
        rels = rel_to_obj[active, h, :]
        c = {}
        for r in rels:
            f = classify_face(r)
            c[f] = c.get(f, 0) + 1
        face_counts.append(c)

    # Also: hand z vs obj z range
    hand_z_means = []
    hand_z_relmean = []
    for h in range(2):
        active = contact[:, h] > 0.5
        if active.sum() == 0:
            hand_z_means.append(None); hand_z_relmean.append(None)
        else:
            hand_z_means.append(float(contact_pos[active, h, 2].mean()))
            hand_z_relmean.append(float((contact_pos[active, h, 2] - obj_xyz[active, 2]).mean()))

    print(f"\n=== {label} ===  (T={T}, nq={nq})")
    print(f"obj xyz frame0: {obj_xyz[0]}")
    print(f"obj xyz framelast: {obj_xyz[-1]}")
    print(f"obj z   range: {z_min:.3f} .. {z_max:.3f}  (init {z0:.3f}, max-init={dz_first_to_max:+.3f})")
    print(f"obj lift % above init+t: " + ", ".join(f"{k}={v:.0f}%" for k, v in lift_pcts.items()))
    print(f"obj horiz path: {horiz_path:.3f} m, end-to-start: {horiz_end_to_start:.3f}")
    print(f"obj yaw swept: {yaw_swept_deg:.1f} deg, end-start: {yaw_change_deg:+.1f} deg")
    print(f"pelvis z min: {pelvis_z_min:.3f}, pelvis xy path: {pelvis_dxy_path:.3f}m")
    print(f"contact per hand (mean): L={contact_per_hand[0]:.3f} R={contact_per_hand[1]:.3f}")
    print(f"contact_pos hand z mean (active): L={hand_z_means[0]} R={hand_z_means[1]}")
    print(f"contact_pos (hand_z - obj_z) mean: L={hand_z_relmean[0]} R={hand_z_relmean[1]}")
    print(f"face counts (active frames):")
    print(f"   L: {face_counts[0]}")
    print(f"   R: {face_counts[1]}")


def main():
    for label, td in CASES.items():
        p = TASKS / td / "0" / "trajectory_kinematic.npz"
        if not p.exists():
            print(f"\n[MISSING] {label}: {p}")
            continue
        d = dict(np.load(p))
        analyze(label, d["qpos"], d["ctrl"], d["contact"], d["contact_pos"])


if __name__ == "__main__":
    main()
