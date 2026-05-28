"""Transform contact_pos to object's local frame and classify face.
Also report hand height in world to expose 'low hand target' issue.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

CASES = {
    "box021_D003_18029_p2": "d003_box021_20231018_029_p2_upperobj_e083_m10_e087",
    "box021_D003_11035_p2": "d003_box021_20231011_035_p2_upperobj_e083",
    "box021_D003_20019_p1": "d003_box021_20231020_019_p1_upperobj_e083",
    "box023_p2_OK": "box023_person2",
    "box025_p2_partial": "box025_person2",
}

def quat_to_rot(q):
    w, x, y, z = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

def parse_object_collision_extents(scene_xml):
    """Find the object collision box geom and return its size (half-extents) and pos (offset)."""
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            for geom in body.iter("geom"):
                cls = geom.attrib.get("class") or geom.attrib.get("group")
                if "collision" in (cls or "").lower() or geom.attrib.get("name", "").endswith("collision"):
                    size = geom.attrib.get("size")
                    pos = geom.attrib.get("pos", "0 0 0")
                    if size:
                        return tuple(float(x) for x in size.split()), tuple(float(x) for x in pos.split())
            # fallback: first box geom
            for geom in body.iter("geom"):
                if geom.attrib.get("type") == "box":
                    size = geom.attrib.get("size")
                    pos = geom.attrib.get("pos", "0 0 0")
                    if size:
                        return tuple(float(x) for x in size.split()), tuple(float(x) for x in pos.split())
    return None, None

def analyze(label, task_dir):
    p_npz = TASKS / task_dir / "0" / "trajectory_kinematic.npz"
    p_scene = TASKS / task_dir / "scene.xml"
    if not p_npz.exists():
        print(f"[MISSING] {label}")
        return
    d = dict(np.load(p_npz))
    qpos = d["qpos"]
    contact = d["contact"]
    contact_pos = d["contact_pos"]
    T = qpos.shape[0]

    obj_xyz = qpos[:, -7:-4]
    obj_quat = qpos[:, -4:]
    pelvis_xyz = qpos[:, :3]

    half_ext, geom_pos = parse_object_collision_extents(p_scene)
    print(f"\n=== {label} ===  T={T}")
    print(f"scene.xml: {p_scene.relative_to(ROOT)}")
    print(f"object collision half-extents (local): {half_ext}, geom pos: {geom_pos}")

    # Convert contact_pos -> object local frame for each (t, hand)
    face_counts = [{}, {}]
    hand_local_means = [np.zeros(3), np.zeros(3)]
    hand_face_dist = [[], []]  # signed distance to chosen face (outside positive)
    hand_world_z = [[], []]
    active_cnt = [0, 0]
    obj_center_offset = np.array(geom_pos) if geom_pos else np.zeros(3)
    half = np.array(half_ext) if half_ext else np.array([0.2, 0.2, 0.2])

    for t in range(T):
        R = quat_to_rot(obj_quat[t])
        for h in range(2):
            if contact[t, h] < 0.5:
                continue
            active_cnt[h] += 1
            world_rel = contact_pos[t, h] - obj_xyz[t]
            local = R.T @ world_rel - obj_center_offset
            hand_local_means[h] += local
            # face = argmax abs of (local / half) — i.e. closest face when normalized
            norm = local / half
            ax = int(np.argmax(np.abs(norm)))
            sgn = "+" if norm[ax] >= 0 else "-"
            names = ["x", "y", "z"]
            face = f"{sgn}{names[ax]}"
            face_counts[h][face] = face_counts[h].get(face, 0) + 1
            # signed dist to that face (positive = outside box past chosen face plane)
            # If sgn>=0: outside means local[ax] > +half; signed = local[ax] - half
            # If sgn<0:  outside means local[ax] < -half; signed = -half - local[ax]
            sign = 1 if norm[ax] >= 0 else -1
            hand_face_dist[h].append(sign * local[ax] - half[ax])
            # also compute "fully inside box" flag
            # (handled outside loop)
            hand_world_z[h].append(float(contact_pos[t, h, 2]))

    for h in range(2):
        if active_cnt[h] == 0:
            continue
        hand_local_means[h] /= active_cnt[h]
    hand_local_strs = [
        f"L_local_mean={hand_local_means[0].round(3).tolist()}",
        f"R_local_mean={hand_local_means[1].round(3).tolist()}",
    ]
    print(f"pelvis z min: {pelvis_xyz[:,2].min():.3f},  obj z init: {obj_xyz[0,2]:.3f},  obj z max: {obj_xyz[:,2].max():.3f}")
    print(f"hand world z mean: L={np.mean(hand_world_z[0]):.3f} R={np.mean(hand_world_z[1]):.3f}")
    print(f"hand world z min:  L={np.min(hand_world_z[0]):.3f} R={np.min(hand_world_z[1]):.3f}")
    for s in hand_local_strs:
        print(f"  {s}")
    print(f"face counts (in object local frame):")
    print(f"  L: {face_counts[0]} (active {active_cnt[0]} frames)")
    print(f"  R: {face_counts[1]} (active {active_cnt[1]} frames)")
    print(f"signed dist to chosen face mean (>0 outside, <0 inside): "
          f"L={np.mean(hand_face_dist[0]):+.3f} R={np.mean(hand_face_dist[1]):+.3f}")
    # vfrac = vertical fraction of contact on the chosen face
    # report obj initial bottom in world
    print(f"object bottom in world (obj_z_init - half_z): {obj_xyz[0,2] - half[2]:.3f}")
    print(f"object top in world (obj_z_init + half_z): {obj_xyz[0,2] + half[2]:.3f}  (LOCAL frame, may differ from physical 'top' after rotation)")


def main():
    for label, td in CASES.items():
        analyze(label, td)


if __name__ == "__main__":
    main()
