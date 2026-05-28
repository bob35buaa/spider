"""Use MuJoCo FK on qpos_ref to compute G1 left/right wrist positions, then express in
object local frame and report face / signed-dist statistics. This mirrors the
`ref_fk` target source used by SPIDER's contact_hdmi reward.

Compare against contact_pos (raw mocap) results from object_local_contact.py.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from pathlib import Path
import numpy as np
import mujoco
import xml.etree.ElementTree as ET

ROOT = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS = ROOT / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

CASES = {
    "box021_18029_p2": "d003_box021_20231018_029_p2_upperobj_e083_m10_e087",
    "box021_11035_p2": "d003_box021_20231011_035_p2_upperobj_e083",
    "box021_20019_p1": "d003_box021_20231020_019_p1_upperobj_e083",
    "box023_p2_OK":    "box023_person2",
    "box025_p2":       "box025_person2",
}

WRIST_BODY = {
    "L": "left_wrist_yaw_link",
    "R": "right_wrist_yaw_link",
}
# Default reward eef offset used in contact_hdmi (E073 onwards)
EEF_OFFSET = np.array([0.05, 0.0, 0.0])


def parse_object_collision(scene_xml):
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            for g in body.iter("geom"):
                if g.attrib.get("type") == "box" and "collision" in g.attrib.get("name", "").lower():
                    return tuple(float(x) for x in g.attrib["size"].split())
    return None


def fk_wrist_pos(model, data, qpos_row, bid):
    data.qpos[:] = qpos_row
    mujoco.mj_forward(model, data)
    pos = data.xpos[bid].copy()
    quat = data.xquat[bid].copy()  # wxyz
    # apply eef offset in wrist local frame
    # convert wxyz to scipy [x,y,z,w]
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pos_eef = pos + rot.apply(EEF_OFFSET)
    return pos, pos_eef


def analyze(label, task_dir):
    p_scene = TASKS / task_dir / "scene.xml"
    p_npz = TASKS / task_dir / "0" / "trajectory_kinematic.npz"
    if not p_scene.exists() or not p_npz.exists():
        print(f"[MISSING] {label}")
        return
    d = dict(np.load(p_npz))
    qpos = d["qpos"]
    T = qpos.shape[0]
    half = np.array(parse_object_collision(p_scene))

    model = mujoco.MjModel.from_xml_path(str(p_scene))
    data = mujoco.MjData(model)
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    bidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY["L"])
    bidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY["R"])

    L_local = np.zeros((T, 3))
    R_local = np.zeros((T, 3))
    L_world_z = np.zeros(T)
    R_world_z = np.zeros(T)
    obj_xyz_arr = np.zeros((T, 3))
    for t in range(T):
        _, L_world = fk_wrist_pos(model, data, qpos[t], bidL)
        # Note: mj_forward was already called; need to call again for R? Actually
        # state didn't change for R; fk_wrist_pos resets data.qpos and re-forwards.
        _, R_world = fk_wrist_pos(model, data, qpos[t], bidR)
        # also obj pose after forward
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].reshape(3, 3)
        L_local[t] = obj_mat.T @ (L_world - obj_pos)
        R_local[t] = obj_mat.T @ (R_world - obj_pos)
        L_world_z[t] = L_world[2]
        R_world_z[t] = R_world[2]
        obj_xyz_arr[t] = obj_pos

    def face_stats(local):
        norm = local / half
        ax = np.argmax(np.abs(norm), axis=1)
        sgn = np.where(np.take_along_axis(norm, ax[:, None], axis=1).squeeze(1) >= 0, 1, -1)
        names = ["x", "y", "z"]
        faces = {}
        signed_dists = []
        for t in range(local.shape[0]):
            a = ax[t]
            s = sgn[t]
            face = ("+" if s >= 0 else "-") + names[a]
            faces[face] = faces.get(face, 0) + 1
            face_plane = half[a] * s
            signed_dists.append(local[t, a] - face_plane)
        return faces, float(np.mean(signed_dists)), float(np.std(signed_dists))

    fcL, sdL, sdL_std = face_stats(L_local)
    fcR, sdR, sdR_std = face_stats(R_local)
    print(f"\n=== {label} ({task_dir})  T={T} ===")
    print(f"object half-extents (local): {half.tolist()}")
    print(f"L wrist (FK+eef_offset) world_z mean={L_world_z.mean():.3f} min={L_world_z.min():.3f} max={L_world_z.max():.3f}")
    print(f"R wrist (FK+eef_offset) world_z mean={R_world_z.mean():.3f} min={R_world_z.min():.3f} max={R_world_z.max():.3f}")
    print(f"L local mean: {L_local.mean(axis=0).round(3).tolist()}, R local mean: {R_local.mean(axis=0).round(3).tolist()}")
    print(f"L face counts: {fcL}")
    print(f"R face counts: {fcR}")
    print(f"L signed dist to chosen face: mean {sdL:+.3f} ± {sdL_std:.3f}")
    print(f"R signed dist to chosen face: mean {sdR:+.3f} ± {sdR_std:.3f}")
    # fraction of frames where wrist is INSIDE the box
    L_in = np.all(np.abs(L_local) < half, axis=1).mean()
    R_in = np.all(np.abs(R_local) < half, axis=1).mean()
    print(f"L INSIDE box (all 3 axes < half) frac: {L_in*100:.1f}%   R INSIDE: {R_in*100:.1f}%")


def main():
    for label, td in CASES.items():
        analyze(label, td)


if __name__ == "__main__":
    main()
