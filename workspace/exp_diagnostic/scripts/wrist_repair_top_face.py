"""E089 B-path implementation: project G1 wrist FK target onto box world-up face + 5cm.

Reads an existing SPIDER trajectory_kinematic.npz + scene.xml, computes per-frame
wrist target as the world-up-most box face + 5cm (preserving the lateral position
of the original wrist target, clipped 2cm inside the face), and runs damped
Gauss-Newton IK on a 17-DoF subset (left arm 7 + right arm 7 + waist 3) using
MuJoCo's analytical Jacobian to land the wrist (with the standard 5cm forward
eef offset) on that target.

All other DOFs (legs, free joint, object freejoint) are preserved frame-by-frame.

Output: a new trajectory_kinematic.npz with patched qpos written to the
specified output directory.

Usage:
    python wrist_repair_top_face.py \
        --task <task_dir_name> \
        [--out-task <new_task_dir_name>] \
        [--source-root spider/example_datasets/.../humanoid_object]

If --out-task is not provided, defaults to <task>_btoprepair.

Notes
- The "top face" is the box-local axis whose outward normal best aligns with
  world +z under the per-frame object quaternion. Sign is +/- of that axis.
- We always project (not only "near"), because the diagnostic showed every box021
  D003 case has wrist deeply embedded or on side faces; the gate explicitly
  requires top_face_frac >= 20%.
- We clip lateral target within (-h+0.02, +h-0.02) so the wrist is on the face,
  not on the edge.
- eef_offset = [0.05, 0, 0] in the wrist body local frame, matching the SPIDER
  reward "ref_fk" target convention.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
import mujoco

REPO = Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider")
TASKS_DIR = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

EEF_OFFSET = np.array([0.05, 0.0, 0.0])
TOP_OFFSET_M = 0.05      # how far above the top face the wrist target sits
EDGE_MARGIN_M = 0.02     # how far inside the face edge to clip

# 17 DoF for IK: left arm (7) + right arm (7) + waist (3).
# DOF addresses for G1 (verified via mj_name2id on the scene):
ARM_WAIST_DOF_LEFT = list(range(21, 28))   # left_shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
ARM_WAIST_DOF_RIGHT = list(range(28, 35))  # right_*
WAIST_DOF = [18, 19, 20]                   # waist_yaw/roll/pitch
ALL_IK_DOFS = WAIST_DOF + ARM_WAIST_DOF_LEFT + ARM_WAIST_DOF_RIGHT

# Same indices for qpos (model has no quat joints in these arms — all hinge,
# so dof_adr - 3 = qpos_adr - 3 across these ranges. We use jnt_qposadr at runtime.)


def parse_box_collision(scene_xml: Path):
    """Return (half_extents_array (3,), object_body_name)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.attrib.get("name") == "object":
            for g in body.iter("geom"):
                if g.attrib.get("type") == "box" and "collision" in g.attrib.get("name", "").lower():
                    half = np.array([float(x) for x in g.attrib["size"].split()])
                    return half, body.attrib["name"]
    raise RuntimeError("No object collision box found in scene.xml")


def quat_to_mat(q_wxyz):
    """MuJoCo-style (w,x,y,z) quat -> 3x3 rotation matrix."""
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q_wxyz)
    return out.reshape(3, 3)


def world_up_face(box_quat_wxyz, half):
    """Return (axis_idx, sign, half_extent_along_axis).

    The face whose outward normal best aligns with world +z, expressed in the
    box local frame. axis_idx in {0,1,2}, sign in {+1,-1}, half = half[axis_idx].
    """
    R = quat_to_mat(box_quat_wxyz)             # world ← local
    local_up = R.T @ np.array([0.0, 0.0, 1.0]) # local components of world +z
    axis = int(np.argmax(np.abs(local_up)))
    sign = float(np.sign(local_up[axis])) or 1.0
    return axis, sign, float(half[axis])


def project_target_to_top(hand_world, box_pos, box_quat_wxyz, half):
    """Compute world-space wrist target projected onto the world-up face + 5cm."""
    R = quat_to_mat(box_quat_wxyz)
    hand_local = R.T @ (hand_world - box_pos)
    axis, sign, h_axis = world_up_face(box_quat_wxyz, half)
    target_local = hand_local.copy()
    other = [a for a in (0, 1, 2) if a != axis]
    for a in other:
        target_local[a] = float(np.clip(hand_local[a],
                                        -half[a] + EDGE_MARGIN_M,
                                        +half[a] - EDGE_MARGIN_M))
    target_local[axis] = sign * (h_axis + TOP_OFFSET_M)
    target_world = R @ target_local + box_pos
    return target_world


def wrist_world(model, data, body_id):
    """Return (pos, jac_pos) for the wrist body's eef point (pos + eef_offset in body frame)."""
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3)
    world_pt = pos + mat @ EEF_OFFSET
    return world_pt


def ik_single_frame(model, data, bidL, bidR, target_L, target_R,
                    dofs, q_lb, q_ub, n_iter=25, damping=1e-2,
                    step_clip=0.25, tol=1e-3):
    """Solve damped LS IK so wrist eef points reach target_L / target_R."""
    dofs = np.array(dofs, dtype=int)
    n = len(dofs)
    # qpos addresses for these dofs (1:1 since they are all hinge joints)
    qpos_adrs = np.array([model.jnt_qposadr[model.dof_jntid[d]] for d in dofs])

    jac_pos_L = np.zeros((3, model.nv))
    jac_rot_L = np.zeros((3, model.nv))
    jac_pos_R = np.zeros((3, model.nv))
    jac_rot_R = np.zeros((3, model.nv))

    for it in range(n_iter):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        # Recompute eef in world frame
        mat_L = data.xmat[bidL].reshape(3, 3)
        mat_R = data.xmat[bidR].reshape(3, 3)
        eef_L_world = data.xpos[bidL] + mat_L @ EEF_OFFSET
        eef_R_world = data.xpos[bidR] + mat_R @ EEF_OFFSET
        err_L = target_L - eef_L_world
        err_R = target_R - eef_R_world
        err = np.concatenate([err_L, err_R])
        if np.linalg.norm(err) < tol:
            break

        # Body-point jacobian: jacobian of (xpos + R*offset) w.r.t. qvel.
        mujoco.mj_jacBody(model, data, jac_pos_L, jac_rot_L, bidL)
        mujoco.mj_jacBody(model, data, jac_pos_R, jac_rot_R, bidR)
        # Approx: shift the body-frame point along the rotation jacobian.
        # For small offset, dp_world = jac_pos + jac_rot x R*offset
        off_L_world = mat_L @ EEF_OFFSET
        off_R_world = mat_R @ EEF_OFFSET
        # Cross product of jacobian rot columns with offset_world: for each dof
        # contribution dp = (jac_rot[:,d] x off_world)
        def shift_jac(jp, jr, off):
            return jp + np.cross(jr.T, off).T
        jp_L_full = shift_jac(jac_pos_L, jac_rot_L, off_L_world)
        jp_R_full = shift_jac(jac_pos_R, jac_rot_R, off_R_world)
        J = np.vstack([jp_L_full[:, dofs], jp_R_full[:, dofs]])  # (6 x n)
        # Damped LS
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # step clip
        nrm = np.linalg.norm(dq)
        if nrm > step_clip:
            dq = dq * (step_clip / nrm)
        # Apply
        for i, adr in enumerate(qpos_adrs):
            v = data.qpos[adr] + dq[i]
            v = float(np.clip(v, q_lb[i], q_ub[i]))
            data.qpos[adr] = v
    return np.linalg.norm(err)


def repair_one_case(task: str, out_task: str | None = None,
                    source_root: Path = TASKS_DIR,
                    verbose: bool = True):
    in_dir = source_root / task
    p_scene = in_dir / "scene.xml"
    p_npz = in_dir / "0" / "trajectory_kinematic.npz"
    if not p_scene.exists() or not p_npz.exists():
        raise FileNotFoundError(f"Missing scene.xml or kinematic npz under {in_dir}")

    half, _ = parse_box_collision(p_scene)
    model = mujoco.MjModel.from_xml_path(str(p_scene))
    data = mujoco.MjData(model)
    bidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    bidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # Joint bounds for IK dofs (closed form)
    q_lb = []
    q_ub = []
    for d in ALL_IK_DOFS:
        jid = model.dof_jntid[d]
        if model.jnt_limited[jid]:
            q_lb.append(model.jnt_range[jid, 0])
            q_ub.append(model.jnt_range[jid, 1])
        else:
            q_lb.append(-np.pi)
            q_ub.append(np.pi)
    q_lb = np.array(q_lb)
    q_ub = np.array(q_ub)

    npz = dict(np.load(p_npz))
    qpos = npz["qpos"].copy()
    T = qpos.shape[0]
    qpos_adrs = np.array([model.jnt_qposadr[model.dof_jntid[d]] for d in ALL_IK_DOFS])

    res = {"T": int(T), "frames_converged_lt_3cm": 0, "max_resid": 0.0,
            "mean_resid": 0.0, "wrist_L_inside_before": 0, "wrist_R_inside_before": 0,
            "wrist_L_inside_after": 0, "wrist_R_inside_after": 0}
    resids = []
    inside_before_L = 0; inside_before_R = 0
    inside_after_L = 0; inside_after_R = 0

    for t in range(T):
        # Reset to original qpos for this frame
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        # Read original wrist eef + object pose
        mat_L = data.xmat[bidL].reshape(3, 3)
        mat_R = data.xmat[bidR].reshape(3, 3)
        eef_L0 = data.xpos[bidL] + mat_L @ EEF_OFFSET
        eef_R0 = data.xpos[bidR] + mat_R @ EEF_OFFSET
        obj_pos = data.xpos[obj_bid].copy()
        obj_quat = data.xquat[obj_bid].copy()
        # Project targets
        target_L = project_target_to_top(eef_L0, obj_pos, obj_quat, half)
        target_R = project_target_to_top(eef_R0, obj_pos, obj_quat, half)
        # IK
        resid = ik_single_frame(model, data, bidL, bidR, target_L, target_R,
                                ALL_IK_DOFS, q_lb, q_ub)
        # Capture new qpos for these dofs
        for i, adr in enumerate(qpos_adrs):
            qpos[t, adr] = data.qpos[adr]
        # Re-evaluate inside-box stats
        mujoco.mj_forward(model, data)
        eef_L1 = data.xpos[bidL] + data.xmat[bidL].reshape(3,3) @ EEF_OFFSET
        eef_R1 = data.xpos[bidR] + data.xmat[bidR].reshape(3,3) @ EEF_OFFSET
        R = quat_to_mat(obj_quat)
        L_loc0 = R.T @ (eef_L0 - obj_pos); R_loc0 = R.T @ (eef_R0 - obj_pos)
        L_loc1 = R.T @ (eef_L1 - obj_pos); R_loc1 = R.T @ (eef_R1 - obj_pos)
        if np.all(np.abs(L_loc0) < half): inside_before_L += 1
        if np.all(np.abs(R_loc0) < half): inside_before_R += 1
        if np.all(np.abs(L_loc1) < half): inside_after_L += 1
        if np.all(np.abs(R_loc1) < half): inside_after_R += 1
        resids.append(resid)
        if resid < 0.03:
            res["frames_converged_lt_3cm"] += 1

    res["max_resid"] = float(np.max(resids))
    res["mean_resid"] = float(np.mean(resids))
    res["wrist_L_inside_before"] = inside_before_L
    res["wrist_R_inside_before"] = inside_before_R
    res["wrist_L_inside_after"] = inside_after_L
    res["wrist_R_inside_after"] = inside_after_R

    # Save patched npz to a new task directory (mirror scene.xml etc.)
    if out_task is None:
        out_task = task + "_btoprepair"
    out_dir = source_root / out_task
    out_dir.mkdir(parents=True, exist_ok=True)
    # Symlink (or copy) scene.xml + auxiliary files; only qpos changes.
    for fname in ["scene.xml", "scene_act.xml", "scene_act_meta.json",
                  "task_info.json", "object_mesh.obj"]:
        sf = in_dir / fname
        df = out_dir / fname
        if sf.exists() and not df.exists():
            # Hard-copy small XMLs (cheap & reproducible)
            try:
                df.write_bytes(sf.read_bytes())
            except Exception:
                pass
    # Copy mesh dir if it exists (Box021 has an assets dir referenced)
    mesh_src = in_dir / "object_models"
    if mesh_src.exists() and not (out_dir / "object_models").exists():
        import shutil
        shutil.copytree(mesh_src, out_dir / "object_models")

    out_npz_dir = out_dir / "0"
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    npz["qpos"] = qpos
    np.savez(out_npz_dir / "trajectory_kinematic.npz", **npz)

    if verbose:
        print(f"[{task}] T={T}  conv<3cm {res['frames_converged_lt_3cm']}/{T}  "
              f"mean_resid={res['mean_resid']*100:.2f}cm  max={res['max_resid']*100:.2f}cm  "
              f"inside L: {inside_before_L}->{inside_after_L}  R: {inside_before_R}->{inside_after_R}")
        print(f"   wrote: {out_dir}/0/trajectory_kinematic.npz")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-task", default=None)
    ap.add_argument("--source-root", type=Path, default=TASKS_DIR)
    args = ap.parse_args()
    repair_one_case(args.task, args.out_task, args.source_root)


if __name__ == "__main__":
    main()
