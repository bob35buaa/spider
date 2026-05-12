# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-snap IK: project G1 palms onto object surface inside an intent window.

Used by Path B (E055+) to inject grasp prior into CEM warmstart, replacing
unreliable mocap hand poses with object-geometry-anchored hand poses.

Solves arm-only IK (left arm = qpos[22:29], right arm = qpos[29:36]) with damped
least squares on the palm site position. Pelvis / leg / torso / object qpos are
left untouched — only arm joints are modified.

Usage (programmatic):
    from spider.preprocess.hand_snap_ik import snap_hands_to_object
    qpos_snap, diag = snap_hands_to_object(
        scene_xml="…/box023_person1/scene.xml",
        qpos_ref=np.load(".../trajectory_kinematic.npz")["qpos"],
        intent_window=(50, 101),
        dominant_hand="both",
        object_mesh_path=".../box023_m.obj",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

# G1 joint layout for the CORE4D scenes (verified for box023_person1):
#   qpos[0:7]    pelvis freejoint
#   qpos[7:13]   left leg (6)
#   qpos[13:19]  right leg (6)
#   qpos[19:22]  waist (yaw, roll, pitch)
#   qpos[22:29]  left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
#   qpos[29:36]  right arm
#   qpos[36:43]  object freejoint
LEFT_ARM_QPOS = slice(22, 29)
RIGHT_ARM_QPOS = slice(29, 36)
LEFT_ARM_DOF = slice(21, 28)   # qvel index = qpos_adr - 1 (freejoint is 6 dof in qvel)
RIGHT_ARM_DOF = slice(28, 35)


@dataclass
class SnapDiag:
    """Per-frame diagnostics for one hand."""
    frame: int
    hand: str                       # "L" or "R"
    in_intent: bool
    palm_to_surface_init: float     # before IK, m
    palm_to_surface_final: float    # after IK, m
    ik_iterations: int
    ik_residual: float
    joint_in_limits: bool
    target_xyz: tuple[float, float, float]


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """MuJoCo (w, x, y, z) → 3x3 rotation matrix."""
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def _closest_surface_point(
    mesh_world: trimesh.Trimesh, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closest point on mesh surface + outward normal + distance.

    `mesh_world` must already be in the world frame (caller transforms it).
    Returns (point, normal_outward, signed_distance). Distance > 0 if query is
    outside the mesh, < 0 if inside (uses signed_distance from trimesh).
    """
    closest, dists, face_ids = trimesh.proximity.closest_point(mesh_world, query[None, :])
    point = closest[0]
    normal = mesh_world.face_normals[face_ids[0]]
    # signed_distance: + outside, - inside
    signed = trimesh.proximity.signed_distance(mesh_world, query[None, :])[0]
    return point, normal, float(signed)


def _solve_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_xyz: np.ndarray,
    arm_qpos_slice: slice,
    arm_dof_slice: slice,
    max_iter: int = 100,
    tol: float = 1e-3,
    damping: float = 0.05,
    step_alpha: float = 0.5,
) -> tuple[int, float]:
    """Damped least-squares IK on the arm joints to reach target_xyz with the
    given site. Modifies data.qpos in-place. Returns (n_iter, best_residual).

    Uses a moderate damping factor and step scaling to avoid the oscillation
    that pure undamped LS shows on near-singular configurations. Always keeps
    the best (lowest-residual) qpos seen across iterations."""
    jacp = np.zeros((3, model.nv))
    best_residual = np.inf
    best_qpos = data.qpos[arm_qpos_slice].copy()

    # find joint ids that fall in the arm slice (for limit clipping)
    arm_jids = [
        jid for jid in range(model.njnt)
        if arm_qpos_slice.start <= model.jnt_qposadr[jid] < arm_qpos_slice.stop
    ]

    for it in range(max_iter):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        cur = data.site_xpos[site_id]
        err = target_xyz - cur
        residual = float(np.linalg.norm(err))
        if residual < best_residual:
            best_residual = residual
            best_qpos = data.qpos[arm_qpos_slice].copy()
        if residual < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, arm_dof_slice]                   # (3, 7)
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)
        data.qpos[arm_qpos_slice] += step_alpha * dq
        # clip to joint limits
        for jid in arm_jids:
            if model.jnt_limited[jid]:
                adr = model.jnt_qposadr[jid]
                lo, hi = model.jnt_range[jid]
                data.qpos[adr] = np.clip(data.qpos[adr], lo, hi)

    # restore best
    data.qpos[arm_qpos_slice] = best_qpos
    mujoco.mj_kinematics(model, data)
    return min(it + 1, max_iter), best_residual


def snap_hands_to_object(
    scene_xml: str,
    qpos_ref: np.ndarray,
    intent_window: tuple[int, int],
    dominant_hand: str,
    object_mesh_path: str,
    object_body_name: str = "object",
    surface_offset: float = 0.005,
    approach_blend_frames: int = 10,
    max_ik_iter: int = 50,
    ik_tol: float = 1e-3,
    ik_damping: float = 1e-2,
) -> tuple[np.ndarray, list[SnapDiag]]:
    """Generate snapped warmstart trajectory.

    Args:
        scene_xml: path to MuJoCo scene XML containing G1 + object.
        qpos_ref: (T, nq) reference trajectory.
        intent_window: (t_start, t_end) inclusive frame indices to snap.
        dominant_hand: "L" / "R" / "both" — which hand(s) to snap.
        object_mesh_path: path to object .obj/.stl file (in object body frame).
        surface_offset: outward normal offset applied to the IK target (m).
        approach_blend_frames: linear blend between ref and snap qpos before
            and after the intent window.
        max_ik_iter, ik_tol, ik_damping: DLS solver knobs.

    Returns:
        qpos_snap: (T, nq) modified qpos
        diag: list of SnapDiag (one per snapped frame × snapped hand)
    """
    assert dominant_hand in ("L", "R", "both"), dominant_hand

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    # disable contacts during IK (we only need kinematics)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

    left_palm_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_palm_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    assert left_palm_sid != -1 and right_palm_sid != -1 and obj_bid != -1

    mesh_local = trimesh.load(object_mesh_path, force="mesh")
    if not isinstance(mesh_local, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load mesh as Trimesh: {object_mesh_path}")

    T, nq = qpos_ref.shape
    assert nq == model.nq, f"qpos_ref nq={nq} mismatch model.nq={model.nq}"

    qpos_snap = qpos_ref.copy()
    diag: list[SnapDiag] = []

    t_start, t_end = intent_window
    hands_to_snap: list[tuple[str, int, slice, slice]] = []
    if dominant_hand in ("L", "both"):
        hands_to_snap.append(("L", left_palm_sid, LEFT_ARM_QPOS, LEFT_ARM_DOF))
    if dominant_hand in ("R", "both"):
        hands_to_snap.append(("R", right_palm_sid, RIGHT_ARM_QPOS, RIGHT_ARM_DOF))

    # store the snapped arm qpos at window boundaries to blend approach/release
    snap_arm_at_start: dict[str, np.ndarray] = {}
    snap_arm_at_end: dict[str, np.ndarray] = {}

    prev_arm_qpos: dict[str, np.ndarray] = {}
    for t in range(t_start, t_end + 1):
        # Use ref qpos as base (pelvis/leg/torso/object), but warm-start arm
        # joints from the previous frame's snap when available — consecutive
        # frames are close, this avoids restarting IK from a far init.
        data.qpos[:] = qpos_ref[t]
        for hand_name, arm_q in prev_arm_qpos.items():
            qslice = LEFT_ARM_QPOS if hand_name == "L" else RIGHT_ARM_QPOS
            data.qpos[qslice] = arm_q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # transform mesh to world via object body pose
        obj_pos = data.xpos[obj_bid].copy()
        obj_mat = data.xmat[obj_bid].copy().reshape(3, 3)
        T_obj = np.eye(4)
        T_obj[:3, :3] = obj_mat
        T_obj[:3, 3] = obj_pos
        mesh_world = mesh_local.copy()
        mesh_world.apply_transform(T_obj)

        for hand, sid, qslice, dslice in hands_to_snap:
            palm_init = data.site_xpos[sid].copy()
            point, normal, signed = _closest_surface_point(mesh_world, palm_init)
            # outward normal: trimesh face_normals already point outward; but
            # if signed < 0 (palm inside mesh) we still want to push out
            target = point + surface_offset * normal
            init_dist = float(np.linalg.norm(palm_init - point))

            n_iter, residual = _solve_arm_ik(
                model, data, sid, target, qslice, dslice,
                max_iter=max_ik_iter, tol=ik_tol, damping=ik_damping,
            )

            # check joint limits
            in_lim = True
            for jid in range(model.njnt):
                adr = model.jnt_qposadr[jid]
                if qslice.start <= adr < qslice.stop and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    if data.qpos[adr] < lo - 1e-6 or data.qpos[adr] > hi + 1e-6:
                        in_lim = False
                        break

            qpos_snap[t, qslice] = data.qpos[qslice]
            mujoco.mj_kinematics(model, data)
            palm_final = data.site_xpos[sid].copy()
            # recompute closest-surface distance for the *final* palm — the
            # surface point can drift as the palm moves, and the static "point"
            # captured before IK is only an approximation.
            closest_final, _, _ = _closest_surface_point(mesh_world, palm_final)
            final_dist_to_surface = float(np.linalg.norm(palm_final - closest_final))

            diag.append(SnapDiag(
                frame=t, hand=hand, in_intent=True,
                palm_to_surface_init=init_dist,
                palm_to_surface_final=final_dist_to_surface,
                ik_iterations=n_iter, ik_residual=residual,
                joint_in_limits=in_lim,
                target_xyz=tuple(target.tolist()),
            ))

            if t == t_start:
                snap_arm_at_start[hand] = qpos_snap[t, qslice].copy()
            if t == t_end:
                snap_arm_at_end[hand] = qpos_snap[t, qslice].copy()
            prev_arm_qpos[hand] = qpos_snap[t, qslice].copy()

    # approach blend: frames [t_start - K, t_start - 1]
    for k in range(1, approach_blend_frames + 1):
        t = t_start - k
        if t < 0:
            break
        alpha = 1.0 - (k / (approach_blend_frames + 1))   # 0 at far edge, →1 near window
        for hand, _, qslice, _ in hands_to_snap:
            if hand not in snap_arm_at_start:
                continue
            ref_arm = qpos_ref[t, qslice]
            blended = (1 - alpha) * ref_arm + alpha * snap_arm_at_start[hand]
            qpos_snap[t, qslice] = blended

    # release blend: frames [t_end + 1, t_end + K]
    for k in range(1, approach_blend_frames + 1):
        t = t_end + k
        if t >= T:
            break
        alpha = 1.0 - (k / (approach_blend_frames + 1))
        for hand, _, qslice, _ in hands_to_snap:
            if hand not in snap_arm_at_end:
                continue
            ref_arm = qpos_ref[t, qslice]
            blended = (1 - alpha) * ref_arm + alpha * snap_arm_at_end[hand]
            qpos_snap[t, qslice] = blended

    return qpos_snap, diag


def find_object_mesh(scene_xml: str) -> Path:
    """Resolve the visual mesh path from a scene.xml's <mesh file=…> entry that
    matches the scene's object body. Returns the absolute Path."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    # find first <mesh> whose name matches the object body's geom mesh attr
    obj_geom = root.find(".//body[@name='object']/geom[@mesh]")
    if obj_geom is None:
        raise RuntimeError(f"No <body name='object'><geom mesh=...> in {scene_xml}")
    mesh_name = obj_geom.attrib["mesh"]
    mesh_elem = root.find(f".//mesh[@name='{mesh_name}']")
    if mesh_elem is None:
        raise RuntimeError(f"<mesh name='{mesh_name}'> not declared in {scene_xml}")
    rel = mesh_elem.attrib["file"]
    # MuJoCo resolves <mesh file=…> relative to <compiler meshdir=…> (which is
    # itself relative to the scene.xml directory). Fall back to scene.xml dir
    # when no meshdir is set.
    compiler = root.find("./compiler")
    base = Path(scene_xml).parent
    if compiler is not None and "meshdir" in compiler.attrib:
        base = (base / compiler.attrib["meshdir"]).resolve()
    abs_path = (base / rel).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {abs_path}")
    return abs_path
