"""Adapter: holosoma v2 kinematic NPZ -> ``EvalInputs``.

Source: ``holosoma/workspace/v2/results/retarget_replace_batch_trimmed/*.npz``
        + companion demo object pose
        ``holosoma/workspace/v2/data/core4d_replace_batch/{seq}-object.npz``

Convention check (verified 2026-05-20 on box025_p2):
  * ``qpos`` shape ``(T, 43)`` float64, layout matches spider
    (root pos 0:3, root quat wxyz 3:7, 29 dof 7:36, object pos 36:39, object
    quat wxyz 39:43).
  * ``fps`` = 30 (holosoma v2 saved rate).
  * ``human_joints`` shape ``(T, 22, 3)`` float32 Z-up — directly usable for
    OmniRetarget Table II contact preservation + foot skating stance detect.
  * Frame counts match spider sim for the same case (e.g. box025_p2 T=124).

2026-05-20 dry-run extended box021_p1 via `retarget_replace_batch_extra_trimmed/`;
``CASE_MAP`` now lists 3 cases. Run holosoma batch (see
``scripts/eval/HOLOSOMA_BATCH_CMDS.md``) to cover the remaining 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import mujoco

from .common_inputs import EvalInputs


HOLOSOMA_ROOTS = [
    Path("/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma"),
    Path("/home/ubuntu/Workspace/holosoma"),
]

HOLOSOMA_RESULT_DIRS = [
    root / "workspace/v2/results/retarget_replace_batch_trimmed"
    for root in HOLOSOMA_ROOTS
] + [
    root / "workspace/v2/results/retarget_replace_batch_extra_trimmed"
    for root in HOLOSOMA_ROOTS
]
HOLOSOMA_RESULT_DIR = HOLOSOMA_RESULT_DIRS[0]  # backward-compat single dir
HOLOSOMA_DEMO_DIRS = [
    root / "workspace/v2/data/core4d_replace_batch"
    for root in HOLOSOMA_ROOTS
] + [
    root / "workspace/v2/data/core4d_replace_batch_extra"
    for root in HOLOSOMA_ROOTS
]
HOLOSOMA_DEMO_DIR = HOLOSOMA_DEMO_DIRS[0]


def _resolve_in_dirs(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        p = d / name
        if p.is_file():
            return p
    return None


# Map: short spider case name -> holosoma v2 retarget NPZ filename.
# 2026-05-20: extended from N=2 (box025 only) to N=12 via batch retarget
# (run_holosoma_batch_remaining10.sh). T_holosoma aligns spider T per case
# (9/10 exact, box023_p2 ±1 frame). desk021_p1 missing: SOCP infeasible
# (CVXPY clarabel "infeasible" on this specific motion; XML identical to
# successful desk005, so it's motion-specific not template).
CASE_MAP: dict[str, str] = {
    "box021_p1": "20231018-030-person1-Box021_with_obj_original.npz",
    "box021_p2": "20231018-030-person2-Box021_with_obj_original.npz",
    "box023_p1": "20231008-045-person1-Box023_with_obj_original.npz",
    "box023_p2": "20231008-045-person2-Box023_with_obj_original.npz",
    "box025_p1": "20231011-048-person1-Box025_with_obj_original.npz",
    "box025_p2": "20231011-048-person2-Box025_with_obj_original.npz",
    "bucket001_p1": "20231030-094-person1-bucket001_with_obj_original.npz",
    "bucket001_p2": "20231030-094-person2-bucket001_with_obj_original.npz",
    "bucket005_s2_p1": "20231002-004-person1-bucket005_with_obj_original.npz",
    "bucket005_s2_p2": "20231002-004-person2-bucket005_with_obj_original.npz",
    "bucket007_p1": "20231020-055-person1-Bucket007_with_obj_original.npz",
    "bucket007_p2": "20231020-055-person2-Bucket007_with_obj_original.npz",
    # "desk021_p1": SOCP infeasible — see comment above.
}

# Companion object NPZ name from the same seq id (for true demo object poses)
def _companion_object_npz(retarget_npz_name: str) -> str:
    # Filename pattern: {seq_id}-personN-{Obj}_with_obj_original.npz
    seq_id = retarget_npz_name.split("-person")[0]
    return f"{seq_id}-object.npz"


def list_available_cases() -> list[str]:
    out = []
    for k, v in CASE_MAP.items():
        if _resolve_in_dirs(v, HOLOSOMA_RESULT_DIRS):
            out.append(k)
    return out


def load_kinematic_inputs(
    case: str,
    model: mujoco.MjModel,
    *,
    case_window: tuple[int, int] | None = None,
) -> EvalInputs:
    """Load a single case as ``EvalInputs(method='holosoma_v2_kinematic')``.

    For kinematic data, ``qpos_sim`` is the retarget output itself and
    ``qpos_ref`` is set to the same trajectory (so body tracking error vs ref
    is ~0). The point of evaluating kinematic data is the *physics* metrics
    (penetration / foot skating / contact preservation), where reference is
    not needed.
    """
    if case not in CASE_MAP:
        raise KeyError(
            f"holosoma v2 has no retarget output for case '{case}'. "
            f"Available: {sorted(CASE_MAP)}"
        )
    retarget_npz_name = CASE_MAP[case]
    retarget_path = _resolve_in_dirs(retarget_npz_name, HOLOSOMA_RESULT_DIRS)
    if retarget_path is None:
        raise FileNotFoundError(
            f"{retarget_npz_name} not found in {HOLOSOMA_RESULT_DIRS}"
        )
    data = np.load(retarget_path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(
            f"holosoma qpos shape {qpos.shape} != model nq={model.nq}"
        )
    human_joints = (
        np.asarray(data["human_joints"], dtype=np.float64)
        if "human_joints" in data.files
        else None
    )
    fps = float(data["fps"]) if "fps" in data.files else 30.0

    # Companion demo object pose: (T_demo, 7) with layout [qw qx qy qz x y z]
    # (verified on holosoma v2 box025). Normalise to MuJoCo [x y z qw qx qy qz]
    # to match qpos[-7:].
    companion = _resolve_in_dirs(
        _companion_object_npz(retarget_npz_name), HOLOSOMA_DEMO_DIRS
    )
    object_poses_norm: np.ndarray | None = None
    if companion is not None and companion.is_file():
        obj_data = np.load(companion, allow_pickle=True)
        if "object_poses" in obj_data.files:
            raw = np.asarray(obj_data["object_poses"], dtype=np.float64)
            if raw.ndim == 2 and raw.shape[1] == 7:
                # Decide layout: holosoma demo writes [qw qx qy qz x y z]
                # (validated by w being the largest element in first frame).
                if abs(raw[0, 0]) > abs(raw[0, 3]):
                    quat = raw[:, :4]
                    pos = raw[:, 4:]
                else:
                    pos = raw[:, :3]
                    quat = raw[:, 3:]
                object_poses_norm = np.concatenate([pos, quat], axis=1)
                # Resize to retarget T
                if len(object_poses_norm) != len(qpos):
                    idx = np.round(
                        np.linspace(0, len(object_poses_norm) - 1, len(qpos))
                    ).astype(int)
                    object_poses_norm = object_poses_norm[idx]

    # Default case window = full trajectory.
    return EvalInputs(
        method="holosoma_v2_kinematic",
        case=case,
        model=model,
        qpos_sim=qpos,
        qpos_ref=qpos,  # kinematic self-ref; body tracking err ≈ 0 (expected)
        fps=fps,
        case_window=case_window,
        human_joints=human_joints,
        object_poses=object_poses_norm,
        extras={
            "source_npz": str(retarget_path),
            "companion_npz": str(companion) if companion else "",
            "fps": fps,
            "cost": float(data["cost"]) if "cost" in data.files else float("nan"),
        },
    )
