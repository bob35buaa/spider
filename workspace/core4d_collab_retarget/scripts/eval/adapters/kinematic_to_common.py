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

Only 2 of the 13 spider E018b cases have a holosoma v2 counterpart:
``box025_p1`` and ``box025_p2``. The remaining 11 cases would need holosoma to
re-run retargeting; ``CASE_MAP`` only registers the 2-case first cohort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import mujoco

from .common_inputs import EvalInputs


HOLOSOMA_RESULT_DIR = Path(
    "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v2/"
    "results/retarget_replace_batch_trimmed"
)
HOLOSOMA_DEMO_DIR = Path(
    "/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/holosoma/workspace/v2/"
    "data/core4d_replace_batch"
)


# Map: short spider case name -> holosoma v2 retarget NPZ filename
# Extend when holosoma covers more cases.
CASE_MAP: dict[str, str] = {
    "box025_p1": "20231011-048-person1-Box025_with_obj_original.npz",
    "box025_p2": "20231011-048-person2-Box025_with_obj_original.npz",
}

# Companion object NPZ name from the same seq id (for true demo object poses)
def _companion_object_npz(retarget_npz_name: str) -> str:
    # Filename pattern: {seq_id}-personN-{Obj}_with_obj_original.npz
    seq_id = retarget_npz_name.split("-person")[0]
    return f"{seq_id}-object.npz"


def list_available_cases() -> list[str]:
    return [k for k, v in CASE_MAP.items() if (HOLOSOMA_RESULT_DIR / v).is_file()]


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
    retarget_path = HOLOSOMA_RESULT_DIR / retarget_npz_name
    if not retarget_path.is_file():
        raise FileNotFoundError(retarget_path)
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
    companion = HOLOSOMA_DEMO_DIR / _companion_object_npz(retarget_npz_name)
    object_poses_norm: np.ndarray | None = None
    if companion.is_file():
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
            "companion_npz": str(companion) if companion.is_file() else "",
            "fps": fps,
            "cost": float(data["cost"]) if "cost" in data.files else float("nan"),
        },
    )
