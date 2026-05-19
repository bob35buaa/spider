"""Common evaluation input schema (E019 P1).

Used by ``paper_metrics`` to evaluate trajectories from heterogeneous sources
(spider physical rollouts, holosoma v2 kinematic outputs, future OmniRetarget
baselines …) under a single schema. The dataclass deliberately keeps the
minimum needed for paper-aligned metrics; per-source extra fields live on the
``extras`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import mujoco


@dataclass
class EvalInputs:
    """Minimum data needed to compute paper-aligned eval metrics.

    Attributes
    ----------
    method:
        Source label (e.g. ``"spider_E018b"``, ``"holosoma_v2_kinematic"``).
        Used for table column / file naming.
    case:
        Case identifier (e.g. ``"box025_p2"``). Should be **the canonical short
        name** without method or canonical_t02 suffix — adapters strip these.
    model:
        Loaded ``mujoco.MjModel`` for FK / penetration / foot xy.
    qpos_sim:
        Trajectory whose paper-aligned metrics we want. Shape ``(T, nq)`` after
        adapters have flattened any parallel-env / substep dimensions.
    qpos_ref:
        Reference trajectory. For spider physical = hydra-reloaded kinematic
        retarget. For holosoma kinematic = ``None`` (or self when sim/ref split
        is not applicable). When ``None``, body tracking metrics are skipped.
    fps:
        Saved frame rate (Hz). spider = 60, holosoma v2 = 30. Per-case so foot
        skating and smoothness use the correct timestep.
    case_window:
        ``(start_frame, end_frame_exclusive)`` for case-window-restricted means.
        ``None`` ⇒ full trajectory.
    human_joints:
        SMPL-X 22-joint Z-up positions ``(T, 22, 3)`` from the source demo.
        Required by OmniRetarget contact preservation 28cm local-frame variant
        and (optionally) by foot skating's demo-based stance detection.
    object_poses:
        Demo object pose ``(T, 7)`` ``[qw qx qy qz x y z]`` or
        ``[x y z qw qx qy qz]`` — adapters should normalise to MuJoCo
        ``[x y z qw qx qy qz]`` to match qpos[-7:].
    extras:
        Per-source extras (config paths, anchor metadata, …) that flow through
        to the per-case JSON.
    """

    method: str
    case: str
    model: mujoco.MjModel
    qpos_sim: np.ndarray
    qpos_ref: np.ndarray | None
    fps: float
    case_window: tuple[int, int] | None = None
    human_joints: np.ndarray | None = None
    object_poses: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.qpos_sim.shape[0])

    def window(self) -> tuple[int, int]:
        if self.case_window is None:
            return 0, self.T
        start, end = self.case_window
        start = max(0, min(int(start), self.T))
        end = max(start, min(int(end), self.T))
        return start, end

    def summary_template(self) -> dict[str, Any]:
        """Mirror of the spider summary dict shape that ``paper_metrics``
        functions expect ``case_window_*`` keys on.
        """
        start, end = self.window()
        return {
            "variant": self.case,
            "case": self.case,
            "T": self.T,
            "case_window_start_frame": start,
            "case_window_end_frame": max(end - 1, start),
            "fps": self.fps,
        }


def detect_object_body(model: mujoco.MjModel) -> int:
    """Return body id of the freejoint object whose qpos sits at ``nq-7``.

    Matches the convention used by ``paper_metrics._add_penetration_metrics_mj``.
    Returns -1 if not found.
    """
    nq = model.nq
    for b in range(model.nbody - 1, 0, -1):
        for j in range(model.body_jntnum[b]):
            jid = int(model.body_jntadr[b] + j)
            if (
                int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
                and int(model.jnt_qposadr[jid]) == nq - 7
            ):
                return int(b)
    return -1


def repo_root() -> Path:
    """Return spider repo root (where ``examples/run_mjwp.py`` lives)."""
    here = Path(__file__).resolve()
    # adapters/common_inputs.py(0) -> eval(1) -> scripts(2) ->
    # core4d_collab_retarget(3) -> workspace(4) -> spider/(5)
    return here.parents[5]  # spider repo root
