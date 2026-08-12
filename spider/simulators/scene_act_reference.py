"""Fail-closed Euler reference contract for pre-built scene-act models."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import loguru
import mujoco
import numpy as np


VALID_EULER_CONVENTIONS = frozenset(
    "".join(order) for order in itertools.permutations("XYZ")
)


@dataclass(frozen=True)
class SceneActReference:
    """Resolved and audited Euler contract for one compiled scene-act model."""

    convention: str
    xml_axis_sequence: str
    meta_path: Path
    meta_sha256: str
    object_body_id: int
    object_body_name: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_object_body(
    model: mujoco.MjModel, body_names: Iterable[str]
) -> tuple[int, str]:
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id), name
    raise ValueError(
        "scene-act reference contract requires an object body named one of "
        f"{tuple(body_names)!r}"
    )


def object_hinge_axis_sequence(model: mujoco.MjModel, body_id: int) -> str:
    """Return the positive XYZ hinge sequence in compiled qpos order."""

    start = int(model.body_jntadr[body_id])
    stop = start + int(model.body_jntnum[body_id])
    hinge_ids = [
        joint_id
        for joint_id in range(start, stop)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    hinge_ids.sort(key=lambda joint_id: int(model.jnt_qposadr[joint_id]))
    letters: list[str] = []
    for joint_id in hinge_ids:
        axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
        axis_index = int(np.argmax(np.abs(axis)))
        expected = np.zeros(3, dtype=np.float64)
        expected[axis_index] = 1.0
        if not np.allclose(axis, expected, atol=1e-9, rtol=0.0):
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            raise ValueError(
                "scene-act object hinge axes must be positive XYZ unit vectors; "
                f"joint={joint_name!r} axis={axis.tolist()}"
            )
        letters.append("XYZ"[axis_index])
    sequence = "".join(letters)
    if len(sequence) != 3 or len(set(sequence)) != 3:
        raise ValueError(
            "scene-act object must expose exactly three unique XYZ hinges; "
            f"got {sequence!r}"
        )
    return sequence


def resolve_scene_act_reference(
    model_path: str | Path,
    model: mujoco.MjModel,
    *,
    body_names: Iterable[str] = ("object", "suitcase"),
    emit_log: bool = True,
) -> SceneActReference:
    """Resolve the Euler convention and reject missing or inconsistent metadata."""

    scene_path = Path(model_path)
    meta_path = scene_path.with_name("scene_act_meta.json")
    if not meta_path.is_file() or meta_path.stat().st_size == 0:
        raise FileNotFoundError(
            "scene-act reference metadata is required; no Euler fallback is allowed: "
            f"{meta_path}"
        )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid scene-act reference metadata {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"scene-act reference metadata must be a JSON object: {meta_path}")
    convention = payload.get("euler_convention")
    if not isinstance(convention, str) or convention not in VALID_EULER_CONVENTIONS:
        raise ValueError(
            "scene-act euler_convention must be one of "
            f"{sorted(VALID_EULER_CONVENTIONS)}; got {convention!r} in {meta_path}"
        )
    body_id, body_name = _find_object_body(model, tuple(body_names))
    axis_sequence = object_hinge_axis_sequence(model, body_id)
    if convention != axis_sequence:
        raise ValueError(
            "scene-act Euler metadata disagrees with compiled object hinges: "
            f"meta={convention} compiled={axis_sequence} path={meta_path}"
        )
    resolved = SceneActReference(
        convention=axis_sequence,
        xml_axis_sequence=axis_sequence,
        meta_path=meta_path,
        meta_sha256=_sha256(meta_path),
        object_body_id=body_id,
        object_body_name=body_name,
    )
    if emit_log:
        loguru.logger.info(
            "scene-act-reference: convention={} source={} meta_sha256={} "
            "xml_axis_sequence={} parity=pass object_body={}",
            resolved.convention,
            resolved.meta_path,
            resolved.meta_sha256,
            resolved.xml_axis_sequence,
            resolved.object_body_name,
        )
    return resolved
