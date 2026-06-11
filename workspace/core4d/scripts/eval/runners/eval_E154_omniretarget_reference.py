#!/usr/bin/env python3
"""Evaluate E154 OmniRetarget references for the E152/E153 3-case set.

The CEM cases use the local OmniRetarget/kinematic reference as input:

  <case>/0/trajectory_kinematic.npz

That file uses the freejoint object scene (`scene.xml`, nq=43).  The shared
metric evaluator expects the evaluated qpos to match the chosen scene, so this
script converts the reference to the corresponding `scene_act.xml` object
parameterization (nq=42), then evaluates it with the same E154 tracking and
real 3cm mask metrics used by E152/E153.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.core_metrics import METRIC_FIELDS, evaluate_sequence, person_idx_from_case  # noqa: E402


REPO = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST = REPO / "workspace/core4d/results/E152/axis1_hand_object_physics_gate/cases_manifest.tsv"
DEFAULT_OUT_DIR = REPO / "workspace/core4d/results/E154/omniretarget_eval"
DEFAULT_OUT_TSV = DEFAULT_OUT_DIR / "e154_omniretarget_method_metrics.tsv"


def repo_path(text: str | Path) -> Path:
    p = Path(text)
    return p if p.is_absolute() else REPO / p


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except Exception:
        return str(path)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else ""
    return str(value)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def scene_euler_convention(scene_act: Path) -> str:
    meta = scene_act.parent / "scene_act_meta.json"
    if meta.is_file():
        return str(json.loads(meta.read_text(encoding="utf-8")).get("euler_convention", "XYZ"))
    return "XYZ"


def convert_freejoint_to_scene_act(qpos: np.ndarray, scene_act: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    if qpos.shape[1] == model.nq:
        return qpos.astype(np.float64, copy=True)
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(f"cannot convert qpos shape={qpos.shape} for scene nq={model.nq}: {scene_act}")

    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_body < 0:
        raise ValueError(f"scene has no object body: {scene_act}")

    obj_pos_world = qpos[:, nq_robot : nq_robot + 3]
    obj_quat_wxyz = qpos[:, nq_robot + 3 : nq_robot + 7]
    body_pos = model.body_pos[obj_body]
    body_quat_wxyz = model.body_quat[obj_body]
    body_quat_xyzw = [body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]]
    r_body = R.from_quat(body_quat_xyzw)

    obj_slide = r_body.inv().apply(obj_pos_world - body_pos[np.newaxis, :])
    obj_quat_xyzw = np.column_stack(
        [obj_quat_wxyz[:, 1], obj_quat_wxyz[:, 2], obj_quat_wxyz[:, 3], obj_quat_wxyz[:, 0]]
    )
    obj_euler = (r_body.inv() * R.from_quat(obj_quat_xyzw)).as_euler(scene_euler_convention(scene_act))

    out = np.zeros((qpos.shape[0], model.nq), dtype=np.float64)
    out[:, :nq_robot] = qpos[:, :nq_robot]
    out[:, nq_robot : nq_robot + 3] = obj_slide
    out[:, nq_robot + 3 : nq_robot + 6] = obj_euler
    return out


def baseline_rows(manifest: Path) -> list[dict[str, str]]:
    rows = [row for row in read_tsv(manifest) if row.get("method") == "baseline"]
    if not rows:
        raise ValueError(f"no baseline rows in manifest: {manifest}")
    return rows


def evaluate_row(row: dict[str, str], out_dir: Path, manifest: Path) -> dict[str, Any]:
    trajectory = repo_path(row["trajectory"])
    scene_act = repo_path(row["base_scene_act"])
    mask_path = repo_path(row["mask_path"])

    data = np.load(trajectory, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.ndim != 2:
        raise ValueError(f"unsupported qpos shape={qpos.shape}: {trajectory}")

    qpos_dir = out_dir / "omni_converted_qpos"
    qpos_dir.mkdir(parents=True, exist_ok=True)
    qpos_path = qpos_dir / f"E154_{row['short_case_id']}_omniretarget_scene_act_qpos.npz"
    np.savez_compressed(qpos_path, qpos=convert_freejoint_to_scene_act(qpos, scene_act))

    eval_row = dict(row)
    eval_row["variant"] = f"E154_{row['short_case_id']}_omniretarget"
    item = evaluate_sequence(
        row=eval_row,
        method="OmniRetarget",
        hand_collision_variant_id="OmniRetarget",
        qpos_path=qpos_path,
        scene_xml=scene_act,
        kin_ref_path=trajectory,
        contact_mask_path=mask_path,
        person_idx=person_idx_from_case(row["short_case_id"]),
    )
    item.update(
        {
            "short_case_id": row["short_case_id"],
            "method_key": "OmniRetarget",
            "source_exp": "E154",
            "source_manifest": rel(manifest),
            "source_trajectory": rel(trajectory),
            "contact_mask_path": rel(mask_path),
            "qpos_path": rel(qpos_path),
            "scene_xml": rel(scene_act),
        }
    )
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-tsv", type=Path, default=DEFAULT_OUT_TSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [evaluate_row(row, args.out_dir, args.manifest) for row in baseline_rows(args.manifest)]
    fields = [
        "method_key",
        "source_exp",
        "short_case_id",
        "source_manifest",
        "source_trajectory",
        "contact_mask_path",
        *METRIC_FIELDS,
    ]
    write_tsv(args.out_tsv, rows, fields)
    print(f"saved -> {args.out_tsv}")


if __name__ == "__main__":
    main()
