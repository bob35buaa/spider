#!/usr/bin/env python3
"""Build E092 Stage B rl_from_spider tasks from Stage A dynamic full results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
OVERRIDE_ROOT = REPO / "examples/config/override"
BASE_VARIANTS = REPO / "workspace/core4d/scripts/E092/variants.tsv"
RL_VARIANTS_OUT = REPO / "workspace/core4d/scripts/E092/variants_rl_spider.tsv"
SPIDER_FULL = REPO / "workspace/core4d/results/E092/spider_dyn/full"
BUILD_META_OUT = REPO / "workspace/core4d/results/E092/rl_from_spider/task_build_meta.json"
E083_DIR = REPO / "workspace/core4d/scripts/E083"

if str(E083_DIR) not in sys.path:
    sys.path.insert(0, str(E083_DIR))

import create_upperobj_cases as upperobj  # type: ignore  # noqa: E402


FIELDS = [
    "route",
    "case_id",
    "variant",
    "source_task",
    "derived_task",
    "person_idx",
    "split",
    "object",
    "role",
    "note",
]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _read_variants(path: Path, route: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(
            (line for line in f if line.strip() and not line.startswith("#")),
            delimiter="\t",
            fieldnames=FIELDS,
        )
        for row in reader:
            if route is None or row["route"] == route:
                rows.append(row)
    return rows


def _read_work_variants(summary: Path) -> set[str]:
    if not summary.is_file():
        return set()
    data = json.loads(summary.read_text(encoding="utf-8"))
    return {
        variant
        for variant, row in data.items()
        if row.get("work_status") == "WORK" or row.get("stage_pass") is True
    }


def _scene_act_to_freejoint_qpos(qpos_act: np.ndarray, scene_act: Path) -> np.ndarray:
    """Invert run_mjwp.py's freejoint->scene_act object conversion."""
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_bid == -1:
        raise ValueError(f"object body not found in {scene_act}")
    meta_path = scene_act.with_name("scene_act_meta.json")
    euler_conv = "XYZ"
    if meta_path.is_file():
        euler_conv = json.loads(meta_path.read_text(encoding="utf-8")).get(
            "euler_convention", "XYZ"
        )
    nq_robot = qpos_act.shape[1] - 6
    body_pos = model.body_pos[obj_bid].copy()
    body_quat_wxyz = model.body_quat[obj_bid].copy()
    r_body = R.from_quat(
        [body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]]
    )
    slide = qpos_act[:, nq_robot : nq_robot + 3]
    euler = qpos_act[:, nq_robot + 3 : nq_robot + 6]
    world_pos = body_pos[None, :] + r_body.apply(slide)
    r_joint = R.from_euler(euler_conv, euler)
    r_world = r_body * r_joint
    quat_xyzw = r_world.as_quat()
    quat_wxyz = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])
    qpos_free = np.zeros((qpos_act.shape[0], nq_robot + 7), dtype=np.float64)
    qpos_free[:, :nq_robot] = qpos_act[:, :nq_robot]
    qpos_free[:, nq_robot : nq_robot + 3] = world_pos
    qpos_free[:, nq_robot + 3 : nq_robot + 7] = quat_wxyz
    return qpos_free


def _validate_task(task_dir: Path) -> dict[str, object]:
    scene_act = task_dir / "scene_act.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    qpos = np.load(task_dir / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    root = ET.parse(scene_act).getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    required_leg = {f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS}
    required_upper = {f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS}
    report = {
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "model_npair": int(model.npair),
        "trajectory_qpos_shape": list(qpos.shape),
        "missing_leg_pairs": sorted(required_leg - pair_names),
        "missing_upper_pairs": sorted(required_upper - pair_names),
    }
    ok = (
        model.nq == 42
        and model.nv == 41
        and model.nu == 35
        and qpos.ndim == 2
        and qpos.shape[1] == 43
        and not report["missing_leg_pairs"]
        and not report["missing_upper_pairs"]
    )
    report["validation_ok"] = ok
    if not ok:
        raise RuntimeError(f"E092 rl_from_spider validation failed for {task_dir}: {report}")
    return report


def _write_override(row: dict[str, str]) -> Path:
    path = OVERRIDE_ROOT / f"core4d_{row['variant']}.yaml"
    text = f"""# @package _global_
# Auto-generated by workspace/core4d/scripts/E092/build_rl_tasks.py.
# E092 Stage B RL-from-SPIDER route; current executable entry is MJWP.
defaults:
  - core4d_E089A_box021_person1_upperobj
  - _self_

task: {row['derived_task']}

contact_hdmi_target_source: ref_fk
contact_hdmi_target_path: ""
contact_hdmi_target_uses_eef_offset: true
contact_hdmi_mask_source: ""
contact_hdmi_mask_path: ""
contact_hdmi_mask_person_idx: {row['person_idx']}
contact_hdmi_mask_time_axis: auto
hold_contact_rew_scale: 0.0
hold_contact_start_eval_time: 0.0
hold_contact_end_eval_time: 0.0
"""
    path.write_text(text, encoding="utf-8")
    return path


def _build_one(row: dict[str, str], *, force: bool) -> dict[str, object]:
    full_npz = SPIDER_FULL / f"{row['variant']}_outdir_full/trajectory_mjwp_act.npz"
    if not full_npz.is_file():
        raise FileNotFoundError(full_npz)
    src = TASK_ROOT / row["derived_task"]
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst_task = f"{row['source_task']}_e092_rl_from_spider"
    dst = TASK_ROOT / dst_task
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[SKIP] derived task exists: {_relative(dst)}")
    if not dst.exists():
        shutil.copytree(src, dst)

    data = np.load(full_npz, allow_pickle=True)
    qpos_act = data["qpos"][:, 0, :].astype(np.float64)
    qvel = data["qvel"][:, 0, :].astype(np.float64)
    qpos_free = _scene_act_to_freejoint_qpos(qpos_act, dst / "scene_act.xml")
    ctrl = qpos_free[:, 7:36].astype(np.float64)
    contact = np.zeros((qpos_free.shape[0], 2), dtype=np.float64)
    contact_pos = np.zeros((qpos_free.shape[0], 2, 3), dtype=np.float64)
    np.savez(
        dst / "0/trajectory_kinematic.npz",
        qpos=qpos_free,
        qvel=qvel,
        ctrl=ctrl,
        contact=contact,
        contact_pos=contact_pos,
    )
    info_path = dst / "task_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    info.update(
        {
            "task": dst_task,
            "source_task_e092_rl_from_spider": row["derived_task"],
            "source_full_rollout_e092": _relative(full_npz),
            "e092_route": "rl_spider",
            "e092_variant": row["variant"].replace("D", "S", 1).replace("_dyn", "_spider"),
            "e092_note": "Reference qpos converted from Stage A dynamic full sim qpos.",
        }
    )
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    new_row = {
        "route": "rl_spider",
        "case_id": row["case_id"],
        "variant": row["variant"].replace("D", "S", 1).replace("_dyn", "_spider"),
        "source_task": row["derived_task"],
        "derived_task": dst_task,
        "person_idx": row["person_idx"],
        "split": row["split"],
        "object": row["object"],
        "role": row["role"],
        "note": f"Built from Stage A WORK rollout {row['variant']}.",
    }
    override = _write_override(new_row)
    validation = _validate_task(dst)
    meta = {
        **new_row,
        "task_dir": _relative(dst),
        "override": _relative(override),
        "source_full_npz": _relative(full_npz),
        "validation": validation,
    }
    print(
        f"{row['variant']} -> {new_row['variant']}: "
        f"T={qpos_free.shape[0]} qpos={qpos_free.shape[1]} validation=ok"
    )
    return meta


def _write_variants(rows: list[dict[str, object]]) -> None:
    RL_VARIANTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with RL_VARIANTS_OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("# E092 rl_from_spider variants generated by build_rl_tasks.py\n")
        f.write(
            "# route\tcase_id\tvariant\tsource_task\tderived_task\tperson_idx\t"
            "split\tobject\trole\tnote\n"
        )
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow([row[field] for field in FIELDS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all-full", action="store_true", help="Build from every full rollout, ignoring WORK status.")
    parser.add_argument("--summary", type=Path, default=SPIDER_FULL / "full_eval_summary.json")
    args = parser.parse_args()

    rows = _read_variants(BASE_VARIANTS, route="spider_dyn")
    work = _read_work_variants(args.summary)
    selected = rows if args.all_full else [row for row in rows if row["variant"] in work]
    if not selected:
        raise SystemExit(
            f"No Stage A WORK variants found in {args.summary}. "
            "Use --all-full only for explicit diagnostic runs."
        )
    built = [_build_one(row, force=args.force) for row in selected]
    _write_variants(built)
    BUILD_META_OUT.parent.mkdir(parents=True, exist_ok=True)
    BUILD_META_OUT.write_text(json.dumps(built, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {_relative(RL_VARIANTS_OUT)}")
    print(f"wrote {_relative(BUILD_META_OUT)}")


if __name__ == "__main__":
    main()
