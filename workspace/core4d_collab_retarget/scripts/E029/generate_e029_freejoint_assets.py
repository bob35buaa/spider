#!/usr/bin/env python3
"""Generate E029 freejoint support-body scenes and references."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e029_common as common  # noqa: E402
import generate_e029_d6_assets as d6_assets  # noqa: E402

E016_DIR = common.REPO / "workspace/core4d_collab_retarget/scripts/E016"
if str(E016_DIR) not in sys.path:
    sys.path.insert(0, str(E016_DIR))
import generate_e016_assets as e016_assets  # noqa: E402


RESULT_ROOT = common.E029_RESULTS / "freejoint"
MANIFEST = RESULT_ROOT / "manifest.tsv"
PREFLIGHT = common.E029_RESULTS / "preflight/axis_contact_summary.csv"
SUPPORT_BODY = "support_dynamic_anchor"
WELD_NAME_PREFIX = "e029_free_support_weld"

PROFILES = {
    "free_locked": {
        "support_dynamic_mass": 2.0,
        "support_dynamic_pos_kp": 400.0,
        "support_dynamic_pos_kd": -1.0,
        "support_dynamic_rot_kp": 60.0,
        "support_dynamic_rot_kd": -1.0,
        "support_dynamic_force_clamp": 250.0,
        "support_dynamic_torque_clamp": 80.0,
        "weld_solref_timeconst": 0.02,
        "weld_solimp_1": 0.9,
        "weld_solimp_2": 0.98,
        "weld_solimp_width": 0.001,
        "full_queue": "sanity",
    },
    "free_compliant": {
        "support_dynamic_mass": 2.0,
        "support_dynamic_pos_kp": 2400.0,
        "support_dynamic_pos_kd": -1.0,
        "support_dynamic_rot_kp": 120.0,
        "support_dynamic_rot_kd": -1.0,
        "support_dynamic_force_clamp": 1200.0,
        "support_dynamic_torque_clamp": 300.0,
        "weld_solref_timeconst": 0.05,
        "weld_solimp_1": 0.75,
        "weld_solimp_2": 0.95,
        "weld_solimp_width": 0.005,
        "full_queue": "local",
    },
}

FIELDNAMES = d6_assets.FIELDNAMES + ["support_joint_layout"]


def _write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    *,
    manifest_path: Path,
    candidates_path: Path,
    preflight_path: Path,
    out_path: Path,
) -> list[dict[str, str]]:
    preflight = d6_assets._read_preflight(preflight_path)
    rows: list[dict[str, str]] = []
    for e028 in common.candidate_rows(manifest_path=manifest_path, candidates_path=candidates_path):
        pf = preflight[e028["variant"]]
        point, confidence = d6_assets._endpoint(e028, pf)
        centroid = d6_assets._parse_point(pf["selected_side_centroid"])
        half = common.object_half(e028)
        mask_slug = Path(e028["mask_path_source"]).parent.name
        for profile, params in PROFILES.items():
            variant = f"{d6_assets._variant_stem(e028['variant'])}_{profile}"
            derived_task = f"{e028['source_task']}_freejoint_legobj_e029free"
            scene_name = f"scene_e029_{profile}_{e028['source_task']}"
            data_relpath = (
                f"workspace/core4d_collab_retarget/results/E029/freejoint/data/"
                f"{variant}/trajectory_kinematic.npz"
            )
            row = {
                "variant": variant,
                "source_e028_variant": e028["variant"],
                "source_task": e028["source_task"],
                "derived_task": derived_task,
                "person_idx": e028["person_idx"],
                "profile": profile,
                "queue": str(params["full_queue"]),
                "scene_name": scene_name,
                "data_relpath": data_relpath,
                "support_endpoint_policy": "axis_remap_selected_side_centroid",
                "height_local_axis": pf["height_local_axis"],
                "selected_side_face": pf["selected_side_face"],
                "selected_side_frac": pf["selected_side_frac"],
                "selected_side_margin": pf["selected_side_margin"],
                "endpoint_confidence": confidence,
                "support_proxy_point_local_x": f"{point[0]:.8g}",
                "support_proxy_point_local_y": f"{point[1]:.8g}",
                "support_proxy_point_local_z": f"{point[2]:.8g}",
                "selected_side_centroid_x": f"{centroid[0]:.8g}",
                "selected_side_centroid_y": f"{centroid[1]:.8g}",
                "selected_side_centroid_z": f"{centroid[2]:.8g}",
                "object_half_x": f"{half[0]:.8g}",
                "object_half_y": f"{half[1]:.8g}",
                "object_half_z": f"{half[2]:.8g}",
                "hold_contact_rew_scale": "0.0",
                "hold_contact_sigma": "0.05",
                "hold_contact_start_eval_time": e028.get("hold_contact_start_eval_time", "0.64"),
                "hold_contact_end_eval_time": e028.get("hold_contact_end_eval_time", "4.08"),
                "hold_contact_require_ref_contact": e028.get("hold_contact_require_ref_contact", "true"),
                "mask_path_source": e028["mask_path_source"],
                "mask_slug": mask_slug,
                "support_joint_layout": "freejoint",
            }
            for key, value in params.items():
                if key == "full_queue":
                    continue
                row[key] = f"{float(value):.8g}"
            rows.append(row)
    _write_manifest(rows, out_path)
    return rows


def read_manifest(path: Path = MANIFEST) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _point_local(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            float(row["support_proxy_point_local_x"]),
            float(row["support_proxy_point_local_y"]),
            float(row["support_proxy_point_local_z"]),
        ],
        dtype=np.float64,
    )


def generate_scene(row: dict[str, str]) -> Path:
    task_dir = common.BASE / row["derived_task"]
    src = task_dir / "scene.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{src} missing worldbody")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    weld_name = f"{WELD_NAME_PREFIX}_{row['profile']}"
    d6_assets._remove_named_children(worldbody, "body", {SUPPORT_BODY, "support_weld_anchor"})
    d6_assets._remove_named_children(
        equality,
        "weld",
        {weld_name, "e029_support_weld", "e028_support_weld", "e018b_support_weld", "e018_support_weld", "object_weld"},
    )

    object_idx = -1
    for idx, child in enumerate(list(worldbody)):
        if child.tag == "body" and child.get("name") == "object":
            object_idx = idx
            break
    if object_idx < 0:
        raise ValueError(f"{src} missing top-level object body")

    mass = float(row["support_dynamic_mass"])
    inertia = max(0.01 * mass, 0.005)
    support = ET.Element("body", {"name": SUPPORT_BODY, "pos": "0 0 0"})
    ET.SubElement(support, "freejoint", {"name": "support_dyn_free"})
    ET.SubElement(
        support,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{mass:.8g}",
            "diaginertia": f"{inertia:.8g} {inertia:.8g} {inertia:.8g}",
        },
    )
    ET.SubElement(
        support,
        "geom",
        {
            "name": "support_dynamic_anchor_geom",
            "type": "sphere",
            "size": "0.04",
            "rgba": "0.0 0.45 1.0 0.45",
            "group": "4",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(support, "site", {"name": "trace_support_dynamic_anchor", "size": "0.025"})
    worldbody.insert(object_idx, support)

    point = _point_local(row)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": weld_name,
            "body1": "object",
            "body2": SUPPORT_BODY,
            "relpose": f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} 1 0 0 0",
            "solref": f"{float(row['weld_solref_timeconst']):.8g} 1",
            "solimp": (
                f"{float(row['weld_solimp_1']):.8g} "
                f"{float(row['weld_solimp_2']):.8g} "
                f"{float(row['weld_solimp_width']):.8g}"
            ),
        },
    )
    d6_assets._indent(root)
    out = task_dir / f"{row['scene_name']}.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    text = out.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        out.write_text(text + "\n", encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(out))
    if (model.nq, model.nv, model.nu) != (50, 47, 29):
        raise ValueError(f"{out} changed dims: nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SUPPORT_BODY)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if support_id < 0 or object_id < 0 or equality_id < 0:
        raise ValueError(f"{out} missing support/object/equality")
    if model.body_mocapid[support_id] >= 0:
        raise ValueError(f"{out} support body is mocap")
    if int(model.body_jntnum[support_id]) != 1:
        raise ValueError(f"{out} support joint count={model.body_jntnum[support_id]}")
    support_jadr = int(model.body_jntadr[support_id])
    if int(model.jnt_type[support_jadr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"{out} support joint is not freejoint")
    object_jadr = int(model.body_jntadr[object_id])
    support_qadr = int(model.jnt_qposadr[support_jadr])
    support_dadr = int(model.jnt_dofadr[support_jadr])
    object_qadr = int(model.jnt_qposadr[object_jadr])
    object_dadr = int(model.jnt_dofadr[object_jadr])
    if support_qadr != 36 or support_dadr != 35:
        raise ValueError(f"{out} support addresses q/d={support_qadr}/{support_dadr}")
    if object_qadr != model.nq - 7 or object_dadr != model.nv - 6:
        raise ValueError(f"{out} object not last q/d={object_qadr}/{object_dadr}")
    return out


def _support_qpos_qvel_from_object(qpos: np.ndarray, qvel: np.ndarray, point_local: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    obj_pos = qpos[:, -7:-4].astype(np.float64)
    obj_quat = qpos[:, -4:].astype(np.float64)
    support_pos = obj_pos + np.asarray([common.quat_apply(q, point_local) for q in obj_quat])
    support_qpos = np.concatenate([support_pos, obj_quat], axis=1)
    support_qvel = np.zeros((qvel.shape[0], 6), dtype=np.float64)
    support_qvel[:, :3] = d6_assets._finite_velocity(support_pos, dt)
    support_qvel[:, 3:] = qvel[:, -3:]
    return support_qpos, support_qvel


def generate_data(row: dict[str, str]) -> Path:
    task_dir = common.BASE / row["derived_task"]
    src = task_dir / "0/trajectory_kinematic.npz"
    data = np.load(src, allow_pickle=True)
    qpos = data["qpos"].astype(np.float64)
    qvel = data["qvel"].astype(np.float64)
    if qpos.shape[1] != 43 or qvel.shape[1] != 41:
        raise ValueError(f"{src} expected 43/41 qpos/qvel, got {qpos.shape}/{qvel.shape}")
    support_qpos, support_qvel = _support_qpos_qvel_from_object(
        qpos, qvel, _point_local(row), d6_assets._source_ref_dt(task_dir)
    )
    qpos_aug = np.zeros((qpos.shape[0], 50), dtype=qpos.dtype)
    qvel_aug = np.zeros((qvel.shape[0], 47), dtype=qvel.dtype)
    qpos_aug[:, :36] = qpos[:, :36]
    qpos_aug[:, 36:43] = support_qpos
    qpos_aug[:, 43:] = qpos[:, -7:]
    qvel_aug[:, :35] = qvel[:, :35]
    qvel_aug[:, 35:41] = support_qvel
    qvel_aug[:, 41:] = qvel[:, -6:]
    out = common.REPO / row["data_relpath"]
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: data[key] for key in data.files if key not in {"qpos", "qvel"}}
    np.savez(out, qpos=qpos_aug, qvel=qvel_aug, **payload)
    return out


def prepare_assets(rows: list[dict[str, str]], *, force: bool) -> None:
    prepared: set[str] = set()
    for row in rows:
        if row["derived_task"] not in prepared:
            task_dir = e016_assets.copy_freejoint_case(row["source_task"], row["derived_task"], force=force)
            e016_assets.patch_leg_object_pairs(task_dir / "scene.xml", row["source_task"], row["derived_task"])
            prepared.add(row["derived_task"])
        scene = generate_scene(row)
        data = generate_data(row)
        print(
            f"{row['variant']}: scene={common.rel_or_abs(scene)} data={common.rel_or_abs(data)} "
            f"endpoint=[{row['support_proxy_point_local_x']},{row['support_proxy_point_local_y']},"
            f"{row['support_proxy_point_local_z']}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=common.MANIFEST)
    parser.add_argument("--candidates", type=Path, default=common.CANDIDATES)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-manifest", action="store_true")
    args = parser.parse_args()

    if args.reuse_manifest and args.manifest.is_file():
        rows = read_manifest(args.manifest)
    else:
        rows = build_manifest(
            manifest_path=args.source_manifest,
            candidates_path=args.candidates,
            preflight_path=args.preflight,
            out_path=args.manifest,
        )
        print(f"Wrote {common.rel_or_abs(args.manifest)} rows={len(rows)}")
    prepare_assets(rows, force=args.force)


if __name__ == "__main__":
    main()
