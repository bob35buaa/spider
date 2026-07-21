#!/usr/bin/env python3
"""Build E170 PRG authority and graded preflight manifests."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rotation

from e170_common import (
    CASE_GPU, E168_METRICS, E168_PRODUCTION, E168_REVIEWED, E169_FULL,
    GPU_QUEUES, LOWER_BODY_GEOMS, OVERRIDE_DIR, REPO, RESULTS, REUSE_CASES,
    SCENE_NAME, SCRIPT_DIR, now, read_tsv, rel, repo_path, safe_id, sha256,
    write_json, write_tsv,
)


PAIR_PREFIX = "E170_"
AXIS_FIELDS = {
    "scene_name", "leg_object_penalty_scale", "leg_object_penalty_margin_m",
    "leg_object_penalty_geom_names", "leg_object_penalty_geom_ids",
    "leg_object_penalty_gate_source", "leg_object_penalty_start_eval_time",
    "leg_object_penalty_end_eval_time", "cem_leg_gate_enabled",
    "cem_leg_gate_geom_names", "cem_leg_gate_geom_ids",
    "cem_leg_gate_min_sdf_m", "cem_leg_gate_max_violation_pct",
    "cem_leg_gate_hard_floor_m", "cem_leg_gate_min_valid_frac",
    "cem_leg_gate_fallback",
}


def tree_signature(element: ET.Element, *, ignore_e170_pairs: bool = False) -> Any:
    children = []
    for child in element:
        if ignore_e170_pairs and child.tag == "pair" and child.get("name", "").startswith(PAIR_PREFIX):
            continue
        children.append(tree_signature(child, ignore_e170_pairs=ignore_e170_pairs))
    return element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(), tuple(children)


def convert_reference_to_scene(qpos: np.ndarray, scene_xml: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    if qpos.ndim == 3:
        qpos = qpos[:, 0, :]
    if qpos.shape[1] == model.nq:
        return qpos.astype(np.float64, copy=True)
    nq_robot = model.nq - 6
    if qpos.shape[1] < nq_robot + 7:
        raise ValueError(f"cannot convert qpos {qpos.shape} to nq={model.nq}")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0:
        raise ValueError("scene has no object body")
    convention = "XYZ"
    meta = scene_xml.with_name("scene_act_meta.json")
    if meta.is_file():
        convention = str(json.loads(meta.read_text(encoding="utf-8")).get("euler_convention", "XYZ"))
    body_pos, body_quat = model.body_pos[object_id], model.body_quat[object_id]
    body_rot = Rotation.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]])
    object_pos = qpos[:, nq_robot:nq_robot + 3]
    object_quat = qpos[:, nq_robot + 3:nq_robot + 7]
    object_slide = body_rot.inv().apply(object_pos - body_pos[np.newaxis, :])
    object_xyzw = np.column_stack([object_quat[:, 1], object_quat[:, 2], object_quat[:, 3], object_quat[:, 0]])
    object_euler = (body_rot.inv() * Rotation.from_quat(object_xyzw)).as_euler(convention)
    converted = np.zeros((qpos.shape[0], model.nq), dtype=np.float64)
    converted[:, :nq_robot] = qpos[:, :nq_robot]
    converted[:, nq_robot:nq_robot + 3] = object_slide
    converted[:, nq_robot + 3:nq_robot + 6] = object_euler
    return converted


def min_lowerbody_distance(model: mujoco.MjModel, frames: np.ndarray) -> float:
    data = mujoco.MjData(model)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in LOWER_BODY_GEOMS]
    if object_id < 0 or any(index < 0 for index in geom_ids):
        raise ValueError("lower-body/object geoms missing")
    minimum = float("inf")
    fromto = np.zeros(6, dtype=np.float64)
    for frame in frames:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for geom_id in geom_ids:
            minimum = min(minimum, float(mujoco.mj_geomDistance(model, data, geom_id, object_id, 10.0, fromto)))
    return minimum


def base_payload(source: dict[str, str]) -> dict[str, Any]:
    return {
        "defaults": [source["override_id"], "_self_"],
        "scene_name": SCENE_NAME,
        "leg_object_penalty_scale": 2.0,
        "leg_object_penalty_margin_m": 0.02,
        "leg_object_penalty_geom_names": list(LOWER_BODY_GEOMS),
        "leg_object_penalty_geom_ids": [],
        "leg_object_penalty_gate_source": "always",
        "leg_object_penalty_start_eval_time": 0.0,
        "leg_object_penalty_end_eval_time": 999.0,
        "cem_leg_gate_enabled": True,
        "cem_leg_gate_geom_names": list(LOWER_BODY_GEOMS),
        "cem_leg_gate_geom_ids": [],
        "cem_leg_gate_min_sdf_m": 0.005,
        "cem_leg_gate_max_violation_pct": 0.02,
        "cem_leg_gate_hard_floor_m": -0.005,
        "cem_leg_gate_min_valid_frac": 0.02,
        "cem_leg_gate_fallback": "least_violation",
    }


def write_override(source: dict[str, str]) -> Path:
    override_id = safe_id(f"core4d_E170_{source['case_id']}_PRG")
    output = OVERRIDE_DIR / f"{override_id}.yaml"
    header = "# @package _global_\n# Auto-generated by E170 graded preflight.\n"
    output.write_text(header + yaml.safe_dump(base_payload(source), sort_keys=False), encoding="utf-8")
    return output


def audit_override(source: dict[str, str], output: Path) -> tuple[bool, str]:
    with initialize_config_dir(version_base=None, config_dir=str((REPO / "examples/config").resolve())):
        baseline = OmegaConf.to_container(compose(config_name="default", overrides=[f"+override={source['override_id']}"]), resolve=True)
        candidate = OmegaConf.to_container(compose(config_name="default", overrides=[f"+override={output.stem}"]), resolve=True)
    failures = []
    for key in sorted(set(baseline) | set(candidate)):
        if key not in AXIS_FIELDS and baseline.get(key) != candidate.get(key):
            failures.append(f"non_axis_drift:{key}")
    expected = base_payload(source)
    expected.pop("defaults")
    for key, value in expected.items():
        if candidate.get(key) != value:
            failures.append(f"axis_mismatch:{key}")
    return not failures, ";".join(failures)


def build_scene(source: dict[str, str], *, overwrite: bool) -> dict[str, Any]:
    base_scene = repo_path(source["scene_act"])
    if not base_scene.is_file():
        raise FileNotFoundError(base_scene)
    tree = ET.parse(base_scene)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if pair.get("name", "").startswith(PAIR_PREFIX):
            contact.remove(pair)
    geoms = {geom.get("name") for geom in root.iter("geom") if geom.get("name")}
    missing = sorted(set(LOWER_BODY_GEOMS) - geoms)
    if "object_collision" not in geoms or missing:
        raise ValueError(f"missing_geoms:object={int('object_collision' not in geoms)}:lower={','.join(missing)}")
    for name in LOWER_BODY_GEOMS:
        ET.SubElement(contact, "pair", {"name": f"{PAIR_PREFIX}{name}_object", "geom1": name, "geom2": "object_collision", "solref": "0.008 1", "margin": "0", "gap": "0", "condim": "1"})
    output = base_scene.with_name(f"{SCENE_NAME}.xml")
    if output.exists() and not overwrite:
        if tree_signature(ET.parse(output).getroot()) != tree_signature(root):
            raise FileExistsError(f"different sidecar exists: {output}")
    else:
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)
    physical_root = ET.parse(output).getroot()
    if tree_signature(ET.parse(base_scene).getroot()) != tree_signature(physical_root, ignore_e170_pairs=True):
        raise AssertionError("semantic_diff_exceeds_e170_pairs")
    pairs = [pair for pair in physical_root.findall("./contact/pair") if pair.get("name", "").startswith(PAIR_PREFIX)]
    if len(pairs) != 16 or len({pair.get("name") for pair in pairs}) != 16:
        raise AssertionError(f"pair_count={len(pairs)}")
    model = mujoco.MjModel.from_xml_path(str(output))
    pair_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, index) for index in range(model.npair)}
    if not {f"{PAIR_PREFIX}{name}_object" for name in LOWER_BODY_GEOMS}.issubset(pair_names):
        raise AssertionError("compiled_pair_missing")
    with np.load(repo_path(source["trajectory"]), allow_pickle=True) as data:
        reference = np.asarray(data["qpos"], dtype=np.float64)
    converted = convert_reference_to_scene(reference, output)
    qpos0_min = min_lowerbody_distance(model, model.qpos0[np.newaxis, :])
    reference_min = min_lowerbody_distance(model, converted[:min(5, len(converted))])
    # run_mjwp seeds the simulator from qpos_ref[0] (setup_env), not from the
    # XML model.qpos0.  Keep qpos0 overlap as a diagnostic because it can still
    # reveal a surprising template default, but hard-block the row on the
    # actual runtime initialization frames.  A qpos0-only warning must pass a
    # dedicated runtime smoke before it is admitted to the production queue.
    if reference_min < -0.005:
        raise ValueError(
            f"runtime_initial_overlap:reference_first5={reference_min:.6f}:"
            f"qpos0_diagnostic={qpos0_min:.6f}"
        )
    qpos0_warning = qpos0_min < -0.005
    snapshot_dir = RESULTS / "scene_snapshot" / source["case_id"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_scene, snapshot_dir / base_scene.name)
    shutil.copy2(output, snapshot_dir / output.name)
    return {
        "base_scene": rel(base_scene), "base_scene_sha256": sha256(base_scene),
        "physical_scene": rel(output), "physical_scene_sha256": sha256(output),
        "compiled_pair_count": 16, "semantic_diff_only_e170_pairs": "true",
        "qpos0_min_lowerbody_object_distance_m": qpos0_min,
        "reference_first5_min_lowerbody_object_distance_m": reference_min,
        "runtime_initialization_source": "reference_qpos_frame0",
        "runtime_init_smoke_required": "true" if qpos0_warning else "false",
        "diagnostic_warning": (
            f"unused_xml_qpos0_overlap:{qpos0_min:.6f}"
            if qpos0_warning else ""
        ),
    }


def validate_file(path: str, expected_sha: str = "") -> tuple[bool, str]:
    value = repo_path(path)
    if not value.is_file():
        return False, f"missing:{path}"
    if expected_sha and sha256(value) != expected_sha:
        return False, f"sha_mismatch:{path}"
    return True, ""


def audit_reuse(source: dict[str, str], e169: dict[str, str]) -> dict[str, Any]:
    failures = []
    for key in ("result_npz", "outdir_npz", "config_act", "video", "scene_act", "trajectory", "contact_mask"):
        ok, detail = validate_file(e169[key])
        if not ok:
            failures.append(detail)
    if not failures:
        with np.load(repo_path(e169["result_npz"]), allow_pickle=True) as root, np.load(repo_path(e169["outdir_npz"]), allow_pickle=True) as out:
            if "qpos" not in root or "qpos" not in out or not np.array_equal(root["qpos"], out["qpos"]):
                failures.append("root_outdir_qpos_mismatch")
            if not np.isfinite(np.asarray(out["qpos"], dtype=np.float64)).all():
                failures.append("nonfinite_qpos")
            required = {"cem_leg_gate_valid_frac", "cem_leg_gate_selected_valid_frac", "cem_leg_gate_fallback_used", "sample_leg_gate_min_sdf_min", "sample_leg_gate_violation_pct_mean", "leg_object_penalty_mean"}
            failures.extend(f"missing_diag:{key}" for key in sorted(required - set(out.files)))
        config = yaml.safe_load(repo_path(e169["config_act"]).read_text(encoding="utf-8"))
        if config.get("scene_name") != "scene_act_E169_lowerbody_physics" or config.get("leg_object_penalty_scale") != 2.0 or config.get("cem_leg_gate_enabled") is not True:
            failures.append("e169_config_prg_mismatch")
        if sha256(e169["scene_act"]) != e169["effective_scene_sha256"]:
            failures.append("e169_scene_sha_drift")
    return {
        "status": "REUSE_AUDITED" if not failures else "preflight_blocked_local",
        "blocker_type": "" if not failures else "reuse_artifact_drift",
        "blocker_detail": ";".join(failures),
    }


def artifact_paths(case_id: str, stage: str) -> dict[str, str]:
    if stage not in {"canary", "recovery_smoke", "full"}:
        raise ValueError(f"unsupported artifact stage: {stage}")
    is_smoke = stage in {"canary", "recovery_smoke"}
    suffix = "smoke" if is_smoke else "full"
    variant_suffix = "_canary" if stage == "canary" else ("_recovery_smoke" if stage == "recovery_smoke" else "")
    variant = f"E170_{case_id}_PRG{variant_suffix}"
    root = f"workspace/core4d/results/E170/s6_downstream/cem/{stage}"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir_{suffix}/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir_{suffix}/config_act.yaml",
        "video": f"workspace/core4d/results/E170/s6_downstream/render/{stage}/{variant}_{suffix}.mp4",
        "log": f"logs/E170/cem/{stage}/{variant}.log",
    }


def audit_runtime_smoke(row: dict[str, Any]) -> tuple[bool, str]:
    """Validate a pulled case-specific smoke before production promotion."""
    paths = {key: repo_path(row[key]) for key in ("result_npz", "outdir_npz", "config_act")}
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        return False, "missing:" + ",".join(missing)
    failures: list[str] = []
    try:
        with np.load(paths["result_npz"], allow_pickle=True) as root, np.load(paths["outdir_npz"], allow_pickle=True) as out:
            if "qpos" not in root or "qpos" not in out or not np.array_equal(root["qpos"], out["qpos"]):
                failures.append("root_outdir_qpos_mismatch")
            elif not np.isfinite(np.asarray(out["qpos"], dtype=np.float64)).all():
                failures.append("nonfinite_qpos")
            required = {
                "cem_leg_gate_valid_frac", "cem_leg_gate_selected_valid_frac",
                "cem_leg_gate_fallback_used", "sample_leg_gate_min_sdf_min",
                "sample_leg_gate_violation_pct_mean", "leg_object_penalty_mean",
            }
            failures.extend(f"missing_diag:{key}" for key in sorted(required - set(out.files)))
        config = yaml.safe_load(paths["config_act"].read_text(encoding="utf-8"))
        if config.get("scene_name") != SCENE_NAME:
            failures.append("config_mismatch:scene_name")
        if config.get("leg_object_penalty_scale") != 2.0:
            failures.append("config_mismatch:leg_object_penalty_scale")
        if config.get("cem_leg_gate_enabled") is not True:
            failures.append("config_mismatch:cem_leg_gate_enabled")
        scene = repo_path(row["scene_act"])
        if not scene.is_file() or sha256(scene) != row["effective_scene_sha256"]:
            failures.append("effective_scene_sha")
    except Exception as exc:
        failures.append(f"validation:{type(exc).__name__}:{exc}")
    return not failures, ";".join(failures)


def build(*, overwrite_scenes: bool) -> tuple[dict[str, Any], int]:
    global_failures = []
    for path in (E168_REVIEWED, E168_PRODUCTION, E168_METRICS, E169_FULL):
        if not path.is_file():
            global_failures.append(f"missing_authority:{rel(path)}")
    if global_failures:
        summary = {"created_at": now(), "status": "global_contract_blocked", "global_failures": global_failures}
        write_json(RESULTS / "preflight/preflight_summary.json", summary)
        return summary, 2

    reviewed_rows = read_tsv(E168_REVIEWED)
    production_rows = read_tsv(E168_PRODUCTION)
    metrics_rows = read_tsv(E168_METRICS)
    e169_rows = read_tsv(E169_FULL)
    reviewed = {row["case_id"]: row for row in reviewed_rows}
    production_all = {row["case_id"]: row for row in production_rows}
    production = {case_id: production_all[case_id] for case_id in reviewed if case_id in production_all}
    metrics = {row["case_id"]: row for row in metrics_rows}
    e169_prg = {row["case_id"]: row for row in e169_rows if row.get("cell_id") == "PRG"}
    cases = sorted(reviewed)
    if len(reviewed_rows) != 28 or len(reviewed) != 28:
        global_failures.append(f"authority_cardinality:rows={len(reviewed_rows)}:unique={len(reviewed)}")
    if set(cases) != set(production) or set(cases) != set(metrics):
        global_failures.append("authority_join_case_set_mismatch")
    if not REUSE_CASES.issubset(cases) or len(set(cases) - REUSE_CASES) != 24:
        global_failures.append("reuse_new_split_mismatch")
    variants = [reviewed[case].get("retarget_variant_id", "") for case in cases]
    if variants.count("omnirt_v1") != 25 or variants.count("omnirt_v2") != 3:
        global_failures.append(f"retarget_distribution:v1={variants.count('omnirt_v1')}:v2={variants.count('omnirt_v2')}")
    if set(CASE_GPU) != set(cases) - REUSE_CASES:
        global_failures.append("fixed_gpu_queue_case_set_mismatch")
    if set(e169_prg) != REUSE_CASES:
        global_failures.append("e169_prg_reuse_set_mismatch")
    if global_failures:
        summary = {"created_at": now(), "status": "global_contract_blocked", "global_failures": global_failures}
        write_json(RESULTS / "preflight/preflight_summary.json", summary)
        return summary, 2

    rows = []
    scene_audit = []
    config_audit = []
    blocker_rows = []
    for ordinal, case_id in enumerate(cases, 1):
        source = production[case_id]
        reviewed_row = reviewed[case_id]
        base = {
            "ordinal": ordinal, "case_id": case_id, "sequence_key": source["sequence_key"],
            "source_person": source["source_person"], "object_key": source["object_key"],
            "source_action": metrics[case_id].get("source_action", ""),
            "source_obstacle_level": metrics[case_id].get("source_obstacle_level", ""),
            "e168_manual_use_decision": metrics[case_id]["manual_use_decision"],
            "execution_source": "E169" if case_id in REUSE_CASES else "E170",
            "reused_full": "true" if case_id in REUSE_CASES else "false",
            "cell_id": "PRG", "p_enabled": "true", "r_enabled": "true", "g_enabled": "true",
            "target_task": source["target_task"], "target_scene": source["target_scene"],
            "trajectory": source["trajectory"], "contact_mask": source["contact_mask"],
            "e168_baseline_result_npz": source["result_npz"],
            "e168_baseline_outdir_npz": source["outdir_npz"],
            "e168_baseline_config": source["config_act"],
            "e168_baseline_video": source["video"],
            "e168_baseline_result_sha256": sha256(source["result_npz"]) if repo_path(source["result_npz"]).is_file() else "",
            "e168_baseline_video_sha256": sha256(source["video"]) if repo_path(source["video"]).is_file() else "",
            "contact_mask_label": source["contact_mask_label"],
            "retarget_variant_id": source["retarget_variant_id"],
            "target_variant_id": source["target_variant_id"],
            "hand_collision_variant_id": source["hand_collision_variant_id"],
            "spider_method_id": source["spider_method_id"],
            "assigned_gpu": CASE_GPU.get(case_id, ""), "gpu_id": "",
            "failure_mode": "",
            "blocker_type": "", "blocker_detail": "", "evidence_path": "",
            "preflight_warning": "", "runtime_init_smoke_required": "false",
            "runtime_init_smoke_status": "not_required",
            "first_seen_at": "", "recovery_status": "not_applicable", "updated_at": now(),
        }
        if case_id in REUSE_CASES:
            old = e169_prg[case_id]
            audit = audit_reuse(source, old)
            base.update({
                "variant": f"E170_{case_id}_PRG_reuse_E169", "execution_mode": "reused",
                "execution_decision": "frozen_e169_prg_sha_reference", "status": audit["status"],
                "blocker_type": audit["blocker_type"], "blocker_detail": audit["blocker_detail"],
                "override_id": old["override_id"], "override_path": old["override_path"],
                "override_sha256": sha256(old["override_path"]), "config_audit_status": "pass" if audit["status"] == "REUSE_AUDITED" else "fail",
                "scene_act": old["scene_act"], "scene_name": old["scene_name"],
                "base_scene_sha256": old["base_scene_sha256"], "effective_scene_sha256": old["effective_scene_sha256"],
                "trajectory_sha256": sha256(source["trajectory"]), "contact_mask_sha256": sha256(source["contact_mask"]),
                "source_variant": old["variant"], "source_result_npz": old["result_npz"],
                "source_outdir_npz": old["outdir_npz"], "source_config": old["config_act"], "source_video": old["video"],
                "source_result_sha256": sha256(old["result_npz"]) if repo_path(old["result_npz"]).is_file() else "",
                "source_outdir_sha256": sha256(old["outdir_npz"]) if repo_path(old["outdir_npz"]).is_file() else "",
                "source_config_sha256": sha256(old["config_act"]) if repo_path(old["config_act"]).is_file() else "",
                "source_video_sha256": sha256(old["video"]) if repo_path(old["video"]).is_file() else "",
                "result_npz": old["result_npz"], "outdir_npz": old["outdir_npz"], "config_act": old["config_act"], "video": old["video"], "log": old["log"],
            })
        else:
            failures = []
            for key in ("target_scene", "trajectory", "scene_act", "contact_mask", "result_npz", "outdir_npz", "config_act"):
                ok, detail = validate_file(source[key])
                if not ok:
                    failures.append(detail)
            scene = {}
            override = OVERRIDE_DIR / f"core4d_E170_{case_id}_PRG.yaml"
            if not failures:
                try:
                    scene = build_scene(source, overwrite=overwrite_scenes)
                except Exception as exc:
                    failures.append(f"scene_preflight:{type(exc).__name__}:{exc}")
                try:
                    override = write_override(source)
                    passed, detail = audit_override(source, override)
                    if not passed:
                        failures.append(f"override_parity:{detail}")
                except Exception as exc:
                    failures.append(f"override_parity:{type(exc).__name__}:{exc}")
            paths = artifact_paths(case_id, "full")
            smoke_required = scene.get("runtime_init_smoke_required") == "true"
            smoke_status = "not_required"
            smoke_detail = ""
            if not failures and smoke_required:
                smoke_probe = {**base, **paths, **artifact_paths(case_id, "recovery_smoke")}
                smoke_probe.update({
                    "scene_act": scene["physical_scene"],
                    "effective_scene_sha256": scene["physical_scene_sha256"],
                })
                smoke_passed, smoke_detail = audit_runtime_smoke(smoke_probe)
                smoke_status = "pass" if smoke_passed else f"pending:{smoke_detail}"
            if failures:
                status = "preflight_blocked_local"
            elif smoke_required and smoke_status != "pass":
                status = "READY_FOR_RECOVERY_SMOKE"
            else:
                status = "READY_FOR_FULL"
            base.update(paths)
            base.update({
                "execution_mode": "production", "execution_decision": "a100_user_allowlist_0_1_2_3",
                "status": status, "blocker_type": "" if not failures else "case_preflight",
                "blocker_detail": ";".join(failures), "first_seen_at": "" if not failures else now(),
                "preflight_warning": scene.get("diagnostic_warning", ""),
                "runtime_init_smoke_required": "true" if smoke_required else "false",
                "runtime_init_smoke_status": smoke_status,
                "recovery_status": (
                    "pending_runtime_smoke" if status == "READY_FOR_RECOVERY_SMOKE"
                    else ("runtime_smoke_passed_ready_for_full" if smoke_required and smoke_status == "pass"
                          else ("not_needed" if not failures else "pending_repair"))
                ),
                "override_id": override.stem, "override_path": rel(override),
                "override_sha256": sha256(override) if override.is_file() else "",
                "config_audit_status": "pass" if not failures else "fail",
                "scene_act": scene.get("physical_scene", ""), "scene_name": SCENE_NAME,
                "base_scene_sha256": scene.get("base_scene_sha256", ""),
                "effective_scene_sha256": scene.get("physical_scene_sha256", ""),
                "trajectory_sha256": sha256(source["trajectory"]) if repo_path(source["trajectory"]).is_file() else "",
                "contact_mask_sha256": sha256(source["contact_mask"]) if repo_path(source["contact_mask"]).is_file() else "",
                "source_variant": source["variant"], "source_result_npz": source["result_npz"],
                "source_outdir_npz": source["outdir_npz"], "source_config": source["config_act"], "source_video": source["video"],
                "source_result_sha256": sha256(source["result_npz"]) if repo_path(source["result_npz"]).is_file() else "",
                "source_outdir_sha256": sha256(source["outdir_npz"]) if repo_path(source["outdir_npz"]).is_file() else "",
                "source_config_sha256": sha256(source["config_act"]) if repo_path(source["config_act"]).is_file() else "",
                "source_video_sha256": sha256(source["video"]) if repo_path(source["video"]).is_file() else "",
            })
            scene_audit.append({
                "case_id": case_id, "status": status, **scene,
                "runtime_init_smoke_status": smoke_status,
                "failures": ";".join(failures),
            })
            config_audit.append({"case_id": case_id, "override_path": rel(override), "status": "pass" if not failures else "fail", "failures": ";".join(failures)})
        if base["status"] == "preflight_blocked_local":
            base["evidence_path"] = "workspace/core4d/results/E170/preflight/local_blockers.tsv"
            blocker_rows.append({key: base[key] for key in ("case_id", "execution_source", "retarget_variant_id", "assigned_gpu", "blocker_type", "blocker_detail", "evidence_path", "first_seen_at", "recovery_status")})
        rows.append(base)

    ready_by_case = {
        row["case_id"]: copy.deepcopy(row)
        for row in rows
        if row["execution_source"] == "E170" and row["status"] == "READY_FOR_FULL"
    }
    # Preserve the plan's fixed within-GPU queue order.  The cases are
    # independent, but deterministic ordering makes queue evidence and timing
    # directly comparable and avoids an incidental authority-sort schedule.
    ready = [
        ready_by_case[case_id]
        for gpu in ("0", "1", "2", "3")
        for case_id in GPU_QUEUES[gpu]
        if case_id in ready_by_case
    ]
    canary = []
    for retarget in ("omnirt_v1", "omnirt_v2"):
        candidates = [row for row in ready if row["retarget_variant_id"] == retarget]
        if candidates:
            row = copy.deepcopy(candidates[0])
            row.update(artifact_paths(row["case_id"], "canary"))
            row["execution_mode"] = "canary"
            row["status"] = "READY_FOR_CANARY"
            canary.append(row)

    recovery_smoke = []
    recovery_full = []
    recovery_rows = []
    for source_row in rows:
        if source_row["execution_source"] != "E170" or source_row["runtime_init_smoke_required"] != "true":
            continue
        recovery_rows.append(copy.deepcopy(source_row))
        if source_row["status"] == "READY_FOR_RECOVERY_SMOKE":
            row = copy.deepcopy(source_row)
            row.update(artifact_paths(row["case_id"], "recovery_smoke"))
            row["execution_mode"] = "canary"
            row["status"] = "READY_FOR_CANARY"
            recovery_smoke.append(row)
        elif source_row["status"] == "READY_FOR_FULL" and source_row["runtime_init_smoke_status"] == "pass":
            row = copy.deepcopy(source_row)
            row.update(artifact_paths(row["case_id"], "full"))
            row["execution_mode"] = "production"
            row["status"] = "READY_FOR_FULL"
            recovery_full.append(row)

    write_tsv(SCRIPT_DIR / "variants.tsv", rows)
    write_tsv(RESULTS / "s6_downstream/manifests/analysis_manifest.tsv", rows)
    write_tsv(RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv", ready)
    write_tsv(RESULTS / "s6_downstream/manifests/cem_canary_manifest.tsv", canary)
    fields = list(rows[0]) if rows else []
    write_tsv(RESULTS / "s6_downstream/manifests/cem_recovery_smoke_manifest.tsv", recovery_smoke, fields)
    write_tsv(RESULTS / "s6_downstream/manifests/cem_recovery_full_manifest.tsv", recovery_full, fields)
    write_tsv(RESULTS / "s6_downstream/manifests/recovery_manifest.tsv", recovery_rows + blocker_rows)
    write_tsv(RESULTS / "preflight/local_blockers.tsv", blocker_rows)
    write_tsv(RESULTS / "preflight/scene_audit.tsv", scene_audit)
    write_tsv(RESULTS / "preflight/config_audit.tsv", config_audit)

    authority_snapshot = [{"case_id": row["case_id"], "retarget_variant_id": row["retarget_variant_id"], "target_variant_id": row["target_variant_id"], "hand_collision_variant_id": row["hand_collision_variant_id"], "e168_manual_use_decision": row["e168_manual_use_decision"], "execution_source": row["execution_source"], "source_result_npz": row["source_result_npz"], "source_result_sha256": row["source_result_sha256"]} for row in rows]
    for relative in (
        "s1_raw_contact/imported_e168_snapshot/authority.tsv",
        "s2_templates/imported_e168_snapshot/authority.tsv",
        "s3_retarget/imported_e168_snapshot/authority.tsv",
        "s5_handoff/imported_e168_snapshot/authority.tsv",
    ):
        write_tsv(RESULTS / relative, authority_snapshot)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()
    environment = {
        "created_at": now(), "git_commit_at_preflight": git_commit,
        "remote": "batchcom@61.172.170.106", "remote_root": "/home/dataset-assist-0/xiayb/workspace/spider",
        "allowed_gpus": [0, 1, 2, 3], "gpu_policy": "user_explicit_overlap_allowed_no_kill",
        "authority_path": rel(E168_REVIEWED), "authority_sha256": sha256(E168_REVIEWED),
    }
    write_json(RESULTS / "s0_environment/environment_manifest.json", environment)
    summary = {
        "created_at": now(), "status": "pass" if not blocker_rows and not recovery_smoke else "partial_ready",
        "global_failures": [], "analysis_rows": len(rows), "reuse_rows": sum(row["execution_source"] == "E169" for row in rows),
        "reuse_audited": sum(row["status"] == "REUSE_AUDITED" for row in rows),
        "new_rows": sum(row["execution_source"] == "E170" for row in rows), "ready_for_full": len(ready),
        "local_blocked": len(blocker_rows), "runtime_smoke_pending": len(recovery_smoke),
        "recovery_full_ready": len(recovery_full), "canary_rows": len(canary),
        "retarget_distribution": {"omnirt_v1": variants.count("omnirt_v1"), "omnirt_v2": variants.count("omnirt_v2")},
        "fixed_gpu_queues": GPU_QUEUES,
    }
    write_json(RESULTS / "preflight/preflight_summary.json", summary)
    write_json(RESULTS / "execution_manifest.json", {"environment": environment, "summary": summary, "rows": rows})
    return summary, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite-scenes", action="store_true")
    args = parser.parse_args()
    summary, status = build(overwrite_scenes=args.overwrite_scenes)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
