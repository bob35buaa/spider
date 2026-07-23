#!/usr/bin/env python3
"""E174: build the rubber-hull + E170-PRG CEM scenes, overrides and manifests.

Input authority is the E174 S5 handoff manifest (the frozen S5_READY_SET) plus
the regenerated dcv3 CEM overrides (chained to the plan-frozen E167A base).

For each S5-ready case this:
  1. builds a FRESH rubber_hull hand sidecar from the target scene_act.xml
     (sphere hands), then adds the 16 E169/E170 lower-body/object collision
     pairs -> scene_act_E174_rubberHull_PRG.xml (never overwrites scene_act.xml);
  2. writes a per-case PRG Hydra override (defaults: [dcv3_override, _self_]) with
     the E170 PRG frozen candidate-gate + soft-penalty payload;
  3. emits canary (opt_steps=4/samples=64) and full (1024/32) manifests with
     local 8-GPU assignment, selected_retarget_variant_id / rescue_of provenance.

Only rows whose object is in the E174 non-box scope (5 bucket + 2 desk) are in scope. Full expected = count(S5_READY).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e174_common as C  # noqa: E402

# shared rubber_hull patch (step 1)
sys.path.insert(0, str(C.DCV3 / "stages/s5_handoff"))
sys.path.insert(0, str(C.DCV3 / "lib"))
sys.path.insert(0, str(C.DCV3 / "state"))
from patch_hand_collision import patch_scene  # noqa: E402

PAIR_PREFIX = "E174_"
NUM_GPUS = int(__import__("os").environ.get("E174_NUM_GPUS", "8"))
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


def tree_signature(element: ET.Element, *, ignore_pairs: bool = False) -> Any:
    children = []
    for child in element:
        if ignore_pairs and child.tag == "pair" and child.get("name", "").startswith(PAIR_PREFIX):
            continue
        children.append(tree_signature(child, ignore_pairs=ignore_pairs))
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
    geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in C.LOWER_BODY_GEOMS]
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


def base_payload(dcv3_override_id: str) -> dict[str, Any]:
    return {
        "defaults": [dcv3_override_id, "_self_"],
        "scene_name": C.SCENE_NAME,
        "leg_object_penalty_scale": C.LEG_OBJECT_PENALTY_SCALE,
        "leg_object_penalty_margin_m": C.LEG_OBJECT_PENALTY_MARGIN_M,
        "leg_object_penalty_geom_names": list(C.LOWER_BODY_GEOMS),
        "leg_object_penalty_geom_ids": [],
        "leg_object_penalty_gate_source": "always",
        "leg_object_penalty_start_eval_time": 0.0,
        "leg_object_penalty_end_eval_time": 999.0,
        "cem_leg_gate_enabled": True,
        "cem_leg_gate_geom_names": list(C.LOWER_BODY_GEOMS),
        "cem_leg_gate_geom_ids": [],
        "cem_leg_gate_min_sdf_m": C.CEM_LEG_GATE_MIN_SDF_M,
        "cem_leg_gate_max_violation_pct": C.CEM_LEG_GATE_MAX_VIOLATION_PCT,
        "cem_leg_gate_hard_floor_m": C.CEM_LEG_GATE_HARD_FLOOR_M,
        "cem_leg_gate_min_valid_frac": C.CEM_LEG_GATE_MIN_VALID_FRAC,
        "cem_leg_gate_fallback": C.CEM_LEG_GATE_FALLBACK,
    }


def write_override(case_id: str, dcv3_override_id: str) -> Path:
    override_id = C.safe_id(f"core4d_E174_{case_id}_PRG")
    output = C.OVERRIDE_DIR / f"{override_id}.yaml"
    header = "# @package _global_\n# Auto-generated by E174 build_prg_cem_manifest.py.\n"
    output.write_text(header + yaml.safe_dump(base_payload(dcv3_override_id), sort_keys=False), encoding="utf-8")
    return output


def audit_override(dcv3_override_id: str, output: Path) -> tuple[bool, str]:
    with initialize_config_dir(version_base=None, config_dir=str((C.REPO / "examples/config").resolve())):
        baseline = OmegaConf.to_container(compose(config_name="default", overrides=[f"+override={dcv3_override_id}"]), resolve=True)
        candidate = OmegaConf.to_container(compose(config_name="default", overrides=[f"+override={output.stem}"]), resolve=True)
    failures = []
    for key in sorted(set(baseline) | set(candidate)):
        if key not in AXIS_FIELDS and baseline.get(key) != candidate.get(key):
            failures.append(f"non_axis_drift:{key}")
    expected = base_payload(dcv3_override_id)
    expected.pop("defaults")
    for key, value in expected.items():
        if candidate.get(key) != value:
            failures.append(f"axis_mismatch:{key}")
    return not failures, ";".join(failures)


def build_scene(case_id: str, base_scene_act: Path, trajectory: Path, *, overwrite: bool) -> dict[str, Any]:
    """sphere scene_act -> rubber_hull patch -> +16 lower-body pairs -> E174 sidecar."""
    if not base_scene_act.is_file():
        raise FileNotFoundError(base_scene_act)
    task_dir = base_scene_act.parent
    # step 1: rubber_hull hand sidecar (intermediate, in the task dir, own name)
    rubber_name = "scene_act_E174_rubberHull"
    patch_scene(
        base_scene_act=base_scene_act, out_dir=task_dir, case_id=case_id,
        hand_collision_variant_id="rubber_hull", scene_name=rubber_name,
        install_dir=None, repo=C.REPO,
    )
    rubber_scene = task_dir / f"{rubber_name}.xml"
    # step 2: add 16 lower-body/object pairs on top of the rubber sidecar
    tree = ET.parse(rubber_scene)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if pair.get("name", "").startswith(PAIR_PREFIX):
            contact.remove(pair)
    geoms = {geom.get("name") for geom in root.iter("geom") if geom.get("name")}
    missing = sorted(set(C.LOWER_BODY_GEOMS) - geoms)
    if "object_collision" not in geoms or missing:
        raise ValueError(f"missing_geoms:object={int('object_collision' not in geoms)}:lower={','.join(missing)}")
    for name in C.LOWER_BODY_GEOMS:
        ET.SubElement(contact, "pair", {
            "name": f"{PAIR_PREFIX}{name}_object", "geom1": name, "geom2": "object_collision",
            "solref": "0.008 1", "margin": "0", "gap": "0", "condim": "1",
        })
    output = task_dir / f"{C.SCENE_NAME}.xml"
    if output.exists() and not overwrite:
        if tree_signature(ET.parse(output).getroot()) != tree_signature(root):
            raise FileExistsError(f"different E174 sidecar exists: {output}")
    else:
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)
    # validation: semantic diff vs rubber intermediate == exactly the 16 pairs
    physical_root = ET.parse(output).getroot()
    if tree_signature(ET.parse(rubber_scene).getroot()) != tree_signature(physical_root, ignore_pairs=True):
        raise AssertionError("semantic_diff_exceeds_e171_pairs")
    pairs = [p for p in physical_root.findall("./contact/pair") if p.get("name", "").startswith(PAIR_PREFIX)]
    if len(pairs) != 16 or len({p.get("name") for p in pairs}) != 16:
        raise AssertionError(f"pair_count={len(pairs)}")
    model = mujoco.MjModel.from_xml_path(str(output))
    pair_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, i) for i in range(model.npair)}
    if not {f"{PAIR_PREFIX}{name}_object" for name in C.LOWER_BODY_GEOMS}.issubset(pair_names):
        raise AssertionError("compiled_pair_missing")
    for side in ("lh", "rh"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side)
        if gid < 0 or mujoco.mjtGeom(int(model.geom_type[gid])).name != "mjGEOM_MESH":
            raise AssertionError(f"{side}_not_mesh_after_rubber_hull")
    # runtime initial overlap on the reference first frames (run_mjwp seeds from qpos_ref[0])
    with np.load(C.repo_path(trajectory), allow_pickle=True) as data:
        reference = np.asarray(data["qpos"], dtype=np.float64)
    converted = convert_reference_to_scene(reference, output)
    reference_min = min_lowerbody_distance(model, converted[:min(5, len(converted))])
    if reference_min < C.CEM_LEG_GATE_HARD_FLOOR_M:
        raise ValueError(f"runtime_initial_overlap:reference_first5={reference_min:.6f}")
    snapshot_dir = C.RESULTS / "scene_snapshot/cem_sidecars" / case_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_scene_act, snapshot_dir / base_scene_act.name)
    shutil.copy2(rubber_scene, snapshot_dir / rubber_scene.name)
    shutil.copy2(output, snapshot_dir / output.name)
    return {
        "base_scene": C.rel(base_scene_act), "base_scene_sha256": C.sha256(base_scene_act),
        "rubber_scene": C.rel(rubber_scene), "rubber_scene_sha256": C.sha256(rubber_scene),
        "physical_scene": C.rel(output), "physical_scene_sha256": C.sha256(output),
        "compiled_pair_count": 16,
        "reference_first5_min_lowerbody_object_distance_m": reference_min,
    }


def artifact_paths(case_id: str, stage: str) -> dict[str, str]:
    is_smoke = stage == "canary"
    suffix = "smoke" if is_smoke else "full"
    variant = f"E174_{case_id}_PRG{'_canary' if is_smoke else ''}"
    root = f"workspace/core4d/results/E174/s6_downstream/cem/{stage}"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir_{suffix}/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir_{suffix}/config_act.yaml",
        "video": f"workspace/core4d/results/E174/s6_downstream/render/{stage}/{variant}_{suffix}.mp4",
        "log": f"logs/E174/cem/{stage}/{variant}.log",
        "cem_samples": C.CEM_CANARY_SAMPLES if is_smoke else C.CEM_FULL_SAMPLES,
        "cem_opt_steps": C.CEM_CANARY_OPT_STEPS if is_smoke else C.CEM_FULL_OPT_STEPS,
        "cem_seed": C.CEM_SEED,
    }


def load_rescue_of() -> dict[str, str]:
    """map v2-rescued case_id -> v1 infeasible manifest ref (rescue provenance)."""
    v1 = C.RESULTS / "s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv"
    out = {}
    if v1.is_file():
        for row in C.read_tsv(v1):
            if row.get("stage2b_status") == C.V1_INFEASIBLE_STATUS:
                out[row["case_id"]] = f"omnirt_v1:{row['case_id']}:omniretarget_infeasible"
    return out


def build(*, overwrite_scenes: bool, include_review: bool) -> tuple[dict[str, Any], int]:
    handoff = C.RESULTS / "s5_handoff/handoff_manifest.tsv"
    if not handoff.is_file():
        return {"status": "global_blocked", "reason": f"missing handoff: {C.rel(handoff)}"}, 2
    rows_in = C.read_tsv(handoff)
    ready_decisions = {"HANDOFF_READY"} if not include_review else {"HANDOFF_READY", "HANDOFF_REVIEW_VISUAL_QC"}
    s5_ready = [
        r for r in rows_in
        if r.get("object_key") in C.OBJECT_KEYS and r.get("handoff_decision") in ready_decisions
    ]
    s5_ready.sort(key=lambda r: (r["object_key"], r["person"], r["date"], r["seq"]))
    rescue_of = load_rescue_of()

    rows: list[dict[str, Any]] = []
    scene_audit: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for ordinal, src in enumerate(s5_ready, 1):
        case_id = src["case_id"]
        retarget = src["retarget_variant_id"]
        dcv3_override_id = C.safe_id(f"core4d_dcv3_{retarget}_{src['target_variant_id']}_{case_id}")
        base = {
            "ordinal": ordinal, "case_id": case_id, "object_key": src["object_key"],
            "date": src["date"], "seq": src["seq"], "person": src["person"],
            "retarget_variant_id": retarget,
            "selected_retarget_variant_id": retarget,
            "rescue_of": rescue_of.get(case_id, "") if retarget == C.RETARGET_RESCUE_VARIANT else "",
            "target_variant_id": src["target_variant_id"],
            "hand_collision_variant_id": C.HAND_COLLISION_VARIANT,
            "spider_method_id": C.E174_METHOD_ID,
            "source_config_id": C.SOURCE_CONFIG_ID,
            "cell_id": "PRG", "p_enabled": "true", "r_enabled": "true", "g_enabled": "true",
            "contact_mask_label": src.get("contact_mask_label", "3cm"),
            "target_task": src.get("stage2b_target_task", ""),
            "target_scene": src.get("target_scene", ""),
            "trajectory": src["trajectory"], "contact_mask": src["contact_mask"],
            "assigned_gpu": str((ordinal - 1) % NUM_GPUS), "gpu_id": "",
            "override_id": "", "override_path": "", "override_sha256": "",
            "scene_act": "", "scene_name": C.SCENE_NAME,
            "base_scene_sha256": "", "effective_scene_sha256": "",
            "trajectory_sha256": C.sha256(src["trajectory"]) if C.repo_path(src["trajectory"]).is_file() else "",
            "contact_mask_sha256": C.sha256(src["contact_mask"]) if C.repo_path(src["contact_mask"]).is_file() else "",
            "status": "", "failure_mode": "", "blocker_detail": "", "updated_at": C.now(),
        }
        failures: list[str] = []
        base_scene_act = C.repo_path(src["scene_act"])
        scene: dict[str, Any] = {}
        override = C.OVERRIDE_DIR / f"{C.safe_id(f'core4d_E174_{case_id}_PRG')}.yaml"
        dcv3_path = C.OVERRIDE_DIR / f"{dcv3_override_id}.yaml"
        if not dcv3_path.is_file():
            failures.append(f"missing_dcv3_override:{dcv3_override_id}")
        for key in ("scene_act", "trajectory", "contact_mask"):
            if not C.repo_path(src[key]).is_file():
                failures.append(f"missing:{key}")
        if not failures:
            try:
                scene = build_scene(case_id, base_scene_act, Path(src["trajectory"]), overwrite=overwrite_scenes)
            except Exception as exc:
                failures.append(f"scene_preflight:{type(exc).__name__}:{exc}")
            try:
                override = write_override(case_id, dcv3_override_id)
                ok, detail = audit_override(dcv3_override_id, override)
                if not ok:
                    failures.append(f"override_parity:{detail}")
            except Exception as exc:
                failures.append(f"override_parity:{type(exc).__name__}:{exc}")
        base.update(artifact_paths(case_id, "full"))
        base.update({
            "execution_mode": "production",
            "status": "READY_FOR_FULL" if not failures else "preflight_blocked",
            "blocker_detail": ";".join(failures),
            "override_id": override.stem, "override_path": C.rel(override),
            "override_sha256": C.sha256(override) if override.is_file() else "",
            "scene_act": scene.get("physical_scene", ""),
            "base_scene_sha256": scene.get("base_scene_sha256", ""),
            "effective_scene_sha256": scene.get("physical_scene_sha256", ""),
        })
        scene_audit.append({"case_id": case_id, "status": base["status"], **scene, "failures": ";".join(failures)})
        if failures:
            blockers.append({"case_id": case_id, "blocker_detail": ";".join(failures)})
        rows.append(base)

    ready = [copy.deepcopy(r) for r in rows if r["status"] == "READY_FOR_FULL"]

    # canary (plan 5.7): >=1 per object_key, >=1 per object CATEGORY (bucket +
    # desk both use different collision proxies -> both runtime paths must be
    # smoke-tested), and >=1 v2 rescue if any v2 is ready. Deterministic order.
    canary: list[dict[str, Any]] = []
    seen_objects: set[str] = set()
    seen_cats: set[str] = set()
    have_v2 = False

    def add_canary(r: dict[str, Any]) -> None:
        row = copy.deepcopy(r)
        row.update(artifact_paths(r["case_id"], "canary"))
        row["execution_mode"] = "canary"
        row["status"] = "READY_FOR_CANARY"
        canary.append(row)
        seen_objects.add(r["object_key"])
        seen_cats.add(C.OBJECT_CATEGORY.get(r["object_key"], ""))

    for r in ready:
        obj = r["object_key"]
        cat = C.OBJECT_CATEGORY.get(obj, "")
        need_object = obj not in seen_objects
        need_cat = cat not in seen_cats
        need_v2 = (r["selected_retarget_variant_id"] == C.RETARGET_RESCUE_VARIANT and not have_v2)
        if need_object or need_cat or need_v2:
            add_canary(r)
            if r["selected_retarget_variant_id"] == C.RETARGET_RESCUE_VARIANT:
                have_v2 = True

    fields = list(rows[0]) if rows else []
    C.write_tsv(C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv", ready, fields)
    C.write_tsv(C.RESULTS / "s6_downstream/manifests/cem_canary_manifest.tsv", canary, fields)
    C.write_tsv(C.RESULTS / "s6_downstream/manifests/cem_analysis_manifest.tsv", rows, fields)
    C.write_tsv(C.RESULTS / "s6_downstream/preflight/scene_audit.tsv", scene_audit)
    C.write_tsv(C.RESULTS / "s6_downstream/preflight/local_blockers.tsv", blockers)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, text=True, capture_output=True, check=True).stdout.strip()
    v2_ready = sum(r["selected_retarget_variant_id"] == C.RETARGET_RESCUE_VARIANT for r in ready)
    summary = {
        "created_at": C.now(), "git_commit": git_commit,
        "status": "pass" if not blockers else "partial_ready",
        "s5_ready_input": len(s5_ready), "full_expected": len(ready),
        "blocked": len(blockers), "canary_rows": len(canary),
        "include_review_mode": include_review,
        "selected_variant_distribution": {
            "omnirt_v1": sum(r["selected_retarget_variant_id"] == "omnirt_v1" for r in ready),
            "omnirt_v2": v2_ready,
        },
        "object_distribution": {
            k: sum(r["object_key"] == k for r in ready) for k in C.OBJECT_KEYS
        },
        "frozen_cem": {"seed": C.CEM_SEED, "full_samples": C.CEM_FULL_SAMPLES, "full_opt_steps": C.CEM_FULL_OPT_STEPS,
                       "canary_samples": C.CEM_CANARY_SAMPLES, "canary_opt_steps": C.CEM_CANARY_OPT_STEPS},
    }
    C.write_json(C.RESULTS / "s6_downstream/manifests/cem_manifest_summary.json", summary)
    return summary, (0 if not blockers else 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-scenes", action="store_true")
    parser.add_argument("--include-review", action="store_true",
                        help="also include HANDOFF_REVIEW_VISUAL_QC rows (pre visual-review dry build)")
    args = parser.parse_args()
    summary, status = build(overwrite_scenes=args.overwrite_scenes, include_review=args.include_review)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
