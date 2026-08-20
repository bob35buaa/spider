#!/usr/bin/env python3
"""Shared contract + helpers for E202 (bucket object augmentation -> E178 full CEM).

E202 plumbs the SAME upstream OmniRetarget/holosoma object-interaction
augmentation as E199 (native translation configs only), but builds the CEM
scene with the **E178 collision body** (per-object contact-aligned five-segment
proxy, `union` SDF) on the **E174 PRG arm** (rubber_hull hand + E170-PRG
reward/gate) and runs the frozen E174/E178 CEM budget (1024x32, seed 0). It is a
sibling of E199 (box, 16-pair single-geom PRG) -- see plan232.

The generic IO / upstream-augmentation helpers are reused verbatim from
`e199_common`; E202 only overrides the CEM-scene build step (`build_prg_scene`)
with the bucket proxy + 18-pair/geom construction and swaps the case registry to
the 27 E178 full-CEM bucket cases. It never overwrites E178 / E174 historical
scenes -- all artifacts land in NEW `__aug_{variant}` task dirs, relabelled E202.
"""

from __future__ import annotations

import copy
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# --- reuse the E199 generic contract (IO, repo roots, upstream aug env) -------
_E199_DIR = Path(__file__).resolve().parents[1] / "E199"
if str(_E199_DIR) not in sys.path:
    sys.path.insert(0, str(_E199_DIR))
import e199_common as E199  # noqa: E402

# re-export the IO / path / upstream helpers E202 shares with E199
REPO = E199.REPO
OVERRIDE_DIR = E199.OVERRIDE_DIR
TASK_ROOT = E199.TASK_ROOT
CORE4D_RAW_ROOT = E199.CORE4D_RAW_ROOT
SMPLX_MODEL_DIR = E199.SMPLX_MODEL_DIR
HOLOSOMA_REPO = E199.HOLOSOMA_REPO
RETARGET_PYTHON_BIN = E199.RETARGET_PYTHON_BIN
SPIDER_PYTHON_BIN = E199.SPIDER_PYTHON_BIN
OMNIRT_V2_ENV = E199.OMNIRT_V2_ENV
RETARGET_VARIANT = E199.RETARGET_VARIANT
E173 = E199.E173

now = E199.now
repo_path = E199.repo_path
rel = E199.rel
sha256 = E199.sha256
safe_id = E199.safe_id
read_tsv = E199.read_tsv
write_tsv = E199.write_tsv
write_json = E199.write_json
load_case_meta = E199.load_case_meta
aug_task_name = E199.aug_task_name
aug_translation = E199.aug_translation
aug_rotation_rad = E199.aug_rotation_rad

# --- E202 result roots -------------------------------------------------------
RESULTS = REPO / "workspace/core4d/results/E202"
DATA_PREPROCESS_REL = "workspace/core4d/results/E202/data_preprocess"

# --- E202 scope: the 27 E178 full-CEM bucket cases (authority = E178 full) ----
# Translation-only (rotation is systematically infeasible per the E199 pilot).
TRANS_VARIANTS: list[tuple[str, str]] = list(E199.TRANS_VARIANTS)
BUCKET_OBJECTS = ("bucket003", "bucket004", "bucket007")
EXPECTED_OBJECT_COUNTS = {"bucket003": 9, "bucket004": 4, "bucket007": 14}
E178_FULL_MANIFEST = (
    REPO / "workspace/core4d/results/E178/s6_downstream/manifests/"
    "semantic_bucket_full_manifest.tsv"
)

# --- frozen E178/E174 CEM scene contract -------------------------------------
# scene_act sidecar name (relabelled E202, never overwrites E178's).
SCENE_NAME = "scene_act_E202_bucketAlignedTop_PRG"
RUBBER_INTERMEDIATE = "scene_act_E202_rubberHull"
PAIR_PREFIX = "E202_"
HAND_COLLISION_VARIANT = "rubber_hull"
# E178 collision axes (both overrides carried on the aug base task yaml).
OBJECT_COLLISION_SDF_MODE = "union"
OBJECT_COLLISION_SDF_BATCH_GROUPS = True

# --- frozen CEM budget + PRG reward/gate contract (single source = E174) ------
_E174_DIR = REPO / "workspace/core4d/scripts/experiments/E174"
if str(_E174_DIR) not in sys.path:
    sys.path.insert(0, str(_E174_DIR))
import e174_common as E174  # noqa: E402

CEM_SEED = E174.CEM_SEED               # 0
CEM_FULL_SAMPLES = E174.CEM_FULL_SAMPLES     # 1024
CEM_FULL_OPT_STEPS = E174.CEM_FULL_OPT_STEPS   # 32
LEG_OBJECT_PENALTY_SCALE = E174.LEG_OBJECT_PENALTY_SCALE
LEG_OBJECT_PENALTY_MARGIN_M = E174.LEG_OBJECT_PENALTY_MARGIN_M
CEM_LEG_GATE_MIN_SDF_M = E174.CEM_LEG_GATE_MIN_SDF_M
CEM_LEG_GATE_MAX_VIOLATION_PCT = E174.CEM_LEG_GATE_MAX_VIOLATION_PCT
CEM_LEG_GATE_HARD_FLOOR_M = E174.CEM_LEG_GATE_HARD_FLOOR_M
CEM_LEG_GATE_MIN_VALID_FRAC = E174.CEM_LEG_GATE_MIN_VALID_FRAC
CEM_LEG_GATE_FALLBACK = E174.CEM_LEG_GATE_FALLBACK

# --- reuse the vetted E175/E177/E178 geometry code as libraries ---------------
_E178_DIR = REPO / "workspace/core4d/scripts/experiments/E178"
_E177_DIR = REPO / "workspace/core4d/scripts/experiments/E177"
_E175_DIR = REPO / "workspace/core4d/scripts/experiments/E175"
for _p in (str(_E178_DIR), str(_E177_DIR), str(_E175_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import build_nonbox_multigeom_production as GEOM  # noqa: E402  (E175 base)
import semantic_bucket_proxy as PROXY  # noqa: E402  (E177)
from contact_aligned_bucket_proxy import build_contact_aligned_boxes  # noqa: E402  (E178)
from patch_hand_collision import patch_scene  # noqa: E402

EXPECTED_BOXES_BY_OBJECT = PROXY.EXPECTED_BOXES_BY_OBJECT  # {003:5, 004:1, 007:5}


def _drop_broken_torch() -> None:
    """The SPIDER venv ships a broken `torch` namespace stub (no `Tensor`); the
    E175/E177/E178 geometry imports pull it into sys.modules, which poisons
    scipy's `Rotation.from_quat` (array-api-compat probes `getattr(torch,'Tensor')`
    for any iterable arg once torch is imported). Drop the broken stub so scipy
    short-circuits -- E199/E174 never imported these modules so never hit this.
    """
    mod = sys.modules.get("torch")
    if mod is not None and not hasattr(mod, "Tensor"):
        del sys.modules["torch"]


_drop_broken_torch()


# manifest schema (consumed by the E199 priority queue, reused by E202).
FIELDS = list(E199.FIELDS)

MANIFEST_DIR = RESULTS / "s6_downstream/manifests"
FULL_MANIFEST = MANIFEST_DIR / "e202_bucket_priority_manifest.tsv"
AUTHORITY_TSV = MANIFEST_DIR / "e202_bucket_authority.tsv"
ARTIFACTS_TSV = RESULTS / "data_preprocess/manifests/e202_bucket_aug_artifacts.tsv"


# --- case registry: derive the 27 E178 full-CEM bucket cases -----------------
def load_e178_bucket_cases(objects: tuple[str, ...] = BUCKET_OBJECTS) -> list[dict[str, str]]:
    """One entry per E178 full-manifest bucket case.

    `base_target_task` is the dcv3 dir owning the E178 orig scene (task_info.json
    + scene.xml geometry template + metadata). `orig_*` fields point at the
    existing E178 full-CEM rollout to reuse as the same-case baseline in eval.
    `e178_scene_act` is the authority for the C1 geometry-parity check.
    """
    rows = read_tsv(E178_FULL_MANIFEST)
    snap_root = REPO / "workspace/core4d/results/E178/scene_snapshot/semantic_bucket_proxy"
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("object_key") not in objects:
            continue
        case_id = row["case_id"]
        if case_id in seen:
            continue
        seen.add(case_id)
        # E178's task-dir scene_act sidecar was often overwritten by later
        # experiments; the E178 snapshot is the surviving authoritative copy.
        orig_scene = row.get("scene_act", "")
        if orig_scene and not repo_path(orig_scene).is_file():
            snap = snap_root / case_id / "scene_act_E178_contactAlignedTop.xml"
            if snap.is_file():
                orig_scene = rel(snap)
        cases.append({
            "object_key": row["object_key"],
            "case_id": case_id,
            "base_target_task": row["target_task"],
            "e178_scene_act": row.get("scene_act", ""),
            "e178_scene_name": row.get("scene_name", ""),
            "e178_object_geom_count": row.get("object_geom_count", ""),
            "e178_pair_count": row.get("compiled_robot_object_pair_count", ""),
            "orig_result_npz": row.get("result_npz", ""),
            "orig_outdir_npz": row.get("outdir_npz", ""),
            "orig_scene_act": orig_scene,
            "orig_trajectory": row.get("trajectory", ""),
            "orig_contact_mask": row.get("contact_mask", ""),
            "orig_override_id": row.get("override_id", ""),
            "orig_retarget_variant_id": row.get("retarget_variant_id", ""),
        })
    cases.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return cases


# --- euler-aware reference conversion (from E174, generalised to N object geoms)
def _convert_reference_to_scene(qpos: np.ndarray, scene_xml: Path) -> np.ndarray:
    import json
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
    from scipy.spatial.transform import Rotation
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


def _reference_first5_union_min_distance(model: mujoco.MjModel, frames: np.ndarray) -> float:
    """Min SDF over ALL object_collision* geoms x lower-body geoms on the first frames."""
    data = mujoco.MjData(model)
    object_ids = [
        gid for gid in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith("object_collision")
    ]
    leg_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in GEOM.LOWER_BODY_GEOMS]
    if not object_ids or any(index < 0 for index in leg_ids):
        raise ValueError("lower-body/object geoms missing")
    minimum = float("inf")
    fromto = np.zeros(6, dtype=np.float64)
    for frame in frames:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for leg_id in leg_ids:
            for obj_id in object_ids:
                minimum = min(minimum, float(mujoco.mj_geomDistance(model, data, leg_id, obj_id, 10.0, fromto)))
    return minimum


def _replace_robot_object_pairs(root: ET.Element, object_names: list[str]) -> int:
    """E202-labelled copy of GEOM.replace_robot_object_pairs (hand condim4 + leg condim1)."""
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for pair in list(contact.findall("pair")):
        if GEOM.is_robot_object_pair(pair):
            contact.remove(pair)
    for object_index, object_name in enumerate(object_names):
        for hand in GEOM.HAND_GEOMS:
            ET.SubElement(contact, "pair", {
                "name": f"{PAIR_PREFIX}{hand}_obj{object_index:03d}",
                "geom1": hand, "geom2": object_name,
                "solref": "0.008 1", "friction": "2 1", "condim": "4",
            })
        for body_geom in GEOM.LOWER_BODY_GEOMS:
            ET.SubElement(contact, "pair", {
                "name": f"{PAIR_PREFIX}{body_geom}_obj{object_index:03d}",
                "geom1": body_geom, "geom2": object_name,
                "solref": "0.008 1", "margin": "0", "gap": "0", "condim": "1",
            })
    return len(GEOM.ROBOT_OBJECT_GEOMS) * len(object_names)


def build_prg_scene(case_id: str, base_scene_act: Path, trajectory: Path, *,
                    overwrite: bool, object_key: str) -> dict[str, Any]:
    """standard scene_act -> rubber_hull hand -> E178 5-seg proxy + 18-pair/geom union -> E202 sidecar.

    Reproduces the E178 collision body on an augmented task by construction: the
    proxy boxes are derived from the object MESH (pose-independent) via the vetted
    E178 `build_contact_aligned_boxes`, and the robot-object pairs are rebuilt with
    the E175/E178 18-pair/geom (hand condim4 + lower-body condim1) contract. Same
    return-dict shape as E199.build_prg_scene, plus bucket geom/pair diagnostics.
    """
    base_scene_act = repo_path(base_scene_act)
    if not base_scene_act.is_file():
        raise FileNotFoundError(base_scene_act)
    if object_key not in EXPECTED_BOXES_BY_OBJECT:
        raise ValueError(f"E202 unexpected object: {object_key}")
    task_dir = base_scene_act.parent

    # step 1: rubber_hull hand sidecar (sphere hands -> mesh hull)
    patch_scene(
        base_scene_act=base_scene_act, out_dir=task_dir, case_id=case_id,
        hand_collision_variant_id=HAND_COLLISION_VARIANT, scene_name=RUBBER_INTERMEDIATE,
        install_dir=None, repo=REPO,
    )
    rubber_scene = task_dir / f"{RUBBER_INTERMEDIATE}.xml"
    rubber_root = ET.parse(rubber_scene).getroot()
    root = copy.deepcopy(rubber_root)

    # step 2: replace single object_collision with the E178 contact-aligned proxy
    object_body = GEOM.find_object_body(root)
    mesh = GEOM.resolve_object_mesh(rubber_scene)
    boxes, metadata = build_contact_aligned_boxes(mesh, object_key)
    generated_xml, _names = PROXY.proxy_xml(boxes)
    metrics = PROXY.fidelity_metrics(mesh, boxes)
    metrics.update(PROXY.union_fidelity_metrics(mesh, boxes))
    if metrics["mesh_to_proxy_p90_m"] > 0.08:
        raise AssertionError(f"{object_key} mesh->proxy p90 exceeds 8cm: {metrics['mesh_to_proxy_p90_m']:.4f}")
    if object_key != "bucket004" and (
        metrics["union_mesh_to_proxy_p90_m"] > 0.04 or metrics["union_proxy_to_mesh_p90_m"] > 0.04
    ):
        raise AssertionError(f"{object_key} union fidelity exceeds 4cm: {metrics}")

    object_names = GEOM.replace_bucket_proxy(object_body, generated_xml)
    expected_count = EXPECTED_BOXES_BY_OBJECT[object_key]
    if len(object_names) != expected_count:
        raise AssertionError(f"{object_key} geom count={len(object_names)} expected={expected_count}")

    # step 3: rebuild robot-object pairs (18 per object geom)
    pair_count = _replace_robot_object_pairs(root, object_names)
    if pair_count != len(GEOM.ROBOT_OBJECT_GEOMS) * expected_count:
        raise AssertionError(f"{object_key} pair count={pair_count}")

    # scene did not drift outside the proxy/pair axes
    if GEOM.stripped_signature(rubber_root) != GEOM.stripped_signature(root):
        raise AssertionError("scene drift outside proxy/pair axes")

    output = task_dir / f"{SCENE_NAME}.xml"
    tree = ET.ElementTree(root)
    if output.exists() and not overwrite:
        if GEOM.semantic_xml_signature(ET.parse(output).getroot()) != GEOM.semantic_xml_signature(root):
            raise FileExistsError(f"different E202 sidecar exists: {output}")
    else:
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)

    # compiled contract: geom count + pair count parity
    compiled = GEOM.compiled_contract(output)
    if compiled["object_geom_count"] != expected_count:
        raise AssertionError("XML/compiled geom count drift")
    if compiled["compiled_robot_object_pair_count"] != pair_count:
        raise AssertionError("XML/compiled pair count drift")
    for side in ("lh", "rh"):
        gid = mujoco.mj_name2id(compiled["model"], mujoco.mjtObj.mjOBJ_GEOM, side)
        if gid < 0 or mujoco.mjtGeom(int(compiled["model"].geom_type[gid])).name != "mjGEOM_MESH":
            raise AssertionError(f"{side}_not_mesh_after_rubber_hull")

    # runtime initial overlap (run_mjwp seeds from qpos_ref[0])
    with np.load(repo_path(trajectory), allow_pickle=True) as data:
        reference = np.asarray(data["qpos"], dtype=np.float64)
    _drop_broken_torch()  # guard: keep scipy's Rotation off the broken torch stub
    converted = _convert_reference_to_scene(reference, output)
    reference_min = _reference_first5_union_min_distance(compiled["model"], converted[:min(5, len(converted))])
    if reference_min < CEM_LEG_GATE_HARD_FLOOR_M:
        raise ValueError(f"runtime_initial_overlap:reference_first5={reference_min:.6f}")

    # snapshot (rule 10b)
    snapshot_dir = RESULTS / "scene_snapshot/cem_sidecars" / case_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_scene_act, snapshot_dir / base_scene_act.name)
    shutil.copy2(rubber_scene, snapshot_dir / rubber_scene.name)
    shutil.copy2(output, snapshot_dir / output.name)

    return {
        "base_scene": rel(base_scene_act), "base_scene_sha256": sha256(base_scene_act),
        "rubber_scene": rel(rubber_scene), "rubber_scene_sha256": sha256(rubber_scene),
        "physical_scene": rel(output), "physical_scene_sha256": sha256(output),
        "object_geom_count": compiled["object_geom_count"],
        "compiled_robot_object_pair_count": pair_count,
        "object_collision_geom_names": ",".join(compiled["object_names"]),
        "proxy_variant": "solid_mesh_aabb_1box" if object_key == "bucket004"
        else "five_body_steps_no_lid_contact_aligned_top",
        "collision_policy": metadata["collision_policy"],
        "mesh_to_proxy_p90_m": metrics["mesh_to_proxy_p90_m"],
        "union_mesh_to_proxy_p90_m": metrics.get("union_mesh_to_proxy_p90_m", ""),
        "union_proxy_to_mesh_p90_m": metrics.get("union_proxy_to_mesh_p90_m", ""),
        "reference_first5_min_lowerbody_object_distance_m": reference_min,
    }


def prg_override_payload(base_task_override_id: str) -> dict[str, Any]:
    """E202 PRG override: E174 PRG reward/gate + E178 union-SDF collision axes."""
    return {
        "defaults": [base_task_override_id, "_self_"],
        "scene_name": SCENE_NAME,
        "object_collision_sdf_mode": OBJECT_COLLISION_SDF_MODE,
        "object_collision_sdf_batch_groups": OBJECT_COLLISION_SDF_BATCH_GROUPS,
        "leg_object_penalty_scale": LEG_OBJECT_PENALTY_SCALE,
        "leg_object_penalty_margin_m": LEG_OBJECT_PENALTY_MARGIN_M,
        "leg_object_penalty_geom_names": list(GEOM.LOWER_BODY_GEOMS),
        "leg_object_penalty_geom_ids": [],
        "leg_object_penalty_gate_source": "always",
        "leg_object_penalty_start_eval_time": 0.0,
        "leg_object_penalty_end_eval_time": 999.0,
        "cem_leg_gate_enabled": True,
        "cem_leg_gate_geom_names": list(GEOM.LOWER_BODY_GEOMS),
        "cem_leg_gate_geom_ids": [],
        "cem_leg_gate_min_sdf_m": CEM_LEG_GATE_MIN_SDF_M,
        "cem_leg_gate_max_violation_pct": CEM_LEG_GATE_MAX_VIOLATION_PCT,
        "cem_leg_gate_hard_floor_m": CEM_LEG_GATE_HARD_FLOOR_M,
        "cem_leg_gate_min_valid_frac": CEM_LEG_GATE_MIN_VALID_FRAC,
        "cem_leg_gate_fallback": CEM_LEG_GATE_FALLBACK,
    }
