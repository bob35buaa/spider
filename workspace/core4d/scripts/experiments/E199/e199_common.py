#!/usr/bin/env python3
"""Shared contract + helpers for E199 (OmniRetarget object augmentation -> full CEM).

E199 plumbs the upstream OmniRetarget/holosoma object-interaction augmentation
(original + 5 native configs: forward/left/right translation + -/+45 deg yaw)
through the SPIDER data pipeline and runs formal full CEM per variant.

Single source of truth for the E199 scope, paths, variant set, the frozen PRG
arm, and small IO helpers. The PRG scene/gate numeric contract is imported from
e173_common so E199 stays field-identical to the canonical E170-PRG candidate;
the rubber_hull hand patch is reused from the s5_handoff module. E199 only
relabels its artifacts (scene_act_E199_rubberHull_PRG, core4d_E199_* overrides)
and writes them into NEW __aug_{variant} task dirs -- it never overwrites the
historical base task scenes/overrides.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# --- Repo / result roots -----------------------------------------------------
REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E199"
RESULTS = REPO / "workspace/core4d/results/E199"
DCV3 = REPO / "workspace/core4d/scripts/data_construction_v3"
OVERRIDE_DIR = REPO / "examples/config/override"
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
DATA_PREPROCESS = REPO / "workspace/core4d/data_preprocess"

# reuse the E173 numeric contract (gate/penalty/CEM/geoms) as single source of truth
_E173_DIR = REPO / "workspace/core4d/scripts/experiments/E173"
import sys as _sys  # noqa: E402
for _p in (str(_E173_DIR), str(DCV3 / "stages/s5_handoff"), str(DCV3 / "lib"), str(DCV3 / "state")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import e173_common as E173  # noqa: E402
from patch_hand_collision import patch_scene  # noqa: E402

# --- live data / model / env paths (inherited from the E173 contract) --------
CORE4D_RAW_ROOT = E173.CORE4D_RAW_ROOT
SMPLX_MODEL_DIR = E173.SMPLX_MODEL_DIR
HOLOSOMA_REPO = E173.HOLOSOMA_REPO
RETARGET_PYTHON_BIN = E173.RETARGET_PYTHON_BIN
SPIDER_PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", str(REPO / ".venv/bin/python")))

# --- E199 scope: 8 objects x 1 case each (one per object) --------------------
# Each entry pins the historical `_original` dcv3 task (base_target_task); its
# task_info.json is the authority for date/seq/person/object_name/model + the
# object-template source scene. box004/021/023/024 reuse the E198 representative
# cases for cross-experiment comparability.
CASES: list[dict[str, str]] = [
    {"object_key": "box001", "base_target_task": "dcv3_omnirt_v1_ref_fk_box001_20231003_1_039_p1"},
    {"object_key": "box004", "base_target_task": "dcv3_omnirt_v1_ref_fk_box004_20231003_2_082_p1"},
    {"object_key": "box021", "base_target_task": "dcv3_omnirt_v1_ref_fk_box021_20231011_034_p1"},
    {"object_key": "box023", "base_target_task": "dcv3_omnirt_v1_ref_fk_box023_20231008_045_p1"},
    {"object_key": "box024", "base_target_task": "dcv3_omnirt_v1_ref_fk_box024_20231011_026_p1"},
    {"object_key": "bucket003", "base_target_task": "dcv3_omnirt_v1_ref_fk_bucket003_20231018_001_p1"},
    {"object_key": "bucket004", "base_target_task": "dcv3_omnirt_v1_ref_fk_bucket004_20231002_021_p1"},
    {"object_key": "bucket007", "base_target_task": "dcv3_omnirt_v1_ref_fk_bucket007_20231003_1_021_p1"},
]

# --- frozen retarget contract: omnirt_v2 (Phase-4) ---------------------------
# Object augmentation shifts the object out of reach for the omnirt_v1 (no
# constraint relaxation) contract, making many IK solves infeasible. E199 uses
# the project's canonical omnirt_v2 variant (all Phase-4 boolean controls on,
# foot-slide weight 1.0, object-penetration tolerance scale 0.8) uniformly for
# ALL 6 variants (incl. orig) so the orig-vs-aug comparison stays single-variable.
RETARGET_VARIANT = "omnirt_v2"
OMNIRT_V2_ENV = {
    "RETARGET_ENABLE_CONSTRAINT_RELAXATION": "1",
    "RETARGET_ENABLE_FOOT_Z_CONSTRAINT": "1",
    "RETARGET_ENABLE_CONTACT_PRESERVATION": "1",
    "RETARGET_FOOT_SLIDE_PENALTY_WEIGHT": "1.0",
    "RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE": "0.8",
}

# E199 short variant name -> upstream holosoma augmentation config name.
# `orig` is re-run under the frozen E199 config as the same-condition baseline.
VARIANTS: list[tuple[str, str]] = [
    ("orig", "original"),
    ("trans0", "trans_0"),   # forward  [0.2, 0, 0]
    ("trans1", "trans_1"),   # left     [0, 0.2, 0]
    ("trans2", "trans_2"),   # right    [0, -0.2, 0]
    ("rot0", "rot_0"),       # +45 deg yaw, +0.2 lateral
    ("rot1", "rot_1"),       # -45 deg yaw, -0.2 lateral
]
AUG_VARIANTS = [v for v in VARIANTS if v[0] != "orig"]

# --- box full-scale scope (plan229): translation-only, all s6-full-CEM box cases -
# Full-scale generates ONLY the 3 native translation variants (rotation is
# systematically infeasible per the pilot; orig is reused from the existing
# E198 A0/PRG full CEM rather than re-run). The case authority is the E198
# four-arm eval A0 arm (a case that has an A0/PRG full-CEM rollout == "reached
# s6 full CEM"); each case's base task dir is read from that arm's scene_xml
# parent (71 under dcv3_omnirt_v1_*, 16 under dcv3_omnirt_v2_*).
TRANS_VARIANTS: list[tuple[str, str]] = [
    ("trans0", "trans_0"), ("trans1", "trans_1"), ("trans2", "trans_2"),
]
BOX_OBJECTS = ("box001", "box004", "box021", "box023", "box024")
E198_ARM_CACHE = REPO / "workspace/core4d/results/E198/s6_downstream/eval/full_factorial/e198_arm_cache.tsv"
FULLSCALE_ARTIFACTS = RESULTS / "data_preprocess/manifests/e199_fullscale_aug_artifacts.tsv"
FULLSCALE_MANIFEST = RESULTS / "s6_downstream/manifests/e199_fullscale_priority_manifest.tsv"
FULLSCALE_AUTHORITY = RESULTS / "s6_downstream/manifests/e199_fullscale_authority.tsv"

# --- frozen PRG arm (relabelled E199, identical numeric contract to E173) -----
SCENE_NAME = "scene_act_E199_rubberHull_PRG"
RUBBER_INTERMEDIATE = "scene_act_E199_rubberHull"
PAIR_PREFIX = "E199_"
HAND_COLLISION_VARIANT = "rubber_hull"
LOWER_BODY_GEOMS = E173.LOWER_BODY_GEOMS
LEG_OBJECT_PENALTY_SCALE = E173.LEG_OBJECT_PENALTY_SCALE
LEG_OBJECT_PENALTY_MARGIN_M = E173.LEG_OBJECT_PENALTY_MARGIN_M
CEM_LEG_GATE_MIN_SDF_M = E173.CEM_LEG_GATE_MIN_SDF_M
CEM_LEG_GATE_MAX_VIOLATION_PCT = E173.CEM_LEG_GATE_MAX_VIOLATION_PCT
CEM_LEG_GATE_HARD_FLOOR_M = E173.CEM_LEG_GATE_HARD_FLOOR_M
CEM_LEG_GATE_MIN_VALID_FRAC = E173.CEM_LEG_GATE_MIN_VALID_FRAC
CEM_LEG_GATE_FALLBACK = E173.CEM_LEG_GATE_FALLBACK

# --- frozen CEM budget (full) ------------------------------------------------
CEM_SEED = E173.CEM_SEED            # 0
CEM_FULL_SAMPLES = E173.CEM_FULL_SAMPLES    # 1024
CEM_FULL_OPT_STEPS = E173.CEM_FULL_OPT_STEPS  # 32

# --- priority tiers: P0 orig(8) -> P1 trans(24) -> P2 rot(16) -----------------
TIER_OF_VARIANT = {"orig": "P0", "trans0": "P1", "trans1": "P1", "trans2": "P1", "rot0": "P2", "rot1": "P2"}
TIER_RANK = {"P0": 0, "P1": 1, "P2": 2}

MANIFEST_DIR = RESULTS / "s6_downstream/manifests"
FULL_MANIFEST = MANIFEST_DIR / "e199_priority_full_manifest.tsv"
AUTHORITY_TSV = MANIFEST_DIR / "e199_aug_authority.tsv"

# manifest schema consumed by run_local_priority_queue.py (E199 A0/PRG arm).
FIELDS = [
    "ordinal", "tier", "experiment", "arm", "object_key", "case_id",
    "aug_variant", "aug_translation", "aug_rotation_rad",
    "base_target_task", "target_task", "target_scene",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "override_id", "override_path", "override_sha256",
    "base_scene_act", "base_scene_sha256", "scene_act", "scene_name", "effective_scene_sha256",
    "extra_overrides", "cem_samples", "cem_opt_steps", "cem_seed",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "gpu_id", "status", "failure_mode", "execution_mode", "updated_at",
]


# --- small IO helpers --------------------------------------------------------
def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    # Prefer the unresolved repo-relative path so symlinked trees
    # (workspace/core4d/results -> tidal storage) stay under REPO.
    path = repo_path(value)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def sha256(value: str | Path) -> str:
    path = repo_path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/empty: {value}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def serial(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t",
                                    lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: serial(row.get(key, "")) for key in fields})
        # JuiceFS occasionally throws a transient EIO on os.replace; retry a few
        # times so the long-running CEM queue survives storage hiccups instead of
        # crashing mid-run (it writes this manifest after every row).
        import time as _time
        for _attempt in range(6):
            try:
                Path(tmp).replace(path)
                break
            except OSError:
                if _attempt == 5:
                    raise
                _time.sleep(0.5 * (_attempt + 1))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(value: str | Path, payload: Any) -> None:
    path = repo_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# --- per-case metadata (authority = base task_info.json) ---------------------
def load_case_meta(base_target_task: str) -> dict[str, str]:
    """Read date/seq/person/object_name/model + template scene from task_info.json."""
    info_path = TASK_ROOT / base_target_task / "task_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing base task_info: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    date, seq, person = info["date"], info["seq"], info["person"]
    object_name = info["object_name"]
    holosoma_task = f"{date}-{seq}-{person}-{object_name}_with_obj"
    return {
        "date": date, "seq": seq, "person": person,
        "object_name": object_name,
        "object_model_rel": info["object_model_rel"],
        "source_scene": info["source_scene"],
        "source_scene_task": Path(info["source_scene"]).parent.name,
        "holosoma_task": holosoma_task,
    }


def aug_task_name(base_target_task: str, e199_variant: str) -> str:
    # relabel the retarget variant to the E199 contract (omnirt_v2); the base
    # task pin (omnirt_v1) is only the metadata/geometry-template authority.
    # Idempotent: a base already on omnirt_v2 (16 full-scale box cases) stays v2.
    base = base_target_task
    if "omnirt_v1" in base:
        base = base.replace("omnirt_v1", RETARGET_VARIANT)
    return f"{base}__aug_{e199_variant}"


def load_fullscale_cases(objects: tuple[str, ...] = BOX_OBJECTS) -> list[dict[str, str]]:
    """Derive the s6-full-CEM box case registry from the E198 A0 arm.

    One entry per unique A0 case_id for the requested objects. `base_target_task`
    is the dcv3 dir that owns the A0 scene (v1 or v2); it carries task_info.json
    + scene.xml (geometry template + metadata). The `orig_*` fields point at the
    existing A0/PRG full-CEM rollout to reuse as the same-case baseline in eval.
    """
    rows = read_tsv(E198_ARM_CACHE)
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("arm") != "A0" or row.get("object_key") not in objects:
            continue
        case_id = row["case_id"]
        if case_id in seen:
            continue
        seen.add(case_id)
        cases.append({
            "object_key": row["object_key"],
            "case_id": case_id,
            "base_target_task": Path(row["scene_xml"]).parent.name,
            "orig_result_npz": row.get("result_npz", ""),
            "orig_outdir_npz": row.get("outdir_npz", ""),
            "orig_scene_act": row.get("scene_xml", ""),
            "orig_trajectory": row.get("trajectory", ""),
            "orig_contact_mask": row.get("contact_mask", ""),
            "orig_retarget_variant_id": row.get("retarget_variant_id", ""),
        })
    cases.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return cases


def aug_translation(e199_variant: str) -> str:
    return {
        "orig": "0,0,0", "trans0": "0.2,0,0", "trans1": "0,0.2,0",
        "trans2": "0,-0.2,0", "rot0": "0,0.2,0", "rot1": "0,-0.2,0",
    }[e199_variant]


def aug_rotation_rad(e199_variant: str) -> str:
    return {"orig": "0", "trans0": "0", "trans1": "0", "trans2": "0",
            "rot0": f"{math.pi / 4:.6f}", "rot1": f"{-math.pi / 4:.6f}"}[e199_variant]


# --- PRG scene builder (ported from E173 build_prg_cem_manifest.build_scene) --
def _tree_signature(element: ET.Element, *, ignore_pairs: bool = False) -> Any:
    children = []
    for child in element:
        if ignore_pairs and child.tag == "pair" and child.get("name", "").startswith(PAIR_PREFIX):
            continue
        children.append(_tree_signature(child, ignore_pairs=ignore_pairs))
    return element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(), tuple(children)


def _convert_reference_to_scene(qpos: np.ndarray, scene_xml: Path) -> np.ndarray:
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


def _min_lowerbody_distance(model: mujoco.MjModel, frames: np.ndarray) -> float:
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


def build_prg_scene(case_id: str, base_scene_act: Path, trajectory: Path, *, overwrite: bool) -> dict[str, Any]:
    """standard scene_act -> rubber_hull hand sidecar -> +16 lower-body pairs -> E199 PRG sidecar.

    Ported verbatim (logic + numeric contract) from E173 build_prg_cem_manifest.build_scene;
    only the sidecar/pair labels are relabelled E199 and snapshots land under E199 results.
    """
    base_scene_act = repo_path(base_scene_act)
    if not base_scene_act.is_file():
        raise FileNotFoundError(base_scene_act)
    task_dir = base_scene_act.parent
    patch_scene(
        base_scene_act=base_scene_act, out_dir=task_dir, case_id=case_id,
        hand_collision_variant_id=HAND_COLLISION_VARIANT, scene_name=RUBBER_INTERMEDIATE,
        install_dir=None, repo=REPO,
    )
    rubber_scene = task_dir / f"{RUBBER_INTERMEDIATE}.xml"
    tree = ET.parse(rubber_scene)
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
        ET.SubElement(contact, "pair", {
            "name": f"{PAIR_PREFIX}{name}_object", "geom1": name, "geom2": "object_collision",
            "solref": "0.008 1", "margin": "0", "gap": "0", "condim": "1",
        })
    output = task_dir / f"{SCENE_NAME}.xml"
    if output.exists() and not overwrite:
        if _tree_signature(ET.parse(output).getroot()) != _tree_signature(root):
            raise FileExistsError(f"different E199 sidecar exists: {output}")
    else:
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)
    physical_root = ET.parse(output).getroot()
    if _tree_signature(ET.parse(rubber_scene).getroot()) != _tree_signature(physical_root, ignore_pairs=True):
        raise AssertionError("semantic_diff_exceeds_pairs")
    pairs = [p for p in physical_root.findall("./contact/pair") if p.get("name", "").startswith(PAIR_PREFIX)]
    if len(pairs) != 16 or len({p.get("name") for p in pairs}) != 16:
        raise AssertionError(f"pair_count={len(pairs)}")
    model = mujoco.MjModel.from_xml_path(str(output))
    pair_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_PAIR, i) for i in range(model.npair)}
    if not {f"{PAIR_PREFIX}{name}_object" for name in LOWER_BODY_GEOMS}.issubset(pair_names):
        raise AssertionError("compiled_pair_missing")
    for side in ("lh", "rh"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, side)
        if gid < 0 or mujoco.mjtGeom(int(model.geom_type[gid])).name != "mjGEOM_MESH":
            raise AssertionError(f"{side}_not_mesh_after_rubber_hull")
    with np.load(repo_path(trajectory), allow_pickle=True) as data:
        reference = np.asarray(data["qpos"], dtype=np.float64)
    converted = _convert_reference_to_scene(reference, output)
    reference_min = _min_lowerbody_distance(model, converted[:min(5, len(converted))])
    if reference_min < CEM_LEG_GATE_HARD_FLOOR_M:
        raise ValueError(f"runtime_initial_overlap:reference_first5={reference_min:.6f}")
    snapshot_dir = RESULTS / "scene_snapshot/cem_sidecars" / case_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(base_scene_act, snapshot_dir / base_scene_act.name)
    shutil.copy2(rubber_scene, snapshot_dir / rubber_scene.name)
    shutil.copy2(output, snapshot_dir / output.name)
    return {
        "base_scene": rel(base_scene_act), "base_scene_sha256": sha256(base_scene_act),
        "rubber_scene": rel(rubber_scene), "rubber_scene_sha256": sha256(rubber_scene),
        "physical_scene": rel(output), "physical_scene_sha256": sha256(output),
        "compiled_pair_count": 16,
        "reference_first5_min_lowerbody_object_distance_m": reference_min,
    }


def prg_override_payload(base_task_override_id: str) -> dict[str, Any]:
    """E199 PRG override payload (defaults -> aug base task yaml; identical gate to E173)."""
    return {
        "defaults": [base_task_override_id, "_self_"],
        "scene_name": SCENE_NAME,
        "leg_object_penalty_scale": LEG_OBJECT_PENALTY_SCALE,
        "leg_object_penalty_margin_m": LEG_OBJECT_PENALTY_MARGIN_M,
        "leg_object_penalty_geom_names": list(LOWER_BODY_GEOMS),
        "leg_object_penalty_geom_ids": [],
        "leg_object_penalty_gate_source": "always",
        "leg_object_penalty_start_eval_time": 0.0,
        "leg_object_penalty_end_eval_time": 999.0,
        "cem_leg_gate_enabled": True,
        "cem_leg_gate_geom_names": list(LOWER_BODY_GEOMS),
        "cem_leg_gate_geom_ids": [],
        "cem_leg_gate_min_sdf_m": CEM_LEG_GATE_MIN_SDF_M,
        "cem_leg_gate_max_violation_pct": CEM_LEG_GATE_MAX_VIOLATION_PCT,
        "cem_leg_gate_hard_floor_m": CEM_LEG_GATE_HARD_FLOOR_M,
        "cem_leg_gate_min_valid_frac": CEM_LEG_GATE_MIN_VALID_FRAC,
        "cem_leg_gate_fallback": CEM_LEG_GATE_FALLBACK,
    }
