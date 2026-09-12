#!/usr/bin/env python3
"""E215 shared contract: bucket+box ROTATION object-augmentation -> per-arm full CEM.

E215 scales the *rotation* augmentation (rot_0/rot_1 = +/-45 deg yaw + 0.2 m
lateral, the upstream native configs) onto 7 bucket + 31 box cases, one aug task
per (case, rot variant), then a formal full CEM under each case's SELECTED arm.

Why rotation is only now runnable
---------------------------------
E199/E202 shipped translation only and recorded rotation as "systematically
infeasible".  E208 found that was an upstream crash (holosoma ``src/utils.py``
called ``R.from_euler("z", (N,))`` where scipy>=1.15 needs ``(N,1)``) that fired
*before any IK* -- rotation had never actually been evaluated.  Fixed in
holosoma ``9e544b1``; E208 then ran rot on desk/chair.  E215 carries that to
bucket/box, reusing every piece of the E199/E202/E208 pipeline unchanged and
only swapping (a) the case set, (b) the five arm builders, (c) rot as the sole
build variant.

Five arm groups, keyed by object (plan247 3.1, design locks in progress 2026-09-12):
  * bucket003            E202 PRG (bucketAlignedTop 5-seg proxy, union SDF)
  * bucket007            E202 PRG + object gravcomp=1 (E210 G1)
  * box021               E199 PRG (rubber_hull, 16-pair single-geom)
  * box023               E167A noPRG (rubber_hull hand, no leg pairs/gate)
  * box001/004/024       E199 PRG + gravcomp=1 (G1) + A2 hand gate  == E200 prg_g1a2

All numeric contracts (reward, gate, CEM budget, proxy geometry) are imported
from the arm commons; E215 re-types nothing.  Its only literals are the frozen
scene basenames, asserted against what the builders actually produce.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = REPO / "workspace/core4d/scripts/experiments"

for _p in (EXPERIMENTS / "E199",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import e199_common as E199  # noqa: E402  (base IO helpers + box PRG scene builder)


def _drop_broken_torch() -> None:
    """Drop the SPIDER venv's broken ``torch`` namespace stub (no ``Tensor``).

    Lifted from e202_common: the E175/E177/E178 geometry imports pull the stub
    into ``sys.modules`` and poison scipy's ``Rotation.from_quat``.
    """
    mod = sys.modules.get("torch")
    if mod is not None and not hasattr(mod, "Tensor"):
        del sys.modules["torch"]


_drop_broken_torch()


def load_e215_module(name: str):
    """Import an E215-local sibling by explicit path (avoids E199/E202 shadowing).

    Both e199_common and e202_common push their own experiment dir onto
    ``sys.path[0]`` and E215 has same-named siblings (``build_augmented_tasks``,
    ``build_aug_manifest``).  This is the sanctioned way for E215 scripts to
    import each other.
    """
    path = SCRIPT_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no such E215 module: {path}")
    spec = importlib.util.spec_from_file_location(f"e215_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E215"
PLAN_SLOT = 247
LOG_SLOT = 306
PLAN_REF = f"workspace/core4d/plan/{PLAN_SLOT}_E215_bucket_box_rot_augmentation_plan.md"
LOG_REF = f"workspace/core4d/log/{LOG_SLOT}_E215_bucket_box_rot_augmentation.md"
RUN_ID = "R302"

# --------------------------------------------------------------------------
# Retarget contract -- omnirt_v2, uniform (plan247 lock 3)
# --------------------------------------------------------------------------
# E199/E202 froze omnirt_v2 for ALL variants (incl. orig), so E215 seeds v2->v2.
# We reuse e199_common.OMNIRT_V2_ENV verbatim (5 keys; REPLACE_WRIST_WITH_FINGERTIP
# is left at pipeline.sh's default of 1, matching the E199/E202 baseline -- NOT
# E208's 6-key desk/chair dict whose wrist=0 came from E206).
RETARGET_VARIANT = "omnirt_v2"
OMNIRT_V2_ENV = dict(E199.OMNIRT_V2_ENV)
PIPELINE_DATASET_ENV = {"SPIDER_DATASET": "core4d", "SPIDER_SOURCE_DATASET": "core4d"}

# --------------------------------------------------------------------------
# Variants -- rotation only (plan247 scope)
# --------------------------------------------------------------------------
ROT_VARIANTS: list[tuple[str, str]] = [("rot0", "rot_0"), ("rot1", "rot_1")]
BUILD_VARIANTS = list(ROT_VARIANTS)
ROTATION_FIX_COMMIT = "9e544b1"  # holosoma; before this, rot_* could not run at all

aug_translation = E199.aug_translation      # rot0 -> "0,0.2,0", rot1 -> "0,-0.2,0"
aug_rotation_rad = E199.aug_rotation_rad     # rot0 -> +pi/4, rot1 -> -pi/4

# Effective-yaw floor: a very late trim_start decays the yaw (rotation_tau=25)
# to near zero before SPIDER's window opens (E208 chair005: 0.58 deg).  Flag,
# do not silently ship such a degenerate "rotation".  Reported, not a hard gate.
EFFECTIVE_YAW_FLOOR_DEG = 30.0
NOMINAL_YAW_DEG = 45.0

# --------------------------------------------------------------------------
# Case registry -- authority: tmp/paper_case_id.txt (box*/bucket* rows), pinned
# here for reproducibility (plan247 3.2; verified against TASK_ROOT + warm-start
# trees 2026-09-12, see memory e215-case-set-authority).
# --------------------------------------------------------------------------
# Each case_id already carries object+date+seq+person.  base_variant is the dcv3
# task-dir prefix that actually exists in TASK_ROOT (task_info.json + scene.xml):
#   v1       -> dcv3_omnirt_v1_ref_fk_{case_id}
#   v2       -> dcv3_omnirt_v2_ref_fk_{case_id}
#   missing  -> no base task/warm-start; Stage 0 must build it first, else excluded.
_CASE_IDS: list[str] = [
    # box001 (6)
    "box001_20231003_1_040_p2", "box001_20231003_2_039_p1", "box001_20231003_2_041_p1",
    "box001_20231020_014_p1", "box001_20231023_108_p1", "box001_20231023_108_p2",
    # box021 (11)
    "box021_20231011_034_p1", "box021_20231011_034_p2", "box021_20231011_036_p1",
    "box021_20231011_037_p2", "box021_20231011_038_p1", "box021_20231020_020_p1",
    "box021_20231020_022_p1", "box021_20231020_023_p1", "box021_20231011_035_p1",
    "box021_20231011_036_p2", "box021_20231018_029_p2",
    # box004 (3)
    "box004_20231003_2_083_p1", "box004_20231003_2_083_p2", "box004_20231003_2_082_p1",
    # box024 (4)
    "box024_20231011_026_p1", "box024_20231011_026_p2", "box024_20231011_027_p2",
    "box024_20231011_028_p2",
    # box023 (7)
    "box023_20231008_046_p1", "box023_20231011_021_p1", "box023_20231011_021_p2",
    "box023_20231020_040_p2", "box023_20231020_041_p1", "box023_20231008_045_p1",
    "box023_20231020_042_p2",
    # bucket003 (3)
    "bucket003_20231018_001_p1", "bucket003_20231018_005_p2", "bucket003_20231020_068_p1",
    # bucket007 (4)
    "bucket007_20231003_2_021_p1", "bucket007_20231023_073_p1",
    "bucket007_20231023_075_p1", "bucket007_20231023_075_p2",
]
_BASE_VARIANT_V2 = frozenset({
    "box001_20231003_1_040_p2", "box001_20231003_2_039_p1", "box001_20231023_108_p1",
    "box021_20231011_034_p2", "box024_20231011_026_p2", "box024_20231011_027_p2",
})
_BASE_MISSING = frozenset({"box021_20231011_035_p1", "box021_20231018_029_p2"})

CASE_SET_AUTHORITY = "tmp/paper_case_id.txt"  # provenance, not read at run time
EXPECTED_CASES = 38
EXPECTED_CASES_BY_OBJECT = {
    "box001": 6, "box021": 11, "box004": 3, "box024": 4, "box023": 7,
    "bucket003": 3, "bucket007": 4,
}


def object_key_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


def base_variant_of(case_id: str) -> str:
    if case_id in _BASE_MISSING:
        return "missing"
    return "omnirt_v2" if case_id in _BASE_VARIANT_V2 else "omnirt_v1"


def base_target_task(case_id: str) -> str:
    bv = base_variant_of(case_id)
    if bv == "missing":
        raise ValueError(f"{case_id} has no base task (Stage 0 required)")
    return f"dcv3_{bv}_ref_fk_{case_id}"


# --------------------------------------------------------------------------
# Arm groups -- keyed by object (plan247 3.1)
# --------------------------------------------------------------------------
# Frozen scene basenames.  The builders assert the file they write matches these,
# so a rename upstream is caught rather than silently accepted.
SCENE_BUCKET_PRG = "scene_act_E202_bucketAlignedTop_PRG"          # E202.SCENE_NAME
SCENE_BUCKET_PRG_GRAVCOMP = "scene_act_E210_bucketAlignedTop_PRG_gravcomp"
SCENE_BOX_RUBBER = "scene_act_E199_rubberHull"                    # E199.RUBBER_INTERMEDIATE
SCENE_BOX_PRG = "scene_act_E199_rubberHull_PRG"                   # E199.SCENE_NAME
SCENE_BOX_PRG_GRAVCOMP = "scene_act_E199_rubberHull_PRG_gravcomp"  # E200.GRAVCOMP_SCENE

ARM_GROUP_BY_OBJECT = {
    "bucket003": "bucket_prg",
    "bucket007": "bucket_prg_gravcomp",
    "box021": "box_prg",
    "box023": "box_noprg",
    "box001": "box_prg_g1a2",
    "box004": "box_prg_g1a2",
    "box024": "box_prg_g1a2",
}

# per-group spec.  `object_line` selects the scene builder (E202 bucket proxy vs
# E199 box rubber_hull).  `arm_override` selects how the CEM override is wired:
#   e202_prg / e199_prg  -> write an arm override yaml (defaults [aug_base,_self_]
#                           + the arm's prg_override_payload)
#   base_only            -> no arm yaml; +override = the aug base task yaml, and
#                           extra_overrides forces the scene (E200 noprg pattern,
#                           the negative leg state is verified from config_act).
# `gravcomp` builds the object-gravcomp sidecar on top of the PRG scene.
# `override_token` is the human tag in the override filename.
GROUPS: dict[str, dict[str, Any]] = {
    "bucket_prg": {  # G1 bucket003
        "object_line": "bucket", "scene_name": SCENE_BUCKET_PRG,
        "override_token": "PRG", "arm_override": "e202_prg",
        "gravcomp": False, "extra_scene": None, "a2_gate": False,
    },
    "bucket_prg_gravcomp": {  # G2 bucket007 (E210)
        "object_line": "bucket", "scene_name": SCENE_BUCKET_PRG_GRAVCOMP,
        "override_token": "PRG_gravcomp", "arm_override": "e202_prg",
        "gravcomp": True, "gravcomp_base": SCENE_BUCKET_PRG,
        "extra_scene": SCENE_BUCKET_PRG_GRAVCOMP, "a2_gate": False,
    },
    "box_prg": {  # G3 box021
        "object_line": "box", "scene_name": SCENE_BOX_PRG,
        "override_token": "PRG", "arm_override": "e199_prg",
        "gravcomp": False, "extra_scene": None, "a2_gate": False,
    },
    "box_noprg": {  # G4 box023
        "object_line": "box", "scene_name": SCENE_BOX_RUBBER,
        "override_token": "noPRG", "arm_override": "base_only",
        "gravcomp": False, "extra_scene": SCENE_BOX_RUBBER, "a2_gate": False,
    },
    "box_prg_g1a2": {  # G5 box001/004/024
        "object_line": "box", "scene_name": SCENE_BOX_PRG_GRAVCOMP,
        "override_token": "G1A2", "arm_override": "e199_prg",
        "gravcomp": True, "gravcomp_base": SCENE_BOX_PRG,
        "extra_scene": SCENE_BOX_PRG_GRAVCOMP, "a2_gate": True,
    },
}


def arm_group_of(case_id: str) -> str:
    obj = object_key_of(case_id)
    if obj not in ARM_GROUP_BY_OBJECT:
        raise ValueError(f"no arm group for object {obj!r} (case {case_id})")
    return ARM_GROUP_BY_OBJECT[obj]


def group_spec(case_id: str) -> dict[str, Any]:
    return GROUPS[arm_group_of(case_id)]


def load_e215_cases(*, include_missing: bool = False) -> list[dict[str, str]]:
    """The pinned registry projected to the fields the pipeline needs.

    Sorted (object, case_id).  ``buildable`` is 0 for the two no-base box021
    cases unless Stage 0 has produced their base (checked at build time, not
    here -- this function only reflects the pin).
    """
    cases: list[dict[str, str]] = []
    for case_id in _CASE_IDS:
        bv = base_variant_of(case_id)
        if bv == "missing" and not include_missing:
            continue
        cases.append({
            "case_id": case_id,
            "object_key": object_key_of(case_id),
            "base_variant": bv,
            "arm_group": arm_group_of(case_id),
            "base_target_task": (
                "" if bv == "missing" else base_target_task(case_id)),
        })
    cases.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return cases


def aug_task_name(case_id: str, variant: str) -> str:
    """``dcv3_omnirt_v2_ref_fk_{case}__aug_{variant}`` -- always v2-relabelled.

    The aug directory is uniformly v2 (retarget contract is v2), even when the
    base task pin is v1 (v1 is only the geometry/metadata template authority).
    """
    return f"dcv3_{RETARGET_VARIANT}_ref_fk_{case_id}__aug_{variant}"


# --------------------------------------------------------------------------
# Warm-start seed sources (FLAT trees already on disk)
# --------------------------------------------------------------------------
# bucket -> E202 data_preprocess; box -> E199 data_preprocess.  Both hold
# converted/ + retargeted/trimmed `_original` (+ trans_*) + trim_window.json.
E199_DP = REPO / "workspace/core4d/results/E199/data_preprocess"
E202_DP = REPO / "workspace/core4d/results/E202/data_preprocess"


def seed_source_dir(case_id: str) -> Path:
    obj = object_key_of(case_id)
    root = E202_DP if obj.startswith("bucket") else E199_DP
    return root / f"holosoma_{base_target_task(case_id)}"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RESULTS = REPO / "workspace/core4d/results" / EXP_ID
DP = RESULTS / "data_preprocess"
DP_MANIFESTS = DP / "manifests"
DP_SENTINELS = DP / "sentinels"
DP_CASE_FILES = DP / "case_files"
DP_LOGS = DP / "logs"
PREFLIGHT_DIR = RESULTS / "preflight"
S5_DIR = RESULTS / "s5_handoff"
S6_DIR = RESULTS / "s6_downstream"
MANIFEST_DIR = S6_DIR / "manifests"
EVAL_DIR = S6_DIR / "eval/aug"
SNAPSHOT_DIR = RESULTS / "scene_snapshot"

TASK_ROOT = E199.TASK_ROOT
OVERRIDE_DIR = E199.OVERRIDE_DIR
PIPELINE_SH = REPO / "workspace/core4d/data_preprocess/pipeline.sh"
SNAPSHOT_SH = REPO / "workspace/core4d/scripts/convert/snapshot_scenes.sh"

FEASIBILITY_TSV = DP_MANIFESTS / "e215_rot_feasibility.tsv"
ARTIFACTS_TSV = DP_MANIFESTS / "e215_rot_artifacts.tsv"
SEED_REPORT_JSON = PREFLIGHT_DIR / "seed_report.json"
BASELINE_AUDIT_JSON = PREFLIGHT_DIR / "baseline_audit.json"
OVERRIDE_AUDIT_JSON = S5_DIR / "aug_override_audit.json"

PRIORITY_MANIFEST = MANIFEST_DIR / "e215_priority_manifest.tsv"
FROZEN_MANIFEST = MANIFEST_DIR / "e215_priority_manifest.frozen.tsv"
FREEZE_JSON = MANIFEST_DIR / "freeze.json"
AUTHORITY_TSV = MANIFEST_DIR / "e215_aug_authority.tsv"

CEM_LOG_DIR = REPO / "logs/E215/cem"


def data_root() -> Path:
    """Single RESULT_ROOT: E215 retargets everything under omnirt_v2, so unlike
    E208 (which split v1/v2) there is no name collision -- base_target_task
    already carries the v1/v2 prefix, and only rot_* npz are produced here."""
    return DP


def holosoma_dir(case_id: str) -> Path:
    return data_root() / f"holosoma_{base_target_task(case_id)}"


def sentinel_path(case_id: str) -> Path:
    return DP_SENTINELS / f"{base_target_task(case_id)}.done"


def cem_out_dir(case_id: str, variant: str, stage: str = "full") -> Path:
    return S6_DIR / "cem" / stage / aug_variant_id(case_id, variant)


def render_path(case_id: str, variant: str, stage: str = "full") -> Path:
    return S6_DIR / "render" / stage / f"{aug_variant_id(case_id, variant)}_{stage}.mp4"


def override_name(case_id: str, variant: str) -> str:
    """Arm override filename stem (G1/G2/G3/G5 only; G4 uses the base yaml)."""
    token = group_spec(case_id)["override_token"]
    return f"core4d_{EXP_ID}_{case_id}_aug_{variant}_{token}"


def aug_variant_id(case_id: str, variant: str) -> str:
    token = group_spec(case_id)["override_token"]
    return E199.safe_id(f"{EXP_ID}_{case_id}_aug_{variant}_{token}")


# --------------------------------------------------------------------------
# Frozen CEM budget (plan247 2)
# --------------------------------------------------------------------------
CEM_SEED = 0
CEM_NUM_SAMPLES = 1024
CEM_MAX_ITERATIONS = 32
CEM_USE_TORCH_COMPILE = False
GPU_DEFAULT = "0,1,2,3,4,5,6,7"
PER_GPU_MEM_MIB = 5000
PER_TASK_TIMEOUT_MIN = 180
TIER_RANK = {"P0": 0, "P1": 1, "P2": 2}


def frozen_budget() -> dict[str, Any]:
    return {
        "num_samples": CEM_NUM_SAMPLES,
        "max_num_iterations": CEM_MAX_ITERATIONS,
        "use_torch_compile": CEM_USE_TORCH_COMPILE,
        "seed": CEM_SEED,
    }


# --------------------------------------------------------------------------
# Manifest schema (E199 dialect + E215 provenance)
# --------------------------------------------------------------------------
EXTRA_FIELDS = [
    "arm_group", "base_variant", "effective_retarget_variant", "gravcomp",
    "object_geom_count", "compiled_robot_object_pair_count",
    "approach_yaw_deg_max", "endpoint_yaw_deg", "degraded_yaw",
    "approach_trans_offset_m_max", "orig_result_npz", "wall_min",
]
FIELDS = list(E199.FIELDS) + [f for f in EXTRA_FIELDS if f not in E199.FIELDS]

FEASIBILITY_FIELDS = [
    "case_id", "object_key", "arm_group", "base_target_task", "variant",
    "holosoma_variant", "npz_present", "npz_sha256", "log_hit", "rotation_bug",
    "state", "built", "wall_s", "log_ref", "updated_at",
]

# States for the upstream rot feasibility scan.
BUILT_STATE = "rot_ok"
STATES = ("rot_ok", "rot_infeasible", "upstream_rotation_bug", "needs_triage")


# --------------------------------------------------------------------------
# Single-instance lock (verbatim discipline from E208 -- both drivers race on
# the shared per-object files pipeline.sh / the manifest rewrites).
# --------------------------------------------------------------------------
class SingleInstance:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "SingleInstance":
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._handle.seek(0)
            holder = self._handle.read().strip() or "<unknown>"
            self._handle.close()
            raise SystemExit(
                f"another {Path(sys.argv[0]).name} is already running ({holder}).\n"
                f"lock: {self.path}\nWait for it or stop it first."
            ) from None
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} started={now()} argv={' '.join(sys.argv[1:])}\n")
        self._handle.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            self._handle.close()


def sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# IO helpers -- single authority, re-exported from E199
# --------------------------------------------------------------------------
now = E199.now
repo_path = E199.repo_path
rel = E199.rel
sha256 = E199.sha256
safe_id = E199.safe_id
read_tsv = E199.read_tsv
read_with_fields = E199.read_with_fields
write_tsv = E199.write_tsv          # carries the JuiceFS EIO retry
write_json = E199.write_json
truth = E199.truth
load_case_meta = E199.load_case_meta

SPIDER_PYTHON_BIN = E199.SPIDER_PYTHON_BIN
RETARGET_PYTHON_BIN = E199.RETARGET_PYTHON_BIN
HOLOSOMA_REPO = E199.HOLOSOMA_REPO
CORE4D_RAW_ROOT = E199.CORE4D_RAW_ROOT
SMPLX_MODEL_DIR = E199.SMPLX_MODEL_DIR
