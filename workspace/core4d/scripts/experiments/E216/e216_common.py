#!/usr/bin/env python3
"""E216 shared contract: per-object-arm TRANSLATION object-augmentation at 0.1 m.

E216 re-runs the translation augmentation (trans_0/1/2 = fwd/left/right) at a
GENTLER **0.1 m** offset (vs the historical 0.2 m of E199/E202/E213), under each
object's downstream-correct CEM arm, over the FULL upstream case set for that arm
(larger than the downstream pre-filtered set).  One aug task per (case, trans
variant), then a formal full CEM.

Why 0.1 m
---------
The downstream RL pre-filters our 0.2 m augmented data (14-gate funnel + Refiner/
Tracker double-gate), keeping only a subset.  A smaller approach offset perturbs
the object less -> stays in the reach envelope, gentler leg-object clearance,
pose closer to orig -> expected higher feasibility and higher 14-gate pass rate,
so more survives the downstream pre-filter.

Per-object arm (verified against downstream provenance + e215_common):
  * box021               E199 PRG (rubber_hull, 16-pair)                 [box_prg]
  * box023               E167A noPRG (rubber_hull hand, no leg pairs)    [box_noprg]
  * box001/004/024       E199 PRG + gravcomp=1 + A2 hand gate (E200)     [box_prg_g1a2]
  * bucket003/007        E202 PRG (bucketAlignedTop 5-seg proxy)         [bucket_prg]
  * chair006/desk*       E213 per-case selected gravcomp arm (Phase 2)   [chair_desk_*]

Phase 1 (this file's active registry) = box + bucket = 104 cases.  chair/desk
(20 cases, per-case fractional gravcomp) land in Phase 2 via a registry append.

Critical: aug task dirs carry a ``_0p1m`` suffix so they never overwrite the
0.2 m E199/E202 tasks (which E216 needs as the 0.2 m baseline for C4).

Numeric contracts (reward, gate, CEM budget, proxy geometry) are imported from
the arm commons (E199/E202/E200); E216 re-types nothing.
"""

from __future__ import annotations

import importlib.util
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

    Lifted from e215_common: geometry imports pull the stub into sys.modules and
    poison scipy's ``Rotation.from_quat``.
    """
    mod = sys.modules.get("torch")
    if mod is not None and not hasattr(mod, "Tensor"):
        del sys.modules["torch"]


_drop_broken_torch()


def load_e216_module(name: str):
    """Import an E216-local sibling by explicit path (avoids E199/E202 shadowing)."""
    path = SCRIPT_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no such E216 module: {path}")
    spec = importlib.util.spec_from_file_location(f"e216_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E216"
PLAN_SLOT = 249
PLAN_REF = f"workspace/core4d/plan/{PLAN_SLOT}_E216_trans_0p1m_perobject_arm_plan.md"
RUN_ID = "R304"

# --------------------------------------------------------------------------
# Retarget contract -- omnirt_v2 + gentler 0.1 m trans offset
# --------------------------------------------------------------------------
RETARGET_VARIANT = "omnirt_v2"
OMNIRT_V2_ENV = dict(E199.OMNIRT_V2_ENV)
PIPELINE_DATASET_ENV = {"SPIDER_DATASET": "core4d", "SPIDER_SOURCE_DATASET": "core4d"}

# The one knob that distinguishes E216 from E199/E202.  Consumed by the patched
# holosoma generate_augmentation_configs (OMNIRT_TRANS_OFFSET_M, commit 3fa6d90).
OMNIRT_TRANS_OFFSET_M = "0.1"
NOMINAL_TRANS_OFFSET_M = 0.1
# A very late trim_start can decay the approach offset below nominal before
# SPIDER's window opens.  Flag (do not hard-gate) a near-zero "translation".
WEAK_TRANS_FLOOR_M = 0.05

# --------------------------------------------------------------------------
# Variants -- translation only, 0.1 m
# --------------------------------------------------------------------------
TRANS_VARIANTS: list[tuple[str, str]] = [
    ("trans0", "trans_0"), ("trans1", "trans_1"), ("trans2", "trans_2"),
]
BUILD_VARIANTS = list(TRANS_VARIANTS)

_TRANS_OFFSET = {
    "trans0": f"{OMNIRT_TRANS_OFFSET_M},0,0",
    "trans1": f"0,{OMNIRT_TRANS_OFFSET_M},0",
    "trans2": f"0,-{OMNIRT_TRANS_OFFSET_M},0",
}


def aug_translation(short: str) -> str:
    return _TRANS_OFFSET[short]


def aug_rotation_rad(short: str) -> str:  # noqa: ARG001 - trans-only, always 0
    return "0.0"


# --------------------------------------------------------------------------
# Case registry -- pinned TSV (generated from E199/E202 authority manifests).
# Phase 1 = registry_box_bucket.tsv (104).  Phase 2 appends chair/desk.
# --------------------------------------------------------------------------
REGISTRY_TSV = SCRIPT_DIR / "registry_box_bucket.tsv"
REGISTRY_FIELDS = ["case_id", "object_key", "base_variant", "base_target_task",
                   "arm_group", "gravcomp"]


def _load_registry() -> list[dict[str, str]]:
    rows = E199.read_tsv(REGISTRY_TSV)
    rows.sort(key=lambda r: (r["object_key"], r["case_id"]))
    return rows


_REGISTRY = _load_registry()
_BY_CASE = {r["case_id"]: r for r in _REGISTRY}

EXPECTED_CASES = len(_REGISTRY)
EXPECTED_CASES_BY_OBJECT: dict[str, int] = {}
for _r in _REGISTRY:
    EXPECTED_CASES_BY_OBJECT[_r["object_key"]] = EXPECTED_CASES_BY_OBJECT.get(_r["object_key"], 0) + 1


def object_key_of(case_id: str) -> str:
    return case_id.split("_", 1)[0]


def base_variant_of(case_id: str) -> str:
    return _BY_CASE[case_id]["base_variant"]


def base_target_task(case_id: str) -> str:
    return _BY_CASE[case_id]["base_target_task"]


def gravcomp_value_of(case_id: str) -> float:
    """Per-case object gravcomp (chair/desk Phase 2 uses 0.4/0.6/0.8/1.0).

    For box/bucket the arm group's own ``gravcomp`` flag governs; this column is
    0 for them and only consulted by fractional-gravcomp groups.
    """
    return float(_BY_CASE[case_id].get("gravcomp", "0") or "0")


# --------------------------------------------------------------------------
# Arm groups -- keyed by object (bucket007 = plain PRG per plan249 Q2, NOT E210
# gravcomp).  box groups verbatim from E215.
# --------------------------------------------------------------------------
SCENE_BUCKET_PRG = "scene_act_E202_bucketAlignedTop_PRG"
SCENE_BOX_RUBBER = "scene_act_E199_rubberHull"
SCENE_BOX_PRG = "scene_act_E199_rubberHull_PRG"
SCENE_BOX_PRG_GRAVCOMP = "scene_act_E199_rubberHull_PRG_gravcomp"

GROUPS: dict[str, dict[str, Any]] = {
    "bucket_prg": {  # bucket003 / bucket007
        "object_line": "bucket", "scene_name": SCENE_BUCKET_PRG,
        "override_token": "PRG", "arm_override": "e202_prg",
        "gravcomp": False, "extra_scene": None, "a2_gate": False,
    },
    "box_prg": {  # box021
        "object_line": "box", "scene_name": SCENE_BOX_PRG,
        "override_token": "PRG", "arm_override": "e199_prg",
        "gravcomp": False, "extra_scene": None, "a2_gate": False,
    },
    "box_noprg": {  # box023
        "object_line": "box", "scene_name": SCENE_BOX_RUBBER,
        "override_token": "noPRG", "arm_override": "base_only",
        "gravcomp": False, "extra_scene": SCENE_BOX_RUBBER, "a2_gate": False,
    },
    "box_prg_g1a2": {  # box001/004/024
        "object_line": "box", "scene_name": SCENE_BOX_PRG_GRAVCOMP,
        "override_token": "G1A2", "arm_override": "e199_prg",
        "gravcomp": True, "gravcomp_base": SCENE_BOX_PRG,
        "extra_scene": SCENE_BOX_PRG_GRAVCOMP, "a2_gate": True,
    },
}


def arm_group_of(case_id: str) -> str:
    return _BY_CASE[case_id]["arm_group"]


def group_spec(case_id: str) -> dict[str, Any]:
    return GROUPS[arm_group_of(case_id)]


def load_e216_cases(*, objects: str = "") -> list[dict[str, str]]:
    """The pinned registry projected to the fields the pipeline needs."""
    keep = {o.strip() for o in objects.split(",") if o.strip()} if objects else None
    cases: list[dict[str, str]] = []
    for r in _REGISTRY:
        if keep and r["object_key"] not in keep:
            continue
        cases.append({
            "case_id": r["case_id"], "object_key": r["object_key"],
            "base_variant": r["base_variant"], "arm_group": r["arm_group"],
            "base_target_task": r["base_target_task"],
        })
    cases.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return cases


def aug_task_name(case_id: str, variant: str) -> str:
    """``dcv3_omnirt_v2_ref_fk_{case}__aug_{variant}_0p1m`` -- 0p1m suffix keeps
    the 0.2 m E199/E202 tasks of the same (case, variant) intact."""
    return f"dcv3_{RETARGET_VARIANT}_ref_fk_{case_id}__aug_{variant}_0p1m"


# --------------------------------------------------------------------------
# Warm-start seed sources (FLAT trees already on disk) -- box E199, bucket E202.
# E216 seeds ONLY `_original` (+converted+trim_window); trans_* are NOT seeded so
# upstream recomputes them at the 0.1 m offset.
# --------------------------------------------------------------------------
E199_DP = REPO / "workspace/core4d/results/E199/data_preprocess"
E202_DP = REPO / "workspace/core4d/results/E202/data_preprocess"


def seed_source_dir(case_id: str) -> Path:
    obj = object_key_of(case_id)
    root = E202_DP if obj.startswith("bucket") else E199_DP
    return root / f"holosoma_{base_target_task(case_id)}"


# --------------------------------------------------------------------------
# Paths (all under results/E216)
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

FEASIBILITY_TSV = DP_MANIFESTS / "e216_trans_feasibility.tsv"
ARTIFACTS_TSV = DP_MANIFESTS / "e216_trans_artifacts.tsv"
SEED_REPORT_JSON = PREFLIGHT_DIR / "seed_report.json"
BASELINE_AUDIT_JSON = PREFLIGHT_DIR / "baseline_audit.json"
OVERRIDE_AUDIT_JSON = S5_DIR / "aug_override_audit.json"

PRIORITY_MANIFEST = MANIFEST_DIR / "e216_priority_manifest.tsv"
FROZEN_MANIFEST = MANIFEST_DIR / "e216_priority_manifest.frozen.tsv"
FREEZE_JSON = MANIFEST_DIR / "freeze.json"
AUTHORITY_TSV = MANIFEST_DIR / "e216_aug_authority.tsv"

CEM_LOG_DIR = REPO / "logs/E216/cem"


def data_root() -> Path:
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
    token = group_spec(case_id)["override_token"]
    return f"core4d_{EXP_ID}_{case_id}_aug_{variant}_0p1m_{token}"


def aug_variant_id(case_id: str, variant: str) -> str:
    token = group_spec(case_id)["override_token"]
    return E199.safe_id(f"{EXP_ID}_{case_id}_aug_{variant}_0p1m_{token}")


# --------------------------------------------------------------------------
# Frozen CEM budget (same as E199/E202/E215)
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
# Manifest schema (E199 dialect + E216 provenance)
# --------------------------------------------------------------------------
EXTRA_FIELDS = [
    "arm_group", "base_variant", "effective_retarget_variant", "gravcomp",
    "object_geom_count", "compiled_robot_object_pair_count",
    "approach_trans_offset_m_max", "endpoint_trans_offset_m", "weak_trans",
    "orig_result_npz", "wall_min",
]
FIELDS = list(E199.FIELDS) + [f for f in EXTRA_FIELDS if f not in E199.FIELDS]

FEASIBILITY_FIELDS = [
    "case_id", "object_key", "arm_group", "base_target_task", "variant",
    "holosoma_variant", "npz_present", "npz_sha256", "log_hit",
    "state", "built", "wall_s", "log_ref", "updated_at",
]

BUILT_STATE = "trans_ok"
STATES = ("trans_ok", "trans_infeasible", "needs_triage")


# --------------------------------------------------------------------------
# Single-instance lock (verbatim from E215)
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
write_tsv = E199.write_tsv
write_json = E199.write_json
truth = E199.truth
load_case_meta = E199.load_case_meta

SPIDER_PYTHON_BIN = E199.SPIDER_PYTHON_BIN
RETARGET_PYTHON_BIN = E199.RETARGET_PYTHON_BIN
HOLOSOMA_REPO = E199.HOLOSOMA_REPO
CORE4D_RAW_ROOT = E199.CORE4D_RAW_ROOT
SMPLX_MODEL_DIR = E199.SMPLX_MODEL_DIR
