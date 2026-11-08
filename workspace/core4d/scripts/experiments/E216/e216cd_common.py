#!/usr/bin/env python3
"""E216 chair/desk (Phase 2) contract: per-case selected-gravcomp arm, trans 0.1m.

Phase 1 (box+bucket) shipped as e216_common.py.  chair/desk diverge enough to
warrant their own module:
  * retarget variant = **omnirt_v1** (E208/E213 used v1-primary for desk/chair;
    box/bucket were v2 -- so E216 is mixed-per-family, matching downstream).
  * CEM scene = **scene_act_E206_lowgeom_PRG** (hand-placed low-geom boxes, via
    E206 ``build_arm_scenes.build_one``) + a per-case **fractional** object
    gravcomp sidecar (G1=1.0 / G0.8 / G0.6 / G0.4), the arm E213 selected per case.
  * seed ``_original`` comes from E208's split v1/v2 data_preprocess tree.

Everything else (IO, ARM builder, scene names, pipeline env plumbing) is imported
from e208_common; e216cd re-types nothing.  Aug task dirs carry ``_0p1m``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = SCRIPT_DIR.parent
for _p in (EXPERIMENTS / "E199", EXPERIMENTS / "E206", EXPERIMENTS / "E208", EXPERIMENTS / "E212", EXPERIMENTS / "E213"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import e208_common as E8  # noqa: E402  (env + ARM builder + IO + scene names)

REPO = E8.REPO

# --------------------------------------------------------------------------
# Identity / contract
# --------------------------------------------------------------------------
EXP_ID = "E216CD"
RUN_ID = "R304"
RETARGET_VARIANT = "omnirt_v1"              # desk/chair primary (E208/E213); per-case below
OMNIRT_TRANS_OFFSET_M = "0.1"
OMNIRT_ENV = dict(E8.OMNIRT_V1_ENV)         # default (v1); use omnirt_env(case_id) per case
OMNIRT_ENV_BY_VARIANT = dict(E8.OMNIRT_ENV_BY_VARIANT)
PIPELINE_DATASET_ENV = dict(E8.PIPELINE_DATASET_ENV)


def omnirt_env(case_id: str) -> dict:
    """Per-case retarget env: a v2-base case (desk021_014_p2) must run under v2,
    the rest under v1 -- matching the variant E208/E213 shipped for that case."""
    return dict(OMNIRT_ENV_BY_VARIANT[base_variant_of(case_id)])

TRANS_VARIANTS: list[tuple[str, str]] = [("trans0", "trans_0"), ("trans1", "trans_1"), ("trans2", "trans_2")]
BUILD_VARIANTS = list(TRANS_VARIANTS)
NOMINAL_TRANS_OFFSET_M = 0.1
WEAK_TRANS_FLOOR_M = 0.05

_TRANS_OFFSET = {"trans0": f"{OMNIRT_TRANS_OFFSET_M},0,0",
                 "trans1": f"0,{OMNIRT_TRANS_OFFSET_M},0",
                 "trans2": f"0,-{OMNIRT_TRANS_OFFSET_M},0"}


def aug_translation(short: str) -> str:
    return _TRANS_OFFSET[short]


def aug_rotation_rad(short: str) -> str:  # noqa: ARG001
    return "0.0"


SCENE_NAME = E8.SCENE_NAME                  # scene_act_E206_lowgeom_PRG
SCENE_GRAVCOMP = "scene_act_E209_lowgeom_PRG_gravcomp"  # per-case fractional sidecar

# --------------------------------------------------------------------------
# Registry (pinned TSV generated from E213 e213_source_arm_cem selected arm)
# --------------------------------------------------------------------------
REGISTRY_TSV = SCRIPT_DIR / "registry_chair_desk.tsv"


def _load_registry() -> list[dict[str, str]]:
    rows = E8.read_tsv(REGISTRY_TSV)
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


def gravcomp_value_of(case_id: str) -> str:
    """Per-case fractional object gravcomp string (verbatim, '1'/'0.8'/'0.6'/'0.4')."""
    v = (_BY_CASE[case_id].get("gravcomp", "") or "").strip()
    # registry stores floats like 1.0 / 0.8; normalize to the scene string form
    if v in ("1", "1.0"):
        return "1"
    return str(float(v)) if v else ""


def load_cases(*, objects: str = "") -> list[dict[str, str]]:
    keep = {o.strip() for o in objects.split(",") if o.strip()} if objects else None
    out: list[dict[str, str]] = []
    for r in _REGISTRY:
        if keep and r["object_key"] not in keep:
            continue
        out.append({
            "case_id": r["case_id"], "object_key": r["object_key"],
            "base_variant": r["base_variant"], "base_target_task": r["base_target_task"],
            "arm_group": r["arm_group"], "gravcomp": gravcomp_value_of(r["case_id"]),
        })
    out.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return out


def aug_task_name(case_id: str, variant: str) -> str:
    """E208-style (no v1->v2 rewrite) + ``_0p1m`` suffix."""
    base = base_target_task(case_id)
    return f"{base}__aug_{variant}_0p1m"


# --------------------------------------------------------------------------
# Warm-start seed: E208's split tree  data_preprocess/omnirt_v{bv}/holosoma_{base}
# --------------------------------------------------------------------------
E208_DP = REPO / "workspace/core4d/results/E208/data_preprocess"


def seed_source_dir(case_id: str) -> Path:
    bv = base_variant_of(case_id)
    return E208_DP / bv / f"holosoma_{base_target_task(case_id)}"


# --------------------------------------------------------------------------
# Paths (own tree under results/E216/chair_desk)
# --------------------------------------------------------------------------
RESULTS = REPO / "workspace/core4d/results/E216/chair_desk"
DP = RESULTS / "data_preprocess"
DP_MANIFESTS = DP / "manifests"
DP_SENTINELS = DP / "sentinels"
DP_CASE_FILES = DP / "case_files"
DP_LOGS = DP / "logs"
PREFLIGHT_DIR = RESULTS / "preflight"
S6_DIR = RESULTS / "s6_downstream"
MANIFEST_DIR = S6_DIR / "manifests"
EVAL_DIR = S6_DIR / "eval/aug"
SNAPSHOT_DIR = RESULTS / "scene_snapshot"

TASK_ROOT = E8.TASK_ROOT
OVERRIDE_DIR = E8.OVERRIDE_DIR
PIPELINE_SH = REPO / "workspace/core4d/data_preprocess/pipeline.sh"

FEASIBILITY_TSV = DP_MANIFESTS / "e216cd_trans_feasibility.tsv"
ARTIFACTS_TSV = DP_MANIFESTS / "e216cd_trans_artifacts.tsv"
PRIORITY_MANIFEST = MANIFEST_DIR / "e216cd_priority_manifest.tsv"
FROZEN_MANIFEST = MANIFEST_DIR / "e216cd_priority_manifest.frozen.tsv"
FREEZE_JSON = MANIFEST_DIR / "freeze.json"
AUTHORITY_TSV = MANIFEST_DIR / "e216cd_aug_authority.tsv"
OVERRIDE_AUDIT_JSON = RESULTS / "s5_handoff/aug_override_audit.json"


def data_root() -> Path:
    return DP


def holosoma_dir(case_id: str) -> Path:
    return DP / f"holosoma_{base_target_task(case_id)}"


def sentinel_path(case_id: str) -> Path:
    return DP_SENTINELS / f"{base_target_task(case_id)}.done"


def aug_variant_id(case_id: str, variant: str) -> str:
    return E8.safe_id(f"{EXP_ID}_{case_id}_aug_{variant}_0p1m_sel")


def override_name(case_id: str, variant: str) -> str:
    return f"core4d_{EXP_ID}_{case_id}_aug_{variant}_0p1m_sel"


# --------------------------------------------------------------------------
# Frozen CEM budget (same 1024x32 seed0)
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
    return {"num_samples": CEM_NUM_SAMPLES, "max_num_iterations": CEM_MAX_ITERATIONS,
            "use_torch_compile": CEM_USE_TORCH_COMPILE, "seed": CEM_SEED}


FIELDS = list(E8.E199.FIELDS) + [f for f in (
    "arm_group", "base_variant", "effective_retarget_variant", "gravcomp",
    "object_geom_count", "compiled_robot_object_pair_count",
    "approach_trans_offset_m_max", "endpoint_trans_offset_m", "weak_trans",
    "orig_result_npz", "wall_min") if f not in E8.E199.FIELDS]

FEASIBILITY_FIELDS = ["case_id", "object_key", "arm_group", "base_target_task", "variant",
                      "holosoma_variant", "npz_present", "npz_sha256", "log_hit",
                      "state", "built", "wall_s", "log_ref", "updated_at"]
BUILT_STATE = "trans_ok"

# --------------------------------------------------------------------------
# Re-exports from e208_common / e199_common
# --------------------------------------------------------------------------
SingleInstance = E8.SingleInstance
sha256_text = E8.sha256_text
now = E8.now
rel = E8.rel
repo_path = E8.repo_path
sha256 = E8.sha256
safe_id = E8.safe_id
read_tsv = E8.read_tsv
read_with_fields = E8.E199.read_with_fields
write_tsv = E8.write_tsv
write_json = E8.write_json
load_case_meta = E8.load_case_meta
ARM = E8.ARM
SPIDER_PYTHON_BIN = E8.SPIDER_PYTHON_BIN
RETARGET_PYTHON_BIN = E8.E199.RETARGET_PYTHON_BIN
HOLOSOMA_REPO = E8.HOLOSOMA_REPO
CORE4D_RAW_ROOT = E8.E199.CORE4D_RAW_ROOT
SMPLX_MODEL_DIR = E8.E199.SMPLX_MODEL_DIR
_drop_broken_torch = E8._drop_broken_torch
