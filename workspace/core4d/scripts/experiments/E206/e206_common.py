#!/usr/bin/env python3
"""E206 shared contract: scope, paths, arm definitions, frozen constants.

E206 = desk+chair through the full dcv3 pipeline with a <=N_MAX box lowgeom
collision proxy, run under two reward arms (noPRG / PRG).

Single source of truth for everything the E206 scripts agree on.  Constants that
already exist elsewhere are IMPORTED, never re-typed (rule: one authority per
constant).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[5]
WS = REPO / "workspace/core4d"
DCV3 = WS / "scripts/data_construction_v3"
EXPERIMENTS = WS / "scripts/experiments"

for _p in (
    DCV3 / "stages/s2_templates",
    EXPERIMENTS / "E175",
    EXPERIMENTS / "E176",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E206"
PLAN_REF = "workspace/core4d/plan/236_E206_desk_chair_move2_dcv3_noprg_prg_plan.md"
RUN_IDS = {"noprg": "R291", "prg": "R292"}
SPIDER_DATASET = "core4d"  # v1, NOT core4d_v2 (locked by user)


# --------------------------------------------------------------------------
# Scope (locked by user 2026-09-02)
# --------------------------------------------------------------------------
DESK_KEYS = ("desk005", "desk007", "desk020", "desk021", "desk023")
CHAIR_KEYS = ("chair005", "chair006", "chair020", "chair021", "chair022")
OBJECT_KEYS = DESK_KEYS + CHAIR_KEYS

# Dropped by the user after the 2026-09-03 hand review: chair021's geometry does
# not admit a proxy worth running.  It was the only object still carrying a
# hand-deleted voxel draft rather than a re-placed box set, and had already been
# queued last for CEM on that basis.  Its 9 S1 cases leave E206 with it
# (74 -> 65 cases, 9 -> 8 objects).  Kept inside OBJECT_KEYS so the exclusion is
# auditable rather than silent, exactly like SIZE_GATE_EXCLUDED_KEYS below.
DROPPED_OBJECT_KEYS = ("chair021",)

# desk001 is hard-rejected by the S1 AABB size gate
# (reject_too_large_box025_or_larger); kept here only so the exclusion is
# auditable rather than silent.
SIZE_GATE_EXCLUDED_KEYS = ("desk001",)

# board/stick are out of scope for E206: terminally rejected at the S1 size gate
# (stick all `too_small`; board020 is a 2.7cm plank misread as `too_large`).
# Deferred to a later experiment that must first fix the gate.
DEFERRED_CATEGORIES = ("board", "stick")

# "double-person collaborative carrying" == all three move2 labels.  obs1/obs3
# only mean an obstacle is present in the room; the task is the same.  This
# matches the bucket line (E174 -> E178 -> E202 -> E204/E205).
MOVE2_ACTIONS = ("move2_obs0", "move2_obs1", "move2_obs3")

# Contact-mask label.  Settled empirically in P1, not by assumption:
# S1 measured 3cm -> 74 cases vs 5cm -> 76 (same 9 objects; only desk007 +1 and
# desk020 +1).  Switching to 5cm would have bought 2 cases while breaking the
# ruler shared with the whole box/bucket line (E173/E174/E178/E199/E202/E203/
# E204/E205) and forcing a separate 3cm bridge run for the C5a-vs-E174
# comparison.  User reverted to 3cm on that evidence -> single label, no bridge.
PRIMARY_CONTACT_LABEL = "3cm"
CONTACT_THRESHOLDS_M = "0.03,0.05"  # still score both; 5cm is reported, not used

# S1 finding (plan236 log-worthy, NOT acted on in E206): the real attrition is
# the `object_rotation >= 45deg` motion gate, which rejects 56/164 candidates
# (34%); `object_lift <= 0.30m` only rejects 8.  Two-person desk/chair carrying
# routinely pivots the object.  The gate is inherited unchanged so E206 stays
# comparable to the bucket line; relaxing it is a follow-up experiment.
S1_ROTATION_GATE_REJECTS = 56
S1_LIFT_GATE_REJECTS = 8

# Minimum landed case count below which the experiment stops and escalates
# rather than silently relaxing (plan236 R1).
MIN_CASES_ESCALATE = 12


# --------------------------------------------------------------------------
# Collision proxy budget
# --------------------------------------------------------------------------
# E176 froze MAX_OBJECT_GEOMS=9 under a throughput gate its own log235 §4.3
# says was not geom-count-bound.  User authorised 16; P3 measures whether it is
# affordable and may fall back to 9 via the A3 ladder.
N_MAX_TARGET = 16
N_MAX_FALLBACK = 9

# Robot side is fixed: 2 hands + 16 lower-body geoms (E170 PRG set).
# Imported from the E175 authority, not re-typed.
from build_nonbox_multigeom_production import (  # noqa: E402
    HAND_GEOMS,
    LOWER_BODY_GEOMS,
    ROBOT_OBJECT_GEOMS,
)

N_HAND_GEOMS = len(HAND_GEOMS)          # 2
N_LEG_GEOMS = len(LOWER_BODY_GEOMS)     # 16
N_ROBOT_GEOMS = len(ROBOT_OBJECT_GEOMS)  # 18


def expected_pair_counts(n_object_geoms: int) -> dict[str, int]:
    """Pair counts each arm must compile to, given N object collision geoms."""
    return {
        "noprg": N_HAND_GEOMS * n_object_geoms,
        "prg": N_ROBOT_GEOMS * n_object_geoms,
        "leg_only": N_LEG_GEOMS * n_object_geoms,
    }


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
ARMS = ("noprg", "prg")

SCENE_RUBBER_HULL = "scene_act_E206_lowgeom_rubberHull"
SCENE_BY_ARM = {
    "noprg": "scene_act_E206_lowgeom_noPRG",
    "prg": "scene_act_E206_lowgeom_PRG",
}

BASE_REWARD_METHOD = "E167A_zOnlyBody"
SOURCE_CONFIG_ID = "E170_PRG_lowerbodyPhysics_softPenalty_candidateGate"
E206_METHOD_ID = "E206_lowgeom16_deskChair_move2_r1"

# The five keys — and ONLY these five — may differ between the two composed
# configs.  Anything else is a confound (plan236 C4, zero tolerance).
ARM_DIFF_KEYS = frozenset(
    {
        "scene_name",
        "leg_object_penalty_scale",
        "leg_object_penalty_geom_names",
        "cem_leg_gate_enabled",
        "cem_leg_gate_geom_names",
    }
)

# noPRG == E167A: leg triple off, base reward E167A_zOnlyBody, no G1, no A2.
NOPRG_OVERRIDES: dict[str, Any] = {
    "scene_name": SCENE_BY_ARM["noprg"],
    "object_collision_sdf_mode": "union",  # NOT inherited, must be re-declared
    "object_collision_sdf_batch_groups": True,
    "leg_object_penalty_scale": 0,
    "leg_object_penalty_geom_names": [],
    "cem_leg_gate_enabled": False,
}

PRG_OVERRIDES: dict[str, Any] = {
    "scene_name": SCENE_BY_ARM["prg"],
    "object_collision_sdf_mode": "union",
    "object_collision_sdf_batch_groups": True,
    "leg_object_penalty_scale": 2.0,
    "leg_object_penalty_margin_m": 0.02,
    "leg_object_penalty_geom_names": list(LOWER_BODY_GEOMS),
    "leg_object_penalty_gate_source": "always",
    "cem_leg_gate_enabled": True,
    "cem_leg_gate_geom_names": list(LOWER_BODY_GEOMS),
    "cem_leg_gate_min_sdf_m": 0.005,
    "cem_leg_gate_max_violation_pct": 0.02,
    "cem_leg_gate_hard_floor_m": -0.005,
    "cem_leg_gate_min_valid_frac": 0.02,
    "cem_leg_gate_fallback": "least_violation",
}

OVERRIDES_BY_ARM = {"noprg": NOPRG_OVERRIDES, "prg": PRG_OVERRIDES}

# E163 default hand gate — BOTH arms must keep this (no A2 in E206).
E163_HAND_GATE = {
    "cem_hand_gate_enabled": True,
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.10,
    "cem_hand_gate_hard_floor_m": -0.020,
}


# --------------------------------------------------------------------------
# Frozen CEM budget (same as the whole box/bucket line)
# --------------------------------------------------------------------------
CEM_NUM_SAMPLES = 1024
CEM_MAX_ITERATIONS = 32
CEM_SEED = 0


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RESULTS = WS / "results" / EXP_ID
PRE_RESULTS = WS / "results" / f"{EXP_ID}_pre"
PROCESSED_ROOT = (
    REPO / "example_datasets/processed" / SPIDER_DATASET / "unitree_g1/humanoid_object"
)
OBJECT_MESH_ROOT = WS / "object_models/object_models"
OVERRIDE_DIR = REPO / "examples/config/override"

S0_DIR = RESULTS / "s0_environment"
S1_DIR = RESULTS / "s1_raw_contact"
S2_PROXY_DIR = RESULTS / "s2_proxy"
S2_TEMPLATE_DIR = RESULTS / "s2_templates"
S3_DIR = RESULTS / "s3_retarget"
S3_BRIDGE_DIR = RESULTS / "s3_retarget_bridge3cm"
S4_DIR = RESULTS / "s4_gate_visual_qc"
S5_DIR = RESULTS / "s5_handoff"
S6_DIR = RESULTS / "s6_downstream"
REGISTRY_DIR = RESULTS / "registries"
PREFLIGHT_DIR = RESULTS / "preflight"

ADMISSION_JSON = S6_DIR / "cem/throughput/admission_decision.json"


def object_mesh_path(object_key: str) -> Path:
    """`chair021` -> workspace/core4d/object_models/object_models/chair/chair021_m.obj"""
    category = object_key.rstrip("0123456789")
    return OBJECT_MESH_ROOT / category / f"{object_key}_m.obj"


def object_category(object_key: str) -> str:
    return object_key.rstrip("0123456789")


def arm_out_dir(arm: str, case_id: str, stage: str = "full") -> Path:
    """Per-arm isolated CEM output dir — never let two arms share one."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    return S6_DIR / "cem" / stage / f"{EXP_ID}_{case_id}_{arm}"


def override_name(case_id: str, arm: str) -> str:
    suffix = {"noprg": "noPRG", "prg": "PRG"}[arm]
    return f"core4d_{EXP_ID}_{case_id}_lowgeom_{suffix}"


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
ENV_DEFAULTS = {
    "CORE4D_RAW_ROOT": "/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real",
    "SMPLX_MODEL_DIR": "/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files",
    "HOLOSOMA_DEPS_DIR": "/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps",
}


def env_path(name: str) -> Path:
    value = os.environ.get(name) or ENV_DEFAULTS.get(name)
    if not value:
        raise SystemExit(f"missing env {name} and no default registered")
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{name}={path} does not exist")
    return path


def retarget_python_bin() -> Path:
    value = os.environ.get("RETARGET_PYTHON_BIN")
    if value:
        return Path(value)
    return env_path("HOLOSOMA_DEPS_DIR") / "miniconda3/envs/hsretargeting/bin/python"


# --------------------------------------------------------------------------
# TSV helpers (same dialect as the dcv3 stages)
# --------------------------------------------------------------------------
def read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> int:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def is_in_scope(row: dict[str, str]) -> bool:
    """Inventory/raw-contact row is an E206 candidate."""
    return (
        row.get("object_key") in OBJECT_KEYS
        and row.get("object_key") not in DROPPED_OBJECT_KEYS
        and row.get("action") in MOVE2_ACTIONS
        and not row.get("hard_reject_reason")
    )
