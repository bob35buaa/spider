#!/usr/bin/env python3
"""Shared contracts and filesystem helpers for E174 (non-box full pipeline).

Single source of truth for E174 paths, data/env locations, frozen config values,
case-id schema, and small IO helpers. Experiment scripts must import from here so
that authority, manifests, CEM overrides and audits stay field-level consistent.

E174 forks the E173 contract (same frozen algorithm: omnirt v1->v2 rescue +
ref_fk + rubber_hull + E170 PRG cross-object candidate), but is the FIRST run to
leave the box topology: scope is 5 bucket + 2 desk objects sized in [box023,
box025], action scope is move2-ONLY (move1 -> EXCLUDE_ACTION_NOT_MOVE2, kept in
registry but not carried into S2-S6; join/leave/rot/pass/strike/raise are Stage0
non-preferred rejects), and NO RL export.

Two orthogonal collision axes (do NOT conflate; see dcv3 doc 04 + doc 15):
  - OBJECT concavity -> object-side collision_policy proxy, per category:
      bucket -> bucket_wall_proxy_aabb        (floor + 4 walls)
      desk   -> desk_surface_voxel_multibox_proxy_draft (surface voxelize)
    Non-box templates are `manual_review_required` by default and MUST reach
    `clean_reviewed` via nonbox_template_review.tsv (review_decision=approve_clean)
    before S3 -- they CANNOT auto-clean from a MuJoCo load/render pass.
  - ROBOT HAND collision -> hand_collision_variant_id=rubber_hull (lh/rh mesh
    convex hull), injected via an E174-scoped sidecar; orthogonal to object shape.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


# --- Repo / result roots -------------------------------------------------
REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E174"
SCRIPT_DIR = REPO / "workspace/core4d/scripts/experiments/E174"
DCV3 = REPO / "workspace/core4d/scripts/data_construction_v3"
OVERRIDE_DIR = REPO / "examples/config/override"

# --- Live data / model / env paths (tidal mount; a0ccc676 mount is dead) --
CORE4D_RAW_ROOT = Path(
    os.environ.get(
        "CORE4D_RAW_ROOT",
        "/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real",
    )
)
# smplx.create(dir, model_type="smplx") appends "smplx/", so this is the PARENT.
SMPLX_MODEL_DIR = Path(
    os.environ.get(
        "SMPLX_MODEL_DIR",
        "/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/human_model_files",
    )
)
HOLOSOMA_REPO = Path(
    os.environ.get("HOLOSOMA_REPO", REPO.parent / "holosoma")
)
HOLOSOMA_DEPS_DIR = Path(
    os.environ.get("HOLOSOMA_DEPS_DIR", "/mnt/ali-sh-1/dataset/zeus/xiayb/.holosoma_deps")
)
# hsretargeting conda env python (S3 OmniRetarget); NOT under /root/miniconda3.
RETARGET_PYTHON_BIN = Path(
    os.environ.get(
        "RETARGET_PYTHON_BIN",
        str(HOLOSOMA_DEPS_DIR / "miniconda3/envs/hsretargeting/bin/python"),
    )
)
SPIDER_PYTHON_BIN = Path(os.environ.get("PYTHON_BIN", str(REPO / ".venv/bin/python")))

# --- Scope (5 bucket + 2 desk; move2-only production) ----------------------
# Priority order (GPU queue + execution): bucket small->large volume, then desk.
OBJECT_KEYS = (
    "bucket004", "bucket009", "bucket010", "bucket007", "bucket003",
    "desk005", "desk007",
)
OBJECT_PRIORITY = OBJECT_KEYS
# Non-box object category + collision policy (dcv3 doc 04). These are the ONLY
# mechanism expressing object concavity; frozen (no target_cells/shrink/geom
# sweep). Orthogonal to hand rubber_hull.
OBJECT_CATEGORY = {
    "bucket004": "bucket", "bucket009": "bucket", "bucket010": "bucket",
    "bucket007": "bucket", "bucket003": "bucket",
    "desk005": "desk", "desk007": "desk",
}
OBJECT_COLLISION_POLICY = {
    "bucket": "bucket_wall_proxy_aabb",
    "desk": "desk_surface_voxel_multibox_proxy_draft",
}
NONBOX_TEMPLATE_ADAPTER = {
    "bucket": "nonbox_proxy_aabb_review",
    "desk": "nonbox_surface_voxel_review",
}
# Action scope: move2-ONLY enters production. move1 -> EXCLUDE_ACTION_NOT_MOVE2
# (registry only). Everything else -> dcv3 Stage0 reject_action_not_first_line.
ACTION_SCOPE = "move2_only"
# Raw authority closes over ALL person-cases (fresh S1 inventory, 2026-07-23).
# Fresh authority = 330 pc / 165 seq for the 7 objects (bucket007 alone is 126 pc
# / 63 seq -- the most-reused bucket in CORE4D; the pre-run plan prior of 240/74
# undercounted it). Production candidate set is the 92 move2 pc (46 seq); move1
# and other actions are retained in the registry but not carried into S2-S6.
EXPECTED_PERSON_CASES = 330
EXPECTED_SEQUENCES = 165
EXPECTED_MOVE_PERSON_CASES = 92   # move2-only production denominator
EXPECTED_MOVE_SEQUENCES = 46
# historical prior only (NOT E174 authority); fresh S1 inventory is authority.
# move_* fields are the move2-only production counts (verified vs fresh inventory).
RAW_INVENTORY_PRIOR = {
    "bucket004": {"person_cases": 32, "sequences": 16, "move_person_cases": 10, "move_sequences": 5,
                  "dates": ("20231002", "20231003_1")},
    "bucket009": {"person_cases": 16, "sequences": 8, "move_person_cases": 6, "move_sequences": 3,
                  "dates": ("20231002",)},
    "bucket010": {"person_cases": 32, "sequences": 16, "move_person_cases": 14, "move_sequences": 7,
                  "dates": ("20231002", "20231003_2", "20231011")},
    "bucket007": {"person_cases": 126, "sequences": 63, "move_person_cases": 30, "move_sequences": 15,
                  "dates": ("20231003_1", "20231003_2", "20231020", "20231023")},
    "bucket003": {"person_cases": 32, "sequences": 16, "move_person_cases": 12, "move_sequences": 6,
                  "dates": ("20231018", "20231020")},
    "desk005": {"person_cases": 18, "sequences": 9, "move_person_cases": 6, "move_sequences": 3,
                "dates": ("20231023",)},
    "desk007": {"person_cases": 74, "sequences": 37, "move_person_cases": 14, "move_sequences": 7,
                "dates": ("20231023", "20231030")},
}

# --- Frozen retarget contract ---------------------------------------------
RETARGET_PRIMARY_VARIANT = "omnirt_v1"
RETARGET_RESCUE_VARIANT = "omnirt_v2"
TARGET_VARIANT = "ref_fk"
V1_INFEASIBLE_STATUS = "omniretarget_infeasible"  # only status eligible for v2 rescue

# --- Frozen CEM / scene contract (E170 PRG cross-object candidate) ---------
HAND_COLLISION_VARIANT = "rubber_hull"
SCENE_NAME = "scene_act_E174_rubberHull_PRG"
BASE_REWARD_METHOD = "E167A_zOnlyBody"
SOURCE_CONFIG_ID = "E170_PRG_lowerbodyPhysics_softPenalty_candidateGate"
E174_METHOD_ID = "E174_E170PRG_nonbox_candidate_r1"

# 16 lower-body/object collision geoms (E169/E170 P config)
LOWER_BODY_GEOMS = (
    "left_hip_collision", "right_hip_collision", "left_thigh_collision",
    "right_thigh_collision", "left_shin_collision", "right_shin_collision",
    "left_linkage_brace_collision", "right_linkage_brace_collision",
    "lf0", "lf1", "lf2", "lf3", "rf0", "rf1", "rf2", "rf3",
)

# lower-body soft penalty (E169/E170 R config)
LEG_OBJECT_PENALTY_SCALE = 2.0
LEG_OBJECT_PENALTY_MARGIN_M = 0.02
# candidate gate
CEM_LEG_GATE_MIN_SDF_M = 0.005
CEM_LEG_GATE_MAX_VIOLATION_PCT = 0.02
CEM_LEG_GATE_HARD_FLOOR_M = -0.005
CEM_LEG_GATE_MIN_VALID_FRAC = 0.02
CEM_LEG_GATE_FALLBACK = "least_violation"

# CEM budget
CEM_SEED = 0
CEM_FULL_SAMPLES = 1024
CEM_FULL_OPT_STEPS = 32
CEM_CANARY_SAMPLES = 64   # E170 proven smoke budget
CEM_CANARY_OPT_STEPS = 4


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def rel(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except (OSError, ValueError):
        return str(value)


def sha256(value: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(value).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_tsv(value: str | Path) -> list[dict[str, str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_tsv_with_fields(value: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with repo_path(value).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_tsv(value: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    output = repo_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        Path(temporary).replace(output)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(value: str | Path, payload: Any) -> None:
    output = repo_path(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
