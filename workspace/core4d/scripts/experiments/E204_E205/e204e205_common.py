#!/usr/bin/env python3
"""Shared contract for E204 (noPRG / E167A) and E205 (G1A2 = PRG+G1+A2).

Both experiments reuse E178's 27 bucket cases (bucket003=9 / bucket004=4 /
bucket007=14) VERBATIM: same omnirt_v1 ref_fk reference trajectory, same 3cm
contact mask, same CEM budget, same contact-aligned 5-segment object proxy
(``scene_act_E178_contactAlignedTop``). Only the reward *arm* changes:

  * E204 (noPRG): strip the leg-object PRG (scene leg<->object pairs removed +
    ``leg_object_penalty_scale=0`` / ``cem_leg_gate_enabled=false``), object
    ``gravcomp=0``, no A2. The 5-segment object proxy is kept.
  * E205 (G1A2): E178 PRG kept + G1 (object ``gravcomp`` 0->1 single-variable
    sidecar) + A2 (hand-gate three-field retune).

Nothing here is copied: geom constants come from the E175 base module, the
A2 pack + gravcomp assertion come from E198/E200, the 27-case authority + scene
build come from the E178 production builder. Single source of truth per rule 13.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
_EXP = REPO / "workspace/core4d/scripts/experiments"
for _d in ("E204_E205", "E178", "E177", "E176", "E175", "E200", "E198"):
    p = str(_EXP / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

# E178 production builder (configures `production` globals for contactAlignedTop).
import build_contact_aligned_production as e178prod  # noqa: E402
production = e178prod.production  # build_semantic_bucket_production, E178-configured
base = production.base  # E175 build_nonbox_multigeom_production

# Reused intervention primitives (single source of truth).
from e198_common import A2_GATE, CEM_SEED, FULL_OPT_STEPS, FULL_SAMPLES  # noqa: E402
from e200_common import assert_gravcomp_diff  # noqa: E402

# Reused geom contracts (E175 base).
HAND_GEOMS = tuple(base.HAND_GEOMS)          # ("lh", "rh")
LOWER_BODY_GEOMS = tuple(base.LOWER_BODY_GEOMS)  # 16 leg/foot geoms (PRG)

# --- arms --------------------------------------------------------------------
ARMS = ("noprg_e204", "g1a2_e205")
ARM_EXP = {"noprg_e204": "E204", "g1a2_e205": "E205"}
E178_SCENE = "scene_act_E178_contactAlignedTop"
E204_SCENE = "scene_act_E204_contactAlignedTop_noPRG"
E205_SCENE = "scene_act_E205_contactAlignedTop_gravcomp"
ARM_SCENE = {"noprg_e204": E204_SCENE, "g1a2_e205": E205_SCENE}
ARM_TAG = {"noprg_e204": "noPRG", "g1a2_e205": "G1A2"}

# --- results / config paths --------------------------------------------------
RESULTS = {"noprg_e204": REPO / "workspace/core4d/results/E204",
           "g1a2_e205": REPO / "workspace/core4d/results/E205"}
OVERRIDE_DIR = REPO / "examples/config/override"
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"

# E163 default hand-gate (inherited baseline); E204 must leave these untouched.
E163_HAND_GATE = {
    "cem_hand_gate_min_sdf_m": -0.010,
    "cem_hand_gate_max_violation_pct": 0.10,
    "cem_hand_gate_hard_floor_m": -0.020,
}


_SOURCES: list[dict] | None = None
_CASE_TASK: dict[str, str] | None = None


def load_sources() -> list[dict]:
    """The 27 E178 bucket source rows (from the E174 manifest, E178 filter)."""
    global _SOURCES
    if _SOURCES is not None:
        return _SOURCES
    rows = base.read_tsv(production.E174_MANIFEST)
    sources = [r for r in rows if r["object_key"] in production.EXPECTED_OBJECT_COUNTS]
    from collections import Counter
    dist = Counter(r["object_key"] for r in sources)
    if len(sources) != 27 or dict(dist) != production.EXPECTED_OBJECT_COUNTS:
        raise ValueError(f"E204/E205 authority drift: n={len(sources)} dist={dict(dist)}")
    _SOURCES = sources
    return sources


def _case_task() -> dict[str, str]:
    """case_id -> dcv3 task dir name. NOTE: 26/27 are omnirt_v1 but 1 case
    (bucket007_20231020_055_p1) is omnirt_v2 (v2-rescued); derive from the source
    row rather than assuming a variant, so we inherit exactly what E178 used."""
    global _CASE_TASK
    if _CASE_TASK is None:
        _CASE_TASK = {s["case_id"]: Path(s["scene_act"]).parent.name for s in load_sources()}
    return _CASE_TASK


def task_of(case_id: str) -> str:
    return _case_task()[case_id]


def task_dir(case_id: str) -> Path:
    return TASK_ROOT / task_of(case_id)


def e178_scene_path(case_id: str) -> Path:
    return task_dir(case_id) / f"{E178_SCENE}.xml"


def arm_scene_path(arm: str, case_id: str) -> Path:
    return task_dir(case_id) / f"{ARM_SCENE[arm]}.xml"


def override_id(arm: str, case_id: str) -> str:
    return f"core4d_{ARM_EXP[arm]}_{case_id}_contactAlignedTop_{ARM_TAG[arm]}"


def override_path(arm: str, case_id: str) -> Path:
    return OVERRIDE_DIR / f"{override_id(arm, case_id)}.yaml"


def e178_override_id(case_id: str) -> str:
    return f"core4d_E178_{case_id}_contactAlignedTop"


def dcv3_override_id(case_id: str) -> str:
    return f"core4d_{task_of(case_id)}"


def arm_out_dir(arm: str, case_id: str, stage: str = "full") -> Path:
    # stage separates canary (smoke) rollouts from full so a small-budget smoke
    # never shadows the full 1024x32 run via skip-already-done.
    return RESULTS[arm] / f"s6_downstream/cem/{stage}" / f"{ARM_EXP[arm]}_{case_id}_{ARM_TAG[arm]}"


def result_npz(arm: str, case_id: str, stage: str = "full") -> Path:
    """CEM rollout the driver checks for skip-already-done (contact_guidance on)."""
    return arm_out_dir(arm, case_id, stage) / "trajectory_mjwp_act.npz"
