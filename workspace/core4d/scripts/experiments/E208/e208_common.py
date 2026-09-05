#!/usr/bin/env python3
"""E208 shared contract: desk/chair translation augmentation -> PRG full CEM.

E208 takes E206's 22 human-USE desk/chair cases and runs the OmniRetarget
object-interaction *translation* augmentation over them (trans_0/1/2), then a
formal full CEM on the PRG arm only.  The orig baseline is E206's existing PRG
rollouts, reused rather than re-run.

Three things make E208 different from the box/bucket augmentation line
(E199/E202) and they are all encoded here:

1. **omnirt_v1 first, omnirt_v2 rescue.**  E199/E202 froze v2 for *every*
   variant.  E206 ran v1 (its variant registry says v2 is "rescue-only, do not
   use before an explicit omnirt_v1 infeasible result"), so E208 keeps v1 as the
   default and only escalates a variant that CVXPY actually declared infeasible.
   Consequence: :data:`OMNIRT_V1_ENV` is a *new* constant -- reusing
   ``e199_common.OMNIRT_V2_ENV`` would silently retarget under a different
   contract than the baseline it is compared against.  Note also that
   ``e199_common.OMNIRT_V2_ENV`` omits ``REPLACE_WRIST_WITH_FINGERTIP``, whose
   ``pipeline.sh`` default is **1** while E206 used **0** -- so both env dicts
   here spell out all six keys.

2. **v1-aware task naming.**  ``e199_common.aug_task_name`` hardcodes a
   ``omnirt_v1 -> omnirt_v2`` relabel.  Under a v1 run that would produce dirs
   whose name claims v2.  :func:`aug_task_name` takes the effective variant
   explicitly, so the directory name *is* the provenance.

3. **The collision proxy already lives upstream.**  E206 installed its
   hand-placed lowgeom boxes into the source template
   ``<obj>_person{1,2}/scene.xml`` (P4), so an ``__aug_*`` task built from that
   template inherits the proxy for free.  What it does *not* inherit are the
   three ``scene_act_E206_lowgeom_*.xml`` sidecars, which E208 rebuilds by
   calling E206's :mod:`build_arm_scenes` unchanged (see :data:`ARM`).

Nothing numeric is re-typed: the arm overrides, hand gate, CEM budget and pair
arithmetic all come from ``e206_common``; the manifest schema and IO helpers
come from ``e199_common``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = REPO / "workspace/core4d/scripts/experiments"

for _p in (EXPERIMENTS / "E199", EXPERIMENTS / "E206"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import e199_common as E199  # noqa: E402
import e206_common as E206  # noqa: E402


def _drop_broken_torch() -> None:
    """Drop the SPIDER venv's broken ``torch`` namespace stub (no ``Tensor``).

    Importing ``e206_common`` pulls in the E175 geometry module, which imports
    that stub; once it is in ``sys.modules`` scipy's ``Rotation.from_quat``
    starts probing ``getattr(torch, 'Tensor')`` through array-api-compat and
    raises.  Lifted verbatim from ``e202_common.py:118-130``.
    """
    mod = sys.modules.get("torch")
    if mod is not None and not hasattr(mod, "Tensor"):
        del sys.modules["torch"]


_drop_broken_torch()


def load_e208_module(name: str):
    """Load an E208-local sibling module by explicit path.

    Both ``e199_common`` and ``e206_common`` push their own experiment dir onto
    ``sys.path[0]``, and E208 has same-named siblings (``build_augmented_tasks``,
    ``build_aug_manifest``).  A bare import would silently resolve to E199's
    copy.  This is the ONLY sanctioned way for E208 scripts to import each other.
    """
    path = SCRIPT_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no such E208 module: {path}")
    spec = importlib.util.spec_from_file_location(f"e208_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E208"
PLAN_SLOT = 238
# Log slot 296 was claimed first by the concurrent E207 session (committed 01:24
# vs this experiment's 15:16 on 2026-09-05), so E208 moved to 297 per plan239's
# adjudication E207=296 / E208=297 / E209=298.  Kept as a constant because A9
# checks it and the doc paths derive from it.
LOG_SLOT = 297
PLAN_REF = f"workspace/core4d/plan/{PLAN_SLOT}_E208_deskchair_translation_augmentation_plan.md"
LOG_REF = f"workspace/core4d/log/{LOG_SLOT}_E208_deskchair_translation_augmentation.md"
RUN_ID = "R294"
BASE_EXP_ID = E206.EXP_ID  # "E206" -- geometry, overrides, budget and orig baseline
SPIDER_DATASET = E206.SPIDER_DATASET  # core4d v1


# --------------------------------------------------------------------------
# Case registry: E206's delivered 22-case paired table
# --------------------------------------------------------------------------
SOURCE_TSV = E206.S6_DIR / "rl_export/paired_rl_export_input.tsv"
EXPECTED_SOURCE_SHA256 = "af17e209a9cbce3cad7597f0782a60c78b8f3c76ac7656efcb60b2a6cb2e1d5e"
EXPECTED_CASES = 22

MANUAL_REVIEW_TSV = E206.S6_DIR / "eval/two_arm/user_manual_review_filled.tsv"
EXPECTED_MANUAL_REVIEW_SHA256 = (
    "7e95901c5b51ca6866fdd74c001e16af1884978ef98aa17e5c30da1a8f41f249"
)

READY_DECISION = "RL_EXPORT_READY"

# Per-object case counts in the 22, pinned so a silently shrinking registry
# cannot slip through (chair005 is the n=1 object that drives the `indicative`
# rendering rule in the eval workbook).
EXPECTED_CASES_BY_OBJECT = {
    "desk021": 7, "chair006": 5, "desk007": 5, "desk023": 4, "chair005": 1,
}

# The one case whose SOURCE side is already an omnirt_v2 rescue: it has no v1
# pass and nothing to escalate to, so it runs v2 from the start and is excluded
# from the v1 feasibility denominator (see `rescue_state` below).
SOURCE_V2_CASES = ("desk021_20231011_014_p2",)


def load_e208_cases() -> list[dict[str, str]]:
    """The 22 `RL_EXPORT_READY` rows, projected onto the fields E208 needs.

    ``base_target_task`` comes from the manifest's own ``stage2b_target_task``
    column and is cross-checked against the ``scene_act`` parent -- never
    rebuilt from ``case_id``, because the rescued case lives under
    ``dcv3_omnirt_v2_ref_fk_*`` (E206 hit that bug once).
    """
    cases: list[dict[str, str]] = []
    for row in E199.read_tsv(SOURCE_TSV):
        if row.get("paired_rl_export_decision") != READY_DECISION:
            continue
        base = (row.get("stage2b_target_task") or "").strip()
        from_scene = Path(row["scene_act"]).parent.name
        if not base:
            base = from_scene
        elif base != from_scene:
            raise AssertionError(
                f"{row['case_id']}: stage2b_target_task={base!r} != scene_act parent {from_scene!r}"
            )
        cases.append({
            "case_id": row["case_id"],
            "object_key": row["object_key"],
            "person": row["person"],
            "person_idx": row["person_idx"],
            "base_target_task": base,
            "source_retarget_variant_id": row["retarget_variant_id"],
            "orig_target_scene": row.get("target_scene", ""),
            "orig_scene_act": row.get("scene_act", ""),
            "orig_trajectory": row.get("trajectory", ""),
            "orig_contact_mask": row.get("contact_mask", ""),
            "orig_result_npz": row.get("cem_result_npz", ""),
            "orig_video": row.get("cem_video", ""),
            "orig_manual_use_decision": row.get("manual_use_decision", ""),
            "orig_manual_quality_label": row.get("manual_quality_label", ""),
            "orig_manual_failure_taxonomy": row.get("manual_failure_taxonomy", ""),
            "orig_numeric_release_pass": row.get("numeric_release_pass", ""),
        })
    cases.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return cases


# --------------------------------------------------------------------------
# Retarget contract -- six keys, always spelled out (plan238 A7 / R6)
# --------------------------------------------------------------------------
# Authority: results/E206/registries/retarget_variant_registry.tsv, cross-checked
# against the literal `env ...` line in
# results/E206/s3_retarget/omnirt_v{1,2}/ref_fk/run_stage2b_omnirt_v{1,2}_ref_fk.sh
OMNIRT_V1_ENV = {
    "REPLACE_WRIST_WITH_FINGERTIP": "0",
    "RETARGET_ENABLE_CONSTRAINT_RELAXATION": "0",
    "RETARGET_ENABLE_FOOT_Z_CONSTRAINT": "0",
    "RETARGET_ENABLE_CONTACT_PRESERVATION": "0",
    "RETARGET_FOOT_SLIDE_PENALTY_WEIGHT": "0.0",
    "RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE": "1.0",
}
OMNIRT_V2_ENV = {
    "REPLACE_WRIST_WITH_FINGERTIP": "0",
    "RETARGET_ENABLE_CONSTRAINT_RELAXATION": "1",
    "RETARGET_ENABLE_FOOT_Z_CONSTRAINT": "1",
    "RETARGET_ENABLE_CONTACT_PRESERVATION": "1",
    "RETARGET_FOOT_SLIDE_PENALTY_WEIGHT": "1.0",
    "RETARGET_OBJECT_PENETRATION_TOLERANCE_SCALE": "0.8",
}
OMNIRT_ENV_BY_VARIANT = {"omnirt_v1": OMNIRT_V1_ENV, "omnirt_v2": OMNIRT_V2_ENV}
RETARGET_VARIANTS = ("omnirt_v1", "omnirt_v2")

# The env keys pipeline.sh consumes that are NOT retarget knobs but still have
# to match E206 or the trajectories are not comparable.
PIPELINE_DATASET_ENV = {
    "SPIDER_DATASET": "core4d",
    "SPIDER_SOURCE_DATASET": "core4d",
}

STAGE2B_SCRIPT = {
    v: E206.S3_DIR / f"{v}/ref_fk/run_stage2b_{v}_ref_fk.sh" for v in RETARGET_VARIANTS
}


# --------------------------------------------------------------------------
# Variants -- translation only (plan238 lock 1; rotation is measured, not built)
# --------------------------------------------------------------------------
TRANS_VARIANTS = list(E199.TRANS_VARIANTS)   # [(trans0, trans_0), ...]
ROT_VARIANTS = [("rot0", "rot_0"), ("rot1", "rot_1")]
ALL_AUG_VARIANTS = TRANS_VARIANTS + ROT_VARIANTS

# Scope amendment (user, 2026-09-05): E208 builds ALL FIVE augmentation variants.
#
# The plan was locked to translation-only because E199 reported rotation as
# "systematically infeasible" and E202 inherited that. P2 found the real cause:
# `augment_object_poses` crashed on a scipy shape requirement *before any IK*
# (holosoma src/utils.py:346, fixed in holosoma 9e544b1). Rotation had never been
# evaluated -- E199 attempted rot_0 97 times, rot_1 zero times, produced zero rot
# npz against 582 trans npz, and logged exactly one true infeasibility.
#
# Note rot_* is NOT a pure rotation: `generate_augmentation_configs` pairs each
# +/-45 deg yaw with a 0.2 m lateral translation, and the yaw decays on
# rotation_tau=25 while the translation decays on translation_tau=50.
BUILD_VARIANTS = TRANS_VARIANTS + ROT_VARIANTS
ROTATION_FIX_COMMIT = "9e544b1"  # holosoma; before this, rot_* could not run at all

aug_translation = E199.aug_translation
aug_rotation_rad = E199.aug_rotation_rad


def aug_task_name(base_target_task: str, variant: str, effective_retarget_variant: str) -> str:
    """``{base with variant relabelled}__aug_{variant}`` -- idempotent, v1-aware.

    Unlike ``e199_common.aug_task_name`` this performs NO implicit v1->v2
    rewrite: the caller passes the variant that actually produced the npz, so
    the directory name is evidence rather than a convention.
    """
    if effective_retarget_variant not in RETARGET_VARIANTS:
        raise ValueError(f"unknown retarget variant {effective_retarget_variant!r}")
    base = base_target_task
    for known in RETARGET_VARIANTS:
        if known in base and known != effective_retarget_variant:
            base = base.replace(known, effective_retarget_variant)
    return f"{base}__aug_{variant}"


# --------------------------------------------------------------------------
# Rescue state machine
# --------------------------------------------------------------------------
# v1_ok                 v1 produced the npz -> use it
# v1_infeasible         npz missing AND the log shows a CVXPY infeasibility
# rescued_v2            v2 produced the npz for a v1_infeasible variant
# rescue_failed         v2 also infeasible -> variant dropped, counted in C2
# needs_triage          npz missing but NO infeasibility in the log (crash / OOM
#                       / disk) -- deliberately NOT auto-escalated, because
#                       calling a bug "mathematically infeasible" would corrupt
#                       both the C2 yield and the L-ladder decision
# source_v2_ok          source side was already v2; it produced the npz
# source_v2_infeasible  ditto but infeasible -- terminal, nothing to escalate to
# upstream_rotation_bug  rot_* only: upstream crashed in augment_object_poses
#                       BEFORE any IK (holosoma src/utils.py:346 calls
#                       R.from_euler("z", (N,)) but scipy 1.17.1 needs (N,1)).
#                       Measured, 2026-09-05: E199 attempted rot_0 97 times,
#                       rot_1 0 times, produced 0 rot npz vs 582 trans npz, and
#                       logged exactly ONE genuine infeasibility -- so E199's
#                       "rotation is systematically infeasible" was this crash,
#                       not an IK result. Kept distinct from *_infeasible so
#                       E208 does not repeat that misattribution, and from
#                       needs_triage so it does not pollute the trans signal.
RESCUE_STATES = (
    "v1_ok", "v1_infeasible", "rescued_v2", "rescue_failed", "needs_triage",
    "source_v2_ok", "source_v2_infeasible", "upstream_rotation_bug",
)
BUILT_STATES = frozenset({"v1_ok", "rescued_v2", "source_v2_ok"})

# Regexes the feasibility detector greps for in the pipeline log (signal B).
INFEASIBLE_LOG_PATTERNS = (
    "SolverError", "infeasible", "DCPError", "cvxpy", "SolverFailure",
)

# Potential aug tasks = 22 cases x 5 variants.  The L ladder below is expressed
# as a FRACTION of this so the 3-variant thresholds from plan238 keep their
# meaning after the 5-variant amendment (L0 was 55/66 = 83%, L1 44/66 = 67%,
# L2 25/66 = 38%).
POTENTIAL_AUG_TASKS = EXPECTED_CASES * len(BUILD_VARIANTS)   # 110

# C2 escalation floor: average of >=1 surviving variant per case.  Mirrors the
# spirit of E206's MIN_CASES_ESCALATE=12.  Unchanged by the amendment: the bar is
# "did every case keep at least one variant", not "how many variants exist".
MIN_AUG_TASKS_ESCALATE = 22

# R10 fallback ladder, keyed on `c` = number of aug tasks actually built.
L_LADDER = (
    ("L0", 0.83, "run as planned; full stratified C3/C4"),
    ("L1", 0.67, "run; per-(object,variant) yield promoted to a first-class claim"),
    ("L2", 0.38, "run CEM but C4 degrades to pooled-only; escalate before spending review budget"),
    ("L3", 0.0, "stop before CEM; triage bug-vs-infeasible, amplitude change needs user sign-off"),
)


def l_tier(n_built: int, potential: int | None = None) -> tuple[str, str]:
    frac = n_built / float(potential or POTENTIAL_AUG_TASKS)
    for name, floor, action in L_LADDER:
        if frac >= floor:
            return name, action
    return L_LADDER[-1][0], L_LADDER[-1][2]


# --------------------------------------------------------------------------
# Effective augmentation floor (P2 finding, not in plan238)
# --------------------------------------------------------------------------
# The augmentation perturbs the APPROACH segment and decays back to the original
# from `object_moving_frame_idx` onwards (translation_tau=50 frames).  SPIDER only
# ever sees the contact-trimmed window, so when `trim_start` lands well after the
# object starts moving, most of the decay has already happened and the variant
# reaches SPIDER with only a fraction of the nominal 0.2 m.
#
# Measured 2026-09-05 -- this is NOT desk/chair-specific and no prior experiment
# gated on it:
#     E199 fullscale  n=249  min 0.083 m   21 below 0.18 m  (8.4%)
#     E202 bucket     n= 73  min 0.074 m   24 below 0.18 m  (33%)
#     E208 probes     n= 15  min 0.023 m    6 below 0.18 m
# So plan238's `approach_trans_offset_m_max in [0.18, 0.22]` is the WRONG gate --
# as an absolute pass/fail it would reject a third of what E202 shipped.
#
# What is worth gating is "this is not augmentation at all".  E208's chair005
# arrives at 0.023 m, an order of magnitude below anything E199/E202 shipped and
# far under E206's own ~12 cm object-tracking error, so it is a near-duplicate of
# orig: it would inflate the dataset without adding diversity AND flatter the C4
# delta (aug ~= orig by construction).  The floor is set below every value prior
# experiments shipped, so it does not retroactively invalidate them.
EFFECTIVE_AUG_FLOOR_M = 0.05
NOMINAL_AUG_OFFSET_M = 0.20

# Measured over all 110 built variants (2026-09-05):
#   chair005  n= 5  every variant 0.0226 m   <- the whole object is degenerate
#   chair006  n=25  min 0.2000
#   desk007   n=25  min 0.1213
#   desk021   n=35  min 0.0935
#   desk023   n=20  min 0.1452
# chair005's trim_start is 113 frames, by far the latest, so both the translation
# (tau=50) and the rotation (tau=25, hence yaw 0.58 deg instead of 45) have almost
# fully decayed before SPIDER's window opens.  All five of its variants are
# near-duplicates of orig.
#
# User decision 2026-09-05: EXCLUDE chair005 from E208 entirely rather than ship
# near-duplicates.  Cost: the per-object stratification drops from 5 objects to 4
# (chair005 was the only n=1 object).  Benefit: no padded dataset, and C4's pooled
# delta is not flattered by five rows where aug == orig by construction.
EXCLUDED_OBJECT_KEYS = ("chair005",)
EXCLUDED_REASON = "decay_degenerate_all_variants_below_effective_floor"


def is_excluded(object_key: str) -> bool:
    return object_key in EXCLUDED_OBJECT_KEYS


# C4 stratification band on the effective offset (user decision 2026-09-05:
# promote it from a reported number to a first-class stratification variable).
# The question it answers -- and that no prior experiment could -- is at what
# augmentation amplitude the tracking cost actually starts being paid.
OFFSET_BANDS = (
    ("full", 0.18, 1e9),      # ~nominal
    ("partial", 0.10, 0.18),
    ("weak", 0.0, 0.10),
)


def offset_band(offset_m: float) -> str:
    for name, lo, hi in OFFSET_BANDS:
        if lo <= offset_m < hi:
            return name
    return "weak"


# --------------------------------------------------------------------------
# Arm: PRG only (plan238 lock 2)
# --------------------------------------------------------------------------
ARM_ID = "prg"
SCENE_NAME = E206.SCENE_BY_ARM["prg"]        # scene_act_E206_lowgeom_PRG
SCENE_RUBBER_HULL = E206.SCENE_RUBBER_HULL
SCENE_NOPRG = E206.SCENE_BY_ARM["noprg"]     # built as evidence, never referenced
PRG_OVERRIDES = dict(E206.PRG_OVERRIDES)     # values, not a re-typed copy
E163_HAND_GATE = dict(E206.E163_HAND_GATE)
expected_pair_counts = E206.expected_pair_counts

# E206's arm-scene builder, reused unchanged.  `build_arm_scenes.task_dir()`
# reads `row["target_task"]` before falling back to a case_id-derived name, so
# feeding it an `__aug_*` target_task lands the whole rubberHull -> noPRG -> PRG
# chain in the aug dir without touching E206's code.
_ARM_SPEC = importlib.util.spec_from_file_location(
    "e208_e206_build_arm_scenes", EXPERIMENTS / "E206/build_arm_scenes.py"
)
ARM = importlib.util.module_from_spec(_ARM_SPEC)
_ARM_SPEC.loader.exec_module(ARM)
_drop_broken_torch()


def override_name(case_id: str, variant: str) -> str:
    return f"core4d_{EXP_ID}_{case_id}_aug_{variant}_lowgeom_PRG"


def aug_variant_id(case_id: str, variant: str) -> str:
    return E199.safe_id(f"{EXP_ID}_{case_id}_aug_{variant}_PRG")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RESULTS = REPO / "workspace/core4d/results" / EXP_ID
DP = RESULTS / "data_preprocess"
DP_MANIFESTS = DP / "manifests"
DP_SENTINELS = DP / "sentinels"
DP_CONTACT_MASKS = DP / "contact_masks"
DP_CASE_FILES = DP / "case_files"
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
E206_OVERRIDE_SRC = E206.S5_DIR / "cem_overrides/overrides"

CONTRACT_SELFCHECK_JSON = PREFLIGHT_DIR / "e208_contract_selfcheck.json"
SEED_INTEGRITY_TSV = PREFLIGHT_DIR / "orig_seed_integrity.tsv"
P2_GATE_JSON = PREFLIGHT_DIR / "p2_gate_decision.json"
RUNTIME_CONFIG_AUDIT_JSON = PREFLIGHT_DIR / "runtime_config_audit.json"

FEASIBILITY_TSV = DP_MANIFESTS / "e208_aug_feasibility.tsv"
ARTIFACTS_TSV = DP_MANIFESTS / "e208_aug_artifacts.tsv"
SCENE_PARITY_TSV = S5_DIR / "scene_parity_vs_e206.tsv"
ARM_SCENE_DIR = S5_DIR / "arm_scenes"
OVERRIDE_AUDIT_JSON = S5_DIR / "aug_override_audit.json"

PRIORITY_MANIFEST = MANIFEST_DIR / "e208_priority_manifest.tsv"
FROZEN_MANIFEST = MANIFEST_DIR / "e208_priority_manifest.frozen.tsv"
FREEZE_JSON = MANIFEST_DIR / "freeze.json"
AUTHORITY_TSV = MANIFEST_DIR / "e208_aug_authority.tsv"
ADMISSION_JSON = S6_DIR / "cem/throughput/admission_decision.json"
SOURCE_ADMISSION_JSON = E206.ADMISSION_JSON

CEM_LOG_DIR = REPO / "logs/E208/cem"


def data_root(retarget_variant: str) -> Path:
    """RESULT_ROOT for one retarget variant.

    v1 and v2 products MUST live in separate trees: upstream short-circuits on
    an existing output filename, and both variants emit the same
    ``{task}_{trans_k}.npz`` name.  Sharing one root would leave
    ``effective_retarget_variant`` guessable only from mtime.  Split, the
    directory *is* the evidence.
    """
    if retarget_variant not in RETARGET_VARIANTS:
        raise ValueError(f"unknown retarget variant {retarget_variant!r}")
    return DP / retarget_variant


def holosoma_dir(base_target_task: str, retarget_variant: str) -> Path:
    return data_root(retarget_variant) / f"holosoma_{base_target_task}"


def sentinel_path(base_target_task: str, pass_id: str) -> Path:
    return DP_SENTINELS / f"{base_target_task}.{pass_id}.done"


def cem_out_dir(case_id: str, variant: str, stage: str = "full") -> Path:
    return S6_DIR / "cem" / stage / f"{EXP_ID}_{case_id}_aug_{variant}_PRG"


def render_path(case_id: str, variant: str, stage: str = "full") -> Path:
    return S6_DIR / "render" / stage / f"{EXP_ID}_{case_id}_aug_{variant}_PRG_{stage}.mp4"


# --------------------------------------------------------------------------
# CEM budget -- inherited from E206's admission decision, never re-typed
# --------------------------------------------------------------------------
def source_admission() -> dict[str, Any]:
    """E206's frozen admission payload; both verdicts must still read `pass`."""
    path = SOURCE_ADMISSION_JSON
    if not path.is_file():
        raise SystemExit(f"missing E206 admission decision: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for gate in ("A1_single_task", "A2_queue"):
        if payload.get(gate, {}).get("verdict") != "pass":
            raise SystemExit(f"E206 admission {gate} is not 'pass'; refusing to inherit")
    return payload


def frozen_budget() -> dict[str, Any]:
    frozen = source_admission()["frozen"]
    return {
        "num_samples": int(frozen["num_samples"]),
        "max_num_iterations": int(frozen["max_num_iterations"]),
        "use_torch_compile": bool(frozen["use_torch_compile"]),
        "seed": E206.CEM_SEED,
    }


# Reference values only -- asserted against `frozen_budget()` by A8, never used
# as the source of truth for a run.
EXPECTED_BUDGET = {
    "num_samples": E206.CEM_NUM_SAMPLES,
    "max_num_iterations": E206.CEM_MAX_ITERATIONS,
    "use_torch_compile": False,
    "seed": E206.CEM_SEED,
}
GPU_DEFAULT = "0,1,2,3,4,5,6,7"
PER_GPU_MEM_MIB = 5000
PER_TASK_TIMEOUT_MIN = 180  # > E206's A2 per_task_bound_min (177.2)


# --------------------------------------------------------------------------
# Manifest schema
# --------------------------------------------------------------------------
# E199's 38 columns (the priority-queue dialect) plus the E208-specific
# provenance the v1/v2 rescue and the desk/chair proxy require.
EXTRA_FIELDS = [
    "pass", "effective_retarget_variant", "source_retarget_variant_id",
    "rescue_state", "rescue_reason",
    "object_geom_count", "compiled_robot_object_pair_count",
    "orig_result_npz", "f15_divergent",
]
FIELDS = list(E199.FIELDS) + [f for f in EXTRA_FIELDS if f not in E199.FIELDS]

FEASIBILITY_FIELDS = [
    "case_id", "object_key", "base_target_task", "variant", "holosoma_variant",
    "pass", "retarget_variant", "npz_present", "npz_sha256", "log_hit",
    "rescue_reason", "rescue_state", "built", "wall_s", "log_ref", "updated_at",
]


# --------------------------------------------------------------------------
# Known caveats inherited from E206 (surfaced in eval output, not hidden in prose)
# --------------------------------------------------------------------------
# F15: the retarget pipeline is not reproducible everywhere -- 19/22 of E206's
# re-run cases were bit-identical, 3 diverged.  Two of those three are in the
# E208 registry.  P1 seeds `_original` byte-for-byte from E206 instead of
# recomputing it, which removes the exposure by construction; these entries stay
# so the eval layer can still flag the rows.
F15_DIVERGENT_CASES = {
    "chair005_20231030_043_p1": 1.096,
    "desk023_20231030_019_p1": 0.097,
}

# F17: chair006's 10-box proxy leaves 42.9% of near-surface contact targets more
# than 3cm outside it.  orig and aug share the same proxy AND the same 3cm mask,
# so the bias is common-mode and the delta stays valid -- but the absolute
# contact numbers are not comparable across objects.  chair006 is 5 of the 22.
BLIND3CM_BY_OBJECT = {"chair006": 0.429}

# Probe set for P2: covers both F15 divergent cases, the cheapest object (n=1,
# N=2 boxes), the most expensive (N=12), the blind3cm object, and the source-v2
# case -- i.e. every axis that could change the L-ladder verdict.
PROBE_CASES = (
    "chair005_20231030_043_p1",
    "desk023_20231030_019_p1",
    "desk007_20231030_028_p1",
    "chair006_20231003_1_003_p1",
    "desk021_20231011_014_p2",
)


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
write_tsv = E199.write_tsv          # carries the JuiceFS EIO retry (R15)
write_json = E199.write_json
truth = E199.truth
load_case_meta = E199.load_case_meta

env_path = E206.env_path
retarget_python_bin = E206.retarget_python_bin
SPIDER_PYTHON_BIN = E199.SPIDER_PYTHON_BIN
HOLOSOMA_REPO = E199.HOLOSOMA_REPO
