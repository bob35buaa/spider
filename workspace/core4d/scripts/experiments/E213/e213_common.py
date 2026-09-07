#!/usr/bin/env python3
"""E213 shared contract: E212 paired-export cases -> selected-arm object augmentation.

E208 already ran the OmniRetarget object augmentation (trans_0/1/2 + rot_0/1) over
the 21 desk/chair source cases that E212 delivered in
``E212/s6_downstream/rl_export/paired_rl_export_input.tsv`` -- but its CEM ran the
plain **PRG** arm only.  E212's mixed-arm export instead selected a *per-case*
gravcomp arm (G1 / G0.8 / G0.6 / G0.4 / PRG; see the ``arm``/``arm_gravcomp``/
``arm_scene_name`` columns).  So except for the one PRG-selected case, E208's aug
rollouts are on the wrong arm.

E213 reuses E208's aug retarget NPZ and ``__aug_*`` SPIDER task dirs verbatim and
only:

1. writes, into each aug task dir, the selected-``g`` gravcomp ``scene_act``
   sidecar (a single-variable diff on the aug's own ``scene_act_E206_lowgeom_PRG``
   base -- exactly E212's ``build_sidecar`` mechanism, but against the aug dir);
2. writes one Hydra override per (case, aug variant) that inherits E208's PRG
   override and swaps ``scene_name`` to the selected sidecar (E212's pattern);
3. runs Full CEM per (case, aug variant) on that arm.

The partner side (Phase B, ``build_partner_aug.py``) generates augmented
kinematic retargets only -- no CEM -- for the paired export's partner-only cases.

Nothing numeric is re-typed: the case list, per-case arm and orig baseline come
from the E212 TSV; the aug variant artifacts, task naming and IO helpers come
from ``e208_common``; the single-variable gravcomp assertion comes from
``e212_common`` (which reuses ``e200_common._signature``).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = REPO / "workspace/core4d/scripts/experiments"

for _d in ("E213", "E212", "E208"):
    _p = str(EXPERIMENTS / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e208_common as E208  # noqa: E402  (pulls E199 + E206, drops broken torch)
import e212_common as E212C  # noqa: E402  (assert_gravcomp_diff_value / scene_gravcomp / _signature)

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
EXP_ID = "E213"
RUN_ID = "R299"
PLAN_SLOT = 243
LOG_SLOT = 301
BASE_AUG_EXP_ID = "E208"  # aug retarget + __aug_* tasks reused wholesale

# --------------------------------------------------------------------------
# Source registry: E212's mixed-arm paired export (21 RL_EXPORT_READY rows)
# --------------------------------------------------------------------------
SOURCE_TSV = REPO / "workspace/core4d/results/E212/s6_downstream/rl_export/paired_rl_export_input.tsv"
EXPECTED_SOURCE_SHA256 = "d71370fa7d1300ac41d28f3780468d2a5075313d44bdc6cc5b1fe97b647f3c1d"
READY_DECISION = "RL_EXPORT_READY"
EXPECTED_CASES = 21
EXPECTED_CASES_BY_OBJECT = {"desk021": 7, "chair006": 5, "desk007": 5, "desk023": 4}

# --------------------------------------------------------------------------
# Reused authority from e208_common / e199_common
# --------------------------------------------------------------------------
TASK_ROOT = E208.TASK_ROOT
OVERRIDE_DIR = E208.OVERRIDE_DIR
SPIDER_PYTHON_BIN = E208.SPIDER_PYTHON_BIN
ARTIFACTS_TSV = E208.ARTIFACTS_TSV            # e208_aug_artifacts.tsv (variant authority)
BUILT_STATUS = "built"                        # e208 artifacts status for adopted+effective rows
GPU_DEFAULT = E208.GPU_DEFAULT
PER_GPU_MEM_MIB = E208.PER_GPU_MEM_MIB
PER_TASK_TIMEOUT_MIN = E208.PER_TASK_TIMEOUT_MIN
SingleInstance = E208.SingleInstance

now = E208.now
rel = E208.rel
repo_path = E208.repo_path
sha256 = E208.sha256
sha256_text = E208.sha256_text
safe_id = E208.safe_id
read_tsv = E208.read_tsv
read_with_fields = E208.read_with_fields
write_tsv = E208.write_tsv
write_json = E208.write_json
truth = E208.truth
_drop_broken_torch = E208._drop_broken_torch

assert_gravcomp_diff_value = E212C.assert_gravcomp_diff_value
scene_gravcomp = E212C.scene_gravcomp

# CEM budget: inherited from E206's frozen admission via e208_common (1024x32 seed0).
frozen_budget = E208.frozen_budget

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
RESULTS = REPO / "workspace/core4d/results" / EXP_ID
DP = RESULTS / "data_preprocess"
PARTNER_DP = DP / "partner_aug"
S6_DIR = RESULTS / "s6_downstream"
MANIFEST_DIR = S6_DIR / "manifests"
EVAL_DIR = S6_DIR / "eval"
PREFLIGHT_DIR = RESULTS / "preflight"
SNAPSHOT_DIR = RESULTS / "scene_snapshot"

SOURCE_MANIFEST = MANIFEST_DIR / "e213_source_arm_cem.tsv"
PARTNER_MANIFEST = DP / "manifests" / "e213_partner_aug_artifacts.tsv"


def manifest_path(shard: str = "") -> Path:
    """Main manifest, or a per-shard manifest.

    Four 8-GPU machines share this /mnt.  The CEM runner rewrites the WHOLE
    manifest on every status change, so two machines on one file would clobber
    each other's status.  Each machine reads/writes only its own shard file;
    merge_shards.py reconciles them into the main manifest afterward.
    """
    if not shard:
        return SOURCE_MANIFEST
    return MANIFEST_DIR / f"e213_source_arm_cem.shard{shard}.tsv"
CONTRACT_SELFCHECK_JSON = PREFLIGHT_DIR / "e213_contract_selfcheck.json"
ARM_SCENE_AUDIT_JSON = PREFLIGHT_DIR / "e213_arm_scene_audit.json"
OVERRIDE_AUDIT_JSON = PREFLIGHT_DIR / "e213_override_audit.json"

CEM_LOG_DIR = REPO / "logs/E213/cem"

PIPELINE_SH = E208.PIPELINE_SH
SNAPSHOT_SH = E208.SNAPSHOT_SH

# --------------------------------------------------------------------------
# Case + arm model
# --------------------------------------------------------------------------
PRG_ARM = "PRG"  # g=0.0: E206 PRG base is the selected arm; E208 already CEM'd it

# Four 8-GPU machines share this /mnt.  A = local (run by Claude); B/C/D = remote
# (user launches one command each).  Rows are round-robined across shards in a
# canonical (arm, object, case, variant) order so every shard spans every arm --
# losing one machine leaves an interpretable cross-section, not "only the G1 rows".
SHARDS = ("A", "B", "C", "D")
LOCAL_SHARD = "A"
N_SHARDS = len(SHARDS)


def load_cases() -> list[dict[str, str]]:
    """The 21 RL_EXPORT_READY source rows, projected onto the fields E213 needs.

    ``base_target_task`` = the source arm-case's dcv3 task dir (``stage2b_target_task``,
    cross-checked against the ``scene_act`` parent).  The selected arm is read
    straight from the E212 export columns; nothing is recomputed.
    """
    cases: list[dict[str, str]] = []
    for row in read_tsv(SOURCE_TSV):
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
            # selected arm (E212 authority)
            "arm": row["arm"],
            "arm_experiment": row.get("arm_experiment", ""),
            "arm_gravcomp": row.get("arm_gravcomp", ""),
            "arm_scene_name": row["arm_scene_name"],
            # orig selected-arm rollout (aug-vs-orig baseline)
            "orig_scene_act": row.get("scene_act", ""),
            "orig_trajectory": row.get("trajectory", ""),
            "orig_contact_mask": row.get("contact_mask", ""),
            "orig_result_npz": row.get("cem_result_npz", ""),
            "orig_video": row.get("cem_video", ""),
            # partner (Phase B)
            "partner_case_id": row.get("partner_case_id", ""),
            "partner_person": row.get("partner_person", ""),
            "partner_person_idx": row.get("partner_person_idx", ""),
            "partner_retarget_variant_id": row.get("partner_retarget_variant_id", ""),
            "partner_generation_mode": row.get("partner_generation_mode", ""),
            "partner_trimmed_npz": row.get("partner_trimmed_npz", ""),
            "partner_omniretarget_output_npz": row.get("partner_omniretarget_output_npz", ""),
        })
    cases.sort(key=lambda c: (c["object_key"], c["case_id"]))
    return cases


def is_prg_case(case: dict[str, str]) -> bool:
    """PRG-selected (g=0.0): E208 already delivered the aug CEM on this arm."""
    return case["arm"] == PRG_ARM


def orig_selected_scene(case: dict[str, str]) -> Path:
    """The original (non-aug) task dir's selected-arm scene_act -- gravcomp authority."""
    return TASK_ROOT / case["base_target_task"] / f"{case['arm_scene_name']}.xml"


def selected_gravcomp_str(case: dict[str, str]) -> str:
    """Object gravcomp string read verbatim from the orig selected-arm scene.

    Read rather than derived so the aug sidecar carries the byte-identical value
    the original arm used ("1" for G1, "0.6"/"0.8"/"0.4" for partials) and CEM
    behaves identically.
    """
    val = scene_gravcomp(orig_selected_scene(case))
    if val in (None, "0", "0.0"):
        raise ValueError(f"{case['case_id']}: orig selected scene has no gravcomp ({val!r})")
    return val


# --------------------------------------------------------------------------
# Aug variant rows (from E208's artifacts) restricted to the 21 cases
# --------------------------------------------------------------------------
def aug_rows_by_case() -> dict[str, list[dict[str, str]]]:
    """case_id -> list of E208 'built' aug variant rows (effective, non-degenerate).

    E208's artifacts carry status 'built' for adopted+effective variants and
    'built_degenerate_offset' for chair005 (excluded here anyway).  We keep only
    'built' rows for the 21 cases.
    """
    if not ARTIFACTS_TSV.is_file():
        raise SystemExit(f"missing E208 artifacts: {rel(ARTIFACTS_TSV)}")
    keep = {c["case_id"] for c in load_cases()}
    out: dict[str, list[dict[str, str]]] = {}
    for row in read_tsv(ARTIFACTS_TSV):
        if row["case_id"] not in keep or row.get("status") != BUILT_STATUS:
            continue
        out.setdefault(row["case_id"], []).append(row)
    for cid in out:
        out[cid].sort(key=lambda r: r["aug_variant"])
    return out


# --------------------------------------------------------------------------
# Aug task dir + sidecar
# --------------------------------------------------------------------------
def aug_task_dir(aug_row: dict[str, str]) -> Path:
    return TASK_ROOT / aug_row["target_task"]


def aug_prg_scene(aug_row: dict[str, str]) -> Path:
    """The aug task's own E206-PRG base scene (E208 built it via build_arm_scenes)."""
    return aug_task_dir(aug_row) / f"{E208.SCENE_NAME}.xml"


def aug_selected_scene(aug_row: dict[str, str], case: dict[str, str]) -> Path:
    """Where the selected-arm gravcomp sidecar lives inside the aug task dir."""
    return aug_task_dir(aug_row) / f"{case['arm_scene_name']}.xml"


def build_selected_sidecar(aug_row: dict[str, str], case: dict[str, str], *, overwrite: bool = False) -> Path:
    """Write ``<arm_scene_name>.xml`` next to the aug task's E206-PRG scene.

    Single-variable diff: the aug PRG scene's `object` body gravcomp absent/0 ->
    the value read from the ORIGINAL selected-arm scene.  Idempotent: an existing
    sidecar is re-verified, not rewritten, unless overwrite.  PRG cases have no
    sidecar (g=0.0 is the base itself).
    """
    if is_prg_case(case):
        raise ValueError(f"{case['case_id']} is PRG (g=0): no sidecar; reuse E208")
    base = aug_prg_scene(aug_row)
    if not base.is_file():
        raise FileNotFoundError(f"{aug_row['target_task']}: missing aug PRG scene {base}")
    value = selected_gravcomp_str(case)
    out = aug_selected_scene(aug_row, case)
    if out.is_file() and not overwrite:
        assert_gravcomp_diff_value(base, out, value)
        return out
    tree = ET.parse(base)
    objs = [b for b in tree.getroot().iter("body") if b.get("name") == "object"]
    if len(objs) != 1:
        raise ValueError(f"expected exactly one object body in {base}, found {len(objs)}")
    if objs[0].get("gravcomp") not in (None, "0", "0.0"):
        raise ValueError(f"aug PRG scene already has gravcomp: {base}")
    objs[0].set("gravcomp", value)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    assert_gravcomp_diff_value(base, out, value)
    return out


# --------------------------------------------------------------------------
# Override naming
# --------------------------------------------------------------------------
def e208_prg_override_id(case_id: str, variant: str) -> str:
    """The E208 PRG override this E213 override inherits from."""
    return E208.override_name(case_id, variant)  # core4d_E208_{case}_aug_{variant}_lowgeom_PRG


def override_id(case_id: str, variant: str, arm: str) -> str:
    return f"core4d_{EXP_ID}_{case_id}_aug_{variant}_{arm}"


def override_path(case_id: str, variant: str, arm: str) -> Path:
    return OVERRIDE_DIR / f"{override_id(case_id, variant, arm)}.yaml"


# --------------------------------------------------------------------------
# CEM output paths
# --------------------------------------------------------------------------
def cem_out_dir(case_id: str, variant: str, arm: str, stage: str = "full") -> Path:
    return S6_DIR / "cem" / stage / f"{EXP_ID}_{case_id}_aug_{variant}_{arm}"


def result_npz(case_id: str, variant: str, arm: str, stage: str = "full") -> Path:
    return cem_out_dir(case_id, variant, arm, stage) / "trajectory_mjwp_act.npz"


def cem_log_path(case_id: str, variant: str, arm: str, stage: str = "full") -> Path:
    return CEM_LOG_DIR / stage / f"{EXP_ID}_{case_id}_aug_{variant}_{arm}.log"


def e208_prg_result_npz(case_id: str, variant: str) -> Path:
    """The already-delivered E208 PRG aug rollout (reuse target for PRG cases)."""
    return E208.cem_out_dir(case_id, variant, "full") / "trajectory_mjwp_act.npz"


# --------------------------------------------------------------------------
# Partner gap (Phase B)
# --------------------------------------------------------------------------
def partner_gap_cases() -> list[dict[str, str]]:
    """Partner cases needing aug retargets: those NOT already a source row.

    A partner that is itself one of the 21 source rows already has E208 aug
    retargets; only the partner-only cases (not in the source set) are the gap.
    """
    cases = load_cases()
    source_ids = {c["case_id"] for c in cases}
    seen: set[str] = set()
    gap: list[dict[str, str]] = []
    for c in cases:
        pid = c["partner_case_id"]
        if not pid or pid in source_ids or pid in seen:
            continue
        seen.add(pid)
        variant = c["partner_retarget_variant_id"] or "omnirt_v1"
        # partner dcv3 base task = derive from the partner_trimmed_npz path if present,
        # else construct from the partner variant + case id.
        base = _partner_base_task(c, variant)
        gap.append({
            "partner_case_id": pid,
            "object_key": c["object_key"],
            "partner_person": c["partner_person"],
            "partner_person_idx": c["partner_person_idx"],
            "retarget_variant": variant,
            "base_target_task": base,
            "partner_trimmed_npz": c["partner_trimmed_npz"],
            "partner_omniretarget_output_npz": c["partner_omniretarget_output_npz"],
            "partner_generation_mode": c["partner_generation_mode"],
            "of_source_case": c["case_id"],
        })
    gap.sort(key=lambda g: (g["object_key"], g["partner_case_id"]))
    return gap


def _partner_base_task(case: dict[str, str], variant: str) -> str:
    """dcv3 task name of the partner, taken from the partner_trimmed_npz path.

    e.g. .../holosoma_dcv3_omnirt_v2_ref_fk_chair006_20231003_1_003_p2/trimmed/...
    -> dcv3_omnirt_v2_ref_fk_chair006_20231003_1_003_p2
    """
    trimmed = case.get("partner_trimmed_npz", "")
    if trimmed:
        for part in Path(trimmed).parts:
            if part.startswith("holosoma_dcv3_"):
                return part[len("holosoma_"):]
    return f"dcv3_{variant}_ref_fk_{case['partner_case_id']}"


# --------------------------------------------------------------------------
# Manifest schema (source CEM)
# --------------------------------------------------------------------------
SOURCE_FIELDS = [
    "ordinal", "shard", "object_key", "case_id", "aug_variant", "arm", "arm_gravcomp",
    "arm_experiment", "base_target_task", "target_task",
    "e208_prg_override_id", "override_id", "override_path", "override_sha256",
    "scene_name", "selected_scene_act", "selected_scene_sha256",
    "trajectory", "trajectory_sha256", "contact_mask", "contact_mask_sha256",
    "target_scene", "base_scene_act",
    "effective_retarget_variant", "object_geom_count", "offset_band",
    "approach_trans_offset_m_max", "aug_translation", "aug_rotation_rad",
    "cem_samples", "cem_opt_steps", "cem_seed",
    "orig_result_npz",
    "result_npz", "outdir_npz", "config_act", "log",
    "execution_mode", "gpu_id", "host", "status", "failure_mode", "wall_min", "updated_at",
]
