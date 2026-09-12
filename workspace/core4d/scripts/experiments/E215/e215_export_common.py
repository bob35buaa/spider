#!/usr/bin/env python3
"""Contract single-source for the E215-export step (plan247 RL-export).

Packages the user-curated E215 ROTATION-augmentation units
(``E215_rot_aug_export.xlsx``) as partner-paired, dcv3 Holosoma-ready RL inputs.
This is the rotation analogue of E213-export (selected-arm aug), reusing the
identical E202/E213 partner-re-anchor semantics:

  * source  = the E215 rot aug CEM rollout on the case's selected arm_group
    (scene_act / trajectory / contact_mask / cem_result), reused as-is (no CEM);
  * partner = the OPPOSITE person's kinematic aug retarget under the IDENTICAL
    rot variant (trimmed NPZ), from one of two sources:
      - the E215 source retarget tree, when the partner is itself a delivered
        E215 source case (rot trimmed npz already on disk), OR
      - the E215 partner_aug tree (build_partner_aug.py) for gap partners;
  * a common-raw-window alignment audit (reused dcv3 machinery).

Two-person object consistency is enforced DOWNSTREAM by the Holosoma exporter's
partner re-anchor (partner hands -> source object frame), verified post-reanchor
([[two-person-aug-partner-reanchor]]). This layer only produces the paired RL
INPUT and hard-gates C4 physical failures.

Nothing numeric is re-typed: the source rollout paths come from E215's 3 CEM
shard manifests; case metadata from each base task's task_info.json; the dcv3
partner row / paired schema from the E206 exporter's dcv3 adapter (imported by
the export script exactly as E213 does).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import openpyxl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e215_common as C  # noqa: E402  (case registry / manifests / io + load_case_meta)

REPO = C.REPO

# --- authorities -------------------------------------------------------------
XLSX = C.EVAL_DIR / "E215_rot_aug_export.xlsx"
EXPECTED_XLSX_SHA256 = "08f3e7d13dd4bb2394b7352f171798bf609dcb3845070d5a487a5512eec201d7"

# E215's 3 CEM shard manifests are the source rollout authority.
SHARD_MANIFESTS = [C.MANIFEST_DIR / f"e215_priority_manifest.shard{s}.tsv" for s in ("A", "B", "C")]
DONE_STATUS = {"cem_ok", "run_complete_pending_eval"}

# --- outputs -----------------------------------------------------------------
OUT = C.S6_DIR / "export"
PARTNER_AUG_ROOT = C.DP / "partner_aug"           # build_partner_aug.py target
PARTNER_MANIFEST = C.DP / "manifests" / "e215_partner_aug_artifacts.tsv"

# --- variant naming ----------------------------------------------------------
VAR_UNDERSCORE = {"rot0": "rot_0", "rot1": "rot_1"}   # E215 is rotation-only
TARGET_VARIANT = "ref_fk"

_PERSON_FROM_SUFFIX = {"p1": "person1", "p2": "person2"}
_PERSON_IDX = {"person1": "0", "person2": "1"}
_FLIP = {"p1": "p2", "p2": "p1"}


def flip_case(case_id: str) -> str:
    base, _, suffix = case_id.rpartition("_")
    return f"{base}_{_FLIP[suffix]}"


def person_of(case_id: str) -> tuple[str, str]:
    person = _PERSON_FROM_SUFFIX[case_id.rsplit("_", 1)[-1]]
    return person, _PERSON_IDX[person]


# --- selection authority (xlsx) ----------------------------------------------
def load_units() -> list[dict[str, str]]:
    """The curated (object_key, case_id, arm_group, aug_variant) units; SHA-pinned."""
    got = C.sha256(XLSX)
    if got != EXPECTED_XLSX_SHA256:
        raise SystemExit(f"xlsx selection authority SHA drift: {got} != {EXPECTED_XLSX_SHA256}")
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    need = ["object_key", "case_id", "arm_group", "aug_variant"]
    if header[:4] != need:
        raise SystemExit(f"xlsx header drift: {header[:4]} != {need}")
    units: list[dict[str, str]] = []
    for r in rows[1:]:
        if not r or not r[1]:
            continue
        obj, case_id, arm_group, variant = (str(r[0]).strip(), str(r[1]).strip(),
                                            str(r[2]).strip(), str(r[3]).strip())
        if variant not in VAR_UNDERSCORE:
            raise SystemExit(f"{case_id}: unknown/unsupported variant {variant!r} (rot only)")
        units.append({"object_key": obj, "case_id": case_id,
                      "arm_group": arm_group, "variant": variant})
    units.sort(key=lambda u: (u["object_key"], u["case_id"], u["variant"]))
    return units


# --- source rollout index (3 shard manifests) --------------------------------
def source_manifest_index() -> dict[tuple[str, str], dict[str, str]]:
    """E215 finished rot rollouts by (case_id, aug_variant). Each case has one arm_group."""
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for mf in SHARD_MANIFESTS:
        if not mf.is_file():
            continue
        for r in C.read_tsv(mf):
            if r.get("status") in DONE_STATUS:
                idx[(r["case_id"], r["aug_variant"])] = r
    return idx


def source_meta(man: dict[str, str]) -> dict[str, str]:
    """Static provenance for a source unit from its base task_info + manifest."""
    base = man["base_target_task"]
    meta = C.load_case_meta(base)   # date/seq/person/object_name/object_model_rel/source_scene
    person, person_idx = person_of(man["case_id"])
    return {
        "case_id": man["case_id"], "object_key": man["object_key"],
        "object_name": meta["object_name"], "date": meta["date"], "seq": meta["seq"],
        "person": person, "person_idx": person_idx,
        "object_model_rel": meta["object_model_rel"],
        "target_scene": man.get("target_scene", meta.get("source_scene", "")),
        "base_target_task": base,
        "effective_retarget_variant": man.get("effective_retarget_variant", "") or C.RETARGET_VARIANT,
        "arm_group": man.get("arm_group", ""),
        "arm_scene_name": Path(man.get("scene_act", "")).stem,
    }


# --- retarget-tree resolution ------------------------------------------------
def _e215_source_holo(case_id: str) -> Path | None:
    """The E215 source retarget dir for a delivered case (glob v1/v2), else None."""
    hits = sorted(C.DP.glob(f"holosoma_dcv3_omnirt_v*_ref_fk_{case_id}"))
    hits = [h for h in hits if (h / "trim_window.json").is_file()]
    return hits[-1] if hits else None   # prefer v2 if both (lexicographic last)


def _gap_partner_holo(partner_case: str) -> Path | None:
    """The E215 partner_aug retarget dir for a gap partner (glob variant subtrees)."""
    if not PARTNER_AUG_ROOT.is_dir():
        return None
    hits = sorted(PARTNER_AUG_ROOT.glob(f"*/holosoma_dcv3_omnirt_v*_ref_fk_{partner_case}"))
    hits = [h for h in hits if (h / "trim_window.json").is_file()]
    return hits[-1] if hits else None


def is_delivered_source(case_id: str) -> bool:
    return _e215_source_holo(case_id) is not None


def _aug_group_from_root(root: Path, variant_u: str) -> dict[str, str]:
    """Assemble converted/retargeted/trimmed/trim_window paths for one variant."""
    base_task = root.name[len("holosoma_"):] if root.name.startswith("holosoma_") else root.name
    variant_id = "omnirt_v2" if "_v2_" in root.name else "omnirt_v1"
    trimmed_dir = root / "trimmed"
    hits = sorted(trimmed_dir.glob(f"*_{variant_u}.npz")) if trimmed_dir.is_dir() else []
    if len(hits) != 1:
        raise SystemExit(f"{root.name}/{variant_u}: expected 1 trimmed npz, found {len(hits)}")
    trimmed = hits[0]
    holo = trimmed.name[: -(len(variant_u) + 1 + 4)]   # strip _<variant>.npz
    converted = root / "converted" / f"{holo}.npz"
    retargeted = root / "retargeted" / f"{holo}_{variant_u}.npz"
    trim_json = root / "trim_window.json"
    for label, p in (("converted", converted), ("retargeted", retargeted),
                     ("trimmed", trimmed), ("trim_window.json", trim_json)):
        if not p.is_file():
            raise SystemExit(f"{root.name}/{variant_u}: missing {label}: {p}")
    return {
        "base_task": base_task, "retarget_variant_id": variant_id,
        "holosoma_case_root": str(root),
        "converted_npz": str(converted), "retargeted_npz": str(retargeted),
        "trimmed_npz": str(trimmed), "trim_window_json": str(trim_json),
    }


def resolve_partner_aug(partner_case: str, variant_u: str) -> dict[str, str]:
    """Locate one partner's kinematic rot retarget for ``variant_u``.

    Prefers the E215 source tree (partner is a delivered case), else the E215
    partner_aug tree (gap partner built by build_partner_aug.py)."""
    root = _e215_source_holo(partner_case)
    source_kind = "e215_source_retarget"
    if root is None:
        root = _gap_partner_holo(partner_case)
        source_kind = "e215_partner_aug"
    if root is None:
        raise SystemExit(f"{partner_case}: no rot retarget in E215 source or partner_aug tree "
                         f"(run build_partner_aug.py for gap partners)")
    aug = _aug_group_from_root(root, variant_u)
    aug["source_kind"] = source_kind
    return aug


def source_aug_evidence(man: dict[str, str]) -> dict[str, str]:
    """Source person's own aug retarget trim window (E215 source tree), for alignment."""
    root = _e215_source_holo(man["case_id"])
    if root is None:
        raise SystemExit(f"{man['case_id']}: no E215 source retarget dir for alignment")
    return {"holosoma_case_root": str(root), "trim_window_json": str(root / "trim_window.json")}


def partner_evidence(source_row: dict[str, Any], partner_case: str,
                     aug: dict[str, str]) -> dict[str, str]:
    """Stage2b-shaped evidence for dcv3 ``build_partner_row`` (E202/E213 convention)."""
    p_person, p_idx = person_of(partner_case)
    return {
        "case_id": partner_case,
        "object_key": source_row["object_key"],
        "object_name": source_row["object_name"],
        "date": source_row["date"],
        "seq": source_row["seq"],
        "person": p_person,
        "person_idx": p_idx,
        "stage2b_status": "pass",
        "retarget_variant_id": aug["retarget_variant_id"],
        "target_variant_id": TARGET_VARIANT,
        "target_task": aug["base_task"],
        "holosoma_case_root": aug["holosoma_case_root"],
        "trim_window_json": aug["trim_window_json"],
        "converted_npz": aug["converted_npz"],
        "omniretarget_output_npz": aug["retargeted_npz"],
        "retargeted_npz": aug["retargeted_npz"],
        "trimmed_npz": aug["trimmed_npz"],
    }


# --- gap partner enumeration (for build_partner_aug.py) ----------------------
def partner_gap_cases() -> list[dict[str, str]]:
    """Partner cases needing a rot retarget: those NOT already a delivered source.

    Seeds from E199_DP (box) / E202_DP (bucket) '_original' retarget + trim_window.
    """
    units = load_units()
    gap: dict[str, dict[str, str]] = {}
    for u in units:
        partner = flip_case(u["case_id"])
        if is_delivered_source(partner) or partner in gap:
            continue
        seed_root = _seed_source_root(partner)
        gap[partner] = {
            "partner_case_id": partner,
            "object_key": u["object_key"],
            "of_source_case": u["case_id"],
            "seed_holosoma_root": str(seed_root) if seed_root else "",
            "seed_variant": ("omnirt_v2" if seed_root and "_v2_" in seed_root.name
                             else "omnirt_v1") if seed_root else "",
            "base_target_task": (seed_root.name[len("holosoma_"):]
                                 if seed_root else f"dcv3_omnirt_v1_ref_fk_{partner}"),
        }
    return [gap[k] for k in sorted(gap)]


def _seed_source_root(case_id: str) -> Path | None:
    """The E199_DP (box) / E202_DP (bucket) holosoma dir with an '_original' seed."""
    roots = [C.E202_DP] if case_id.startswith("bucket") else [C.E199_DP, C.E202_DP]
    for base in roots:
        if not base.is_dir():
            continue
        for d in sorted(base.glob(f"holosoma_dcv3_omnirt_v*_ref_fk_{case_id}")):
            ret = list((d / "retargeted").glob("*_original.npz")) if (d / "retargeted").is_dir() else []
            if ret and (d / "trim_window.json").is_file():
                return d
    return None
