#!/usr/bin/env python3
"""Contract single-source for the E213-export step (plan244).

Packages the user-curated E213 selected-arm object-augmentation units
(``E213_export_aug_cases.xlsx``) as partner-paired, dcv3 Holosoma-ready RL inputs.

This is the augmentation analogue of E212's mixed-arm paired export, using the
E202-export partner-re-anchor semantics:

  * source  = the E213 selected-arm aug CEM rollout (each case on its own E212
    gravcomp arm; scene_act / trajectory / contact_mask / cem_result), reused
    as-is (no new CEM);
  * partner = the OPPOSITE person's kinematic aug retarget under the IDENTICAL
    variant (trimmed NPZ), from one of two sources:
      - E213 partner_aug tree (partner-only cases, 5 variants each), OR
      - E208 source aug retarget tree (when the partner is itself a source case);
  * a common-raw-window alignment audit (reused dcv3 machinery).

Two-person object consistency is enforced DOWNSTREAM by the Holosoma exporter's
partner re-anchor (partner hands -> source object frame), verified post-reanchor
(matches E200/E202 practice). This layer only produces the paired RL INPUT and
hard-gates C4 physical failures.

The selection authority is the xlsx (SHA-pinned); the per-case selected arm and
static provenance come from E212's SHA-pinned paired export; the aug rollout
paths come from E213's merged source CEM manifest; the aug retarget trimmed NPZ
come from E213 partner_aug / E208 aug artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import openpyxl

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e213_common as C  # noqa: E402  (load_cases / manifest / artifacts / io helpers)

REPO = C.REPO

# --- authorities -------------------------------------------------------------
XLSX = C.EVAL_DIR / "aug" / "E213_export_aug_cases.xlsx"
EXPECTED_XLSX_SHA256 = "ddfc63fef83612ad0eeff361980e72d031f06dfc9269394117cf9210a4df824e"

# E212's paired export is the static-provenance + selected-arm authority; its SHA
# is already pinned in e213_common (== e209_common.EXPECTED_SOURCE_SHA256).
SEED_TSV = C.SOURCE_TSV
EXPECTED_SEED_SHA256 = C.EXPECTED_SOURCE_SHA256

# E213 merged source CEM manifest (selected-arm aug rollouts).
SOURCE_MANIFEST = C.SOURCE_MANIFEST
# E213 partner-only aug artifacts (11 cases x 5 variants).
PARTNER_MANIFEST = C.PARTNER_MANIFEST
# E208 source-side aug retarget artifacts (for partners that are themselves sources).
E208_ARTIFACTS = C.ARTIFACTS_TSV
PARTNER_AUG_ROOT = C.PARTNER_DP  # results/E213/data_preprocess/partner_aug

# --- outputs -----------------------------------------------------------------
OUT = C.S6_DIR / "export"

# --- variant naming ----------------------------------------------------------
# xlsx uses trans0/trans1/trans2/rot0/rot1; the trimmed NPZ use trans_0 .. rot_1.
VAR_UNDERSCORE = {"trans0": "trans_0", "trans1": "trans_1", "trans2": "trans_2",
                  "rot0": "rot_0", "rot1": "rot_1"}
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
    """The curated (object, case_id, arm, variant) units; SHA-pinned."""
    got = C.sha256(XLSX)
    if got != EXPECTED_XLSX_SHA256:
        raise SystemExit(f"xlsx selection authority SHA drift: {got} != {EXPECTED_XLSX_SHA256}")
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    need = ["object", "case_id", "arm", "variant"]
    if header[:4] != need:
        raise SystemExit(f"xlsx header drift: {header[:4]} != {need}")
    units: list[dict[str, str]] = []
    for r in rows[1:]:
        if not r or not r[1]:
            continue
        obj, case_id, arm, variant = (str(r[0]).strip(), str(r[1]).strip(),
                                      str(r[2]).strip(), str(r[3]).strip())
        if variant not in VAR_UNDERSCORE:
            raise SystemExit(f"{case_id}: unknown variant {variant!r}")
        units.append({"object_key": obj, "case_id": case_id, "arm": arm, "variant": variant})
    units.sort(key=lambda u: (u["object_key"], u["case_id"], u["variant"]))
    return units


# --- indices -----------------------------------------------------------------
def seed_index() -> dict[str, dict[str, str]]:
    """E212 paired export rows by case_id (static provenance + selected arm)."""
    got = C.sha256(SEED_TSV)
    if got != EXPECTED_SEED_SHA256:
        raise SystemExit(f"E212 seed authority SHA drift: {got} != {EXPECTED_SEED_SHA256}")
    return {r["case_id"]: r for r in C.read_tsv(SEED_TSV)}


def source_manifest_index() -> dict[tuple[str, str, str], dict[str, str]]:
    """E213 cem_ok aug rollouts by (case_id, aug_variant, arm)."""
    idx: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in C.read_tsv(SOURCE_MANIFEST):
        if r.get("status") != "cem_ok":
            continue
        idx[(r["case_id"], r["aug_variant"], r["arm"])] = r
    return idx


def partner_manifest_index() -> dict[str, dict[str, str]]:
    """E213 partner-only aug artifacts by partner_case_id."""
    return {r["partner_case_id"]: r for r in C.read_tsv(PARTNER_MANIFEST)}


def e208_aug_index() -> dict[tuple[str, str], dict[str, str]]:
    """E208 'built' aug rows by (case_id, aug_variant) -- source-side trimmed NPZ."""
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for r in C.read_tsv(E208_ARTIFACTS):
        if r.get("status") != "built":
            continue
        idx[(r["case_id"], r["aug_variant"])] = r
    return idx


def _parse_provenance(text: str) -> dict[str, str]:
    """'rot_0=omnirt_v2;trans_0=omnirt_v1' -> {rot_0: omnirt_v2, ...}."""
    out: dict[str, str] = {}
    for tok in (text or "").split(";"):
        tok = tok.strip()
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# --- partner trimmed resolver (two sources) ----------------------------------
def resolve_partner_aug(partner_case: str, variant_u: str, *,
                        pm_idx: dict[str, dict[str, str]],
                        e208_idx: dict[tuple[str, str], dict[str, str]],
                        source_ids: set[str]) -> dict[str, str]:
    """Locate one partner's kinematic aug retarget for ``variant_u``.

    Returns the file group + retarget_variant needed to synthesize partner
    evidence. Picks the E213 partner_aug tree for partner-only cases (choosing
    the per-variant adopted v1/v2 subtree from the manifest provenance) or the
    E208 source aug tree when the partner is itself a source case.
    """
    variant_short = variant_u.replace("_", "")  # trans_0 -> trans0
    if partner_case in pm_idx:
        row = pm_idx[partner_case]
        base_task = row["base_target_task"]  # dcv3_omnirt_vX_ref_fk_<case>
        prov = _parse_provenance(row.get("provenance", ""))
        variant_tree = prov.get(variant_u) or row["primary_variant"]
        root = PARTNER_AUG_ROOT / variant_tree / f"holosoma_{base_task}"
        source_kind = "e213_partner_aug"
    elif partner_case in source_ids:
        row = e208_idx.get((partner_case, variant_short))
        if row is None:
            raise SystemExit(f"{partner_case}/{variant_short}: no E208 built aug row")
        trimmed = C.repo_path(row["trimmed_npz"])
        root = trimmed.parent.parent
        variant_tree = row["effective_retarget_variant"]
        base_task = root.name[len("holosoma_"):] if root.name.startswith("holosoma_") else root.name
        source_kind = "e208_source_aug"
    else:
        raise SystemExit(f"{partner_case}: partner is neither partner-only nor a source case")

    trimmed_dir = root / "trimmed"
    hits = sorted(trimmed_dir.glob(f"*_{variant_u}.npz")) if trimmed_dir.is_dir() else []
    hits = [h for h in hits if h.name.endswith(f"_{variant_u}.npz")]
    if len(hits) != 1:
        raise SystemExit(f"{partner_case}/{variant_u}: expected 1 trimmed npz in {trimmed_dir}, found {len(hits)}")
    trimmed = hits[0]
    holo = trimmed.name[: -(len(variant_u) + 1 + 4)]  # strip _<variant>.npz
    converted = root / "converted" / f"{holo}.npz"
    retargeted = root / "retargeted" / f"{holo}_{variant_u}.npz"
    trim_json = root / "trim_window.json"
    for label, p in (("converted", converted), ("retargeted", retargeted),
                     ("trimmed", trimmed), ("trim_window.json", trim_json)):
        if not p.is_file():
            raise SystemExit(f"{partner_case}/{variant_u}: missing {label}: {p}")
    return {
        "source_kind": source_kind,
        "base_task": base_task,
        "retarget_variant_id": variant_tree,
        "holosoma_case_root": str(root),
        "converted_npz": str(converted),
        "retargeted_npz": str(retargeted),
        "trimmed_npz": str(trimmed),
        "trim_window_json": str(trim_json),
    }


E208_DP = REPO / "workspace/core4d/results/E208/data_preprocess"


def source_aug_evidence(manifest_row: dict[str, str]) -> dict[str, str]:
    """Source person's own aug retarget trim window (E208 tree), for alignment.

    The source aug kinematic retarget + its ``trim_window.json`` live in E208's
    data_preprocess under the *effective* retarget variant parent dir; the
    holosoma subdir keeps the case's *primary* variant name even when the aug
    variant was rescued to v2 (so target_task's variant tag is unreliable). The
    trim window is per-case (shared across variants, fixed-window trim).
    """
    case_id = manifest_row["case_id"]
    eff = manifest_row.get("effective_retarget_variant", "") or "*"
    hits = sorted(E208_DP.glob(f"{eff}/holosoma_dcv3_*_ref_fk_{case_id}"))
    hits = [h for h in hits if (h / "trim_window.json").is_file()]
    if len(hits) != 1:
        raise SystemExit(
            f"{case_id} ({eff}): cannot locate source aug trim_window.json ({len(hits)} candidates)")
    root = hits[0]
    return {"holosoma_case_root": str(root), "trim_window_json": str(root / "trim_window.json")}


def partner_evidence(source_row: dict[str, Any], partner_case: str,
                     aug: dict[str, str]) -> dict[str, str]:
    """Stage2b-shaped evidence for dcv3 ``build_partner_row``.

    Identity fields must match the source row; artifact paths point at the
    resolved aug retarget group. ``omniretarget_output_npz`` aliases the
    retargeted NPZ (the E202-export convention).
    """
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
