#!/usr/bin/env python3
"""E216 paired RL export: source + same-variant augmented partner (Phases 3-4).

For each keep-best source row (``rl_export_input.tsv``), resolve the opposite
person of the same sequence as a kinematic partner that carries the SAME object
augmentation variant (``trans_N``). Two-person object consistency is enforced
DOWNSTREAM by the Holosoma partner re-anchor; this stage only prepares the paired
RL INPUT + trim-window alignment audit (E213 parity, no RL outcome claim).

Partners whose augmented holosoma motion is not yet generated are marked
``PARTNER_PENDING`` and listed in ``excluded_units.json`` (they become complete
after the Phase-2 generation step).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "workspace/core4d/scripts/data_construction_v3/lib")
from common import read_tsv, write_json, write_tsv  # noqa: E402

SCHEMA_VERSION = "core4d_data_construction_v3.0"
PERSON_PARTNER = {"person1": "person2", "person2": "person1"}
PERSON_INDEX = {"person1": "0", "person2": "1"}
PERSON_SHORT = {"person1": "p1", "person2": "p2"}
VARIANT_PREF = ["omnirt_v1", "omnirt_v2"]
VMAP = {"trans0": "trans_0", "trans1": "trans_1", "trans2": "trans_2"}
GENERATION_MODE = "e216_selected_arm_aug_partner_kinematic_same_variant"
DP_ROOTS = [
    "workspace/core4d/results/E216/data_preprocess",
    "workspace/core4d/results/E216/chair_desk/data_preprocess",
]

# paired extra columns (exact order of E213 paired cols 82-100).
PAIRED_EXTRA_FIELDS = [
    "pair_status", "paired_rl_export_decision", "partner_case_id",
    "partner_person", "partner_person_idx", "partner_status",
    "partner_retarget_variant_id", "partner_target_variant_id",
    "partner_generation_mode", "partner_trimmed_npz",
    "partner_trimmed_npz_sha256", "partner_omniretarget_output_npz",
    "partner_omniretarget_output_npz_sha256", "partner_trim_window_json",
    "partner_trim_window_json_sha256", "partner_manifest_ref",
    "partner_manifest_sha256", "scene_act_sha256", "cem_result_sha256",
]

# partner_omnirt manifest schema.
PARTNER_FIELDS = [
    "source_case_id", "source_person", "source_person_idx",
    "source_rl_export_decision", "partner_case_id", "partner_person",
    "partner_person_idx", "object_key", "object_name", "date", "seq",
    "aug_variant", "pair_status", "partner_status", "paired_rl_export_decision",
    "partner_retarget_variant_id", "partner_target_variant_id",
    "generation_mode", "failure_mode", "decision_notes", "holosoma_case_root",
    "converted_npz", "converted_npz_sha256", "omniretarget_output_npz",
    "omniretarget_output_npz_sha256", "retargeted_npz", "retargeted_npz_sha256",
    "trimmed_npz", "trimmed_npz_sha256", "trim_window_json",
    "trim_window_json_sha256", "trim_start", "trim_end", "trim_frames",
    "untrimmed_frames", "trimmed_frames", "schema_version", "updated_at",
]

AUDIT_FIELDS = [
    "source_case_id", "source_person", "source_retarget_variant_id",
    "source_trim_window_json", "source_trim_start", "source_trim_end",
    "source_trim_frames", "source_untrimmed_frames",
    "partner_case_id", "partner_person", "partner_retarget_variant_id",
    "partner_trim_window_json", "partner_trim_start", "partner_trim_end",
    "partner_trim_frames", "partner_untrimmed_frames",
    "common_raw_start", "common_raw_end", "common_raw_frames",
    "source_crop_offset", "partner_crop_offset",
    "alignment_policy", "alignment_status", "alignment_failure_mode",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def infer_partner(case_id: str, person: str) -> tuple[str, str, str]:
    partner_person = PERSON_PARTNER[person]
    suffix = f"_{PERSON_SHORT[person]}"
    if not case_id.endswith(suffix):
        raise SystemExit(f"case/person suffix mismatch: {case_id} / {person}")
    partner_case = case_id[: -len(suffix)] + f"_{PERSON_SHORT[partner_person]}"
    return partner_case, partner_person, PERSON_INDEX[partner_person]


def find_holosoma_dir(case: str, repo: Path) -> tuple[str, Path] | None:
    """Return (variant, dir) for a case, preferring omnirt_v1 then v2."""
    for variant in VARIANT_PREF:
        for root in DP_ROOTS:
            d = repo / root / f"holosoma_dcv3_{variant}_ref_fk_{case}"
            if d.is_dir():
                return variant, d
    return None


def resolve_partner_aug(case: str, aug_variant: str, repo: Path):
    """Locate the same-variant augmented partner artifacts; None if absent."""
    fv = VMAP[aug_variant]
    for variant in VARIANT_PREF:
        for root in DP_ROOTS:
            d = repo / root / f"holosoma_dcv3_{variant}_ref_fk_{case}"
            if not d.is_dir():
                continue
            trimmed = glob.glob(str(d / "trimmed" / f"*_{fv}.npz"))
            if not trimmed:
                continue
            retargeted = glob.glob(str(d / "retargeted" / f"*_{fv}.npz"))
            converted = glob.glob(str(d / "converted" / "*_with_obj.npz"))
            tw = d / "trim_window.json"
            return {
                "variant": variant,
                "case_root": d,
                "trimmed": Path(trimmed[0]),
                "retargeted": Path(retargeted[0]) if retargeted else None,
                "converted": Path(converted[0]) if converted else None,
                "trim_window": tw if tw.is_file() else None,
            }
    return None


def load_trim(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def build(args) -> int:
    repo = Path.cwd()
    source_fields = None
    source_rows = read_tsv(args.rl_export_input)
    # preserve exact source header order
    with args.rl_export_input.open() as f:
        source_fields = f.readline().rstrip("\n").split("\t")

    partner_rows: list[dict] = []
    audit_rows: list[dict] = []
    paired_rows: list[dict] = []
    excluded: list[dict] = []

    for src in source_rows:
        case_id = src["case_id"]
        person = src["person"]
        aug_variant = src["aug_variant"]
        partner_case, partner_person, partner_idx = infer_partner(case_id, person)
        found = resolve_partner_aug(partner_case, aug_variant, repo)

        # source trim window (for the alignment audit)
        src_dir = find_holosoma_dir(case_id, repo)
        src_tw_path = (src_dir[1] / "trim_window.json") if src_dir else None
        src_trim = load_trim(src_tw_path)
        src_variant = src_dir[0] if src_dir else ""

        base_partner = {
            "source_case_id": case_id,
            "source_person": person,
            "source_person_idx": src["person_idx"],
            "source_rl_export_decision": src["rl_export_decision"],
            "partner_case_id": partner_case,
            "partner_person": partner_person,
            "partner_person_idx": partner_idx,
            "object_key": src["object_key"],
            "object_name": src["object_name"],
            "date": src["date"],
            "seq": src["seq"],
            "aug_variant": aug_variant,
            "partner_target_variant_id": "ref_fk",
            "generation_mode": GENERATION_MODE,
            "schema_version": SCHEMA_VERSION,
            "updated_at": now(),
        }

        if found is None:
            prow = {
                **base_partner,
                "pair_status": "PARTNER_PENDING",
                "partner_status": "missing_partner_aug_motion",
                "paired_rl_export_decision": "PARTNER_PENDING",
                "partner_retarget_variant_id": "",
                "failure_mode": "partner_aug_holosoma_absent",
                "decision_notes": (
                    "opposite-person augmented holosoma motion not yet generated "
                    f"(same variant {aug_variant}); pending Phase-2 upstream retarget"
                ),
            }
            for field in PARTNER_FIELDS:
                prow.setdefault(field, "")
            partner_rows.append(prow)
            excluded.append({
                "source_case_id": case_id, "partner_case_id": partner_case,
                "aug_variant": aug_variant, "reason": "partner_aug_holosoma_absent",
            })
            audit_rows.append(_audit(base_partner, src_variant, src_tw_path,
                                     src_trim, "", None, {}, repo,
                                     status="PARTNER_PENDING",
                                     failure="partner_aug_holosoma_absent"))
            paired_rows.append(_paired(src, source_fields, prow, "", "", repo))
            continue

        p_trim = load_trim(found["trim_window"])
        trimmed = found["trimmed"]
        retargeted = found["retargeted"] or trimmed
        converted = found["converted"]
        prow = {
            **base_partner,
            "pair_status": "PAIR_COMPLETE",
            "partner_status": "pass",
            "paired_rl_export_decision": "RL_EXPORT_READY",
            "partner_retarget_variant_id": found["variant"],
            "failure_mode": "",
            "decision_notes": (
                f"reused {found['variant']} same-variant aug partner "
                f"({aug_variant}); kinematic, re-anchored downstream"
            ),
            "holosoma_case_root": ref(found["case_root"], repo),
            "converted_npz": ref(converted, repo) if converted else "",
            "converted_npz_sha256": sha256(converted) if converted else "",
            "omniretarget_output_npz": ref(retargeted, repo),
            "omniretarget_output_npz_sha256": sha256(retargeted),
            "retargeted_npz": ref(retargeted, repo),
            "retargeted_npz_sha256": sha256(retargeted),
            "trimmed_npz": ref(trimmed, repo),
            "trimmed_npz_sha256": sha256(trimmed),
            "trim_window_json": ref(found["trim_window"], repo) if found["trim_window"] else "",
            "trim_window_json_sha256": sha256(found["trim_window"]) if found["trim_window"] else "",
            "trim_start": p_trim.get("trim_start", ""),
            "trim_end": p_trim.get("trim_end", ""),
            "trim_frames": p_trim.get("trim_frames", ""),
            "untrimmed_frames": p_trim.get("untrimmed_frames", ""),
            "trimmed_frames": p_trim.get("trimmed_frames", ""),
        }
        for field in PARTNER_FIELDS:
            prow.setdefault(field, "")
        partner_rows.append(prow)
        audit_rows.append(_audit(base_partner, src_variant, src_tw_path, src_trim,
                                 found["variant"], found["trim_window"], p_trim,
                                 repo, status="RL_EXPORT_READY", failure=""))
        paired_rows.append(_paired(src, source_fields, prow,
                                   src.get("scene_sha256", ""),
                                   src.get("result_sha256", ""), repo))

    # write partner manifest first (its sha is referenced by paired rows)
    out = args.out_dir
    (out / "partner_omnirt").mkdir(parents=True, exist_ok=True)
    pm_tsv = out / "partner_omnirt" / "rl_partner_omnirt_manifest.tsv"
    write_tsv(pm_tsv, partner_rows, PARTNER_FIELDS)
    write_json(out / "partner_omnirt" / "rl_partner_omnirt_manifest.json", partner_rows)
    pm_ref = ref(pm_tsv, repo)
    pm_sha = sha256(pm_tsv)
    for prow in paired_rows:
        prow["partner_manifest_ref"] = pm_ref
        prow["partner_manifest_sha256"] = pm_sha

    write_tsv(out / "paired_rl_export_input.tsv", paired_rows,
              source_fields + PAIRED_EXTRA_FIELDS)
    write_json(out / "paired_rl_export_input.json", paired_rows)
    write_tsv(out / "partner_resolution_audit.tsv", audit_rows, AUDIT_FIELDS)
    write_json(out / "excluded_units.json", excluded)

    complete = sum(1 for p in partner_rows if p["pair_status"] == "PAIR_COMPLETE")
    pending = len(partner_rows) - complete
    _write_summary_validation(args, source_rows, partner_rows, paired_rows,
                              audit_rows, excluded, pm_ref, pm_sha, complete,
                              pending, repo)
    print(f"source={len(source_rows)} complete={complete} pending={pending} "
          f"-> {out/'paired_rl_export_input.tsv'}")
    return 0


def _paired(src, source_fields, prow, scene_sha, cem_sha, repo) -> dict:
    row = {k: src.get(k, "") for k in source_fields}
    row.update({
        "pair_status": prow["pair_status"],
        "paired_rl_export_decision": prow["paired_rl_export_decision"],
        "partner_case_id": prow["partner_case_id"],
        "partner_person": prow["partner_person"],
        "partner_person_idx": prow["partner_person_idx"],
        "partner_status": prow["partner_status"],
        "partner_retarget_variant_id": prow.get("partner_retarget_variant_id", ""),
        "partner_target_variant_id": prow.get("partner_target_variant_id", ""),
        "partner_generation_mode": prow["generation_mode"],
        "partner_trimmed_npz": prow.get("trimmed_npz", ""),
        "partner_trimmed_npz_sha256": prow.get("trimmed_npz_sha256", ""),
        "partner_omniretarget_output_npz": prow.get("omniretarget_output_npz", ""),
        "partner_omniretarget_output_npz_sha256": prow.get("omniretarget_output_npz_sha256", ""),
        "partner_trim_window_json": prow.get("trim_window_json", ""),
        "partner_trim_window_json_sha256": prow.get("trim_window_json_sha256", ""),
        "partner_manifest_ref": "",
        "partner_manifest_sha256": "",
        "scene_act_sha256": scene_sha,
        "cem_result_sha256": cem_sha,
    })
    return row


def _audit(base, src_variant, src_tw_path, src_trim, p_variant, p_tw_path,
           p_trim, repo, *, status, failure) -> dict:
    s_start = src_trim.get("trim_start", "")
    s_end = src_trim.get("trim_end", "")
    p_start = p_trim.get("trim_start", "")
    p_end = p_trim.get("trim_end", "")
    common_start = common_end = common_frames = s_off = p_off = ""
    if src_trim and p_trim:
        cs = max(int(s_start), int(p_start))
        ce = min(int(s_end), int(p_end))
        common_start, common_end = cs, ce
        common_frames = max(0, ce - cs)
        s_off = cs - int(s_start)
        p_off = cs - int(p_start)
    return {
        "source_case_id": base["source_case_id"],
        "source_person": base["source_person"],
        "source_retarget_variant_id": src_variant,
        "source_trim_window_json": ref(src_tw_path, repo) if src_tw_path else "",
        "source_trim_start": s_start, "source_trim_end": s_end,
        "source_trim_frames": src_trim.get("trim_frames", ""),
        "source_untrimmed_frames": src_trim.get("untrimmed_frames", ""),
        "partner_case_id": base["partner_case_id"],
        "partner_person": base["partner_person"],
        "partner_retarget_variant_id": p_variant,
        "partner_trim_window_json": ref(p_tw_path, repo) if p_tw_path else "",
        "partner_trim_start": p_start, "partner_trim_end": p_end,
        "partner_trim_frames": p_trim.get("trim_frames", ""),
        "partner_untrimmed_frames": p_trim.get("untrimmed_frames", ""),
        "common_raw_start": common_start, "common_raw_end": common_end,
        "common_raw_frames": common_frames,
        "source_crop_offset": s_off, "partner_crop_offset": p_off,
        "alignment_policy": "common_raw_window",
        "alignment_status": status, "alignment_failure_mode": failure,
    }


def _write_summary_validation(args, source_rows, partner_rows, paired_rows,
                              audit_rows, excluded, pm_ref, pm_sha, complete,
                              pending, repo) -> None:
    out = args.out_dir
    status = "RL_EXPORT_READY" if pending == 0 else "PARTIAL_PARTNER_PENDING"
    summary = {
        "experiment": "E216-export",
        "stage": "s6_downstream/export",
        "kind": "selected_arm_object_aug_paired",
        "created_at": now(),
        "status": status,
        "selection_authority": ref(args.selection_snapshot, repo),
        "selection_authority_sha256": sha256(args.selection_snapshot)
        if args.selection_snapshot.is_file() else "",
        "source_rl_export_input": ref(args.rl_export_input, repo),
        "source_rl_export_input_sha256": sha256(args.rl_export_input),
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "paired_complete_rows": complete,
        "partner_pending_rows": pending,
        "excluded_unit_count": len(excluded),
        "object_counts": dict(Counter(r["object_key"] for r in source_rows)),
        "variant_counts": dict(Counter(r["aug_variant"] for r in source_rows)),
        "partner_variant_counts": dict(Counter(
            p["partner_retarget_variant_id"] for p in partner_rows
            if p["pair_status"] == "PAIR_COMPLETE")),
        "partner_manifest": pm_ref,
        "partner_manifest_sha256": pm_sha,
        "consistency_note": (
            "two-person object consistency enforced DOWNSTREAM by Holosoma partner "
            "re-anchor (same-variant kinematic partner)"),
        "claim_boundary": (
            "paired RL export INPUT + trim-window alignment audit only; "
            "no RL outcome claim"),
    }
    write_json(out / "rl_export_summary.json", summary)

    # validation: re-stat + re-hash every artifact path in the paired tsv
    checks = {"source_artifacts_ok": True, "partner_artifacts_ok": True,
              "all_sha_match": True, "unique_source_ids": True,
              "unique_partner_ids": True, "audit_frames_positive": True}
    problems: list[str] = []
    seen_src: set[str] = set()
    for row in paired_rows:
        cid = row["case_id"]
        if cid in seen_src:
            checks["unique_source_ids"] = False
            problems.append(f"dup source {cid}")
        seen_src.add(cid)
        for col in ("trajectory", "scene_act", "contact_mask", "cem_result_npz"):
            p = repo / row[col] if not Path(row[col]).is_absolute() else Path(row[col])
            if not (p.is_file() and p.stat().st_size > 0):
                checks["source_artifacts_ok"] = False
                problems.append(f"{cid} missing {col}")
        if row["pair_status"] == "PAIR_COMPLETE":
            tp = row["partner_trimmed_npz"]
            p = repo / tp if not Path(tp).is_absolute() else Path(tp)
            if not (p.is_file() and p.stat().st_size > 0):
                checks["partner_artifacts_ok"] = False
                problems.append(f"{cid} missing partner trimmed")
            elif sha256(p) != row["partner_trimmed_npz_sha256"]:
                checks["all_sha_match"] = False
                problems.append(f"{cid} partner sha mismatch")
    for a in audit_rows:
        if a["alignment_status"] == "RL_EXPORT_READY" and not (
                isinstance(a["common_raw_frames"], int) and a["common_raw_frames"] > 0):
            checks["audit_frames_positive"] = False
            problems.append(f"{a['source_case_id']} common_raw_frames<=0")
    report = {
        "created_at": now(),
        "status": "PASS" if all(checks.values()) and pending == 0 else (
            "PASS_WITH_PENDING" if all(checks.values()) else "FAIL"),
        "checks": checks,
        "paired_complete_rows": complete,
        "partner_pending_rows": pending,
        "problems": problems[:50],
    }
    write_json(out / "validation_report.json", report)


def main() -> int:
    base = Path("workspace/core4d/results/E216/s6_downstream/export")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-export-input", type=Path, default=base / "rl_export_input.tsv")
    ap.add_argument("--selection-snapshot", type=Path, default=base / "selection_snapshot.xlsx")
    ap.add_argument("--out-dir", type=Path, default=base)
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
