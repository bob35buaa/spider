#!/usr/bin/env python3
"""Build a small git-trackable seed registry from trusted historical results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import SCHEMA_VERSION, sha256_file, timestamp, write_tsv
from update_case_state_registry import FIELDS, current_decision_from_registry, normalize_row, row_key


SPIDER_REPO = SCRIPT_ROOT.parents[3]


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def first_existing(repo: Path, rel_paths: list[str]) -> Path:
    for rel_path in rel_paths:
        path = repo / rel_path
        if path.is_file():
            return path
    return repo / rel_paths[0]


def rel(path: Path | str) -> str:
    text = str(path)
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute():
        try:
            return str(p.relative_to(SPIDER_REPO))
        except ValueError:
            return text
    return text


def parse_case_id(case_id: str) -> dict[str, str]:
    parts = case_id.split("_")
    if len(parts) >= 5 and parts[0] in {"e091", "d003"}:
        object_key = parts[1]
        date = parts[2]
        seq = parts[3]
        person_token = parts[4]
    elif len(parts) >= 2 and parts[1].startswith("person"):
        object_key = parts[0]
        date = ""
        seq = ""
        person_token = parts[1]
    else:
        object_key = parts[0] if parts else ""
        date = ""
        seq = ""
        person_token = ""
    person_idx = person_token[1:] if person_token.startswith("p") else person_token.replace("person", "")
    return {
        "object_key": object_key,
        "object_name": object_key.replace("box", "Box") if object_key.startswith("box") else object_key,
        "date": date,
        "seq": seq,
        "person": f"person{person_idx}" if person_idx else "",
        "person_idx": person_idx,
    }


def target_variant(row: dict[str, str]) -> str:
    text = " ".join([row.get("target_route", ""), row.get("route", ""), row.get("variant", "")]).lower()
    if "fingertip" in text:
        return "fingertip_aware"
    if "adaptive" in text or "hbproj" in text:
        return "adaptive"
    return "ref_fk"


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "work"}


def float_value(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def root_npz_for(row: dict[str, str], summary_csv: Path) -> str:
    root_npz = row.get("root_npz_path", "")
    if root_npz:
        return rel(root_npz)
    candidate = summary_csv.parent / f"{row.get('variant', '')}.npz"
    if candidate.is_file():
        return rel(candidate)
    return rel(row.get("npz_path", ""))


def video_for(row: dict[str, str], summary_csv: Path) -> str:
    video = row.get("video_path", "")
    if video:
        return rel(video)
    candidate = summary_csv.parent / f"{row.get('variant', '')}_full.mp4"
    if candidate.is_file():
        return rel(candidate)
    candidate = summary_csv.parent / f"{row.get('variant', '')}.mp4"
    if candidate.is_file():
        return rel(candidate)
    return ""


def cem_pass(row: dict[str, str]) -> bool:
    for key in [
        "E081_success_legobj_strict_proxy",
        "E081_success_numeric",
        "E080_success_numeric",
        "E079_success_numeric",
    ]:
        if row.get(key, ""):
            return boolish(row.get(key, ""))
    strict = row.get("work_status_lowerbody_strict", "")
    if strict:
        return strict == "WORK" or boolish(row.get("advance_to_rl_lowerbody_strict", ""))
    if row.get("lowerbody_strict_pass", ""):
        return boolish(row.get("lowerbody_strict_pass", ""))
    return row.get("work_status", "") == "WORK" and boolish(row.get("stage_pass", row.get("advance_to_rl", "")))


def downstream_failure_mode(row: dict[str, str]) -> str:
    if cem_pass(row):
        return ""
    if float_value(row.get("case_window_sim_leg_box_interference_frames_pct", "")) > 5.0:
        return "leg_object_interference"
    if any(row.get(key, "") for key in ["E081_success_numeric", "E080_success_numeric", "E079_success_numeric"]):
        return "legacy_numeric_fail"
    if float_value(row.get("leg_box_interference_frac", "")) > 0.05:
        return "leg_object_interference"
    if boolish(row.get("gate_lie_on_box", "")) or float_value(row.get("lie_on_box_frac", "")) >= 0.30:
        return "lie_on_box"
    if boolish(row.get("gate_pelvis_low", "")):
        return "pelvis_low"
    if boolish(row.get("gate_pelvis_tilt", "")):
        return "pelvis_tilt"
    if row.get("work_status") == "FAIL":
        return "cem_work_fail"
    return "cem_strict_fail"


def downstream_decision_from_cem(row: dict[str, str]) -> str:
    if cem_pass(row):
        return "DOWNSTREAM_CEM_PASS"
    mode = downstream_failure_mode(row)
    if mode in {"pelvis_low", "pelvis_tilt", "lie_on_box"}:
        return "DOWNSTREAM_POSTURE_FAIL"
    if mode == "leg_object_interference":
        return "DOWNSTREAM_MOTION_BINDING_FAIL"
    return "DOWNSTREAM_CEM_FAIL"


def base_case_row(case_id: str) -> dict[str, str]:
    meta = parse_case_id(case_id)
    row = {
        "case_id": case_id,
        **meta,
        "raw_inventory_status": "present",
        "raw_contact_3cm_status": "pass",
        "raw_contact_5cm_status": "pass",
        "template_status": "clean",
        "retarget_variant_id": "omnirt_v1",
        "target_variant_id": "ref_fk",
        "stage2b_status": "pass",
        "target_gate_status": "pass",
        "visual_qc_status": "pass",
        "cem_status": "not_run",
        "rl_status": "not_run",
        "diagnostic_contracts": "E098_global",
        "source_type": "historical_seed",
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    return row


def finalize(row: dict[str, str]) -> dict[str, str]:
    if row.get("target_variant_id") == "fingertip_aware":
        row.setdefault("diagnostic_contracts", "E098_global+E099_E100_E101_fingertip_aware")
        row.setdefault("route_diagnostic_status", "pass")
        row.setdefault("fingertip_vote_status", "pass")
        row.setdefault("palm_vote_status", "pass")
        row.setdefault("quat_audit_status", "pass")
        row.setdefault("target_active_mask_status", "pass")
        row.setdefault("target_gap_status", "pass")
        row.setdefault("e101_route_evidence_status", "pass")
    if row.get("target_npz") and Path(row["target_npz"]).is_file() and not row.get("target_npz_sha256"):
        row["target_npz_sha256"] = sha256_file(Path(row["target_npz"]))
    norm = normalize_row(row)
    norm["current_decision"] = current_decision_from_registry(norm)
    return norm


def row_from_full_eval(row: dict[str, str], summary_csv: Path, experiment: str) -> dict[str, str]:
    case_id = row.get("source_task", "") or row.get("case_id", "")
    out = base_case_row(case_id)
    tv = target_variant(row)
    out.update(
        {
            "target_variant_id": tv,
            "visual_qc_status": "pass",
            "cem_status": "pass" if cem_pass(row) else "fail",
            "downstream_decision": downstream_decision_from_cem(row),
            "downstream_failure_mode": downstream_failure_mode(row),
            "downstream_evidence_root": rel(summary_csv.parent),
            "cem_run_id": row.get("variant", ""),
            "cem_result_npz": root_npz_for(row, summary_csv),
            "cem_video": video_for(row, summary_csv),
            "cem_metrics_ref": rel(summary_csv),
            "target_npz": rel(row.get("target_npz", "")),
            "evidence_root": rel(summary_csv.parent),
            "source_ref": rel(summary_csv),
            "notes": (
                f"{experiment}; variant={row.get('variant', '')}; route={row.get('route', '')}; "
                f"work_status={row.get('work_status', '')}; strict={row.get('work_status_lowerbody_strict', row.get('lowerbody_strict_pass', ''))}; "
                f"contact={row.get('contact_frac_either', '')}; leg_interference={row.get('leg_box_interference_frac', '')}"
            ),
        }
    )
    if tv == "fingertip_aware":
        out["route_diagnostic_ref"] = rel(summary_csv.parent.parent.parent / "fingertip_target_summary.tsv")
    return finalize(out)


def row_from_legacy_eval(row: dict[str, str], summary_csv: Path, experiment: str) -> dict[str, str]:
    case_id = row.get("source_task", "") or row.get("case", "") or row.get("case_id", "")
    out = base_case_row(case_id)
    out.update(
        {
            "retarget_variant_id": "omnirt_legacy",
            "target_variant_id": "ref_fk",
            "visual_qc_status": "pass",
            "cem_status": "pass" if cem_pass(row) else "fail",
            "downstream_decision": downstream_decision_from_cem(row),
            "downstream_failure_mode": downstream_failure_mode(row),
            "downstream_evidence_root": rel(summary_csv.parent),
            "cem_run_id": row.get("variant", ""),
            "cem_result_npz": root_npz_for(row, summary_csv),
            "cem_video": video_for(row, summary_csv),
            "cem_metrics_ref": rel(summary_csv),
            "evidence_root": rel(summary_csv.parent),
            "source_ref": rel(summary_csv),
            "notes": (
                f"{experiment}; legacy CEM cache; case={row.get('case', '')}; "
                f"source_task={row.get('source_task', '')}; variant={row.get('variant', '')}; "
                f"E079_numeric={row.get('E079_success_numeric', '')}; "
                f"E080_numeric={row.get('E080_success_numeric', '')}; "
                f"E081_numeric={row.get('E081_success_numeric', '')}; "
                f"E081_legobj_strict={row.get('E081_success_legobj_strict_proxy', '')}; "
                f"sim_leg_interference_pct={row.get('case_window_sim_leg_box_interference_frames_pct', '')}; "
                "not post-E103 clean rerun"
            ),
        }
    )
    return finalize(out)


def row_from_e106_preprocess(row: dict[str, str], source_tsv: Path) -> dict[str, str]:
    out = base_case_row(row["source_task"])
    out.update(
        {
            "stage2b_status": "omniretarget_infeasible",
            "target_gate_status": "not_run",
            "visual_qc_status": "not_run",
            "cem_status": "not_run",
            "evidence_root": rel(source_tsv.parent),
            "source_ref": rel(source_tsv),
            "notes": f"E106 preprocess failure; variant={row.get('variant', '')}; reason={row.get('reason', '')}",
        }
    )
    return finalize(out)


def status_from_raw_decision(value: str) -> str:
    if value == "raw_contact_pass":
        return "pass"
    if value == "raw_contact_review":
        return "review"
    if value in {"raw_contact_fail", "raw_contact_error", "raw_contact_reject_motion"}:
        return "reject"
    return "not_run"


def row_from_e107_gate(row: dict[str, str], source_tsv: Path) -> dict[str, str]:
    out = base_case_row(row["target_task"])
    out.update(
        {
            "raw_contact_3cm_status": status_from_raw_decision(row.get("raw_decision_3cm", "")),
            "raw_contact_5cm_status": status_from_raw_decision(row.get("raw_decision_5cm", "")),
            "template_status": "clean" if row.get("source_scene_clean") == "True" else "audit_fail",
            "stage2b_status": "pass" if row.get("failure_mode") == "cem_ready" else "omniretarget_infeasible",
            "target_gate_status": "pass" if row.get("failure_mode") == "cem_ready" else "not_run",
            "visual_qc_status": "review" if row.get("failure_mode") == "cem_ready" else "not_run",
            "evidence_root": rel(source_tsv.parent),
            "source_ref": rel(source_tsv),
            "notes": f"E107 clean reconstruction gate; failure_mode={row.get('failure_mode', '')}; reason={row.get('reason', '')}",
        }
    )
    return finalize(out)


def persistent_e108_path(value: str) -> str:
    if not value:
        return ""
    return rel(value.replace("/tmp/core4d_dcv3_E108_nonbox", "workspace/core4d/results/E108"))


def row_from_e108_registry(row: dict[str, str], source_tsv: Path) -> dict[str, str]:
    out = normalize_row(row)
    for field in [
        "downstream_evidence_root",
        "cem_result_npz",
        "cem_video",
        "cem_metrics_ref",
        "evidence_root",
        "source_ref",
    ]:
        out[field] = persistent_e108_path(out.get(field, ""))
    out.update(
        {
            "source_type": "historical_seed",
            "source_ref": rel(source_tsv),
            "schema_version": SCHEMA_VERSION,
            "updated_at": timestamp(),
        }
    )
    if out.get("downstream_decision") == "DOWNSTREAM_NOT_RUN":
        out["downstream_decision"] = ""
    if out.get("retarget_variant_id") == "omnirt_v1":
        out["notes"] = "E108_nonbox_bucket004; " + out.get("notes", "")
    norm = normalize_row(out)
    norm["current_decision"] = current_decision_from_registry(norm)
    return norm


def prefixed_note(prefix: str, note: str) -> str:
    return f"{prefix}; {note}" if note else prefix


def build_rows(repo: Path) -> list[dict[str, str]]:
    sources: list[tuple[str, Path, str]] = [
        ("E092_box004_ref_fk", repo / "workspace/core4d/results/E092/spider_dyn/full/full_eval_summary.csv", "box004_only"),
        ("E094_box004_adaptive", repo / "workspace/core4d/results/E094/cem/full/full_eval_summary.csv", "box004_only"),
        ("E096b_box004_ref_fk", repo / "workspace/core4d/results/E096b/cem/full/full_eval_summary.csv", "all"),
        ("E105_box026_clean", repo / "workspace/core4d/results/E105/cem/full/full_eval_summary.csv", "all"),
        ("E106_box026_clean_batch", repo / "workspace/core4d/results/E106/cem/full/full_eval_summary.csv", "all"),
        ("E107_box021_selected4", repo / "workspace/core4d/results/E107/cem/full/full_eval_summary.csv", "all"),
    ]
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for experiment, path, mode in sources:
        if not path.is_file():
            continue
        for row in read_rows(path):
            if mode == "box004_only" and row.get("object") != "box004":
                continue
            built = row_from_full_eval(row, path, experiment)
            by_key[row_key(built)] = built

    legacy_sources: list[tuple[str, Path]] = [
        ("E079_box023_person1", repo / "workspace/core4d/results/E079/eval_summary_E079_box023_p1.csv"),
        ("E079_box023_person2", repo / "workspace/core4d/results/E079/eval_summary_E079_box023_p2.csv"),
        ("E080_box025_person1", repo / "workspace/core4d/results/E080/eval_summary_E080_box025_p1.csv"),
        ("E080_box025_person2", repo / "workspace/core4d/results/E080/eval_summary_E080_box025_p2.csv"),
        ("E081_box023_person2_legobj", repo / "workspace/core4d/results/E081/eval_summary_E081_box023_p2_legobj.csv"),
        ("E081_box025_person2_legobj", repo / "workspace/core4d/results/E081/eval_summary_E081_box025_p2_legobj.csv"),
    ]
    for experiment, path in legacy_sources:
        if not path.is_file():
            continue
        for row in read_rows(path):
            built = row_from_legacy_eval(row, path, experiment)
            by_key[row_key(built)] = built

    e106_fail = repo / "workspace/core4d/results/E106/preprocess_failures.tsv"
    if e106_fail.is_file():
        for row in read_rows(e106_fail, delimiter="\t"):
            built = row_from_e106_preprocess(row, e106_fail)
            by_key[row_key(built)] = built

    e107_gate = repo / "workspace/core4d/results/E107/box021_clean_gate_summary.tsv"
    if e107_gate.is_file():
        for row in read_rows(e107_gate, delimiter="\t"):
            built = row_from_e107_gate(row, e107_gate)
            by_key.setdefault(row_key(built), built)

    e108_candidate_registry = first_existing(
        repo,
        [
            "workspace/core4d/results/E108/registries/case_state_registry_all_nonbox_candidates.tsv",
            "workspace/core4d/results/E108/archive_legacy/E108_nonbox_candidate_mining_smoke/registries/case_state_registry.tsv",
            "workspace/core4d/results/E108/E108_nonbox_candidate_mining_smoke/registries/case_state_registry.tsv",
        ],
    )
    if e108_candidate_registry.is_file():
        for row in read_rows(e108_candidate_registry, delimiter="\t"):
            built = row_from_e108_registry(row, e108_candidate_registry)
            if built.get("retarget_variant_id") == "shared":
                built["notes"] = prefixed_note("E108_nonbox_candidate_mining", built.get("notes", ""))
            by_key[row_key(built)] = built

    e108_registry = first_existing(
        repo,
        [
            "workspace/core4d/results/E108/registries/case_state_registry.tsv",
            "workspace/core4d/results/E108/archive_legacy/registry_bucket004_person1_final/case_state_registry.tsv",
            "workspace/core4d/results/E108/registry_bucket004_person1_final/case_state_registry.tsv",
        ],
    )
    if e108_registry.is_file():
        for row in read_rows(e108_registry, delimiter="\t"):
            built = row_from_e108_registry(row, e108_registry)
            by_key[row_key(built)] = built

    return sorted(by_key.values(), key=lambda row: (row["object_key"], row["case_id"], row["target_variant_id"]))


def write_summary(path: Path, rows: list[dict[str, str]], source_status: dict[str, str]) -> None:
    counts = {
        "rows": len(rows),
        "object_counts": dict(Counter(row["object_key"] for row in rows)),
        "target_variant_counts": dict(Counter(row["target_variant_id"] for row in rows)),
        "current_decision_counts": dict(Counter(row["current_decision"] for row in rows)),
        "cem_status_counts": dict(Counter(row["cem_status"] for row in rows)),
        "downstream_decision_counts": dict(Counter(row["downstream_decision"] for row in rows)),
        "source_status": source_status,
    }
    lines = [
        "# Existing Cases Seed Summary",
        "",
        "本文件由 `build_existing_cases_seed.py` 从可信历史结果生成。旧污染 scene 的 Box021/Box026 结果不纳入；E105 ref-FK 与 E106 重复的 Box026 case 由 E106 覆盖。",
        "",
        f"- rows: `{counts['rows']}`",
        "",
        "## Counts",
        "",
    ]
    for title, key in [
        ("object", "object_counts"),
        ("target_variant", "target_variant_counts"),
        ("current_decision", "current_decision_counts"),
        ("cem_status", "cem_status_counts"),
        ("downstream_decision", "downstream_decision_counts"),
    ]:
        lines.extend([f"### {title}", "", "| value | count |", "|---|---:|"])
        for value, count in counts[key].items():
            lines.append(f"| `{value}` | {count} |")
        lines.append("")
    lines.extend(["## Sources", "", "| source | status |", "|---|---|"])
    for source, status in source_status.items():
        lines.append(f"| `{source}` | `{status}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider-repo", type=Path, default=SPIDER_REPO)
    parser.add_argument("--out-tsv", type=Path, default=SPIDER_REPO / "workspace/core4d/data_construction_v3/existing_cases.tsv")
    args = parser.parse_args()

    repo = args.spider_repo.expanduser().resolve()
    rows = build_rows(repo)
    out_tsv = args.out_tsv.expanduser().resolve()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(out_tsv, rows, FIELDS)
    source_status = {
        "E092_box004_ref_fk": "included_box004_only",
        "E094_box004_adaptive": "included_box004_only",
        "E096b_box004_ref_fk": "included",
        "E105_box026_clean": "included_nonduplicate_routes; duplicated ref_fk superseded by E106",
        "E106_box026_clean_batch": "included",
        "E107_box021_gate_and_selected4": "included",
        "E108_nonbox_candidate_mining": "included_full_candidate_state_cache; final bucket004 registry supersedes matching rows",
        "E108_nonbox_bucket004": "included_final_registry; 1 RL smoke pass, 1 CEM pass, 1 CEM fail, 1 visual reject",
        "E079_E080_E081_box023_box025": "included_as_legacy_cem_cache; E081 leg-object rows supersede E079/E080 person2 rows",
        "pre_E103_box021_box026": "excluded_invalidated_by_template_inertial_bug",
    }
    write_summary(out_tsv.with_name("existing_cases_summary.md"), rows, source_status)
    print(f"wrote {out_tsv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
