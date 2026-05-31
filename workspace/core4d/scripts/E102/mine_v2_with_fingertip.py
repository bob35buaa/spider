"""E102 Phase 2 candidate mining with fingertip and evidence-aware negatives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
SPIDER_SCENE_ROOT = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fnum(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canonical_case(task: str) -> str:
    out = task.replace("_upperobj_e083", "").replace("_e092_dyn", "")
    if out.startswith("e091_box021_"):
        out = out.replace("e091_box021_", "d003_box021_", 1)
    return out


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    reg = {}
    for row in read_tsv(path):
        reg[row["case_name"]] = row
    return reg


def load_fingertip(path: Path) -> dict[str, dict[str, str]]:
    return {r["case"]: r for r in read_tsv(path)}


def load_quat(path: Path) -> dict[str, dict[str, str]]:
    return {r["case"]: r for r in read_tsv(path)}


def load_excluded(path: Path) -> dict[str, dict[str, str]]:
    return {r["target_task"]: r for r in read_tsv(path)}


def load_box022_preflight(path: Path) -> dict[str, dict[str, str]]:
    return {r["target_task"]: r for r in read_tsv(path)}


def load_invalidated_tasks(path: Path) -> set[str]:
    out = set()
    for row in read_tsv(path):
        if row.get("action") == "quarantine_invalidated_by_scene_inertial_bug":
            out.add(row.get("task", ""))
    return out


def size_in_target_range(row: dict[str, str]) -> bool:
    # User clarified that size only needs to be in the box023-to-box025 range.
    return row.get("size_band") == "target_medium_between_box023_and_box025"


def source_scene_path(row: dict[str, str], scene_root: Path = SPIDER_SCENE_ROOT) -> Path | None:
    task = row.get("source_scene_task", "")
    if not task:
        return None
    return scene_root / task / "scene.xml"


def source_scene_exists(row: dict[str, str], scene_root: Path = SPIDER_SCENE_ROOT) -> bool:
    path = source_scene_path(row, scene_root)
    if path is not None and path.is_file():
        return True
    return row.get("source_scene_exists") == "True" or row.get("has_template") == "true"


def raw_pass(row: dict[str, str]) -> bool:
    return row.get("stage1_decision") == "raw_contact_pass"


def score(row: dict[str, str], fingertip: dict[str, str], registry_row: dict[str, str] | None) -> float:
    raw = fnum(row.get("stage1_raw_contact_score", "0"))
    both = fnum(row.get("target_both_active_frac", row.get("target_both_active_frac_3cm", "0")))
    longest = fnum(
        row.get("target_both_longest_run_active_frac", row.get("target_both_longest_run_active_frac_3cm", "0"))
    )
    support_bonus = 0.0
    if fingertip:
        support_bonus = min(fnum(fingertip.get("L_contact", "0")), fnum(fingertip.get("R_contact", "0"))) / 2.0
    legacy_penalty = 10.0 if registry_row and registry_row.get("evidence_level") == "legacy_failure_prior" else 0.0
    return raw + 20.0 * both + 10.0 * longest + support_bonus - legacy_penalty


def classify_row(
    row: dict[str, str],
    registry: dict[str, dict[str, str]],
    excluded: dict[str, dict[str, str]],
    box022_preflight: dict[str, dict[str, str]],
    invalidated_tasks: set[str],
    scene_root: Path = SPIDER_SCENE_ROOT,
) -> tuple[str, str]:
    task = row.get("planned_target_task", "") or row.get("target_task", "")
    canonical = canonical_case(task)
    reg = registry.get(canonical)
    exc = excluded.get(task)
    obj = row.get("object_key", "").lower()

    if reg and reg.get("evidence_level") in {"current_negative", "legacy_replayed_negative"}:
        return "reject_current_negative", reg.get("notes", "")
    if exc and exc.get("known_status") == "verified_positive":
        return "existing_positive_not_new", exc.get("feature_note", "")
    if exc and exc.get("known_stage") == "omniretarget_infeasible":
        return "reject_preprocess_infeasible", exc.get("feature_note", "")
    if exc and exc.get("known_status") == "verified_reject":
        if task in invalidated_tasks or canonical in invalidated_tasks:
            # E103: polluted-scene dynamics labels are priors only, not hard negatives.
            pass
        else:
            return "reject_verified_legacy", exc.get("feature_note", "")
    has_source_scene = source_scene_exists(row, scene_root)

    if obj == "box022":
        pf = box022_preflight.get(task)
        if not pf or pf.get("decision") != "PREFLIGHT_PASS":
            decision = (pf or {}).get("decision", "missing_preflight")
            return "reject_box022_preflight_not_pass", f"{decision}: {(pf or {}).get('reason', 'no PREFLIGHT_PASS')}"
    if not size_in_target_range(row):
        return "reject_outside_size_prior", row.get("size_band", "")
    if not has_source_scene:
        if raw_pass(row):
            return "needs_source_scene_template_then_preflight", row.get("source_scene_task", "")
        return "needs_source_scene_template", row.get("source_scene_task", "")
    if not raw_pass(row):
        return "reject_raw_contact_not_pass", row.get("stage1_decision", "")
    if reg and reg.get("evidence_level") == "legacy_failure_prior":
        return "candidate_legacy_risk_needs_visual", reg.get("notes", "")
    return "candidate_executable", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-registry", type=Path, required=True)
    parser.add_argument("--box022-preflight", type=Path, default=Path("workspace/core4d/results/E102/box022_preflight.tsv"))
    parser.add_argument("--manifest", type=Path, default=None, help="override medium_box_manifest.tsv path")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--fingertip-stats", type=Path, default=Path("workspace/core4d/results/E099/fingertip_face_stats.tsv"))
    parser.add_argument("--quat-audit", type=Path, default=Path("workspace/core4d/results/E099/quat_audit.tsv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reject-out", type=Path, default=Path("workspace/core4d/results/E102/v2_candidates_rejected.tsv"))
    parser.add_argument("--summary", type=Path, default=Path("workspace/core4d/results/E102/v2_candidate_mining_summary.md"))
    parser.add_argument("--scene-root", type=Path, default=SPIDER_SCENE_ROOT)
    parser.add_argument(
        "--invalidated-registry",
        type=Path,
        default=Path("workspace/core4d/results/E103/affected_scene_registry_phase0_pre_rebuild.tsv"),
        help="pre-rebuild E103 registry; polluted dynamics labels from these tasks are not hard rejects",
    )
    args = parser.parse_args()

    manifest = read_tsv(args.manifest or (args.v2_root / "inputs" / "medium_box_manifest.tsv"))
    registry = load_registry(args.negative_registry)
    fingertip = load_fingertip(args.fingertip_stats)
    quat = load_quat(args.quat_audit)
    excluded = load_excluded(args.v2_root / "results" / "e097_feature_candidates" / "excluded_verified_cases.tsv")
    box022_pf = load_box022_preflight(args.box022_preflight)
    invalidated_tasks = load_invalidated_tasks(args.invalidated_registry)

    candidate_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for row in manifest:
        if row.get("object_category") != "box":
            continue
        task = row.get("planned_target_task", "") or row.get("mask_slug", "")
        canonical = canonical_case(task)
        reg = registry.get(canonical)
        ft = fingertip.get(task) or fingertip.get(canonical) or {}
        qa = quat.get(task) or quat.get(canonical) or {}
        route, reason = classify_row(row, registry, excluded, box022_pf, invalidated_tasks, args.scene_root)
        live_source_scene = source_scene_path(row, args.scene_root)
        live_source_scene_exists = bool(live_source_scene and live_source_scene.is_file())
        counts[route] = counts.get(route, 0) + 1
        out = {
            "target_task": task,
            "canonical_case": canonical,
            "route": route,
            "reason": reason,
            "score": f"{score(row, ft, reg):.3f}",
            "object_key": row.get("object_key", ""),
            "object_name": row.get("object_name", ""),
            "sequence": row.get("sequence", ""),
            "person": row.get("person", ""),
            "action": row.get("action", ""),
            "size_band": row.get("size_band", ""),
            "extent_x_m": row.get("extent_x_m", ""),
            "extent_y_m": row.get("extent_y_m", ""),
            "extent_z_m": row.get("extent_z_m", ""),
            "size_vs_box023_volume_ratio": row.get("size_vs_box023_volume_ratio", ""),
            "size_vs_box025_volume_ratio": row.get("size_vs_box025_volume_ratio", ""),
            "stage1_decision": row.get("stage1_decision", ""),
            "stage1_raw_contact_score": row.get("stage1_raw_contact_score", ""),
            "contact_threshold_label": row.get("contact_threshold_label", ""),
            "target_both_active_frac": row.get("target_both_active_frac", ""),
            "target_both_longest_run_active_frac": row.get("target_both_longest_run_active_frac", ""),
            "target_both_active_frac_3cm": row.get("target_both_active_frac_3cm", ""),
            "target_both_active_frac_5cm": row.get("target_both_active_frac_5cm", ""),
            "target_both_longest_run_active_frac_3cm": row.get("target_both_longest_run_active_frac_3cm", ""),
            "target_both_longest_run_active_frac_5cm": row.get("target_both_longest_run_active_frac_5cm", ""),
            "source_scene_task": row.get("source_scene_task", ""),
            "source_scene_exists": row.get("source_scene_exists", ""),
            "source_scene_exists_live": str(live_source_scene_exists),
            "source_scene_xml_live": str(live_source_scene or ""),
            "raw_contact_proxy_path": row.get("raw_contact_proxy_path", ""),
            "fingertip_status": ft.get("status", ""),
            "L_vote": ft.get("L_vote", ""),
            "L_contact": ft.get("L_contact", ""),
            "R_vote": ft.get("R_vote", ""),
            "R_contact": ft.get("R_contact", ""),
            "disable_world_up": qa.get("disable_world_up", ""),
            "registry_evidence_level": (reg or {}).get("evidence_level", ""),
            "registry_status": (reg or {}).get("registry_status", ""),
            "legacy_label_invalidated_by_e103": str(task in invalidated_tasks or canonical in invalidated_tasks),
            "notes": row.get("notes", ""),
        }
        if route.startswith("candidate"):
            candidate_rows.append(out)
        else:
            rejected_rows.append(out)

    candidate_rows.sort(key=lambda r: float(r["score"]), reverse=True)
    for i, row in enumerate(candidate_rows, start=1):
        row["rank"] = str(i)
    for i, row in enumerate(rejected_rows, start=1):
        row["rank"] = str(i)

    fields = [
        "rank",
        "target_task",
        "canonical_case",
        "route",
        "reason",
        "score",
        "object_key",
        "object_name",
        "sequence",
        "person",
        "action",
        "size_band",
        "extent_x_m",
        "extent_y_m",
        "extent_z_m",
        "size_vs_box023_volume_ratio",
        "size_vs_box025_volume_ratio",
        "stage1_decision",
        "stage1_raw_contact_score",
        "contact_threshold_label",
        "target_both_active_frac",
        "target_both_longest_run_active_frac",
        "target_both_active_frac_3cm",
        "target_both_active_frac_5cm",
        "target_both_longest_run_active_frac_3cm",
        "target_both_longest_run_active_frac_5cm",
        "source_scene_task",
        "source_scene_exists",
        "source_scene_exists_live",
        "source_scene_xml_live",
        "raw_contact_proxy_path",
        "fingertip_status",
        "L_vote",
        "L_contact",
        "R_vote",
        "R_contact",
        "disable_world_up",
        "registry_evidence_level",
        "registry_status",
        "legacy_label_invalidated_by_e103",
        "notes",
    ]
    write_tsv(args.out, candidate_rows, fields)
    write_tsv(args.reject_out, rejected_rows, fields)

    summary_lines = [
        "# Candidate Mining Summary",
        "",
        f"- Manifest: `{args.manifest or (args.v2_root / 'inputs' / 'medium_box_manifest.tsv')}`",
        f"- Source rows: {len(manifest)}",
        f"- Candidate rows: {len(candidate_rows)}",
        f"- Rejected/held rows: {len(rejected_rows)}",
        "",
        "## Route Counts",
        "",
        "| route | count |",
        "|---|---:|",
    ]
    for route, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        summary_lines.append(f"| `{route}` | {count} |")
    summary_lines += [
        "",
        "## Decision",
        "",
    ]
    if len(candidate_rows) < 5:
        summary_lines.append(
            "- Fewer than 5 executable candidates were found. Causes are separated in the rejected TSV: current negatives, existing positives, Box022 preflight failures, source-template backlog rows, raw-contact failures, invalidated legacy labels, and Box026 large-reach holdouts."
        )
    if len(candidate_rows) < 2:
        summary_lines.append("- Full CEM should be skipped under the current data-gate rule because executable candidates < 2.")
    else:
        summary_lines.append("- The next phase can select typical-2 from `v2_candidates_with_fingertip.tsv`.")
    args.summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out} rows={len(candidate_rows)}")
    print(f"wrote {args.reject_out} rows={len(rejected_rows)}")
    print(f"summary -> {args.summary}")
    print(f"route_counts={counts}")


if __name__ == "__main__":
    main()
