"""Build E102 evidence-aware negative registry.

E101 post-fix failures become current negatives. Historical rows are retained as
legacy priors unless they are replayed later under the current gate/target
contract.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def case_key(task: str) -> str:
    # Normalize experiment-derived task names to the source case id used by
    # manifests/mining inventories.
    return task.replace("_e092_dyn", "").replace("_upperobj_e083", "")


def legacy_evidence_level(row: dict[str, str]) -> str:
    owner = row.get("owner_stage", "")
    outcome = row.get("last_known_outcome", "")
    if owner in {"guard", "guard_pos", "guard_h1"} or "WORK" in outcome:
        return "positive_guard"
    return "legacy_failure_prior"


def hard_exclusion_reason(row: dict[str, str]) -> str:
    source = row.get("source_npz", "")
    scene = row.get("scene_xml", "")
    outcome = row.get("last_known_outcome", "")
    if source.startswith("pending") or scene.startswith("pending"):
        return "source_pending"
    if "infeasible" in outcome.lower():
        return "preprocess_infeasible"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e101-taxonomy", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy-replay-out",
        type=Path,
        default=Path("workspace/core4d/results/E102/legacy_failure_replay.tsv"),
    )
    args = parser.parse_args()

    taxonomy = read_tsv(args.e101_taxonomy)
    legacy = read_tsv(args.legacy_manifest)
    rows_by_case: dict[str, dict[str, str]] = {}

    for row in legacy:
        case = row.get("case_name", "")
        level = legacy_evidence_level(row)
        rows_by_case[case] = {
            "case_name": case,
            "task": case,
            "box_family": row.get("box_family", ""),
            "person": row.get("person", ""),
            "evidence_level": level,
            "registry_status": "ALLOW" if level == "positive_guard" else "PRIOR_ONLY",
            "primary_failure_tag": "other" if level == "positive_guard" else "other",
            "secondary_failure_tags": "",
            "source_npz": row.get("source_npz", ""),
            "scene_xml": row.get("scene_xml", ""),
            "representative_variant": "",
            "mp4_path": "",
            "sheet_path": "",
            "hard_exclusion_reason": hard_exclusion_reason(row),
            "provenance": f"historical_manifest:{row.get('owner_stage', '')}",
            "notes": row.get("last_known_outcome", ""),
        }

    for row in taxonomy:
        task = case_key(row.get("task", ""))
        if not task:
            continue
        level = row.get("evidence_level", "")
        if level == "current_negative":
            status = "HARD_EXCLUDE"
        elif level == "positive_guard":
            status = "ALLOW"
        else:
            status = "PRIOR_ONLY"
        rows_by_case[task] = {
            "case_name": task,
            "task": row.get("task", ""),
            "box_family": "box021_d003" if "box021" in task else ("box004" if "box004" in task else ""),
            "person": "",
            "evidence_level": level,
            "registry_status": status,
            "primary_failure_tag": row.get("primary_failure_tag", ""),
            "secondary_failure_tags": row.get("secondary_failure_tags", ""),
            "source_npz": "",
            "scene_xml": "",
            "representative_variant": row.get("variant", ""),
            "mp4_path": row.get("mp4_path", ""),
            "sheet_path": row.get("sheet_path", ""),
            "hard_exclusion_reason": "",
            "provenance": "E101_post_E098_E100",
            "notes": row.get("reason", ""),
        }

    fieldnames = [
        "case_name",
        "task",
        "box_family",
        "person",
        "evidence_level",
        "registry_status",
        "primary_failure_tag",
        "secondary_failure_tags",
        "source_npz",
        "scene_xml",
        "representative_variant",
        "mp4_path",
        "sheet_path",
        "hard_exclusion_reason",
        "provenance",
        "notes",
    ]
    rows = sorted(rows_by_case.values(), key=lambda r: (r["evidence_level"], r["case_name"]))
    write_tsv(args.out, rows, fieldnames)

    replay_rows = []
    for row in rows:
        if row["evidence_level"] == "legacy_failure_prior":
            replay_rows.append(
                {
                    "case_name": row["case_name"],
                    "source_npz": row["source_npz"],
                    "scene_xml": row["scene_xml"],
                    "legacy_evidence": row["notes"],
                    "replay_status": "not_replayed_in_E102_phase0",
                    "evidence_level_after_replay": "legacy_failure_prior",
                    "reason": "historical failure predates E098-E100 fixes; retained as mechanism prior only",
                }
            )
    write_tsv(
        args.legacy_replay_out,
        replay_rows,
        [
            "case_name",
            "source_npz",
            "scene_xml",
            "legacy_evidence",
            "replay_status",
            "evidence_level_after_replay",
            "reason",
        ],
    )
    print(f"wrote {args.out} rows={len(rows)}")
    print(f"wrote {args.legacy_replay_out} rows={len(replay_rows)}")


if __name__ == "__main__":
    main()
