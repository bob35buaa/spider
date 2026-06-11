"""Build E102 RL-ready handoff tables from existing SPIDER-WORK positives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POSITIVE_CASES = [
    {
        "case_name": "e091_box004_20231003_2_083_p1",
        "provenance": "E096b",
        "summary": "workspace/core4d/results/E096b/cem/full/full_eval_summary.csv",
        "variant": "E096bP1_box004_083_p1_mask_cem",
        "video_path": "workspace/core4d/results/E096b/cem/full/E096bP1_box004_083_p1_mask_cem_full.mp4",
    },
    {
        "case_name": "e091_box004_20231003_2_082_p1",
        "provenance": "E096b",
        "summary": "workspace/core4d/results/E096b/cem/full/full_eval_summary.csv",
        "variant": "E096bP2_box004_082_p1_mask_cem",
        "video_path": "workspace/core4d/results/E096b/cem/full/E096bP2_box004_082_p1_mask_cem_full.mp4",
    },
    {
        "case_name": "e091_box004_20231003_2_083_p2",
        "provenance": "E101_guard_E100_target",
        "summary": "workspace/core4d/results/E101/cem_outcome_matrix.tsv",
        "variant": "E101P1_box004_083_p2_fingertip_seed0",
        "video_path": "workspace/core4d/results/E101/phase1/E101P1_box004_083_p2_fingertip_seed0.mp4",
    },
]


def read_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def find_summary_row(summary: Path, variant: str) -> dict[str, str]:
    for row in read_table(summary):
        if row.get("variant") == variant:
            return row
    return {}


def path_exists(path: str) -> str:
    return "True" if path and Path(path).is_file() else "False"


def build_positive_rows() -> list[dict[str, str]]:
    rows = []
    for spec in POSITIVE_CASES:
        case = spec["case_name"]
        summary = Path(spec["summary"])
        row = find_summary_row(summary, spec["variant"])
        source_npz = f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{case}/0/trajectory_kinematic.npz"
        scene_xml = row.get("scene_xml") or f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{case}/scene_act.xml"
        contact_target_npz = f"workspace/core4d/results/E100/fingertip_targets/{case}/spider_contact_target_object_local.npz"
        gate_metrics = ""
        if spec["provenance"].startswith("E096b"):
            gate_metrics = (
                f"pelvis_min={row.get('pelvis_min_m')}; obj_mean={row.get('obj_err_mean_m')}; "
                f"obj_max={row.get('obj_err_max_m')}; head={row.get('head_pen_frac')}; "
                f"upper={row.get('upper_pen_frac')}; hand_floor={row.get('handL_floor_lt_5cm_frac')}/{row.get('handR_floor_lt_5cm_frac')}"
            )
        else:
            gate_metrics = (
                f"pelvis_min={row.get('pelvis_min_z_m')}; pelvis_end={row.get('pelvis_end_z_m')}; "
                f"tilt_end={row.get('pelvis_tilt_end_deg')}; lie={row.get('lie_on_box_frac')}; gate_pass={row.get('gate_pass')}"
            )
        rows.append(
            {
                "case_name": case,
                "source_npz": source_npz,
                "scene_xml": scene_xml,
                "contact_target_npz": contact_target_npz,
                "gate_metrics": gate_metrics,
                "video_path": spec["video_path"],
                "provenance": spec["provenance"],
                "status": "READY" if all(path_exists(p) == "True" for p in [source_npz, scene_xml, contact_target_npz, spec["video_path"]]) else "MISSING_ARTIFACT",
                "source_npz_exists": path_exists(source_npz),
                "scene_xml_exists": path_exists(scene_xml),
                "contact_target_exists": path_exists(contact_target_npz),
                "video_exists": path_exists(spec["video_path"]),
                "notes": "existing box004 SPIDER-WORK positive; no new E102 data expansion positive",
            }
        )
    return rows


def build_dont_try(negative_registry: Path, rejected: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_table(negative_registry):
        if row.get("registry_status") == "HARD_EXCLUDE":
            rows.append(
                {
                    "case_name": row.get("case_name", ""),
                    "source": "negative_case_registry",
                    "status": row.get("registry_status", ""),
                    "reason": row.get("primary_failure_tag", "") + (";" + row.get("secondary_failure_tags", "") if row.get("secondary_failure_tags") else ""),
                    "evidence_level": row.get("evidence_level", ""),
                    "notes": row.get("notes", ""),
                }
            )
    hard_routes = {
        "reject_box022_preflight_not_pass",
        "reject_preprocess_infeasible",
        "reject_verified_legacy",
        "reject_box026_large_reach_holdout",
    }
    for row in read_table(rejected):
        if row.get("route") in hard_routes:
            rows.append(
                {
                    "case_name": row.get("target_task", ""),
                    "source": "v2_candidate_mining",
                    "status": row.get("route", ""),
                    "reason": row.get("reason", ""),
                    "evidence_level": row.get("registry_evidence_level", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def write_handoff(path: Path, rl_rows: list[dict[str, str]], dont_rows: list[dict[str, str]]) -> None:
    ready = [r for r in rl_rows if r["status"] == "READY"]
    status = "PASS" if len(ready) >= 5 else "PARTIAL"
    lines = [
        "# E102 Holosoma Handoff",
        "",
        f"Status: **{status}**",
        "",
        f"- RL-ready WORK cases: {len(ready)}",
        f"- Required target: >= 5 case",
        "- Data expansion did not yield enough new WORK case; E102 Phase 3 was skipped because Phase 2 found 0 executable candidates.",
        "",
        "## RL-ready Set",
        "",
        "| case | status | provenance | source | scene | target | video |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rl_rows:
        lines.append(
            f"| `{row['case_name']}` | {row['status']} | {row['provenance']} | {row['source_npz']} | {row['scene_xml']} | {row['contact_target_npz']} | {row['video_path']} |"
        )
    lines += [
        "",
        "## Do Not Retry Without New Plan",
        "",
        f"- Rows: {len(dont_rows)}",
        "- Current negatives come from E101 after E098-E100 fixes.",
        "- Box022 rows are rejected by E102 raw fingertip preflight and still lack source scene templates.",
        "- Box026 remains a large-reach/posture holdout.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dont-out", type=Path, default=Path("workspace/core4d/results/E102/dont_try_this_list.tsv"))
    parser.add_argument("--negative-registry", type=Path, default=Path("workspace/core4d/results/E102/negative_case_registry.tsv"))
    parser.add_argument("--rejected", type=Path, default=Path("workspace/core4d/results/E102/v2_candidates_rejected.tsv"))
    parser.add_argument("--handoff", type=Path, default=Path("workspace/core4d/results/E102/holosoma_handoff.md"))
    args = parser.parse_args()

    rl_rows = build_positive_rows()
    dont_rows = build_dont_try(args.negative_registry, args.rejected)
    rl_fields = [
        "case_name",
        "source_npz",
        "scene_xml",
        "contact_target_npz",
        "gate_metrics",
        "video_path",
        "provenance",
        "status",
        "source_npz_exists",
        "scene_xml_exists",
        "contact_target_exists",
        "video_exists",
        "notes",
    ]
    dont_fields = ["case_name", "source", "status", "reason", "evidence_level", "notes"]
    write_tsv(args.out, rl_rows, rl_fields)
    write_tsv(args.dont_out, dont_rows, dont_fields)
    write_handoff(args.handoff, rl_rows, dont_rows)
    print(f"wrote {args.out} rows={len(rl_rows)} ready={sum(1 for r in rl_rows if r['status'] == 'READY')}")
    print(f"wrote {args.dont_out} rows={len(dont_rows)}")
    print(f"handoff -> {args.handoff}")


if __name__ == "__main__":
    main()
