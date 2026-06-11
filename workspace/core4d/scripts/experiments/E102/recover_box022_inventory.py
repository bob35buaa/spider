"""Recover Box022 inventory for E102 Phase 1.

The output separates discovered rows from executable preflight scope. E091/E097
data_construction_v2 contains six intended Box022 stress-test/review cases
(125/126/127 x person1/person2) plus two non-selected strike rows (124). The
historical E098 manifest also has two older pending rows. All are recorded, but
only the six data_construction_v2 stress-test rows are marked in E102 scope by
default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "E099"))
from case_to_raw import parse_case  # noqa: E402


DEFAULT_V2_ROOT = Path("/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2")
SPIDER_SCENE_BASE = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object")
OBJECT_MODEL_BASE = Path("workspace/core4d/object_models/object_models")
RAW_ROOT_CANDIDATES = [
    Path("/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real/human_object_motions"),
    Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions"),
]


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


def bool_str(v: bool) -> str:
    return "True" if v else "False"


def choose_raw_paths(date: str, seq: str, person: str, core4d_root: Path | None) -> tuple[Path, Path, Path]:
    roots = [core4d_root] if core4d_root is not None else []
    roots += [r for r in RAW_ROOT_CANDIDATES if r not in roots]
    fallback_root = roots[0]
    for root in roots:
        person_npz = root / date / seq / f"{person}_poses.npz"
        obj_poses = root / date / seq / "smooth_objposes.npy"
        if person_npz.is_file() and obj_poses.is_file():
            return root, person_npz, obj_poses
    return (
        fallback_root,
        fallback_root / date / seq / f"{person}_poses.npz",
        fallback_root / date / seq / "smooth_objposes.npy",
    )


def raw_paths(case: str, core4d_root: Path | None) -> dict[str, str]:
    try:
        spec = parse_case(case)
    except Exception as exc:  # noqa: BLE001 - diagnostics script should not abort inventory
        return {
            "raw_parse_status": "parse_error",
            "raw_parse_error": str(exc),
            "raw_person_npz": "",
            "raw_obj_poses": "",
            "raw_person_exists": "False",
            "raw_obj_exists": "False",
            "raw_ok": "False",
        }
    raw_root, person_npz, obj_poses = choose_raw_paths(spec.date, spec.seq, spec.person, core4d_root)
    raw_ok = person_npz.is_file() and obj_poses.is_file() and not spec.note.startswith("SKIP")
    return {
        "raw_parse_status": "ok",
        "raw_parse_error": "",
        "raw_root": str(raw_root),
        "raw_person_npz": str(person_npz),
        "raw_obj_poses": str(obj_poses),
        "raw_person_exists": bool_str(person_npz.is_file()),
        "raw_obj_exists": bool_str(obj_poses.is_file()),
        "raw_ok": bool_str(raw_ok),
    }


def enrich(row: dict[str, str], core4d_root: Path | None) -> dict[str, str]:
    target_task = row["target_task"]
    source_scene_task = row.get("source_scene_task", "")
    source_scene = SPIDER_SCENE_BASE / source_scene_task / "scene.xml" if source_scene_task else Path("")
    object_model_rel = row.get("object_model_rel", "")
    object_model = OBJECT_MODEL_BASE / object_model_rel if object_model_rel else Path("")
    out = dict(row)
    out.update(raw_paths(target_task, core4d_root))
    out["source_scene_xml"] = str(source_scene) if source_scene_task else ""
    out["source_scene_xml_exists"] = bool_str(bool(source_scene_task) and source_scene.is_file())
    out["object_model_path"] = str(object_model) if object_model_rel else ""
    out["object_model_exists"] = bool_str(bool(object_model_rel) and object_model.is_file())
    return out


def from_medium_manifest(path: Path, core4d_root: Path | None) -> list[dict[str, str]]:
    rows = []
    for row in read_tsv(path):
        if row.get("object_key", "").lower() != "box022":
            continue
        case_role = row.get("case_role", "")
        in_scope = case_role == "review_stress_test_selected"
        rows.append(
            enrich(
                {
                    "inventory_source": str(path),
                    "target_task": row.get("planned_target_task", "") or row.get("mask_slug", ""),
                    "date": row.get("date", ""),
                    "seq": row.get("seq", ""),
                    "person": row.get("person", ""),
                    "object_name": row.get("object_name", ""),
                    "object_key": row.get("object_key", ""),
                    "case_role": case_role,
                    "action": row.get("action", ""),
                    "size_band": row.get("size_band", ""),
                    "extent_x_m": row.get("extent_x_m", ""),
                    "extent_y_m": row.get("extent_y_m", ""),
                    "extent_z_m": row.get("extent_z_m", ""),
                    "size_vs_box023_volume_ratio": row.get("size_vs_box023_volume_ratio", ""),
                    "size_vs_box025_volume_ratio": row.get("size_vs_box025_volume_ratio", ""),
                    "source_scene_task": row.get("source_scene_task", ""),
                    "object_model_rel": row.get("object_model_rel", ""),
                    "stage0_v2_decision": row.get("stage0_v2_decision", ""),
                    "stage1_decision": row.get("stage1_decision", ""),
                    "stage2_route": row.get("stage2_route", ""),
                    "e102_scope": bool_str(in_scope),
                    "notes": row.get("notes", ""),
                },
                core4d_root,
            )
        )
    return rows


def from_historical_manifest(path: Path, core4d_root: Path | None) -> list[dict[str, str]]:
    rows = []
    for row in read_tsv(path):
        if "box022" not in row.get("case_name", "").lower() and "box022" not in row.get("box_family", "").lower():
            continue
        case = row["case_name"]
        rows.append(
            enrich(
                {
                    "inventory_source": str(path),
                    "target_task": case,
                    "date": "",
                    "seq": "",
                    "person": f"person{row.get('person', '')}" if row.get("person", "") else "",
                    "object_name": "Box022",
                    "object_key": "box022",
                    "case_role": row.get("owner_stage", ""),
                    "action": "",
                    "size_band": "",
                    "extent_x_m": "",
                    "extent_y_m": "",
                    "extent_z_m": "",
                    "size_vs_box023_volume_ratio": "",
                    "size_vs_box025_volume_ratio": "",
                    "source_scene_task": "",
                    "object_model_rel": "box/box022_m.obj",
                    "stage0_v2_decision": "",
                    "stage1_decision": "",
                    "stage2_route": "",
                    "e102_scope": "False",
                    "notes": row.get("last_known_outcome", ""),
                },
                core4d_root,
            )
        )
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        case = row["target_task"]
        if case not in by_case or row.get("e102_scope") == "True":
            by_case[case] = row
    return sorted(by_case.values(), key=lambda r: (r.get("e102_scope") != "True", r["target_task"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--core4d-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=Path("workspace/core4d/results/E102/box022_missing_sources.tsv"),
    )
    args = parser.parse_args()

    rows = []
    rows += from_medium_manifest(args.v2_root / "inputs" / "medium_box_manifest.tsv", args.core4d_root)
    rows += from_historical_manifest(args.manifest, args.core4d_root)
    rows = dedupe(rows)

    fieldnames = [
        "target_task",
        "e102_scope",
        "inventory_source",
        "date",
        "seq",
        "person",
        "object_name",
        "object_key",
        "case_role",
        "action",
        "size_band",
        "extent_x_m",
        "extent_y_m",
        "extent_z_m",
        "size_vs_box023_volume_ratio",
        "size_vs_box025_volume_ratio",
        "source_scene_task",
        "source_scene_xml",
        "source_scene_xml_exists",
        "object_model_rel",
        "object_model_path",
        "object_model_exists",
        "raw_parse_status",
        "raw_parse_error",
        "raw_root",
        "raw_person_npz",
        "raw_obj_poses",
        "raw_person_exists",
        "raw_obj_exists",
        "raw_ok",
        "stage0_v2_decision",
        "stage1_decision",
        "stage2_route",
        "notes",
    ]
    write_tsv(args.out, rows, fieldnames)

    missing_rows = []
    for row in rows:
        missing = []
        if row["raw_person_exists"] != "True":
            missing.append("raw_person_npz")
        if row["raw_obj_exists"] != "True":
            missing.append("raw_obj_poses")
        if row["source_scene_xml_exists"] != "True":
            missing.append("source_scene_xml")
        if row["object_model_exists"] != "True":
            missing.append("object_model")
        if missing:
            miss = dict(row)
            miss["missing_fields"] = ",".join(missing)
            missing_rows.append(miss)
    write_tsv(args.missing_out, missing_rows, fieldnames + ["missing_fields"])

    scoped = [r for r in rows if r["e102_scope"] == "True"]
    executable = [r for r in scoped if r["raw_ok"] == "True" and r["source_scene_xml_exists"] == "True"]
    print(f"wrote {args.out} rows={len(rows)} scoped={len(scoped)} executable={len(executable)}")
    print(f"wrote {args.missing_out} rows={len(missing_rows)}")


if __name__ == "__main__":
    main()
