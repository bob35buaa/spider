"""Run best-effort Box022 preflight from recovered inventory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "E099"))
import case_to_raw  # noqa: E402
from fingertip_face_vote import vote_case  # noqa: E402


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def decide_source_block(row: dict[str, str]) -> tuple[bool, str]:
    missing = []
    if row.get("raw_person_exists") != "True":
        missing.append("raw_person_npz")
    if row.get("raw_obj_exists") != "True":
        missing.append("raw_obj_poses")
    if missing:
        return True, "missing:" + ",".join(missing)
    return False, ""


def preflight_row(
    row: dict[str, str],
    json_dir: Path,
    contact_threshold_m: float,
    contact_threshold_label: str,
    min_contact_frames: int,
) -> dict[str, str]:
    out = {
        "target_task": row["target_task"],
        "e102_scope": row.get("e102_scope", ""),
        "case_role": row.get("case_role", ""),
        "decision": "",
        "reason": "",
        "contact_threshold_m": f"{contact_threshold_m:.3f}",
        "contact_threshold_label": contact_threshold_label,
        "min_contact_frames": str(min_contact_frames),
        "raw_ok": row.get("raw_ok", ""),
        "source_scene_xml_exists": row.get("source_scene_xml_exists", ""),
        "object_model_exists": row.get("object_model_exists", ""),
        "L_vote": "",
        "L_frac": "",
        "L_contact": "",
        "L_contact_frac": "",
        "R_vote": "",
        "R_frac": "",
        "R_contact": "",
        "R_contact_frac": "",
        "T": "",
        "quat_disable_world_up": "",
        "reach_support_status": row.get("stage1_decision", "") or "not_run",
        "video_path": "",
        "json_path": "",
        "raw_person_npz": row.get("raw_person_npz", ""),
        "raw_obj_poses": row.get("raw_obj_poses", ""),
        "source_scene_xml": row.get("source_scene_xml", ""),
        "notes": row.get("notes", ""),
    }
    blocked, reason = decide_source_block(row)
    if blocked:
        out["decision"] = "SOURCE_BLOCKED"
        out["reason"] = reason
        return out

    scene = Path(row["source_scene_xml"])
    scene_arg = scene if scene.is_file() else None
    try:
        result = vote_case(row["target_task"], scene_xml=scene_arg, contact_thresh=contact_threshold_m)
    except Exception as exc:  # noqa: BLE001 - diagnostics should emit row
        out["decision"] = "PREFLIGHT_ERROR"
        out["reason"] = str(exc)
        return out

    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{row['target_task']}_fingertip_vote_{contact_threshold_label}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out["json_path"] = str(json_path)

    if result.get("status") != "ok":
        out["decision"] = "SOURCE_BLOCKED" if result.get("status") == "missing_raw" else "PREFLIGHT_ERROR"
        out["reason"] = result.get("status", "unknown")
        return out

    total_frames = int(result.get("T", 0) or 0)
    out["T"] = str(total_frames)
    for hand in ("L", "R"):
        h = result.get(hand, {})
        out[f"{hand}_vote"] = h.get("vote_face", "")
        out[f"{hand}_frac"] = f"{float(h.get('vote_frac', 0.0)):.3f}"
        contact_frames = int(h.get("contact_frames", 0) or 0)
        out[f"{hand}_contact"] = str(contact_frames)
        out[f"{hand}_contact_frac"] = f"{(contact_frames / total_frames if total_frames else 0.0):.3f}"

    l_contact = int(out["L_contact"] or "0")
    r_contact = int(out["R_contact"] or "0")
    if l_contact >= min_contact_frames and r_contact >= min_contact_frames and out["L_vote"] and out["R_vote"]:
        if row.get("source_scene_xml_exists") == "True":
            out["decision"] = "PREFLIGHT_PASS"
            out["reason"] = (
                f"both hands have fingertip contact evidence at {contact_threshold_label} "
                f"with min_contact_frames={min_contact_frames}"
            )
        else:
            out["decision"] = "RAW_PREFLIGHT_PASS_NEEDS_SCENE_TEMPLATE"
            out["reason"] = (
                f"raw fingertip contact exists at {contact_threshold_label}; "
                "source_scene_template missing"
            )
    else:
        out["decision"] = "REJECT"
        out["reason"] = (
            f"insufficient fingertip contact evidence at {contact_threshold_label} "
            f"with min_contact_frames={min_contact_frames}"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, default=Path("workspace/core4d/results/E102/box022_preflight_json"))
    parser.add_argument(
        "--core4d-root",
        type=Path,
        default=Path("/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real/human_object_motions"),
    )
    parser.add_argument("--include-out-of-scope", action="store_true")
    parser.add_argument("--contact-threshold-m", type=float, default=0.02)
    parser.add_argument("--contact-threshold-label", type=str, default="")
    parser.add_argument("--min-contact-frames", type=int, default=20)
    args = parser.parse_args()
    if args.core4d_root is not None:
        case_to_raw.CORE4D_ROOT = args.core4d_root

    threshold_label = args.contact_threshold_label or f"{int(round(args.contact_threshold_m * 100)):d}cm"
    rows = []
    for row in read_tsv(args.inventory):
        if row.get("e102_scope") != "True" and not args.include_out_of_scope:
            continue
        rows.append(preflight_row(row, args.json_dir, args.contact_threshold_m, threshold_label, args.min_contact_frames))

    fieldnames = [
        "target_task",
        "e102_scope",
        "case_role",
        "decision",
        "reason",
        "contact_threshold_m",
        "contact_threshold_label",
        "min_contact_frames",
        "raw_ok",
        "source_scene_xml_exists",
        "object_model_exists",
        "L_vote",
        "L_frac",
        "L_contact",
        "L_contact_frac",
        "R_vote",
        "R_frac",
        "R_contact",
        "R_contact_frac",
        "T",
        "quat_disable_world_up",
        "reach_support_status",
        "video_path",
        "json_path",
        "raw_person_npz",
        "raw_obj_poses",
        "source_scene_xml",
        "notes",
    ]
    write_tsv(args.out, rows, fieldnames)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    print(f"wrote {args.out} rows={len(rows)} counts={counts}")


if __name__ == "__main__":
    main()
