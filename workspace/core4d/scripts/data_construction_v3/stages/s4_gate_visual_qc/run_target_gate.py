#!/usr/bin/env python3
"""Run S4 target gate checks for Stage2b outputs."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import sys

SCRIPT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "data_construction_v3")
for _path in (SCRIPT_ROOT / "lib", SCRIPT_ROOT / "state", SCRIPT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from typing import Any

import mujoco
import numpy as np

from common import SCHEMA_VERSION, find_spider_repo, json_dumps, read_tsv, timestamp, write_json, write_tsv
from geometry import normalize_contact_pos_source


def task_name(row: dict[str, str]) -> str:
    return f"{row['date']}-{row['seq']}-{row['person']}-{row['object_name']}_with_obj"


def expected_paths(row: dict[str, str], spider_repo: Path, task_root: Path | None = None) -> dict[str, Path | None]:
    result_root = Path(row["result_root"]).expanduser()
    target_task = row["target_task"]
    task = task_name(row)
    root = task_root or (spider_repo / "example_datasets/processed/core4d/unitree_g1/humanoid_object")
    return {
        "converted_npz": result_root / f"holosoma_{target_task}" / "converted" / f"{task}.npz",
        "retargeted_npz": result_root / f"holosoma_{target_task}" / "retargeted" / f"{task}_original.npz",
        "trimmed_npz": result_root / f"holosoma_{target_task}" / "trimmed" / f"{task}_original.npz",
        "contact_mask_npz": Path(row["contact_mask_npz"]).expanduser() if row.get("contact_mask_npz") else None,
        "verify_summary": result_root / f"{target_task}_verify_summary.json",
        "target_scene": root / target_task / "scene.xml",
        "scene_act": root / target_task / "scene_act.xml",
        "trajectory": root / target_task / str(row.get("data_id", "0") or "0") / "trajectory_kinematic.npz",
    }


def fnum(text: str | None) -> float:
    try:
        return float(text or "nan")
    except ValueError:
        return float("nan")


def robot_inertial_polluted(scene: Path) -> tuple[str, str]:
    if not scene.is_file():
        return "unknown", "missing_scene"
    try:
        root = ET.parse(scene).getroot()
    except Exception as exc:  # noqa: BLE001
        return "unknown", f"xml_parse_error:{type(exc).__name__}"
    polluted = False
    unique_pairs: set[tuple[str, str]] = set()
    for body in root.iter("body"):
        if body.get("name") == "object":
            continue
        inertial = next((child for child in body if child.tag == "inertial"), None)
        if inertial is None:
            continue
        mass = inertial.get("mass", "")
        inertia = inertial.get("diaginertia", "")
        unique_pairs.add((mass, inertia))
        if abs(fnum(mass) - 29.632) < 1e-3:
            polluted = True
    return str(polluted), str(len(unique_pairs))


def load_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def check_outputs(row: dict[str, str], spider_repo: Path, task_root: Path | None) -> dict[str, Any]:
    paths = expected_paths(row, spider_repo, task_root)
    exists = {key: (path.is_file() if path is not None else False) for key, path in paths.items()}
    missing_required = [key for key in ("trimmed_npz", "target_scene", "scene_act", "trajectory") if not exists[key]]
    out: dict[str, Any] = {
        "paths": {key: str(path) if path is not None else "" for key, path in paths.items()},
        "exists": {key: str(value) for key, value in exists.items()},
        "missing_required": ",".join(missing_required),
    }
    if missing_required:
        out.update(
            {
                "target_gate_status": "not_run",
                "failure_mode": "stage2b_outputs_missing",
                "decision_notes": f"missing required outputs: {','.join(missing_required)}",
            }
        )
        return out

    try:
        scene_model = mujoco.MjModel.from_xml_path(str(paths["target_scene"]))
        act_model = mujoco.MjModel.from_xml_path(str(paths["scene_act"]))
        traj = load_npz(paths["trajectory"])
        trimmed = load_npz(paths["trimmed_npz"])
        qpos = np.asarray(traj["qpos"])
        qvel = np.asarray(traj["qvel"]) if "qvel" in traj else None
        ctrl = np.asarray(traj["ctrl"]) if "ctrl" in traj else None
        contact = np.asarray(traj["contact"]) if "contact" in traj else None
        contact_pos_source = normalize_contact_pos_source("contact_pos" in traj)
        trimmed_match = bool("qpos" in trimmed and trimmed["qpos"].shape == qpos.shape and np.allclose(trimmed["qpos"], qpos))
        qpos_ok = qpos.ndim == 2 and qpos.shape[1] == scene_model.nq and np.isfinite(qpos).all()
        qvel_ok = qvel is not None and qvel.ndim == 2 and qvel.shape[0] == qpos.shape[0] and np.isfinite(qvel).all()
        ctrl_ok = ctrl is not None and ctrl.ndim == 2 and ctrl.shape[0] == qpos.shape[0] and np.isfinite(ctrl).all()
        contact_ok = contact is not None and contact.shape[0] == qpos.shape[0]
        scene_polluted, scene_unique = robot_inertial_polluted(paths["target_scene"])
        act_polluted, act_unique = robot_inertial_polluted(paths["scene_act"])
        hard_failures: list[str] = []
        if not qpos_ok:
            hard_failures.append("qpos_shape_or_finite_fail")
        if not trimmed_match:
            hard_failures.append("trimmed_qpos_mismatch")
        if scene_model.nq != 43 or scene_model.nv != 41:
            hard_failures.append("target_scene_dims_unexpected")
        if act_model.nq != 42 or act_model.nv != 41:
            hard_failures.append("scene_act_dims_unexpected")
        if scene_polluted == "True" or act_polluted == "True":
            hard_failures.append("robot_inertial_polluted")
        status = "pass" if not hard_failures else "reject"
        out.update(
            {
                "target_gate_status": status,
                "failure_mode": "" if status == "pass" else ";".join(hard_failures),
                "decision_notes": "machine gate pass" if status == "pass" else "machine gate reject",
                "qpos_shape": str(list(qpos.shape)),
                "qvel_shape": str(list(qvel.shape)) if qvel is not None else "",
                "ctrl_shape": str(list(ctrl.shape)) if ctrl is not None else "",
                "contact_shape": str(list(contact.shape)) if contact is not None else "",
                "qpos_ok": str(qpos_ok),
                "qvel_ok": str(qvel_ok),
                "ctrl_ok": str(ctrl_ok),
                "contact_ok": str(contact_ok),
                "trimmed_qpos_matches_spider_qpos": str(trimmed_match),
                "scene_dims": f"nq={scene_model.nq},nv={scene_model.nv},nu={scene_model.nu}",
                "scene_act_dims": f"nq={act_model.nq},nv={act_model.nv},nu={act_model.nu}",
                "scene_robot_polluted_mass_29_632": scene_polluted,
                "scene_robot_inertial_unique_pairs": scene_unique,
                "scene_act_robot_polluted_mass_29_632": act_polluted,
                "scene_act_robot_inertial_unique_pairs": act_unique,
                "frames": str(int(qpos.shape[0])),
                "contact_pos_source": contact_pos_source,
                "pelvis_end_z": f"{float(qpos[-1, 2]):.6f}" if qpos_ok else "",
                "pelvis_tilt_end": "",
            }
        )
    except Exception as exc:  # noqa: BLE001 - keep per-case gate evidence.
        out.update(
            {
                "target_gate_status": "reject",
                "failure_mode": "target_gate_exception",
                "decision_notes": f"{type(exc).__name__}: {exc}",
            }
        )
    return out


def build_rows(stage2b_rows: list[dict[str, str]], spider_repo: Path, task_root: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stage2b_rows:
        if row.get("stage2b_decision") != "stage2b_ready":
            gate = {
                "target_gate_status": "not_run",
                "failure_mode": row.get("stage2b_decision", "stage2b_not_ready"),
                "decision_notes": row.get("decision_notes", ""),
                "paths": {},
                "exists": {},
            }
        else:
            gate = check_outputs(row, spider_repo, task_root)
        rows.append(
            {
                "stage": "S4_target_gate",
                "case_id": row.get("case_id", ""),
                "retarget_variant_id": row.get("retarget_variant_id", ""),
                "target_variant_id": row.get("target_variant_id", "ref_fk"),
                "target_task": row.get("target_task", ""),
                "object_key": row.get("object_key", ""),
                "object_name": row.get("object_name", ""),
                "date": row.get("date", ""),
                "seq": row.get("seq", ""),
                "person": row.get("person", ""),
                "person_idx": row.get("person_idx", ""),
                "stage2b_decision": row.get("stage2b_decision", ""),
                "target_gate_status": gate.get("target_gate_status", "not_run"),
                "visual_qc_status": "not_run",
                "failure_mode": gate.get("failure_mode", ""),
                "decision_notes": gate.get("decision_notes", ""),
                "target_scene": gate.get("paths", {}).get("target_scene", ""),
                "scene_act": gate.get("paths", {}).get("scene_act", ""),
                "trajectory": gate.get("paths", {}).get("trajectory", ""),
                "trimmed_npz": gate.get("paths", {}).get("trimmed_npz", ""),
                "retargeted_npz": gate.get("paths", {}).get("retargeted_npz", ""),
                "contact_mask_npz": gate.get("paths", {}).get("contact_mask_npz", ""),
                "contact_mask_label": row.get("contact_mask_label", row.get("raw_contact_threshold_label", "")),
                "contact_mask_person_idx": row.get("contact_mask_person_idx", row.get("person_idx", "")),
                "contact_mask_status": row.get("contact_mask_status", ""),
                "contact_mask_time_axis": row.get("contact_mask_time_axis", ""),
                "stage2b_contact_mask_npz_expected": row.get("stage2b_contact_mask_npz_expected", ""),
                "raw_contact_artifact_npz": row.get("raw_contact_artifact_npz", ""),
                "raw_contact_time_axis": row.get("raw_contact_time_axis", ""),
                "raw_to_trimmed_mapping_status": row.get("raw_to_trimmed_mapping_status", ""),
                "left_active_frac": row.get("left_active_frac", ""),
                "right_active_frac": row.get("right_active_frac", ""),
                "both_active_frac": row.get("both_active_frac", ""),
                "left_longest_run_frac": row.get("left_longest_run_frac", ""),
                "right_longest_run_frac": row.get("right_longest_run_frac", ""),
                "both_longest_run_frac": row.get("both_longest_run_frac", ""),
                "contact_target_status": row.get("contact_target_status", ""),
                "contact_target_npz": row.get("contact_target_npz", row.get("target_npz", "")),
                "contact_target_source": row.get("contact_target_source", ""),
                "contact_target_frame": row.get("contact_target_frame", ""),
                "contact_target_time_axis": row.get("contact_target_time_axis", ""),
                "contact_route_diagnostic_ref": row.get("contact_route_diagnostic_ref", row.get("route_diagnostic_ref", "")),
                "verify_summary": gate.get("paths", {}).get("verify_summary", ""),
                "missing_required": gate.get("missing_required", ""),
                "qpos_shape": gate.get("qpos_shape", ""),
                "qvel_shape": gate.get("qvel_shape", ""),
                "ctrl_shape": gate.get("ctrl_shape", ""),
                "contact_shape": gate.get("contact_shape", ""),
                "qpos_ok": gate.get("qpos_ok", ""),
                "qvel_ok": gate.get("qvel_ok", ""),
                "ctrl_ok": gate.get("ctrl_ok", ""),
                "contact_ok": gate.get("contact_ok", ""),
                "trimmed_qpos_matches_spider_qpos": gate.get("trimmed_qpos_matches_spider_qpos", ""),
                "scene_dims": gate.get("scene_dims", ""),
                "scene_act_dims": gate.get("scene_act_dims", ""),
                "scene_robot_polluted_mass_29_632": gate.get("scene_robot_polluted_mass_29_632", ""),
                "scene_act_robot_polluted_mass_29_632": gate.get("scene_act_robot_polluted_mass_29_632", ""),
                "frames": gate.get("frames", ""),
                "contact_pos_source": gate.get("contact_pos_source", ""),
                "pelvis_end_z": gate.get("pelvis_end_z", ""),
                "pelvis_tilt_end": gate.get("pelvis_tilt_end", ""),
                "path_exists_json": json.dumps(gate.get("exists", {}), sort_keys=True),
                "schema_version": SCHEMA_VERSION,
                "updated_at": timestamp(),
            }
        )
    rows.sort(key=lambda r: (r["target_gate_status"], r["case_id"], r["retarget_variant_id"]))
    return rows


def summarize(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    return {
        "stage": "S4_target_gate",
        "created_at": timestamp(),
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "target_gate_counts": dict(Counter(row["target_gate_status"] for row in rows)),
        "visual_qc_counts": dict(Counter(row["visual_qc_status"] for row in rows)),
        "failure_mode_counts": dict(Counter(row["failure_mode"] for row in rows if row["failure_mode"])),
        "out_dir": str(out_dir),
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# S4 target gate summary",
        "",
        f"- rows: `{summary['rows']}`",
        "",
        "## target gate counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, count in summary["target_gate_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## non-pass rows", "", "| case | variant | status | failure | notes |", "|---|---|---|---|---|"])
    for row in rows:
        if row["target_gate_status"] != "pass":
            lines.append(
                f"| `{row['case_id']}` | `{row['retarget_variant_id']}` | `{row['target_gate_status']}` | `{row['failure_mode']}` | `{row['decision_notes']}` |"
            )
    lines.extend(
        [
            "",
            "说明：`pelvis_tilt_end` 只作为 diagnostic；本脚本不把 visual QC 当作机器硬 gate。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2b-manifest-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spider-repo", type=Path, default=None)
    parser.add_argument("--task-root", type=Path, default=None)
    args = parser.parse_args()

    spider_repo = (args.spider_repo or find_spider_repo()).resolve()
    task_root = args.task_root.resolve() if args.task_root else None
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_tsv(args.stage2b_manifest_tsv), spider_repo, task_root)
    fields = list(rows[0].keys()) if rows else []
    write_tsv(out_dir / "target_gate_manifest.tsv", rows, fields)
    write_json(out_dir / "target_gate_manifest.json", rows)
    summary = summarize(rows, out_dir)
    write_json(out_dir / "target_gate_summary.json", summary)
    (out_dir / "target_gate_summary.md").write_text(markdown_summary(summary, rows), encoding="utf-8")
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
