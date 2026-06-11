#!/usr/bin/env python3
"""Rebuild Box021 D003 targets from clean E103 templates and run CEM-entry gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO = Path(__file__).resolve().parents[4]
TASK_ROOT = REPO / "example_datasets/processed/core4d/unitree_g1/humanoid_object"
RESULTS_ROOT = REPO / "workspace/core4d/results/E107"
HOLOSOMA_D003 = Path(
    "/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/results/d003_omniretarget_spider_production/summary.json"
)
D002_ROOT = Path(
    "/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/results/d002_stage1_raw_contact_v2"
)

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "E083"))
sys.path.insert(0, str(THIS.parent / "E103"))
sys.path.insert(0, str(THIS.parent / "convert"))

import create_upperobj_cases as upperobj  # type: ignore  # noqa: E402
import generate_scene_act  # type: ignore  # noqa: E402
from audit_scene_inertials import read_scene as audit_scene  # type: ignore  # noqa: E402


THRESHOLDS = (0.03, 0.05)
PERSON_INDEX = {"person1": 0, "person2": 1}
FIELDS = [
    "target_task",
    "derived_task",
    "person",
    "sequence",
    "d003_status",
    "d003_decision",
    "failure_mode",
    "cem_gate",
    "source_scene_task",
    "source_scene_clean",
    "reconstruction_ok",
    "qpos_shape",
    "qpos_matches_legacy_spider",
    "scene_dims",
    "scene_act_dims",
    "scene_robot_polluted_mass_29_632",
    "scene_act_robot_polluted_mass_29_632",
    "missing_leg_pairs",
    "missing_upper_pairs",
    "raw_decision_3cm",
    "raw_decision_5cm",
    "target_both_active_frac_3cm",
    "target_both_active_frac_5cm",
    "target_left_active_frac_3cm",
    "target_right_active_frac_3cm",
    "target_left_active_frac_5cm",
    "target_right_active_frac_5cm",
    "partner_any_active_frac_3cm",
    "partner_any_active_frac_5cm",
    "raw_contact_score_3cm",
    "raw_contact_score_5cm",
    "old_spider_trajectory_path",
    "raw_contact_proxy_path",
    "reason",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def fmt(values: np.ndarray, digits: int = 6) -> str:
    return " ".join(f"{float(v):.{digits}f}".rstrip("0").rstrip(".") for v in values)


def read_d003_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    out = [row for row in rows if row.get("object_name") == "Box021" and row.get("target_task", "").startswith("d003_box021_")]
    out.sort(key=lambda r: (r.get("date", ""), r.get("seq", ""), r.get("person", "")))
    return out


def patch_object_pose(scene: Path, qpos0: np.ndarray) -> None:
    tree = ET.parse(scene)
    root = tree.getroot()
    obj = next((body for body in root.iter("body") if body.get("name") == "object"), None)
    if obj is None:
        raise ValueError(f"object body missing: {scene}")
    obj.set("pos", fmt(qpos0[-7:-4]))
    obj.set("quat", fmt(qpos0[-4:]))
    tree.write(scene, encoding="unicode")


def copy_clean_target(row: dict[str, Any], force: bool) -> tuple[Path, dict[str, Any]]:
    source_task = row["source_scene_task"]
    target_task = row["target_task"]
    derived_task = f"{target_task}_e107_clean"
    source_dir = TASK_ROOT / source_task
    old_task_dir = TASK_ROOT / target_task
    derived_dir = TASK_ROOT / derived_task
    old_traj = Path(row["spider_trajectory_path"])
    if not old_traj.is_file():
        old_traj = old_task_dir / "0/trajectory_kinematic.npz"
    if not (source_dir / "scene.xml").is_file():
        raise FileNotFoundError(source_dir / "scene.xml")
    if not old_traj.is_file():
        raise FileNotFoundError(old_traj)

    if derived_dir.exists():
        if force:
            shutil.rmtree(derived_dir)
        else:
            raise FileExistsError(f"{derived_dir} exists; rerun with --force")

    (derived_dir / "0").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "scene.xml", derived_dir / "scene.xml")
    old_data = np.load(old_traj, allow_pickle=True)
    qpos = old_data["qpos"]
    trajectory_payload = {key: old_data[key] for key in old_data.files}
    np.savez_compressed(derived_dir / "0/trajectory_kinematic.npz", **trajectory_payload)
    patch_object_pose(derived_dir / "scene.xml", qpos[0])

    source_info = {}
    source_info_path = source_dir / "task_info.json"
    if source_info_path.is_file():
        source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    task_info = {
        "task": derived_task,
        "source_task_e107": source_task,
        "legacy_d003_target_task": target_task,
        "legacy_spider_trajectory_path": str(old_traj),
        "e107_clean_reconstruction": True,
        "e107_note": "Clean Box021 target rebuilt from E103 source template; old target scene/scene_act are not copied.",
        "e107_source_template_info": source_info,
        "object_initial_pos": qpos[0, -7:-4].tolist(),
        "object_initial_quat": qpos[0, -4:].tolist(),
    }
    (derived_dir / "task_info.json").write_text(
        json.dumps(task_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    generate_scene_act.generate_scene_act(derived_task)
    added_leg, added_upper = upperobj.patch_scene_act(derived_dir / "scene_act.xml", source_task, derived_task)
    meta = {
        "derived_task": derived_task,
        "old_traj": str(old_traj),
        "qpos_shape": list(qpos.shape),
        "added_leg_pairs": added_leg,
        "added_upper_pairs": added_upper,
    }
    return derived_dir, meta


def dims(model: mujoco.MjModel) -> str:
    return f"nq={model.nq},nv={model.nv},nu={model.nu}"


def validate_rebuilt(row: dict[str, Any], derived_dir: Path) -> dict[str, Any]:
    target_task = row["target_task"]
    old_traj = Path(row["spider_trajectory_path"])
    if not old_traj.is_file():
        old_traj = TASK_ROOT / target_task / "0/trajectory_kinematic.npz"
    new_qpos = np.load(derived_dir / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    old_qpos = np.load(old_traj, allow_pickle=True)["qpos"]
    model_scene = mujoco.MjModel.from_xml_path(str(derived_dir / "scene.xml"))
    model_act = mujoco.MjModel.from_xml_path(str(derived_dir / "scene_act.xml"))
    scene_audit = audit_scene(derived_dir / "scene.xml")
    act_audit = audit_scene(derived_dir / "scene_act.xml")
    root = ET.parse(derived_dir / "scene_act.xml").getroot()
    pair_names = {pair.get("name") for pair in root.iter("pair")}
    missing_leg = sorted({f"{geom}_object" for geom in upperobj.LEG_FOOT_GEOMS} - pair_names)
    missing_upper = sorted({f"{geom}_object" for geom in upperobj.UPPER_BODY_GEOMS} - pair_names)
    ok = (
        model_scene.nq == 43
        and model_scene.nv == 41
        and model_scene.nu == 29
        and model_act.nq == 42
        and model_act.nv == 41
        and model_act.nu == 35
        and new_qpos.shape == old_qpos.shape
        and bool(np.allclose(new_qpos, old_qpos))
        and scene_audit.get("robot_polluted_mass_29_632") == "False"
        and act_audit.get("robot_polluted_mass_29_632") == "False"
        and not missing_leg
        and not missing_upper
    )
    return {
        "ok": ok,
        "qpos_shape": str(list(new_qpos.shape)),
        "qpos_matches_legacy_spider": str(bool(np.allclose(new_qpos, old_qpos))),
        "scene_dims": dims(model_scene),
        "scene_act_dims": dims(model_act),
        "scene_robot_polluted_mass_29_632": scene_audit.get("robot_polluted_mass_29_632", ""),
        "scene_act_robot_polluted_mass_29_632": act_audit.get("robot_polluted_mass_29_632", ""),
        "missing_leg_pairs": ",".join(missing_leg),
        "missing_upper_pairs": ",".join(missing_upper),
    }


def fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else 0.0


def longest_run_fraction(mask: np.ndarray) -> float:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / mask.size) if mask.size else 0.0


def threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100))}cm"


def raw_metrics(row: dict[str, Any], threshold: float, d002_root: Path) -> dict[str, Any]:
    slug = f"{row['date']}_{row['seq']}_box021"
    path = d002_root / "per_sequence" / slug / "raw_contact_proxy.npz"
    if not path.is_file():
        return {"decision": "raw_contact_missing", "path": str(path), "reason": "raw_contact_proxy_missing"}
    data = np.load(path, allow_pickle=True)
    thresholds = [float(x) for x in data["thresholds_m"].tolist()]
    if threshold not in thresholds:
        return {"decision": "raw_contact_missing", "path": str(path), "reason": f"threshold_{threshold}_missing"}
    ti = thresholds.index(threshold)
    pi = PERSON_INDEX[row["person"]]
    partner = 1 - pi
    active = np.asarray(data["active_mask"], dtype=bool)
    masks = np.asarray(data["masks"])[:, :, :, ti]
    target_left = masks[:, pi, 0]
    target_right = masks[:, pi, 1]
    target_any = target_left | target_right
    target_both = target_left & target_right
    partner_any = masks[:, partner, 0] | masks[:, partner, 1]
    target_left_active = fraction(target_left[active])
    target_right_active = fraction(target_right[active])
    target_any_active = fraction(target_any[active])
    target_both_active = fraction(target_both[active])
    partner_any_active = fraction(partner_any[active])
    balanced_active = min(target_left_active, target_right_active)
    both_longest = longest_run_fraction(target_both[active])
    score = (
        35.0 * target_both_active
        + 20.0 * balanced_active
        + 20.0 * target_any_active
        + 15.0 * partner_any_active
        + 10.0 * both_longest
    )
    if target_both_active >= 0.25 and balanced_active >= 0.35 and partner_any_active >= 0.25:
        decision = "raw_contact_pass"
    elif target_any_active >= 0.40 and balanced_active >= 0.20 and partner_any_active >= 0.15:
        decision = "raw_contact_review"
    else:
        decision = "raw_contact_fail"
    return {
        "decision": decision,
        "path": str(path),
        "target_left_active_frac": round(target_left_active, 4),
        "target_right_active_frac": round(target_right_active, 4),
        "target_both_active_frac": round(target_both_active, 4),
        "partner_any_active_frac": round(partner_any_active, 4),
        "score": round(max(0.0, score), 3),
        "reason": "",
    }


def source_clean(source_task: str) -> str:
    scene = TASK_ROOT / source_task / "scene.xml"
    if not scene.is_file():
        return "missing"
    audit = audit_scene(scene)
    clean = audit.get("robot_polluted_mass_29_632") == "False" and int(audit.get("robot_inertial_unique_pairs") or 0) > 5
    return str(clean)


def classify_gate(d003_ok: bool, reconstruction_ok: bool, raw: dict[str, dict[str, Any]], reason: str) -> tuple[str, str]:
    if not d003_ok:
        return "preprocess_infeasible", reason or "D003 OmniRetarget/SPIDER preprocess failed"
    if not reconstruction_ok:
        return "reconstruction_failed", reason or "clean reconstruction or validation failed"
    decisions = {label: value.get("decision") for label, value in raw.items()}
    if "raw_contact_pass" in decisions.values():
        return "cem_ready", "D003 pass; clean reconstruction pass; raw-contact pass at 3cm or 5cm"
    if "raw_contact_review" in decisions.values():
        return "gate_review", "raw contact is review-only; visual review required before CEM"
    if all(v == "raw_contact_fail" for v in decisions.values()):
        return "raw_contact_fail_3cm_5cm", "3cm and 5cm raw-contact gates both fail"
    return "raw_contact_missing", "raw contact proxy or requested threshold missing"


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d003_rows = read_d003_rows(args.d003_summary)
    rows_out: list[dict[str, Any]] = []
    build_meta: list[dict[str, Any]] = []
    for row in d003_rows:
        d003_ok = row.get("d003_status") == "ok"
        target_task = row["target_task"]
        derived_task = f"{target_task}_e107_clean"
        base = {
            "target_task": target_task,
            "derived_task": derived_task,
            "person": row.get("person", ""),
            "sequence": row.get("sequence", ""),
            "d003_status": row.get("d003_status", ""),
            "d003_decision": row.get("d003_decision", ""),
            "source_scene_task": row.get("source_scene_task", ""),
            "source_scene_clean": source_clean(row.get("source_scene_task", "")),
            "old_spider_trajectory_path": row.get("spider_trajectory_path", ""),
        }
        validation: dict[str, Any] = {}
        rebuild_reason = ""
        if d003_ok:
            try:
                derived_dir, meta = copy_clean_target(row, args.force)
                validation = validate_rebuilt(row, derived_dir)
                build_meta.append({"target_task": target_task, **meta, "validation": validation})
            except Exception as exc:  # noqa: BLE001
                validation = {"ok": False}
                rebuild_reason = f"{type(exc).__name__}: {exc}"
                build_meta.append({"target_task": target_task, "derived_task": derived_task, "error": rebuild_reason})
        raw = {threshold_label(t): raw_metrics(row, t, args.d002_root) for t in THRESHOLDS}
        gate, reason = classify_gate(d003_ok, bool(validation.get("ok")), raw, rebuild_reason or row.get("failure_reason", ""))
        out = {
            **base,
            "failure_mode": gate,
            "cem_gate": str(gate == "cem_ready"),
            "reconstruction_ok": str(bool(validation.get("ok"))),
            "qpos_shape": validation.get("qpos_shape", ""),
            "qpos_matches_legacy_spider": validation.get("qpos_matches_legacy_spider", ""),
            "scene_dims": validation.get("scene_dims", ""),
            "scene_act_dims": validation.get("scene_act_dims", ""),
            "scene_robot_polluted_mass_29_632": validation.get("scene_robot_polluted_mass_29_632", ""),
            "scene_act_robot_polluted_mass_29_632": validation.get("scene_act_robot_polluted_mass_29_632", ""),
            "missing_leg_pairs": validation.get("missing_leg_pairs", ""),
            "missing_upper_pairs": validation.get("missing_upper_pairs", ""),
            "raw_contact_proxy_path": raw["3cm"].get("path", "") or raw["5cm"].get("path", ""),
            "reason": reason,
        }
        for label in ("3cm", "5cm"):
            metrics = raw[label]
            out[f"raw_decision_{label}"] = metrics.get("decision", "")
            out[f"target_both_active_frac_{label}"] = metrics.get("target_both_active_frac", "")
            out[f"target_left_active_frac_{label}"] = metrics.get("target_left_active_frac", "")
            out[f"target_right_active_frac_{label}"] = metrics.get("target_right_active_frac", "")
            out[f"partner_any_active_frac_{label}"] = metrics.get("partner_any_active_frac", "")
            out[f"raw_contact_score_{label}"] = metrics.get("score", "")
        rows_out.append(out)
    return rows_out, build_meta


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(row["failure_mode"] for row in rows)
    ready = [row for row in rows if row["cem_gate"] == "True"]
    lines = [
        "# E107 Box021 Clean Reconstruction Gate Summary",
        "",
        f"- total case-person: `{len(rows)}`",
        f"- clean rebuilt and CEM-ready: `{len(ready)}`",
        f"- failure mode counts: `{dict(counts)}`",
        "",
        "| failure_mode | count |",
        "|---|---:|",
    ]
    for key, value in counts.most_common():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## CEM-ready rows", "", "| target | derived | raw 3cm | raw 5cm | reason |", "|---|---|---|---|---|"])
    if ready:
        for row in ready:
            lines.append(
                f"| `{row['target_task']}` | `{row['derived_task']}` | `{row['raw_decision_3cm']}` | "
                f"`{row['raw_decision_5cm']}` | {row['reason']} |"
            )
    else:
        lines.append("| - | - | - | - | no CEM-ready Box021 rows |")
    lines.extend(["", "## All rows", "", "| target | d003 | rebuild | gate | raw 3cm | raw 5cm | reason |", "|---|---|---|---|---|---|---|"])
    for row in rows:
        lines.append(
            f"| `{row['target_task']}` | `{row['d003_decision']}` | `{row['reconstruction_ok']}` | "
            f"`{row['failure_mode']}` | `{row['raw_decision_3cm']}` | `{row['raw_decision_5cm']}` | {row['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d003-summary", type=Path, default=HOLOSOMA_D003)
    parser.add_argument("--d002-root", type=Path, default=D002_ROOT)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows, build_meta = build_rows(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "box021_clean_gate_summary.tsv", rows)
    (args.out_dir / "box021_clean_gate_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "box021_clean_build_meta.json").write_text(
        json.dumps(build_meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(rows, args.out_dir / "box021_clean_gate_summary.md")
    counts = Counter(row["failure_mode"] for row in rows)
    print(json.dumps({"rows": len(rows), "failure_modes": dict(counts)}, indent=2, ensure_ascii=False, sort_keys=True))
    ready = [row["derived_task"] for row in rows if row["cem_gate"] == "True"]
    if ready:
        print("cem_ready:", " ".join(ready))
    else:
        print("cem_ready: none")


if __name__ == "__main__":
    main()
