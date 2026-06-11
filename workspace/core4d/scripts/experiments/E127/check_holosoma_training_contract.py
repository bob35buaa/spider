#!/usr/bin/env python3
"""Static Holosoma training-contract preflight for E126 fragment exports."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/e126_adapter_manifest.tsv"
DEFAULT_OUT = REPO / "workspace/core4d/results/E127/holosoma_training_contract_preflight"
DEFAULT_HOLOSOMA = Path("/home/ubuntu/Workspace/holosoma")
EXPECTED_CONFIG_KEY = "g1_29dof_wbt_w_object_r135_box021_handbox_exp0601_v4_3"
EXPECTED_CLI_ALIAS = "exp:g1-29dof-wbt-w-object-r135-box021-handbox-exp0601-v4-3"
EXPECTED_OBJECT_NAME = "Box021"
EXPECTED_DECISION = "FRAGMENT_HOLDOUT_ONLY"
EXPECTED_ALIGNMENT = "min_frames_head_crop_fragment_pair_no_raw_window"

REQUIRED_NUMERIC_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
    "partner_hand_pos_w",
    "partner_hand_quat_w",
)
REQUIRED_METADATA_KEYS = ("body_names", "joint_names")


@dataclass
class ContractRow:
    case_id: str
    partner_case_id: str
    source_decision: str
    rl_train_allowed: str
    object_name: str
    paired_export_npz: str
    paired_frames_manifest: int
    output_fps_manifest: float
    config_key: str
    cli_alias: str
    object_urdf_path: str
    handbox_robot_urdf_path: str
    partner_hand_urdf_path: str
    motion_frames: int
    fps: float
    joint_pos_shape: str
    joint_vel_shape: str
    body_pos_shape: str
    partner_hand_pos_shape: str
    partner_hand_quat_shape: str
    numeric_nan_count: int
    numeric_nonfinite_count: int
    body_names_count: int
    joint_names_count: int
    object_contact_present: bool
    config_registered: bool
    cli_alias_in_train_scripts: bool
    object_urdf_exists: bool
    handbox_robot_urdf_exists: bool
    partner_hand_urdf_exists: bool
    structural_status: str
    rl_smoke_allowed: str
    training_launched: str
    failure_mode: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holosoma-root", type=Path, default=DEFAULT_HOLOSOMA)
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO / path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(rows: list[ContractRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ContractRow.__dataclass_fields__), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def contains(path: Path, text: str) -> bool:
    if not path.exists():
        return False
    return text in path.read_text(encoding="utf-8", errors="replace")


def shape_str(arr: np.ndarray) -> str:
    return "x".join(str(v) for v in arr.shape)


def validate_motion(path: Path, expected_frames: int, expected_fps: float) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if not path.exists():
        return {}, [f"missing_paired_export:{path}"]

    data = np.load(path, allow_pickle=True)
    missing = [key for key in (*REQUIRED_NUMERIC_KEYS, *REQUIRED_METADATA_KEYS) if key not in data.files]
    if missing:
        failures.append("missing_keys:" + ",".join(missing))

    if missing:
        return {}, failures

    frames = int(np.asarray(data["joint_pos"]).shape[0])
    fps_arr = np.asarray(data["fps"]).reshape(-1)
    fps = float(fps_arr[0]) if fps_arr.size else float("nan")
    expected_shapes = {
        "joint_vel": (frames, 35),
        "body_pos_w": (frames, 52, 3),
        "body_quat_w": (frames, 52, 4),
        "body_lin_vel_w": (frames, 52, 3),
        "body_ang_vel_w": (frames, 52, 3),
        "object_pos_w": (frames, 3),
        "object_quat_w": (frames, 4),
        "object_lin_vel_w": (frames, 3),
        "object_ang_vel_w": (frames, 3),
        "partner_hand_pos_w": (frames, 2, 3),
        "partner_hand_quat_w": (frames, 2, 4),
    }
    if tuple(np.asarray(data["joint_pos"]).shape) != (frames, 36):
        failures.append(f"bad_shape:joint_pos:{np.asarray(data['joint_pos']).shape}")
    for key, expected in expected_shapes.items():
        actual = tuple(np.asarray(data[key]).shape)
        if actual != expected:
            failures.append(f"bad_shape:{key}:{actual}:expected:{expected}")
    if frames != expected_frames:
        failures.append(f"frame_mismatch:{frames}:manifest:{expected_frames}")
    if abs(fps - expected_fps) > 1e-6:
        failures.append(f"fps_mismatch:{fps}:manifest:{expected_fps}")

    nan_count = 0
    nonfinite_count = 0
    for key in REQUIRED_NUMERIC_KEYS:
        arr = np.asarray(data[key])
        if not np.issubdtype(arr.dtype, np.number):
            failures.append(f"non_numeric_required_key:{key}:{arr.dtype}")
            continue
        nan_count += int(np.isnan(arr).sum())
        nonfinite_count += int((~np.isfinite(arr)).sum())
    if nan_count:
        failures.append(f"nan_count:{nan_count}")
    if nonfinite_count:
        failures.append(f"nonfinite_count:{nonfinite_count}")

    body_names = np.asarray(data["body_names"])
    joint_names = np.asarray(data["joint_names"])
    metrics = {
        "frames": frames,
        "fps": fps,
        "joint_pos_shape": shape_str(np.asarray(data["joint_pos"])),
        "joint_vel_shape": shape_str(np.asarray(data["joint_vel"])),
        "body_pos_shape": shape_str(np.asarray(data["body_pos_w"])),
        "partner_hand_pos_shape": shape_str(np.asarray(data["partner_hand_pos_w"])),
        "partner_hand_quat_shape": shape_str(np.asarray(data["partner_hand_quat_w"])),
        "numeric_nan_count": nan_count,
        "numeric_nonfinite_count": nonfinite_count,
        "body_names_count": int(body_names.shape[0]),
        "joint_names_count": int(joint_names.shape[0]),
        "object_contact_present": "object_contact" in data.files,
    }
    return metrics, failures


def build_summary(rows: list[ContractRow], holosoma_root: Path) -> dict[str, Any]:
    pass_rows = [row for row in rows if row.structural_status == "pass"]
    blocked_rows = [row for row in rows if row.rl_smoke_allowed == "false"]
    return {
        "experiment": "E127",
        "rows": len(rows),
        "structural_pass_rows": len(pass_rows),
        "rl_smoke_allowed_rows": sum(row.rl_smoke_allowed == "true" for row in rows),
        "rl_ready_rows": 0,
        "training_launched": False,
        "holosoma_root": str(holosoma_root),
        "config_key": EXPECTED_CONFIG_KEY,
        "cli_alias": EXPECTED_CLI_ALIAS,
        "blocked_rows": len(blocked_rows),
        "status": "pass" if len(pass_rows) == len(rows) and len(blocked_rows) == len(rows) else "fail",
    }


def write_markdown(rows: list[ContractRow], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# E127 Holosoma Training-Contract Preflight Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- structural pass rows: `{summary['structural_pass_rows']}`",
        f"- RL smoke allowed rows: `{summary['rl_smoke_allowed_rows']}`",
        f"- RL-ready rows: `{summary['rl_ready_rows']}`",
        f"- training launched: `{summary['training_launched']}`",
        f"- status: `{summary['status']}`",
        "",
        "| case | partner | structural | source decision | rl smoke allowed | frames | fps | config | notes |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.case_id}` | `{row.partner_case_id}` | `{row.structural_status}` | "
            f"`{row.source_decision}` | `{row.rl_smoke_allowed}` | {row.motion_frames} | "
            f"{row.fps:.1f} | `{row.config_key}` | {row.notes} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: E126 fragment exports satisfy the static Holosoma Box021 partner motion contract, "
            "but E127 keeps RL smoke/training blocked because the upstream rows remain `FRAGMENT_HOLDOUT_ONLY`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_rows = read_tsv(args.input_tsv)
    if not manifest_rows:
        raise SystemExit(f"no rows in {args.input_tsv}")

    holosoma_root = args.holosoma_root
    config_registry = holosoma_root / "src/holosoma/holosoma/config_values/experiment.py"
    wbt_experiment = holosoma_root / "src/holosoma/holosoma/config_values/wbt/g1/experiment.py"
    train_script_a = holosoma_root / "workspace/v3/scripts/train/train_core4d_r135_r137_exp0601_box021_partner.sh"
    train_script_b = holosoma_root / "workspace/v3/scripts/train/train_core4d_r135o_r136o_exp0602_box021_035_omnirt_partner.sh"
    object_urdf = (
        holosoma_root
        / "src/holosoma_retargeting/holosoma_retargeting/models/Box021/Box021.urdf"
    )
    handbox_robot_urdf = holosoma_root / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_handbox_m5.urdf"
    partner_hand_urdf = holosoma_root / "src/holosoma/holosoma/data/models/partner_hand/partner_hand.urdf"

    config_registered = contains(config_registry, EXPECTED_CONFIG_KEY) and contains(wbt_experiment, EXPECTED_CONFIG_KEY)
    cli_alias_in_train_scripts = contains(train_script_a, EXPECTED_CLI_ALIAS) and contains(train_script_b, EXPECTED_CLI_ALIAS)
    object_urdf_exists = object_urdf.exists() and object_urdf.stat().st_size > 0
    handbox_robot_urdf_exists = handbox_robot_urdf.exists() and handbox_robot_urdf.stat().st_size > 0
    partner_hand_urdf_exists = partner_hand_urdf.exists() and partner_hand_urdf.stat().st_size > 0

    out_rows: list[ContractRow] = []
    for src in manifest_rows:
        failures: list[str] = []
        case_id = src.get("case_id", "")
        partner_case_id = src.get("partner_case_id", "")
        source_decision = src.get("source_decision", "")
        rl_train_allowed = src.get("rl_train_allowed", "")
        object_name = src.get("object_name", "")
        paired_export = resolve_path(src.get("paired_export_npz", ""))
        paired_frames = int(float(src.get("paired_frames") or 0))
        output_fps = float(src.get("output_fps") or 0.0)

        if source_decision != EXPECTED_DECISION:
            failures.append(f"unexpected_source_decision:{source_decision}")
        if rl_train_allowed.lower() != "false":
            failures.append(f"rl_train_allowed_not_false:{rl_train_allowed}")
        if object_name != EXPECTED_OBJECT_NAME:
            failures.append(f"unexpected_object_name:{object_name}")
        if src.get("adapter_status") != "pass":
            failures.append(f"adapter_not_pass:{src.get('adapter_status')}")
        if src.get("alignment_policy") != EXPECTED_ALIGNMENT:
            failures.append(f"unexpected_alignment:{src.get('alignment_policy')}")
        if not config_registered:
            failures.append("holosoma_config_not_registered")
        if not cli_alias_in_train_scripts:
            failures.append("cli_alias_missing_from_train_scripts")
        if not object_urdf_exists:
            failures.append("missing_box021_urdf")
        if not handbox_robot_urdf_exists:
            failures.append("missing_handbox_robot_urdf")
        if not partner_hand_urdf_exists:
            failures.append("missing_partner_hand_urdf")

        metrics, motion_failures = validate_motion(paired_export, paired_frames, output_fps)
        failures.extend(motion_failures)
        structural_status = "pass" if not failures else "fail"
        rl_smoke_allowed = "false"
        notes = "structural contract ok; fragment-only label blocks RL smoke"
        if failures:
            notes = "blocked by: " + ";".join(failures)

        out_rows.append(
            ContractRow(
                case_id=case_id,
                partner_case_id=partner_case_id,
                source_decision=source_decision,
                rl_train_allowed=rl_train_allowed,
                object_name=object_name,
                paired_export_npz=str(paired_export),
                paired_frames_manifest=paired_frames,
                output_fps_manifest=output_fps,
                config_key=EXPECTED_CONFIG_KEY,
                cli_alias=EXPECTED_CLI_ALIAS,
                object_urdf_path=str(object_urdf),
                handbox_robot_urdf_path=str(handbox_robot_urdf),
                partner_hand_urdf_path=str(partner_hand_urdf),
                motion_frames=int(metrics.get("frames", 0)),
                fps=float(metrics.get("fps", 0.0)),
                joint_pos_shape=str(metrics.get("joint_pos_shape", "")),
                joint_vel_shape=str(metrics.get("joint_vel_shape", "")),
                body_pos_shape=str(metrics.get("body_pos_shape", "")),
                partner_hand_pos_shape=str(metrics.get("partner_hand_pos_shape", "")),
                partner_hand_quat_shape=str(metrics.get("partner_hand_quat_shape", "")),
                numeric_nan_count=int(metrics.get("numeric_nan_count", -1)),
                numeric_nonfinite_count=int(metrics.get("numeric_nonfinite_count", -1)),
                body_names_count=int(metrics.get("body_names_count", 0)),
                joint_names_count=int(metrics.get("joint_names_count", 0)),
                object_contact_present=bool(metrics.get("object_contact_present", False)),
                config_registered=config_registered,
                cli_alias_in_train_scripts=cli_alias_in_train_scripts,
                object_urdf_exists=object_urdf_exists,
                handbox_robot_urdf_exists=handbox_robot_urdf_exists,
                partner_hand_urdf_exists=partner_hand_urdf_exists,
                structural_status=structural_status,
                rl_smoke_allowed=rl_smoke_allowed,
                training_launched="false",
                failure_mode=";".join(failures),
                notes=notes,
            )
        )

    summary = build_summary(out_rows, holosoma_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_rows, args.output_dir / "e127_training_contract_manifest.tsv")
    write_json(summary, args.output_dir / "e127_training_contract_summary.json")
    write_markdown(out_rows, summary, args.output_dir / "e127_training_contract_summary.md")
    print(
        "wrote "
        f"{args.output_dir / 'e127_training_contract_summary.md'} "
        f"rows={summary['rows']} status={summary['status']}"
    )
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
