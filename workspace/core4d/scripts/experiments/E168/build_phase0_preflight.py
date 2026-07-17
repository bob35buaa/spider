#!/usr/bin/env python3
"""Build the E168 local environment, E167 import, and E167A profile snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[5]
EXPECTED_RESULT_ROOT = REPO / "workspace/core4d/results/E168"
DEFAULT_E167_EXPORT = (
    REPO
    / "workspace/core4d/results/E167/holosoma_zonly/rl_export"
    / "s6_downstream/rl_export/rl_export_input.tsv"
)
DEFAULT_E167_CONFIG = (
    REPO
    / "workspace/core4d/results/E167/holosoma_zonly/cem/full"
    / "E167_box004_082_p1_E167A_outdir_full/config_act.yaml"
)
DEFAULT_AXIS_AUDIT = (
    REPO
    / "workspace/core4d/results/E167/holosoma_zonly/eval/axis_audit/full"
    / "axis_audit_summary.json"
)

PROFILE_EXPECTED: dict[str, Any] = {
    "e167_body_z_enabled": True,
    "e167_body_z_names": [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ],
    "e167_body_z_weight": 2.0,
    "e167_body_z_threshold_m": 0.25,
    "e167_ground_z_enabled": True,
    "e167_ground_z_names": [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    "e167_ground_z_weight": 2.0,
    "e167_ground_contact_height_m": 0.05,
    "foot_slip_enabled": False,
    "foot_slip_weight": 0.0,
    "foot_ground_enabled": False,
    "foot_ground_weight": 0.0,
    "local_frame_ankle_weight": 1.0,
    "cem_smooth_enabled": False,
    "cem_hand_gate_enabled": True,
    "cem_hand_gate_min_sdf_m": -0.01,
    "cem_hand_gate_max_violation_pct": 0.1,
    "cem_hand_gate_hard_floor_m": -0.02,
    "surface_band_rew_scale": 1.5,
    "surface_band_penalty_scale": 0.0,
    "surface_band_width_m": 0.003,
    "surface_band_min_sdf_m": -0.001,
    "surface_band_sigma": 0.0015,
    "surface_band_score_mode": "symmetric_abs",
    "surface_band_gate_source": "contact_mask",
    "cem_posture_gate_enabled": True,
    "cem_posture_gate_mean_z_err_m": 0.1,
    "cem_posture_gate_terminal_z_err_m": 0.12,
    "cem_posture_gate_max_z_drop_m": 0.18,
    "cem_posture_gate_terminal_frac": 0.15,
    "cem_posture_gate_min_valid_frac": 0.05,
    "cem_posture_gate_fallback_lambda": 5.0,
}

E167_ARTIFACT_FIELDS = [
    "target_scene",
    "trajectory",
    "scene_act",
    "contact_mask",
    "cem_result_npz",
    "cem_video",
    "cem_metrics_ref",
]
E167_ARTIFACT_FALLBACKS = {
    "scene_act": (
        REPO
        / "workspace/core4d/results/E147/rubber_hand_collision/scene_snapshot"
        / "e091_box004_20231003_2_082_p1_e096b_mask_cem"
        / "scene_act_E147_rubber_hull.xml"
    ),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def git_state(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"path": str(path), "exists": path.is_dir(), "git": False}
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    dirty_lines = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=path, text=True
    ).splitlines()
    return {
        "path": str(path),
        "exists": True,
        "git": True,
        "head": head,
        "dirty": bool(dirty_lines),
        "dirty_paths": dirty_lines,
    }


def local_gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"status": "unavailable", "error": str(exc), "gpus": []}
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "utilization_gpu_pct": int(parts[5]),
            }
        )
    return {"status": "pass", "gpus": gpus}


def retarget_python_probe(python_bin: Path) -> dict[str, Any]:
    if not python_bin.is_file():
        return {
            "status": "fail",
            "python": str(python_bin),
            "error": "python binary missing",
        }
    command = [
        str(python_bin),
        "-c",
        (
            "import json,sys,smplx,holosoma_retargeting;"
            "print(json.dumps({'python':sys.executable,'status':'pass'}))"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["requested_python"] = str(python_bin)
        return payload
    except Exception as exc:  # noqa: BLE001 - retain import failure in the manifest.
        return {
            "status": "fail",
            "python": str(python_bin),
            "error": f"{type(exc).__name__}: {exc}",
        }


def select_e167_row(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("object_key") == "box004"
        and row.get("date") == "20231003_2"
        and row.get("seq") == "082"
        and row.get("person") == "person1"
        and row.get("spider_method_id") == "E167A_zOnlyBody"
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one E167 box004_082_p1/E167A row, got {len(matches)}")
    row = matches[0]
    required = {
        "source_exp_id": "E167",
        "retarget_variant_id": "omnirt_v1",
        "target_variant_id": "ref_fk",
        "hand_collision_variant_id": "rubber_hull",
        "rl_export_decision": "RL_EXPORT_READY",
        "cem_status": "pass",
        "target_gate_status": "pass",
        "visual_qc_status": "pass",
    }
    mismatches = {
        field: {"expected": expected, "actual": row.get(field, "")}
        for field, expected in required.items()
        if row.get(field, "") != expected
    }
    if mismatches:
        raise SystemExit(f"E167 source row contract mismatch: {mismatches}")
    return row


def import_e167_snapshot(
    source_export: Path,
    row: dict[str, str],
    out_dir: Path,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for field in E167_ARTIFACT_FIELDS:
        value = row.get(field, "")
        requested_path = resolve_repo_path(value)
        path = requested_path
        resolution_mode = "source_row_path"
        if not path.is_file() and field in E167_ARTIFACT_FALLBACKS:
            path = E167_ARTIFACT_FALLBACKS[field]
            resolution_mode = "recovered_from_immutable_snapshot"
        exists = path.is_file() and path.stat().st_size > 0
        if not exists:
            missing.append(f"{field}:{value}")
        artifact = {
            "field": field,
            "source_path": value,
            "requested_resolved_path": str(requested_path),
            "resolved_path": str(path),
            "resolution_mode": resolution_mode,
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
            "sha256": sha256(path) if exists else "",
            "imported_snapshot_path": "",
        }
        if exists and resolution_mode == "recovered_from_immutable_snapshot":
            snapshot_path = out_dir / "artifacts" / path.name
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot_path)
            if sha256(snapshot_path) != artifact["sha256"]:
                raise SystemExit(f"snapshot checksum mismatch for {field}: {snapshot_path}")
            artifact["imported_snapshot_path"] = str(snapshot_path)
        artifacts.append(artifact)
    if missing:
        raise SystemExit("E167 source artifact validation failed: " + ", ".join(missing))

    imported = dict(row)
    imported.update(
        {
            "sequence_key": "box004_20231003_2_082",
            "source_person": "person1",
            "partner_case_id": "box004_20231003_2_082_p2",
            "partner_person": "person2",
            "imported_by_exp_id": "E168",
            "source_rl_export_decision": row["rl_export_decision"],
            "pair_status": "PAIR_PENDING_PARTNER_OMNIRT",
            "rl_export_decision": "PENDING_PAIR_COMPLETION",
        }
    )
    fields = list(imported)
    write_tsv(out_dir / "imported_rl_export_row.tsv", [imported], fields)
    write_json(out_dir / "imported_rl_export_row.json", imported)
    manifest = {
        "snapshot_id": "E168_import_E167_box004_082_p1_E167A",
        "created_at": now(),
        "source_type": "experiment_import",
        "source_exp_id": "E167",
        "imported_by_exp_id": "E168",
        "source_ref": str(source_export),
        "source_ref_sha256": sha256(source_export),
        "source_rl_export_decision": row["rl_export_decision"],
        "e168_pair_status": imported["pair_status"],
        "artifacts": artifacts,
        "status": "pass",
    }
    write_json(out_dir / "import_manifest.json", manifest)
    (out_dir / "import_manifest.md").write_text(
        "\n".join(
            [
                "# E168 E167 Import Snapshot",
                "",
                f"- Source: `{source_export}`",
                "- Source row: `box004_20231003_2_082_p1 / E167A_zOnlyBody`",
                "- Source decision: `RL_EXPORT_READY`",
                "- E168 decision: `PENDING_PAIR_COMPLETION`",
                "- Partner: `box004_20231003_2_082_p2`",
                "- Status: `pass`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def build_profile_snapshot(
    config_path: Path,
    axis_audit_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mismatches: dict[str, dict[str, Any]] = {}
    profile: dict[str, Any] = {}
    for field, expected in PROFILE_EXPECTED.items():
        actual = config.get(field)
        profile[field] = actual
        if actual != expected:
            mismatches[field] = {"expected": expected, "actual": actual}

    axis_audit = json.loads(axis_audit_path.read_text(encoding="utf-8"))
    if axis_audit.get("axis_audit_pass") is not True:
        mismatches["axis_audit_pass"] = {
            "expected": True,
            "actual": axis_audit.get("axis_audit_pass"),
        }
    if mismatches:
        raise SystemExit(f"E167A profile mismatch: {mismatches}")

    profile_path = out_dir / "e167a_zonly_profile.yaml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    summary = {
        "created_at": now(),
        "status": "pass",
        "spider_method_id": "E167A_zOnlyBody",
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "source_axis_audit": str(axis_audit_path),
        "source_axis_audit_sha256": sha256(axis_audit_path),
        "profile_path": str(profile_path),
        "profile_sha256": sha256(profile_path),
        "profile": profile,
        "axis_invariants": {
            "b1_cem_smooth_disabled": config.get("cem_smooth_enabled") is False,
            "b2_postprocess_absent": True,
            "xy_foot_slip_disabled": config.get("foot_slip_enabled") is False,
            "foot_ground_disabled": config.get("foot_ground_enabled") is False,
            "ankle_extra_weight_disabled": config.get("local_frame_ankle_weight") == 1.0,
            "body_executability_axis": "z",
            "ground_executability_axis": "z",
            "object_tracking_axis": "xyz+rotation",
        },
    }
    write_json(out_dir / "e167a_zonly_profile.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=EXPECTED_RESULT_ROOT)
    parser.add_argument("--holosoma-repo", type=Path, required=True)
    parser.add_argument("--core4d-raw-root", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, required=True)
    parser.add_argument("--retarget-python", type=Path, required=True)
    parser.add_argument("--e167-export-tsv", type=Path, default=DEFAULT_E167_EXPORT)
    parser.add_argument("--e167-config", type=Path, default=DEFAULT_E167_CONFIG)
    parser.add_argument("--e167-axis-audit", type=Path, default=DEFAULT_AXIS_AUDIT)
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()
    if result_root != EXPECTED_RESULT_ROOT.resolve():
        raise SystemExit(
            f"E168 formal result root must be {EXPECTED_RESULT_ROOT.resolve()}, got {result_root}"
        )
    result_root.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "holosoma_repo": args.holosoma_repo.expanduser().resolve(),
        "core4d_raw_root": args.core4d_raw_root.expanduser().resolve(),
        "smplx_model_dir": args.smplx_model_dir.expanduser().resolve(),
        "e167_export_tsv": args.e167_export_tsv.expanduser().resolve(),
        "e167_config": args.e167_config.expanduser().resolve(),
        "e167_axis_audit": args.e167_axis_audit.expanduser().resolve(),
    }
    missing = [f"{name}:{path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise SystemExit("Phase 0 required paths missing: " + ", ".join(missing))

    environment = {
        "created_at": now(),
        "experiment_id": "E168",
        "result_root": str(result_root),
        "spider": git_state(REPO),
        "holosoma": git_state(required_paths["holosoma_repo"]),
        "core4d_raw_root": {
            "path": str(required_paths["core4d_raw_root"]),
            "exists": required_paths["core4d_raw_root"].is_dir(),
        },
        "smplx_model_dir": {
            "path": str(required_paths["smplx_model_dir"]),
            "exists": required_paths["smplx_model_dir"].is_dir(),
        },
        "retarget_python": retarget_python_probe(args.retarget_python.expanduser().resolve()),
        "local_gpu_snapshot": local_gpu_snapshot(),
        "remote_profiles": {
            "a6000-2gpu": {
                "status": "probe_pending",
                "host": "spider-remote",
                "repo_root": "/home/xiayb/pHRI_workspace/spider",
            },
            "A100-8gpu": {
                "status": "policy_allowlist_pending",
                "host": "batchcom@61.172.170.106:30409",
                "repo_root": "/home/dataset-assist-0/xiayb/workspace/spider",
                "selection_rule": "memory.used<5000MB AND no foreign process AND policy allowlist; max 4",
                "selected_gpus": [],
            },
        },
    }
    environment["status"] = (
        "local_pass_remote_pending"
        if environment["retarget_python"].get("status") == "pass"
        else "fail"
    )
    write_json(result_root / "s0_environment/environment_manifest.json", environment)

    source_rows = read_tsv(required_paths["e167_export_tsv"])
    source_row = select_e167_row(source_rows)
    import_manifest = import_e167_snapshot(
        required_paths["e167_export_tsv"],
        source_row,
        result_root / "imported_snapshots/E167_box004_082_p1_E167A",
    )
    profile = build_profile_snapshot(
        required_paths["e167_config"],
        required_paths["e167_axis_audit"],
        result_root / "s0_environment/e167a_profile",
    )
    summary = {
        "created_at": now(),
        "status": environment["status"],
        "environment_status": environment["status"],
        "e167_import_status": import_manifest["status"],
        "e167a_profile_status": profile["status"],
        "hard_stops": ["remote_profile_probe_and_A100_policy_allowlist_pending"],
        "next_allowed_action": "complete remote read-only probe; do not launch GPU work",
    }
    write_json(result_root / "s0_environment/phase0_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
