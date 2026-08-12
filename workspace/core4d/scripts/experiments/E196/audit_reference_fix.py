#!/usr/bin/env python3
"""Fail-closed preflight and landed-artifact auditor for E196."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e196_reference_fix_common as C  # noqa: E402

from spider.simulators.scene_act_reference import resolve_scene_act_reference  # noqa: E402


CONFIG_DIR = str((C.REPO / "examples/config").resolve())
RUNTIME_RE = re.compile(
    r"scene-act-reference: convention=([A-Z]{3}).*?meta_sha256=([0-9a-f]{64}).*?"
    r"xml_axis_sequence=([A-Z]{3}) parity=pass"
)
PROJECTION_DROP = {"output_dir", "video_output_path"}


def compose_config(row: dict[str, str]) -> dict[str, Any]:
    overrides = [f"+override={row['override_id']}"] + row["extra_overrides"].split()
    overrides.extend(
        [
            "+use_torch_compile=false",
            "save_video=false",
            "video_camera=auto",
            f"seed={int(row['cem_seed'])}",
            f"num_samples={int(row['cem_samples'])}",
            f"max_num_iterations={int(row['cem_opt_steps'])}",
        ]
    )
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return OmegaConf.to_container(
            compose(config_name="default", overrides=overrides), resolve=True
        )


def normalized_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalized_value(item)
            for key, item in sorted(value.items())
            if key not in PROJECTION_DROP
        }
    if isinstance(value, list):
        return [normalized_value(item) for item in value]
    if isinstance(value, str):
        for marker in ("example_datasets/", "workspace/", "logs/"):
            if marker in value:
                return marker + value.split(marker, 1)[1]
    return value


def projection_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        normalized_value(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def config_failures(row: dict[str, str], landed: bool) -> list[str]:
    failures: list[str] = []
    if landed:
        path = C.repo_path(row["config_act"])
        if not path.is_file():
            return ["missing:config_act"]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        e194_path = C.repo_path(row["e194_config_act"])
        if not e194_path.is_file():
            return ["missing:e194_config_act"]
        e194_config = yaml.safe_load(e194_path.read_text(encoding="utf-8"))
        if projection_sha(config) != projection_sha(e194_config):
            failures.append("scientific_config_projection")
    else:
        config = compose_config(row)
    checks = {
        "scene_name": (config.get("scene_name"), C.SCENE_NAME),
        "kp_pos": (float(config.get("init_pos_actuator_gain", -1)), C.KP_POS),
        "kp_rot": (float(config.get("init_rot_actuator_gain", -1)), C.KP_ROT),
        "samples": (int(config.get("num_samples", -1)), C.FULL_SAMPLES),
        "iterations": (int(config.get("max_num_iterations", -1)), C.FULL_OPT_STEPS),
        "seed": (int(config.get("seed", -1)), C.CEM_SEED),
        "prg": (float(config.get("leg_object_penalty_scale", 0) or 0), 2.0),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            failures.append(f"config:{name}:{actual}!={expected}")
    return failures


def input_failures(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for label, field, sha_field in (
        ("scene", "scene_act", "effective_scene_sha256"),
        ("meta", "scene_act_meta_path", "scene_act_meta_sha256"),
        ("trajectory", "trajectory", "trajectory_sha256"),
        ("contact", "contact_mask", "contact_mask_sha256"),
        ("override", "override_path", "override_sha256"),
        ("e194_config", "e194_config_act", "e194_config_sha256"),
    ):
        path = C.repo_path(row[field])
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing:{label}")
        elif C.sha256(path) != row[sha_field]:
            failures.append(f"sha:{label}")
    if row.get("reference_contract_version") != C.REFERENCE_CONTRACT_VERSION:
        failures.append("reference_contract_version")
    if failures:
        return failures
    scene = C.repo_path(row["scene_act"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    try:
        resolved = resolve_scene_act_reference(scene, model, emit_log=False)
    except (FileNotFoundError, ValueError) as exc:
        failures.append(f"reference_contract:{exc}")
        return failures
    if resolved.convention != row["resolved_euler_convention"]:
        failures.append("resolved_convention")
    if resolved.xml_axis_sequence != row["compiled_xml_axis_sequence"]:
        failures.append("compiled_axis_sequence")
    if resolved.meta_sha256 != row["scene_act_meta_sha256"]:
        failures.append("resolved_meta_sha")
    if float(row["axis_target_vs_raw_ori_err_deg_max"]) >= 1e-4:
        failures.append("axis_world_orientation_parity")
    if float(row["axis_target_vs_raw_pos_err_cm_max"]) >= 1e-9:
        failures.append("axis_world_position_parity")
    obj = resolved.object_body_id
    if float(model.body_gravcomp[obj]) != C.GRAVCOMP:
        failures.append("object_gravcomp")
    if C.compiled_physical_sha256(model) != row["compiled_physical_sha256"]:
        failures.append("compiled_physical_sha")
    snapshot_scene = C.SNAPSHOT_ROOT / row["case_id"] / scene.name
    if not snapshot_scene.is_file() or C.sha256(snapshot_scene) != row["effective_scene_sha256"]:
        failures.append("snapshot_scene")
    snapshot_meta = C.SNAPSHOT_ROOT / row["case_id"] / "scene_act_meta.json"
    if not snapshot_meta.is_file() or C.sha256(snapshot_meta) != row["scene_act_meta_sha256"]:
        failures.append("snapshot_meta")
    failures.extend(config_failures(row, landed=False))
    return failures


def landed_failures(row: dict[str, str], *, require_video: bool) -> list[str]:
    failures: list[str] = []
    for label, field in (
        ("result", "result_npz"),
        ("outdir", "outdir_npz"),
        ("config", "config_act"),
        ("log", "log"),
    ):
        path = C.repo_path(row[field])
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing:{label}")
    if require_video:
        video = C.repo_path(row["video"])
        if not video.is_file() or video.stat().st_size == 0:
            failures.append("missing:video")
    if failures:
        return failures
    for field in ("result_npz", "outdir_npz"):
        with np.load(C.repo_path(row[field]), allow_pickle=True) as archive:
            if "qpos" not in archive.files:
                failures.append(f"npz_missing_qpos:{field}")
            elif not np.isfinite(np.asarray(archive["qpos"], dtype=np.float64)).all():
                failures.append(f"npz_nonfinite_qpos:{field}")
    log_text = C.repo_path(row["log"]).read_text(errors="replace")
    matches = RUNTIME_RE.findall(log_text)
    unique = sorted(set(matches))
    expected = (
        row["resolved_euler_convention"],
        row["scene_act_meta_sha256"],
        row["compiled_xml_axis_sequence"],
    )
    if unique != [expected]:
        failures.append(f"runtime_reference_log:{unique!r}!={[expected]!r}")
    if "fallback" in log_text.lower():
        failures.append("runtime_log_contains_fallback")
    failures.extend(config_failures(row, landed=True))
    return failures


def audit(
    manifest: Path,
    *,
    scope: str,
    allow_subset: bool,
    require_video: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = C.read_tsv(manifest)
    details: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    for row in rows:
        failures = input_failures(row)
        for field in ("result_npz", "outdir_npz", "config_act", "video", "log"):
            path = row[field]
            if path in seen_outputs:
                failures.append(f"output_collision:{field}")
            seen_outputs.add(path)
            if "E194" in path or "/E194/" in path:
                failures.append(f"e194_output_collision:{field}")
        if scope in {"wave0", "full"}:
            failures.extend(landed_failures(row, require_video=require_video))
        details.append(
            {
                "case_id": row["case_id"],
                "object_key": row["object_key"],
                "worker": row["worker"],
                "wave": row["wave"],
                "status": "pass" if not failures else "fail",
                "failures": ";".join(failures),
            }
        )
    cardinal_failures: list[str] = []
    case_ids = {row["case_id"] for row in rows}
    if not allow_subset:
        expected = 3 if scope == "wave0" else C.N_CASES
        if len(rows) != expected or len(case_ids) != expected:
            cardinal_failures.append(f"cardinality:{len(rows)}!={expected}")
        if scope != "wave0":
            try:
                C.validate_frozen_queues(case_ids)
            except ValueError as exc:
                cardinal_failures.append(str(exc))
        elif case_ids != {case for cases in C.WAVE0.values() for case in cases}:
            cardinal_failures.append("wave0_set")
    summary = {
        "created_at": C.now(),
        "manifest": C.rel(manifest),
        "scope": scope,
        "rows": len(rows),
        "objects": dict(Counter(row["object_key"] for row in rows)),
        "workers": dict(Counter(row["worker"] for row in rows)),
        "row_failures": sum(row["status"] == "fail" for row in details),
        "cardinal_failures": cardinal_failures,
        "case_set_sha256": C.case_set_sha256(case_ids),
        "status": "pass"
        if not cardinal_failures and all(row["status"] == "pass" for row in details)
        else "fail",
    }
    return details, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scope", choices=("prelaunch", "wave0", "full"), default="prelaunch")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    default = C.WAVE0_MANIFEST if args.scope == "wave0" else C.FULL_MANIFEST
    manifest = C.repo_path(args.manifest or default)
    details, summary = audit(
        manifest,
        scope=args.scope,
        allow_subset=args.allow_subset,
        require_video=args.require_video,
    )
    suffix = Path(manifest).stem
    output = C.PREFLIGHT_ROOT / f"{args.scope}_{suffix}"
    C.write_tsv(output.with_suffix(".tsv"), details)
    C.write_json(output.with_suffix(".json"), summary)
    for row in details:
        if row["status"] == "fail":
            print(f"[FAIL] {row['case_id']}: {row['failures']}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if args.require_all and summary["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
