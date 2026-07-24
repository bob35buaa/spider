#!/usr/bin/env python3
"""Strict preflight/runtime audit for E179 E167A without PRG."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e179_common as C  # noqa: E402


LOWER_BODY_GEOMS = {
    "left_hip_collision",
    "right_hip_collision",
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "left_linkage_brace_collision",
    "right_linkage_brace_collision",
    "lf0",
    "lf1",
    "lf2",
    "lf3",
    "rf0",
    "rf1",
    "rf2",
    "rf3",
}


def compose_config(override_id: str) -> dict[str, Any]:
    with initialize_config_dir(
        version_base=None,
        config_dir=str((C.REPO / "examples/config").resolve()),
    ):
        config = compose(
            config_name="default",
            overrides=[f"+override={override_id}"],
        )
    return dict(
        OmegaConf.to_container(config, resolve=True)  # type: ignore[arg-type]
    )


def scene_pair_leaks(path: Path) -> list[dict[str, str]]:
    root = ET.parse(path).getroot()
    leaks = []
    for pair in root.findall("./contact/pair"):
        geom1, geom2 = pair.get("geom1", ""), pair.get("geom2", "")
        if (
            geom1 == "object_collision"
            and geom2 in LOWER_BODY_GEOMS
        ) or (
            geom2 == "object_collision"
            and geom1 in LOWER_BODY_GEOMS
        ):
            leaks.append(dict(pair.attrib))
        elif pair.get("name", "").startswith(("E170_", "E173_")):
            leaks.append(dict(pair.attrib))
    return leaks


def output_diagnostic_leaks(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=True) as archive:
        return sorted(
            key
            for key in archive.files
            if key.startswith(C.PRG_DIAGNOSTIC_PREFIXES)
        )


def audit_row(
    row: dict[str, str],
    authority: dict[str, str],
    profile: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    override = C.repo_path(row["override_path"])
    scene = C.repo_path(row["scene_act"])
    trajectory = C.repo_path(row["trajectory"])
    contact_mask = C.repo_path(row["contact_mask"])
    required = {
        "override": override,
        "scene": scene,
        "trajectory": trajectory,
        "contact_mask": contact_mask,
    }
    for label, path in required.items():
        if not path.is_file():
            failures.append(f"missing:{label}")
    hash_checks = {}
    for field, path, expected in (
        ("override", override, row["override_sha256"]),
        ("scene", scene, row["effective_scene_sha256"]),
        ("trajectory", trajectory, row["trajectory_sha256"]),
        ("contact_mask", contact_mask, row["contact_mask_sha256"]),
    ):
        actual = C.sha256(path) if path.is_file() else ""
        hash_checks[field] = {
            "actual": actual,
            "expected": expected,
            "pass": actual == expected,
        }
        if actual != expected:
            failures.append(f"sha_mismatch:{field}")

    composed: dict[str, Any] = {}
    override_doc: dict[str, Any] = {}
    profile_mismatch = {}
    prg_composed = []
    prg_override = []
    if override.is_file():
        override_doc = yaml.safe_load(
            override.read_text(encoding="utf-8")
        )
        prg_override = sorted(
            set(override_doc) & set(C.PRG_CONFIG_FIELDS)
        )
        if prg_override:
            failures.append("prg_override_field_leak")
        defaults = override_doc.get("defaults", [])
        if not defaults or defaults[0] != authority["dcv3_override_id"]:
            failures.append("dcv3_override_parent_mismatch")
        composed = compose_config(row["override_id"])
        prg_composed = sorted(
            key for key in C.PRG_CONFIG_FIELDS if key in composed
        )
        if prg_composed:
            failures.append("prg_composed_field_leak")
        if composed.get("scene_name") != C.SCENE_NAME:
            failures.append("scene_name_mismatch")
        for key, expected in profile.items():
            actual = composed.get(
                key, False if key == "cem_smooth_enabled" else None
            )
            if actual != expected:
                profile_mismatch[key] = {
                    "actual": actual,
                    "expected": expected,
                }
        if profile_mismatch:
            failures.append("e167a_profile_mismatch")

    leaks = scene_pair_leaks(scene) if scene.is_file() else []
    if leaks:
        failures.append("prg_scene_pair_leak")
    hand_types = {}
    if scene.is_file():
        model = mujoco.MjModel.from_xml_path(str(scene))
        for hand in ("lh", "rh"):
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, hand
            )
            hand_types[hand] = (
                mujoco.mjtGeom(int(model.geom_type[geom_id])).name
                if geom_id >= 0
                else "missing"
            )
        if set(hand_types.values()) != {"mjGEOM_MESH"}:
            failures.append("rubber_hand_not_mesh")

    if any(C.boolish(row.get(flag)) for flag in ("p_enabled", "r_enabled", "g_enabled")):
        failures.append("manifest_prg_flag_enabled")
    if row.get("base_reward_method") != C.SPIDER_METHOD_ID:
        failures.append("base_reward_method_mismatch")
    if row.get("scene_name") != C.SCENE_NAME:
        failures.append("manifest_scene_name_mismatch")
    if (
        authority["trajectory_sha256"]
        != row["trajectory_sha256"]
        or authority["contact_mask_sha256"]
        != row["contact_mask_sha256"]
    ):
        failures.append("paired_input_hash_drift")

    diagnostic_leaks = output_diagnostic_leaks(
        C.repo_path(row["outdir_npz"])
    )
    if diagnostic_leaks:
        failures.append("prg_runtime_diagnostic_leak")
    return {
        "case_id": row["case_id"],
        "assigned_worker": row["assigned_worker"],
        "retarget_variant_id": row["retarget_variant_id"],
        "method_parity_pass": not profile_mismatch,
        "profile_fields_checked": len(profile),
        "profile_mismatch_json": json.dumps(
            profile_mismatch, sort_keys=True
        ),
        "prg_override_fields_json": json.dumps(prg_override),
        "prg_composed_fields_json": json.dumps(prg_composed),
        "prg_scene_pair_count": len(leaks),
        "prg_scene_pairs_json": json.dumps(leaks, sort_keys=True),
        "prg_runtime_diagnostic_fields_json": json.dumps(
            diagnostic_leaks
        ),
        "lh_geom_type": hand_types.get("lh", ""),
        "rh_geom_type": hand_types.get("rh", ""),
        "hash_checks_json": json.dumps(
            hash_checks, sort_keys=True
        ),
        "no_prg_pass": not (
            prg_override
            or prg_composed
            or leaks
            or diagnostic_leaks
        ),
        "audit_status": "pass" if not failures else "fail",
        "failures": ";".join(failures),
        "audited_at": C.now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=C.RESULTS
        / "s6_downstream/manifests/cem_full_manifest.tsv",
    )
    args = parser.parse_args()
    manifest = C.repo_path(args.manifest)
    rows = C.read_tsv(manifest)
    authority_rows = C.read_tsv(
        C.RESULTS / "input_authority/input_authority.tsv"
    )
    authority = {row["case_id"]: row for row in authority_rows}
    profile_doc = json.loads(C.E167A_PROFILE.read_text(encoding="utf-8"))
    if (
        profile_doc["profile_sha256"]
        != C.EXPECTED_E167A_PROFILE_SHA256
    ):
        raise ValueError("E167A profile SHA drift")
    profile = dict(profile_doc["profile"])
    profile["cem_smooth_enabled"] = False
    if len(rows) != C.EXPECTED_PAIRED_ROWS:
        raise ValueError(f"Full manifest must have 16 rows: {len(rows)}")
    if set(authority) != {row["case_id"] for row in rows}:
        raise ValueError("manifest/authority case set mismatch")
    C.validate_queue_contract(set(authority))

    audit_rows = [
        audit_row(row, authority[row["case_id"]], profile)
        for row in rows
    ]
    canary_rows = C.read_tsv(
        C.RESULTS / "s6_downstream/manifests/cem_canary_manifest.tsv"
    )
    canary_set = {row["case_id"] for row in canary_rows}
    failures = [
        row["case_id"]
        for row in audit_rows
        if row["audit_status"] != "pass"
    ]
    summary = {
        "created_at": C.now(),
        "status": "pass" if not failures else "fail",
        "manifest": C.rel(manifest),
        "manifest_sha256": C.sha256(manifest),
        "rows": len(rows),
        "unique_rows": len({row["case_id"] for row in rows}),
        "method_parity_pass": sum(
            C.boolish(row["method_parity_pass"])
            for row in audit_rows
        ),
        "no_prg_pass": sum(
            C.boolish(row["no_prg_pass"]) for row in audit_rows
        ),
        "audit_pass": sum(
            row["audit_status"] == "pass" for row in audit_rows
        ),
        "audit_failures": failures,
        "canary_rows": len(canary_rows),
        "canary_set_exact": canary_set == set(C.CANARY_CASES),
        "worker_distribution": dict(
            Counter(row["assigned_worker"] for row in rows)
        ),
        "e167a_profile_sha256": profile_doc["profile_sha256"],
        "prg_runtime_diagnostics_expected": False,
    }
    if not summary["canary_set_exact"]:
        summary["status"] = "fail"
        summary["audit_failures"].append("canary_set_mismatch")
    out_dir = C.RESULTS / "s5_handoff/config_scene_audit"
    C.write_tsv(out_dir / "e167a_no_prg_audit.tsv", audit_rows)
    C.write_json(out_dir / "e167a_no_prg_audit.json", audit_rows)
    C.write_json(
        out_dir / "e167a_no_prg_audit_summary.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_all and summary["status"] != "pass":
        return 2
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
