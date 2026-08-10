#!/usr/bin/env python3
"""Positive audit that E192 A2 changes only two effective gate fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e192_common as C  # noqa: E402


CONFIG_DIR = str((C.REPO / "examples/config").resolve())
ALLOWED_DIFFS = {
    "cem_hand_gate_max_violation_pct",
    "cem_hand_gate_hard_floor_m",
}
FROZEN_CONFIG = {
    "cem_hand_gate_min_sdf_m",
    "cem_leg_gate_enabled",
    "cem_leg_gate_min_sdf_m",
    "cem_leg_gate_max_violation_pct",
    "cem_leg_gate_hard_floor_m",
    "cem_posture_gate_enabled",
    "cem_safety_gate_enabled",
    "surface_band_rew_scale",
    "surface_band_penalty_scale",
    "object_lift_sigma",
    "leg_object_penalty_scale",
    "init_pos_actuator_gain",
    "init_rot_actuator_gain",
    "contact_hdmi_target_source",
    "scene_name",
}


def resolved(override_id: str, extra: str = "") -> dict:
    overrides = [f"+override={override_id}"] + (extra.split() if extra else [])
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="default", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def audit_row(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    base = resolved(row["override_id"])
    arm = resolved(row["override_id"], row["extra_overrides"])
    changed = {
        key for key in set(base) | set(arm)
        if base.get(key) != arm.get(key)
    }
    expected_changed = set() if row["arm"] == "A0" else ALLOWED_DIFFS
    if changed != expected_changed:
        failures.append(f"resolved_diff={sorted(changed)} expected={sorted(expected_changed)}")
    gate = C.A0_GATE if row["arm"] == "A0" else C.A2_GATE
    for key, expected in gate.items():
        if not np.isclose(float(arm.get(key, np.nan)), expected):
            failures.append(f"{key}={arm.get(key)}!={expected}")
    for key in FROZEN_CONFIG:
        if key not in ALLOWED_DIFFS and base.get(key) != arm.get(key):
            failures.append(f"frozen_config_changed:{key}")
    if float(arm.get("init_pos_actuator_gain", -1)) != C.KP_POS:
        failures.append("kp_pos_not_500")
    if float(arm.get("init_rot_actuator_gain", -1)) != C.KP_ROT:
        failures.append("kp_rot_not_50")
    if float(arm.get("leg_object_penalty_scale", 0) or 0) != 2.0:
        failures.append("prg_disabled")
    scene = C.repo_path(row["scene_act"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0 or float(model.body_gravcomp[object_id]) != 0.0:
        failures.append("object_gravcomp_not_zero")
    for key, sha_key in (
        ("scene_act", "scene_sha256"),
        ("scene_snapshot_path", "scene_snapshot_sha256"),
        ("trajectory", "trajectory_sha256"),
        ("contact_mask", "contact_mask_sha256"),
        ("override_path", "override_sha256"),
    ):
        path = C.repo_path(row[key]) if row[key] else Path()
        if not row[key] or not path.is_file():
            failures.append(f"missing_physical:{key}")
        elif C.sha256(path) != row[sha_key]:
            failures.append(f"sha_mismatch:{key}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    root = C.RESULTS / "s6_downstream/manifests"
    manifests = [
        root / "cem_baseline_sentinel_manifest.tsv",
        root / "cem_canary_manifest.tsv",
        root / "cem_full_manifest.tsv",
    ]
    rows = [row for path in manifests for row in C.read_tsv(path)]
    failures = []
    for row in rows:
        current = audit_row(row)
        if current:
            failures.append({"variant": row["variant"], "failures": current})
    full = C.read_tsv(manifests[-1])
    if len(full) != 15 or len({row["case_id"] for row in full}) != 15:
        failures.append({"variant": "full_contract", "failures": ["not_15_unique"]})
    if len({(row["worker_id"], row["case_id"]) for row in full}) != 15:
        failures.append({"variant": "worker_contract", "failures": ["duplicate_claim"]})
    summary = {
        "created_at": C.now(),
        "rows": len(rows),
        "failed_rows": len(failures),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    C.write_json(C.RESULTS / "preflight/gate_override_audit.json", summary)
    print(json.dumps({k: summary[k] for k in ("rows", "failed_rows", "status")}, indent=2))
    return 1 if args.require_all and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
