#!/usr/bin/env python3
"""Verify E195 differs from its paired E192 A2 row only at the planned gate values."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_common as C  # noqa: E402


CONFIG_DIR = str((C.REPO / "examples/config").resolve())
ALLOWED_DIFFS_FROM_E192 = {
    "cem_hand_gate_min_sdf_m",
    "cem_hand_gate_hard_floor_m",
}


def resolved(override_id: str, extra: str) -> dict:
    overrides = [f"+override={override_id}"] + extra.split()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="default", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def audit_row(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    baseline = resolved(row["override_id"], row["source_extra_overrides"])
    current = resolved(row["override_id"], row["extra_overrides"])
    changed = {
        key for key in set(baseline) | set(current)
        if baseline.get(key) != current.get(key)
    }
    if changed != ALLOWED_DIFFS_FROM_E192:
        failures.append(
            f"resolved_diff={sorted(changed)} expected={sorted(ALLOWED_DIFFS_FROM_E192)}"
        )
    for key, expected in C.E192_GATE.items():
        if not np.isclose(float(baseline.get(key, np.nan)), expected):
            failures.append(f"E192:{key}={baseline.get(key)}!={expected}")
    for key, expected in C.E195_GATE.items():
        if not np.isclose(float(current.get(key, np.nan)), expected):
            failures.append(f"E195:{key}={current.get(key)}!={expected}")
    if float(current.get("init_pos_actuator_gain", -1)) != C.KP_POS:
        failures.append("kp_pos_not_500")
    if float(current.get("init_rot_actuator_gain", -1)) != C.KP_ROT:
        failures.append("kp_rot_not_50")
    if float(current.get("leg_object_penalty_scale", 0) or 0) != 2.0:
        failures.append("prg_disabled")
    if int(row["cem_samples"]) != C.FULL_SAMPLES:
        failures.append("samples_not_1024")
    if int(row["cem_opt_steps"]) != C.FULL_OPT_STEPS:
        failures.append("iterations_not_32")
    if int(row["cem_seed"]) != C.CEM_SEED:
        failures.append("seed_not_zero")
    for key in ("target_scene", "trajectory", "contact_mask", "override_path", "scene_act"):
        if not C.repo_path(row[key]).is_file():
            failures.append(f"missing_input:{key}")
    scene = C.repo_path(row["scene_act"])
    if scene.is_file():
        model = mujoco.MjModel.from_xml_path(str(scene))
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        if object_id < 0 or float(model.body_gravcomp[object_id]) != 0.0:
            failures.append("object_gravcomp_not_zero")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    manifest = C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"
    rows = C.read_tsv(manifest)
    failures = []
    for row in rows:
        current = audit_row(row)
        if current:
            failures.append({"variant": row["variant"], "failures": current})
    counts = Counter(row["worker_id"] for row in rows)
    expected_counts = {"local-gpu0": 7, "ada-gpu0": 4, "ada-gpu1": 4}
    if len(rows) != 15 or len({row["case_id"] for row in rows}) != 15:
        failures.append({"variant": "manifest", "failures": ["not_15_unique"]})
    if set(row["case_id"] for row in rows) != set(C.all_case_ids()):
        failures.append({"variant": "case_set", "failures": ["case_set_drift"]})
    if dict(counts) != expected_counts:
        failures.append({"variant": "workers", "failures": [f"counts={dict(counts)}"]})
    for row in rows:
        expected = C.worker_for(row["case_id"])
        if any(row[key] != expected[key] for key in ("worker_id", "host", "gpu_id")):
            failures.append({"variant": row["variant"], "failures": ["worker_mapping_drift"]})
    summary = {
        "created_at": C.now(),
        "rows": len(rows),
        "worker_counts": dict(counts),
        "failed_rows": len(failures),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    C.write_json(C.RESULTS / "preflight/config_audit.json", summary)
    print(json.dumps({key: summary[key] for key in ("rows", "worker_counts", "failed_rows", "status")}, indent=2))
    return 1 if args.require_all and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

