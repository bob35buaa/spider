#!/usr/bin/env python3
"""Positive 72-row preflight audit for E194 G1 expansion."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_g1_expansion_common as C  # noqa: E402

CONFIG_DIR = str((C.REPO / "examples/config").resolve())
MODEL_ARRAYS = (
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_condim",
    "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
    "jnt_type", "jnt_axis", "dof_damping", "dof_armature", "pair_geom1", "pair_geom2",
    "pair_solref", "pair_margin", "pair_gap", "pair_dim", "actuator_gainprm", "actuator_biasprm", "actuator_trnid",
)


def compose_config(row: dict[str, str]) -> dict:
    overrides = [f"+override={row['override_id']}"] + row["extra_overrides"].split()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return OmegaConf.to_container(compose(config_name="default", overrides=overrides), resolve=True)


def audit_row(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for label, field, sha_field in (
        ("override", "override_path", "override_sha256"), ("trajectory", "trajectory", "trajectory_sha256"),
        ("contact", "contact_mask", "contact_mask_sha256"), ("base_scene", "base_scene_act", "source_effective_scene_sha256"),
        ("g1_scene", "scene_act", "effective_scene_sha256"),
    ):
        path = C.repo_path(row[field])
        if not path.is_file(): failures.append(f"missing:{label}")
        elif C.sha256(path) != row[sha_field]: failures.append(f"sha:{label}")
    if not C.repo_path(row["target_scene"]).is_file(): failures.append("missing:target_scene")
    if failures: return failures
    cfg = compose_config(row)
    if cfg.get("scene_name") != C.SCENE_NAME: failures.append(f"scene_name:{cfg.get('scene_name')}")
    if float(cfg.get("init_pos_actuator_gain", -1)) != C.KP_POS: failures.append(f"kp_pos:{cfg.get('init_pos_actuator_gain')}")
    if float(cfg.get("init_rot_actuator_gain", -1)) != C.KP_ROT: failures.append(f"kp_rot:{cfg.get('init_rot_actuator_gain')}")
    if float(cfg.get("leg_object_penalty_scale", 0) or 0) != 2.0: failures.append("prg_off")
    base = mujoco.MjModel.from_xml_path(str(C.repo_path(row["base_scene_act"])))
    g1 = mujoco.MjModel.from_xml_path(str(C.repo_path(row["scene_act"])))
    obj = mujoco.mj_name2id(g1, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj < 0: return failures + ["missing:object_body"]
    if float(g1.body_gravcomp[obj]) != C.GRAVCOMP: failures.append(f"object_gravcomp:{g1.body_gravcomp[obj]}")
    delta = np.asarray(g1.body_gravcomp) - np.asarray(base.body_gravcomp)
    if np.flatnonzero(delta).tolist() != [obj] or float(delta[obj]) != 1.0: failures.append("gravcomp_delta")
    for name in MODEL_ARRAYS:
        if not np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(g1, name))):
            failures.append(f"model_diff:{name}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, default=C.FULL_MANIFEST)
    parser.add_argument("--require-all", action="store_true"); args = parser.parse_args()
    rows = C.read_tsv(args.manifest); details = []; failures = 0
    for row in rows:
        errors = audit_row(row); failures += bool(errors)
        details.append({"case_id": row["case_id"], "object_key": row["object_key"], "worker": row["worker"],
                        "status": "pass" if not errors else "fail", "failures": ";".join(errors)})
        if errors: print(f"[FAIL] {row['case_id']}: {';'.join(errors)}")
    counts = Counter(row["object_key"] for row in rows); workers = Counter(row["worker"] for row in rows)
    if len(rows) != C.N_CASES or dict(counts) != C.OBJECT_COUNTS: failures += 1
    try: C.validate_worker_balance(rows)
    except ValueError as exc: print(f"[FAIL] {exc}"); failures += 1
    summary = {"created_at": C.now(), "manifest": C.rel(args.manifest), "rows": len(rows), "objects": dict(counts),
               "workers": dict(workers), "row_failures": sum(row["status"] == "fail" for row in details),
               "status": "pass" if failures == 0 else "fail"}
    out = C.RESULTS / "s6_downstream/evidence/g1_expansion/preflight"
    C.write_tsv(out / "audit_rows.tsv", details); C.write_json(out / "audit_summary.json", summary)
    print(summary); return 1 if args.require_all and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
