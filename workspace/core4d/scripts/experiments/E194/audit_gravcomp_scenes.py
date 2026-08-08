#!/usr/bin/env python3
"""Positive per-arm audit of the E194 gravity-compensation manifest.

For every one of the 45 Full rows this asserts, from first principles, that the
arm the row *claims* to be is exactly what run_mjwp will actually execute:

  Config side (Hydra compose of `+override=<override_id>` + the row's
  extra_overrides, exactly as the runner builds it):
    - scene_name resolves to the gravcomp sidecar for G1/G3, the base for G2
    - init_pos_actuator_gain == 2500 for G2/G3, == 500 for G1
    - init_rot_actuator_gain == 50 for ALL arms  (C3 guard: rotation untouched)
    - leg_object_penalty_scale == 2.0            (PRG still on)

  Model side (compiled MjModel):
    - G1/G3: body gravcomp[object] == 1.0 AND every other model array is
      byte-identical to the A0 base sidecar (geom/mass/inertia/pair/dof/...)
      -- i.e. the sidecar adds gravity comp and NOTHING else
    - G2:    gravcomp[object] == 0.0 (base sidecar used verbatim)

Exit non-zero (with --require-all) if any row fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_common as C  # noqa: E402

CONFIG_DIR = str((C.REPO / "examples/config").resolve())

# MjModel arrays that MUST be identical between the gravcomp sidecar and the A0
# base (everything about the physics except the one gravcomp entry).
MODEL_ARRAYS = [
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
    "geom_condim", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
    "geom_margin", "geom_gap",
    "jnt_type", "jnt_axis", "dof_damping", "dof_armature",
    "pair_geom1", "pair_geom2", "pair_solref", "pair_margin", "pair_gap", "pair_dim",
    "actuator_gainprm", "actuator_biasprm", "actuator_trnid",
]


def compose_config(override_id: str, extra: str) -> dict:
    overrides = [f"+override={override_id}"] + (extra.split() if extra else [])
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return OmegaConf.to_container(compose(config_name="default", overrides=overrides), resolve=True)


def scene_xml_for(scene_act_rel: str) -> Path:
    return C.repo_path(scene_act_rel)


def audit_row(row: dict[str, str]) -> list[str]:
    arm = row["arm"]
    spec = C.ARMS[arm]
    failures: list[str] = []

    # --- config side --------------------------------------------------------
    cfg = compose_config(row["override_id"], row["extra_overrides"])
    exp_scene = C.GRAVCOMP_SCENE_NAME if spec["gravcomp"] else row["scene_name"]
    if cfg.get("scene_name") != exp_scene:
        failures.append(f"scene_name={cfg.get('scene_name')}!={exp_scene}")
    if float(cfg.get("init_pos_actuator_gain", -1)) != float(spec["kp"]):
        failures.append(f"init_pos={cfg.get('init_pos_actuator_gain')}!={spec['kp']}")
    if float(cfg.get("init_rot_actuator_gain", -1)) != C.ROT_GAIN:
        failures.append(f"init_rot={cfg.get('init_rot_actuator_gain')}!={C.ROT_GAIN}")
    if float(cfg.get("leg_object_penalty_scale", -1)) != 2.0:
        failures.append(f"prg_off:leg_object_penalty_scale={cfg.get('leg_object_penalty_scale')}")

    # --- model side ---------------------------------------------------------
    scene = scene_xml_for(row["scene_act"])
    if not scene.is_file():
        failures.append(f"scene_missing:{row['scene_act']}")
        return failures
    model = mujoco.MjModel.from_xml_path(str(scene))
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_id < 0:
        failures.append("no_object_body")
        return failures
    gravcomp = float(model.body_gravcomp[obj_id])
    if spec["gravcomp"] and gravcomp != 1.0:
        failures.append(f"gravcomp[object]={gravcomp}!=1.0")
    if not spec["gravcomp"] and gravcomp != 0.0:
        failures.append(f"gravcomp[object]={gravcomp}!=0.0(base should be untouched)")

    # --- gravcomp sidecar must equal base + only gravcomp -------------------
    if spec["gravcomp"]:
        base_model = _base_model_for(row)
        if base_model is None:
            failures.append("base_model_unresolved")
        else:
            if base_model.nbody != model.nbody or base_model.ngeom != model.ngeom or base_model.npair != model.npair:
                failures.append("model_shape_diff_vs_base")
            else:
                # gravcomp differs only at object
                dg = np.asarray(model.body_gravcomp) - np.asarray(base_model.body_gravcomp)
                nz = [i for i in range(model.nbody) if dg[i] != 0.0]
                if nz != [obj_id]:
                    failures.append(f"gravcomp_delta_bodies={nz}!=[{obj_id}]")
                for name in MODEL_ARRAYS:
                    a, b = getattr(model, name, None), getattr(base_model, name, None)
                    if a is None or b is None:
                        continue
                    if not np.array_equal(np.asarray(a), np.asarray(b)):
                        failures.append(f"model_array_diff:{name}")
    return failures


_SOURCE_ROWS = None


def _base_model_for(row: dict[str, str]):
    """Compile the A0 base PRG sidecar for this case (from the source manifest)."""
    global _SOURCE_ROWS
    if _SOURCE_ROWS is None:
        _SOURCE_ROWS = C.load_source_rows()
    src = _SOURCE_ROWS.get(row["case_id"])
    if src is None:
        return None
    base = C.repo_path(src["scene_act"])
    if not base.is_file():
        return None
    return mujoco.MjModel.from_xml_path(str(base))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true", help="exit non-zero if any row fails")
    parser.add_argument("--manifest", default=str(C.RESULTS / "s6_downstream/manifests/cem_full_manifest.tsv"))
    args = parser.parse_args()

    rows = C.read_tsv(args.manifest)
    if len(rows) != 45:
        print(f"[audit] WARNING manifest has {len(rows)} rows (expected 45)", file=sys.stderr)

    per_arm_pass = {"G1": 0, "G2": 0, "G3": 0}
    n_fail = 0
    for row in rows:
        failures = audit_row(row)
        tag = "PASS" if not failures else "FAIL"
        if failures:
            n_fail += 1
            print(f"[{tag}] {row['variant']}: {';'.join(failures)}")
        else:
            per_arm_pass[row["arm"]] = per_arm_pass.get(row["arm"], 0) + 1
    print(f"[audit] per-arm PASS: {per_arm_pass}  total_fail={n_fail}/{len(rows)}")
    C.write_json(C.RESULTS / "preflight/audit_summary.json", {
        "created_at": C.now(), "manifest": C.rel(args.manifest),
        "rows": len(rows), "per_arm_pass": per_arm_pass, "n_fail": n_fail,
        "status": "pass" if n_fail == 0 else "fail",
    })
    if args.require_all and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
