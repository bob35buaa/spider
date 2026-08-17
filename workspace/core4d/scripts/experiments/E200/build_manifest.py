#!/usr/bin/env python3
"""Build the per-arm 8-GPU priority manifest for E200.

One manifest per arm (noprg / prg_g1a2), each with one row per feasible E199
full-scale trans variant. All CEM inputs (trajectory / contact_mask / target_scene)
are reused verbatim from E199; only the downstream arm (scene_act + override +
extra_overrides) changes. Output paths are arm-tagged so noprg / prg_g1a2 / the
existing E199 A0/PRG rollouts never collide on the shared filesystem.

Requires build_arm_scenes.py to have produced the per-arm scene registry first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e200_common as C  # noqa: E402

CEM_ROOT = "workspace/core4d/results/E200/s6_downstream/cem"
RENDER_ROOT = "workspace/core4d/results/E200/s6_downstream/render"
LOG_ROOT = "logs/E200/cem"


def override_fields(arm: str, aug_row: dict[str, str]) -> dict[str, str]:
    override_id = C.arm_override_id(arm, aug_row)
    if arm == "prg_g1a2":
        return {"override_id": override_id, "override_path": aug_row["override_path"],
                "override_sha256": aug_row.get("override_sha256", "")}
    # noprg: the E199 augmented base task yaml (chains E167A)
    path = C.REPO / "examples/config/override" / f"{override_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"noprg base override yaml missing: {path}")
    return {"override_id": override_id, "override_path": C.rel(path),
            "override_sha256": C.sha256(path)}


def build_rows(arm: str, aug_rows: list[dict[str, str]],
               registry: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, str]]:
    tag = C.ARM_TAG[arm]
    rows: list[dict[str, str]] = []
    for ordinal, aug in enumerate(aug_rows):
        key = (aug["case_id"], aug["aug_variant"])
        if key not in registry:
            raise KeyError(f"[{arm}] no scene registry entry for {key}; run build_arm_scenes.py first")
        reg = registry[key]
        stem = f"E200_{aug['case_id']}_aug_{aug['aug_variant']}_{tag}"
        outdir = f"{CEM_ROOT}/{arm}/{stem}_outdir"
        rows.append({
            "ordinal": ordinal, "tier": "P1", "experiment": "E200", "arm": arm,
            "object_key": aug["object_key"], "case_id": aug["case_id"],
            "aug_variant": aug["aug_variant"], "aug_translation": aug.get("aug_translation", ""),
            "aug_rotation_rad": aug.get("aug_rotation_rad", ""),
            "base_target_task": aug.get("base_target_task", ""), "target_task": aug["target_task"],
            "target_scene": aug["target_scene"],
            "trajectory": aug["trajectory"], "trajectory_sha256": aug["trajectory_sha256"],
            "contact_mask": aug["contact_mask"], "contact_mask_sha256": aug["contact_mask_sha256"],
            **override_fields(arm, aug),
            "base_scene_act": aug["base_scene_act"], "base_scene_sha256": aug.get("base_scene_sha256", ""),
            "scene_act": reg["scene_act"], "scene_name": reg["scene_name"],
            "effective_scene_sha256": reg["scene_sha256"],
            "extra_overrides": C.arm_extra_overrides(arm),
            "cem_samples": C.CEM_FULL_SAMPLES, "cem_opt_steps": C.CEM_FULL_OPT_STEPS, "cem_seed": C.CEM_SEED,
            "variant": "full",
            "result_npz": f"{CEM_ROOT}/{arm}/{stem}.npz",
            "outdir_npz": f"{outdir}/trajectory_mjwp_act.npz",
            "config_act": f"{outdir}/config_act.yaml",
            "video": f"{RENDER_ROOT}/{arm}/{stem}.mp4",
            "log": f"{LOG_ROOT}/{arm}/{stem}.log",
            "gpu_id": "", "status": "", "failure_mode": "",
            "execution_mode": "full", "updated_at": C.now(),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(C.ARMS))
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    aug_rows = C.load_aug_rows()
    for arm in arms:
        if arm not in C.ARMS:
            raise SystemExit(f"unknown arm {arm}; expected {C.ARMS}")
        reg_rows = C.read_tsv(C.scene_registry_path(arm))
        registry = {(r["case_id"], r["aug_variant"]): r for r in reg_rows}
        rows = build_rows(arm, aug_rows, registry)
        C.write_tsv(C.manifest_path(arm), rows, C.FIELDS)
        by_obj = {o: sum(1 for r in rows if r["object_key"] == o)
                  for o in sorted({r["object_key"] for r in rows})}
        print(f"[{arm}] manifest -> {C.rel(C.manifest_path(arm))}  ({len(rows)} rows) {by_obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
