#!/usr/bin/env python3
"""E206 P6: build the manifest that feeds the G8 contact-fidelity gate.

G8 asks: do the ref-FK contact targets actually land on the collision proxy?
`eval_E176_contact_fidelity.py` answers it, and its `load_reference_qpos` reads
the rollout npz ONLY when the trajectory file is missing -- so with a real
trajectory on disk this runs **before** any CEM, which is the whole point
(finding out the proxy misses the contact targets after 108 GPU runs would be
too late).

Columns match what that runner reads:
  case_id, object_key, scene_act, trajectory, contact_mask, object_geom_count,
  compiled_robot_object_pair_count, source_e206_config_act

`source_e206_config_act` uses the generalised `source_<exp>_config_act` lookup
added alongside `--max-object-geoms` (U7/R10) -- the runner no longer hardcodes
E174 column names.

The noPRG scene is used: both arms carry the identical 2N hand<->object pairs
and the identical proxy, and G8 only looks at hand contact targets vs proxy
surface, so the leg pairs are irrelevant here. Using the smaller scene keeps the
compile cheap.

Usage:
    .venv/bin/python .../build_proxy_manifest.py
    ... --arm prg --limit N
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e206_common as C  # noqa: E402

V1_MANIFEST = C.S3_DIR / "omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv"
FIELDS = [
    "case_id", "object_key", "scene_act", "trajectory", "contact_mask",
    "object_geom_count", "compiled_robot_object_pair_count",
    "source_e206_config_act", "arm", "scene_name",
]


def dump_resolved_config(case_id: str, arm: str, out_dir: Path) -> Path:
    """Hydra-compose the arm override and dump it as a resolved `config_act.yaml`.

    The G8 runner reads `contact_hdmi_dynamic_target` / `contact_hdmi_target_source`
    off this file. Those live in the BASE override, so pointing the manifest at the
    raw arm YAML makes every case fail with "expected contact_hdmi_dynamic_target=
    true" -- E176's column pointed at the resolved config run_mjwp dumps at runtime.
    Pre-CEM there is no runtime dump, so compose it here (no GPU, no scene load).
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{case_id}_{arm}_config_act.yaml"
    with initialize_config_dir(version_base=None,
                               config_dir=str((C.REPO / "examples/config").resolve())):
        cfg = compose(config_name="default",
                      overrides=[f"+override={C.override_name(case_id, arm)}"])
    out.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return out


def compiled_counts(scene: Path) -> tuple[int, int]:
    """(object collision geoms, robot<->object pairs) as MuJoCo actually compiles them."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    names = {}
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("object_collision"):
            names[gid] = name
    object_names = set(names.values())
    robot = set(C.ROBOT_OBJECT_GEOMS)
    pairs = 0
    for p in range(model.npair):
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.pair_geom1[p]) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.pair_geom2[p]) or ""
        if (n1 in robot and n2 in object_names) or (n2 in robot and n1 in object_names):
            pairs += 1
    return len(names), pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="noprg", choices=list(C.ARMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=C.S6_DIR / "manifests")
    args = ap.parse_args()

    rows_in = [r for r in C.read_tsv(V1_MANIFEST) if r.get("stage2b_status") == "pass"]
    if args.limit:
        rows_in = rows_in[: args.limit]
    if not rows_in:
        raise SystemExit("no Stage2b-passing cases; run S3 first")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in rows_in:
        case_id = row["case_id"]
        tdir = C.PROCESSED_ROOT / f"dcv3_omnirt_v1_ref_fk_{case_id}"
        scene = tdir / f"{C.SCENE_BY_ARM[args.arm]}.xml"
        trajectory = tdir / "0" / "trajectory_kinematic.npz"
        override = C.OVERRIDE_DIR / f"{C.override_name(case_id, args.arm)}.yaml"
        mask = row.get("contact_mask_npz") or row.get("contact_mask") or ""
        missing = [
            label for label, path in
            (("scene_act", scene), ("trajectory", trajectory), ("override", override))
            if not path.is_file()
        ]
        if missing or not mask:
            skipped.append({"case_id": case_id,
                            "missing": ",".join(missing + ([] if mask else ["contact_mask"]))})
            continue
        n_geoms, n_pairs = compiled_counts(scene)
        expected = C.expected_pair_counts(n_geoms)[args.arm]
        if n_pairs != expected:
            skipped.append({"case_id": case_id,
                            "missing": f"pair_count {n_pairs}!={expected}"})
            continue
        resolved = dump_resolved_config(case_id, args.arm,
                                        args.out_dir / "resolved_configs")
        rows.append({
            "case_id": case_id,
            "object_key": row["object_key"],
            "scene_act": str(scene.relative_to(C.REPO)),
            "trajectory": str(trajectory.relative_to(C.REPO)),
            "contact_mask": mask,
            "object_geom_count": n_geoms,
            "compiled_robot_object_pair_count": n_pairs,
            # generalised source_<exp>_config_act column (see U7/R10); must be the
            # RESOLVED config, not the arm override -- see dump_resolved_config.
            "source_e206_config_act": str(resolved.relative_to(C.REPO)),
            "arm": args.arm,
            "scene_name": C.SCENE_BY_ARM[args.arm],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"lowgeom_full_manifest_{args.arm}.tsv"
    C.write_tsv(out, rows, FIELDS)
    objects = sorted({r["object_key"] for r in rows})
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "arm": args.arm,
        "manifest": str(out.relative_to(C.REPO)),
        "rows": len(rows),
        "objects": objects,
        "n_objects": len(objects),
        "max_object_geom_count": max((r["object_geom_count"] for r in rows), default=0),
        "skipped": skipped,
    }
    (args.out_dir / f"lowgeom_full_manifest_{args.arm}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nG8 next:\n  .venv/bin/python workspace/core4d/scripts/eval/runners/"
          f"eval_E176_contact_fidelity.py \\\n"
          f"    --manifest {out.relative_to(C.REPO)} \\\n"
          f"    --out-dir workspace/core4d/results/E206/s2_proxy/contact_fidelity \\\n"
          f"    --experiment-id E206 --expected-cases {len(rows)} "
          f"--expected-objects {len(objects)} \\\n"
          f"    --max-object-geoms {payload['max_object_geom_count']}")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
