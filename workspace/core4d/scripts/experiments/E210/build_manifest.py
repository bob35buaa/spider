#!/usr/bin/env python3
"""E210 P4: build the priority-queue manifest for the 15 aug x G1only rollouts.

Schema is E199's (the queue driver reads it verbatim). Every value that is NOT
the gravcomp axis is copied from the corresponding E202 row and then re-verified
against the file on disk, so a silent upstream edit shows up here rather than
after 15 CEM runs.

Exit checks enforced in-process:
  * ``trajectory_sha256`` / ``contact_mask_sha256`` equal E202's AND equal the
    current file hash -> the reference and mask have not drifted;
  * ``effective_scene_sha256`` equals the P2 sidecar's real hash AND differs
    from E202's base scene -> we are actually running the gravcomp variant.
    (plan237 P3 originally demanded scene-sha *equality* with the baseline,
    which was a typo -- the whole point is that it differs. E207 F2 corrected
    it; this is the corrected form.)

Usage:
    .venv/bin/python .../E210/build_manifest.py
    ... --stage smoke --limit 1 --samples 64 --opt-steps 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e210_common as C  # noqa: E402

EXTRA_FIELDS = ("object_geom_count", "compiled_robot_object_pair_count")


def build_row(row: dict[str, str], *, stage: str, samples: int, opt_steps: int) -> dict[str, Any]:
    case_id, av = row["case_id"], row["aug_variant"]
    override = C.override_path(case_id, av)
    base_scene = C.base_scene_path(row)
    scene = C.scene_path(row)
    for label, path in (("override", override), ("E202 base scene", base_scene),
                        ("gravcomp sidecar", scene)):
        if not path.is_file():
            raise FileNotFoundError(f"{case_id}/{av}: missing {label}: {C.rel(path)}")

    out: dict[str, Any] = {
        "tier": "P1",
        "experiment": C.EXP,
        "arm": C.ARM,
        "object_key": row["object_key"],
        "case_id": case_id,
        "aug_variant": av,
        "aug_translation": row.get("aug_translation", ""),
        "aug_rotation_rad": row.get("aug_rotation_rad", ""),
        "base_target_task": row["base_target_task"],
        "target_task": row["target_task"],
        "target_scene": row["target_scene"],
        "trajectory": row["trajectory"],
        "trajectory_sha256": row["trajectory_sha256"],
        "contact_mask": row["contact_mask"],
        "contact_mask_sha256": row["contact_mask_sha256"],
        "override_id": override.stem,
        "override_path": C.rel(override),
        "override_sha256": C.sha256(override),
        "base_scene_act": C.rel(base_scene),
        "base_scene_sha256": C.sha256(base_scene),
        "scene_act": C.rel(scene),
        "scene_name": C.SCENE_NAME,
        "effective_scene_sha256": C.sha256(scene),
        "object_geom_count": row.get("object_geom_count", ""),
        "compiled_robot_object_pair_count": row.get("compiled_robot_object_pair_count", ""),
        "extra_overrides": "",
        "cem_samples": samples,
        "cem_opt_steps": opt_steps,
        "cem_seed": C.CEM_SEED,
        "execution_mode": "production" if stage == "full" else stage,
        "status": "",
        "failure_mode": "",
        "gpu_id": "",
        "updated_at": C.now(),
        **C.artifact_paths(case_id, av, stage=stage),
    }
    return out


def verify(rows: list[dict[str, Any]], src: list[dict[str, str]]) -> list[str]:
    """C0/C1 exit checks. Returns a list of human-readable failures."""
    fails: list[str] = []
    for out, ref in zip(rows, src, strict=True):
        tag = f"{out['case_id']}/{out['aug_variant']}"

        # C0: reference + mask byte-identical to E202, and still on disk unchanged.
        for field, sha_field in (("trajectory", "trajectory_sha256"),
                                 ("contact_mask", "contact_mask_sha256")):
            if out[sha_field] != ref[sha_field]:
                fails.append(f"{tag}: {sha_field} != E202")
            path = C.repo_path(out[field])
            if not path.is_file():
                fails.append(f"{tag}: missing {field} {out[field]}")
            elif C.sha256(path) != out[sha_field]:
                fails.append(f"{tag}: {field} on disk != recorded {sha_field}")

        # C1: the scene really is the gravcomp one, and really differs from E202's.
        if out["effective_scene_sha256"] == out["base_scene_sha256"]:
            fails.append(f"{tag}: gravcomp sidecar sha == E202 base sha (no diff applied)")
        if out["base_scene_sha256"] != ref["effective_scene_sha256"]:
            fails.append(f"{tag}: E202 base scene changed since its manifest was written")
        C.assert_gravcomp_diff(C.repo_path(out["base_scene_act"]), C.repo_path(out["scene_act"]))

        # Budget frozen at the E178/E202/E207 contract for the full stage.
        if out["execution_mode"] == "production":
            if (out["cem_samples"], out["cem_opt_steps"], out["cem_seed"]) != (
                C.CEM_FULL_SAMPLES, C.CEM_FULL_OPT_STEPS, C.CEM_SEED
            ):
                fails.append(f"{tag}: CEM budget drift {out['cem_samples']}x{out['cem_opt_steps']}")
            if (int(ref["cem_samples"]), int(ref["cem_opt_steps"]), int(ref["cem_seed"])) != (
                C.CEM_FULL_SAMPLES, C.CEM_FULL_OPT_STEPS, C.CEM_SEED
            ):
                fails.append(f"{tag}: E202 ran a different budget than E210 is about to")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("full", "smoke"), default="full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--opt-steps", type=int, default=0)
    args = ap.parse_args()

    stage = args.stage
    samples = args.samples or C.CEM_FULL_SAMPLES
    opt_steps = args.opt_steps or C.CEM_FULL_OPT_STEPS

    src = C.load_variants()
    if args.limit:
        src = src[: args.limit]
    rows = [build_row(r, stage=stage, samples=samples, opt_steps=opt_steps) for r in src]
    rows.sort(key=lambda r: (r["case_id"], r["aug_variant"]))
    for ordinal, row in enumerate(rows, 1):
        row["ordinal"] = ordinal

    if fails := verify(rows, sorted(src, key=lambda r: (r["case_id"], r["aug_variant"]))):
        for f in fails:
            print(f"  FAIL {f}")
        raise SystemExit(f"manifest verification FAILED ({len(fails)} problems)")

    fields = list(C.FIELDS) + [f for f in EXTRA_FIELDS if f not in C.FIELDS]
    out_path = C.FULL_MANIFEST if stage == "full" else C.SMOKE_MANIFEST
    out_path.parent.mkdir(parents=True, exist_ok=True)
    C.write_tsv(out_path, rows, fields)

    summary = {
        "created_at": C.now(),
        "stage": stage,
        "rows": len(rows),
        "cases": sorted({r["case_id"] for r in rows}),
        "excluded_cases": C.EXCLUDED_CASES,
        "by_variant": {v: sum(r["aug_variant"] == v for r in rows) for v in C.TRANS_VARIANTS},
        "scene_name": C.SCENE_NAME,
        "base_scene_name": C.BASE_SCENE_NAME,
        "frozen_cem": {"seed": C.CEM_SEED, "samples": samples, "opt_steps": opt_steps},
        "manifest": C.rel(out_path),
        "verified": [
            "trajectory/contact_mask sha == E202 and == on-disk",
            "effective_scene_sha != base_scene_sha (gravcomp applied)",
            "base_scene_sha == E202 manifest's effective_scene_sha (no upstream drift)",
            "assert_gravcomp_diff per row",
        ],
    }
    C.write_json(out_path.parent / f"e210_manifest_summary_{stage}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
