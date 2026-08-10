#!/usr/bin/env python3
"""Build the frozen 15-row E195 Full manifest from completed E192 A2 rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e195_common as C  # noqa: E402


FIELDS = [
    "ordinal", "case_id", "object_key", "arm", "stage", "source_exp",
    "retarget_variant_id", "target_variant_id", "person",
    "hand_collision_variant_id", "spider_method_id", "target_task",
    "target_scene", "trajectory", "contact_mask", "override_id",
    "override_path", "scene_act", "scene_name", "source_extra_overrides",
    "extra_overrides", "cem_hand_gate_min_sdf_m",
    "cem_hand_gate_max_violation_pct", "cem_hand_gate_hard_floor_m",
    "kp_pos", "kp_rot", "gravcomp", "worker_id", "host", "gpu_id",
    "remote_staging_path", "cem_samples", "cem_opt_steps", "cem_seed",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "execution_mode", "updated_at",
]


def artifact_paths(case_id: str) -> dict[str, str]:
    variant = f"E195_{case_id}_A3"
    root = "workspace/core4d/results/E195/s6_downstream/cem/full"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir_full/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir_full/config_act.yaml",
        "video": f"workspace/core4d/results/E195/s6_downstream/render/full/{variant}_full.mp4",
        "log": f"logs/E195/cem/full/{variant}.log",
    }


def make_row(source: dict[str, str], ordinal: int) -> dict[str, Any]:
    case_id = source["case_id"]
    worker = C.worker_for(case_id)
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "case_id": case_id,
        "object_key": C.object_key_of(case_id),
        "arm": "A3",
        "stage": "full",
        "source_exp": "E192",
        "retarget_variant_id": source.get("retarget_variant_id", ""),
        "target_variant_id": source.get("target_variant_id", ""),
        "person": source.get("person", ""),
        "hand_collision_variant_id": source.get(
            "hand_collision_variant_id", C.HAND_COLLISION_VARIANT_ID
        ),
        "spider_method_id": C.METHOD_ID,
        "target_task": source["target_task"],
        "target_scene": C.rel(source["target_scene"]),
        "trajectory": C.rel(source["trajectory"]),
        "contact_mask": C.rel(source["contact_mask"]),
        "override_id": source["override_id"],
        "override_path": C.rel(source["override_path"]),
        "scene_act": C.rel(source["scene_act"]),
        "scene_name": source["scene_name"],
        "source_extra_overrides": source["extra_overrides"],
        "extra_overrides": C.e195_overrides(),
        **C.E195_GATE,
        "kp_pos": C.KP_POS,
        "kp_rot": C.KP_ROT,
        "gravcomp": False,
        **worker,
        "remote_staging_path": (
            f"workspace/core4d/results/E195/remote_staging/{worker['worker_id']}/"
            f"full/{case_id}"
        ),
        "cem_samples": C.FULL_SAMPLES,
        "cem_opt_steps": C.FULL_OPT_STEPS,
        "cem_seed": C.CEM_SEED,
        "status": "READY_FOR_RUN",
        "failure_mode": "",
        "execution_mode": "production",
        "updated_at": C.now(),
    }
    row.update(artifact_paths(case_id))
    return row


def build() -> None:
    sources = C.load_e192_rows()
    rows = [make_row(sources[case_id], i) for i, case_id in enumerate(C.all_case_ids(), 1)]
    manifest_dir = C.RESULTS / "s6_downstream/manifests"
    manifest = manifest_dir / "cem_full_manifest.tsv"
    C.write_tsv(manifest, rows, FIELDS)
    counts = {
        worker: sum(row["worker_id"] == worker for row in rows)
        for worker in C.WORKERS
    }
    C.write_json(
        manifest_dir / "build_summary.json",
        {
            "created_at": C.now(),
            "method_id": C.METHOD_ID,
            "source": C.rel(C.E192_MANIFEST),
            "rows": len(rows),
            "worker_counts": counts,
            "gate": C.E195_GATE,
            "cem_samples": C.FULL_SAMPLES,
            "cem_opt_steps": C.FULL_OPT_STEPS,
            "cem_seed": C.CEM_SEED,
        },
    )
    print(f"E195 manifest rows={len(rows)} workers={counts}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print("--apply is required")
        return 2
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

