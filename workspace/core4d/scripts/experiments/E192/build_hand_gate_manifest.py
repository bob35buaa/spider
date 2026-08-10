#!/usr/bin/env python3
"""Build and snapshot the frozen E192 A0-sentinel/A2 manifests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e192_common as C  # noqa: E402


FIELDS = [
    "ordinal", "case_id", "object_key", "arm", "stage", "source_exp",
    "retarget_variant_id", "target_variant_id", "person", "hand_collision_variant_id",
    "spider_method_id", "target_task", "target_scene", "trajectory",
    "contact_mask", "override_id", "override_path", "override_sha256",
    "scene_act", "scene_name", "scene_sha256", "scene_snapshot_path",
    "scene_snapshot_sha256", "trajectory_sha256", "contact_mask_sha256",
    "trajectory_physical_state", "trajectory_physical_sha256",
    "contact_mask_physical_state", "contact_mask_physical_sha256",
    "extra_overrides", "cem_hand_gate_min_sdf_m",
    "cem_hand_gate_max_violation_pct", "cem_hand_gate_hard_floor_m",
    "kp_pos", "kp_rot", "gravcomp", "worker_id", "host", "gpu_id",
    "remote_staging_path", "cem_samples", "cem_opt_steps", "cem_seed",
    "variant", "result_npz", "outdir_npz", "config_act", "video", "log",
    "status", "failure_mode", "execution_mode", "updated_at",
]


def snapshot(sources: dict[str, dict[str, str]]) -> None:
    tasks = [sources[case_id]["target_task"] for case_id in C.all_case_ids()]
    subprocess.run(
        ["bash", "workspace/core4d/scripts/convert/snapshot_scenes.sh", "E192", *tasks],
        cwd=C.REPO,
        check=True,
    )


def artifact_paths(case_id: str, arm: str, stage: str) -> dict[str, str]:
    variant = f"E192_{case_id}_{arm}"
    if stage == "canary":
        variant += "_canary"
    root = f"workspace/core4d/results/E192/s6_downstream/cem/{stage}"
    suffix = "canary" if stage == "canary" else "full"
    return {
        "variant": variant,
        "result_npz": f"{root}/{variant}.npz",
        "outdir_npz": f"{root}/{variant}_outdir_{suffix}/trajectory_mjwp_act.npz",
        "config_act": f"{root}/{variant}_outdir_{suffix}/config_act.yaml",
        "video": f"workspace/core4d/results/E192/s6_downstream/render/{stage}/{variant}_{suffix}.mp4",
        "log": f"logs/E192/cem/{stage}/{variant}.log",
    }


def make_row(
    src: dict[str, str], arm: str, stage: str, ordinal: int, require_snapshot: bool
) -> dict[str, Any]:
    case_id = src["case_id"]
    worker = C.worker_for(stage, ordinal)
    gate = C.A0_GATE if arm == "A0" else C.A2_GATE
    scene = C.repo_path(src["scene_act"])
    snapshot_path = (
        C.RESULTS / "scene_snapshot" / src["target_task"] / scene.name
    )
    if require_snapshot and not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)

    def input_provenance(path_key: str, sha_key: str) -> tuple[Path, str, str, str]:
        """Resolve the recorded authority file and report physical readiness.

        Some historical manifests point at ``.../0/trajectory_kinematic.npz``
        while the authority archive is stored one directory above.  Prefer a
        sibling candidate whose bytes match the frozen SHA; this preserves the
        authority without treating a numerically close re-export as identical.
        """
        path = C.repo_path(src[path_key])
        recorded = src.get(sha_key, "")
        if len(recorded) != 64:
            raise ValueError(f"{case_id} invalid source authority {sha_key}={recorded!r}")
        candidates = [path]
        if path.name == "trajectory_kinematic.npz" and path.parent.name == "0":
            candidates.append(path.parent.parent / path.name)
        for candidate in candidates:
            if candidate.is_file() and C.sha256(candidate) == recorded:
                return candidate, recorded, "MATCH", recorded
        if path.is_file():
            actual = C.sha256(path)
            return path, recorded, "DRIFT", actual
        return path, recorded, "MISSING", ""

    trajectory_path, trajectory_sha, trajectory_state, trajectory_physical_sha = input_provenance(
        "trajectory", "trajectory_sha256"
    )
    contact_path, contact_sha, contact_state, contact_physical_sha = input_provenance(
        "contact_mask", "contact_mask_sha256"
    )
    samples = C.CANARY_SAMPLES if stage == "canary" else C.FULL_SAMPLES
    steps = C.CANARY_OPT_STEPS if stage == "canary" else C.FULL_OPT_STEPS
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "case_id": case_id,
        "object_key": C.object_key_of(case_id),
        "arm": arm,
        "stage": stage,
        "source_exp": src["_source_exp"],
        "retarget_variant_id": src.get("retarget_variant_id", ""),
        "target_variant_id": src.get("target_variant_id", ""),
        "person": src.get("person", ""),
        "hand_collision_variant_id": src.get("hand_collision_variant_id", C.HAND_COLLISION_VARIANT_ID),
        "spider_method_id": C.METHOD_ID,
        "target_task": src["target_task"],
        "target_scene": C.rel(src["target_scene"]),
        "trajectory": C.rel(trajectory_path),
        "contact_mask": C.rel(contact_path),
        "override_id": src["override_id"],
        "override_path": C.rel(src["override_path"]),
        "override_sha256": C.sha256(src["override_path"]),
        "scene_act": C.rel(scene),
        "scene_name": src["scene_name"],
        "scene_sha256": C.sha256(scene),
        "scene_snapshot_path": C.rel(snapshot_path) if snapshot_path.is_file() else "",
        "scene_snapshot_sha256": C.sha256(snapshot_path) if snapshot_path.is_file() else "",
        "trajectory_sha256": trajectory_sha,
        "contact_mask_sha256": contact_sha,
        "trajectory_physical_state": trajectory_state,
        "trajectory_physical_sha256": trajectory_physical_sha,
        "contact_mask_physical_state": contact_state,
        "contact_mask_physical_sha256": contact_physical_sha,
        "extra_overrides": "" if arm == "A0" else C.a2_overrides(),
        **gate,
        "kp_pos": C.KP_POS,
        "kp_rot": C.KP_ROT,
        "gravcomp": False,
        **worker,
        "remote_staging_path": (
            f"workspace/core4d/results/E192/remote_staging/{worker['worker_id']}/"
            f"{stage}/{case_id}"
        ),
        "cem_samples": samples,
        "cem_opt_steps": steps,
        "cem_seed": C.CEM_SEED,
        "status": "READY_FOR_RUN",
        "failure_mode": "",
        "execution_mode": "canary" if stage == "canary" else "production",
        "updated_at": C.now(),
    }
    row.update(artifact_paths(case_id, arm, stage))
    return row


def build(*, do_snapshot: bool) -> None:
    sources = C.load_source_rows()
    if do_snapshot:
        snapshot(sources)
    sentinel = [
        make_row(sources[cid], "A0", "baseline_sentinel", i, do_snapshot)
        for i, cid in enumerate(C.SENTINEL_CASES, 1)
    ]
    canary = [
        make_row(sources[cid], "A2", "canary", i, do_snapshot)
        for i, cid in enumerate(C.CANARY_CASES, 1)
    ]
    full = [
        make_row(sources[cid], "A2", "full", i, do_snapshot)
        for i, cid in enumerate(C.all_case_ids(), 1)
    ]
    manifests = C.RESULTS / "s6_downstream/manifests"
    C.write_tsv(manifests / "cem_baseline_sentinel_manifest.tsv", sentinel, FIELDS)
    C.write_tsv(manifests / "cem_canary_manifest.tsv", canary, FIELDS)
    C.write_tsv(manifests / "cem_full_manifest.tsv", full, FIELDS)
    C.write_json(
        manifests / "build_summary.json",
        {
            "created_at": C.now(),
            "snapshot": do_snapshot,
            "method_id": C.METHOD_ID,
            "sentinel_rows": len(sentinel),
            "canary_rows": len(canary),
            "full_rows": len(full),
            "full_worker_counts": {
                worker["worker_id"]: sum(
                    row["worker_id"] == worker["worker_id"] for row in full
                )
                for worker in C.WORKERS
            },
            "full_budget_rows": len(sentinel) + len(full),
            "a0_gate": C.A0_GATE,
            "a2_gate": C.A2_GATE,
            "unique_trajectory_physical_states": {
                state: sum(
                    row["trajectory_physical_state"] == state for row in full
                )
                for state in ("MATCH", "MISSING", "DRIFT")
            },
        },
    )
    print(
        f"E192 manifests: sentinel={len(sentinel)} canary={len(canary)} "
        f"full={len(full)} full_budget={len(sentinel) + len(full)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write frozen manifests")
    parser.add_argument("--snapshot", action="store_true", help="snapshot all 15 source scenes")
    args = parser.parse_args()
    if not args.apply:
        print("--apply is required; builder never mutates source scene/config files")
        return 2
    build(do_snapshot=args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
