#!/usr/bin/env python3
"""Independent E179 completion audit for the 16-row paired experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e179_common as C  # noqa: E402
import run_cem_queue as queue  # noqa: E402


def record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": passed,
            "detail": detail,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []
    manifest = C.read_tsv(
        C.RESULTS
        / "s6_downstream/manifests/cem_full_manifest.tsv"
    )
    cases = {row["case_id"] for row in manifest}
    record(
        checks,
        "full_manifest_16_unique",
        len(manifest) == 16 and len(cases) == 16,
        {"rows": len(manifest), "unique": len(cases)},
    )
    expected_statuses = {
        "run_complete_pending_eval",
        "run_complete",
        "eval_complete",
    }
    status_failures = [
        f"{row['case_id']}:{row['status']}"
        for row in manifest
        if row["status"] not in expected_statuses
    ]
    record(
        checks,
        "full_execution_complete_16",
        not status_failures,
        status_failures,
    )
    runtime_failures = {
        row["case_id"]: issues
        for row in manifest
        if (issues := queue.validate_runtime_outputs(row))
    }
    record(
        checks,
        "runtime_e167a_no_prg_16",
        not runtime_failures,
        runtime_failures,
    )

    authority = C.read_tsv(
        C.RESULTS / "input_authority/input_authority.tsv"
    )
    record(
        checks,
        "paired_authority_exact",
        len(authority) == 16
        and {row["case_id"] for row in authority} == cases,
        {
            "authority_rows": len(authority),
            "manifest_rows": len(manifest),
        },
    )
    workers = {
        worker: sum(row["assigned_worker"] == worker for row in manifest)
        for worker in C.WORKER_QUEUES
    }
    record(
        checks,
        "fixed_worker_distribution",
        workers
        == {
            "local-gpu0": 4,
            "a100-gpu2": 3,
            "a100-gpu3": 3,
            "a100-gpu6": 3,
            "a100-gpu7": 3,
        },
        workers,
    )

    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", "-z"], text=False
        )
        .decode()
        .split("\0")
    )
    required_tracked = []
    for row in manifest:
        task = Path(row["scene_act"]).parent
        required_tracked.extend(
            [
                task / "scene.xml",
                task / "scene_act.xml",
                Path(row["scene_act"]),
                Path(row["override_path"]),
            ]
        )
    missing_tracked = [
        str(path)
        for path in required_tracked
        if str(path) not in tracked
    ]
    record(
        checks,
        "active_scene_override_tracked_64",
        len(required_tracked) == 64 and not missing_tracked,
        {
            "required": len(required_tracked),
            "missing": missing_tracked,
        },
    )
    snapshot_manifest = C.RESULTS / "scene_snapshot/manifest.txt"
    sidecars = list(
        (
            C.RESULTS
            / "scene_snapshot/e179_rubber_hull_sidecars"
        ).glob("*/scene_act_E179_rubberHull.xml")
    )
    record(
        checks,
        "scene_snapshot_complete",
        snapshot_manifest.is_file() and len(sidecars) == 16,
        {
            "manifest": snapshot_manifest.is_file(),
            "rubber_sidecars": len(sidecars),
        },
    )

    eval_dir = C.RESULTS / "s6_downstream/eval/full"
    metrics = C.read_tsv(eval_dir / "e179_case_metrics.tsv") if (
        eval_dir / "e179_case_metrics.tsv"
    ).is_file() else []
    paired = C.read_tsv(eval_dir / "e179_vs_e173_paired.tsv") if (
        eval_dir / "e179_vs_e173_paired.tsv"
    ).is_file() else []
    gates = C.read_tsv(eval_dir / "e179_vs_e173_gate_matrix.tsv") if (
        eval_dir / "e179_vs_e173_gate_matrix.tsv"
    ).is_file() else []
    record(
        checks,
        "paired_12gate_metrics_16x12",
        len(metrics) == 16
        and len(paired) == 16
        and len(gates) == 192
        and {row["case_id"] for row in metrics} == cases,
        {
            "metrics": len(metrics),
            "paired": len(paired),
            "gate_cells": len(gates),
        },
    )

    e173_videos = {}
    for row in C.read_tsv(C.E173_FULL_MANIFEST):
        if row.get("object_key") == "box023":
            e173_videos[row["case_id"]] = C.repo_path(row["video"])
    e179_videos = {
        row["case_id"]: C.repo_path(row["video"])
        for row in manifest
    }
    paired_videos = {
        case_id: (
            C.RESULTS
            / "s6_downstream/render/full/paired"
            / f"{case_id}_E173_PRG_vs_E179_noPRG.mp4"
        )
        for case_id in cases
    }
    video_counts = {
        "e173": sum(path.is_file() for path in e173_videos.values()),
        "e179": sum(path.is_file() for path in e179_videos.values()),
        "paired": sum(path.is_file() for path in paired_videos.values()),
    }
    record(
        checks,
        "videos_16_baseline_16_new_16_paired",
        video_counts == {"e173": 16, "e179": 16, "paired": 16},
        video_counts,
    )
    review_path = (
        C.RESULTS
        / "s6_downstream/render/full/visual_review.tsv"
    )
    reviews = C.read_tsv(review_path) if review_path.is_file() else []
    record(
        checks,
        "visual_review_16",
        len(reviews) == 16
        and {row["case_id"] for row in reviews} == cases
        and all(row.get("review_status") == "reviewed" for row in reviews),
        {
            "rows": len(reviews),
            "reviewed": sum(
                row.get("review_status") == "reviewed"
                for row in reviews
            ),
        },
    )

    remote_pointer = (
        C.RESULTS / "s0_environment/a100_full_latest.json"
    )
    local_pointer = (
        C.RESULTS / "s0_environment/local_full_latest.json"
    )
    execution_detail: dict[str, Any] = {
        "local_pointer": local_pointer.is_file(),
        "remote_pointer": remote_pointer.is_file(),
    }
    execution_pass = local_pointer.is_file() and remote_pointer.is_file()
    if remote_pointer.is_file():
        remote = json.loads(remote_pointer.read_text(encoding="utf-8"))
        execution_detail["remote_gpus"] = remote.get("selected_gpus")
        execution_pass = execution_pass and remote.get(
            "selected_gpus"
        ) == ["2", "3", "6", "7"]
    record(
        checks,
        "execution_manifests_fixed_five_cards",
        execution_pass,
        execution_detail,
    )

    failures = [
        item["check"] for item in checks if not item["passed"]
    ]
    payload = {
        "created_at": C.now(),
        "experiment_id": "E179",
        "status": "pass" if not failures else "incomplete",
        "checks": checks,
        "failures": failures,
    }
    output = C.RESULTS / "completion_audit"
    C.write_json(output / "completion_audit.json", payload)
    lines = [
        "# E179 completion audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Checks: `{len(checks)-len(failures)}/{len(checks)}`",
        "",
        "| Check | Pass | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        lines.append(
            f"| `{item['check']}` | "
            f"{'PASS' if item['passed'] else 'FAIL'} | "
            f"`{json.dumps(item['detail'], ensure_ascii=False)}` |"
        )
    (output / "completion_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require_all and failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
