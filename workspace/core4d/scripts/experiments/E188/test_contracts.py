#!/usr/bin/env python3
"""Direct contract tests for E188 mass, authority, queues, and commands."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import build_mass5_scenes as scenes
import common
import deploy_remote_a100 as deploy
import run_full_queue as runner


def test_mass_and_scene_contract() -> None:
    rows = common.read_tsv(common.S0 / "scene_manifest.tsv")
    assert len(rows) == len({row["case_id"] for row in rows}) == 15
    assert {row["case_id"] for row in rows} == common.queue_case_set()
    for row in rows:
        target = common.repo_path(row["scene_act"])
        inertial = scenes.object_inertial(ET.parse(target).getroot())
        assert float(inertial.attrib["mass"]) == 5.0
        old = scenes.floats(row["old_diaginertia"])
        new = scenes.floats(inertial.attrib["diaginertia"])
        assert all(abs(n / o - 2.5) < 1e-8 for o, n in zip(old, new, strict=True))


def test_authority_and_queue_contract() -> None:
    authority = common.read_tsv(common.S0 / "authority_manifest.tsv")
    assert len(authority) == 15
    assert sum(row["object_key"] == "bucket003" for row in authority) == 2
    assert sum(row["object_key"] == "bucket004" for row in authority) == 0
    assert sum(row["object_key"] == "bucket007" for row in authority) == 13
    queue = runner.queue_payload()
    assert queue["queue_revision"] == 2
    assert {worker: len(queue["queues"][worker]) for worker in common.WORKERS} == {"local-0": 7, "a100-4": 4, "a100-5": 4}
    assert queue["remaining_counts_by_worker"] == {"local-0": 6, "a100-4": 3, "a100-5": 3}
    assert queue["gpu_overlap_policy"] == "USER_AUTHORIZED_NO_MEMORY_OR_COMPUTE_PROCESS_GATE"
    cases = [row["case_id"] for worker in common.WORKERS for row in queue["queues"][worker]]
    assert len(cases) == len(set(cases)) == 15
    assert all(queue["queues"][worker][0]["case_id"] == common.CANARY_BY_WORKER[worker] for worker in common.WORKERS)
    assert {"bucket003_20231018_003_p1", "bucket007_20231018_021_p2", "bucket007_20231018_021_p1", "bucket007_20231018_019_p2"} <= {row["case_id"] for row in queue["queues"]["local-0"]}


def test_full_budget_commands() -> None:
    total = 0
    for worker in common.WORKERS:
        render_mode = runner.expected_render_mode(worker)
        for row in runner.load_queue(worker):
            command = runner.build_command(row, python_bin=sys.executable, device_id=0, output_dir=Path("/tmp/e188/out"), video=Path("/tmp/e188/video.mp4"), render_mode=render_mode)
            expected_video = "save_video=true" if worker == "local-0" else "save_video=false"
            assert {"num_samples=1024", "max_num_iterations=32", "seed=0", expected_video, "save_info=true", "+query_tape_enabled=false", "+query_tape_record_geometry_state=false", "device=cuda:0"} <= set(command)
            assert not any(value.startswith("max_sim_steps=") for value in command)
            total += 1
    assert total == 15


def test_remote_runtime_snapshot_contract() -> None:
    paths = deploy.snapshot_paths("full")
    assert "spider/query_tape.py" in paths
    assert "examples/run_mjwp.py" in paths
    assert "workspace/core4d/scripts/experiments/E188/run_full_queue.py" in paths
    assert "workspace/core4d/results/E188/s5_full/queue_speed_rebalanced_v2/queue_manifest.json" in paths
    launcher = (common.REPO / "workspace/core4d/scripts/launch/active/run_E188_remote_a100.sh").read_text()
    assert 'REMOTE_PYTHONPATH="$REMOTE_ROOT:$REMOTE_ROOT/workspace/core4d/scripts/experiments/E188"' in launcher
    assert 'import spider.query_tape; print(spider.query_tape.__file__)' in launcher
    assert runner.manifest_render_mode({"worker": "local-0", "video": {"path": "legacy.mp4"}}) == "INLINE_CEM"


def main() -> int:
    tests = (
        test_mass_and_scene_contract,
        test_authority_and_queue_contract,
        test_full_budget_commands,
        test_remote_runtime_snapshot_contract,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E188_CONTRACT_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
