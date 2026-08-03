#!/usr/bin/env python3
"""Build and audit short same-device E178 baselines for the E187 A3 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from statistics import median
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E187/s4_canary"
E178_MANIFEST = (
    REPO
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
OUTPUT = RESULTS / "efficiency_gate.json"
MAX_SIM_STEPS = 30
PLAN_PATTERN = re.compile(
    r"plan time:\s*([0-9.]+)s,\s*sim_steps:\s*\d+/\d+,\s*opt_steps:\s*(\d+)"
)
ASSIGNMENTS = {
    "bucket003_20231018_003_p1": {"worker": "local-0", "gpu_id": 0},
    "bucket004_20231002_021_p1": {"worker": "remote-0", "gpu_id": 0},
    "bucket007_20231020_055_p1": {"worker": "remote-1", "gpu_id": 1},
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path = REPO) -> str:
    """Serialize a path relative to an audit root."""
    return path.absolute().relative_to(root.absolute()).as_posix()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def authority(case_id: str) -> dict[str, str]:
    """Validate one frozen E178 baseline row."""
    if case_id not in ASSIGNMENTS:
        raise ValueError(f"not an A3 representative: {case_id}")
    rows = {row["case_id"]: row for row in read_tsv(E178_MANIFEST)}
    row = rows[case_id]
    if (row["cem_samples"], row["cem_opt_steps"], row["cem_seed"]) != (
        "1024",
        "32",
        "0",
    ):
        raise RuntimeError(f"{case_id}: E178 budget authority changed")
    for value, expected, label in (
        (row["override_path"], row["override_sha256"], "override"),
        (row["scene_act"], row["effective_scene_sha256"], "scene"),
        (row["trajectory"], row["trajectory_sha256"], "trajectory"),
        (row["contact_mask"], row["contact_mask_sha256"], "contact"),
    ):
        if sha256(REPO / value) != expected:
            raise RuntimeError(f"{case_id}: E178 {label} SHA changed")
    return row


def baseline_root(case_id: str, root: Path = REPO) -> Path:
    """Return the isolated baseline row root."""
    return (
        root
        / "workspace/core4d/results/E187/s4_canary/efficiency_baseline/rows"
        / case_id
    )


def build_command(
    case_id: str, *, python_bin: str, gpu_id: int, repo_root: Path
) -> list[str]:
    """Build one short full-density E178 same-device timing command."""
    row = authority(case_id)
    output_dir = baseline_root(case_id, repo_root) / "outdir"
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "save_info=false",
        f"output_dir={output_dir}",
        "num_samples=1024",
        "max_num_iterations=32",
        "seed=0",
        f"device=cuda:{gpu_id}",
        f"max_sim_steps={MAX_SIM_STEPS}",
        "+query_tape_enabled=false",
        "+query_tape_record_geometry_state=false",
    ]


def optimized_times(path: Path) -> list[float]:
    """Extract only fully optimized records from a CEM log."""
    return [
        float(value)
        for value, iterations in PLAN_PATTERN.findall(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if int(iterations) == 32
    ]


def verify_baseline_row(root: Path, case_id: str) -> dict[str, Any]:
    """Verify one isolated baseline config and timing log."""
    row = authority(case_id)
    row_root = baseline_root(case_id, root)
    log = row_root / "run.log"
    config_path = row_root / "outdir/config_act.yaml"
    if not log.is_file() or not config_path.is_file():
        return {"status": "FAIL", "case_id": case_id, "error": "missing_artifact"}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "num_samples": 1024,
        "max_num_iterations": 32,
        "seed": 0,
        "max_sim_steps": MAX_SIM_STEPS,
        "query_tape_enabled": False,
        "query_tape_record_geometry_state": False,
        "scene_name": row["scene_name"],
    }
    mismatches = {
        key: [config.get(key), value]
        for key, value in expected.items()
        if config.get(key) != value
    }
    times = optimized_times(log)
    status = "PASS" if not mismatches and len(times) >= 5 else "FAIL"
    return {
        "status": status,
        "case_id": case_id,
        "optimized_record_count": len(times),
        "median_plan_time_seconds": median(times) if times else None,
        "config_mismatches": mismatches,
        "e178_override_sha256": row["override_sha256"],
        "e178_scene_sha256": row["effective_scene_sha256"],
        "log": {"path": relative(log, root), "sha256": sha256(log)},
        "config": {
            "path": relative(config_path, root),
            "sha256": sha256(config_path),
        },
    }


def gpu_rows(path: Path) -> dict[int, dict[str, str]]:
    """Parse a passive nvidia-smi CSV keyed by physical index."""
    rows: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for values in csv.reader(stream):
            if len(values) < 3:
                continue
            rows[int(values[0].strip())] = {
                "name": values[1].strip(),
                "uuid": values[2].strip(),
            }
    return rows


def device_identity(root: Path, worker: str, gpu_id: int) -> dict[str, Any]:
    """Prove canary and baseline used the same physical GPU UUID."""
    location = "local" if worker == "local-0" else "remote"
    canary_path = (
        root
        / "workspace/core4d/results/E187/s4_canary/process_snapshots"
        / location
        / "gpu_before.csv"
    )
    baseline_path = (
        root
        / "workspace/core4d/results/E187/s4_canary/process_snapshots/efficiency"
        / location
        / "gpu_before.csv"
    )
    canary = gpu_rows(canary_path)[gpu_id]
    baseline = gpu_rows(baseline_path)[gpu_id]
    return {
        "status": "PASS" if canary == baseline else "FAIL",
        "gpu_id": gpu_id,
        "canary": canary,
        "baseline": baseline,
        "canary_snapshot_sha256": sha256(canary_path),
        "baseline_snapshot_sha256": sha256(baseline_path),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one immutable audit manifest atomically."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to replace efficiency gate: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def audit(root: Path, output: Path) -> dict[str, Any]:
    """Audit all three same-device E187/E178 plan-time ratios."""
    records = []
    for case_id, assignment in ASSIGNMENTS.items():
        baseline = verify_baseline_row(root, case_id)
        canary_manifest = (
            root
            / "workspace/core4d/results/E187/s4_canary/rows"
            / case_id
            / "manifest.json"
        )
        canary = json.loads(canary_manifest.read_text(encoding="utf-8"))
        canary_log = root / canary["log"]["path"]
        if sha256(canary_log) != canary["log"]["sha256"]:
            raise RuntimeError(f"{case_id}: canary log SHA changed")
        canary_times = optimized_times(canary_log)
        if len(canary_times) < 5:
            raise RuntimeError(f"{case_id}: insufficient optimized canary records")
        identity = device_identity(
            root, assignment["worker"], int(assignment["gpu_id"])
        )
        baseline_median = float(baseline["median_plan_time_seconds"] or 0.0)
        canary_median = median(canary_times)
        ratio = (
            canary_median / baseline_median if baseline_median > 0.0 else float("inf")
        )
        passed = (
            baseline["status"] == "PASS"
            and canary.get("status") == "PASS"
            and identity["status"] == "PASS"
            and ratio <= 1.5
            and int(canary["peak_total_gpu_memory_mib"]) <= 6144
        )
        records.append(
            {
                "case_id": case_id,
                "worker": assignment["worker"],
                "status": "PASS" if passed else "FAIL",
                "same_device": identity,
                "e178_baseline": baseline,
                "e187_optimized_record_count": len(canary_times),
                "e187_median_plan_time_seconds": canary_median,
                "plan_time_ratio": ratio,
                "plan_time_ratio_le_1p5": ratio <= 1.5,
                "e187_peak_total_gpu_memory_mib": canary["peak_total_gpu_memory_mib"],
                "peak_total_gpu_memory_le_6144_mib": int(
                    canary["peak_total_gpu_memory_mib"]
                )
                <= 6144,
            }
        )
    payload = {
        "schema": "e187_a3_efficiency_gate_v1",
        "experiment_id": "E187",
        "stage": "A3_S4_EFFICIENCY",
        "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "baseline_budget": {
            "samples": 1024,
            "iterations": 32,
            "seed": 0,
            "max_sim_steps": MAX_SIM_STEPS,
            "minimum_optimized_records": 5,
        },
        "thresholds": {"plan_time_ratio": 1.5, "peak_total_gpu_memory_mib": 6144},
        "records": records,
    }
    atomic_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--python-bin", default=sys.executable)
    preflight.add_argument("--repo-root", type=Path, default=REPO)
    emit = sub.add_parser("command")
    emit.add_argument("--case-id", required=True, choices=tuple(ASSIGNMENTS))
    emit.add_argument("--python-bin", required=True)
    emit.add_argument("--gpu-id", required=True, type=int)
    emit.add_argument("--repo-root", required=True, type=Path)
    verify = sub.add_parser("verify-row")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--case-id", required=True, choices=tuple(ASSIGNMENTS))
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--root", type=Path, default=REPO)
    audit_parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> int:
    """Preflight commands, verify baseline rows, or close C9."""
    args = build_parser().parse_args()
    if args.command == "command":
        print(
            shlex.join(
                build_command(
                    args.case_id,
                    python_bin=args.python_bin,
                    gpu_id=args.gpu_id,
                    repo_root=args.repo_root,
                )
            )
        )
        return 0
    if args.command == "preflight":
        payload: dict[str, Any] = {
            case_id: build_command(
                case_id,
                python_bin=args.python_bin,
                gpu_id=int(assignment["gpu_id"]),
                repo_root=args.repo_root,
            )
            for case_id, assignment in ASSIGNMENTS.items()
        }
        print(json.dumps({"status": "PREFLIGHT_PASS", "commands": payload}, indent=2))
        return 0
    if args.command == "verify-row":
        payload = verify_baseline_row(args.root, args.case_id)
    else:
        output = args.output if args.output.is_absolute() else REPO / args.output
        payload = audit(args.root, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
