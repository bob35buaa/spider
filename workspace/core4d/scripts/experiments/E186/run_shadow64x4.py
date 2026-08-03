#!/usr/bin/env python3
"""Run one bounded E186 64x4 CEM query-tape shadow per frozen object."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from spider.query_tape import finalize_cem_query_tape

HERE = Path(__file__).absolute().parent
REPO = HERE.parents[4]
SCENE_MANIFEST = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
OUTPUT = REPO / "workspace/core4d/results/E186/s3_prg_audit/shadow64x4_v7"
REPRESENTATIVE_CASES = (
    "bucket003_20231018_003_p1",
    "bucket004_20231002_021_p1",
    "bucket007_20231020_055_p1",
)
RECORD_START_SIM_STEP = {
    "bucket003_20231018_003_p1": 16,
    "bucket004_20231002_021_p1": 12,
    "bucket007_20231020_055_p1": 22,
}
E178_TIMING_SOURCE_SHA256 = {
    "bucket003_20231018_003_p1": "cfc1c1c3f06eba2b88aec01d2b514cb2d249c03c8a8353ad9e95927cba169030",
    "bucket004_20231002_021_p1": "1f9c2f3f9ff7ec57cb77b7f852cbc8fd8e4408ec0da41f69329c2c62d2efa01b",
    "bucket007_20231020_055_p1": "d72af0f315b51811b912033f83874b05c2fe7eaed56dc923f4a59fd987c0de84",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    """Serialize a lexical repository-relative path."""
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(REPO.absolute()))
    except ValueError:
        return str(absolute)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def rows_by_case() -> dict[str, dict[str, str]]:
    """Load and validate the three preregistered representative rows."""
    with SCENE_MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    selected = {
        row["case_id"]: row for row in rows if row["case_id"] in REPRESENTATIVE_CASES
    }
    if (
        tuple(case for case in REPRESENTATIVE_CASES if case in selected)
        != REPRESENTATIVE_CASES
    ):
        raise RuntimeError("one or more representative cases left the frozen keep22")
    if {selected[case]["object_key"] for case in REPRESENTATIVE_CASES} != {
        "bucket003",
        "bucket004",
        "bucket007",
    }:
        raise RuntimeError("representative object coverage changed")
    return selected


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    gpu_id: int,
    output_dir: Path,
    tape_root: Path,
) -> list[str]:
    """Build the fixed single-control-tick E186 shadow command."""
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "save_info=true",
        f"output_dir={output_dir}",
        "num_samples=64",
        "max_num_iterations=4",
        "seed=0",
        f"device=cuda:{gpu_id}",
        "+query_tape_enabled=true",
        f"+query_tape_output_dir={tape_root}",
        f"+query_tape_run_id={row['case_id']}",
        "+query_tape_max_chunks=1",
        "+query_tape_stop_after_chunks=1",
        f"+query_tape_record_start_sim_step={RECORD_START_SIM_STEP[row['case_id']]}",
        "+query_tape_record_geometry_state=true",
    ]


def run_one(
    row: dict[str, str], *, python_bin: str, gpu_id: int, output_root: Path
) -> dict[str, Any]:
    """Run or resume-validate one immutable bounded shadow case."""
    case_id = row["case_id"]
    output_dir = output_root / "runs" / f"{case_id}_outdir"
    result_path = output_dir / "trajectory_mjwp_act.npz"
    config_path = output_dir / "config_act.yaml"
    tape_root = output_root / "raw_chunks"
    chunk_manifest = tape_root / case_id / "chunk_manifest.json"
    log_path = output_root / "logs" / f"{case_id}.log"
    case_manifest = output_root / "manifests" / f"{case_id}.json"
    if case_manifest.is_file():
        payload = json.loads(case_manifest.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS":
            for key in ("result", "config", "chunk_manifest"):
                artifact = REPO / payload[key]["path"]
                if not artifact.is_file() or sha256(artifact) != payload[key]["sha256"]:
                    raise RuntimeError(f"complete shadow artifact changed: {artifact}")
            return payload
    if output_dir.exists() or chunk_manifest.parent.exists():
        raise RuntimeError(f"refusing to append to incomplete shadow: {case_id}")
    output_dir.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        row,
        python_bin=python_bin,
        gpu_id=gpu_id,
        output_dir=output_dir,
        tape_root=tape_root,
    )
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("# " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    if result.returncode != 0 or not result_path.is_file() or not config_path.is_file():
        raise RuntimeError(
            f"shadow failed case={case_id} rc={result.returncode} log={log_path}"
        )
    tape = finalize_cem_query_tape(
        SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=str(tape_root),
            query_tape_run_id=case_id,
        ),
        provenance={
            "experiment_id": "E186",
            "stage": "S3_shadow64x4",
            "case_id": case_id,
            "object_key": row["object_key"],
            "override_id": row["override_id"],
            "override_sha256": row["override_sha256"],
            "scene_sha256": row["effective_scene_sha256"],
            "grid_manifest_sha256": row["object_distance_manifest_sha256"],
            "budget": {
                "samples": 64,
                "opt_steps": 4,
                "seed": 0,
                "max_sim_steps": "PRODUCTION_CONFIG",
                "warmup_steps": 0.2,
                "recorded_query": "E178_FROZEN_CONTACT_TIMED_CEM",
                "record_start_sim_step": RECORD_START_SIM_STEP[case_id],
                "record_timing_source": "E178_FIRST_FINAL_ITER_SURFACE_MEAN_GE_0P3_AND_VALID_GT_0",
                "record_timing_source_sha256": E178_TIMING_SOURCE_SHA256[case_id],
            },
        },
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "num_samples": 64,
        "max_num_iterations": 4,
        "seed": 0,
        "query_tape_enabled": True,
        "query_tape_max_chunks": 1,
        "query_tape_stop_after_chunks": 1,
        "query_tape_record_start_sim_step": RECORD_START_SIM_STEP[case_id],
        "query_tape_record_geometry_state": True,
        "object_distance_backend": "grid_sdf",
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if (
        mismatch
        or int(config.get("max_sim_steps", 0)) <= 14
        or tape.get("status") != "COMPLETE"
        or tape.get("chunk_count") != 1
    ):
        raise RuntimeError(f"shadow runtime contract mismatch: {mismatch} tape={tape}")
    payload = {
        "schema": "e186_shadow64x4_run_v7",
        "experiment_id": "E186",
        "stage": "S3_shadow64x4",
        "status": "PASS",
        "case_id": case_id,
        "object_key": row["object_key"],
        "gpu_id": gpu_id,
        "wall_seconds": wall_seconds,
        "command": command,
        "result": {"path": rel(result_path), "sha256": sha256(result_path)},
        "config": {"path": rel(config_path), "sha256": sha256(config_path)},
        "chunk_manifest": {
            "path": rel(chunk_manifest),
            "sha256": sha256(chunk_manifest),
        },
        "chunk_content_sha256": tape["content_sha256"],
        "log": rel(log_path),
    }
    write_json(case_manifest, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse a bounded local/remote worker invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True, choices=REPRESENTATIVE_CASES)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run one representative shadow and persist its immutable manifest."""
    args = parse_args()
    row = rows_by_case()[args.case_id]
    payload = run_one(
        row,
        python_bin=args.python_bin,
        gpu_id=args.gpu_id,
        output_root=args.output_root
        if args.output_root.is_absolute()
        else REPO / args.output_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
