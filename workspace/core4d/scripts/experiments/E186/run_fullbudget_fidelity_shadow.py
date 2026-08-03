#!/usr/bin/env python3
"""Run formal-budget transform shadows for nontrivial E186 R/G fidelity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from run_fullbudget_feasibility_probe import run_process
from run_shadow64x4 import (
    RECORD_START_SIM_STEP,
    REPO,
    rel,
    rows_by_case,
    sha256,
    write_json,
)

from spider.query_tape import finalize_cem_query_tape

OUTPUT = REPO / "workspace/core4d/results/E186/s3_prg_audit/fullbudget_fidelity_v1"
FIDELITY_CASES = (
    "bucket003_20231018_003_p1",
    "bucket007_20231020_055_p1",
)


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    gpu_id: int,
    output_dir: Path,
    tape_root: Path,
) -> list[str]:
    """Build one formal-budget transform-shadow command."""
    case_id = row["case_id"]
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
        "num_samples=1024",
        "max_num_iterations=32",
        "seed=0",
        f"device=cuda:{gpu_id}",
        "+query_tape_enabled=true",
        f"+query_tape_output_dir={tape_root}",
        f"+query_tape_run_id={case_id}",
        "+query_tape_max_chunks=1",
        "+query_tape_stop_after_chunks=1",
        f"+query_tape_record_start_sim_step={RECORD_START_SIM_STEP[case_id]}",
        "+query_tape_record_geometry_state=true",
    ]


def summarize_chunk(path: Path) -> dict[str, Any]:
    """Validate transform schema and summarize active feasibility axes."""
    required = {
        "geometry_geom_xpos",
        "geometry_geom_xmat",
        "geometry_body_xpos",
        "geometry_body_xmat",
    }
    with np.load(path, allow_pickle=False) as chunk:
        if not required <= set(chunk.files):
            raise RuntimeError("formal fidelity chunk lacks MJWarp transforms")
        combined = np.asarray(chunk["sample_gate_valid_mask"], dtype=bool)
        posture = np.asarray(chunk["sample_posture_valid_mask"], dtype=bool)
        surface = np.asarray(chunk["reward_trace_surface_band_rew"], dtype=float)
        rewards = np.asarray(chunk["rewards"], dtype=float)
        return {
            "sample_count": int(len(combined)),
            "combined_valid_count": int(combined.sum()),
            "posture_valid_count": int(posture.sum()),
            "surface_active_candidate_count": int(
                (np.abs(surface).max(axis=1) > 1e-12).sum()
            ),
            "reward_finite": bool(np.isfinite(rewards).all()),
            "selected_count": int(len(chunk["selected_indices"])),
            "chunk_size_bytes": int(path.stat().st_size),
        }


def run_one(
    row: dict[str, str], *, python_bin: str, gpu_id: int, output_root: Path
) -> dict[str, Any]:
    """Run or immutable-resume one formal-budget fidelity shadow."""
    case_id = row["case_id"]
    output_dir = output_root / "runs" / f"{case_id}_outdir"
    result_path = output_dir / "trajectory_mjwp_act.npz"
    config_path = output_dir / "config_act.yaml"
    tape_root = output_root / "raw_chunks"
    chunk_path = tape_root / case_id / "chunk_000000.npz"
    chunk_manifest = tape_root / case_id / "chunk_manifest.json"
    log_path = output_root / "logs" / f"{case_id}.log"
    case_manifest = output_root / "manifests" / f"{case_id}.json"
    if case_manifest.is_file():
        payload = json.loads(case_manifest.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS":
            for key in ("result", "config", "chunk_manifest"):
                artifact = REPO / payload[key]["path"]
                if not artifact.is_file() or sha256(artifact) != payload[key]["sha256"]:
                    raise RuntimeError(
                        f"complete fidelity artifact changed: {artifact}"
                    )
            return payload
    if output_dir.exists() or chunk_manifest.parent.exists():
        raise RuntimeError(
            f"refusing to append to incomplete fidelity shadow: {case_id}"
        )
    output_dir.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        row,
        python_bin=python_bin,
        gpu_id=gpu_id,
        output_dir=output_dir,
        tape_root=tape_root,
    )
    returncode, wall_seconds, peak_memory_mib = run_process(command, log_path, gpu_id)
    if returncode != 0 or not result_path.is_file() or not config_path.is_file():
        raise RuntimeError(
            f"fidelity shadow failed case={case_id} rc={returncode} log={log_path}"
        )
    tape = finalize_cem_query_tape(
        SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=str(tape_root),
            query_tape_run_id=case_id,
        ),
        provenance={
            "experiment_id": "E186",
            "stage": "S3c_fullbudget_fidelity",
            "case_id": case_id,
            "budget": {"samples": 1024, "iterations": 32, "seed": 0},
            "record_start_sim_step": RECORD_START_SIM_STEP[case_id],
            "geometry_state_recorded": True,
        },
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "num_samples": 1024,
        "max_num_iterations": 32,
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
    if mismatch or tape.get("status") != "COMPLETE" or tape.get("chunk_count") != 1:
        raise RuntimeError(
            f"fidelity runtime contract mismatch: {mismatch} tape={tape}"
        )
    summary = summarize_chunk(chunk_path)
    payload = {
        "schema": "e186_fullbudget_fidelity_shadow_v1",
        "experiment_id": "E186",
        "stage": "S3c_fullbudget_fidelity",
        "status": "PASS",
        "case_id": case_id,
        "object_key": row["object_key"],
        "gpu_id": gpu_id,
        "wall_seconds": wall_seconds,
        "peak_total_gpu_memory_mib": peak_memory_mib,
        "summary": summary,
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
    """Parse one fidelity shadow invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True, choices=FIDELITY_CASES)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run one shadow and persist its immutable manifest."""
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
