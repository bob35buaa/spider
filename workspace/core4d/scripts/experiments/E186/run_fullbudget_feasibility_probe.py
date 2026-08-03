#!/usr/bin/env python3
"""Run bounded formal-budget probes for E186 low-budget false negatives."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from run_shadow64x4 import REPO, rel, rows_by_case, sha256, write_json

from spider.query_tape import finalize_cem_query_tape

OUTPUT = REPO / "workspace/core4d/results/E186/s3_prg_audit/fullbudget_probe_v1"
PROBE_CASES = (
    "bucket003_20231018_003_p1",
    "bucket007_20231020_055_p1",
)
RECORD_START_SIM_STEP = {
    "bucket003_20231018_003_p1": 12,
    "bucket007_20231020_055_p1": 22,
}


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    gpu_id: int,
    output_dir: Path,
    tape_root: Path,
) -> list[str]:
    """Build one exact-budget, bounded feasibility command."""
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
        "+query_tape_record_geometry_state=false",
    ]


def gpu_memory_mib(gpu_id: int) -> int | None:
    """Read total device memory use without interacting with other processes."""
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def run_process(
    command: list[str], log_path: Path, gpu_id: int
) -> tuple[int, float, int | None]:
    """Run one worker while passively polling total GPU memory use."""
    started = time.perf_counter()
    peak = gpu_memory_mib(gpu_id)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("# " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            current = gpu_memory_mib(gpu_id)
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(0.25)
    return process.returncode, time.perf_counter() - started, peak


def summarize_chunk(path: Path) -> dict[str, Any]:
    """Extract the feasibility axes without exact geometry replay."""
    with np.load(path, allow_pickle=False) as chunk:
        forbidden = {key for key in chunk.files if key.startswith("geometry_geom_")}
        forbidden |= {key for key in chunk.files if key.startswith("geometry_body_")}
        if forbidden:
            raise RuntimeError(
                f"full-budget probe recorded forbidden transforms: {forbidden}"
            )
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
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "selected_count": int(len(chunk["selected_indices"])),
        }


def run_one(
    row: dict[str, str], *, python_bin: str, gpu_id: int, output_root: Path
) -> dict[str, Any]:
    """Run or immutable-resume one formal-budget probe."""
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
                    raise RuntimeError(f"complete probe artifact changed: {artifact}")
            return payload
    if output_dir.exists() or chunk_manifest.parent.exists():
        raise RuntimeError(f"refusing to append to incomplete probe: {case_id}")
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
            f"probe failed case={case_id} rc={returncode} log={log_path}"
        )
    tape = finalize_cem_query_tape(
        SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=str(tape_root),
            query_tape_run_id=case_id,
        ),
        provenance={
            "experiment_id": "E186",
            "stage": "S3b_fullbudget_feasibility",
            "case_id": case_id,
            "budget": {"samples": 1024, "iterations": 32, "seed": 0},
            "record_start_sim_step": RECORD_START_SIM_STEP[case_id],
            "geometry_state_recorded": False,
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
        "query_tape_record_geometry_state": False,
        "object_distance_backend": "grid_sdf",
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch or tape.get("status") != "COMPLETE" or tape.get("chunk_count") != 1:
        raise RuntimeError(f"probe runtime contract mismatch: {mismatch} tape={tape}")
    feasibility = summarize_chunk(chunk_path)
    payload = {
        "schema": "e186_fullbudget_feasibility_probe_v1",
        "experiment_id": "E186",
        "stage": "S3b_fullbudget_feasibility",
        "status": "PASS",
        "feasibility_status": (
            "VALID_RECOVERED" if feasibility["combined_valid_count"] > 0 else "NO_VALID"
        ),
        "case_id": case_id,
        "object_key": row["object_key"],
        "gpu_id": gpu_id,
        "wall_seconds": wall_seconds,
        "peak_total_gpu_memory_mib": peak_memory_mib,
        "feasibility": feasibility,
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
    """Parse one local probe invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True, choices=PROBE_CASES)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run one probe and persist its immutable manifest."""
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
