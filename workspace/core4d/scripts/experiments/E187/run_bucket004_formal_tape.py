#!/usr/bin/env python3
"""Capture E187's missing bucket004 formal reward-aligned query tape."""

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

import numpy as np
import yaml

from spider.query_tape import finalize_cem_query_tape

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
CASE_ID = "bucket004_20231002_021_p1"
TASK_ID = "dcv3_omnirt_v1_ref_fk_bucket004_20231002_021_p1"
RECORD_START_SIM_STEP = 12
SCENE_MANIFEST = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO / "workspace/core4d/results/E187/s2_canonical_grid_sdf/bucket004_formal_tape"
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Render a lexical repository-relative path."""
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def load_row() -> dict[str, str]:
    """Load and SHA-check the frozen bucket004 E186 production row."""
    with SCENE_MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    matches = [row for row in rows if row["case_id"] == CASE_ID]
    if len(matches) != 1:
        raise RuntimeError("bucket004 formal authority row is not unique")
    row = matches[0]
    expected = {
        "object_key": "bucket004",
        "target_task": TASK_ID,
        "cem_samples": "1024",
        "cem_opt_steps": "32",
        "cem_seed": "0",
        "physics_status": "COMPOUND_CPU_MJWARP_PASS",
        "object_distance_backend": "grid_sdf",
    }
    mismatch = {
        key: row.get(key) for key, value in expected.items() if row.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"bucket004 authority mismatch: {mismatch}")
    artifacts = {
        "scene_act": "effective_scene_sha256",
        "override_path": "override_sha256",
        "object_distance_manifest": "object_distance_manifest_sha256",
    }
    for path_key, sha_key in artifacts.items():
        path = REPO / row[path_key]
        if not path.is_file() or sha256(path) != row[sha_key]:
            raise RuntimeError(f"bucket004 authority artifact changed: {path}")
    return row


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    gpu_id: int,
    output_dir: Path,
    tape_root: Path,
) -> list[str]:
    """Build the frozen 1024x32 seed0 production-fixed contact capture."""
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
        f"+query_tape_run_id={CASE_ID}",
        "+query_tape_max_chunks=1",
        "+query_tape_stop_after_chunks=1",
        f"+query_tape_record_start_sim_step={RECORD_START_SIM_STEP}",
        "+query_tape_record_geometry_state=true",
    ]


def gpu_memory_mib(gpu_id: int) -> int | None:
    """Passively read total memory use without interacting with processes."""
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
    """Run the capture while only polling aggregate GPU memory."""
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
    """Validate the formal transform/reward/gate tape schema."""
    required = {
        "geometry_geom_xpos",
        "geometry_geom_xmat",
        "geometry_body_xpos",
        "geometry_body_xmat",
        "rewards",
        "selected_indices",
        "sample_gate_valid_mask",
        "sample_posture_valid_mask",
        "reward_trace_surface_band_rew",
    }
    with np.load(path, allow_pickle=False) as chunk:
        if not required <= set(chunk.files):
            raise RuntimeError(
                f"bucket004 formal tape lacks: {sorted(required - set(chunk.files))}"
            )
        geom = np.asarray(chunk["geometry_geom_xpos"])
        rewards = np.asarray(chunk["rewards"])
        combined = np.asarray(chunk["sample_gate_valid_mask"], dtype=bool)
        posture = np.asarray(chunk["sample_posture_valid_mask"], dtype=bool)
        surface = np.asarray(chunk["reward_trace_surface_band_rew"], dtype=float)
        if geom.shape[:2] != (1024, 48) or rewards.shape != (1024,):
            raise RuntimeError("bucket004 formal tape shape mismatch")
        return {
            "sample_count": 1024,
            "horizon": 48,
            "selected_count": int(len(chunk["selected_indices"])),
            "combined_valid_count": int(combined.sum()),
            "posture_valid_count": int(posture.sum()),
            "legacy_surface_active_candidate_count": int(
                (np.abs(surface).max(axis=1) > 1e-12).sum()
            ),
            "reward_finite": bool(np.isfinite(rewards).all()),
            "chunk_size_bytes": path.stat().st_size,
        }


def verify_complete(payload: dict[str, Any]) -> None:
    """Fail closed when resuming an existing completed capture."""
    if payload.get("status") != "PASS":
        raise RuntimeError("existing bucket004 capture manifest is not PASS")
    for key in ("result", "config", "chunk_manifest"):
        artifact = REPO / payload[key]["path"]
        if not artifact.is_file() or sha256(artifact) != payload[key]["sha256"]:
            raise RuntimeError(f"completed bucket004 capture changed: {artifact}")


def run_capture(*, python_bin: str, gpu_id: int, output_root: Path) -> dict[str, Any]:
    """Run or immutable-resume the single bucket004 formal capture."""
    row = load_row()
    output_dir = output_root / "runs" / f"{CASE_ID}_outdir"
    result_path = output_dir / "trajectory_mjwp_act.npz"
    config_path = output_dir / "config_act.yaml"
    tape_root = output_root / "raw_chunks"
    chunk_path = tape_root / CASE_ID / "chunk_000000.npz"
    chunk_manifest_path = tape_root / CASE_ID / "chunk_manifest.json"
    log_path = output_root / "logs" / f"{CASE_ID}.log"
    manifest_path = output_root / "manifests" / f"{CASE_ID}.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_complete(payload)
        return payload
    if output_dir.exists() or chunk_manifest_path.parent.exists():
        raise RuntimeError("refusing to append to incomplete bucket004 capture")
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
            f"bucket004 capture failed rc={returncode}; inspect {log_path}"
        )
    tape = finalize_cem_query_tape(
        SimpleNamespace(
            query_tape_enabled=True,
            query_tape_output_dir=str(tape_root),
            query_tape_run_id=CASE_ID,
        ),
        provenance={
            "experiment_id": "E187",
            "stage": "A1_S2_BUCKET004_FORMAL_TAPE",
            "case_id": CASE_ID,
            "object_key": "bucket004",
            "budget": {"samples": 1024, "iterations": 32, "seed": 0},
            "record_start_sim_step": RECORD_START_SIM_STEP,
            "recorded_query": "PRODUCTION_FIXED_CONTACT_QUERY",
            "geometry_state_recorded": True,
            "source_override_sha256": row["override_sha256"],
            "source_scene_sha256": row["effective_scene_sha256"],
            "source_grid_manifest_sha256": row["object_distance_manifest_sha256"],
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
        "query_tape_record_start_sim_step": RECORD_START_SIM_STEP,
        "query_tape_record_geometry_state": True,
        "object_distance_backend": "grid_sdf",
        "surface_band_score_mode": "symmetric_abs",
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch or tape.get("status") != "COMPLETE" or tape.get("chunk_count") != 1:
        raise RuntimeError(
            f"bucket004 runtime contract mismatch: {mismatch} tape={tape}"
        )
    summary = summarize_chunk(chunk_path)
    if not summary["reward_finite"] or summary["selected_count"] != 102:
        raise RuntimeError(f"bucket004 capture summary failed: {summary}")
    payload = {
        "schema": "e187_bucket004_formal_tape_v1",
        "experiment_id": "E187",
        "stage": "A1_S2_BUCKET004_FORMAL_TAPE",
        "status": "PASS",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "case_id": CASE_ID,
        "object_key": "bucket004",
        "gpu_id": gpu_id,
        "wall_seconds": wall_seconds,
        "peak_total_gpu_memory_mib": peak_memory_mib,
        "summary": summary,
        "command": command,
        "source": {
            "override_sha256": row["override_sha256"],
            "scene_sha256": row["effective_scene_sha256"],
            "grid_manifest_sha256": row["object_distance_manifest_sha256"],
        },
        "result": {"path": relative(result_path), "sha256": sha256(result_path)},
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "chunk_manifest": {
            "path": relative(chunk_manifest_path),
            "sha256": sha256(chunk_manifest_path),
        },
        "chunk_content_sha256": tape["content_sha256"],
        "log": relative(log_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse the single local capture invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Preflight or run the missing bucket004 formal tape."""
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO / output_root
    if args.gpu_id != 0:
        raise ValueError("local bucket004 formal tape is frozen to GPU0")
    if args.preflight:
        row = load_row()
        command = build_command(
            row,
            python_bin=args.python_bin,
            gpu_id=args.gpu_id,
            output_dir=output_root / "runs" / f"{CASE_ID}_outdir",
            tape_root=output_root / "raw_chunks",
        )
        print(json.dumps({"status": "PASS", "command": command}, indent=2))
        return 0
    payload = run_capture(
        python_bin=args.python_bin, gpu_id=args.gpu_id, output_root=output_root
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
