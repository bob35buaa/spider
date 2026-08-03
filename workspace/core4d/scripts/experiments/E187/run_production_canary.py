#!/usr/bin/env python3
"""Run or preflight one immutable E187 production canary row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E187"
OVERRIDES = RESULTS / "s3_prg_audit/production_overrides.json"
A2_LOCK = RESULTS / "s3_prg_audit/production_integration_lock.json"
SCENES = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
E178_MANIFEST = (
    REPO
    / "workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv"
)
DEFAULT_OUTPUT = RESULTS / "s4_canary"
CANARY_ASSIGNMENT = {
    "bucket003_20231018_003_p1": "local-0",
    "bucket004_20231002_021_p1": "remote-0",
    "bucket007_20231020_055_p1": "remote-1",
}
PLAN_TIME_PATTERN = re.compile(r"plan time:\s*([0-9.]+)s")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Serialize a path relative to the current snapshot root."""
    return path.absolute().relative_to(REPO.absolute()).as_posix()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated manifest."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def authority(case_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one canary row and fail closed on A2 plus input SHA."""
    if case_id not in CANARY_ASSIGNMENT:
        raise ValueError(f"case is not an A3 representative: {case_id}")
    lock = json.loads(A2_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN" or lock.get("full_cem_started_rows") != 0:
        raise RuntimeError("A2 lock is not frozen before canary")
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    if lock["production_overrides"]["sha256"] != sha256(OVERRIDES):
        raise RuntimeError("production override manifest changed after A2")
    override_by_case = {row["case_id"]: row for row in overrides["rows"]}
    scene_by_case = {row["case_id"]: row for row in read_tsv(SCENES)}
    override = override_by_case[case_id]
    scene = scene_by_case[case_id]
    for value, expected, label in (
        (REPO / override["override_path"], override["override_sha256"], "override"),
        (REPO / scene["scene_act"], scene["effective_scene_sha256"], "scene"),
        (REPO / scene["trajectory"], scene["trajectory_sha256"], "trajectory"),
        (REPO / scene["contact_mask"], scene["contact_mask_sha256"], "contact"),
        (
            REPO / override["grid_manifest"],
            override["grid_manifest_sha256"],
            "grid manifest",
        ),
    ):
        if sha256(value) != expected:
            raise RuntimeError(f"{case_id}: {label} SHA changed")
    return override, scene


def build_command(
    case_id: str,
    *,
    python_bin: str,
    gpu_id: int,
    output_dir: Path,
    video_path: Path,
) -> list[str]:
    """Build one Full-budget, recorder-off production command."""
    override, scene = authority(case_id)
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={override['override_id']}",
        f"task={scene['target_task']}",
        "+use_torch_compile=false",
        "save_video=true",
        "save_info=true",
        "video_camera=auto",
        f"output_dir={output_dir}",
        f"video_output_path={video_path}",
        "num_samples=1024",
        "max_num_iterations=32",
        "seed=0",
        f"device=cuda:{gpu_id}",
        "+query_tape_enabled=false",
        "+query_tape_record_geometry_state=false",
    ]


def gpu_memory_mib(gpu_id: int) -> int | None:
    """Passively read total device memory use."""
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
    command: list[str], log: Path, gpu_id: int
) -> tuple[int, float, int | None]:
    """Run one canary while only observing total GPU memory."""
    started = time.perf_counter()
    peak = gpu_memory_mib(gpu_id)
    with log.open("w", encoding="utf-8") as stream:
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


def numeric_finite(path: Path) -> tuple[bool, int]:
    """Verify all numeric result arrays are finite."""
    checked = 0
    with np.load(path, allow_pickle=True) as payload:
        for key in payload.files:
            values = np.asarray(payload[key])
            if np.issubdtype(values.dtype, np.number):
                checked += 1
                if not np.isfinite(values).all():
                    return False, checked
    return True, checked


def video_contract(path: Path) -> dict[str, Any]:
    """Use ffprobe to verify a non-empty readable video."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
    return {
        "readable": probe.returncode == 0 and duration > 0.0,
        "duration_seconds": duration,
    }


def plan_times(log: Path) -> list[float]:
    """Extract all per-tick planning times from a run log."""
    return [
        float(value)
        for value in PLAN_TIME_PATTERN.findall(log.read_text(errors="replace"))
    ]


def run_one(
    case_id: str, *, python_bin: str, gpu_id: int, output_root: Path
) -> dict[str, Any]:
    """Run or immutable-resume one E187 canary."""
    override, scene = authority(case_id)
    worker = CANARY_ASSIGNMENT[case_id]
    row_root = output_root / "rows" / case_id
    output_dir = row_root / "outdir"
    video = row_root / "video.mp4"
    log = row_root / "run.log"
    manifest = row_root / "manifest.json"
    result = output_dir / "trajectory_mjwp_act.npz"
    config_path = output_dir / "config_act.yaml"
    if manifest.is_file():
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        if frozen.get("status") == "PASS":
            for item in ("result", "config", "video", "log"):
                artifact = REPO / frozen[item]["path"]
                if sha256(artifact) != frozen[item]["sha256"]:
                    raise RuntimeError(f"complete canary artifact changed: {artifact}")
            return frozen
    if row_root.exists():
        raise RuntimeError(f"refusing to overwrite incomplete canary: {case_id}")
    output_dir.mkdir(parents=True)
    command = build_command(
        case_id,
        python_bin=python_bin,
        gpu_id=gpu_id,
        output_dir=output_dir,
        video_path=video,
    )
    returncode, wall_seconds, peak_memory = run_process(command, log, gpu_id)
    if (
        returncode != 0
        or not result.is_file()
        or not config_path.is_file()
        or not video.is_file()
    ):
        raise RuntimeError(f"canary failed case={case_id} rc={returncode} log={log}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "num_samples": 1024,
        "max_num_iterations": 32,
        "seed": 0,
        "query_tape_enabled": False,
        "query_tape_record_geometry_state": False,
        "surface_band_score_mode": "distance_continuation",
        "object_distance_manifest": override["grid_manifest"],
        "object_distance_error_bound_m": float(override["epsilon_grid_m"]),
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    finite, checked_arrays = numeric_finite(result)
    video_check = video_contract(video)
    timings = plan_times(log)
    if mismatch or not finite or not video_check["readable"] or not timings:
        raise RuntimeError(
            f"canary runtime contract failed mismatch={mismatch} finite={finite} "
            f"video={video_check} plan_times={len(timings)}"
        )
    payload = {
        "schema": "e187_production_canary_v1",
        "experiment_id": "E187",
        "stage": "A3_S4_PRODUCTION_CANARY",
        "status": "PASS",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "case_id": case_id,
        "object_key": scene["object_key"],
        "worker": worker,
        "gpu_id": gpu_id,
        "budget": {"samples": 1024, "iterations": 32, "seed": 0},
        "a2_lock_sha256": sha256(A2_LOCK),
        "override_sha256": override["override_sha256"],
        "scene_sha256": scene["effective_scene_sha256"],
        "grid_manifest_sha256": override["grid_manifest_sha256"],
        "wall_seconds": wall_seconds,
        "peak_total_gpu_memory_mib": peak_memory,
        "plan_time_count": len(timings),
        "plan_time_median_seconds": median(timings),
        "numeric_array_count": checked_arrays,
        "video_contract": video_check,
        "full_promotable": True,
        "command": command,
        "result": {"path": relative(result), "sha256": sha256(result)},
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "video": {"path": relative(video), "sha256": sha256(video)},
        "log": {"path": relative(log), "sha256": sha256(log)},
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse one canary invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--case-id", required=True, choices=tuple(CANARY_ASSIGNMENT))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Preflight or run one production canary."""
    args = parse_args()
    root = (
        args.output_root if args.output_root.is_absolute() else REPO / args.output_root
    )
    if args.mode == "preflight":
        command = build_command(
            args.case_id,
            python_bin=args.python_bin,
            gpu_id=args.gpu_id,
            output_dir=root / "rows" / args.case_id / "outdir",
            video_path=root / "rows" / args.case_id / "video.mp4",
        )
        print(json.dumps({"status": "PREFLIGHT_PASS", "command": command}, indent=2))
        return 0
    print(
        json.dumps(
            run_one(
                args.case_id,
                python_bin=args.python_bin,
                gpu_id=args.gpu_id,
                output_root=root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
