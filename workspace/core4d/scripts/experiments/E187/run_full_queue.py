#!/usr/bin/env python3
"""Run immutable E187 Full CEM queues and atomically promote A3 canaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
A1_LOCK = RESULTS / "s2_canonical_grid_sdf/reward_grid_lock.json"
A2_LOCK = RESULTS / "s3_prg_audit/production_integration_lock.json"
A3_LOCK = RESULTS / "s4_canary/canary_gate_lock.json"
OVERRIDES = RESULTS / "s3_prg_audit/production_overrides.json"
SCENES = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
QUEUE_ROOT = RESULTS / "s5_full/queue"
QUEUE_MANIFEST = QUEUE_ROOT / "queue_manifest.json"
FULL_ROOT = RESULTS / "s5_full"
PROMOTION_MANIFEST = FULL_ROOT / "promotion_manifest.json"
WORKERS = ("local-0", "remote-0", "remote-1")
A1_LOCK_SHA256 = "2cc949e19cb313e58883fd5d0446872220188ea9ecce06cb23d39e2a6b42194f"
A2_LOCK_SHA256 = "400c98b422ac4458eccffb589eecc1d776fe5cd3968eb855cd89a485bb7d5441"
A3_LOCK_SHA256 = "65d53086d7e82b6f4faa384788509055d505abc1302709ab2f9003484eacb18a"
QUEUE_MANIFEST_SHA256 = (
    "836b5388420f968e9c88968349a87799ae1ea3bd43d6276e631847c7956dcd92"
)
QUEUE_TSV_SHA256 = {
    "local-0": "3bbd3e6f0ae9727dfc5e664636579688f7f628898db8ad6f82ee8ad9396a1c9d",
    "remote-0": "15b97a6eecd7d421a9b82c9c62fbd4974d173a518684b190db14ad8176814e44",
    "remote-1": "ce132777efb04577ec960588e7196148b247d7a3e1d6c7e761c7d44437d63afd",
}
PLAN_TIME_PATTERN = re.compile(r"plan time:\s*([0-9.]+)s")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path = REPO) -> str:
    """Serialize a verified path relative to the active snapshot root."""
    return path.absolute().relative_to(root.absolute()).as_posix()


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one tab-separated authority file."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a JSON artifact or verify the identical frozen value."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to replace frozen artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def frozen_authority(root: Path = REPO) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the fixed A1/A2/A3 and queue identities."""
    paths = {
        "A1": root / A1_LOCK.relative_to(REPO),
        "A2": root / A2_LOCK.relative_to(REPO),
        "A3": root / A3_LOCK.relative_to(REPO),
        "queue": root / QUEUE_MANIFEST.relative_to(REPO),
    }
    expected = {
        "A1": A1_LOCK_SHA256,
        "A2": A2_LOCK_SHA256,
        "A3": A3_LOCK_SHA256,
        "queue": QUEUE_MANIFEST_SHA256,
    }
    for label, path in paths.items():
        if sha256(path) != expected[label]:
            raise RuntimeError(f"{label} frozen authority SHA changed: {path}")
    a3 = json.loads(paths["A3"].read_text(encoding="utf-8"))
    queue = json.loads(paths["queue"].read_text(encoding="utf-8"))
    if (
        a3.get("status") != "FROZEN"
        or not a3.get("c8_pass")
        or a3.get("c9_technical_pass") is not False
        or not a3.get("c9_progression_allowed")
        or a3.get("c9_progression_authority") != "USER_WAIVED"
        or a3.get("full_cem_started_rows") != 0
    ):
        raise RuntimeError("A3 does not authorize USER_WAIVED Full progression")
    if (
        queue.get("status") != "FROZEN"
        or queue.get("row_count") != 22
        or queue.get("promoted_canary_count") != 3
        or queue.get("not_run_count") != 19
        or queue.get("full_cem_started_rows") != 0
        or queue.get("a3_lock", {}).get("sha256") != A3_LOCK_SHA256
    ):
        raise RuntimeError("Full queue authority is not frozen keep22/3+19")
    return a3, queue


def load_worker_queue(worker: str, root: Path = REPO) -> list[dict[str, str]]:
    """Load one worker TSV and cross-check it against the frozen JSON queue."""
    if worker not in WORKERS:
        raise ValueError(f"unknown E187 worker: {worker}")
    _, queue = frozen_authority(root)
    path = root / QUEUE_ROOT.relative_to(REPO) / f"{worker}.tsv"
    if sha256(path) != QUEUE_TSV_SHA256[worker]:
        raise RuntimeError(f"{worker} queue TSV SHA changed")
    rows = read_tsv(path)
    manifest_rows = queue["queues"][worker]
    if [row["case_id"] for row in rows] != [row["case_id"] for row in manifest_rows]:
        raise RuntimeError(f"{worker} TSV row order differs from queue manifest")
    if [int(row["queue_position"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"{worker} queue positions are not contiguous")
    if rows[0]["initial_status"] != "PROMOTED_CANARY_PENDING_ATOMIC_REGISTRATION":
        raise RuntimeError(f"{worker} queue does not begin with its frozen canary")
    if any(row["initial_status"] != "NOT_RUN" for row in rows[1:]):
        raise RuntimeError(f"{worker} non-canary rows are not all NOT_RUN")
    return rows


def row_authority(
    row: dict[str, str], root: Path = REPO
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify one row against production override and scene authorities."""
    overrides_path = root / OVERRIDES.relative_to(REPO)
    scenes_path = root / SCENES.relative_to(REPO)
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    override = {item["case_id"]: item for item in overrides["rows"]}[row["case_id"]]
    scene = {item["case_id"]: item for item in read_tsv(scenes_path)}[row["case_id"]]
    checks = (
        (root / override["override_path"], row["override_sha256"], "override"),
        (root / scene["scene_act"], row["scene_sha256"], "scene"),
        (root / scene["trajectory"], row["trajectory_sha256"], "trajectory"),
        (root / scene["contact_mask"], scene["contact_mask_sha256"], "contact"),
        (
            root / override["grid_manifest"],
            row["grid_manifest_sha256"],
            "grid manifest",
        ),
    )
    for path, expected, label in checks:
        if sha256(path) != expected:
            raise RuntimeError(f"{row['case_id']}: {label} SHA changed")
    if (
        override["override_id"] != row["override_id"]
        or scene["target_task"] != row["target_task"]
        or override["grid_manifest"] != row["grid_manifest"]
    ):
        raise RuntimeError(f"{row['case_id']}: queue runtime authority changed")
    return override, scene


def build_command(
    row: dict[str, str],
    *,
    python_bin: str,
    gpu_id: int,
    output_dir: Path,
    video_path: Path,
    root: Path = REPO,
) -> list[str]:
    """Build one exact Full-budget, video-on, recorder-off command."""
    override, scene = row_authority(row, root)
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
    """Passively read total memory use for one visible GPU."""
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
    """Run one row while passively observing total GPU memory."""
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
    """Verify a non-empty readable video with ffprobe."""
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


def verify_row_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify one completed Full row without trusting external paths."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("schema") != "e187_full_row_v1":
        raise RuntimeError(f"row manifest is not E187 Full PASS: {manifest_path}")
    for key in ("result", "config", "video", "log"):
        artifact = (root / payload[key]["path"]).absolute()
        try:
            artifact.relative_to(root.absolute())
        except ValueError as exc:
            raise RuntimeError(f"row artifact escapes root: {artifact}") from exc
        if not artifact.is_file() or sha256(artifact) != payload[key]["sha256"]:
            raise RuntimeError(f"completed Full artifact changed: {artifact}")
    return payload


def promotion_payload(
    row: dict[str, str], canary: dict[str, Any], root: Path = REPO
) -> dict[str, Any]:
    """Build one Full row manifest that references an immutable A3 canary."""
    source_manifest = root / canary["manifest"]["path"]
    if sha256(source_manifest) != canary["manifest"]["sha256"]:
        raise RuntimeError(f"canary manifest changed: {row['case_id']}")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "PASS" or not source.get("full_promotable"):
        raise RuntimeError(f"canary is not promotable: {row['case_id']}")
    if (
        source.get("case_id") != row["case_id"]
        or source.get("override_sha256") != row["override_sha256"]
        or source.get("scene_sha256") != row["scene_sha256"]
        or source.get("grid_manifest_sha256") != row["grid_manifest_sha256"]
    ):
        raise RuntimeError(f"canary authority differs from Full row: {row['case_id']}")
    artifacts = canary["artifacts"]
    for key in ("result", "config", "video", "log"):
        path = root / artifacts[key]["path"]
        if sha256(path) != artifacts[key]["sha256"]:
            raise RuntimeError(f"canary {key} changed: {row['case_id']}")
    return {
        "schema": "e187_full_row_v1",
        "experiment_id": "E187",
        "stage": "A4_S5_KEEP22_FULL_CEM",
        "status": "PASS",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "c9_technical_status": "FAIL",
        "c9_progression_authority": "USER_WAIVED",
        "execution_kind": "PROMOTED_CANARY",
        "case_id": row["case_id"],
        "worker": row["worker"],
        "queue_position": int(row["queue_position"]),
        "ordinal": int(row["ordinal"]),
        "budget": {"samples": 1024, "iterations": 32, "seed": 0},
        "a3_lock_sha256": A3_LOCK_SHA256,
        "queue_manifest_sha256": QUEUE_MANIFEST_SHA256,
        "source_canary_manifest": canary["manifest"],
        "wall_seconds": source["wall_seconds"],
        "peak_total_gpu_memory_mib": source["peak_total_gpu_memory_mib"],
        "plan_time_count": source["plan_time_count"],
        "plan_time_median_seconds": source["plan_time_median_seconds"],
        "numeric_array_count": source["numeric_array_count"],
        "video_contract": source["video_contract"],
        **artifacts,
    }


def register_promotions(root: Path = REPO) -> dict[str, Any]:
    """Register all three canaries and commit the 0-to-3 transition last."""
    a3, _ = frozen_authority(root)
    canary_by_case = {row["case_id"]: row for row in a3["canaries"]}
    registrations = []
    for worker in WORKERS:
        row = load_worker_queue(worker, root)[0]
        canary = canary_by_case[row["case_id"]]
        payload = promotion_payload(row, canary, root)
        manifest = (
            root
            / FULL_ROOT.relative_to(REPO)
            / "rows"
            / row["case_id"]
            / "manifest.json"
        )
        immutable_json(manifest, payload)
        registrations.append(
            {
                "case_id": row["case_id"],
                "worker": worker,
                "full_row_manifest": {
                    "path": relative(manifest, root),
                    "sha256": sha256(manifest),
                },
                "source_canary_manifest": canary["manifest"],
            }
        )
    payload = {
        "schema": "e187_full_canary_promotion_v1",
        "experiment_id": "E187",
        "stage": "A4_S5_KEEP22_FULL_CEM",
        "status": "FROZEN",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "c9_technical_status": "FAIL",
        "c9_progression_authority": "USER_WAIVED",
        "a3_lock_sha256": A3_LOCK_SHA256,
        "queue_manifest_sha256": QUEUE_MANIFEST_SHA256,
        "promoted_rows": 3,
        "full_cem_started_rows_before": 0,
        "full_cem_started_rows_after": 3,
        "remaining_not_run_rows": 19,
        "registrations": registrations,
    }
    immutable_json(root / PROMOTION_MANIFEST.relative_to(REPO), payload)
    return payload


def verify_promotions(root: Path = REPO) -> dict[str, Any]:
    """Verify the committed three-row promotion transaction."""
    path = root / PROMOTION_MANIFEST.relative_to(REPO)
    if not path.is_file():
        raise RuntimeError("Full canary promotion manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "FROZEN"
        or payload.get("full_cem_started_rows_after") != 3
        or payload.get("remaining_not_run_rows") != 19
        or payload.get("a3_lock_sha256") != A3_LOCK_SHA256
        or payload.get("queue_manifest_sha256") != QUEUE_MANIFEST_SHA256
    ):
        raise RuntimeError("Full canary promotion transaction is invalid")
    for registration in payload["registrations"]:
        manifest = root / registration["full_row_manifest"]["path"]
        if sha256(manifest) != registration["full_row_manifest"]["sha256"]:
            raise RuntimeError("promoted Full row manifest SHA changed")
        verify_row_manifest(root, manifest)
    return payload


def plan_times(log: Path) -> list[float]:
    """Extract all per-tick plan times from a row log."""
    return [
        float(value)
        for value in PLAN_TIME_PATTERN.findall(log.read_text(errors="replace"))
    ]


def run_one(
    row: dict[str, str], *, python_bin: str, gpu_id: int, output_root: Path
) -> dict[str, Any]:
    """Run or immutable-resume one non-canary Full row."""
    override, scene = row_authority(row)
    row_root = output_root / "rows" / row["case_id"]
    output_dir = row_root / "outdir"
    video = row_root / "video.mp4"
    log = row_root / "run.log"
    manifest = row_root / "manifest.json"
    result = output_dir / "trajectory_mjwp_act.npz"
    config_path = output_dir / "config_act.yaml"
    if manifest.is_file():
        frozen = verify_row_manifest(REPO, manifest)
        if frozen.get("worker") != row["worker"]:
            raise RuntimeError(f"completed row worker changed: {row['case_id']}")
        return frozen
    if row_root.exists():
        raise RuntimeError(
            f"refusing to overwrite incomplete Full row: {row['case_id']}"
        )
    output_dir.mkdir(parents=True)
    command = build_command(
        row,
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
        raise RuntimeError(
            f"Full row failed case={row['case_id']} rc={returncode} log={log}"
        )
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
            f"Full runtime contract failed mismatch={mismatch} finite={finite} "
            f"video={video_check} plan_times={len(timings)}"
        )
    payload = {
        "schema": "e187_full_row_v1",
        "experiment_id": "E187",
        "stage": "A4_S5_KEEP22_FULL_CEM",
        "status": "PASS",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "c9_technical_status": "FAIL",
        "c9_progression_authority": "USER_WAIVED",
        "execution_kind": "FULL_CEM",
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "worker": row["worker"],
        "queue_position": int(row["queue_position"]),
        "ordinal": int(row["ordinal"]),
        "gpu_id": gpu_id,
        "budget": {"samples": 1024, "iterations": 32, "seed": 0},
        "a3_lock_sha256": A3_LOCK_SHA256,
        "queue_manifest_sha256": QUEUE_MANIFEST_SHA256,
        "override_sha256": row["override_sha256"],
        "scene_sha256": row["scene_sha256"],
        "grid_manifest_sha256": row["grid_manifest_sha256"],
        "wall_seconds": wall_seconds,
        "peak_total_gpu_memory_mib": peak_memory,
        "plan_time_count": len(timings),
        "plan_time_median_seconds": median(timings),
        "numeric_array_count": checked_arrays,
        "video_contract": video_check,
        "command": command,
        "result": {"path": relative(result), "sha256": sha256(result)},
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "video": {"path": relative(video), "sha256": sha256(video)},
        "log": {"path": relative(log), "sha256": sha256(log)},
    }
    immutable_json(manifest, payload)
    return payload


def write_worker_state(output_root: Path, worker: str, payload: dict[str, Any]) -> None:
    """Atomically update the bounded mutable worker status file."""
    path = output_root / "worker_state" / f"{worker}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_worker(
    worker: str, *, python_bin: str, gpu_id: int, output_root: Path
) -> dict[str, Any]:
    """Run one frozen worker queue sequentially after promotion verification."""
    verify_promotions()
    rows = load_worker_queue(worker)
    promoted = verify_row_manifest(
        REPO, output_root / "rows" / rows[0]["case_id"] / "manifest.json"
    )
    completed = [promoted["case_id"]]
    for row in rows[1:]:
        write_worker_state(
            output_root,
            worker,
            {
                "status": "RUNNING",
                "worker": worker,
                "gpu_id": gpu_id,
                "current_case_id": row["case_id"],
                "completed_case_ids": completed,
                "queue_manifest_sha256": QUEUE_MANIFEST_SHA256,
            },
        )
        completed.append(
            run_one(
                row,
                python_bin=python_bin,
                gpu_id=gpu_id,
                output_root=output_root,
            )["case_id"]
        )
    payload = {
        "status": "PASS",
        "worker": worker,
        "gpu_id": gpu_id,
        "completed_case_ids": completed,
        "completed_rows": len(completed),
        "queue_manifest_sha256": QUEUE_MANIFEST_SHA256,
    }
    write_worker_state(output_root, worker, payload)
    return payload


def verify_rows(root: Path, case_ids: list[str]) -> dict[str, Any]:
    """Verify selected Full row manifests and all referenced artifacts."""
    failures = []
    checked = 0
    for case_id in case_ids:
        manifest = (
            root / FULL_ROOT.relative_to(REPO) / "rows" / case_id / "manifest.json"
        )
        try:
            payload = verify_row_manifest(root, manifest)
            if payload.get("case_id") != case_id:
                raise RuntimeError("case_id mismatch")
            checked += 1
        except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
            failures.append({"case_id": case_id, "error": str(exc)})
    return {
        "status": "PASS" if checked == len(case_ids) and not failures else "FAIL",
        "root": str(root),
        "case_ids": case_ids,
        "checked_rows": checked,
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--worker", required=True, choices=WORKERS)
    preflight.add_argument("--gpu-id", type=int, default=0)
    preflight.add_argument("--python-bin", default=sys.executable)
    sub.add_parser("register-promotions")
    run = sub.add_parser("run")
    run.add_argument("--worker", required=True, choices=WORKERS)
    run.add_argument("--gpu-id", type=int, default=0)
    run.add_argument("--python-bin", default=sys.executable)
    run.add_argument("--output-root", type=Path, default=FULL_ROOT)
    verify = sub.add_parser("verify-rows")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--case-id", action="append", required=True)
    return root


def main() -> int:
    """Preflight, promote, run, or verify E187 Full rows."""
    args = parser().parse_args()
    if args.mode == "register-promotions":
        payload = register_promotions()
    elif args.mode == "preflight":
        rows = load_worker_queue(args.worker)
        commands = [
            build_command(
                row,
                python_bin=args.python_bin,
                gpu_id=args.gpu_id,
                output_dir=FULL_ROOT / "rows" / row["case_id"] / "outdir",
                video_path=FULL_ROOT / "rows" / row["case_id"] / "video.mp4",
            )
            for row in rows[1:]
        ]
        payload = {
            "status": "PREFLIGHT_PASS",
            "worker": args.worker,
            "promoted_case_id": rows[0]["case_id"],
            "not_run_rows": len(rows) - 1,
            "commands": commands,
        }
    elif args.mode == "run":
        output_root = (
            args.output_root
            if args.output_root.is_absolute()
            else REPO / args.output_root
        )
        payload = run_worker(
            args.worker,
            python_bin=args.python_bin,
            gpu_id=args.gpu_id,
            output_root=output_root,
        )
    else:
        payload = verify_rows(args.root, args.case_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"PASS", "FROZEN", "PREFLIGHT_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
