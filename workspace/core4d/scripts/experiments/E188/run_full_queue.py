#!/usr/bin/env python3
"""Run E188 full-budget canaries, promote them, and complete fixed queues."""

from __future__ import annotations

import argparse
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

from common import (
    CANARY_BY_WORKER,
    PHYSICAL_GPU,
    RESULTS,
    S0,
    WORKERS,
    read_tsv,
    relative,
    repo_path,
    sha256,
    write_json,
)

REPO = Path(__file__).resolve().parents[5]
QUEUE_ROOT_V1 = RESULTS / "s5_full/queue"
QUEUE_ROOT_V2 = RESULTS / "s5_full/queue_speed_rebalanced_v2"
AUTHORITY = S0 / "authority_manifest.tsv"
OVERRIDES = S0 / "overrides_manifest.json"
CANARY_ROOT = RESULTS / "s4_canary"
FULL_ROOT = RESULTS / "s5_full"
CANARY_GATE = CANARY_ROOT / "canary_gate.json"
PROMOTION = FULL_ROOT / "promotion_manifest.json"
PLAN_TIME_PATTERN = re.compile(r"plan time:\s*([0-9.]+)s")
RENDER_MODES = ("inline", "deferred-local")


def expected_render_mode(worker: str) -> str:
    return "inline" if worker == "local-0" else "deferred-local"


def validate_render_mode(worker: str, render_mode: str) -> None:
    expected = expected_render_mode(worker)
    if render_mode != expected:
        raise RuntimeError(
            f"worker render mode differs from approved E188 contract: "
            f"worker={worker} got={render_mode} expected={expected}"
        )


def active_queue_root(root: Path = REPO) -> Path:
    v2 = root / QUEUE_ROOT_V2.relative_to(REPO)
    return v2 if (v2 / "queue_manifest.json").is_file() else root / QUEUE_ROOT_V1.relative_to(REPO)


def active_queue_manifest(root: Path = REPO) -> Path:
    return active_queue_root(root) / "queue_manifest.json"


def queue_payload(root: Path = REPO) -> dict[str, Any]:
    path = active_queue_manifest(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or payload.get("row_count") != 15:
        raise RuntimeError("E188 queue is not frozen 15-row authority")
    return payload


def load_queue(worker: str, root: Path = REPO) -> list[dict[str, str]]:
    if worker not in WORKERS:
        raise ValueError(f"unknown worker: {worker}")
    payload = queue_payload(root)
    path = active_queue_root(root) / f"{worker}.tsv"
    if sha256(path) != payload["queue_tsv_sha256"][worker]:
        raise RuntimeError(f"{worker}: queue TSV SHA changed")
    rows = read_tsv(path)
    expected = [row["case_id"] for row in payload["queues"][worker]]
    if [row["case_id"] for row in rows] != expected:
        raise RuntimeError(f"{worker}: TSV/JSON order mismatch")
    if rows[0]["initial_status"] != "CANARY_NOT_RUN" or any(
        row["initial_status"] != "NOT_RUN" for row in rows[1:]
    ):
        raise RuntimeError(f"{worker}: queue initial states changed")
    return rows


def row_authority(row: dict[str, str], root: Path = REPO) -> tuple[dict[str, Any], dict[str, str]]:
    authority = {
        item["case_id"]: item
        for item in read_tsv(root / AUTHORITY.relative_to(REPO))
    }
    overrides = json.loads(
        (root / OVERRIDES.relative_to(REPO)).read_text(encoding="utf-8")
    )
    override = {item["case_id"]: item for item in overrides["rows"]}[row["case_id"]]
    source = authority[row["case_id"]]
    checks = (
        (override["override_path"], row["override_sha256"], "override"),
        (source["scene_act"], row["scene_sha256"], "scene"),
        (source["trajectory"], row["trajectory_sha256"], "trajectory"),
        (source["contact_mask"], row["contact_mask_sha256"], "contact mask"),
        (source["grid_manifest"], row["grid_manifest_sha256"], "grid manifest"),
    )
    for value, expected, label in checks:
        path = root / value
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{row['case_id']}: {label} authority changed")
    if float(source["new_mass_kg"]) != 5.0 or float(source["inertia_scale"]) != 2.5:
        raise RuntimeError(f"{row['case_id']}: mass authority changed")
    return override, source


def build_command(
    row: dict[str, str], *, python_bin: str, device_id: int, output_dir: Path,
    video: Path, render_mode: str, root: Path = REPO
) -> list[str]:
    if render_mode not in RENDER_MODES:
        raise ValueError(render_mode)
    override, source = row_authority(row, root)
    save_video = render_mode == "inline"
    return [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={override['override_id']}",
        f"task={source['target_task']}",
        "+use_torch_compile=false",
        f"save_video={str(save_video).lower()}",
        "save_info=true",
        "video_camera=auto",
        f"output_dir={output_dir}",
        f"video_output_path={video}",
        "num_samples=1024",
        "max_num_iterations=32",
        "seed=0",
        f"device=cuda:{device_id}",
        "+query_tape_enabled=false",
        "+query_tape_record_geometry_state=false",
    ]


def gpu_memory_mib(physical_gpu: int) -> int | None:
    result = subprocess.run(
        ["nvidia-smi", f"--id={physical_gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def run_process(command: list[str], log: Path, physical_gpu: int) -> tuple[int, float, int | None]:
    started = time.perf_counter()
    peak = gpu_memory_mib(physical_gpu)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("# " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            current = gpu_memory_mib(physical_gpu)
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(0.25)
    return int(process.returncode), time.perf_counter() - started, peak


def numeric_finite(path: Path) -> tuple[bool, int]:
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
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except ValueError:
        duration = 0.0
    return {"readable": result.returncode == 0 and duration > 0, "duration_seconds": duration}


def manifest_render_mode(payload: dict[str, Any]) -> str:
    render_mode = payload.get("render_mode")
    if render_mode is None and payload.get("worker") == "local-0" and isinstance(payload.get("video"), dict):
        # The local canary was already running when the user approved the A100-only
        # deferred-render amendment. Preserve its immutable inline-video manifest.
        return "INLINE_CEM"
    if render_mode not in {"INLINE_CEM", "DEFERRED_LOCAL_RENDER"}:
        raise RuntimeError(f"unknown row render mode: {render_mode}")
    return str(render_mode)


def verify_row_manifest(root: Path, manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != "e188_full_row_v1" or payload.get("status") != "PASS":
        raise RuntimeError(f"invalid E188 row manifest: {manifest}")
    for key in ("result", "config", "log"):
        path = root / payload[key]["path"]
        if not path.is_file() or sha256(path) != payload[key]["sha256"]:
            raise RuntimeError(f"row artifact changed: {path}")
    render_mode = manifest_render_mode(payload)
    video = payload.get("video")
    if render_mode == "INLINE_CEM":
        if not isinstance(video, dict):
            raise RuntimeError(f"inline video metadata missing: {manifest}")
        path = root / video["path"]
        if not path.is_file() or sha256(path) != video["sha256"]:
            raise RuntimeError(f"row artifact changed: {path}")
    elif render_mode == "DEFERRED_LOCAL_RENDER":
        if video is not None or not payload.get("deferred_local_video_required"):
            raise RuntimeError(f"invalid deferred-video contract: {manifest}")
    return payload


def plan_times(log: Path) -> list[float]:
    return [float(value) for value in PLAN_TIME_PATTERN.findall(log.read_text(errors="replace"))]


def run_one(
    row: dict[str, str], *, stage: str, python_bin: str, device_id: int,
    physical_gpu: int, render_mode: str, root: Path = REPO
) -> dict[str, Any]:
    if stage not in {"canary", "full"}:
        raise ValueError(stage)
    base = CANARY_ROOT if stage == "canary" else FULL_ROOT
    row_root = root / base.relative_to(REPO) / "rows" / row["case_id"]
    manifest = row_root / "manifest.json"
    if manifest.is_file():
        return verify_row_manifest(root, manifest)
    if row_root.exists():
        raise RuntimeError(f"refusing to overwrite incomplete row: {row_root}")
    output_dir = row_root / "outdir"
    output_dir.mkdir(parents=True)
    video = row_root / "video.mp4"
    log = row_root / "run.log"
    command = build_command(
        row, python_bin=python_bin, device_id=device_id, output_dir=output_dir,
        video=video, render_mode=render_mode, root=root
    )
    returncode, wall, peak = run_process(command, log, physical_gpu)
    result = output_dir / "trajectory_mjwp_act.npz"
    config_path = output_dir / "config_act.yaml"
    required = (result, config_path, video) if render_mode == "inline" else (result, config_path)
    if returncode != 0 or not all(path.is_file() for path in required):
        raise RuntimeError(f"E188 row failed case={row['case_id']} rc={returncode} log={log}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "num_samples": 1024,
        "max_num_iterations": 32,
        "seed": 0,
        "query_tape_enabled": False,
        "query_tape_record_geometry_state": False,
        "surface_band_score_mode": "distance_continuation",
        "object_distance_manifest": row["grid_manifest"],
        "object_distance_error_bound_m": float(row["epsilon_grid_m"]),
        "scene_name": "scene_act_E188_mass5kg",
        "save_video": render_mode == "inline",
    }
    mismatch = {key: [config.get(key), value] for key, value in expected.items() if config.get(key) != value}
    finite, arrays = numeric_finite(result)
    video_check = video_contract(video) if render_mode == "inline" else {
        "readable": False,
        "duration_seconds": 0.0,
        "status": "DEFERRED_LOCAL_RENDER",
    }
    timings = plan_times(log)
    if mismatch or not finite or (render_mode == "inline" and not video_check["readable"]) or not timings:
        raise RuntimeError(f"runtime contract failed case={row['case_id']} mismatch={mismatch} finite={finite} video={video_check} plan_times={len(timings)}")
    queue_sha = sha256(active_queue_manifest(root))
    payload = {
        "schema": "e188_full_row_v1",
        "experiment_id": "E188",
        "stage": "S4_CANARY" if stage == "canary" else "S5_FULL",
        "status": "PASS",
        "execution_kind": "FULL_BUDGET_CANARY" if stage == "canary" else "FULL_CEM",
        "case_id": row["case_id"],
        "object_key": row["object_key"],
        "worker": row["worker"],
        "queue_position": int(row["queue_position"]),
        "ordinal": int(row["ordinal"]),
        "physical_gpu": physical_gpu,
        "visible_device": device_id,
        "budget": {"samples": 1024, "iterations": 32, "seed": 0},
        "render_mode": "INLINE_CEM" if render_mode == "inline" else "DEFERRED_LOCAL_RENDER",
        "deferred_local_video_required": render_mode == "deferred-local",
        "queue_manifest_sha256": queue_sha,
        "override_sha256": row["override_sha256"],
        "scene_sha256": row["scene_sha256"],
        "trajectory_sha256": row["trajectory_sha256"],
        "contact_mask_sha256": row["contact_mask_sha256"],
        "grid_manifest_sha256": row["grid_manifest_sha256"],
        "wall_seconds": wall,
        "peak_total_gpu_memory_mib": peak,
        "plan_time_count": len(timings),
        "plan_time_median_seconds": median(timings),
        "numeric_array_count": arrays,
        "video_contract": video_check,
        "command": command,
        "result": {"path": relative(result, root), "sha256": sha256(result)},
        "config": {"path": relative(config_path, root), "sha256": sha256(config_path)},
        "video": {"path": relative(video, root), "sha256": sha256(video)} if render_mode == "inline" else None,
        "log": {"path": relative(log, root), "sha256": sha256(log)},
    }
    write_json(manifest, payload, immutable=True)
    return payload


def freeze_canary_gate(root: Path = REPO) -> dict[str, Any]:
    rows = []
    for worker in WORKERS:
        queue = load_queue(worker, root)
        case_id = queue[0]["case_id"]
        manifest = root / CANARY_ROOT.relative_to(REPO) / "rows" / case_id / "manifest.json"
        payload = verify_row_manifest(root, manifest)
        if payload["execution_kind"] != "FULL_BUDGET_CANARY" or payload["worker"] != worker:
            raise RuntimeError(f"invalid canary identity: {case_id}")
        rows.append({"case_id": case_id, "worker": worker, "render_mode": manifest_render_mode(payload), "manifest": {"path": relative(manifest, root), "sha256": sha256(manifest)}})
    gate = {"schema": "e188_canary_gate_v1", "experiment_id": "E188", "status": "PASS", "canary_count": 3, "inline_video_rows": 1, "deferred_local_render_rows": 2, "full_progression_allowed": True, "queue_manifest_sha256": sha256(active_queue_manifest(root)), "canaries": rows}
    write_json(root / CANARY_GATE.relative_to(REPO), gate, immutable=True)
    return gate


def promote_canaries(root: Path = REPO) -> dict[str, Any]:
    gate = json.loads((root / CANARY_GATE.relative_to(REPO)).read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("canary_count") != 3:
        raise RuntimeError("E188 canary gate is not PASS 3/3")
    registrations = []
    for worker in WORKERS:
        row = load_queue(worker, root)[0]
        source_path = root / CANARY_ROOT.relative_to(REPO) / "rows" / row["case_id"] / "manifest.json"
        source = verify_row_manifest(root, source_path)
        promoted = {**source, "stage": "S5_FULL", "execution_kind": "PROMOTED_CANARY", "render_mode": manifest_render_mode(source), "deferred_local_video_required": manifest_render_mode(source) == "DEFERRED_LOCAL_RENDER", "source_canary_manifest": {"path": relative(source_path, root), "sha256": sha256(source_path)}}
        target = root / FULL_ROOT.relative_to(REPO) / "rows" / row["case_id"] / "manifest.json"
        write_json(target, promoted, immutable=True)
        registrations.append({"case_id": row["case_id"], "worker": worker, "manifest": {"path": relative(target, root), "sha256": sha256(target)}})
    payload = {"schema": "e188_canary_promotion_v1", "experiment_id": "E188", "status": "FROZEN", "promoted_rows": 3, "remaining_rows": 12, "canary_gate_sha256": sha256(root / CANARY_GATE.relative_to(REPO)), "registrations": registrations}
    write_json(root / PROMOTION.relative_to(REPO), payload, immutable=True)
    return payload


def verify_promotion(root: Path = REPO) -> dict[str, Any]:
    payload = json.loads((root / PROMOTION.relative_to(REPO)).read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or payload.get("promoted_rows") != 3:
        raise RuntimeError("promotion manifest is not frozen")
    for row in payload["registrations"]:
        manifest = root / row["manifest"]["path"]
        if sha256(manifest) != row["manifest"]["sha256"]:
            raise RuntimeError("promoted manifest SHA changed")
        verify_row_manifest(root, manifest)
    return payload


def run_worker(
    worker: str, phase: str, *, python_bin: str, device_id: int,
    physical_gpu: int, render_mode: str, root: Path = REPO
) -> dict[str, Any]:
    validate_render_mode(worker, render_mode)
    rows = load_queue(worker, root)
    selected = rows[:1] if phase == "canary" else rows[1:]
    if phase == "full":
        verify_promotion(root)
    completed = []
    for row in selected:
        completed.append(run_one(
            row, stage=phase, python_bin=python_bin, device_id=device_id,
            physical_gpu=physical_gpu, render_mode=render_mode, root=root
        )["case_id"])
    state = {"status": "PASS", "worker": worker, "phase": phase, "physical_gpu": physical_gpu, "render_mode": render_mode, "completed_case_ids": completed, "completed_rows": len(completed)}
    state_root = CANARY_ROOT if phase == "canary" else FULL_ROOT
    write_json(root / state_root.relative_to(REPO) / "worker_state" / f"{worker}.json", state)
    return state


def verify_rows(root: Path, stage: str, case_ids: list[str]) -> dict[str, Any]:
    base = CANARY_ROOT if stage == "canary" else FULL_ROOT
    failures = []
    checked = 0
    for case_id in case_ids:
        manifest = root / base.relative_to(REPO) / "rows" / case_id / "manifest.json"
        try:
            payload = verify_row_manifest(root, manifest)
            if payload["case_id"] != case_id:
                raise RuntimeError("case_id mismatch")
            checked += 1
        except Exception as exc:
            failures.append({"case_id": case_id, "error": str(exc)})
    return {"status": "PASS" if checked == len(case_ids) and not failures else "FAIL", "stage": stage, "checked_rows": checked, "failures": failures}


def closure(root: Path = REPO) -> dict[str, Any]:
    cases = [row["case_id"] for worker in WORKERS for row in load_queue(worker, root)]
    result = verify_rows(root, "full", cases)
    result.update({"expected_rows": 15, "unique_rows": len(set(cases))})
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--worker", choices=WORKERS, required=True)
    preflight.add_argument("--phase", choices=("canary", "full"), required=True)
    preflight.add_argument("--python-bin", default=sys.executable)
    preflight.add_argument("--device-id", type=int, default=0)
    preflight.add_argument("--render-mode", choices=RENDER_MODES, required=True)
    run = sub.add_parser("run")
    run.add_argument("--worker", choices=WORKERS, required=True)
    run.add_argument("--phase", choices=("canary", "full"), required=True)
    run.add_argument("--python-bin", default=sys.executable)
    run.add_argument("--device-id", type=int, default=0)
    run.add_argument("--physical-gpu", type=int, required=True)
    run.add_argument("--render-mode", choices=RENDER_MODES, required=True)
    sub.add_parser("freeze-canary-gate")
    sub.add_parser("promote-canaries")
    verify = sub.add_parser("verify-rows")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--stage", choices=("canary", "full"), required=True)
    verify.add_argument("--case-id", action="append", required=True)
    sub.add_parser("closure")
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.mode == "preflight":
        validate_render_mode(args.worker, args.render_mode)
        rows = load_queue(args.worker)
        selected = rows[:1] if args.phase == "canary" else rows[1:]
        payload = {"status": "PREFLIGHT_PASS", "worker": args.worker, "phase": args.phase, "render_mode": args.render_mode, "rows": len(selected), "commands": [build_command(row, python_bin=args.python_bin, device_id=args.device_id, output_dir=(CANARY_ROOT if args.phase == 'canary' else FULL_ROOT) / 'rows' / row['case_id'] / 'outdir', video=(CANARY_ROOT if args.phase == 'canary' else FULL_ROOT) / 'rows' / row['case_id'] / 'video.mp4', render_mode=args.render_mode) for row in selected]}
    elif args.mode == "run":
        if args.physical_gpu != PHYSICAL_GPU[args.worker]:
            raise RuntimeError("worker physical GPU differs from plan212")
        payload = run_worker(args.worker, args.phase, python_bin=args.python_bin, device_id=args.device_id, physical_gpu=args.physical_gpu, render_mode=args.render_mode)
    elif args.mode == "freeze-canary-gate":
        payload = freeze_canary_gate()
    elif args.mode == "promote-canaries":
        payload = promote_canaries()
    elif args.mode == "verify-rows":
        payload = verify_rows(args.root.resolve(), args.stage, args.case_id)
    else:
        payload = closure()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"PASS", "PREFLIGHT_PASS", "FROZEN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
