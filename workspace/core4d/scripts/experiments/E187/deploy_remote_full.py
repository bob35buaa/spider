#!/usr/bin/env python3
"""Freeze, deploy, and verify the immutable E187 Full Ada snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
RESULTS = REPO / "workspace/core4d/results/E187"
DEPLOY_ROOT = RESULTS / "s5_full/deployment"
SOURCE_MANIFEST = DEPLOY_ROOT / "source_snapshot_manifest.json"
SOURCE_FILE_LIST = DEPLOY_ROOT / "source_snapshot_files.txt"
DEPLOYMENT_MANIFEST = DEPLOY_ROOT / "remote_deployment_manifest.json"
A1_LOCK = RESULTS / "s2_canonical_grid_sdf/reward_grid_lock.json"
A2_LOCK = RESULTS / "s3_prg_audit/production_integration_lock.json"
A3_LOCK = RESULTS / "s4_canary/canary_gate_lock.json"
OVERRIDES = RESULTS / "s3_prg_audit/production_overrides.json"
QUEUE_ROOT = RESULTS / "s5_full/queue"
QUEUE_MANIFEST = QUEUE_ROOT / "queue_manifest.json"
PROMOTION_MANIFEST = RESULTS / "s5_full/promotion_manifest.json"
SCENES = (
    REPO
    / "workspace/core4d/results/E186/s2_compound_physics/compound_scene_manifest.tsv"
)
REMOTE_PARENT = "/home/xiayb/pHRI_workspace/e187_runs"
REMOTE_PYTHON = "/home/xiayb/pHRI_workspace/spider/.venv/bin/python"
REMOTE_MANIFEST_RELATIVE = (
    "workspace/core4d/results/E187/s5_full/deployment/source_snapshot_manifest.json"
)
A1_LOCK_SHA256 = "2cc949e19cb313e58883fd5d0446872220188ea9ecce06cb23d39e2a6b42194f"
A2_LOCK_SHA256 = "400c98b422ac4458eccffb589eecc1d776fe5cd3968eb855cd89a485bb7d5441"
A3_LOCK_SHA256 = "65d53086d7e82b6f4faa384788509055d505abc1302709ab2f9003484eacb18a"
QUEUE_MANIFEST_SHA256 = (
    "836b5388420f968e9c88968349a87799ae1ea3bd43d6276e631847c7956dcd92"
)
SOURCE_PREFIXES = ("spider", "examples", "src")
ROOT_SOURCE_FILES = (
    ".python-version",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "uv.lock",
)
E187_RUNTIME_SCRIPTS = (
    "workspace/core4d/scripts/experiments/E187/deploy_remote_full.py",
    "workspace/core4d/scripts/experiments/E187/run_full_queue.py",
    "workspace/core4d/scripts/experiments/E187/test_full_queue_runner.py",
    "workspace/core4d/scripts/experiments/E187/test_remote_full_deployment.py",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Return SHA-256 for an in-memory payload."""
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable JSON artifact or verify the identical value."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to replace frozen manifest: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def command(
    args: Sequence[str], *, cwd: Path = REPO, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one bounded local command and capture output."""
    return subprocess.run(
        list(args), cwd=cwd, check=check, text=True, capture_output=True
    )


def ssh(
    remote_host: str, script: str, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one bounded non-interactive remote command."""
    return command(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            remote_host,
            script,
        ),
        check=check,
    )


def relative_file(path: Path) -> str:
    """Return a repository-relative regular-file path."""
    absolute = (path if path.is_absolute() else REPO / path).absolute()
    try:
        relative = absolute.relative_to(REPO.absolute()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {path}") from exc
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return relative


def tree_files(path: Path) -> list[str]:
    """List regular files under one repository-owned tree."""
    root = path if path.is_absolute() else REPO / path
    if not root.is_dir():
        raise FileNotFoundError(root)
    return [
        relative_file(candidate)
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file() and "__pycache__" not in candidate.parts
    ]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read one tab-separated authority file."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def source_files() -> set[str]:
    """Discover bounded runtime source without including the dirty worktree broadly."""
    result = command(
        (
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_PREFIXES,
        )
    )
    files = {
        relative_file(REPO / value)
        for value in result.stdout.splitlines()
        if value and (REPO / value).is_file() and "__pycache__" not in Path(value).parts
    }
    files.update(relative_file(REPO / value) for value in ROOT_SOURCE_FILES)
    files.update(relative_file(REPO / value) for value in E187_RUNTIME_SCRIPTS)
    return files


def runtime_files() -> tuple[set[str], list[dict[str, Any]]]:
    """Resolve all 22 Full rows, locks, queues, grids, and canary evidence."""
    for path, expected, label in (
        (A1_LOCK, A1_LOCK_SHA256, "A1"),
        (A2_LOCK, A2_LOCK_SHA256, "A2"),
        (A3_LOCK, A3_LOCK_SHA256, "A3"),
        (QUEUE_MANIFEST, QUEUE_MANIFEST_SHA256, "queue"),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"{label} frozen authority SHA changed")
    a3 = json.loads(A3_LOCK.read_text(encoding="utf-8"))
    queue = json.loads(QUEUE_MANIFEST.read_text(encoding="utf-8"))
    promotion = json.loads(PROMOTION_MANIFEST.read_text(encoding="utf-8"))
    if (
        a3.get("status") != "FROZEN"
        or not a3.get("c8_pass")
        or a3.get("c9_technical_pass") is not False
        or not a3.get("c9_progression_allowed")
    ):
        raise RuntimeError("A3 does not authorize Full deployment")
    if queue.get("row_count") != 22 or queue.get("not_run_count") != 19:
        raise RuntimeError("queue is not frozen keep22")
    if (
        promotion.get("status") != "FROZEN"
        or promotion.get("full_cem_started_rows_after") != 3
        or promotion.get("remaining_not_run_rows") != 19
        or len(promotion.get("registrations", [])) != 3
    ):
        raise RuntimeError("three canaries were not atomically registered")
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    override_by_case = {row["case_id"]: row for row in overrides["rows"]}
    scene_by_case = {row["case_id"]: row for row in read_tsv(SCENES)}
    files = {
        relative_file(path)
        for path in (
            A1_LOCK,
            A2_LOCK,
            A3_LOCK,
            OVERRIDES,
            SCENES,
            PROMOTION_MANIFEST,
        )
    }
    files.update(tree_files(QUEUE_ROOT))
    files.update(tree_files(REPO / "example_datasets/processed/core4d/assets"))
    for canary in a3["canaries"]:
        manifest = REPO / canary["manifest"]["path"]
        if sha256(manifest) != canary["manifest"]["sha256"]:
            raise RuntimeError(f"canary manifest changed: {canary['case_id']}")
        files.update(tree_files(manifest.parent))
    files.update(tree_files(RESULTS / "s5_full/rows"))
    records = []
    for worker in queue["worker_order"]:
        for row in queue["queues"][worker]:
            case_id = row["case_id"]
            override = override_by_case[case_id]
            scene = scene_by_case[case_id]
            for value, expected, label in (
                (override["override_path"], row["override_sha256"], "override"),
                (scene["scene_act"], row["scene_sha256"], "scene"),
                (scene["trajectory"], row["trajectory_sha256"], "trajectory"),
                (scene["contact_mask"], scene["contact_mask_sha256"], "contact"),
                (
                    override["grid_manifest"],
                    row["grid_manifest_sha256"],
                    "grid manifest",
                ),
            ):
                path = REPO / value
                if sha256(path) != expected:
                    raise RuntimeError(f"{case_id}: {label} SHA changed")
                files.add(relative_file(path))
            task_dir = (REPO / scene["target_scene"]).parent
            grid_dir = (REPO / override["grid_manifest"]).parent
            files.update(tree_files(task_dir))
            files.update(tree_files(grid_dir))
            records.append(
                {
                    "case_id": case_id,
                    "worker": worker,
                    "queue_position": int(row["queue_position"]),
                    "promoted_canary": bool(row["promoted_canary"]),
                    "object_key": row["object_key"],
                    "task_dir": task_dir.relative_to(REPO).as_posix(),
                    "grid_manifest": override["grid_manifest"],
                    "grid_manifest_sha256": row["grid_manifest_sha256"],
                    "override_sha256": row["override_sha256"],
                    "scene_sha256": row["scene_sha256"],
                    "trajectory_sha256": row["trajectory_sha256"],
                    "contact_mask_sha256": scene["contact_mask_sha256"],
                }
            )
    if len(records) != 22 or len({row["case_id"] for row in records}) != 22:
        raise RuntimeError("Full runtime closure is not 22 unique rows")
    return files, records


def build_snapshot(
    *,
    output_path: Path = SOURCE_MANIFEST,
    file_list_path: Path = SOURCE_FILE_LIST,
) -> dict[str, Any]:
    """Freeze a new Full-only source and runtime inventory."""
    source = source_files()
    runtime, records = runtime_files()
    paths = sorted(source | runtime)
    files = []
    for value in paths:
        path = REPO / value
        files.append(
            {
                "path": value,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
                "layer": "runtime_input" if value in runtime else "source",
            }
        )
    identity = {"full_rows": records, "files": files}
    snapshot_sha = sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    payload = {
        "schema": "e187_remote_full_snapshot_v1",
        "experiment_id": "E187",
        "stage": "A4_S5_KEEP22_FULL_CEM_DEPLOYMENT",
        "status": "FROZEN",
        "gate0_technical_status": "FAIL",
        "progression_authority": "USER_WAIVED",
        "c9_technical_status": "FAIL",
        "c9_progression_authority": "USER_WAIVED",
        "a1_lock_sha256": sha256(A1_LOCK),
        "a2_lock_sha256": sha256(A2_LOCK),
        "a3_lock_sha256": sha256(A3_LOCK),
        "queue_manifest_sha256": sha256(QUEUE_MANIFEST),
        "promotion_manifest_sha256": sha256(PROMOTION_MANIFEST),
        "full_row_count": len(records),
        "promoted_canary_count": 3,
        "not_run_count": 19,
        "row_records": records,
        "source_file_count": len(source),
        "runtime_file_count": len(runtime),
        "snapshot_file_count": len(files),
        "snapshot_size_bytes": sum(item["size_bytes"] for item in files),
        "source_snapshot_sha256": snapshot_sha,
        "files": files,
    }
    atomic_json(output_path, payload)
    file_list_path.parent.mkdir(parents=True, exist_ok=True)
    listing = "\n".join(paths) + "\n"
    if (
        file_list_path.is_file()
        and file_list_path.read_text(encoding="utf-8") != listing
    ):
        raise RuntimeError(f"refusing to replace frozen file list: {file_list_path}")
    if not file_list_path.exists():
        temporary = file_list_path.with_suffix(file_list_path.suffix + ".tmp")
        temporary.write_text(listing, encoding="utf-8")
        os.replace(temporary, file_list_path)
    return payload


def derive_remote_root(snapshot_sha: str) -> str:
    """Derive an isolated Full root from the snapshot identity."""
    if len(snapshot_sha) != 64:
        raise ValueError("snapshot SHA must contain 64 hex characters")
    int(snapshot_sha, 16)
    return f"{REMOTE_PARENT}/e187_full_{snapshot_sha[:16]}/spider"


def validate_remote_root(remote_root: str) -> None:
    """Reject broad, canary, shared-checkout, or malformed remote roots."""
    prefix = f"{REMOTE_PARENT}/e187_full_"
    if not remote_root.startswith(prefix) or not remote_root.endswith("/spider"):
        raise RuntimeError(f"unsafe E187 Full remote root: {remote_root}")
    identity = remote_root[len(prefix) : -len("/spider")]
    if len(identity) != 16:
        raise RuntimeError(f"unexpected E187 Full snapshot ID: {identity}")
    int(identity, 16)


def verify_snapshot(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every immutable snapshot file."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            mismatches.append({"path": entry["path"], "error": "missing"})
        elif (actual := sha256(path)) != entry["sha256"]:
            mismatches.append(
                {
                    "path": entry["path"],
                    "error": "sha256_mismatch",
                    "actual": actual,
                    "expected": entry["sha256"],
                }
            )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "root": str(root),
        "checked_files": len(manifest["files"]),
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "mismatches": mismatches,
    }


def parse_json_output(output: str) -> dict[str, Any]:
    """Parse the final compact JSON line from a remote helper."""
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("helper returned no output")
    return json.loads(lines[-1])


def remote_verify_command(remote_root: str) -> str:
    """Build the exact remote snapshot verification command."""
    validate_remote_root(remote_root)
    module = "workspace/core4d/scripts/experiments/E187/deploy_remote_full.py"
    manifest = f"{remote_root}/{REMOTE_MANIFEST_RELATIVE}"
    return (
        f"cd {shlex.quote(remote_root)} && {shlex.quote(REMOTE_PYTHON)} "
        f"{shlex.quote(module)} verify --root {shlex.quote(remote_root)} "
        f"--manifest {shlex.quote(manifest)}"
    )


def remote_environment(remote_host: str) -> dict[str, Any]:
    """Verify the existing shared environment without modifying it."""
    code = (
        "import importlib.metadata as m,json,torch;"
        "names=('torch','mujoco','mujoco-warp','warp-lang','hydra-core','open3d');"
        "print(json.dumps({'versions':{n:m.version(n) for n in names},"
        "'cuda_available':torch.cuda.is_available()},separators=(',',':')))"
    )
    result = parse_json_output(
        ssh(remote_host, f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)}").stdout
    )
    if not result["cuda_available"]:
        raise RuntimeError("remote shared environment has no CUDA")
    ffprobe = ssh(remote_host, "command -v ffprobe", check=False)
    if ffprobe.returncode != 0:
        raise RuntimeError("remote ffprobe is unavailable")
    result["python"] = REMOTE_PYTHON
    result["ffprobe"] = ffprobe.stdout.strip()
    result["shared_environment_modified"] = False
    return result


def deploy(
    *,
    remote_host: str,
    source_manifest: Path = SOURCE_MANIFEST,
    file_list: Path = SOURCE_FILE_LIST,
    output_path: Path = DEPLOYMENT_MANIFEST,
) -> dict[str, Any]:
    """Deploy one immutable Full snapshot without touching the shared checkout."""
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    local = verify_snapshot(REPO, source_manifest)
    if local["status"] != "PASS":
        raise RuntimeError(f"local snapshot verification failed: {local}")
    remote_root = derive_remote_root(manifest["source_snapshot_sha256"])
    validate_remote_root(remote_root)
    remote_manifest = f"{remote_root}/{REMOTE_MANIFEST_RELATIVE}"
    has_manifest = (
        ssh(
            remote_host, f"test -f {shlex.quote(remote_manifest)}", check=False
        ).returncode
        == 0
    )
    root_exists = (
        ssh(remote_host, f"test -e {shlex.quote(remote_root)}", check=False).returncode
        == 0
    )
    if root_exists and not has_manifest:
        raise RuntimeError("Full snapshot root exists without frozen manifest")
    if not root_exists:
        ssh(
            remote_host,
            f"mkdir -p {shlex.quote(remote_root)} "
            f"{shlex.quote(str(Path(remote_manifest).parent))}",
        )
        command(
            (
                "rsync",
                "--archive",
                "--relative",
                f"--files-from={file_list}",
                "./",
                f"{remote_host}:{remote_root}/",
            )
        )
        command(
            (
                "rsync",
                "--archive",
                str(source_manifest),
                f"{remote_host}:{remote_manifest}",
            )
        )
    verification = parse_json_output(
        ssh(remote_host, remote_verify_command(remote_root)).stdout
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"remote Full snapshot verification failed: {verification}")
    payload = {
        "schema": "e187_remote_full_deployment_v1",
        "experiment_id": "E187",
        "stage": "A4_S5_KEEP22_FULL_CEM_DEPLOYMENT",
        "status": "PASS",
        "remote_host": remote_host,
        "remote_root": remote_root,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "source_manifest_sha256": sha256(source_manifest),
        "snapshot_file_count": manifest["snapshot_file_count"],
        "snapshot_size_bytes": manifest["snapshot_size_bytes"],
        "frozen_remote_root_verified": True,
        "source_verification": verification,
        "environment": remote_environment(remote_host),
        "rsync_delete_used": False,
        "shared_checkout_mutated": False,
        "existing_processes_modified": False,
        "canary_remote_root_reused": False,
    }
    atomic_json(output_path, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", type=Path, default=SOURCE_MANIFEST)
    freeze.add_argument("--file-list", type=Path, default=SOURCE_FILE_LIST)
    deploy_parser = sub.add_parser("deploy")
    deploy_parser.add_argument("--remote-host", default="spider-remote")
    deploy_parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    deploy_parser.add_argument("--file-list", type=Path, default=SOURCE_FILE_LIST)
    deploy_parser.add_argument("--output", type=Path, default=DEPLOYMENT_MANIFEST)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return root


def main() -> int:
    """Freeze, deploy, or verify E187 Full inputs."""
    args = parser().parse_args()
    if args.command == "freeze":
        payload = build_snapshot(output_path=args.output, file_list_path=args.file_list)
    elif args.command == "deploy":
        payload = deploy(
            remote_host=args.remote_host,
            source_manifest=args.manifest,
            file_list=args.file_list,
            output_path=args.output,
        )
    else:
        payload = verify_snapshot(args.root, args.manifest)
    printable = payload
    if args.command == "freeze":
        printable = {
            "status": payload["status"],
            "snapshot_file_count": payload["snapshot_file_count"],
            "snapshot_size_bytes": payload["snapshot_size_bytes"],
            "source_snapshot_sha256": payload["source_snapshot_sha256"],
        }
    print(json.dumps(printable, sort_keys=True, separators=(",", ":")))
    return 0 if payload["status"] in {"PASS", "FROZEN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
