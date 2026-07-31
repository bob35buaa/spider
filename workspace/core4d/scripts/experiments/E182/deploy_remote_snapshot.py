#!/usr/bin/env python3
"""Deploy an immutable E182 source snapshot to the isolated Ada run root."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from e182_common import (
    REPO_ROOT,
    atomic_json,
    relative_to_repo,
    sha256_bytes,
    sha256_file,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/s0_environment"
DEFAULT_SOURCE_MANIFEST = DEFAULT_OUTPUT_ROOT / "source_snapshot_manifest.json"
DEFAULT_FILE_LIST = DEFAULT_OUTPUT_ROOT / "source_snapshot_files.txt"
DEFAULT_DEPLOYMENT_MANIFEST = DEFAULT_OUTPUT_ROOT / "remote_deployment_manifest.json"
DEFAULT_COACD_FILE_LIST = DEFAULT_OUTPUT_ROOT / "coacd_distribution_files.txt"
DEFAULT_COACD_MANIFEST = DEFAULT_OUTPUT_ROOT / "coacd_distribution_manifest.json"
REMOTE_PARENT = "/home/xiayb/pHRI_workspace/e182_runs"
REMOTE_BASE_PYTHON = "/home/xiayb/pHRI_workspace/spider/.venv/bin/python"
REMOTE_MANIFEST_RELATIVE = (
    "workspace/core4d/results/E182/s0_environment/source_snapshot_manifest.json"
)


def command(
    args: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture text output."""
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def source_paths() -> list[str]:
    """Return tracked plus non-ignored untracked files in stable order."""
    tracked = command(("git", "ls-files")).stdout.splitlines()
    untracked = command(
        ("git", "ls-files", "--others", "--exclude-standard")
    ).stdout.splitlines()
    paths = sorted({path for path in (*tracked, *untracked) if path})
    return [path for path in paths if (REPO_ROOT / path).is_file()]


def build_source_manifest(
    *,
    output_path: Path = DEFAULT_SOURCE_MANIFEST,
    file_list_path: Path = DEFAULT_FILE_LIST,
) -> dict[str, Any]:
    """Freeze the exact local files that remote workers will execute."""
    paths = source_paths()
    files = []
    for relative in paths:
        path = REPO_ROOT / relative
        mode = stat.S_IMODE(path.stat().st_mode)
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "mode": f"{mode:04o}",
            }
        )
    git_head = command(("git", "rev-parse", "HEAD")).stdout.strip()
    dirty_patch = command(("git", "diff", "--binary", "HEAD")).stdout.encode()
    identity = {
        "git_head": git_head,
        "dirty_patch_sha256": sha256_bytes(dirty_patch),
        "uv_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "files": files,
    }
    source_sha = sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest = {
        "experiment_id": "E182",
        "status": "FROZEN",
        "git_head": git_head,
        "dirty_patch_sha256": identity["dirty_patch_sha256"],
        "uv_lock_sha256": identity["uv_lock_sha256"],
        "source_file_count": len(files),
        "source_snapshot_sha256": source_sha,
        "files": files,
    }
    atomic_json(output_path, manifest)
    file_list_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_list_path.with_suffix(file_list_path.suffix + ".tmp")
    temporary.write_text("\n".join(paths) + "\n", encoding="utf-8")
    os.replace(temporary, file_list_path)
    return manifest


def derive_remote_root(source_snapshot_sha256: str) -> str:
    """Derive the immutable remote root from source identity."""
    if len(source_snapshot_sha256) != 64:
        raise ValueError("source snapshot SHA must contain 64 hex characters")
    int(source_snapshot_sha256, 16)
    return f"{REMOTE_PARENT}/e182_{source_snapshot_sha256[:16]}/spider"


def validate_remote_root(remote_root: str) -> None:
    """Reject broad or unexpected remote mutation targets."""
    prefix = f"{REMOTE_PARENT}/e182_"
    if not remote_root.startswith(prefix) or not remote_root.endswith("/spider"):
        raise RuntimeError(f"unsafe E182 remote root: {remote_root}")
    relative = remote_root[len(prefix) : -len("/spider")]
    if len(relative) != 16:
        raise RuntimeError(f"unexpected E182 execution ID: {relative}")
    int(relative, 16)


def verify_snapshot(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify all source files against a frozen manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str]] = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            mismatches.append({"path": entry["path"], "error": "missing"})
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
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
        "manifest": str(manifest_path),
        "checked_files": len(manifest["files"]),
        "mismatches": mismatches,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
    }


def ssh(
    remote_host: str, script: str, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a bounded remote shell command."""
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


def remote_verify_command(remote_root: str) -> str:
    """Return the exact verifier command for an isolated remote root."""
    validate_remote_root(remote_root)
    module = "workspace/core4d/scripts/experiments/E182/deploy_remote_snapshot.py"
    manifest = f"{remote_root}/{REMOTE_MANIFEST_RELATIVE}"
    return (
        f"cd {shlex.quote(remote_root)} && {shlex.quote(REMOTE_BASE_PYTHON)} "
        f"{shlex.quote(module)} verify --root {shlex.quote(remote_root)} "
        f"--manifest {shlex.quote(manifest)}"
    )


def parse_json_output(output: str) -> dict[str, Any]:
    """Parse the last JSON object emitted by a remote helper."""
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote helper returned no output")
    return json.loads(lines[-1])


def ensure_remote_dependencies(remote_host: str, remote_root: str) -> dict[str, Any]:
    """Provision the locked local CoACD distribution into the isolated layer."""
    validate_remote_root(remote_root)
    dependency_root = f"{remote_root}/.e182_deps"
    python = shlex.quote(REMOTE_BASE_PYTHON)
    quoted_dependency = shlex.quote(dependency_root)
    remote_manifest = f"{dependency_root}/coacd_distribution_manifest.json"

    distribution = importlib.metadata.distribution("coacd")
    distribution_root = Path(distribution.locate_file("")).resolve()
    distribution_files = []
    for package_path in distribution.files or ():
        relative = Path(package_path)
        source = Path(distribution.locate_file(package_path)).resolve()
        if ".." in relative.parts or not source.is_file():
            continue
        distribution_files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    distribution_files.sort(key=lambda entry: entry["path"])
    if not any(entry["path"] == "coacd/lib_coacd.so" for entry in distribution_files):
        raise RuntimeError("local CoACD distribution lacks lib_coacd.so")
    distribution_manifest = {
        "name": "coacd",
        "version": distribution.version,
        "files": distribution_files,
    }
    atomic_json(DEFAULT_COACD_MANIFEST, distribution_manifest)
    DEFAULT_COACD_FILE_LIST.write_text(
        "\n".join(entry["path"] for entry in distribution_files) + "\n",
        encoding="utf-8",
    )

    verifier_code = (
        "import hashlib,json,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);m=json.load(open(sys.argv[2]));bad=[];"
        "[(bad.append(e['path']) if (not (p:=root/e['path']).is_file() or "
        "hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']) else None) "
        "for e in m['files']];print(json.dumps({'status':'PASS' if not bad else "
        "'FAIL','checked':len(m['files']),'bad':bad},separators=(',',':')))"
    )
    verify_command = (
        f"{python} -c {shlex.quote(verifier_code)} {quoted_dependency} "
        f"{shlex.quote(remote_manifest)}"
    )
    frozen_probe = ssh(
        remote_host,
        f"test -f {shlex.quote(remote_manifest)}",
        check=False,
    )
    reused = frozen_probe.returncode == 0
    if reused:
        existing_verify = ssh(remote_host, verify_command, check=False)
        if existing_verify.returncode != 0:
            raise RuntimeError(existing_verify.stderr.strip())
        existing_payload = parse_json_output(existing_verify.stdout)
        if existing_payload["status"] != "PASS":
            raise RuntimeError("existing frozen CoACD dependency layer is corrupted")
    else:
        ssh(remote_host, f"mkdir -p {quoted_dependency}")
        command(
            (
                "rsync",
                "--archive",
                "--relative",
                f"--files-from={DEFAULT_COACD_FILE_LIST}",
                "./",
                f"{remote_host}:{dependency_root}/",
            ),
            cwd=distribution_root,
        )
        command(
            (
                "rsync",
                "--archive",
                str(DEFAULT_COACD_MANIFEST),
                f"{remote_host}:{remote_manifest}",
            )
        )
        transferred_verify = ssh(remote_host, verify_command)
        transferred_payload = parse_json_output(transferred_verify.stdout)
        if transferred_payload["status"] != "PASS":
            raise RuntimeError(
                f"remote CoACD verification failed: {transferred_payload}"
            )

    version_probe = f"PYTHONPATH={quoted_dependency} {python} -c " + shlex.quote(
        "import importlib.metadata as m; "
        "print(m.version('coacd')); print(m.version('trimesh')); "
        "print(m.version('mujoco')); print(m.version('mujoco-warp')); "
        "print(m.version('warp-lang'))"
    )
    after = ssh(remote_host, version_probe)
    versions = after.stdout.splitlines()
    expected = ["1.0.11", "4.11.5", "3.7.0", "3.7.0.1", "1.12.1"]
    if versions != expected:
        raise RuntimeError(f"remote dependency mismatch: {versions} != {expected}")
    return {
        "status": "PASS",
        "dependency_root": dependency_root,
        "coacd_distribution_manifest_sha256": sha256_file(DEFAULT_COACD_MANIFEST),
        "coacd_file_count": len(distribution_files),
        "reused_existing_frozen_layer": reused,
        "versions": dict(
            zip(
                ("coacd", "trimesh", "mujoco", "mujoco-warp", "warp-lang"),
                versions,
                strict=True,
            )
        ),
        "shared_environment_modified": False,
    }


def deploy(
    *,
    remote_host: str,
    source_manifest: Path,
    file_list: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Deploy and verify one immutable E182 source snapshot."""
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    remote_root = derive_remote_root(manifest["source_snapshot_sha256"])
    validate_remote_root(remote_root)
    remote_manifest = f"{remote_root}/{REMOTE_MANIFEST_RELATIVE}"

    frozen_probe = ssh(
        remote_host,
        f"test -f {shlex.quote(remote_manifest)}",
        check=False,
    )
    reused = frozen_probe.returncode == 0
    if reused:
        verification = parse_json_output(
            ssh(remote_host, remote_verify_command(remote_root)).stdout
        )
        if verification["status"] != "PASS":
            raise RuntimeError("existing frozen E182 remote root failed verification")
    else:
        manifest_parent = str(Path(remote_manifest).parent)
        ssh(
            remote_host,
            f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(manifest_parent)}",
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
            raise RuntimeError(f"remote source verification failed: {verification}")

    dependencies = ensure_remote_dependencies(remote_host, remote_root)
    deployment = {
        "experiment_id": "E182",
        "status": "PASS",
        "remote_host": remote_host,
        "remote_root": remote_root,
        "source_manifest": relative_to_repo(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "reused_existing_frozen_root": reused,
        "source_verification": verification,
        "dependencies": dependencies,
        "rsync_delete_used": False,
        "shared_checkout_mutated": False,
        "existing_processes_modified": False,
    }
    atomic_json(output_path, deployment)
    return deployment


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    freeze_parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--remote-host", default="spider-remote")
    deploy_parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    deploy_parser.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    deploy_parser.add_argument(
        "--output", type=Path, default=DEFAULT_DEPLOYMENT_MANIFEST
    )
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    """Freeze, deploy, or verify an E182 source snapshot."""
    args = build_parser().parse_args()
    if args.command == "freeze":
        payload = build_source_manifest(
            output_path=args.output,
            file_list_path=args.file_list,
        )
    elif args.command == "deploy":
        payload = deploy(
            remote_host=args.remote_host,
            source_manifest=args.source_manifest,
            file_list=args.file_list,
            output_path=args.output,
        )
    else:
        payload = verify_snapshot(args.root, args.manifest)
    printable = payload
    if args.command == "freeze":
        printable = {
            "status": payload["status"],
            "source_file_count": payload["source_file_count"],
            "source_snapshot_sha256": payload["source_snapshot_sha256"],
        }
    print(json.dumps(printable, sort_keys=True, separators=(",", ":")))
    return 0 if payload["status"] in {"PASS", "FROZEN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
