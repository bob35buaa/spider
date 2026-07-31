#!/usr/bin/env python3
"""Freeze, deploy, and verify git-ignored E182 S1 runtime inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from pathlib import Path
from typing import Any

from deploy_remote_snapshot import (
    REMOTE_BASE_PYTHON,
    command,
    parse_json_output,
    ssh,
    validate_remote_root,
)
from e182_common import (
    REPO_ROOT,
    atomic_json,
    sha256_bytes,
    sha256_file,
)

DEFAULT_RESULT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/s1_query_tape"
DEFAULT_MANIFEST = DEFAULT_RESULT_ROOT / "runtime_inputs_manifest.json"
DEFAULT_FILE_LIST = DEFAULT_RESULT_ROOT / "runtime_input_files.txt"
DEFAULT_DEPLOYMENT = DEFAULT_RESULT_ROOT / "runtime_input_deployment.json"
DEFAULT_AUTHORITY = (
    REPO_ROOT / "workspace/core4d/results/E182/authority/dev3/manifest.tsv"
)
DEFAULT_E178_SOURCE = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
EXPECTED_E178_SOURCE_SHA256 = (
    "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8"
)
REMOTE_MANIFEST_RELATIVE = (
    "workspace/core4d/results/E182/s1_query_tape/runtime_inputs_manifest.json"
)


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    """Read one TSV into dictionaries."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _safe_relative_path(repo_root: Path, value: str | Path) -> str:
    """Return a normalized repository-relative file path."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"runtime input must be repository-relative: {value}")
    normalized = path.as_posix()
    if not normalized or not (repo_root / normalized).is_file():
        raise FileNotFoundError(repo_root / normalized)
    return normalized


def _tree_files(repo_root: Path, relative_root: str | Path) -> list[str]:
    """List all files under one repository-relative directory."""
    root = Path(relative_root)
    if root.is_absolute() or ".." in root.parts:
        raise RuntimeError(f"runtime tree must be repository-relative: {relative_root}")
    absolute = repo_root / root
    if not absolute.is_dir():
        raise FileNotFoundError(absolute)
    return [
        path.relative_to(repo_root).as_posix()
        for path in sorted(
            candidate for candidate in absolute.rglob("*") if candidate.is_file()
        )
    ]


def _discover_real_runtime_paths(
    *,
    repo_root: Path,
    authority_path: Path,
    e178_source_path: Path,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Discover the exact dev3 runtime closure from frozen authority fields."""
    if sha256_file(e178_source_path) != EXPECTED_E178_SOURCE_SHA256:
        raise RuntimeError("E178 source manifest SHA changed")
    authority_rows = read_tsv_rows(authority_path)
    source_by_case = {row["case_id"]: row for row in read_tsv_rows(e178_source_path)}
    paths = {
        authority_path.relative_to(repo_root).as_posix(),
        e178_source_path.relative_to(repo_root).as_posix(),
    }
    case_records: list[dict[str, Any]] = []
    for row in authority_rows:
        case_id = row["case_id"]
        if case_id not in source_by_case:
            raise RuntimeError(f"dev case missing from E178 source: {case_id}")
        source = source_by_case[case_id]
        if (source["cem_samples"], source["cem_opt_steps"], source["cem_seed"]) != (
            "1024",
            "32",
            "0",
        ):
            raise RuntimeError(f"{case_id}: E178 budget authority changed")

        task_dir = Path(row["target_scene"]).parent
        object_dir = Path(
            f"example_datasets/processed/core4d/assets/objects/{row['object_key']}"
        )
        case_paths = set(_tree_files(repo_root, task_dir))
        case_paths.update(_tree_files(repo_root, object_dir))
        case_paths.update(
            {
                _safe_relative_path(repo_root, row["contact_mask"]),
                _safe_relative_path(repo_root, source["result_npz"]),
                _safe_relative_path(repo_root, source["config_act"]),
            }
        )
        trajectory = repo_root / row["trajectory"]
        contact_mask = repo_root / row["contact_mask"]
        if sha256_file(trajectory) != row["trajectory_sha256"]:
            raise RuntimeError(f"{case_id}: trajectory SHA changed")
        if sha256_file(contact_mask) != row["contact_mask_sha256"]:
            raise RuntimeError(f"{case_id}: contact-mask SHA changed")
        paths.update(case_paths)
        case_records.append(
            {
                "case_id": case_id,
                "object_key": row["object_key"],
                "task_dir": task_dir.as_posix(),
                "file_count": len(case_paths),
                "e178_result_npz": source["result_npz"],
                "e178_config_act": source["config_act"],
            }
        )
    return sorted(paths), [row["case_id"] for row in authority_rows], case_records


def build_runtime_input_manifest(
    *,
    output_path: Path = DEFAULT_MANIFEST,
    file_list_path: Path = DEFAULT_FILE_LIST,
    repo_root: Path = REPO_ROOT,
    authority_path: Path = DEFAULT_AUTHORITY,
    e178_source_path: Path = DEFAULT_E178_SOURCE,
    explicit_paths: tuple[str, ...] | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Freeze all runtime inputs needed by S1 while preserving repo-relative paths."""
    if explicit_paths is None:
        paths, discovered_case_ids, case_records = _discover_real_runtime_paths(
            repo_root=repo_root,
            authority_path=authority_path,
            e178_source_path=e178_source_path,
        )
    else:
        paths = sorted(
            {_safe_relative_path(repo_root, path) for path in explicit_paths}
        )
        discovered_case_ids = list(case_ids or ())
        case_records = []
    files = [
        {
            "path": path,
            "sha256": sha256_file(repo_root / path),
            "size_bytes": (repo_root / path).stat().st_size,
        }
        for path in paths
    ]
    identity = {"case_ids": discovered_case_ids, "files": files}
    runtime_sha = sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    manifest = {
        "experiment_id": "E182",
        "stage": "S1_runtime_inputs",
        "status": "FROZEN",
        "case_count": len(discovered_case_ids),
        "case_ids": discovered_case_ids,
        "case_records": case_records,
        "runtime_file_count": len(files),
        "runtime_snapshot_sha256": runtime_sha,
        "files": files,
    }
    atomic_json(output_path, manifest)
    file_list_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_list_path.with_suffix(file_list_path.suffix + ".tmp")
    temporary.write_text("\n".join(paths) + "\n", encoding="utf-8")
    os.replace(temporary, file_list_path)
    return manifest


def verify_runtime_inputs(
    root: Path,
    manifest_path: Path,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Verify a runtime-input tree; optionally permit files not deployed yet."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str]] = []
    missing: list[str] = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
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
    passed = not mismatches and (allow_missing or not missing)
    return {
        "status": "PASS" if passed else "FAIL",
        "root": str(root),
        "manifest": str(manifest_path),
        "checked_files": len(manifest["files"]) - len(missing),
        "missing_count": len(missing),
        "missing": missing,
        "mismatches": mismatches,
        "runtime_snapshot_sha256": manifest["runtime_snapshot_sha256"],
    }


def assert_compatible_frozen_manifest(
    existing: dict[str, Any], requested: dict[str, Any]
) -> None:
    """Reject attempts to replace a frozen runtime-input identity."""
    if existing.get("runtime_snapshot_sha256") != requested.get(
        "runtime_snapshot_sha256"
    ):
        raise RuntimeError(
            "runtime snapshot mismatch; refusing to overwrite frozen remote input"
        )
    if existing.get("files") != requested.get("files"):
        raise RuntimeError(
            "runtime file inventory mismatch; refusing to overwrite frozen remote input"
        )


def remote_verify_command(
    remote_root: str, manifest_path: str, *, allow_missing: bool = False
) -> str:
    """Build a remote verifier command rooted in the immutable source snapshot."""
    validate_remote_root(remote_root)
    module = "workspace/core4d/scripts/experiments/E182/runtime_inputs.py"
    suffix = " --allow-missing" if allow_missing else ""
    return (
        f"cd {shlex.quote(remote_root)} && {shlex.quote(REMOTE_BASE_PYTHON)} "
        f"{shlex.quote(module)} verify --root {shlex.quote(remote_root)} "
        f"--manifest {shlex.quote(manifest_path)}{suffix}"
    )


def deploy_runtime_inputs(
    *,
    remote_host: str,
    remote_root: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    file_list_path: Path = DEFAULT_FILE_LIST,
    output_path: Path = DEFAULT_DEPLOYMENT,
) -> dict[str, Any]:
    """Deploy runtime inputs without overwriting any inconsistent remote file."""
    validate_remote_root(remote_root)
    requested = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_verification = verify_runtime_inputs(REPO_ROOT, manifest_path)
    if local_verification["status"] != "PASS":
        raise RuntimeError(
            f"local runtime inputs failed verification: {local_verification}"
        )
    remote_manifest = f"{remote_root}/{REMOTE_MANIFEST_RELATIVE}"
    probe = ssh(remote_host, f"test -f {shlex.quote(remote_manifest)}", check=False)
    reused = probe.returncode == 0
    if reused:
        existing = parse_json_output(
            ssh(
                remote_host,
                f"{shlex.quote(REMOTE_BASE_PYTHON)} -c "
                + shlex.quote(
                    "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),separators=(',',':')))"
                )
                + f" {shlex.quote(remote_manifest)}",
            ).stdout
        )
        assert_compatible_frozen_manifest(existing, requested)
    else:
        staging_dir = f"{remote_root}/.e182_runtime_staging"
        staged_manifest = (
            f"{staging_dir}/runtime_{requested['runtime_snapshot_sha256']}.json"
        )
        ssh(remote_host, f"mkdir -p {shlex.quote(staging_dir)}")
        command(
            (
                "rsync",
                "--archive",
                str(manifest_path),
                f"{remote_host}:{staged_manifest}",
            )
        )
        preflight = parse_json_output(
            ssh(
                remote_host,
                remote_verify_command(remote_root, staged_manifest, allow_missing=True),
            ).stdout
        )
        if preflight["status"] != "PASS" or preflight["mismatches"]:
            raise RuntimeError(
                f"remote runtime path conflict; refusing overwrite: {preflight}"
            )
        command(
            (
                "rsync",
                "--archive",
                "--relative",
                "--ignore-existing",
                f"--files-from={file_list_path}",
                "./",
                f"{remote_host}:{remote_root}/",
            )
        )
        remote_parent = str(Path(remote_manifest).parent)
        ssh(
            remote_host,
            f"mkdir -p {shlex.quote(remote_parent)} && "
            f"cp -n {shlex.quote(staged_manifest)} {shlex.quote(remote_manifest)}",
        )
    verification = parse_json_output(
        ssh(
            remote_host,
            remote_verify_command(remote_root, remote_manifest),
        ).stdout
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"remote runtime verification failed: {verification}")
    payload = {
        "experiment_id": "E182",
        "stage": "S1_runtime_input_deployment",
        "status": "PASS",
        "remote_host": remote_host,
        "remote_root": remote_root,
        "runtime_snapshot_sha256": requested["runtime_snapshot_sha256"],
        "runtime_file_count": requested["runtime_file_count"],
        "reused_existing_frozen_runtime": reused,
        "verification": verification,
        "rsync_delete_used": False,
        "rsync_ignore_existing_used": True,
        "shared_checkout_mutated": False,
        "existing_processes_modified": False,
    }
    atomic_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    freeze.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--allow-missing", action="store_true")
    deploy = subparsers.add_parser("deploy")
    deploy.add_argument("--remote-host", default="spider-remote")
    deploy.add_argument("--remote-root", required=True)
    deploy.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    deploy.add_argument("--file-list", type=Path, default=DEFAULT_FILE_LIST)
    deploy.add_argument("--output", type=Path, default=DEFAULT_DEPLOYMENT)
    return parser


def main() -> int:
    """Freeze, verify, or deploy the runtime-input layer."""
    args = build_parser().parse_args()
    if args.command == "freeze":
        payload = build_runtime_input_manifest(
            output_path=args.output,
            file_list_path=args.file_list,
        )
    elif args.command == "verify":
        payload = verify_runtime_inputs(
            args.root,
            args.manifest,
            allow_missing=args.allow_missing,
        )
    else:
        payload = deploy_runtime_inputs(
            remote_host=args.remote_host,
            remote_root=args.remote_root,
            manifest_path=args.manifest,
            file_list_path=args.file_list,
            output_path=args.output,
        )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload["status"] in {"PASS", "FROZEN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
