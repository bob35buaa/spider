#!/usr/bin/env python3
"""Capture E182 local/remote environment evidence without touching other jobs."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import coacd
import mujoco
import mujoco_warp as mjwarp
import trimesh
import warp
from e182_common import (
    REPO_ROOT,
    atomic_json,
    relative_to_repo,
    sha256_bytes,
    sha256_file,
)

DEFAULT_OUTPUT = (
    REPO_ROOT / "workspace/core4d/results/E182/s0_environment/environment_manifest.json"
)
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "workspace/core4d/results/E182/s0_environment/source_snapshot_manifest.json"
)
GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"
    "utilization.gpu --format=csv,noheader,nounits"
)
PROCESS_QUERY = (
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
    "--format=csv,noheader,nounits"
)
CUBE_OBJ = b"""\
v -0.1 -0.1 -0.1
v 0.1 -0.1 -0.1
v 0.1 0.1 -0.1
v -0.1 0.1 -0.1
v -0.1 -0.1 0.1
v 0.1 -0.1 0.1
v 0.1 0.1 0.1
v -0.1 0.1 0.1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def command(
    args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess from the repository and capture text output."""
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    """Parse the fixed nvidia-smi GPU query."""
    fields = (
        "index",
        "name",
        "uuid",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_gpu_percent",
    )
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        row: dict[str, Any] = dict(zip(fields, values, strict=True))
        for field in (
            "index",
            "memory_total_mib",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_gpu_percent",
        ):
            row[field] = int(row[field])
        rows.append(row)
    return rows


def validate_gpu_contract(
    rows: Sequence[dict[str, Any]],
    *,
    expected_ids: Sequence[int],
    required_name: str | None,
    allow_existing_compute_overlap: bool,
    compute_processes: Sequence[str],
) -> dict[str, Any]:
    """Validate selected GPUs while treating authorized jobs as coexistence evidence."""
    actual_ids = tuple(int(row["index"]) for row in rows)
    name_pass = required_name is None or all(
        required_name in str(row["name"]) for row in rows
    )
    existing = bool(compute_processes)
    overlap_pass = not existing or allow_existing_compute_overlap
    status = (
        "PASS"
        if actual_ids == tuple(expected_ids) and name_pass and overlap_pass
        else "FAIL"
    )
    return {
        "status": status,
        "expected_ids": list(expected_ids),
        "actual_ids": list(actual_ids),
        "required_name": required_name,
        "name_pass": name_pass,
        "existing_compute_overlap": existing,
        "allow_existing_compute_overlap": allow_existing_compute_overlap,
        "kill_existing_processes": False,
        "pause_existing_processes": False,
        "preempt_existing_processes": False,
    }


def dependency_versions() -> dict[str, str]:
    """Return versions used by the geometry and CEM runtime."""
    names = ("coacd", "trimesh", "mujoco", "mujoco-warp", "warp-lang", "torch")
    return {name: importlib.metadata.version(name) for name in names}


def compile_probe(geom_type: str) -> dict[str, Any]:
    """Compile a minimal mesh/SDF model in MuJoCo and MJWarp."""
    xml = f"""\
<mujoco>
  <asset><mesh name="cube" file="cube.obj"/></asset>
  <worldbody><body name="object"><freejoint/>
    <geom name="object_collision" type="{geom_type}" mesh="cube" mass="1"/>
  </body></worldbody>
</mujoco>
"""
    cpu_model = mujoco.MjModel.from_xml_string(xml, {"cube.obj": CUBE_OBJ})
    warp_model = mjwarp.put_model(cpu_model)
    return {
        "status": "PASS",
        "geom_type": geom_type,
        "cpu_ngeom": int(cpu_model.ngeom),
        "cpu_nmesh": int(cpu_model.nmesh),
        "warp_ngeom": int(warp_model.ngeom),
        "warp_nmesh": int(warp_model.nmesh),
    }


def git_snapshot() -> dict[str, Any]:
    """Capture local source identity, including current dirty work."""
    head = command(("git", "rev-parse", "HEAD")).stdout.strip()
    status = command(("git", "status", "--porcelain=v1")).stdout.splitlines()
    patch = command(("git", "diff", "--binary", "HEAD")).stdout.encode()
    untracked = command(
        ("git", "ls-files", "--others", "--exclude-standard")
    ).stdout.splitlines()
    entries: list[dict[str, Any]] = []
    for status_line in status:
        relative = status_line[3:]
        path = REPO_ROOT / relative
        entry: dict[str, Any] = {"status": status_line[:2], "path": relative}
        if path.is_file():
            entry.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
        entries.append(entry)
    identity = {
        "git_head": head,
        "dirty_patch_sha256": sha256_bytes(patch),
        "dirty_entries": entries,
        "untracked_paths": sorted(path for path in untracked if path),
    }
    identity["source_identity_sha256"] = sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return identity


def local_gpu_snapshot(
    selected_gpu_id: int,
    allow_existing_compute_overlap: bool,
) -> dict[str, Any]:
    """Capture and validate the selected local GPU."""
    gpu_result = command(GPU_QUERY.split())
    process_result = command(PROCESS_QUERY.split(), check=False)
    all_rows = parse_gpu_csv(gpu_result.stdout)
    selected = [row for row in all_rows if row["index"] == selected_gpu_id]
    processes = tuple(
        line for line in process_result.stdout.splitlines() if line.strip()
    )
    contract = validate_gpu_contract(
        selected,
        expected_ids=(selected_gpu_id,),
        required_name=None,
        allow_existing_compute_overlap=allow_existing_compute_overlap,
        compute_processes=processes,
    )
    return {
        "all_gpus": all_rows,
        "selected_gpus": selected,
        "compute_processes_raw": list(processes),
        "process_query_returncode": process_result.returncode,
        "contract": contract,
    }


def ssh_command(remote_host: str, script: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only probe script over SSH."""
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
        check=False,
    )


def remote_gpu_snapshot(
    remote_host: str,
    allow_existing_compute_overlap: bool,
) -> dict[str, Any]:
    """Capture and validate the two remote Ada GPUs."""
    gpu_result = ssh_command(remote_host, GPU_QUERY)
    process_result = ssh_command(remote_host, PROCESS_QUERY)
    if gpu_result.returncode != 0:
        raise RuntimeError(f"remote GPU query failed: {gpu_result.stderr.strip()}")
    rows = parse_gpu_csv(gpu_result.stdout)
    processes = tuple(
        line for line in process_result.stdout.splitlines() if line.strip()
    )
    contract = validate_gpu_contract(
        rows,
        expected_ids=(0, 1),
        required_name="RTX 6000 Ada",
        allow_existing_compute_overlap=allow_existing_compute_overlap,
        compute_processes=processes,
    )
    return {
        "host": remote_host,
        "gpus": rows,
        "compute_processes_raw": list(processes),
        "gpu_query_returncode": gpu_result.returncode,
        "process_query_returncode": process_result.returncode,
        "contract": contract,
    }


def remote_deployment_snapshot(
    remote_host: str,
    remote_root: str,
    source_manifest: Path,
) -> dict[str, Any]:
    """Verify the isolated source/lock/dependency layer created by deployment."""
    if not remote_root.startswith("/home/xiayb/pHRI_workspace/e182_runs/"):
        raise RuntimeError(f"unexpected E182 remote root: {remote_root}")
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    relative_manifest = (
        "workspace/core4d/results/E182/s0_environment/source_snapshot_manifest.json"
    )
    quoted_root = remote_root.replace("'", "'\\''")
    script = (
        "set -eu; "
        f"root='{quoted_root}'; "
        'test -d "$root"; test -f "$root/uv.lock"; '
        f'test -f "$root/{relative_manifest}"; '
        "printf 'uv_lock_sha='; sha256sum \"$root/uv.lock\" | awk '{print $1}'; "
        f"printf 'source_manifest_sha='; sha256sum \"$root/{relative_manifest}\" | awk '{{print $1}}'; "
        "printf 'tmux='; command -v tmux; printf 'rsync='; command -v rsync; "
        'cd "$root"; PYTHONPATH="$root/.e182_deps${PYTHONPATH:+:$PYTHONPATH}" '
        "/home/xiayb/pHRI_workspace/spider/.venv/bin/python -c "
        "'import importlib.metadata as m; "
        'print("coacd="+m.version("coacd")); '
        'print("trimesh="+m.version("trimesh")); '
        'print("mujoco="+m.version("mujoco")); '
        'print("mujoco-warp="+m.version("mujoco-warp")); '
        'print("warp-lang="+m.version("warp-lang"))\''
    )
    result = ssh_command(remote_host, script)
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    expected = {
        "uv_lock_sha": sha256_file(REPO_ROOT / "uv.lock"),
        "source_manifest_sha": sha256_file(source_manifest),
        "coacd": "1.0.11",
        "trimesh": "4.11.5",
        "mujoco": "3.7.0",
        "mujoco-warp": "3.7.0.1",
        "warp-lang": "1.12.1",
    }
    parity = result.returncode == 0 and all(
        values.get(key) == value for key, value in expected.items()
    )
    return {
        "status": "PASS" if parity else "FAIL",
        "remote_root": remote_root,
        "source_manifest": relative_to_repo(source_manifest),
        "values": values,
        "expected": expected,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "shared_checkout_mutated": False,
    }


def build_manifest(
    *,
    remote_host: str,
    remote_root: str,
    selected_local_gpu: int,
    allow_existing_compute_overlap: bool,
    source_manifest: Path,
) -> dict[str, Any]:
    """Build and validate E182 S0 environment evidence."""
    versions = dependency_versions()
    expected_versions = {"coacd": "1.0.11", "trimesh": "4.11.5"}
    if any(versions[name] != version for name, version in expected_versions.items()):
        raise RuntimeError(f"local dependency mismatch: {versions}")
    if "real_metric" not in inspect.signature(coacd.run_coacd).parameters:
        raise RuntimeError("CoACD lacks real_metric")

    source = git_snapshot()
    local_gpu = local_gpu_snapshot(selected_local_gpu, allow_existing_compute_overlap)
    remote_gpu = remote_gpu_snapshot(remote_host, allow_existing_compute_overlap)
    deployment = remote_deployment_snapshot(remote_host, remote_root, source_manifest)
    compile_probes = {kind: compile_probe(kind) for kind in ("mesh", "sdf")}
    statuses = (
        local_gpu["contract"]["status"],
        remote_gpu["contract"]["status"],
        deployment["status"],
    )
    status = "PASS" if set(statuses) == {"PASS"} else "FAIL"
    manifest = {
        "experiment_id": "E182",
        "gate": "S0_environment",
        "status": status,
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
        "versions": versions,
        "coacd_signature": str(inspect.signature(coacd.run_coacd)),
        "trimesh_import_version": trimesh.__version__,
        "mujoco_import_version": mujoco.__version__,
        "warp_import_version": warp.__version__,
        "source": source,
        "local_gpu": local_gpu,
        "remote_gpu": remote_gpu,
        "remote_deployment": deployment,
        "compile_probes": compile_probes,
        "execution_contract": {
            "allow_existing_compute_overlap": allow_existing_compute_overlap,
            "kill_existing_processes": False,
            "pause_existing_processes": False,
            "preempt_existing_processes": False,
            "local_worker_count": 1,
            "remote_worker_count": 2,
        },
    }
    if status != "PASS":
        raise RuntimeError(f"E182 environment contract failed: {json.dumps(manifest)}")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", default="spider-remote")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-gpu-id", type=int, default=0)
    parser.add_argument("--allow-existing-compute-overlap", action="store_true")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run S0 environment probes and persist evidence."""
    args = parse_args()
    manifest = build_manifest(
        remote_host=args.remote_host,
        remote_root=args.remote_root,
        selected_local_gpu=args.local_gpu_id,
        allow_existing_compute_overlap=args.allow_existing_compute_overlap,
        source_manifest=args.source_manifest,
    )
    atomic_json(args.output, manifest)
    print(
        "E182_ENVIRONMENT=PASS "
        f"local_gpu={args.local_gpu_id} remote_gpus=0,1 "
        f"source={manifest['source']['source_identity_sha256']} overlap=allowed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
