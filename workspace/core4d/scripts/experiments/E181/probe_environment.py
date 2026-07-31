#!/usr/bin/env python3
"""Capture the E181 environment and run S0 compile probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import coacd
import mujoco
import mujoco_warp as mjwarp
import trimesh
import warp

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT = (
    REPO_ROOT / "workspace/core4d/results/E181/s0_environment/dependency_manifest.json"
)
REMOTE_GPU_COMMAND = (
    "nvidia-smi "
    "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"
    "utilization.gpu --format=csv,noheader,nounits"
)
REMOTE_PROCESS_COMMAND = (
    "nvidia-smi "
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory "
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


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(
    args: list[str],
    *,
    cwd: Path = REPO_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture text output."""
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def git_snapshot() -> dict[str, Any]:
    """Capture the reproducible local source snapshot identity."""
    head = command(["git", "rev-parse", "HEAD"]).stdout.strip()
    status = command(["git", "status", "--porcelain=v1"]).stdout.splitlines()
    patch = command(["git", "diff", "--binary", "HEAD"]).stdout.encode()
    untracked_output = command(
        ["git", "ls-files", "--others", "--exclude-standard"]
    ).stdout
    untracked = [line for line in untracked_output.splitlines() if line]
    dirty_entries: list[dict[str, Any]] = []
    for status_line in status:
        relative = status_line[3:]
        path = REPO_ROOT / relative
        entry: dict[str, Any] = {
            "status": status_line[:2],
            "path": relative,
        }
        if path.is_file():
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
        dirty_entries.append(entry)
    untracked_entries = [
        {
            "path": relative,
            "size_bytes": (REPO_ROOT / relative).stat().st_size,
            "sha256": sha256_file(REPO_ROOT / relative),
        }
        for relative in untracked
        if (REPO_ROOT / relative).is_file()
    ]
    identity = {
        "git_head": head,
        "dirty_patch_sha256": sha256_bytes(patch),
        "dirty_entries": dirty_entries,
        "untracked_entries": untracked_entries,
    }
    identity["source_snapshot_sha256"] = sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return identity


def parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    """Parse the fixed nvidia-smi GPU CSV query."""
    fields = (
        "index",
        "name",
        "uuid",
        "memory_total_mib",
        "memory_used_mib",
        "memory_free_mib",
        "utilization_gpu_percent",
    )
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
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


def local_gpu_snapshot() -> dict[str, Any]:
    """Capture local GPU and compute-process state."""
    gpu_result = command(REMOTE_GPU_COMMAND.split())
    process_result = command(
        REMOTE_PROCESS_COMMAND.split(),
        check=False,
    )
    return {
        "gpus": parse_gpu_csv(gpu_result.stdout),
        "compute_processes_raw": process_result.stdout.splitlines(),
        "process_query_returncode": process_result.returncode,
    }


def remote_snapshot(remote_host: str) -> dict[str, Any]:
    """Capture remote state without mutating or stopping any process."""
    prefix = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        remote_host,
    ]
    gpu_result = command(prefix + [REMOTE_GPU_COMMAND], check=False)
    process_result = command(prefix + [REMOTE_PROCESS_COMMAND], check=False)
    head_result = command(
        prefix + ["cd /home/xiayb/pHRI_workspace/spider && git rev-parse HEAD"],
        check=False,
    )
    status = (
        "PASS" if gpu_result.returncode == 0 and head_result.returncode == 0 else "FAIL"
    )
    gpus = parse_gpu_csv(gpu_result.stdout) if gpu_result.returncode == 0 else []
    expected_gpu_ids = {0, 1}
    if {row["index"] for row in gpus} != expected_gpu_ids:
        status = "FAIL"
    return {
        "status": status,
        "host": remote_host,
        "shared_checkout_head": head_result.stdout.strip(),
        "gpus": gpus,
        "compute_processes_raw": process_result.stdout.splitlines(),
        "gpu_stderr": gpu_result.stderr.strip(),
        "process_stderr": process_result.stderr.strip(),
        "authorization": (
            "user explicitly allows E181 to coexist; never kill, pause, "
            "or preempt existing processes"
        ),
    }


def compile_probe(geom_type: str) -> dict[str, Any]:
    """Compile a minimal mesh/SDF model in MuJoCo and MJWarp."""
    xml = f"""\
<mujoco>
  <asset><mesh name="cube" file="cube.obj"/></asset>
  <worldbody>
    <body name="object">
      <freejoint/>
      <geom name="object_collision" type="{geom_type}" mesh="cube" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model_cpu = mujoco.MjModel.from_xml_string(xml, {"cube.obj": CUBE_OBJ})
    model_wp = mjwarp.put_model(model_cpu)
    return {
        "status": "PASS",
        "geom_type": geom_type,
        "cpu_ngeom": int(model_cpu.ngeom),
        "cpu_nmesh": int(model_cpu.nmesh),
        "warp_ngeom": int(model_wp.ngeom),
        "warp_nmesh": int(model_wp.nmesh),
    }


def dependency_versions() -> dict[str, str]:
    """Return the pinned runtime package versions."""
    names = (
        "coacd",
        "trimesh",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "torch",
    )
    return {name: importlib.metadata.version(name) for name in names}


def build_manifest(remote_host: str) -> dict[str, Any]:
    """Build and validate the S0 environment evidence."""
    signature = inspect.signature(coacd.run_coacd)
    if "real_metric" not in signature.parameters:
        raise RuntimeError("CoACD run_coacd lacks real_metric")
    versions = dependency_versions()
    if versions["coacd"] != "1.0.11":
        raise RuntimeError(f"unexpected CoACD version: {versions['coacd']}")
    if versions["trimesh"] != "4.11.5":
        raise RuntimeError(f"unexpected trimesh version: {versions['trimesh']}")

    # Capture every subprocess-based snapshot before Warp initializes CUDA.
    # Forking after CUDA initialization can terminate the interpreter on exit.
    source = git_snapshot()
    local_gpu = local_gpu_snapshot()
    remote = remote_snapshot(remote_host)
    if remote["status"] != "PASS":
        raise RuntimeError(f"remote environment probe failed: {remote}")
    compile_probes = {
        geom_type: compile_probe(geom_type) for geom_type in ("mesh", "sdf")
    }
    return {
        "experiment_id": "E181",
        "gate": "S0_environment",
        "status": "PASS",
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
        "versions": versions,
        "coacd_run_signature": str(signature),
        "coacd_real_metric_supported": True,
        "trimesh_import_version": trimesh.__version__,
        "mujoco_import_version": mujoco.__version__,
        "warp_import_version": warp.__version__,
        "cuda_authority": "nvidia-smi snapshots; torch import is not required by S0",
        "source": source,
        "local_gpu": local_gpu,
        "remote_gpu": remote,
        "compile_probes": compile_probes,
        "remote_execution_contract": {
            "shared_checkout_mutated": False,
            "isolated_root_template": (
                "/home/xiayb/pHRI_workspace/e181_runs/<execution_id>/spider"
            ),
            "kill_existing_processes": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", default="spider-remote")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run S0 environment probes and persist evidence."""
    args = parse_args()
    manifest = build_manifest(args.remote_host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        "E181_ENVIRONMENT=PASS "
        f"source_snapshot={manifest['source']['source_snapshot_sha256']} "
        f"local_gpus={len(manifest['local_gpu']['gpus'])} "
        f"remote_gpus={len(manifest['remote_gpu']['gpus'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
