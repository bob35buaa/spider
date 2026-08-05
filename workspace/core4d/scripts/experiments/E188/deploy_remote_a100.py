#!/usr/bin/env python3
"""Preflight A100 GPU4/5 and deploy isolated E188 canary/full snapshots."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from common import REPO, RESULTS, S0, WORKERS, read_tsv, relative, sha256, write_json

REMOTE_PARENT = "/home/dataset-assist-0/xiayb/workspace/e188_spider_runs"
REMOTE_PYTHON = "/home/dataset-assist-0/xiayb/workspace/spider/.venv/bin/python"
REMOTE_SHARED = "/home/dataset-assist-0/xiayb/workspace/spider"
DEFAULT_HOST = "61.172.170.106"
DEFAULT_PORT = 30409
DEFAULT_IDENTITY = Path("/home/ubuntu/.ssh/id_rsa_tianyiyun")
REQUESTED_GPUS = (4, 5)
MEMORY_LIMIT_MIB = 5000
ROOT_FILES = (".python-version", "pyproject.toml", "requirements.txt", "setup.py", "uv.lock")


def command(args: Sequence[str], *, check: bool = True, cwd: Path = REPO) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, check=check, text=True, capture_output=True)


def ssh_args(host: str, port: int, identity: Path) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", str(port), "-i", str(identity), f"batchcom@{host}"]


def ssh(host: str, port: int, identity: Path, script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return command((*ssh_args(host, port, identity), script), check=check)


def rsync_remote(host: str, port: int, identity: Path) -> str:
    return " ".join(shlex.quote(value) for value in ("ssh", "-p", str(port), "-i", str(identity), "-o", "BatchMode=yes"))


def parse_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for values in csv.reader(raw.splitlines()):
        if not values:
            continue
        index, name, uuid, used, total = (value.strip() for value in values)
        rows.append({"index": int(index), "name": name, "uuid": uuid, "memory_used_mib": int(used), "memory_total_mib": int(total)})
    return rows


def parse_compute_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for values in csv.reader(raw.splitlines()):
        if len(values) < 4:
            continue
        uuid, pid, process, memory = (value.strip() for value in values[:4])
        rows.append({"gpu_uuid": uuid, "pid": pid, "process_name": process, "used_gpu_memory_mib": memory})
    return rows


def remote_gpu_snapshot(host: str, port: int, identity: Path) -> dict[str, Any]:
    gpu = ssh(host, port, identity, "nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv,noheader,nounits").stdout
    compute = ssh(host, port, identity, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits", check=False).stdout
    gpu_rows = parse_gpu_rows(gpu)
    compute_rows = parse_compute_rows(compute)
    by_index = {row["index"]: row for row in gpu_rows}
    if set(REQUESTED_GPUS) - set(by_index):
        raise RuntimeError("remote GPU4/5 are missing")
    requested_uuids = {by_index[index]["uuid"] for index in REQUESTED_GPUS}
    allowed = [
        index
        for index in REQUESTED_GPUS
        if "A100" in by_index[index]["name"]
        and by_index[index]["memory_used_mib"] < MEMORY_LIMIT_MIB
        and not any(row["gpu_uuid"] == by_index[index]["uuid"] for row in compute_rows)
    ]
    return {"gpu_rows": gpu_rows, "compute_rows": compute_rows, "requested_uuids": sorted(requested_uuids), "low_memory_compute_free_gpus": allowed}


def execution_manifest(stage: str) -> Path:
    return S0 / "execution_manifest.json" if stage == "canary" else RESULTS / "s5_full/deployment/full_execution_manifest.json"


def gpu_check(stage: str, round_name: str, *, host: str, port: int, identity: Path, policy_gpus: str) -> dict[str, Any]:
    policy = sorted(int(value) for value in policy_gpus.replace(",", " ").split())
    if policy != list(REQUESTED_GPUS):
        raise RuntimeError(f"policy set must be exactly GPU4/5, got {policy}")
    snapshot = remote_gpu_snapshot(host, port, identity)
    allowed = sorted(set(policy) & set(snapshot["low_memory_compute_free_gpus"]))
    if allowed != list(REQUESTED_GPUS):
        raise RuntimeError(f"A100 GPU4/5 preflight blocked: allowed={allowed}")
    path = execution_manifest(stage)
    if round_name == "first":
        payload = {
            "schema": "e188_a100_execution_manifest_v1",
            "experiment_id": "E188",
            "stage": stage,
            "status": "PREFLIGHT_PASS_1",
            "remote": {"host": host, "port": port, "user": "batchcom"},
            "policy_source": "plan212_and_user_authorization",
            "requested_gpu_set": list(REQUESTED_GPUS),
            "policy_gpu_set": policy,
            "allowed_gpus": allowed,
            "memory_limit_mib": MEMORY_LIMIT_MIB,
            "first_snapshot": snapshot,
            "workers": {"a100-4": {"physical_gpu": 4, "visible_device": 0}, "a100-5": {"physical_gpu": 5, "visible_device": 0}},
            "fallback_allowed": False,
            "preemption_allowed": False,
        }
    else:
        if not path.is_file():
            raise RuntimeError("first GPU check is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PREFLIGHT_PASS_1" or payload.get("allowed_gpus") != list(REQUESTED_GPUS):
            raise RuntimeError("first GPU check is not valid")
        if snapshot["requested_uuids"] != payload["first_snapshot"]["requested_uuids"]:
            raise RuntimeError("GPU4/5 UUID mapping changed between checks")
        payload["second_snapshot"] = snapshot
        payload["status"] = "FROZEN"
    output = path if round_name == "first" else deployment_root(stage) / "launch_execution_manifest.json"
    write_json(output, payload)
    return payload


def freeze_overlap_policy(stage: str) -> dict[str, Any]:
    payload = {
        "schema": "e188_user_authorized_gpu_overlap_v1",
        "experiment_id": "E188",
        "stage": stage,
        "status": "USER_AUTHORIZED_OVERLAP",
        "physical_gpus": list(REQUESTED_GPUS),
        "memory_gate_enabled": False,
        "compute_process_gate_enabled": False,
        "external_workload_overlap_allowed": True,
        "fallback_allowed": False,
        "preemption_allowed": False,
        "authority": "explicit_user_instruction_2026-08-05",
    }
    write_json(execution_manifest(stage), payload)
    return payload


def relative_file(path: Path) -> str:
    path = path.absolute()
    value = path.relative_to(REPO.absolute()).as_posix()
    if not path.is_file():
        raise FileNotFoundError(path)
    return value


def tree_files(path: Path) -> set[str]:
    return {relative_file(candidate) for candidate in sorted(path.rglob("*")) if candidate.is_file() and "__pycache__" not in candidate.parts}


def snapshot_paths(stage: str) -> set[str]:
    result = command(("git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "spider", "examples", "src", "workspace/core4d/scripts/experiments/E188"))

    def required_source(value: str) -> bool:
        path = Path(value)
        if value.startswith("spider/assets/robots/unitree_g1/"):
            return True
        if value.startswith("spider/"):
            return path.suffix in {".py", ".json", ".yaml", ".yml"}
        if value.startswith("examples/config/"):
            return True
        if value.startswith("examples/"):
            return path.suffix == ".py"
        if value.startswith("src/"):
            return path.suffix in {".py", ".json", ".yaml", ".yml"}
        return value.startswith("workspace/core4d/scripts/experiments/E188/")

    files = {
        relative_file(REPO / value)
        for value in result.stdout.splitlines()
        if value
        and required_source(value)
        and (REPO / value).is_file()
        and "__pycache__" not in Path(value).parts
    }
    files.update(relative_file(REPO / value) for value in ROOT_FILES if (REPO / value).is_file())
    files.update(tree_files(S0))
    files.update(tree_files(RESULTS / "s5_full/queue"))
    files.update(tree_files(RESULTS / "s5_full/queue_speed_rebalanced_v2"))
    if stage == "full":
        files.update(tree_files(RESULTS / "s4_canary/rows"))
        files.add(relative_file(RESULTS / "s4_canary/canary_gate.json"))
        files.add(relative_file(RESULTS / "s5_full/promotion_manifest.json"))
        files.update(tree_files(RESULTS / "s5_full/rows"))
    authority = read_tsv(S0 / "authority_manifest.tsv")
    for row in authority:
        task_dir = (REPO / row["scene_act"]).parent
        files.update(tree_files(task_dir))
        files.add(relative_file(REPO / row["contact_mask"]))
        files.update(tree_files((REPO / row["grid_manifest"]).parent))
    assets = REPO / "example_datasets/processed/core4d/assets"
    if assets.is_dir():
        files.update(tree_files(assets))
    return files


def deployment_root(stage: str) -> Path:
    return RESULTS / ("s4_canary/deployment" if stage == "canary" else "s5_full/deployment")


def build_snapshot(stage: str) -> dict[str, Any]:
    paths = sorted(snapshot_paths(stage))
    entries = []
    for value in paths:
        path = REPO / value
        entries.append({"path": value, "sha256": sha256(path), "size_bytes": path.stat().st_size, "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}"})
    identity = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    snapshot_sha = hashlib.sha256(identity).hexdigest()
    payload = {"schema": "e188_a100_snapshot_v1", "experiment_id": "E188", "stage": stage, "status": "FROZEN", "snapshot_file_count": len(entries), "snapshot_size_bytes": sum(row["size_bytes"] for row in entries), "source_snapshot_sha256": snapshot_sha, "files": entries}
    root = deployment_root(stage)
    manifest = root / "source_snapshot_manifest.json"
    listing = root / "source_snapshot_files.txt"
    write_json(manifest, payload, immutable=True)
    text = "\n".join(paths) + "\n"
    if listing.is_file() and listing.read_text(encoding="utf-8") != text:
        raise RuntimeError("refusing to replace frozen source file list")
    if not listing.exists():
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(text, encoding="utf-8")
    return payload


def remote_root(stage: str, snapshot_sha: str) -> str:
    return f"{REMOTE_PARENT}/e188_{stage}_{snapshot_sha[:16]}/spider"


def validate_remote_root(value: str, stage: str) -> None:
    prefix = f"{REMOTE_PARENT}/e188_{stage}_"
    if not value.startswith(prefix) or not value.endswith("/spider") or len(value[len(prefix):-len("/spider")]) != 16:
        raise RuntimeError(f"unsafe E188 remote root: {value}")


def verify_snapshot(root: Path, manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mismatches = []
    for row in payload["files"]:
        path = root / row["path"]
        if not path.is_file():
            mismatches.append({"path": row["path"], "error": "missing"})
        elif sha256(path) != row["sha256"]:
            mismatches.append({"path": row["path"], "error": "sha256"})
    return {"status": "PASS" if not mismatches else "FAIL", "checked_files": len(payload["files"]), "mismatches": mismatches}


def compress_selected(paths: list[str]) -> tuple[Path, str, int, Path]:
    """Build one zstd-compressed tar containing only remote-missing/mismatched files."""
    list_descriptor, raw_list = tempfile.mkstemp(prefix="e188_overlay_", suffix=".txt")
    os.close(list_descriptor)
    listing = Path(raw_list)
    listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
    archive_descriptor, raw_archive = tempfile.mkstemp(prefix="e188_overlay_", suffix=".tar.zst")
    os.close(archive_descriptor)
    archive = Path(raw_archive)
    tar_process = subprocess.Popen(
        ["tar", "--create", "--file", "-", f"--files-from={listing}"],
        cwd=REPO,
        stdout=subprocess.PIPE,
    )
    assert tar_process.stdout is not None
    compressed = subprocess.run(
        ["zstd", "-3", "--threads=0", "--quiet", "--force", "-o", str(archive)],
        stdin=tar_process.stdout,
        check=True,
    )
    tar_process.stdout.close()
    if tar_process.wait() != 0 or compressed.returncode != 0:
        raise RuntimeError("manifest-selected tar/zstd overlay failed")
    return archive, sha256(archive), archive.stat().st_size, listing


def shared_mismatches(
    *, host: str, port: int, identity: Path, source_manifest: Path, snapshot_sha: str
) -> tuple[list[str], str, str]:
    """Compare the frozen manifest against the remote shared checkout read-only."""
    staging_manifest = f"{REMOTE_PARENT}/.staging_{snapshot_sha[:16]}_manifest.json"
    staging_list = f"{REMOTE_PARENT}/.staging_{snapshot_sha[:16]}_files.txt"
    file_list = source_manifest.with_name("source_snapshot_files.txt")
    remote_shell = rsync_remote(host, port, identity)
    command(("rsync", "--archive", "-e", remote_shell, str(source_manifest), f"batchcom@{host}:{staging_manifest}"))
    command(("rsync", "--archive", "-e", remote_shell, str(file_list), f"batchcom@{host}:{staging_list}"))
    code = (
        "import hashlib,json,os,sys;"
        "m=json.load(open(sys.argv[1]));root=sys.argv[2];out=[];"
        "exec(\"for e in m['files']:\\n p=os.path.join(root,e['path'])\\n h=hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.isfile(p) else ''\\n if h!=e['sha256']: out.append(e['path'])\");"
        "print(json.dumps(out,separators=(',',':')))"
    )
    output = ssh(
        host,
        port,
        identity,
        f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(code)} {shlex.quote(staging_manifest)} {shlex.quote(REMOTE_SHARED)}",
    ).stdout
    return json.loads(output.splitlines()[-1]), staging_manifest, staging_list


def parallel_chunk_rsync(
    archive: Path,
    *,
    archive_sha: str,
    incoming: str,
    host: str,
    port: int,
    identity: Path,
) -> int:
    """Transfer one archive as bounded parallel chunks and verify reconstruction."""
    chunk_dir = Path(tempfile.mkdtemp(prefix="e188_chunks_"))
    remote_chunk_dir = incoming + ".chunks"
    try:
        prefix = chunk_dir / "part_"
        command(("split", "--bytes=8M", "--numeric-suffixes=0", "--suffix-length=3", str(archive), str(prefix)))
        chunks = sorted(chunk_dir.glob("part_*"))
        if not chunks:
            raise RuntimeError("overlay split produced no chunks")
        ssh(host, port, identity, f"test ! -e {shlex.quote(remote_chunk_dir)} && mkdir -p {shlex.quote(remote_chunk_dir)}")
        remote_shell = rsync_remote(host, port, identity)

        def send(path: Path) -> None:
            command(("rsync", "--archive", "--partial", "-e", remote_shell, str(path), f"batchcom@{host}:{remote_chunk_dir}/{path.name}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(chunks))) as pool:
            futures = [pool.submit(send, path) for path in chunks]
            for future in futures:
                future.result()
        remote_parts = " ".join(shlex.quote(f"{remote_chunk_dir}/{path.name}") for path in chunks)
        result = ssh(
            host,
            port,
            identity,
            f"cat {remote_parts} > {shlex.quote(incoming)} && sha256sum {shlex.quote(incoming)}",
        ).stdout.split()[0]
        if result != archive_sha:
            raise RuntimeError(f"remote reconstructed overlay SHA mismatch: {result}")
        ssh(host, port, identity, f"rm -f {remote_parts} && rmdir {shlex.quote(remote_chunk_dir)}")
        return len(chunks)
    finally:
        shutil.rmtree(chunk_dir)


def deploy(stage: str, *, host: str, port: int, identity: Path) -> dict[str, Any]:
    root = deployment_root(stage)
    source_manifest = root / "source_snapshot_manifest.json"
    file_list = root / "source_snapshot_files.txt"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if verify_snapshot(REPO, source_manifest)["status"] != "PASS":
        raise RuntimeError("local source snapshot verification failed")
    destination = remote_root(stage, manifest["source_snapshot_sha256"])
    validate_remote_root(destination, stage)
    remote_manifest = f"{destination}/{relative(source_manifest)}"
    exists = ssh(host, port, identity, f"test -e {shlex.quote(destination)}", check=False).returncode == 0
    has_manifest = ssh(host, port, identity, f"test -f {shlex.quote(remote_manifest)}", check=False).returncode == 0
    if exists and not has_manifest:
        raise RuntimeError("remote snapshot root exists without manifest")
    archive_sha = ""
    archive_size = 0
    overlay_file_count = 0
    shared_reused_file_count = 0
    transfer_chunk_count = 0
    if not exists:
        snapshot_sha = manifest["source_snapshot_sha256"]
        mismatches, staging_manifest, staging_list = shared_mismatches(
            host=host,
            port=port,
            identity=identity,
            source_manifest=source_manifest,
            snapshot_sha=snapshot_sha,
        )
        overlay_file_count = len(mismatches)
        shared_reused_file_count = manifest["snapshot_file_count"] - overlay_file_count
        incoming = f"{REMOTE_PARENT}/.incoming_e188_{stage}_{snapshot_sha[:16]}.tar.zst"
        archive = Path()
        mismatch_list = Path()
        try:
            ssh(
                host,
                port,
                identity,
                f"mkdir -p {shlex.quote(destination)} {shlex.quote(str(Path(remote_manifest).parent))} && "
                f"rsync --archive --relative --ignore-missing-args --files-from={shlex.quote(staging_list)} {shlex.quote(REMOTE_SHARED + '/')} {shlex.quote(destination + '/')}",
            )
            if mismatches:
                archive, archive_sha, archive_size, mismatch_list = compress_selected(mismatches)
                transfer_chunk_count = parallel_chunk_rsync(
                    archive,
                    archive_sha=archive_sha,
                    incoming=incoming,
                    host=host,
                    port=port,
                    identity=identity,
                )
                ssh(
                    host,
                    port,
                    identity,
                    f"zstd --decompress --stdout {shlex.quote(incoming)} | tar --extract --file - --directory {shlex.quote(destination)}",
                )
            command(("rsync", "--archive", "-e", rsync_remote(host, port, identity), str(source_manifest), f"batchcom@{host}:{remote_manifest}"))
        finally:
            if archive != Path():
                archive.unlink(missing_ok=True)
            if mismatch_list != Path():
                mismatch_list.unlink(missing_ok=True)
    module = "workspace/core4d/scripts/experiments/E188/deploy_remote_a100.py"
    remote_cmd = f"cd {shlex.quote(destination)} && {shlex.quote(REMOTE_PYTHON)} {module} verify --root {shlex.quote(destination)} --manifest {shlex.quote(remote_manifest)}"
    verification = json.loads(ssh(host, port, identity, remote_cmd).stdout.splitlines()[-1])
    if verification["status"] != "PASS":
        raise RuntimeError(f"remote snapshot verification failed: {verification}")
    if not exists:
        cleanup = [staging_manifest, staging_list]
        if overlay_file_count:
            cleanup.append(incoming)
        ssh(host, port, identity, "rm -f " + " ".join(shlex.quote(value) for value in cleanup))
    env = ssh(host, port, identity, f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote('import json,torch,mujoco; print(json.dumps({\"cuda\":torch.cuda.is_available(),\"torch\":torch.__version__,\"mujoco\":mujoco.__version__}))')}").stdout.strip()
    payload = {"schema": "e188_a100_deployment_v1", "experiment_id": "E188", "stage": stage, "status": "PASS", "remote_host": host, "remote_port": port, "remote_root": destination, "remote_python": REMOTE_PYTHON, "source_snapshot_sha256": manifest["source_snapshot_sha256"], "source_manifest_sha256": sha256(source_manifest), "snapshot_file_count": manifest["snapshot_file_count"], "snapshot_size_bytes": manifest["snapshot_size_bytes"], "transfer_mode": "remote_shared_sha_reuse_plus_manifest_selected_zstd_parallel_chunks", "shared_reused_file_count": shared_reused_file_count, "overlay_file_count": overlay_file_count, "transfer_chunk_count": transfer_chunk_count, "transfer_archive_sha256": archive_sha, "transfer_archive_size_bytes": archive_size, "verification": verification, "environment": json.loads(env), "rsync_delete_used": False, "shared_checkout_mutated": False, "existing_processes_modified": False}
    write_json(root / "remote_deployment_manifest.json", payload, immutable=True)
    return payload


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("gpu-check")
    check.add_argument("--stage", choices=("canary", "full"), required=True)
    check.add_argument("--round", choices=("first", "second"), required=True)
    check.add_argument("--policy-gpus", default="4,5")
    overlap = sub.add_parser("overlap-policy")
    overlap.add_argument("--stage", choices=("canary", "full"), required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--stage", choices=("canary", "full"), required=True)
    deploy_parser = sub.add_parser("deploy")
    deploy_parser.add_argument("--stage", choices=("canary", "full"), required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.mode == "gpu-check":
        payload = gpu_check(args.stage, args.round, host=args.host, port=args.port, identity=args.identity, policy_gpus=args.policy_gpus)
        printable = {"status": payload["status"], "stage": args.stage, "allowed_gpus": payload["allowed_gpus"]}
    elif args.mode == "overlap-policy":
        payload = freeze_overlap_policy(args.stage)
        printable = {"status": payload["status"], "stage": args.stage, "physical_gpus": payload["physical_gpus"]}
    elif args.mode == "freeze":
        payload = build_snapshot(args.stage)
        printable = {key: payload[key] for key in ("status", "stage", "snapshot_file_count", "snapshot_size_bytes", "source_snapshot_sha256")}
    elif args.mode == "deploy":
        payload = deploy(args.stage, host=args.host, port=args.port, identity=args.identity)
        printable = {key: payload[key] for key in ("status", "stage", "remote_root", "snapshot_file_count", "source_snapshot_sha256")}
    else:
        payload = verify_snapshot(args.root, args.manifest)
        printable = payload
    print(json.dumps(printable, sort_keys=True, separators=(",", ":")))
    return 0 if payload["status"] in {"PASS", "FROZEN", "PREFLIGHT_PASS_1", "USER_AUTHORIZED_OVERLAP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
