#!/usr/bin/env python3
"""Read-only remote environment probes for E168 worker profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO / "workspace/core4d/results/E168"
ENVIRONMENT_MANIFEST = RESULT_ROOT / "s0_environment/environment_manifest.json"
PHASE0_SUMMARY = RESULT_ROOT / "s0_environment/phase0_summary.json"

REMOTE_PROBE = r"""
import json
import os
import subprocess

repo_root = os.environ["E168_REPO_ROOT"]
holosoma_root = os.environ["E168_HOLOSOMA_ROOT"]
raw_root = os.environ.get("E168_RAW_ROOT", "")
smplx_root = os.environ.get("E168_SMPLX_ROOT", "")

def command(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR:{type(exc).__name__}:{exc}"

def git_state(path):
    if not os.path.isdir(path):
        return {"path": path, "exists": False}
    head = command(["git", "rev-parse", "HEAD"], cwd=path)
    dirty = command(["git", "status", "--porcelain"], cwd=path)
    return {
        "path": path,
        "exists": True,
        "head": head,
        "dirty": bool(dirty) and not dirty.startswith("ERROR:"),
        "dirty_paths": dirty.splitlines() if dirty and not dirty.startswith("ERROR:") else [],
    }

gpu_raw = command([
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
])
compute_raw = command([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
])
gpus = []
uuid_to_index = {}
if not gpu_raw.startswith("ERROR:"):
    for line in gpu_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        gpu = {
            "index": int(parts[0]),
            "uuid": parts[1],
            "name": parts[2],
            "memory_total_mb": int(parts[3]),
            "memory_used_mb": int(parts[4]),
            "utilization_gpu_pct": int(parts[5]),
            "compute_processes": [],
        }
        gpus.append(gpu)
        uuid_to_index[parts[1]] = gpu["index"]

if compute_raw and not compute_raw.startswith("ERROR:"):
    for line in compute_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or parts[0] not in uuid_to_index:
            continue
        pid = parts[1]
        owner = command(["ps", "-o", "user=", "-p", pid]).strip()
        process = {
            "pid": int(pid),
            "owner": owner if owner and not owner.startswith("ERROR:") else "unknown",
            "process_name": parts[2],
            "used_gpu_memory_mb": int(parts[3]),
        }
        gpus[uuid_to_index[parts[0]]]["compute_processes"].append(process)

print(json.dumps({
    "hostname": command(["hostname"]),
    "repo": git_state(repo_root),
    "holosoma": git_state(holosoma_root),
    "raw_root": {"path": raw_root, "exists": bool(raw_root) and os.path.isdir(raw_root)},
    "smplx_root": {"path": smplx_root, "exists": bool(smplx_root) and os.path.isdir(smplx_root)},
    "gpus": gpus,
}))
"""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def probe(
    ssh_args: list[str],
    *,
    repo_root: str,
    holosoma_root: str,
    raw_root: str,
    smplx_root: str,
) -> dict[str, Any]:
    remote_command = (
        f"E168_REPO_ROOT={repo_root!r} "
        f"E168_HOLOSOMA_ROOT={holosoma_root!r} "
        f"E168_RAW_ROOT={raw_root!r} "
        f"E168_SMPLX_ROOT={smplx_root!r} "
        "python3 -"
    )
    try:
        result = subprocess.run(
            [*ssh_args, remote_command],
            input=REMOTE_PROBE,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["probe_status"] = "pass"
        return payload
    except Exception as exc:  # noqa: BLE001 - connection errors belong in evidence.
        return {
            "probe_status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }


def gpu_sets(
    gpus: list[dict[str, Any]],
    *,
    memory_limit_mb: int,
) -> tuple[list[int], list[int]]:
    low_memory = [
        gpu["index"] for gpu in gpus if gpu["memory_used_mb"] < memory_limit_mb
    ]
    no_process = [
        gpu["index"]
        for gpu in gpus
        if gpu["memory_used_mb"] < memory_limit_mb and not gpu["compute_processes"]
    ]
    return low_memory, no_process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a100-policy-gpus", default="")
    parser.add_argument("--a100-memory-limit-mb", type=int, default=5000)
    parser.add_argument("--a100-max-gpus", type=int, default=4)
    args = parser.parse_args()

    if not ENVIRONMENT_MANIFEST.is_file() or not PHASE0_SUMMARY.is_file():
        raise SystemExit("run E168 phase0 before probing remote profiles")

    a6000 = probe(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "spider-remote"],
        repo_root="/home/xiayb/pHRI_workspace/spider",
        holosoma_root="/home/xiayb/pHRI_workspace/holosoma",
        raw_root="",
        smplx_root="",
    )
    a6000["profile"] = "a6000-2gpu"
    a6000["role"] = "CEM worker; exact staged inputs only"
    if a6000.get("probe_status") == "pass":
        _, a6000_available = gpu_sets(a6000["gpus"], memory_limit_mb=5000)
        a6000["available_gpus_snapshot"] = a6000_available
        a6000["scheduling_status"] = (
            "available_snapshot" if a6000_available else "unavailable_busy_snapshot"
        )

    a100 = probe(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=12",
            "-p",
            "30409",
            "-i",
            str(Path("~/.ssh/id_rsa_tianyiyun").expanduser()),
            "batchcom@61.172.170.106",
        ],
        repo_root="/home/dataset-assist-0/xiayb/workspace/spider",
        holosoma_root="/home/dataset-assist-0/xiayb/workspace/holosoma",
        raw_root="",
        smplx_root="",
    )
    a100["profile"] = "A100-8gpu"
    a100["role"] = "CEM worker; exact staged inputs only"
    policy = sorted(
        {
            int(value)
            for value in args.a100_policy_gpus.split(",")
            if value.strip()
        }
    )
    if a100.get("probe_status") == "pass":
        low_memory, no_process = gpu_sets(
            a100["gpus"], memory_limit_mb=args.a100_memory_limit_mb
        )
        selected = sorted(set(no_process) & set(policy))[: args.a100_max_gpus]
        a100.update(
            {
                "memory_limit_mb": args.a100_memory_limit_mb,
                "max_gpus": args.a100_max_gpus,
                "low_memory_candidates": low_memory,
                "no_process_candidates": no_process,
                "policy_allowlist": policy,
                "selected_gpus_snapshot": selected,
                "scheduling_status": (
                    "available_snapshot"
                    if selected
                    else (
                        "disabled_policy_allowlist_missing"
                        if not policy
                        else "unavailable_after_policy_intersection"
                    )
                ),
                "launch_requires_fresh_recheck": True,
            }
        )

    probe_root = RESULT_ROOT / "s0_environment/remote_profiles"
    write_json(probe_root / "a6000_probe.json", a6000)
    write_json(probe_root / "a100_probe.json", a100)

    environment = json.loads(ENVIRONMENT_MANIFEST.read_text(encoding="utf-8"))
    environment["remote_profiles"] = {
        "a6000-2gpu": a6000,
        "A100-8gpu": a100,
    }
    probes_pass = all(
        profile.get("probe_status") == "pass" for profile in (a6000, a100)
    )
    environment["status"] = (
        "pass_with_a100_disabled"
        if probes_pass and not policy
        else ("pass" if probes_pass else "fail")
    )
    environment["updated_at"] = now()
    write_json(ENVIRONMENT_MANIFEST, environment)

    summary = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "updated_at": now(),
            "environment_status": environment["status"],
            "status": environment["status"],
            "hard_stops": (
                ["A100 launch disabled until policy allowlist is supplied"]
                if probes_pass and not policy
                else ([] if probes_pass else ["remote profile probe failed"])
            ),
            "next_allowed_action": (
                "local CPU data stages are allowed; all GPU launchers must recheck availability"
                if probes_pass
                else "repair remote connectivity before execution"
            ),
        }
    )
    write_json(PHASE0_SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
