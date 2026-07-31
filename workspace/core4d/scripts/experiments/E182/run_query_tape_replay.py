#!/usr/bin/env python3
"""Run E178-matched dev3 CEM replays for E182 query-tape validation."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from e182_common import REPO_ROOT, atomic_json, relative_to_repo, sha256_file

from spider.query_tape import finalize_cem_query_tape

AUTHORITY = REPO_ROOT / "workspace/core4d/results/E182/authority/dev3/manifest.tsv"
E178_SOURCE = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
EXPECTED_SOURCE_SHA256 = (
    "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8"
)
DEFAULT_RESULT_ROOT = REPO_ROOT / "workspace/core4d/results/E182/s1_query_tape"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read TSV rows."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_dev_rows() -> list[dict[str, str]]:
    """Join immutable dev3 authority to E178 runtime fields."""
    if sha256_file(E178_SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("E178 source manifest SHA changed")
    authority_rows = read_rows(AUTHORITY)
    source_by_case = {row["case_id"]: row for row in read_rows(E178_SOURCE)}
    joined = []
    for authority_row in authority_rows:
        source = source_by_case[authority_row["case_id"]]
        if source["cem_samples"] != "1024" or source["cem_opt_steps"] != "32":
            raise RuntimeError(f"{source['case_id']}: invalid E178 budget authority")
        joined.append(
            {
                **authority_row,
                "override_id": source["override_id"],
                "source_config_id": source["source_config_id"],
                "e178_result_npz": source["result_npz"],
                "e178_config_act": source["config_act"],
            }
        )
    return joined


def build_command(
    *,
    row: dict[str, str],
    mode: str,
    python_bin: str,
    output_dir: Path,
    tape_root: Path,
    gpu_id: int,
) -> list[str]:
    """Build one exact E178-configured 64x4 replay command."""
    if mode not in {"off", "on_a", "on_b"}:
        raise ValueError(mode)
    command = [
        python_bin,
        "-u",
        "examples/run_mjwp.py",
        f"+override={row['override_id']}",
        f"task={row['target_task']}",
        "+use_torch_compile=false",
        "save_video=false",
        "video_camera=auto",
        f"output_dir={output_dir}",
        "num_samples=64",
        "max_num_iterations=4",
        "seed=0",
        f"device=cuda:{gpu_id}",
    ]
    if mode != "off":
        command.extend(
            (
                "+query_tape_enabled=true",
                f"+query_tape_output_dir={tape_root}",
                f"+query_tape_run_id={mode}/{row['case_id']}",
            )
        )
    return command


def run_one(
    *,
    row: dict[str, str],
    mode: str,
    python_bin: str,
    result_root: Path,
    gpu_id: int,
) -> dict[str, Any]:
    """Run or validate one resume-safe replay row."""
    output_dir = result_root / "replays" / mode / f"{row['case_id']}_outdir"
    result_path = output_dir / "trajectory_mjwp_act.npz"
    tape_root = result_root / "raw_chunks"
    chunk_manifest = tape_root / mode / row["case_id"] / "chunk_manifest.json"
    log_path = result_root / "logs" / mode / f"{row['case_id']}.log"
    tape_manifest = None
    if mode != "off" and result_path.is_file() and chunk_manifest.is_file():
        tape_manifest = finalize_cem_query_tape(
            SimpleNamespace(
                query_tape_enabled=True,
                query_tape_output_dir=str(tape_root),
                query_tape_run_id=f"{mode}/{row['case_id']}",
            ),
            provenance={
                "experiment_id": "E182",
                "stage": "S1_query_tape_replay",
                "case_id": row["case_id"],
                "mode": mode,
                "override_id": row["override_id"],
                "target_task": row["target_task"],
                "samples": 64,
                "opt_steps": 4,
                "seed": 0,
                "result": relative_to_repo(result_path),
                "result_sha256": sha256_file(result_path),
            },
        )
    complete = result_path.is_file() and (
        mode == "off" or (tape_manifest or {}).get("status") == "COMPLETE"
    )
    if complete:
        return {
            "case_id": row["case_id"],
            "mode": mode,
            "status": "SKIPPED_COMPLETE",
            "result": relative_to_repo(result_path),
            "result_sha256": sha256_file(result_path),
            "tape_content_sha256": (
                tape_manifest["content_sha256"] if tape_manifest is not None else None
            ),
        }
    if output_dir.exists() or (mode != "off" and chunk_manifest.parent.exists()):
        raise RuntimeError(
            f"refusing to append to incomplete replay; use a new run ID: {row['case_id']} {mode}"
        )
    output_dir.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        row=row,
        mode=mode,
        python_bin=python_bin,
        output_dir=output_dir,
        tape_root=tape_root,
        gpu_id=gpu_id,
    )
    env = os.environ.copy()
    started = __import__("time").time()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("# " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_seconds = __import__("time").time() - started
    if result.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            f"query replay failed case={row['case_id']} mode={mode} rc={result.returncode} log={log_path}"
        )
    if mode != "off" and not chunk_manifest.is_file():
        raise RuntimeError(f"query replay produced no chunk manifest: {chunk_manifest}")
    if mode != "off":
        tape_manifest = finalize_cem_query_tape(
            SimpleNamespace(
                query_tape_enabled=True,
                query_tape_output_dir=str(tape_root),
                query_tape_run_id=f"{mode}/{row['case_id']}",
            ),
            provenance={
                "experiment_id": "E182",
                "stage": "S1_query_tape_replay",
                "case_id": row["case_id"],
                "mode": mode,
                "override_id": row["override_id"],
                "target_task": row["target_task"],
                "samples": 64,
                "opt_steps": 4,
                "seed": 0,
                "result": relative_to_repo(result_path),
                "result_sha256": sha256_file(result_path),
            },
        )
    return {
        "case_id": row["case_id"],
        "mode": mode,
        "status": "PASS",
        "wall_seconds": wall_seconds,
        "command": command,
        "result": relative_to_repo(result_path),
        "result_sha256": sha256_file(result_path),
        "tape_content_sha256": (
            tape_manifest["content_sha256"] if tape_manifest is not None else None
        ),
        "chunk_manifest": (
            relative_to_repo(chunk_manifest) if chunk_manifest.is_file() else None
        ),
        "log": relative_to_repo(log_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("off", "on_a", "on_b"), required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Run selected dev3 replay rows sequentially on one GPU."""
    args = parse_args()
    rows = load_dev_rows()
    if args.case_id:
        requested = set(args.case_id)
        rows = [row for row in rows if row["case_id"] in requested]
        if {row["case_id"] for row in rows} != requested:
            raise RuntimeError("one or more requested cases are outside dev3")
    results = [
        run_one(
            row=row,
            mode=args.mode,
            python_bin=args.python_bin,
            result_root=args.result_root,
            gpu_id=args.gpu_id,
        )
        for row in rows
    ]
    manifest = {
        "experiment_id": "E182",
        "stage": "S1_query_tape_replay",
        "mode": args.mode,
        "status": "PASS",
        "budget": {"samples": 64, "opt_steps": 4, "seed": 0},
        "rows": results,
    }
    output = args.result_root / "replays" / args.mode / "run_manifest.json"
    atomic_json(output, manifest)
    print(f"E182_QUERY_REPLAY=PASS mode={args.mode} rows={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
