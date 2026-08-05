#!/usr/bin/env python3
"""Shared frozen authority helpers for the E188 2-to-5 kg experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
E187 = REPO / "workspace/core4d/results/E187"
RESULTS = REPO / "workspace/core4d/results/E188"
S0 = RESULTS / "s0_environment"
E187_EVAL = E187 / "s6_downstream/manifests/e187_full_evaluation_manifest.tsv"
E187_QUEUE = E187 / "s5_full/queue/queue_manifest.json"
E187_OVERRIDES = E187 / "s3_prg_audit/production_overrides.json"
E187_METRICS = E187 / "s6_downstream/eval/full/e187_case_metrics.tsv"
UPSTREAM_SHA256 = {
    E187 / "s0_environment/keep22_protocol_manifest.tsv": "b1122a3a99d9760ff2bcd22b90783ca93999b17f620923e8ecef44801d6a4924",
    E187_QUEUE: "836b5388420f968e9c88968349a87799ae1ea3bd43d6276e631847c7956dcd92",
    E187_EVAL: "507c3a79766e61b49d5e298432bd0a766d9566f931ec9a2382f1a5b6e82c6fc7",
    E187 / "s2_canonical_grid_sdf/reward_grid_lock.json": "2cc949e19cb313e58883fd5d0446872220188ea9ecce06cb23d39e2a6b42194f",
    E187 / "s3_prg_audit/production_integration_lock.json": "400c98b422ac4458eccffb589eecc1d776fe5cd3968eb855cd89a485bb7d5441",
    E187_METRICS: "77405ebd72a02c7135d3b28945ffcf65e1fa2c0e2394bbba687e3c699000cb06",
}
WORKERS = ("local-0", "a100-4", "a100-5")
PHYSICAL_GPU = {"local-0": 0, "a100-4": 4, "a100-5": 5}
CANARY_BY_WORKER = {
    "local-0": "bucket003_20231018_003_p1",
    "a100-4": "bucket007_20231003_2_021_p1",
    "a100-5": "bucket007_20231020_055_p1",
}
QUEUE_CASES = {
    "local-0": (
        "bucket003_20231018_003_p1",
        "bucket007_20231018_021_p2",
        "bucket007_20231018_021_p1",
        "bucket007_20231018_019_p2",
    ),
    "a100-4": (
        "bucket007_20231003_2_021_p1",
        "bucket007_20231023_075_p2",
        "bucket007_20231003_1_021_p2",
        "bucket003_20231020_064_p1",
        "bucket007_20231003_1_021_p1",
        "bucket007_20231003_2_023_p1",
    ),
    "a100-5": (
        "bucket007_20231020_055_p1",
        "bucket007_20231023_073_p1",
        "bucket007_20231023_075_p1",
        "bucket007_20231020_059_p1",
        "bucket007_20231018_019_p1",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path = REPO) -> str:
    return path.absolute().relative_to(root.absolute()).as_posix()


def repo_path(value: str | Path, root: Path = REPO) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty TSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any], *, immutable: bool = False) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to replace frozen artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def verify_upstream(root: Path = REPO) -> dict[str, str]:
    verified: dict[str, str] = {}
    for source, expected in UPSTREAM_SHA256.items():
        path = root / source.relative_to(REPO)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen upstream SHA changed: {path} {actual} != {expected}")
        verified[relative(source)] = actual
    return verified


def e187_eval_rows(root: Path = REPO) -> list[dict[str, str]]:
    rows = read_tsv(root / E187_EVAL.relative_to(REPO))
    if len(rows) != 22 or len({row["case_id"] for row in rows}) != 22:
        raise RuntimeError("E187 evaluation authority is not 22 unique rows")
    return rows


def e187_queue_index(root: Path = REPO) -> dict[str, dict[str, Any]]:
    queue = json.loads((root / E187_QUEUE.relative_to(REPO)).read_text(encoding="utf-8"))
    output: dict[str, dict[str, Any]] = {}
    for worker in queue["worker_order"]:
        for row in queue["queues"][worker]:
            output[row["case_id"]] = {**row, "e187_worker": worker}
    if len(output) != 22:
        raise RuntimeError("E187 queue authority is not 22 unique rows")
    return output


def e187_override_index(root: Path = REPO) -> dict[str, dict[str, Any]]:
    payload = json.loads(
        (root / E187_OVERRIDES.relative_to(REPO)).read_text(encoding="utf-8")
    )
    rows = {row["case_id"]: row for row in payload["rows"]}
    if payload.get("status") != "PASS" or len(rows) != 22:
        raise RuntimeError("E187 production overrides are not 22 PASS rows")
    return rows


def queue_case_set() -> set[str]:
    rows = [case_id for worker in WORKERS for case_id in QUEUE_CASES[worker]]
    if len(rows) != 15 or len(set(rows)) != 15:
        raise RuntimeError("E188 fixed queues are not 15 unique rows")
    return set(rows)
