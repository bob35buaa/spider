"""Observational-only query-tape artifact writer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_COUNTERS: dict[Path, int] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _next_index(run_root: Path) -> int:
    if run_root in _COUNTERS:
        index = _COUNTERS[run_root]
    else:
        manifest_path = run_root / "chunk_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index = int(manifest["chunk_count"])
        else:
            index = 0
    _COUNTERS[run_root] = index + 1
    return index


def _run_root(config: Any) -> tuple[Path, str]:
    output_dir = str(config.query_tape_output_dir)
    run_id = str(config.query_tape_run_id)
    if not output_dir or not run_id:
        raise ValueError("query_tape_output_dir and query_tape_run_id are required")
    return Path(output_dir).resolve() / run_id, run_id


def cem_query_tape_chunk_count(config: Any) -> int:
    """Return the persisted chunk count for optional bounded-run control."""
    if not bool(config.query_tape_enabled):
        return 0
    run_root, _ = _run_root(config)
    manifest_path = run_root / "chunk_manifest.json"
    if not manifest_path.is_file():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(manifest.get("chunk_count", 0))


def record_cem_query_chunk(config: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Write one final-iteration CEM sample chunk without changing optimization."""
    if not bool(config.query_tape_enabled):
        raise RuntimeError("query tape recorder called while disabled")
    run_root, run_id = _run_root(config)
    if "qpos" not in payload or "rewards" not in payload:
        raise ValueError("query tape payload requires qpos and rewards")

    start_step = int(getattr(config, "query_tape_record_start_sim_step", 0))
    current_step = int(getattr(config, "_query_tape_current_sim_step", 0))
    if start_step < 0:
        raise ValueError("query_tape_record_start_sim_step must be non-negative")
    if current_step < start_step:
        return {
            "status": "SKIPPED_BEFORE_START",
            "current_sim_step": current_step,
            "record_start_sim_step": start_step,
        }

    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "chunk_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "COMPLETE":
            raise RuntimeError(f"refusing to append to complete query tape: {run_root}")
    else:
        existing_manifest = {"chunks": [], "chunk_count": 0}
    maximum_chunks = int(getattr(config, "query_tape_max_chunks", 0))
    if maximum_chunks < 0:
        raise ValueError("query_tape_max_chunks must be non-negative")
    if (
        maximum_chunks
        and int(existing_manifest.get("chunk_count", 0)) >= maximum_chunks
    ):
        return {
            "status": "SKIPPED_MAX_CHUNKS",
            "chunk_count": int(existing_manifest["chunk_count"]),
        }
    chunk_index = _next_index(run_root)
    chunk_path = run_root / f"chunk_{chunk_index:06d}.npz"
    if chunk_path.exists():
        raise RuntimeError(f"refusing to overwrite query chunk: {chunk_path}")
    arrays = {
        key: _to_numpy(value)
        for key, value in payload.items()
        if isinstance(value, (torch.Tensor, np.ndarray, int, float, bool))
    }
    temporary = chunk_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, chunk_path)
    entry = {
        "chunk_index": chunk_index,
        "path": str(chunk_path),
        "sha256": _sha256(chunk_path),
        "size_bytes": chunk_path.stat().st_size,
        "qpos_shape": list(arrays["qpos"].shape),
        "reward_shape": list(arrays["rewards"].shape),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"status": "RECORDING", "run_id": run_id, "chunks": []}
    manifest["chunks"].append(entry)
    manifest["chunk_count"] = len(manifest["chunks"])
    _atomic_json(manifest_path, manifest)
    return entry


def finalize_cem_query_tape(
    config: Any,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify and freeze one ordered query-tape manifest after a successful run."""
    if not bool(config.query_tape_enabled):
        raise RuntimeError("query tape finalizer called while disabled")
    run_root, run_id = _run_root(config)
    manifest_path = run_root / "chunk_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "COMPLETE":
        if provenance is not None and manifest.get("provenance") != provenance:
            raise RuntimeError("complete query tape provenance mismatch")
        return manifest
    if manifest.get("status") != "RECORDING":
        raise RuntimeError(f"invalid query tape status: {manifest.get('status')}")

    chunks = manifest.get("chunks", [])
    if not chunks or manifest.get("chunk_count") != len(chunks):
        raise RuntimeError("query tape has no chunks or an invalid chunk count")
    canonical_entries = []
    for expected_index, entry in enumerate(chunks):
        if int(entry["chunk_index"]) != expected_index:
            raise RuntimeError("query tape chunk indices are not contiguous")
        chunk_path = Path(entry["path"]).resolve()
        if chunk_path.parent != run_root or not chunk_path.is_file():
            raise RuntimeError(f"query tape chunk path is invalid: {chunk_path}")
        if chunk_path.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"query tape chunk size changed: {chunk_path}")
        actual_sha256 = _sha256(chunk_path)
        if actual_sha256 != entry["sha256"]:
            raise RuntimeError(f"query tape chunk SHA changed: {chunk_path}")
        canonical_entries.append(
            {
                "chunk_index": expected_index,
                "sha256": actual_sha256,
                "size_bytes": int(entry["size_bytes"]),
                "qpos_shape": entry["qpos_shape"],
                "reward_shape": entry["reward_shape"],
            }
        )

    content = json.dumps(
        canonical_entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["status"] = "COMPLETE"
    manifest["run_id"] = run_id
    manifest["content_sha256"] = hashlib.sha256(content).hexdigest()
    manifest["provenance"] = provenance or {}
    _atomic_json(manifest_path, manifest)
    return manifest
