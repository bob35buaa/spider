#!/usr/bin/env python3
"""Re-audit E186 static-P selection with analytic convex-union sign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_ROOT = REPO_ROOT / "workspace/core4d/scripts/experiments/E186"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from bake_canonical_grid_sdf import (  # noqa: E402
    _convex_union_halfspaces,
    _load_parts,
    _scene,
    _signed_distance,
    build_exact_union_mesh,
    load_collider_lock,
)
from freeze_authority import P_FLOOR, p_gate  # noqa: E402

from spider.geometry.grid_sdf import sha256_file  # noqa: E402

E183_ROOT = REPO_ROOT / "workspace/core4d/results/E183/full27_static_p"
QUERY_AGGREGATE = E183_ROOT / "query_aggregate.json"
OLD_CASE_METRICS = E183_ROOT / "case_candidate_metrics.tsv"
E186_SELECTION = (
    REPO_ROOT / "workspace/core4d/results/E186/s0_environment/p_selection_evidence.tsv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "workspace/core4d/results/E186/s0_environment/robust_p_reaudit_v3"
)
EXPECTED_QUERY_AGGREGATE_SHA256 = (
    "be891a41b76e3a7cfba6e3a801dfe6faf2f98bd9c9a89b78707328808a3c61c8"
)
EXPECTED_OLD_CASE_METRICS_SHA256 = (
    "28a3c8118fe10767856b5d6bcdb93430dc9ba629d7f4bb58eead9924168a949c"
)
EXPECTED_SELECTION_SHA256 = (
    "0efcbb98909295390de34013bb3631728e3043aab80ac1b27156ff881e08f090"
)
OUTPUT_FIELDS = (
    "authority_row_index",
    "case_id",
    "object_key",
    "candidate_key",
    "pose_count",
    "oracle_contact_count",
    "old_true_positive_count",
    "old_phantom_contact_count",
    "old_missed_contact_count",
    "old_precision",
    "old_recall",
    "old_gate_pass",
    "robust_true_positive_count",
    "robust_phantom_contact_count",
    "robust_missed_contact_count",
    "robust_true_negative_count",
    "robust_precision",
    "robust_recall",
    "robust_gate_pass",
    "selection_changed",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _display_path(path: Path) -> str:
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _tsv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        delimiter="\t",
        fieldnames=list(OUTPUT_FIELDS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable robust-P artifact mismatch: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def confusion(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    """Return pose-level confusion and zero-aware P-gate fields."""
    predicted = np.asarray(candidate, dtype=bool).reshape(-1)
    target = np.asarray(oracle, dtype=bool).reshape(-1)
    if predicted.shape != target.shape:
        raise ValueError("candidate/oracle contact shape mismatch")
    tp = int(np.count_nonzero(predicted & target))
    phantom = int(np.count_nonzero(predicted & ~target))
    missed = int(np.count_nonzero(~predicted & target))
    tn = int(np.count_nonzero(~predicted & ~target))
    precision = tp / (tp + phantom) if tp + phantom else 1.0
    recall = tp / (tp + missed) if tp + missed else 1.0
    return {
        "true_positive_count": tp,
        "phantom_contact_count": phantom,
        "missed_contact_count": missed,
        "true_negative_count": tn,
        "precision": precision,
        "recall": recall,
        "gate_pass": p_gate(tp, phantom, missed),
    }


def load_sources() -> tuple[
    list[dict[str, str]], dict[str, Any], dict[tuple[str, str], dict[str, str]]
]:
    """Load exact E183 query and E186 selection sources after SHA closure."""
    for path, expected in (
        (QUERY_AGGREGATE, EXPECTED_QUERY_AGGREGATE_SHA256),
        (OLD_CASE_METRICS, EXPECTED_OLD_CASE_METRICS_SHA256),
        (E186_SELECTION, EXPECTED_SELECTION_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"robust-P source SHA changed: {path}")
    selection = _read_tsv(E186_SELECTION)
    query = json.loads(QUERY_AGGREGATE.read_text(encoding="utf-8"))
    old_rows = _read_tsv(OLD_CASE_METRICS)
    if len(selection) != 27 or query.get("case_count") != 27:
        raise RuntimeError("robust-P Full27 source closure changed")
    if [row["case_id"] for row in selection] != [
        row["case_id"] for row in query["cases"]
    ]:
        raise RuntimeError("selection/query case order changed")
    old = {(row["candidate_key"], row["case_id"]): row for row in old_rows}
    return selection, query, old


def _load_query_case(
    entry: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = REPO_ROOT / entry["manifest"]["path"]
    if sha256_file(manifest_path) != entry["manifest"]["sha256"]:
        raise RuntimeError(f"query manifest SHA changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    point_blocks: list[np.ndarray] = []
    oracle_blocks: list[np.ndarray] = []
    radii: np.ndarray | None = None
    for family in ("reference", "e178_final"):
        family_entry = manifest["families"][family]
        path = root / family_entry["relative_path"]
        if sha256_file(path) != family_entry["sha256"]:
            raise RuntimeError(f"query family SHA changed: {path}")
        with np.load(path, allow_pickle=False) as values:
            point_blocks.append(np.asarray(values["points"], dtype=np.float32))
            oracle_blocks.append(np.asarray(values["oracle_contact"], dtype=bool))
            current = np.asarray(values["radii"], dtype=np.float32)
            if radii is None:
                radii = current
            elif not np.array_equal(radii, current):
                raise RuntimeError(f"query radii changed: {entry['case_id']}")
    assert radii is not None
    return np.concatenate(point_blocks), radii, np.concatenate(oracle_blocks)


def audit(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Recompute all 27 object-matched candidate contacts and freeze the audit."""
    selection, query, old = load_sources()
    lock = load_collider_lock()
    geometry: dict[str, tuple[Any, list[dict[str, np.ndarray]]]] = {}
    for object_key, collider in lock["objects"].items():
        parts = _load_parts(collider)
        union = build_exact_union_mesh(parts)
        geometry[object_key] = (_scene(union), _convex_union_halfspaces(parts))
    query_by_case = {row["case_id"]: row for row in query["cases"]}
    rows: list[dict[str, Any]] = []
    for selection_row in selection:
        case_id = selection_row["case_id"]
        object_key = selection_row["object_key"]
        candidate_key = selection_row["collider_candidate_key"]
        old_row = old.get((candidate_key, case_id))
        if old_row is None:
            raise RuntimeError(f"missing old candidate-case row: {case_id}")
        points, radii, oracle = _load_query_case(query_by_case[case_id])
        scene, volumes = geometry[object_key]
        signed = _signed_distance(scene, volumes, points, chunk_size=250_000)
        candidate = (signed - radii[None, :]).min(axis=1) <= 0.0
        robust = confusion(candidate, oracle)
        old_pass = selection_row["selection_decision"] == "KEEP_P_PASS"
        rows.append(
            {
                "authority_row_index": selection_row["authority_row_index"],
                "case_id": case_id,
                "object_key": object_key,
                "candidate_key": candidate_key,
                "pose_count": len(candidate),
                "oracle_contact_count": int(np.count_nonzero(oracle)),
                "old_true_positive_count": old_row["true_positive_count"],
                "old_phantom_contact_count": old_row["phantom_contact_count"],
                "old_missed_contact_count": old_row["missed_contact_count"],
                "old_precision": old_row["precision"],
                "old_recall": old_row["recall"],
                "old_gate_pass": str(old_pass).lower(),
                "robust_true_positive_count": robust["true_positive_count"],
                "robust_phantom_contact_count": robust["phantom_contact_count"],
                "robust_missed_contact_count": robust["missed_contact_count"],
                "robust_true_negative_count": robust["true_negative_count"],
                "robust_precision": f"{robust['precision']:.9f}",
                "robust_recall": f"{robust['recall']:.9f}",
                "robust_gate_pass": str(robust["gate_pass"]).lower(),
                "selection_changed": str(old_pass != robust["gate_pass"]).lower(),
            }
        )
    evidence_path = output_root / "case_metrics.tsv"
    _write_immutable(evidence_path, _tsv_bytes(rows))
    robust_keep = [row["case_id"] for row in rows if row["robust_gate_pass"] == "true"]
    old_keep = [row["case_id"] for row in rows if row["old_gate_pass"] == "true"]
    changed = [row["case_id"] for row in rows if row["selection_changed"] == "true"]
    aggregate = {
        "experiment_id": "E186",
        "stage": "S0_ROBUST_P_REAUDIT_V3",
        "status": "CONFIRMED" if robust_keep == old_keep else "AUTHORITY_INVALIDATED",
        "sign_authority": "ordered_convex_part_halfspace_union",
        "p_floor": P_FLOOR,
        "case_count": len(rows),
        "old_keep_count": len(old_keep),
        "robust_keep_count": len(robust_keep),
        "old_keep_by_object": dict(
            Counter(row["object_key"] for row in rows if row["old_gate_pass"] == "true")
        ),
        "robust_keep_by_object": dict(
            Counter(
                row["object_key"] for row in rows if row["robust_gate_pass"] == "true"
            )
        ),
        "selection_changed_case_ids": changed,
        "old_keep_case_ids": old_keep,
        "robust_keep_case_ids": robust_keep,
        "evidence": {
            "path": _display_path(evidence_path),
            "sha256": sha256_file(evidence_path),
        },
        "sources": {
            "query_aggregate_sha256": EXPECTED_QUERY_AGGREGATE_SHA256,
            "old_case_metrics_sha256": EXPECTED_OLD_CASE_METRICS_SHA256,
            "selection_sha256": EXPECTED_SELECTION_SHA256,
        },
    }
    aggregate_path = output_root / "aggregate.json"
    _write_immutable(aggregate_path, _json_bytes(aggregate))
    return aggregate


def parse_args() -> argparse.Namespace:
    """Parse the robust-P audit command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Run and print the robust-P authority audit."""
    result = audit(parse_args().output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
