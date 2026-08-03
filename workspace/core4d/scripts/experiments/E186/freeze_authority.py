#!/usr/bin/env python3
"""Freeze the E186 keep22 authority and three object-specific colliders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ID = "E186"
METHOD_ID = "E186_object_specific_coacd_compound_grid_sdf_r1"
P_FLOOR = 0.70
E178_MANIFEST = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
E183_CASE_TABLE = (
    REPO_ROOT
    / "workspace/core4d/results/E183/full27_static_p"
    / "case_candidate_metrics.tsv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E186"
PLAN_PATH = (
    REPO_ROOT / "workspace/core4d/plan/204_E186_22case_object_specific_prg_full_plan.md"
)
EXPECTED_E178_SHA256 = (
    "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8"
)
EXPECTED_E183_CASE_SHA256 = (
    "28a3c8118fe10767856b5d6bcdb93430dc9ba629d7f4bb58eead9924168a949c"
)
EXPECTED_PLAN_SHA256 = (
    "25091ba7183f081f0fe0585d34de49772dc1cde089db438839cdd02904ae4fc3"
)
EXPECTED_OBJECT_COUNTS = {"bucket003": 9, "bucket004": 4, "bucket007": 14}
EXPECTED_KEEP_COUNTS = {"bucket003": 5, "bucket004": 4, "bucket007": 13}
EXPECTED_DROPPED = {
    "bucket003_20231018_001_p1",
    "bucket003_20231018_005_p1",
    "bucket003_20231020_068_p1",
    "bucket003_20231018_003_p2",
    "bucket007_20231003_2_021_p2",
}
COLLIDER_SPECS = {
    "bucket003": {
        "candidate_key": "E181__bucket003__t020_k16_v032",
        "candidate_id": "t020_k16_v032",
        "manifest": (
            "workspace/core4d/results/E181/s2_coacd/bucket003/"
            "t020_k16_v032/manifest.json"
        ),
        "manifest_sha256": (
            "10b0bb91f0fb48d85b62b6b0dd066a8627c794d0814b7527176ddab1ae543803"
        ),
        "candidate_asset_sha256": (
            "ca7fbe33c82180636889dc8639e1de1a28b1814feae1b750f0874f734351c4e1"
        ),
        "max_hulls": 16,
        "actual_hulls": 16,
    },
    "bucket004": {
        "candidate_key": "E181__bucket004__t005_k08_v064",
        "candidate_id": "t005_k08_v064",
        "manifest": (
            "workspace/core4d/results/E181/s2_coacd/bucket004/"
            "t005_k08_v064/manifest.json"
        ),
        "manifest_sha256": (
            "18acdf1b6970f0c769fa304547f40dd29663b0d22aad4f6ac3dbb8f7ddc4af1b"
        ),
        "candidate_asset_sha256": (
            "327999b33a8a32aad79df8343c0393b94350072402fa25bbb23bf56951f68853"
        ),
        "max_hulls": 8,
        "actual_hulls": 8,
    },
    "bucket007": {
        "candidate_key": "E181__bucket007__t020_k08_v064",
        "candidate_id": "t020_k08_v064",
        "manifest": (
            "workspace/core4d/results/E181/s2_coacd/bucket007/"
            "t020_k08_v064/manifest.json"
        ),
        "manifest_sha256": (
            "0c5dd38d76514d2b944061169ce80d132fedc4fc05808b0a77d07d17c5ea406a"
        ),
        "candidate_asset_sha256": (
            "9b04f068f580259e74d0e74c0a57f9174f8f18710fe2e87e6a88c0e4b13abfed"
        ),
        "max_hulls": 8,
        "actual_hulls": 8,
    },
}
KEEP_FIELDS = (
    "authority_row_index",
    "source_ordinal",
    "case_id",
    "object_key",
    "date",
    "seq",
    "person",
    "retarget_variant_id",
    "selected_retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
    "source_exp_id",
    "spider_method_id",
    "contact_mask_label",
    "target_task",
    "target_scene",
    "trajectory",
    "contact_mask",
    "source_e178_scene_act",
    "source_e178_scene_sha256",
    "trajectory_sha256",
    "contact_mask_sha256",
    "cem_samples",
    "cem_opt_steps",
    "cem_seed",
    "collider_candidate_key",
    "collider_candidate_id",
    "collider_manifest",
    "collider_manifest_sha256",
    "collider_asset_sha256",
    "collider_actual_hulls",
    "p_gate_floor",
    "p_true_positive_count",
    "p_phantom_contact_count",
    "p_missed_contact_count",
    "p_precision",
    "p_recall",
    "p_zero_oracle",
    "selection_decision",
    "e186_status",
)
DROP_FIELDS = KEEP_FIELDS


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"missing TSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _tsv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        delimiter="\t",
        fieldnames=list(fields),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_immutable(path: Path, payload: bytes) -> None:
    """Create an artifact or require byte-identical content on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable E186 artifact mismatch: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _precision_recall(tp: int, phantom: int, missed: int) -> tuple[float, float, bool]:
    if min(tp, phantom, missed) < 0:
        raise ValueError("negative confusion count")
    oracle = tp + missed
    predicted = tp + phantom
    precision = tp / predicted if predicted else 1.0
    recall = tp / oracle if oracle else 1.0
    return precision, recall, oracle == 0


def p_gate(tp: int, phantom: int, missed: int, floor: float = P_FLOOR) -> bool:
    """Apply positive P/R floors or the explicit zero-oracle contract."""
    precision, recall, zero_oracle = _precision_recall(tp, phantom, missed)
    return phantom == 0 if zero_oracle else precision >= floor and recall >= floor


def _validate_source() -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    if sha256_file(E178_MANIFEST) != EXPECTED_E178_SHA256:
        raise RuntimeError("E178 Full manifest SHA changed")
    if sha256_file(E183_CASE_TABLE) != EXPECTED_E183_CASE_SHA256:
        raise RuntimeError("E183 case table SHA changed")
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("E186 plan changed after preregistration")
    source_fields, source_rows = _read_tsv(E178_MANIFEST)
    _, metric_rows = _read_tsv(E183_CASE_TABLE)
    if len(source_rows) != 27 or len({row["case_id"] for row in source_rows}) != 27:
        raise RuntimeError("E178 Full27 closure changed")
    counts = Counter(row["object_key"] for row in source_rows)
    if dict(counts) != EXPECTED_OBJECT_COUNTS:
        raise RuntimeError(f"E178 object counts changed: {dict(counts)}")
    for row in source_rows:
        case_id = row["case_id"]
        if (row["cem_samples"], row["cem_opt_steps"], row["cem_seed"]) != (
            "1024",
            "32",
            "0",
        ):
            raise RuntimeError(f"{case_id}: CEM contract changed")
        for field, sha_field in (
            ("trajectory", "trajectory_sha256"),
            ("contact_mask", "contact_mask_sha256"),
            ("scene_act", "effective_scene_sha256"),
        ):
            path = _repo_path(row[field])
            if not path.is_file() or sha256_file(path) != row[sha_field]:
                raise RuntimeError(f"{case_id}: {field} missing or SHA changed")
    return source_fields, source_rows, metric_rows


def _load_colliders() -> dict[str, dict[str, Any]]:
    locks: dict[str, dict[str, Any]] = {}
    for object_key, spec in COLLIDER_SPECS.items():
        manifest_path = _repo_path(str(spec["manifest"]))
        if sha256_file(manifest_path) != spec["manifest_sha256"]:
            raise RuntimeError(f"{object_key}: candidate manifest SHA changed")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "BUILD_PASS"
            or payload.get("object_key") != object_key
            or payload.get("candidate_id") != spec["candidate_id"]
            or payload.get("candidate_asset_sha256") != spec["candidate_asset_sha256"]
            or int(payload.get("hull_count", -1)) != spec["actual_hulls"]
            or int(payload["parameters"]["max_convex_hull"]) != spec["max_hulls"]
        ):
            raise RuntimeError(f"{object_key}: candidate manifest contract changed")
        parts: list[dict[str, Any]] = []
        for expected_index, part in enumerate(payload["parts"]):
            path = _repo_path(part["path"])
            if int(part["part_index"]) != expected_index:
                raise RuntimeError(f"{object_key}: non-canonical part ordering")
            if not path.is_file() or sha256_file(path) != part["sha256"]:
                raise RuntimeError(f"{object_key}: part SHA changed: {path}")
            parts.append(
                {
                    "part_index": expected_index,
                    "path": _display_path(path),
                    "sha256": part["sha256"],
                    "vertex_count": int(part["vertex_count"]),
                    "face_count": int(part["face_count"]),
                }
            )
        if len(parts) != spec["actual_hulls"]:
            raise RuntimeError(f"{object_key}: part count changed")
        locks[object_key] = {
            "object_key": object_key,
            "candidate_key": spec["candidate_key"],
            "candidate_id": spec["candidate_id"],
            "manifest": _display_path(manifest_path),
            "manifest_sha256": spec["manifest_sha256"],
            "candidate_asset_sha256": spec["candidate_asset_sha256"],
            "max_hulls": spec["max_hulls"],
            "actual_hulls": spec["actual_hulls"],
            "max_ch_vertex": int(payload["parameters"]["max_ch_vertex"]),
            "threshold_m": float(payload["parameters"]["threshold_m"]),
            "ordered_parts": parts,
            "ordered_parts_sha256": _canonical_digest(parts),
        }
    return locks


def _selection_rows(
    source_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    colliders: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    selected_metrics = {
        (row["candidate_key"], row["case_id"]): row
        for row in metric_rows
        if row["candidate_key"]
        in {spec["candidate_key"] for spec in COLLIDER_SPECS.values()}
    }
    if len(selected_metrics) != 27:
        raise RuntimeError(
            f"selected P evidence is not 27 rows: {len(selected_metrics)}"
        )
    evidence: list[dict[str, str]] = []
    keeps: list[dict[str, str]] = []
    drops: list[dict[str, str]] = []
    for authority_index, source in enumerate(source_rows, start=1):
        object_key = source["object_key"]
        collider = colliders[object_key]
        metric = selected_metrics.get((collider["candidate_key"], source["case_id"]))
        if metric is None or metric["object_key"] != object_key:
            raise RuntimeError(
                f"missing object-matched P evidence: {source['case_id']}"
            )
        tp = int(metric["true_positive_count"])
        phantom = int(metric["phantom_contact_count"])
        missed = int(metric["missed_contact_count"])
        precision, recall, zero_oracle = _precision_recall(tp, phantom, missed)
        passed = p_gate(tp, phantom, missed)
        decision = "KEEP_P_PASS" if passed else "DROP_P_REJECTED"
        row = {
            "authority_row_index": str(authority_index),
            "source_ordinal": source["ordinal"],
            "case_id": source["case_id"],
            "object_key": object_key,
            "date": source["date"],
            "seq": source["seq"],
            "person": source["person"],
            "retarget_variant_id": source["retarget_variant_id"],
            "selected_retarget_variant_id": source["selected_retarget_variant_id"],
            "target_variant_id": source["target_variant_id"],
            "hand_collision_variant_id": source["hand_collision_variant_id"],
            "source_exp_id": "E178",
            "spider_method_id": METHOD_ID,
            "contact_mask_label": source["contact_mask_label"],
            "target_task": source["target_task"],
            "target_scene": source["target_scene"],
            "trajectory": source["trajectory"],
            "contact_mask": source["contact_mask"],
            "source_e178_scene_act": source["scene_act"],
            "source_e178_scene_sha256": source["effective_scene_sha256"],
            "trajectory_sha256": source["trajectory_sha256"],
            "contact_mask_sha256": source["contact_mask_sha256"],
            "cem_samples": source["cem_samples"],
            "cem_opt_steps": source["cem_opt_steps"],
            "cem_seed": source["cem_seed"],
            "collider_candidate_key": str(collider["candidate_key"]),
            "collider_candidate_id": str(collider["candidate_id"]),
            "collider_manifest": str(collider["manifest"]),
            "collider_manifest_sha256": str(collider["manifest_sha256"]),
            "collider_asset_sha256": str(collider["candidate_asset_sha256"]),
            "collider_actual_hulls": str(collider["actual_hulls"]),
            "p_gate_floor": f"{P_FLOOR:.2f}",
            "p_true_positive_count": str(tp),
            "p_phantom_contact_count": str(phantom),
            "p_missed_contact_count": str(missed),
            "p_precision": f"{precision:.9f}",
            "p_recall": f"{recall:.9f}" if not zero_oracle else "NA",
            "p_zero_oracle": str(zero_oracle).lower(),
            "selection_decision": decision,
            "e186_status": "P_FROZEN_R_G_PENDING" if passed else "P_REJECTED",
        }
        evidence.append(row)
        (keeps if passed else drops).append(row)
    if len(keeps) != 22 or len(drops) != 5:
        raise RuntimeError(f"unexpected keep/drop counts: {len(keeps)}/{len(drops)}")
    if {row["case_id"] for row in drops} != EXPECTED_DROPPED:
        raise RuntimeError("drop5 case set changed")
    keep_counts = Counter(row["object_key"] for row in keeps)
    if dict(keep_counts) != EXPECTED_KEEP_COUNTS:
        raise RuntimeError(f"keep object counts changed: {dict(keep_counts)}")
    return evidence, keeps, drops


def protocol_payload() -> dict[str, Any]:
    """Return the result-independent frozen protocol."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN",
        "scope": "KEEP22_OBJECT_SPECIFIC_P_R_G_FULL",
        "method_id": METHOD_ID,
        "selection_gate": {
            "positive_case": "precision>=0.70 AND recall>=0.70",
            "zero_oracle": "phantom==0",
        },
        "expected_counts": {
            "full": 27,
            "keep": 22,
            "drop": 5,
            "keep_by_object": EXPECTED_KEEP_COUNTS,
        },
        "sources": {
            "e178_manifest": {
                "path": _display_path(E178_MANIFEST),
                "sha256": EXPECTED_E178_SHA256,
            },
            "e183_case_table": {
                "path": _display_path(E183_CASE_TABLE),
                "sha256": EXPECTED_E183_CASE_SHA256,
            },
            "plan": {
                "path": _display_path(PLAN_PATH),
                "sha256": EXPECTED_PLAN_SHA256,
            },
        },
        "colliders": {
            key: {
                name: value
                for name, value in spec.items()
                if name != "candidate_asset_sha256"
            }
            | {"candidate_asset_sha256": spec["candidate_asset_sha256"]}
            for key, spec in COLLIDER_SPECS.items()
        },
        "full_contract": {
            "retarget_variant_id": "omnirt_v1",
            "target_variant_id": "ref_fk",
            "hand_collision_variant_id": "rubber_hull",
            "cem_samples": 1024,
            "cem_opt_steps": 32,
            "cem_seed": 0,
            "recorder": "off",
            "resources": [
                "local_gpu0",
                "spider_remote_ada_gpu0",
                "spider_remote_ada_gpu1",
            ],
            "existing_process_policy": "coexist_no_kill_pause_or_preempt",
        },
        "freeze_rules": {
            "drop_cases_may_enter_downstream": False,
            "full_results_may_reselect_collider": False,
            "full_results_may_retune_grid": False,
            "k4_in_scope": False,
        },
        "runner": {
            "path": _display_path(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def freeze(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Freeze protocol, collider lock, keep/drop manifests, and aggregate."""
    _, source_rows, metric_rows = _validate_source()
    colliders = _load_colliders()
    evidence, keeps, drops = _selection_rows(source_rows, metric_rows, colliders)

    protocol_path = output_root / "s0_environment/protocol_manifest.json"
    collider_path = output_root / "s0_environment/collider_lock.json"
    evidence_path = output_root / "s0_environment/p_selection_evidence.tsv"
    keep_path = output_root / "s5_handoff/keep22_authority_manifest.tsv"
    drop_path = output_root / "s5_handoff/p_rejected5_manifest.tsv"
    _write_immutable(protocol_path, _json_bytes(protocol_payload()))
    collider_payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN",
        "collider_count": 3,
        "source_of_truth": "ordered E181 CoACD part union",
        "objects": colliders,
        "collider_set_sha256": _canonical_digest(colliders),
    }
    _write_immutable(collider_path, _json_bytes(collider_payload))
    _write_immutable(evidence_path, _tsv_bytes(evidence, KEEP_FIELDS))
    _write_immutable(keep_path, _tsv_bytes(keeps, KEEP_FIELDS))
    _write_immutable(drop_path, _tsv_bytes(drops, DROP_FIELDS))
    aggregate = {
        "experiment_id": EXPERIMENT_ID,
        "status": "AUTHORITY_FROZEN",
        "stage": "S0_AUTHORITY",
        "case_counts": {
            "full": 27,
            "keep": len(keeps),
            "drop": len(drops),
            "keep_by_object": dict(Counter(row["object_key"] for row in keeps)),
        },
        "dropped_case_ids": [row["case_id"] for row in drops],
        "collider_set_sha256": collider_payload["collider_set_sha256"],
        "artifacts": {
            "protocol": _artifact(protocol_path),
            "collider_lock": _artifact(collider_path),
            "p_selection_evidence": _artifact(evidence_path),
            "keep22_authority": _artifact(keep_path),
            "p_rejected5": _artifact(drop_path),
        },
        "next_gate": "S1_CANONICAL_GRID_SDF",
    }
    aggregate_path = output_root / "s0_environment/authority_aggregate.json"
    _write_immutable(aggregate_path, _json_bytes(aggregate))
    return aggregate


def validate(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Recompute authority and require every immutable artifact to match."""
    aggregate = freeze(output_root)
    keep_path = output_root / "s5_handoff/keep22_authority_manifest.tsv"
    drop_path = output_root / "s5_handoff/p_rejected5_manifest.tsv"
    _, keeps = _read_tsv(keep_path)
    _, drops = _read_tsv(drop_path)
    keep_ids = {row["case_id"] for row in keeps}
    drop_ids = {row["case_id"] for row in drops}
    if len(keep_ids) != 22 or len(drop_ids) != 5 or keep_ids & drop_ids:
        raise RuntimeError("E186 keep/drop partition is invalid")
    if any(row["e186_status"] != "P_FROZEN_R_G_PENDING" for row in keeps):
        raise RuntimeError("keep22 contains a non-frozen row")
    if any(row["e186_status"] != "P_REJECTED" for row in drops):
        raise RuntimeError("drop5 contains a non-rejected row")
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": "S0_AUTHORITY",
        "status": "PASS",
        "keep": len(keeps),
        "drop": len(drops),
        "aggregate_sha256": sha256_file(
            output_root / "s0_environment/authority_aggregate.json"
        ),
        "collider_set_sha256": aggregate["collider_set_sha256"],
    }


def parse_args() -> argparse.Namespace:
    """Parse the authority-freeze command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "validate", "all"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Run the requested authority stage."""
    args = parse_args()
    if args.command in {"freeze", "all"}:
        result = freeze(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.command in {"validate", "all"}:
        result = validate(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
