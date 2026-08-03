#!/usr/bin/env python3
"""Freeze E187's E178-compatible keep22 authority and isolation inventory."""

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
EXPERIMENT_ID = "E187"
METHOD_ID = "E187_canonical_distance_continuation_reward_r1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "workspace/core4d/results/E187"
PLAN_PATH = (
    REPO_ROOT
    / "workspace/core4d/plan/205_E187_canonical_distance_continuation_reward_plan.md"
)
E178_MANIFEST = (
    REPO_ROOT
    / "workspace/core4d/results/E178/s6_downstream/manifests"
    / "semantic_bucket_full_manifest.tsv"
)
E186_KEEP = (
    REPO_ROOT / "workspace/core4d/results/E186/s5_handoff/keep22_authority_manifest.tsv"
)
E186_DROP = (
    REPO_ROOT / "workspace/core4d/results/E186/s5_handoff/p_rejected5_manifest.tsv"
)
E186_COLLIDER_LOCK = (
    REPO_ROOT / "workspace/core4d/results/E186/s0_environment/collider_lock.json"
)
E186_PROTOCOL = (
    REPO_ROOT / "workspace/core4d/results/E186/s0_environment/protocol_manifest.json"
)

EXPECTED_SHA256 = {
    "e178_manifest": "de9a3d165301318049da208aad52b1f5bc4d3ed671631f0097958734a0f022a8",
    "e186_keep22": "35d028f3ae6465ca8602d5973ead782000115f50f463247ab44ac581c6aeb535",
    "e186_drop5": "baa2dc1184de7e8d25fb9acb7026e6ebd2f53bb563cb8ff6829c789d80fea604",
    "e186_collider_lock": "6a20df7c2b47d0d5a5ea104bdd18ab129753db7c3a14e5f82aa0ce93a1a66065",
    "e186_protocol": "e74ca8904f6a55aa7b76ff96815c8d4b089f0d9087b4327b1cc2f88e22495795",
    "e187_plan": "9cc2fc9d275af5a53c850a1f99a4fef919d3d60195a0569837c1837a553b15fe",
}
EXPECTED_KEEP_COUNTS = {"bucket003": 5, "bucket004": 4, "bucket007": 13}
E178_ROW_ARTIFACT_FIELDS = (
    ("target_scene", "target_scene"),
    ("trajectory", "trajectory"),
    ("contact_mask", "contact_mask"),
    ("override", "override_path"),
    ("scene_act", "scene_act"),
    ("result_npz", "result_npz"),
    ("outdir_npz", "outdir_npz"),
    ("config_act", "config_act"),
    ("video", "video"),
    ("log", "log"),
)
E178_SHARED_ARTIFACTS = (
    E178_MANIFEST,
    REPO_ROOT / "workspace/core4d/log/237_E178_bucket_contact_aligned_proxy_gates.md",
    REPO_ROOT / "workspace/core4d/log/238_E178_canary_waiver_full_launch.md",
    REPO_ROOT / "workspace/core4d/log/239_E178_local_5090_hybrid_rebalance.md",
    REPO_ROOT / "workspace/core4d/log/240_E178_tracking_error_numeric_gates.md",
    REPO_ROOT / "workspace/core4d/log/241_E178_bucket_user_manual_review_results.md",
)
KEEP_OUTPUT_FIELDS = (
    "authority_row_index",
    "source_ordinal",
    "case_id",
    "object_key",
    "retarget_variant_id",
    "target_variant_id",
    "hand_collision_variant_id",
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
    "collider_manifest",
    "collider_manifest_sha256",
    "collider_asset_sha256",
    "collider_actual_hulls",
    "source_exp_id",
    "source_method_id",
    "spider_method_id",
    "reward_score_mode",
    "e187_status",
)
INVENTORY_FIELDS = ("case_id", "artifact_kind", "path", "sha256", "size_bytes")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


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
        stream, delimiter="\t", fieldnames=list(fields), lineterminator="\n"
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
            raise RuntimeError(f"immutable E187 artifact mismatch: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _verify_source_hashes() -> None:
    sources = {
        "e178_manifest": E178_MANIFEST,
        "e186_keep22": E186_KEEP,
        "e186_drop5": E186_DROP,
        "e186_collider_lock": E186_COLLIDER_LOCK,
        "e186_protocol": E186_PROTOCOL,
        "e187_plan": PLAN_PATH,
    }
    for name, path in sources.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_SHA256[name]:
            raise RuntimeError(f"{name} missing or SHA changed: {path}")


def _validate_authority() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    _verify_source_hashes()
    _, e178_rows = _read_tsv(E178_MANIFEST)
    _, keep_rows = _read_tsv(E186_KEEP)
    _, drop_rows = _read_tsv(E186_DROP)
    if len(e178_rows) != 27 or len(keep_rows) != 22 or len(drop_rows) != 5:
        raise RuntimeError("E178/E186 authority counts changed")
    if len({row["case_id"] for row in e178_rows}) != 27:
        raise RuntimeError("E178 case IDs are not unique")
    keep_ids = [row["case_id"] for row in keep_rows]
    drop_ids = {row["case_id"] for row in drop_rows}
    projected = [row["case_id"] for row in e178_rows if row["case_id"] not in drop_ids]
    if keep_ids != projected or set(keep_ids) & drop_ids:
        raise RuntimeError("E186 keep22 is not the ordered E178 minus drop5 projection")
    if dict(Counter(row["object_key"] for row in keep_rows)) != EXPECTED_KEEP_COUNTS:
        raise RuntimeError("keep22 object counts changed")

    e178_by_case = {row["case_id"]: row for row in e178_rows}
    compare_fields = (
        ("object_key", "object_key"),
        ("retarget_variant_id", "retarget_variant_id"),
        ("target_variant_id", "target_variant_id"),
        ("hand_collision_variant_id", "hand_collision_variant_id"),
        ("target_task", "target_task"),
        ("target_scene", "target_scene"),
        ("trajectory", "trajectory"),
        ("contact_mask", "contact_mask"),
        ("source_e178_scene_act", "scene_act"),
        ("source_e178_scene_sha256", "effective_scene_sha256"),
        ("trajectory_sha256", "trajectory_sha256"),
        ("contact_mask_sha256", "contact_mask_sha256"),
        ("cem_samples", "cem_samples"),
        ("cem_opt_steps", "cem_opt_steps"),
        ("cem_seed", "cem_seed"),
    )
    for keep in keep_rows:
        source = e178_by_case[keep["case_id"]]
        for keep_field, source_field in compare_fields:
            if keep[keep_field] != source[source_field]:
                raise RuntimeError(
                    f"{keep['case_id']}: authority mismatch {keep_field}/{source_field}"
                )
        if (keep["cem_samples"], keep["cem_opt_steps"], keep["cem_seed"]) != (
            "1024",
            "32",
            "0",
        ):
            raise RuntimeError(f"{keep['case_id']}: CEM contract changed")
    return e178_rows, keep_rows


def _inventory(e178_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in e178_rows:
        case_id = source["case_id"]
        for kind, field in E178_ROW_ARTIFACT_FIELDS:
            path = _repo_path(source[field])
            if not path.is_file():
                raise RuntimeError(f"{case_id}: missing E178 {kind}: {path}")
            expected_field = {
                "override": "override_sha256",
                "scene_act": "effective_scene_sha256",
                "trajectory": "trajectory_sha256",
                "contact_mask": "contact_mask_sha256",
            }.get(kind)
            digest = sha256_file(path)
            if expected_field and digest != source[expected_field]:
                raise RuntimeError(f"{case_id}: E178 {kind} SHA changed")
            rows.append(
                {
                    "case_id": case_id,
                    "artifact_kind": kind,
                    "path": _display_path(path),
                    "sha256": digest,
                    "size_bytes": str(path.stat().st_size),
                }
            )
    for path in E178_SHARED_ARTIFACTS:
        if not path.is_file():
            raise RuntimeError(f"missing E178 shared artifact: {path}")
        rows.append(
            {
                "case_id": "__shared__",
                "artifact_kind": "shared_authority",
                "path": _display_path(path),
                "sha256": sha256_file(path),
                "size_bytes": str(path.stat().st_size),
            }
        )
    return rows


def _keep_projection(keep_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    projected = []
    for source in keep_rows:
        row = {field: source.get(field, "") for field in KEEP_OUTPUT_FIELDS}
        row.update(
            {
                "source_exp_id": "E186",
                "source_method_id": source["spider_method_id"],
                "spider_method_id": METHOD_ID,
                "reward_score_mode": "distance_continuation",
                "e187_status": "AUTHORITY_FROZEN_CONFIG_PENDING",
            }
        )
        projected.append(row)
    return projected


def protocol_payload() -> dict[str, Any]:
    """Return E187's result-independent preregistered protocol."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "method_id": METHOD_ID,
        "status": "AUTHORITY_FROZEN",
        "scope": "E178_COMPATIBLE_KEEP22_DISTANCE_CONTINUATION_FULL",
        "sources": {
            name: {"path": _display_path(path), "sha256": EXPECTED_SHA256[name]}
            for name, path in {
                "e178_manifest": E178_MANIFEST,
                "e186_keep22": E186_KEEP,
                "e186_drop5": E186_DROP,
                "e186_collider_lock": E186_COLLIDER_LOCK,
                "e186_protocol": E186_PROTOCOL,
                "e187_plan": PLAN_PATH,
            }.items()
        },
        "expected_counts": {
            "full": 27,
            "keep": 22,
            "drop": 5,
            "keep_by_object": EXPECTED_KEEP_COUNTS,
        },
        "reward": {
            "mode": "distance_continuation",
            "far_weight": 0.25,
            "near_weight": 0.75,
            "far_scale_m": 0.050,
            "near_scale_m": 0.015,
            "smooth_delta_m": 0.001,
            "scale": 1.5,
            "temporal_gate": "contact_mask",
            "decay_frac": 0.15,
        },
        "full_contract": {
            "cem_samples": 1024,
            "cem_opt_steps": 32,
            "cem_seed": 0,
            "recorder": "off",
            "resources": [
                "local_gpu0",
                "spider_remote_ada_gpu0",
                "spider_remote_ada_gpu1",
            ],
            "use_a100": False,
            "existing_process_policy": "coexist_no_kill_pause_or_preempt",
        },
        "freeze_rules": {
            "full_results_may_retune_reward": False,
            "full_results_may_retune_grid": False,
            "full_results_may_reselect_collider": False,
            "drop_cases_may_enter_downstream": False,
        },
    }


def freeze(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Create or byte-validate all immutable S0 authority artifacts."""
    e178_rows, keep_rows = _validate_authority()
    environment = output_root / "s0_environment"
    outputs = {
        "protocol": environment / "protocol_manifest.json",
        "keep22": environment / "keep22_protocol_manifest.tsv",
        "inventory": environment / "e178_authority_sha_inventory.tsv",
    }
    _write_immutable(outputs["protocol"], _json_bytes(protocol_payload()))
    _write_immutable(
        outputs["keep22"],
        _tsv_bytes(_keep_projection(keep_rows), KEEP_OUTPUT_FIELDS),
    )
    inventory = _inventory(e178_rows)
    _write_immutable(outputs["inventory"], _tsv_bytes(inventory, INVENTORY_FIELDS))
    return {
        "status": "PASS",
        "keep": len(keep_rows),
        "inventory_rows": len(inventory),
        "artifacts": {
            name: {
                "path": _display_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }


def validate(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Rebuild authority in memory and require byte-identical frozen outputs."""
    return freeze(output_root)


def main() -> int:
    """CLI entry point for the S0 authority freeze."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
