#!/usr/bin/env python3
"""Build a paired S6 RL input by reusing passing Stage2b partner motions.

The standard ``rl_export_input.tsv`` remains the source-person/CEM authority.
This adapter adds the opposite person from the same object/date/sequence and
writes a separate paired input. It is downstream-only and does not modify the
S1-S5 registry or CEM evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "core4d_data_construction_v3.0"
PERSON_PARTNER = {"person1": "person2", "person2": "person1"}
PERSON_INDEX = {"person1": "0", "person2": "1"}
PERSON_SHORT = {"person1": "p1", "person2": "p2"}
DEFAULT_VARIANT_PREFERENCE = ["omnirt_v1", "omnirt_v2"]

PARTNER_FIELDS = [
    "source_case_id",
    "source_person",
    "source_person_idx",
    "source_rl_export_decision",
    "partner_case_id",
    "partner_person",
    "partner_person_idx",
    "object_key",
    "object_name",
    "date",
    "seq",
    "pair_status",
    "partner_status",
    "paired_rl_export_decision",
    "partner_retarget_variant_id",
    "partner_target_variant_id",
    "generation_mode",
    "stage2b_status",
    "failure_mode",
    "decision_notes",
    "partner_target_task",
    "partner_provenance_ref",
    "partner_provenance_sha256",
    "partner_params_json",
    "holosoma_case_root",
    "converted_npz",
    "converted_npz_sha256",
    "omniretarget_output_npz",
    "omniretarget_output_npz_sha256",
    "retargeted_npz",
    "retargeted_npz_sha256",
    "trimmed_npz",
    "trimmed_npz_sha256",
    "trim_window_json",
    "trim_window_json_sha256",
    "trim_start",
    "trim_end",
    "trim_frames",
    "untrimmed_frames",
    "trimmed_frames",
    "source_rl_export_input",
    "source_rl_export_input_sha256",
    "schema_version",
    "updated_at",
]

PAIRED_EXTRA_FIELDS = [
    "pair_status",
    "paired_rl_export_decision",
    "partner_case_id",
    "partner_person",
    "partner_person_idx",
    "partner_status",
    "partner_retarget_variant_id",
    "partner_target_variant_id",
    "partner_generation_mode",
    "partner_trimmed_npz",
    "partner_trimmed_npz_sha256",
    "partner_omniretarget_output_npz",
    "partner_omniretarget_output_npz_sha256",
    "partner_trim_window_json",
    "partner_trim_window_json_sha256",
    "partner_manifest_ref",
    "partner_manifest_sha256",
    "trajectory_sha256",
    "scene_act_sha256",
    "contact_mask_sha256",
    "cem_result_sha256",
]


def timestamp() -> str:
    """Return a timezone-aware timestamp with second precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a TSV and preserve its declared field order."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a stable TSV schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    """Hash one required artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(value: str | Path, repo: Path) -> Path:
    """Resolve local, repo-relative, and historical remote absolute paths."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return repo / path
    if path.exists():
        return path
    text = path.as_posix()
    for marker, prefix in (
        ("/workspace/core4d/", "workspace/core4d/"),
        ("/example_datasets/", "example_datasets/"),
    ):
        if marker in text:
            return repo / f"{prefix}{text.split(marker, 1)[1]}"
    return path


def require_file(value: str | Path, label: str, repo: Path) -> Path:
    """Resolve and require a non-empty artifact."""
    path = local_path(value, repo)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty {label}: {path}")
    return path


def path_ref(value: str | Path, repo: Path) -> str:
    """Prefer stable repo-relative references in generated manifests."""
    path = local_path(value, repo).absolute()
    try:
        return path.relative_to(repo.absolute()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def cli_path(value: Path, repo: Path) -> Path:
    """Resolve CLI paths without dereferencing repo-internal symlinks."""
    expanded = value.expanduser()
    return expanded if expanded.is_absolute() else repo / expanded


def infer_partner(source: dict[str, str]) -> tuple[str, str, str]:
    """Infer the opposite CORE4D person in the same sequence."""
    partner_person = PERSON_PARTNER.get(source.get("person", ""))
    if not partner_person:
        raise SystemExit(
            f"unsupported source person for {source.get('case_id', '')}: "
            f"{source.get('person', '')}"
        )
    case_id = source.get("case_id", "")
    expected_suffix = f"_{PERSON_SHORT.get(source.get('person', ''), '')}"
    if not expected_suffix.strip("_") or not case_id.endswith(expected_suffix):
        raise SystemExit(f"case/person suffix mismatch: {case_id}")
    partner_case = case_id[: -len(expected_suffix)] + f"_{PERSON_SHORT[partner_person]}"
    return partner_case, partner_person, PERSON_INDEX[partner_person]


def validate_source_rows(
    rows: list[dict[str, str]],
    *,
    object_key: str,
    expected_rows: int | None,
) -> None:
    """Require one exact object scope containing only standard ready rows."""
    case_ids = [row.get("case_id", "") for row in rows]
    if not rows or len(case_ids) != len(set(case_ids)) or not all(case_ids):
        raise SystemExit("source RL input must contain non-empty unique case_id rows")
    if expected_rows is not None and len(rows) != expected_rows:
        raise SystemExit(
            f"source row count mismatch: expected {expected_rows}, got {len(rows)}"
        )
    invalid = [
        case_id
        for case_id, row in zip(case_ids, rows, strict=True)
        if row.get("object_key") != object_key
        or row.get("rl_export_decision") != "RL_EXPORT_READY"
    ]
    if invalid:
        raise SystemExit(
            f"source RL input contains out-of-scope/non-ready rows: {invalid}"
        )


def validate_review_authority(
    review_path: Path | None,
    source_rows: list[dict[str, str]],
    repo: Path,
) -> dict[str, Any]:
    """Optionally prove that the source set equals the manual USE set."""
    if review_path is None:
        return {
            "authority_review_snapshot": "",
            "authority_review_snapshot_sha256": "",
            "approved_source_set_exact": None,
        }
    _, reviews = read_tsv(review_path)
    approved = {
        row.get("case_id", "")
        for row in reviews
        if row.get("manual_use_decision") == "USE"
    }
    source_ids = {row["case_id"] for row in source_rows}
    if source_ids != approved:
        raise SystemExit(
            "source RL input does not exactly match manual USE authority: "
            f"source_only={sorted(source_ids - approved)}, "
            f"authority_only={sorted(approved - source_ids)}"
        )
    return {
        "authority_review_snapshot": path_ref(review_path, repo),
        "authority_review_snapshot_sha256": sha256(review_path),
        "approved_source_set_exact": True,
    }


def load_stage2b_index(
    manifests: list[Path],
) -> dict[str, list[tuple[Path, dict[str, str]]]]:
    """Index only passing Stage2b rows while retaining provenance."""
    index: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    for manifest in manifests:
        _, rows = read_tsv(manifest)
        for row in rows:
            if row.get("stage2b_status") == "pass":
                index.setdefault(row.get("case_id", ""), []).append((manifest, row))
    return index


def choose_partner(
    candidates: list[tuple[Path, dict[str, str]]],
    variant_preference: list[str],
    partner_case: str,
) -> tuple[Path, dict[str, str]]:
    """Choose one passing route deterministically, preferring production v1."""
    preference = {variant: rank for rank, variant in enumerate(variant_preference)}
    eligible = [
        item
        for item in candidates
        if item[1].get("retarget_variant_id") in preference
        and item[1].get("target_variant_id") == "ref_fk"
    ]
    eligible.sort(
        key=lambda item: (
            preference[item[1]["retarget_variant_id"]],
            item[1].get("updated_at", ""),
            item[0].as_posix(),
        )
    )
    if not eligible:
        raise SystemExit(f"no passing ref_fk Stage2b partner: {partner_case}")
    best_rank = preference[eligible[0][1]["retarget_variant_id"]]
    same_rank = [
        item
        for item in eligible
        if preference[item[1]["retarget_variant_id"]] == best_rank
    ]
    if len(same_rank) != 1:
        raise SystemExit(
            f"ambiguous passing Stage2b partner at preferred variant: {partner_case}"
        )
    return same_rank[0]


def trim_window_path(
    evidence: dict[str, str],
    partner_case: str,
    repo: Path,
) -> Path:
    """Resolve the trim sidecar from the row or its Holosoma case root."""
    explicit = evidence.get("trim_window_json", "")
    if explicit:
        return require_file(explicit, f"{partner_case} trim_window_json", repo)
    case_root = local_path(evidence.get("holosoma_case_root", ""), repo)
    return require_file(
        case_root / "trim_window.json",
        f"{partner_case} trim_window_json",
        repo,
    )


def build_partner_row(
    source: dict[str, str],
    *,
    partner_case: str,
    partner_person: str,
    partner_person_idx: str,
    evidence: dict[str, str],
    provenance: Path,
    source_rl: Path,
    repo: Path,
    generation_mode: str,
) -> dict[str, Any]:
    """Build one hash-pinned partner row from passing Stage2b evidence."""
    for key, expected in (
        ("case_id", partner_case),
        ("object_key", source["object_key"]),
        ("date", source["date"]),
        ("seq", source["seq"]),
        ("person", partner_person),
        ("person_idx", partner_person_idx),
        ("stage2b_status", "pass"),
    ):
        if evidence.get(key, "") != expected:
            raise SystemExit(
                f"partner identity mismatch for {partner_case}: "
                f"{key}={evidence.get(key, '')!r}, expected={expected!r}"
            )

    artifacts = {
        field: require_file(evidence.get(field, ""), f"{partner_case} {field}", repo)
        for field in (
            "converted_npz",
            "omniretarget_output_npz",
            "retargeted_npz",
            "trimmed_npz",
        )
    }
    trim_path = trim_window_path(evidence, partner_case, repo)
    with trim_path.open("r", encoding="utf-8") as stream:
        trim = json.load(stream)

    variant = evidence["retarget_variant_id"]
    decision_notes = (
        f"reused passing Stage2b partner; selected {variant} by "
        f"production-v1-then-v2 preference"
    )
    row: dict[str, Any] = {
        "source_case_id": source["case_id"],
        "source_person": source["person"],
        "source_person_idx": source["person_idx"],
        "source_rl_export_decision": source["rl_export_decision"],
        "partner_case_id": partner_case,
        "partner_person": partner_person,
        "partner_person_idx": partner_person_idx,
        "object_key": source["object_key"],
        "object_name": source["object_name"],
        "date": source["date"],
        "seq": source["seq"],
        "pair_status": "PAIR_COMPLETE",
        "partner_status": "pass",
        "paired_rl_export_decision": "RL_EXPORT_READY",
        "partner_retarget_variant_id": variant,
        "partner_target_variant_id": evidence["target_variant_id"],
        "generation_mode": generation_mode,
        "stage2b_status": evidence["stage2b_status"],
        "failure_mode": "",
        "decision_notes": decision_notes,
        "partner_target_task": evidence.get("target_task", ""),
        "partner_provenance_ref": path_ref(provenance, repo),
        "partner_provenance_sha256": sha256(provenance),
        "partner_params_json": evidence.get("params_json", ""),
        "holosoma_case_root": path_ref(evidence["holosoma_case_root"], repo),
        "trim_window_json": path_ref(trim_path, repo),
        "trim_window_json_sha256": sha256(trim_path),
        "trim_start": trim.get("trim_start", ""),
        "trim_end": trim.get("trim_end", ""),
        "trim_frames": trim.get("trim_frames", ""),
        "untrimmed_frames": trim.get("untrimmed_frames", ""),
        "trimmed_frames": trim.get("trimmed_frames", ""),
        "source_rl_export_input": path_ref(source_rl, repo),
        "source_rl_export_input_sha256": sha256(source_rl),
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp(),
    }
    for field, path in artifacts.items():
        row[field] = path_ref(path, repo)
        row[f"{field}_sha256"] = sha256(path)
    return row


def source_hashes(source: dict[str, str], repo: Path) -> dict[str, str]:
    """Hash the four standard source-person RL assets."""
    mapping = {
        "trajectory_sha256": "trajectory",
        "scene_act_sha256": "scene_act",
        "contact_mask_sha256": "contact_mask",
        "cem_result_sha256": "cem_result_npz",
    }
    return {
        output: sha256(
            require_file(
                source.get(field, ""),
                f"{source['case_id']} source {field}",
                repo,
            )
        )
        for output, field in mapping.items()
    }


def paired_row(
    source: dict[str, str],
    partner: dict[str, Any],
    *,
    manifest_ref: str,
    manifest_hash: str,
    repo: Path,
) -> dict[str, Any]:
    """Join one standard source row with its partner contract."""
    return {
        **source,
        "pair_status": partner["pair_status"],
        "paired_rl_export_decision": partner["paired_rl_export_decision"],
        "partner_case_id": partner["partner_case_id"],
        "partner_person": partner["partner_person"],
        "partner_person_idx": partner["partner_person_idx"],
        "partner_status": partner["partner_status"],
        "partner_retarget_variant_id": partner["partner_retarget_variant_id"],
        "partner_target_variant_id": partner["partner_target_variant_id"],
        "partner_generation_mode": partner["generation_mode"],
        "partner_trimmed_npz": partner["trimmed_npz"],
        "partner_trimmed_npz_sha256": partner["trimmed_npz_sha256"],
        "partner_omniretarget_output_npz": partner["omniretarget_output_npz"],
        "partner_omniretarget_output_npz_sha256": partner[
            "omniretarget_output_npz_sha256"
        ],
        "partner_trim_window_json": partner["trim_window_json"],
        "partner_trim_window_json_sha256": partner["trim_window_json_sha256"],
        "partner_manifest_ref": manifest_ref,
        "partner_manifest_sha256": manifest_hash,
        **source_hashes(source, repo),
    }


def markdown_summary(
    *,
    experiment_id: str,
    object_key: str,
    partner_rows: list[dict[str, Any]],
    paired_path: str,
) -> str:
    """Render a compact human audit summary."""
    variant_counts = Counter(row["partner_retarget_variant_id"] for row in partner_rows)
    lines = [
        f"# {experiment_id} {object_key} paired RL-ready export",
        "",
        f"- source rows: `{len(partner_rows)}`",
        f"- partner rows: `{len(partner_rows)}`",
        f"- paired `RL_EXPORT_READY`: `{len(partner_rows)}`",
        f"- partner variants: `{dict(variant_counts)}`",
        f"- paired input: `{paired_path}`",
        "",
        "| source | partner | variant | status |",
        "|---|---|---|---|",
    ]
    for row in partner_rows:
        lines.append(
            f"| `{row['source_case_id']}` | `{row['partner_case_id']}` | "
            f"`{row['partner_retarget_variant_id']}` | `{row['pair_status']}` |"
        )
    lines.extend(
        [
            "",
            "This package reuses passing Stage2b partner artifacts. It prepares "
            "paired RL inputs only; it does not claim RL training success or "
            "rewrite S1-S5 facts.",
        ]
    )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    value = argparse.ArgumentParser()
    value.add_argument("--repo", type=Path, default=Path.cwd())
    value.add_argument("--experiment-id", required=True)
    value.add_argument("--object-key", required=True)
    value.add_argument("--rl-export-input-tsv", type=Path, required=True)
    value.add_argument("--out-dir", type=Path, required=True)
    value.add_argument(
        "--stage2b-manifest-tsv", type=Path, action="append", required=True
    )
    value.add_argument("--manual-review-snapshot", type=Path)
    value.add_argument("--expected-rows", type=int)
    value.add_argument(
        "--variant-preference",
        action="append",
        default=[],
        help="Repeat in preferred order; defaults to omnirt_v1 then omnirt_v2.",
    )
    return value


def main() -> int:
    """Create and audit the paired S6 package."""
    args = parser().parse_args()
    repo = args.repo.expanduser().resolve()
    source_rl = require_file(
        cli_path(args.rl_export_input_tsv, repo), "source RL input", repo
    )
    out_dir = cli_path(args.out_dir, repo)
    manifests = [
        require_file(cli_path(path, repo), "Stage2b manifest", repo)
        for path in args.stage2b_manifest_tsv
    ]
    review_path = (
        require_file(
            cli_path(args.manual_review_snapshot, repo),
            "manual review snapshot",
            repo,
        )
        if args.manual_review_snapshot
        else None
    )
    variant_preference = args.variant_preference or DEFAULT_VARIANT_PREFERENCE

    source_fields, source_rows = read_tsv(source_rl)
    validate_source_rows(
        source_rows,
        object_key=args.object_key,
        expected_rows=args.expected_rows,
    )
    authority_audit = validate_review_authority(review_path, source_rows, repo)
    stage2b_index = load_stage2b_index(manifests)
    generation_mode = f"reuse_{args.experiment_id.lower()}_stage2b_partner"

    partner_rows: list[dict[str, Any]] = []
    for source in source_rows:
        partner_case, partner_person, partner_person_idx = infer_partner(source)
        provenance, evidence = choose_partner(
            stage2b_index.get(partner_case, []),
            variant_preference,
            partner_case,
        )
        partner_rows.append(
            build_partner_row(
                source,
                partner_case=partner_case,
                partner_person=partner_person,
                partner_person_idx=partner_person_idx,
                evidence=evidence,
                provenance=provenance,
                source_rl=source_rl,
                repo=repo,
                generation_mode=generation_mode,
            )
        )

    source_ids = {row["source_case_id"] for row in partner_rows}
    partner_ids = {row["partner_case_id"] for row in partner_rows}
    if len(source_ids) != len(partner_rows) or len(partner_ids) != len(partner_rows):
        raise SystemExit("paired output requires unique source and partner case ids")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_tsv = out_dir / "rl_partner_omnirt_manifest.tsv"
    manifest_json = out_dir / "rl_partner_omnirt_manifest.json"
    write_tsv(manifest_tsv, partner_rows, PARTNER_FIELDS)
    write_json(manifest_json, partner_rows)
    manifest_hash = sha256(manifest_tsv)
    manifest_ref = path_ref(manifest_tsv, repo)

    paired_rows = [
        paired_row(
            source,
            partner,
            manifest_ref=manifest_ref,
            manifest_hash=manifest_hash,
            repo=repo,
        )
        for source, partner in zip(source_rows, partner_rows, strict=True)
    ]
    paired_tsv = out_dir / "paired_rl_export_input.tsv"
    paired_json = out_dir / "paired_rl_export_input.json"
    write_tsv(paired_tsv, paired_rows, source_fields + PAIRED_EXTRA_FIELDS)
    write_json(paired_json, paired_rows)

    variant_counts = Counter(row["partner_retarget_variant_id"] for row in partner_rows)
    audit = {
        "experiment_id": args.experiment_id,
        "object_key": args.object_key,
        "created_at": timestamp(),
        "source_rl_export_input": path_ref(source_rl, repo),
        "source_rl_export_input_sha256": sha256(source_rl),
        "source_rows": len(source_rows),
        "partner_rows": len(partner_rows),
        "same_sequence_opposite_person": True,
        "all_partner_artifacts_nonempty": True,
        "all_hashes_recomputed": True,
        "paired_rl_ready_rows": len(paired_rows),
        "partner_variant_counts": dict(variant_counts),
        "partner_manifest": manifest_ref,
        "partner_manifest_sha256": manifest_hash,
        "paired_rl_export_input": path_ref(paired_tsv, repo),
        "paired_rl_export_input_sha256": sha256(paired_tsv),
        **authority_audit,
    }
    write_json(out_dir / "paired_rl_export_audit.json", audit)
    write_json(out_dir / "rl_partner_omnirt_summary.json", audit)
    (out_dir / "rl_partner_omnirt_summary.md").write_text(
        markdown_summary(
            experiment_id=args.experiment_id,
            object_key=args.object_key,
            partner_rows=partner_rows,
            paired_path=path_ref(paired_tsv, repo),
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
