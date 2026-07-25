"""Stable TSV and JSON writers for the InterPose audit."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interpose_audit_core import SequenceRecord


def _serialize_cell(value: Any) -> str | int | float:
    """Convert tuples and nested values into stable TSV cells."""
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_tsv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write stable UTF-8 TSV output."""
    row_list = list(rows)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row_list[0].keys()),
            delimiter="\t",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in row_list:
            writer.writerow({key: _serialize_cell(value) for key, value in row.items()})


def _inventory_row(
    record: SequenceRecord,
    multi_group_ids: frozenset[str],
    candidate_group_ids: frozenset[str],
) -> dict[str, Any]:
    """Add the conservative target decision to one sequence inventory row."""
    row = asdict(record)
    if record.group_id in candidate_group_ids:
        decision = "UNRESOLVED_HOH_CANDIDATE"
    elif record.group_id in multi_group_ids:
        decision = "UNRESOLVED_MULTI_PERSON"
    else:
        decision = "NON_TARGET_SINGLE_TRACK_GROUP"
    row["target_decision"] = decision
    return row


def _counter_rows(counter: Mapping[str, int], field: str) -> list[dict[str, Any]]:
    """Convert an ordered count mapping into rank rows."""
    return [
        {"rank": rank, field: label, "sequence_count": count}
        for rank, (label, count) in enumerate(counter.items(), start=1)
    ]


def write_outputs(
    output_root: Path,
    records: Sequence[SequenceRecord],
    summary: Mapping[str, Any],
    group_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    """Persist the machine-readable R001 audit authority."""
    output_root.mkdir(parents=True, exist_ok=True)
    multi_group_ids = frozenset(row["group_id"] for row in group_rows)
    candidate_group_ids = frozenset(row["group_id"] for row in pair_rows)
    write_tsv(
        output_root / "sequence_inventory.tsv",
        (
            _inventory_row(record, multi_group_ids, candidate_group_ids)
            for record in records
        ),
    )
    write_tsv(output_root / "multi_person_groups.tsv", group_rows)
    write_tsv(output_root / "hoh_pair_candidates.tsv", pair_rows)
    write_tsv(
        output_root / "object_counts.tsv",
        _counter_rows(summary["labels"]["object_sequence_counts"], "object_label"),
    )
    write_tsv(
        output_root / "action_counts.tsv",
        _counter_rows(summary["labels"]["action_sequence_counts"], "action_label"),
    )
    write_tsv(
        output_root / "verb_counts.tsv",
        _counter_rows(summary["labels"]["verb_lemma_counts"], "verb_lemma"),
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "scan_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
