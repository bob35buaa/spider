"""Release-level aggregation for the InterPose audit."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from interpose_audit_core import SequenceRecord, build_shared_object_pairs
from interpose_vocab import EXPLICIT_HUMAN_CUES, JOINT_CUES, TRANSFER_CUES


def _source_summary(records: Sequence[SequenceRecord]) -> dict[str, Any]:
    """Aggregate release counts, durations, and quality states by source."""
    valid = [record for record in records if record.schema_status != "ERROR"]
    total_frames = sum(record.frames for record in valid)
    total_duration = sum(record.duration_seconds for record in valid)
    return {
        "sequences": len(records),
        "valid_sequences": len(valid),
        "frames": total_frames,
        "duration_seconds": total_duration,
        "duration_hours": total_duration / 3600.0,
        "mean_frames": total_frames / len(valid) if valid else 0.0,
        "mean_duration_seconds": total_duration / len(valid) if valid else 0.0,
        "bytes": sum(record.file_size_bytes for record in records),
    }


def _build_group_row(
    group_id: str,
    records: Sequence[SequenceRecord],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one multi-person clip-group audit row."""
    person_ids = sorted({record.person_id for record in records})
    strict_count = sum(row["tier"] == "HOH_TEXT_STRICT_CANDIDATE" for row in pair_rows)
    shared_count = len(pair_rows) - strict_count
    if strict_count:
        decision = "UNRESOLVED_TEXT_STRICT"
    elif shared_count:
        decision = "UNRESOLVED_SHARED_OBJECT"
    else:
        decision = "UNRESOLVED_MULTI_PERSON"
    return {
        "source": records[0].source,
        "group_id": group_id,
        "clip_id": records[0].clip_id,
        "sequence_count": len(records),
        "distinct_person_ids": len(person_ids),
        "person_ids": ",".join(str(value) for value in person_ids),
        "total_frames_unaligned": sum(record.frames for record in records),
        "strict_pair_candidates": strict_count,
        "shared_object_pair_candidates": shared_count,
        "decision": decision,
        "missing_required_evidence": "frame_ids,object_identity,object_trajectory",
    }


def _is_high_priority_text_candidate(row: Mapping[str, Any]) -> bool:
    """Require an explicit second-human cue plus transfer or joint-action cue."""
    cues = frozenset(str(row["relation_cues"]).split(","))
    has_explicit_human = bool(cues.intersection(EXPLICIT_HUMAN_CUES))
    has_interaction = bool(cues.intersection(TRANSFER_CUES | JOINT_CUES))
    return has_explicit_human and has_interaction


def _build_local_release(
    records: Sequence[SequenceRecord],
    groups: Mapping[str, Sequence[SequenceRecord]],
    by_source: Mapping[str, Sequence[SequenceRecord]],
) -> dict[str, Any]:
    """Build local release volume and schema metrics."""
    duration_seconds = sum(record.duration_seconds for record in records)
    return {
        "sequence_count": len(records),
        "clip_group_count": len(groups),
        "frame_count": sum(record.frames for record in records),
        "duration_seconds": duration_seconds,
        "duration_hours": duration_seconds / 3600.0,
        "npz_bytes": sum(record.file_size_bytes for record in records),
        "sources": {
            source: _source_summary(values)
            for source, values in sorted(by_source.items())
        },
        "schema_status_counts": dict(
            sorted(Counter(record.schema_status for record in records).items())
        ),
        "text_status_counts": dict(
            sorted(Counter(record.text_status for record in records).items())
        ),
        "fps_counts": dict(
            sorted(
                Counter(str(record.fps) for record in records if record.fps > 0).items()
            )
        ),
        "gender_counts": dict(
            sorted(
                Counter(record.gender for record in records if record.gender).items()
            )
        ),
    }


def _build_track_summary(
    groups: Mapping[str, Sequence[SequenceRecord]],
    multi_groups: Mapping[str, Sequence[SequenceRecord]],
) -> tuple[dict[str, Any], int]:
    """Build track-count distribution and possible pair count."""
    track_distribution = Counter(
        len({record.person_id for record in values if record.person_id >= 0})
        for values in groups.values()
    )
    possible_pairs = sum(
        math.comb(track_count, 2) * group_count
        for track_count, group_count in track_distribution.items()
    )
    summary = {
        "track_count_distribution": {
            str(key): value for key, value in sorted(track_distribution.items())
        },
        "multi_person_group_count": len(multi_groups),
        "multi_person_sequence_count": sum(
            len(values) for values in multi_groups.values()
        ),
        "exactly_two_person_group_count": sum(
            len({record.person_id for record in values}) == 2
            for values in multi_groups.values()
        ),
        "max_person_ids_in_group": max(track_distribution, default=0),
        "possible_person_pair_count": possible_pairs,
    }
    return summary, possible_pairs


def _build_hoh_summary(
    pair_rows: Sequence[Mapping[str, Any]],
    possible_pairs: int,
) -> dict[str, Any]:
    """Build conservative H-O-H screening metrics."""
    tier_counts = Counter(row["tier"] for row in pair_rows)
    tier_counts["MULTI_PERSON_NO_SHARED_OBJECT"] = possible_pairs - len(pair_rows)
    strict_group_ids = {
        row["group_id"]
        for row in pair_rows
        if row["tier"] == "HOH_TEXT_STRICT_CANDIDATE"
    }
    high_priority_rows = [
        row for row in pair_rows if _is_high_priority_text_candidate(row)
    ]
    return {
        "confirmed_target_hoh_count": 0,
        "strict_text_pair_candidate_count": tier_counts["HOH_TEXT_STRICT_CANDIDATE"],
        "shared_object_pair_candidate_count": tier_counts[
            "HOH_SHARED_OBJECT_CANDIDATE"
        ],
        "multi_person_no_shared_object_pair_count": tier_counts[
            "MULTI_PERSON_NO_SHARED_OBJECT"
        ],
        "strict_text_candidate_group_count": len(strict_group_ids),
        "high_priority_text_pair_candidate_count": len(high_priority_rows),
        "high_priority_text_candidate_group_count": len(
            {row["group_id"] for row in high_priority_rows}
        ),
        "any_shared_object_candidate_group_count": len(
            {row["group_id"] for row in pair_rows}
        ),
        "decision": "NO_CONFIRMED_HOH_IN_RELEASE",
        "blocking_evidence": [
            "no object identity or trajectory",
            "no original frame IDs after per-track segmentation",
            "no guaranteed cross-track temporal synchronization",
            "text labels are heuristic rather than geometric contact evidence",
        ],
    }


def _build_label_summary(records: Sequence[SequenceRecord]) -> dict[str, Any]:
    """Build multi-label caption frequency maps."""
    return {
        "object_sequence_counts": dict(
            Counter(
                label for record in records for label in record.object_labels
            ).most_common()
        ),
        "action_sequence_counts": dict(
            Counter(
                label for record in records for label in record.action_labels
            ).most_common()
        ),
        "verb_lemma_counts": dict(
            Counter(
                lemma for record in records for lemma in record.verb_lemmas
            ).most_common()
        ),
    }


def aggregate_records(
    records: Sequence[SequenceRecord],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate release-level metrics and conservative pair evidence."""
    groups: dict[str, list[SequenceRecord]] = defaultdict(list)
    by_source: dict[str, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
        by_source[record.source].append(record)
    multi_groups = {
        group_id: values
        for group_id, values in groups.items()
        if len({record.person_id for record in values if record.person_id >= 0}) >= 2
    }
    pair_rows = [
        row
        for group_id, values in sorted(multi_groups.items())
        for row in build_shared_object_pairs(group_id, values)
    ]
    pairs_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_group[row["group_id"]].append(row)
    group_rows = [
        _build_group_row(group_id, values, pairs_by_group.get(group_id, []))
        for group_id, values in sorted(multi_groups.items())
    ]
    track_summary, possible_pairs = _build_track_summary(groups, multi_groups)
    summary = {
        "local_release": _build_local_release(records, groups, by_source),
        "track_groups": track_summary,
        "hoh_screen": _build_hoh_summary(pair_rows, possible_pairs),
        "labels": _build_label_summary(records),
    }
    return summary, group_rows, pair_rows
