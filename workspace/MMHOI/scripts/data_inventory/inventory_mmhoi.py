#!/usr/bin/env python3
"""Build MMHOI inventory evidence, including C_2/C_8 and C_2+box scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger("inventory_mmhoi")

SPLIT_PATH = "MMHOI/splits/train_val_test_split.json"
ACTION_SUFFIX = "/PARAM/action.csv"
OBJECT_PREFIX = "MMHOI/object/"
NO_INTERACTION = "no-interaction"
SCENARIO_PATTERN = re.compile(r"C_(\d+)")
SUBJECT_PATTERN = re.compile(r"person([A-Z])")

# Figure A.1 gives the five Collaborative work scenario names. The code mapping
# follows the official README's ordered 12-scenario list and split keys.
COLLABORATIVE_SCENARIOS: Mapping[str, str] = {
    "C_2": "Moving heavy stuffs 1",
    "C_8": "Moving heavy stuffs 2",
    "C_9": "Meeting 1",
    "C_10": "Moving stuffs",
    "C_9_r2": "Meeting 2",
}
PRIMARY_SCENARIOS: Mapping[str, str] = {
    "C_2": "Moving heavy stuffs 1",
    "C_8": "Moving heavy stuffs 2",
}
PILOT_SCENARIO = "C_2"
PILOT_OBJECT = "box"
DENSE_CANDIDATE_SUFFIXES = {
    ".avi",
    ".bvh",
    ".c3d",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
}


@dataclass(frozen=True)
class ActionRow:
    """One person-object-verb label from PARAM/action.csv."""

    person: str
    object_name: str
    verb: str


@dataclass(frozen=True)
class SampleEntry:
    """One annotated frame folder backed by PARAM/action.csv."""

    sequence: str
    scenario_folder: str
    scenario_code: str
    frame_folder: str
    action_path: str

    @property
    def capture_id(self) -> str:
        """Return the stable sequence/scenario identifier."""
        return f"{self.sequence}/{self.scenario_folder}"

    @property
    def sample_id(self) -> str:
        """Return the stable sequence/scenario/frame identifier."""
        return f"{self.capture_id}/{self.frame_folder}"


@dataclass(frozen=True)
class SampleRecord:
    """A sample with split assignment and parsed action labels."""

    entry: SampleEntry
    split: str
    actions: tuple[ActionRow, ...]


def parse_scenario_code(folder_name: str) -> str:
    """Return canonical C_N or C_N_r2 from an archive scenario folder."""
    match = SCENARIO_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(f"cannot parse scenario code from {folder_name!r}")
    base_code = f"C_{int(match.group(1))}"
    is_three_person = "3person" in folder_name or "_r2" in folder_name
    return f"{base_code}_r2" if is_three_person else base_code


def normalize_sequence_key(sequence: str) -> str:
    """Map the one archive/split naming discrepancy to the split key."""
    if sequence == "Single20240531_personA_all_30skip_start-end":
        return "Single20240531_single_personA_all_30skip_start-end"
    return sequence


def assign_split(index: int, counts: tuple[int, int, int]) -> str:
    """Assign a zero-based chronological sample index to train/val/test."""
    train_count, val_count, test_count = counts
    total_count = train_count + val_count + test_count
    if index < 0 or index >= total_count:
        raise ValueError(
            f"sample index {index} outside split counts {counts} (total={total_count})"
        )
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def parse_action_rows(text: str) -> tuple[ActionRow, ...]:
    """Parse PARAM/action.csv and validate its person/object/verb columns."""
    parsed: list[ActionRow] = []
    for row_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row:
            continue
        if len(row) < 11:
            raise ValueError(
                f"action row {row_number}: expected at least 11 columns, got {len(row)}"
            )
        person, object_name, verb = (value.strip() for value in row[-3:])
        if not person or not object_name or not verb:
            raise ValueError(
                f"action row {row_number}: empty person/object/verb field"
            )
        parsed.append(
            ActionRow(
                person=person,
                object_name=object_name,
                verb=verb,
            )
        )
    if not parsed:
        raise ValueError("action CSV contains no rows")
    return tuple(parsed)


def _parse_sample_entry(path: str) -> SampleEntry:
    parts = path.split("/")
    if len(parts) != 7 or parts[0:2] != ["MMHOI", "sequences"]:
        raise ValueError(f"unexpected action path layout: {path}")
    sequence, scenario_folder, frame_folder = parts[2:5]
    if not frame_folder.isdigit():
        raise ValueError(f"non-numeric frame folder in {path}")
    return SampleEntry(
        sequence=sequence,
        scenario_folder=scenario_folder,
        scenario_code=parse_scenario_code(scenario_folder),
        frame_folder=frame_folder,
        action_path=path,
    )


def _read_split_data(archive: zipfile.ZipFile) -> dict[str, dict[str, list[int]]]:
    try:
        payload = archive.read(SPLIT_PATH)
    except KeyError as error:
        raise ValueError(f"archive is missing {SPLIT_PATH}") from error
    raw_data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(raw_data, dict):
        raise ValueError("split JSON root must be an object")
    return raw_data


def _split_counts(
    split_data: Mapping[str, Mapping[str, Sequence[int]]],
    sequence: str,
    scenario_code: str,
) -> tuple[int, int, int] | None:
    split_sequence = normalize_sequence_key(sequence)
    try:
        raw_counts = split_data[split_sequence][scenario_code]
    except KeyError:
        return None
    if len(raw_counts) != 3 or any(
        not isinstance(value, int) or value < 0 for value in raw_counts
    ):
        raise ValueError(
            f"invalid split counts for {split_sequence}/{scenario_code}: {raw_counts}"
        )
    return tuple(raw_counts)


def _assign_all_splits(
    entries: Sequence[SampleEntry],
    split_data: Mapping[str, Mapping[str, Sequence[int]]],
) -> tuple[dict[str, str], list[dict[str, str | int]]]:
    grouped: dict[tuple[str, str], list[SampleEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.sequence, entry.scenario_code)].append(entry)

    assignments: dict[str, str] = {}
    mismatches: list[dict[str, str | int]] = []
    for (sequence, scenario_code), group in sorted(grouped.items()):
        counts = _split_counts(split_data, sequence, scenario_code)
        ordered = sorted(group, key=lambda item: int(item.frame_folder))
        if counts is None:
            mismatches.append(
                {
                    "sequence": sequence,
                    "scenario_code": scenario_code,
                    "reason": "missing_split_key",
                    "archive_sample_count": len(ordered),
                    "split_train": "",
                    "split_val": "",
                    "split_test": "",
                    "split_total": "",
                    "delta_archive_minus_split": "",
                }
            )
            for entry in ordered:
                assignments[entry.action_path] = "unspecified"
            continue
        if len(ordered) != sum(counts):
            mismatches.append(
                {
                    "sequence": sequence,
                    "scenario_code": scenario_code,
                    "reason": "count_mismatch",
                    "archive_sample_count": len(ordered),
                    "split_train": counts[0],
                    "split_val": counts[1],
                    "split_test": counts[2],
                    "split_total": sum(counts),
                    "delta_archive_minus_split": len(ordered) - sum(counts),
                }
            )
        for index, entry in enumerate(ordered):
            if index < sum(counts):
                assignments[entry.action_path] = assign_split(index, counts)
            else:
                assignments[entry.action_path] = "unspecified"
    return assignments, mismatches


def _object_names(infos: Iterable[zipfile.ZipInfo]) -> tuple[str, ...]:
    names: list[str] = []
    for info in infos:
        path = info.filename
        if not path.startswith(OBJECT_PREFIX) or not path.endswith(".ply"):
            continue
        if path.count("/") != 2:
            continue
        stem = Path(path).stem
        name_parts = stem.split("_", maxsplit=1)
        if len(name_parts) != 2 or not name_parts[0].isdigit():
            raise ValueError(f"unexpected object mesh name: {path}")
        names.append(name_parts[1])
    return tuple(sorted(names))


def _read_sample_records(
    archive: zipfile.ZipFile,
    entries: Sequence[SampleEntry],
    assignments: Mapping[str, str],
) -> tuple[SampleRecord, ...]:
    records: list[SampleRecord] = []
    for index, entry in enumerate(entries, start=1):
        try:
            text = archive.read(entry.action_path).decode("utf-8-sig")
            actions = parse_action_rows(text)
        except (KeyError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"failed to parse {entry.action_path}: {error}") from error
        records.append(
            SampleRecord(
                entry=entry,
                split=assignments[entry.action_path],
                actions=actions,
            )
        )
        if index % 1000 == 0:
            LOGGER.info("parsed %d/%d action CSVs", index, len(entries))
    return tuple(records)


def _is_cooperative_sample(record: SampleRecord) -> bool:
    by_object_verb: dict[tuple[str, str], set[str]] = defaultdict(set)
    for action in record.actions:
        if "together" not in action.verb:
            continue
        by_object_verb[(action.object_name, action.verb)].add(action.person)
    return any(len(persons) >= 2 for persons in by_object_verb.values())


def _subject_ids(records: Sequence[SampleRecord]) -> list[str]:
    subjects = {
        subject
        for record in records
        for subject in SUBJECT_PATTERN.findall(record.entry.sequence)
    }
    return sorted(subjects)


def _class_rows(
    records: Sequence[SampleRecord],
) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    sample_sets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in records:
        for action in record.actions:
            key = (
                record.entry.scenario_code,
                action.object_name,
                action.verb,
            )
            counts[key] += 1
            sample_sets[key].add(record.entry.sample_id)
    return [
        {
            "scenario_code": code,
            "scenario_name": COLLABORATIVE_SCENARIOS[code],
            "object_name": object_name,
            "verb": verb,
            "action_row_count": count,
            "sample_count": len(sample_sets[(code, object_name, verb)]),
        }
        for (code, object_name, verb), count in sorted(counts.items())
    ]


def _capture_rows(records: Sequence[SampleRecord]) -> list[dict[str, str | int]]:
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.entry.capture_id].append(record)
    rows: list[dict[str, str | int]] = []
    for capture_id, group in sorted(grouped.items()):
        first = group[0]
        action_rows = sum(len(record.actions) for record in group)
        active_objects = sorted(
            {
                action.object_name
                for record in group
                for action in record.actions
                if action.verb != NO_INTERACTION
            }
        )
        rows.append(
            {
                "capture_id": capture_id,
                "sequence": first.entry.sequence,
                "scenario_folder": first.entry.scenario_folder,
                "scenario_code": first.entry.scenario_code,
                "scenario_name": COLLABORATIVE_SCENARIOS[
                    first.entry.scenario_code
                ],
                "annotated_sample_count": len(group),
                "action_row_count": action_rows,
                "active_object_types": ",".join(active_objects),
                "cooperative_sample_count": sum(
                    _is_cooperative_sample(record) for record in group
                ),
            }
        )
    return rows


def _sample_rows(records: Sequence[SampleRecord]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.entry.sequence,
            item.entry.scenario_code,
            int(item.entry.frame_folder),
        ),
    ):
        active_objects = sorted(
            {
                action.object_name
                for action in record.actions
                if action.verb != NO_INTERACTION
            }
        )
        rows.append(
            {
                "sample_id": record.entry.sample_id,
                "sequence": record.entry.sequence,
                "scenario_code": record.entry.scenario_code,
                "scenario_name": COLLABORATIVE_SCENARIOS[
                    record.entry.scenario_code
                ],
                "frame_folder": record.entry.frame_folder,
                "split": record.split,
                "person_count": len({action.person for action in record.actions}),
                "action_row_count": len(record.actions),
                "active_object_types": ",".join(active_objects),
                "has_cooperative_action": int(_is_cooperative_sample(record)),
                "action_csv": record.entry.action_path,
            }
        )
    return rows


def _object_rows(
    records: Sequence[SampleRecord],
) -> list[dict[str, str | int]]:
    action_counts: Counter[str] = Counter()
    active_counts: Counter[str] = Counter()
    cooperative_counts: Counter[str] = Counter()
    sample_sets: dict[str, set[str]] = defaultdict(set)
    scenario_sets: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for action in record.actions:
            name = action.object_name
            action_counts[name] += 1
            sample_sets[name].add(record.entry.sample_id)
            scenario_sets[name].add(record.entry.scenario_code)
            if action.verb != NO_INTERACTION:
                active_counts[name] += 1
            if "together" in action.verb:
                cooperative_counts[name] += 1
    return [
        {
            "object_name": name,
            "action_row_count": action_counts[name],
            "active_action_row_count": active_counts[name],
            "cooperative_action_row_count": cooperative_counts[name],
            "sample_count": len(sample_sets[name]),
            "scenario_codes": ",".join(sorted(scenario_sets[name])),
        }
        for name in sorted(action_counts)
    ]


def _scenario_split_rows(
    records: Sequence[SampleRecord],
    scenario_codes: Iterable[str] | None = None,
) -> list[dict[str, str | int]]:
    counts = Counter(
        (record.entry.scenario_code, record.split) for record in records
    )
    return [
        {
            "scenario_code": code,
            "scenario_name": COLLABORATIVE_SCENARIOS[code],
            "train": counts[(code, "train")],
            "val": counts[(code, "val")],
            "test": counts[(code, "test")],
            "unspecified": counts[(code, "unspecified")],
            "total": sum(
                counts[(code, split)]
                for split in ("train", "val", "test", "unspecified")
            ),
        }
        for code in (
            COLLABORATIVE_SCENARIOS if scenario_codes is None else scenario_codes
        )
    ]


def _frame_gap_distribution(
    records: Sequence[SampleRecord],
) -> dict[str, int]:
    """Count adjacent source-frame-id gaps within each capture."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[record.entry.capture_id].append(
            int(record.entry.frame_folder)
        )
    gaps: Counter[int] = Counter()
    for frames in grouped.values():
        ordered = sorted(frames)
        gaps.update(
            current - previous
            for previous, current in zip(
                ordered,
                ordered[1:],
                strict=False,
            )
        )
    return {str(gap): count for gap, count in sorted(gaps.items())}


def _scope_summary(
    records: Sequence[SampleRecord],
    filter_description: str = (
        "exactly two distinct person labels in PARAM/action.csv"
    ),
) -> dict[str, object]:
    """Summarize an already-filtered collaborative record scope."""
    actions = [action for record in records for action in record.actions]
    active_actions = [
        action for action in actions if action.verb != NO_INTERACTION
    ]
    split_counts = Counter(record.split for record in records)
    scenario_codes = sorted(
        {record.entry.scenario_code for record in records}
    )
    return {
        "filter": filter_description,
        "scenario_codes": {
            code: COLLABORATIVE_SCENARIOS[code] for code in scenario_codes
        },
        "scenario_capture_count": len(
            {record.entry.capture_id for record in records}
        ),
        "sequence_root_count": len(
            {record.entry.sequence for record in records}
        ),
        "annotated_sample_count": len(records),
        "split_sample_counts": {
            split: split_counts[split]
            for split in ("train", "val", "test", "unspecified")
        },
        "subject_ids": _subject_ids(records),
        "subject_count": len(_subject_ids(records)),
        "action_row_count": len(actions),
        "active_action_row_count": len(active_actions),
        "unique_verb_count": len({action.verb for action in actions}),
        "unique_verbs": sorted({action.verb for action in actions}),
        "active_unique_verb_count": len(
            {action.verb for action in active_actions}
        ),
        "unique_action_class_count": len(
            {(action.object_name, action.verb) for action in actions}
        ),
        "active_unique_action_class_count": len(
            {(action.object_name, action.verb) for action in active_actions}
        ),
        "active_object_types": sorted(
            {action.object_name for action in active_actions}
        ),
        "cooperative_sample_count": sum(
            _is_cooperative_sample(record) for record in records
        ),
        "consecutive_frame_gap_distribution": _frame_gap_distribution(
            records
        ),
    }


def _has_active_object(record: SampleRecord, object_name: str) -> bool:
    return any(
        action.object_name == object_name
        and action.verb != NO_INTERACTION
        for action in record.actions
    )


def _is_cooperative_for_object(
    record: SampleRecord,
    object_name: str,
) -> bool:
    by_verb: dict[str, set[str]] = defaultdict(set)
    for action in record.actions:
        if action.object_name == object_name and "together" in action.verb:
            by_verb[action.verb].add(action.person)
    return any(len(persons) >= 2 for persons in by_verb.values())


def _target_object_verb_rows(
    records: Sequence[SampleRecord],
    object_name: str,
) -> list[dict[str, str | int]]:
    counts: Counter[str] = Counter()
    sample_sets: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for action in record.actions:
            if action.object_name != object_name:
                continue
            counts[action.verb] += 1
            sample_sets[action.verb].add(record.entry.sample_id)
    return [
        {
            "scenario_code": PILOT_SCENARIO,
            "object_name": object_name,
            "verb": verb,
            "action_row_count": count,
            "sample_count": len(sample_sets[verb]),
        }
        for verb, count in sorted(counts.items())
    ]


def _pilot_sample_rows(
    records: Sequence[SampleRecord],
    object_name: str,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.entry.sequence,
            int(item.entry.frame_folder),
        ),
    ):
        target_actions = [
            action
            for action in record.actions
            if action.object_name == object_name
        ]
        active_actions = [
            action
            for action in target_actions
            if action.verb != NO_INTERACTION
        ]
        rows.append(
            {
                "sample_id": record.entry.sample_id,
                "sequence": record.entry.sequence,
                "capture_id": record.entry.capture_id,
                "frame_folder": record.entry.frame_folder,
                "split": record.split,
                "target_object": object_name,
                "target_verbs": ",".join(
                    sorted({action.verb for action in target_actions})
                ),
                "active_person_count": len(
                    {action.person for action in active_actions}
                ),
                "has_together_label": int(
                    any("together" in action.verb for action in target_actions)
                ),
                "has_cooperative_label": int(
                    _is_cooperative_for_object(record, object_name)
                ),
                "action_csv": record.entry.action_path,
            }
        )
    return rows


def _pilot_summary(
    c2_records: Sequence[SampleRecord],
) -> dict[str, object]:
    target_actions = [
        action
        for record in c2_records
        for action in record.actions
        if action.object_name == PILOT_OBJECT
    ]
    active_target_actions = [
        action
        for action in target_actions
        if action.verb != NO_INTERACTION
    ]
    selected_records = [
        record
        for record in c2_records
        if _has_active_object(record, PILOT_OBJECT)
    ]
    return {
        "scenario_code": PILOT_SCENARIO,
        "target_object": PILOT_OBJECT,
        "filter": (
            "C_2 sample with at least one box action row whose verb is not "
            "no-interaction"
        ),
        "c2_annotated_sample_count": len(c2_records),
        "target_object_sample_count": sum(
            any(action.object_name == PILOT_OBJECT for action in record.actions)
            for record in c2_records
        ),
        "target_object_active_sample_count": len(selected_records),
        "target_object_action_row_count": len(target_actions),
        "target_object_active_action_row_count": len(active_target_actions),
        "target_object_together_action_row_count": sum(
            "together" in action.verb for action in target_actions
        ),
        "target_object_cooperative_sample_count": sum(
            _is_cooperative_for_object(record, PILOT_OBJECT)
            for record in c2_records
        ),
        "target_object_verb_distribution": _target_object_verb_rows(
            c2_records,
            PILOT_OBJECT,
        ),
        "selection": _scope_summary(
            selected_records,
            filter_description=(
                "C_2 with at least one active box action row; exactly two "
                "distinct person labels"
            ),
        ),
    }


def _primary_temporal_audit(
    infos: Sequence[zipfile.ZipInfo],
    primary_records: Sequence[SampleRecord],
) -> dict[str, object]:
    frame_roots = {
        record.entry.action_path.removesuffix(ACTION_SUFFIX)
        for record in primary_records
    }
    person1_paths = {f"{root}/PARAM/person1.json" for root in frame_roots}
    person2_paths = {f"{root}/PARAM/person2.json" for root in frame_roots}
    found_person1: set[str] = set()
    found_person2: set[str] = set()
    final_object_roots: set[str] = set()
    dense_candidates: list[str] = []
    nested_archives: list[dict[str, str | int]] = []
    outside_numeric_frame_paths: Counter[str] = Counter()
    capture_ids = {
        record.entry.capture_id for record in primary_records
    }

    for info in infos:
        path = info.filename
        if path in person1_paths:
            found_person1.add(path.removesuffix("/PARAM/person1.json"))
        if path in person2_paths:
            found_person2.add(path.removesuffix("/PARAM/person2.json"))
        if "/final/" in path and path.endswith(".ply"):
            root, filename = path.rsplit("/final/", maxsplit=1)
            stem = Path(filename).stem
            if (
                root in frame_roots
                and stem != "all"
                and not stem.startswith("person")
            ):
                final_object_roots.add(root)
        suffix = Path(path).suffix.lower()
        if suffix in DENSE_CANDIDATE_SUFFIXES:
            dense_candidates.append(path)
        if suffix == ".zip":
            nested_archives.append(
                {
                    "path": path,
                    "uncompressed_size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                }
            )
        parts = path.split("/")
        if (
            len(parts) >= 5
            and parts[0:2] == ["MMHOI", "sequences"]
            and f"{parts[2]}/{parts[3]}" in capture_ids
            and not parts[4].isdigit()
            and not info.is_dir()
        ):
            outside_numeric_frame_paths["/".join(parts[4:])] += 1

    capture_folder_names = {
        record.entry.capture_id: record.entry.scenario_folder
        for record in primary_records
    }
    normalized_capture_names = {
        name.lower().replace("_", "")
        for name in capture_folder_names.values()
    }
    return {
        "scope": dict(PRIMARY_SCENARIOS),
        "scenario_capture_count": len(
            {record.entry.capture_id for record in primary_records}
        ),
        "numeric_frame_folder_count": len(frame_roots),
        "param_action_frame_folder_count": len(primary_records),
        "param_two_person_json_frame_folder_count": len(
            found_person1 & found_person2
        ),
        "final_object_mesh_frame_folder_count": len(final_object_roots),
        "capture_folder_count": len(capture_folder_names),
        "all_capture_folders_marked_30skip": all(
            "30skip" in name for name in normalized_capture_names
        ),
        "files_outside_numeric_frame_folders": dict(
            sorted(outside_numeric_frame_paths.items())
        ),
        "consecutive_frame_gap_distribution": _frame_gap_distribution(
            primary_records
        ),
        "dense_candidate_suffixes": sorted(DENSE_CANDIDATE_SUFFIXES),
        "dense_candidate_file_count": len(dense_candidates),
        "dense_candidate_files": dense_candidates,
        "nested_archive_entries": nested_archives,
        "interpretation": (
            "The public release provides 30skip annotations. At a 30 fps "
            "capture rate, adjacent released GT samples are approximately 1 s "
            "apart; capture fps must not be used as released GT fps."
        ),
    }


def _summarize_records(
    records: Sequence[SampleRecord],
    object_names: Sequence[str],
    infos: Sequence[zipfile.ZipInfo],
    split_mismatches: Sequence[Mapping[str, str | int]],
) -> dict[str, object]:
    collaborative = [
        record
        for record in records
        if record.entry.scenario_code in COLLABORATIVE_SCENARIOS
    ]
    strict_two_person = [
        record
        for record in collaborative
        if len({action.person for action in record.actions}) == 2
    ]
    primary_scope = [
        record
        for record in strict_two_person
        if record.entry.scenario_code in PRIMARY_SCENARIOS
    ]
    c2_records = [
        record
        for record in primary_scope
        if record.entry.scenario_code == PILOT_SCENARIO
    ]
    pilot_records = [
        record
        for record in c2_records
        if _has_active_object(record, PILOT_OBJECT)
    ]
    actions = [action for record in collaborative for action in record.actions]
    active_actions = [
        action for action in actions if action.verb != NO_INTERACTION
    ]
    global_actions = [action for record in records for action in record.actions]
    split_counts = Counter(record.split for record in collaborative)
    person_counts = Counter(
        len({action.person for action in record.actions})
        for record in collaborative
    )
    captures = {record.entry.capture_id for record in collaborative}
    scenario_captures = {record.entry.capture_id for record in records}

    return {
        "archive": {"entry_count": len(infos)},
        "split_audit": {
            "mismatch_count": len(split_mismatches),
            "policy": (
                "Archive samples beyond declared split totals are preserved as "
                "unspecified; missing archive samples remain visible via negative delta."
            ),
        },
        "global": {
            "sequence_root_count": len(
                {record.entry.sequence for record in records}
            ),
            "scenario_capture_count": len(scenario_captures),
            "annotated_sample_count": len(records),
            "action_row_count": len(global_actions),
            "unique_verb_count": len({action.verb for action in global_actions}),
            "unique_action_class_count": len(
                {(action.object_name, action.verb) for action in global_actions}
            ),
            "object_mesh_count": len(object_names),
            "object_mesh_types": list(object_names),
        },
        "collaborative": {
            "scenario_codes": dict(COLLABORATIVE_SCENARIOS),
            "sequence_root_count": len(
                {record.entry.sequence for record in collaborative}
            ),
            "scenario_capture_count": len(captures),
            "annotated_sample_count": len(collaborative),
            "split_sample_counts": {
                split: split_counts[split]
                for split in ("train", "val", "test", "unspecified")
            },
            "subject_ids": _subject_ids(collaborative),
            "subject_count": len(_subject_ids(collaborative)),
            "person_count_distribution": {
                str(count): total for count, total in sorted(person_counts.items())
            },
            "action_row_count": len(actions),
            "active_action_row_count": len(active_actions),
            "unique_verb_count": len({action.verb for action in actions}),
            "unique_verbs": sorted({action.verb for action in actions}),
            "active_unique_verb_count": len(
                {action.verb for action in active_actions}
            ),
            "unique_action_class_count": len(
                {(action.object_name, action.verb) for action in actions}
            ),
            "active_unique_action_class_count": len(
                {(action.object_name, action.verb) for action in active_actions}
            ),
            "object_types": sorted({action.object_name for action in actions}),
            "active_object_types": sorted(
                {action.object_name for action in active_actions}
            ),
            "cooperative_object_types": sorted(
                {
                    action.object_name
                    for action in actions
                    if "together" in action.verb
                }
            ),
            "cooperative_sample_count": sum(
                _is_cooperative_sample(record) for record in collaborative
            ),
            "consecutive_frame_gap_distribution": _frame_gap_distribution(
                collaborative
            ),
            "strict_two_person": _scope_summary(strict_two_person),
        },
        "primary_scope": _scope_summary(
            primary_scope,
            filter_description=(
                "scenario in C_2/C_8 and exactly two distinct person labels "
                "in PARAM/action.csv"
            ),
        ),
        "pilot_c2_box": _pilot_summary(c2_records),
        "primary_temporal_audit": _primary_temporal_audit(
            infos,
            primary_scope,
        ),
        "tables": {
            "scenario_split": _scenario_split_rows(collaborative),
            "captures": _capture_rows(collaborative),
            "samples": _sample_rows(collaborative),
            "action_classes": _class_rows(collaborative),
            "objects": _object_rows(collaborative),
            "primary_scenario_split": _scenario_split_rows(
                primary_scope,
                PRIMARY_SCENARIOS,
            ),
            "primary_captures": _capture_rows(primary_scope),
            "primary_samples": _sample_rows(primary_scope),
            "primary_action_classes": _class_rows(primary_scope),
            "primary_objects": _object_rows(primary_scope),
            "pilot_samples": _pilot_sample_rows(
                pilot_records,
                PILOT_OBJECT,
            ),
            "pilot_action_classes": _target_object_verb_rows(
                c2_records,
                PILOT_OBJECT,
            ),
            "split_mismatches": list(split_mismatches),
        },
    }


def inventory_archive(archive_path: Path) -> dict[str, object]:
    """Read the MMHOI release archive and return full/collaborative statistics."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"MMHOI archive not found: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"not a readable ZIP archive: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        split_data = _read_split_data(archive)
        entries = tuple(
            sorted(
                (
                    _parse_sample_entry(info.filename)
                    for info in infos
                    if info.filename.endswith(ACTION_SUFFIX)
                ),
                key=lambda item: (
                    item.sequence,
                    item.scenario_code,
                    int(item.frame_folder),
                ),
            )
        )
        if not entries:
            raise ValueError("archive contains no PARAM/action.csv entries")
        assignments, split_mismatches = _assign_all_splits(entries, split_data)
        records = _read_sample_records(archive, entries, assignments)
        return _summarize_records(
            records=records,
            object_names=_object_names(infos),
            infos=infos,
            split_mismatches=split_mismatches,
        )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _file_signature(path: Path) -> dict[str, object]:
    stat_result = path.stat()
    identity = f"{stat_result.st_size}:{stat_result.st_mtime_ns}".encode()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "size_mtime_sha256": hashlib.sha256(identity).hexdigest(),
        "note": "Fast identity signature; not a content hash of the 93.7 GB archive.",
    }


def write_outputs(
    result: Mapping[str, object],
    archive_path: Path,
    output_dir: Path,
    command_line: Sequence[str],
) -> None:
    """Write JSON and TSV evidence into the requested persistent directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = result["tables"]
    if not isinstance(tables, Mapping):
        raise TypeError("result tables must be a mapping")
    table_paths = {
        "scenario_split": "collaborative_samples_by_scenario_split.tsv",
        "captures": "collaborative_scenario_inventory.tsv",
        "samples": "collaborative_sample_inventory.tsv",
        "action_classes": "collaborative_action_classes.tsv",
        "objects": "collaborative_objects.tsv",
        "primary_scenario_split": "primary_samples_by_scenario_split.tsv",
        "primary_captures": "primary_scenario_inventory.tsv",
        "primary_samples": "primary_sample_inventory.tsv",
        "primary_action_classes": "primary_action_classes.tsv",
        "primary_objects": "primary_objects.tsv",
        "pilot_samples": "pilot_c2_box_sample_inventory.tsv",
        "pilot_action_classes": "pilot_c2_box_action_classes.tsv",
        "split_mismatches": "split_mismatches.tsv",
    }
    for table_name, filename in table_paths.items():
        rows = tables[table_name]
        if not isinstance(rows, list):
            raise TypeError(f"table {table_name} must be a list")
        _write_tsv(output_dir / filename, rows)

    summary = {key: value for key, value in result.items() if key != "tables"}
    summary["source_archive"] = _file_signature(archive_path)
    summary["generated_at"] = datetime.now(UTC).isoformat()
    summary["command_line"] = list(command_line)
    with (output_dir / "inventory_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build inventory outputs and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    try:
        result = inventory_archive(args.archive)
        write_outputs(
            result=result,
            archive_path=args.archive,
            output_dir=args.output_dir,
            command_line=sys.argv,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, zipfile.BadZipFile):
        LOGGER.exception("MMHOI inventory failed")
        return 1
    primary = result["primary_scope"]
    if not isinstance(primary, Mapping):
        raise TypeError("primary scope summary must be a mapping")
    LOGGER.info(
        "done: primary scope has %s captures, %s samples, %s active object types",
        primary["scenario_capture_count"],
        primary["annotated_sample_count"],
        len(primary["active_object_types"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
