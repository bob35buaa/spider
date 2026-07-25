"""Core scanner for the released InterPose motion package.

The released package stores one SMPL-X track per NPZ. This scanner reads NPY
headers and small scalar fields without materializing pose arrays, reconstructs
the exporter ``<person_idx>_<segment_idx>`` suffix, and generates conservative
text-evidence candidates. It never treats a text candidate as confirmed paired
human-object-human data because the release omits object trajectories and
original frame IDs.
"""

from __future__ import annotations

import ast
import io
import itertools
import json
import logging
import math
import re
import subprocess
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from interpose_vocab import (
    ACTION_LABELS,
    EXPECTED_KEYS,
    OBJECT_LABELS,
    RELATION_CUES,
    SOURCE_DIRS,
)
from numpy.lib import format as npy_format

LOGGER = logging.getLogger("audit_interpose")
SEQUENCE_SUFFIX = re.compile(
    r"^(?P<clip_id>.+)_(?P<person_id>\d+)_(?P<segment_id>\d+)$"
)


@dataclass(frozen=True)
class ParsedSequenceId:
    """Exporter identity reconstructed from a released filename."""

    clip_id: str
    person_id: int
    segment_id: int


@dataclass(frozen=True)
class ParsedTextRecord:
    """HumanML3D-style caption and NLP annotation."""

    caption: str
    lemma_tokens: tuple[str, ...]
    verb_lemmas: tuple[str, ...]
    noun_lemmas: tuple[str, ...]
    start_time: float | None
    end_time: float | None
    parse_status: str


@dataclass(frozen=True)
class SequenceEvidence:
    """Deterministic text evidence used for conservative candidate screening."""

    object_labels: tuple[str, ...]
    action_labels: tuple[str, ...]
    relation_cues: tuple[str, ...]


@dataclass(frozen=True)
class PairClassification:
    """Evidence tier for two tracks from one exported clip group."""

    tier: str
    shared_objects: tuple[str, ...]
    relation_cues: tuple[str, ...]


@dataclass(frozen=True)
class SequenceRecord:
    """One released InterPose NPZ plus its paired text metadata."""

    source: str
    relative_path: str
    sequence_id: str
    group_id: str
    clip_id: str
    person_id: int
    segment_id: int
    frames: int
    fps: float
    duration_seconds: float
    pose_shape: tuple[int, ...]
    pose_dtype: str
    trans_shape: tuple[int, ...]
    trans_dtype: str
    betas_shape: tuple[int, ...]
    gender: str
    num_betas: int
    caption: str
    verb_lemmas: tuple[str, ...]
    noun_lemmas: tuple[str, ...]
    object_labels: tuple[str, ...]
    action_labels: tuple[str, ...]
    relation_cues: tuple[str, ...]
    schema_status: str
    text_status: str
    file_size_bytes: int
    error: str


def parse_sequence_id(filename: str) -> ParsedSequenceId:
    """Parse the official ``<clip>_<person>_<segment>.npz`` export suffix."""
    stem = Path(filename).stem
    match = SEQUENCE_SUFFIX.fullmatch(stem)
    if match is None:
        raise ValueError(f"missing official person/segment suffix: {filename}")
    return ParsedSequenceId(
        clip_id=match.group("clip_id"),
        person_id=int(match.group("person_id")),
        segment_id=int(match.group("segment_id")),
    )


def _parse_float(value: str) -> float | None:
    """Return a finite float or ``None`` for malformed optional timing."""
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _split_lemma_token(token: str) -> tuple[str, str] | None:
    """Split one ``lemma/POS`` token while tolerating slash punctuation."""
    if "/" not in token:
        return None
    lemma, pos = token.rsplit("/", 1)
    normalized = lemma.strip().lower()
    return (normalized, pos.strip().upper()) if normalized else None


def parse_text_record(raw_text: str) -> ParsedTextRecord:
    """Parse a released text row using a right split for the last three fields."""
    parts = raw_text.strip().rsplit("#", 3)
    if len(parts) != 4:
        return ParsedTextRecord(
            caption=raw_text.strip(),
            lemma_tokens=(),
            verb_lemmas=(),
            noun_lemmas=(),
            start_time=None,
            end_time=None,
            parse_status="MALFORMED_TEXT_ROW",
        )
    caption, token_text, start_text, end_text = parts
    tagged_tokens = tuple(
        parsed
        for token in token_text.split()
        if (parsed := _split_lemma_token(token)) is not None
    )
    return ParsedTextRecord(
        caption=caption.strip(),
        lemma_tokens=tuple(token[0] for token in tagged_tokens),
        verb_lemmas=tuple(token[0] for token in tagged_tokens if token[1] == "VERB"),
        noun_lemmas=tuple(token[0] for token in tagged_tokens if token[1] == "NOUN"),
        start_time=_parse_float(start_text),
        end_time=_parse_float(end_text),
        parse_status="OK",
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a normalized phrase on word boundaries."""
    pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def extract_labels(
    caption: str,
    lemma_tokens: Sequence[str],
    verb_lemmas: Sequence[str],
    noun_lemmas: Sequence[str],
    object_vocabulary: Sequence[str] = OBJECT_LABELS,
    action_vocabulary: Sequence[str] = ACTION_LABELS,
) -> SequenceEvidence:
    """Extract deterministic labels from caption surface text and POS lemmas."""
    lemma_text = " ".join(lemma_tokens)
    noun_set = frozenset(noun_lemmas)
    verb_set = frozenset(verb_lemmas)
    objects = tuple(
        sorted(
            label
            for label in object_vocabulary
            if label.lower() in noun_set
            or _contains_phrase(caption, label)
            or _contains_phrase(lemma_text, label)
        )
    )
    actions = tuple(
        sorted(
            label
            for label in action_vocabulary
            if (" " not in label and label.lower() in verb_set)
            or _contains_phrase(lemma_text, label)
            or _contains_phrase(caption, label)
        )
    )
    cues = tuple(
        sorted(
            cue
            for cue in RELATION_CUES
            if _contains_phrase(caption, cue) or (" " not in cue and cue in verb_set)
        )
    )
    return SequenceEvidence(
        object_labels=objects,
        action_labels=actions,
        relation_cues=cues,
    )


def classify_pair(
    first: SequenceEvidence,
    second: SequenceEvidence,
) -> PairClassification:
    """Classify a same-clip person pair without claiming geometric confirmation."""
    shared_objects = tuple(
        sorted(set(first.object_labels).intersection(second.object_labels))
    )
    relation_cues = tuple(sorted(set(first.relation_cues).union(second.relation_cues)))
    if shared_objects and relation_cues:
        tier = "HOH_TEXT_STRICT_CANDIDATE"
    elif shared_objects:
        tier = "HOH_SHARED_OBJECT_CANDIDATE"
    else:
        tier = "MULTI_PERSON_NO_SHARED_OBJECT"
    return PairClassification(
        tier=tier,
        shared_objects=shared_objects,
        relation_cues=relation_cues,
    )


def _read_array_header(
    archive: zipfile.ZipFile,
    key: str,
) -> tuple[tuple[int, ...], str]:
    """Read an NPY member header without loading its array payload."""
    with archive.open(f"{key}.npy", "r") as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = npy_format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, _, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY version {version} for {key}")
    return tuple(int(value) for value in shape), str(dtype)


def _read_scalar(archive: zipfile.ZipFile, key: str) -> Any:
    """Load one scalar NPY member with pickle disabled."""
    payload = io.BytesIO(archive.read(f"{key}.npy"))
    value = np.load(payload, allow_pickle=False)
    if value.shape != ():
        raise ValueError(f"{key} is not scalar: shape={value.shape}")
    return value.item()


def _empty_record(path: Path, data_root: Path, error: Exception) -> SequenceRecord:
    """Build a stable error row when one file cannot be audited."""
    source = path.parent.name
    return SequenceRecord(
        source=source,
        relative_path=str(path.relative_to(data_root)),
        sequence_id=path.stem,
        group_id=f"{source}/{path.stem}",
        clip_id=path.stem,
        person_id=-1,
        segment_id=-1,
        frames=0,
        fps=0.0,
        duration_seconds=0.0,
        pose_shape=(),
        pose_dtype="",
        trans_shape=(),
        trans_dtype="",
        betas_shape=(),
        gender="",
        num_betas=0,
        caption="",
        verb_lemmas=(),
        noun_lemmas=(),
        object_labels=(),
        action_labels=(),
        relation_cues=(),
        schema_status="ERROR",
        text_status="ERROR",
        file_size_bytes=path.stat().st_size,
        error=f"{type(error).__name__}: {error}",
    )


def _read_npz_metadata(path: Path) -> Mapping[str, Any]:
    """Read the released NPZ schema, headers, and small scalar fields."""
    with zipfile.ZipFile(path, "r") as archive:
        keys = tuple(sorted(Path(name).stem for name in archive.namelist()))
        pose_shape, pose_dtype = _read_array_header(archive, "poses")
        trans_shape, trans_dtype = _read_array_header(archive, "trans")
        betas_shape, _ = _read_array_header(archive, "betas")
        return {
            "keys": keys,
            "pose_shape": pose_shape,
            "pose_dtype": pose_dtype,
            "trans_shape": trans_shape,
            "trans_dtype": trans_dtype,
            "betas_shape": betas_shape,
            "gender": str(_read_scalar(archive, "gender")),
            "num_betas": int(_read_scalar(archive, "num_betas")),
            "fps": float(_read_scalar(archive, "mocap_frame_rate")),
            "npz_text": str(_read_scalar(archive, "text")),
        }


def _schema_status(metadata: Mapping[str, Any]) -> str:
    """Validate the observed schema against the released exporter contract."""
    if tuple(metadata["keys"]) != EXPECTED_KEYS:
        return "KEY_MISMATCH"
    pose_shape = metadata["pose_shape"]
    trans_shape = metadata["trans_shape"]
    if len(pose_shape) != 2 or pose_shape[1] != 165:
        return "POSE_SHAPE_MISMATCH"
    if len(trans_shape) != 2 or trans_shape[1] != 3:
        return "TRANS_SHAPE_MISMATCH"
    if pose_shape[0] != trans_shape[0]:
        return "FRAME_COUNT_MISMATCH"
    if metadata["betas_shape"] != (10,) or metadata["num_betas"] != 10:
        return "BETAS_MISMATCH"
    if not math.isfinite(metadata["fps"]) or metadata["fps"] <= 0:
        return "INVALID_FPS"
    return "OK"


def _read_text_evidence(
    text_path: Path,
    npz_text: str,
    object_vocabulary: Sequence[str],
    action_vocabulary: Sequence[str],
) -> tuple[ParsedTextRecord, SequenceEvidence, str]:
    """Read a paired text file and compare its caption with the NPZ scalar."""
    if not text_path.exists():
        parsed = ParsedTextRecord(
            caption=npz_text,
            lemma_tokens=(),
            verb_lemmas=(),
            noun_lemmas=(),
            start_time=None,
            end_time=None,
            parse_status="TXT_MISSING",
        )
        status = "TXT_MISSING"
    else:
        parsed = parse_text_record(
            text_path.read_text(encoding="utf-8", errors="replace")
        )
        status = parsed.parse_status
        if status == "OK":
            status = "MATCH" if parsed.caption == npz_text else "CAPTION_MISMATCH"
    labels = extract_labels(
        caption=parsed.caption,
        lemma_tokens=parsed.lemma_tokens,
        verb_lemmas=parsed.verb_lemmas,
        noun_lemmas=parsed.noun_lemmas,
        object_vocabulary=object_vocabulary,
        action_vocabulary=action_vocabulary,
    )
    return parsed, labels, status


def _build_sequence_record(
    path: Path,
    data_root: Path,
    parsed_id: ParsedSequenceId,
    metadata: Mapping[str, Any],
    parsed_text: ParsedTextRecord,
    labels: SequenceEvidence,
    text_status: str,
) -> SequenceRecord:
    """Build a successful inventory row from parsed release metadata."""
    frames = int(metadata["pose_shape"][0])
    fps = float(metadata["fps"])
    source = path.parent.name
    return SequenceRecord(
        source=source,
        relative_path=str(path.relative_to(data_root)),
        sequence_id=path.stem,
        group_id=f"{source}/{parsed_id.clip_id}",
        clip_id=parsed_id.clip_id,
        person_id=parsed_id.person_id,
        segment_id=parsed_id.segment_id,
        frames=frames,
        fps=fps,
        duration_seconds=frames / fps,
        pose_shape=metadata["pose_shape"],
        pose_dtype=metadata["pose_dtype"],
        trans_shape=metadata["trans_shape"],
        trans_dtype=metadata["trans_dtype"],
        betas_shape=metadata["betas_shape"],
        gender=metadata["gender"],
        num_betas=metadata["num_betas"],
        caption=parsed_text.caption,
        verb_lemmas=parsed_text.verb_lemmas,
        noun_lemmas=parsed_text.noun_lemmas,
        object_labels=labels.object_labels,
        action_labels=labels.action_labels,
        relation_cues=labels.relation_cues,
        schema_status=_schema_status(metadata),
        text_status=text_status,
        file_size_bytes=path.stat().st_size,
        error="",
    )


def scan_sequence(
    path: Path,
    data_root: Path,
    object_vocabulary: Sequence[str],
    action_vocabulary: Sequence[str],
) -> SequenceRecord:
    """Audit one released NPZ while keeping failures in the inventory."""
    try:
        parsed_id = parse_sequence_id(path.name)
        metadata = _read_npz_metadata(path)
        text_path = data_root / "texts" / f"{path.stem}.txt"
        parsed_text, labels, text_status = _read_text_evidence(
            text_path=text_path,
            npz_text=metadata["npz_text"],
            object_vocabulary=object_vocabulary,
            action_vocabulary=action_vocabulary,
        )
        return _build_sequence_record(
            path=path,
            data_root=data_root,
            parsed_id=parsed_id,
            metadata=metadata,
            parsed_text=parsed_text,
            labels=labels,
            text_status=text_status,
        )
    except Exception as error:  # noqa: BLE001 - every bad source row must survive.
        return _empty_record(path=path, data_root=data_root, error=error)


def _merge_person_evidence(records: Sequence[SequenceRecord]) -> SequenceEvidence:
    """Union all post-process segments belonging to one tracker ID."""
    return SequenceEvidence(
        object_labels=tuple(
            sorted({label for record in records for label in record.object_labels})
        ),
        action_labels=tuple(
            sorted({label for record in records for label in record.action_labels})
        ),
        relation_cues=tuple(
            sorted({cue for record in records for cue in record.relation_cues})
        ),
    )


def _person_caption_map(
    records: Sequence[SequenceRecord],
) -> Mapping[int, tuple[str, ...]]:
    """Collect unique captions per tracker ID in stable order."""
    captions: dict[int, list[str]] = defaultdict(list)
    for record in records:
        if record.caption and record.caption not in captions[record.person_id]:
            captions[record.person_id].append(record.caption)
    return {person_id: tuple(values) for person_id, values in captions.items()}


def build_shared_object_pairs(
    group_id: str,
    records: Sequence[SequenceRecord],
) -> list[dict[str, Any]]:
    """Build only pairs with shared text-object evidence, avoiding all-pairs blowup."""
    by_person: dict[int, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        by_person[record.person_id].append(record)
    evidence = {
        person_id: _merge_person_evidence(person_records)
        for person_id, person_records in by_person.items()
    }
    object_people: dict[str, set[int]] = defaultdict(set)
    for person_id, item in evidence.items():
        for object_label in item.object_labels:
            object_people[object_label].add(person_id)
    pair_objects: dict[tuple[int, int], set[str]] = defaultdict(set)
    for object_label, people in object_people.items():
        for pair in itertools.combinations(sorted(people), 2):
            pair_objects[pair].add(object_label)
    captions = _person_caption_map(records)
    rows: list[dict[str, Any]] = []
    source = records[0].source
    for (first_id, second_id), shared_objects in sorted(pair_objects.items()):
        classification = classify_pair(evidence[first_id], evidence[second_id])
        rows.append(
            {
                "source": source,
                "group_id": group_id,
                "clip_id": records[0].clip_id,
                "person_a": first_id,
                "person_b": second_id,
                "shared_objects": ",".join(sorted(shared_objects)),
                "relation_cues": ",".join(classification.relation_cues),
                "tier": classification.tier,
                "person_a_segments": len(by_person[first_id]),
                "person_b_segments": len(by_person[second_id]),
                "person_a_captions_json": json.dumps(
                    captions.get(first_id, ()), ensure_ascii=False
                ),
                "person_b_captions_json": json.dumps(
                    captions.get(second_id, ()), ensure_ascii=False
                ),
            }
        )
    return rows


def _git_commit(repo: Path) -> str:
    """Resolve one repository commit without invoking a shell."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"
    return result.stdout.strip()


def _load_official_vocab(
    collection_root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read official label lists through AST without importing project code."""
    constants_path = collection_root / "utils" / "analyze_results" / "constant_class.py"
    if not constants_path.is_file():
        LOGGER.warning(
            "Official label file missing; using embedded fallback vocabulary"
        )
        return OBJECT_LABELS, ACTION_LABELS
    tree = ast.parse(constants_path.read_text(encoding="utf-8"))
    values: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {
            "object_class_list",
            "action_class_list",
        }:
            parsed = ast.literal_eval(node.value)
            values[target.id] = tuple(str(item).lower() for item in parsed)
    objects = tuple(
        label
        for label in values.get("object_class_list", OBJECT_LABELS)
        if label not in {"human", "unknown", "none"}
    )
    actions = tuple(dict.fromkeys(values.get("action_class_list", ACTION_LABELS)))
    return objects, actions


def _discover_npz_files(data_root: Path, limit: int | None) -> list[Path]:
    """Discover only the four released source directories."""
    paths = [
        path
        for source in SOURCE_DIRS
        for path in sorted((data_root / source).glob("*.npz"))
    ]
    return paths[:limit] if limit is not None else paths


def scan_release(
    data_root: Path,
    collection_root: Path,
    workers: int,
    limit: int | None,
) -> list[SequenceRecord]:
    """Scan all release files with bounded thread parallelism."""
    object_vocabulary, action_vocabulary = _load_official_vocab(collection_root)
    paths = _discover_npz_files(data_root=data_root, limit=limit)
    LOGGER.info("Discovered %d NPZ files", len(paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(
            executor.map(
                lambda path: scan_sequence(
                    path=path,
                    data_root=data_root,
                    object_vocabulary=object_vocabulary,
                    action_vocabulary=action_vocabulary,
                ),
                paths,
            )
        )
    return records
