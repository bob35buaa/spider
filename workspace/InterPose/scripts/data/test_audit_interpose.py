"""Unit tests for the InterPose dataset auditor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import audit_interpose as auditor  # noqa: E402
from audit_interpose import (  # noqa: E402
    SequenceEvidence,
    classify_pair,
    extract_labels,
    parse_sequence_id,
    parse_text_record,
)


@pytest.mark.unit
def test_parse_sequence_id_recovers_track_suffix() -> None:
    """The final two integers are person and post-process segment IDs."""
    parsed = parse_sequence_id("hdvila_video_part_001_12_3.npz")

    assert parsed.clip_id == "hdvila_video_part_001"
    assert parsed.person_id == 12
    assert parsed.segment_id == 3


@pytest.mark.unit
def test_parse_sequence_id_rejects_missing_numeric_suffix() -> None:
    """Files without the official exporter suffix must fail explicitly."""
    with pytest.raises(ValueError, match="person/segment suffix"):
        parse_sequence_id("hdvila_video_part_001.npz")


@pytest.mark.unit
def test_parse_text_record_uses_right_split() -> None:
    """The parser preserves caption text and extracts lemmatized POS tokens."""
    raw_text = (
        "The person passes a ball to another person."
        "#The/DET person/NOUN pass/VERB a/DET ball/NOUN to/ADP "
        "another/DET person/NOUN#0.0#0.0\n"
    )

    parsed = parse_text_record(raw_text)

    assert parsed.caption == "The person passes a ball to another person."
    assert parsed.verb_lemmas == ("pass",)
    assert parsed.noun_lemmas == ("person", "ball", "person")
    assert parsed.start_time == pytest.approx(0.0)
    assert parsed.end_time == pytest.approx(0.0)


@pytest.mark.unit
def test_extract_labels_detects_shared_object_and_relation() -> None:
    """Official labels and cooperation cues are deterministic."""
    labels = extract_labels(
        caption="The person passes a ball to another person and they move together.",
        lemma_tokens=("the", "person", "pass", "a", "ball", "to", "another", "person"),
        verb_lemmas=("pass", "move"),
        noun_lemmas=("person", "ball", "person"),
    )

    assert labels.object_labels == ("ball",)
    assert labels.action_labels == ("move", "pass")
    assert labels.relation_cues == ("another person", "pass", "together")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("objects_a", "objects_b", "cues", "expected_tier"),
    [
        (("ball",), ("ball",), ("together",), "HOH_TEXT_STRICT_CANDIDATE"),
        (("box",), ("box",), (), "HOH_SHARED_OBJECT_CANDIDATE"),
        (("ball",), ("racket",), ("together",), "MULTI_PERSON_NO_SHARED_OBJECT"),
    ],
)
def test_classify_pair_uses_shared_object_and_relation_cues(
    objects_a: tuple[str, ...],
    objects_b: tuple[str, ...],
    cues: tuple[str, ...],
    expected_tier: str,
) -> None:
    """Pair tiers distinguish strict text cues from broad co-occurrence."""
    first = SequenceEvidence(
        object_labels=objects_a,
        action_labels=("hold",),
        relation_cues=cues,
    )
    second = SequenceEvidence(
        object_labels=objects_b,
        action_labels=("hold",),
        relation_cues=(),
    )

    classification = classify_pair(first, second)

    assert classification.tier == expected_tier


def _write_sequence(
    data_root: Path,
    source: str,
    stem: str,
    caption: str,
    token_text: str,
    *,
    write_text: bool = True,
) -> None:
    """Create one minimal released-style NPZ and optional HumanML3D text row."""
    np.savez(
        data_root / source / f"{stem}.npz",
        poses=np.zeros((4, 165), dtype=np.float64),
        trans=np.zeros((4, 3), dtype=np.float64),
        betas=np.zeros((10,), dtype=np.float32),
        num_betas=10,
        gender="neutral",
        mocap_frame_rate=30,
        text=caption,
    )
    if write_text:
        (data_root / "texts" / f"{stem}.txt").write_text(
            f"{caption}#{token_text}#0.0#0.0\n",
            encoding="utf-8",
        )


SYNTHETIC_SEQUENCES = (
    (
        "Charades",
        "Charades_shared_0_0",
        "The person passes a ball to another person.",
        "The/DET person/NOUN pass/VERB a/DET ball/NOUN to/ADP another/DET person/NOUN",
        True,
    ),
    (
        "Charades",
        "Charades_shared_1_0",
        "The person receives the ball and they move together.",
        "The/DET person/NOUN receive/VERB the/DET ball/NOUN and/CCONJ "
        "they/PRON move/VERB together/ADV",
        True,
    ),
    (
        "hdvila",
        "hdvila_single_0_0",
        "The person opens a door.",
        "The/DET person/NOUN open/VERB a/DET door/NOUN",
        True,
    ),
    (
        "kinetics",
        "kinetics_single_0_0",
        "The person holds a racket.",
        "The/DET person/NOUN hold/VERB a/DET racket/NOUN",
        False,
    ),
    (
        "online_video",
        "online_single_0_0",
        "The person carries a box.",
        "The/DET person/NOUN carry/VERB a/DET box/NOUN",
        True,
    ),
    (
        "online_video",
        "invalid_suffix",
        "The person moves.",
        "The/DET person/NOUN move/VERB",
        True,
    ),
)


def _write_synthetic_release(data_root: Path) -> None:
    """Create a four-source release with one multi-person clip and one bad row."""
    for directory in (*auditor.SOURCE_DIRS, "texts"):
        (data_root / directory).mkdir(parents=True)
    for source, stem, caption, token_text, write_text in SYNTHETIC_SEQUENCES:
        _write_sequence(
            data_root,
            source,
            stem,
            caption,
            token_text,
            write_text=write_text,
        )


def _write_synthetic_code_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal official and collection repository boundaries."""
    official_root = tmp_path / "official"
    collection_root = tmp_path / "collection"
    constants_dir = collection_root / "utils" / "analyze_results"
    official_root.mkdir()
    constants_dir.mkdir(parents=True)
    (constants_dir / "constant_class.py").write_text(
        "object_class_list = ['ball', 'door', 'racket', 'box', 'human', "
        "'unknown', 'None']\n"
        "action_class_list = ['pass', 'receive', 'move', 'open', 'hold', "
        "'carry']\n",
        encoding="utf-8",
    )
    return official_root, collection_root


@pytest.mark.unit
def test_full_cli_audits_synthetic_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny four-source release exercises scan, aggregate, and output contracts."""
    data_root = tmp_path / "InterPose"
    _write_synthetic_release(data_root)
    official_root, collection_root = _write_synthetic_code_roots(tmp_path)
    output_root = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_interpose.py",
            "--data-root",
            str(data_root),
            "--official-code-root",
            str(official_root),
            "--collection-code-root",
            str(collection_root),
            "--output-root",
            str(output_root),
            "--workers",
            "2",
        ],
    )

    assert auditor.main() == 0

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["local_release"]["sequence_count"] == 6
    assert summary["local_release"]["schema_status_counts"] == {"ERROR": 1, "OK": 5}
    assert summary["track_groups"]["multi_person_group_count"] == 1
    assert summary["hoh_screen"]["strict_text_pair_candidate_count"] == 1
    assert summary["hoh_screen"]["high_priority_text_pair_candidate_count"] == 1
    assert summary["hoh_screen"]["confirmed_target_hoh_count"] == 0
    assert (output_root / "sequence_inventory.tsv").is_file()
    assert (output_root / "multi_person_groups.tsv").is_file()
    assert (output_root / "hoh_pair_candidates.tsv").is_file()
