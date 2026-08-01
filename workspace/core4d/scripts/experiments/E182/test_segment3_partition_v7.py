#!/usr/bin/env python3
"""Direct-main contracts for segment3 partial-merge partition search v7."""

from __future__ import annotations

from search_segment3_partition_v7 import (
    _parent_v6_evidence,
    best_near_miss_context,
    canonical_partitions,
    source_dependencies,
)


def test_v7_parent_authority_contract() -> None:
    """V7 must start only from the complete 0/6561 v6 search evidence."""
    evidence = _parent_v6_evidence()
    assert evidence["status"] == "V6_COMPLETE_NO_SEGMENT_THRESHOLD_HYBRID"
    assert evidence["selected_candidate_count"] == 0
    assert evidence["p_floor_passing_counts"] == {"K16": 0, "K32": 0, "K8": 0}


def test_four_part_canonical_partition_family() -> None:
    """All 15 set partitions must appear once in deterministic canonical order."""
    partitions = canonical_partitions((0, 1, 2, 3))
    assert len(partitions) == 15
    assert len(set(partitions)) == 15
    assert ((0,), (1,), (2,), (3,)) in partitions
    assert ((0, 1, 2, 3),) in partitions
    for partition in partitions:
        assert tuple(sorted(value for block in partition for value in block)) == (
            0,
            1,
            2,
            3,
        )
        assert all(tuple(sorted(block)) == block for block in partition)


def test_real_v6_near_miss_contexts_are_frozen_inputs() -> None:
    """The context selector must reproduce the audited K16/K32 near misses."""
    k16 = best_near_miss_context(16)
    assert k16["threshold_indices_by_segment"] == [0, 0, 0, 2, 2, 0, 0, 1]
    assert k16["actual_hulls"] == 15
    assert k16["true_positive_count"] == 18
    assert k16["phantom_contact_count"] == 8
    assert k16["missed_contact_count"] == 9
    k32 = best_near_miss_context(32)
    assert k32["threshold_indices_by_segment"] == [0, 2, 2, 2, 2, 2, 2, 0]
    assert k32["actual_hulls"] == 18
    assert k32["true_positive_count"] == 18
    assert k32["phantom_contact_count"] == 8
    assert k32["missed_contact_count"] == 9


def test_v7_directly_binds_all_runtime_math_sources() -> None:
    """The frozen protocol must reject later changes to core, v6, or v7 math."""
    sources = source_dependencies()
    assert [source["role"] for source in sources] == [
        "UNCHANGED_P_CONTACT_MATH",
        "V6_RUNTIME_QUERY_CONTACT_PARENT_HELPERS",
        "V7_PARTITION_SEARCH",
    ]
    assert len({source["path"] for source in sources}) == 3
    assert all(len(source["sha256"]) == 64 for source in sources)


def main() -> int:
    """Run v7 contracts without pytest discovery."""
    tests = (
        test_v7_parent_authority_contract,
        test_four_part_canonical_partition_family,
        test_real_v6_near_miss_contexts_are_frozen_inputs,
        test_v7_directly_binds_all_runtime_math_sources,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"E182_SEGMENT3_PARTITION_V7_TESTS=PASS count={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
