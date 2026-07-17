#!/usr/bin/env python3
"""Build the exact E168 object/action/seed recall and production inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


TARGET_OBJECTS = {"box004", "box021", "bucket004"}
MOVE_ACTION_RE = re.compile(r"^move[12]_obs[013]$")
SEED_CASES = {
    "box004_20231003_2_083_p1",
    "box004_20231003_2_083_p2",
    "box021_20231018_029_p2",
    "box021_20231011_035_p1",
    "box021_20231011_035_p2",
    "bucket004_20231002_022_p1",
}
EXPECTED_OBJECT_COUNTS = {
    "box004": {"recalled": 20, "move": 14, "candidate": 12, "obs0": 2, "obs13": 10},
    "box021": {"recalled": 50, "move": 36, "candidate": 33, "obs0": 13, "obs13": 20},
    "bucket004": {"recalled": 32, "move": 22, "candidate": 21, "obs0": 7, "obs13": 14},
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obstacle_level(action: str) -> str:
    return action.rsplit("_", 1)[-1] if MOVE_ACTION_RE.fullmatch(action) else ""


def partner_identity(row: dict[str, str]) -> tuple[str, str]:
    if row["person"] == "person1":
        return "person2", f"{row['object_key']}_{row['date']}_{row['seq']}_p2"
    if row["person"] == "person2":
        return "person1", f"{row['object_key']}_{row['date']}_{row['seq']}_p1"
    raise ValueError(f"unexpected person: {row['person']}")


def classify(row: dict[str, str]) -> dict[str, str]:
    action = row.get("action", "")
    case_id = row.get("case_id", "")
    in_scope = MOVE_ACTION_RE.fullmatch(action) is not None
    is_seed = case_id in SEED_CASES
    partner_person, partner_case_id = partner_identity(row)
    obstacle = obstacle_level(action)
    if not in_scope:
        scope_status = "excluded_non_move_action"
        decision = "OUT_OF_SCOPE_E168"
        tier = ""
    elif is_seed:
        scope_status = "move_seed"
        decision = "EXCLUDED_SEED"
        tier = ""
    else:
        scope_status = "move_candidate"
        decision = "S1_RAW_CONTACT_PENDING"
        tier = "tier_a_obs0" if obstacle == "obs0" else "tier_b_obs1_obs3"
    return {
        "sequence_key": f"{row['object_key']}_{row['date']}_{row['seq']}",
        "source_person": row["person"],
        "partner_person": partner_person,
        "partner_case_id": partner_case_id,
        "source_action": action,
        "source_obstacle_level": obstacle,
        "obstacle_context_not_reconstructed": str(obstacle in {"obs1", "obs3"}).lower(),
        "scope_status": scope_status,
        "seed_status": "seed" if is_seed else "non_seed",
        "candidate_tier": tier,
        "execution_decision": decision,
    }


def assert_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        duplicates = [case for case, count in Counter(case_ids).items() if count > 1]
        raise SystemExit(f"duplicate source person cases: {duplicates}")

    move_rows = [row for row in rows if MOVE_ACTION_RE.fullmatch(row["action"])]
    seed_rows = [row for row in move_rows if row["case_id"] in SEED_CASES]
    candidates = [row for row in move_rows if row["case_id"] not in SEED_CASES]
    missing_seeds = sorted(SEED_CASES - {row["case_id"] for row in seed_rows})
    if missing_seeds:
        raise SystemExit(f"E168 seed cases missing from recall: {missing_seeds}")

    totals = {
        "recalled": len(rows),
        "move": len(move_rows),
        "seed": len(seed_rows),
        "candidate": len(candidates),
        "obs0": sum(obstacle_level(row["action"]) == "obs0" for row in candidates),
        "obs13": sum(obstacle_level(row["action"]) in {"obs1", "obs3"} for row in candidates),
    }
    expected_totals = {
        "recalled": 102,
        "move": 72,
        "seed": 6,
        "candidate": 66,
        "obs0": 22,
        "obs13": 44,
    }
    if totals != expected_totals:
        raise SystemExit(f"E168 recall total mismatch: expected={expected_totals}, actual={totals}")

    by_object: dict[str, dict[str, int]] = {}
    for object_key in sorted(TARGET_OBJECTS):
        object_rows = [row for row in rows if row["object_key"] == object_key]
        object_move = [row for row in object_rows if MOVE_ACTION_RE.fullmatch(row["action"])]
        object_candidates = [
            row for row in object_move if row["case_id"] not in SEED_CASES
        ]
        actual = {
            "recalled": len(object_rows),
            "move": len(object_move),
            "candidate": len(object_candidates),
            "obs0": sum(
                obstacle_level(row["action"]) == "obs0" for row in object_candidates
            ),
            "obs13": sum(
                obstacle_level(row["action"]) in {"obs1", "obs3"}
                for row in object_candidates
            ),
        }
        if actual != EXPECTED_OBJECT_COUNTS[object_key]:
            raise SystemExit(
                f"E168 {object_key} count mismatch: "
                f"expected={EXPECTED_OBJECT_COUNTS[object_key]}, actual={actual}"
            )
        by_object[object_key] = actual

    sequence_person_pairs = {
        (row["sequence_key"], row["source_person"])
        for row in rows
        if row["execution_decision"] == "S1_RAW_CONTACT_PENDING"
    }
    if len(sequence_person_pairs) != 66:
        raise SystemExit(
            "source-person uniqueness mismatch: "
            f"expected=66 actual={len(sequence_person_pairs)}"
        )
    return {
        "totals": totals,
        "by_object": by_object,
        "unique_candidate_sequence_person": len(sequence_person_pairs),
        "p1_p2_are_independent_rows": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-tsv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    inventory_path = args.inventory_tsv.expanduser().resolve()
    rows, inventory_fields = read_tsv(inventory_path)
    recalled: list[dict[str, str]] = []
    for row in rows:
        object_key = row.get("object_key", "").lower()
        if object_key not in TARGET_OBJECTS:
            continue
        row = dict(row)
        row["object_key"] = object_key
        row.update(classify(row))
        recalled.append(row)
    recalled.sort(
        key=lambda row: (
            row["object_key"],
            row["date"],
            row["seq"],
            int(row["person_idx"]),
        )
    )
    summary = assert_counts(recalled)

    out_dir = args.out_dir.expanduser().resolve()
    recall_fields = inventory_fields + [
        "sequence_key",
        "source_person",
        "partner_person",
        "partner_case_id",
        "source_action",
        "source_obstacle_level",
        "obstacle_context_not_reconstructed",
        "scope_status",
        "seed_status",
        "candidate_tier",
        "execution_decision",
    ]
    recall_tsv = out_dir / "e168_same_object_recall.tsv"
    write_tsv(recall_tsv, recalled, recall_fields)
    candidates = [
        row for row in recalled if row["execution_decision"] == "S1_RAW_CONTACT_PENDING"
    ]
    candidate_tsv = out_dir / "e168_same_object_candidates.tsv"
    write_tsv(candidate_tsv, candidates, inventory_fields)

    summary.update(
        {
            "created_at": now(),
            "status": "pass",
            "inventory_tsv": str(inventory_path),
            "inventory_sha256": sha256(inventory_path),
            "recall_tsv": str(recall_tsv),
            "candidate_inventory_tsv": str(candidate_tsv),
            "move_action_regex": MOVE_ACTION_RE.pattern,
            "target_objects": sorted(TARGET_OBJECTS),
            "seed_cases": sorted(SEED_CASES),
        }
    )
    write_json(out_dir / "e168_same_object_recall_summary.json", summary)
    (out_dir / "e168_same_object_recall_summary.md").write_text(
        "\n".join(
            [
                "# E168 Same-object Recall",
                "",
                f"- Status: `{summary['status']}`",
                f"- Recalled source-person rows: `{summary['totals']['recalled']}`",
                f"- Move rows: `{summary['totals']['move']}`",
                f"- Seed rows: `{summary['totals']['seed']}`",
                f"- Non-seed candidates: `{summary['totals']['candidate']}`",
                f"- Tier A obs0: `{summary['totals']['obs0']}`",
                f"- Tier B obs1/obs3: `{summary['totals']['obs13']}`",
                "- Identity unit: `(sequence_key, source_person)`; p1/p2 are independent.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
