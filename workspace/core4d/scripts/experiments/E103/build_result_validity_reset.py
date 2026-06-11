#!/usr/bin/env python3
"""Write E103 result validity reset note from affected scene registry."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_tsv(args.registry)
    action_counts = Counter(row["action"] for row in rows)
    invalid = [row for row in rows if row["action"] == "quarantine_invalidated_by_scene_inertial_bug"]
    rebuild = [
        row
        for row in rows
        if row["action"] in {"rebuild_source_template", "rebuild_missing_source_template"}
    ]
    keep = [row for row in rows if row["action"] == "keep_clean"]
    review = [row for row in rows if row["action"] == "review_before_use"]

    lines = [
        "# E103 Result Validity Reset",
        "",
        "This file records the validity policy after the scene inertial audit.",
        "It does not delete or rewrite historical logs; it defines how their labels may be used going forward.",
        "",
        "## Counts",
        "",
        "| action | count | meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "keep_clean": "scene audit supports continued use, subject to task-specific evidence",
        "review_before_use": "robot inertial is not polluted, but geometry/mass policy needs explicit review before reuse",
        "rebuild_source_template": "canonical template must be rebuilt before any downstream target regeneration",
        "rebuild_missing_source_template": "canonical source template is missing and must be created before use",
        "quarantine_invalidated_by_scene_inertial_bug": "do not use dynamics/CEM outcome as positive or negative label",
    }
    for action, count in action_counts.most_common():
        lines.append(f"| `{action}` | {count} | {meanings.get(action, '')} |")

    lines.extend(
        [
            "",
            "## Invalidated Dynamics Labels",
            "",
            "All rows with `quarantine_invalidated_by_scene_inertial_bug` used a scene where robot link inertials were polluted by the Box021 object inertial. Their geometry/semantic observations can still inform hypotheses, but CEM/RL success or failure labels must not be used as hard labels.",
            "",
            f"- Invalidated scene count: `{len(invalid)}`",
            "",
            "Representative invalidated tasks:",
            "",
        ]
    )
    for row in invalid[:20]:
        lines.append(f"- `{row['task']}`")
    if len(invalid) > 20:
        lines.append(f"- ... plus `{len(invalid) - 20}` more rows in `affected_scene_registry.tsv`")

    lines.extend(
        [
            "",
            "## Source Templates To Rebuild",
            "",
        ]
    )
    if rebuild:
        for row in rebuild:
            lines.append(
                f"- `{row['task']}`: inertial=`{row['inertial_status']}`, geometry=`{row['geometry_status']}`"
            )
    else:
        lines.append("- None in the current registry.")

    lines.extend(
        [
            "",
            "## Guard Rows",
            "",
            "Rows marked `keep_clean` are not automatically WORK cases; they only pass the scene-level audit. Their behavioral labels still require the original gate/video evidence.",
            "",
        ]
    )
    for row in keep[:20]:
        lines.append(f"- `{row['task']}`")
    if len(keep) > 20:
        lines.append(f"- ... plus `{len(keep) - 20}` more")

    lines.extend(
        [
            "",
            "## Review Rows",
            "",
            "`review_before_use` means the robot inertial audit did not find the Box021 pollution, but geometry policy differs from strict mesh-AABB or mass policy needs explicit justification.",
            "",
        ]
    )
    for row in review[:20]:
        lines.append(
            f"- `{row['task']}`: inertial=`{row['inertial_status']}`, geometry=`{row['geometry_status']}`"
        )
    if len(review) > 20:
        lines.append(f"- ... plus `{len(review) - 20}` more rows in `affected_scene_registry.tsv`")

    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Do not hard-exclude or hard-accept candidates using polluted-scene dynamics outcomes.",
            "- Canonical Box021/Box022/Box026 source templates must remain clean before target regeneration.",
            "- Re-run raw/contact/quat/target audits after target regeneration.",
            "- Open a new dynamics/CEM experiment only after E103 clean data gates pass.",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
