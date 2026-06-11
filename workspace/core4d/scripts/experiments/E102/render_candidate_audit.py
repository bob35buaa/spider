"""Render E102 Phase 2 candidate mining REVIEW.md."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_tsv(args.candidates)
    rejected = read_tsv(args.rejected)
    route_counts: dict[str, int] = {}
    for row in rejected:
        route_counts[row["route"]] = route_counts.get(row["route"], 0) + 1

    lines = [
        "# Candidate Audit",
        "",
        f"- Candidate rows: {len(candidates)}",
        f"- Rejected/held rows: {len(rejected)}",
        "",
        "## Candidates",
        "",
        "| rank | target | route | score | object | raw | fingertip | note |",
        "|---:|---|---|---:|---|---|---|---|",
    ]
    if candidates:
        for row in candidates[:20]:
            fingertip = f"L {row.get('L_vote', '')}/{row.get('L_contact', '')}; R {row.get('R_vote', '')}/{row.get('R_contact', '')}"
            lines.append(
                f"| {row.get('rank', '')} | `{row.get('target_task', '')}` | `{row.get('route', '')}` | {row.get('score', '')} | {row.get('object_key', '')} | {row.get('stage1_decision', '')} | {fingertip} | {row.get('notes', '').replace('|', '/')} |"
            )
    else:
        lines.append("|  |  | `none` |  |  |  |  | no executable candidate after E099/E101 constraints |")

    lines += [
        "",
        "## Rejection Counts",
        "",
        "| route | count |",
        "|---|---:|",
    ]
    for route, count in sorted(route_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{route}` | {count} |")

    lines += [
        "",
        "## Top Held Rows",
        "",
        "| target | route | object | raw | source_scene_live | reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in rejected[:30]:
        source_state = row.get("source_scene_exists_live", "") or row.get("source_scene_exists", "")
        lines.append(
            f"| `{row.get('target_task', '')}` | `{row.get('route', '')}` | {row.get('object_key', '')} | {row.get('stage1_decision', '')} | {source_state} | {row.get('reason', '').replace('|', '/')} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "- `v2_candidates_with_fingertip.tsv` is the authority for Phase 3 selection.",
    ]
    if len(candidates) < 2:
        lines.append("- With fewer than 2 executable candidates, full CEM must be skipped under the current data-gate rule.")
    else:
        lines.append("- At least 2 executable candidates are available; the next phase should select typical-2 after visual/fingertip review.")
    lines.append("")
    if args.summary.is_file():
        lines += ["## Mining Summary File", "", f"- {args.summary}", ""]
    (args.out_dir / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"review -> {args.out_dir / 'REVIEW.md'}")


if __name__ == "__main__":
    main()
