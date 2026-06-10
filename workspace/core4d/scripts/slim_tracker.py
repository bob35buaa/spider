#!/usr/bin/env python3
"""Slim down EXPERIMENT_TRACKER.md to a compact index table.

Reads the full tracker, extracts the table rows, compresses descriptions
to <=80 chars, and optionally links to log files found in workspace/core4d/log/.

Usage:
    python scripts/slim_tracker.py            # overwrite EXPERIMENT_TRACKER.md
    python scripts/slim_tracker.py --dry-run  # print stats only
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent  # workspace/core4d/
TRACKER_PATH = WORKSPACE / "EXPERIMENT_TRACKER.md"
LOG_DIR = WORKSPACE / "log"


def extract_summary(desc: str, max_len: int = 80) -> str:
    """Extract a short summary from the description column.

    Priority:
      1. First bold segment (**...**).
      2. Text up to the first colon (：or :) or period (。or .).
      3. Truncate to max_len with ellipsis.
    """
    # Try bold text first
    bold_match = re.search(r"\*\*(.+?)\*\*", desc)
    if bold_match:
        summary = bold_match.group(1).strip()
    else:
        # Split on first colon or period (Chinese or ASCII)
        split_match = re.split(r"[：:。.]", desc, maxsplit=1)
        summary = split_match[0].strip()

    # Truncate if needed
    if len(summary) > max_len:
        summary = summary[: max_len - 1] + "…"
    return summary


def find_log_for_experiment(run_id: str) -> str | None:
    """Find the last matching log file for a given experiment run ID.

    run_id can be like 'E151', 'E037-E039', 'E084-audit', etc.
    We extract the primary experiment number and search for log files containing it.
    """
    if not LOG_DIR.is_dir():
        return None

    # Extract the primary experiment number (e.g., 'E151' from 'E151' or 'E037' from 'E037-E039')
    exp_match = re.search(r"E(\d+)", run_id)
    if not exp_match:
        return None

    exp_num = exp_match.group(1)  # e.g., '151'
    # Also try without leading zeros for matching
    exp_num_int = int(exp_num)

    # Collect all matching log files
    matches: list[str] = []
    for fname in os.listdir(LOG_DIR):
        if not fname.endswith(".md"):
            continue
        # Match patterns like: 191_E151_..., 01_E001_..., or files containing _ENNN_
        if re.search(rf"(?:^|\D)E0*{exp_num_int}(?:\D|$)", fname, re.IGNORECASE):
            matches.append(fname)
        elif re.search(rf"(?:^|\D){exp_num_int}(?:\D)", fname):
            # Also try just the number prefix like "191_E151..."
            matches.append(fname)

    if not matches:
        return None

    # Sort and return the last (most recent / highest number) match
    matches.sort()
    last_match = matches[-1]
    # Generate relative link from tracker location
    # Extract the numeric prefix for display
    num_prefix_match = re.match(r"(\d+)", last_match)
    display = num_prefix_match.group(1) if num_prefix_match else last_match[:20]
    return f"[{display}](log/{last_match})"


def parse_and_slim(content: str) -> tuple[str, list[str], int, int]:
    """Parse the tracker content, slim it down.

    Returns:
        (header, slimmed_lines, original_row_count, new_row_count)
    """
    lines = content.split("\n")

    # Find the table: first line starting with "| Run"
    table_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("| Run"):
            table_start = i
            break

    if table_start == -1:
        raise ValueError("Could not find table header '| Run ...' in tracker")

    # Find table end: first non-table line after table start (skip header + separator)
    # Table lines start with '|'
    table_end = table_start
    for i in range(table_start, len(lines)):
        if lines[i].strip().startswith("|") or lines[i].strip() == "":
            table_end = i + 1
        else:
            # Check if this is still part of table (blank lines within table)
            # Actually, stop at first non-pipe line after we've seen data rows
            if i > table_start + 2:  # past header + separator
                table_end = i
                break
    else:
        table_end = len(lines)

    # Header = everything before table
    header_lines = lines[:table_start]

    # Parse table rows (skip header and separator)
    data_rows: list[dict[str, str]] = []
    for i in range(table_start + 2, table_end):  # +2 skips header and |---|---|...|
        line = lines[i].strip()
        if not line.startswith("|"):
            continue
        # Split by | and strip
        cols = [c.strip() for c in line.split("|")]
        # cols[0] and cols[-1] are empty strings from leading/trailing |
        cols = cols[1:-1]
        if len(cols) < 5:
            continue

        run = cols[0].strip()
        date = cols[1].strip()
        phase = cols[2].strip()
        desc = cols[3].strip()
        status = cols[4].strip()

        # Skip empty/placeholder rows
        if run == "—" or not run:
            # Still include planning/audit rows
            if not desc:
                continue

        data_rows.append(
            {
                "run": run,
                "date": date,
                "phase": phase,
                "desc": desc,
                "status": status,
            }
        )

    original_row_count = len(data_rows)

    # Build slimmed table
    new_header = "| Run | 日期 | Phase | 描述 | 状态 | Log |"
    new_separator = "|-----|------|-------|------|------|-----|"

    slimmed_rows: list[str] = []
    for row in data_rows:
        summary = extract_summary(row["desc"])
        log_link = find_log_for_experiment(row["run"]) or ""
        slimmed_rows.append(
            f"| {row['run']} | {row['date']} | {row['phase']} | {summary} | {row['status']} | {log_link} |"
        )

    # Compose output
    output_lines = header_lines + [new_header, new_separator] + slimmed_rows + [""]

    return "\n".join(output_lines), slimmed_rows, original_row_count, len(slimmed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Slim EXPERIMENT_TRACKER.md")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print statistics, do not overwrite",
    )
    args = parser.parse_args()

    content = TRACKER_PATH.read_text(encoding="utf-8")
    original_line_count = len(content.split("\n"))

    output, slimmed_rows, orig_rows, new_rows = parse_and_slim(content)
    new_line_count = len(output.split("\n"))

    # Compute description lengths
    desc_lens = []
    for row in slimmed_rows:
        cols = [c.strip() for c in row.split("|")]
        cols = cols[1:-1]
        if len(cols) >= 4:
            desc_lens.append(len(cols[3]))

    avg_desc = sum(desc_lens) / len(desc_lens) if desc_lens else 0
    max_desc = max(desc_lens) if desc_lens else 0

    print(f"Original: {original_line_count} lines, {orig_rows} data rows")
    print(f"Slimmed:  {new_line_count} lines, {new_rows} data rows")
    print(f"Description: avg={avg_desc:.0f} chars, max={max_desc} chars")
    print(f"Reduction: {original_line_count - new_line_count} lines removed")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
    else:
        TRACKER_PATH.write_text(output, encoding="utf-8")
        print(f"\nWritten to: {TRACKER_PATH}")


if __name__ == "__main__":
    main()
