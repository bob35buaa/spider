#!/usr/bin/env python3
"""Build INDEX.md for the log/ directory.

Scans all *.md files in workspace/core4d/log/, parses sequence number and
experiment ID from filenames, cross-references Phase/date from
EXPERIMENT_TRACKER.md, groups by Phase, and writes log/INDEX.md.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent  # workspace/core4d
LOG_DIR = WORKSPACE / "log"
TRACKER_PATH = WORKSPACE / "EXPERIMENT_TRACKER.md"
OUTPUT_PATH = LOG_DIR / "INDEX.md"


def parse_tracker(tracker_path: Path) -> dict[str, dict]:
    """Parse EXPERIMENT_TRACKER.md to extract experiment metadata.

    Returns a dict keyed by experiment ID (e.g. 'E151') with values:
        {'date': str, 'phase': str, 'phase_num': int, 'description': str, 'status': str}
    """
    experiments: dict[str, dict] = {}
    text = tracker_path.read_text(encoding="utf-8")

    # Match table rows: | Run | 日期 | Phase | 描述 | 状态 |
    # The tracker has multiple table sections; we want the main one.
    row_pattern = re.compile(
        r"^\|\s*(E\d+[a-z]?(?:\s*[-/]\s*\w+)?|Audit|Strategic|Correction|Diagnosis|Update|R4-diag)\s*\|"
        r"\s*(\d{4}-\d{2}-\d{2})\s*\|"
        r"\s*(Phase\s*\d+)\s*\|"
        r"\s*(.*?)\s*\|"
        r"\s*(.*?)\s*\|",
        re.MULTILINE,
    )

    for m in row_pattern.finditer(text):
        run_id = m.group(1).strip()
        date = m.group(2).strip()
        phase_str = m.group(3).strip()
        description = m.group(4).strip()
        status = m.group(5).strip()

        phase_num_match = re.search(r"\d+", phase_str)
        phase_num = int(phase_num_match.group()) if phase_num_match else 0

        experiments[run_id] = {
            "date": date,
            "phase": phase_str,
            "phase_num": phase_num,
            "description": description,
            "status": status,
        }

    # Also handle grouped rows like "E037-E039" in the tracker
    grouped_pattern = re.compile(
        r"^\|\s*(E(\d+)\s*[-–]\s*E(\d+))\s*\|"
        r"\s*(\d{4}-\d{2}-\d{2}(?:~\d{2})?)\s*\|"
        r"\s*(Phase\s*\d+)\s*\|"
        r"\s*(.*?)\s*\|"
        r"\s*(.*?)\s*\|",
        re.MULTILINE,
    )
    for m in grouped_pattern.finditer(text):
        start_num = int(m.group(2))
        end_num = int(m.group(3))
        date = m.group(4).strip()
        phase_str = m.group(5).strip()
        description = m.group(6).strip()
        status = m.group(7).strip()

        phase_num_match = re.search(r"\d+", phase_str)
        phase_num = int(phase_num_match.group()) if phase_num_match else 0

        # Expand the range
        for n in range(start_num, end_num + 1):
            eid = f"E{n:03d}"
            if eid not in experiments:
                experiments[eid] = {
                    "date": date,
                    "phase": phase_str,
                    "phase_num": phase_num,
                    "description": description,
                    "status": status,
                }

    return experiments


def build_phase_ranges(tracker_data: dict[str, dict]) -> dict[int, tuple[int, int]]:
    """Build a mapping from phase_num to (min_exp_num, max_exp_num).

    Used to infer phase for experiments not explicitly in the tracker.
    """
    phase_exp_nums: dict[int, list[int]] = {}
    for k, v in tracker_data.items():
        m = re.match(r"E(\d+)", k)
        if m:
            phase_exp_nums.setdefault(v["phase_num"], []).append(int(m.group(1)))

    return {
        p: (min(nums), max(nums)) for p, nums in phase_exp_nums.items() if nums
    }


def infer_phase_from_ranges(
    exp_num: int, phase_ranges: dict[int, tuple[int, int]]
) -> int | None:
    """Infer phase number for an experiment by checking which phase range it falls in.

    For numbers in gaps between phases, assign to the nearest phase
    (preferring the phase whose upper bound is closest below, i.e. the
    preceding phase, since experiments in gaps are usually late additions).
    """
    # Exact match first
    for phase_num, (lo, hi) in phase_ranges.items():
        if lo <= exp_num <= hi:
            return phase_num

    # Gap handling: find the phase whose range is closest
    best_phase = None
    best_dist = float("inf")
    for phase_num, (lo, hi) in phase_ranges.items():
        # Distance to range
        if exp_num < lo:
            dist = lo - exp_num
        else:  # exp_num > hi
            dist = exp_num - hi
        if dist < best_dist:
            best_dist = dist
            best_phase = phase_num

    # Only assign if gap is small (within 2 experiments of a range boundary)
    if best_dist <= 2:
        return best_phase
    return None


def extract_exp_ids(filename: str) -> list[str]:
    """Extract experiment IDs from filename.

    Handles patterns like:
      - 187_E147_... -> ['E147']
      - 53_E044_E047_... -> ['E044', 'E047']
      - 108_E085_E086_... -> ['E085', 'E086']
      - 70_pre_E060_... -> ['E060']
      - 39_hdmi_vs_mjwp_... -> []
      - 54_collision_box_bug_fix.md -> []
    """
    stem = Path(filename).stem
    return re.findall(r"E\d+[a-z]?", stem, re.IGNORECASE)


def parse_seq(filename: str) -> int:
    """Extract leading sequence number from filename."""
    m = re.match(r"(\d+)", filename)
    return int(m.group(1)) if m else 0


def summarize_filename(filename: str) -> str:
    """Derive a human-readable summary from the filename."""
    stem = Path(filename).stem
    # Remove leading sequence number + underscore
    no_seq = re.sub(r"^\d+_", "", stem)
    # Remove experiment IDs
    no_exp = re.sub(r"E\d+[a-z]?_?", "", no_seq)
    # Clean up leading/trailing underscores
    no_exp = no_exp.strip("_")
    # Replace underscores with spaces, capitalize first letter
    summary = no_exp.replace("_", " ")
    if summary:
        summary = summary[0].upper() + summary[1:]
    return summary


def get_phase_title(phase_num: int, tracker_data: dict[str, dict]) -> str:
    """Build a phase title from tracker descriptions.

    Finds all experiments in this phase and derives a short title from the
    first and last experiment descriptions.
    """
    phase_exps = [
        (k, v) for k, v in tracker_data.items() if v["phase_num"] == phase_num
    ]
    if not phase_exps:
        return ""

    # Get experiment number range
    exp_nums = []
    for k, _ in phase_exps:
        m = re.match(r"E(\d+)", k)
        if m:
            exp_nums.append(int(m.group(1)))

    if exp_nums:
        exp_range = f"E{min(exp_nums):03d}-E{max(exp_nums):03d}"
    else:
        exp_range = ""

    return exp_range


def build_index() -> str:
    """Build the INDEX.md content."""
    tracker_data = parse_tracker(TRACKER_PATH)
    phase_ranges = build_phase_ranges(tracker_data)

    # Collect all log files (excluding INDEX.md)
    log_files = sorted(
        [f.name for f in LOG_DIR.glob("*.md") if f.name != "INDEX.md"],
        key=parse_seq,
    )

    # Group files by phase
    # For each file, determine its phase from the experiment IDs
    phase_groups: dict[int, list[dict]] = {}  # phase_num -> list of file info dicts
    unphased: list[dict] = []

    for filename in log_files:
        seq = parse_seq(filename)
        exp_ids = extract_exp_ids(filename)

        # Find phase from tracker
        phase_num = None
        date = ""
        for eid in exp_ids:
            if eid in tracker_data:
                phase_num = tracker_data[eid]["phase_num"]
                date = tracker_data[eid]["date"]
                break

        # For multi-experiment files, try all IDs
        if phase_num is None:
            # Try case-insensitive and sub-experiment match (e.g. E096b -> E096)
            for eid in exp_ids:
                base_eid = re.match(r"(E\d+)", eid)
                if base_eid and base_eid.group(1) in tracker_data:
                    phase_num = tracker_data[base_eid.group(1)]["phase_num"]
                    date = tracker_data[base_eid.group(1)]["date"]
                    break

        # Fallback: infer phase from experiment number range
        if phase_num is None:
            for eid in exp_ids:
                m = re.match(r"E(\d+)", eid)
                if m:
                    inferred = infer_phase_from_ranges(int(m.group(1)), phase_ranges)
                    if inferred is not None:
                        phase_num = inferred
                        break

        summary = summarize_filename(filename)
        exp_label = ", ".join(exp_ids) if exp_ids else "-"

        entry = {
            "seq": seq,
            "exp_ids": exp_ids,
            "exp_label": exp_label,
            "date": date,
            "summary": summary,
            "filename": filename,
        }

        if phase_num is not None:
            phase_groups.setdefault(phase_num, []).append(entry)
        else:
            unphased.append(entry)

    # Build markdown
    lines: list[str] = []
    lines.append("# 实验日志索引")
    lines.append("")
    lines.append(f"> 自动生成，共 {len(log_files)} 个日志文件。")
    lines.append(f"> 运行 `python workspace/core4d/scripts/build_log_index.py` 更新。")
    lines.append("")

    # Sort phases in descending order (newest first)
    for phase_num in sorted(phase_groups.keys(), reverse=True):
        entries = phase_groups[phase_num]
        entries.sort(key=lambda e: e["seq"])

        phase_range = get_phase_title(phase_num, tracker_data)
        header = f"## Phase {phase_num}"
        if phase_range:
            header += f" ({phase_range})"
        lines.append(header)
        lines.append("")
        lines.append("| # | 实验 | 日期 | 摘要 | 文件 |")
        lines.append("|---|------|------|------|------|")

        for entry in entries:
            link = f"[{entry['filename'][:40]}{'...' if len(entry['filename']) > 40 else ''}]({entry['filename']})"
            lines.append(
                f"| {entry['seq']} | {entry['exp_label']} | {entry['date']} | {entry['summary']} | {link} |"
            )

        lines.append("")

    # Unphased files (no matching experiment in tracker)
    if unphased:
        unphased.sort(key=lambda e: e["seq"])
        lines.append("## 未分类")
        lines.append("")
        lines.append("| # | 实验 | 日期 | 摘要 | 文件 |")
        lines.append("|---|------|------|------|------|")

        for entry in unphased:
            link = f"[{entry['filename'][:40]}{'...' if len(entry['filename']) > 40 else ''}]({entry['filename']})"
            lines.append(
                f"| {entry['seq']} | {entry['exp_label']} | {entry['date']} | {entry['summary']} | {link} |"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    content = build_index()
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Generated {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")

    # Count files covered
    log_files = [f.name for f in LOG_DIR.glob("*.md") if f.name != "INDEX.md"]
    print(f"Total log files: {len(log_files)}")


if __name__ == "__main__":
    main()
