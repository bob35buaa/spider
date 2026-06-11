"""Collect existing RL-ready videos into the E102 visual handoff directory."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-ready", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in read_tsv(args.rl_ready):
        src = Path(row.get("video_path", ""))
        if not src.is_file():
            continue
        dst = args.out_dir / f"{row['case_name']}_{row['provenance']}.mp4"
        shutil.copy2(src, dst)
        copied.append((row["case_name"], dst))
    lines = ["# E102 RL-ready Visuals", ""]
    for case, dst in copied:
        lines.append(f"- `{case}`: `{dst.name}`")
    (args.out_dir / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"copied {len(copied)} videos -> {args.out_dir}")


if __name__ == "__main__":
    main()
