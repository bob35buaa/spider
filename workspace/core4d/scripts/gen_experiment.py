#!/usr/bin/env python3
"""Generate experiment script scaffolding from templates.

Usage:
    python scripts/gen_experiment.py \
        --exp-id E153 \
        --description "test experiment" \
        --splits "local-gpu0,remote-gpu0,remote-gpu1" \
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate experiment scripts from templates."
    )
    parser.add_argument(
        "--exp-id",
        required=True,
        help="Experiment ID, e.g. E153",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Short description for script header comments",
    )
    parser.add_argument(
        "--variants-tsv",
        default=None,
        help="Path to variants TSV (relative to repo root). "
        "Default: workspace/core4d/scripts/{EXP_ID}/variants.tsv",
    )
    parser.add_argument(
        "--result-root",
        default=None,
        help="Result output root (relative to repo root). "
        "Default: workspace/core4d/results/{EXP_ID}",
    )
    parser.add_argument(
        "--splits",
        default="local-gpu0,remote-gpu0,remote-gpu1",
        help="Comma-separated split names for the train script mode dispatch",
    )
    parser.add_argument(
        "--override-prefix",
        default=None,
        help="Override YAML prefix for rsync. Default: core4d_{EXP_ID}",
    )
    parser.add_argument(
        "--exp-slug",
        default=None,
        help="Slug for naming files, e.g. 'eef_offset_sweep'. "
        "Default: derived from --description",
    )
    parser.add_argument(
        "--split-col",
        type=int,
        default=12,
        help="TSV column index (1-based) for the 'split' field (default: 12)",
    )
    parser.add_argument(
        "--status-col",
        type=int,
        default=13,
        help="TSV column index (1-based) for 'run_status' field (default: 13)",
    )
    parser.add_argument(
        "--task-col",
        type=int,
        default=21,
        help="TSV column index (1-based) for the 'task' field (default: 21)",
    )
    parser.add_argument(
        "--override-col",
        type=int,
        default=27,
        help="TSV column index (1-based) for the 'override' path field (default: 27)",
    )
    parser.add_argument(
        "--with-watcher",
        action="store_true",
        help="Also generate a watch_and_pull script from template",
    )
    parser.add_argument(
        "--remote-host",
        default="spider-remote",
        help="Remote SSH host for watcher (default: spider-remote)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=600,
        help="Poll interval in seconds for watcher (default: 600)",
    )
    parser.add_argument(
        "--expected-npz-count",
        type=int,
        default=None,
        help="Expected NPZ artifact count for watcher. "
        "Default: inferred from number of splits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    return parser.parse_args()


def slug_from_description(description: str) -> str:
    """Convert description to a file-name slug."""
    slug = description.lower().strip()
    slug = slug.replace(" ", "_").replace("-", "_")
    # Keep only alphanumeric and underscores
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    # Collapse multiple underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "experiment"


def render_template(template_path: Path, variables: dict[str, str]) -> str:
    """Read template and substitute {{VARIABLE}} placeholders."""
    content = template_path.read_text(encoding="utf-8")
    for key, value in variables.items():
        content = content.replace("{{" + key + "}}", value)
    return content


def make_executable(path: Path) -> None:
    """Add executable permission to a file."""
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    args = parse_args()

    exp_id = args.exp_id.upper() if not args.exp_id[0].isalpha() else args.exp_id
    exp_slug = args.exp_slug or slug_from_description(args.description)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits_pipe = "|".join(splits)
    default_split = splits[0] if splits else "local-gpu0"
    variants_tsv = args.variants_tsv or f"workspace/core4d/scripts/{exp_id}/variants.tsv"
    result_root = args.result_root or f"workspace/core4d/results/{exp_id}"
    override_prefix = args.override_prefix or f"core4d_{exp_id}"
    exp_id_lower = exp_id.lower()

    train_script_name = f"train_{exp_id}_{exp_slug}.sh"

    variables = {
        "EXP_ID": exp_id,
        "EXP_ID_LOWER": exp_id_lower,
        "DESCRIPTION": args.description or f"{exp_id} experiment",
        "EXP_SLUG": exp_slug,
        "VARIANTS_TSV_PATH": variants_tsv,
        "RESULT_ROOT": result_root,
        "OVERRIDE_PREFIX": override_prefix,
        "SPLITS_PIPE": splits_pipe,
        "DEFAULT_SPLIT": default_split,
        "TRAIN_SCRIPT": train_script_name,
        "SPLIT_COL": str(args.split_col),
        "STATUS_COL": str(args.status_col),
        "TASK_COL": str(args.task_col),
        "OVERRIDE_COL": str(args.override_col),
    }

    # Define output files
    outputs: list[tuple[Path, str]] = []

    # 1. Train script
    train_template = TEMPLATE_DIR / "train_template.sh"
    train_output = SCRIPT_DIR / "train" / train_script_name
    outputs.append((train_output, render_template(train_template, variables)))

    # 2. Remote launcher
    remote_template = TEMPLATE_DIR / "remote_template.sh"
    remote_output = SCRIPT_DIR / f"run_{exp_id}_remote.sh"
    outputs.append((remote_output, render_template(remote_template, variables)))

    # 3. Pull script
    pull_template = TEMPLATE_DIR / "pull_template.sh"
    pull_output = SCRIPT_DIR / f"pull_{exp_id}_remote_results.sh"
    outputs.append((pull_output, render_template(pull_template, variables)))

    # 4. Eval wrapper
    eval_template = TEMPLATE_DIR / "eval_wrapper_template.sh"
    eval_output = SCRIPT_DIR / "eval" / f"eval_{exp_id}_{exp_slug}.sh"
    outputs.append((eval_output, render_template(eval_template, variables)))

    # 5. Watch-and-pull script (optional)
    if args.with_watcher:
        watcher_template = TEMPLATE_DIR / "watch_and_pull_template.sh"
        watcher_output = SCRIPT_DIR / f"watch_and_pull_{exp_id_lower}.sh"

        # Infer expected NPZ count from splits if not explicitly given.
        # Total splits count as a rough proxy for expected artifacts.
        expected_npz = args.expected_npz_count or len(splits)

        # Use repo-relative paths (scripts run from repo root via cd "$(git rev-parse ...)")
        pull_rel = f"workspace/core4d/scripts/pull_{exp_id}_remote_results.sh"
        eval_rel = f"workspace/core4d/scripts/eval/eval_{exp_id}_{exp_slug}.sh"

        watcher_variables = {
            "EXP_ID": exp_id,
            "REMOTE_HOST": args.remote_host,
            "LOCAL_TMUX": f"{exp_id_lower}_local",
            "REMOTE_TMUX": f"{exp_id_lower}_remote",
            "EXPECTED_NPZ_COUNT": str(expected_npz),
            "PULL_SCRIPT": pull_rel,
            "EVAL_SCRIPT": eval_rel,
            "RESULT_ROOT": result_root,
            "POLL_INTERVAL": str(args.poll_interval),
        }
        outputs.append(
            (watcher_output, render_template(watcher_template, watcher_variables))
        )

    if args.dry_run:
        print(f"[dry-run] Would generate {len(outputs)} files for {exp_id}:")
        print()
        for path, content in outputs:
            lines = content.splitlines()
            preview = "\n".join(lines[:5])
            print(f"  {path}")
            print(f"    lines: {len(lines)}")
            print(f"    preview:")
            for line in preview.splitlines():
                print(f"      {line}")
            print()
        return 0

    # Write files
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        make_executable(path)
        print(f"  created: {path}")

    # Also create the experiment scripts directory
    exp_scripts_dir = SCRIPT_DIR / exp_id
    exp_scripts_dir.mkdir(parents=True, exist_ok=True)
    print(f"  created dir: {exp_scripts_dir}/")

    print(f"\nGenerated {len(outputs)} scripts for {exp_id}.")
    print(f"Next steps:")
    print(f"  1. Create {variants_tsv}")
    print(f"  2. Add experiment-specific logic to run_one() in {train_output.name}")
    print(f"  3. Create eval Python script: workspace/core4d/scripts/eval/eval_{exp_id}_{exp_slug}.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
