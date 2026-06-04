#!/usr/bin/env python3
"""Evaluate E118 carry-corridor soft-gate diagnostic outputs."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
E115_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E115_lowerbody_contact.py"
E118_VARIANTS = REPO / "workspace/core4d/scripts/E118/variants.tsv"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def load_e115_eval():
    spec = importlib.util.spec_from_file_location("eval_E115_reused_for_E118", E115_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {E115_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VARIANTS_TSV = E118_VARIANTS
    return module


def retitle_summary(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "E115 Lower-Body-Aware Contact",
        "E118 Carry Corridor",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full", choices=["smoke", "full"])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    e115 = load_e115_eval()
    results_dir = repo_path_from_env(
        "RESULTS",
        args.results_dir or REPO / f"workspace/core4d/results/E118/cem/{args.stage}",
    )
    rows = e115.read_variants()
    missing = e115.fail_if_missing(args.stage, rows, results_dir, args.allow_missing)
    method_rows, decisions = e115.evaluate(args.stage, rows, results_dir)
    e115.write_outputs(args.stage, method_rows, decisions, missing, results_dir)
    retitle_summary(results_dir / f"{args.stage}_eval_summary.md")
    print(f"wrote {rel(results_dir / f'{args.stage}_eval_summary.md')}")


if __name__ == "__main__":
    main()
