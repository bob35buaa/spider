#!/usr/bin/env python3
"""Evaluate E124 SBTO carry-horizon outputs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


THIS = Path(__file__).resolve()
REPO = THIS.parents[4]
E121_EVAL = REPO / "workspace/core4d/scripts/eval/eval_E121_terminal_carry_gate.py"
E124_VARIANTS = REPO / "workspace/core4d/scripts/E124/variants.tsv"


def load_e121_eval():
    spec = importlib.util.spec_from_file_location("eval_E121_reused_for_E124", E121_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {E121_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.E121_VARIANTS = E124_VARIANTS
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="full", choices=["smoke", "full"])
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    e121 = load_e121_eval()
    e115 = e121.load_e115_eval()
    rows = e115.read_variants()
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else REPO / f"workspace/core4d/results/E124/cem/{args.stage}"
    )
    if not results_dir.is_absolute():
        results_dir = REPO / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    missing = e115.fail_if_missing(args.stage, rows, results_dir, args.allow_missing)
    method_rows, decisions = e115.evaluate(args.stage, rows, results_dir)
    e121.augment_method_rows(method_rows, results_dir)
    e121.augment_decisions(decisions, method_rows)
    e115.write_outputs(args.stage, method_rows, decisions, missing, results_dir)
    e121.rewrite_summary(args.stage, decisions, method_rows, missing, results_dir)
    summary = results_dir / f"{args.stage}_eval_summary.md"
    if summary.is_file():
        text = summary.read_text(encoding="utf-8")
        text = text.replace(
            f"# E121 Terminal Carry Gate {args.stage} Summary",
            f"# E124 SBTO Carry Horizon {args.stage} Summary",
        )
        summary.write_text(text, encoding="utf-8")
    print(f"wrote {e121.rel(summary)}")


if __name__ == "__main__":
    main()
