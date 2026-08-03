#!/usr/bin/env python3
"""E187 Full adapter for paired evaluation against frozen E178 metrics."""

from __future__ import annotations

import sys
from pathlib import Path

from eval_E176_lowgeom import main as run_lowgeom_eval

REPO = Path(__file__).resolve().parents[5]
E187 = REPO / "workspace/core4d/results/E187/s6_downstream"
MANIFEST = E187 / "manifests/e187_full_evaluation_manifest.tsv"
BASELINE = (
    REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/e178_case_metrics.tsv"
)
OUT_DIR = E187 / "eval/full"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args.extend(
        (
            "full",
            "--manifest",
            str(MANIFEST),
            "--baseline",
            str(BASELINE),
            "--baseline-prefix",
            "e178",
            "--out-dir",
            str(OUT_DIR),
            "--experiment-id",
            "E187",
            "--expected-rows",
            "22",
            "--output-prefix",
            "e187",
            "--enable-tracking-gates",
            "--root-pos-max-cm",
            "20",
            "--root-ori-max-deg",
            "20",
            "--hand-pos-max-cm",
            "20",
            "--hand-ori-max-deg",
            "20",
            "--object-pos-max-cm",
            "20",
            "--object-ori-max-deg",
            "10",
            "--require-all",
        )
    )
    return run_lowgeom_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
