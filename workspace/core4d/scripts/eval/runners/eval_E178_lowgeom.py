#!/usr/bin/env python3
"""E178 adapter for the shared E176/E178 low-geom CEM evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

from eval_E176_lowgeom import main as run_lowgeom_eval


REPO = Path(__file__).resolve().parents[5]
RESULT_ROOT = REPO / "workspace/core4d/results/E178/s6_downstream"
MANIFESTS = {
    "canary": RESULT_ROOT / "manifests/semantic_bucket_canary_manifest.tsv",
    "full": RESULT_ROOT / "manifests/semantic_bucket_full_manifest.tsv",
}
EXPECTED_ROWS = {"canary": 3, "full": 27}


def _has_option(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in EXPECTED_ROWS:
        raise SystemExit("usage: eval_E178_lowgeom.py {canary,full} [options]")
    mode = args[0]
    if not _has_option(args, "--manifest"):
        args.extend(("--manifest", str(MANIFESTS[mode])))
    if not _has_option(args, "--out-dir"):
        args.extend(("--out-dir", str(RESULT_ROOT / "eval" / mode)))
    if mode == "full" and not _has_option(args, "--video-dir"):
        args.extend(("--video-dir", str(RESULT_ROOT / "render/full")))
    args.extend(
        (
            "--experiment-id",
            "E178",
            "--expected-rows",
            str(EXPECTED_ROWS[mode]),
            "--output-prefix",
            "e178",
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
        )
    )
    return run_lowgeom_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
