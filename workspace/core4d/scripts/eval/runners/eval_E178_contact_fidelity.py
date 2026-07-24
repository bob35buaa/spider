#!/usr/bin/env python3
"""Pre-CEM ref-FK contact-target fidelity gate for E178 bucket proxies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))

from eval_E176_contact_fidelity import repo_path, run  # noqa: E402


DEFAULT_MANIFEST = (
    REPO
    / "workspace/core4d/results/E178/s6_downstream/manifests/"
    "semantic_bucket_full_manifest.tsv"
)
DEFAULT_OUT = (
    REPO
    / "workspace/core4d/results/E178/s2_proxy/contact_fidelity"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = run(
        repo_path(args.manifest),
        repo_path(args.out_dir),
        experiment_id="E178",
        expected_cases=27,
        expected_objects=3,
    )
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
