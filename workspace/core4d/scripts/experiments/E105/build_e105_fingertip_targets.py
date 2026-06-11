#!/usr/bin/env python3
"""Build/revalidate E105 E100/E101-style fingertip-aware external targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent / "E100"))

import build_fingertip_aware_target as e100_target  # type: ignore  # noqa: E402
from e105_common import REPO, RESULTS_ROOT, SOURCE_CASES, TASK_ROOT, rel, rows  # noqa: E402


OUT_ROOT = RESULTS_ROOT / "fingertip_targets"
VOTE_DIR = REPO / "workspace/core4d/results/E099/fingertip_vote_per_case"
SUMMARY_TSV = RESULTS_ROOT / "fingertip_target_summary.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_one(case: str, force: bool) -> dict[str, Any]:
    scene = TASK_ROOT / case / "scene.xml"
    traj = TASK_ROOT / case / "0/trajectory_kinematic.npz"
    vote_path = VOTE_DIR / f"{case}.json"
    if not scene.is_file() or not traj.is_file() or not vote_path.is_file():
        raise FileNotFoundError(f"Missing fingertip inputs for {case}: {scene}, {traj}, {vote_path}")
    vote = json.loads(vote_path.read_text(encoding="utf-8"))
    if vote.get("status") != "ok":
        raise RuntimeError(f"{case} vote status is {vote.get('status')}")

    case_out = OUT_ROOT / case
    if force and case_out.exists():
        import shutil

        shutil.rmtree(case_out)
    case_out.mkdir(parents=True, exist_ok=True)
    res = e100_target.build_target(case, scene, traj, vote)
    npz_path = case_out / "spider_contact_target_object_local.npz"
    np.savez(
        npz_path,
        spider_contact_target_object_local=res["spider_contact_target_object_local"],
        palm_local_record=res["palm_local_record"],
        active=res["active"],
    )
    summary = dict(res["summary"])
    summary.update(
        {
            "scene_xml": rel(scene),
            "trajectory_npz": rel(traj),
            "vote_json": rel(vote_path),
            "target_npz": rel(npz_path),
            "scene_sha256": sha256(scene),
            "trajectory_sha256": sha256(traj),
            "vote_sha256": sha256(vote_path),
            "target_sha256": sha256(npz_path),
            "e105_authoritative_target": True,
        }
    )
    (case_out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = [SOURCE_CASES["039"]["source_task"], SOURCE_CASES["135"]["source_task"]]
    summaries = [build_one(case, args.force) for case in cases]

    expected = [REPO / row["target_npz"] for row in rows() if row["route"] == "fingertip_clean"]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing E105 fingertip targets: " + ", ".join(map(str, missing)))

    fields = sorted({k for row in summaries for k in row})
    with SUMMARY_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[E105-fingertip] wrote {rel(SUMMARY_TSV)} rows={len(summaries)}")


if __name__ == "__main__":
    main()
