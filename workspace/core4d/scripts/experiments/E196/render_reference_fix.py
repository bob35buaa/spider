#!/usr/bin/env python3
"""Render E196 corrected G1 self replays and build three-arm visual manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e196_reference_fix_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))
from render_a100_cem_videos import render_row  # noqa: E402

sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E194"))
import e194_g1_expansion_common as E194  # noqa: E402

from spider.simulators.scene_act_reference import resolve_scene_act_reference  # noqa: E402
import mujoco  # noqa: E402


OUT = C.RESULTS / "s6_downstream/render/full_reference_fix"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=C.FULL_MANIFEST)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    rows = C.read_tsv(C.repo_path(args.manifest))
    sources = {row["case_id"]: row for row in E194.source_rows()}
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = []
    failures = []
    for row in rows:
        output = C.repo_path(row["video"])
        try:
            for key in ("outdir_npz", "config_act", "scene_act", "trajectory"):
                C.require_file(row[key], key)
            model = mujoco.MjModel.from_xml_path(str(C.repo_path(row["scene_act"])))
            resolved = resolve_scene_act_reference(C.repo_path(row["scene_act"]), model, emit_log=False)
            if resolved.convention != row["resolved_euler_convention"] or resolved.meta_sha256 != row["scene_act_meta_sha256"]:
                raise ValueError("render reference contract differs from manifest")
            status = "existing"
            if args.overwrite or not output.is_file() or output.stat().st_size == 0:
                render_row(row, out_path=output, max_frames=args.max_frames)
                status = "rendered"
            source = sources[row["case_id"]]
            prg_video = C.require_file(source.get("video", ""), "paired PRG video")
            contaminated_video = C.require_file(row["e194_video"], "paired contaminated G1 video")
            evidence.append(
                {
                    "case_id": row["case_id"],
                    "object_key": row["object_key"],
                    "worker": row["worker"],
                    "wave": row["wave"],
                    "prg_video": C.rel(prg_video),
                    "contaminated_g1_video": C.rel(contaminated_video),
                    "corrected_g1_video": C.rel(output),
                    "corrected_status": status,
                    "visual_review_status": "pending",
                }
            )
            print(f"[{status}] {row['case_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[failed] {row['case_id']}: {exc}", file=sys.stderr)
    C.write_tsv(OUT / "three_arm_video_manifest.tsv", evidence)
    C.write_tsv(OUT / "render_failures.tsv", failures)
    summary = {
        "created_at": C.now(),
        "selected": len(rows),
        "complete": len(evidence),
        "failed": len(failures),
        "status": "pass" if len(evidence) == len(rows) and not failures else "incomplete",
    }
    C.write_json(OUT / "render_summary.json", summary)
    print(summary)
    return 1 if args.require_all and summary["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
