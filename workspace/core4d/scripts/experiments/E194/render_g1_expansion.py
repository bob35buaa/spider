#!/usr/bin/env python3
"""Render all completed E194 G1 expansion self replays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e194_g1_expansion_common as C  # noqa: E402
sys.path.insert(0, str(C.REPO / "workspace/core4d/scripts/experiments/E168"))
from render_a100_cem_videos import render_row  # noqa: E402

COMPLETE = {"run_complete_pending_eval", "run_complete", "eval_complete"}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, default=C.FULL_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=C.RESULTS / "s6_downstream/render/full_g1_expansion")
    parser.add_argument("--max-frames", type=int, default=0); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variants", nargs="*", default=[]); args = parser.parse_args()
    manifest = C.repo_path(args.manifest)
    rows = C.read_tsv(manifest); out = C.repo_path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    expected = C.N_CASES if not args.variants and manifest.resolve() == C.FULL_MANIFEST.resolve() else None
    rendered=[]; skipped=[]; failed=[]; evidence=[]
    sources = {row["case_id"]: row for row in C.source_rows()}
    for row in rows:
        if args.variants and row["variant"] not in args.variants: continue
        output = out / f"{row['variant']}.mp4"
        if output.is_file() and output.stat().st_size > 0 and not args.overwrite:
            rendered.append(row["variant"])
        elif row.get("status") not in COMPLETE and not C.repo_path(row["outdir_npz"]).is_file():
            skipped.append(row["variant"]); continue
        else:
            try:
                render_row(row, out_path=output, max_frames=args.max_frames); rendered.append(row["variant"])
                print(f"[rendered] {row['variant']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append({"variant":row["variant"],"error":f"{type(exc).__name__}: {exc}"}); print(f"[failed] {row['variant']}: {exc}",file=sys.stderr)
                continue
        source=sources[row["case_id"]]
        evidence.append({"case_id":row["case_id"],"object_key":row["object_key"],"worker":row["worker"],
                         "a0_video":source.get("video",""),"g1_video":C.rel(output),"status":"ready_for_visual_review"})
    C.write_tsv(out/"paired_video_manifest.tsv",evidence)
    C.write_json(out/"render_summary.json",{"created_at":C.now(),"rendered":len(rendered),"skipped":skipped,"failed":failed,"paired_rows":len(evidence)})
    contract_ok = not failed
    if expected is not None:
        contract_ok = contract_ok and len(rows) == expected and len(rendered) == expected and not skipped and len(evidence) == expected
    print(f"rendered={len(rendered)} skipped={len(skipped)} failed={len(failed)} paired={len(evidence)} expected={expected}")
    return 0 if contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
