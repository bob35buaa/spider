#!/usr/bin/env python3
"""E202 visual QC (C7): render a representative sample of bucket aug rollouts.

Reuses E199 render_qc.render_row (generic: only uses repo-relative paths) on the
E202 priority manifest. Emits mp4 + keyframe strips into the E202 render dir so
penetration / floating / jitter / fall can be eyeballed vs the E178 orig.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E199"))
import e202_common as C  # noqa: E402
import render_qc as RQC  # noqa: E402  (E199 render_row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default="bucket003,bucket004,bucket007")
    ap.add_argument("--max-per-object", type=int, default=3)
    ap.add_argument("--only-cases", default="", help="comma case_ids to force-include")
    ap.add_argument("--n-keyframes", type=int, default=5)
    ap.add_argument("--out", type=Path, default=C.RESULTS / "s6_downstream/render/qc")
    args = ap.parse_args()

    wanted = {o.strip() for o in args.objects.split(",") if o.strip()}
    force = {c.strip() for c in args.only_cases.split(",") if c.strip()}
    rows = [r for r in C.read_tsv(C.FULL_MANIFEST)
            if r["object_key"] in wanted and C.repo_path(r["outdir_npz"]).is_file()]
    rows.sort(key=lambda r: (r["object_key"], r.get("case_id", ""), r["aug_variant"]))

    # forced problem cases: render all their trans variants
    selected: list[dict] = [r for r in rows if r.get("case_id", "") in force]
    # healthy sample: first `max_per_object` distinct cases per object, trans0 only
    obj_cases: dict[str, list[str]] = {}
    for r in rows:
        cid = r.get("case_id", "")
        if cid in force:
            continue
        picked = obj_cases.setdefault(r["object_key"], [])
        if cid not in picked and len(picked) < args.max_per_object:
            picked.append(cid)
        if cid in picked and r["aug_variant"] == "trans0":
            selected.append(r)

    done = []
    for r in selected:
        try:
            # per-case subdir: render_row's tag is object_key+variant only, which
            # collides across cases of the same object; isolate by case_id.
            case_out = args.out / r.get("case_id", "case")
            info = RQC.render_row(r, case_out, args.n_keyframes)
            info["case_id"] = r.get("case_id", "")
            done.append(info)
            print(f"[render] {r.get('case_id')} {info['tag']} frames={info['frames']} -> {info['mp4']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {r.get('case_id')} {r['aug_variant']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    C.write_json(args.out / "render_index.json", done)
    print(f"[done] rendered {len(done)} rollouts -> {C.rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
