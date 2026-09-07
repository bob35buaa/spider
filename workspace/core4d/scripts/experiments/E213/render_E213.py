#!/usr/bin/env python3
"""E213: render selected-arm aug rollouts to mp4 (sim left / reference-FK right).

Reads the merged E213 source manifest and renders each aug (case, variant) that
finished CEM.  CPU work (osmesa), does not touch GPUs; reads each run's own
config_act.yaml.

**Cross-video pose comparison is INVALID** -- `_auto_video_camera`
(spider/viewers/__init__.py) recomputes lookat/radius per frame from the sim-union
bbox, so two rollouts' mp4s use different, moving cameras (E209 F6 / E210 F3).
Compare sim-vs-ref WITHIN one video, or use viser_replay_E213.py for a
shared-camera aug-vs-orig A/B.

Usage:
    MUJOCO_GL=osmesa .venv/bin/python \
      workspace/core4d/scripts/experiments/E213/render_E213.py
    ... --cases <id> --variants trans0,rot0 --limit N --overwrite --max-frames 200 --watch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
for _d in ("E213", "E168"):
    sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments" / _d))

import e213_common as C  # noqa: E402


def load_rows() -> list[dict[str, str]]:
    """All cem_ok aug rows across the 4 shards (or merged main), mapped for render_row."""
    raw: list[dict[str, str]] = []
    shard_files = [C.manifest_path(s) for s in C.SHARDS if C.manifest_path(s).is_file()]
    src = shard_files or ([C.SOURCE_MANIFEST] if C.SOURCE_MANIFEST.is_file() else [])
    for f in src:
        raw += [r for r in C.read_tsv(f) if r.get("status") == "cem_ok"]
    rows: list[dict[str, str]] = []
    for r in raw:
        rows.append({
            "case_id": r["case_id"], "object_key": r["object_key"], "aug_variant": r["aug_variant"],
            "arm": r["arm"],
            "variant": f"{C.EXP_ID}_{r['case_id']}_aug_{r['aug_variant']}_{r['arm']}",
            "outdir_npz": r["outdir_npz"], "config_act": r["config_act"],
            "scene_act": r["selected_scene_act"], "trajectory": r["trajectory"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--variants", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--watch", action="store_true", help="poll for newly finished rollouts")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise SystemExit("MUJOCO_GL must be 'osmesa' on this host (egl->EGLError, glfw->no _mjr_context).")

    from render_a100_cem_videos import render_row  # noqa: E402

    out_dir = args.out_dir or (C.S6_DIR / "render" / "full")
    C.repo_path(out_dir).mkdir(parents=True, exist_ok=True)
    want_c = {c.strip() for c in args.cases.split(",") if c.strip()}
    want_v = {v.strip() for v in args.variants.split(",") if v.strip()}
    rendered, reused, failed = [], [], []

    def pass_once() -> int:
        rows = load_rows()
        rows = [r for r in rows if C.repo_path(r["outdir_npz"]).is_file()]
        if want_c:
            rows = [r for r in rows if r["case_id"] in want_c]
        if want_v:
            rows = [r for r in rows if r["aug_variant"] in want_v]
        if args.limit:
            rows = rows[: args.limit]
        for i, row in enumerate(rows, 1):
            mp4 = C.repo_path(out_dir) / f"{row['variant']}.mp4"
            if row["variant"] in rendered or row["variant"] in [x["variant"] for x in failed]:
                continue
            if mp4.is_file() and not args.overwrite:
                if row["variant"] not in reused:
                    reused.append(row["variant"])
                continue
            try:
                render_row(row, out_path=mp4, max_frames=args.max_frames)
                rendered.append(row["variant"])
                print(f"[{i:03d}/{len(rows)}] rendered {row['variant']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append({"variant": row["variant"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{i:03d}/{len(rows)}] FAILED {row['variant']}: {exc}", file=sys.stderr, flush=True)
        return len(rows)

    if args.watch:
        stable = 0
        while stable < 3:
            n = pass_once()
            done = len(rendered) + len(reused) + len(failed)
            stable = stable + 1 if done >= n and n >= 100 else 0
            if done < 100:
                time.sleep(60)
            else:
                break
    else:
        pass_once()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rendered": rendered, "reused": reused, "failed": failed,
        "camera_caveat": ("auto camera per-frame from sim-union bbox; cross-video pose "
                          "comparison INVALID (E209 F6/E210 F3). Use viser_replay_E213.py."),
    }
    (C.repo_path(out_dir) / "e213_render_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nrendered={len(rendered)} reused={len(reused)} failed={len(failed)} -> {C.rel(out_dir)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
