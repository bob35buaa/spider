#!/usr/bin/env python3
"""E174 S2: non-box source-template proxy review queue + explicit approve flow.

Non-box templates (bucket / desk) are `manual_review_required` by default and
CANNOT auto-clean from a MuJoCo load/render pass (dcv3 doc 04). This script:

  build mode (default):
    - reads the S2 template_backlog.tsv,
    - for every in-scope non-box template, renders an OBJECT-ONLY mesh/collision
      overlay sheet (stages/s2_templates/render_template_mesh_collision_review_package.py
      --object-only) so a reviewer can see whether the collision proxy is larger
      than the mesh / bridges desk-leg gaps / wraps the wrong topology,
    - computes doc-04 hard-fail flags (mujoco load, nq/nv/nu present, 2 hand
      contact sites, no 29.632 inertial pollution),
    - writes/updates nonbox_template_review.tsv with review_decision left blank
      (pending) for anything not already approved.

  approve mode (--approve TASK ...):
    - marks the named source_scene_task rows review_decision=approve_clean with a
      reviewer + notes, ONLY if they have no hard-fail. This is the explicit,
      git-tracked, reproducible reviewer decision required before S3. run_stage2b
      then consumes this TSV via --template-review-tsv and lifts the template to
      template_status=clean_reviewed.

The proxy adapter / collision_policy per category is frozen in e174_common
(bucket -> bucket_wall_proxy_aabb via nonbox_proxy_aabb_review; desk ->
desk_surface_voxel_multibox_proxy_draft via nonbox_surface_voxel_review).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e174_common as C

REVIEW_TSV_NAME = "nonbox_template_review.tsv"
OVERLAY_SUBDIR = "object_only_overlay"

FIELDS = [
    "source_scene_task", "object_key", "object_category", "person",
    "proxy_scene_xml", "template_adapter", "expected_collision_policy",
    "approved_collision_policy", "backlog_collision_policy",
    "mujoco_load_ok", "nq", "nv", "nu", "hand_contact_site_count",
    "robot_polluted_mass_29_632", "proxy_template", "hard_fail",
    "object_only_overlay", "review_decision", "reviewer", "review_notes",
    "reviewed_at",
]


def hard_fail_reasons(row: dict[str, str]) -> str:
    reasons: list[str] = []
    if row.get("mujoco_load_ok") != "True":
        reasons.append("mujoco_load_error")
    if not row.get("nq") or not row.get("nv") or not row.get("nu"):
        reasons.append("missing_nq_nv_nu")
    if row.get("hand_contact_site_count") not in {"2"}:
        reasons.append("hand_contact_sites_not_2")
    if row.get("robot_polluted_mass_29_632") == "True":
        reasons.append("robot_inertial_29_632_pollution")
    return ";".join(reasons)


def load_existing(out_review: Path) -> dict[str, dict[str, str]]:
    if not out_review.is_file():
        return {}
    return {r["source_scene_task"]: r for r in C.read_tsv(out_review) if r.get("source_scene_task")}


def render_overlays(backlog_tsv: Path, out_dir: Path) -> None:
    """Render object-only mesh/collision overlays for manual_review_required rows."""
    script = C.DCV3 / "stages/s2_templates/render_template_mesh_collision_review_package.py"
    cmd = [
        str(C.SPIDER_PYTHON_BIN), str(script),
        "--input-tsv", str(backlog_tsv),
        "--out-dir", str(out_dir / OVERLAY_SUBDIR),
        "--render-statuses", "manual_review_required",
        "--object-only",
    ]
    print("[render]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(C.REPO))
    if result.returncode != 0:
        print(f"[warn] object-only overlay render returned {result.returncode} (review manually)")


def build(backlog_tsv: Path, out_dir: Path, do_render: bool) -> dict:
    out_review = out_dir / REVIEW_TSV_NAME
    backlog = [
        r for r in C.read_tsv(backlog_tsv)
        if r.get("object_key") in C.OBJECT_KEYS and r.get("object_category") != "box"
    ]
    if do_render:
        render_overlays(backlog_tsv, out_dir)
    existing = load_existing(out_review)

    rows: list[dict[str, str]] = []
    for src in sorted(backlog, key=lambda r: r.get("source_scene_task", "")):
        task = src.get("source_scene_task", "")
        cat = src.get("object_category", "")
        prev = existing.get(task, {})
        hard = hard_fail_reasons(src)
        overlay = out_dir / OVERLAY_SUBDIR / f"{task}_object_only_mesh_collision.mp4"
        expected_policy = C.OBJECT_COLLISION_POLICY.get(cat, "")
        # preserve a prior approval (idempotent), but never keep an approval that
        # now has a hard-fail.
        decision = prev.get("review_decision", "")
        if decision == "approve_clean" and hard:
            decision = ""
        rows.append({
            "source_scene_task": task,
            "object_key": src.get("object_key", ""),
            "object_category": cat,
            "person": src.get("person", ""),
            "proxy_scene_xml": src.get("scene_xml", ""),
            "template_adapter": src.get("template_adapter", C.NONBOX_TEMPLATE_ADAPTER.get(cat, "")),
            "expected_collision_policy": expected_policy,
            "approved_collision_policy": prev.get("approved_collision_policy", expected_policy),
            "backlog_collision_policy": src.get("collision_policy", ""),
            "mujoco_load_ok": src.get("mujoco_load_ok", ""),
            "nq": src.get("nq", ""), "nv": src.get("nv", ""), "nu": src.get("nu", ""),
            "hand_contact_site_count": src.get("hand_contact_site_count", ""),
            "robot_polluted_mass_29_632": src.get("robot_polluted_mass_29_632", ""),
            "proxy_template": src.get("proxy_template", ""),
            "hard_fail": hard,
            "object_only_overlay": C.rel(overlay) if overlay.is_file() else "",
            "review_decision": decision,
            "reviewer": prev.get("reviewer", ""),
            "review_notes": prev.get("review_notes", ""),
            "reviewed_at": prev.get("reviewed_at", ""),
        })
    C.write_tsv(out_review, rows, FIELDS)
    summary = {
        "created_at": C.now(),
        "review_tsv": C.rel(out_review),
        "total": len(rows),
        "hard_fail": sum(bool(r["hard_fail"]) for r in rows),
        "approved": sum(r["review_decision"] == "approve_clean" for r in rows),
        "pending": sum(not r["review_decision"] for r in rows),
        "tasks": [r["source_scene_task"] for r in rows],
    }
    C.write_json(out_dir / "nonbox_template_review_summary.json", summary)
    return summary


def approve(out_dir: Path, tasks: list[str], reviewer: str, notes: str) -> dict:
    out_review = out_dir / REVIEW_TSV_NAME
    rows = C.read_tsv(out_review)
    index = {r["source_scene_task"]: r for r in rows}
    applied, refused = [], []
    for task in tasks:
        row = index.get(task)
        if row is None:
            refused.append(f"{task}:not_in_queue")
            continue
        if row.get("hard_fail"):
            refused.append(f"{task}:hard_fail={row['hard_fail']}")
            continue
        row["review_decision"] = "approve_clean"
        row["reviewer"] = reviewer
        row["review_notes"] = notes
        row["reviewed_at"] = C.now()
        if not row.get("approved_collision_policy"):
            row["approved_collision_policy"] = C.OBJECT_COLLISION_POLICY.get(row.get("object_category", ""), "")
        applied.append(task)
    C.write_tsv(out_review, rows, FIELDS)
    return {"applied": applied, "refused": refused,
            "approved_total": sum(r["review_decision"] == "approve_clean" for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-backlog-tsv", type=Path,
                        default=C.RESULTS / "s2_templates/template_backlog.tsv")
    parser.add_argument("--out-dir", type=Path, default=C.RESULTS / "s2_templates/review")
    parser.add_argument("--render", action="store_true", help="render object-only overlays")
    parser.add_argument("--approve", action="append", default=[], metavar="TASK",
                        help="mark source_scene_task approve_clean (repeatable)")
    parser.add_argument("--reviewer", default="codex")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.approve:
        result = approve(args.out_dir, args.approve, args.reviewer, args.notes)
    else:
        result = build(args.template_backlog_tsv, args.out_dir, args.render)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
