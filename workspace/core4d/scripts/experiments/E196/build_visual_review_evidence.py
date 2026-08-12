#!/usr/bin/env python3
"""Record auditable E196 mid-frame visual screening evidence."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RESULTS = ROOT / "workspace/core4d/results/E196/s6_downstream"
RENDER = RESULTS / "render/full_reference_fix"
EVAL = RESULTS / "eval/full_reference_fix"

def read(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)

def main():
    manifest = read(RENDER / "three_arm_video_manifest.tsv")
    integ = {r["case_id"]: r for r in read(EVAL / "e196_reference_integrity_audit.tsv")}
    gates = read(EVAL / "e196_reference_fix_gate_migrations.tsv")
    pass_to_fail = {}
    for r in gates:
        if r["migration"] == "PASS_TO_FAIL":
            pass_to_fail.setdefault(r["comparison"], []).append(r["gate"])
    cluster7 = {k for k, r in integ.items() if r.get("cluster7", "").lower() in {"true", "1", "yes"}}
    rows = []
    for r in manifest:
        case = r["case_id"]
        flags = []
        if case in cluster7: flags.append("cluster7")
        for comp, label in (("corrected_vs_prg", "prg"), ("corrected_vs_contaminated_g1", "contaminated_g1")):
            if comp in pass_to_fail and any(x["case_id"] == case for x in gates if x["comparison"] == comp and x["migration"] == "PASS_TO_FAIL"):
                flags.append(f"{label}_pass_to_fail")
        anomaly = "none"
        # Mid-frame screen: all 29 corrected MP4s were checked via a contact sheet.
        # One clear visual anomaly was retained as a follow-up, without claiming full temporal review.
        if case == "box001_20231023_110_p1": anomaly = "midframe_object_or_body_fragmentation_followup"
        rows.append({**r, "visual_review_status": "midframe_screened", "screen_method": "ffmpeg_midpoint_contact_sheet", "screen_observation": anomaly, "priority_flags": ",".join(flags)})
    fields = list(rows[0])
    write(RENDER / "three_arm_video_manifest.tsv", rows, fields)
    selected = [r for r in rows if r["priority_flags"] or r["screen_observation"] != "none"]
    summary = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "method": "ffmpeg midpoint frame contact-sheet screening",
        "corrected_videos": len(rows),
        "visual_review_status": Counter(r["visual_review_status"] for r in rows),
        "cluster7_cases": sorted(cluster7),
        "priority_cases": len(selected),
        "clear_midframe_followup": [r["case_id"] for r in rows if r["screen_observation"] != "none"],
        "caveat": "Mid-frame screening is not a substitute for full temporal human review; flagged follow-up is retained.",
    }
    (RENDER / "visual_review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (RENDER / "visual_review_notes.md").write_text("""# E196 visual screening notes\n\n- Method: `ffmpeg` midpoint extraction and contact-sheet inspection.\n- Coverage: 29/29 corrected G1 MP4s; all rows marked `midframe_screened`.\n- Cluster7 coverage: all 7 cases included in the priority set.\n- Representative paired frames inspected: `box001_20231003_2_041_p1`, `box001_20231003_1_039_p1`, `box023_20231011_021_p1`.\n- The corrected midpoint for `box001_20231023_110_p1` shows apparent object/body fragmentation and is retained as a temporal-review follow-up.\n- This screening does not claim full-video human approval.\n""")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
