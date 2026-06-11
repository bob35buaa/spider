#!/usr/bin/env python3
"""Probe Holosoma MotionLoader object_contact behavior for E126/E131 exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.managers.command.terms.wbt import MotionLoader


REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "workspace/core4d/results/E132/holosoma_motionloader_object_contact_probe"

ROWS = [
    {
        "case_id": "box021_035_p1",
        "partner_case_id": "box021_035_p2",
        "source": "E126",
        "motion_path": REPO
        / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/"
        / "E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz",
        "expected_has_object_contact": False,
    },
    {
        "case_id": "box021_035_p2",
        "partner_case_id": "box021_035_p1",
        "source": "E126",
        "motion_path": REPO
        / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/"
        / "E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz",
        "expected_has_object_contact": False,
    },
    {
        "case_id": "box021_035_p1",
        "partner_case_id": "box021_035_p2",
        "source": "E131",
        "motion_path": REPO
        / "workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/"
        / "E126_box021_035_p1_with_partner_box021_035_p2_object_contact_proxy5cm.npz",
        "expected_has_object_contact": True,
    },
    {
        "case_id": "box021_035_p2",
        "partner_case_id": "box021_035_p1",
        "source": "E131",
        "motion_path": REPO
        / "workspace/core4d/results/E131/holosoma_object_contact_proxy/exports/"
        / "E126_box021_035_p2_with_partner_box021_035_p1_object_contact_proxy5cm.npz",
        "expected_has_object_contact": True,
    },
]

FIELDS = [
    "source",
    "case_id",
    "partner_case_id",
    "motion_path",
    "frames",
    "fps",
    "has_object",
    "has_partner",
    "has_object_contact",
    "expected_has_object_contact",
    "object_contact_shape",
    "object_contact_dtype",
    "left_active_frac",
    "right_active_frac",
    "both_active_frac",
    "either_active_frac",
    "left_longest_run_frames",
    "right_longest_run_frames",
    "both_longest_run_frames",
    "runtime_contract_status",
    "semantic_ref_mask_ready",
    "rl_ready",
    "notes",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def names(arr: np.ndarray) -> list[str]:
    out = []
    for item in arr.tolist():
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in values.astype(bool).tolist():
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def probe(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["motion_path"])
    with np.load(path, allow_pickle=True) as data:
        body_names = names(data["body_names"])
        joint_names = names(data["joint_names"])
        frames = int(data["joint_pos"].shape[0])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    motion = MotionLoader(str(path), body_names, joint_names, device="cpu")
    contact = motion.object_contact.cpu().numpy().astype(bool)
    left = contact[:, 0]
    right = contact[:, 1]
    both = left & right
    either = left | right
    expected = bool(row["expected_has_object_contact"])
    status = "pass" if bool(motion.has_object_contact) == expected else "fail"
    if expected and contact.shape != (frames, 2):
        status = "fail"
    if expected and not bool(either.any()):
        status = "fail"
    return {
        "source": row["source"],
        "case_id": row["case_id"],
        "partner_case_id": row["partner_case_id"],
        "motion_path": rel(path),
        "frames": frames,
        "fps": f"{fps:.3f}",
        "has_object": str(bool(motion.has_object)).lower(),
        "has_partner": str(bool(motion.has_partner)).lower(),
        "has_object_contact": str(bool(motion.has_object_contact)).lower(),
        "expected_has_object_contact": str(expected).lower(),
        "object_contact_shape": "x".join(str(x) for x in contact.shape),
        "object_contact_dtype": str(contact.dtype),
        "left_active_frac": f"{float(np.mean(left)):.6f}",
        "right_active_frac": f"{float(np.mean(right)):.6f}",
        "both_active_frac": f"{float(np.mean(both)):.6f}",
        "either_active_frac": f"{float(np.mean(either)):.6f}",
        "left_longest_run_frames": longest_true_run(left),
        "right_longest_run_frames": longest_true_run(right),
        "both_longest_run_frames": longest_true_run(both),
        "runtime_contract_status": status,
        "semantic_ref_mask_ready": "false",
        "rl_ready": "false",
        "notes": "MotionLoader CPU contract probe only; E131 masks are geometry proxies" if row["source"] == "E131" else "E126 missing-mask baseline",
    }


def write_summary(rows: list[dict[str, Any]]) -> None:
    e126 = [row for row in rows if row["source"] == "E126"]
    e131 = [row for row in rows if row["source"] == "E131"]
    pass_rows = [row for row in rows if row["runtime_contract_status"] == "pass"]
    structural_ready = [
        row
        for row in e131
        if row["runtime_contract_status"] == "pass"
        and row["has_object_contact"] == "true"
        and row["object_contact_shape"] == f"{row['frames']}x2"
    ]
    summary = {
        "experiment": "E132",
        "status": "pass" if len(pass_rows) == len(rows) and len(structural_ready) == len(e131) else "fail",
        "rows": len(rows),
        "runtime_contract_pass_rows": len(pass_rows),
        "e126_missing_mask_rows": sum(1 for row in e126 if row["has_object_contact"] == "false"),
        "e131_object_contact_rows": sum(1 for row in e131 if row["has_object_contact"] == "true"),
        "structural_ref_mask_runtime_ready_rows": len(structural_ready),
        "semantic_ref_mask_ready_rows": 0,
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "checkpoint_created": False,
        "remote_jobs_launched": False,
        "main_release_evidence": False,
        "notes": [
            "E132 uses Holosoma MotionLoader under hssim/source_isaacsim_setup dependencies on CPU.",
            "E131 geometry proxies load as object_contact, but remain non-semantic proxy masks.",
            "No IsaacSim stepping, PPO, CEM, or checkpoint creation was launched.",
        ],
    }
    (OUT_DIR / "e132_motionloader_object_contact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# E132 Holosoma MotionLoader Object-Contact Probe Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| rows | {summary['rows']} |",
        f"| runtime contract pass rows | {summary['runtime_contract_pass_rows']} |",
        f"| E126 missing-mask rows | {summary['e126_missing_mask_rows']} |",
        f"| E131 object-contact rows | {summary['e131_object_contact_rows']} |",
        f"| structural ref-mask runtime-ready rows | {summary['structural_ref_mask_runtime_ready_rows']} |",
        "| semantic ref-mask ready rows | 0 |",
        "| RL-ready rows | 0 |",
        "| training launched | false |",
        "| CEM launched | false |",
        "",
        "## Loader Rows",
        "",
        "| source | case | has object contact | shape | left active | right active | both active | status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['source']}`",
                    f"`{row['case_id']}`",
                    f"`{row['has_object_contact']}`",
                    f"`{row['object_contact_shape']}`",
                    row["left_active_frac"],
                    row["right_active_frac"],
                    row["both_active_frac"],
                    f"`{row['runtime_contract_status']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "E132 proves the structural loader distinction between E126 missing-mask exports and E131 proxy-contact exports. It does not prove raw-contact semantics, simulator force behavior, policy quality, or main-case release readiness.",
        ]
    )
    (OUT_DIR / "e132_motionloader_object_contact_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [probe(row) for row in ROWS]
    write_tsv(OUT_DIR / "e132_motionloader_object_contact_manifest.tsv", rows, FIELDS)
    write_summary(rows)


if __name__ == "__main__":
    main()
