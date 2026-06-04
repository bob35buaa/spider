#!/usr/bin/env python3
"""Add labeled object_contact proxy masks to E126 Holosoma fragment exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
INPUT_DIR = REPO / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports"
OUT_DIR = REPO / "workspace/core4d/results/E131/holosoma_object_contact_proxy"
EXPORT_DIR = OUT_DIR / "exports"
BOX021_HALF_EXTENTS = np.array([0.195495, 0.15155, 0.25004], dtype=np.float64)
HAND_NAMES = ("left_rubber_hand_link", "right_rubber_hand_link")
CONTACT_THRESHOLD_M = 0.05
MASK_SOURCE = "actor_rubber_hand_box021_surface_proxy_5cm"

INPUT_EXPORTS = [
    INPUT_DIR / "E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz",
    INPUT_DIR / "E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz",
]

MANIFEST_FIELDS = [
    "source_export_npz",
    "proxy_export_npz",
    "frames",
    "fps",
    "object_contact_shape",
    "object_contact_dtype",
    "mask_source",
    "threshold_m",
    "hand_body_names",
    "left_active_frac",
    "right_active_frac",
    "both_active_frac",
    "either_active_frac",
    "left_longest_run_frames",
    "right_longest_run_frames",
    "both_longest_run_frames",
    "left_min_distance_m",
    "right_min_distance_m",
    "left_mean_distance_m",
    "right_mean_distance_m",
    "has_existing_object_contact",
    "structural_ref_mask_ready",
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


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = q[..., :1]
    xyz = q[..., 1:]
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (w * uv + uuv)


def box_signed_surface_distance(points: np.ndarray, obj_pos: np.ndarray, obj_quat: np.ndarray) -> np.ndarray:
    local = quat_apply_wxyz(quat_conj_wxyz(obj_quat), points - obj_pos)
    delta = np.abs(local) - BOX021_HALF_EXTENTS
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(np.max(delta, axis=-1), 0.0)
    return outside + inside


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def body_names(data: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in np.asarray(data["body_names"]).tolist():
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def inspect_required(data: dict[str, Any], path: Path) -> None:
    required = [
        "fps",
        "joint_pos",
        "body_names",
        "body_pos_w",
        "object_pos_w",
        "object_quat_w",
        "partner_hand_pos_w",
        "partner_hand_quat_w",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    frames = int(np.asarray(data["joint_pos"]).shape[0])
    for key in ("body_pos_w", "object_pos_w", "object_quat_w", "partner_hand_pos_w"):
        if int(np.asarray(data[key]).shape[0]) != frames:
            raise ValueError(f"{path}: {key} frame count does not match joint_pos")


def export_proxy(path: Path) -> dict[str, Any]:
    data = load_npz(path)
    inspect_required(data, path)
    names = body_names(data)
    index = {name: i for i, name in enumerate(names)}
    missing_hands = [name for name in HAND_NAMES if name not in index]
    if missing_hands:
        raise ValueError(f"{path}: missing actor hand body names {missing_hands}")

    frames = int(data["joint_pos"].shape[0])
    obj_pos = np.asarray(data["object_pos_w"], dtype=np.float64)
    obj_quat = np.asarray(data["object_quat_w"], dtype=np.float64)
    distances = []
    for name in HAND_NAMES:
        points = np.asarray(data["body_pos_w"][:, index[name], :], dtype=np.float64)
        distances.append(box_signed_surface_distance(points, obj_pos, obj_quat))
    dist_arr = np.stack(distances, axis=1)
    object_contact = dist_arr <= CONTACT_THRESHOLD_M

    out_name = path.name.replace("_mj_w_obj_w_partner.npz", "_object_contact_proxy5cm.npz")
    out_path = EXPORT_DIR / out_name
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_data = dict(data)
    out_data["object_contact"] = object_contact.astype(np.bool_)
    out_data["object_contact_source"] = np.asarray(MASK_SOURCE)
    out_data["object_contact_threshold_m"] = np.asarray(CONTACT_THRESHOLD_M, dtype=np.float32)
    out_data["object_contact_hand_body_names"] = np.asarray(HAND_NAMES)
    np.savez(out_path, **out_data)

    left = object_contact[:, 0]
    right = object_contact[:, 1]
    both = left & right
    either = left | right
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    return {
        "source_export_npz": rel(path),
        "proxy_export_npz": rel(out_path),
        "frames": frames,
        "fps": f"{fps:.3f}",
        "object_contact_shape": f"{frames}x2",
        "object_contact_dtype": "bool",
        "mask_source": MASK_SOURCE,
        "threshold_m": f"{CONTACT_THRESHOLD_M:.3f}",
        "hand_body_names": ",".join(HAND_NAMES),
        "left_active_frac": f"{float(np.mean(left)):.6f}",
        "right_active_frac": f"{float(np.mean(right)):.6f}",
        "both_active_frac": f"{float(np.mean(both)):.6f}",
        "either_active_frac": f"{float(np.mean(either)):.6f}",
        "left_longest_run_frames": longest_true_run(left),
        "right_longest_run_frames": longest_true_run(right),
        "both_longest_run_frames": longest_true_run(both),
        "left_min_distance_m": f"{float(np.min(dist_arr[:, 0])):.6f}",
        "right_min_distance_m": f"{float(np.min(dist_arr[:, 1])):.6f}",
        "left_mean_distance_m": f"{float(np.mean(dist_arr[:, 0])):.6f}",
        "right_mean_distance_m": f"{float(np.mean(dist_arr[:, 1])):.6f}",
        "has_existing_object_contact": str("object_contact" in data).lower(),
        "structural_ref_mask_ready": "true",
        "semantic_ref_mask_ready": "false",
        "rl_ready": "false",
        "notes": "proxy contact mask for bounded loader/reward debugging only; not raw-contact ground truth",
    }


def write_summary(rows: list[dict[str, Any]]) -> None:
    structural = sum(1 for row in rows if row["structural_ref_mask_ready"] == "true")
    semantic = sum(1 for row in rows if row["semantic_ref_mask_ready"] == "true")
    summary = {
        "experiment": "E131",
        "status": "pass",
        "rows": len(rows),
        "proxy_exports": [row["proxy_export_npz"] for row in rows],
        "mask_source": MASK_SOURCE,
        "threshold_m": CONTACT_THRESHOLD_M,
        "structural_ref_mask_ready_rows": structural,
        "semantic_ref_mask_ready_rows": semantic,
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "checkpoint_created": False,
        "remote_jobs_launched": False,
        "main_release_evidence": False,
        "fragment_label": "FRAGMENT_HOLDOUT_ONLY",
        "notes": [
            "object_contact is a method-side surface-distance proxy, not raw-contact ground truth.",
            "E126 source exports are not modified.",
            "Proxy exports are valid only for bounded loader/ref-mask wiring inspection.",
        ],
    }
    (OUT_DIR / "e131_object_contact_proxy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# E131 Holosoma Object-Contact Proxy Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| rows | {summary['rows']} |",
        f"| structural ref-mask ready rows | {structural} |",
        f"| semantic ref-mask ready rows | {semantic} |",
        "| RL-ready rows | 0 |",
        "| training launched | false |",
        "| CEM launched | false |",
        "",
        "## Proxy Rows",
        "",
        "| source | left active | right active | both active | left run | right run | proxy export |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{Path(row['source_export_npz']).name}`",
                    row["left_active_frac"],
                    row["right_active_frac"],
                    row["both_active_frac"],
                    str(row["left_longest_run_frames"]),
                    str(row["right_longest_run_frames"]),
                    f"`{row['proxy_export_npz']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These exports close the Holosoma loader shape contract for `object_contact`, but the mask is derived from actor rubber-hand distance to the Box021 OBB. It is not a real raw/trimmed contact label and must not be used as release or PPO evidence.",
        ]
    )
    (OUT_DIR / "e131_object_contact_proxy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [export_proxy(path) for path in INPUT_EXPORTS]
    write_tsv(OUT_DIR / "e131_object_contact_proxy_manifest.tsv", rows, MANIFEST_FIELDS)
    write_summary(rows)
    print(json.dumps(json.loads((OUT_DIR / "e131_object_contact_proxy_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
