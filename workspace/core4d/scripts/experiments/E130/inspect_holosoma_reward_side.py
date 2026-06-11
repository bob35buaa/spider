#!/usr/bin/env python3
"""Inspect Holosoma Box021 reward-side evidence for E126 paired fragments."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[4]
HOLOSOMA = Path("/home/ubuntu/Workspace/holosoma")
REWARD_SRC = HOLOSOMA / "src/holosoma/holosoma/config_values/wbt/g1/reward.py"
EXPERIMENT_SRC = HOLOSOMA / "src/holosoma/holosoma/config_values/wbt/g1/experiment.py"
OUT_DIR = REPO / "workspace/core4d/results/E130/holosoma_reward_side_inspection"

BOX021_HALF_EXTENTS = np.array([0.195495, 0.15155, 0.25004], dtype=np.float64)
HAND_FALLBACK_NAMES = ("left_rubber_hand_link", "right_rubber_hand_link")
HOLOSOMA_HANDBOX_NAMES = ("left_handbox_link", "right_handbox_link")
MOTIONS = [
    REPO
    / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/"
    / "E126_box021_035_p1_with_partner_box021_035_p2_mj_w_obj_w_partner.npz",
    REPO
    / "workspace/core4d/results/E126/holosoma_fragment_adapter_preflight/exports/"
    / "E126_box021_035_p2_with_partner_box021_035_p1_mj_w_obj_w_partner.npz",
]

REWARD_BLOCKS = [
    "g1_29dof_wbt_reward_w_object_r119_box021_handbox_omnirt_v4_3",
    "g1_29dof_wbt_reward_w_object_r138_box021_r095_loadpath_v4_3",
]
EXPERIMENT_CONFIGS = [
    "g1_29dof_wbt_w_object_r135_box021_handbox_exp0601_v4_3",
    "g1_29dof_wbt_w_object_r138_box021_r095_loadpath_partner_v4_3",
]

REWARD_FIELDS = [
    "source",
    "reward_config",
    "inherits_from",
    "term",
    "func",
    "weight",
    "offline_status",
    "reason",
]

PROXY_FIELDS = [
    "motion",
    "frames",
    "fps",
    "has_object",
    "has_partner",
    "has_object_contact",
    "object_contact_shape",
    "ref_mask_reward_allowed",
    "actor_hand_body_mapping",
    "sample",
    "mean_surface_distance_m",
    "p50_surface_distance_m",
    "p90_surface_distance_m",
    "min_surface_distance_m",
    "penetration_frac",
    "near_2cm_frac",
    "near_5cm_frac",
    "near_10cm_frac",
    "clean_near_5cm_frac",
    "longest_clean_near_5cm_run_frames",
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


def block_text(text: str, symbol: str) -> str:
    start = text.find(f"{symbol} = RewardManagerCfg(")
    if start < 0:
        return ""
    depth = 0
    seen = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
            seen = True
        elif ch == ")":
            depth -= 1
            if seen and depth == 0:
                return text[start : i + 1]
    return text[start:]


def experiment_block(text: str, symbol: str) -> str:
    start = text.find(f"{symbol} = replace(")
    if start < 0:
        return ""
    depth = 0
    seen = False
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
            seen = True
        elif ch == ")":
            depth -= 1
            if seen and depth == 0:
                return text[start : i + 1]
    return text[start:]


def classify_reward(func: str, term: str) -> tuple[str, str]:
    joined = f"{term} {func}"
    if "RefMasked" in joined:
        return ("blocked_requires_object_contact", "requires motion_command.ref_object_contact/object_contact mask")
    if "Contact" in joined or "Force" in joined:
        return ("blocked_requires_sim_contact", "requires IsaacSim contact sensors/forces, not NPZ-only geometry")
    if "SurfaceProximity" in joined or "Proximity" in joined:
        return ("offline_proxy_available", "approximated from object pose and hand point surface distance")
    if "Object" in joined:
        return ("motion_reference_only", "can inspect reference object trajectories but not learned policy behavior")
    return ("static_only", "static reward term listing only")


def parse_reward_terms() -> list[dict[str, Any]]:
    text = REWARD_SRC.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for config in REWARD_BLOCKS:
        block = block_text(text, config)
        inherit_match = re.search(r"\*\*([A-Za-z0-9_]+)\.terms", block)
        inherits_from = inherit_match.group(1) if inherit_match else ""
        for match in re.finditer(
            r'"([^"]+)":\s*RewardTermCfg\(\s*func="([^"]+)".*?weight=([-0-9.]+)',
            block,
            flags=re.S,
        ):
            term, func, weight = match.groups()
            status, reason = classify_reward(func, term)
            rows.append(
                {
                    "source": rel(REWARD_SRC),
                    "reward_config": config,
                    "inherits_from": inherits_from,
                    "term": term,
                    "func": func,
                    "weight": weight,
                    "offline_status": status,
                    "reason": reason,
                }
            )
    exp_text = EXPERIMENT_SRC.read_text(encoding="utf-8")
    for config in EXPERIMENT_CONFIGS:
        block = experiment_block(exp_text, config)
        reward_match = re.search(r"reward=reward\.([A-Za-z0-9_]+)", block)
        project_match = re.search(r'project="([^"]+)"', block)
        rows.append(
            {
                "source": rel(EXPERIMENT_SRC),
                "reward_config": config,
                "inherits_from": "",
                "term": "experiment_reward_mapping",
                "func": reward_match.group(1) if reward_match else "",
                "weight": "",
                "offline_status": "static_config_mapping",
                "reason": f"project={project_match.group(1) if project_match else ''}",
            }
        )
    return rows


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


def summarize_distances(base: dict[str, Any], sample: str, dist: np.ndarray) -> dict[str, Any]:
    clean_near_5cm = (dist >= 0.0) & (dist <= 0.05)
    return {
        **base,
        "sample": sample,
        "mean_surface_distance_m": f"{float(np.mean(dist)):.6f}",
        "p50_surface_distance_m": f"{float(np.percentile(dist, 50)):.6f}",
        "p90_surface_distance_m": f"{float(np.percentile(dist, 90)):.6f}",
        "min_surface_distance_m": f"{float(np.min(dist)):.6f}",
        "penetration_frac": f"{float(np.mean(dist < 0.0)):.6f}",
        "near_2cm_frac": f"{float(np.mean(dist <= 0.02)):.6f}",
        "near_5cm_frac": f"{float(np.mean(dist <= 0.05)):.6f}",
        "near_10cm_frac": f"{float(np.mean(dist <= 0.10)):.6f}",
        "clean_near_5cm_frac": f"{float(np.mean(clean_near_5cm)):.6f}",
        "longest_clean_near_5cm_run_frames": longest_true_run(clean_near_5cm),
    }


def inspect_motion(path: Path) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    body_names = [str(x) for x in data["body_names"].tolist()]
    body_index = {name: i for i, name in enumerate(body_names)}
    has_object = "object_pos_w" in data.files and "object_quat_w" in data.files
    has_partner = "partner_hand_pos_w" in data.files
    has_contact = "object_contact" in data.files
    contact_shape = "x".join(str(x) for x in data["object_contact"].shape) if has_contact else ""
    expected_contact = has_contact and tuple(data["object_contact"].shape[-1:]) == (2,)
    frames = int(data["body_pos_w"].shape[0])
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    actor_names = HAND_FALLBACK_NAMES
    mapping = "rubber_hand_fallback"
    if all(name in body_index for name in HOLOSOMA_HANDBOX_NAMES):
        actor_names = HOLOSOMA_HANDBOX_NAMES
        mapping = "holosoma_handbox"
    elif not all(name in body_index for name in actor_names):
        actor_names = tuple(name for name in actor_names if name in body_index)
        mapping = "partial_rubber_hand_fallback"

    base = {
        "motion": rel(path),
        "frames": frames,
        "fps": fps,
        "has_object": str(has_object).lower(),
        "has_partner": str(has_partner).lower(),
        "has_object_contact": str(has_contact).lower(),
        "object_contact_shape": contact_shape,
        "ref_mask_reward_allowed": str(expected_contact).lower(),
        "actor_hand_body_mapping": mapping,
    }
    rows: list[dict[str, Any]] = []
    if not has_object:
        return [summarize_distances(base, "missing_object_pose", np.array([np.nan]))]

    obj_pos = data["object_pos_w"].astype(np.float64)
    obj_quat = data["object_quat_w"].astype(np.float64)
    for name in actor_names:
        points = data["body_pos_w"][:, body_index[name], :].astype(np.float64)
        dist = box_signed_surface_distance(points, obj_pos, obj_quat)
        rows.append(summarize_distances(base, f"actor:{name}", dist))

    if has_partner:
        partner = data["partner_hand_pos_w"].astype(np.float64)
        for idx, side in enumerate(("left", "right")):
            dist = box_signed_surface_distance(partner[:, idx, :], obj_pos, obj_quat)
            rows.append(summarize_distances(base, f"partner:{side}_hand", dist))
    return rows


def summary_json(reward_rows: list[dict[str, Any]], proxy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    motions = sorted({row["motion"] for row in proxy_rows})
    ref_mask_allowed = sum(1 for row in proxy_rows if row["ref_mask_reward_allowed"] == "true")
    offline_terms = sum(1 for row in reward_rows if row["offline_status"] == "offline_proxy_available")
    blocked_terms = sum(1 for row in reward_rows if row["offline_status"].startswith("blocked"))
    return {
        "experiment": "E130",
        "status": "pass",
        "reward_term_rows": len(reward_rows),
        "motion_proxy_rows": len(proxy_rows),
        "motion_exports": motions,
        "paired_exports_audited": len(motions),
        "offline_proxy_reward_terms": offline_terms,
        "blocked_contact_or_force_terms": blocked_terms,
        "ref_mask_reward_allowed_rows": ref_mask_allowed,
        "rl_ready_rows": 0,
        "training_launched": False,
        "cem_launched": False,
        "checkpoint_created": False,
        "remote_jobs_launched": False,
        "main_release_evidence": False,
        "fragment_label": "FRAGMENT_HOLDOUT_ONLY",
        "notes": [
            "E126 paired exports include object pose and partner hand pose.",
            "E126 paired exports do not include object_contact, so ref-masked contact rewards are blocked.",
            "Actor handbox links are absent from exported body_names; actor geometry proxy uses rubber-hand fallback links.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any], proxy_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E130 Holosoma Reward-Side Inspection Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| reward term rows | {summary['reward_term_rows']} |",
        f"| motion proxy rows | {summary['motion_proxy_rows']} |",
        f"| paired exports audited | {summary['paired_exports_audited']} |",
        f"| offline proxy reward terms | {summary['offline_proxy_reward_terms']} |",
        f"| blocked contact/force terms | {summary['blocked_contact_or_force_terms']} |",
        f"| ref-mask allowed rows | {summary['ref_mask_reward_allowed_rows']} |",
        f"| RL-ready rows | {summary['rl_ready_rows']} |",
        f"| training launched | {str(summary['training_launched']).lower()} |",
        f"| CEM launched | {str(summary['cem_launched']).lower()} |",
        "",
        "## Motion Proxy Rows",
        "",
        "| motion | sample | mean dist | clean near 5cm | longest clean run |",
        "|---|---|---:|---:|---:|",
    ]
    for row in proxy_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{Path(row['motion']).name}`",
                    f"`{row['sample']}`",
                    row["mean_surface_distance_m"],
                    row["clean_near_5cm_frac"],
                    str(row["longest_clean_near_5cm_run_frames"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "E130 is inspection evidence only. The paired fragments can support an offline hand/object surface-distance proxy, but they cannot validate simulator contact-force rewards or ref-masked contact rewards because `object_contact` is absent. The actor-side proxy uses `left_rubber_hand_link` and `right_rubber_hand_link` because Holosoma handbox links are not present in the E126 export body list.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reward_rows = parse_reward_terms()
    proxy_rows: list[dict[str, Any]] = []
    for motion in MOTIONS:
        proxy_rows.extend(inspect_motion(motion))
    summary = summary_json(reward_rows, proxy_rows)

    write_tsv(OUT_DIR / "e130_reward_terms.tsv", reward_rows, REWARD_FIELDS)
    write_tsv(OUT_DIR / "e130_motion_reward_proxy.tsv", proxy_rows, PROXY_FIELDS)
    (OUT_DIR / "e130_reward_side_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(OUT_DIR / "e130_reward_side_summary.md", summary, proxy_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
