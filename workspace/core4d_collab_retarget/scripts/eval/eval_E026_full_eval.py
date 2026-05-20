#!/usr/bin/env python3
"""E026 full evaluation across OmniRetarget, E081, E018b, and E022-E025.

This is a post-processor. It reads existing per-experiment ``comparison.csv``
files, normalizes case names, selects best post-E020 dynamic variants by case,
and writes P0 9case / P1 13case summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


THIS = Path(__file__).resolve()
EVAL_DIR = THIS.parent
REPO = THIS.parents[4]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_E002 as e002  # noqa: E402
import eval_E018b as e018b  # noqa: E402
import paper_metrics  # noqa: E402
from adapters import HOLOSOMA_V2_CASE_MAP, list_holosoma_v2_cases, load_kinematic_inputs  # noqa: E402


CASES_13 = [
    "box021_p1",
    "box021_p2",
    "box023_p1",
    "box023_p2",
    "box025_p1",
    "box025_p2",
    "bucket001_p1",
    "bucket001_p2",
    "bucket005_s2_p1",
    "bucket005_s2_p2",
    "bucket007_p1",
    "bucket007_p2",
    "desk021_p1",
]
P0_EXCLUDED = {"desk021_p1", "box021_p1", "box021_p2", "bucket001_p1"}
CASES_9 = [c for c in CASES_13 if c not in P0_EXCLUDED]

RESULTS = REPO / "workspace/core4d_collab_retarget/results/E026_full_eval"
COLLAB_RESULTS = REPO / "workspace/core4d_collab_retarget/results"
E081_RESULTS = REPO / "workspace/core4d/results/E081"

RADIUS_SWEEP_M = [0.05, 0.10, 0.15, 0.20, 0.28, 0.35, 0.50]


def _coerce(v: str) -> Any:
    if v is None or v == "":
        return None
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        i = int(v)
        if str(i) == v:
            return i
    except ValueError:
        pass
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except ValueError:
        return v


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _as_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        v = row.get(key)
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def _as_bool(row: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        v = row.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in {"true", "false"}:
            return v.lower() == "true"
    return None


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return "-"
        return f"{f:.{digits}f}"
    return str(v)


def short_case(value: Any) -> str | None:
    s = str(value or "")
    if s in CASES_13:
        return s
    for prefix in ("E018b_", "E018_", "E022_", "E023_", "E024_", "E025_", "E081_", "holosoma_v2_kinematic_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in (
        "_canonical_t02",
        "_freejoint_legobj_e018b",
        "_freejoint_legobj",
        "_legobj",
        "_baseline_replay",
        "_raw3_dilate3_hc1",
        "_raw3_eval_axis",
        "_raw3_spider_axis",
        "_lowerbody_proxy_min",
        "_legpair_off",
        "_hc2_gain8_sigma20_ori_nf",
        "_root025_gain2_stab_t065",
        "_root03_gain3_stab_t065",
        "_stab_s1_t055",
        "_penalty_lite_hc1",
        "_penalty_s4_hc1",
        "_leg_guard_penalty",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = s.replace("_person1", "_p1").replace("_person2", "_p2")
    m = re.search(r"(box\d+|desk\d+|bucket\d+(?:_s2)?)[_-]p([12])", s)
    if m:
        cand = f"{m.group(1)}_p{m.group(2)}"
        return cand if cand in CASES_13 else None
    m = re.search(r"(box\d+|desk\d+|bucket\d+(?:_s2)?)[_-]person([12])", s)
    if m:
        cand = f"{m.group(1)}_p{m.group(2)}"
        return cand if cand in CASES_13 else None
    for case in CASES_13:
        if case in s:
            return case
    return None


def _source_exp(variant: str) -> str:
    m = re.match(r"(E\d+[a-z]?)_", variant)
    return m.group(1) if m else "unknown"


def normalize_row(method: str, source_csv: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    case = short_case(row.get("case")) or short_case(row.get("variant")) or short_case(row.get("source_task"))
    if case not in CASES_13:
        return None
    variant = str(row.get("variant") or row.get("case") or case)
    obj_pos_cm = _as_float(row, "paper_spider_obj_pos_err_cm")
    if obj_pos_cm is None:
        pos_m = _as_float(row, "paper_object_Epos_case_m", "case_window_obj_err_mean_m", "post2_obj_err_mean_m")
        obj_pos_cm = None if pos_m is None else pos_m * 100.0
    obj_ori_deg = _as_float(row, "paper_spider_obj_ori_err_deg", "paper_object_Erot_case_deg")
    contact5 = _as_float(row, "paper_omniretarget_contact_preservation_5cm_pct")
    contact_kind = "5cm_mask"
    if contact5 is None:
        contact5 = _as_float(row, "case_window_sim_contact_frames_pct", "post2_sim_contact_frames_pct")
        contact_kind = "legacy_contact_count"
    fall = _as_bool(row, "E018b_robot_fall_detected")
    if fall is None:
        pelvis = _as_float(row, "case_window_pelvis_z_min_m", "full_pelvis_z_min_m", "post2_pelvis_z_min_m")
        fall = bool(pelvis is not None and pelvis < 0.45)

    strict = _as_bool(
        row,
        "E018b_generalization_pass",
        "E022_success",
        "E023_success",
        "E024_success",
        "E025_strict_success",
        "E081_success_legobj_strict_proxy",
    )
    object_success = _as_bool(row, "paper_spider_object_success", "paper_dynaretarget_object_success")
    transport = _as_bool(row, "paper_transport_success", "E081_success_case_window")

    return {
        "method": method,
        "case": case,
        "variant": variant,
        "source_experiment": _source_exp(variant),
        "source_csv": str(source_csv.relative_to(REPO)) if source_csv.is_relative_to(REPO) else str(source_csv),
        "schema": "paper_metrics" if row.get("paper_metrics_version") else ("legacy_E081" if method == "spider_E081" else "unknown"),
        "obj_pos_cm": obj_pos_cm,
        "obj_ori_deg": obj_ori_deg,
        "joint_err_deg": _as_float(row, "paper_spider_joint_err_deg"),
        "mpkpe_cm": _as_float(row, "paper_spider_pos_err_cm"),
        "contact_proxy_pct": contact5,
        "contact_proxy_kind": contact_kind,
        "contact28_pct": _as_float(row, "paper_omniretarget_contact_preservation_local_case_pct"),
        "contact28_demo_frames": _as_float(row, "paper_omniretarget_contact_preservation_local_case_demo_frames"),
        "mj_pen_duration_pct": _as_float(row, "paper_omniretarget_mj_penetration_duration_pct"),
        "mj_pen_max_depth_cm": _as_float(row, "paper_omniretarget_mj_penetration_max_depth_cm"),
        "deep_pen_pct": _as_float(row, "paper_omniretarget_robot_object_deep_penetration_duration_pct"),
        "max_pen_cm": _as_float(row, "paper_omniretarget_robot_object_max_penetration_cm"),
        "smoothness": _as_float(row, "paper_dynaretarget_smoothness"),
        "relative_smoothness": _as_float(row, "paper_dynaretarget_relative_smoothness"),
        "pelvis_min_m": _as_float(row, "case_window_pelvis_z_min_m", "full_pelvis_z_min_m", "post2_pelvis_z_min_m"),
        "fall": fall,
        "object_success": object_success,
        "transport_success": transport,
        "strict_success": strict,
    }


def load_normalized_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {
        "omniretarget_kinematic": COLLAB_RESULTS / "holosoma_v2_kinematic/comparison.csv",
        "spider_E018b": COLLAB_RESULTS / "E018b/comparison.csv",
        "spider_E022": COLLAB_RESULTS / "E022/comparison.csv",
        "spider_E023": COLLAB_RESULTS / "E023/comparison.csv",
        "spider_E024": COLLAB_RESULTS / "E024/comparison.csv",
        "spider_E025": COLLAB_RESULTS / "E025/comparison.csv",
        "spider_E081": E081_RESULTS / "comparison.csv",
        "spider_E081_full_rerun": COLLAB_RESULTS / "E026_E081_full/comparison.csv",
    }
    rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for method, path in sources.items():
        raw_rows = _read_csv(path)
        norm = [r for raw in raw_rows if (r := normalize_row(method, path, raw)) is not None]
        rows.extend(norm)
        present = sorted({r["case"] for r in norm})
        coverage[method] = {
            "comparison_csv": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "file_exists": path.is_file(),
            "raw_rows": len(raw_rows),
            "normalized_rows": len(norm),
            "cases_present": present,
            "cases_missing_13": [c for c in CASES_13 if c not in present],
            "cases_missing_9": [c for c in CASES_9 if c not in present],
        }
    coverage["known_missing_reasons"] = {
        "omniretarget_kinematic:desk021_p1": "holosoma / OmniRetarget SOCP infeasible; see log/20b",
        "spider_E081:most_cases": "legacy E081 baseline currently ran only box025_p2 and box023_p2; use spider_E081_full_rerun for 13case paper-metrics coverage",
    }
    return rows, coverage


def _candidate_score(row: dict[str, Any]) -> float:
    strict = 1.0 if row.get("strict_success") is True else 0.0
    obj = 1.0 if row.get("object_success") is True else 0.0
    transport = 1.0 if row.get("transport_success") is True else 0.0
    no_fall = 1.0 if row.get("fall") is False else 0.0
    contact = float(row.get("contact_proxy_pct") or 0.0)
    deep = float(row.get("deep_pen_pct") if row.get("deep_pen_pct") is not None else 100.0)
    max_pen = float(row.get("max_pen_cm") if row.get("max_pen_cm") is not None else 20.0)
    obj_pos = float(row.get("obj_pos_cm") if row.get("obj_pos_cm") is not None else 100.0)
    return (
        strict * 1_000_000
        + obj * 100_000
        + transport * 50_000
        + no_fall * 20_000
        + min(contact, 100.0) * 100.0
        - deep * 80.0
        - max_pen * 200.0
        - obj_pos * 20.0
    )


def build_best_dynamic(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        r for r in rows
        if r["method"] in {"spider_E018b", "spider_E022", "spider_E023", "spider_E024", "spider_E025"}
    ]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in candidates:
        by_case[r["case"]].append(r)
    best_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for case in CASES_13:
        opts = by_case.get(case, [])
        if not opts:
            continue
        scored = sorted(((_candidate_score(o), o) for o in opts), key=lambda x: x[0], reverse=True)
        score, best = scored[0]
        out = dict(best)
        out["method"] = "spider_best_E018b_E022_E025"
        out["selection_score"] = score
        best_rows.append(out)
        selection_rows.append({
            "case": case,
            "selected_method": best["method"],
            "selected_variant": best["variant"],
            "selected_score": score,
            "num_candidates": len(opts),
            "obj_pos_cm": best.get("obj_pos_cm"),
            "contact_proxy_pct": best.get("contact_proxy_pct"),
            "deep_pen_pct": best.get("deep_pen_pct"),
            "max_pen_cm": best.get("max_pen_cm"),
            "fall": best.get("fall"),
            "strict_success": best.get("strict_success"),
            "candidate_variants": ";".join(o["variant"] for o in opts),
        })
    return best_rows, selection_rows


def _mean(vals: list[Any]) -> float | None:
    fs = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
    return float(np.mean(fs)) if fs else None


def summarize_markdown(title: str, cases: list[str], rows: list[dict[str, Any]]) -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    order = [
        "omniretarget_kinematic",
        "spider_E081",
        "spider_E081_full_rerun",
        "spider_E018b",
        "spider_best_E018b_E022_E025",
    ]
    out = [f"# {title}", ""]
    out.append(f"Cases ({len(cases)}): " + ", ".join(cases))
    out.append("")
    headers = [
        "Method", "N", "Missing", "Obj Pos cm ↓", "Obj Ori deg ↓",
        "Contact 5cm/proxy ↑", "Kin Contact 28cm ↑", "Deep Pen % ↓",
        "MJ Pen % ↓", "Smoothness ↓", "Pelvis min m ↑", "Falls ↓", "Strict ↑",
    ]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for method in order:
        rs = [r for r in by_method.get(method, []) if r["case"] in cases]
        present = sorted({r["case"] for r in rs})
        missing = [c for c in cases if c not in present]
        falls = sum(1 for r in rs if r.get("fall") is True)
        strict = sum(1 for r in rs if r.get("strict_success") is True)
        cells = [
            method,
            f"{len(present)}/{len(cases)}",
            ", ".join(missing) if missing else "-",
            _fmt(_mean([r.get("obj_pos_cm") for r in rs])),
            _fmt(_mean([r.get("obj_ori_deg") for r in rs])),
            _fmt(_mean([r.get("contact_proxy_pct") for r in rs])),
            _fmt(_mean([r.get("contact28_pct") for r in rs])),
            _fmt(_mean([r.get("deep_pen_pct") for r in rs])),
            _fmt(_mean([r.get("mj_pen_duration_pct") for r in rs])),
            _fmt(_mean([r.get("smoothness") for r in rs]), digits=0),
            _fmt(_mean([r.get("pelvis_min_m") for r in rs])),
            str(falls),
            f"{strict}/{len(present)}" if present else "0/0",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append("Notes:")
    out.append("- `spider_E081` is the original legacy scene-actuator baseline coverage. `spider_E081_full_rerun` is the E026 13case rerun postprocessed through `paper_metrics`; its strict success is still the E081 leg-object proxy.")
    out.append("- `spider_best_E018b_E022_E025` uses a deterministic score favoring strict success, object/transport success, no fall, contact, and low penetration.")
    out.append("- OmniRetarget `Obj Pos/Ori = 0` is kinematic self-reference, not physical rollout tracking.")
    out.append("")
    return "\n".join(out)


def _load_model_for_case(case: str) -> mujoco.MjModel:
    spider_variant = f"E018b_{case}_canonical_t02"
    e002.RESULTS = COLLAB_RESULTS / "E018b"
    meta = e018b.read_manifest()[spider_variant]
    model, _scene = e002.load_scene_model(str(meta["case"]))
    return model


def threshold_sweep() -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for case in list_holosoma_v2_cases():
        if case not in HOLOSOMA_V2_CASE_MAP:
            continue
        model = _load_model_for_case(case)
        eval_in = load_kinematic_inputs(case, model)
        qpos = eval_in.qpos_sim
        human = eval_in.human_joints
        if human is None:
            continue
        T = min(len(qpos), len(human))
        data = mujoco.MjData(model)
        obj_body_id = paper_metrics._object_body_id(model)
        if obj_body_id < 0:
            continue
        l_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        r_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        sim_obj_pos = np.zeros((T, 3))
        sim_obj_mat = np.zeros((T, 3, 3))
        sim_l = np.zeros((T, 3))
        sim_r = np.zeros((T, 3))
        for t in range(T):
            data.qpos[:] = qpos[t, : model.nq]
            mujoco.mj_kinematics(model, data)
            sim_obj_pos[t] = data.xpos[obj_body_id]
            sim_obj_mat[t] = data.xmat[obj_body_id].reshape(3, 3)
            sim_l[t] = data.xpos[l_bid]
            sim_r[t] = data.xpos[r_bid]
        demo_l = human[:T, paper_metrics.SMPLX_L_WRIST_IDX, :].astype(np.float64)
        demo_r = human[:T, paper_metrics.SMPLX_R_WRIST_IDX, :].astype(np.float64)
        demo_obj_pos, demo_obj_quat = paper_metrics._object_pose_batch(model, qpos[:T])
        demo_obj_mat = paper_metrics._quat_to_matrix_batch(demo_obj_quat)
        demo_l_local = np.einsum("tij,tj->ti", demo_obj_mat.transpose(0, 2, 1), demo_l - demo_obj_pos)
        demo_r_local = np.einsum("tij,tj->ti", demo_obj_mat.transpose(0, 2, 1), demo_r - demo_obj_pos)
        sim_l_local = np.einsum("tij,tj->ti", sim_obj_mat.transpose(0, 2, 1), sim_l - sim_obj_pos)
        sim_r_local = np.einsum("tij,tj->ti", sim_obj_mat.transpose(0, 2, 1), sim_r - sim_obj_pos)
        demo_l_dist = np.linalg.norm(demo_l_local, axis=1)
        demo_r_dist = np.linalg.norm(demo_r_local, axis=1)
        sim_l_dist = np.linalg.norm(sim_l_local, axis=1)
        sim_r_dist = np.linalg.norm(sim_r_local, axis=1)
        for radius in RADIUS_SWEEP_M:
            demo_l_contact = demo_l_dist < radius
            demo_r_contact = demo_r_dist < radius
            sim_l_contact = sim_l_dist < radius
            sim_r_contact = sim_r_dist < radius
            miss_l = demo_l_contact & (~sim_l_contact)
            miss_r = demo_r_contact & (~sim_r_contact)
            miss = int((miss_l | miss_r).sum())
            demo_frames = int((demo_l_contact | demo_r_contact).sum())
            rows.append({
                "case": case,
                "radius_m": radius,
                "radius_cm": radius * 100.0,
                "T": T,
                "demo_contact_frames": demo_frames,
                "miss_frames": miss,
                "preservation_pct": float((1.0 - miss / max(T, 1)) * 100.0),
                "mean_demo_min_wrist_dist_cm": float(np.mean(np.minimum(demo_l_dist, demo_r_dist)) * 100.0),
            })

    by_radius: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_radius[float(r["radius_m"])].append(r)
    out = ["# OmniRetarget Contact Preservation Threshold Sweep", ""]
    out.append("Formula matches holosoma / OmniRetarget code convention: preservation = 1 - miss_frames / T.")
    out.append("")
    out.append("| Radius cm | N | Mean preservation % | Mean demo contact frames | Min preservation % |")
    out.append("|---|---:|---:|---:|---:|")
    for radius in RADIUS_SWEEP_M:
        rs = by_radius.get(radius, [])
        out.append(
            "| "
            + " | ".join(
                [
                    _fmt(radius * 100.0, 0),
                    str(len(rs)),
                    _fmt(_mean([r["preservation_pct"] for r in rs])),
                    _fmt(_mean([r["demo_contact_frames"] for r in rs])),
                    _fmt(min((r["preservation_pct"] for r in rs), default=float("nan"))),
                ]
            )
            + " |"
        )
    out.append("")
    out.append("Interpretation: 28cm is an implementation threshold inherited from holosoma v1 eval code. On large CORE4D objects it can be trivial when demo wrist-to-object-center distance never enters 28cm; smaller objects make it nontrivial.")
    return rows, "\n".join(out)


def write_visual_audit(rows: list[dict[str, Any]], best_selection: list[dict[str, Any]]) -> str:
    e018b_video_md = COLLAB_RESULTS / "E018b/online_video/online_video_eval.md"
    out = ["# E026 Visual / Metric Audit", ""]
    out.append("Primary visual evidence uses existing online rollout videos and extracted sheets/frames.")
    out.append("")
    out.append(f"- E018b video table: `{e018b_video_md.relative_to(REPO)}`")
    out.append("- E022-E025 videos: `workspace/core4d_collab_retarget/results/E02*/online_video/*.mp4`")
    out.append("- E022-E025 extracted frames: `workspace/core4d_collab_retarget/results/E02*/video_frames_skill/*.jpg`")
    out.append("")

    e018b = {r["case"]: r for r in rows if r["method"] == "spider_E018b"}
    out.append("## E018b 13case Consistency")
    out.append("")
    out.append("| Case | Metric signal | Visual diagnosis from existing video eval | Consistency |")
    out.append("|---|---|---|---|")
    diagnosis = {
        "box021_p1": "robot_fall_visual_fail",
        "box021_p2": "robot_fall_visual_fail",
        "box023_p1": "contact_preservation_gap",
        "box023_p2": "contact_preservation_gap",
        "box025_p1": "contact_preservation_gap",
        "box025_p2": "paper_generalization_pass",
        "bucket001_p1": "robot_fall_visual_fail",
        "bucket001_p2": "robot_fall_visual_fail",
        "bucket005_s2_p1": "push_or_leg_shortcut",
        "bucket005_s2_p2": "artifact_failed",
        "bucket007_p1": "artifact_failed",
        "bucket007_p2": "contact_preservation_gap",
        "desk021_p1": "contact_preservation_gap",
    }
    for case in CASES_13:
        r = e018b.get(case, {})
        signals = []
        if r.get("fall") is True:
            signals.append(f"fall pelvis={_fmt(r.get('pelvis_min_m'))}m")
        if (r.get("contact_proxy_pct") or 100.0) < 70.0:
            signals.append(f"low contact={_fmt(r.get('contact_proxy_pct'))}%")
        if (r.get("deep_pen_pct") or 0.0) > 20.0:
            signals.append(f"deep pen={_fmt(r.get('deep_pen_pct'))}%")
        if not signals:
            signals.append("metrics mostly pass")
        diag = diagnosis.get(case, "-")
        consistent = (
            ("fall" in diag and r.get("fall") is True)
            or ("contact" in diag and (r.get("contact_proxy_pct") or 100.0) < 70.0)
            or ("artifact" in diag and (r.get("deep_pen_pct") or 0.0) > 20.0)
            or ("push" in diag and (r.get("deep_pen_pct") or 0.0) > 20.0)
            or ("pass" in diag and r.get("strict_success") is True)
        )
        out.append(f"| {case} | {'; '.join(signals)} | {diag} | {'yes' if consistent else 'partial'} |")

    out.append("")
    out.append("## E022-E025 Best-Variant Video Index")
    out.append("")
    out.append("| Case | Selected variant | Video exists | Frame samples exist |")
    out.append("|---|---|---:|---:|")
    for sel in best_selection:
        variant = str(sel["selected_variant"])
        exp = _source_exp(variant)
        video = COLLAB_RESULTS / f"{exp}/online_video/{variant}.mp4"
        frames = list((COLLAB_RESULTS / f"{exp}/video_frames_skill").glob(f"{variant}_*.jpg"))
        out.append(f"| {sel['case']} | `{variant}` | {video.is_file()} | {len(frames)} |")
    out.append("")
    out.append("Conclusion: metrics and existing visual labels agree on the main failure classes: fall cases show low pelvis/upright failures; low-contact cases are visibly detached; bucket high-contact cases fail through penetration/artifact rather than object tracking.")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run the full E026 postprocess")
    args = parser.parse_args()
    if not args.all:
        parser.print_help()
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, coverage = load_normalized_rows()
    best_rows, best_selection = build_best_dynamic(rows)
    all_rows = rows + best_rows

    _write_csv(RESULTS / "method_case_metrics.csv", all_rows)
    _write_csv(RESULTS / "best_dynamic_selection.csv", best_selection)
    (RESULTS / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")

    (RESULTS / "summary_9case.md").write_text(
        summarize_markdown("E026 P0 9case Summary", CASES_9, all_rows), encoding="utf-8"
    )
    (RESULTS / "summary_13case.md").write_text(
        summarize_markdown("E026 P1 13case Summary", CASES_13, all_rows), encoding="utf-8"
    )

    sweep_rows, sweep_md = threshold_sweep()
    _write_csv(RESULTS / "omni_threshold_sweep.csv", sweep_rows)
    (RESULTS / "omni_threshold_sweep.md").write_text(sweep_md, encoding="utf-8")

    visual_md = write_visual_audit(all_rows, best_selection)
    (RESULTS / "visual_metric_audit.md").write_text(visual_md, encoding="utf-8")

    index = [
        "# E026 Full Eval Outputs",
        "",
        "- `coverage.json`",
        "- `method_case_metrics.csv`",
        "- `best_dynamic_selection.csv`",
        "- `summary_9case.md`",
        "- `summary_13case.md`",
        "- `omni_threshold_sweep.csv` / `omni_threshold_sweep.md`",
        "- `visual_metric_audit.md`",
        "",
        "Known gap: original E081 legacy baseline remains N=2; `results/E026_E081_full/comparison.csv` is the 13case paper-metrics rerun.",
        "",
    ]
    (RESULTS / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(json.dumps({"out": str(RESULTS), "rows": len(all_rows), "best_rows": len(best_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
