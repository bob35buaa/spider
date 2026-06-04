#!/usr/bin/env python3
"""Build E125 RL hand-support preflight artifacts from recent CEM smoke rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO / "workspace/core4d/results/E125/rl_hand_support_preflight"
RECENT_EXPERIMENTS = ("E120", "E121", "E122", "E123", "E124")
MAIN_CASE = "box021_029_p2"

ALL_FIELDS = [
    "experiment",
    "case_id",
    "variant",
    "ablation",
    "method",
    "phase_scope",
    "work_status",
    "decision",
    "decision_reason",
    "rank_score",
    "physics_contact_frac",
    "lowerbody_contact_frac",
    "nonhand_support_frac",
    "hand_near_zero_frac",
    "deep_pen_frac",
    "pelvis_min_m",
    "obj_err_mean_m",
    "npz_path",
    "scene_xml",
    "video_path",
    "converter_input_npz",
    "converter_meta_json",
    "qpos43_shape",
    "fps",
]


@dataclass
class Candidate:
    experiment: str
    case_id: str
    variant: str
    ablation: str
    method: str
    phase_scope: str
    work_status: str
    physics_contact_frac: float
    lowerbody_contact_frac: float
    nonhand_support_frac: float
    hand_near_zero_frac: float
    deep_pen_frac: float
    pelvis_min_m: float
    obj_err_mean_m: float
    npz_path: str
    scene_xml: str
    video_path: str
    decision: str = ""
    decision_reason: str = ""
    rank_score: float = 0.0
    converter_input_npz: str = ""
    converter_meta_json: str = ""
    qpos43_shape: str = ""
    fps: str = ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def candidate_from_row(experiment: str, row: dict[str, str]) -> Candidate | None:
    method = row.get("method", "")
    ablation = row.get("ablation", "")
    if not method or method != ablation:
        return None
    physics = safe_float(row, "hand_object_physics_contact")
    if physics == 0.0:
        physics = safe_float(row, "hand_object_contact_physics_frac")
    cand = Candidate(
        experiment=experiment,
        case_id=row.get("case_id", ""),
        variant=row.get("variant", ""),
        ablation=ablation,
        method=method,
        phase_scope=row.get("phase_scope", ""),
        work_status=row.get("work_status", ""),
        physics_contact_frac=physics,
        lowerbody_contact_frac=safe_float(row, "leg_object_contact_frac"),
        nonhand_support_frac=safe_float(row, "nonhand_object_support_frac"),
        hand_near_zero_frac=safe_float(row, "hand_support_near_zero_frac"),
        deep_pen_frac=safe_float(row, "hand_geom_deep_penetration_2cm"),
        pelvis_min_m=safe_float(row, "pelvis_min_m"),
        obj_err_mean_m=safe_float(row, "obj_err_mean_m"),
        npz_path=row.get("npz_path", ""),
        scene_xml=row.get("scene_xml", ""),
        video_path=row.get("video_path", ""),
    )
    classify(cand)
    return cand


def strict_support_pass(cand: Candidate) -> bool:
    return (
        cand.physics_contact_frac >= 0.57
        and cand.lowerbody_contact_frac <= 0.05
        and cand.nonhand_support_frac <= 0.05
        and cand.hand_near_zero_frac >= 0.50
        and cand.deep_pen_frac <= 0.03
        and cand.pelvis_min_m >= 0.55
        and cand.obj_err_mean_m <= 0.02
        and cand.work_status == "PASS"
    )


def classify(cand: Candidate) -> None:
    posture_gap = max(0.0, 0.55 - cand.pelvis_min_m)
    cand.rank_score = (
        1.5 * cand.physics_contact_frac
        + cand.hand_near_zero_frac
        - 2.0 * cand.lowerbody_contact_frac
        - 2.0 * cand.nonhand_support_frac
        - 2.0 * cand.deep_pen_frac
        - posture_gap
    )
    if strict_support_pass(cand) and cand.case_id == MAIN_CASE:
        cand.decision = "RL_EXPORT_READY"
        cand.decision_reason = "main case satisfies strict hand-support preflight"
    elif strict_support_pass(cand):
        cand.decision = "FRAGMENT_HOLDOUT_ONLY"
        cand.decision_reason = "support geometry passes but row is not the main release case"
    elif cand.case_id == MAIN_CASE:
        cand.decision = "BLOCK_MAIN_GATE_FAIL"
        cand.decision_reason = "main case still fails strict support/posture gate"
    elif cand.physics_contact_frac < 0.20:
        cand.decision = "NO_CONTACT"
        cand.decision_reason = "physics hand-object contact below 20%"
    elif cand.lowerbody_contact_frac > 0.10 or cand.nonhand_support_frac > 0.10:
        cand.decision = "SHORTCUT_SUPPORT"
        cand.decision_reason = "contact uses lower-body or other non-hand support"
    else:
        cand.decision = "REVIEW_FRAGMENT"
        cand.decision_reason = "partial support signal but not strict enough"


def load_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for exp in RECENT_EXPERIMENTS:
        path = REPO / f"workspace/core4d/results/{exp}/cem/smoke/smoke_method_metrics.csv"
        for row in read_rows(path):
            cand = candidate_from_row(exp, row)
            if cand is not None:
                out.append(cand)
    return out


def infer_fps(data: np.lib.npyio.NpzFile) -> float:
    if "time" not in data.files:
        return 30.0
    time = np.asarray(data["time"], dtype=np.float64)
    if time.ndim == 2:
        time = time[:, 0]
    if time.shape[0] < 2:
        return 30.0
    dt = float(np.median(np.diff(time)))
    if dt <= 0.0 or math.isnan(dt):
        return 30.0
    return float(round(1.0 / dt, 3))


def scene_act_to_freejoint_qpos(qpos_act: np.ndarray, scene_act: Path) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_act))
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if obj_bid == -1:
        raise ValueError(f"object body not found in {scene_act}")
    meta_path = scene_act.with_name("scene_act_meta.json")
    euler_conv = "XYZ"
    if meta_path.is_file():
        euler_conv = json.loads(meta_path.read_text(encoding="utf-8")).get("euler_convention", "XYZ")

    nq_robot = qpos_act.shape[1] - 6
    if nq_robot != 36:
        raise ValueError(f"expected 36 robot qpos + 6 object qpos, got {qpos_act.shape}")
    body_pos = model.body_pos[obj_bid].copy()
    body_quat_wxyz = model.body_quat[obj_bid].copy()
    body_rot = R.from_quat([body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]])

    slide = qpos_act[:, nq_robot : nq_robot + 3]
    euler = qpos_act[:, nq_robot + 3 : nq_robot + 6]
    world_pos = body_pos[None, :] + body_rot.apply(slide)
    quat_xyzw = (body_rot * R.from_euler(euler_conv, euler)).as_quat()
    quat_wxyz = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])

    qpos_free = np.zeros((qpos_act.shape[0], nq_robot + 7), dtype=np.float64)
    qpos_free[:, :nq_robot] = qpos_act[:, :nq_robot]
    qpos_free[:, nq_robot : nq_robot + 3] = world_pos
    qpos_free[:, nq_robot + 3 : nq_robot + 7] = quat_wxyz
    return qpos_free


def load_sim_qpos42(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as data:
        if "qpos" not in data.files:
            raise KeyError(f"{path} missing qpos")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        fps = infer_fps(data)
    if qpos.ndim == 3 and qpos.shape[1] >= 1 and qpos.shape[2] == 42:
        return qpos[:, 0, :], fps
    if qpos.ndim == 2 and qpos.shape[1] == 42:
        return qpos, fps
    raise ValueError(f"{path}: expected qpos shape (T,2,42) or (T,42), got {qpos.shape}")


def export_converter_input(cand: Candidate, out_dir: Path) -> None:
    npz = repo_path(cand.npz_path)
    scene = repo_path(cand.scene_xml)
    qpos42, fps = load_sim_qpos42(npz)
    qpos43 = scene_act_to_freejoint_qpos(qpos42, scene)
    if qpos43.ndim != 2 or qpos43.shape[1] != 43:
        raise ValueError(f"{cand.variant}: converted qpos shape invalid: {qpos43.shape}")
    if not np.isfinite(qpos43).all():
        raise ValueError(f"{cand.variant}: converted qpos contains non-finite values")

    stem = safe_id(f"{cand.experiment}_{cand.variant}")
    converter_dir = out_dir / "converter_inputs"
    converter_dir.mkdir(parents=True, exist_ok=True)
    qpos_path = converter_dir / f"{stem}_qpos43.npz"
    meta_path = converter_dir / f"{stem}_qpos43.json"
    np.savez(
        qpos_path,
        qpos=qpos43.astype(np.float32),
        fps=np.asarray(fps, dtype=np.float32),
    )
    meta = {
        "experiment": cand.experiment,
        "case_id": cand.case_id,
        "variant": cand.variant,
        "decision": cand.decision,
        "source_npz": cand.npz_path,
        "scene_xml": cand.scene_xml,
        "video_path": cand.video_path,
        "qpos42_shape": list(qpos42.shape),
        "qpos43_shape": list(qpos43.shape),
        "fps": fps,
        "metrics": {
            "physics_contact_frac": cand.physics_contact_frac,
            "lowerbody_contact_frac": cand.lowerbody_contact_frac,
            "nonhand_support_frac": cand.nonhand_support_frac,
            "hand_near_zero_frac": cand.hand_near_zero_frac,
            "deep_pen_frac": cand.deep_pen_frac,
            "pelvis_min_m": cand.pelvis_min_m,
            "obj_err_mean_m": cand.obj_err_mean_m,
        },
        "rl_train_allowed": cand.decision == "RL_EXPORT_READY",
        "notes": "qpos43 converter input only; not a Holosoma training launch.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cand.converter_input_npz = rel(qpos_path)
    cand.converter_meta_json = rel(meta_path)
    cand.qpos43_shape = str(list(qpos43.shape))
    cand.fps = f"{fps:.3f}"


def select_rows(candidates: list[Candidate]) -> list[Candidate]:
    by_case: dict[str, Candidate] = {}
    for cand in candidates:
        prev = by_case.get(cand.case_id)
        if prev is None or cand.rank_score > prev.rank_score:
            by_case[cand.case_id] = cand
    return [by_case[key] for key in sorted(by_case)]


def write_tsv(path: Path, rows: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: asdict(row).get(key, "") for key in ALL_FIELDS})


def write_summary(out_dir: Path, candidates: list[Candidate], selected: list[Candidate]) -> None:
    decision_counts = Counter(c.decision for c in candidates)
    selected_counts = Counter(c.decision for c in selected)
    summary = {
        "stage": "E125_rl_hand_support_preflight",
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "rl_ready_rows": decision_counts.get("RL_EXPORT_READY", 0),
        "decision_counts": dict(decision_counts),
        "selected_decision_counts": dict(selected_counts),
        "outputs": {
            "candidate_metrics": rel(out_dir / "candidate_metrics.tsv"),
            "selected_preflight": rel(out_dir / "selected_preflight.tsv"),
            "rl_objective_manifest": rel(out_dir / "rl_objective_manifest.tsv"),
            "converter_inputs_dir": rel(out_dir / "converter_inputs"),
        },
        "training_launched": False,
        "subagent_status": "spawn attempted but thread limit reached; local audit used",
    }
    (out_dir / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# E125 RL Hand-Support Preflight Summary",
        "",
        f"- candidate rows: `{len(candidates)}`",
        f"- selected rows: `{len(selected)}`",
        f"- RL-ready rows: `{summary['rl_ready_rows']}`",
        "- training launched: `False`",
        "",
        "## Decision Counts",
        "",
        "| decision | count |",
        "|---|---:|",
    ]
    for key, count in sorted(decision_counts.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Selected Rows",
            "",
            "| case | experiment | variant | decision | physics | lower-body | non-hand | hand near-zero | pelvis | converter input |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected:
        lines.append(
            f"| `{row.case_id}` | `{row.experiment}` | `{row.variant}` | `{row.decision}` | "
            f"{row.physics_contact_frac * 100:.1f}% | {row.lowerbody_contact_frac * 100:.1f}% | "
            f"{row.nonhand_support_frac * 100:.1f}% | {row.hand_near_zero_frac * 100:.1f}% | "
            f"{row.pelvis_min_m:.3f}m | `{row.converter_input_npz}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: `FRAGMENT_HOLDOUT_ONLY` rows can be used to inspect downstream hand-support rewards, but they are not release/RL-ready rows. The main Box021 row remains blocked.",
        ]
    )
    (out_dir / "preflight_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-export", action="store_true", help="Only write metric tables; skip qpos43 export.")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    if not candidates:
        raise SystemExit("No E125 candidate rows found.")
    selected = select_rows(candidates)
    if not args.no_export:
        for cand in selected:
            export_converter_input(cand, out_dir)
    write_tsv(out_dir / "candidate_metrics.tsv", candidates)
    write_tsv(out_dir / "selected_preflight.tsv", selected)
    write_tsv(out_dir / "rl_objective_manifest.tsv", selected)
    write_summary(out_dir, candidates, selected)
    print(f"wrote {rel(out_dir / 'preflight_summary.md')} candidates={len(candidates)} selected={len(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
