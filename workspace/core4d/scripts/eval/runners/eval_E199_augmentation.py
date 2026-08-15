#!/usr/bin/env python3
"""E199 public-core evaluation: score each augmented full-CEM run and compare
every augmented variant against its same-case `orig` baseline (C4).

Self-contained: uses only the public eval.core API (evaluate_sequence, run_health,
EvalConfig). Reports the FULL distribution (mean+std+worst across all variants) --
no cherry-picking -- and per (object, aug_variant) orig-vs-aug deltas.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E199"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
)
from eval.core.motion_health import run_health  # noqa: E402
import e199_common as C  # noqa: E402

METHOD = "E167A_zOnlyBody"
HAND_VARIANT = "rubber_hull"

KEY_METRICS = (
    "track_obj_pos_err_cm_mean", "track_obj_ori_err_deg_mean", "track_obj_z_abs_err_cm_mean",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean",
    "track_eef_pos_err_cm_mean", "track_eef_ori_err_deg_mean", "body_z_err_p95_m",
    "hand_object_physics_contact_in_mask_frac", "hand_object_physics_penetration_3mm_frame_frac",
    "leg_penetration_frac", "fall_flag",
)
GATE_THRESHOLDS = {
    "object_pos": 20.0, "object_ori": 10.0, "contact": 0.50,
    "hand_penetration": 0.30, "lower_body": 0.10,
}


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def person_idx(case_id: str) -> int:
    return 0 if case_id.lower().endswith("_p1") else 1


def score(row: dict[str, str], cfg: EvalConfig) -> dict[str, Any]:
    scoring = dict(row)
    scoring.setdefault("spider_method_id", METHOD)
    scoring.setdefault("hand_collision_variant_id", HAND_VARIANT)
    qpos_path = C.repo_path(row["outdir_npz"])
    scene = C.repo_path(row["scene_act"])
    trajectory = C.repo_path(row["trajectory"])
    mask = C.repo_path(row["contact_mask"])
    for label, path in (("outdir_npz", qpos_path), ("scene_act", scene),
                        ("trajectory", trajectory), ("contact_mask", mask)):
        if not path.is_file():
            raise FileNotFoundError(f"{row['case_id']}:{label}:{path}")
    item = evaluate_sequence(row=scoring, method=METHOD, hand_collision_variant_id=HAND_VARIANT,
                             qpos_path=qpos_path, scene_xml=scene, config=cfg, kin_ref_path=trajectory,
                             contact_mask_path=mask, person_idx=person_idx(row["case_id"]))
    item.update(run_health(qpos_path, scene, cfg))
    gates = {
        "fall": not bool(item.get("fall_flag")),
        "object_pos": finite(item.get("track_obj_pos_err_cm_mean")) <= GATE_THRESHOLDS["object_pos"],
        "object_ori": finite(item.get("track_obj_ori_err_deg_mean")) <= GATE_THRESHOLDS["object_ori"],
        "contact": finite(item.get("hand_object_physics_contact_in_mask_frac")) >= GATE_THRESHOLDS["contact"],
        "hand_penetration": finite(item.get("hand_object_physics_penetration_3mm_frame_frac")) <= GATE_THRESHOLDS["hand_penetration"],
        "lower_body": finite(item.get("leg_penetration_frac")) <= GATE_THRESHOLDS["lower_body"],
    }
    for name, ok in gates.items():
        item[f"{name}_gate_pass"] = ok
    item["all_gates_pass"] = all(gates.values())
    item["numeric_failure_modes"] = ",".join(n for n, ok in gates.items() if not ok)
    item.update({
        "object_key": row["object_key"], "case_id": row["case_id"], "aug_variant": row["aug_variant"],
        "target_task": row["target_task"], "tier": row["tier"],
        "metric_standard_id": EVAL_METRIC_STANDARD_ID,
        "result_npz": C.rel(row["result_npz"]), "scene_act": C.rel(scene),
        "scene_sha256": C.sha256(row["scene_act"]), "outdir_sha256": C.sha256(row["outdir_npz"]),
    })
    return item


def deltas_vs_orig(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_variant = {(r["object_key"], r["aug_variant"]): r for r in rows}
    out: list[dict[str, Any]] = []
    for (obj, variant), item in sorted(by_case_variant.items()):
        if variant == "orig":
            continue
        base = by_case_variant.get((obj, "orig"))
        if base is None:
            continue
        row: dict[str, Any] = {"object_key": obj, "aug_variant": variant,
                               "orig_all_gates_pass": base.get("all_gates_pass", ""),
                               "aug_all_gates_pass": item.get("all_gates_pass", ""),
                               "aug_failure_modes": item.get("numeric_failure_modes", "")}
        for metric in KEY_METRICS:
            b, a = finite(base.get(metric)), finite(item.get(metric))
            row[f"orig_{metric}"] = b
            row[f"aug_{metric}"] = a
            row[f"delta_{metric}"] = a - b if math.isfinite(a) and math.isfinite(b) else math.nan
            row[f"pct_{metric}"] = (a - b) / b * 100.0 if math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-9 else math.nan
        out.append(row)
    return out


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    aug = [r for r in rows if r["aug_variant"] != "orig"]
    orig = [r for r in rows if r["aug_variant"] == "orig"]
    dist: dict[str, Any] = {"n_aug": len(aug), "n_orig": len(orig)}
    for group, tag in ((orig, "orig"), (aug, "aug")):
        for metric in KEY_METRICS:
            vals = np.array([finite(r.get(metric)) for r in group], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                dist[f"{tag}_{metric}_mean"] = float(vals.mean())
                dist[f"{tag}_{metric}_std"] = float(vals.std())
                dist[f"{tag}_{metric}_worst"] = float(vals.max())
    dist["aug_gate_pass_frac"] = float(np.mean([1.0 if r.get("all_gates_pass") else 0.0 for r in aug])) if aug else math.nan
    dist["orig_gate_pass_frac"] = float(np.mean([1.0 if r.get("all_gates_pass") else 0.0 for r in orig])) if orig else math.nan
    return dist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest = C.read_tsv(C.FULL_MANIFEST)
    cfg = EvalConfig()
    complete = [r for r in manifest
                if C.repo_path(r["outdir_npz"]).is_file() and C.repo_path(r["result_npz"]).is_file()]
    if args.require_all and len(complete) != len(manifest):
        raise SystemExit(f"complete rows={len(complete)} expected={len(manifest)}")

    scored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in complete:
        try:
            scored.append(score(row, cfg))
            print(f"[scored] {row['object_key']} {row['aug_variant']}")
        except Exception as exc:  # noqa: BLE001
            errors.append({"case_id": row["case_id"], "aug_variant": row["aug_variant"],
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {row['case_id']} {row['aug_variant']}: {errors[-1]['error']}", file=sys.stderr)

    out = C.RESULTS / "s6_downstream/eval/full_augmentation"
    deltas = deltas_vs_orig(scored)
    dist = distribution(scored)
    C.write_tsv(out / "e199_aug_case_metrics.tsv", scored)
    C.write_tsv(out / "e199_aug_orig_deltas.tsv", deltas)
    if errors:
        C.write_tsv(out / "e199_aug_eval_errors.tsv", errors)
    summary = {"created_at": C.now(), "metric_standard_id": EVAL_METRIC_STANDARD_ID,
               "scored": len(scored), "errors": len(errors), "paired_deltas": len(deltas),
               "distribution": dist,
               "status": "pass" if not errors and len(scored) == len(manifest) else "incomplete"}
    C.write_json(out / "summary.json", summary)
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.require_all and summary["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
