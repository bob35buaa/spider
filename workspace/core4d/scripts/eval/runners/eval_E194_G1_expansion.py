#!/usr/bin/env python3
"""Paired public-core evaluation for the 72-case E194 G1 expansion."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))

from eval.core.core_metrics import (  # noqa: E402
    EVAL_METRIC_STANDARD_ID,
    EvalConfig,
    evaluate_sequence,
    npz_qpos,
)
from eval.core.motion_health import fps_from_npz, run_health  # noqa: E402
import e194_g1_expansion_common as C  # noqa: E402

KEY_METRICS = (
    "track_obj_z_abs_err_cm_mean", "track_obj_pos_err_cm_mean", "track_obj_z_err_m_lifted_mean",
    "track_obj_xy_err_cm_lifted_mean", "track_obj_z_err_share_lifted", "body_z_err_p95_m",
    "hand_object_physics_contact_3mm_in_mask_frac", "hand_object_physics_contact_in_mask_frac",
    "hand_object_release_false_contact_3mm_frac", "hand_object_physics_penetration_3mm_frame_frac",
    "leg_penetration_frac", "fall_flag", "qpos_accel_l2_p95", "qpos_jerk_l2_p95",
    "trackbody_jerk_p95", "ankle_jerk_p95", "track_pelvis_z_err_terminal_m",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean", "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean", "track_obj_ori_err_deg_mean",
)
PHYSICS_GATES = (
    "fall", "body_z", "contact", "release", "hand_penetration", "lower_body",
)
TRACKING_GATES = (
    "root_pos", "root_ori", "hand_pos", "hand_ori", "object_pos", "object_ori",
)
ALL_GATES = PHYSICS_GATES + TRACKING_GATES
GATE_FIELDS = tuple(f"{gate}_gate_pass" for gate in ALL_GATES) + (
    "legacy_physics6_pass", "numeric_release_pass_12gate", "numeric_release_pass", "numeric_failure_modes",
)
GATE_THRESHOLDS = {
    "body_z": 0.20,
    "contact": 0.50,
    "release": 0.30,
    "hand_penetration": 0.30,
    "lower_body": 0.10,
    "root_pos": 20.0,
    "root_ori": 20.0,
    "hand_pos": 20.0,
    "hand_ori": 20.0,
    "object_pos": 20.0,
    "object_ori": 10.0,
}
MONITORED_BODIES = (
    "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)
LEG_GATE_KEYS = (
    "cem_leg_gate_valid_frac", "cem_leg_gate_selected_valid_frac", "cem_leg_gate_fallback_used",
    "cem_leg_gate_min_sdf_min_m", "cem_leg_gate_min_sdf_p05_m", "cem_leg_gate_violation_pct_mean",
    "cem_leg_gate_selected_all_valid",
)


def person_idx(row: dict[str, str]) -> int:
    person = str(row.get("source_person", row.get("person_idx", ""))).strip().lower()
    if person in {"person1", "p1", "0"}: return 0
    if person in {"person2", "p2", "1"}: return 1
    return 0 if row["case_id"].lower().endswith("_p1") else 1


def p95(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.percentile(data, 95)) if data.size else math.nan


def fixed_reference_z_metrics(qpos_path: Path, scene: Path, trajectory: Path) -> dict[str, Any]:
    sim_qpos, intra_tick_qpos = npz_qpos(qpos_path)
    with np.load(trajectory, allow_pickle=True) as archive:
        ref_qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if ref_qpos.ndim == 3: ref_qpos = ref_qpos[:, 0, :]
    model = mujoco.MjModel.from_xml_path(str(scene)); nq_robot = min(36, model.nq, ref_qpos.shape[1])
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MONITORED_BODIES]
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError(f"missing monitored body in {scene}: {body_ids}")

    def positions(qpos: np.ndarray, reference: bool = False) -> np.ndarray:
        data = mujoco.MjData(model); output = np.zeros((len(qpos), len(body_ids), 3), dtype=np.float64)
        for index, frame in enumerate(qpos):
            if reference and frame.shape[0] != model.nq:
                data.qpos[:] = sim_qpos[min(index, len(sim_qpos) - 1)]
                data.qpos[:nq_robot] = frame[:nq_robot]
            else:
                data.qpos[:] = frame[:model.nq]
            data.qvel[:] = 0.0; mujoco.mj_forward(model, data); output[index] = data.xpos[body_ids]
        return output

    frames = min(len(sim_qpos), len(ref_qpos)); sim_pos = positions(sim_qpos[:frames]); ref_pos = positions(ref_qpos[:frames], True)
    diff = sim_pos - ref_pos; z_err = np.abs(diff[..., 2]); xy_err = np.linalg.norm(diff[..., :2], axis=-1)
    err_3d = np.linalg.norm(diff, axis=-1); fps = fps_from_npz(qpos_path, 50.0)
    z_accel = np.diff(sim_pos[..., 2], n=2, axis=0) * fps**2 if frames >= 3 else np.empty((0,))
    z_jerk = np.diff(sim_pos[..., 2], n=3, axis=0) * fps**3 if frames >= 4 else np.empty((0,))
    legacy_peak = math.nan
    if intra_tick_qpos is not None:
        legacy_frames = min(len(sim_qpos), len(intra_tick_qpos)); legacy_ref = positions(intra_tick_qpos[:legacy_frames])
        legacy_peak = float(np.max(np.abs(sim_pos[:legacy_frames, :, 2] - legacy_ref[..., 2])))
    return {
        "z_reference_source": "fixed_kinematic_trajectory", "z_reference_path": C.rel(trajectory),
        "z_eval_frames": frames, "monitored_bodies": ",".join(MONITORED_BODIES),
        "body_z_gate_metric": "body_z_err_p95_m", "body_z_gate_threshold_m": 0.20,
        "body_z_err_peak_m": float(np.max(z_err)), "body_z_err_p95_m": p95(z_err),
        "body_z_over_frac": float(np.mean(z_err > 0.20)), "holosoma_z_gate_pass": p95(z_err) <= 0.20,
        "sugar_3d_err_peak_m": float(np.max(err_3d)), "sugar_3d_err_p95_m": p95(err_3d),
        "sugar_3d_over_frac": float(np.mean(err_3d > 0.30)), "sugar_3d_gate_pass": float(np.max(err_3d)) <= 0.30,
        "xy_err_peak_m": float(np.max(xy_err)), "xy_err_p95_m": p95(xy_err),
        "xy_only_failure": float(np.max(err_3d)) > 0.30 and p95(z_err) <= 0.20,
        "body_z_accel_p95": p95(np.abs(z_accel)), "body_z_jerk_p95": p95(np.abs(z_jerk)),
        "legacy_intra_tick_z_err_peak_m": legacy_peak,
        "legacy_intra_tick_z_gate_pass": math.isfinite(legacy_peak) and legacy_peak <= 0.25,
    }


def release_window_info(mask_path: Path, selected_person: int, frame_count: int) -> dict[str, Any]:
    with np.load(mask_path, allow_pickle=True) as archive:
        if "spider_contact_mask_3cm" not in archive.files:
            return {"release_window_frame_count": 0, "release_gate_applicable": False,
                    "release_gate_status": "NOT_APPLICABLE_MASK_MISSING_KEY"}
        mask = np.asarray(archive["spider_contact_mask_3cm"])
    if mask.ndim != 3 or mask.shape[2] != 2 or not 0 <= selected_person < mask.shape[1]:
        raise ValueError(f"invalid contact mask/person: {mask_path} shape={mask.shape} person={selected_person}")
    frames = min(frame_count, mask.shape[0]); active = np.any(mask[:frames, selected_person].astype(bool), axis=1)
    if not active.any():
        return {"release_window_frame_count": 0, "release_gate_applicable": False,
                "release_gate_status": "NOT_APPLICABLE_NO_REFERENCE_CONTACT"}
    release_frames = frames - int(np.flatnonzero(active)[-1]) - 1
    return {"release_window_frame_count": release_frames, "release_gate_applicable": release_frames > 0,
            "release_gate_status": "PENDING_EVALUATION" if release_frames > 0 else "NOT_APPLICABLE_NO_RELEASE_WINDOW"}


def array_stat(archive: np.lib.npyio.NpzFile, key: str, mode: str) -> float:
    if key not in archive.files: return math.nan
    values = np.asarray(archive[key], dtype=np.float64)
    if mode == "last":
        if values.ndim != 2 or not values.shape[1]: return math.nan
        values = values[:, -1]
    values = values[np.isfinite(values)]
    return float(values.min() if mode == "min" else values.mean()) if values.size else math.nan


def gate_health(path: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as archive:
        for key in LEG_GATE_KEYS:
            output[f"{key}_mean"] = array_stat(archive, key, "mean")
            output[f"{key}_last_iter_mean"] = array_stat(archive, key, "last")
        output["cem_leg_gate_min_sdf_worst_m"] = array_stat(archive, "cem_leg_gate_min_sdf_min_m", "min")
    output["leg_gate_fallback_pass"] = finite(output["cem_leg_gate_fallback_used_mean"]) <= 0.10
    output["leg_gate_valid_frac_pass"] = finite(output["cem_leg_gate_valid_frac_last_iter_mean"]) >= 0.05
    output["leg_gate_selected_valid_pass"] = finite(output["cem_leg_gate_selected_all_valid_mean"]) >= 1.0 - 1e-9
    output["leg_gate_health_pass"] = all(output[key] for key in ("leg_gate_fallback_pass", "leg_gate_valid_frac_pass", "leg_gate_selected_valid_pass"))
    return output


def apply_12gates(item: dict[str, Any]) -> None:
    release_applicable = bool(item.get("release_gate_applicable"))
    gates = {
        "fall": not bool(item.get("fall_flag")),
        "body_z": finite(item.get("body_z_err_p95_m")) <= GATE_THRESHOLDS["body_z"],
        "contact": finite(item.get("hand_object_physics_contact_in_mask_frac")) >= GATE_THRESHOLDS["contact"],
        "release": (not release_applicable) or finite(item.get("hand_object_release_false_contact_3mm_frac")) <= GATE_THRESHOLDS["release"],
        "hand_penetration": finite(item.get("hand_object_physics_penetration_3mm_frame_frac")) <= GATE_THRESHOLDS["hand_penetration"],
        "lower_body": finite(item.get("leg_penetration_frac")) <= GATE_THRESHOLDS["lower_body"],
        "root_pos": finite(item.get("track_root_pos_err_cm_mean")) <= GATE_THRESHOLDS["root_pos"],
        "root_ori": finite(item.get("track_root_ori_err_deg_mean")) <= GATE_THRESHOLDS["root_ori"],
        "hand_pos": finite(item.get("track_eef_pos_err_cm_mean")) <= GATE_THRESHOLDS["hand_pos"],
        "hand_ori": finite(item.get("track_eef_ori_err_deg_mean")) <= GATE_THRESHOLDS["hand_ori"],
        "object_pos": finite(item.get("track_obj_pos_err_cm_mean")) <= GATE_THRESHOLDS["object_pos"],
        "object_ori": finite(item.get("track_obj_ori_err_deg_mean")) <= GATE_THRESHOLDS["object_ori"],
    }
    for name, passed in gates.items(): item[f"{name}_gate_pass"] = passed
    item["legacy_physics6_pass"] = all(gates[name] for name in PHYSICS_GATES)
    item["numeric_release_pass_12gate"] = all(gates.values()); item["numeric_release_pass"] = all(gates.values())
    item["numeric_failure_modes"] = ",".join(name for name in ALL_GATES if not gates[name])
    item["scoring_contract_id"] = "core4d-e194-g1-expansion-12gate-v1"; item["tracking_gates_enabled"] = True


def finite(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return math.nan
    return output if math.isfinite(output) else math.nan


def score(row: dict[str, str], arm: str, cfg: EvalConfig) -> dict[str, Any]:
    scoring = dict(row)
    scoring.setdefault("spider_method_id", "E167A_zOnlyBody_PRG_gravcomp_G1" if arm == "G1" else "E167A_zOnlyBody")
    scoring.setdefault("hand_collision_variant_id", "rubber_hull")
    qpos_path = C.repo_path(row["outdir_npz"]); result_path = C.repo_path(row["result_npz"])
    scene = C.repo_path(row["scene_act"]); trajectory = C.repo_path(row["trajectory"])
    mask = C.repo_path(row["contact_mask"]); config_act = C.repo_path(row["config_act"])
    for label, path in (("outdir_npz", qpos_path), ("result_npz", result_path), ("scene_act", scene),
                        ("trajectory", trajectory), ("contact_mask", mask), ("config_act", config_act)):
        if not path.is_file(): raise FileNotFoundError(f"{row['case_id']}:{label}:{path}")
    selected_person = person_idx(row)
    item = evaluate_sequence(row=scoring, method=scoring["spider_method_id"],
                             hand_collision_variant_id=scoring["hand_collision_variant_id"],
                             qpos_path=qpos_path, scene_xml=scene, config=cfg, kin_ref_path=trajectory,
                             contact_mask_path=mask, person_idx=selected_person)
    item.update(run_health(qpos_path, scene, cfg)); item.update(fixed_reference_z_metrics(qpos_path, scene, trajectory))
    item.update(release_window_info(mask, selected_person, len(npz_qpos(qpos_path)[0]))); item.update(gate_health(result_path))
    apply_12gates(item)
    item.update({"arm": arm, "case_id": row["case_id"], "object_key": row["object_key"],
                 "source_exp": row.get("source_exp", ""), "execution_source": row.get("execution_source", ""),
                 "reused_full": row.get("reused_full", "false"), "retarget_variant_id": row.get("retarget_variant_id", ""),
                 "worker": row.get("worker", "A0-source"), "execution_profile": row.get("execution_profile", "A0-source"),
                 "result_sha256": C.sha256(row["result_npz"]), "outdir_sha256": C.sha256(row["outdir_npz"]),
                 "scene_sha256": C.sha256(row["scene_act"]), "trajectory_sha256": C.sha256(row["trajectory"]),
                 "metric_standard_id": EVAL_METRIC_STANDARD_ID, "result_npz": C.rel(result_path),
                 "outdir_npz": C.rel(qpos_path), "config_act": C.rel(config_act), "scene_xml": C.rel(scene),
                 "trajectory": C.rel(trajectory), "contact_mask": C.rel(mask)})
    return item


def paired(a0: dict[str, dict[str, Any]], g1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in g1:
        base = a0[item["case_id"]]
        row: dict[str, Any] = {"case_id": item["case_id"], "object_key": item["object_key"],
            "source_exp": item["source_exp"], "execution_source": item["execution_source"], "reused_full": item["reused_full"],
            "retarget_variant_id": item["retarget_variant_id"], "worker": item["worker"], "execution_profile": item["execution_profile"]}
        for metric in KEY_METRICS:
            before, after = finite(base.get(metric)), finite(item.get(metric))
            row[f"a0_{metric}"] = before; row[f"g1_{metric}"] = after
            row[f"delta_{metric}"] = after - before if math.isfinite(before) and math.isfinite(after) else math.nan
        for gate in GATE_FIELDS:
            row[f"a0_{gate}"] = base.get(gate, ""); row[f"g1_{gate}"] = item.get(gate, "")
        row.update({"a0_result_sha256": base["result_sha256"], "g1_result_sha256": item["result_sha256"],
                    "a0_scene_sha256": base["scene_sha256"], "g1_scene_sha256": item["scene_sha256"]})
        output.append(row)
    return output


def a0_authority_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    frozen = {row["case_id"]: row for row in C.read_tsv(C.A0_METRICS)}; output = []
    for row in rows:
        expected = finite(frozen.get(row["case_id"], {}).get("track_obj_z_abs_err_cm_mean"))
        actual = finite(row.get("track_obj_z_abs_err_cm_mean")); difference = abs(actual - expected)
        output.append({"case_id": row["case_id"], "object_key": row["object_key"], "expected_z_cm": expected,
                       "recomputed_z_cm": actual, "abs_diff_cm": difference,
                       "status": "pass" if math.isfinite(difference) and difference <= 1e-4 else "fail"})
    complete = len(rows) == len(frozen) == C.N_CASES and len({row["case_id"] for row in rows}) == C.N_CASES
    return output, complete and all(row["status"] == "pass" for row in output)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", nargs="?", default="full", choices=("full",))
    parser.add_argument("--require-all", action="store_true"); args = parser.parse_args()
    full = C.read_tsv(C.FULL_MANIFEST); sources = C.source_rows(); cfg = EvalConfig()
    complete = [row for row in full if C.repo_path(row["outdir_npz"]).is_file() and C.repo_path(row["result_npz"]).is_file()]
    if args.require_all and len(complete) != C.N_CASES:
        raise SystemExit(f"G1 complete rows={len(complete)} expected={C.N_CASES}")
    a0_rows: list[dict[str, Any]] = []; g1_rows: list[dict[str, Any]] = []; errors: list[dict[str, Any]] = []
    for arm, rows in (("A0", sources), ("G1", complete)):
        for row in rows:
            try:
                item = score(row, arm, cfg); (a0_rows if arm == "A0" else g1_rows).append(item)
                z_error = finite(item.get("track_obj_z_abs_err_cm_mean")); pos_error = finite(item.get("track_obj_pos_err_cm_mean"))
                if not math.isfinite(z_error) or not math.isfinite(pos_error) or z_error > pos_error + 1e-9:
                    errors.append({"arm": arm, "case_id": row["case_id"],
                                   "error": f"metric_contract:z_cm={z_error}:pos_cm={pos_error}"})
                print(f"[scored] {arm} {row['case_id']}")
            except Exception as exc:  # noqa: BLE001
                errors.append({"arm": arm, "case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[error] {arm} {row['case_id']}: {errors[-1]['error']}", file=sys.stderr)
    a0 = {row["case_id"]: row for row in a0_rows}; deltas = paired(a0, g1_rows) if set(a0) >= {row["case_id"] for row in g1_rows} else []
    out = C.RESULTS / "s6_downstream/eval/full_g1_expansion"
    a0_audit, a0_audit_pass = a0_authority_audit(a0_rows)
    combined = a0_rows + g1_rows; C.write_tsv(out / "e194_g1_expansion_case_metrics.tsv", combined)
    C.write_tsv(out / "e194_g1_expansion_paired_deltas.tsv", deltas); C.write_tsv(out / "e194_g1_expansion_eval_errors.tsv", errors)
    C.write_tsv(out / "e194_g1_expansion_a0_authority_audit.tsv", a0_audit)
    summary = {"created_at": C.now(), "metric_standard_id": EVAL_METRIC_STANDARD_ID, "a0_scored": len(a0_rows),
               "g1_scored": len(g1_rows), "paired": len(deltas), "errors": len(errors),
               "a0_authority_tolerance_cm": 1e-4, "a0_authority_pass": a0_audit_pass,
               "status": "pass" if len(a0_rows) == len(g1_rows) == len(deltas) == C.N_CASES and not errors and a0_audit_pass else "incomplete"}
    C.write_json(out / "summary.json", summary); print(summary)
    return 1 if args.require_all and summary["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
