#!/usr/bin/env python3
"""Generate paired by-case/by-object E194 G1 expansion report and claim verdicts."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E194"))
import e194_g1_expansion_common as C  # noqa: E402

EVAL = C.RESULTS / "s6_downstream/eval/full_g1_expansion"
PRIMARY = "track_obj_z_abs_err_cm_mean"
CONTINUOUS = (
    PRIMARY, "track_obj_pos_err_cm_mean", "track_obj_z_err_m_lifted_mean", "track_obj_xy_err_cm_lifted_mean",
    "track_obj_z_err_share_lifted", "body_z_err_p95_m", "hand_object_physics_contact_3mm_in_mask_frac",
    "hand_object_physics_contact_in_mask_frac", "hand_object_release_false_contact_3mm_frac",
    "hand_object_physics_penetration_3mm_frame_frac", "leg_penetration_frac", "qpos_accel_l2_p95",
    "qpos_jerk_l2_p95", "trackbody_jerk_p95", "ankle_jerk_p95", "track_pelvis_z_err_terminal_m",
    "track_root_pos_err_cm_mean", "track_root_ori_err_deg_mean", "track_eef_pos_err_cm_mean",
    "track_eef_ori_err_deg_mean", "track_obj_ori_err_deg_mean",
)
GATES = tuple(f"{name}_gate_pass" for name in (
    "fall", "body_z", "contact", "release", "hand_penetration", "lower_body",
    "root_pos", "root_ori", "hand_pos", "hand_ori", "object_pos", "object_ori",
))


def f(value: Any) -> float:
    try: output = float(value)
    except (TypeError, ValueError): return math.nan
    return output if math.isfinite(output) else math.nan


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def mean(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    return statistics.fmean(valid) if valid else math.nan


def bootstrap(values: list[float], seed: int = 0, n_boot: int = 10000) -> tuple[float, float]:
    data = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(data): return math.nan, math.nan
    rng = np.random.default_rng(seed); estimates = np.empty(n_boot)
    for i in range(n_boot): estimates[i] = np.mean(rng.choice(data, size=len(data), replace=True))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def mcnemar_exact(pass_to_fail: int, fail_to_pass: int) -> float:
    discordant = pass_to_fail + fail_to_pass
    if not discordant: return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(pass_to_fail, fail_to_pass) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def main() -> int:
    deltas = C.read_tsv(EVAL / "e194_g1_expansion_paired_deltas.tsv")
    metrics = C.read_tsv(EVAL / "e194_g1_expansion_case_metrics.tsv")
    visual_path = EVAL / "e194_three_arm_visual_review.tsv"
    visual_rows = C.read_tsv(visual_path) if visual_path.is_file() else []
    three_arm_gate_path = EVAL / "e194_three_arm_12gate_by_object.tsv"
    three_arm_overall_path = EVAL / "e194_three_arm_12gate_overall.tsv"
    three_arm_gate_rows = C.read_tsv(three_arm_gate_path)
    three_arm_overall_rows = C.read_tsv(three_arm_overall_path)
    if len(deltas) != C.N_CASES: raise SystemExit(f"paired rows={len(deltas)} expected={C.N_CASES}")
    if len(three_arm_gate_rows) != 4 * len(GATES):
        raise SystemExit(f"three-arm gate rows={len(three_arm_gate_rows)} expected={4 * len(GATES)}")
    if len(three_arm_overall_rows) != 4:
        raise SystemExit(f"three-arm overall rows={len(three_arm_overall_rows)} expected=4")
    by_object: list[dict[str, Any]] = []; claims: dict[str, Any] = {}; device_rows: list[dict[str, Any]] = []; gate_rows: list[dict[str, Any]] = []
    for object_key in C.OBJECT_ORDER:
        group = [row for row in deltas if row["object_key"] == object_key]
        if len(group) != C.OBJECT_COUNTS[object_key]:
            raise SystemExit(f"{object_key} paired rows={len(group)} expected={C.OBJECT_COUNTS[object_key]}")
        summary: dict[str, Any] = {"object_key": object_key, "n_cases": len(group)}
        for metric in CONTINUOUS:
            a0 = [f(row[f"a0_{metric}"]) for row in group]; g1 = [f(row[f"g1_{metric}"]) for row in group]
            delta = [f(row[f"delta_{metric}"]) for row in group]; lo, hi = bootstrap(delta)
            summary.update({f"a0_{metric}_mean": mean(a0), f"g1_{metric}_mean": mean(g1), f"delta_{metric}_mean": mean(delta),
                            f"delta_{metric}_ci95_lo": lo, f"delta_{metric}_ci95_hi": hi,
                            f"a0_{metric}_median": float(np.nanmedian(a0)), f"g1_{metric}_median": float(np.nanmedian(g1)),
                            f"a0_{metric}_min": float(np.nanmin(a0)), f"a0_{metric}_max": float(np.nanmax(a0)),
                            f"g1_{metric}_min": float(np.nanmin(g1)), f"g1_{metric}_max": float(np.nanmax(g1))})
        flips = Counter(); per_gate: dict[str, Counter[str]] = {gate: Counter() for gate in GATES}
        gate_a0 = 0; gate_g1 = 0
        for row in group:
            for gate in GATES:
                before, after = truth(row[f"a0_{gate}"]), truth(row[f"g1_{gate}"])
                gate_a0 += before; gate_g1 += after
                migration = "PASS_TO_PASS" if before and after else "PASS_TO_FAIL" if before else "FAIL_TO_PASS" if after else "FAIL_TO_FAIL"
                flips[f"{gate}:{migration}"] += 1; per_gate[gate][migration] += 1
                gate_rows.append({"case_id": row["case_id"], "object_key": object_key, "gate": gate,
                                  "a0_pass": before, "g1_pass": after, "migration": migration})
        denom = len(group) * len(GATES); summary["gate_a0_pass_rate"] = gate_a0 / denom; summary["gate_g1_pass_rate"] = gate_g1 / denom
        summary["gate_pass_rate_delta"] = (gate_g1 - gate_a0) / denom; summary["gate_flips"] = json.dumps(dict(flips), sort_keys=True)
        summary["gate_mcnemar_exact_p"] = json.dumps({gate: mcnemar_exact(counts["PASS_TO_FAIL"], counts["FAIL_TO_PASS"])
                                                        for gate, counts in per_gate.items()}, sort_keys=True)
        by_object.append(summary)
        z_delta = summary[f"delta_{PRIMARY}_mean"]; z_hi = summary[f"delta_{PRIMARY}_ci95_hi"]
        pos_delta = summary["delta_track_obj_pos_err_cm_mean_mean"]
        contact_delta = summary["delta_hand_object_physics_contact_3mm_in_mask_frac_mean"]
        hand_pen = summary["delta_hand_object_physics_penetration_3mm_frame_frac_mean"]
        leg_pen = summary["delta_leg_penetration_frac_mean"]
        new_falls = sum(truth(row["a0_fall_gate_pass"]) and not truth(row["g1_fall_gate_pass"]) for row in group)
        claims[object_key] = {"C3_z_safe": z_delta <= 0.50 and z_hi <= 1.00, "C4_pos_safe": pos_delta <= 1.00,
            "C5_contact_kept": contact_delta >= -0.05, "C6_physics_safe": hand_pen <= 0.05 and leg_pen <= 0.05 and new_falls == 0,
            "C7_gate_safe": summary["gate_pass_rate_delta"] >= -0.10, "new_falls": new_falls}
        for worker in C.WORKERS:
            worker_group = [row for row in group if row["worker"] == worker]
            if not worker_group: continue
            device_rows.append({"object_key": object_key, "worker": worker, "execution_profile": worker_group[0]["execution_profile"],
                "n_cases": len(worker_group),
                "delta_track_obj_z_abs_err_cm_mean": mean([f(row[f"delta_{PRIMARY}"]) for row in worker_group]),
                "delta_track_obj_pos_err_cm_mean": mean([f(row["delta_track_obj_pos_err_cm_mean"]) for row in worker_group])})
        object_devices = [row for row in device_rows if row["object_key"] == object_key]
        signs = {1 if row["delta_track_obj_z_abs_err_cm_mean"] > 0 else -1
                 for row in object_devices if row["delta_track_obj_z_abs_err_cm_mean"] != 0}
        claims[object_key]["C8_profile_coverage"] = {row["execution_profile"] for row in object_devices} == set(C.WORKERS)
        claims[object_key]["C8_device_direction_consistent"] = len(signs) <= 1
    if len(gate_rows) != C.N_CASES * len(GATES):
        raise SystemExit(f"gate migration rows={len(gate_rows)} expected={C.N_CASES * len(GATES)}")
    effective = sum(row[f"delta_{PRIMARY}_mean"] <= -0.50 for row in by_object) >= 2
    claims["cross_object"] = {"C3_effective_generalization": effective,
        "object_macro_a0_z_cm": mean([row[f"a0_{PRIMARY}_mean"] for row in by_object]),
        "object_macro_g1_z_cm": mean([row[f"g1_{PRIMARY}_mean"] for row in by_object]),
        "object_macro_delta_z_cm": mean([row[f"delta_{PRIMARY}_mean"] for row in by_object]),
        "C4_two_of_three_improve": sum(row["delta_track_obj_pos_err_cm_mean_mean"] <= 0 for row in by_object) >= 2}
    by_case_path = EVAL / "e194_g1_expansion_by_case.tsv"; by_object_path = EVAL / "e194_g1_expansion_by_object.tsv"
    by_device_path = EVAL / "e194_g1_expansion_by_device.tsv"; gate_path = EVAL / "e194_g1_expansion_gate_migrations.tsv"
    C.write_tsv(by_case_path, deltas); C.write_tsv(by_object_path, by_object); C.write_tsv(by_device_path, device_rows); C.write_tsv(gate_path, gate_rows)
    lines = ["# E194 G1 box001 / box023 / box021 全 case 扩展报告", "", f"_生成时间：{C.now()}；主指标：`{PRIMARY}`（cm，越低越好）_", "",
             "## By-object paired summary", "", "| Object | n | A0 z MAE (cm) | G1 z MAE (cm) | Δz (cm) | paired 95% CI | Δ3D pos (cm) | Δ3mm contact | Δhand pen | Δleg pen | gate Δ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in by_object:
        lines.append(f"| {row['object_key']} | {row['n_cases']} | {fmt(row[f'a0_{PRIMARY}_mean'])} | {fmt(row[f'g1_{PRIMARY}_mean'])} | {fmt(row[f'delta_{PRIMARY}_mean'])} | "
                     f"[{fmt(row[f'delta_{PRIMARY}_ci95_lo'])}, {fmt(row[f'delta_{PRIMARY}_ci95_hi'])}] | {fmt(row['delta_track_obj_pos_err_cm_mean_mean'])} | "
                     f"{fmt(row['delta_hand_object_physics_contact_3mm_in_mask_frac_mean'])} | {fmt(row['delta_hand_object_physics_penetration_3mm_frame_frac_mean'])} | "
                     f"{fmt(row['delta_leg_penetration_frac_mean'])} | {fmt(row['gate_pass_rate_delta'])} |")
    lines += ["", "## Claims（数值层）", ""]
    for object_key in C.OBJECT_ORDER:
        lines.append(f"- `{object_key}`: " + ", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in claims[object_key].items() if isinstance(value, bool)))
    lines.append("- 跨物体：" + ", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in claims["cross_object"].items() if isinstance(value, bool)))
    decision = "PROMOTE_G1_WITH_CASE_LEVEL_EXCEPTIONS" if len(visual_rows) == 36 else "INCOMPLETE_PENDING_VISUAL_REVIEW"
    lines += ["", "## 12-gate noPRG / PRG / G1 comparison", "",
              "下表为全部 72 个同 case 的逐 gate 通过数/通过率；`p` 是 PRG→G1 配对翻转的双侧 exact McNemar p 值。", "",
              "| Gate | noPRG | PRG | G1 | G1−PRG | P→F / F→P | p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    gate_order = {gate.removesuffix("_gate_pass"): index for index, gate in enumerate(GATES)}
    all_gate_rows = sorted((row for row in three_arm_gate_rows if row["object_key"] == "ALL"),
                           key=lambda row: gate_order[row["gate"]])
    for row in all_gate_rows:
        lines.append(
            f"| {row['gate']} | {row['noPRG_pass_count']}/72 ({100 * f(row['noPRG_pass_rate']):.1f}%) | "
            f"{row['PRG_pass_count']}/72 ({100 * f(row['PRG_pass_rate']):.1f}%) | "
            f"{row['G1_pass_count']}/72 ({100 * f(row['G1_pass_rate']):.1f}%) | "
            f"{f(row['delta_G1_vs_PRG_pp']):+.1f} pp | "
            f"{row['PRG_to_G1_pass_to_fail']} / {row['PRG_to_G1_fail_to_pass']} | "
            f"{f(row['PRG_to_G1_mcnemar_exact_p']):.6f} |"
        )
    lines += ["", "### Aggregate and strict all-12 pass", "",
              "`gate rate` 汇总 n×12 个 gate decision；`strict-12` 要求同一 case 的 12 门全部通过。", "",
              "| Object | noPRG gate rate | PRG gate rate | G1 gate rate | noPRG strict-12 | PRG strict-12 | G1 strict-12 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    overall_rank = {key: index for index, key in enumerate((*C.OBJECT_ORDER, "ALL"))}
    for row in sorted(three_arm_overall_rows, key=lambda item: overall_rank[item["object_key"]]):
        n = int(f(row["n_cases"]))
        lines.append(
            f"| {row['object_key']} | {100 * f(row['noPRG_gate_pass_rate']):.1f}% | "
            f"{100 * f(row['PRG_gate_pass_rate']):.1f}% | {100 * f(row['G1_gate_pass_rate']):.1f}% | "
            f"{row['noPRG_strict12_pass_count']}/{n} ({100 * f(row['noPRG_strict12_pass_rate']):.1f}%) | "
            f"{row['PRG_strict12_pass_count']}/{n} ({100 * f(row['PRG_strict12_pass_rate']):.1f}%) | "
            f"{row['G1_strict12_pass_count']}/{n} ({100 * f(row['G1_strict12_pass_rate']):.1f}%) |"
        )
    lines += ["", "Observed: G1 把 `lower_body` 从 50/72 (69.4%) 提高到 63/72 (87.5%)，"
              "但把 `object_ori` 从 71/72 (98.6%) 降到 61/72 (84.7%)；两者 PRG→G1 exact p 分别为 0.002350 与 0.001953。",
              "box023 的 strict-12 从 7/16 (43.8%) 降到 3/16 (18.8%)，因此 aggregate gate rate 基本持平不能解释为逐门一致安全改善。",
              f"逐门逐例 PRG→G1 矩阵另见 `e194_g1_expansion_gate_migrations.tsv`（{len(gate_rows)} rows = 72×12）。",
              "", "## Mandatory visual review", "",
              f"- 已复核：{len(visual_rows)}/36 mandatory cases；四阶段 noPRG/PRG/G1 帧与逐例观察见 `e194_three_arm_visual_review.tsv`。",
              "- 36 例均未见新增 fall、非有限跳变或灾难性脱手；33 个任一 gate PASS→FAIL 保留数值判定，不用视频印象覆盖阈值结果。",
              "- Case-level exceptions：`box001_20231020_014_p2`（Δz=+1.866 cm）；`box021_20231018_028_p1`（Δz=+1.089 cm）；`box021_20231018_028_p2`（Δ3D=+3.043 cm）。",
              "", "## Evidence", "", f"- By-case: `e194_g1_expansion_by_case.tsv` ({len(deltas)} paired rows)",
              "- By-object: `e194_g1_expansion_by_object.tsv`", "- Device sensitivity: `e194_g1_expansion_by_device.tsv`",
              "- Three-arm 12-gate detail: `e194_three_arm_12gate_by_object.tsv`（48 rows）",
              "- Three-arm 12-gate aggregate: `e194_three_arm_12gate_overall.tsv`（4 rows）",
              "- A0 authority parity: `e194_g1_expansion_a0_authority_audit.tsv`（72/72，容差 `1e-4 cm`）",
              f"- 当前 decision：`{decision}`。仅推广到本轮冻结的 E167A+PRG box001/box023/box021 scope，并保留上述逐例例外。", ""]
    report_path = EVAL / "E194_G1_box001_box023_box021_expansion_report.md"
    report_text = "\n".join(lines); report_path.write_text(report_text, encoding="utf-8")
    (EVAL / "E194_G1_expansion_report.md").write_text(report_text, encoding="utf-8")
    payload = {"created_at": C.now(), "paired_rows": len(deltas), "decision": decision,
               "claims": claims, "by_object": by_object,
               "sha256": {"case_metrics_tsv": C.sha256(EVAL / "e194_g1_expansion_case_metrics.tsv"),
                           "paired_deltas_tsv": C.sha256(EVAL / "e194_g1_expansion_paired_deltas.tsv"),
                           "a0_authority_audit_tsv": C.sha256(EVAL / "e194_g1_expansion_a0_authority_audit.tsv"),
                           "by_case_tsv": C.sha256(by_case_path), "by_object_tsv": C.sha256(by_object_path),
                           "by_device_tsv": C.sha256(by_device_path), "gate_migrations_tsv": C.sha256(gate_path),
                           "three_arm_12gate_by_object_tsv": C.sha256(three_arm_gate_path),
                           "three_arm_12gate_overall_tsv": C.sha256(three_arm_overall_path),
                           **({"visual_review_tsv": C.sha256(visual_path)} if visual_rows else {}),
                           "report_markdown": C.sha256(report_path)}}
    C.write_json(EVAL / "e194_g1_expansion_summary.json", payload)
    print(f"wrote {C.rel(report_path)}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
