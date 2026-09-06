#!/usr/bin/env python3
"""E207 object z-height diagnostic: PRG (E178) vs G1only (E207) vs G1A2 (E205).

This is the diagnostic that motivated E207 (see plan237): E178's object sags
below the kinematic reference on 24/27 bucket cases, with the shortfall scaling
with how high the reference lifts. Object ``gravcomp`` should remove the
``sag = m*g/kp`` component of that.

The per-frame metric is imported from gen_E178_object_z_diff_report.py so all
three arms are measured by exactly the same code (rule 13):

    z_diff(t) = z_sim(t) - z_ref(t)      cm, + = retargeted object above reference
    z_bias    = mean_t(z_diff)           <- what gravcomp is expected to fix
    z_mae     = mean_t(|z_diff|)         <- total tracking error, NOT the target

``z_ref`` is the same fixed omnirt_v1 ref_fk kinematic trajectory for every arm,
so the arms are directly comparable.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/eval/reports/gen_E207_object_z_diff.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[5]
for _p in ("workspace/core4d/scripts",
           "workspace/core4d/scripts/eval/reports",
           "workspace/core4d/scripts/experiments/E204_E205",
           "workspace/core4d/scripts/experiments/E207"):
    sys.path.insert(0, str(REPO / _p))

from eval.core.core_metrics import npz_qpos  # noqa: E402
from gen_E178_object_z_diff_report import object_z_series, sha256  # noqa: E402
import e204e205_common as C205  # noqa: E402
import e207_common as C207  # noqa: E402

ARMS = ("PRG", "G1only", "G1A2")
ARM_NOTE = {
    "PRG": "E178 baseline — A0 hand-gate, no gravcomp",
    "G1only": "E207 — A0 hand-gate + object gravcomp (this experiment)",
    "G1A2": "E205 — A2 hand-gate + object gravcomp",
}
E178_MANIFEST = (
    REPO / "workspace/core4d/results/E178/s6_downstream/eval/full/evaluated_manifest_snapshot.tsv"
)


def arm_paths(arm: str, case_id: str) -> tuple[Path, Path]:
    if arm == "G1only":
        return C207.result_npz(case_id, "full"), C207.scene_path(case_id)
    if arm == "G1A2":
        return C205.result_npz("g1a2_e205", case_id, "full"), C205.arm_scene_path("g1a2_e205", case_id)
    row = _e178_row(case_id)
    return REPO / row["outdir_npz"], REPO / row["scene_act"]


_E178: dict[str, dict[str, str]] | None = None


def _e178_row(case_id: str) -> dict[str, str]:
    global _E178
    if _E178 is None:
        with E178_MANIFEST.open(encoding="utf-8", newline="") as stream:
            _E178 = {r["case_id"]: r for r in csv.DictReader(stream, delimiter="\t")}
    return _E178[case_id]


def measure(arm: str, case_id: str) -> dict[str, Any]:
    qpos_path, scene_path = arm_paths(arm, case_id)
    trajectory = REPO / _e178_row(case_id)["trajectory"]
    for label, path in (("rollout", qpos_path), ("scene", scene_path), ("ref", trajectory)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{case_id}:{arm}:{label}:{path}")
    run, _ = npz_qpos(qpos_path)
    kin = np.asarray(np.load(trajectory, allow_pickle=True)["qpos"], dtype=np.float64)
    if kin.ndim == 3:
        kin = kin[:, 0, :]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    z_sim, z_ref, err_3d = object_z_series(run, kin, model)
    dz = (z_sim - z_ref) * 100.0
    return {
        "arm": arm,
        "case_id": case_id,
        "frames": int(len(dz)),
        "z_bias_cm": float(np.mean(dz)),
        "z_mae_cm": float(np.mean(np.abs(dz))),
        "z_rmse_cm": float(np.sqrt(np.mean(dz**2))),
        "z_abs_p95_cm": float(np.percentile(np.abs(dz), 95)),
        "z_abs_max_cm": float(np.max(np.abs(dz))),
        "z_abs_gt5_frac": float(np.mean(np.abs(dz) > 5.0)),
        "obj_pos_err_3d_cm": float(np.mean(err_3d) * 100.0),
        "ref_z_range_m": float(np.max(z_ref) - np.min(z_ref)),
    }


def agg(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    group = [r for r in rows if r["arm"] == arm]
    return {
        "n": len(group),
        "z_bias_cm": statistics.fmean(r["z_bias_cm"] for r in group),
        "z_mae_cm": statistics.fmean(r["z_mae_cm"] for r in group),
        "z_rmse_cm": statistics.fmean(r["z_rmse_cm"] for r in group),
        "z_abs_max_cm": max(r["z_abs_max_cm"] for r in group),
        "z_abs_gt5_frac": statistics.fmean(r["z_abs_gt5_frac"] for r in group),
        "obj_pos_err_3d_cm": statistics.fmean(r["obj_pos_err_3d_cm"] for r in group),
    }


def render(rows: list[dict[str, Any]], arms: list[str], generated_at: str) -> str:
    by = {(r["arm"], r["case_id"]): r for r in rows}
    cases = [c for c in C207.CASES if ("PRG", c) in by]
    summary = {a: agg(rows, a) for a in arms}
    base, cand = summary.get("PRG"), summary.get("G1only")

    lines = [
        "# E207 物体 z 高度：G1only vs E178 PRG vs E205 G1A2",
        "",
        f"_{len(cases)} 个 bucket case（bucket003=3 / bucket007=6，**无 bucket004**）；生成于 {generated_at}_",
        "",
        "## 📐 指标",
        "",
        "```text",
        "z_diff(t) = z_sim(t) - z_ref(t)     # cm，正 = 重定向物体高于参考",
        "z_bias    = mean_t(z_diff)          # gravcomp 针对的系统性下沉",
        "z_mae     = mean_t(|z_diff|)        # 总跟踪误差，非本次干预目标",
        "```",
        "",
        "三臂共用同一条 omnirt_v1 ref_fk 参考轨迹与同一段公共帧窗，逐 case 帧均值后等权宏平均。",
        "",
        "## 📊 三臂汇总",
        "",
        "| Arm | 说明 | n | z bias (cm) | z MAE (cm) | z RMSE (cm) | 最差单帧 (cm) | >5cm 帧占比 | 3D pos (cm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        s = summary[arm]
        lines.append(
            f"| **{arm}** | {ARM_NOTE[arm]} | {s['n']} | {s['z_bias_cm']:+.3f} | {s['z_mae_cm']:.3f} | "
            f"{s['z_rmse_cm']:.3f} | {s['z_abs_max_cm']:.3f} | {100*s['z_abs_gt5_frac']:.2f}% | "
            f"{s['obj_pos_err_3d_cm']:.3f} |"
        )

    if base and cand:
        improved = sum(
            abs(by[("G1only", c)]["z_bias_cm"]) < abs(by[("PRG", c)]["z_bias_cm"]) for c in cases
        )
        mae_better = sum(by[("G1only", c)]["z_mae_cm"] < by[("PRG", c)]["z_mae_cm"] for c in cases)
        lines += [
            "",
            "## 🎯 C4 判定（gravcomp 是否消除欠抬升）",
            "",
            f"- z bias 宏平均：**{base['z_bias_cm']:+.3f} → {cand['z_bias_cm']:+.3f} cm**"
            f"（|bias| {abs(base['z_bias_cm']):.3f} → {abs(cand['z_bias_cm']):.3f}）",
            f"- |bias| 下降的 case：**{improved}/{len(cases)}**",
            f"- C4 门：|z bias 宏平均| ≤ 0.8 cm 且 ≥7/9 改善 → "
            f"**{'PASS' if abs(cand['z_bias_cm']) <= 0.8 and improved >= 7 else 'FAIL'}**",
            "",
            f"- （诊断，不作判据）z MAE {base['z_mae_cm']:.3f} → {cand['z_mae_cm']:.3f} cm，"
            f"{mae_better}/{len(cases)} 个 case 改善",
        ]

    lines += [
        "",
        "## 📋 逐 case",
        "",
        "| Case | 参考抬升 (m) | " + " | ".join(f"{a} bias" for a in arms) + " | "
        + " | ".join(f"{a} MAE" for a in arms) + " |",
        "| --- | ---: |" + " ---: |" * (2 * len(arms)),
    ]
    for case_id in cases:
        lift = by[("PRG", case_id)]["ref_z_range_m"]
        bias = " | ".join(f"{by[(a, case_id)]['z_bias_cm']:+.3f}" for a in arms)
        mae = " | ".join(f"{by[(a, case_id)]['z_mae_cm']:.3f}" for a in arms)
        lines.append(f"| `{case_id}` | {lift:.3f} | {bias} | {mae} |")

    lines += [
        "",
        "## ⚠️ 口径",
        "",
        "- 本报告为诊断，主判定见四臂 14-gate（`eval_E207_g1only.py`）。",
        "- 9 case **不含 bucket004**（抬升最高、E178 下欠抬升最重的物体），结论不外推到它。",
        "- gravcomp 使物体在 CEM 优化中失重；下游 RL/真机在真实重力下工作，该建模落差已在 plan237 声明接受。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out-dir", type=Path, default=C207.RESULTS / "s6_downstream/eval/four_arm")
    ap.add_argument("--generated-at", default="2026-09-04")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a.strip()]

    C207.audit(verbose=False)
    rows = [measure(arm, case_id) for case_id in C207.CASES for arm in arms]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case_path = args.out_dir / "e207_object_z_diff_by_case.tsv"
    fields = list(rows[0])
    with case_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.out_dir / "E207_object_z_diff_report.md"
    report_path.write_text(render(rows, arms, args.generated_at), encoding="utf-8")

    summary_path = args.out_dir / "e207_object_z_diff_summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": "E207",
        "metric": "object_world_z_diff_vs_kinematic_reference",
        "unit": "cm",
        "sign_convention": "positive = retargeted object higher than reference",
        "arms": {a: ARM_NOTE[a] for a in arms},
        "cases": list(C207.CASES),
        "by_arm": {a: agg(rows, a) for a in arms},
        "generated_at": args.generated_at,
        "generator": "workspace/core4d/scripts/eval/reports/gen_E207_object_z_diff.py",
        "generator_sha256": sha256(Path(__file__)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for arm in arms:
        s = agg(rows, arm)
        print(f"  {arm:7s} n={s['n']}  bias={s['z_bias_cm']:+.3f}  mae={s['z_mae_cm']:.3f}  "
              f"3d={s['obj_pos_err_3d_cm']:.3f}")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
