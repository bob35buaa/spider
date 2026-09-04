#!/usr/bin/env python3
"""E209 object z-height diagnostic: E206 PRG vs E209 G1 (PRG + object gravcomp).

This is the diagnostic that motivated E209 (plan239). The per-frame metric is
imported from gen_E178_object_z_diff_report.py so E178, E207 and E209 are all
measured by exactly the same code (rule 13):

    z_diff(t) = z_sim(t) - z_ref(t)      cm, + = retargeted object above reference
    z_bias    = mean_t(z_diff)           <- what gravcomp is expected to fix
    z_mae     = mean_t(|z_diff|)         <- total tracking error, NOT the target

Beyond re-running E207's C4, this report settles a question E207 could not:
**is gravcomp's correction a shrink toward zero, or an additive offset?**

E207 reported "+1.945 cm constant", but in its 9 cases object mass is collinear
with pre-bias, so the two model families are indistinguishable there. All 22
E209 cases weigh 5.000 kg while pre-bias spans -5.6 .. +2.7, which separates
them. The three candidate transfer functions were frozen into
``e209_common.PREREG_MODELS`` before the run; this script only scores them.

Usage:
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/eval/reports/gen_E209_object_z_diff.py
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
for _p in (
    "workspace/core4d/scripts",
    "workspace/core4d/scripts/eval/reports",
    "workspace/core4d/scripts/experiments/E209",
):
    _s = str(REPO / _p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from eval.core.core_metrics import npz_qpos  # noqa: E402
from gen_E178_object_z_diff_report import object_z_series  # noqa: E402

import e209_common as C  # noqa: E402

ARMS = ("PRG", "G1")
ARM_NOTE = {
    "PRG": "E206 baseline — A0 hand-gate, no gravcomp (reused, not re-run)",
    "G1": "E209 — A0 hand-gate + object gravcomp (this experiment)",
}

CASE_FIELDS = [
    "arm", "case_id", "object_key", "stratum", "frames",
    "z_bias_cm", "z_mae_cm", "z_rmse_cm", "z_abs_p95_cm", "z_abs_max_cm",
    "z_abs_gt5_frac", "obj_pos_err_3d_cm", "ref_z_range_m", "object_mass_kg",
]


def arm_paths(arm: str, row: dict[str, str]) -> tuple[Path, Path]:
    case_id = row["case_id"]
    if arm == "G1":
        return C.result_npz(case_id), C.scene_path(row)
    return C.baseline_npz(case_id), C.base_scene_path(row)


def measure(arm: str, row: dict[str, str]) -> dict[str, Any]:
    case_id = row["case_id"]
    qpos_path, scene_path = arm_paths(arm, row)
    trajectory = C.kinematic_npz(row)
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
        "object_key": row["object_key"],
        "stratum": "S+" if case_id in C.S_PLUS else "S-",
        "frames": int(len(dz)),
        "z_bias_cm": float(np.mean(dz)),
        "z_mae_cm": float(np.mean(np.abs(dz))),
        "z_rmse_cm": float(np.sqrt(np.mean(dz**2))),
        "z_abs_p95_cm": float(np.percentile(np.abs(dz), 95)),
        "z_abs_max_cm": float(np.max(np.abs(dz))),
        "z_abs_gt5_frac": float(np.mean(np.abs(dz) > 5.0)),
        "obj_pos_err_3d_cm": float(np.mean(err_3d) * 100.0),
        "ref_z_range_m": float(np.max(z_ref) - np.min(z_ref)),
        "object_mass_kg": C.object_mass(scene_path),
    }


def agg(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
    g = [r for r in rows if r["arm"] == arm]
    return {
        "n": len(g),
        "z_bias_cm": statistics.fmean(r["z_bias_cm"] for r in g),
        "mean_abs_bias_cm": statistics.fmean(abs(r["z_bias_cm"]) for r in g),
        "z_mae_cm": statistics.fmean(r["z_mae_cm"] for r in g),
        "z_rmse_cm": statistics.fmean(r["z_rmse_cm"] for r in g),
        "z_abs_max_cm": max(r["z_abs_max_cm"] for r in g),
        "obj_pos_err_3d_cm": statistics.fmean(r["obj_pos_err_3d_cm"] for r in g),
    }


def score_models(pre: dict[str, float], post: dict[str, float]) -> dict[str, dict[str, float]]:
    """RMSE of each pre-registered transfer function against the measured post-bias."""
    out: dict[str, dict[str, float]] = {}
    cases = sorted(post)
    for name in C.PREREG_MODELS:
        resid = np.array([C.predict(name, pre[c]) - post[c] for c in cases])
        out[name] = {
            "rmse_cm": float(np.sqrt(np.mean(resid**2))),
            "mae_cm": float(np.mean(np.abs(resid))),
            "bias_cm": float(np.mean(resid)),
            "max_abs_cm": float(np.max(np.abs(resid))),
        }
    return out


def evaluate_claims(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["arm"], r["case_id"]): r for r in rows}
    cases = [c for c in C.CASES if ("G1", c) in by and ("PRG", c) in by]
    pre = {c: by[("PRG", c)]["z_bias_cm"] for c in cases}
    post = {c: by[("G1", c)]["z_bias_cm"] for c in cases}

    def macro(subset: list[str], src: dict[str, float]) -> float:
        return float(np.mean([src[c] for c in subset])) if subset else float("nan")

    sm = [c for c in cases if c in C.S_MINUS]
    sp = [c for c in cases if c in C.S_PLUS]
    g = C.GATES

    # C4 -- S- stratum
    sm_macro = macro(sm, post)
    sm_improved = sum(abs(post[c]) < abs(pre[c]) for c in sm)
    sm_overshoot = [c for c in sm if abs(post[c]) > g["C4_s_minus_overshoot_abs_cm"]]
    c4 = {
        "n": len(sm),
        "macro_bias_cm": sm_macro,
        "baseline_macro_bias_cm": macro(sm, pre),
        "improved": sm_improved,
        "overshoot_cases": sm_overshoot,
        "sub_magnitude": abs(sm_macro) <= g["C4_s_minus_abs_macro_bias_max_cm"],
        "sub_improved": sm_improved >= g["C4_s_minus_improved_min"],
        "sub_overshoot": len(sm_overshoot) <= g["C4_s_minus_overshoot_max_cases"],
        "stretch_E207_gate_0p8": abs(sm_macro) <= 0.8,
    }
    c4["pass"] = c4["sub_magnitude"] and c4["sub_improved"] and c4["sub_overshoot"]

    # C4b -- S+ stratum (harm ceiling)
    sp_macro = macro(sp, post)
    sp_worst = max((abs(post[c]) for c in sp), default=float("nan"))
    c4b = {
        "n": len(sp),
        "macro_bias_cm": sp_macro,
        "baseline_macro_bias_cm": macro(sp, pre),
        "worst_abs_cm": sp_worst,
        "sub_macro": abs(sp_macro) <= g["C4b_s_plus_abs_macro_bias_max_cm"],
        "sub_case": sp_worst <= g["C4b_s_plus_abs_case_max_cm"],
    }
    c4b["pass"] = c4b["sub_macro"] and c4b["sub_case"]

    # C4c -- all 22, anti-cherry-pick
    all_macro = macro(cases, post)
    all_mean_abs = float(np.mean([abs(post[c]) for c in cases]))
    try:
        from scipy import stats

        wil = stats.wilcoxon(
            [abs(pre[c]) for c in cases], [abs(post[c]) for c in cases], alternative="greater"
        )
        pvalue = float(wil.pvalue)
    except Exception as exc:  # pragma: no cover - scipy always present here
        pvalue = float("nan")
        print(f"  (wilcoxon unavailable: {exc})")
    c4c = {
        "n": len(cases),
        "macro_bias_cm": all_macro,
        "mean_abs_bias_cm": all_mean_abs,
        "baseline_macro_bias_cm": macro(cases, pre),
        "baseline_mean_abs_bias_cm": float(np.mean([abs(pre[c]) for c in cases])),
        "improved": sum(abs(post[c]) < abs(pre[c]) for c in cases),
        "wilcoxon_p_abs_bias_decreased": pvalue,
        "sub_macro": abs(all_macro) < g["C4c_all_abs_macro_bias_max_cm"],
        "sub_mean_abs": all_mean_abs < g["C4c_all_mean_abs_bias_max_cm"],
        "sub_wilcoxon": pvalue < 0.05,
    }
    c4c["pass"] = c4c["sub_macro"] and c4c["sub_mean_abs"] and c4c["sub_wilcoxon"]

    # C4d -- transfer-function discrimination
    scores = score_models(pre, post)
    ranked = sorted(scores.items(), key=lambda kv: kv[1]["rmse_cm"])
    winner, runner = ranked[0], ranked[1]
    ratio = runner[1]["rmse_cm"] / winner[1]["rmse_cm"] if winner[1]["rmse_cm"] > 0 else float("inf")
    c4d = {
        "scores": scores,
        "winner": winner[0],
        "winner_rmse_cm": winner[1]["rmse_cm"],
        "runner_up": runner[0],
        "runner_up_rmse_cm": runner[1]["rmse_cm"],
        "rmse_ratio": ratio,
        "sub_ratio": ratio >= g["C4d_rmse_win_ratio_min"],
        "sub_rmse": winner[1]["rmse_cm"] <= g["C4d_rmse_max_cm"],
    }
    c4d["pass"] = c4d["sub_ratio"] and c4d["sub_rmse"]

    # Empirical delta -- the number E207 reported as "+1.945 constant"
    deltas = np.array([post[c] - pre[c] for c in cases])
    pre_arr = np.array([pre[c] for c in cases])
    fit_slope, fit_int = (np.polyfit(pre_arr, [post[c] for c in cases], 1) if len(cases) > 2 else (float("nan"),) * 2)
    return {
        "cases": cases,
        "C4_s_minus": c4,
        "C4b_s_plus": c4b,
        "C4c_all": c4c,
        "C4d_model": c4d,
        "empirical": {
            "delta_mean_cm": float(deltas.mean()),
            "delta_sd_cm": float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan"),
            "r_pre_vs_delta": float(np.corrcoef(pre_arr, deltas)[0, 1]) if len(cases) > 2 else float("nan"),
            "observed_fit_slope": float(fit_slope),
            "observed_fit_intercept": float(fit_int),
        },
    }


def render(rows: list[dict[str, Any]], claims: dict[str, Any], generated_at: str) -> str:
    by = {(r["arm"], r["case_id"]): r for r in rows}
    cases = claims["cases"]
    summary = {a: agg(rows, a) for a in ARMS}
    c4, c4b, c4c, c4d = (
        claims["C4_s_minus"], claims["C4b_s_plus"], claims["C4c_all"], claims["C4d_model"]
    )
    emp = claims["empirical"]
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731

    lines = [
        "# E209 物体 z 高度：G1（PRG + object gravcomp）vs E206 PRG",
        "",
        f"_{len(cases)} 例 desk/chair 交付 case；全部 object mass = 5.000 kg；生成于 {generated_at}_",
        "",
        "## 📐 指标",
        "",
        "```text",
        "z_diff(t) = z_sim(t) - z_ref(t)     # cm，正 = 重定向物体高于参考",
        "z_bias    = mean_t(z_diff)          # gravcomp 针对的系统性下沉",
        "z_mae     = mean_t(|z_diff|)        # 总跟踪误差，非本次干预目标",
        "```",
        "",
        "两臂共用同一条参考轨迹与同一段公共帧窗，逐 case 帧均值后等权宏平均。",
        "指标函数与 E178/E207 同源（`gen_E178_object_z_diff_report.object_z_series`）。",
        "",
        "## 📊 两臂汇总",
        "",
        "| Arm | 说明 | n | z bias (cm) | mean\\|bias\\| | z MAE (cm) | 最差单帧 (cm) | 3D pos (cm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        s = summary[arm]
        lines.append(
            f"| **{arm}** | {ARM_NOTE[arm]} | {s['n']} | {s['z_bias_cm']:+.3f} | "
            f"{s['mean_abs_bias_cm']:.3f} | {s['z_mae_cm']:.3f} | {s['z_abs_max_cm']:.3f} | "
            f"{s['obj_pos_err_3d_cm']:.3f} |"
        )

    lines += [
        "",
        "## 🎯 C4 / C4b / C4c 判定（分层在 P0 由基线数据冻结，非事后划线）",
        "",
        f"### C4 · S⁻ 子群（baseline bias < 0，n={c4['n']}）— **{ok(c4['pass'])}**",
        "",
        f"- z bias 宏平均：**{c4['baseline_macro_bias_cm']:+.3f} → {c4['macro_bias_cm']:+.3f} cm**",
        f"- 门① \\|宏 bias\\| ≤ {C.GATES['C4_s_minus_abs_macro_bias_max_cm']} cm → "
        f"{abs(c4['macro_bias_cm']):.3f}，**{ok(c4['sub_magnitude'])}**"
        f"（E207 的 0.8 stretch 线：{ok(c4['stretch_E207_gate_0p8'])}）",
        f"- 门② ≥{C.GATES['C4_s_minus_improved_min']}/{c4['n']} 改善 → "
        f"**{c4['improved']}/{c4['n']}**，{ok(c4['sub_improved'])}",
        f"- 门③ \\|post\\|>{C.GATES['C4_s_minus_overshoot_abs_cm']}cm 的 case ≤"
        f"{C.GATES['C4_s_minus_overshoot_max_cases']} → {len(c4['overshoot_cases'])}，"
        f"{ok(c4['sub_overshoot'])}"
        + (f"（{', '.join('`'+x+'`' for x in c4['overshoot_cases'])}）" if c4["overshoot_cases"] else ""),
        "",
        f"### C4b · S⁺ 子群（baseline bias ≥ 0，n={c4b['n']}，全为 chair006）— **{ok(c4b['pass'])}**",
        "",
        "> 这是**可失败的预注册硬门**，不是注脚。gravcomp 对本来就偏高的 case 必然继续推高；",
        "> C4b 限定的是「危害上限」而非「改善」。若 FAIL，结论须写成"
        "「gravcomp 对基线 bias≥0 的 case 禁用」。",
        "",
        f"- z bias 宏平均：**{c4b['baseline_macro_bias_cm']:+.3f} → {c4b['macro_bias_cm']:+.3f} cm**",
        f"- 门① \\|宏 bias\\| ≤ {C.GATES['C4b_s_plus_abs_macro_bias_max_cm']} cm → "
        f"{abs(c4b['macro_bias_cm']):.3f}，{ok(c4b['sub_macro'])}",
        f"- 门② 无单例 \\|post\\| > {C.GATES['C4b_s_plus_abs_case_max_cm']} cm → "
        f"最差 {c4b['worst_abs_cm']:.3f}，{ok(c4b['sub_case'])}",
        "",
        f"### C4c · 全 {c4c['n']} 例反挑拣（强制报告）— **{ok(c4c['pass'])}**",
        "",
        f"- z bias 宏平均：**{c4c['baseline_macro_bias_cm']:+.3f} → {c4c['macro_bias_cm']:+.3f} cm**"
        f"（门 < {C.GATES['C4c_all_abs_macro_bias_max_cm']}，{ok(c4c['sub_macro'])}）",
        f"- mean\\|bias\\|：**{c4c['baseline_mean_abs_bias_cm']:.3f} → {c4c['mean_abs_bias_cm']:.3f}**"
        f"（门 < {C.GATES['C4c_all_mean_abs_bias_max_cm']}，{ok(c4c['sub_mean_abs'])}）",
        f"- 改善 case：**{c4c['improved']}/{c4c['n']}**",
        f"- Wilcoxon 配对符号秩（\\|bias\\| 下降，单尾）：p = {c4c['wilcoxon_p_abs_bias_decreased']:.5f}"
        f"（门 < 0.05，{ok(c4c['sub_wilcoxon'])}）",
        "",
        "## 🔬 C4d · 传递函数判别（E209 的核心科学产出）— **" + ok(c4d["pass"]) + "**",
        "",
        "三个候选模型的系数在跑之前就冻结进 `e209_common.PREREG_MODELS`，此处只算 RMSE。",
        "E207 的 9 例里物体质量与 pre-bias 共线，无法区分「收缩」与「加性」；",
        "E209 全部 22 例 mass=5.000 kg 而 pre-bias 跨 −5.6…+2.7，可以。",
        "",
        "| 模型 | 形式 | RMSE (cm) | MAE (cm) | 残差偏置 (cm) | 最差 (cm) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    forms = {
        "M_shrink": "post = 1.0939 + 0.3827·pre（E207 OLS，收缩型）",
        "M_pooled": "post = pre + 1.945（E207 池化常量，加性型）",
        "M_mass": "post = pre + 2.836（E207 mass=5 单元，加性型）",
    }
    for name, s in sorted(c4d["scores"].items(), key=lambda kv: kv[1]["rmse_cm"]):
        mark = " ⬅ **胜**" if name == c4d["winner"] else ""
        lines.append(
            f"| `{name}`{mark} | {forms[name]} | **{s['rmse_cm']:.3f}** | {s['mae_cm']:.3f} | "
            f"{s['bias_cm']:+.3f} | {s['max_abs_cm']:.3f} |"
        )
    lines += [
        "",
        f"- 胜者 **{c4d['winner']}** RMSE {c4d['winner_rmse_cm']:.3f} cm；次者 {c4d['runner_up']} "
        f"{c4d['runner_up_rmse_cm']:.3f} cm；比值 **{c4d['rmse_ratio']:.2f}×**",
        f"- 门 ①比值 ≥ {C.GATES['C4d_rmse_win_ratio_min']}× → {ok(c4d['sub_ratio'])}；"
        f"②胜者 RMSE ≤ {C.GATES['C4d_rmse_max_cm']} cm → {ok(c4d['sub_rmse'])}",
        "",
        "**实测传递函数**（事后拟合，仅供对照，不参与判定）：",
        "",
        f"- delta = post − pre：均值 **{emp['delta_mean_cm']:+.3f} ± {emp['delta_sd_cm']:.3f} cm**"
        f"（E207 池化值 +1.945 ± 0.711；E207 mass=5 单元 +2.836 ± 0.550）",
        f"- r(pre, delta) = **{emp['r_pre_vs_delta']:+.3f}**"
        "（加性型预测 ≈ 0；收缩型预测显著为负）",
        f"- OLS：post = {emp['observed_fit_intercept']:+.4f} + {emp['observed_fit_slope']:.4f}·pre"
        "（加性型预测 slope ≈ 1；收缩型预测 slope < 1）",
        "",
        "## 📋 逐 case",
        "",
        "| Case | 层 | 参考抬升 (m) | PRG bias | G1 bias | Δ | \\|bias\\| 改善 | PRG MAE | G1 MAE |",
        "| --- | :-: | ---: | ---: | ---: | ---: | :-: | ---: | ---: |",
    ]
    for case_id in cases:
        p, q = by[("PRG", case_id)], by[("G1", case_id)]
        better = "✅" if abs(q["z_bias_cm"]) < abs(p["z_bias_cm"]) else "❌"
        lines.append(
            f"| `{case_id}` | {p['stratum']} | {p['ref_z_range_m']:.3f} | {p['z_bias_cm']:+.3f} | "
            f"{q['z_bias_cm']:+.3f} | {q['z_bias_cm'] - p['z_bias_cm']:+.3f} | {better} | "
            f"{p['z_mae_cm']:.3f} | {q['z_mae_cm']:.3f} |"
        )

    lines += [
        "",
        "## ⚠️ 口径",
        "",
        "- 本报告为诊断；主判定见 14-gate（`eval_E209_g1_gravcomp.py`）。",
        "- S⁺/S⁻ 分层由 `e209_common.BASELINE_Z_BIAS_CM`（P0 冻结、随代码提交）机械决定，"
        "**不按物体名切**：`chair006_20231003_2_015_p1`(−0.726) 属 S⁻。",
        "- z **MAE** 的降幅显式不作判据（E207 已证 gravcomp 修 bias 不修 MAE）。",
        "- gravcomp 使物体在 CEM 优化中失重；下游 RL/真机在真实重力下工作，"
        "该建模落差沿用 plan237 的声明式接受。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out-dir", type=Path, default=C.S6_DIR / "eval/two_arm")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows: list[dict[str, Any]] = []
    for row in C.sources():
        for arm in arms:
            rows.append(measure(arm, row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.out_dir / "e209_object_z_diff_by_case.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=CASE_FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    generated_at = C.sources()[0].get("updated_at", "")
    claims = evaluate_claims(rows) if set(arms) >= {"PRG", "G1"} else {}
    if claims:
        (args.out_dir / "e209_object_z_diff_summary.json").write_text(
            json.dumps(
                {"arms": {a: agg(rows, a) for a in arms}, "claims": claims},
                indent=2,
                ensure_ascii=False,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )
        md = args.out_dir / "E209_object_z_diff_report.md"
        md.write_text(render(rows, claims, generated_at) + "\n", encoding="utf-8")
        c4, c4b, c4c, c4d = (
            claims["C4_s_minus"], claims["C4b_s_plus"], claims["C4c_all"], claims["C4d_model"]
        )
        print(f"\n{len(rows)} rows -> {tsv.relative_to(REPO)}")
        print(f"report -> {md.relative_to(REPO)}\n")
        print(f"  C4  (S-, n={c4['n']:2d}) {c4['baseline_macro_bias_cm']:+.3f} -> "
              f"{c4['macro_bias_cm']:+.3f} cm, improved {c4['improved']}/{c4['n']}  "
              f"{'PASS' if c4['pass'] else 'FAIL'}")
        print(f"  C4b (S+, n={c4b['n']:2d}) {c4b['baseline_macro_bias_cm']:+.3f} -> "
              f"{c4b['macro_bias_cm']:+.3f} cm, worst {c4b['worst_abs_cm']:.3f}  "
              f"{'PASS' if c4b['pass'] else 'FAIL'}")
        print(f"  C4c (all,n={c4c['n']:2d}) {c4c['baseline_macro_bias_cm']:+.3f} -> "
              f"{c4c['macro_bias_cm']:+.3f} cm, p={c4c['wilcoxon_p_abs_bias_decreased']:.5f}  "
              f"{'PASS' if c4c['pass'] else 'FAIL'}")
        print(f"  C4d winner={c4d['winner']} rmse={c4d['winner_rmse_cm']:.3f} vs "
              f"{c4d['runner_up']} {c4d['runner_up_rmse_cm']:.3f} ({c4d['rmse_ratio']:.2f}x)  "
              f"{'PASS' if c4d['pass'] else 'FAIL'}")
    else:
        print(f"{len(rows)} rows -> {tsv.relative_to(REPO)} (single arm, no claims evaluated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
