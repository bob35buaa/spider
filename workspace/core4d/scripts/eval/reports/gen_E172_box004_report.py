#!/usr/bin/env python3
"""E172: generate the refined Markdown main report + machine JSON.

Synthesizes the pipeline funnel, Stage2b v1/v2 rescue provenance, PRG-scene
contract outcomes, and the Full-CEM numeric evaluation into a user-facing
Markdown report (plan 8.3 questions) plus a machine-readable JSON summary.
Reads only committed E172 evidence; does not write manual_* user fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E172"))
import e172_common as C

E106_OVERLAP = {  # E106 ran 30 box026 candidates; historical reference only (diff scene/method)
    "note": "E106 used a different scene (pre-rubber-hull) and method; comparison is same-case historical reference, NOT a strict paired A/B.",
}


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=C.RESULTS / "s6_downstream/eval/full/E172_report.md")
    args = ap.parse_args()
    R = C.RESULTS
    funnel = json.loads((R / "registries/pipeline_funnel.json").read_text())
    audit = json.loads((R / "completion_audit/completion_audit.json").read_text())
    esummary = json.loads((R / "s6_downstream/eval/full/summary.json").read_text())
    metrics = C.read_tsv(R / "s6_downstream/eval/full/e171_case_metrics.tsv")
    authority = C.read_tsv(R / "registries/pipeline_authority.tsv")

    numeric_pass = [m for m in metrics if m.get("numeric_release_pass") in ("True", "true", True)]
    by_obj_pass = Counter(m["case_id"].split("_")[0] for m in numeric_pass)

    L = []
    L.append("# E172 结果报告：Box004 全流程筛选 + Full CEM（move-only）")
    L.append("")
    L.append("_Core4D Phase 35 · rubber_hull + E170 PRG cross-object candidate · machine recommendation = `PENDING_USER_REVIEW`_")
    L.append("")
    L.append("> 本报告为机器核验 + 视觉复核交付物；最终 `USE/DO_NOT_USE` 由用户裁决。E172 沿用 E170/E171 冻结算法，move-only，不做 reward/route sweep，不导出 RL；science yield 与执行完成解耦。")
    L.append("")
    # Q1 funnel
    L.append("## 1. 20 条 raw row 的终态分布（funnel）")
    L.append("")
    L.append(md_table(["阶段", "计数"], [
        ["raw expected / seen", f"{funnel['raw_expected']} / {funnel['raw_seen']}"],
        ["box004 (single object)", funnel['by_object'].get('box004', 0)],
        ["move eligible / 非首选 Stage0 reject", "14 / 6 (pass2×4 + strike×2)"],
        ["raw-contact pass (3cm=5cm)", funnel["raw_contact_pass"]],
        ["Stage2b v1 pass", funnel["stage2b_v1_pass"]],
        ["Stage2b v1 infeasible", funnel["stage2b_v1_infeasible"]],
        ["v2 rescued pass", funnel["stage2b_v2_rescued_pass"]],
        ["dual-infeasible", funnel["dual_infeasible"]],
        ["target gate pass", funnel["target_gate_pass"]],
        ["visual QC pass", funnel["visual_qc_pass"]],
        ["PRG scene-contract reject", funnel["prg_scene_contract_reject"]],
        ["**CEM eligible (S5_READY)**", f"**{funnel['cem_eligible']}**"],
    ]))
    L.append("")
    L.append("终态闭合：" + " + ".join(f"{k}={v}" for k, v in funnel["terminal_states"].items()) +
             f" = {sum(funnel['terminal_states'].values())}（raw_closure_pass；其中 REJECT_RAW_CONTACT=14 含 6 条 Stage0 非首选动作 + 8 条 move raw-contact fail）")
    L.append("")
    # Q2 per-object
    L.append("## 2. 产出汇总")
    L.append("")
    L.append(md_table(["对象", "raw", "move eligible", "S5-ready(CEM)", "Full complete", "numeric pass"], [
        ["box004", funnel['by_object'].get('box004', 0), 14, funnel["cem_eligible"], audit["full_completed"], len(numeric_pass)],
    ]))
    L.append("")
    L.append(f"- box004：{funnel['cem_eligible']} 条进入 Full CEM，{audit['full_completed']} 条完成，**{len(numeric_pass)} 条 numeric pass**（6 S5-ready，PRG scene contract 0 reject）。")
    L.append("")
    # Q3 usable/not
    L.append("## 3. 各 case 可用性（numeric 层，最终以用户裁决为准）")
    L.append("")
    rows = []
    for m in metrics:
        rows.append([m["case_id"], m.get("selected_retarget_variant_id", "") or m.get("retarget_variant_id", ""),
                     "✅" if m in numeric_pass else "❌",
                     m.get("numeric_failure_modes", "") or "-",
                     "pass" if m.get("leg_gate_health_pass") in ("True", "true") else "fail"])
    L.append(md_table(["case", "variant", "numeric", "failure modes", "gate-health"], rows))
    L.append("")
    L.append(f"- numeric failure 分布：{esummary.get('numeric_failure_counts', {})}")
    L.append(f"- gate-health pass：{esummary['recommendation']['gate_health_pass']}/{len(metrics)}（与 E170 0/28、E171 0/12 一致，candidate gate 已知弱项，与质量判定分列）")
    L.append("")
    # Q4 history overlap
    L.append("## 4. 历史 overlap（仅参考，非因果 A/B）")
    L.append("")
    L.append("box004_082_p1 曾有 E167A `RL_EXPORT_READY`；082_p2 历史 v1 infeasible/missing。E172 fresh 重跑：082_p1 v1 numeric pass；082_p2 v1 infeasible → **v2 rescue pass 且 numeric pass**（v2 在 box004 产生真实下游 yield，区别于 E171 box026 v2 的 0 净产出）。E091/E167 旧 scene/CEM 不计入 E172 completion。")
    L.append("")
    # Q5 v1/v2
    L.append("## 5. v1→v2 rescue")
    L.append("")
    L.append(f"- v1 infeasible：{funnel['stage2b_v1_infeasible']}（box004_082_p2，命中历史 prior）")
    L.append(f"- v2 救回 pass：{funnel['stage2b_v2_rescued_pass']}；dual-infeasible：{funnel['dual_infeasible']}")
    L.append(f"- v2-rescued 082_p2 进入 Full CEM 且 **numeric pass + 视觉干净** → 有效下游产出。")
    L.append("")
    # Q6 PRG
    L.append("## 6. PRG 跨物体是否成立 + contact trade-off")
    L.append("")
    L.append(f"- **lower-body 定向**：PRG 16 lower-body/object pair + soft penalty 在 {len(numeric_pass)}/6 条 numeric-pass（含 v2）上成立，首帧 lower-body/object 距离 0.07–0.17m，PRG scene contract 0 reject（优于 E171 box026 的 5 reject）。")
    L.append("- **contact/lower-body trade-off**：唯一 fail 086_p2 为深蹲跨箱低位姿，Full CEM 中 fall + leg_penetration 0.27 + release 0.53 + body_z 0.46，是 PRG 下肢/接触 trade-off 的数据+视觉体现。")
    L.append(f"- **gate-health 0/{len(metrics)}**：candidate gate 未达 full validity（082_p2/086_p2 用 fallback），属已知弱项。")
    L.append(f"- 结论倾向 **PARTIAL_YIELD**（box004 {len(numeric_pass)}/6 case-level positive），不足以将 PRG 升级为跨物体统一默认；最终以用户裁决为准。")
    L.append("")
    L.append("## 7. 复现锚点")
    L.append("")
    L.append(f"- funnel：`registries/pipeline_funnel.json`；authority：`registries/pipeline_authority.tsv`")
    L.append(f"- completion audit：`completion_audit/completion_audit.json`（{audit['status']}）")
    L.append(f"- metrics：`s6_downstream/eval/full/e171_case_metrics.tsv`（内容为 E172）；summary：`.../summary.json`")
    L.append(f"- CEM manifest：`s6_downstream/manifests/cem_full_manifest.tsv`；scenes：`scene_snapshot/cem_sidecars/`")
    L.append("")

    out = C.repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    machine = {
        "created_at": C.now(), "machine_recommendation": "PENDING_USER_REVIEW",
        "funnel": funnel, "numeric_pass": len(numeric_pass), "by_object_numeric_pass": dict(by_obj_pass),
        "completion_audit_status": audit["status"], "yield_class_hint": "PARTIAL_YIELD_pending_user",
    }
    C.write_json(out.with_suffix(".json"), machine)
    print(json.dumps(machine, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
