#!/usr/bin/env python3
"""E174: generate the refined Markdown main report + machine JSON.

Synthesizes the pipeline funnel, Stage2b v1/v2 rescue provenance, PRG-scene
contract outcomes, and the Full-CEM numeric evaluation into a user-facing
Markdown report (plan 8.3 questions) plus a machine-readable JSON summary.
Reads only committed E174 evidence; does not write manual_* user fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E174"))
import e174_common as C

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
    ap.add_argument("--out", type=Path, default=C.RESULTS / "s6_downstream/eval/full/E174_report.md")
    args = ap.parse_args()
    R = C.RESULTS
    funnel = json.loads((R / "registries/pipeline_funnel.json").read_text())
    audit = json.loads((R / "completion_audit/completion_audit.json").read_text())
    esummary = json.loads((R / "s6_downstream/eval/full/summary.json").read_text())
    metrics = C.read_tsv(R / "s6_downstream/eval/full/e174_case_metrics.tsv")
    authority = C.read_tsv(R / "registries/pipeline_authority.tsv")

    numeric_pass = [m for m in metrics if m.get("numeric_release_pass") in ("True", "true", True)]
    by_obj_pass = Counter(m["case_id"].split("_")[0] for m in numeric_pass)
    OBJS = C.OBJECT_KEYS
    CATS = {o: C.OBJECT_CATEGORY.get(o, "") for o in OBJS}

    def cat_pass(cat: str) -> int:
        return sum(by_obj_pass.get(o, 0) for o in OBJS if CATS[o] == cat)

    L = []
    L.append("# E174 结果报告：bucket / desk 非 box 物体 move2 全流程筛选 + Full CEM")
    L.append("")
    L.append("_Core4D Phase 37 · rubber_hull(手) + 物体侧 proxy(bucket_wall/desk_surface_voxel) + E170 PRG · machine recommendation = `PENDING_USER_REVIEW`_")
    L.append("")
    L.append("> 本报告为机器核验 + 视觉复核交付物；最终 `USE/DO_NOT_USE` 由用户裁决。E174 沿用 E170–E173 冻结算法，**move2-only**，首次离开 box 拓扑（5 bucket 0.045–0.19m³ 空心 + 2 desk 0.24m³ 桌腿）。物体凹几何由 object-side collision_policy proxy 表达（bucket_wall / desk_surface_voxel），与 rubber_hull（手侧）正交；不做 reward/route/geometry sweep，不导出 RL；science yield 与执行完成解耦。")
    L.append("")
    # Q1 funnel
    L.append("## 1. raw row 的终态分布（funnel）")
    L.append("")
    L.append(md_table(["阶段", "计数"], [
        ["raw expected / seen", f"{funnel['raw_expected']} / {funnel['raw_seen']}"],
        ["by object (raw)", ", ".join(f"{o}={funnel['by_object'].get(o, 0)}" for o in OBJS)],
        ["raw-contact pass (move2-only, 3cm)", funnel["raw_contact_pass"]],
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
             f" = {sum(funnel['terminal_states'].values())}（raw_closure_pass={funnel.get('raw_closure_pass')}）")
    L.append("")
    # Q2 per-object
    L.append("## 2. 逐物体产出汇总")
    L.append("")
    bo = funnel["by_object"]
    cem_by = funnel.get("cem_eligible_by_object", {})
    rows2 = []
    for o in OBJS:
        rows2.append([o, bo.get(o, 0), cem_by.get(o, ""), by_obj_pass.get(o, 0)])
    rows2.append(["**合计**", sum(bo.get(o, 0) for o in OBJS), funnel["cem_eligible"], len(numeric_pass)])
    L.append(md_table(["对象", "raw pc", "S5-ready(CEM)", "numeric pass"], rows2))
    L.append("")
    L.append(f"- 分组 numeric pass：bucket={cat_pass('bucket')}、desk={cat_pass('desk')}。")
    L.append(f"- Full CEM：{funnel['cem_eligible']} 条 S5-ready，{audit['full_completed']} 条完成，**{len(numeric_pass)} 条 numeric pass**（PRG scene-contract reject={funnel['prg_scene_contract_reject']}）。")
    L.append("")
    # Q3 usable/not
    L.append("## 3. 各 case 可用性（numeric 层，最终以用户裁决为准）")
    L.append("")
    rows = []
    for m in sorted(metrics, key=lambda x: x["case_id"]):
        rows.append([m["case_id"], m.get("selected_retarget_variant_id", "") or m.get("retarget_variant_id", ""),
                     "✅" if m in numeric_pass else "❌",
                     m.get("numeric_failure_modes", "") or "-",
                     "pass" if m.get("leg_gate_health_pass") in ("True", "true") else "fail"])
    L.append(md_table(["case", "variant", "numeric", "failure modes", "gate-health"], rows))
    L.append("")
    L.append(f"- numeric failure 分布：{esummary.get('numeric_failure_counts', {})}")
    L.append(f"- gate-health pass：{esummary['recommendation']['gate_health_pass']}/{len(metrics)}（与 E170 0/28、E171 0/12、E172 0/6 一致，candidate gate 已知弱项，与质量判定分列）")
    L.append("")
    # Q5 v1/v2
    L.append("## 4. v1→v2 rescue")
    L.append("")
    L.append(f"- v1 infeasible：{funnel['stage2b_v1_infeasible']}；v2 救回 pass：{funnel['stage2b_v2_rescued_pass']}；dual-infeasible：{funnel['dual_infeasible']}")
    L.append("- 分组比较 bucket vs desk 的 v1-infeasible 率；v2 Phase4 rescue 是否在非 box 产生下游 numeric-pass 见 §3。")
    L.append("")
    # Q6 PRG + concavity proxy
    L.append("## 5. PRG+非box proxy 是否成立 + 凹几何 proxy 惩罚")
    L.append("")
    L.append(f"- numeric-pass {len(numeric_pass)}/{funnel['cem_eligible']}（逐物体：" + ", ".join(f"{o}={by_obj_pass.get(o,0)}" for o in OBJS) + "）。")
    L.append(f"- PRG scene-contract reject={funnel['prg_scene_contract_reject']}；gate-health {esummary['recommendation']['gate_health_pass']}/{len(metrics)}。")
    L.append("- **凹几何 proxy 惩罚**：对照 E173 尺寸-通过率先验（小箱~80%、大箱~35%），bucket/desk 实际通过率与尺寸先验之差 = 物体 proxy（bucket_wall / desk_surface_voxel）的失配代价；rubber_hull（手侧）影响单独记录。详见日志分析。")
    L.append("")
    L.append("## 6. 复现锚点")
    L.append("")
    L.append("- funnel：`registries/pipeline_funnel.json`；authority：`registries/pipeline_authority.tsv`")
    L.append(f"- completion audit：`completion_audit/completion_audit.json`（{audit['status']}）")
    L.append("- metrics：`s6_downstream/eval/full/e174_case_metrics.tsv`；summary：`.../summary.json`")
    L.append("- CEM manifest：`s6_downstream/manifests/cem_full_manifest.tsv`；scenes：`scene_snapshot/cem_sidecars/`")
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
