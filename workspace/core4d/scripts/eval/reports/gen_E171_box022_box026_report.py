#!/usr/bin/env python3
"""E171: generate the refined Markdown main report + machine JSON.

Synthesizes the pipeline funnel, Stage2b v1/v2 rescue provenance, PRG-scene
contract outcomes, and the Full-CEM numeric evaluation into a user-facing
Markdown report (plan 8.3 questions) plus a machine-readable JSON summary.
Reads only committed E171 evidence; does not write manual_* user fields.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/E171"))
import e171_common as C

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
    ap.add_argument("--out", type=Path, default=C.RESULTS / "s6_downstream/eval/full/E171_report.md")
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
    L.append("# E171 结果报告：Box022/Box026 全流程筛选 + Full CEM")
    L.append("")
    L.append("_Core4D Phase 34 · rubber_hull + E170 PRG cross-object candidate · machine recommendation = `PENDING_USER_REVIEW`_")
    L.append("")
    L.append("> 本报告为机器核验 + 视觉复核交付物；最终 `USE/DO_NOT_USE` 由用户裁决。E171 不做 reward/route sweep，science yield 与执行完成解耦。")
    L.append("")
    # Q1 funnel
    L.append("## 1. 60 条 raw row 的终态分布（funnel）")
    L.append("")
    L.append(md_table(["阶段", "计数"], [
        ["raw expected / seen", f"{funnel['raw_expected']} / {funnel['raw_seen']}"],
        ["Box022 / Box026", f"{funnel['by_object']['box022']} / {funnel['by_object']['box026']}"],
        ["raw-contact pass (3cm)", funnel["raw_contact_pass"]],
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
             f" = {sum(funnel['terminal_states'].values())}（raw_closure_pass）")
    L.append("")
    # Q2 per-object
    L.append("## 2. 分对象产出")
    L.append("")
    L.append(md_table(["对象", "raw", "S5-ready(CEM)", "Full complete", "numeric pass"], [
        ["Box022", 8, 0, 0, 0],
        ["Box026", 52, funnel["cem_eligible"], audit["full_completed"], len(numeric_pass)],
    ]))
    L.append("")
    L.append(f"- Box022：raw-contact 全 fail → **DATA_NEGATIVE**（0 下游 row，合法数据层负结果）。")
    L.append(f"- Box026：{funnel['cem_eligible']} 条进入 Full CEM，{audit['full_completed']} 条完成，{len(numeric_pass)} 条 numeric pass。")
    L.append("")
    # Q3 usable/not
    L.append("## 3. 各 case 可用性（numeric 层，最终以用户裁决为准）")
    L.append("")
    rows = []
    for m in metrics:
        rows.append([m["case_id"], m.get("retarget_variant_id", ""),
                     "✅" if m in numeric_pass else "❌",
                     m.get("numeric_failure_modes", "") or "-",
                     "pass" if m.get("leg_gate_health_pass") in ("True", "true") else "fail"])
    L.append(md_table(["case", "variant", "numeric", "failure modes", "gate-health"], rows))
    L.append("")
    L.append(f"- numeric failure 分布：{esummary.get('numeric_failure_counts', {})}")
    L.append(f"- gate-health pass：{esummary['recommendation']['gate_health_pass']}/{len(metrics)}（与 E170 0/28 一致，candidate gate 已知弱项，与质量判定分列）")
    L.append("")
    # Q4 E106
    L.append("## 4. 相比 E106 的历史 funnel/yield")
    L.append("")
    L.append("E106 曾在 30 个 box026 candidate 上 28 Stage2b runnable / 28 Full CEM / 4 strict positive。E171 fresh：" +
             f"17 raw-contact pass → 17 Stage2b pass（14 v1 + 3 v2）→ {funnel['cem_eligible']} CEM eligible（5 条被 PRG scene contract 拒绝）→ {len(numeric_pass)} numeric pass。")
    L.append("")
    L.append(f"> {E106_OVERLAP['note']} 不声称严格因果 A/B。")
    L.append("")
    # Q5 v1/v2
    L.append("## 5. v1→v2 rescue")
    L.append("")
    v2rows = [a for a in authority if a.get("selected_retarget_variant_id") == "omnirt_v2"]
    L.append(f"- v1 infeasible：{funnel['stage2b_v1_infeasible']}（040_p2 / 043_p2 / 137_p2，其中 043_p2、137_p2 命中 E106 prior）")
    L.append(f"- v2 救回 pass：{funnel['stage2b_v2_rescued_pass']}/3；dual-infeasible：{funnel['dual_infeasible']}")
    L.append(f"- 3 条 v2-rescued 中，2 条（043_p2/137_p2）后续被 PRG scene contract 拒绝，1 条（040_p2）进入 CEM 但 numeric fail。")
    L.append("")
    # Q6 PRG
    L.append("## 6. PRG 跨物体是否成立 + contact trade-off")
    L.append("")
    L.append("- **contact trade-off 显著**：17 条 Stage2b-pass 中 5 条因 reference 首帧 lower-body/object 初始穿透被 PRG scene contract 拒绝（box026 低位贴腿搬箱），Full CEM 7/12 fail 中 contact×4 + lower_body×4 为主因。")
    L.append("- **lower-body 定向**：PRG 的 16 lower-body/object pair + soft penalty 在 5 条 numeric-pass 上成立，但 gate-health 0/12 说明 candidate gate 未达 full validity。")
    L.append("- 结论倾向 **PARTIAL_YIELD**（Box026 有 case-level positive，Box022 DATA_NEGATIVE），不足以将 PRG 升级为跨物体统一默认；最终以用户裁决为准。")
    L.append("")
    L.append("## 7. 复现锚点")
    L.append("")
    L.append(f"- funnel：`registries/pipeline_funnel.json`；authority：`registries/pipeline_authority.tsv`")
    L.append(f"- completion audit：`completion_audit/completion_audit.json`（{audit['status']}）")
    L.append(f"- metrics：`s6_downstream/eval/full/e171_case_metrics.tsv`；summary：`.../summary.json`")
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
