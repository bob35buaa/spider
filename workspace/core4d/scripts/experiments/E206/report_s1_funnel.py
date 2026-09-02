#!/usr/bin/env python3
"""E206 P1: honest S1 funnel + the case authority.

Emits, per object key, the 3cm and 5cm columns SIDE BY SIDE.  That comparison is
what settled the label choice empirically: 5cm buys only 2 cases over 3cm
(74 -> 76, same 9 objects), so the primary label stayed at 3cm and E206 keeps the
ruler shared with the whole box/bucket line.  The 5cm column is retained as
reported evidence for that decision.

Every in-scope inventory row gets exactly one terminal reason per threshold
(plan236 C0).

Outputs (under <run>/s1_raw_contact/):
  e206_s1_funnel.tsv / .md / .json        per-object funnel + totals + gate analysis
  e206_s1_case_terminal.tsv               one row per in-scope case, terminal reason @3cm and @5cm
  raw_contact/raw_contact_pass_3cm_move2only.tsv   THE case authority
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402


def load_pass_rows(raw_dir: Path, label: str) -> dict[str, dict[str, str]]:
    path = raw_dir / f"raw_contact_pass_{label}.tsv"
    if not path.exists():
        return {}
    return {row["case_id"]: row for row in C.read_tsv(path)}


def load_candidate_rows(raw_dir: Path, label: str) -> dict[str, dict[str, str]]:
    """All scored rows for a threshold (pass + review + fail + reject)."""
    path = raw_dir / f"raw_contact_candidates_{label}.tsv"
    if not path.exists():
        return {}
    return {row["case_id"]: row for row in C.read_tsv(path)}


def terminal_reason(case_id: str, cand: dict[str, dict[str, str]], passed: dict[str, dict[str, str]]) -> tuple[str, str]:
    """(decision, human-readable reason) for one case at one threshold."""
    if case_id in passed:
        return "pass", ""
    row = cand.get(case_id)
    if row is None:
        return "not_scored", "case never reached raw-contact scoring"
    decision = row.get("raw_contact_decision") or "unknown"
    reason = row.get("motion_quality_fail_reasons") or row.get("raw_contact_notes") or ""
    return decision, reason


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=C.RESULTS)
    ap.add_argument("--write-authorities", action="store_true",
                    help="also write the primary case-authority TSV")
    args = ap.parse_args()

    s1 = args.run_root / "s1_raw_contact"
    inv_path = s1 / "inventory/inventory.tsv"
    raw_dir = s1 / "raw_contact"
    if not inv_path.exists():
        raise SystemExit(f"missing inventory: {inv_path}")

    inventory = C.read_tsv(inv_path)

    # ---- layer 1/2: inventory -> move2_* -> size gate -------------------
    per_object: dict[str, dict[str, object]] = {}
    for key in C.OBJECT_KEYS:
        per_object[key] = {
            "object_key": key,
            "category": C.object_category(key),
            "inventory_rows": 0,
            "move2_rows": 0,
            "size_gate_rejected": 0,
            "in_scope": 0,
        }

    in_scope_rows: list[dict[str, str]] = []
    size_rejects: Counter[str] = Counter()
    for row in inventory:
        key = row.get("object_key", "")
        if key not in per_object:
            continue
        per_object[key]["inventory_rows"] += 1  # type: ignore[operator]
        if row.get("action") not in C.MOVE2_ACTIONS:
            continue
        per_object[key]["move2_rows"] += 1  # type: ignore[operator]
        if row.get("hard_reject_reason"):
            per_object[key]["size_gate_rejected"] += 1  # type: ignore[operator]
            size_rejects[row["hard_reject_reason"]] += 1
            continue
        per_object[key]["in_scope"] += 1  # type: ignore[operator]
        in_scope_rows.append(row)

    # ---- layer 3: raw contact @3cm and @5cm -----------------------------
    labels = ("3cm", "5cm")
    cand = {lb: load_candidate_rows(raw_dir, lb) for lb in labels}
    passed = {lb: load_pass_rows(raw_dir, lb) for lb in labels}
    s1_ran = any(cand[lb] for lb in labels)

    for lb in labels:
        for key in C.OBJECT_KEYS:
            per_object[key][f"pass_{lb}"] = 0
            per_object[key][f"nonpass_{lb}"] = 0

    case_terminal: list[dict[str, str]] = []
    for row in in_scope_rows:
        case_id = row["case_id"]
        key = row["object_key"]
        entry = {
            "case_id": case_id,
            "object_key": key,
            "category": C.object_category(key),
            "action": row.get("action", ""),
            "size_band": row.get("size_band", ""),
        }
        for lb in labels:
            decision, reason = terminal_reason(case_id, cand[lb], passed[lb])
            entry[f"decision_{lb}"] = decision
            entry[f"reason_{lb}"] = reason
            if decision == "pass":
                per_object[key][f"pass_{lb}"] += 1  # type: ignore[operator]
            else:
                per_object[key][f"nonpass_{lb}"] += 1  # type: ignore[operator]
        entry["rescued_by_5cm"] = str(
            entry["decision_3cm"] != "pass" and entry["decision_5cm"] == "pass"
        ).lower()
        case_terminal.append(entry)

    # ---- funnel table ---------------------------------------------------
    funnel_rows = [per_object[k] for k in C.OBJECT_KEYS]
    totals = {
        "object_key": "TOTAL",
        "category": "",
        "inventory_rows": sum(int(r["inventory_rows"]) for r in funnel_rows),
        "move2_rows": sum(int(r["move2_rows"]) for r in funnel_rows),
        "size_gate_rejected": sum(int(r["size_gate_rejected"]) for r in funnel_rows),
        "in_scope": sum(int(r["in_scope"]) for r in funnel_rows),
    }
    for lb in labels:
        totals[f"pass_{lb}"] = sum(int(r[f"pass_{lb}"]) for r in funnel_rows)
        totals[f"nonpass_{lb}"] = sum(int(r[f"nonpass_{lb}"]) for r in funnel_rows)
    funnel_rows_out = funnel_rows + [totals]

    fields = [
        "object_key", "category", "inventory_rows", "move2_rows",
        "size_gate_rejected", "in_scope",
        "pass_3cm", "nonpass_3cm", "pass_5cm", "nonpass_5cm",
    ]
    C.write_tsv(s1 / "e206_s1_funnel.tsv", funnel_rows_out, fields)
    C.write_tsv(
        s1 / "e206_s1_case_terminal.tsv",
        case_terminal,
        ["case_id", "object_key", "category", "action", "size_band",
         "decision_3cm", "reason_3cm", "decision_5cm", "reason_5cm", "rescued_by_5cm"],
    )

    rescued = sum(1 for e in case_terminal if e["rescued_by_5cm"] == "true")

    # ---- markdown -------------------------------------------------------
    md = [
        "# E206 S1 funnel — desk+chair move2_* (3cm vs 5cm 并排)",
        "",
        f"run_root: `{args.run_root}`",
        f"scope: {len(C.OBJECT_KEYS)} object keys, actions={list(C.MOVE2_ACTIONS)}",
        f"contact label: **{C.PRIMARY_CONTACT_LABEL}**（单一口径，与 box/bucket 全线一致）"
        " · 5cm 列仅作为该选择的依据保留",
        "",
        "| object | cat | inventory | move2_* | 尺寸门拒 | 在范围 | pass@3cm | pass@5cm | 5cm 救回 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in funnel_rows:
        key = str(r["object_key"])
        obj_rescued = sum(
            1 for e in case_terminal if e["object_key"] == key and e["rescued_by_5cm"] == "true"
        )
        md.append(
            f"| {key} | {r['category']} | {r['inventory_rows']} | {r['move2_rows']} | "
            f"{r['size_gate_rejected']} | {r['in_scope']} | "
            f"{r.get('pass_3cm', 0)} | {r.get('pass_5cm', 0)} | {obj_rescued} |"
        )
    md.append(
        f"| **TOTAL** | | **{totals['inventory_rows']}** | **{totals['move2_rows']}** | "
        f"**{totals['size_gate_rejected']}** | **{totals['in_scope']}** | "
        f"**{totals.get('pass_3cm', 0)}** | **{totals.get('pass_5cm', 0)}** | **{rescued}** |"
    )
    md += [
        "",
        "## 尺寸门拒因分布 (move2_* 内)",
        "",
    ]
    if size_rejects:
        for reason, count in size_rejects.most_common():
            md.append(f"- `{reason}`: {count}")
    else:
        md.append("- (none)")

    if not s1_ran:
        md += ["", "> ⚠️ raw_contact 尚未运行，3cm/5cm 列全为 0。先跑 run_raw_contact.py。"]

    md += [
        "",
        "## 非 pass 终态原因分布",
        "",
        "| label | decision | count |",
        "|---|---|---:|",
    ]
    for lb in labels:
        counts = Counter(e[f"decision_{lb}"] for e in case_terminal if e[f"decision_{lb}"] != "pass")
        for decision, count in counts.most_common():
            md.append(f"| {lb} | `{decision}` | {count} |")

    # ---- what actually causes the attrition -----------------------------
    lb = C.PRIMARY_CONTACT_LABEL
    rot_rejects = sum(1 for e in case_terminal if "object_rotation" in e[f"reason_{lb}"])
    lift_only = sum(
        1
        for e in case_terminal
        if "object_lift" in e[f"reason_{lb}"] and "object_rotation" not in e[f"reason_{lb}"]
    )
    contact_fail = sum(
        1 for e in case_terminal if e[f"decision_{lb}"] == "raw_contact_fail"
    )
    md += [
        "",
        "## 衰减归因（真正的瓶颈）",
        "",
        f"| 原因 | 数量 | 占在范围 {totals['in_scope']} |",
        "|---|---:|---:|",
        f"| `object_rotation >= 45°` 运动硬门 | **{rot_rejects}** | {rot_rejects / max(1, int(totals['in_scope'])):.0%} |",
        f"| `object_lift <= 0.30m`（不含同时旋转超限） | {lift_only} | {lift_only / max(1, int(totals['in_scope'])):.0%} |",
        f"| 接触质量不足（weak_two_hand_overlap / unbalanced 等） | {contact_fail} | {contact_fail / max(1, int(totals['in_scope'])):.0%} |",
        "",
        "**结论：瓶颈是旋转门，不是接触阈值。** 双人搬桌/椅常需转向（绕门、调头），"
        "而 45° 硬门是为 box 搬运设定的。E206 **不动此门**（保持与 bucket 线 "
        "E174/E178/E202/E204/E205 同一 S1 口径，结论可横向比），作为后续实验方向记录。",
    ]

    primary_total = int(totals.get(f"pass_{lb}", 0))
    zero_objects = [
        k for k in C.OBJECT_KEYS if int(per_object[k].get(f"pass_{lb}", 0)) == 0  # type: ignore[arg-type]
    ]
    md += [
        "",
        "## 退出检查",
        "",
        f"- 在范围候选（尺寸门后）: **{totals['in_scope']}**（每个在两个阈值下各有唯一终态）",
        f"- **主口径 {lb} 落地: {primary_total}**，覆盖 "
        f"**{len(C.OBJECT_KEYS) - len(zero_objects)}/{len(C.OBJECT_KEYS)}** 个物体",
        f"- 0 case 因而退出 E206 的物体: {zero_objects or '（无）'}",
        f"- 对照：5cm 落地 **{int(totals.get('pass_5cm', 0))}**，仅多 **{rescued}** 个 "
        "→ 放宽接触阈值几乎无收益，故维持 3cm 单一口径、不建 3cm/5cm 桥接层",
        f"- 停机线 MIN_CASES_ESCALATE={C.MIN_CASES_ESCALATE} → "
        + ("**PASS**" if primary_total >= C.MIN_CASES_ESCALATE else "**STOP，需升级讨论**"),
    ]
    (s1 / "e206_s1_funnel.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- case authority --------------------------------------------------
    written: dict[str, object] = {}
    if args.write_authorities and s1_ran:
        in_scope_ids = {row["case_id"] for row in in_scope_rows}
        primary = [
            row for cid, row in passed[C.PRIMARY_CONTACT_LABEL].items() if cid in in_scope_ids
        ]
        primary.sort(key=lambda r: r["case_id"])
        written["primary"] = C.write_tsv(
            raw_dir / f"raw_contact_pass_{C.PRIMARY_CONTACT_LABEL}_move2only.tsv", primary
        )
        written["primary_objects"] = len({r["object_key"] for r in primary})
        dropped = sorted(set(C.OBJECT_KEYS) - {r["object_key"] for r in primary})
        written["objects_with_zero_cases"] = dropped
        if dropped:
            print(
                f"NOTE: {len(dropped)} object key(s) landed 0 cases and drop out of "
                f"E206: {dropped}",
                file=sys.stderr,
            )

    summary = {
        "exp_id": C.EXP_ID,
        "object_keys": list(C.OBJECT_KEYS),
        "actions": list(C.MOVE2_ACTIONS),
        "totals": totals,
        "rescued_by_5cm": rescued,
        "size_reject_reasons": dict(size_rejects),
        "s1_ran": s1_ran,
        "primary_label": C.PRIMARY_CONTACT_LABEL,
        "primary_landed": primary_total,
        "objects_with_zero_cases": zero_objects,
        "attrition": {
            "rotation_gate_rejects": rot_rejects,
            "lift_gate_only_rejects": lift_only,
            "contact_quality_fails": contact_fail,
        },
        "min_cases_escalate": C.MIN_CASES_ESCALATE,
        "escalate": primary_total < C.MIN_CASES_ESCALATE,
        "authorities_written": written,
    }
    (s1 / "e206_s1_funnel.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
