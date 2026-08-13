#!/usr/bin/env python3
"""Render the E198 G1xA2 factorial report from the eval outputs.

Reads e198_factorial_by_object.tsv, e198_gate_migrations.tsv, e198_factorial_summary.json
and writes E198_G1xA2_factorial_report.md + SHA manifest. No GPU / no GL.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "experiments/E198"))
import e198_common as C  # noqa: E402

OUT = C.RESULTS_E198 / "s6_downstream/eval/full_factorial"
REPORT = OUT / "E198_G1xA2_factorial_report.md"
OBJECT_ORDER = ("box024", "box021", "box023", "box004")
PRIMARY = [
    ("track_obj_z_abs_err_cm_mean", "obj z |err| (cm)", "lower"),
    ("track_obj_pos_err_cm_mean", "obj 3D pos err (cm)", "lower"),
    ("hand_object_physics_penetration_3mm_frame_frac", "hand 3mm penetration", "lower"),
    ("leg_penetration_frac", "leg penetration", "lower"),
    ("hand_object_physics_contact_3mm_in_mask_frac", "3mm in-mask contact", "higher"),
    ("track_obj_ori_err_deg_mean", "obj ori err (deg)", "lower"),
]


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def f(v, nd=3):
    try:
        x = float(v)
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    by_object = {r["object_key"]: r for r in C.read_tsv(OUT / "e198_factorial_by_object.tsv")}
    migr = C.read_tsv(OUT / "e198_gate_migrations.tsv")
    summary = json.loads((OUT / "e198_factorial_summary.json").read_text(encoding="utf-8"))

    L: list[str] = []
    L.append("# E198 G1×A2 因子探索结果：box004/021/023/024 完整 2×2 交互项\n")
    L.append(f"_Core4D · Phase 61 · 生成于 {C.now()} · evaluator `{summary.get('metric_standard_id')}`_\n")
    L.append("> **纯因子探索**：估计 G1(gravcomp)×A2(hand-gate) 交互项 "
             "`INT = M(G1+A2) − M(A2) − M(G1) + M(A0)`。不做升级判决；A2 的 "
             "`INCONCLUSIVE_GATE_COLLAPSE` governance 不被推翻。\n")

    n_by = summary.get("objects", {})
    L.append("## 1. 2×2 单元均值（逐物体）\n")
    for m, label, better in PRIMARY:
        L.append(f"### {label}（{'越低越好' if better == 'lower' else '越高越好'}）\n")
        L.append("| Object | n | A0 | G1 | A2 | G1+A2 | INT [95% CI] |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for obj in OBJECT_ORDER:
            r = by_object[obj]
            ci = f"[{f(r.get(f'{m}__INT_ci_lo'))}, {f(r.get(f'{m}__INT_ci_hi'))}]"
            L.append(f"| {obj} | {n_by.get(obj,'')} | {f(r.get(f'{m}__A0_mean'))} | {f(r.get(f'{m}__G1_mean'))} "
                     f"| {f(r.get(f'{m}__A2_mean'))} | {f(r.get(f'{m}__G1A2_mean'))} "
                     f"| **{f(r.get(f'{m}__INT_mean'))}** {ci} |")
        L.append("")

    L.append("## 2. 交互项判读\n")
    L.append("对越低越好的指标：`INT<0` 且 CI 不含 0 ⇒ **协同**（组合优于两单干预相加的预期）；"
             "`INT>0` ⇒ **相互抵消/冗余**；CI 含 0 ⇒ 近似可加，无显著交互。\n")
    for m, label, better in PRIMARY[:4]:
        L.append(f"- **{label}**：")
        for obj in OBJECT_ORDER:
            r = by_object[obj]
            lo, hi = r.get(f"{m}__INT_ci_lo"), r.get(f"{m}__INT_ci_hi")
            try:
                sig = "含0" if float(lo) <= 0 <= float(hi) else ("协同(<0)" if float(hi) < 0 else "抵消(>0)")
            except (TypeError, ValueError):
                sig = "—"
            L.append(f"  {obj} INT={f(r.get(f'{m}__INT_mean'))} [{f(lo)},{f(hi)}] → {sig};")
        L.append("")

    L.append("## 3. 12-gate 迁移（McNemar exact，ALL 汇总）\n")
    L.append("| Transition | Gate | before→after | Δpp | P→F / F→P | exact p |")
    L.append("|---|---|---:|---:|---:|---:|")
    for r in migr:
        if r["object_key"] != "ALL":
            continue
        try:
            if abs(float(r["delta_pp"])) < 2.0 and int(r["p2f"]) == 0 and int(r["f2p"]) == 0:
                continue
        except (TypeError, ValueError):
            pass
        L.append(f"| {r['transition']} | {r['gate']} | {r['before_pass']}→{r['after_pass']} "
                 f"| {r['delta_pp']} | {r['p2f']} / {r['f2p']} | {r['mcnemar_p']} |")
    L.append("")

    L.append("## 4. 产物与哈希\n")
    L.append("| 文件 | SHA256 |")
    L.append("|---|---|")
    for name in ("e198_factorial_by_case.tsv", "e198_factorial_by_object.tsv",
                 "e198_gate_migrations.tsv", "e198_factorial_summary.json", "e198_arm_cache.tsv"):
        p = OUT / name
        if p.is_file():
            L.append(f"| {name} | `{sha256(p)[:16]}…` |")
    L.append("")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
