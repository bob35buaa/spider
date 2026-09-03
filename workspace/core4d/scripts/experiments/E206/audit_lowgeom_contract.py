#!/usr/bin/env python3
"""E206 P2.2: the lowgeom collision-proxy geometry contract (G1..G9).

Builds a <=N_MAX box proxy for every in-scope object key and scores it against
pre-declared numeric bars.  G1-G5/G7 are hard gates; G6 (cavity over-fill) is
report-and-waive by design — at a coarse budget a chair legitimately fills its
under-seat volume, and that is a physics-relevant approximation that must be
declared and reviewed, not silently passed or silently failed.

G8 (reference contact targets -> proxy surface) needs trajectories and runs in
P6.  G9 compares against the live 26-cell draft already on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402
import lowgeom_proxy_v2 as L  # noqa: E402
import manual_boxes as MB  # noqa: E402
import semantic_proxy as S  # noqa: E402

# ---- pre-declared numeric bars (plan236 P2.2) ---------------------------
BARS = {
    "G3_mesh_to_proxy_p90_max": 0.08,   # E176's own bar
    "G4_mesh_to_proxy_max_max": 0.16,
    "G5_proxy_to_mesh_p90_max": 0.14,
    "G6_overfill_warn": 0.30,           # report + waive, NOT a hard gate
    "G7_overfill_pitch_max": 0.02,
    "G9_p90_regression_max": 0.04,      # vs the live 26-cell draft
    # G10 (added 2026-09-03 at the user's request): every load-bearing box
    # bottom within 5 mm of the true mesh floor.  Catches "one leg lower than
    # the others" (the object rocks on one leg, a tilt the proxy invented) and
    # "box reaches below the mesh" (the object floats).  No aggregate distance
    # gate sees either: measured spreads of 5-16 mm moved G3's p90 by nothing.
    "G10_support_offset_max": 0.005,
}

# Proxy kinds with no voxel grid behind them: `target_cells` and the
# one-voxel-based G7 are undefined for these, and reported as "n/a".
NON_VOXEL_KINDS = ("semantic", "manual")

GEOM_RE = re.compile(r'name="(object_collision[^"]*)"')


def draft_geom_stats(object_key: str) -> dict[str, Any]:
    """The live dcv3 surface-voxel draft, for the G9 regression check."""
    for person in ("person1", "person2"):
        scene = C.PROCESSED_ROOT / f"{object_key}_{person}" / "scene.xml"
        if scene.exists():
            names = GEOM_RE.findall(scene.read_text(encoding="utf-8"))
            return {"draft_scene": str(scene), "draft_geom_count": len(names)}
    return {"draft_scene": "", "draft_geom_count": 0}


def evaluate(object_key: str, n_max: int, frozen_tc: int | None = None) -> dict[str, Any]:
    """Score one object's proxy.

    `frozen_tc` skips the target_cells sweep and re-measures only the selected
    configuration.  The sweep costs ~10 fidelity+cavity evaluations per object
    per budget (~35 min for the full 9x2 matrix), which is fine for the initial
    freeze but far too slow for the edit->re-measure loop the 3D reviewer needs.
    Once target_cells is frozen, re-scoring after a box deletion only requires
    the one configuration.
    """
    mesh_path = C.object_mesh_path(object_key)
    row: dict[str, Any] = {"object_key": object_key, "n_max": n_max}
    try:
        boxes, meta = S.build_effective_proxy(object_key, n_max, frozen_tc)
    except Exception as exc:  # noqa: BLE001 - an infeasible object is a datum
        row.update({"build_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return row

    sweep = meta.pop("sweep", [])
    scored = meta.pop("scored", [])
    row.update({k: v for k, v in meta.items() if not isinstance(v, (list, dict))})
    row["build_ok"] = True
    row["error"] = ""
    row["feasible_target_cells"] = ",".join(
        str(s["target_cells"]) for s in sweep if s.get("feasible")
    )
    row["selection_rule"] = meta.get("selection_rule", "")
    row["proxy_kind"] = meta.get("proxy_kind", "")
    row["parts"] = ",".join(meta.get("parts", []))
    row["edited"] = str(bool(meta.get("edited"))).lower()
    row["removed_indices"] = ",".join(str(i) for i in meta.get("removed_indices", []))
    row["n_boxes_before_edit"] = str(meta.get("n_boxes_before_edit", ""))
    row["editor"] = meta.get("editor", "")
    row["bar_passing_target_cells"] = ",".join(
        str(s["target_cells"]) for s in scored if s.get("bars_pass")
    )
    row.update(draft_geom_stats(object_key))

    n = int(row["object_geom_count"])
    checks: dict[str, bool] = {
        "G1_box_budget": 1 <= n <= n_max,
        "G3_mesh_to_proxy_p90": row["mesh_to_proxy_p90_m"] <= BARS["G3_mesh_to_proxy_p90_max"],
        "G4_mesh_to_proxy_max": row["mesh_to_proxy_max_m"] <= BARS["G4_mesh_to_proxy_max_max"],
        "G5_proxy_to_mesh_p90": row["proxy_to_mesh_p90_m"] <= BARS["G5_proxy_to_mesh_p90_max"],
    }
    # G7 asks "is any interior point further than ONE VOXEL from the mesh" — a
    # sanity check on the voxel merge, meaningless for a semantic part proxy or a
    # hand-placed one, both of which are deliberately part-spanning AABBs with no
    # voxel notion at all.  Applying it there is a category error, not a finding.
    if row.get("proxy_kind") in NON_VOXEL_KINDS:
        row["G7_overfill_pitch"] = "n/a"
    else:
        checks["G7_overfill_pitch"] = (
            row["interior_overfill_frac_pitch"] <= BARS["G7_overfill_pitch_max"]
        )
    support = L.support_contact_metrics(mesh_path, boxes, list(meta.get("parts") or []))
    row.update(
        {
            "support_box_count": support["support_box_count"],
            "support_bottom_spread_m": support["support_bottom_spread_m"],
            "support_bottom_offset_min_m": support["support_bottom_offset_min_m"],
            "support_bottom_offset_max_m": support["support_bottom_offset_max_m"],
            "support_bottom_worst_abs_m": support["support_bottom_worst_abs_m"],
            "floating_leg_labels": ",".join(support["floating_leg_labels"]),
        }
    )
    checks["G10_support_coplanar"] = (
        support["support_box_count"] >= 1
        and support["support_bottom_worst_abs_m"] <= BARS["G10_support_offset_max"]
    )

    row.update({k: str(v).lower() for k, v in checks.items()})
    row["G6_overfill_5cm"] = row["interior_overfill_frac_5cm"]
    row["G6_needs_waiver"] = str(
        row["interior_overfill_frac_5cm"] > BARS["G6_overfill_warn"]
    ).lower()
    row["geom_reduction"] = (
        f"{row['draft_geom_count']}->{n}" if row["draft_geom_count"] else f"?->{n}"
    )
    if row.get("proxy_kind") in NON_VOXEL_KINDS:
        row["target_cells"] = "n/a"
    row["hard_gates_pass"] = str(all(checks.values())).lower()
    row["failed_gates"] = ",".join(k for k, v in checks.items() if not v)
    row["_sweep"] = {"feasibility": sweep, "scored": scored}
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=C.N_MAX_TARGET)
    ap.add_argument("--compare-n-max", type=int, default=C.N_MAX_FALLBACK,
                    help="second budget to tabulate alongside (0 to skip)")
    ap.add_argument("--object-keys", default="",
                    help="comma-separated; default = the S1-landed keys")
    ap.add_argument("--out-dir", type=Path, default=C.S2_PROXY_DIR)
    ap.add_argument(
        "--frozen-target-cells", action="store_true",
        help="reuse target_cells from the existing contract instead of re-sweeping "
             "(fast path for the edit->re-measure loop)",
    )
    args = ap.parse_args()

    if args.object_keys:
        keys = tuple(k.strip() for k in args.object_keys.split(",") if k.strip())
    else:
        keys = landed_object_keys()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [args.n_max] + ([args.compare_n_max] if args.compare_n_max else [])

    frozen: dict[int, dict[str, int]] = {}
    if args.frozen_target_cells:
        for n_max in budgets:
            prev = args.out_dir / f"lowgeom_contract_n{n_max}.tsv"
            if not prev.exists():
                raise SystemExit(
                    f"--frozen-target-cells needs an existing {prev}; run a full audit once first"
                )
            # Semantic/manual rows carry target_cells="n/a" — there is nothing to
            # freeze for them, and int() would abort the whole run.
            frozen[n_max] = {
                r["object_key"]: int(r["target_cells"])
                for r in C.read_tsv(prev)
                if (r.get("target_cells") or "").lstrip("-").isdigit()
            }

    all_rows: dict[int, list[dict[str, Any]]] = {}
    for n_max in budgets:
        rows = [evaluate(k, n_max, frozen.get(n_max, {}).get(k)) for k in keys]
        all_rows[n_max] = rows
        sweeps = {r["object_key"]: r.pop("_sweep", []) for r in rows}
        if not args.frozen_target_cells:
            (args.out_dir / f"lowgeom_sweep_n{n_max}.json").write_text(
                json.dumps(sweeps, indent=2), encoding="utf-8"
            )
        fields = [
            "object_key", "object_category", "n_max", "build_ok", "error",
            "target_cells", "object_geom_count", "draft_geom_count", "geom_reduction",
            "voxel_pitch_m", "occupied_voxels",
            "mesh_to_proxy_p50_m", "mesh_to_proxy_p90_m", "mesh_to_proxy_max_m",
            "proxy_to_mesh_p50_m", "proxy_to_mesh_p90_m", "proxy_to_mesh_max_m",
            "interior_overfill_frac_5cm", "interior_overfill_frac_pitch",
            "interior_dist_p90_m", "proxy_volume_over_mesh_aabb",
            "G1_box_budget", "G3_mesh_to_proxy_p90", "G4_mesh_to_proxy_max",
            "G5_proxy_to_mesh_p90", "G7_overfill_pitch",
            "G6_overfill_5cm", "G6_needs_waiver",
            "G10_support_coplanar", "support_box_count", "support_bottom_spread_m",
            "support_bottom_offset_min_m", "support_bottom_offset_max_m",
            "support_bottom_worst_abs_m", "floating_leg_labels",
            "hard_gates_pass", "failed_gates", "feasible_target_cells",
            "bar_passing_target_cells", "selection_rule",
            "edited", "removed_indices", "n_boxes_before_edit", "editor",
            "proxy_kind", "parts",
        ]
        C.write_tsv(args.out_dir / f"lowgeom_contract_n{n_max}.tsv", rows, fields)

    # Cache the actual box geometry so the 3D viewer opens instantly instead of
    # re-running the 80k-point semantic decomposition for every object.
    # IMPORTANT: cache the PRE-edit boxes.  Reviewer deletions are recorded as
    # original build-order indices, so a viewer that renders post-edit boxes
    # would map "delete index 3" onto the wrong box.
    cache: dict[str, Any] = {}
    edits_now = L.load_box_edits()
    tc_by_key = {
        r["object_key"]: r.get("target_cells")
        for r in all_rows[args.n_max]
        if r.get("build_ok")
    }
    for k in keys:
        try:
            manual = MB.manual_boxes_for(k)
            if manual is not None:
                boxes, parts = manual
                kind, tc = "manual", -1
            elif k in S.SEMANTIC_SPEC:
                boxes, meta = S.build_semantic_boxes(k)
                kind = "semantic"
                parts = list(meta.get("parts", []))
                tc = -1
            else:
                raw = tc_by_key.get(k) or frozen.get(args.n_max, {}).get(k)
                tc = int(raw) if str(raw).lstrip("-").isdigit() else 0
                if tc <= 0:
                    continue
                boxes, _pitch = L._boxes_at(C.object_mesh_path(k), tc, args.n_max)
                kind = "voxel"
                parts = [f"v{i:02d}" for i in range(len(boxes))]
        except Exception:  # noqa: BLE001
            continue
        cache[k] = {
            "proxy_kind": kind,
            "parts": parts,
            "target_cells": tc,
            "removed": sorted(int(i) for i in edits_now.get(k, {}).get("removed", [])),
            "boxes": [
                {"center": list(map(float, b.center)), "half_size": list(map(float, b.half_size))}
                for b in boxes
            ],
        }
    (args.out_dir / "effective_boxes.json").write_text(
        json.dumps(cache, indent=1), encoding="utf-8"
    )

    primary = all_rows[args.n_max]
    ok = [r for r in primary if r.get("hard_gates_pass") == "true"]
    waivers = [r for r in primary if r.get("G6_needs_waiver") == "true"]

    md = [
        f"# E206 lowgeom 碰撞代理契约 (N_MAX={args.n_max})",
        "",
        f"物体: {len(keys)} 个 — {', '.join(keys)}",
        f"硬门 G1/G3/G4/G5/G7 通过: **{len(ok)}/{len(primary)}**",
        f"G6 腔体过填需豁免 (>{BARS['G6_overfill_warn']:.0%}): **{len(waivers)}** — "
        + (", ".join(r["object_key"] for r in waivers) if waivers else "无"),
        "",
        f"## 契约表 (N_MAX={args.n_max})",
        "",
        "| 物体 | 类型 | tc | boxes | 草稿→现在 | mesh→proxy p90 | max | proxy→mesh p90 | 过填>5cm | 过填>pitch | 硬门 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in primary:
        if not r.get("build_ok"):
            md.append(f"| {r['object_key']} | — | — | — | — | — | — | — | — | — | ❌ {r['error'][:60]} |")
            continue
        flag = "✅" if r["hard_gates_pass"] == "true" else f"❌ {r['failed_gates']}"
        warn = " ⚠️" if r["G6_needs_waiver"] == "true" else ""
        edit_tag = f" ✎-{len(r['removed_indices'].split(','))}" if r.get("edited") == "true" else ""
        md.append(
            f"| {r['object_key']}{edit_tag} | {r.get('proxy_kind','')[:4]} | {r['target_cells']} | {r['object_geom_count']} | "
            f"{r['geom_reduction']} | {r['mesh_to_proxy_p90_m']:.3f} | "
            f"{r['mesh_to_proxy_max_m']:.3f} | {r['proxy_to_mesh_p90_m']:.3f} | "
            f"{r['interior_overfill_frac_5cm']:.2f}{warn} | {flag} | {r.get('parts','')} |"
        )

    if args.compare_n_max:
        cmp_rows = {r["object_key"]: r for r in all_rows[args.compare_n_max]}
        md += [
            "",
            f"## N_MAX={args.n_max} vs {args.compare_n_max}（腔体过填是抬预算的唯一理由）",
            "",
            f"| 物体 | boxes@{args.compare_n_max} | boxes@{args.n_max} | "
            f"过填@{args.compare_n_max} | 过填@{args.n_max} | Δ过填 | "
            f"p90@{args.compare_n_max} | p90@{args.n_max} |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in primary:
            o = cmp_rows.get(r["object_key"], {})
            if not (r.get("build_ok") and o.get("build_ok")):
                continue
            d = r["interior_overfill_frac_5cm"] - o["interior_overfill_frac_5cm"]
            md.append(
                f"| {r['object_key']} | {o['object_geom_count']} | {r['object_geom_count']} | "
                f"{o['interior_overfill_frac_5cm']:.2f} | {r['interior_overfill_frac_5cm']:.2f} | "
                f"**{d:+.2f}** | {o['mesh_to_proxy_p90_m']:.3f} | {r['mesh_to_proxy_p90_m']:.3f} |"
            )

    md += [
        "",
        "## 门限出处",
        "",
        f"- G3 `mesh→proxy p90 ≤ {BARS['G3_mesh_to_proxy_p90_max']}` — 沿用 E176 "
        "`build_lowgeom_production.py:215` 原门",
        f"- G4 `mesh→proxy max ≤ {BARS['G4_mesh_to_proxy_max_max']}` — 单侧 Hausdorff 上限",
        f"- G5 `proxy→mesh p90 ≤ {BARS['G5_proxy_to_mesh_p90_max']}` — 外凸上限",
        f"- G6 `过填>5cm` — **报告+豁免**，>{BARS['G6_overfill_warn']:.0%} 告警。"
        "粗预算下椅子填掉座下空间是有物理后果的近似（腿无法从椅下摆过），"
        "必须显式声明并复审，不能静默通过",
        f"- G7 `过填>pitch ≤ {BARS['G7_overfill_pitch_max']}` — **仅对体素代理适用**"
        "（检查体素合并逻辑：box 来自表面体素，内部不该有点离 mesh 超过一个体素）。"
        "语义代理是按部件切的 AABB，没有体素概念，故标 `n/a`。",
        "- G2（全 box）在场景装配后由 `union_geoms_are_boxes()` 断言（P4/P7）",
        "- G8（接触目标→代理表面 p90 ≤ 0.08）需轨迹，P6 执行",
        f"- G9（相对草稿 p90 退化 ≤ {BARS['G9_p90_regression_max']}）需草稿代理的同尺度量，P4 执行",
    ]
    (args.out_dir / "lowgeom_contract.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    summary = {
        "n_max": args.n_max,
        "object_keys": list(keys),
        "hard_gates_pass": len(ok),
        "total": len(primary),
        "all_pass": len(ok) == len(primary),
        "waivers_needed": [r["object_key"] for r in waivers],
        "bars": BARS,
    }
    (args.out_dir / "lowgeom_contract.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["all_pass"] else 1


def landed_object_keys() -> tuple[str, ...]:
    """Object keys still in play: landed cases in S1 and not user-dropped.

    desk005 fell out at S1 (0 cases); chair021 was dropped by hand review.
    """
    in_play = tuple(k for k in C.OBJECT_KEYS if k not in C.DROPPED_OBJECT_KEYS)
    authority = (
        C.S1_DIR / "raw_contact" / f"raw_contact_pass_{C.PRIMARY_CONTACT_LABEL}_move2only.tsv"
    )
    if not authority.exists():
        return in_play
    keys = {row["object_key"] for row in C.read_tsv(authority)}
    return tuple(k for k in in_play if k in keys)


if __name__ == "__main__":
    raise SystemExit(main())
