#!/usr/bin/env python3
"""E206 P2.3: interactive 3D review of the lowgeom collision proxies.

`docs/data_construction_v3/04_scene_template_policy.md:63-77` forbids releasing a
non-box template on "MuJoCo loads and renders" alone — a human must look at the
mesh/collision overlay and write `review_decision=approve_clean`.  This is that
viewer.

What you are judging, per object:
  1. Does the proxy bridge a real opening?  (under-desk space, chair seat-to-floor)
     -> that is where a robot leg would swing; filling it fabricates collisions.
  2. Is the collision visibly LARGER than the mesh anywhere?  -> phantom contact.
  3. Is the carried surface covered?  (table top / seat / rim)  -> that is where
     the reference contact targets live; a gap there loses real contact.

Run:
    .venv/bin/python workspace/core4d/scripts/experiments/E206/review_proxy_3d.py
    # then open http://<host>:8080

Verdicts are written to `<run>/s2_templates/review/nonbox_template_review.tsv`
in the same schema E145/E174 used, so the S2 gate can consume them unchanged.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e206_common as C  # noqa: E402
import lowgeom_proxy_v2 as L  # noqa: E402
import semantic_proxy as S  # noqa: E402

MESH_COLOR = (170, 170, 178)
DELETED_COLOR = (120, 120, 120)
BOX_COLORS = [
    (233, 78, 60), (52, 168, 226), (95, 200, 120), (245, 176, 55),
    (176, 116, 222), (238, 130, 178), (86, 205, 199), (200, 200, 90),
]
REVIEW_FIELDS = [
    "object_key", "source_scene_task", "object_category", "template_status",
    "recommended_action", "proxy_variant", "proxy_policy", "mesh_path",
    "scene_xml", "object_geom_count", "target_cells",
    "mesh_to_proxy_p90_m", "proxy_to_mesh_p90_m", "interior_overfill_frac_5cm",
    "contract_status", "waived_checks", "boxes_removed", "object_geom_count_final",
    "review_decision", "reviewer", "reviewed_at", "notes",
]


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_contract(n_max: int) -> dict[str, dict[str, str]]:
    path = C.S2_PROXY_DIR / f"lowgeom_contract_n{n_max}.tsv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run audit_lowgeom_contract.py first")
    return {r["object_key"]: r for r in C.read_tsv(path)}


def source_task(object_key: str) -> str:
    for person in ("person1", "person2"):
        if (C.PROCESSED_ROOT / f"{object_key}_{person}" / "scene.xml").exists():
            return f"{object_key}_{person}"
    return f"{object_key}_person1"


_BOX_CACHE: dict[str, Any] | None = None


def _cached(object_key: str) -> dict[str, Any] | None:
    """Boxes frozen by the last audit run — avoids re-deriving them per launch."""
    global _BOX_CACHE
    if _BOX_CACHE is None:
        path = C.S2_PROXY_DIR / "effective_boxes.json"
        import json as _json

        _BOX_CACHE = _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _BOX_CACHE.get(object_key)


def build_payload(object_key: str, row: dict[str, str], n_max: int) -> dict[str, Any]:
    mesh_path = C.object_mesh_path(object_key)
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Show the proxy that actually ships.  Built PRE-edit so the viewer's box
    # indices line up with what `box_edits.json` records.
    hit = _cached(object_key)
    if hit is not None:
        # The cache holds PRE-edit boxes in original build order, so the
        # viewer's indices line up with box_edits.json exactly.
        boxes = [
            L.ProxyBox(center=np.asarray(b["center"]), half_size=np.asarray(b["half_size"]))
            for b in hit["boxes"]
        ]
        return {
            "mesh": mesh,
            "boxes": boxes,
            "parts": hit.get("parts") or [f"v{i:02d}" for i in range(len(boxes))],
            "kind": hit.get("proxy_kind", "voxel"),
            "row": row,
            "mesh_path": mesh_path,
            "from_cache": True,
        }
    if object_key in S.SEMANTIC_SPEC:
        boxes, meta = S.build_semantic_boxes(object_key)
        kind, parts = "semantic", list(meta.get("parts", []))
    else:
        boxes, _pitch = L._boxes_at(mesh_path, int(row["target_cells"]), n_max)
        kind, parts = "voxel", [f"v{i:02d}" for i in range(len(boxes))]
    return {
        "mesh": mesh,
        "boxes": boxes,
        "parts": parts,
        "kind": kind,
        "row": row,
        "mesh_path": mesh_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-max", type=int, default=C.N_MAX_TARGET)
    ap.add_argument("--reviewer", default="user")
    ap.add_argument("--out-dir", type=Path, default=C.S2_TEMPLATE_DIR / "review")
    args = ap.parse_args()

    import viser

    contract = load_contract(args.n_max)
    keys = [k for k in C.OBJECT_KEYS if k in contract and contract[k].get("build_ok") in ("True", "true")]
    if not keys:
        raise SystemExit("no buildable objects in the contract")

    print(f"[E206] loading {len(keys)} object proxies ...", flush=True)
    payloads = {k: build_payload(k, contract[k], args.n_max) for k in keys}
    print("[E206] ready", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.out_dir / "nonbox_template_review.tsv"
    verdicts: dict[str, dict[str, str]] = {}
    if review_path.exists():
        verdicts = {r["object_key"]: r for r in C.read_tsv(review_path)}

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    state = {"key": keys[0]}
    handles: list[Any] = []
    edits: dict[str, Any] = L.load_box_edits()

    def flush_reviews() -> None:
        rows = []
        for k in keys:
            r = contract[k]
            v = verdicts.get(k, {})
            rows.append(
                {
                    "object_key": k,
                    "source_scene_task": source_task(k),
                    "object_category": C.object_category(k),
                    "template_status": (
                        "clean_reviewed" if v.get("review_decision") == "approve_clean"
                        else "manual_review_required"
                    ),
                    "recommended_action": "review_mesh_collision_overlay",
                    "proxy_variant": f"{C.object_category(k)}_lowgeom{args.n_max}",
                    "proxy_policy": f"{C.object_category(k)}_lowgeom{args.n_max}_proxy",
                    "mesh_path": str(C.object_mesh_path(k).relative_to(C.REPO)),
                    "scene_xml": f"example_datasets/processed/core4d/unitree_g1/humanoid_object/{source_task(k)}/scene.xml",
                    "object_geom_count": r["object_geom_count"],
                    "target_cells": r["target_cells"],
                    "mesh_to_proxy_p90_m": r["mesh_to_proxy_p90_m"],
                    "proxy_to_mesh_p90_m": r["proxy_to_mesh_p90_m"],
                    "interior_overfill_frac_5cm": r["interior_overfill_frac_5cm"],
                    "contract_status": "pass" if r["hard_gates_pass"] == "true" else r["failed_gates"],
                    "waived_checks": "G6_overfill" if r["G6_needs_waiver"] == "true" else "",
                    "boxes_removed": ",".join(str(i) for i in sorted(removed_set(k))),
                    "object_geom_count_final": str(
                        int(r["object_geom_count"]) - len(removed_set(k))
                    ),
                    "review_decision": v.get("review_decision", ""),
                    "reviewer": v.get("reviewer", ""),
                    "reviewed_at": v.get("reviewed_at", ""),
                    "notes": v.get("notes", ""),
                }
            )
        C.write_tsv(review_path, rows, REVIEW_FIELDS)

    # ---- UI -------------------------------------------------------------
    with server.gui.add_folder("物体"):
        picker = server.gui.add_dropdown("object", tuple(keys), initial_value=keys[0])
        info = server.gui.add_markdown("")
    with server.gui.add_folder("显示"):
        show_mesh = server.gui.add_checkbox("真实 mesh", True)
        mesh_opacity = server.gui.add_slider("mesh 透明度", 0.05, 1.0, 0.05, 0.35)
        show_boxes = server.gui.add_checkbox("碰撞 box", True)
        box_opacity = server.gui.add_slider("box 透明度", 0.05, 1.0, 0.05, 0.55)
        wireframe = server.gui.add_checkbox("box 线框", False)
        show_deleted = server.gui.add_checkbox("显示已删 box(灰线框)", True)
        show_floor = server.gui.add_checkbox("地面网格", True)
    with server.gui.add_folder("编辑碰撞体（点 3D 里的 box 即可删/恢复）"):
        edit_status = server.gui.add_markdown("")
        btn_reset = server.gui.add_button("↩ 恢复本物体全部 box")
        box_list_folder = server.gui.add_folder("逐 box 开关")
    with server.gui.add_folder("判定"):
        notes = server.gui.add_text("notes", "")
        btn_ok = server.gui.add_button("✅ approve_clean")
        btn_no = server.gui.add_button("❌ needs_manual_edit")
        progress = server.gui.add_markdown("")

    box_checkboxes: list[Any] = []

    def removed_set(key: str) -> set[int]:
        return set(int(i) for i in edits.get(key, {}).get("removed", []))

    def set_removed(key: str, removed: set[int]) -> None:
        p = payloads[key]
        if removed:
            edits[key] = {
                "n_max": args.n_max,
                "proxy_kind": p["kind"],
                "target_cells": -1 if p["kind"] == "semantic" else int(p["row"]["target_cells"]),
                "n_boxes_original": len(p["boxes"]),
                "removed": sorted(removed),
                "editor": args.reviewer,
                "edited_at": now(),
                "notes": notes.value,
            }
        else:
            edits.pop(key, None)
        L.save_box_edits(edits)

    def toggle_box(key: str, index: int) -> None:
        removed = removed_set(key)
        if len(removed) + 1 >= len(payloads[key]["boxes"]) and index not in removed:
            print(f"[E206] refusing to delete the last remaining box of {key}", flush=True)
            return
        removed.symmetric_difference_update({index})
        set_removed(key, removed)
        render()
        refresh_info()

    def render() -> None:
        for h in handles:
            h.remove()
        handles.clear()
        key = state["key"]
        p = payloads[key]
        removed = removed_set(key)
        if show_mesh.value:
            handles.append(
                server.scene.add_mesh_simple(
                    "/mesh",
                    vertices=np.asarray(p["mesh"].vertices),
                    faces=np.asarray(p["mesh"].faces),
                    color=MESH_COLOR,
                    opacity=float(mesh_opacity.value),
                    material="standard",
                )
            )
        if show_boxes.value:
            for i, b in enumerate(p["boxes"]):
                is_removed = i in removed
                if is_removed and not show_deleted.value:
                    continue
                handle = server.scene.add_box(
                    f"/box/{i:03d}",
                    color=DELETED_COLOR if is_removed else BOX_COLORS[i % len(BOX_COLORS)],
                    dimensions=tuple(2.0 * np.asarray(b.half_size)),
                    position=tuple(np.asarray(b.center)),
                    opacity=0.12 if is_removed else float(box_opacity.value),
                    wireframe=True if is_removed else bool(wireframe.value),
                )
                handle.on_click(lambda _, idx=i: toggle_box(state["key"], idx))
                handles.append(handle)
        if show_floor.value:
            ext = float(np.max(p["mesh"].extents))
            handles.append(
                server.scene.add_grid(
                    "/grid",
                    width=ext * 2.5,
                    height=ext * 2.5,
                    position=(0.0, 0.0, float(p["mesh"].bounds[0][2])),
                )
            )

    def rebuild_box_list() -> None:
        for cb in box_checkboxes:
            cb.remove()
        box_checkboxes.clear()
        key = state["key"]
        p = payloads[key]
        removed = removed_set(key)
        with box_list_folder:
            for i, b in enumerate(p["boxes"]):
                dims = 2.0 * np.asarray(b.half_size)
                ctr = np.asarray(b.center)
                part = p["parts"][i] if i < len(p["parts"]) else f"{i:02d}"
                label = (
                    f"{i:02d} {part:8s} {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}"
                    f" @y={ctr[1]:+.2f}"
                )
                cb = server.gui.add_checkbox(label, i not in removed)
                cb.on_update(lambda _, idx=i: toggle_box(state["key"], idx))
                box_checkboxes.append(cb)

    def refresh_info() -> None:
        key = state["key"]
        r = contract[key]
        v = verdicts.get(key, {})
        waiver = (
            "\n\n> ⚠️ **需要 G6 豁免判定** — 腔体过填 "
            f"{float(r['interior_overfill_frac_5cm']):.0%}，代表座下/桌下空间被填实，"
            "机器人腿无法从中摆过。抬预算对它无效。"
            if r["G6_needs_waiver"] == "true" else ""
        )
        pl = payloads[key]
        info.content = (
            f"### {key}  ({C.object_category(key)})\n\n"
            f"**代理类型: `{pl['kind']}`**"
            + (f"  部件: {', '.join(pl['parts'])}\n\n" if pl["kind"] == "semantic" else "\n\n")
            + f"| | |\n|---|---|\n"
            f"| box 数 | **{r['object_geom_count']}** (草稿 {r['draft_geom_count']}) |\n"
            f"| target_cells | {r['target_cells']} |\n"
            f"| mesh→proxy p90 | {float(r['mesh_to_proxy_p90_m']):.3f} m |\n"
            f"| mesh→proxy max | {float(r['mesh_to_proxy_max_m']):.3f} m |\n"
            f"| proxy→mesh p90 | {float(r['proxy_to_mesh_p90_m']):.3f} m |\n"
            f"| 腔体过填>5cm | **{float(r['interior_overfill_frac_5cm']):.0%}** |\n"
            f"| 硬门 | {'✅ pass' if r['hard_gates_pass']=='true' else '❌ '+r['failed_gates']} |\n"
            f"| 当前判定 | **{v.get('review_decision') or '（未判）'}** |"
            + waiver
        )
        rm = sorted(removed_set(key))
        kept = len(payloads[key]["boxes"]) - len(rm)
        edit_status.content = (
            f"**{key}**: 保留 **{kept}** / {len(payloads[key]['boxes'])} box"
            + (f"\n\n已删索引: `{rm}`" if rm else "\n\n（未删任何 box）")
            + "\n\n点 3D 里的 box 或下面的勾选框即可删/恢复。删除会在下一步"
              "**按编辑后的 box 集重算全部保真指标并重装模板**。"
        )
        done = sum(1 for k in keys if verdicts.get(k, {}).get("review_decision"))
        approved = sum(1 for k in keys if verdicts.get(k, {}).get("review_decision") == "approve_clean")
        pending = [k for k in keys if not verdicts.get(k, {}).get("review_decision")]
        progress.content = (
            f"**{done}/{len(keys)} 已判** · approve_clean {approved}\n\n"
            + ("待判: " + ", ".join(pending) if pending else "✅ 全部判完")
        )

    def record(decision: str) -> None:
        key = state["key"]
        verdicts[key] = {
            "object_key": key,
            "review_decision": decision,
            "reviewer": args.reviewer,
            "reviewed_at": now(),
            "notes": notes.value,
        }
        if key in edits:
            edits[key]["notes"] = notes.value
            L.save_box_edits(edits)
        flush_reviews()
        refresh_info()
        rm = sorted(removed_set(key))
        print(f"[E206] {key} -> {decision}  removed={rm}  ({notes.value})", flush=True)
        remaining = [k for k in keys if not verdicts.get(k, {}).get("review_decision")]
        if remaining:
            picker.value = remaining[0]

    def reset_boxes(_) -> None:
        set_removed(state["key"], set())
        render()
        rebuild_box_list()
        refresh_info()

    btn_reset.on_click(reset_boxes)

    @picker.on_update
    def _(_) -> None:
        state["key"] = picker.value
        notes.value = verdicts.get(picker.value, {}).get("notes", "")
        render()
        rebuild_box_list()
        refresh_info()

    for ctl in (show_mesh, mesh_opacity, show_boxes, box_opacity, wireframe, show_floor):
        ctl.on_update(lambda _: render())

    btn_ok.on_click(lambda _: record("approve_clean"))
    btn_no.on_click(lambda _: record("needs_manual_edit"))

    render()
    rebuild_box_list()
    refresh_info()
    flush_reviews()

    print(f"\n[E206] 3D 复审已启动 -> http://{args.host}:{args.port}", flush=True)
    print(f"[E206] 判定写入 {review_path}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        flush_reviews()
        print("\n[E206] 已保存判定，退出", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
