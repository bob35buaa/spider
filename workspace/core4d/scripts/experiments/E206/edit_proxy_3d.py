#!/usr/bin/env python3
"""E206 P2.3b: free-form 3D editor for the object collision proxy.

`review_proxy_3d.py` only lets a reviewer DELETE auto-generated boxes.  That is
not enough for the objects that actually need help: chair005 and chair022 sit at
41% cavity over-fill because neither the voxel merge nor the 4-way leg split can
leave the under-seat gap open — there is no subset of those boxes that is
correct, so no amount of deleting fixes them.  The boxes have to be re-placed.

This tool lets you place them:

  * click a box to select it; drag the centre gizmo to move it, drag either
    corner gizmo to resize it (viser has no scale gizmo, so two corners define
    the AABB)
  * new / duplicate / delete / undo
  * ⇲ fit    — shrink the selected box onto the mesh actually inside it
  * ⇔ mirror — copy it across the object's X or Z mid-plane (chair legs, arms)
  * red dots show mesh area the proxy does NOT cover, so under-coverage is
    visible rather than inferred from a p90
  * fidelity + cavity numbers refresh after every drag, with the G1/G3/G4/G5
    bars from `audit_lowgeom_contract.py` as a live pass/fail

Edits are stored as absolute geometry in `s2_proxy/manual_boxes.json` and win
over both automatic paths (`semantic_proxy.build_effective_proxy`), so the
normal audit -> install chain picks them up unchanged.

Everything emitted is an axis-aligned box: `object_collision_sdf_mode='union'`
is fail-closed on non-box geoms (`spider/config.py:69-79`).

Run:
    .venv/bin/python workspace/core4d/scripts/experiments/E206/edit_proxy_3d.py \
        --reviewer <name> --port 8080
    # then open http://<host>:8080

Verdicts still go to `<run>/s2_templates/review/nonbox_template_review.tsv` in
the E145/E174 schema, so the S2 gate consumes them unchanged.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lowgeom_contract as A  # noqa: E402
import e206_common as C  # noqa: E402
import lowgeom_proxy_v2 as L  # noqa: E402
import manual_boxes as MB  # noqa: E402
import review_proxy_3d as R  # noqa: E402  (REVIEW_FIELDS / now / source_task)
import semantic_proxy as S  # noqa: E402

MESH_COLOR = (170, 170, 178)
SELECTED_COLOR = (255, 214, 64)
UNCOVERED_COLOR = (235, 64, 52)
BOX_COLORS = R.BOX_COLORS

UNDO_DEPTH = 30
MIN_HALF = MB.MIN_HALF_M
# Interactive sample budgets.  The exact contract numbers are produced by the
# audit; these only have to be stable enough to steer an edit.
LIVE_MESH_SAMPLES = 2_000
LIVE_FACE_SAMPLES = 16
LIVE_CAVITY_SAMPLES = 2_500
PER_BOX_SAMPLES = 250
UNCOVERED_SAMPLES = 4_000


@dataclass
class Box:
    center: np.ndarray
    half: np.ndarray
    label: str = ""

    def copy(self) -> Box:
        return Box(self.center.copy(), self.half.copy(), self.label)

    @property
    def lo(self) -> np.ndarray:
        return self.center - self.half

    @property
    def hi(self) -> np.ndarray:
        return self.center + self.half


@dataclass
class Work:
    """Per-object editing state."""

    mesh: trimesh.Trimesh
    mesh_path: Path
    surface: np.ndarray  # cached surface samples, for the uncovered-point cloud
    fit_points: np.ndarray  # surface samples + true vertices, for ⇲ fit
    boxes: list[Box]
    seed_name: str
    dirty: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    per_box_overfill: list[float] = field(default_factory=list)
    undo: list[list[Box]] = field(default_factory=list)


def to_proxy(boxes: list[Box]) -> list[L.ProxyBox]:
    return [L.ProxyBox(center=b.center.copy(), half_size=b.half.copy()) for b in boxes]


def signed_union_distance(points: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Signed distance to the box union: negative inside, positive outside.

    Deliberately NOT E176's `point_to_proxy_surface_distance`, which takes the
    absolute value — that answers "how far from the nearest box SURFACE", so a
    mesh point buried deep inside a large box scores as far away.  For the
    red "not covered" overlay we need "how far OUTSIDE the union", or every
    thick proxy would paint its own interior red.
    """
    best = np.full(points.shape[0], np.inf)
    for box in boxes:
        q = np.abs(points - box.center) - box.half
        signed = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
            np.max(q, axis=1), 0.0
        )
        best = np.minimum(best, signed)
    return best


# --------------------------------------------------------------------------
# Seeds — the automatic proxies are starting points, nothing more
# --------------------------------------------------------------------------
def installed_boxes(object_key: str) -> list[Box] | None:
    """The boxes currently written into the source template, if any."""
    scene = C.PROCESSED_ROOT / R.source_task(object_key) / "scene.xml"
    if not scene.exists():
        return None
    root = ET.parse(scene).getroot()
    body = next(
        (b for b in root.iter("body") if b.get("name") == "object"), None
    )
    if body is None:
        return None
    out: list[Box] = []
    for geom in body.findall("geom"):
        name = geom.get("name") or ""
        if not name.startswith("object_collision") or geom.get("type") != "box":
            continue
        pos = np.array([float(v) for v in (geom.get("pos") or "0 0 0").split()])
        size = np.array([float(v) for v in (geom.get("size") or "").split()])
        if size.shape != (3,):
            continue
        out.append(Box(pos, size, name.replace("object_collision", "inst") or "inst"))
    return out or None


EFFECTIVE_SEED = "effective(当前生效)"


def person_tasks(object_key: str) -> list[str]:
    """Every `<obj>_person{1,2}` template that exists on disk.

    The review TSV is keyed by `source_scene_task`, and `run_stage2b.py:77`
    flips `template_status` to `clean_reviewed` per TASK, not per object.
    `review_proxy_3d.source_task()` returns person1 only, so a one-row-per-object
    file silently held every person2 case at Stage2b — 34 of E206's 65 — while
    approving desk020_person1, which has no cases at all.  Both persons share the
    same object mesh and the same installed proxy, so one judgement covers both;
    it just has to be written to both rows.
    """
    return [
        f"{object_key}_{person}"
        for person in ("person1", "person2")
        if (C.PROCESSED_ROOT / f"{object_key}_{person}" / "scene.xml").exists()
    ] or [f"{object_key}_person1"]


def seed_names(object_key: str, contract: dict[str, dict[str, str]]) -> list[str]:
    # "effective" is what `build_effective_proxy` returns right now — manual if
    # one exists, else semantic, else voxel WITH the recorded deletions applied.
    # Starting anywhere else would silently discard prior review work.
    names: list[str] = [EFFECTIVE_SEED]
    if object_key in S.SEMANTIC_SPEC:
        names.append("semantic")
    row = contract.get(object_key, {})
    tc = (row.get("target_cells") or "").lstrip("-")
    if tc.isdigit() and int(row["target_cells"]) > 0:
        names.append(f"voxel(tc={row['target_cells']})")
    if installed_boxes(object_key) is not None:
        names.append("installed")
    names.append("空白(单个全局 AABB)")
    return names


def load_seed(
    object_key: str,
    name: str,
    n_max: int,
    contract: dict[str, dict[str, str]],
    mesh: trimesh.Trimesh,
) -> list[Box]:
    if name == EFFECTIVE_SEED:
        row = contract.get(object_key, {})
        tc_raw = (row.get("target_cells") or "").lstrip("-")
        tc = int(row["target_cells"]) if tc_raw.isdigit() and int(row["target_cells"]) > 0 else None
        boxes, meta = S.build_effective_proxy(object_key, n_max, tc, measure_metrics=False)
        parts = list(meta.get("parts") or [])
        return [
            Box(b.center.copy(), b.half_size.copy(), parts[i] if i < len(parts) else f"b{i:02d}")
            for i, b in enumerate(boxes)
        ]
    if name == "semantic":
        boxes, meta = S.build_semantic_boxes(object_key)
        parts = list(meta.get("parts", []))
        return [
            Box(b.center.copy(), b.half_size.copy(), parts[i] if i < len(parts) else f"p{i:02d}")
            for i, b in enumerate(boxes)
        ]
    if name.startswith("voxel(tc="):
        tc = int(name.split("=")[1].rstrip(")"))
        boxes, _pitch = L._boxes_at(C.object_mesh_path(object_key), tc, n_max)
        return [Box(b.center.copy(), b.half_size.copy(), f"v{i:02d}") for i, b in enumerate(boxes)]
    if name == "installed":
        got = installed_boxes(object_key)
        if got is None:
            raise ValueError(f"{object_key}: no installed template")
        return got
    lo, hi = mesh.bounds
    return [Box((lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, MIN_HALF), "all")]


def default_seed(object_key: str, options: list[str]) -> str:
    return EFFECTIVE_SEED if EFFECTIVE_SEED in options else options[0]


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def support_info(work: Work) -> dict[str, Any]:
    return L.support_contact_metrics(
        work.mesh_path, to_proxy(work.boxes), [b.label for b in work.boxes]
    )


def measure_live(work: Work) -> tuple[dict[str, Any], list[float]]:
    """Fidelity + cavity on the current boxes, fast enough to run per drag."""
    proxy = to_proxy(work.boxes)
    out: dict[str, Any] = dict(
        L.fidelity_metrics_fast(
            work.mesh_path,
            proxy,
            mesh_sample_count=LIVE_MESH_SAMPLES,
            proxy_samples_per_face=LIVE_FACE_SAMPLES,
        )
    )
    out.update(
        L.cavity_metrics(
            work.mesh_path,
            proxy,
            np.array([0.02, 0.02, 0.02]),
            sample_count=LIVE_CAVITY_SAMPLES,
            fast=True,
        )
    )
    # Per-box over-fill: which single box is filling a cavity?  A union-level
    # number cannot say, and that is exactly what the reviewer needs to know.
    per_box: list[float] = []
    for box in work.boxes:
        pts = L._sample_inside_boxes([L.ProxyBox(box.center, box.half)], PER_BOX_SAMPLES, seed=0)
        dist = L.mesh_surface_distance(work.mesh, pts, fast=True)
        per_box.append(float(np.mean(dist > 0.05)))
    return out, per_box


def gate_status(
    metrics: dict[str, Any],
    n_boxes: int,
    n_max: int,
    support: dict[str, Any] | None = None,
) -> list[tuple[str, bool, str]]:
    bars = A.BARS
    extra: list[tuple[str, bool, str]] = []
    if support is not None:
        worst = support["support_bottom_worst_abs_m"]
        extra = [
            (
                "G10 支撑面共面",
                support["support_box_count"] >= 1 and worst <= bars["G10_support_offset_max"],
                f"{support['support_box_count']} 个支撑 box · 高低差 "
                f"{support['support_bottom_spread_m'] * 1000:.1f} mm · 最大离地 "
                f"{worst * 1000:.1f} mm ≤ {bars['G10_support_offset_max'] * 1000:.0f}",
            )
        ]
    return extra + [
        ("G1 box 数", 1 <= n_boxes <= n_max, f"{n_boxes} / {n_max}"),
        (
            "G3 mesh→proxy p90",
            metrics["mesh_to_proxy_p90_m"] <= bars["G3_mesh_to_proxy_p90_max"],
            f"{metrics['mesh_to_proxy_p90_m']:.3f} ≤ {bars['G3_mesh_to_proxy_p90_max']}",
        ),
        (
            "G4 mesh→proxy max",
            metrics["mesh_to_proxy_max_m"] <= bars["G4_mesh_to_proxy_max_max"],
            f"{metrics['mesh_to_proxy_max_m']:.3f} ≤ {bars['G4_mesh_to_proxy_max_max']}",
        ),
        (
            "G5 proxy→mesh p90",
            metrics["proxy_to_mesh_p90_m"] <= bars["G5_proxy_to_mesh_p90_max"],
            f"{metrics['proxy_to_mesh_p90_m']:.3f} ≤ {bars['G5_proxy_to_mesh_p90_max']}",
        ),
        (
            "G6 腔体过填",
            metrics["interior_overfill_frac_5cm"] <= bars["G6_overfill_warn"],
            f"{metrics['interior_overfill_frac_5cm']:.0%} ≤ {bars['G6_overfill_warn']:.0%}（可豁免）",
        ),
    ]


def main() -> int:  # noqa: PLR0915 - a single-screen GUI is one cohesive unit
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-max", type=int, default=C.N_MAX_TARGET)
    ap.add_argument("--reviewer", default="user")
    ap.add_argument("--out-dir", type=Path, default=C.S2_TEMPLATE_DIR / "review")
    ap.add_argument("--object", default="", help="open just this object")
    args = ap.parse_args()

    import viser

    contract_path = C.S2_PROXY_DIR / f"lowgeom_contract_n{args.n_max}.tsv"
    contract: dict[str, dict[str, str]] = {}
    if contract_path.exists():
        contract = {r["object_key"]: r for r in C.read_tsv(contract_path)}
    else:
        print(f"[E206] no {contract_path}; running without contract columns", flush=True)

    if args.object:
        keys = [k.strip() for k in args.object.split(",") if k.strip()]
    else:
        # Every in-scope object, including ones whose auto build fails — those
        # are precisely the ones that need hand editing.
        keys = [k for k in C.OBJECT_KEYS if C.object_mesh_path(k).exists()]
        landed = set(A.landed_object_keys())
        keys = [k for k in keys if k in landed] or keys
    if not keys:
        raise SystemExit("no objects to edit")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.out_dir / "nonbox_template_review.tsv"
    verdicts: dict[str, dict[str, str]] = {}
    # Rows for objects this session is not editing (e.g. launched with
    # --object): they must survive verbatim, or a single-object run would
    # silently drop every other object's verdict from the S2 gate's input.
    foreign_rows: dict[str, dict[str, str]] = {}
    if review_path.exists():
        for r in C.read_tsv(review_path):
            verdicts[r["object_key"]] = r
            if r["object_key"] not in keys:
                # Keyed by task, not object: an object has one row per person.
                foreign_rows[r.get("source_scene_task") or r["object_key"]] = r

    lock = threading.RLock()
    work: dict[str, Work] = {}

    def get_work(key: str) -> Work:
        if key in work:
            return work[key]
        mesh_path = C.object_mesh_path(key)
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        options = seed_names(key, contract)
        seed = default_seed(key, options)
        try:
            boxes = load_seed(key, seed, args.n_max, contract, mesh)
        except Exception as exc:  # noqa: BLE001 - fall back to a usable canvas
            print(f"[E206] {key}: seed '{seed}' failed ({exc}); starting from AABB", flush=True)
            seed = "空白(单个全局 AABB)"
            boxes = load_seed(key, seed, args.n_max, contract, mesh)
        state_ = np.random.get_state()
        try:
            np.random.seed(0)
            surface, _ = trimesh.sample.sample_surface(mesh, UNCOVERED_SAMPLES)
        finally:
            np.random.set_state(state_)
        surface = np.asarray(surface, dtype=np.float64)
        w = Work(
            mesh=mesh,
            mesh_path=mesh_path,
            surface=surface,
            # Vertices are exact geometry; a thin chair leg may catch only a
            # handful of the 4k surface samples, which would make ⇲ fit jittery.
            fit_points=np.vstack([surface, np.asarray(mesh.vertices, dtype=np.float64)]),
            boxes=boxes,
            seed_name=seed,
            # `dirty` == "this object has a hand-authored record on disk".
            dirty=MB.manual_boxes_for(key) is not None,
        )
        w.metrics, w.per_box_overfill = measure_live(w)
        work[key] = w
        return w

    print(f"[E206] preparing {len(keys)} object(s) ...", flush=True)
    get_work(keys[0])

    server = viser.ViserServer(host=args.host, port=args.port)
    # Object-local +Y is up for every CORE4D desk/chair (semantic_proxy.py:37).
    # review_proxy_3d.py set "+z", which renders every object lying on its side.
    server.scene.set_up_direction("+y")

    state: dict[str, Any] = {"key": keys[0], "sel": None, "guard": False}
    mesh_handle: list[Any] = []
    box_handles: list[Any] = []
    gizmos: list[Any] = []
    cloud_handle: list[Any] = []
    grid_handle: list[Any] = []
    row_handles: list[Any] = []

    # ---- UI -------------------------------------------------------------
    with server.gui.add_folder("物体"):
        picker = server.gui.add_dropdown("object", tuple(keys), initial_value=keys[0])
        info = server.gui.add_markdown("")
    with server.gui.add_folder("显示"):
        show_mesh = server.gui.add_checkbox("真实 mesh", True)
        mesh_opacity = server.gui.add_slider("mesh 透明度", 0.05, 1.0, 0.05, 0.30)
        show_boxes = server.gui.add_checkbox("碰撞 box", True)
        box_opacity = server.gui.add_slider("box 透明度", 0.05, 1.0, 0.05, 0.45)
        wireframe = server.gui.add_checkbox("box 线框", False)
        show_uncovered = server.gui.add_checkbox("未覆盖 mesh 点(红)", True)
        uncovered_thr = server.gui.add_slider("未覆盖阈值 m", 0.01, 0.15, 0.005, 0.05)
        show_floor = server.gui.add_checkbox("地面网格", True)
    with server.gui.add_folder("种子"):
        seed_pick = server.gui.add_dropdown("seed", tuple(seed_names(keys[0], contract)))
        btn_seed = server.gui.add_button("↻ 用该种子重置（丢弃当前编辑）")
    with server.gui.add_folder("碰撞体编辑"):
        sel_md = server.gui.add_markdown("")
        btn_new = server.gui.add_button("＋ 新建 box")
        btn_dup = server.gui.add_button("⧉ 复制选中")
        btn_del = server.gui.add_button("🗑 删除选中")
        btn_undo = server.gui.add_button("↶ 撤销")
        btn_fit = server.gui.add_button("⇲ 贴合（收到内部 mesh）")
        btn_mx = server.gui.add_button("⇔ 镜像 X")
        btn_mz = server.gui.add_button("⇔ 镜像 Z")
        btn_align = server.gui.add_button("⊻ 对齐支撑面（腿底踩到地）")
        label_in = server.gui.add_text("label", "")
        num_cx = server.gui.add_number("center x", 0.0, step=0.005)
        num_cy = server.gui.add_number("center y", 0.0, step=0.005)
        num_cz = server.gui.add_number("center z", 0.0, step=0.005)
        num_hx = server.gui.add_number("half x", 0.05, min=MIN_HALF, step=0.005)
        num_hy = server.gui.add_number("half y", 0.05, min=MIN_HALF, step=0.005)
        num_hz = server.gui.add_number("half z", 0.05, min=MIN_HALF, step=0.005)
        box_folder = server.gui.add_folder("box 列表（点击选中）")
    with server.gui.add_folder("判定"):
        notes = server.gui.add_text("notes", "")
        btn_exact = server.gui.add_button("🎯 全精度重算")
        btn_ok = server.gui.add_button("✅ approve_clean")
        btn_no = server.gui.add_button("❌ needs_manual_edit")
        btn_drop = server.gui.add_button("⟲ 丢弃手编，回到自动代理")
        progress = server.gui.add_markdown("")

    # ---- persistence ----------------------------------------------------
    def persist(key: str) -> None:
        w = work[key]
        MB.record_manual(
            key,
            to_proxy(w.boxes),
            [b.label for b in w.boxes],
            seed={"kind": w.seed_name},
            editor=args.reviewer,
            notes=notes.value,
            n_max=args.n_max,
        )
        w.dirty = True

    def push_undo(key: str) -> None:
        w = work[key]
        w.undo.append([b.copy() for b in w.boxes])
        del w.undo[:-UNDO_DEPTH]

    def flush_reviews() -> None:
        rows = [foreign_rows[k] for k in sorted(foreign_rows)]
        for k in keys:
            r = contract.get(k, {})
            v = verdicts.get(k, {})
            w = work.get(k)
            m = w.metrics if w else {}

            def pick(live_key: str, row_key: str) -> str:
                if m:
                    return f"{float(m[live_key]):.6f}"
                return r.get(row_key, "")

            # One row per person template — see person_tasks().
            for task in person_tasks(k):
                rows.append(
                    {
                        "object_key": k,
                        "source_scene_task": task,
                        "object_category": C.object_category(k),
                        "template_status": (
                            "clean_reviewed"
                            if v.get("review_decision") == "approve_clean"
                            else "manual_review_required"
                        ),
                        "recommended_action": "review_mesh_collision_overlay",
                        "proxy_variant": f"{C.object_category(k)}_lowgeom{args.n_max}",
                        "proxy_policy": f"{C.object_category(k)}_lowgeom{args.n_max}_proxy",
                        "mesh_path": str(C.object_mesh_path(k).relative_to(C.REPO)),
                        "scene_xml": (
                            "example_datasets/processed/core4d/unitree_g1/humanoid_object/"
                            f"{task}/scene.xml"
                        ),
                        "object_geom_count": str(len(w.boxes)) if w else r.get("object_geom_count", ""),
                        "target_cells": "n/a" if (w and w.dirty) else r.get("target_cells", ""),
                        "mesh_to_proxy_p90_m": pick("mesh_to_proxy_p90_m", "mesh_to_proxy_p90_m"),
                        "proxy_to_mesh_p90_m": pick("proxy_to_mesh_p90_m", "proxy_to_mesh_p90_m"),
                        "interior_overfill_frac_5cm": pick(
                            "interior_overfill_frac_5cm", "interior_overfill_frac_5cm"
                        ),
                        "contract_status": (
                            "pass"
                            if all(
                                ok
                                for name, ok, _ in gate_status(
                                    m, len(w.boxes), args.n_max, support_info(w)
                                )
                                # G6 is report-and-waive, never a hard gate.
                                if not name.startswith("G6")
                            )
                            else "gate_fail"
                        )
                        if m and w
                        else r.get("failed_gates", ""),
                        "waived_checks": (
                            "G6_overfill"
                            if m and float(m["interior_overfill_frac_5cm"]) > A.BARS["G6_overfill_warn"]
                            else ("G6_overfill" if r.get("G6_needs_waiver") == "true" else "")
                        ),
                        # Hand-authored proxies have no "removed index" notion; the
                        # box set itself is the record.
                        "boxes_removed": "",
                        "object_geom_count_final": str(len(w.boxes))
                        if w
                        else r.get("object_geom_count", ""),
                        "review_decision": v.get("review_decision", ""),
                        "reviewer": v.get("reviewer", ""),
                        "reviewed_at": v.get("reviewed_at", ""),
                        "notes": v.get("notes", ""),
                    }
                )
        rows.sort(key=lambda r: r["source_scene_task"])
        C.write_tsv(review_path, rows, R.REVIEW_FIELDS)

    # ---- rendering ------------------------------------------------------
    def clear(handles: list[Any]) -> None:
        for h in handles:
            h.remove()
        handles.clear()

    def render_mesh() -> None:
        clear(mesh_handle)
        if not show_mesh.value:
            return
        w = work[state["key"]]
        mesh_handle.append(
            server.scene.add_mesh_simple(
                "/mesh",
                vertices=np.asarray(w.mesh.vertices),
                faces=np.asarray(w.mesh.faces),
                color=MESH_COLOR,
                opacity=float(mesh_opacity.value),
                material="standard",
            )
        )

    def render_floor() -> None:
        clear(grid_handle)
        if not show_floor.value:
            return
        w = work[state["key"]]
        ext = float(np.max(w.mesh.extents))
        grid_handle.append(
            server.scene.add_grid(
                "/grid",
                width=ext * 2.5,
                height=ext * 2.5,
                plane="xz",
                position=(0.0, float(w.mesh.bounds[0][1]), 0.0),
            )
        )

    def render_boxes() -> None:
        clear(box_handles)
        if not show_boxes.value:
            return
        w = work[state["key"]]
        for i, b in enumerate(w.boxes):
            selected = i == state["sel"]
            handle = server.scene.add_box(
                f"/box/{i:03d}",
                color=SELECTED_COLOR if selected else BOX_COLORS[i % len(BOX_COLORS)],
                dimensions=tuple(2.0 * b.half),
                position=tuple(b.center),
                opacity=min(1.0, float(box_opacity.value) + (0.2 if selected else 0.0)),
                wireframe=bool(wireframe.value),
            )
            handle.on_click(lambda _, idx=i: select(idx))
            box_handles.append(handle)

    def render_cloud() -> None:
        clear(cloud_handle)
        if not show_uncovered.value:
            return
        w = work[state["key"]]
        dist = signed_union_distance(w.surface, w.boxes)
        pts = w.surface[dist > float(uncovered_thr.value)]
        if pts.shape[0] == 0:
            return
        cloud_handle.append(
            server.scene.add_point_cloud(
                "/uncovered",
                points=pts,
                colors=np.tile(np.array(UNCOVERED_COLOR, dtype=np.uint8), (pts.shape[0], 1)),
                point_size=0.006,
            )
        )

    def render_gizmos() -> None:
        clear(gizmos)
        idx = state["sel"]
        if idx is None:
            return
        w = work[state["key"]]
        if idx >= len(w.boxes):
            return
        b = w.boxes[idx]
        span = float(np.max(b.half))
        centre = server.scene.add_transform_controls(
            "/gizmo/center",
            scale=max(0.10, span * 1.4),
            disable_rotations=True,
            depth_test=False,
            position=tuple(b.center),
        )
        lo = server.scene.add_transform_controls(
            "/gizmo/lo",
            scale=max(0.05, span * 0.6),
            disable_rotations=True,
            disable_sliders=True,
            depth_test=False,
            position=tuple(b.lo),
        )
        hi = server.scene.add_transform_controls(
            "/gizmo/hi",
            scale=max(0.05, span * 0.6),
            disable_rotations=True,
            disable_sliders=True,
            position=tuple(b.hi),
        )
        centre.on_update(lambda _: on_gizmo("center"))
        lo.on_update(lambda _: on_gizmo("lo"))
        hi.on_update(lambda _: on_gizmo("hi"))
        for g in (centre, lo, hi):
            g.on_drag_end(lambda _: commit_drag())
        gizmos.extend([centre, lo, hi])

    def sync_gizmos() -> None:
        """Push the box geometry back onto the three gizmos."""
        idx = state["sel"]
        if idx is None or len(gizmos) != 3:
            return
        b = work[state["key"]].boxes[idx]
        state["guard"] = True
        try:
            gizmos[0].position = tuple(b.center)
            gizmos[1].position = tuple(b.lo)
            gizmos[2].position = tuple(b.hi)
        finally:
            state["guard"] = False

    def sync_numbers() -> None:
        idx = state["sel"]
        if idx is None:
            return
        b = work[state["key"]].boxes[idx]
        state["guard"] = True
        try:
            num_cx.value, num_cy.value, num_cz.value = (float(v) for v in b.center)
            num_hx.value, num_hy.value, num_hz.value = (float(v) for v in b.half)
            label_in.value = b.label
        finally:
            state["guard"] = False

    def update_selected_visual() -> None:
        """Cheap per-frame update while dragging: move the one box that moved."""
        idx = state["sel"]
        if idx is None or idx >= len(box_handles):
            return
        b = work[state["key"]].boxes[idx]
        box_handles[idx].position = tuple(b.center)
        box_handles[idx].dimensions = tuple(2.0 * b.half)

    # ---- edit operations -------------------------------------------------
    def on_gizmo(which: str) -> None:
        if state["guard"] or state["sel"] is None or len(gizmos) != 3:
            return
        with lock:
            w = work[state["key"]]
            b = w.boxes[state["sel"]]
            if which == "center":
                b.center = np.asarray(gizmos[0].position, dtype=np.float64)
            else:
                lo = np.asarray(gizmos[1].position if which == "lo" else b.lo, dtype=np.float64)
                hi = np.asarray(gizmos[2].position if which == "hi" else b.hi, dtype=np.float64)
                # Keep the dragged corner authoritative and never let the box
                # invert: clamp the opposite corner instead.
                if which == "lo":
                    hi = np.maximum(hi, lo + 2.0 * MIN_HALF)
                else:
                    lo = np.minimum(lo, hi - 2.0 * MIN_HALF)
                b.center = (lo + hi) / 2.0
                b.half = np.maximum((hi - lo) / 2.0, MIN_HALF)
            update_selected_visual()
            sync_gizmos()
            sync_numbers()

    def commit_drag() -> None:
        """Drag finished — now pay for the metrics and write to disk."""
        with lock:
            key = state["key"]
            persist(key)
            refresh_metrics()

    def refresh_metrics() -> None:
        w = work[state["key"]]
        w.metrics, w.per_box_overfill = measure_live(w)
        render_cloud()
        rebuild_rows()
        refresh_info()

    def select(idx: int | None) -> None:
        with lock:
            state["sel"] = idx
            render_boxes()
            render_gizmos()
            sync_numbers()
            rebuild_rows()
            refresh_info()

    def mutate(fn) -> None:
        """Run an edit, then persist + re-render + re-measure."""
        with lock:
            key = state["key"]
            push_undo(key)
            fn(work[key])
            persist(key)
            render_boxes()
            render_gizmos()
            sync_numbers()
            refresh_metrics()

    def op_new(w: Work) -> None:
        lo, hi = w.mesh.bounds
        w.boxes.append(Box((lo + hi) / 2.0, np.full(3, 0.05), f"m{len(w.boxes):02d}"))
        state["sel"] = len(w.boxes) - 1

    def op_dup(w: Work) -> None:
        if state["sel"] is None:
            return
        src = w.boxes[state["sel"]]
        w.boxes.insert(state["sel"] + 1, src.copy())
        state["sel"] = state["sel"] + 1

    def op_del(w: Work) -> None:
        if state["sel"] is None or len(w.boxes) <= 1:
            print("[E206] refusing to delete the last box", flush=True)
            return
        w.boxes.pop(state["sel"])
        state["sel"] = min(state["sel"], len(w.boxes) - 1)

    def op_fit(w: Work) -> None:
        """Shrink the selected box onto the mesh surface actually inside it.

        The workflow this enables: drop a rough box over a chair leg, hit fit,
        get a tight one — instead of dialling six numbers by hand.
        """
        if state["sel"] is None:
            return
        b = w.boxes[state["sel"]]
        inside = w.fit_points[
            np.all((w.fit_points >= b.lo) & (w.fit_points <= b.hi), axis=1)
        ]
        if inside.shape[0] < 8:
            print("[E206] fit: fewer than 8 mesh samples inside the box, leaving it alone", flush=True)
            return
        lo, hi = inside.min(axis=0), inside.max(axis=0)
        b.center = (lo + hi) / 2.0
        b.half = np.maximum((hi - lo) / 2.0, MIN_HALF)

    def op_mirror(axis: int):
        def run(w: Work) -> None:
            if state["sel"] is None:
                return
            src = w.boxes[state["sel"]]
            mid = float((w.mesh.bounds[0][axis] + w.mesh.bounds[1][axis]) / 2.0)
            new = src.copy()
            new.center[axis] = 2.0 * mid - src.center[axis]
            new.label = f"{src.label}_m"
            w.boxes.insert(state["sel"] + 1, new)
            state["sel"] = state["sel"] + 1

        return run

    def op_align(w: Work) -> None:
        """Drop every load-bearing box onto the floor, tops held fixed (G10)."""
        aligned, changes = L.align_support_boxes(w.mesh_path, to_proxy(w.boxes))
        for i, box in enumerate(aligned):
            w.boxes[i].center = box.center
            w.boxes[i].half = box.half_size
        for ch in changes:
            lab = w.boxes[ch["index"]].label
            print(f"[E206] align {lab}: bottom {ch['delta_mm']:+.1f} mm", flush=True)
        if not changes:
            print("[E206] align: 已经贴地，无需改动", flush=True)

    def op_undo(_) -> None:
        with lock:
            key = state["key"]
            w = work[key]
            if not w.undo:
                print("[E206] nothing to undo", flush=True)
                return
            w.boxes = w.undo.pop()
            state["sel"] = min(state["sel"] or 0, len(w.boxes) - 1)
            persist(key)
            render_boxes()
            render_gizmos()
            sync_numbers()
            refresh_metrics()

    def on_number(_) -> None:
        if state["guard"] or state["sel"] is None:
            return
        with lock:
            b = work[state["key"]].boxes[state["sel"]]
            b.center = np.array([num_cx.value, num_cy.value, num_cz.value], dtype=np.float64)
            b.half = np.maximum(
                np.array([num_hx.value, num_hy.value, num_hz.value], dtype=np.float64), MIN_HALF
            )
            persist(state["key"])
            update_selected_visual()
            sync_gizmos()
            refresh_metrics()

    def on_label(_) -> None:
        if state["guard"] or state["sel"] is None:
            return
        with lock:
            work[state["key"]].boxes[state["sel"]].label = label_in.value
            persist(state["key"])
            rebuild_rows()

    # ---- panels ---------------------------------------------------------
    def rebuild_rows() -> None:
        clear(row_handles)
        w = work[state["key"]]
        floor = float(w.mesh.bounds[0][L.UP_AXIS])
        with box_folder:
            for i, b in enumerate(w.boxes):
                dims = 2.0 * b.half
                of = w.per_box_overfill[i] if i < len(w.per_box_overfill) else float("nan")
                mark = "▶" if i == state["sel"] else " "
                gap_mm = (float(b.center[L.UP_AXIS] - b.half[L.UP_AXIS]) - floor) * 1000.0
                # Only annotate boxes near the floor: a seat's "gap" is noise.
                gap = f" 离地{gap_mm:+.0f}mm" if gap_mm <= L.SUPPORT_BAND_M * 1000.0 else ""
                btn = server.gui.add_button(
                    f"{mark}{i:02d} {b.label or '-':7s} "
                    f"{dims[0]:.2f}×{dims[1]:.2f}×{dims[2]:.2f} 空{of:.0%}{gap}"
                )
                btn.on_click(lambda _, idx=i: select(idx))
                row_handles.append(btn)

    def refresh_info() -> None:
        key = state["key"]
        w = work[key]
        m = w.metrics
        sup = support_info(w)
        gates = gate_status(m, len(w.boxes), args.n_max, sup)
        table = "\n".join(f"| {'✅' if ok else '❌'} {n} | {d} |" for n, ok, d in gates)
        floating = (
            f"\n\n> ⚠️ 标为 leg 但离地 >{L.SUPPORT_BAND_M * 1000:.0f}mm（不参与对齐）: "
            f"`{', '.join(sup['floating_leg_labels'])}`"
            if sup["floating_leg_labels"]
            else ""
        )
        v = verdicts.get(key, {})
        info.content = (
            f"### {key} ({C.object_category(key)})\n\n"
            f"种子 `{w.seed_name}` · {'**已手编**' if w.dirty else '未编辑（仍走自动代理）'}\n\n"
            f"| 门 | 值 |\n|---|---|\n{table}\n"
            f"| mesh→proxy p50 | {m['mesh_to_proxy_p50_m']:.3f} m |\n"
            f"| proxy→mesh max | {m['proxy_to_mesh_max_m']:.3f} m |\n"
            f"| 当前判定 | **{v.get('review_decision') or '（未判）'}** |\n"
            + floating
            + "\n\n_数字为交互口径（降采样 + R-tree）。合同数字以 audit 为准，"
            "可点「🎯 全精度重算」核对。_"
        )
        idx = state["sel"]
        if idx is None:
            sel_md.content = "**未选中** — 点 3D 里的 box 或下面的列表选中它。"
        else:
            b = w.boxes[idx]
            of = w.per_box_overfill[idx] if idx < len(w.per_box_overfill) else float("nan")
            sel_md.content = (
                f"**选中 #{idx} `{b.label or '-'}`** · 该 box 内部离 mesh >5cm 的比例 **{of:.0%}**\n\n"
                "拖中心 gizmo 移动；拖两个角 gizmo 改尺寸；`⇲ 贴合` 收到内部 mesh。"
            )
        done = sum(1 for k in keys if verdicts.get(k, {}).get("review_decision"))
        approved = sum(
            1 for k in keys if verdicts.get(k, {}).get("review_decision") == "approve_clean"
        )
        pending = [k for k in keys if not verdicts.get(k, {}).get("review_decision")]
        edited = sorted(MB.load_manual())
        progress.content = (
            f"**{done}/{len(keys)} 已判** · approve_clean {approved}\n\n"
            + ("待判: " + ", ".join(pending) + "\n\n" if pending else "✅ 全部判完\n\n")
            + (f"手编记录: `{', '.join(edited)}`" if edited else "手编记录: （无）")
        )

    # ---- verdicts / seeds / object switch --------------------------------
    def record(decision: str) -> None:
        key = state["key"]
        verdicts[key] = {
            "object_key": key,
            "review_decision": decision,
            "reviewer": args.reviewer,
            "reviewed_at": R.now(),
            "notes": notes.value,
        }
        if work[key].dirty:
            persist(key)
        flush_reviews()
        refresh_info()
        print(
            f"[E206] {key} -> {decision}  boxes={len(work[key].boxes)}  ({notes.value})",
            flush=True,
        )
        remaining = [k for k in keys if not verdicts.get(k, {}).get("review_decision")]
        if remaining:
            picker.value = remaining[0]

    def exact_recompute(_) -> None:
        with lock:
            key = state["key"]
            w = work[key]
            print(f"[E206] {key}: exact recompute ...", flush=True)
            w.metrics = S.measure(key, to_proxy(w.boxes))
            render_cloud()
            refresh_info()
            print(
                f"[E206] {key}: m2p_p90={w.metrics['mesh_to_proxy_p90_m']:.4f} "
                f"p2m_p90={w.metrics['proxy_to_mesh_p90_m']:.4f} "
                f"of5={w.metrics['interior_overfill_frac_5cm']:.3f}",
                flush=True,
            )

    def drop_manual(_) -> None:
        with lock:
            key = state["key"]
            MB.drop_manual(key)
            work.pop(key, None)
            switch(key)
            print(f"[E206] {key}: manual record dropped, back to the auto proxy", flush=True)

    def apply_seed(_) -> None:
        with lock:
            key = state["key"]
            w = work[key]
            push_undo(key)
            try:
                w.boxes = load_seed(key, seed_pick.value, args.n_max, contract, w.mesh)
            except Exception as exc:  # noqa: BLE001
                print(f"[E206] seed '{seed_pick.value}' failed: {exc}", flush=True)
                return
            w.seed_name = seed_pick.value
            state["sel"] = None
            # A seed load alone is not an edit: writing it out would silently
            # freeze this object onto a manual copy of the auto proxy.
            if w.dirty:
                persist(key)
            render_boxes()
            render_gizmos()
            refresh_metrics()

    def switch(key: str) -> None:
        with lock:
            state["key"] = key
            state["sel"] = None
            w = get_work(key)
            options = seed_names(key, contract)
            seed_pick.options = tuple(options)
            seed_pick.value = w.seed_name if w.seed_name in options else options[0]
            notes.value = verdicts.get(key, {}).get("notes", "")
            render_mesh()
            render_floor()
            render_boxes()
            render_gizmos()
            render_cloud()
            rebuild_rows()
            refresh_info()

    # ---- wiring ----------------------------------------------------------
    picker.on_update(lambda _: switch(picker.value))
    btn_seed.on_click(apply_seed)
    btn_new.on_click(lambda _: mutate(op_new))
    btn_dup.on_click(lambda _: mutate(op_dup))
    btn_del.on_click(lambda _: mutate(op_del))
    btn_undo.on_click(op_undo)
    btn_fit.on_click(lambda _: mutate(op_fit))
    btn_mx.on_click(lambda _: mutate(op_mirror(0)))
    btn_mz.on_click(lambda _: mutate(op_mirror(2)))
    btn_align.on_click(lambda _: mutate(op_align))
    for ctl in (num_cx, num_cy, num_cz, num_hx, num_hy, num_hz):
        ctl.on_update(on_number)
    label_in.on_update(on_label)
    for ctl in (show_mesh, mesh_opacity):
        ctl.on_update(lambda _: render_mesh())
    for ctl in (show_boxes, box_opacity, wireframe):
        ctl.on_update(lambda _: (render_boxes(), render_gizmos()))
    for ctl in (show_uncovered, uncovered_thr):
        ctl.on_update(lambda _: render_cloud())
    show_floor.on_update(lambda _: render_floor())
    btn_exact.on_click(exact_recompute)
    btn_ok.on_click(lambda _: record("approve_clean"))
    btn_no.on_click(lambda _: record("needs_manual_edit"))
    btn_drop.on_click(drop_manual)

    switch(keys[0])
    flush_reviews()

    print(f"\n[E206] 碰撞体编辑器已启动 -> http://{args.host}:{args.port}", flush=True)
    print(f"[E206] 手编 box 写入 {MB.MANUAL_PATH}", flush=True)
    print(f"[E206] 判定写入 {review_path}", flush=True)
    print("[E206] 编辑完成后跑: workspace/core4d/scripts/experiments/E206/refreeze_after_edit.sh", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        flush_reviews()
        print("\n[E206] 已保存，退出", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
