#!/usr/bin/env python3
"""Interactive viser review player for PRG retargeting results (E170-E194).

Browse every CEM-complete case across the registered experiments, filter by object /
numeric pass / failure mode / retarget variant, play back the executed 3D
trajectory (robot + object, MuJoCo), and annotate sample quality online. The
annotation writes ``USE / DO_NOT_USE`` decisions into a non-destructive
``user_manual_review_filled.tsv`` per experiment (the ``..._template.tsv`` is
never touched).

Reuses:
  * spider.viewers.viser_viewer  — stateless geom->trimesh helpers only.
  * E168 render_a100_cem_videos   — config + npz load path (rollout_qpos,
    converted_reference_qpos, load_render_config).
  * holosoma viser_player pattern — swap-safe playback with the
    ``updating_programmatically`` slider guard.

We deliberately do NOT use viser_viewer's build_and_log_scene_from_spec /
log_frame: those track handles in a module-global singleton and append to a
never-reset timeline, which is not safe for case switching. Instead we own a
private ViserServer and rebuild the scene under a fresh root on every swap.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
_HERE = Path(__file__).resolve().parent
for _p in (
    str(REPO),
    str(_HERE),
    str(REPO / "workspace/core4d/scripts/experiments/E168"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import review_index as idx  # noqa: E402

USE_DECISIONS = ("PENDING", "USE", "DO_NOT_USE")
QUALITY_LABELS = ("", "CLEAN", "MINOR_ACCEPTABLE", "MAJOR_DEFECT", "UNUSABLE")


def case_name_matches(case_id: str, query: str) -> bool:
    """Case-insensitive substring search; whitespace separates required tokens."""
    normalized = case_id.casefold()
    return all(token in normalized for token in query.casefold().split())

# top metrics bar: (metric column, 中文, threshold key, direction, unit)
#   direction "max" → red if value > threshold; "min" → red if value < threshold;
#   "flag" → red if value >= 0.5 (fall).
METRIC_BAR = (
    ("body_z_err_p95_m", "身高误差", "body_z_err_p95_m_max", "max", "m"),
    ("leg_penetration_frac", "腿穿透", "leg_penetration_max", "max", ""),
    (
        "hand_object_physics_penetration_3mm_frame_frac",
        "手穿透",
        "hand_penetration_3mm_max",
        "max",
        "",
    ),
    (
        "hand_object_release_false_contact_3mm_frac",
        "释放误触",
        "release_false_3mm_max",
        "max",
        "",
    ),
    (
        "hand_object_physics_contact_3mm_in_mask_frac",
        "接触率",
        "raw_contact_min",
        "min",
        "",
    ),
    ("fall_flag", "摔倒", "", "flag", ""),
)

# tracking-error bar: (metric column, 中文, threshold, unit, decimals).
# These red lines mirror E178's numeric gates. Legacy experiments may still use
# them as display-only thresholds because pass/fail always comes from their TSV.
TRACK_BAR = (
    (
        "track_root_pos_err_cm_mean",
        "根位",
        float(os.environ.get("CORE4D_REVIEW_ROOT_POS_MAX_CM", "20")),
        "cm",
        1,
    ),
    (
        "track_root_ori_err_deg_mean",
        "根向",
        float(os.environ.get("CORE4D_REVIEW_ROOT_ORI_MAX_DEG", "20")),
        "°",
        1,
    ),
    (
        "track_eef_pos_err_cm_mean",
        "手位",
        float(os.environ.get("CORE4D_REVIEW_HAND_POS_MAX_CM", "20")),
        "cm",
        1,
    ),
    (
        "track_eef_ori_err_deg_mean",
        "手向",
        float(os.environ.get("CORE4D_REVIEW_HAND_ORI_MAX_DEG", "20")),
        "°",
        1,
    ),
    (
        "track_obj_pos_err_cm_mean",
        "物位",
        float(os.environ.get("CORE4D_REVIEW_OBJECT_POS_MAX_CM", "20")),
        "cm",
        1,
    ),
    (
        "track_obj_ori_err_deg_mean",
        "物向",
        float(os.environ.get("CORE4D_REVIEW_OBJECT_ORI_MAX_DEG", "10")),
        "°",
        1,
    ),
    ("foot_slip_max_m", "脚滑", 1.0, "m", 3),
)


# ---------------------------------------------------------------------------
# Case loading (MuJoCo) — precompute per-frame body transforms for fast scrub.
# ---------------------------------------------------------------------------
ROBOT_MESH_DIR = REPO / "spider/assets/robots/unitree_g1/meshes"
OBJECT_MESH_ROOT = REPO / "workspace/core4d/object_models/object_models"

# --- performance knobs ------------------------------------------------------
# The heavy `spider.config` + trimesh import (~25s cold on JuiceFS) and per-case
# model compile / mesh decode used to run inside the very first (and every) case
# load, freezing the page. We now (a) warm the heavy imports in a startup thread,
# (b) LRU-cache compiled models + precomputed frames per case, (c) cache decoded
# meshes across cases (the G1 robot geometry is identical everywhere), and
# (d) decimate high-poly visual meshes so the browser scene stays light.
# CORE4D_REVIEW_MAX_FACES=0 disables decimation.
MAX_FACES = int(os.environ.get("CORE4D_REVIEW_MAX_FACES", "4000") or "0")
_CASE_CACHE: dict = {}          # (rec.key, want_ref) -> loaded tuple  (LRU, bounded)
_CASE_CACHE_MAX = 24
_MESH_CACHE: dict = {}          # (path, scale, max_faces) -> base trimesh (uncolored)
_WARMED = threading.Event()


def _warmup() -> None:
    """Import the heavy stack once, off the request path, then flag ready."""
    try:
        import trimesh  # noqa: F401
        import render_a100_cem_videos  # noqa: F401  (pulls spider.config)
        import spider.viewers.viser_viewer  # noqa: F401
        if MAX_FACES > 0:
            try:
                import open3d  # noqa: F401
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[review] warmup import failed (non-fatal): {exc}")
    finally:
        _WARMED.set()


def _decimate(tm, max_faces: int):
    """Reduce a trimesh to ~max_faces via open3d quadric decimation (best effort)."""
    try:
        n = len(tm.faces)
    except Exception:
        return tm
    if max_faces <= 0 or n <= max_faces:
        return tm
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
        )
        red = mesh.simplify_quadric_decimation(int(max_faces))
        import trimesh

        return trimesh.Trimesh(
            vertices=np.asarray(red.vertices),
            faces=np.asarray(red.triangles),
            process=False,
        )
    except Exception:
        return tm  # open3d missing / decimation failed -> keep full mesh


def _cached_mesh(mf, scale, max_faces: int):
    """Decode (+scale +decimate) a mesh file once and reuse across cases."""
    import trimesh

    key = (str(mf), None if scale is None else tuple(np.ravel(scale).tolist()), max_faces)
    base = _MESH_CACHE.get(key)
    if base is None:
        tm = trimesh.load(str(mf), force="mesh")
        if isinstance(tm, trimesh.Scene):
            tm = tm.to_mesh()
        if scale is not None:
            try:
                tm.apply_scale(scale)
            except Exception:
                pass
        base = _decimate(tm, max_faces)
        _MESH_CACHE[key] = base
    return base.copy()


@functools.lru_cache(maxsize=None)
def _object_mesh_fallback(filename: str) -> Path | None:
    """Return an unambiguous canonical object mesh with this basename."""
    if not OBJECT_MESH_ROOT.is_dir():
        return None
    matches = sorted(
        p.resolve() for p in OBJECT_MESH_ROOT.rglob(filename) if p.is_file()
    )
    return matches[0] if len(matches) == 1 else None


def _repo_asset_candidate(raw: str) -> Path | None:
    """Resolve a path containing a known repo marker after relocation."""
    normalized = raw.replace("\\", "/")
    for marker in ("example_datasets/", "workspace/", "spider/"):
        if marker in normalized:
            return (REPO / (marker + normalized.split(marker, 1)[1])).resolve()
    return None


def _load_portable_spec(scene_path: Path):
    """Load a scene without depending on its original directory depth.

    Archived ``scene_snapshot`` XMLs retain paths relative to their former
    task directory.  Re-root assets in memory so the snapshots remain
    read-only and portable after relocation.
    """
    import mujoco

    root = ET.parse(scene_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(ROBOT_MESH_DIR.resolve()))

    for mesh in root.findall("./asset/mesh"):
        raw = (mesh.get("file") or "").strip()
        if not raw:
            continue
        path = Path(raw)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        repo_candidate = _repo_asset_candidate(raw)
        if repo_candidate is not None:
            candidates.append(repo_candidate)
        candidates.extend((ROBOT_MESH_DIR / raw, scene_path.parent / raw))

        resolved = next((p.resolve() for p in candidates if p.is_file()), None)
        if resolved is None:
            resolved = _object_mesh_fallback(path.name)
        if resolved is not None:
            mesh.set("file", str(resolved))

    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def _load_case_data(rec: idx.CaseRecord, want_ref: bool):
    """Cached front for `_load_case_data_raw` (compile + qpos are expensive)."""
    key = (rec.key, bool(want_ref))
    hit = _CASE_CACHE.get(key)
    if hit is not None:
        _CASE_CACHE[key] = _CASE_CACHE.pop(key)  # mark most-recently-used
        return hit
    val = _load_case_data_raw(rec, want_ref)
    _CASE_CACHE[key] = val
    while len(_CASE_CACHE) > _CASE_CACHE_MAX:
        _CASE_CACHE.pop(next(iter(_CASE_CACHE)))  # evict least-recently-used
    return val


def _load_case_data_raw(rec: idx.CaseRecord, want_ref: bool):
    """Return (spec, model, sim_qpos, ref_qpos|None, frame_ids, fps)."""
    from render_a100_cem_videos import (  # noqa: E402
        converted_reference_qpos,
        load_render_config,
        rollout_qpos,
    )
    from spider.viewers.viser_viewer import _ensure_names

    row = {"scene_act": rec.scene_xml, "trajectory": rec.trajectory}
    config = load_render_config(row, Path(rec.config_act))
    spec = _load_portable_spec(Path(config.model_path))
    _ensure_names(spec)
    model = spec.compile()

    sim_qpos = rollout_qpos(Path(rec.outdir_npz))
    if sim_qpos.shape[1] != model.nq:
        raise ValueError(f"rollout nq={sim_qpos.shape[1]} != model nq={model.nq}")

    stride = max(1, int(round(float(config.render_dt) / float(config.sim_dt))))
    fps = max(1, int(round(1.0 / float(config.render_dt))))
    frame_ids = list(range(0, len(sim_qpos), stride))

    ref_qpos = None
    if want_ref:
        try:
            rq = converted_reference_qpos(config)
            if rq.shape[1] == model.nq:
                ref_qpos = rq
        except Exception as exc:  # reference is best-effort (E170 traj may be gone)
            print(f"[review] reference unavailable for {rec.key}: {exc}")
    if ref_qpos is not None:
        frame_ids = [f for f in frame_ids if f < len(ref_qpos)]
    if not frame_ids:
        raise ValueError("no replay frames")
    return spec, model, sim_qpos, ref_qpos, frame_ids, fps


def _compute_xforms(model, qpos, body_ids, frame_ids):
    import mujoco

    data = mujoco.MjData(model)
    out = []
    for fid in frame_ids:
        data.qpos[:] = qpos[fid]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        out.append(
            {bid: (data.xpos[bid].copy(), data.xquat[bid].copy()) for bid in body_ids}
        )
    return out


def _build_scene(server, spec, model, root: str, ref_color=None, include_collision=True):
    """Add one frame per body + one mesh per geom under ``root``.

    Mirrors spider.viewers.viser_viewer geom handling but writes under a root we
    own so it can be cleared on case swap. Returns
    (body_handles[(handle, body_id)], visual_handles, collision_handles).

    ``include_collision=False`` skips uploading collision geoms entirely (they are
    hidden by default, so uploading them just wastes transfer + browser memory);
    they are built on demand the first time the collision toggle is enabled.
    """
    import mujoco

    from spider.viewers.viser_viewer import (
        _get_mesh_file,
        _get_mesh_scale,
        _mujoco_mesh_to_trimesh,
        _set_mesh_color,
        _trimesh_from_primitive,
    )

    body_handles, visual, collision = [], [], []
    for body in spec.bodies[1:]:
        bpath = f"{root}/{body.name}"
        bh = server.scene.add_frame(bpath, show_axes=False)
        try:
            bid = model.body(body.name).id
        except Exception:
            bid = body.id
        body_handles.append((bh, bid))

        for geom in body.geoms:
            gname = geom.name
            if "_object_mass" in gname:
                continue
            try:
                gv = (
                    int(np.asarray(geom.group).ravel()[0])
                    if hasattr(geom, "group")
                    else 0
                )
            except Exception:
                gv = 0
            if gv >= 5:
                continue
            is_collision = ("collision" in (gname or "").lower()) or gv >= 3
            if is_collision and not include_collision:
                continue
            try:
                mg = model.geom(gname)
            except Exception:
                mg = None

            rgba = ref_color
            if rgba is None:
                for src in (mg, geom):
                    if src is None:
                        continue
                    try:
                        rgba = np.asarray(src.rgba, dtype=np.float32)
                        break
                    except Exception:
                        rgba = None

            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                tm = None
                mf = _get_mesh_file(spec, geom)
                ms = _get_mesh_scale(spec, geom)
                if mf is not None and mf.exists():
                    try:
                        tm = _cached_mesh(mf, ms, MAX_FACES)
                    except Exception:
                        tm = None
                if tm is None:
                    try:
                        tm = _mujoco_mesh_to_trimesh(
                            model, mg.id if mg is not None else -1
                        )
                        if tm is not None and ms is not None:
                            tm.apply_scale(ms)
                    except Exception:
                        tm = None
                if tm is None:
                    continue
                if rgba is not None:
                    _set_mesh_color(tm, rgba)
            else:
                size = geom.size
                if mg is not None:
                    try:
                        msz = model.geom_size[mg.id]
                        if np.any(np.asarray(size) == 0) or np.any(np.isnan(size)):
                            size = msz
                    except Exception:
                        pass
                tm = _trimesh_from_primitive(geom.type, size, rgba=rgba)
            if tm is None:
                continue

            if geom.type != mujoco.mjtGeom.mjGEOM_MESH and mg is not None:
                gpos = np.asarray(model.geom_pos[mg.id], dtype=np.float32)
                gquat = np.asarray(model.geom_quat[mg.id], dtype=np.float32)
            else:
                gpos = np.asarray(geom.pos, dtype=np.float32)
                gquat = np.asarray(geom.quat, dtype=np.float32)

            try:
                h = server.scene.add_mesh_trimesh(
                    f"{bpath}/g_{gname}", tm, position=gpos, wxyz=gquat
                )
            except Exception:
                continue
            (collision if is_collision else visual).append(h)
    return body_handles, visual, collision


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class ReviewApp:
    def __init__(self, records, host, port, reviewer, show_reference):
        import viser

        self.records = records
        self.by_key = {r.key: r for r in records}
        self.reviewer = reviewer
        self.want_ref = show_reference
        self.thresholds = {
            e: idx.load_thresholds(e) for e in {r.exp_id for r in records}
        }
        self.server = viser.ViserServer(host=host, port=port)
        try:
            self.server.gui.configure_theme(control_width="large")
        except Exception:
            pass

        # playback state (shared with the daemon loop)
        self.lock = threading.RLock()
        self.frames = []  # list[dict[bid -> (pos, quat)]] for sim
        self.ref_frames = []
        self.sim_bodies = []  # [(handle, bid)]
        self.ref_bodies = []
        self.visual_handles = []
        self.collision_handles = []
        self.ref_geom_handles = []
        self.playing = False
        self._next_t = 0.0
        self._prog = False  # programmatic slider write guard
        self.current = None  # current CaseRecord
        self._collision_built = False

        self._build_gui()
        self._refresh_case_list(initial=True)  # populates dropdown; defers geometry
        threading.Thread(target=self._warm_then_load, daemon=True).start()
        threading.Thread(target=self._player_loop, daemon=True).start()

    def _warm_then_load(self):
        """Import the heavy stack off the request path, then load the first case."""
        self._set_info("_首次启动：正在预热渲染依赖（约 15–25s），随后自动载入首个样本…_")
        _warmup()
        with self.lock:
            rec = self._filtered_recs[0] if (self.current is None and self._filtered_recs) else None
        if rec is not None:
            self._load_case(rec)
        elif not self._filtered_recs:
            self._set_info(
                "_当前筛选下没有可回放的样本。请放宽筛选条件，或换一个已评估的实验。_"
            )

    # -- GUI ----------------------------------------------------------------
    def _build_gui(self):
        s = self.server
        exps = ["全部"] + sorted({r.exp_id for r in self.records})
        objs = ["全部"] + idx.objects_for(self.records)
        modes = ["全部"] + idx.failure_modes_for(self.records)
        variants = ["全部"] + idx.variants_for(self.records)

        with s.gui.add_folder("数值指标"):
            self.metrics_md = s.gui.add_markdown("")

        with s.gui.add_folder("筛选"):
            self.f_case = s.gui.add_text(
                "Case 名称（支持子串）", initial_value=""
            )
            self.f_exp = s.gui.add_dropdown("实验", options=exps, initial_value="全部")
            self.f_obj = s.gui.add_dropdown("物体", options=objs, initial_value="全部")
            self.f_num = s.gui.add_dropdown(
                "数值", options=["全部", "达标", "不达标"], initial_value="全部"
            )
            self.f_mode = s.gui.add_dropdown(
                "失败原因", options=modes, initial_value="全部"
            )
            self.f_var = s.gui.add_dropdown(
                "变体", options=variants, initial_value="全部"
            )
            for w in (
                self.f_case,
                self.f_exp,
                self.f_obj,
                self.f_num,
                self.f_mode,
                self.f_var,
            ):
                w.on_update(lambda _=None: self._refresh_case_list())

        with s.gui.add_folder("样本"):
            self.case_dd = s.gui.add_dropdown(
                "选择", options=["(无)"], initial_value="(无)"
            )
            self.case_dd.on_update(self._on_case_pick)
            btn_prev = s.gui.add_button("◀ 上一个")
            btn_next = s.gui.add_button("下一个 ▶")
            btn_prev.on_click(lambda _: self._step_case(-1))
            btn_next.on_click(lambda _: self._step_case(+1))
            self.info_md = s.gui.add_markdown("")

        with s.gui.add_folder("播放"):
            self.frame_slider = s.gui.add_slider(
                "帧", min=0, max=1, step=1, initial_value=0
            )
            self.frame_slider.on_update(self._on_slider)
            self.play_btn = s.gui.add_button("播放 / 暂停")
            self.play_btn.on_click(self._toggle_play)
            self.cb_autoplay = s.gui.add_checkbox("切换后自动播放", initial_value=True)
            self.fps_num = s.gui.add_number(
                "帧率", initial_value=30, min=1, max=120, step=1
            )

        with s.gui.add_folder("显示"):
            self.cb_collision = s.gui.add_checkbox("显示碰撞体", initial_value=False)
            self.cb_reference = s.gui.add_checkbox(
                "显示参考残影", initial_value=self.want_ref
            )
            self.cb_grid = s.gui.add_checkbox("显示网格", initial_value=True)
            self.cb_collision.on_update(self._on_collision_toggle)
            self.cb_reference.on_update(lambda _: self._apply_visibility())
            self.cb_grid.on_update(lambda _: self._apply_visibility())

        with s.gui.add_folder("标注"):
            self.a_use = s.gui.add_dropdown(
                "使用裁决", options=list(USE_DECISIONS), initial_value="PENDING"
            )
            self.a_quality = s.gui.add_dropdown(
                "质量等级", options=list(QUALITY_LABELS), initial_value=""
            )
            self.a_taxo = s.gui.add_text("失败分类", initial_value="")
            self.a_note = s.gui.add_text("备注", initial_value="")
            self.a_reviewer = s.gui.add_text("审阅人", initial_value=self.reviewer)
            self.save_btn = s.gui.add_button("💾 保存标注")
            self.save_btn.on_click(self._save)
            self.progress_md = s.gui.add_markdown("")
        self._update_progress()

    # -- filtering ----------------------------------------------------------
    def _filtered(self):
        out = []
        for r in self.records:
            if not case_name_matches(r.case_id, self.f_case.value):
                continue
            if self.f_exp.value != "全部" and r.exp_id != self.f_exp.value:
                continue
            if self.f_obj.value != "全部" and r.object_key != self.f_obj.value:
                continue
            if self.f_num.value == "达标" and not r.numeric_release_pass:
                continue
            if self.f_num.value == "不达标" and r.numeric_release_pass:
                continue
            if (
                self.f_mode.value != "全部"
                and self.f_mode.value not in r.numeric_failure_modes
            ):
                continue
            if self.f_var.value != "全部" and r.retarget_variant_id != self.f_var.value:
                continue
            out.append(r)
        return out

    def _ann_tag(self, r: idx.CaseRecord):
        a = r.annotation or {}
        if (a.get("user_manual_review_status") or "") != "reviewed":
            return "⬜未标"
        dec = a.get("manual_use_decision", "")
        if dec == "USE":
            return (
                "🟡勉强"
                if a.get("manual_quality_label") == "MINOR_ACCEPTABLE"
                else "✅可用"
            )
        if dec == "DO_NOT_USE":
            return "⛔禁用"
        return "🔲待定"

    def _label(self, r: idx.CaseRecord):
        mark = "✓" if r.numeric_release_pass else "✗"
        v = r.retarget_variant_id.replace("omnirt_", "")
        source = f"{r.exp_id}/{r.arm}" if r.arm else r.exp_id
        return f"{self._ann_tag(r)} {mark} {source} {r.case_id} [{v}]"

    def _refresh_case_list(self, initial=False):
        self._filtered_recs = self._filtered()
        self._labels = [self._label(r) for r in self._filtered_recs]
        opts = self._labels or ["(无)"]
        self.case_dd.options = opts
        target = opts[0]
        self.case_dd.value = target
        if self._filtered_recs and not initial:
            self._load_case(self._filtered_recs[0])
        elif not self._filtered_recs and not initial:
            self._set_info("_当前筛选无匹配样本_")
        # initial load is deferred to _warm_then_load (avoids a cold-import stall
        # blocking server startup / the whole page).

    def _on_case_pick(self, _=None):
        if self.case_dd.value in self._labels:
            self._load_case(self._filtered_recs[self._labels.index(self.case_dd.value)])

    def _step_case(self, delta):
        if not self._filtered_recs:
            return
        i = (
            self._labels.index(self.case_dd.value)
            if self.case_dd.value in self._labels
            else 0
        )
        i = max(0, min(len(self._filtered_recs) - 1, i + delta))
        self.case_dd.value = self._labels[i]  # fires _on_case_pick

    # -- case load ----------------------------------------------------------
    def _load_case(self, rec: idx.CaseRecord):
        with self.lock:
            self.playing = False
            self.current = rec
            self._update_metrics(rec)
            self.server.scene.reset()
            self.frames, self.ref_frames = [], []
            self.sim_bodies, self.ref_bodies = [], []
            self.visual_handles, self.collision_handles = [], []
            self.ref_geom_handles = []
            self._prefill_annotation(rec)

            if not rec.playable:
                self._set_info(
                    self._info_text(
                        rec, note="⚠ 3D 数据已归档不可用；请对照 MP4 标注。"
                    )
                )
                self.frame_slider.max = 1
                self._set_slider(0)
                return
            try:
                spec, model, sim_qpos, ref_qpos, frame_ids, fps = _load_case_data(
                    rec, self.cb_reference.value
                )
            except Exception as exc:
                self._set_info(self._info_text(rec, note=f"⚠ 加载失败: {exc}"))
                return

            if self.cb_grid.value:
                try:
                    self.server.scene.add_grid("/grid")
                except Exception:
                    pass
            want_collision = bool(self.cb_collision.value)
            self._collision_built = want_collision
            self.sim_bodies, self.visual_handles, self.collision_handles = _build_scene(
                self.server, spec, model, "/sim", include_collision=want_collision
            )
            self.frames = _compute_xforms(
                model, sim_qpos, [b for _, b in self.sim_bodies], frame_ids
            )
            if ref_qpos is not None:
                self.ref_bodies, rv, rc = _build_scene(
                    self.server,
                    spec,
                    model,
                    "/ref",
                    ref_color=np.array([0, 0, 1, 0.25], np.float32),
                    include_collision=want_collision,
                )
                self.ref_geom_handles = rv + rc
                self.ref_frames = _compute_xforms(
                    model, ref_qpos, [b for _, b in self.ref_bodies], frame_ids
                )

            self.fps_num.value = fps
            self.frame_slider.max = max(1, len(self.frames) - 1)
            self._set_slider(0)
            self._apply(0)
            self._apply_visibility()
            self._set_info(self._info_text(rec))
            # auto-play the newly-loaded sequence (opt-out via the checkbox)
            if len(self.frames) > 1 and self.cb_autoplay.value:
                self.playing = True
                self._next_t = time.perf_counter()

    # -- playback -----------------------------------------------------------
    def _player_loop(self):
        while True:
            if not self.playing or len(self.frames) <= 1:
                time.sleep(0.03)
                continue
            now = time.perf_counter()
            if now >= self._next_t:
                i = int(self.frame_slider.value) + 1
                if i >= len(self.frames):
                    i = 0
                self._apply(i)
                self._set_slider(i)
                self._next_t = now + 1.0 / max(1, int(self.fps_num.value))
            else:
                time.sleep(min(0.003, max(0.0, self._next_t - now)))

    def _apply(self, i):
        with self.lock:
            if i < 0 or i >= len(self.frames):
                return
            with self.server.atomic():
                for h, bid in self.sim_bodies:
                    pos, quat = self.frames[i][bid]
                    h.position = tuple(float(x) for x in pos)
                    h.wxyz = tuple(float(x) for x in quat)
                if self.ref_frames and i < len(self.ref_frames):
                    for h, bid in self.ref_bodies:
                        pos, quat = self.ref_frames[i][bid]
                        h.position = tuple(float(x) for x in pos)
                        h.wxyz = tuple(float(x) for x in quat)

    def _set_slider(self, i):
        self._prog = True
        try:
            self.frame_slider.value = int(i)
        finally:
            self._prog = False

    def _on_slider(self, _=None):
        if self._prog:
            return
        self.playing = False
        self._apply(int(self.frame_slider.value))

    def _toggle_play(self, _=None):
        self.playing = not self.playing
        self._next_t = time.perf_counter()

    def _on_collision_toggle(self, _=None):
        # Collision geoms are not uploaded until first needed; build them lazily by
        # reloading the current case, then just toggle visibility thereafter.
        if self.cb_collision.value and self.current is not None and not getattr(
            self, "_collision_built", False
        ):
            self._load_case(self.current)
            return
        self._apply_visibility()

    def _apply_visibility(self):
        for h in self.collision_handles:
            h.visible = bool(self.cb_collision.value)
        for h in self.visual_handles:
            h.visible = True
        want_ref = bool(self.cb_reference.value)
        for h in self.ref_geom_handles:
            try:
                h.visible = want_ref
            except Exception:
                pass

    # -- annotation ---------------------------------------------------------
    def _prefill_annotation(self, rec: idx.CaseRecord):
        a = rec.annotation or {}
        self.a_use.value = a.get("manual_use_decision") or "PENDING"
        self.a_quality.value = a.get("manual_quality_label") or ""
        self.a_taxo.value = a.get("manual_failure_taxonomy") or ""
        self.a_note.value = a.get("manual_review_note") or ""
        if a.get("manual_reviewer"):
            self.a_reviewer.value = a["manual_reviewer"]

    def _save(self, _=None):
        rec = self.current
        if rec is None:
            return
        values = {
            "manual_use_decision": self.a_use.value,
            "manual_quality_label": self.a_quality.value,
            "manual_failure_taxonomy": self.a_taxo.value,
            "manual_review_note": self.a_note.value,
            "manual_reviewer": self.a_reviewer.value,
        }
        path = idx.save_annotation(rec.exp_id, rec.ann_id, values)
        rec.annotation = idx.load_annotations(rec.exp_id).get(rec.ann_id, {})
        # refresh label (reviewed marker) in the dropdown
        if rec in self._filtered_recs:
            j = self._filtered_recs.index(rec)
            self._labels[j] = self._label(rec)
            keep = self._labels[j]
            self.case_dd.options = self._labels
            self.case_dd.value = keep
        self._update_progress(saved=f"已保存 {rec.ann_id} → {rec.exp_id}/{path.name}")
        self._set_info(self._info_text(rec))

    def _update_progress(self, saved=""):
        n = len(self.records)
        done = sum(1 for r in self.records if r.reviewed)
        use = sum(
            1
            for r in self.records
            if (r.annotation or {}).get("manual_use_decision") == "USE"
        )
        dnu = sum(
            1
            for r in self.records
            if (r.annotation or {}).get("manual_use_decision") == "DO_NOT_USE"
        )
        msg = f"**已审阅 {done} / {n}**  ·  USE {use}  ·  DO_NOT_USE {dnu}"
        if saved:
            msg += f"\n\n_{saved}_"
        self.progress_md.content = msg

    # -- info panel ---------------------------------------------------------
    def _update_metrics(self, r: idx.CaseRecord):
        thr = self.thresholds.get(r.exp_id, {})
        h1, v1 = [], []
        for col, name, tkey, direction, unit in METRIC_BAR:
            t = thr.get(tkey)
            if direction == "max" and t is not None:
                h1.append(f"{name}≤{t:g}{unit}")
            elif direction == "min" and t is not None:
                h1.append(f"{name}≥{t:g}{unit}")
            else:
                h1.append(name)
            if direction == "flag":
                fell = r.gates.get("fall_gate_pass") is False
                v1.append("🔴**是**" if fell else "否")
                continue
            v = r.metrics.get(col)
            if v is None:
                v1.append("—")
                continue
            txt = f"{v:.3f}" if unit == "m" else f"{v:.2f}"
            breach = t is not None and (
                (direction == "max" and v > t) or (direction == "min" and v < t)
            )
            v1.append(f"🔴**{txt}**" if breach else txt)

        h2, v2 = [], []
        for col, name, t, unit, dec in TRACK_BAR:
            h2.append(f"{name}≤{t:g}{unit}")
            v = r.metrics.get(col)
            if v is None:
                v2.append("—")
                continue
            txt = f"{v:.{dec}f}"
            v2.append(f"🔴**{txt}**" if v > t else txt)

        def table(h, v):
            return (
                "| " + " | ".join(h) + " |\n"
                "|" + "|".join(["---"] * len(h)) + "|\n"
                "| " + " | ".join(v) + " |"
            )

        self.metrics_md.content = (
            "**数值门**\n\n" + table(h1, v1) + "\n\n**跟踪误差**\n\n" + table(h2, v2)
        )

    def _set_info(self, md):
        self.info_md.content = md

    def _info_text(self, r: idx.CaseRecord, extra="", note=""):
        gate_names = {
            "fall_gate_pass": "跌倒",
            "body_z_gate_pass": "身高",
            "contact_gate_pass": "接触",
            "release_gate_pass": "释放",
            "hand_penetration_gate_pass": "手穿透",
            "lower_body_gate_pass": "下肢",
            "root_pos_gate_pass": "根位置",
            "root_ori_gate_pass": "根朝向",
            "hand_pos_gate_pass": "手位置",
            "hand_ori_gate_pass": "手朝向",
            "object_pos_gate_pass": "物位置",
            "object_ori_gate_pass": "物朝向",
        }
        known_gates = {
            key: value for key, value in r.gates.items() if value is not None
        }
        total = len(known_gates)
        n_pass = sum(1 for value in known_gates.values() if value is True)
        failed = [
            gate_names.get(key, key)
            for key, value in known_gates.items()
            if value is False
        ]
        if failed:
            gate_line = f"{n_pass}/{total} 通过 · 未过 " + " ".join(
                f"🔴{n}" for n in failed
            )
        else:
            gate_line = f"🟢 {n_pass}/{total} 全部通过"
        modes = ", ".join(r.numeric_failure_modes) or "—"
        a = r.annotation or {}
        if (a.get("user_manual_review_status") or "") == "reviewed":
            ann = (
                f"{self._ann_tag(r)}  裁决=**{a.get('manual_use_decision', '')}** "
                f"质量={a.get('manual_quality_label', '') or '—'} "
                f"审阅人={a.get('manual_reviewer', '') or '—'}"
            )
        else:
            ann = "⬜ 尚未人工标注"
        pass_mark = "✅ 达标" if r.numeric_release_pass else "❌ 不达标"
        lines = [
            f"### {r.exp_id}{f' · {r.arm}' if r.arm else ''} · {r.case_id}",
            f"- 人工标注: {ann}",
            f"- 物体 **{r.object_key}** · 变体 **{r.retarget_variant_id}**",
            f"- 数值: {pass_mark}",
            f"- 失败原因: {modes}",
            f"- 物理门: {gate_line}",
            f"- 状态: {r.status}",
        ]
        if note:
            lines.append(f"\n**{note}**")
        return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--exps", default=",".join(idx.DEFAULT_EXPS))
    ap.add_argument("--reviewer", default=os.environ.get("USER", "user"))
    ap.add_argument(
        "--no-reference", action="store_true", help="disable reference ghost"
    )
    ap.add_argument("--check", action="store_true", help="print index audit and exit")
    ap.add_argument(
        "--arm", default="",
        help="only load these arm(s), comma-separated (e.g. G1A2); empty = all arms",
    )
    args = ap.parse_args()

    exps = tuple(e.strip() for e in args.exps.split(",") if e.strip())
    if args.check:
        return idx._check(exps)

    records = idx.build_index(exps)
    if args.arm:
        arms = {a.strip() for a in args.arm.split(",") if a.strip()}
        records = [r for r in records if r.arm in arms]
        print(f"[review] arm filter {sorted(arms)} -> {len(records)} cases")
    print(f"[review] indexed {len(records)} cases across {exps}")
    if not records:
        extra = [e for e in ("E194", "E199", "E199P", "E200N", "E200G") if e not in idx.DEFAULT_EXPS]
        avail = ", ".join(list(idx.DEFAULT_EXPS) + extra)
        print(
            f"[review] no reviewable records for {exps} — this experiment has no "
            f"*_case_metrics.tsv under results/<exp>/s6_downstream/eval/, so it was "
            f"never wired into the review player.\n"
            f"[review] available review sets: {avail}",
            file=sys.stderr,
        )
        return 2
    app = ReviewApp(  # noqa: F841
        records,
        host=args.host,
        port=args.port,
        reviewer=args.reviewer,
        show_reference=not args.no_reference,
    )
    print(f"[review] viser server on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
