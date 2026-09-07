#!/usr/bin/env python3
"""Replay E213 selected-arm aug variants (orig + trans/rot) side-by-side in a
headless viser server, with a case picker and looping playback.

Per case, every entity is on the case's OWN E212-selected arm (G1/gc08/gc06/gc04/
PRG), so the only thing that differs across the entities is the object-approach
augmentation -- which is exactly the comparison we want to eyeball.  Loads each
rollout's `trajectory_mjwp_act.npz` + its scene, runs CPU FK per frame, drives
viser body frames on one shared timeline under one camera. No GPU/OpenGL.

**Why not the mp4s.** `_auto_video_camera` recomputes the camera per frame from
the sim-union bbox, so two rollouts' mp4s are shot from different moving cameras
(E209 F6 / E210 F3) -- cross-video pose comparison is invalid. Here every variant
sits in one scene under one camera on one timeline.

Entities (`--variants`, left to right):
  ref     the reference trajectory (from orig's qpos[:, 1, :])
  orig    the case's selected-arm original rollout (cem_result_npz)
  trans0/1/2, rot0/1   the E213 selected-arm aug rollouts

Usage (SSH port-forward):
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E213/viser_replay_E213.py \
      [--variants ref,orig,trans0,rot0] [--case <id>] [--port 8213] [--stride 1]
    ssh -L 8213:localhost:8213 <host>   then open http://localhost:8213
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E213"))

import mujoco  # noqa: E402

import e213_common as C  # noqa: E402

_vv_path = REPO / "spider/viewers/viser_viewer.py"
_spec = _ilu.spec_from_file_location("viser_viewer_standalone", _vv_path)
V = _ilu.module_from_spec(_spec)
sys.modules[_spec.name] = V
_spec.loader.exec_module(V)  # type: ignore

AUG_VARIANTS = ("trans0", "trans1", "trans2", "rot0", "rot1")
ALL_ENTITIES = ("ref", "orig") + AUG_VARIANTS


def _cases() -> dict[str, dict[str, str]]:
    return {c["case_id"]: c for c in C.load_cases()}


def _aug_by_case() -> dict[str, dict[str, dict[str, str]]]:
    """case_id -> aug_variant -> E208 artifacts row (for target_task lookup)."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for cid, rows in C.aug_rows_by_case().items():
        out[cid] = {r["aug_variant"]: r for r in rows}
    return out


def entity_scene_and_npz(entity: str, case: dict[str, str], aug: dict[str, dict[str, str]]) -> tuple[Path, Path, int]:
    """(scene_xml, rollout_npz, qpos_slice) for one entity of a case.

    qpos_slice: 1 for `ref` (reference channel), else 0 (sim channel).
    """
    cid, arm = case["case_id"], case["arm"]
    if entity in ("ref", "orig"):
        scene = C.orig_selected_scene(case)
        npz = C.repo_path(case["orig_result_npz"])
        return scene, npz, (1 if entity == "ref" else 0)
    if entity in AUG_VARIANTS:
        row = aug.get(cid, {}).get(entity)
        if row is None:
            raise SystemExit(f"{cid}/{entity}: no E208 aug artifact row")
        scene = C.aug_selected_scene(row, case)
        npz = (C.e208_prg_result_npz(cid, entity) if C.is_prg_case(case)
               else C.result_npz(cid, entity, arm))
        return scene, npz, 0
    raise ValueError(f"unknown entity {entity}; known {ALL_ENTITIES}")


def entity_label(entity: str, case: dict[str, str]) -> str:
    if entity == "ref":
        return "reference"
    if entity == "orig":
        return f"orig  {case['arm']}"
    return f"{entity}  {case['arm']}"


def load_qpos(npz_path: Path, nq: int, which: int = 0) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as d:
        a = np.asarray(d["qpos"], dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == nq:
        return a[:, which, :]
    if a.ndim == 2 and a.shape[1] == nq:
        return a
    raise KeyError(f"qpos shape {a.shape} vs model.nq={nq} in {npz_path}")


def fk_frames(model: mujoco.MjModel, qpos: np.ndarray, body_ids: list[int]):
    data = mujoco.MjData(model)
    T, B = len(qpos), len(body_ids)
    xpos = np.zeros((T, B, 3), np.float64)
    xquat = np.zeros((T, B, 4), np.float64)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        xpos[t] = data.xpos[body_ids]
        xquat[t] = data.xquat[body_ids]
    return xpos, xquat


class CaseScene:
    def __init__(self, server: Any, case: dict[str, str], entities: list[str],
                 aug: dict[str, dict[str, str]], stride: int, spacing: float) -> None:
        self.case_id = case["case_id"]
        self.entries: list[tuple] = []
        self.labels: list[Any] = []
        n_frames: int | None = None
        for i, entity in enumerate(entities):
            scene, npz, which = entity_scene_and_npz(entity, case, aug)
            if not npz.is_file():
                print(f"  [skip] {self.case_id}/{entity}: rollout missing {npz}", flush=True)
                continue
            spec = mujoco.MjSpec.from_file(str(scene))
            V._ensure_names(spec)
            model = spec.compile()
            qpos = load_qpos(npz, model.nq, which=which)[::stride]
            entity_root = f"{self.case_id}__{entity}"
            handles = V.build_and_log_scene_from_spec(
                spec=spec, model=model, xml_path=scene, entity_root=entity_root, build_ref=False)
            try:
                server.scene.add_grid(f"{entity_root}/ground_plane", visible=False)
            except Exception:  # noqa: BLE001
                pass
            body_ids = [bid for _h, bid in handles]
            xpos, xquat = fk_frames(model, qpos, body_ids)
            offset = np.array([i * spacing - (len(entities) - 1) * spacing / 2.0, 0.0, 0.0])
            self.entries.append((entity, handles, xpos, xquat, offset))
            n_frames = len(qpos) if n_frames is None else min(n_frames, len(qpos))
            try:
                self.labels.append(server.scene.add_label(
                    f"{entity_root}/label", text=entity_label(entity, case),
                    position=tuple(offset + np.array([0.0, 0.0, 1.9]))))
            except Exception:  # noqa: BLE001
                pass
            print(f"  [loaded] {self.case_id} {entity_label(entity, case):16s} frames={len(qpos)}", flush=True)
        self.n_frames = n_frames or 1

    def set_visible(self, value: bool) -> None:
        for _e, handles, _xp, _xq, _o in self.entries:
            for handle, _bid in handles:
                handle.visible = value
        for label in self.labels:
            label.visible = value

    def render(self, frame: int) -> None:
        f = max(0, min(frame, self.n_frames - 1))
        for _e, handles, xpos, xquat, offset in self.entries:
            for j, (handle, _bid) in enumerate(handles):
                handle.position = tuple(xpos[f, j] + offset)
                handle.wxyz = tuple(xquat[f, j])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--cases", default="")
    ap.add_argument("--variants", default="ref,orig,trans0,trans1,trans2,rot0,rot1")
    ap.add_argument("--port", type=int, default=8213)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--spacing", type=float, default=1.8)
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    cases = _cases()
    aug = _aug_by_case()
    pick = [c.strip() for c in args.cases.split(",") if c.strip()] or list(cases)
    unknown = [c for c in pick if c not in cases]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}")
    first = args.case or pick[0]
    entities = [e.strip() for e in args.variants.split(",") if e.strip()]
    bad = [e for e in entities if e not in ALL_ENTITIES]
    if bad:
        raise SystemExit(f"unknown entity(ies) {bad}; known {list(ALL_ENTITIES)}")

    _viser = V._lazy_import_viser()
    V._STATE.server = _viser.ViserServer(host=args.host, port=args.port, label="E213 aug")
    server = V._get_server()
    try:
        server.scene.add_grid("ground_plane")
    except Exception:  # noqa: BLE001
        pass

    built: dict[str, CaseScene] = {}
    state: dict[str, Any] = {"case": None, "dir": 0, "slider": None}

    def build(cid: str) -> CaseScene:
        if cid not in built:
            print(f"[build] {cid} ...", flush=True)
            t0 = time.time()
            built[cid] = CaseScene(server, cases[cid], entities, aug, args.stride, args.spacing)
            print(f"[build] {cid} done in {time.time() - t0:.1f}s", flush=True)
        return built[cid]

    case_dd = server.gui.add_dropdown("Case", options=tuple(pick), initial_value=first)
    info = server.gui.add_text("Frames", initial_value="", disabled=True)
    loop_cb = server.gui.add_checkbox("Loop", initial_value=True)
    fps_slider = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=int(args.fps))
    b_rev = server.gui.add_button("◀ Play")
    b_pause = server.gui.add_button("⏸ Pause")
    b_fwd = server.gui.add_button("Play ▶")
    frame_folder = server.gui.add_folder("Timeline")

    def rebuild_slider(n_frames: int) -> None:
        old = state.get("slider")
        if old is not None:
            old.remove()
        with frame_folder:
            slider = server.gui.add_slider("Frame", min=0, max=max(1, n_frames - 1), step=1, initial_value=0)
        slider.on_update(lambda _=None: _draw())
        state["slider"] = slider

    def _draw() -> None:
        scene = built.get(state["case"])
        slider = state.get("slider")
        if scene is not None and slider is not None:
            scene.render(int(slider.value))

    def select(cid: str) -> None:
        prev = state.get("case")
        if prev == cid:
            return
        scene = build(cid)
        if prev is not None and prev in built:
            built[prev].set_visible(False)
        scene.set_visible(True)
        state["case"] = cid
        rebuild_slider(scene.n_frames)
        info.value = f"{scene.n_frames} frames (stride {args.stride})"
        scene.render(0)

    case_dd.on_update(lambda _=None: select(case_dd.value))
    b_rev.on_click(lambda _=None: state.update(dir=-1))
    b_pause.on_click(lambda _=None: state.update(dir=0))
    b_fwd.on_click(lambda _=None: state.update(dir=1))
    select(first)

    def loop() -> None:
        while True:
            time.sleep(1.0 / max(1.0, float(fps_slider.value)))
            if state["dir"] == 0:
                continue
            slider = state.get("slider")
            scene = built.get(state["case"])
            if slider is None or scene is None:
                continue
            v = int(slider.value) + state["dir"]
            if loop_cb.value:
                v %= scene.n_frames
            elif not 0 <= v < scene.n_frames:
                state["dir"] = 0
                continue
            slider.value = v

    threading.Thread(target=loop, daemon=True).start()
    bound = server.get_port()
    print(f"\n[viser] serving on http://localhost:{bound}  (ssh -L {bound}:localhost:{bound} <host>)", flush=True)
    print(f"[viser] {len(pick)} case(s) x {len(entities)} entities: {entities}. Loop on. Ctrl-C to stop.", flush=True)
    print("[viser] one scene/camera/timeline -- cross-entity pose comparison is valid here.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
