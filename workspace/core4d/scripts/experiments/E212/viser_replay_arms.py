#!/usr/bin/env python3
"""Replay E212 g-curve arms side-by-side in a headless viser server, with a
case picker and looping playback.

Loads each arm's CEM rollout (`trajectory_mjwp_act.npz`) + its scene, runs CPU FK
(`mj_forward`) per frame, and drives viser body frames on a single shared
timeline. Arms are offset along X so they play synchronised side by side. No GPU
/ OpenGL needed -- viser renders in the browser, so this works on GL-less boxes
where osmesa is the only other option.

**Why this exists and the mp4s are not enough.** `_auto_video_camera`
(spider/viewers/__init__.py:262-291) recomputes lookat and radius every frame
from the union bbox of sim and ref bodies, so two arms' mp4s are shot from two
different, silently-moving cameras. E209 F6 and E210 F3 both established that the
SAME reference trajectory renders as visibly different poses across two videos --
one crouched, one upright. Cross-video pose comparison is therefore invalid.
Here every arm sits in ONE scene under ONE camera on ONE timeline, which is
exactly the comparison mp4 cannot give.

GUI
  Case      dropdown over the 4 desk023 cases; switching rebuilds the timeline.
            Scenes are built lazily on first visit and then cached, so the first
            switch to a case costs a few seconds and later ones are instant.
  Loop      wrap at the ends instead of stopping (default on).
  Frame     scrub; the slider is rebuilt per case because viser slider bounds are
            immutable (only `.value` is settable) and cases differ in length
            (varies per case).
  FPS       playback rate.

Arms (`--arms`, in the order they appear left to right):
  ref   the reference trajectory (from a run's qpos[:, 1, :] -- byte-identical
        across arms, verified in E209 log298 section 4-2)
  prg   g=0.0   E206 PRG baseline rollout
  G04   g=0.4   E212 Stage A          (robot side fully preserved, narrow 4/4)
  G06   g=0.6   E212 Stage A          (C1-closest: 5/6 clauses, misses z by 0.26mm)
  G08   g=0.8   E212 Stage A          (best object z, eef_ori starts breaking)
  g1    g=1.0   E209 G1 rollout

Usage (access via SSH port-forward):
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E212/viser_replay_arms.py \
      [--arms ref,G06,G08] [--case <id>] [--port 8080] [--stride 1]

    ssh -L 8080:localhost:8080 <host>   then open http://localhost:8080
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
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E212"))

import mujoco  # noqa: E402

import e212_common as C  # noqa: E402

# Load viser_viewer.py DIRECTLY (bypass spider.viewers.__init__, which imports the
# warp/torch-backed mjwp viewer -> heavy + fragile on loaded boxes). viser_viewer
# itself only needs mujoco/trimesh/viser/numpy/loguru.
_vv_path = REPO / "spider/viewers/viser_viewer.py"
_spec = _ilu.spec_from_file_location("viser_viewer_standalone", _vv_path)
V = _ilu.module_from_spec(_spec)
# Register BEFORE exec_module. viser_viewer defines `@dataclass _ViserState`, and
# dataclasses resolves field types via `sys.modules[cls.__module__].__dict__`; an
# unregistered dynamic module makes that None and the import dies with
# "AttributeError: 'NoneType' object has no attribute '__dict__'". The
# E204_E205 copy of this loader omits the registration and has the same latent bug.
sys.modules[_spec.name] = V
_spec.loader.exec_module(V)  # type: ignore

#: gravcomp value per arm; `ref` is the kinematic reference, not a rollout.
ARM_G: dict[str, float | None] = {
    "ref": None, "prg": 0.0, "G04": 0.4, "G06": 0.6, "G08": 0.8, "g1": 1.0,
}
#: Which rollout `ref` is read out of. Any arm works (the reference is identical
#: across arms), but pinning it makes the choice explicit and reproducible.
REF_SOURCE_ARM = "prg"


def arm_label(arm: str) -> str:
    g = ARM_G[arm]
    return "reference" if g is None else f"{arm}  g={g:.1f}"


def arm_scene_and_npz(arm: str, case_id: str) -> tuple[Path, Path]:
    row = C.source_row(case_id)
    if arm == "prg":
        return C.base_scene_path(row), C.prg_out_dir(case_id) / "trajectory_mjwp_act.npz"
    if arm == "g1":
        return (C.task_dir(row) / f"{C.G1_SCENE}.xml",
                C.g1_out_dir(case_id) / "trajectory_mjwp_act.npz")
    if arm in C.ARMS:
        return C.scene_path(row, arm), C.result_npz(case_id, arm)
    raise ValueError(f"unknown arm {arm}; known {list(ARM_G)}")


def load_qpos(npz_path: Path, nq: int, which: int = 0) -> np.ndarray:
    """Return (T, nq) qpos from a rollout npz.

    run_mjwp saves qpos as (T, 2, nq) = [sim, reference]; `which` picks the slice.
    Also handles a plain (T, nq) array (then `which` is ignored).
    """
    with np.load(npz_path, allow_pickle=True) as d:
        a = np.asarray(d["qpos"], dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == nq:
        return a[:, which, :]
    if a.ndim == 2 and a.shape[1] == nq:
        return a
    raise KeyError(f"qpos shape {a.shape} incompatible with model.nq={nq} in {npz_path}")


def fk_frames(model: mujoco.MjModel, qpos: np.ndarray,
              body_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """CPU FK -> (T, B, 3) xpos, (T, B, 4) xquat (wxyz) for the given bodies."""
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
    """One case's arms, built once and then shown/hidden.

    Every (case, arm) gets its own `entity_root`, so its body frames and their
    child geoms live under a private path prefix and can be hidden as a unit.
    Rebuilding on every case switch would re-upload all the meshes, which is the
    slow part; caching trades memory for an instant second visit.
    """

    def __init__(self, server: Any, case_id: str, arms: list[str],
                 stride: int, spacing: float) -> None:
        self.case_id = case_id
        self.entries: list[tuple[str, list[tuple[Any, int]], np.ndarray, np.ndarray, np.ndarray]] = []
        self.labels: list[Any] = []
        n_frames: int | None = None
        for i, arm in enumerate(arms):
            src_arm = REF_SOURCE_ARM if arm == "ref" else arm
            scene, npz = arm_scene_and_npz(src_arm, case_id)
            if not npz.is_file():
                raise SystemExit(f"{case_id}/{arm}: rollout not found -> {npz}")
            spec = mujoco.MjSpec.from_file(str(scene))
            V._ensure_names(spec)
            model = spec.compile()
            qpos = load_qpos(npz, model.nq, which=1 if arm == "ref" else 0)[::stride]
            entity_root = f"{case_id}__{arm}"
            handles = V.build_and_log_scene_from_spec(
                spec=spec, model=model, xml_path=scene,
                entity_root=entity_root, build_ref=False)
            # build_and_log_scene_from_spec adds a floor grid at
            # "{entity_root}/ground_plane" and throws the handle away
            # (viser_viewer.py:271-278). With one entity_root per (case, arm)
            # that is 15 overlapping grids nobody can hide -- z-fighting, and
            # they would stay on screen after a case switch. Re-adding at the
            # same path replaces the node and hands back a handle; hide it and
            # use the single shared floor added at startup instead.
            try:
                server.scene.add_grid(f"{entity_root}/ground_plane", visible=False)
            except Exception:  # noqa: BLE001 - cosmetic only
                pass
            body_ids = [bid for _h, bid in handles]
            xpos, xquat = fk_frames(model, qpos, body_ids)
            offset = np.array(
                [i * spacing - (len(arms) - 1) * spacing / 2.0, 0.0, 0.0])
            self.entries.append((arm, handles, xpos, xquat, offset))
            n_frames = len(qpos) if n_frames is None else min(n_frames, len(qpos))
            try:
                self.labels.append(server.scene.add_label(
                    f"{case_id}__{arm}/label", text=arm_label(arm),
                    position=tuple(offset + np.array([0.0, 0.0, 1.9]))))
            except Exception:  # noqa: BLE001 - a missing label must not kill the replay
                pass
            print(f"  [loaded] {case_id} {arm_label(arm):16s} scene={scene.name} "
                  f"frames={len(qpos)} bodies={len(body_ids)}", flush=True)
        assert n_frames is not None
        self.n_frames = n_frames
        counts = {a: len(x) for a, _h, x, _q, _o in self.entries}
        if len(set(counts.values())) > 1:
            print(f"  [warn] {case_id}: arms differ in length {counts}; "
                  f"timeline truncated to {n_frames}", flush=True)

    def set_visible(self, value: bool) -> None:
        for _arm, handles, _xp, _xq, _off in self.entries:
            for handle, _bid in handles:
                handle.visible = value
        for label in self.labels:
            label.visible = value

    def render(self, frame: int) -> None:
        f = max(0, min(frame, self.n_frames - 1))
        for _arm, handles, xpos, xquat, offset in self.entries:
            for j, (handle, _bid) in enumerate(handles):
                handle.position = tuple(xpos[f, j] + offset)
                handle.wxyz = tuple(xquat[f, j])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="case shown first (default: first in scope)")
    ap.add_argument("--cases", default="", help="restrict the picker (default: all 4)")
    ap.add_argument("--arms", default="ref,G06,G08",
                    help="left-to-right; ref|prg|G04|G06|G08|g1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--stride", type=int, default=1, help="frame subsample")
    ap.add_argument("--spacing", type=float, default=1.8, help="X offset between arms (m)")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--preload", action="store_true",
                    help="build every case up front instead of on first selection")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()] or list(C.CASES)
    unknown_c = [c for c in cases if c not in C.CASES]
    if unknown_c:
        raise SystemExit(f"unknown case(s) {unknown_c}; E212 scope is {list(C.CASES)}")
    first = args.case or cases[0]
    if first not in cases:
        raise SystemExit(f"--case {first} is not in the picker set {cases}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown_a = [a for a in arms if a not in ARM_G]
    if unknown_a:
        raise SystemExit(f"unknown arm(s) {unknown_a}; known {list(ARM_G)}")

    # Construct the server directly rather than via `init_viser`, which takes no
    # host/port and so always uses ViserServer's default 8080 -- making a --port
    # flag silently a no-op. viser also auto-increments off a busy port, so the
    # requested port is not necessarily the bound one; report `get_port()`.
    _viser = V._lazy_import_viser()
    V._STATE.server = _viser.ViserServer(
        host=args.host, port=args.port, label="E212 g-sweep")
    server = V._get_server()

    # One floor for the whole session; the per-entity grids are suppressed in
    # CaseScene so they cannot stack or linger across case switches.
    try:
        server.scene.add_grid("ground_plane")
    except Exception:  # noqa: BLE001 - cosmetic only
        pass

    built: dict[str, CaseScene] = {}
    state: dict[str, Any] = {"case": None, "dir": 0, "slider": None}

    def build(case_id: str) -> CaseScene:
        if case_id not in built:
            print(f"[build] {case_id} ...", flush=True)
            t0 = time.time()
            built[case_id] = CaseScene(server, case_id, arms, args.stride, args.spacing)
            print(f"[build] {case_id} done in {time.time() - t0:.1f}s", flush=True)
        return built[case_id]

    # --- GUI -----------------------------------------------------------------
    case_dd = server.gui.add_dropdown("Case", options=tuple(cases), initial_value=first)
    info = server.gui.add_text("Frames", initial_value="", disabled=True)
    loop_cb = server.gui.add_checkbox("Loop", initial_value=True)
    fps_slider = server.gui.add_slider("FPS", min=1, max=120, step=1,
                                       initial_value=int(args.fps))
    b_rev = server.gui.add_button("◀ Play")
    b_pause = server.gui.add_button("⏸ Pause")
    b_fwd = server.gui.add_button("Play ▶")
    # The frame slider lives in its own folder so it can be destroyed and rebuilt
    # per case: viser slider bounds are immutable (only `.value` is settable) and
    # the cases run 80..188 frames, so one shared slider would either clip the
    # long cases or leave a dead tail on the short ones.
    frame_folder = server.gui.add_folder("Timeline")

    def rebuild_slider(n_frames: int) -> None:
        old = state.get("slider")
        if old is not None:
            old.remove()
        with frame_folder:
            slider = server.gui.add_slider(
                "Frame", min=0, max=max(1, n_frames - 1), step=1, initial_value=0)
        slider.on_update(lambda _=None: _draw())
        state["slider"] = slider

    def _draw() -> None:
        scene = built.get(state["case"])
        slider = state.get("slider")
        if scene is not None and slider is not None:
            scene.render(int(slider.value))

    def select(case_id: str) -> None:
        prev = state.get("case")
        if prev == case_id:
            return
        scene = build(case_id)
        if prev is not None and prev in built:
            built[prev].set_visible(False)
        scene.set_visible(True)
        state["case"] = case_id
        rebuild_slider(scene.n_frames)
        info.value = f"{scene.n_frames} frames  (stride {args.stride})"
        scene.render(0)

    case_dd.on_update(lambda _=None: select(case_dd.value))
    b_rev.on_click(lambda _=None: state.update(dir=-1))
    b_pause.on_click(lambda _=None: state.update(dir=0))
    b_fwd.on_click(lambda _=None: state.update(dir=1))

    if args.preload:
        for c in cases:
            build(c).set_visible(False)
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
            n = scene.n_frames
            v = int(slider.value) + state["dir"]
            if loop_cb.value:
                v %= n                      # wrap both ways
            elif not 0 <= v < n:
                state["dir"] = 0            # stop at the end
                continue
            slider.value = v                # fires on_update -> _draw()

    threading.Thread(target=loop, daemon=True).start()
    bound = server.get_port()
    # flush=True: stdout is block-buffered when redirected or piped, which is
    # exactly how this gets launched (nohup / tee / background). Without it the
    # one line that matters -- the URL -- never reaches the log.
    if bound != args.port:
        print(f"[warn] port {args.port} was busy; viser bound {bound} instead", flush=True)
    print(f"\n[viser] serving on http://localhost:{bound}  "
          f"(SSH: ssh -L {bound}:localhost:{bound} <host>)", flush=True)
    print(f"[viser] {len(cases)} case(s) x {len(arms)} arms: {arms}. "
          f"Pick a case in the GUI; Loop is on by default. Ctrl-C to stop.", flush=True)
    print("[viser] one scene, one camera, one timeline -- cross-arm pose comparison "
          "is valid here (unlike the mp4s, see module docstring).", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
