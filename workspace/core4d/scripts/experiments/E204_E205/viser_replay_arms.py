#!/usr/bin/env python3
"""Replay the three arms (noPRG / PRG / G1A2) of one E178 bucket case side-by-side
in a headless viser server (browser-rendered -> works on GL-less boxes).

Loads each arm's CEM rollout (trajectory_mjwp_act.npz) + its scene, runs CPU FK
(mj_forward) per frame, and drives viser body frames on a single shared timeline.
Arms are offset along X so they play synchronized side by side; a "Frame" slider
+ play/pause let you scrub. No GPU / OpenGL needed (viser renders in the browser).

Arm -> (scene, rollout npz):
  noPRG = E204 (scene_act_E204_contactAlignedTop_noPRG + results/E204 rollout)
  PRG   = E178 (scene_act_E178_contactAlignedTop        + results/E178 rollout)
  G1A2  = E205 (scene_act_E205_contactAlignedTop_gravcomp + results/E205 rollout)

Usage (on a machine that can reach the buckets; access via SSH port-forward 8080):
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E204_E205/viser_replay_arms.py \
      --case bucket003_20231018_001_p1 [--arms noPRG,PRG,G1A2] [--port 8080] [--stride 1]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E204_E205"))

import mujoco  # noqa: E402

import e204e205_common as C  # noqa: E402

# Load viser_viewer.py DIRECTLY (bypass spider.viewers.__init__, which imports the
# warp/torch-backed mjwp viewer -> heavy + fragile on loaded boxes). viser_viewer
# itself only needs mujoco/trimesh/viser/numpy/loguru.
import importlib.util as _ilu  # noqa: E402
_vv_path = REPO / "spider/viewers/viser_viewer.py"
_spec = _ilu.spec_from_file_location("viser_viewer_standalone", _vv_path)
V = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(V)  # type: ignore

ARM_COLOR = {"noPRG": (0.90, 0.35, 0.35), "PRG": (0.45, 0.65, 0.95), "G1A2": (0.45, 0.85, 0.55)}
E178_CEM_GLOB = "workspace/core4d/results/E178/s6_downstream/cem/full/*{case}*/trajectory_mjwp_act.npz"


def arm_scene_and_npz(arm: str, case_id: str) -> tuple[Path, Path]:
    if arm == "noPRG":
        return C.arm_scene_path("noprg_e204", case_id), C.result_npz("noprg_e204", case_id, "full")
    if arm == "G1A2":
        return C.arm_scene_path("g1a2_e205", case_id), C.result_npz("g1a2_e205", case_id, "full")
    if arm == "PRG":
        scene = C.e178_scene_path(case_id)
        hits = sorted(REPO.glob(E178_CEM_GLOB.format(case=case_id)))
        if not hits:
            raise FileNotFoundError(f"E178 PRG rollout not found for {case_id} "
                                    f"(glob {E178_CEM_GLOB.format(case=case_id)})")
        return scene, hits[0]
    raise ValueError(f"unknown arm {arm}")


def load_qpos(npz_path: Path, nq: int) -> np.ndarray:
    """Return the simulated (T, nq) qpos from a rollout npz.

    run_mjwp saves qpos as (T, 2, nq) = [sim, reference]; take index 0 (sim).
    Also handles a plain (T, nq) array.
    """
    with np.load(npz_path, allow_pickle=True) as d:
        a = np.asarray(d["qpos"], dtype=np.float64)
    if a.ndim == 3 and a.shape[2] == nq:
        return a[:, 0, :]           # [sim, ref] -> sim
    if a.ndim == 2 and a.shape[1] == nq:
        return a
    raise KeyError(f"qpos shape {a.shape} incompatible with model.nq={nq} in {npz_path}")


def fk_frames(model: mujoco.MjModel, qpos: np.ndarray, body_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="bucket003_20231018_001_p1")
    ap.add_argument("--arms", default="noPRG,PRG,G1A2")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--stride", type=int, default=1, help="frame subsample")
    ap.add_argument("--spacing", type=float, default=1.6, help="X offset between arms (m)")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    V.init_viser(app_name=f"E204E205 arms · {args.case}")
    server = V._get_server()

    loaded = []  # (arm, handles[(handle,bid)], xpos, xquat, offset)
    n_frames = None
    for i, arm in enumerate(arms):
        scene, npz = arm_scene_and_npz(arm, args.case)
        spec = mujoco.MjSpec.from_file(str(scene))
        V._ensure_names(spec)
        model = spec.compile()
        qpos = load_qpos(npz, model.nq)[:: args.stride]
        handles = V.build_and_log_scene_from_spec(spec=spec, model=model, xml_path=scene,
                                                  entity_root=f"arm_{arm}", build_ref=False)
        body_ids = [bid for _h, bid in handles]
        xpos, xquat = fk_frames(model, qpos, body_ids)
        offset = np.array([i * args.spacing - (len(arms) - 1) * args.spacing / 2.0, 0.0, 0.0])
        loaded.append((arm, handles, xpos, xquat, offset))
        n_frames = len(qpos) if n_frames is None else min(n_frames, len(qpos))
        try:
            server.scene.add_label(f"arm_{arm}/label", text=arm, position=tuple(offset + [0, 0, 1.7]))
        except Exception:
            pass
        print(f"[loaded] {arm}: scene={scene.name} npz={npz.parent.name} frames={len(qpos)} bodies={len(body_ids)}", flush=True)

    def render(frame: int) -> None:
        f = max(0, min(frame, n_frames - 1))
        for _arm, handles, xpos, xquat, offset in loaded:
            for j, (handle, _bid) in enumerate(handles):
                handle.position = tuple(xpos[f, j] + offset)
                handle.wxyz = tuple(xquat[f, j])

    slider = server.gui.add_slider("Frame", min=0, max=max(1, n_frames - 1), step=1, initial_value=0)
    fps_slider = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=int(args.fps))
    play = {"dir": 0}
    b_rev = server.gui.add_button("◀ Play")
    b_pause = server.gui.add_button("⏸ Pause")
    b_fwd = server.gui.add_button("Play ▶")
    b_rev.on_click(lambda _=None: play.update(dir=-1))
    b_pause.on_click(lambda _=None: play.update(dir=0))
    b_fwd.on_click(lambda _=None: play.update(dir=1))
    slider.on_update(lambda _=None: render(int(slider.value)))
    render(0)

    def loop() -> None:
        while True:
            time.sleep(1.0 / max(1.0, float(fps_slider.value)))
            if play["dir"] != 0:
                v = int(slider.value) + play["dir"]
                slider.value = max(0, min(v, n_frames - 1))

    threading.Thread(target=loop, daemon=True).start()
    print(f"\n[viser] serving on http://localhost:{args.port}  (SSH: ssh -L {args.port}:localhost:{args.port} <host>)")
    print(f"[viser] {len(arms)} arms x {n_frames} frames. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
