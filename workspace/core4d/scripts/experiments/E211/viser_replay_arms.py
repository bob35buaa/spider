#!/usr/bin/env python3
"""Replay E211 g-curve arms of one desk007 case side-by-side in a headless viser server.

Loads each arm's CEM rollout (`trajectory_mjwp_act.npz`) + its scene, runs CPU FK
(`mj_forward`) per frame, and drives viser body frames on a single shared
timeline. Arms are offset along X so they play synchronised side by side; a
"Frame" slider + play/pause let you scrub. No GPU / OpenGL needed -- viser
renders in the browser, so this works on GL-less boxes where osmesa is the only
other option.

**Why this exists and the mp4s are not enough.** `_auto_video_camera`
(spider/viewers/__init__.py:262-291) recomputes lookat and radius every frame
from the union bbox of sim and ref bodies, so two arms' mp4s are shot from two
different, silently-moving cameras. E209 F6 and E210 F3 both established that the
SAME reference trajectory renders as visibly different poses across two videos --
one crouched, one upright. Cross-video pose comparison is therefore invalid.
Here every arm sits in ONE scene under ONE camera on ONE timeline, which is
exactly the comparison mp4 cannot give.

Arms (`--arms`, in the order they appear left to right):
  ref   the reference trajectory (from any run's qpos[:, 1, :] -- byte-identical
        across arms, verified in E209 log298 section 4-2)
  prg   g=0.0   E206 PRG baseline rollout
  G04   g=0.4   E211 Stage A          (the non-monotone arm)
  G06   g=0.6   E211 Stage A
  G08   g=0.8   E211 Stage A
  g1    g=1.0   E209 G1 rollout

Usage (access via SSH port-forward):
    MUJOCO_GL=disable .venv/bin/python \
      workspace/core4d/scripts/experiments/E211/viser_replay_arms.py \
      --case desk007_20231030_034_p1 [--arms ref,G06,G08] [--port 8080] [--stride 1]

    ssh -L 8080:localhost:8080 <host>   then open http://localhost:8080
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace/core4d/scripts/experiments/E211"))

import mujoco  # noqa: E402

import e211_common as C  # noqa: E402

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
ARM_G = {"ref": None, "prg": 0.0, "G04": 0.4, "G06": 0.6, "G08": 0.8, "g1": 1.0}
#: ref grey, baseline blue, then warmer as g rises.
ARM_COLOR = {
    "ref": (0.70, 0.70, 0.70), "prg": (0.45, 0.65, 0.95), "G04": (0.95, 0.80, 0.35),
    "G06": (0.45, 0.85, 0.55), "G08": (0.35, 0.80, 0.85), "g1": (0.90, 0.35, 0.35),
}
#: Which rollout `ref` is read out of. Any arm works (the reference is identical
#: across arms), but pinning it makes the choice explicit and reproducible.
REF_SOURCE_ARM = "prg"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="desk007_20231030_034_p1")
    ap.add_argument("--arms", default="ref,G06,G08",
                    help="left-to-right; ref|prg|G04|G06|G08|g1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--stride", type=int, default=1, help="frame subsample")
    ap.add_argument("--spacing", type=float, default=1.8, help="X offset between arms (m)")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    if args.case not in C.CASES:
        raise SystemExit(f"unknown case {args.case}; E211 scope is {list(C.CASES)}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_G]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known {list(ARM_G)}")

    # Construct the server directly rather than via `init_viser`, which takes no
    # host/port and so always uses ViserServer's default 8080 -- making a --port
    # flag silently a no-op. viser also auto-increments off a busy port, so the
    # requested port is not necessarily the bound one; report `get_port()`.
    _viser = V._lazy_import_viser()
    V._STATE.server = _viser.ViserServer(
        host=args.host, port=args.port, label=f"E211 g-sweep · {args.case}")
    server = V._get_server()

    loaded = []  # (arm, handles, xpos, xquat, offset)
    n_frames: int | None = None
    for i, arm in enumerate(arms):
        # `ref` rides on the baseline scene and reads the reference slice.
        src_arm = REF_SOURCE_ARM if arm == "ref" else arm
        scene, npz = arm_scene_and_npz(src_arm, args.case)
        if not npz.is_file():
            raise SystemExit(f"{arm}: rollout not found -> {npz}")
        spec = mujoco.MjSpec.from_file(str(scene))
        V._ensure_names(spec)
        model = spec.compile()
        qpos = load_qpos(npz, model.nq, which=1 if arm == "ref" else 0)[:: args.stride]
        handles = V.build_and_log_scene_from_spec(
            spec=spec, model=model, xml_path=scene,
            entity_root=f"arm_{arm}", build_ref=False)
        body_ids = [bid for _h, bid in handles]
        xpos, xquat = fk_frames(model, qpos, body_ids)
        offset = np.array(
            [i * args.spacing - (len(arms) - 1) * args.spacing / 2.0, 0.0, 0.0])
        loaded.append((arm, handles, xpos, xquat, offset))
        n_frames = len(qpos) if n_frames is None else min(n_frames, len(qpos))
        g = ARM_G[arm]
        label = "reference" if g is None else f"{arm}  g={g:.1f}"
        try:
            server.scene.add_label(f"arm_{arm}/label", text=label,
                                   position=tuple(offset + np.array([0.0, 0.0, 1.9])))
        except Exception:  # noqa: BLE001 - a missing label must not kill the replay
            pass
        print(f"[loaded] {label:16s} scene={scene.name} npz={npz.parent.name} "
              f"frames={len(qpos)} bodies={len(body_ids)}", flush=True)

    assert n_frames is not None
    if len({len(x[2]) for x in loaded}) > 1:
        print(f"[warn] arms have different frame counts "
              f"{ {a: len(x) for a, _h, x, _q, _o in loaded} }; "
              f"timeline truncated to {n_frames}", flush=True)

    def render(frame: int) -> None:
        f = max(0, min(frame, n_frames - 1))
        for _arm, handles, xpos, xquat, offset in loaded:
            for j, (handle, _bid) in enumerate(handles):
                handle.position = tuple(xpos[f, j] + offset)
                handle.wxyz = tuple(xquat[f, j])

    slider = server.gui.add_slider("Frame", min=0, max=max(1, n_frames - 1),
                                   step=1, initial_value=0)
    fps_slider = server.gui.add_slider("FPS", min=1, max=120, step=1,
                                       initial_value=int(args.fps))
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
    bound = server.get_port()
    if bound != args.port:
        print(f"[warn] port {args.port} was busy; viser bound {bound} instead")
    print(f"\n[viser] serving on http://localhost:{bound}  "
          f"(SSH: ssh -L {bound}:localhost:{bound} <host>)")
    print(f"[viser] {len(arms)} arms x {n_frames} frames, case={args.case}. Ctrl-C to stop.")
    print("[viser] one scene, one camera, one timeline -- cross-arm pose comparison "
          "is valid here (unlike the mp4s, see module docstring).")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
