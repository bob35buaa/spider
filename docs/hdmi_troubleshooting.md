# HDMI Workflow Troubleshooting Guide

This document records all issues encountered when setting up and running the HDMI (Humanoid-Object Interaction) workflow with SPIDER, and their solutions.

## Environment

- OS: Ubuntu with NVIDIA RTX 6000 Ada GPUs
- Python: 3.10 (conda env `hdmi`)
- HDMI path: `/home/xiayb/pHRI_workspace/Loco-Manipulation-projects/HDMI`
- IsaacLab path: `/home/xiayb/pHRI_workspace/Loco-Manipulation-projects/IsaacLab`
- Command: `python examples/run_hdmi.py task=move_suitcase joint_noise_scale=0.2 knot_dt=0.2 ctrl_dt=0.04 horizon=0.8 +data_id=1 viewer="rerun" rerun_spawn=true +save_rerun=true +save_metrics=false max_sim_steps=-1`

---

## Issue 1: Missing Config fields (`num_resamples`, `resample_ratio`)

**Error:**
```
TypeError: Config.__init__() got an unexpected keyword argument 'num_resamples'
```

**Cause:** `examples/config/hdmi.yaml` defines `num_resamples` and `resample_ratio` but these fields don't exist in `spider/config.py`.

**Fix:** Add fields to `spider/spider/config.py`:
```python
num_resamples: int = 0
resample_ratio: float = 0.2
```

---

## Issue 2: `isaaclab` package not installed

**Error:**
```
ModuleNotFoundError: No module named 'isaaclab'
```

**Cause:** The core `isaaclab` package exists at `IsaacLab/source/isaaclab/` but wasn't installed into the hdmi conda env.

**Fix:**
```bash
conda run -n hdmi pip install --no-deps -e /path/to/IsaacLab/source/isaaclab
```

---

## Issue 3: Missing Python packages (`toml`, `prettytable`)

**Error:**
```
ModuleNotFoundError: No module named 'toml'
ModuleNotFoundError: No module named 'prettytable'
```

**Cause:** These are isaaclab/HDMI transitive dependencies not listed in `setup.py`.

**Fix:**
```bash
conda run -n hdmi pip install toml prettytable
```

---

## Issue 4: IsaacSim EULA not accepted

**Error:**
```
EOFError: EOF when reading a line
```
(from `omni/kit_app.py` EULA prompt)

**Cause:** IsaacSim requires EULA acceptance, which fails in non-interactive mode.

**Fix:** Create the acceptance marker file:
```bash
echo "yes" > /path/to/miniconda3/envs/hdmi/lib/python3.10/site-packages/omni/EULA_ACCEPTED
```

Or set env var: `OMNI_KIT_ACCEPT_EULA=Y`

---

## Issue 5: HDMI backend defaults to `isaac`, triggering heavy Omniverse imports

**Error:**
```
ModuleNotFoundError: No module named 'omni.client'
```

**Cause:** `active_adaptation.get_backend()` defaults to `"isaac"`, which causes isaaclab modules to import the full Omniverse SDK (omni.physics, omni.client, etc.). SPIDER only uses the MuJoCo backend.

**Fix (spider side):** In `spider/spider/simulators/hdmi.py`, set backend before importing:
```python
import active_adaptation
active_adaptation.set_backend("mujoco")
from active_adaptation.envs.locomotion import SimpleEnv
```

---

## Issue 6: Unconditional isaaclab imports in HDMI code

**Error:** Various `ModuleNotFoundError` for `omni.*`, `isaacsim`, `carb`, etc.

**Cause:** Several HDMI files import isaaclab/omniverse modules at the top level without checking the backend. These imports cascade into heavy Omniverse SDK dependencies.

**Affected files and fixes:**

### `active_adaptation/envs/mdp/base.py`

Made `isaacsim`, `carb`, `omni`, and `isaaclab.utils.math.quat_mul` imports conditional:
```python
import active_adaptation

if active_adaptation.get_backend() == "isaac":
    import isaacsim
    import carb
    import omni
    from isaaclab.utils.math import quat_mul
else:
    carb = None
    omni = None
    def quat_mul(q1, q2):
        """Minimal quat_mul for mujoco backend (w,x,y,z convention)."""
        w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
        w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
        return torch.cat([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)
```

### `active_adaptation/envs/mdp/__init__.py`

Wrapped `observations` import in try/except (it imports `isaaclab.assets` which requires full Omniverse SDK):
```python
from .base import *
from .randomizations import *
try:
    from .observations import *
except (ImportError, ModuleNotFoundError):
    pass
from .rewards import *
from .terminations import *
from .commands import *
from .action import *
from .addons import *
```

### `active_adaptation/envs/__init__.py`

Made `MJArticulationCfg` import optional (it triggers `isaaclab` imports):
```python
try:
    from .mujoco import MJArticulationCfg
except ImportError:
    pass
from .locomotion import SimpleEnv
```

---

## Issue 7: Omniverse stub modules needed

**Cause:** Even with conditional imports, some isaaclab utility modules (e.g., `isaaclab.utils.math`, `isaaclab.utils.string`) are used by both backends but import `omni.log` at the top level.

**Fix:** Create minimal stub modules in the conda env's site-packages:

### `omni/log.py`
```python
import logging
_logger = logging.getLogger("omni.log")

def warn(msg, *args, **kwargs):
    _logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    _logger.error(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    _logger.info(msg, *args, **kwargs)
```

### `omni/client.py`
```python
class Result:
    OK = 0
    ERROR = 1

def stat(url): return Result(), None
def list(url): return Result(), []
def read_file(url): return Result(), b"", None
```

### `omni/appwindow.py`
```python
def get_default_app_window():
    class _Window:
        def get_keyboard(self): return None
    return _Window()
```

### `omni/physics/tensors/impl/api.py`
```python
# Empty stub
```
(with `__init__.py` files in each directory)

### `carb/__init__.py`
```python
class _Input:
    class KeyboardEventType:
        KEY_PRESS = 0
        KEY_RELEASE = 1
    def acquire_input_interface(self): return self
    def subscribe_to_keyboard_events(self, *args, **kwargs): return None

input = _Input()
```

### `pxr/__init__.py`
```python
class UsdGeom: pass
class UsdPhysics: pass
```

---

## Issue 8: GLFW crashes without X11 display (headless server)

**Error:**
```
ERROR: could not initialize GLFW
GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'
```

**Cause:** `import mujoco.viewer` and `mujoco.viewer.launch_passive()` trigger GLFW initialization, which calls C-level `exit()` without a display — Python `try/except` cannot catch it.

**Fix (HDMI side):** In `active_adaptation/envs/mujoco.py`:

1. Remove top-level `import mujoco.viewer` (keep only `import mujoco`)
2. Guard viewer creation with DISPLAY check:
```python
self.viewer = None
if os.environ.get("DISPLAY") and not os.environ.get("HDMI_HEADLESS"):
    try:
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(...)
    except Exception:
        pass
```
3. Guard all `self.viewer` usages with `if self.viewer is not None`
4. Make `MJSim.has_gui()` return `self.scene.viewer is not None`
5. Add `import os` to the file

---

## Issue 9: Wrong asset path and missing JSON config

**Error:**
```
FileNotFoundError: .../assets_mjcf/g1_23dof/g1_29dof_nohand.json
```

**Cause:** `assets_mjcf/__init__.py` references directory `g1_23dof` but the actual directory is `g1_29dof_nohand`. Also, the `.json` config file was missing.

**Fix:**

1. Fix path in `active_adaptation/assets_mjcf/__init__.py`:
```python
# Before:
ROBOTS["g1_29dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_23dof", "g1_29dof_nohand.xml"),
    **json.load(open(os.path.join(PATH, "g1_23dof", "g1_29dof_nohand.json"))),

# After:
ROBOTS["g1"] = MJArticulationCfg(
    mjcf_path=os.path.join(PATH, "g1_29dof_nohand", "g1_29dof_nohand.xml"),
    **json.load(open(os.path.join(PATH, "g1_29dof_nohand", "g1_29dof_nohand.json"))),
```

2. The key was also changed from `"g1_29dof"` to `"g1"` to match the YAML config (`robot.name: g1`).

3. The JSON file (`g1_29dof_nohand.json`) was created with `init_state`, `actuators`, `body_names_isaac`, and `joint_names_isaac` extracted from the IsaacLab robot config.

---

## Issue 10: Body/joint name mismatch between IsaacLab and MuJoCo

**Error:**
```
ValueError: 'pelvis_contour_link' is not in list
```

**Cause:** `body_names_isaac` in the JSON contains bodies that exist in the IsaacLab URDF but not in the simplified MJCF model (e.g., `pelvis_contour_link`, `d435_link`, `head_link`, etc.).

**Fix:** In `active_adaptation/envs/mujoco.py`, filter the name lists to intersection before building index mappings:
```python
# Filter to only include those present in both
self.joint_names_isaac = [j for j in self.joint_names_isaac if j in self.joint_names_mjc]
self.body_names_isaac = [b for b in self.body_names_isaac if b in self.body_names_mjc]
```

---

## Issue 11: `MJScene` missing `rigid_objects` attribute

**Error:**
```
AttributeError: 'MJScene' object has no attribute 'rigid_objects'
KeyError: 'suitcase'
```

**Cause:** The `RobotObjectTracking` command tries to access `self.env.scene.rigid_objects["suitcase"]` but `MJScene` doesn't have this attribute.

**Fix:** Multiple changes:
1. Created `MJRigidObject` class in `mujoco.py` that wraps a free-body in the shared MuJoCo scene
2. Created `MjFilteredContactSensor` stub for eef-object contact detection
3. Modified `locomotion.py` to detect `object_asset_name` in config and use combined XML (`g1_29dof_nohand-suitcase.xml`)
4. Register rigid object and contact sensors in `MJScene`

---

## Issue 12: Missing IsaacLab-compatible attributes on MuJoCo classes

**Errors:**
```
AttributeError: 'MJArticulationData' object has no attribute 'body_link_pos_w'
AttributeError: 'MJArticulation' object has no attribute 'write_root_link_pose_to_sim'
AttributeError: 'MJArticulationData' object has no attribute 'soft_joint_pos_limits'
```

**Cause:** HDMI's `RobotObjectTracking` and other commands use IsaacLab API names (`body_link_pos_w`, `write_root_link_pose_to_sim`, `soft_joint_pos_limits`) but MuJoCo wrappers use different names.

**Fix:** Added compatibility aliases to `MJArticulationData` and `MJArticulation`:
- `body_link_pos_w` → `body_pos_w` (property)
- `body_link_quat_w` → `body_quat_w` (property)
- `root_link_pos_w` → `root_pos_w` (property)
- `root_link_quat_w` → `root_quat_w` (property)
- `body_com_lin_vel_w` → `body_lin_vel_w` (property)
- Added `write_root_link_pose_to_sim()` and `write_root_com_velocity_to_sim()` methods
- Added `soft_joint_pos_limits` and `soft_joint_vel_limits` fields

---

## Issue 13: `cfg.num_envs` vs MuJoCo single-env mismatch

**Error:**
```
RuntimeError: shape '[1024, -1]' is invalid for input of size 3
```

**Cause:** SPIDER sets `cfg.num_envs = num_samples` (1024) but MuJoCo backend is single-env. HDMI observations/commands create tensors sized `[num_envs, ...]` but data is `[1, ...]`.

**Fix:** Set `cfg.num_envs = 1` in spider's `hdmi.py` since SPIDER handles batching via Warp, not through HDMI's env.

---

## Issue 14: `MJSim` missing `cfg`, `data`, `wp_data` attributes

**Errors:**
```
AttributeError: 'MJSim' object has no attribute 'cfg'
AttributeError: 'MJSim' object has no attribute 'data'
AttributeError: 'MJSim' object has no attribute 'wp_data'
```

**Cause:** Spider's `run_hdmi.py` and `hdmi.py` expect Warp-style sim attributes that HDMI's `MJSim` doesn't have.

**Fix:**
- `cfg.nconmax`/`njmax`: Use fallback from `config.nconmax_per_env`/`config.njmax_per_env`
- `sim.data`: Read directly from `env.sim.mj_data` when Warp data not available
- `sim.wp_data`: Create via `mjwarp.put_data()` in `setup_env` alongside `data_wp_prev`

---

## Issue 15: `run_hdmi.py` missing `get_terminate` in `make_rollout_fn` call

**Error:**
```
TypeError: make_rollout_fn() missing 1 required positional argument: 'copy_sample_state'
```

**Cause:** `make_rollout_fn` requires 10 args (including `get_terminate`), but `run_hdmi.py` only passed 9.

**Fix:** Added `get_terminate` import and parameter in the call.

---

## Issue 16: Device mismatch (CPU vs CUDA)

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Cause:** `ctrls` initialized from reference trajectory on CPU, but sampling happens on CUDA.

**Fix:** `ctrls = ctrl_ref[: config.horizon_steps].to(config.device)`

---

## Issue 17 (Current): Batch simulation architecture gap

**Error:**
```
RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([1]) and value.shape=torch.Size([1024, 23])
```

**Cause:** SPIDER's `step_env` sends 1024 parallel samples to HDMI's env, but HDMI's MuJoCo backend only supports 1 environment. SPIDER's MJWP simulator uses Warp for GPU-batched simulation (1024 worlds in parallel), but `step_env` in `hdmi.py` currently calls HDMI's `env.step()` which is single-env.

**Status: BLOCKING** — `step_env` needs to be rewritten to use Warp-based batch stepping (via `mjwarp.step()`) instead of HDMI's `env.step()`. The Warp data structures (`wp_data`) are already created; the stepping logic needs to bypass HDMI's env and use Warp directly for physics, while still using HDMI's reward/command system for evaluation.
