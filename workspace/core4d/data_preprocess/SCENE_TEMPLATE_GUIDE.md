# CORE4D SPIDER scene 模板制作指南

这个文档说明当一个 CORE4D case 没有现成 `source_scene_task` 时，如何制作 SPIDER 需要的 MuJoCo scene 模板。这里的模板指：

```text
example_datasets/processed/core4d/unitree_g1/humanoid_object/<task>/
  scene.xml
  scene_act.xml
  scene_act_meta.json
  task_info.json   # 推荐保留，主要用于 provenance/audit
```

其中真正被 SPIDER 数据转换硬依赖的是 `scene.xml`。`scene_act.xml` 是 MJWP/CEM 运行时需要的 actuator 版本场景，应在 `trajectory_kinematic.npz` 生成后再由 `generate_scene_act.py` 生成。

## 什么时候需要制作模板

如果 TSV 里的 `source_scene_task` 已经存在，例如 `box023_person1`，E077 pipeline 可以直接复制模板并更新 object 初始 pose。

如果没有同物体或同 case 的 SPIDER scene，就需要先制作一个新的 `scene.xml`。不能把 `source_scene_task` 留空，因为 `spider/process_datasets/core4d.py` 会在生成 `trajectory_kinematic.npz` 前加载：

```text
example_datasets/processed/core4d/unitree_g1/humanoid_object/<target_task>/scene.xml
```

## 推荐制作流程

1. 选择一个已有 SPIDER scene 作为基础模板。
2. 复制到新 task 目录。
3. 修改 object 相关的 mesh、material、body 初始位姿、collision、mass/inertia。
4. 保持 Unitree G1 robot、joint/actuator 顺序、hand contact sites 不变。
5. 用 MuJoCo load 校验 `scene.xml`。
6. 跑 Holosoma/OmniRetarget 和 `spider/process_datasets/core4d.py` 生成 `trajectory_kinematic.npz`。
7. 用 `workspace/core4d/scripts/convert/generate_scene_act.py` 生成 `scene_act.xml`。
8. 做最终校验：`scene.xml` / `scene_act.xml` 都能 load，qpos 维度和 object qpos layout 一致。

## 模板选择依据

优先级如下：

| 优先级 | 模板来源 | 适用情况 |
|--------|----------|----------|
| 1 | 同一物体、不同 person/session | 最稳，例如 `box023_person1 -> box023_person2`，只需要更新 object 初始 pose 和 provenance。 |
| 2 | 同类别近似物体 | 例如 box 到 box、bucket 到 bucket；需要替换 mesh/collision/mass/inertia。 |
| 3 | 其它 CORE4D humanoid_object scene | 可以作为 robot skeleton 基础，但 object 相关字段必须完整重建。 |

不要直接从 HDMI scene 作为 CORE4D SPIDER scene 模板，除非明确知道 robot/body/site/joint 命名、qpos layout 和 MJWP 配置都兼容。当前 CORE4D pipeline 默认 qpos 为：

```text
robot root freejoint: qpos[0:7]
robot joints:         qpos[7:36]
object position:      qpos[36:39]
object quaternion:    qpos[39:43]  # wxyz
```

## 必须修改的内容

### 1. task 目录

新 task 目录：

```text
example_datasets/processed/core4d/unitree_g1/humanoid_object/<target_task>/
```

依据：

- `<target_task>` 应与后续 `spider/process_datasets/core4d.py --task <target_task>` 一致。
- 也应与实验 config 里的 `task=<target_task>` 一致。

### 2. object mesh asset

`scene.xml` 的 `<asset>` 中需要有新物体 mesh，例如：

```xml
<mesh name="box023"
      file="../../../../../example_datasets/processed/core4d/assets/objects/box023/box023_m.obj"
      scale="1 1 1" />
```

依据：

- mesh 文件来自 CORE4D object model，例如：
  `CORE4D_Real/object_models/box/box023_m.obj`
- repo 内 scene 通常引用：
  `example_datasets/processed/core4d/assets/objects/<object_slug>/<mesh_file>`

怎么改：

- 替换 `mesh name`，避免沿用旧物体名。
- 替换 `file` 到新物体 obj。
- 默认 `scale="1 1 1"`，除非你明确知道上游 retarget qpos 和 mesh 使用了同一缩放。

注意：

- mesh 路径必须能被 MuJoCo 从 `scene.xml` 所在目录解析。
- visual mesh 不等于 collision geom，collision 需要单独设置。

### 3. material

`scene.xml` 的 `<asset>` 中通常有物体材质，例如：

```xml
<material name="box_material" rgba="0.6 0.4 0.2 1" />
```

依据：

- 只影响可视化，不决定动力学。
- 可沿用同类物体材质，也可以按物体类别命名。

怎么改：

- 如果 `object_visual` 引用新 material，就同步改 `<material name=...>`。
- 保持名称不冲突即可。

### 4. object body 初始位姿

`worldbody` 中必须有名为 `object` 的 body：

```xml
<body name="object" pos="0.1550 -0.1240 0.3100" quat="1.000000 0.000000 0.000000 0.000000">
  <freejoint name="object_joint" />
  ...
</body>
```

依据：

- `pos` 取 retarget 后、并按最终 SPIDER 窗口裁剪后的第一帧：
  `trimmed_npz["qpos"][0, 36:39]`
- `quat` 取：
  `trimmed_npz["qpos"][0, 39:43]`
- 如果 scene 是在 retarget 前就要创建，可以先用 converted/raw 物体初始位姿占位，随后在 pipeline 中用 retarget qpos 第一帧 patch 一次。

怎么改：

- 保持 body 名称必须是 `object`。
- 保持 object 为 freejoint 版本 scene 时使用：
  `<freejoint name="object_joint" />`
- 写入 `pos` 和 `quat`。quat 是 MuJoCo/wxyz 顺序。

注意：

- `scene.xml` 是 freejoint 版本，`nq` 应为 43。
- `scene_act.xml` 不是手工改 object quat，它会把 object freejoint 改成 3 slide + 3 hinge。

### 5. object visual geom

推荐保留：

```xml
<geom name="object_visual"
      type="mesh"
      mesh="<mesh_name>"
      material="<material_name>"
      group="2"
      contype="0"
      conaffinity="0" />
```

依据：

- visual geom 只用于显示，不参与碰撞。
- `mesh` 必须引用 asset 中的新物体 mesh。

### 6. object collision geom

当前 CORE4D SPIDER scene 通常用 box collision proxy：

```xml
<geom name="object_collision"
      type="box"
      size="0.1531 0.1568 0.1766"
      rgba="0.6 0.4 0.2 0.3"
      group="3"
      contype="1"
      conaffinity="1"
      friction="1 0.005 0.0001"
      condim="3" />
```

依据：

- `size` 是 MuJoCo box half-extents。
- 可以从 object mesh 的 bounding box 得到：

```python
import trimesh
mesh = trimesh.load("box023_m.obj")
half_extents = mesh.bounding_box.extents / 2
```

怎么改：

- 替换 `size` 为新物体 half-extents。
- 保持 geom 名称为 `object_collision`，因为很多评估和接触逻辑按这个名字查找。
- 通常沿用现有 friction/condim/contype/conaffinity。

注意：

- box collision 是近似。对 chair/desk 这种非凸或空心物体，box proxy 可能会显著改变接触行为。
- 如果后续实验依赖精细接触，需要另开实验验证 mesh collision、多个 box proxy 或 hand/object contact proxy 的合理性。

### 7. object mass 和 inertia

`object` body 内需要 inertial：

```xml
<inertial pos="0 0 0" mass="5.0" diaginertia="0.15 0.15 0.1" />
```

依据：

- 如果 CORE4D 或实物信息有质量，优先使用真实质量。
- 如果没有，使用同类别经验值，并在日志中记录假设。
- box inertia 可以用 box 近似：

```python
a, b, c = half_extents * 2
Ix = mass / 12 * (b * b + c * c)
Iy = mass / 12 * (a * a + c * c)
Iz = mass / 12 * (a * a + b * b)
```

怎么改：

- 替换 `mass`。
- 替换 `diaginertia`。
- 保持 `inertial pos="0 0 0"`，除非你明确知道 mesh 的质心偏移。

注意：

- mass/inertia 会直接影响 CEM 动力学。没有真实依据时，要在实验日志中把它标记为建模假设。

### 8. hand contact sites

robot 手部必须保留两个 contact site，当前 `spider/process_datasets/core4d.py` 会按名称自动找：

```text
contact_left_hand
contact_right_hand
```

依据：

- SPIDER conversion 会遍历 MuJoCo sites，选择名字里同时包含 `contact` 和 `hand` 的 site。
- 期望数量为 2。

怎么改：

- 通常不要改 robot 部分。
- 如果从非常旧的模板复制，确认左右手 site 存在，且 site 位置与当前 hand collision 设置一致。

注意：

- 不要改 robot joint 名称和 actuator 顺序。MJWP 的 robot control 默认对应 29 个 robot joint target。

### 9. contact pairs

`scene.xml` 应包含 hand-object 和 object-floor 接触 pair：

```xml
<pair name="left_hand_object" geom1="lh" geom2="object_collision" ... />
<pair name="right_hand_object" geom1="rh" geom2="object_collision" ... />
<pair name="object_floor" geom1="object_collision" geom2="floor" ... />
```

依据：

- `object_collision` 是物体碰撞代理。
- `lh/rh` 是当前模板里的手部 collision geom 名称。

怎么改：

- 如果沿用现有 Unitree G1 CORE4D 模板，通常不需要改。
- 如果换了 hand collision 方案，例如 sphere 到 3-box，需要同步确认 contact pair 的 `geom1` 名称。

注意：

- scene 能 load 不代表接触 pair 正确。后续需要用视频或 contact metric 检查接触行为。

## 可以参考的现有脚本

`workspace/core4d/scripts/convert/setup_new_cases.py` 是历史上为 `bucket010/chair022/desk005` 生成 scene 的例子。它做了这些事：

- 从已有 `bucket005_person1/scene.xml` 读取模板。
- 替换 object mesh asset。
- 替换 material。
- 用 retarget qpos 第一帧设置 object 初始 pose。
- 用 mesh bounding box 生成 `object_collision` half-extents。
- 用 box 近似计算 inertia。
- MuJoCo load 校验 `nq=43`。

这个脚本可以作为写新生成器的参考，但它不是完全通用工具：里面的路径、物体列表、模板物体名都是硬编码的。

## 最小校验清单

### scene.xml

必须通过：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import mujoco

task = "box023_person2"
scene = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object") / task / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(scene))
site_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    for i in range(model.nsite)
]
contact_hand_sites = [s for s in site_names if s and "contact" in s and "hand" in s]
print("nq/nv/nu =", model.nq, model.nv, model.nu)
print("contact hand sites =", contact_hand_sites)
assert model.nq == 43
assert len(contact_hand_sites) == 2
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object") >= 0
assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision") >= 0
PY
```

### trajectory_kinematic.npz

生成 SPIDER trajectory 后必须确认：

```bash
.venv/bin/python - <<'PY'
import numpy as np

path = "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/0/trajectory_kinematic.npz"
data = np.load(path, allow_pickle=True)
print(data["qpos"].shape, data["qvel"].shape, data["ctrl"].shape)
assert data["qpos"].shape[1] == 43
assert data["ctrl"].shape[1] == 29
PY
```

### scene_act.xml

`scene_act.xml` 应在 `trajectory_kinematic.npz` 存在后生成，因为 `generate_scene_act.py` 会用 object quaternion 序列选择 Euler convention。

```bash
.venv/bin/python workspace/core4d/scripts/convert/generate_scene_act.py
```

如果只想给某个 task 生成，可以直接调用函数：

```bash
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "workspace/core4d/scripts/convert")
from generate_scene_act import generate_scene_act

generate_scene_act("box023_person2")
PY
```

校验：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import mujoco

task = "box023_person2"
scene = Path("example_datasets/processed/core4d/unitree_g1/humanoid_object") / task / "scene_act.xml"
model = mujoco.MjModel.from_xml_path(str(scene))
print("nq/nv/nu =", model.nq, model.nv, model.nu)
assert model.nu == 35  # 29 robot position actuators + 6 object position actuators
PY
```

## 常见错误

| 错误 | 现象 | 处理 |
|------|------|------|
| mesh 路径不对 | MuJoCo load 失败，找不到 obj | 用相对 `scene.xml` 的路径，或确认 mesh 已复制到 `example_datasets/processed/core4d/assets/objects/` |
| object body 名称不是 `object` | 评估脚本或 scene_act 生成失败 | 保持 `<body name="object">` |
| object collision 名称不是 `object_collision` | contact/eval 脚本找不到物体碰撞体 | 保持 `<geom name="object_collision">` |
| qpos layout 不一致 | `spider/process_datasets/core4d.py` assert 或 MJWP 初始化错位 | 保持 freejoint scene `nq=43`，object qpos 为 `36:43` |
| contact hand sites 数量不是 2 | SPIDER conversion assert | 保留 `contact_left_hand` 和 `contact_right_hand` |
| 手部 collision geom 名称和 contact pair 不一致 | 视觉上手碰不到物体或接触异常 | 检查 `lh/rh` 或 3-box hand collision 的 geom 名称 |
| mass/inertia 纯拍脑袋 | CEM 行为不稳定，物体过轻/过重 | 记录假设，优先找真实质量，至少用 bbox + mass 计算一致 inertia |
| 对 p1/p2 retarget qpos 直接合并 | 双人场景物体轨迹不一致 | E077 已发现 Holosoma retarget preprocess 会按 person `smpl_scale` 改 object motion；双机器人前需要 common-scale/common-world alignment |

## 建议写入 `task_info.json` 的内容

`task_info.json` 不是当前 `spider/process_datasets/core4d.py` 的硬依赖，但建议保留，方便追踪模板来源：

```json
{
  "source_scene": "example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene.xml",
  "source_qpos": "workspace/core4d/results/E077/holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz",
  "object_mesh": "CORE4D_Real/object_models/box/box023_m.obj",
  "object_collision": "bbox_half_extents_from_mesh",
  "object_mass_source": "assumed_or_measured",
  "object_initial_pos": [0.155, -0.124, 0.310],
  "object_initial_quat": [1.0, 0.0, 0.0, 0.0]
}
```

如果 mass/inertia 是假设值，要明确写入 `object_mass_source`，后续实验分析时不能把它当作数据真值。
