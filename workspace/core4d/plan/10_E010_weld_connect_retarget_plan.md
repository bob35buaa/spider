# E010 实验计划：Weld(Connect)约束重定向 — G1 运动学可行性验证

## Context

### 背景：为什么需要这次实验

E001-E009 完整探索后，确认了一个事实：

> G1 单人无法物理抬起 box025。根因不是力（gravcomp 也只离地 8mm），而是**接触几何错位**——CORE4D 要双人从 ±x 端对夹，G1 从 -y 侧推。

但 E009 没有回答一个更根本的问题：**G1 的运动学能力（臂展、关节范围）是否足以完成 box025 搬运的身体动作？**

如果连运动学都不可行（手够不到正确位置、关节范围不够），那所有后续方案（mocap partner、双机器人、RL）都会在重定向阶段失败。

### 实验思路

用 MuJoCo 的 `connect` 等式约束将 G1 的手和物体"连接"起来，消除"抓握"这个变量。在此条件下跑 CEM 重定向，观察：
1. G1 的身体能否完成弯腰-抬起-搬运的动作序列
2. 物体是否能真正离地（qpos 实测）
3. 身体跟踪质量是否维持（pelvis_err）

### 为什么用 connect 而不是 weld

| 约束类型 | 约束内容 | 自由度 | 适用场景 |
|---------|---------|--------|---------|
| `weld` | 位置 + 姿态 (6DOF) | 0 | 刚性焊接 |
| `connect` | 仅位置 (3DOF) | 保留旋转3DOF | 类似球关节连接 |

G1 的 wrist_yaw_link 需要在搬运过程中调整朝向来适应箱子角度变化。`connect` 约束只锁定位置（确保手贴在物体上），保留手的旋转自由度，更接近真实的"用前臂托住"场景。

### 关键 insight

这不是"让物体能搬起来"的 hack，而是一个**诊断实验**：

- 如果 E010 成功 → 问题确实只在抓握，后续 E011(mocap partner) 有意义
- 如果 E010 失败 → G1 运动学本身不够，需要换物体/换机器人

## Claims（可验证声明）

| Claim | 最低证据 |
|-------|---------|
| C1: 强connect约束下, obj z_max ≥ 0.40m (qpos实测, 非reward) | `npz['qpos'][:, 36:39]` 中 z 分量 |
| C2: pelvis_err ≤ 0.15m (约束不拉垮身体跟踪) | `\|sim_pelvis - ref_pelvis\|` 均值 |
| C3: 视频确认物体被搬起且非旋转/翻滚假象 | 视频帧 + box底面4角z |
| C4: joint_err ≤ 0.10rad (关节不过应力) | 关节角度差异均值 |

## 改动清单

### 1. 生成 scene_connect.xml

**文件**: `workspace/core4d/scripts/generate_scene_connect.py`（新增）

**目的**: 在 scene_forearm.xml 基础上，添加手-物体 connect 等式约束。

```python
"""
从 scene_forearm.xml 生成 scene_connect.xml:
1. 读取现有 scene XML
2. 在物体 body 上添加接触 site（位于 -x 面中心偏左/右）
3. 添加 equality > connect 约束: hand_site ↔ object_contact_site
4. 写入新 XML
"""

import mujoco
import numpy as np
from lxml import etree

def generate_connect_scene(
    base_xml: str,           # scene_forearm.xml 路径
    output_xml: str,         # scene_connect.xml 输出路径
    contact_local_left: np.ndarray,   # 左手在物体局部坐标系的接触点 (3,)
    contact_local_right: np.ndarray,  # 右手在物体局部坐标系的接触点 (3,)
    solref: tuple = (-500, -50),      # 约束刚度 (可调)
):
    tree = etree.parse(base_xml)
    root = tree.getroot()

    # 在 object body 上添加接触 site
    obj_body = root.find(".//body[@name='object']")
    etree.SubElement(obj_body, "site", {
        "name": "obj_contact_left",
        "pos": f"{contact_local_left[0]} {contact_local_left[1]} {contact_local_left[2]}",
        "size": "0.01",
    })
    etree.SubElement(obj_body, "site", {
        "name": "obj_contact_right",
        "pos": f"{contact_local_right[0]} {contact_local_right[1]} {contact_local_right[2]}",
        "size": "0.01",
    })

    # 添加 equality constraints
    eq = root.find("equality")
    if eq is None:
        eq = etree.SubElement(root, "equality")

    for side in ["left", "right"]:
        etree.SubElement(eq, "connect", {
            "body1": f"{side}_wrist_yaw_link",
            "body2": "object",
            "anchor": "0.0 0.0 0.0",
            "solref": f"{solref[0]} {solref[1]}",
            "solimp": "0.95 0.99 0.001",
            "name": f"hand_obj_connect_{side}",
        })

    tree.write(output_xml, xml_declaration=True, encoding="utf-8")
```

**设计考量**:
- 接触点位置从参考数据反算（Step 1 中提取）
- `solref` 控制约束刚度，负值表示阻尼弹簧（可以有 violation）
- `solimp` 控制约束的"柔软度"——允许小量穿透避免过大关节力
- 保留 scene_forearm 的所有碰撞设置（3-box 手部等）

### 2. 提取接触点位置

**文件**: `workspace/core4d/scripts/compute_contact_local.py`（新增）

```python
"""
从 trajectory_kinematic.npz 提取 person1 手在物体局部坐标系中的平均接触位置。
用于确定 connect 约束的 anchor 点。
"""
import numpy as np

def compute_contact_local(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    qpos = data['qpos']           # (T, 43) — pelvis(7) + 29joints + obj(7)
    contact_pos = data['contact_pos']  # (T, 2, 3) — hand world positions

    # 提取物体位姿
    obj_pos = qpos[:, 36:39]       # (T, 3)
    obj_quat = qpos[:, 39:43]      # (T, 4) wxyz

    # 找接触帧（手距物体中心 < 0.5m 的帧）
    T = qpos.shape[0]
    contact_local = np.zeros((T, 2, 3))
    for t in range(T):
        for h in range(2):
            # 世界坐标转物体局部坐标
            delta = contact_pos[t, h] - obj_pos[t]
            # quat_inv_apply: 用物体四元数的逆旋转delta
            contact_local[t, h] = quat_inv_apply(obj_quat[t], delta)

    # 取距离最近的帧的中位数作为稳定估计
    dists = np.linalg.norm(contact_pos - obj_pos[:, None, :], axis=-1)  # (T, 2)
    close_mask = dists < 0.5  # 手距物体中心 < 0.5m

    left_contact = np.median(contact_local[close_mask[:, 0], 0], axis=0)
    right_contact = np.median(contact_local[close_mask[:, 1], 1], axis=0)

    return left_contact, right_contact


def quat_inv_apply(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply inverse quaternion rotation to vector."""
    from scipy.spatial.transform import Rotation as R
    # scipy uses xyzw, mujoco uses wxyz
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(quat_xyzw)
    return rot.inv().apply(vec)
```

### 3. 配置文件

**文件**: `examples/config/override/core4d_box025_weld.yaml`（新增）

```yaml
# @package _global_
dataset_name: core4d
task: box025_person1
data_id: 0
robot_type: unitree_g1
embodiment_type: humanoid_object
ref_dt: 0.0333333
trace_dt: 0.0333333

# 使用 connect 约束场景
scene_name: scene_connect

# 继承 E006 最优的 reward 配置
contact_rew_scale: 1.0
pos_rew_scale: 3.0
rot_rew_scale: 1.0
base_pos_rew_scale: 3.0
base_rot_rew_scale: 1.0

# CEM 参数
num_samples: 2048
terminate_resample: true
```

### 4. 运行脚本

**文件**: `workspace/core4d/scripts/retarget/retarget_core4d_weld.sh`（新增）

```bash
#!/bin/bash
TASK=${1:-box025_person1}
MODE=${2:-strong}  # strong | anneal | strong_contact

echo "=== E010: Weld(Connect) 约束重定向 ==="
echo "Task: $TASK, Mode: $MODE"

# Step 1: 计算接触点
python workspace/core4d/scripts/compute_contact_local.py \
    --npz example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_kinematic.npz

# Step 2: 生成 connect 场景 XML
python workspace/core4d/scripts/generate_scene_connect.py \
    --base example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/scene_forearm.xml \
    --output example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/scene_connect.xml \
    --mode $MODE

# Step 3: 重定向
uv run examples/run_mjwp.py +override=core4d_box025_weld \
    task=${TASK} data_id=0 viewer=none

# Step 4: 客观验证 (npz qpos)
python -c "
import numpy as np
d = np.load('example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_mjwp.npz')
q = d['qpos']
obj_z = q[:, 38]  # object z in qpos
ref = np.load('example_datasets/processed/core4d/unitree_g1/humanoid_object/${TASK}/0/trajectory_kinematic.npz')
ref_z = ref['qpos'][:, 38]
print(f'sim obj z: min={obj_z.min():.3f}, max={obj_z.max():.3f}, mean={obj_z.mean():.3f}')
print(f'ref obj z: min={ref_z.min():.3f}, max={ref_z.max():.3f}, mean={ref_z.mean():.3f}')
# pelvis
sim_pelvis = q[:, :3]
ref_pelvis = ref['qpos'][:, :3]
err = np.linalg.norm(sim_pelvis - ref_pelvis[:len(sim_pelvis)], axis=-1)
print(f'pelvis err: mean={err.mean():.3f}, max={err.max():.3f}')
"
```

## 需要修改/新增的文件

| # | 文件 | 改动 | 新增/修改 |
|---|------|------|----------|
| 1 | `workspace/core4d/scripts/compute_contact_local.py` | 提取手在物体局部坐标的接触点 | 新增 |
| 2 | `workspace/core4d/scripts/generate_scene_connect.py` | 生成 scene_connect.xml | 新增 |
| 3 | `examples/config/override/core4d_box025_weld.yaml` | E010 配置 | 新增 |
| 4 | `workspace/core4d/scripts/retarget/retarget_core4d_weld.sh` | E010 运行脚本 | 新增 |
| 5 | `spider/simulators/mjwp.py` | 可能需要: eq_active 运行时控制 | 条件性修改 |

## Reward 权重

| 类别 | E006f/E009 | **E010** | 备注 |
|------|-----------|---------|------|
| Body pos tracking | 3.0 | 3.0 | 不变 |
| Body rot tracking | 1.0 | 1.0 | 不变 |
| Object pos tracking | 3.0 | 3.0 | 不变 |
| Object rot tracking | 1.0 | 1.0 | 不变 |
| Contact reward | 1.0 | 1.0 | 不变 |
| Joint regularization | 0.003 | 0.003 | 不变 |

**E010 不改 reward**——唯一的变量是 connect 约束。隔离变量，纯诊断实验。

## 成功标准

| 指标 | E009最佳 (gravcomp) | **E010 目标** | Claim |
|------|-------------------|---------------|-------|
| obj z_max (qpos实测) | 0.381m (质心, 底面0.008m) | **≥ 0.40m** | C1 |
| obj 底面离地 | ≤ 0.008m | **≥ 0.05m** | C3 |
| pelvis_err | 0.115m (E009a) | **≤ 0.15m** | C2 |
| joint_err | 未记录 | **≤ 0.10rad** | C4 |
| 视频 | 箱子旋转但不离地 | **箱子离地+被搬运** | C3 |

## 特别需要注意的点

### 1. connect 约束的 anchor 位置很关键

如果 anchor 设在 wrist_yaw_link 的几何中心，而实际接触点在前臂内侧，约束会把手拉到不自然的位置。

**缓解**: 先用 `compute_contact_local.py` 看看参考中手的接触位置，然后在 wrist_yaw_link 的局部坐标系中找到最近点作为 body1 的 anchor。或者直接用 `(0, 0, 0)` 作为 anchor（link 原点），看效果。

### 2. 物体初始位置需要注意

E001-E009 的物体初始 z = 0.305m（地面高度 + 半高）。connect 约束会在第一帧就尝试把手拉到物体上。如果此时手距离物体很远，约束力可能导致不稳定。

**缓解**: 第一帧检查手-物体距离。如果 > 0.3m，考虑在前几步用弱约束（大 solimp width）。

### 3. 验证方法必须严格

绝对不看 reward 字段。只看：
1. `npz['qpos'][:, 36:39]` (物体世界坐标)
2. 视频帧分析（用 /video-frames）
3. box 底面 4 角 z（用 E009 的 eval_e009_lift.py 逻辑）

### 4. 不破坏已有实验

所有改动都是新增文件，不修改任何已有 XML/yaml/脚本。scene_connect.xml 是新文件。

## 失败兜底

如果 E010a (强约束) 物体仍不离地 (z_max < 0.35m)：

**诊断路径**:
1. 检查约束是否生效（打印 eq_active, efc_force）
2. 检查手是否到达物体位置（打印 hand_site_xpos vs obj_pos）
3. 如果手到不了 → G1 臂展不够 → **切换到 bucket005**
4. 如果手到了但物体不动 → 约束刚度不够 → 增大 solref

如果确认 G1 运动学不够 box025：
- 不继续在 box025 上浪费时间
- 转向 bucket005（直径 ~0.3m，G1 单人可环抱）
- 走 E006 风格的前臂接触路线
