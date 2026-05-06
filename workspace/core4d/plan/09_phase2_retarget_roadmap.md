# Phase 2 实验路线图：死磕重定向

## 1. Phase 1 复盘（E001-E009）

### 1.1 实验时间线与核心结论

| Run | 做了什么 | 学到了什么 |
|-----|---------|-----------|
| E001 | 数据管线搭建 | CORE4D → SPIDER 格式转换正确, 124帧 30fps |
| E002-E003 | 基线重定向（无/有引导） | 机器人身体跟踪优秀(pelvis<0.1m), 但物体从未离地 |
| E004 | 强增益 kp=100/1000 | 确认根因: CEM 最终迭代强制归零设计 → 物体自由落体 |
| E005 | 混合轨迹导出 | SPIDER→Holosoma 格式转换 OK, 但绕过了物理搬运问题 |
| E006 | 3-box前臂+contact_rew | **虚假突破**: reward字段看似改善61%, 视频证实箱子被推开而非搬起 |
| E007 | 路径Y物理参数对齐 | physics_dt/PD对齐破坏重定向能力 — 重定向和RL的PD天然冲突 |
| E008 | 视频驱动诊断 | E006 obj z实测 max=0.307m(从未离地), 衰减PD引导期OK撤除即落 |
| E009 | Person2力支撑(5方案) | **几何错位**: CORE4D要双人±x端对夹, G1从-y/顶面接触 → 即使gravcomp也只离地8mm |

### 1.2 三层根因（从表层到本质）

```
表层: "物体抬不起来"
  ↓
中层: G1 球形手/前臂几何只能产生推力, 无法形成力闭合夹持
  ↓
深层: CORE4D box025 是协作任务 — 双人从±x端对夹
      G1 单人从-y侧接近, 接触方向与参考完全错位
      臂展0.5m < box长0.61m, 单人对夹不可解
```

### 1.3 已验证的死路（不再重复）

| 方向 | 为什么不行 | 实验依据 |
|------|-----------|---------|
| 调 reward 权重 | 接触方向错, 再大权重也是推不是夹 | E006a-f |
| 调 PD 增益 | 高kp振荡, 低kp力不足 | E006c, E007b |
| 改 physics_dt | 破坏接触时机, 控制频率不匹配 | E007a |
| 衰减PD引导 | 引导期能拽住, 撤除即落 | E008a |
| 减质量/gravcomp | 问题不是力, 是接触几何 | E009b/c |
| 残余 actuator | 残余力=虚拟弹簧, 不解决几何 | E009a/a2/a3 |

### 1.4 仍然有效的资产

| 资产 | 状态 | 可复用 |
|------|------|--------|
| 数据管线 (core4d.py) | 稳定 | person1/person2 转换 |
| scene_forearm.xml (3-box碰撞) | 稳定 | 所有后续实验 |
| contact_rew 奖励路径 | 稳定 | 后续实验基础 |
| Holosoma 导出脚本 | 稳定 | Phase 2 导出 |
| mjwp_eq.py 等式约束框架 | 已有但未用于此任务 | E010 核心 |
| partner_hand 数据 (_w_partner.npz) | 已有 | E011 核心 |
| 机器人身体跟踪能力 | pelvis<0.1m, joint<0.08rad | 所有实验基线 |

### 1.5 关键教训（写入规则）

1. **必须用 qpos 实测 + 视频验证** — reward字段不可信（E006/E008教训）
2. **几何分析先于参数调优** — 接触方向比力大小重要（E009教训）
3. **理解数据集语义** — 协作数据需要协作建模（E009教训）
4. **重定向≠RL, PD不需对齐** — 两阶段天然分离（E007教训）

---

## 2. Phase 2 核心策略

### 2.1 问题重新定义

Phase 1 的问题定义是："让 G1 单人搬起 box025"
Phase 2 的问题定义是："让重定向产生**物体真实离地**的物理合规轨迹"

关键区别：Phase 2 **接受协作建模**——不再假设单人能完成协作任务。

### 2.2 实验路线图

```
E010: Weld约束重定向
  "假设抓住了, G1的臂/身运动能否完成搬运轨迹?"
  → 回答运动学可行性问题
  │
  ├─ 成功 → E011: Mocap Partner 协作
  │         "有真实伙伴手的物理力, 能否不需要weld?"
  │         → 回答物理协作可行性
  │         │
  │         ├─ 成功 → E012: 导出+泛化
  │         │         导出到Holosoma, 尝试bucket
  │         │
  │         └─ 失败 → E012-alt: Weld+Mocap组合
  │                   weld保证抓握 + mocap提供伙伴力
  │
  └─ 失败 → 换物体(bucket) or 换机器人
            G1运动学不够 → 根本性方向调整
```

### 2.3 每个实验的核心假设

| 实验 | 假设 | 如果假设错了 |
|------|------|-------------|
| E010 | G1 运动学足以完成搬运（如果抓握不是问题） | 换更小物体或换机器人 |
| E011 | Person2 的手提供物理推力后, CEM能找到正确的协作姿态 | Weld+Mocap组合 |
| E012 | 成功轨迹可导出到Holosoma且对其他物体泛化 | 需要物体特定调优 |

---

## 3. E010 详细计划: Weld 约束重定向

### 3.1 Context

E009 证明 G1 单人无法物理抬起 box025（即使消除重力）。但 E009 没有回答一个更基本的问题：

> **G1 的关节范围和臂展, 是否在运动学上足以完成 box025 的搬运动作？**

如果运动学都不行（手够不到参考要求的位置），那后续所有方案都会失败。Weld 约束实验就是要消除"抓握"这个变量，单独验证运动学可行性。

### 3.2 方法

在手(wrist_yaw_link)-物体之间添加 MuJoCo `connect` 等式约束（不是 `weld`，connect 只约束位置不约束姿态，给手更多自由度）。

**为什么用 connect 而不是 weld**：
- `weld` = 位置+姿态完全锁定 → 手被焊死在物体上，关节可能过应力
- `connect` = 只约束位置（点重合）→ 手的朝向可以自由调整，更接近真实抓握

**约束配置**：
- 左手 wrist_yaw_link site → 物体表面 -x 端 site（person1 接触面）
- 右手 wrist_yaw_link site → 物体表面 -x 端 site
- 约束刚度：从弱到强退火（先让CEM找到好的身体姿态，再逐步锁定）

### 3.3 技术路径

**Step 1: 确定接触点位置**

从参考数据提取 person1 手在物体局部坐标系中的位置：
```python
# 从 trajectory_kinematic.npz 的 contact_pos (124,2,3) 提取
# 转换到物体局部坐标系
contact_local = quat_inv_apply(obj_quat, hand_world_pos - obj_pos)
# 取接触帧的平均值作为 connect site 位置
```

**Step 2: 生成 scene_eq_connect.xml**

在 scene_forearm.xml 基础上添加：
```xml
<equality>
    <!-- 左手-物体 connect (只约束位置) -->
    <connect body1="left_wrist_yaw_link" body2="object"
             anchor="0.0 0.0 0.0"  <!-- 手 link 局部坐标 -->
             solref="-1000 -100"    <!-- 可退火 -->
             solimp="0.95 0.99 0.001"/>
    <!-- 右手-物体 connect -->
    <connect body1="right_wrist_yaw_link" body2="object"
             anchor="0.0 0.0 0.0"
             solref="-1000 -100"
             solimp="0.95 0.99 0.001"/>
</equality>
```

**Step 3: 运行 mjwp（不用 mjwp_eq）**

关键决策：**用 mjwp.py 而不是 mjwp_eq.py**。

原因：
- mjwp_eq.py 的退火框架是为 IK 场景设计的（track↔ref site 对），修改成本高
- mjwp.py + 在 XML 里静态写死 equality constraint 更简单
- MuJoCo 的 equality constraint 在 mjwarp 里自动生效（不需要额外代码）
- 如果需要退火，可以在 `setup_env` 中调整 `eq_active` 数组

**Step 4: 分阶段验证**

| Sub-run | 约束 | 目的 |
|---------|------|------|
| E010a | connect 始终开启, 强刚度 | 验证"完美抓握"下G1能否搬 |
| E010b | connect 强→弱退火 | 验证不靠约束G1能否自己维持 |
| E010c | E010a + contact_rew | 在约束辅助下优化接触质量 |

### 3.4 Claims

| Claim | 最低证据 |
|-------|---------|
| C1: 强connect约束下, obj z_max ≥ 0.40m (qpos实测) | trajectory npz 中 obj qpos z |
| C2: pelvis_err ≤ 0.15m (约束不拉垮身体) | qpos pelvis vs ref |
| C3: 视频确认物体被搬起(不是翻滚/旋转) | 视频帧分析 |
| C4: 约束退火后, 物体是否能短暂维持 | E010b obj z 在退火后的帧 |

### 3.5 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/generate_scene_eq_connect.py` | **新增**: 从NPZ提取接触点, 生成带connect约束的scene XML |
| 2 | `examples/config/override/core4d_box025_weld.yaml` | **新增**: E010配置 |
| 3 | `workspace/core4d/scripts/retarget/retarget_core4d_weld.sh` | **新增**: E010运行脚本 |
| 4 | `spider/simulators/mjwp.py` (可能) | 如需运行时控制eq_active |

### 3.6 成功标准

| 指标 | E008 (最后实测) | **E010 目标** | 验证方法 |
|------|----------------|---------------|---------|
| obj z_max (qpos实测) | 0.307m (从未离地) | **≥ 0.40m** | `npz['qpos'][:, obj_z_idx]` |
| obj 底面离地高度 | ≤ 0.008m | **≥ 0.05m** | 4角最低点 z |
| pelvis_err | 0.06m (E006f) | **≤ 0.15m** | qpos pelvis vs ref |
| 视频确认 | 箱子被推开 | **箱子离地+被搬运** | 关键帧分析 |

### 3.7 风险与缓解

| 风险 | 可能性 | 缓解 |
|------|--------|------|
| connect约束导致关节过应力(手被拉到不可达位置) | 中 | 用弱solimp, 允许约束violation |
| G1臂展不够→手到不了box -x端 | 高 | 如果失败, 改为只约束一只手, 或改到box -y端(近侧) |
| mjwarp不支持eq_active运行时控制 | 低 | 检查mjwarp API, 备选方案: 多个XML切换 |
| CEM优化变慢(约束增加计算) | 低 | 可接受, E006已~50s |

### 3.8 训练命令

```bash
# E010a: 强connect约束
bash workspace/core4d/scripts/retarget/retarget_core4d_weld.sh box025_person1 strong

# E010b: 退火connect
bash workspace/core4d/scripts/retarget/retarget_core4d_weld.sh box025_person1 anneal

# E010c: 强connect + contact_rew
bash workspace/core4d/scripts/retarget/retarget_core4d_weld.sh box025_person1 strong_contact
```

---

## 4. E011 详细计划: Mocap Partner 协作重定向

### 4.1 Context

如果 E010 验证了 G1 运动学可行（有约束辅助能搬），E011 的问题是：

> **用 person2 的真实手轨迹作为 mocap body 提供物理推力, G1 (person1) 能否在没有 weld 约束的情况下协作搬运？**

### 4.2 方法

在场景中添加 2 个 mocap body（胶囊体或球体），代表 person2 的左右手：
- 轨迹来源：`_w_partner.npz` 的 `partner_hand_pos_w` (205,2,3) 和 `partner_hand_quat_w` (205,2,4)
- 重采样到 SPIDER 的帧率（30fps）
- mocap body 有碰撞几何，能与箱子产生物理接触力
- G1 (person1) 用 CEM 优化，目标是配合 mocap partner 搬运

### 4.3 技术路径

**Step 1: 提取并转换 partner 数据**

```python
# 从 Holosoma w_partner.npz
partner_pos = npz['partner_hand_pos_w']  # (205, 2, 3) @ 50fps
partner_quat = npz['partner_hand_quat_w']  # (205, 2, 4) @ 50fps
# 重采样到 30fps (SPIDER ref_dt=0.0333)
# 存入 SPIDER 格式的 NPZ 作为额外字段
```

**Step 2: 生成 scene_mocap_partner.xml**

```xml
<!-- Person2 左手 mocap body -->
<body name="p2_left_hand" mocap="true" pos="0 0 0">
    <geom type="capsule" size="0.04 0.08"
           contype="1" conaffinity="1" friction="1.5 0.005 0.001"/>
</body>
<!-- Person2 右手 mocap body -->
<body name="p2_right_hand" mocap="true" pos="0 0 0">
    <geom type="capsule" size="0.04 0.08"
           contype="1" conaffinity="1" friction="1.5 0.005 0.001"/>
</body>
```

**Step 3: 在 mjwp step 中更新 mocap body 位置**

```python
# 在 step_env 中, 每步设置 mocap body 位置
data.mocap_pos[left_mocap_id] = partner_pos[t, 0]
data.mocap_quat[left_mocap_id] = partner_quat[t, 0]
data.mocap_pos[right_mocap_id] = partner_pos[t, 1]
data.mocap_quat[right_mocap_id] = partner_quat[t, 1]
```

**Step 4: 运行 CEM 优化 person1**

CEM 只优化 G1 (person1) 的关节，mocap partner 按参考轨迹固定运动。

### 4.4 Claims

| Claim | 最低证据 |
|-------|---------|
| C1: mocap partner + G1, obj z_max ≥ 0.35m (qpos实测) | trajectory npz |
| C2: G1 pelvis_err ≤ 0.15m | qpos vs ref |
| C3: 视频确认双方从±x端协作夹箱 | 关键帧 |
| C4: 物体在3帧以上保持 z>0.35m | 帧计数 |

### 4.5 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/process_datasets/core4d.py` | 提取partner_hand数据到NPZ |
| 2 | `workspace/core4d/scripts/generate_scene_mocap_partner.py` | **新增**: 生成带mocap body的scene |
| 3 | `spider/simulators/mjwp.py` | 在step_env中更新mocap body位置(条件性) |
| 4 | `spider/config.py` | +`use_mocap_partner`, +`mocap_partner_data_key` |
| 5 | `examples/config/override/core4d_box025_mocap_partner.yaml` | **新增** |
| 6 | `workspace/core4d/scripts/retarget/retarget_core4d_mocap_partner.sh` | **新增** |

### 4.6 风险

| 风险 | 缓解 |
|------|------|
| partner数据帧率不匹配 | 线性插值重采样 |
| mocap body穿过箱子 | 调碰撞参数, 或用大几何体 |
| CEM搜索空间不变但物理更复杂 | 可能需要增加num_samples |
| partner数据是50fps, ref是30fps | 在core4d.py中处理对齐 |

---

## 5. E012 计划（依赖 E010/E011 结果）

### 如果 E010+E011 成功：
- **E012a**: 最优方案导出到 Holosoma 格式（更新 E005 导出脚本）
- **E012b**: 在 bucket005 上验证泛化性（更小物体，可能单人可行）
- **E012c**: 在 chair022 上验证泛化性（不同形状）

### 如果 E010 失败（G1 运动学不够）：
- **E012-alt**: 切换到 bucket005（直径~0.3m, G1单人可抱）
- 需要新的数据转换（bucket CORE4D 数据）
- 验证 E006 风格的前臂接触是否对小物体有效

### 如果 E011 失败（mocap partner 物理力不够）：
- **E012-alt2**: Weld + Mocap Partner 组合
  - Weld 保证 G1 侧抓握
  - Mocap partner 提供另一侧物理支撑
  - 两个机制互补

---

## 6. 时间预算

| 实验 | 预估 | 依赖 |
|------|------|------|
| E010 (Weld约束) | 0.5天代码 + 0.5天跑实验 | 无 |
| E011 (Mocap Partner) | 1天代码 + 0.5天跑实验 | E010结果(不阻塞, 可并行) |
| E012 (导出/泛化) | 0.5天 | E010 or E011成功 |
| **Total** | **~3天** | |

## 7. 优先级排序

**立即执行**: E010（回答最基本的问题：G1 运动学是否可行）

**原因**：
1. 工程量最小（只需要改 scene XML + 一个配置）
2. 信息价值最大（如果运动学都不行，后续所有方案都白费）
3. 不需要额外数据处理（用现有 person1 数据）
4. 代码改动局部可控（只加 XML equality 约束）

如果 E010 成功 → E011 → E012
如果 E010 失败 → 立即切换到 bucket（更小物体）
