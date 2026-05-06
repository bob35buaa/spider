# E006: 前臂接触重定向 — 实验计划

## Context

E002-E004 证明 G1 球形手 (r=5cm, 点接触) 无法物理搬起 61×89cm/5kg 的箱子。但 HDMI 工作流中机器人成功搬运物体——**关键差异是碰撞几何和接触策略**：

| 维度 | HDMI (成功) | CORE4D (失败) |
|------|------------|--------------|
| 手部碰撞 | 3个box geom (17cm长平面) | 1个sphere (r=5cm点接触) |
| 前臂接触 | elbow_yaw_collision 可接触物体 | 仅 lh/rh 允许接触 |
| 碰撞系统 | broadphase (contype=1/conaffinity=1 default) | explicit pair only (contype=0/conaffinity=0 default) |
| 物体尺寸 | suitcase 20×30×40cm/2kg, largebox 36cm/0.1kg | box 61×61×89cm/5kg |
| condim | 1 (法向力) → 面支撑 | 4 (全摩擦) → 但只有点接触无用 |

**核心洞察**: 人类用手掌接触箱子侧面，但 G1 没有灵巧手自由度。实际上机器人应该用**前臂内侧**托住大型箱子——这是一种"接触重映射"策略：
- 人类: 手掌 → 箱子侧面
- 机器人: 前臂内侧 → 箱子底部/侧面

## Claims (可验证声明)

1. **C1**: 将手部碰撞从球体改为 HDMI 风格的 3-box 前臂/手掌面，物体搬起概率 > 0%（vs 当前 0%）
2. **C2**: 添加前臂-物体接触对（elbow collision + wrist collision），配合非归零增益策略，物体 z_max > 0.35m
3. **C3**: 在保持机器人身体跟踪质量 (pelvis_err < 0.15m) 的前提下实现物体搬运

## 改动方案

### 方案 A: 碰撞几何升级 + 接触扩展 (最小侵入)

**不修改 SPIDER 代码**，只修改场景 XML：

1. **替换手部碰撞几何**: `hand_collision` 从球体改为 HDMI 风格 3-box
   ```xml
   <!-- 替换 <geom name="lh" class="hand_collision" /> 为: -->
   <geom name="lh_1" type="box" size="0.05 0.025 0.025" pos="0.02 0 0" group="3" />
   <geom name="lh_2" type="box" size="0.05 0.01 0.05" pos="0.09 0 0" group="3" />
   <geom name="lh_3" type="box" size="0.025 0.01 0.05" pos="0.15 -0.01 0" 
         quat="0.980067 0 0 -0.198669" group="3" />
   ```

2. **添加前臂-物体接触对**: 在 `<contact>` 中添加 elbow collision 与 object 的 pair
   ```xml
   <pair name="left_elbow_object" geom1="left_elbow_yaw_collision" geom2="object_collision" 
         friction="2 1" condim="4" />
   <pair name="right_elbow_object" geom1="right_elbow_yaw_collision" geom2="object_collision" 
         friction="2 1" condim="4" />
   <pair name="left_wrist_1_object" geom1="lh_1" geom2="object_collision" 
         friction="2 1" condim="4" />
   <!-- ... 所有 wrist geom 同理 -->
   ```

3. **减轻物体质量**: 5kg → 2kg (更接近 HDMI suitcase)

### 方案 B: 非归零增益 (保留执行时刻的残余引导力)

**修改 `run_mjwp.py`**: 最后一个 CEM 迭代不归零，保留衰减后的残余增益：

```python
# 原: if i == config.max_num_iterations - 1: kp_i = zeros
# 改: 不做特殊处理，让自然衰减给出极小但非零的增益
# 或: kp_i = base_kp * final_residual_ratio (e.g. 0.05)
```

这使得执行阶段物体仍有微小引导力——不完全物理，但比完全归零更现实（类似另一个人在协助）。

**关键**: 对于人机协作场景，这有物理意义——另一个人（人类伙伴）在帮忙托着物体！残余增益可以模拟人类伙伴的支撑力。

### 方案 C: 协作伙伴建模 (研究方向)

在仿真中显式加入协作伙伴的贡献：
- 选项1: 在物体上施加等效外力 (`xfrc_applied`)，大小和方向从 CORE4D 第二人的动作推算
- 选项2: 在仿真中加入 SMPL-X mocap body 作为运动学约束
- 选项3: 同时重定向两个人为两个 G1

## 实验设计

### E006a: 碰撞几何升级 (方案 A)
- 生成新 scene XML: `scene_forearm.xml`
- 手部改为 3-box, 前臂加接触对
- 物体质量 2kg (HDMI comparable)
- 运行 SPIDER MJWP 无引导 (baseline)

### E006b: 碰撞 + 非归零增益 (方案 A+B)
- 使用 `scene_forearm_act.xml`
- `guidance_decay_ratio=0.9`, 最后迭代保留 `residual_gain_ratio=0.05`
- 运行 SPIDER MJWP with contact guidance

### E006c: 碰撞 + 协作伙伴外力 (方案 A+C)
- 计算 person2 在物体上的等效支撑力
- 通过 `xfrc_applied` 施加到物体 body
- 运行 SPIDER MJWP

## 成功标准

| 指标 | E006a目标 | E006b目标 | E006c目标 |
|------|----------|----------|----------|
| obj z_max | > 0.35m | > 0.40m | > 0.45m |
| obj_pos_err | < 0.5m | < 0.3m | < 0.2m |
| pelvis_err | < 0.15m | < 0.15m | < 0.15m |
| 物体搬起? | 部分 | 是 | 是 |

## 训练/重定向命令

```bash
# E006a: 前臂接触基线
uv run examples/run_mjwp.py +override=core4d_box025 \
    task=box025_person1 data_id=0 viewer=rerun \
    scene_name=scene_forearm

# E006b: 前臂 + 非归零引导
uv run examples/run_mjwp.py +override=core4d_box025_act \
    task=box025_person1 data_id=0 viewer=rerun \
    scene_name=scene_forearm_act

# E006c: 前臂 + 外力
uv run examples/run_mjwp.py +override=core4d_box025 \
    task=box025_person1 data_id=0 viewer=rerun \
    scene_name=scene_forearm \
    perturb_body=object perturb_force="0 0 24.5"  # 抵消50%重力
```

## 实施步骤

1. 生成 `scene_forearm.xml` — 3-box 手部 + 前臂接触对 + 2kg 物体
2. 生成 `scene_forearm_act.xml` — 同上 + 6DOF 物体执行器
3. 修改 `run_mjwp.py` — 添加 `residual_gain_ratio` 配置参数
4. 运行 E006a, 观察前臂是否接触物体
5. 运行 E006b, 观察物体是否被抬起
6. 如果 E006a/b 不足, 实施 E006c (外力方案)
7. 分析: 哪种组合能产生足够的物体搬运质量用于 RL 训练

## 风险

| 风险 | 缓解 |
|------|------|
| 3-box 手部改变了 SPIDER 跟踪的 site 位置 | 保持 track_hand_* site 不变 |
| 前臂碰撞导致 CEM 优化变慢/不收敛 | 增加 max_geom_pairs, 减少 num_samples if needed |
| 非归零增益使结果不可迁移到真实机器人 | 对于协作场景，解释为"人类伙伴力" |
| 物体减轻后场景不真实 | 后续 E007 恢复 5kg 测试 |
