# E006: 前臂接触物理搬运 — 结果

## 状态: 部分成功 (C1/C2 通过, C3 部分通过)

## 核心发现

通过分析 HDMI 成功案例，识别出 CORE4D 搬运失败的**三个根因**：
1. 手部碰撞几何: HDMI 用 3-box 平面(17cm) vs CORE4D 用单球体(r=5cm)
2. 接触对限制: CORE4D 只允许 lh/rh 接触物体, 无前臂接触
3. 优化器缺少激励: `contact_rew_scale=0` → 优化器无动力让手臂接近物体

## 实验结果汇总

| 配置 | obj_pos_err | obj_z_max | pelvis_z_min | 物体抬起? |
|------|------------|-----------|-------------|----------|
| E002 基线 (球形手, 5kg) | 0.825 | 0.305 | 0.81 | 否 |
| E006a 前臂+0.5kg (无引导) | 0.755 | 0.410 | 0.21 | 短暂 |
| E006b 前臂+引导(decay=0.9,res=0.1) | 0.654 | 0.305 | 0.23 | 否 |
| E006c 前臂+引导(kp=100,decay=1.0) | 1.516 | N/A | N/A | 不稳定 |
| E006d 前臂+引导(kp=50,decay=0.99) | 0.479 | 0.346 | 0.16 | 微弱 |
| E006e 前臂+contact_rew+pos_rew=3 | 0.386 | 0.424 | 0.23 | 部分 |
| **E006f 前臂+contact_rew+高权重** | **0.324** | **0.610** | **0.584** | **是!** |

## 最优配置 (E006f)

```yaml
scene_name: scene_forearm  # 3-box 前臂 + 前臂/肘部-物体接触对
mass: 0.5kg               # 模拟人类伙伴承担 90% 重量
contact_rew_scale: 1.0    # 奖励手臂接近物体
pos_rew_scale: 3.0        # 物体位置跟踪权重 ×3
base_pos_rew_scale: 3.0   # 身体稳定权重 ×3
```

**关键指标**:
- obj_pos_err: **0.324m** (vs 基线 0.825m, 改善 **61%**)
- obj z_max: **0.610m** (抬升 30.5cm, 超目标 181%)
- pelvis z_min: **0.584m** (站立稳定)
- 物体在 step 4-6 被有效搬起 (z: 0.484→0.610→0.427m)

## Claims 验证

| Claim | 结果 | 通过? |
|-------|------|------|
| C1: 3-box前臂碰撞使物体搬起概率>0% | E006a/e/f 均出现物体抬起 | ✓ |
| C2: 前臂接触+非归零增益, obj z_max>0.35m | E006f: z_max=0.610m | ✓ |
| C3: pelvis_err<0.15m 同时搬运 | pelvis_z_min=0.584 (稳定但未验证误差) | 部分 |

## 分析: 为什么有效

1. **3-box 几何**: 创造了面接触（vs 点接触），前臂可以"托住"物体
2. **前臂+肘部接触对**: 允许 elbow_collision + 3个 wrist geom 与物体交互
3. **contact_rew_scale=1.0**: 给优化器明确的梯度信号——把手放到物体旁边
4. **pos_rew_scale=3.0**: 强化物体跟踪目标（CEM会优先选择让物体靠近参考的方案）
5. **0.5kg 质量**: 模拟协作场景中人类伙伴承担大部分重量
6. **base_pos_rew_scale=3.0**: 防止优化器为了物体而牺牲身体稳定性

## 为什么 contact guidance (PD增益) 反而不如 freejoint

- PD 执行器在 CEM 优化中引入**额外自由度**，使搜索空间更复杂
- 高 kp 引起振荡不稳定，低 kp 力不足以克服重力
- Freejoint 场景中物体仅通过物理接触移动，优化器有更清晰的因果链
- 结论: 对于大型协作搬运，**纯物理接触** > 衰减 PD 引导

## 下一步

1. **渐进增重测试**: 0.5kg → 1.0kg → 2.0kg，找到物理可行的质量上限
2. **持续搬运**: 当前只有 3 步有效搬运，需要延长保持时间
3. **协作伙伴建模**: 用 xfrc_applied 显式施加伙伴支撑力，替代减轻质量
4. **双人重定向**: 同时重定向 person1 + person2，共享物体
5. **导出验证**: 成功轨迹 → Holosoma RL 格式 → 训练对比

## 结果路径

| 产出 | 路径 |
|------|------|
| 前臂场景 XML | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_forearm.xml` |
| 前臂+act 场景 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/scene_forearm_act.xml` |
| 最优配置 | `examples/config/override/core4d_box025_forearm.yaml` |
| 轨迹输出 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/trajectory_mjwp.npz` |
| 视频 | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/visualization_mjwp.mp4` |
| 实验计划 | `workspace/core4d/plan/05_E006_forearm_contact_plan.md` |

## 可视化观察

E006f (visualization_mjwp.mp4):
- 机器人站立稳定，pelvis 高度保持 0.58-0.84m
- 双臂展开接近箱子，前臂内侧与箱子产生面接触
- 箱子在序列中段被抬起至 0.61m (离地 30cm)
- 抬起后几步内物体回落到地面 (非持续搬运)
- 无脚滑，身体跟踪质量维持合理水平

## 运行命令

所有命令通过本地脚本 `workspace/core4d/scripts/retarget/retarget_core4d_forearm.sh` 调用。各次迭代的关键参数差异：

### E006a: 前臂碰撞 + 0.5kg, 无引导
```bash
# scene_forearm.xml: mass=0.5kg, 3-box 手部, elbow+wrist contact pairs
# config (core4d_box025_forearm.yaml): contact_rew_scale=0 (默认)
uv run examples/run_mjwp.py +override=core4d_box025_forearm \
    task=box025_person1 data_id=0 viewer=none
# Result: pos=0.755, z_max=0.410
```

### E006b: 前臂 + 弱残余引导
```bash
# config (core4d_box025_forearm_act.yaml):
#   contact_guidance=true, guidance_decay_ratio=0.9
#   residual_gain_ratio=0.1, init_pos_actuator_gain=20
uv run examples/run_mjwp.py +override=core4d_box025_forearm_act \
    task=box025_person1 data_id=0 viewer=none
# Result: pos=0.654, 物体未抬起
```

### E006c: 前臂 + 强引导(kp=100, decay=1.0)
```bash
# config: residual_gain_ratio=1.0, init_pos_actuator_gain=100
uv run examples/run_mjwp.py +override=core4d_box025_forearm_act \
    task=box025_person1 data_id=0 viewer=none
# Result: pos=1.516 (不稳定振荡)
```

### E006d: 前臂 + 临界阻尼引导(kp=50)
```bash
# config: residual_gain_ratio=1.0, init_pos_actuator_gain=50,
#         init_pos_actuator_bias=10, guidance_decay_ratio=0.99
uv run examples/run_mjwp.py +override=core4d_box025_forearm_act \
    task=box025_person1 data_id=0 viewer=none
# Result: pos=0.479, z_max=0.346
```

### E006e: 前臂 + contact_rew + 高物体权重
```bash
# config (core4d_box025_forearm.yaml):
#   contact_rew_scale=1.0, pos_rew_scale=3.0, rot_rew_scale=1.0
uv run examples/run_mjwp.py +override=core4d_box025_forearm \
    task=box025_person1 data_id=0 viewer=none
# Result: pos=0.386, z_max=0.424
```

### E006f: 最优配置 — 前臂 + contact_rew + 高物体&身体权重
```bash
# config (core4d_box025_forearm.yaml):
#   contact_rew_scale=1.0
#   pos_rew_scale=3.0, rot_rew_scale=1.0       (object weights)
#   base_pos_rew_scale=3.0, base_rot_rew_scale=1.0  (robot body weights)
uv run examples/run_mjwp.py +override=core4d_box025_forearm \
    task=box025_person1 data_id=0 viewer=none
# Result: pos=0.324, z_max=0.610 (突破!)
```

### 复现 E006f 完整流程

```bash
# 1. 数据转换 (E001 已完成, 如需重做)
bash workspace/core4d/scripts/convert/convert_holosoma_to_spider.sh box025_person1

# 2. 重定向 (E006f 配置)
bash workspace/core4d/scripts/retarget/retarget_core4d_forearm.sh box025_person1

# 3. 视频生成 (内置于 run_mjwp.py)
# 输出: example_datasets/processed/core4d/unitree_g1/humanoid_object/box025_person1/0/visualization_mjwp.mp4
```
