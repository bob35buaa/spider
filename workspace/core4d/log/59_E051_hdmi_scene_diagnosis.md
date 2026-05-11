# E051: HDMI Scene 物理配置诊断 + Euler Convention 修复尝试

## 状态: ✅ 诊断完成, 修复失败 → 方向明确 (2026-05-11)

## 背景

E048-E050 确认 HDMI 在 CORE4D 上 body tracking 好 (7.3°) 但 object tracking 差 (24-91cm).
本实验全面诊断原因并尝试修复.

---

## Part 1: 数据验证 (E051a+b)

### 发现 1: box023 Euler Convention Mismatch = 180° (非 < 1°!)

E050 报告 "box023 euler 差异 < 1°" 是**错误的** — 那是 middle angle (gimbal lock 风险指标).

实际情况:
- box023 物体 quaternion = [-0.007, -0.001, 0.701, 0.713] = **179.3° from identity** (≈90° Z旋转)
- Extrinsic xyz 给 intrinsic XYZ joints 的 FK 误差 = **178°**
- **Euler convention bug 影响所有 CORE4D cases**, 不仅仅是 box025

| Case | 物体旋转(离identity) | Convention mismatch | 最佳 convention |
|------|---------------------|--------------------| --------------|
| box023 | 179.3° | 178° | XZY (middle=8.6°) |
| box025 | 119.4° | 120-140° | YXZ (middle=6.3°) |

### 发现 2: box023 Scene XML body_pos 偏移 145cm

| | HDMI scene XML body_pos | Motion data frame 0 pos | 偏移 |
|---|---|---|---|
| box023 | [0.155, -0.124, 0.310] | [-0.826, -1.184, 0.140] | **145 cm** |
| box025 | [0.155, -0.124, 0.310] | [0.155, -0.124, 0.310] | 0.04 cm |

box023 scene 的 body_pos 是从模板复制的 (box025 的值), 未按实际数据更新.

### 发现 3: Motion.npz FK 完全正确

- body_pos_w 与 scene.xml + qpos FK 匹配: position 0.000cm, rotation 0.000° (frame 0)
- 帧数正确: 136@30fps → 226@50fps
- Joint positions 匹配: max_diff = 0.000°
- **convert_core4d_to_hdmi.py 数据转换无 bug**

---

## Part 2: 修复尝试 (全部失败)

### E051a: body_pos 修正 + euler fix (XZY) + mass=2.0

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E051/E051_box023_euler_fix \
    use_torch_compile=false euler_convention=XZY
# body_pos 改为 [-0.826, -1.183, 0.140]
```

| 指标 | E051a | E048a baseline | 变化 |
|------|-------|---------------|------|
| ObjPos | **131.8cm** | 24.4cm | ❌ 5.4× 恶化 |
| ObjRot | 92.1° | 66.4° | ❌ 恶化 |
| Joint | 14.8° | 7.3° | ❌ 恶化 |
| Stability | 98.8% | 100% | ≈ |

**失败原因**: body_pos 偏移 (145cm) 不是 bug, 是 HDMI PD 追踪的**隐式恢复力来源**:
- kp=20 × 1.5m_offset = 30N 持续拉力 → 帮助物体跟随参考轨迹
- 修正 body_pos 后 offset≈0 → PD 无恢复力 → 物体在 guidance_decay 后漂移

### E051b: 仅 euler fix (XZY) + mass=2.0 (保持原 body_pos)

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E051/E051b_box023_euler_only \
    use_torch_compile=false euler_convention=XZY
```

| 指标 | E051b | E048a baseline | 变化 |
|------|-------|---------------|------|
| ObjPos | **142.5cm** | 24.4cm | ❌ 5.8× 恶化 |
| ObjRot | **113.0°** | 66.4° | ❌ 恶化 |
| Joint | 14.8° | 7.3° | ❌ 恶化 |
| Stability | 98.8% | 100% | ≈ |

**失败原因**: 旧系统虽然 euler 错了 178°, 但**内部自洽**:
- PD 初始化和目标都用同一个"错误"约定 → 物体在"错误"方向上被稳定
- CEM 已适应这个方向上的碰撞动力学
- 修正 euler 后物体方向翻转 178° → 碰撞关系完全改变 → CEM 无法适应

验证: MuJoCo FK round-trip 确认:
- OLD (extrinsic xyz on XYZ joints): FK error = **178.03°** (错但自洽)
- NEW (intrinsic XZY on XZY joints): FK error = **0.00°** (正确但 CEM 不适应)

### E051d: 直接用 scene_act.xml (方案B)

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E051/E051d_box023_scene_act \
    use_torch_compile=false \
    +use_scene_act=example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person1/scene_act.xml
```

| 指标 | E051d | E048a baseline | 变化 |
|------|-------|---------------|------|
| ObjPos | 73.0cm | 24.4cm | ❌ 3× 恶化 |
| ObjRot | 65.2° | 66.4° | ≈ |
| Joint | **19.4°** | 7.3° | ❌ 2.7× 恶化 |
| Stability | **57.2%** | 100% | ❌❌ 崩溃 |

**失败原因**: scene_act.xml 是为 MJWP (E041c) 设计的:
- Object joints 有高 damping (slides=100, hinges=20) → PD kp=20 推不动
- Robot 物理参数也不同 → Stability 崩溃

---

## Part 3: 根因分析 — 为什么 HDMI suitcase 能 work 而 CORE4D 不行

### 核心发现: Scene XML 物理配置差异

`hdmi.py` setup_env 已覆盖: timestep(→0.002), integrator(→implicitfast), actuator gains(→Isaac PD)

**但以下从 scene XML 读取, 未被覆盖**:

| 部件 | 原始 HDMI suitcase | CORE4D HDMI scene | 影响 |
|------|-------------------|-------------------|------|
| **Hand collision** | **3 boxes/wrist** (模拟手掌包裹) | 1 sphere r=0.05 (点接触) | ★★★ 无法抓握 |
| **Foot collision** | **7 capsules/foot** (全足底覆盖) | 4 spheres r=0.005 (点接触) | ★★ 站立不稳 |
| **Joint armature** | per-joint 0.007-0.025 | **全部 1.0** | ★★★ 惯量虚高100x |
| **Contact exclusions** | 4对 (pelvis/hip, elbow/wrist) | 无 | ★ 虚假自碰撞 |
| **Object mass** | 2.0 kg | 5.0 kg (已修→2.0) | 已修复 |

### 为什么 Hand Collision 是关键

HDMI contact guidance 机制:
1. Iter 0-30 (PD active): PD 拉住物体, CEM 优化机器人动作使手接近物体
2. **Iter 31 (最后): PD = 0, 物体完全自由, 只靠机器人手的物理接触支撑**

如果手是 3 boxes (大面积, 手掌形状):
→ 能形成稳定包裹/夹持 → 物体在最后 iteration 被手托住 → CEM 学到有效策略

如果手是 1 sphere (点接触):
→ 无法形成稳定夹持 → 物体在最后 iteration 掉落 → CEM 无解 → 放弃 object tracking

### Scene XML 来源问题

CORE4D HDMI scene 是从 CORE4D 的 `scene.xml` 直接复制+重命名得来的.
CORE4D `scene.xml` 的 robot 格式来自不同的 URDF 转换流程, 与原始 HDMI suitcase scene 完全不同.

---

## 结论

1. **Euler convention 修复虽然数学正确, 但无法单独解决问题** — 旧系统内部自洽, 修一处破全局
2. **真正的瓶颈是 scene 物理配置** — hand collision (1 sphere vs 3 boxes) 使 HDMI 的 contact guidance 机制无法 work
3. **正确方向: 从原始 suitcase scene 模板重建 CORE4D scene** — 保留 suitcase 的 robot 物理 (hand/foot/armature), 只替换 object body

---

## 代码改动 (保留, 用于后续)

| 文件 | 改动 | 状态 |
|------|------|------|
| `spider/simulators/hdmi.py` | `_make_contact_guidance_model(euler_convention)` 参数化 | 保留 |
| `spider/simulators/hdmi.py` | `_load_scene_act_for_hdmi()` 新增 | 保留 |
| `spider/simulators/hdmi.py` | `get_reference()` 使用 config euler_convention | 保留 |
| `spider/config.py` | 添加 `euler_convention`, `use_scene_act` | 保留 |
| `examples/config/hdmi.yaml` | 添加 `euler_convention: "XYZ"` | 保留 |
| HDMI scene XMLs | mesh 路径改为相对路径 + mass=2.0 | 保留 |

## 结果路径

| 实验 | 路径 |
|------|------|
| E051a (body_pos+euler+mass) | `workspace/core4d/results/E051/E051_box023_euler_fix/` |
| E051b (euler only+mass) | `workspace/core4d/results/E051/E051b_box023_euler_only/` |
| E051d (scene_act.xml) | `workspace/core4d/results/E051/E051d_box023_scene_act/` |
| E051 box025 (远程) | `workspace/core4d/results/E051/E051_box025_euler_fix/` (运行中) |
| 数据验证脚本 | `workspace/core4d/scripts/eval/verify_hdmi_data.py` |
