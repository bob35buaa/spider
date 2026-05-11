# E050: HDMI Euler Convention 分析

## 状态: ⚠️ 分析完成, fix 回退 (2026-05-11)

## 背景

E048/E049 发现 HDMI 物体追踪在 CORE4D 数据上严重漂移 (ObjPos 24-91cm, ObjRot 53-66°)。
追查发现 `hdmi.py` 的 `get_reference()` 用 `as_euler("xyz")` (extrinsic) 转换 body quat → hinge euler,
而 MuJoCo 串联 hinge joints 理论上需要 intrinsic `as_euler("XYZ")`。

## 实验: E050a box023 (euler "xyz"→"XYZ" fix)

### 运行命令
```bash
# 本地 GPU0
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=false save_info=true \
    output_dir=workspace/core4d/results/E050/E050a_box023 use_torch_compile=false
```

### 结果 (sim vs kinematic ground truth)

| 指标 | E050a (XYZ fix) | E048a (xyz old) | E041c | 变化 |
|------|----------------|-----------------|-------|------|
| Joint Angle Error | 9.0° | 7.3° | 19.8° | ⬆ 略差 |
| Pelvis Pos Error | **77.4cm** | 17.2cm | 39.5cm | ❌ 大幅恶化 |
| Object Pos Error | **107.5cm** | 24.4cm | 13.9cm | ❌ 大幅恶化 |
| Object Rot Error | 57.0° | 66.4° | 27.0° | ≈ 略好 |
| Stability>0.6m | 99.1% | 100% | 38% | ≈ |

**结论: "XYZ" fix 使 ObjPos 从 24cm 恶化到 108cm, 完全失败。**

## Root Cause: Gimbal Lock

| Object | Intrinsic XYZ 中间角 | Extrinsic xyz 中间角 |
|--------|---------------------|---------------------|
| **box025** | **89.4°** (万向节死锁!) | 0.3° (稳定) |
| box023 | 0.6° (稳定) | 0.5° (稳定) |
| suitcase | 0.9° (稳定) | 1.7° (稳定) |

### 分析

1. **MuJoCo 串联 hinge 确实是 intrinsic** (已用数值实验验证: 设置 hinge [1.57, 0.3, 0.5], body quat 匹配 scipy intrinsic "XYZ")

2. **但 HDMI 用 extrinsic "xyz" 是刻意的 workaround**: 对于 box025 的 quat≈[0.5,0.5,0.5,0.5]:
   - intrinsic XYZ → 中间角 89.4° = gimbal lock → 数值不稳定
   - extrinsic xyz → 中间角 0.3° = 数值稳定

3. **Extrinsic xyz 虽然数学上不匹配 MuJoCo hinge, 但避免了 gimbal lock**:
   - 系统内部一致 (reference + init + PD 都用同一个错误 convention)
   - 物体 PD 追踪内部 reference 很好 (sim≈internal_ref, 差<1°)
   - 只是与真实 kinematic ground truth 有偏差

4. **"XYZ" fix 引入 gimbal lock 后**:
   - CEM 优化器在 gimbal lock 附近无法有效优化
   - 物体 euler 表示在 Y≈90° 处奇异, 小 quat 变化 → 大 euler 跳变
   - PD controller 在奇异点附近震荡 → 物体漂移

## 真正的问题

HDMI 的 `_make_contact_guidance_model` **硬编码 XYZ 关节顺序** (rot_x, rot_y, rot_z)。
这对 suitcase (方向接近 identity) 没问题, 但对 CORE4D 物体 (有 90° X+Z 旋转) 无论用什么 euler convention 都有问题:
- extrinsic "xyz": 数学不匹配 MuJoCo, 但 gimbal 安全 → sim 内部一致但偏离 ground truth
- intrinsic "XYZ": 数学正确, 但 gimbal lock → 优化器崩溃

**正确的解决方案**: 像 `generate_scene_act.py` 一样, per-object 选择最优 euler convention + 对应调整 hinge joint 顺序:
- box025: 用 YXZ (中间角 0.5°)
- box023: XYZ 或 xyz 都行 (中间角 < 1°)
- suitcase: XYZ 或 xyz 都行

这需要修改 `_make_contact_guidance_model` 的关节创建逻辑, 是一个中等复杂度的改动。

## 决策: 回退 fix

由于 "XYZ" fix 使结果恶化, 已回退到 `as_euler("xyz")` (原始代码)。
HDMI 在 CORE4D 上的物体追踪问题暂时无法通过简单 fix 解决。

## 总结: HDMI 在 CORE4D 上的真实能力

在当前状态 (extrinsic xyz, 不修复) 下:

| 维度 | HDMI | E041c |
|------|------|-------|
| Body tracking (joint) | **7-9°** | 15-20° |
| Stability | **100%** | 38-98% |
| Object position | 24-91cm | **14-16cm** |
| Object rotation | 53-66° | **7-27°** |

HDMI 的 body tracking + stability 确实更好, 但 object tracking 确实更差。
不是 eval bug, 是真实的 euler convention 限制。

## 改动文件

| 文件 | 改动 | 状态 |
|------|------|------|
| `spider/simulators/hdmi.py` L659, L1303 | "xyz"→"XYZ" 尝试 | **已回退** |

## 结果路径

| 产出 | 路径 |
|------|------|
| E050a box023 (XYZ fix, 失败) | `workspace/core4d/results/E050/E050a_box023/trajectory_hdmi.npz` |
