# E048-E049 修正: HDMI 评估指标严重失真 + 真实对比

## 状态: 🔴 重大修正 (2026-05-11)

## 发现的评估 Bug

**trajectory_hdmi.npz 的双通道不是 sim vs ground-truth ref**:
- Channel 0 = sim (物理仿真)
- Channel 1 = HDMI 内部 ref (随仿真漂移, 非真实 kinematic ref!)

HDMI 内部 ref 与 kinematic ground truth 的偏差:
- box023: t=2s 偏移 45°, t=3s 偏移 422°
- box025: t=2s 偏移 45°, t=4s 偏移 108°

**之前报的 MPKPE=0.3-0.7cm / Contact=93-99% 全部是虚假指标。**

---

## 正确评估 (sim vs kinematic ground truth)

### box023 (小箱子, 34×35×39cm)

| 指标 | HDMI (真实) | E041c | 谁更好 |
|------|------------|-------|--------|
| Joint Angle Error | **7.3°** | 19.8° | HDMI 2.7× |
| Pelvis Pos Error | **17.2cm** | 39.5cm | HDMI 2.3× |
| Object Pos Error | 24.4cm | **13.9cm** | E041c 1.8× |
| Object Rot Error | 66.4° | **27.0°** | E041c 2.5× |
| Stability>0.6m | **100%** | 38% | HDMI 显著 |

### box025 (大箱子, 75×76×94cm)

| 指标 | HDMI (真实) | E041c | 谁更好 |
|------|------------|-------|--------|
| Joint Angle Error | **5.3°** | 21.5° | HDMI 4× |
| Pelvis Pos Error | 78.7cm | **24.8cm** | E041c 3.2× |
| Object Pos Error | 90.9cm | **15.5cm** | E041c 5.9× |
| Object Rot Error | 53.2° | **6.6°** | E041c 8× |
| Stability>0.6m | **100%** | 98% | 持平 |

---

## 修正后的结论

### 1. HDMI ≠ 全面优于 E041c, 而是各有所长

| 维度 | HDMI 优势 | E041c 优势 |
|------|----------|-----------|
| 机器人 body tracking | ✅ Joint err 5-7° (好 3-4x) | |
| 机器人稳定性 | ✅ 100% (不摔倒) | |
| 物体位置追踪 | | ✅ 14-16cm (好 2-6x) |
| 物体方向追踪 | | ✅ 7-27° (好 2-8x) |
| 全局位置 (pelvis) | 小箱子好 | 大箱子好 |

### 2. HDMI 的物体 PD 控制器严重失效

HDMI 的 contact guidance 使用 PD actuator 控制物体, 但:
- 物体位置漂移 24-91cm (vs E041c 的 14-16cm)
- 物体旋转漂移 53-66° (vs E041c 的 7-27°)
- 物体越大失效越严重 (box025 比 box023 差 3.7x)

根因: HDMI 的 object PD gains (kp=20, kd=20) 可能不足以对抗碰撞力, 或者 gain schedule (guidance_decay) 太激进。

### 3. "算法是瓶颈" 的结论需要修正

之前说 "HDMI 碾压 E041c, 算法是瓶颈" — **错误**。正确结论:
- 两种方法在不同维度各有优势
- HDMI 在 body tracking + stability 上更好 (得益于 Isaac PD gains + wrist damping)
- E041c 在 object tracking 上更好 (得益于 stronger contact guidance gains)
- 物体交互的核心问题 (box025 单人无法搬) 两者都未解决

### 4. 可视化 Bug 总结

render_trajectory_video.py 存在多个 bug:
1. euler convention 用了 YXZ (CORE4D scene_act 的) 而非 XYZ (HDMI 的)
2. `flatten_mjwp` 取 `qpos[:, -1, :]` = channel 1 (内部 ref), 不是 sim
3. 时间对齐: kin (542 frames) vs phys (250 frames) 线性 resample 导致时间错位

---

## 被撤回的错误结论

| 之前的结论 | 修正为 |
|-----------|--------|
| HDMI box023 Contact=93% | ❌ 虚假 (sim vs drifted ref) |
| HDMI box025 Contact=99% | ❌ 虚假 (sim vs drifted ref) |
| HDMI MPKPE=0.3-0.7cm | ❌ 虚假 (应为 17-79cm pelvis err) |
| "HDMI 碾压 E041c" | ❌ 各有所长, 不是碾压 |
| "算法是瓶颈, 非数据" | ⚠️ 需更审慎: 两种算法都有严重缺陷 |

---

## 结果路径 (不变)

视频 v5 (正确渲染): `workspace/core4d/results/E049/E049e_hdmi_box025_comparison_v5.mp4`

---

## 运行命令记录

### E048 碰撞盒修复
```bash
# 本地修复碰撞盒
uv run workspace/core4d/scripts/convert/fix_collision_boxes.py
# 本地重新生成 scene_act.xml (21 case)
uv run workspace/core4d/scripts/convert/generate_scene_act.py
```

### E048 E041c Baseline 重跑 (远程 2-GPU 并行)
```bash
# 远程: spider-remote (10.100.71.70:58122, user xiayb)
# 脚本: workspace/core4d/scripts/run_E048_remote.sh
# GPU0: box025_person1, bucket010_person1, box001_person1
# GPU1: desk005_person2, box023_person1, box024_person1
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && \
    tmux new-session -d -s e048 "bash /tmp/run_E048.sh"'
# 结果回收:
scp spider-remote:/home/xiayb/pHRI_workspace/spider/workspace/core4d/results/E048/E048_*.{npz,mp4} \
    workspace/core4d/results/E048/
```

### E048a HDMI box023
```bash
# 1. 数据转换
uv run workspace/core4d/scripts/convert/convert_core4d_to_hdmi.py --case box023_person1
# 输出: /home/ubuntu/Workspace/HDMI/data/motion/g1/core4d/box023_person1/

# 2. 运行 HDMI
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=false save_info=true \
    output_dir=workspace/core4d/results/E048/E048a_hdmi use_torch_compile=false
```

### E049 HDMI 优化移植 (远程)
```bash
# 脚本: workspace/core4d/scripts/run_E049_remote.sh
# Config: examples/config/override/core4d_e049.yaml
# GPU0: box023_person1, box025_person1
# GPU1: bucket010_person1, desk005_person2
ssh spider-remote 'cd /home/xiayb/pHRI_workspace/spider && \
    tmux new-session -d -s e049 "bash workspace/core4d/scripts/run_E049_remote.sh"'
scp spider-remote:/home/xiayb/pHRI_workspace/spider/workspace/core4d/results/E049/E049_*.{npz,mp4} \
    workspace/core4d/results/E049/
```

### E049e HDMI box025
```bash
# 1. 数据转换
uv run workspace/core4d/scripts/convert/convert_core4d_to_hdmi.py --case box025_person1
# 2. 运行 HDMI
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box025 +data_id=0 viewer=none save_video=false save_info=true \
    output_dir=workspace/core4d/results/E049/E049e_hdmi_box025 use_torch_compile=false
```

### 评估命令
```bash
# E041c (外部 ref):
uv run workspace/core4d/scripts/eval/eval_comprehensive.py <task> <sim.npz> --ref <ref.npz>
# HDMI (双通道, 注意: channel 1 是漂移的内部 ref, 不可信!):
# 正确方法: 手动比较 sim (channel 0) vs trajectory_kinematic.npz
```

### 渲染命令
```bash
# 正确渲染 (使用 CORE4D freejoint scene, 避免 euler 转换问题):
MUJOCO_GL=egl python3 render_v5.py  # 见 v5 内联脚本
# 或修复后的脚本:
MUJOCO_GL=egl uv run workspace/hdmi_reproduce/scripts/render_trajectory_video.py \
    --scene <scene.xml> --kin <kin.npz> --phys <phys.npz> --output <out.mp4> --euler XYZ
```

---

## Root Cause: HDMI euler convention bug

**文件**: `spider/simulators/hdmi.py` 第 659 行和第 1303 行
**Bug**: `as_euler("xyz")` (extrinsic) 应为 `as_euler("XYZ")` (intrinsic)
**影响**: 物体方向大旋转时 (CORE4D boxes 有 90° X+Z), 参考方向错误达 119-278°
**为何未暴露**: HDMI 原始 suitcase 方向接近 identity (仅 -88° 绕 Z), extrinsic/intrinsic 差异仅 2-6°
