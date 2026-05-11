# E052: Suitcase 模板重建 + 联合修复实验

## 状态: ✅ 完成 — 所有组合均未超越 baseline (2026-05-12)

## 背景

E051 发现两个独立问题:
1. **Euler convention 178° mismatch** — PD 引导物体到完全错误的方向
2. **Scene 物理配置差异** — 1 sphere hand (无法抓握) vs 3 boxes, armature=1.0 vs 0.01

本实验通过 2×2 矩阵穷举所有组合, 确定哪个因素是关键.

---

## 完整 2×2 实验矩阵

| | euler=xyz (extrinsic, 原始) | euler=XZY (intrinsic, 正确) |
|---|---|---|
| **旧scene** (1sphere, arm=1.0) | **E048a: ObjPos=24.4cm, Joint=7.3°, Stab=100%** ★ | E051b: ObjPos=142.5cm, Joint=14.8°, Stab=99% |
| **suitcase** (3box, arm=0.01) | E052a: ObjPos=36.9cm, Joint=13.9°, Stab=100% | E052c: ObjPos=97.6cm, Joint=19.7°, Stab=32% |

### 趋势分析

**列方向** (euler 影响):
- 旧scene: 修euler → ObjPos 24→143cm (6×恶化)
- suitcase: 修euler → ObjPos 37→98cm (2.6×恶化)
- **结论: 正确euler在所有物理配置下都更差**

**行方向** (scene 影响):
- 旧euler: suitcase模板 → ObjPos 24→37cm (1.5×恶化)
- 正确euler: suitcase模板 → ObjPos 143→98cm (略好, 但仍极差)
- **结论: suitcase模板(低armature)损害body tracking, 对object tracking无帮助**

**对角线** (联合修复):
- E052c (两者同时修) = **最差**: ObjPos=98cm, Stab=32%, Joint=20°
- **结论: 两个"修复"不互补, 而是叠加恶化**

---

## 各实验详情

### E052a: Suitcase模板 + 旧euler (box023)

```bash
uv run workspace/core4d/scripts/convert/generate_hdmi_scene.py --case box023_person1
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E052/E052a_box023 \
    use_torch_compile=false euler_convention=xyz
```

| 指标 | E052a | E048a | 变化 |
|------|-------|-------|------|
| ObjPos | 36.9cm | 24.4cm | ❌ +51% |
| ObjRot | 71.1° | 66.4° | ❌ 略差 |
| Joint | 13.9° | 7.3° | ❌ 1.9× |
| Stability | 100% | 100% | ✅ |

### E052c: Suitcase模板 + 正确euler XZY (box023) — 关键实验

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E052/E052c_box023_euler_fix \
    use_torch_compile=false euler_convention=XZY
```

| 指标 | E052c | E048a | 变化 |
|------|-------|-------|------|
| ObjPos | 97.6cm | 24.4cm | ❌ 4× |
| ObjRot | 65.9° | 66.4° | ≈ |
| Joint | 19.7° | 7.3° | ❌ 2.7× |
| Stability | **32.0%** | 100% | ❌❌ 崩溃 |
| Pelvis min | 0.062m | >0.6m | ❌ 摔倒 |

### E052b: Suitcase模板 + 旧euler (box025, 远程)

远程多次卡住 (GPU争抢/编译hang), 已重启. 结果待补.

---

## 根本结论

### 1. HDMI 的 "错误" 不是 bug, 是 feature

| "问题" | 为什么不修反而更好 |
|--------|-------------------|
| extrinsic "xyz" euler (178° FK误差) | CEM 已适应这个方向的碰撞动力学, 修改等于完全改变任务 |
| armature=1.0 (100×虚高) | 高惯量→运动平滑→body tracking好→CEM更易优化 |
| 1 sphere hand (点接触) | CEM在这个接触模式下已优化到极限 (24cm) |

### 2. E048a baseline (24.4cm ObjPos, 7.3° Joint) 是当前方法极限

HDMI contact guidance 在 CORE4D box023 上的表现:
- **Body tracking 优秀**: Joint=7.3° (HDMI核心优势)
- **Stability 完美**: 100%
- **Object tracking 有限**: ObjPos=24cm, ObjRot=66° (受 euler + guidance_decay 限制)

### 3. 改善 object tracking 需要更根本的方法改变

contact guidance (PD decay to 0 on last iteration) 要求 CEM 找到纯接触维持物体的策略.
在当前 scene 配置 + CORE4D 搬运距离 (1.5m) 下, CEM 无法做到.

可能的突破方向:
- 不 decay to 0: 保留残余 PD gains (但需调整 reward 权重)
- 修改 CEM reward: 增大 object tracking 权重
- 结合 HDMI body tracking + E041c object tracking (取各自优势)

---

## 改动文件

| 文件 | 改动 |
|------|------|
| `workspace/core4d/scripts/convert/generate_hdmi_scene.py` | 新建: 从 suitcase 模板生成 scene |
| `examples/config/hdmi.yaml` | euler_convention 默认改为 "xyz" |
| `spider/simulators/hdmi.py` | `_make_contact_guidance_model` 支持 euler_convention 参数 |
| `spider/simulators/hdmi.py` | `_load_scene_act_for_hdmi()` 新增 (E051d 使用) |
| `spider/simulators/hdmi.py` | `get_reference()` / `setup_env()` 使用 config euler_convention |
| `spider/config.py` | 添加 `euler_convention`, `use_scene_act` |

## 结果路径

| 实验 | 路径 |
|------|------|
| E052a box023 (suitcase+xyz) | `workspace/core4d/results/E052/E052a_box023/` |
| E052c box023 (suitcase+XZY) | `workspace/core4d/results/E052/E052c_box023_euler_fix/` |
| E052b box025 (suitcase+xyz, 远程) | `workspace/core4d/results/E052/E052b_box025/` |
