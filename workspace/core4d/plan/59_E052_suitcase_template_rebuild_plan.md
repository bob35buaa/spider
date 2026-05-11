# E052: 从 Suitcase 模板重建 CORE4D HDMI Scene

## Context

E051 确认 HDMI object tracking 差 (24-91cm) 的根因是 **scene 物理配置与原始 HDMI suitcase 场景严重不匹配**:

| 部件 | 原始 HDMI suitcase (能 work) | 当前 CORE4D HDMI scene (不 work) |
|------|----------------------------|--------------------------------|
| Hand collision | **3 boxes/wrist** (手掌形状, 面接触) | 1 sphere r=0.05 (点接触) |
| Foot collision | **7 capsules/foot** (全足底) | 4 spheres r=0.005 (点接触) |
| Joint armature | **per-joint 0.007-0.025** | 全部 1.0 (虚高 100×) |
| Contact exclusions | **4对** (pelvis/hip, elbow/wrist) | 无 |
| Actuator type | **general** (affine PD, gainprm/biasprm) | position (kp=500) |

**历史**: E006 (2026-05-01) 已发现 hand collision 差异并为 MJWP 创建了 `scene_forearm.xml`,
但 E048 创建 HDMI scene 时直接从 `scene.xml` (1 sphere) 复制, 未应用此修复.

**Euler convention**: E051 证明修正 euler 会打破内部自洽, 使结果更差.
原始 HDMI suitcase 也用错误的 extrinsic "xyz" convention, 误差 2-6° (可接受).
**本实验保持旧 euler convention 不变**, 专注于物理配置对齐.

---

## Claims

- C1: box023 Object Pos Error < 10cm (当前 24.4cm)
- C2: box023 Object Rot Error < 20° (当前 66.4°)  
- C3: Stability ≥ 95% (当前 100%, 不退化)
- C4: Joint Error < 10° (当前 7.3°, 不退化)

**参照系**: HDMI suitcase 在其原始 motion 数据上的效果 (几乎完美追踪).
如果 C1-C4 都满足, 说明物理配置是唯一瓶颈; 如果仍不满足, 则还有其他因素
(如 euler convention 对大旋转物体的影响, motion 数据质量等).

---

## 实现方案

### Step 1: `generate_hdmi_scene.py` — 从 suitcase 模板生成 CORE4D scene

**输入**:
- 模板: `example_datasets/processed/hdmi/unitree_g1/humanoid_object/move_suitcase/scene/mjlab scene.xml`
- 物体信息: CORE4D case 的 mesh path, collision AABB, initial pos/quat

**处理流程**:
```
suitcase scene.xml
    │
    ├── Robot 部分 (完全保留)
    │   ├── body hierarchy (robot/ namespace prefix)
    │   ├── per-joint armature (0.007-0.025)
    │   ├── general actuators (affine PD)
    │   ├── hand collision (3 boxes × 2)
    │   ├── foot collision (7 capsules × 2)
    │   └── contact exclusions (4对)
    │
    └── Object 部分 (替换)
        ├── body name: suitcase/suitcase (保持)
        ├── geom type=box, size → CORE4D AABB × 1.05
        ├── mesh file → CORE4D object .obj path
        ├── mass → 2.0 kg
        ├── inertial → 根据 AABB 计算
        ├── body pos → 保持原始 [0.4, 0.05, 0] (不修改! 提供 PD 追踪力)
        └── contact pairs (hand-suitcase) → 保持
```

**关键设计决策**:

1. **body_pos 不改**: 保持 suitcase 原始的 `pos="0.4 0.05 0"`.
   - body_pos 决定 slide joints 的 offset: `slide = 物体实际位置 - body_pos`
   - PD force = kp × slide_offset. 当 offset 大 (1-2m) 时, 持续拉力 20-40N
   - E051a 实验证明: 将 body_pos 修正为 frame-0 精确位置 (offset≈0) → PD 无恢复力 → ObjPos 从 24cm 恶化到 132cm
   - 原始 suitcase 也用 [0.4, 0.05, 0] (不是精确位置), 这个 offset 提供的拉力是 contact guidance 的隐式设计
   - 对所有 CORE4D cases 统一用同一个 body_pos, 无需 per-case 调整

2. **Euler convention 不改**: 使用默认 XYZ joints + extrinsic "xyz" euler.
   对 box023 (178° mismatch) 虽然 FK 错, 但系统内部自洽.
   改善物理配置后, CEM 应能在"错误"方向上也建立有效 contact.

3. **Mesh 路径**: 使用相对路径, 确保本地和远程都能运行.

4. **Actuator type**: suitcase 用 `<general>` (不是 `<position>`). 
   `hdmi.py` setup_env L563-585 会用 Isaac PD gains 覆盖, 所以 XML 里的 type 不影响运行时.

### Step 2: 回退 `hdmi.py` euler 相关代码

E051 的 euler 修改需要回退, 确保与旧行为一致:
- `_make_contact_guidance_model`: 回退到不接受 euler_convention 参数 (或保留参数但默认 "XYZ" 且硬编码 XYZ joints)
- `get_reference()`: 回退到 `as_euler("xyz")` (extrinsic)
- `setup_env()`: 回退到原始行为

**实际做法**: 保留新代码但设 `euler_convention="xyz"` (小写) 作为默认值.
在 `get_reference()` 中直接使用 `config.euler_convention` — 小写 "xyz" = extrinsic = 旧行为.

### Step 3: 验证

**E052a: box023** (本地, ~55min)
```bash
uv run workspace/core4d/scripts/convert/generate_hdmi_scene.py --case box023_person1
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/run_hdmi.py \
    task=move_box023 +data_id=0 viewer=none save_video=true save_info=true \
    output_dir=workspace/core4d/results/E052/E052a_box023 use_torch_compile=false
```

**E052b: box025** (远程, ~55min, 可并行)
```bash
uv run workspace/core4d/scripts/convert/generate_hdmi_scene.py --case box025_person1
ssh spider-remote '... python examples/run_hdmi.py task=move_box025 ...'
```

### Step 4: 评估

使用 FK 对比: sim xpos/xquat vs kinematic ground truth xpos/xquat
(不用 euler qpos 对比, 避免 convention 混淆)

---

## 风险分析

### 风险 1: Euler mismatch (178°) 即使物理配置正确也无法被 CEM 克服

**分析**: 对 suitcase, euler 误差仅 2-6° → PD 引导方向接近正确 → CEM 容易找到 contact.
对 box023, euler 误差 178° → PD 引导方向完全相反 → CEM 可能无法在完全错误的方向上建立有效 contact.

**如果发生**: 需要同时修 euler + 物理配置. 但 E051b 表明修 euler 后 ObjPos=142cm (更差).
这可能是因为 E051b 仍用 1 sphere hand. 如果用 3 boxes hand + 正确 euler, 效果可能不同.

**Plan B**: E052 失败后, 尝试 E052c: suitcase 模板 + 正确 euler (XZY).

### 风险 2: Namespace prefix (robot/) 导致 body 匹配失败

**分析**: suitcase scene 用 `robot/pelvis`, CORE4D motion.npz 用 `pelvis`.
`hdmi.py _find_in_scene()` 已处理 prefix stripping.

**验证**: 打印 body 匹配日志确认.

### 风险 3: 原始 suitcase 的 `<general>` actuator 与 HDMI Isaac PD gains 覆盖冲突

**分析**: `setup_env()` L563-585 遍历所有 actuators, 按 joint name 查找 Isaac gains 并覆盖.
suitcase scene 的 actuator 命名为 `robot/{joint_name}`, 代码 L566 做 `split("/")[-1]`.
应该能正确匹配.

**验证**: 检查日志 "Actuator gains overridden" 确认所有 29 个 joints 都被覆盖.

---

## 执行顺序

```
1. 回退 euler convention 到旧行为 (config默认值改为 "xyz")     [5min]
2. 创建 generate_hdmi_scene.py                                [30min]
3. 生成 box023 / box025 的新 scene                            [1min]  
4. 验证 MuJoCo 加载成功 + body 匹配                           [2min]
5. 本地运行 E052a box023                                      [55min]
6. 远程运行 E052b box025 (并行)                               [55min]
7. 评估 + 视频分析                                            [10min]
8. 记录实验日志                                               [10min]
```

总预计: ~2h (含 1h 等待 GPU)

---

## 关键文件

| 文件 | 操作 |
|------|------|
| `workspace/core4d/scripts/convert/generate_hdmi_scene.py` | 新建: 从 suitcase 模板生成 scene |
| `example_datasets/processed/hdmi/.../move_box023/scene/mjlab scene.xml` | 重新生成 |
| `example_datasets/processed/hdmi/.../move_box025/scene/mjlab scene.xml` | 重新生成 |
| `examples/config/hdmi.yaml` | 修改: euler_convention 默认值改为 "xyz" |
| `spider/simulators/hdmi.py` | 修改: get_reference() 支持小写 convention = extrinsic |

## 成功标准

| 场景 | 如果成功 | 如果失败 |
|------|---------|---------|
| C1-C4 全部通过 | 物理配置是唯一问题, euler 可忽略 → 批量生成所有 case scene | Euler 也是问题 → E052c 同时修 euler + 物理 |
| ObjPos < 10cm 但 ObjRot > 40° | 物理配置解决了位置追踪, 但旋转受 euler 限制 → 需要修 euler |
| Stability 退化 | per-joint armature (0.01) 太低 → 需要调整或保持 1.0 |
