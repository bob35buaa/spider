# E048: 碰撞盒修复后 Baseline + box023 HDMI 跨算法对比

## 状态: ✅ 完成 (2026-05-11)

## 背景

Phase 12 (E044-E047) 三方向探索均未超越 E041c。核心问题: contact 上限 ~66% 是**数据质量**还是**算法限制**?

本实验组修复了严重的碰撞盒模板 Bug (详见 log/54), 在修正后的物理场景上:
1. 重跑 E041c baseline (box025/bucket010/desk005)
2. 新增 box023 (小箱子) + box001/box024 (大箱子)
3. **用 HDMI workflow 跑 box023, 与 E041c 直接对比**

## 碰撞盒修复总结

| Case | 旧碰撞盒 (half-m) | 新碰撞盒 (mesh×1.05) | 影响 |
|------|-------------------|---------------------|------|
| box025 | 0.305×0.305×0.446 | 0.377×0.378×0.469 | 增大24% — 碰撞盒匹配视觉 |
| box023 | 0.305×0.305×0.446 | 0.179×0.183×0.206 | **缩小41%** — 修复1.8x过大 |
| bucket010 | 0.201×0.371×0.202 | 0.209×0.209×0.390 | Y/Z形状修正 |
| box001 | 0.305×0.305×0.446 | 0.269×0.330×0.430 | 微调 |
| box024 | 0.305×0.305×0.446 | 0.266×0.273×0.517 | 修正形状 |

---

## 核心结果

### A. E041c Baseline 重跑 (碰撞盒修复后)

| Case | MPKPE(cm)† | ObjPos(cm) | Stability>0.6 | Contact<10cm | 旧E041c Contact |
|------|-----------|------------|---------------|-------------|----------------|
| box025 | 24.8 | 15.5 | **98%** | 54% | 66% |
| bucket010 | 27.4 | 18.6 | **100%** | 7% | 57% |
| desk005 | 156.8‡ | 92.1 | **100%** | 0% | 4% |

†MPKPE 为世界坐标系值, 不可与 E041c 的局部坐标系 1.4cm 直接对比 (局部帧漂移是正常的)
‡desk005 机器人行走偏移 156cm, 物理上稳定但完全偏离 ref

**碰撞盒修复的影响**:
- bucket010: Contact 从 57% 暴跌至 7% — 旧碰撞盒 Y/Z 互换导致"形状不对", 旧结果实际上是在错误物理下取得的
- box025: Contact 54% (旧 66%) — 碰撞盒增大 24%, 物体更难靠近
- desk005: 机器人完全偏移 — 需要进一步调查

### B. box023 HDMI vs E041c 对比 (★核心实验★)

| 指标 | HDMI (E048a) | E041c (E048b) | 判定 |
|------|-------------|-------------|------|
| MPKPE (cm) | **0.7** | 39.5 | HDMI ≫ E041c |
| Stability>0.6 | **100%** | 38% ❌ | E041c 摔倒 |
| Contact<10cm | **93%** | 82%* | HDMI 真接触 |
| Contact<5cm | **53%** | 66%* | *E041c是摔倒导致 |
| ObjPos (cm) | **0.7** | 13.9 | HDMI ≫ E041c |
| Pelvis min (m) | 0.614 | 0.193 | E041c pelvis触地 |

*注意: E041c 的 82% Contact 是因为机器人**摔倒在物体上**, 不是真正的手部接触。视频证实:
- E048a (HDMI): 机器人稳定行走, 双手持续接触小箱子
- E048b (E041c): 机器人 t≈2s 前倾倒地, 身体压在箱子上

### C. 新 Case 评估

| Case | MPKPE(cm) | Stability>0.6 | Contact<10cm | 备注 |
|------|-----------|-------------|-------------|------|
| box001 | 38.2 | **100%** | 0% | ref 无接触帧 — 数据质量问题 |
| box024 | 25.4 | **100%** | 0% | ref 无接触帧 — 数据质量问题 |

box001/box024 的 OmniRetarget ref 中手-物体距离 > 15cm — ref 本身没有接触, 自然无法产生 sim 中的接触。

---

## 可视化分析

### box023 E041c (t=2s)
- ref (左): 机器人站立, 双手前伸搬小箱子
- sim (右): 机器人严重前倾, 即将摔倒, 手臂失控, 箱子在地上
- 判定: **E041c 在 box023 上完全失败**

### box025 Baseline (t=2s)
- ref (左): 机器人站在大箱子后面
- sim (右): 机器人俯卧在箱子顶部 — 碰撞盒修正后箱子是 75×76×94cm (正确尺寸), 太大了
- 判定: 碰撞盒修正揭示了 box025 真正的物理尺寸, 单人无法处理

### bucket010 (t=2s)
- ref (左): 机器人站在桶旁, 手臂接触桶壁
- sim (右): 机器人站在桶旁但手未接触 — 碰撞盒形状修正后接触力学改变
- 判定: 物理上合理, 但之前的 57% contact 是在错误碰撞盒下取得的

---

## Claims 验证

| Claim | 结果 | 判定 |
|-------|------|------|
| C1: Stability ≥ 95% (HDMI+E041c box023) | HDMI=100%, E041c=38% | ❌ E041c 摔倒 |
| C2: Contact<10cm ≥ 50% (小箱子) | HDMI=93% ✅, E041c=82%(假) | ✅ HDMI 达标 |
| C3: 判定 data vs algorithm | HDMI ≫ E041c on same data | **✅ 算法问题, 非数据问题** |

## 关键结论

### 1. **算法是瓶颈, 不是数据** (核心发现)

同样的 CORE4D box023 数据:
- HDMI: MPKPE=0.7cm, Contact=93%, Stability=100%
- E041c: MPKPE=39.5cm, Contact=82%(假), Stability=38%

差距是数量级的。HDMI 能在 CORE4D 数据上实现优秀的 body tracking + contact quality。E041c 在 box023 上完全失败。

### 2. **碰撞盒修复揭示了旧结果的不可靠性**

- 旧 E041c 的 66% contact (box025) 和 57% contact (bucket010) 是在错误碰撞盒下取得的
- bucket010 的碰撞盒 Y/Z 互换 — 旧结果本质上是在错误物理环境下的伪结果
- 修正后 bucket010 contact 从 57% → 7%, box025 从 66% → 54%

### 3. **HDMI 算法的核心优势**

HDMI 相比 E041c MJWP 的关键差异:
1. PD 增益从 Isaac Lab 精确导入 (vs MJWP 的 XML 默认值)
2. 手腕阻尼修复 (dof_damping=5.0, 临界阻尼)
3. 手腕噪声归零 (wrist noise=0)
4. 动态 XML surgery 生成 contact guidance model (vs 预构建 scene_act.xml)
5. 预计算 reward reference (GPU-native vs per-step CPU 计算)

### 4. **新 case box001/box024 的 ref 质量问题**

OmniRetarget 输出的 ref 中手-物体距离 > 15cm → 没有接触帧。需要:
- 检查 OmniRetarget 的 `--replace_wrist_with_fingertip` 是否正确生效
- 或者这些 case 在原始 SMPLX 数据中本身手就远离物体 (单人视角看不到手接触)

---

## 下一步方向

1. **将 HDMI 的关键优化移植到 MJWP**: PD 增益导入、手腕阻尼修复、手腕噪声归零
2. **在更多 case 上验证 HDMI**: 用 convert_core4d_to_hdmi.py 转换 box025/bucket010, 验证 HDMI 在大物体上也优于 E041c
3. **修复 box001/box024 ref 质量**: 检查 OmniRetarget 输出

---

## 改动文件

| 文件 | 操作 | 实验 |
|------|------|------|
| `workspace/core4d/scripts/convert/fix_collision_boxes.py` | 新建 | 碰撞盒修复 |
| `workspace/core4d/scripts/convert/convert_core4d_to_hdmi.py` | 新建 | E048a |
| `workspace/core4d/scripts/convert/generate_scene_act.py` | 扩展 CASES | 全部 |
| `/home/ubuntu/Workspace/HDMI/cfg/task/G1/hdmi/move_box023.yaml` | 新建 | E048a |
| `example_datasets/processed/hdmi/.../move_box023/scene/mjlab scene.xml` | 新建 | E048a |

## 结果路径

| 产出 | 路径 |
|------|------|
| E048 baseline box025 | `workspace/core4d/results/E048/E048_box025_baseline.{npz,mp4}` |
| E048 baseline bucket010 | `workspace/core4d/results/E048/E048_bucket010_baseline.{npz,mp4}` |
| E048 baseline desk005 | `workspace/core4d/results/E048/E048_desk005_baseline.{npz,mp4}` |
| E048 box023 E041c | `workspace/core4d/results/E048/E048_box023.{npz,mp4}` |
| E048 box001 E041c | `workspace/core4d/results/E048/E048_box001.{npz,mp4}` |
| E048 box024 E041c | `workspace/core4d/results/E048/E048_box024.{npz,mp4}` |
| E048a box023 HDMI | `workspace/core4d/results/E048/E048a_hdmi/trajectory_hdmi.npz` |
| HDMI motion data | `/home/ubuntu/Workspace/HDMI/data/motion/g1/core4d/box023_person1/` |
