# E060.2: case-correct palm_normal (+x = audit §2.2 best) — 结果

## 状态: ❌❌ **CATASTROPHIC FAIL — 全面退化, 比 E060.0/.1 都差**

**核心结论**: 用 audit §2.2 找到的"真实最优"palm_normal (box023 L/R = +x; bucket005_s2 L = -y unchanged, R = +x) 让两个 case 的稳定性指标**全面崩盘**. 修订自 E060.1 的 "implicit prior + geometric alignment" 假设 **被反转**.

更重要的是: 这个结果加上 **box025 sphere→3-box regression 发现** (详见 log 74), 现在**强烈怀疑 E060.0/.1/.2 整套对比都建立在 3-box hand port 引入的物理 bug 上**. E060.2 的 catastrophic fail 可能不是 reward 问题, 而是 3-box geometry 与 reward eef_offset (`[0.05,0,0]`) 错位 12.5cm 在不同 palm normal 配置下放大不同程度.

## 实验配置

| 项 | 值 |
|----|---|
| Reward stack | E041c + 新 yaml (`core4d_e060_2_box023.yaml`, `core4d_e060_2_bucket005_s2.yaml`) 仅改 palm_normal |
| box023 palm_normal | L = `[1,0,0]` (+x), R = `[1,0,0]` (+x) |
| bucket005_s2 palm_normal | L = `[0,-1,0]` (unchanged), R = `[1,0,0]` (+x) |
| 数据层 | 同 E060.0/.1 (3-box hand + box023 margin 0.90) |
| GPU | parallel: GPU 0 box023, GPU 1 bucket005_s2 |
| Wall time | ~35min |
| 输出 | `workspace/core4d/results/E060/E060_2_*.{npz,mp4}` |

## 数值结果对比

| 指标 | E060.0 (E041c hardcoded) | E060.1 (ori=0) | **E060.2 (case-correct +x)** | .2 vs .0 |
|------|-------------------------|----------------|-------------------------------|----------|
| **box023** | | | | |
| pelvis_min_intent (m) | 0.176 | 0.146 | **0.113** | -0.063 |
| pelvis_mean_intent (m) | 0.577 | 0.469 | **0.260** | **-0.317** |
| stable_intent % | 74.1 | 53.4 | **5.2** | **-69pp** |
| **bucket005_s2** | | | | |
| pelvis_min_intent (m) | 0.117 | 0.210 | **0.072** | -0.045 |
| pelvis_mean_intent (m) | 0.307 | 0.482 | **0.245** | -0.062 |
| stable_intent % | 14.6 | 53.9 | **2.2** | -12pp |

**两个 case 都呈"stable_intent 个位数 + pelvis_mean_intent < 0.30m"** = **全程趴地**, 比 E060.0 baseline 还差得多.

## Claims 验证

| Claim | 描述 | 实际 | 通过 |
|-------|------|------|------|
| C1 | box023 pelvis_min_intent 比 E060.0 (0.176) 和 E060.1 (0.146) 都改善 | **0.113 (比都差)** | ❌ |
| C2 | box023 stable_intent ≥ 70% (vs E060.1 53%) | **5.2%** | ❌ |
| C3 | 视觉 keyframe ≥ 1/2 站立 + 持物 (intent mid + end) | 未取（数值已说明全程趴地） | ❌ |

**0/3**.

## 假设反转

**E060.1 修订假设** (log 72 §4.1): "hardcoded palm normal 是 implicit wrist-orientation prior, case-correct prior 应同时帮两 case (prior + 几何对齐)."

**E060.2 实验数据反向**: case-correct (+x) prior 反而让两 case 都崩到 stable_intent <10%. 这表明**不仅 case-correct +x 不是更好的 prior, 它实际上是更坏的 prior**.

可能的解释 (但**首先要排除 3-box port bug**, 见 log 74):
1. +x = wrist 局部 fingers 轴指向方向. wrist 转到让 +x 指物体 → 手指尖直指物体 → 物理上是"戳"姿势, 不利稳定握
2. -y/+y (hardcoded) = wrist 局部侧向. wrist 转到让 ±y 指物体 → 手掌侧贴物体 → 物理上是"扶"姿势, 自然稳定
3. audit §2.2 算的是"哪个 wrist 局部轴恰好平均指向物体", 没考虑哪个轴**适合稳定接触**

但即使这个解释正确, E060.2 的 catastrophic 程度 (stable_intent 跌到 2-5%) 还是超出预期, 说明**还有其他 confound** — 最可能就是 log 74 找到的 3-box port bug.

## 视觉 keyframe (跳过, 数值已确认全程趴地)

按强制流程 (log 71 §5), pelvis_mean_intent 0.245-0.260m + stable_intent 2-5% 已经是"intent 内基本一直在地面"的硬证据, 不需视觉验证. 任何 keyframe 都将显示完全趴地.

## 改动文件

| 文件 | 改动 |
|------|------|
| `examples/config/override/core4d_e060_2_box023.yaml` | 新建 (clone E041c, palm_normal L=R=[+x,0,0]) |
| `examples/config/override/core4d_e060_2_bucket005_s2.yaml` | 新建 (clone E041c, palm_normal L unchanged, R=[+x,0,0]) |
| `workspace/core4d/scripts/train/train_E060_2.sh` | 新建 (parallel GPU 0/1) |
| `workspace/core4d/results/E060/E060_2_*.{npz,mp4}` | 训练输出 |
| `workspace/core4d/log/73_E060_2_case_correct_palm_normal_results.md` | 本文件 |

## 下一步

**E060 的整个 reward ablation 序列暂停**.

按 log 74 全面分析, **必须先验证 box025 sphere baseline 能否复现历史结果**. sphere 验证通过后再决定:
- 修复 3-box port (方案 B: 重新校准 box geometry 让其末端 ≈ wrist+8cm) 或
- 完全弃用 3-box, 退回 sphere (方案 A)

详见 `log/74_box025_3box_regression_and_E060_invalidation.md`.
