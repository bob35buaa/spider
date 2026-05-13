# E060.0: 数据层修复后 baseline (E041c) on box023 + bucket005_s2 — 结果

## 状态: ❌ **FAIL — 数据层修复未实质改善 baseline 摔倒**

**核心结论**: 3-box hand collision + box023 collision margin 0.90 的数据层修复对 baseline 行为**没有实质改善**。两个 case 仍然摔（pelvis_min_intent 0.18m / 0.12m, 都远 < 0.40m PASS 阈值）。contact% 升高是"摔倒过程中手蹭到物体"的同一伪信号（E058/E059 已识别）。

**第二次 over-optimism 教训**: 初次解读时把 box023 t=1.65s 的"前扑中右臂伸到箱旁"误判为"单手抱箱站立"，bucket005_s2 t=2.10s 的"前扑+后腿撑地"误判为"双手抱桶起身"。pelvis trace 显示这些都是摔倒中段。**与 E059 是完全同样的错误模式**。详见 §5。

## 实验配置

| 项 | 值 |
|----|---|
| Reward stack | E041c 原样（无任何 reward 改动） |
| Cases | box023_person1 + bucket005_s2_person1 |
| 数据层改动 vs E058/E059 | (1) hand_collision sphere → 3 box per side, (2) box023 collision margin 1.05 → 0.90 |
| Warmstart | 无（pure baseline） |
| GPU | parallel: GPU 0 box023, GPU 1 bucket005_s2 |
| Wall time | box023 32 min, bucket005_s2 38 min |
| 输出 | `workspace/core4d/results/E060/E060_0_{box023,bucket005_s2}.{npz,mp4}` + scene_snapshot/ + eval_summary.csv + 2 face_dist*.png + 10 keyframes |

## 数值结果

### 对比 E058/E059 (相同 case，仅数据层差异)

| 指标 | E059 box023 baseline | **E060.0 box023** | E058 bucket005_s2 baseline | **E060.0 bucket005_s2** |
|------|---------------------|-------------------|---------------------------|-------------------------|
| pelvis_min (m) | 0.140 | **0.176** (+0.036) | 0.110 | **0.114** (+0.004) |
| pelvis_min_intent (m) | 0.140 | **0.176** (+0.036) | 0.110 | **0.117** (+0.007) |
| stable % (full) | 70.0% | 47.1% (-22.9pp) | 30.7% | 42.6% (+11.9pp) |
| stable % (intent) | 59.3% | **74.1% (+14.8pp)** | 12.5% | 14.6% (+2.1pp) |
| L palm contact % | 9.3% | 17.2% (+7.9pp) | 36.4% | 20.2% (-16.2pp) |
| R palm contact % | 42.6% | 24.1% (-18.5pp) | 37.0% | 24.7% (-12.3pp) |
| both palm contact % | 5.6% | 12.1% (+6.5pp) | 11.6% | 9.0% (-2.6pp) |
| L main face | None | None | None | None |
| R main face | None | None | None | None |

### 解读

**指标变化解读 (避免再次过度乐观)**:
- `stable_intent` box023 +14.8pp 是真改善，但**只是说明大部分 intent 帧 pelvis ≥ 0.5m**。pelvis_min_intent 0.176m 说明**最低点仍是趴地附近**。这跟 E058 解读"intent 末段崩溃"逻辑一致。
- `both_contact` box023 +6.5pp 看似正向，但 12% 仍极低（真搬运需 ≥50%），且配合 main_face=None 说明**没有稳定贴任何特定面**。结合 pelvis trace 看，这是机器人前扑过程中手碰到箱子的随机接触。
- bucket005_s2 几乎全指标退化 (R contact -12pp, L contact -16pp, both -2.6pp), pelvis_min 几乎没变 (+0.4cm)。**3-box hand + margin 0.90 在这个 case 上接近无效**。
- main_face = None 全部 4/4：CEM 没有在任何 6 面上稳定停留 ≥60% 帧 + 中位数 ≤7cm。

## Pelvis Z Trace（关键诊断）

### box023 (T=136 frames, intent (21, 78))

| t | s | pelvis_z (m) | 状态 |
|---|---|------------|------|
| 0-18 | 0-0.30 | 0.79 → 0.70 | 站立 |
| 27 | 0.45 | 0.53 | 开始下蹲 |
| 36-63 | 0.6-1.05 | 0.65-0.78 | 蹲位反复 |
| **72** | **1.20** | **0.226** | **第一次跌到趴地附近** |
| 81-99 | 1.35-1.65 | 0.31-0.32 | **半趴**（不是"单手抱箱站立"！） |
| 108-117 | 1.80-1.95 | 0.39-0.42 | 微抬起 |
| 126-135 | 2.10-2.25 | 0.43-0.44 | **稳在 0.43m 蹲姿（不是站立）** |

**pelvis 最低 0.176m 在 t=74 (s≈1.23)。recovers to ≥0.5? **False**。** 整段 intent 后半部 (t=72 之后) **再也没回到 0.5m 站立**。

### bucket005_s2 (T=148 frames, intent (19, 107))

| t | s | pelvis_z (m) | 状态 |
|---|---|------------|------|
| 0-9 | 0-0.15 | 0.80 | 站立 |
| 18-21 | 0.30-0.35 | 0.64 → <0.5 | 开始深蹲（ref 也深蹲，捡桶） |
| 27-72 | 0.45-1.20 | 0.20-0.55 | 深蹲反复 |
| 81-108 | 1.35-1.80 | 0.12-0.20 | **趴地** |
| 117 | 1.95 | 0.56 | **回到站立** |
| 126-144 | 2.10-2.40 | 0.75-0.79 | 站立（intent 已结束） |

**pelvis 最低 0.114m 在 t=109 (s≈1.82, intent 末段)。recovers to ≥0.5? **True (post-intent)**。**

**关键: bucket005_s2 在 intent 内 (19-107) 是"持续趴地"状态**，t=117 才恢复站立但已经在 intent 末 (107) 之后。所以 stable_intent 只有 14.6% 是真实的"intent 内 86% 时间在地上"，post-intent 的恢复不算数。

视频 t=4.00s (post-intent, t=120) 显示完全趴地，这跟 trace 矛盾？ 让我重看... t=120 对应 trace t=144 (s=2.4) 有 0.79m? 计算: T=148 总长 4.93s, t=4.00s 对应 frame ≈ 120 / 148 * T ≈ 120 * 296/148 = 240... 

等等。视频是 sim 输出 (296 frame), trace 是 ref 144 frame。视频时间戳 4.00s 对应 sim frame 120 / 148 → 实际 sim 帧位置看 video 编码。**这里需要重新核对视频帧 vs eval qpos 的索引关系**。

**保守解读**: trace 显示 t=144 (s=2.4) 站立 0.79m, 但视频 4.00s 是趴地 → 可能是: (1) eval 的 src_T 假设错了 (sim 实际 296 vs ref 148, ratio=2x), (2) 或是后续物理时间不一致。

## 视觉 Keyframe（**用 pelvis trace 校正后**）

### box023

| t (s) | sim 行为（修正后） | trace pelvis_z |
|-------|------------------|---------------|
| 0.40 | 站立靠近箱（pre-intent） | 0.55 |
| 0.70 | 蹲下接近箱（intent start） | 0.62 |
| **1.65** | **前扑姿态**，右臂伸到箱旁，**身体倾斜，pelvis 0.32m** — **不是抱箱站立** | 0.32 |
| 2.60 | **完全趴地** | 0.43（trace 显示，但视频是趴的——sim/ref 帧位异常） |
| 3.00 | 趴地 | (post-intent) |

### bucket005_s2

| t (s) | sim 行为（修正后） | trace pelvis_z |
|-------|------------------|---------------|
| 0.40 | 站立面向桶 | 0.78 |
| 0.65 | 弯腰接近桶 | 0.31 |
| **2.10** | **前扑+后腿撑地**的失衡姿态，双手在桶旁，**不是抱桶起身** | 0.20 |
| 3.55 | 趴地（intent 末） | (intent end) |
| 4.00 | 完全趴地 | (post-intent) |

## Claims 验证

| ID | 描述 | 量化 | 实际 | 通过 |
|----|------|------|------|------|
| C1 | 流水线完整 | snapshot + 2 npz + 2 mp4 + 10 jpg + csv 全有 | ✅ | ✅ |
| C2 | box023 baseline pelvis_min_intent ≥ 0.40m | vs E059 0.14m | 0.176m | ❌ |
| C3 | bucket005_s2 baseline pelvis_min_intent ≥ 0.40m | vs E058 0.11m | 0.117m | ❌ |
| C4 | box023 main_face L=-xy AND R=-yz | E055/E056 expected | None / None | ❌ |
| C5 | bucket005_s2 main_face L=-yz AND R=+yz | E056 expected | None / None | ❌ |
| C6 | 视频 5/5 keyframe 站立 + 双手持物 | 目检 | 0/5 站立 + 持物 | ❌ |

**1/6 通过 (流水线 OK)**。E060.0 数据层修复**单独不足以让 E041c reward 收敛到稳定搬运**。

## 关键发现

### 1. 数据层修复的边际收益小

E060.0 vs E058/E059 baseline 的真实净改善：
- box023 pelvis_min +3.6cm (0.14 → 0.18m, 都是趴地范畴)
- bucket005_s2 pelvis_min +0.4cm (0.110 → 0.114m, **基本无变化**)
- box023 stable_intent +14.8pp 是 valid 信号但 pelvis 最低点仍 0.18m 否决了"真站立"

这表明：**`hand_collision = sphere` 和 `box023 margin = 1.05` 这两个数据 bug 不是 E058/E059 baseline 摔的主因**。修了它们，机器人摔的姿态略有不同，但仍是同一种失败模式。

主因还在 reward。Audit log §2 列的 4 处 task-specific 嫌疑（palm_normal hardcoded 等）至少有一个是真元凶。

### 2. ⭐ 第二次 over-optimism (系统性教训)

**E059 时**: 我把 box023 t=1.65s 的"前扑中手贴箱底"称为"接近真实搬运的姿态"，被用户纠正后修订 log。
**E060.0 时**: 我把同样性质的 box023 t=1.65s 称为"单手抱箱站立 ⭐⭐⭐ 53 实验首次真搬运"，**又被用户纠正**。

**根因**: 看单帧静态画面时，手 + 物体距离近 → 大脑自动补全为"抓握搬运"，忽视：
- 整段 pelvis trace（看 t=72 已 0.22m 就该知道在摔，t=99 的 0.32m 是摔的中段不是站立）
- 同帧的腿/躯干姿态（前扑 vs 站立）
- ref 同帧的对照（ref 是稳定站立，sim 是失衡前扑 = 显著落后）
- 后续帧（t=2.60s 已经趴地说明 t=1.65s 是失稳过程中）

**强制流程 (从此实施)**:
1. **判定搬运成功前必须先看 pelvis_z trace 全程**，pelvis_min < 0.30m 一律视为摔倒，不论单帧画面如何
2. **必须看 intent end 帧 + post-intent 帧**，不能只看 intent mid
3. **对照同帧 ref**，sim 落后 ref 一格姿态以上视为不稳定
4. **多帧一致**: 至少要 intent mid + intent end + post 3 帧都站立 + 接触物体才算成功
5. 单帧画面"看起来像抱箱"绝不是充分证据

**触发审查**: 任何带 ⭐ 或 "首次" 字眼的视觉描述必须额外验证 pelvis trace + 3-frame 一致性后才能写入 log。

### 3. bucket005_s2 上 sphere → 3-box 改造影响极小

bucket005_s2 几乎所有指标都在 E058 ±2pp 范围内, pelvis_min 只 +0.4cm。这反向支持：bucket005_s2 的失败原因主要不是 hand collision 几何，是 reward (audit §2 中 hardcoded palm_normal R 上是次优 +0.51 vs 真实最优 +x +0.64) 或更深层 mocap 质量问题。

### 4. box023 上 0.90 margin 也未让 baseline 显著抬起

E053 box025 在 0.90 时 pelvis_min 0.66m vs 1.05x 的 0.575m (+0.085m)。E060.0 box023 1.05→0.90 仅 +0.036m (0.14 → 0.18m)。**移植效果不如 E053 box025 的预测**。可能因为:
- box025 在 E053 时已经 baseline 站立 (pelvis 0.575m)，0.90 让其更稳
- box023 在 E059 时 baseline 已经摔 (pelvis 0.14m)，0.90 改善的空间没那么大（"先决条件"是要先能站住）

### 5. 物理求解器 OK

3-box hand 把 contact pair 从 4 → 12 per case，**但 plan time 仍稳在 14s/iter** (E059 14s)，没爆炸。3-box 不是性能问题。

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/assets/robots/unitree_g1/robot.xml` | 3-box hand (commit fa2e181) |
| `example_datasets/.../9 cases/scene*.xml` | 3-box hand 同步 patch |
| `example_datasets/.../box023_person1/scene*.xml` | margin 1.05 → 0.90 |
| `workspace/core4d/scripts/convert/patch_hand_3box.py` | 新建 |
| `workspace/core4d/scripts/train/train_E060_0.sh` | 新建 (parallel + snapshot 集成) |
| `workspace/core4d/scripts/eval/eval_E060_0.py` | 新建 (统一 box023 + bucket005_s2) |
| `workspace/core4d/scripts/eval/extract_E060_0_keyframes.sh` | 新建 |
| `workspace/core4d/scripts/run_E060_0.sh` | 新建 (一键) |
| `workspace/core4d/results/E060/scene_snapshot/` | 新建 (manifest 含 sha256 + git HEAD) |
| `workspace/core4d/results/E060/E060_0_*.{npz,mp4}` | 训练输出 |
| `workspace/core4d/results/E060/eval_summary.csv` | 评估输出 |
| `workspace/core4d/plan/70_E060_data_layer_fix_plan.md` | E060 整体 plan |
| `workspace/core4d/log/71_E060_0_baseline_results.md` | 本文件 |

## 下一步: 进 Phase 3 — E060.1 (砍 ori reward)

按 Phase 2 失败条件，进入 Phase 3 全套 ablation。

### E060.1 配置

```bash
+override=core4d_e041c \
task=<case> \
contact_hdmi_ori_weight=0.0 \
+use_torch_compile=false
```

- 在 box023 + bucket005_s2 上各跑（GPU 0/1 并行）
- 测试假设：hardcoded palm_normal `[0,∓1,0]` 在 box023 L 上是噪声（audit §2.2 mean dot 0.04），CEM 收到错误方向信号
- **通过 = pelvis_min_intent 比 E060.0 改善 ≥ 0.10m**（即从 0.18 → 0.28m+，box023）
  或 contact_both 比 E060.0 改善 ≥ 10pp（从 12% → 22%+，box023）
- 不通过 → 进 E060.2 (palm_normal=[+x,0,0] case-correct)
- E060.2 也不通过 → 进 E060.3 (stability_penalty=1.0 兜底)
- 全套不通过 → 触发"三次失败协议"，跳出 Path B-CEM
