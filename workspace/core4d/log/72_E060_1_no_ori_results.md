# E060.1: contact_hdmi_ori_weight=0 ablation on box023 + bucket005_s2 — 结果

## 状态: ❌ **FAIL (Phase 3 step 1 不通过严格 PASS criteria)**, 但**反直觉 case-divergent 发现**

**核心结论**: 砍掉 ori reward 的影响**两个 case 完全相反**，不是单向的"关掉就好"或"关掉就坏":
- **bucket005_s2**: 显著改善 (+39pp stable_intent, +17cm pelvis_mean_intent, intent mid 视觉从"接近平躺"→"深蹲弓步")，但 Δpelvis_min_intent 只 +9.3cm 差严格 PASS 阈值 7mm
- **box023**: 显著退化 (-21pp stable_intent, -3cm pelvis_min_intent), 但奇怪地 **recovers=True**（E060.0 是 False）

两个 case 都没真搬运（4/4 keyframe 摔），只是摔的姿态/时机不同。

**这部分纠正了 audit log §2.2 的简单预测** ("hardcoded palm normal 在 box023 L 是噪声 → 关掉应改善")。实际上 box023 关掉反而退化 — 即使数学上对 L 是噪声/对 R 是次优，**有方向 prior 比没有更好**，CEM 没了任何朝向引导反而更乱。详见 §4。

## 实验配置

| 项 | 值 |
|----|---|
| Reward stack | E041c with `contact_hdmi_ori_weight=0.0` (CLI override) |
| Cases | box023_person1 + bucket005_s2_person1 |
| 数据层 | 同 E060.0 (3-box hand + box023 margin 0.90), 复用 commit fa2e181 |
| GPU | parallel: GPU 0 box023, GPU 1 bucket005_s2 |
| Wall time | box023 ~33min, bucket005_s2 ~38min |
| 输出 | `workspace/core4d/results/E060/E060_1_*.{npz,mp4}` + 10 keyframes |

## 数值结果对比 (E060.0 → E060.1)

| 指标 | box023 E060.0 | box023 E060.1 | Δ box023 | bucket005_s2 E060.0 | bucket005_s2 E060.1 | Δ bucket005_s2 |
|------|---------------|---------------|----------|---------------------|---------------------|----------------|
| pelvis_min (m) | 0.176 | 0.105 | **-0.071** | 0.114 | **0.210** | +0.096 |
| pelvis_min_intent (m) | 0.176 | 0.146 | **-0.030** | 0.117 | **0.210** | **+0.093** |
| pelvis_mean_intent (m) | 0.577 | 0.469 | **-0.108** | 0.307 | **0.482** | **+0.175** |
| stable_intent % | 74.1 | 53.4 | **-20.7pp** | 14.6 | **53.9** | **+39.3pp** |
| stable_full % | 47.1 | 72.8 | +25.7pp | 42.6 | 71.6 | +29.0pp |
| L palm contact % | 17.2 | 10.3 | -6.9pp | 20.2 | 34.8 | +14.6pp |
| R palm contact % | 24.1 | 29.3 | +5.2pp | 24.7 | 27.0 | +2.3pp |
| both palm contact % | 12.1 | 6.9 | -5.2pp | 9.0 | 10.1 | +1.1pp |
| recovers (pz≥0.5 after argmin) | False | **True** | ⚠️ | True | True | — |
| argmin time | t=74 (s=1.23) | t=83 (s=1.38) | 后移 | t=109 (s=1.82) | **t=33 (s=0.55)** | **大幅前移** |
| L main face | None | None | — | None | None | — |
| R main face | None | None | — | None | None | — |

### 解读重点

**bucket005_s2** (改善方向):
- pelvis_mean_intent +17.5cm 是真信号，意味着 intent 内**整体姿态更接近站立**
- argmin 移到 intent 极早 (t=33, s=0.55) — 早期蹲下去碰桶时 dip，之后大部分时间在 0.4-0.5m 范围
- L contact +14.6pp 是真改善（CEM 能更稳定地把左手放到桶上）
- 仍 fail 严格 pelvis_min_intent ≥ 0.40m (实测 0.21m)，但只差 19cm

**box023** (退化方向):
- pelvis_min_intent -3cm 但 pelvis_mean_intent -10.8cm 更显著
- stable_intent -21pp 表明大部分 intent 帧都跌出站立区
- L contact -6.9pp, both -5.2pp 全面退化
- argmin 后移 + recovers=True 是 anomaly: 摔得晚但能爬起来。但 recover 帧是 post-intent（不计 stable_intent）

## 视觉 keyframe (4 关键帧 + pelvis_z trace 校正)

按修订后的"3-frame 一致性"规则（强制流程, 见 log 71 §5）:

### box023

| t (s) | sim 行为 | trace pelvis_z | 与 E060.0 同帧对比 |
|-------|---------|---------------|-------------------|
| 1.65 (intent mid) | 前扑+后腿后蹬, 右臂伸到箱旁 | ~0.32 | E060.0 同帧 0.32 — **基本一致** |
| 2.60 (intent end) | **完全趴地**, 右臂伸出碰箱 | (post-intent ~0.43) | E060.0 同帧也趴 — 一致 |

→ box023 关 ori_weight 视觉上跟 E060.0 没区别，都是 "intent mid 前扑 → intent end 趴地"。数值 stable_intent -21pp 反映为 intent 内更多帧低于 0.5m，但视觉上 endpoint 一致。

### bucket005_s2

| t (s) | sim 行为 | trace pelvis_z | 与 E060.0 同帧对比 |
|-------|---------|---------------|-------------------|
| 2.10 (intent mid) | **深蹲弓步**, 右臂伸到桶旁, 左腿后伸 | ~0.45 | E060.0 同帧 0.20 (接近平躺) — **明显改善** |
| 3.55 (intent end) | 完全趴地 (右臂前伸碰桶, 左腿伸出) | (intent end) | E060.0 同帧也趴 — endpoint 一致 |

→ bucket005_s2 关 ori_weight 视觉上 intent mid **真改善**（深蹲弓步 vs 接近平躺），但 intent end 仍是趴地。**改善的"窗口"在 intent 中段，end 仍崩**。

**严格判别**: 4/4 关键帧 sim 都是失稳/摔倒姿态，0/4 是站立持物。**没有真搬运**。

## Claims 验证 (E060.1 PASS criteria from plan §3)

| ID | 描述 | 实际 | 通过 |
|----|------|------|------|
| C1 | box023: pelvis_min_intent 改善 ≥ 0.10m vs E060.0 | -0.030m (退化) | ❌ |
| C2 | bucket005_s2: pelvis_min_intent 改善 ≥ 0.10m vs E060.0 | +0.093m (差 7mm) | ❌ |
| C3 | box023: contact_both 改善 ≥ 10pp | -5.2pp | ❌ |
| C4 | bucket005_s2: contact_both 改善 ≥ 10pp | +1.1pp | ❌ |

**0/4 严格通过**。但 C2 极接近 + bucket005_s2 全面 stable% 改善，**partial signal 显著**。

## 关键发现

### 1. ⭐ 反直觉: ori reward 的作用 case-divergent

Audit log §2.2 数值预测 ("box023 L 是噪声 (mean dot 0.04) → 关掉应改善"，"bucket005_s2 L 是对的 +0.79 → 关掉可能无影响或退化") **被实验数据反转**:

| Case | Audit 预测 (基于 hardcoded palm normal 与 wrist→obj 方向的点积) | 实际结果 |
|------|--------------------------------------------------------|---------|
| box023 | 噪声 → 关掉应改善 | **退化 -21pp stable_intent** |
| bucket005_s2 | 对 (L=+0.79) → 关掉应无影响 | **改善 +39pp stable_intent** |

**深一层解释 (新假设)**: hardcoded palm normal 起作用的不是"指向物体方向" (audit §2.2 量的)，而是 **"给 CEM 一个 wrist 朝向的 prior，约束探索空间"**:
- box023: wrist 局部 -y 方向虽然不指向物体中心 (mean dot 0.04)，但 CEM 用它作为 prior 来约束 wrist 不乱转。关掉 → CEM 没有 wrist 朝向先验 → wrist 自由旋转 → 姿态更乱
- bucket005_s2: hardcoded -y 方向虽然指向物体 (+0.79)，但跟实际接触几何 (侧地物体 + quat 90°) **冲突**。关掉 → CEM 自由探索 wrist → 找到更适合此 case 的接触姿态

如果这个假设对，则 case-correct palm normal (audit 提的下一步 E060.2) 应该 **同时帮 box023 和 bucket005_s2** — 提供正确方向的 prior 既约束 wrist 又对齐物体几何。这是 E060.2 的核心 claim。

### 2. bucket005_s2 + ori_weight=0 看到的"深蹲弓步"是新姿态

E058/E059/E060.0 都是"intent mid 前扑/平躺"，E060.1 第一次出现 "深蹲弓步" (左腿后伸 + 双手向桶 + 上半身前倾但 pelvis ~0.45m)。这个姿态接近 ref 的"弯腰捡桶"动作，且 pelvis 不至于贴地。

但 intent end 仍崩。说明 **CEM 在 intent 中段能维持深蹲，但放下/转身阶段的 reward 不够 stable**。这是 ref motion intent boundary 设计或 reward post-intent 部分的问题，是次要议题。

### 3. box023 的 recovers=True 是 anomaly

E060.0 box023 argmin 后再也回不到 0.5m，E060.1 box023 argmin 在 t=83，之后 recover 到 0.5m+。但 recover 发生在 post-intent，**不计入 stable_intent**。视觉上 t=2.60 (intent end) 仍是趴地。

可能解释: ori reward 在 box023 上让 CEM 一直试图把 wrist 拉到 -y 方向，即使在摔倒过程中也保持这个尝试 → 永久前扑。关掉后 CEM 在摔倒后能"放手"重新组织姿态 → 能爬起来。但这是无意义的 recovery（intent 已结束）。

### 4. main_face 仍全部 None

E060.0/E060.1 都是 4/4 main_face=None。说明 CEM 不论关不关 ori reward，**都没有稳定贴任何特定 box 面 60% 帧 + 中位数 ≤7cm**。这是**接触行为的根本问题**，不在 ori reward 这一项。

可能在: contact reward 的距离阈值 (`contact_hdmi_threshold=0.15`) 过松? 或 contact gain (5.0) 不够? 或 ref qpos 本身让 wrist 接触面不稳定?

### 5. 物理验证

E060.1 plan time 仍稳在 14s/iter。3-box hand 12 pair 不是性能瓶颈（已在 E060.0 验证）。

## 教训

### 1. 数值消融的预测能力有限

Audit §2.2 用"wrist 各轴 vs wrist→obj 方向的点积"来预测 hardcoded palm normal 的有效性，但这个指标 **没考虑 CEM 优化动力学**。点积低不意味着 reward 项无用 — 它可能作为"约束 wrist 自由度"的隐式 prior。这个洞察是 audit 静态分析时漏掉的。

**强制规则**: 任何 reward ablation 实验必须在**至少 2 case** 上跑 (单 case 结论不可推广)。E060.1 同时跑 box023 + bucket005_s2 抓到 case-divergent 才发现这个隐式 prior 假设。

### 2. 静态分析 + 实验数据反向才能找到真机制

Audit §2.2 数值分析→预测一个方向 + E060.1 实验数据→反向 → 我得到比 audit 更深的"隐式 prior"机制假设。这个假设要 E060.2 验证（case-correct palm normal 既保留 prior 又对齐几何）。

### 3. 部分 over-optimism 风险被强制流程拦下

按 log 71 §5 的"3-frame + ref 对比"强制规则，我看 4 帧前先校验 pelvis_z trace，避免了把 bucket005_s2 t=2.10s "深蹲弓步" 误判为"双手抱桶起身"。trace 显示 pelvis 0.45m → 这是深蹲，不是站立。同样 box023 t=1.65s 是 0.32m → 前扑中段，不是站立。视觉上 bucket005_s2 真有改善（深蹲 vs 平躺），但够不上"搬运成功"标准。

## 改动文件

| 文件 | 改动 |
|------|------|
| `workspace/core4d/scripts/train/train_E060_1.sh` | 新建 (clone train_E060_0.sh + `contact_hdmi_ori_weight=0.0` CLI override) |
| `workspace/core4d/results/E060/E060_1_{box023,bucket005_s2}.{npz,mp4}` | 训练输出 |
| `workspace/core4d/results/E060/E060_1_*_kf*.jpg` | 10 关键帧 |
| `workspace/core4d/results/E060/eval_summary.csv` | E060.1 eval (覆盖了 E060.0 — 应改 eval 脚本支持多 exp_id, 见下) |
| `workspace/core4d/results/E060/face_dist_E060_1_*.png` | E060.1 face distance time-series |
| `workspace/core4d/log/72_E060_1_no_ori_results.md` | 本文件 |

**遗留小问题**: eval_E060_0.py 改了 CASES 跑了 E060.1 npz，把 E060.0 的 eval_summary.csv 覆盖了。E060.0 数值仍可从 log 71 / git history 恢复，下次应在 eval 脚本里支持 `--exp_id` 参数。

## 下一步: E060.2 (case-correct palm normal)

按修订后的假设 (§4.1 "隐式 prior 机制")，case-correct palm normal 应同时帮两个 case。

### E060.2 配置 (audit §2.2 找到的真实最优轴)

box023 上 L/R 真实最优都是 +x:
```yaml
# examples/config/override/core4d_e060_2_box023.yaml
defaults:
  - core4d_e041c
contact_hdmi_palm_normal_left:  [1.0, 0.0, 0.0]   # +x = fingers axis
contact_hdmi_palm_normal_right: [1.0, 0.0, 0.0]
```

bucket005_s2 上 L 已经匹配 (-y +0.79)，但 R 的真实最优是 +x (+0.64 vs hardcoded +y +0.60)。鉴于 +y 跟 +x 差距小，**先只测 box023**，bucket005_s2 留作 fallback:
```yaml
# 仅 box023 跑 E060.2 第一轮
```

### E060.2 PASS criteria

- box023: pelvis_min_intent 比 E060.0 (0.176m) 和 E060.1 (0.146m) 都改善, 目标 ≥ 0.30m
- box023: stable_intent 比 E060.1 (53%) 改善, 目标 ≥ 70%
- 视觉 keyframe (intent mid + end) 站立 + 持物 ≥ 1/2

如果 box023 通过 → 验证"隐式 prior 假设"，单独再测 bucket005_s2 用 R=+x。
如果 box023 不通过 → 进 E060.3 (stability_penalty 兜底)
