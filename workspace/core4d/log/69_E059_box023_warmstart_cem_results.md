# E059: box023 + Snap Warmstart CEM (Path D 区分实验) — 结果

## 状态: ❌ **失败 — Baseline 摔倒 + Warm 摔倒**, Claims 1.5/6

**核心结论 (写在最前避免误读)**: E059 的真正发现是 **E041c reward stack 在 box023 上 baseline 自己就站不住** (pelvis_min=0.14m, 站立应 ≥0.7m), warmstart 加上去**也摔** (pelvis_min=0.10m, 倒得更深)。

数值上的"L contact +35pp" 和 "stable_intent +20pp" 是**两种摔法之间的比较**:
- baseline 摔后 L 手飘走 → contact 9% (低是因为摔)
- warm 摔后 L 手被 warmstart 钉在 box 底面 → contact 44% (高是因为 warmstart 锁住, 不是搬运成功)
- 视频 t=1.65s "看起来像搬运的姿态" 实际是机器人**前倾摔倒过程**的一个瞬间, 1 秒后 (t=2.60s) 就完全趴在地上

**与 E058 (bucket005_s2) 对照**:
- E058: baseline 摔 + warm 摔 + warm 物理爆炸 (plan time 7x)
- E059: baseline 摔 + warm 摔 + warm 物理稳定 (plan time 一致)
- **共同点: 双方都摔**。E041c reward stack 在两个 case 上都不能产生稳定搬运。

**真正的 D 路径结论 = 情景 3 (双方都摔)**, 不是中间情景 1+2。warmstart 不能挽救 reward stack 本身的不稳定。下一步必须先在 baseline 上加 stability 机制 (E034 引入但 E041c 关闭的 stability_penalty), 让 baseline 不摔, 再讨论 warmstart 的边际贡献。

## 实验配置

| 项 | 值 |
|---|---|
| Case | `box023_person1` (E055 已做 snap, grasp_type=垂直) |
| Reward stack | E041c (与 E058 完全相同) |
| Warmstart | E055 `warmstart_qpos.npz` (intent 21-78 = 58f, snap mask 116/322 帧) |
| GPU | parallel: GPU 0 baseline, GPU 1 warm |
| Wall time | baseline **32 min**, warm **33 min** ⭐ (vs E058 warm 4h, **没爆炸**) |
| 输出 | `workspace/core4d/results/E059/E059_{baseline,warm}.{npz,mp4}` + outdir + eval_summary + face_dist_E059.png + 10 keyframes |

## 数值结果 (eval_summary.csv) — **解读重点: 大部分"正信号"是两种摔法的对比, 不是搬运成功的证据**

| 指标 | baseline | warm | Δ | 真实解读 |
|------|----------|------|---|---------|
| pelvis_min (m, 整段) | 0.140 | 0.079 | -0.06 | **两个都摔** (站立应 ≥ 0.7m), warm 倒得更深 |
| pelvis_min_intent (m) | 0.140 | 0.101 | -0.04 | 同上, intent 内也是倒地状态 |
| stable % (full episode) | 70.0% | 45.6% | -24.4pp | warm 整段稳定性反而 **-24pp** (intent 后崩溃更厉害) |
| stable % (intent only) | 59.3% | 79.6% | +20.4pp | intent 内**站立时间分数**多了, 但**最低点**还是倒地 — 两种"半摔"姿态对比 |
| L palm contact % | 9.3% | 44.4% | +35.2pp | baseline 摔后 L 飘走 (9%), warm 摔但 L 被 warmstart 钉住 (44%) — **不是搬运, 是固定姿态** |
| R palm contact % | 42.6% | 22.2% | -20.4pp | warm R 反而下降 (CEM 牺牲 R 自由度试图稳身体, 但仍没稳住) |
| both palm contact % | 5.6% | 3.7% | -1.9pp | 真正"双手都贴 box" 的帧率 < 6% — **没有真实搬运** |
| L main face | None | -xy | — | warm L 接触面命中 E055/E056 expected, 但因为机器人摔了, 这个"贴" 是被动的 |
| R main face | None | None | — | R 没稳定贴任何面 |

**关键诊断**: `pelvis_min` 是检验"是否摔倒"的硬指标。 0.07-0.14m 都是地面附近 (G1 站立时 pelvis ≈ 0.7m, 蹲下 ≈ 0.4m)。两个 run 都摔了, 后续所有"contact / face / stable_intent" 的差异本质是**两种倒地姿态的差异**, 而不是"搬运 vs 没搬运"的差异。

### 与 E058 (bucket005_s2) 对比

| 指标 | E058 baseline | E058 warm | E059 baseline | E059 warm |
|------|---------------|-----------|---------------|-----------|
| pelvis_min (整段) | 0.110 | 0.109 | **0.140** | 0.079 |
| stable_intent% | 12.5% | 33.0% | **59.3%** | **79.6%** |
| L main face | None | None | None | **-xy** ✅ |
| L contact% | 36.4% | 35.2% | 9.3% | **44.4%** |
| warm wall time | — | 4h (爆炸) | — | **33min** (正常) |

**E059 baseline 比 E058 baseline 显著更稳** — 验证了 E058 失败假设的 (a) **case 适配问题**: E041c reward 在 box023 上工作得多, bucket005_s2 不行。**情景介于 1 和 2 之间**: warm 改善 intent 稳定 + L 几何, 但破坏 intent 后稳定。

## Claims 验证

| ID | 描述 | 量化标准 | 实际 | 通过 |
|---|---|---|---|---|
| C1 | 流水线 OK (3 步) | npz + mp4 + 10 jpg + csv 全有 | ✅ | ✅ |
| C2 | baseline pelvis_min_intent ≥ 0.40m | 区分情景 1+2 vs 3 | 0.140 m (倒了) | ❌ |
| C3 | warm pelvis_min_intent ≥ 0.40m | 区分情景 1 vs 2 | 0.101 m (倒得更深) | ❌ |
| C4 | warm contact (both) ≥ baseline +5pp | warmstart 真有正贡献 | both -1.9pp, **L +35pp 单独看 ⭐** | ❌ both, ⭐ L |
| C5 | warm main_face = (-xy, -yz) | warmstart 几何 in CEM 保留 | L=**-xy** ✅, R=None ❌ | **0.5 ✅** |
| C6 | 视频 warm ≥ 3/5 视觉优势 | A/B 5 帧目检 | **3-4/5** (见下) | ✅ |

**1.5/6 严格通过, 但内容信号显著正向**。

## C6 视觉详情 (5 keyframe × 2 traj = 10 帧目检) — **修正前次过度乐观解读**

| t | baseline | warm | 真实评价 |
|---|---|---|---|
| 0.40s (pre) | 弯腰对箱 | 弯腰对箱 | 都没问题, 还没开始摔 |
| 0.70s (intent start) | 弯腰单手碰箱 | 弯腰双手在箱两侧 | warm 双手位置由 warmstart 强制, **不证明 CEM 学到搬运** |
| 1.65s (intent mid) | **侧倒, 物体压在身上** ❌ | **前倾, 双手在 box, 脚还在地** | **两个都在摔倒过程中**。warm 是"前倾摔" (mid-fall), baseline 是"侧倒摔" (already-fallen)。warm 1 秒后就完全趴地 (见 t=2.60s) |
| 2.60s (intent end) | 跪姿被箱压 | **完全趴在地上**, 单手延伸到箱顶 | warm 比 baseline 倒得更彻底 |
| 3.00s (post) | 趴下 | 趴下 | 都在地上 |

**真实评价**: 5 帧里 5 帧 baseline 摔, 5 帧里 5 帧 warm 也摔。差别仅在**摔的姿态**:
- baseline 倾向"侧倒被压" (lateral fall)
- warm 倾向"前倾摔倒后继续保持手部接触" (forward fall + sticky hand)

这跟 **E048 box023 视频复查** 结论 ("box023摔倒, ObjPos=14cm 是假象") 完全一致。**E059 没改变这个结论**, 只是把 warmstart 加进来后, 摔法变了一点 (warm 因为 ref 锁住手, 摔的瞬间手仍贴 box, 看起来更像"搬"过)。

**之前我说的 "53 个 CEM 实验中第一次接近真实搬运" 是过度乐观**。t=1.65s 那一帧 isolated 看像搬运, 但放在整个轨迹里看是 mid-fall 瞬间, **不构成搬运成功**。

## 关键发现 (修正版)

### 1. **真正的发现: E041c reward stack 在 box023 + bucket005_s2 上都站不住** ❌

**对照**:
- E058 bucket005_s2: baseline pelvis_min = 0.110m, warm = 0.109m → 都摔
- **E059 box023: baseline pelvis_min = 0.140m, warm = 0.079m → 都摔**

baseline 在两个 case 上都不能维持站立, 说明 **E041c reward stack 缺乏 stability 约束**。E034 引入过 `stability_penalty`, 但 E041c (在 E036/E041 系列调参时) 把它关掉了 (`stability_penalty_scale: 0.0`)。

这跟 **E048 视频复查结论** 完全吻合: "box023摔倒 (ObjPos=14cm 是假象)", "desk005 丢下桌子走" — 当时已经发现 E041c stack 在多个 case 上视觉摔倒, E058+E059 只是用 pelvis_min 把这个事实数值化了。

### 2. warmstart 不能挽救 reward 不稳定

E057 snap 在 box023 上验证了几何可行 (5 cm 内, 关节限位 100%)。但**几何可行 ≠ 物理可行**:
- snap 只改 ref qpos 和 ctrl_ref, 不改 reward stack
- CEM 还是用同一个 E041c reward (无 stability term) 来评分
- 机器人在搬箱过程中倾向前倾失稳, 没有 reward 拉它回来 → 摔
- warmstart 把双手锁在 box 上, 反而**让前倾摔倒过程中手仍贴 box**, 看起来像搬运但物理上仍是摔

### 3. Path B-CEM 失败原因 = E041c 失败原因 (不是 warmstart 设计问题)

**之前我的 D 路径 4 个情景**:
- 情景 1: baseline 站 + warm 站, contact 提升 → warmstart 设计 OK
- 情景 2: baseline 站 + warm 摔 → warmstart 破坏 baseline
- 情景 3: 两个都摔 → reward stack 问题

**实际答案是情景 3**。E058 是情景 3, **E059 也是情景 3** (我之前误读为 1+2 之间)。两个 case 的 baseline 都摔, 说明问题不在 warmstart 而在 reward stack 缺 stability。

### 4. R contact 不对称失败仍然存在

box023 expected R 在 -yz 远侧 (高位), CEM 难以同时保持 L 锁住 -xy 底面 + 维持 R 贴 -yz + 身体平衡。warm R contact 22% 比 baseline 42% 显著降, 说明 CEM 在 warmstart 锁住 L 后, 把 R 的"自由度"用来稳身体了 (但还是没稳住)。

### 3. Intent 后崩溃比 baseline 更严重

warm pelvis_min 0.079m vs baseline 0.140m: **warm 在 intent 内更稳 (+20pp), 但 intent 后倒得更深**。可能因为:
- warmstart blend 区间 (intent 边界 10 帧) 之后 ref 突然回到 mocap qpos, 姿态有跳变, CEM 需要重新规划 → 不稳
- intent 末端 release 动作的 ref 是 mocap retarget, 物理可能不可行

### 4. plan time 没爆炸 — box023 物理稳定

E058 warm: plan time 14s → 99s @ sim_step 14 (intent 内, 物理求解器爆炸)。
**E059 warm: 全程 14-15s 一致**。box023 collision box 较小 (17.9 × 18.3 × 20.6 cm half) + intent 短 (58f vs 88f) + 垂直握姿 (一手在底面只有 1 个接触点, vs 对侧握有 2 个) → 接触约束数量更少, 求解器轻松。

### 5. L main_face = -xy 命中 — 边际信号, 不构成"突破"

之前我把这点写成"历史性突破", 但客观上:
- L 命中 -xy 是因为 warmstart **强制**把 L 投到 box 底面, CEM 32 iter 没把它推开
- 但机器人摔倒了, 所以"L 在 box 底面 67% 帧"实际是"机器人摔倒过程中手被锁在底面"
- 没构成搬运成功, 只证明 ref 替换能影响 CEM 解的几何

边际意义: warmstart hook 设计正确 (能传递几何到 CEM), 没有什么实现 bug。**但仅此而已**。

## 教训

1. **case 选择 > reward 调参**: 同一个 reward stack (E041c), bucket005_s2 完全摔倒, box023 接近真实搬运。E057-E058 在 bucket005_s2 上 5+ 小时调试不如**先验**: box023 是 E041c 训练时见过的 case 之一 (E048 也用过), bucket005_s2 是新引入的, 没磨合。

2. **L 单边成功比 both 更易达到**: contact "both" 5/6 < 4% 难度极高, 但 "L 单边" 44% 可达。Path B-CEM 在多接触面对侧握上的难度可能本质上比单接触面垂直握高得多。

3. **post-intent 稳定是新瓶颈**: warm 在 intent 内 +20pp stable, 但 intent 后立即反过来 -25pp。这是过去 CEM 实验里没特别关注的"释放阶段"问题, 需要 E060 的 intent boundary blend 改进。

4. **buffered output 是隐形 trap**: E058+E059 第一次跑 Python stdout block-buffered, 1.5h 看不到任何进度, 看着像 "stuck"。`PYTHONUNBUFFERED=1 python -u` 后实时进度。**所有未来训练脚本都应加这两个**。

## 改动文件

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/train/train_E059.sh` | 新建 (克隆 E058, +unbuffered Python) |
| `workspace/core4d/scripts/eval/eval_E059.py` | 新建 (克隆 E058, 改 case=box023, INTENT=(21,78), expected=(-xy,-yz)) |
| `workspace/core4d/scripts/eval/extract_E059_keyframes.sh` | 新建 (5 时间戳: 0.40/0.70/1.65/2.60/3.00) |
| `workspace/core4d/scripts/run_E059.sh` | 新建 (一键) |
| `workspace/core4d/results/E059/` | 新生成 (2 npz + 2 mp4 + 10 jpg + csv + face dist png + 2 outdir) |
| `workspace/core4d/plan/69_E059_*.md` | 新建 |
| `workspace/core4d/log/69_E059_*.md` | 本文件 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | E059 行 + scripts |

**spider/ 不动** (E058 已加 hook + output_dir 修)。

## 下一步路径分析 (E060+) — 修正后

E059 验证了 warmstart 不能挽救 reward 不稳定, **必须先把 baseline 修稳**才能讨论后续。

### A. **(必须先做) 修 baseline reward stack — 加回 stability_penalty**

**问题**: E041c `stability_penalty_scale: 0.0` (E041 系列调参时关掉了), 没有任何机制阻止机器人前倾失稳 → box023 / bucket005_s2 baseline 都摔。

**方案**:
- 改 `core4d_e041c.yaml` 的衍生 `core4d_e060_stab.yaml`: `stability_penalty_scale: 1.0`, 阈值 0.55 (E034 默认)
- 在 box023 上跑 baseline (无 warmstart) 单 GPU
- 验证 pelvis_min_intent 能否从 0.14m 抬到 ≥0.40m (蹲下姿态合格), 理想到 ≥0.50m
- 如果 baseline 站住, 再加 warmstart 测增量 — 这才是真正"区分 reward vs warmstart 贡献"的实验

**预算**: 1 baseline (~30 min) + 1 warm (~30 min) = 1 GPU 1 小时

### B. (依赖 A) 边界 blend / R face anchor

A 通过后, intent 后 stable -24pp 才值得修。R contact 不对称同理 — baseline 站不住时 R 不对称是次要问题。

### C. 暂不推广其他 case

box023/bucket005_s2 都摔, 没必要在 bucket007/desk021/bucket001 上重复同样错误。等 A 通过后再推广。

### D. 重新审视 Path B 整体方案

如果 A 之后 baseline 站住但加 warmstart **仍**不能让 contact_both > 50%, 说明 CEM + warmstart 这条 Path 上限有限, 应该早进 RL pipeline (Holosoma / HDMI 训练框架, 有 PPO + GAN reward + interaction term)。

## **推荐: E060 = 在 box023 上加 stability_penalty, 单 GPU 跑 baseline → 看能否站住**, 是这条路的 go/no-go 节点。
