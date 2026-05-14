# E067 Results — narrow body partition CATASTROPHIC FAIL + 3-strike synthesis (PAUSE)

## 状态: ❌❌ **CATASTROPHIC FAIL — narrow partition 让 sim 做手倒立 (B1=1.30m, C3=6.9%). 3 ablation 全 FAIL, 触发暂停 #3 + #5. 等用户决策方向.**

**TL;DR**: log 85 假设 "MJWP `local_frame_lower_ids` 12 bodies vs HDMI 6 bodies, mean dilution 是元凶". E067 把 lower=[2,5,7,8,11,13] upper=[17,20,23,24,27,30] (HDMI exact). **Hypothesis 完全反了**: 双变体 B1 ≈ 1.30m (E063 baseline 0.48m, E066 0.53m, **2.7× WORSE**), C3 = 6.9% (baseline 67%, 几乎完全不稳). 视觉: sim t=0.8s 在做**手倒立** (右腿向上指天 1.30m), 没有 lunge 而是更激进的"handstand". 解释: narrow partition 移除了对 hip_yaw/roll + ankle_pitch 等中间 link 的约束, CEM 反而找到 "handstand" 比 lunge 更好的局部最优. 教训 #15: more body tracking constraints HELP because they cover more DoFs, even with dilution.

## 1. 实验配置

| 变体 | partition | task_obj | actuator | yaml |
|------|-----------|---------|----------|------|
| E067-N | HDMI narrow (6+6) | L2 (E062 default) | strong (kp=500) | core4d_e067n_box023.yaml |
| E067-NS | HDMI narrow (6+6) | exp form (E065-D) | soft (kp=20) | core4d_e067ns_box023.yaml |

E067-N = single-variable (only partition changed)
E067-NS = full HDMI clone (partition + actuator + task_obj all aligned)

## 2. 量化结果 (3 round 全表)

| 指标 | baseline | E065-A | E065-D | E066-A | E066-D | **E067-N** | **E067-NS** | 阈值 |
|------|----------|--------|--------|--------|--------|-----------|------------|------|
| **B1** max foot_z [0-2s] (m) | 0.48 | 0.30 | 0.69 | 0.53 | 0.75 | **1.30** ❌❌ | **1.25** ❌❌ | ≤0.10 |
| B2 single-foot runs | 4 | 1 | 2 | 3 | 1 | 2 | 1 | =0 |
| C1 pelvis_min_intent | 0.19 | 0.12 | 0.11 | 0.34 | 0.10 | 0.12 | 0.13 | ≥0.50 |
| C2 pelvis_mean_intent | 0.54 | 0.48 | 0.47 | 0.60 | 0.45 | **0.26** ❌ | **0.23** ❌ | ≥0.50 |
| C3 stable% intent | 67% | 64% | 60% | 93% | 62% | **6.9%** ❌❌ | **6.9%** ❌❌ | ≥80% |
| C4 final_obj_pos_err (cm) | 0.81 | 3.10 | 1.75 | 112.01 | 138.18 | 1.75 ✓ | 160.15 ❌ | ≤30/20 |

E067-N C4=1.75cm 仍 PASS 因强 actuator 拽 box. 但身体姿态完全崩.

## 3. 视觉观察 (规则 9)

### E067-N
- t=0.8s: sim **倒立 handstand** — 右腿伸直朝上 1.30m, 左腿即将抬起 1.04m, 头朝下双手撑地, 箱子在旁边
- t=1.5-2.5s: pelvis 在 0.13-0.35m 之间挣扎, 身体几乎平躺
- t=3.0s+: pelvis 飙升 0.89m 起身站立, 但箱子仍未真正搬运 (虽然 C4 1.75cm 是 strong actuator 把 box 拖到了 ref 终点)

→ keyframes: `workspace/core4d/results/E067/keyframes/E067N_box023_kf*.jpg`
→ foot trace: `workspace/core4d/results/E067/foot_z_trace_E067N_box023.png` (B1 最高 1.30m)

### E067-NS  
- 类似 E067-N 但更糟: B1 = 1.25m, 加上 soft actuator → C4 = 160cm box 完全弃置

## 4. 修正的诊断 (推翻 log 85 §4)

| log 85 假设 | E067 数据 | 修正版 (教训 #15) |
|------------|----------|-------------------|
| MJWP 12 lower bodies mean 稀释 outlier 反 reward | E067-N B1=1.30m (handstand), C3=6.9% | **完全反了**: more bodies tracking 提供更多 DoF 约束, 防止 CEM 找到极端 pose; 移除中间 link 约束让 CEM 自由探索, 找到 "handstand" 比 "lunge" 更优 |
| Narrow partition 对齐 HDMI 是 net positive | sim 完全崩溃 | partition 不是问题; HDMI 用 6 bodies 因 PPO 已学到 prior, MJWP CEM 没 prior 需要更多约束 |

## 5. 3 round 综合分析 — 真正的元凶在哪里?

| 已 ablate (排除 prime suspect) | 结论 |
|-------------------------------|------|
| task_obj reward form (L2 vs exp vs drop) | 影响 B1 ±20%, 不是 root cause |
| actuator stiffness (kp 500 vs 20) | 改变 stability/carry trade-off, 不消除 lunge |
| body partition (12+17 vs 6+6) | 反方向 — wide 比 narrow 好, dilution 不是元凶 |

**所有 reward + dynamics 单点修改都未消除 pre-contact lunge**. 这强烈暗示元凶在更深层:

### 候选 A: Solver 层差异 (PPO learned prior vs CEM no-prior)
HDMI 用 PPO 训练好的策略 — 已学到"双脚平地" prior. MJWP 用 CEM 每帧采样 1024 candidates 在 48-knot horizon. CEM 没 learned prior, 每个 episode 从 zero ctrl noise 采样. 这本质差异让 MJWP 即使在同 reward + 同 physics 下找不同局部最优.

→ 可能 fix: hot-start CEM with ref ctrl + 更小 noise / 更大 horizon / smoothness penalty

### 候选 B: knot_dt + horizon 时间分辨率
HDMI knot_dt=0.20, MJWP knot_dt=0.10 (2× 密). 在同 horizon 0.8s 下 MJWP 用 8 knots vs HDMI 4 knots. 更密的 control 给 CEM 更多自由度变 pose.

### 候选 C: CEM noise schedule + temperature
MJWP `first_ctrl_noise_scale=0.5, last_ctrl_noise_scale=1.0` — noise 反向递增 (随 iter 增大), HDMI/RL 一般是 anneal down. CEM 在最后几 iter 大 noise 可能跳进 lunge basin 后稳住.

### 候选 D: contact_hdmi reward / hand_approach 等其他 reward
E065-A 显示这些都=0 (E041c 默认), 但是否有些其他 reward 一直 active 没注意到? Need full reward trace dump.

### 候选 E: Initial pose (sim t=0 vs ref t=0)
我已 verified pelvis_z 都 0.79m + foot grounded, 看似 OK. 但 pelvis quat 有差异 — sim 0.50 vs ref 0.66 (different orientation). 也许是 robot t=0 朝向不对.

## 6. 暂停理由 (plan §6)

触发器:
- ✋#3 (新 failure mode 第 1 次出现): "handstand" — B1 = 1.30m 完全超出预期
- ✋#5 (累计 4 轮仍 box023 不 work) — 已用 3 round (E065/E066/E067), 各自不同新 failure mode (deep-dive / abandon-box / handstand), 4 round 后该停

按 plan §6 "**累计 4 轮仍 box023 不 work — 触发 3-strike 协议升级版, 停下系统性 review**" — 我应该暂停, 不应继续盲调.

## 7. 等用户决策的关键问题

1. **当前已知 fact**: 3 个 ablation 全失败, 元凶不在 reward form / actuator stiffness / body partition. 你认为哪个候选 (A/B/C/D/E §5) 最可能?
2. **R4 方向选择**:
   - **R4-direct**: 直接跑 `run_hdmi.py on box023` (HDMI 已存在 trajectory, 无需训练), 拿 sim qpos + foot_z 跟 MJWP 直接 trace 对比, 找剩余差异. 这是诊断, 不是 fix.
   - **R4-hot-start**: 试候选 A — 把 ref qpos/ctrl 作为 CEM warm start, 减小 noise scale (0.1 → 0.02). 配套 yaml override 即可, 无需改代码.
   - **R4-knot**: 试候选 B — knot_dt 0.1 → 0.20 + horizon 加长 (0.8 → 1.6s). yaml override.
   - **R4-noise**: 试候选 C — noise schedule anneal down (0.5→0.1 而非 0.5→1.0). yaml override.
   - **R4-data-fix**: 用户曾说 ref 左手位置不对发不上力 — 这个不解释 pre-contact lunge (ref 左手在弯腰阶段没接触), 但可以试.
3. **是否考虑放弃 box023 在 MJWP 框架内**: HDMI 在类似 omomo suitcase 上 work (see `workspace/hdmi_reproduce/results/R013/R013_comparison.mp4`). 也许 box023 应该用 HDMI 而非 MJWP solve.

## 8. 教训汇总 (R1-R3)

- **#11** (sharper 版): reward 中 unbounded / 总是 active 项扰动 CEM (但 ablate task_obj 后仍 lunge → 不只是 reward form 问题)
- **#12**: 调 reward 前先对齐 dynamics
- **#13** (推翻): "reward.mean 下 body 数加多反向稀释" — 实际反过来, more bodies HELP
- **#14**: soft actuator 是 trade-off, 必须配套 task_obj 加大
- **#15** (新): more body tracking constraints HELP CEM 即使有 mean dilution; HDMI 用少 body 因 PPO 已学 prior, MJWP CEM 没 prior 需要更多约束
- **#16** (新): 3 round ablation 全失败强烈暗示元凶在更深层 (solver / time resolution / noise schedule), 不在 reward / dynamics 单点; 应停下 review 而非继续 ablate

## 9. 改动文件 / 结果路径

| 类型 | 路径 |
|------|------|
| 训练脚本 | `workspace/core4d/scripts/train/train_E067.sh` |
| 评估脚本 | `workspace/core4d/scripts/eval/eval_E067.py` (clone E065) |
| 关键帧脚本 | `workspace/core4d/scripts/eval/extract_E067_keyframes.sh` |
| Yamls | `examples/config/override/core4d_e067{n,ns}_box023.yaml` |
| Code | (无, 纯 yaml override) |
| Scene snapshot | `workspace/core4d/results/E067/scene_snapshot/` |
| 结果 | `workspace/core4d/results/E067/E067{N,NS}_box023.{npz,mp4}` |
| Plots | `workspace/core4d/results/E067/{foot_z_trace,pelvis_obj}_E067{N,NS}_box023.png` |
| Keyframes | `workspace/core4d/results/E067/keyframes/E067{N,NS}_box023_kf*.jpg` |
| CSV | `workspace/core4d/results/E067/eval_summary.csv` |
