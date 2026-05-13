# E065 Plan — task_obj_rew form ablation (zero / exp / L2) on box023

## 状态: 📋 **PLANNED — 代码改动完成 (config + mjwp.py 加 task_obj_use_exp), 4 yamls 创建. 待跑 2 round 并行训练 (~66min).**

**TL;DR**: log 82 诊断定位 MJWP `task_obj_rew = -1.0*err²` (unbounded L2) 是 box023 弯腰阶段抬腿 60cm 的元凶 (HDMI 同 ref motion + 同 body-tracking 公式 + 但用 saturating `exp(-err/0.5)` 就 work). 用户提议除了"完全关掉 task_obj"还要试一种"对齐 HDMI exp 形式". 设计 E065-A (drop) + E065-D (exp) 两个变体并行验证, 加 box025 regression guard. 实验前已改 spider 加 `task_obj_use_exp` 配置 flag (默认 false 保留原 L2 行为, 完全 backward-compatible).

## 1. 上下文 (从 log 80/81/82 凝练)

| 实验 | reward 改动 | failure mode | 教训 |
|------|------------|--------------|------|
| E062 | X1 only baseline | fall (pelvis 0.058m) | — |
| E063 | + stability=1.0 + task_obj=0.5 | superman lunge | #9 (height proxy) |
| E064 | + root_σ=0.3 + gain=3.0 + thresh=0.65 | prone (永不起身) | #10 (3-strike) |
| Diag | 用户视觉 catch sim 弯腰单脚 60cm | — | #11 (3 失败 mode 不同 = reward 缺维度 / unbounded 项) |
| HDMI A/B | 同 ref + box-tracking 公式但 work | (没 task_obj L2) | 元凶定位 |

E063/E064 三次都调 stability + contact + root_sigma, **从未改 task_obj 这个真元凶**. 用户视觉直觉 + HDMI 对照让方向归正.

## 2. 用户讨论关键观察

> "感觉很奇怪啊, 我看到box023在一系列实验中, 最大的问题是, 在一开始没有接触物体的时候 (弯腰的时候), 身体就已经摔倒了, 左腿有一个抬起的动作, 而ref双腿始终贴地"
> "hdmi workflow 在 omomo suitcase 上 work, 而 suitcase 很接近 box023... 它在接触物体之前的 body tracking 是好的, 它的问题应该是接触 or 数据问题 (ref 左手搬箱子的位置不太对, 发不上力)"
> "起码得再加一个实验: 对齐 hdmi 的 rew_obj_pos (指数形式, 不是无界的)"

→ HDMI workflow 在 box023 pre-contact 双脚平地弯腰 ✅, MJWP 同时段右脚悬空 25-62cm ❌. 元凶**只能**在 MJWP 比 HDMI 多的 reward 项 / config 上.

## 3. Reward stack diff 复盘 (log 82 §10.2 提炼)

**body-tracking 公式完全相同** (MJWP 的 `local_frame_rew` 是 HDMI `get_reward` 的逐字 port — 同 sigma 同 body 划分同 W_TRACK=0.5).

**MJWP E041c 独有 + 总是 active**:
```python
# MJWP (spider/simulators/mjwp.py):
task_obj_rew = -task_obj_pos_scale * SUM_xyz(err²)   # ⚠️ UNBOUNDED L2
              -task_obj_rot_scale  * SUM(quat_err²)
```

**HDMI 等价位置 (saturating)**:
```python
# HDMI (spider/simulators/hdmi.py):
rew_obj_pos = exp(-||obj_err||/0.5)   # ∈ [0,1] saturating
rew_obj_ori = exp(-quat_err/0.5)      # ∈ [0,1]
```

数学差异:
- 物体偏离 25cm: MJWP penalty -1.0*0.0625 = **-0.063** (小)
- 物体偏离 50cm: MJWP penalty -1.0*0.25 = -0.25
- 物体偏离 100cm: MJWP penalty -1.0*1.0 = **-1.0** (大且越来越大)
- HDMI 同样情况: exp(-0.25/0.5)=0.61 / exp(-0.5/0.5)=0.37 / exp(-1/0.5)=0.135 (saturating, 永远在 [0,1])

→ 物体跟丢得越远, MJWP 越急, HDMI 已经不再加压. CEM 在 unbounded penalty 下选 "急追物体 → 抬腿前倾" 局部最优.

## 4. E065 实验设计 (3 个变体 + 2 case regression guard)

### 4.1 变体定义

| 变体 | task_obj_rew 形式 | scale | 假设验证目标 |
|------|------------------|-------|------------|
| **E065-A** | **关闭** (scale=0) | 0.0/0.0 | task_obj 项是否真是元凶 — 完全去掉它 sim 是否双脚平地弯腰 |
| **E065-D** | **HDMI exp** | 1.0/1.0 sigma=0.5 | 是 unbounded 性质有问题, 还是 task_obj 信号本身有问题 — 保留信号但 saturating |
| (E062 baseline 复用) | -L2 unbounded | 1.0/1.0 | 已有数据, sim 弯腰右脚 62cm 抬高 |

### 4.2 期望结果矩阵

| E065-A | E065-D | 元凶定位 |
|--------|--------|---------|
| ✅ 双脚平地 | ✅ 双脚平地 | task_obj 信号本身没问题, **形式 (unbounded)** 是元凶 → 永久 port HDMI exp form |
| ✅ 双脚平地 | ❌ 仍抬腿 | **task_obj 信号本身有问题** (即使 saturating), 应永久关闭 |
| ❌ 仍抬腿 | ✅ 双脚平地 | 不太可能 (反直觉, 但可能是 task_obj exp 提供了 "靠近就奖励" 的引导) |
| ❌ 仍抬腿 | ❌ 仍抬腿 | task_obj **不是**唯一元凶, 还有其他差异 (actuator gain / knot_dt / ...). 进 E066 = port HDMI actuator 配置 |

### 4.3 Pass criteria

**主指标 (新增)**: pre-contact 阶段双脚高度
- B1: t=0-2.0s 内 max(Lf_z, Rf_z) ≤ 0.10m (sigma=0.05m, 允许 walking 微抬)
- B2: 单脚撑帧数 (一脚 z>0.05m 而另一脚 z<0.02m 持续 ≥ 5 帧) = 0
- 对照: E062-E064 在 t=0.5-1.8s 都是 max(Rf) ≥ 0.20m 的单脚撑模式

**保留 (沿用 E063 阈值)**:
- C1: pelvis_min_intent ≥ 0.50m
- C2: pelvis_mean_intent ≥ 0.50m
- C3: stable% intent ≥ 80%
- C4: final_obj_pos_err ≤ 20cm
- C5: 5-frame 视觉无摔倒

**box025 regression guard**:
- R1: pelvis_min ≥ 0.65m, R2: stable% ≥ 99%, R3: final_obj ≤ 25cm

### 4.4 决策树

- **A 通过 + D 通过**: 设 task_obj_use_exp=true 为 spider 默认; 之后所有 case 走 D; 如 box023 contact 阶段仍失败 (HDMI 也失败了), 进入 contact 数据修正轨道 (用户说 "ref 左手位置不对发不上力")
- **A 通过 + D 不通过**: 永久关 task_obj_pos/rot_rew_scale=0; 把 spider 默认改为 0; 进入 contact 数据修正
- **A 不通过 + D 不通过**: task_obj 不是唯一元凶; **E066 = port actuator gain (500→20, decay 1.0→0.85)**; 仍不通过 → E067 = 直接跑 run_hdmi.py 在 box023 (用现有的 trajectory_hdmi.npz, 无需训练) 拿 sim qpos 跟 MJWP 对比 foot_z trace, 找出剩余差异
- **任一 box025 R 失败**: 该变体反向破坏 box025 baseline, 不可用 (类似 E064 教训)

## 5. 实施

### 5.1 代码改动 (spider, backward-compatible)

**spider/config.py** 加 3 个字段:
```python
# Separate object pos/rot tracking weights (DynaRetarget uses obj_pos=40)
task_obj_pos_rew_scale: float = 0.0
task_obj_rot_rew_scale: float = 0.0
# E065: switch task_obj_rew form from unbounded L2 (default) to HDMI-style
# saturating exp(-err/sigma). When True, reward becomes
#   scale * exp(-||obj_err||/pos_sigma)  ∈ [0, scale]
# instead of  -scale * sum(err²)  (unbounded). Diagnosis log 82 §10.2.
task_obj_use_exp: bool = False
task_obj_pos_sigma: float = 0.5  # HDMI default
task_obj_rot_sigma: float = 0.5
```

**spider/simulators/mjwp.py** 在 `task_obj_rew` 计算块 (行 ~698-720) 加 `if use_exp: ... else: ...` 分支, 处理 nq_obj=7 和 nq_obj=6 两种情况下的 pos + rot reward. Defaults 保留旧行为 (use_exp=False), 历史 yaml 不改也能跑.

Backward-compat 验证: `Config()` 默认 `task_obj_use_exp=False`, `task_obj_pos_sigma=0.5`. 所有 historical override 不触动 → 等价于改前.

### 5.2 Yaml 文件 (4 个新建)

| 文件 | 继承 | overrides |
|------|------|-----------|
| `examples/config/override/core4d_e065a_box023.yaml` | `core4d_e062_box023` | task_obj_pos/rot_rew_scale = 0.0/0.0 |
| `examples/config/override/core4d_e065a_box025.yaml` | `core4d_e062_box025` | 同上 (regression guard) |
| `examples/config/override/core4d_e065d_box023.yaml` | `core4d_e062_box023` | task_obj_use_exp=true + sigma 0.5 + scale 1.0/1.0 |
| `examples/config/override/core4d_e065d_box025.yaml` | `core4d_e062_box025` | 同上 (regression guard) |

注: 都继承自 E062 (X1 only), **不带 E063/E064 的 stability/threshold 改动** — 因为 log 80/81 已证明那些是错方向, 我们要在干净 baseline 上测 task_obj.

### 5.3 训练 (2 round 并行)

- Round 1: E065-A box023 (GPU0) + E065-D box023 (GPU1), 并行 ~33min
- Round 2: E065-A box025 (GPU0) + E065-D box025 (GPU1), 并行 ~33min
- Total: ~66min

(也可以 E065-A box023+box025 一起跑然后 E065-D box023+box025 一起跑, 时间一样, 但 round 1 拿到 box023 主指标更早.)

### 5.4 评估

- Eval 脚本: `eval_E065.py` clone 自 `eval_E063.py`, 加新指标:
  - **B1 (NEW)**: t=0-2.0s max(Lf_z, Rf_z)
  - **B2 (NEW)**: 单脚撑帧数
  - 沿用 C1-C5 + R1-R3
  - 输出 4 行 CSV (E065-A box023, E065-A box025, E065-D box023, E065-D box025) + 跟 E062 baseline 对比
  - 输出 foot_z trace plot (Lf, Rf vs ref over time, 4 traces 叠加)
- Keyframes: t=0.4/0.6/0.8/1.0/1.5/2.0/2.5/3.0/4.0s × 2 case × 2 variant = 36 jpg

## 6. 待新建文件清单

| 类型 | 路径 | 状态 |
|------|------|------|
| 代码改动 | `spider/config.py` (+5 行 fields) | ✅ done |
| 代码改动 | `spider/simulators/mjwp.py` (+~20 行 if branch) | ✅ done |
| Yaml | `examples/config/override/core4d_e065{a,d}_box{023,025}.yaml` (4 个) | ✅ done |
| Train | `workspace/core4d/scripts/train/train_E065.sh` (2 round) | ⏳ 待建 |
| Eval | `workspace/core4d/scripts/eval/eval_E065.py` (B1/B2 新指标) | ⏳ 待建 |
| Keyframes | `workspace/core4d/scripts/eval/extract_E065_keyframes.sh` | ⏳ 待建 |
| 一键 | `workspace/core4d/scripts/run_E065.sh` | ⏳ 待建 |
| Output | `workspace/core4d/results/E065/` (含 scene_snapshot, npz, mp4, csv, png, jpg) | 训练后产出 |
| 结果 log | `workspace/core4d/log/8X_E065_results.md` | 评估后写 |
| Tracker | EXPERIMENT_TRACKER 加 E065 行 | 评估后更新 |

## 7. 与之前提议的差异

| 之前 plan (log 82 §6) | 现在 plan (本 log) | 修改原因 |
|----------------------|-------------------|---------|
| **重启 task_body_rew_scale=5.0 给 feet** | **task_obj_pos/rot_rew_scale=0 关掉, 或 use_exp=true 改形式** | 用户 HDMI 对照证明 task_obj **形式**是元凶, 不是缺 foot tracking. 加 task_body 是猜的, 现在有更直接的诊断证据 |
| 单变体 (重启 task_body) | A/D 双变体 ablation | 用户提议加 exp 形式作为对照, 区分 "task_obj 信号有问题" vs "L2 unbounded 形式有问题" |
| 改 spider 加新 reward 项 | 不加新 reward, 改现有 reward 形式 | 更小改动, 更可逆, 更精准定位 |

## 8. 教训 #11 完整版 (取代 log 81 §10 旧版)

> **3 次失败但 failure mode 不同的真正信号是: reward 中存在某个 unbounded / 总是 active 的项, CEM 在它的支配下被反复推到不同 dead-end pit**. 解法:
> - **优先检查 reward 是否 saturating** (exp 形式) 而不是 unbounded (-L2 形式)
> - **横向 A/B 对照**已知 work 的同任务 baseline (HDMI vs MJWP) 是定位元凶最快的方法
> - **不要在没找到根因前继续调 weight**, 每次调 weight 只是换 dead-end pit, 不解 root cause
> - 用户的"看起来不对"直觉常常 catch 数值 KPI 漏掉的物理 issue, 应优先采纳并系统验证

## 9. 风险与 guardrails

- **risk**: code change 引入 bug, 历史 baseline 跑不通. **guard**: defaults `use_exp=False` 完全 backward-compat, 已 import 测试通过. E065 训练前先在 box025 上跑一次 confirm 数值对得上 E062 (use_exp 默认 False 时)
- **risk**: A 和 D 都通过, 难以区分元凶是 "信号" 还是 "形式". **guard**: 重新读 reward 总贡献 (E065 训练后看 task_obj_rew_mean/min/max), 若 D 的 task_obj_rew 在 [0,1] 区间 sim 仍 work 就是 "形式" 问题
- **risk**: A 和 D 都不通过. **guard**: 不调 weight 救场, 直接进 E066 port actuator gain (decision tree §4.4)
- **anti-pattern**: 不再做 "再调一组 weight 试试" — 已经 3-strike, 必须根因解决