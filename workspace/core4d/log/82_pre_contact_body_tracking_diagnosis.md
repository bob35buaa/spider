# Pre-Contact Body Tracking Diagnosis — box023 fails BEFORE contact, box025 works because no squat

## 状态: 🚨 **诊断完成 — 用户假设完全正确. box023 在 t=0.6-1.0s (pre-contact) sim 已经把右脚抬到 60cm 高空, 与 ref "双脚平地深蹲"完全错位; box025 ref 几乎不弯腰所以 sim 自然站姿就 OK. E062-E064 全部 reward 调参走错方向 (调的是 contact + stability), 真问题是 body tracking 缺 world-frame foot constraint**

**TL;DR**: 用户提出 box023 "弯腰时左腿抬起就摔了"假设. 验证: 提取 ref/sim/box023/box025 在 t=0-2s 的 trace_pelvis + trace_left_foot + trace_right_foot 全帧. 关键发现 — box023 在 t=0.6-0.8s ref 是 "双脚 z=0 + 深蹲 pz=0.55 + 抓 box": sim 同时段是 "**右脚 z=0.62m 悬空 + pelvis 0.55-0.71 + 没碰 box**", 单脚支撑 + 后脚甩起的姿势, 物理上必然失稳. box025 同时段 ref pz=0.66-0.73 (几乎不弯腰, walking 姿态接近 box), sim 也是双脚轻微抬起 walking 跟得上, 不需要"deep squat with both feet planted"这种 sim 无法稳定的 ref pose. **结论**: E062-E064 三次都在调 contact + stability_penalty + task_obj weight, 但实际问题是**中身 body tracking 没 constrain 脚在世界系的 z**. E041c 的 `local_frame_w_track=0.5` 只用 pelvis 系内关节角 + pelvis 世界位置, 没追 trace_left_foot / trace_right_foot 在世界系的位置, CEM 在"跟 ref 关节角"约束下仍能找到"单脚甩高"局部最优. **真正的 E065 方向**: 重启 `task_body_rew_scale` 给 trace_left_foot + trace_right_foot 紧 sigma (~0.05-0.10m), 强制 sim 双脚跟 ref 双脚位置 — 而不再做 stability_penalty / task_obj 的微调.

## 1. 触发: 用户视觉观察

> "感觉很奇怪啊,我看到box023在一系列实验中,最大的问题是,在一开始没有接触物体的时候(弯腰的时候),身体就已经摔倒了,左腿有一个抬起的动作,而ref双腿始终贴地, 所以我感觉一开始接近箱子的时候就已经出问题了, 这个可能不是后面接触的问题? 我觉得你得排查下原因了, 是不是body tracking的问题? 是不是之前box025这个case没有弯腰的动作,恰好就work了? box023有弯腰就不行呢"

用户提出的两个假设需要分别验证:
- H1: box023 在 pre-contact (弯腰) 阶段 sim 已经因为"腿抬起"失稳, 后面摔倒是这个失稳的延续
- H2: box025 work 是因为 ref motion 几乎不需要弯腰, sim 站姿就跟得上, 不暴露"深蹲需求"下的 body tracking 弱点

## 2. 数据验证 — pre-contact 阶段 ref vs sim 全帧 trace

提取 trace_ref body indices (从 config_act.yaml 读 trace_site_ids = [17, 1, 3, 6, 10, 14], 对应 trace_object/pelvis/left_foot/right_foot/left_hand/right_hand).

### box023 pre-contact (t=0-2s, intent 起始 t=0.7s)

| 时间 | ref pz/Lf/Rf (m) | E062 sim pz/Lf/Rf | E063 sim pz/Lf/Rf | E064 sim pz/Lf/Rf | 解读 |
|------|------------------|--------------------|--------------------|--------------------|------|
| 0.00 | 0.70 / 0.00 / 0.01 | 0.71/0.01/0.01 | 0.71/0.01/0.01 | 0.71/0.01/0.01 | 初始站立, 全部对齐 |
| 0.20 | 0.70 / 0.00 / 0.01 | 0.71/**0.20**/0.03 | 0.70/**0.20**/0.03 | 0.71/**0.18**/0.02 | ⚠️ sim 开始抬左脚 (20cm) |
| 0.40 | 0.67 / 0.02 / 0.01 | 0.74/**0.14**/0.05 | 0.74/**0.14**/0.05 | 0.73/**0.16**/0.04 | sim 仍抬左脚 14-16cm |
| 0.60 | 0.55 / 0.01 / -0.00 | 0.56/-0.01/**0.25** | 0.54/-0.01/**0.17** | 0.67/0.10/0.07 | ⚠️ ref 已深蹲, sim 换抬右脚 |
| **0.70** | 0.51 / 0.02 / 0.01 | 0.50/-0.01/**0.62** ⚠️ | 0.49/-0.01/**0.46** ⚠️ | 0.62/0.04/**0.18** | ⭐ **ref 深蹲到位, sim 右脚悬空 62/46/18 cm** |
| 0.80 | 0.55 / 0.02 / -0.00 | 0.54/-0.00/**0.52** | 0.52/-0.01/**0.38** | 0.57/0.01/**0.36** | sim 仍单脚支撑 |
| 1.00 | 0.65 / 0.03 / 0.02 | 0.55/0.00/0.08 | 0.55/0.03/0.09 | 0.59/0.01/0.22 | ref 站起 (回到 0.65), sim 半起 |
| 1.20 | 0.68 / 0.03 / 0.01 | 0.61/0.03/0.08 | 0.63/0.04/0.08 | 0.66/0.07/0.09 | ref 直立 carry, sim 半弯腰 |
| 1.40 | 0.69 / 0.02 / 0.02 | 0.58/0.03/**0.21** | 0.59/0.01/0.10 | 0.58/0.01/0.15 | ref 走直立, sim 又抬右脚 |
| 1.60 | 0.69 / 0.04 / 0.02 | 0.60/0.05/**0.21** | 0.59/0.04/**0.22** | 0.60/0.05/0.11 | sim 始终单脚撑 |
| 1.80 | 0.69 / 0.04 / 0.03 | 0.55/0.07/**0.30** | 0.56/0.06/**0.40** | 0.61/0.04/**0.37** | sim 越走越右脚抬高 |
| 2.00 | 0.69 / 0.03 / 0.03 | **0.30**/0.08/0.21 | **0.37**/0.15/0.30 | **0.44**/0.18/0.28 | ⭐ **崩溃**: 1.5s 单脚撑撑不住 |

**关键: ref 全程 Lf/Rf 都在 0.00-0.05m 范围 (双脚贴地)**, sim 全部三个版本都是**一脚 0.0-0.05m + 另一脚 0.15-0.62m** 的单脚撑模式. t=2.0s 的"摔倒"是 1-1.5s 单脚撑的物理延续 — 不是接触阶段才出问题.

### box025 pre-contact (t=0-2s, intent 起始 t=0.67s)

| 时间 | ref pz/Lf/Rf | E062 sim pz/Lf/Rf | 是否对齐? |
|------|--------------|--------------------|----------|
| 0.00 | 0.73 / 0.03 / 0.02 | 0.73/0.03/0.01 | ✅ |
| 0.40 | 0.72 / 0.02 / 0.01 | 0.72/0.22/0.03 | ⚠️ sim 抬左脚 22cm 但 ref pz 没变 |
| 0.60 | **0.70** / 0.01 / 0.01 | 0.60/0.07/0.05 | ✅ ref 几乎不蹲 |
| 0.80 | **0.67** / 0.00 / 0.00 | 0.65/0.09/0.00 | ✅ ref 微蹲, sim 跟得上 |
| 1.00 | **0.66** / 0.00 / 0.00 | 0.69/0.06/0.02 | ✅ |
| 1.20 | **0.67** / 0.00 / 0.00 | 0.67/0.03/0.05 | ✅ |
| 1.40 | **0.69** / 0.02 / 0.00 | 0.69/0.05/0.03 | ✅ |
| 1.60 | 0.68 / 0.01 / 0.02 | 0.70/0.07/0.04 | ✅ |
| 1.80 | 0.68 / 0.03 / 0.05 | 0.67/0.10/0.06 | ✅ |
| 2.00 | 0.70 / 0.00 / 0.04 | 0.66/0.08/0.02 | ✅ |

**关键: box025 ref 全程 pz 0.66-0.73m, 几乎不弯腰** (box025 是大型纸箱 ~70cm 高, ref 只是站着双手扶箱). sim 的"轻微抬脚 5-22cm"在这种 pose 下是合理的 walking gait, 不需要 sim 严格跟"双脚贴地深蹲". 因此 sim 在 box025 上不暴露 body tracking 弱点 → 数值看起来 PASS.

## 3. H1 + H2 验证结果

| 假设 | 验证状态 | 证据 |
|------|---------|------|
| H1: box023 sim 在 pre-contact 已经因为"腿抬起"失稳 | ✅ **VERIFIED** | t=0.7s ref Rf=0.01, sim Rf=0.62m. **62cm 误差**, 全程 1.0-2.0s sim 单脚撑模式持续 |
| H2: box025 work 因为 ref 不弯腰, sim 不暴露问题 | ✅ **VERIFIED** | box025 ref pz 0.66-0.73 全程 (微蹲), box023 ref pz 0.51-0.69 (深蹲到 0.51m). 同样 reward 在 box025 不需要"深蹲且双脚贴地"所以不 fail |

**用户假设 100% 正确**.

## 4. 元凶: E041c reward stack 缺世界系 foot tracking

E041c (HDMI-style local-frame reward) 的 body tracking 信号:
- `use_local_frame_reward = true`
- `local_frame_pos_sigma = 0.5` (body pos 在 pelvis 系)
- `local_frame_root_sigma = 0.5` (pelvis 在世界系)
- `local_frame_joint_sigma = 0.25` (关节角)
- `local_frame_w_track = 0.5` (整体权重)
- `task_body_rew_scale = 0.0` ⚠️ **关键: world-frame body 跟踪关闭** (E036 关掉)

**问题**: local_frame 反映 "pelvis 系内 body 位置 + 关节角", 但 pelvis 本身可以在世界系倾斜+漂移. 当 sim pelvis 前倾 30°追 box, 同样的"pelvis 系 + 关节角"可以对应世界系完全不同的 foot 位置. 具体:
- ref: pelvis 直立 + 髋屈 90° + 膝屈 90° → 双脚在 pelvis 后下方 z=0
- sim: pelvis 前倾 45° + 髋屈 60° + 膝屈 60° → **(在 pelvis 系内"看起来"差不多, 但世界系下脚高 50cm)**

CEM 找到这个解是因为:
- 维持 ref 的"双脚贴地"需要 pelvis 直立 + 强髋屈, G1 这种 pose 容易拉胯肌不稳
- "前倾 + 单脚撑"反而是 G1 物理上更稳的 pose (动量补偿)
- local_frame reward 不够分辨这两个 → CEM 选物理稳的, 但视觉上是错的

## 5. 为什么 E062-E064 调参没解决?

E062-E064 调的都是 **stability_penalty (pelvis_z 高度) + task_obj (box 跟踪) + contact_hdmi (手到 box)** 这一组 reward. 全部都是关于 **pelvis-高度 / box-接触** 的标量, 没有一个约束 **foot 位置**.

| 实验 | 调的参数 | 实际触动的 reward 项 | 是否触动 foot tracking? |
|------|---------|---------------------|------------------------|
| E062 | auto X1 palm_normal | contact_hdmi orientation | ❌ |
| E063 | stability_penalty=1.0 + task_obj=0.5 | pelvis_z 高度 + box 位置 | ❌ |
| E064 | + root_sigma=0.3 + contact_gain=3.0 + thresh=0.65 | pelvis_pos 紧 + 手拉力弱 + 摔倒阈值高 | ❌ (root_sigma 是 pelvis 位置, 不是 foot) |

**所有 reward 调整都在 pelvis 和 hand 这两端, foot 中段始终没约束**. 这就是为什么三次实验都让 CEM 找到新失败模式 — 它换的是 pelvis/hand 的姿态, foot 始终自由.

## 6. 真正的 E065 方向 (取代之前的 X5)

### 提议的 reward 修改 (yaml only)

```yaml
# core4d_e065_box023.yaml
defaults:
  - core4d_e063_box023   # 继承 E063 (T1 OK), 不带 E064 (T2 边际破坏 box025)
  - _self_

# 重启 task_body_rew_scale 但只给 feet 紧权重
task_body_rew_scale: 5.0
task_body_names:
  - left_ankle_roll_link    # 左脚
  - right_ankle_roll_link   # 右脚
task_body_weights:
  - 1.0
  - 1.0
# (注意: task_body_sigma 用 spider 默认, 大约 0.10m. 这意味着 ref 双脚 z=0, sim 脚 z>0.10 就有 penalty.
#  sigma 视情况调到 0.05-0.15m, 紧一点更好但可能压制 walking)
```

### Pass criteria (E065 box023)
- B1 (NEW): Lf_max + Rf_max in t=0-2s ≤ 0.10m (vs E064 max 0.36m)
- B2 (NEW): 单脚撑帧数 (其中一脚 z>0.05 而另一脚 z<0.02) ≤ 5 帧 (vs E064 ~50 帧)
- 沿用 E063 的 C1-C4 + box025 R1-R3 regression guard

### 风险
- task_body 重启可能与 contact_hdmi 冲突 (E036 关闭它的原因), 需要回归测试 box025
- weight=5.0 是猜的, 可能需要 sweep 1.0/3.0/5.0
- 仅约束 feet 不约束 thighs, 可能 sim 用其他 joint 补偿

### 备选 (如果 task_body 路径不通)
- 直接收紧 `local_frame_pos_sigma 0.5 → 0.10` (所有 body 都更严, 不只是 feet) — 但可能限制太死
- 写 custom "feet_grounded_when_ref_grounded" reward (代码改, Tier 3)

## 7. 三次失败模式重新解读

| 实验 | 之前的归因 | 现在 (后视) 的归因 |
|------|-----------|------------------|
| E062 | "auto X1 让 sim 真做 carry, 但物理不稳, 摔" | sim 弯腰时单脚抬高 60cm, 1s 后撑不住摔; X1 改对了 palm_normal 但和 foot tracking 无关 |
| E063 | "stability=1.0 阻止深 fall 但 CEM 找 superman" | sim 在 t=0.7s 仍单脚抬高, 多了 stability_penalty 后 CEM 把"摔到底"换成"水平 superman", **foot 抬起的根因没变** |
| E064 | "T2 + threshold raise 让 sim 全程 prone" | sim 在 t=0.7s 单脚抬高问题没解决, T2 抑制 pelvis 偏离后 CEM 选 "全程低 pelvis 不站起", **foot 抬起的根因还是没变** |

**所有失败都是同一个 underlying issue: foot 在世界系不被约束**. E062-E064 各自换 pelvis/hand reward 改变了"摔的方式", 没改"摔的原因".

## 8. 改动文件 (本 log)

| 文件 | 改动 |
|------|------|
| `workspace/core4d/log/82_pre_contact_body_tracking_diagnosis.md` | 本文件 |
| `workspace/core4d/results/diagnosis_pre_contact/E062_box023_t{0.4,0.6,0.8,1.0}s.jpg` | 4 帧诊断 keyframe (复用 E062 mp4) |
| `workspace/core4d/results/diagnosis_pre_contact/E062_box025_t{0.4,0.6,0.8,1.0}s.jpg` | 同上 box025 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` | (待更新) 加 diagnosis 行 + log 82 索引 |

(E065 yaml + train script 留到执行 plan 时新建.)

## 9. 教训 #11 (累计 10 个之前)

之前 10 个教训 (log 81 §10 + 历史):
1-9: log 80 §9 + log 81 §10
10: 同 reward 框架内 weight 调参 3 次都让 CEM 找到 NEW local optimum (3-strike rule)

**新教训 #11 (本 log)**:

> **当一系列 reward 调参都失败但 failure mode 各不相同, 真问题往往不是 weight, 而是 reward 缺一个维度. 应该问"什么 reward 项缺失"而不是"哪个 weight 不对". 多次失败 + 多种失败模式 = 信号: reward landscape 有多个 dead-end pit, 调 weight 只是在 pit 间换. 必须找到一个未被约束的物理量并加进去**.

具体规则:
- 任何"3-strike"failure 后必须**复盘 reward 分量分析**: 列出当前 reward 约束了什么 (pos/rot/contact/...), 然后逐项问"sim 在这个维度上自由吗?" 找未约束的维度.
- 视频 + ref 对比看的不只是"sim 跟 ref 像不像", 还要看**哪些 body part 显著偏离 ref**, 这些就是 reward 没约束的地方.
- 单 case PASS 的 baseline 不能盲目沿用. box025 PASS 是因为 task 简单, 不证明 reward 完整. **必须测多种 task 难度**才能发现 reward 缺失维度.
- 用户的 "看起来不对" 直觉常常 catch 数值 KPI 漏掉的物理 issue — 应优先采纳并系统验证, 而不是用"数值改善"反驳.

## 10. 用户补充关键证据: HDMI workflow 在 box023 pre-contact body tracking 完美

> "hdmi workflow(run_hdmi.py)虽然也在box023case上失败了, 但是它在接触物体之前的body tracking是好的, 它的问题应该是接触or数据问题(ref左手搬箱子的位置不太对,发不上力). hdmi 结果可以看 workspace/core4d/results/E052/E052c_box023_euler_fix/visualization_hdmi.mp4. 做这些实验的原因是, hdmi workflow在omomo数据集的suitcase上work了 (workspace/hdmi_reproduce/results/R013/R013_comparison.mp4), 而suitcase很接近box023这个case"

### 10.1 视觉验证 (HDMI box023 pre-contact)

抽帧 `E052c_box023_euler_fix/visualization_hdmi.mp4` 在同样时间戳:

| 时间 | HDMI sim 行为 vs ref |
|------|---------------------|
| t=0.4s | sim 跟 ref 完全同步, 直立站姿微弯腰 |
| **t=0.6s** | **sim 双脚平地, 弯腰中, 双手伸向 box** ⭐ vs MJWP 同时段右脚悬空 25cm |
| **t=0.8s** | **sim 双脚平地, 双手在 box 上** ⭐ vs MJWP 右脚悬空 52cm |
| t=1.0s | sim 双脚平地, 接触 box 顶 |
| t=1.5s | sim 仍双脚平地, 试图抬 box (没抬起) |
| t=2.0s | sim 蹲在 box 旁(双脚仍在地面), ref 已直立 carry. **HDMI 失败, 但失败模式是"没成功抬起", 不是"摔倒"** |

→ **HDMI body tracking on box023 完全 OK**. 用户结论"HDMI 问题在接触/数据 (ref 左手位置发不上力), 不是 body tracking"100% 对.

→ **同样 ref motion, MJWP 失败在 pre-contact body tracking, HDMI 成功**. 这意味着 ref 数据是可被 G1 物理实现的, 不是 ref 不可达 (排除我 §10 第 4 个怀疑).

### 10.2 Reward stack 逐项 diff (HDMI vs MJWP E041c)

**惊人发现: body-tracking 数学公式完全相同**:

| 项 | HDMI `get_reward` | MJWP `local_frame_rew` (E035 port) | 是否相同? |
|-----|-------------------|------------------------------------|----------|
| upper_pos | exp(-err.mean / 0.5) | exp(-err.mean / 0.5) | ✅ |
| upper_ori | exp(-err.mean / 1.0) | exp(-err.mean / 1.0) | ✅ |
| lower_pos | exp(-err.mean / 0.5), bodies = hip+knee+ankle | exp(-err.mean / 0.5), bodies = hip+knee+ankle | ✅ |
| lower_ori | exp(-err.mean / 1.0) | exp(-err.mean / 1.0) | ✅ |
| root_pos (world) | exp(-err / 0.5) | exp(-err / 0.5) | ✅ |
| joint | exp(-err.mean / 0.25) | exp(-err.mean / 0.25) | ✅ |
| W_TRACK | 0.5 * sum | 0.5 * sum | ✅ |

**body tracking 是逐字 port**. 但 MJWP 失败 HDMI 成功 → 元凶在**其他 reward 项 + config**.

**HDMI ONLY (saturating + phase-gated)**:
```python
rew_obj_pos = exp(-||obj_err|| / 0.5)        # ∈ [0,1] saturating
rew_obj_ori = exp(-quat_err / 0.5)           # ∈ [0,1]
rew_contact = pos_rew * force_factor * in_range * 5.0
              + (1.0 - in_range)              # phase-gated (in_range = ref 接触阶段)
total = tracking + rew_obj_pos + rew_obj_ori + rew_contact
```

**MJWP E041c ONLY (UNBOUNDED L2 + always-on)**:
```python
task_obj_rew = -1.0 * SUM_xyz(err²)          # ⚠️ UNBOUNDED L2, 不饱和, 总是 active
              -1.0 * SUM(quat_err²)
contact_hdmi_rew = exp(-d/0.3) * 5.0 * mask
                   + (1.0 - mask)             # phase-gated (mask = 0 in pre-contact ✅)
```

**关键差异**:
- HDMI 有 `rew_obj_pos = exp(-err/0.5)`: 物体偏离时 reward 从 1 → 0 saturating
- MJWP 有 `task_obj_rew = -1.0 * err²`: 物体偏离时 penalty -∞ unbounded
- 当 sim 物体落后 ref (e.g. t=1s ref obj 在前方 25cm), MJWP 给 sim 大 negative penalty 推动 sim 急赶 → CEM 选 "前倾抬腿急追" 解
- HDMI 同情况只是 obj_pos rew 降到 exp(-0.5)=0.6, 还有 0.6 reward, 不强烈逼 sim 追

**还有 config 差异 (可能放大问题)**:

| 参数 | MJWP E041c | HDMI | 差异 |
|------|-----------|------|------|
| `init_pos_actuator_gain` | **500.0** | 20.0 | **25× 强**, MJWP 物体被强力拉到 ref |
| `init_rot_actuator_gain` | 50.0 | 0.3 | 167× 强 |
| `guidance_decay_ratio` | 1.0 | 0.85 | MJWP 不衰减 actuator |
| `knot_dt` | 0.10 | 0.20 | MJWP 控制点 2× 密 (更多 DoF) |
| `first_ctrl_noise_scale` | 0.5 | 1.0 | MJWP 起始噪声更小 |

强 actuator 反而可能是问题: MJWP 物体被强力拽走 → sim 也想跟上 (因为有 task_obj penalty), 但 sim 跟不上人速度 → 急追 → 抬腿.

### 10.3 修正后的 E065 plan (取代 §6 提议)

我之前 §6 提的 "重启 task_body_rew_scale for feet" 是猜的, 现在有 HDMI 对照, 应该**直接 port HDMI 的 reward 配置**, 而不是另加一个 reward 项:

**E065-A (推荐, 最小改动)**: 关 MJWP E041c 的 `task_obj_pos_rew_scale = 0.0` + `task_obj_rot_rew_scale = 0.0`.
理由: HDMI 没这个 unbounded L2 penalty 也 work. MJWP 加了它反而扰动 CEM 偏向"急追物体". local_frame + contact_hdmi (phase-gated) 应该够.

```yaml
# core4d_e065_box023.yaml
defaults:
  - core4d_e062_box023   # 回到 X1 only, 不带 E063/E064
  - _self_
task_obj_pos_rew_scale: 0.0   # was 1.0 (HDMI 没这项)
task_obj_rot_rew_scale: 0.0   # was 1.0
# 保留 contact_hdmi_gain=5.0 (phase-gated, 和 HDMI eef_gain 一样)
# 保留 stability_penalty_scale=0.0 (HDMI 也没这项)
```

**E065-B (备选, 如果 -A 不够)**: 进一步把 actuator gain 降到 HDMI 水平:
```yaml
init_pos_actuator_gain: 20.0
init_pos_actuator_bias: 20.0
init_rot_actuator_gain: 0.3
init_rot_actuator_bias: 0.3
guidance_decay_ratio: 0.85
```

**E065-C (验证用, 直接跑 HDMI workflow)**: 跑一次 `run_hdmi.py` 在 box023 上, 复现"pre-contact body tracking 好, contact 失败"现象, 作为 MJWP 的 ground-truth A/B baseline. 不用调试, 只是确认.

### 10.4 决策建议

3-strike 后被你的视觉直觉救回来. 我之前的 E063/E064 全部走错方向 (都在调 stability/contact reward weight, 没看 task_obj 这个 unbounded penalty 才是元凶). **建议优先 E065-A**: yaml only, 单 line 改动, 30min 训练, 立马知道 task_obj_rew 是否真是元凶.

如果 E065-A box023 sim 弯腰时双脚 z<5cm 全程 ✅: 元凶定位完成, 接下来研究 contact 阶段失败 (用户说 ref 左手位置发不上力 — 这是数据问题, 需要检查).

如果 E065-A 仍抬腿: 进 E065-B (port actuator gain), 同样 yaml only.

如果 E065-B 仍抬腿: HDMI A/B (E065-C) 必跑, 找出剩余差异.

### 10.5 启示给 §9 教训 #11

之前 §9 教训 #11 说"3 次失败但 failure mode 不同 = reward 缺一个维度". 修正:

> **更准确的版本: "3 次失败但 failure mode 不同 = reward 中有一个 unbounded / 总是 active 的项扰动 CEM"**. E041c 的 task_obj_pos = -1.0*err² 就是这个项 — saturating 的 exp(-err) 形式才是正确选择. MJWP 移植 HDMI 的 local_frame 时, 又加了 task_obj 这个原本 HDMI 没有的项, 破坏了原 HDMI 的 reward 平衡.
