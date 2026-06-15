# E164 — bimanual global mask + bimanual reward plan

日期：2026-06-15

## 0. 背景

E163 clean8 的失败分析把问题定位到两个语义缺口：

1. mask 仍是左右手分开的局部语义；`box004_082_p1` 这类 case 会出现接触手侧切换，导致 reward/eval 的 active window 与“双手一起搬运”的真实意图不一致。
2. 当前 surfaceBand/contact reward 不是严格双手同时满足：`surface_band_sdf` 使用 `min_sdf(["lh", "rh"], object)`，任一只手贴近就可得分；`contact_hdmi_rew` 是 per-hand reward 的平均，也没有硬性 L/R AND。

用户指定 E164 要改两件事：

```text
1. raw mask 和 spider mask 重新处理：
   先取 max(R, L)，再填充中间空洞，得到双手全局 mask；
   然后左右手 mask 都设置为这个全局 mask。

2. reward 改成不能一只手接触就给分；
   必须双手同时接触、同时满足这个 reward，才给分。
```

本计划固化 E164 方案；当前只更新计划，不改代码、不启动 CEM。

## 1. 实验范围

E164 只跑用户指定的三个 case：

| case | 选择原因 |
|---|---|
| `box023_person2` | E161 曾导致下游 RL 失败，E163 已恢复，必须做回归保护 |
| `box004_082_p1` | E163 clean8 唯一 raw contact hard gate fail |
| `box026_139_p1` | clean8 扩展 case，验证新双手语义是否泛化 |

结果目录建议：

```text
workspace/core4d/results/E164/bimanual_global_mask_reward/
```

## 2. 方法定义

E164 以 E163 narrow surfaceBand 为 base，不改 E163 已确认的安全栈：

```text
base = E163 narrowSurfaceBandReleaseDecay

surface_band_min_sdf_m = -0.001
surface_band_width_m = 0.003
surface_band_sigma = 0.0015
surface_band_score_mode = symmetric_abs
surface_band_score = exp(-abs(sdf) / sigma)

cem_hand_gate_min_sdf_m = -0.010
cem_hand_gate_max_violation_pct = 0.10
cem_hand_gate_hard_floor_m = -0.020

cem_posture_gate_* = E160/E163 same
contact_hdmi_mask_source = core4d_3cm
```

E164 的增量只包含：

1. E164 专属 mask 预处理。
2. E164 专属 bimanual reward gate/score。

历史 E147-E163 的原始 mask 和结果不覆盖。

## 3. Mask 预处理计划

新增脚本：

```text
workspace/core4d/scripts/experiments/E164/build_bimanual_global_masks.py
```

输入：

```text
workspace/core4d/results/E143/contact_masks/<case>/raw_contact_mask_3cm.npz
```

输出：

```text
workspace/core4d/results/E164/bimanual_global_mask_reward/contact_masks/<case>/raw_contact_mask_3cm.npz
```

已确认处理逻辑如下：

```text
selected mask keys = raw_contact_mask_3cm, spider_contact_mask_3cm

for each selected mask key:
    per_person_lr = mask[:, person_idx, :]          # (T, 2)
    global_mask = max(per_person_lr[:, 0], per_person_lr[:, 1])
    global_mask = fill all internal holes between first active and last active
    mask[:, person_idx, 0] = global_mask
    mask[:, person_idx, 1] = global_mask
```

`eval_contact_mask_3cm` 不作为 E164 预处理对象；E164 override 强制：

```text
contact_hdmi_mask_time_axis = spider
```

这样 CEM reward 和 strict eval 都使用同一个 E164 spider global mask，不再走 `auto` 误选 eval mask。

mask 只处理当前 case 的目标 `person_idx`；另一个 person 的 mask 原样保留。

输出需要附带 metadata：

```text
source_mask_path
processed_keys
case_id
person_idx
raw_active_before_L/R
raw_active_after_L/R
spider_active_before_L/R
spider_active_after_L/R
filled_hole_segments
```

同时生成可视化：

```text
diagnostics/masks/<case>_raw_spider_before_after.png
diagnostics/masks/<case>_mask_segments.tsv
```

## 4. Reward 改动计划

新增默认关闭配置，保证旧实验行为不变；E164 override 显式打开双手约束：

```text
surface_band_bimanual_required: false       # default
surface_band_bimanual_score_reduce: "min"   # E164 confirmed
contact_hdmi_bimanual_required: false       # default; E164 confirmed true
```

### 4.1 surfaceBand bimanual 方案

当前逻辑是：

```text
surface_band_sdf = min_sdf(["lh", "rh"], object)
surface_band_score = f(surface_band_sdf)
```

E164 计划改为分别计算左右手：

```text
left_sdf = min_sdf(["lh"], object)
right_sdf = min_sdf(["rh"], object)

left_in_band = -1mm <= left_sdf <= 3mm
right_in_band = -1mm <= right_sdf <= 3mm
both_in_band = left_in_band AND right_in_band

left_score = exp(-abs(left_sdf) / sigma)
right_score = exp(-abs(right_sdf) / sigma)

surface_band_score = 0 if not both_in_band
surface_band_score = reduce(left_score, right_score) if both_in_band
```

已确认 `reduce=min(left_score, right_score)`，因为它会把弱接触手作为瓶颈，比平均值更符合“必须双手同时满足”的要求。

新增诊断字段：

```text
surface_band_left_sdf
surface_band_right_sdf
surface_band_left_score
surface_band_right_score
surface_band_bimanual_gate
surface_band_bimanual_score
```

### 4.2 contact_hdmi bimanual 方案

已确认 `contact_hdmi_rew` 也必须加双手 AND；E164 不能只依赖 global mask 让两只手 active。

E164 计划语义为：

```text
left_active = mask_L > 0
right_active = mask_R > 0
both_active = left_active AND right_active

left_rew/right_rew 仍按原 target proximity 计算
contact_hdmi_rew = 0 when active window expects bimanual contact but not both_active
contact_hdmi_rew = reduce(left_rew, right_rew) when both_active
```

其中 `reduce(left_rew, right_rew)` 默认也按瓶颈手处理，即 `min(left_rew, right_rew)`。

`mask` 非 active 的帧继续按 legacy neutral behavior 忽略，不在非接触阶段额外惩罚；active 窗口内任一手不满足则该 reward term 不给分，不额外加 penalty。

E164 不把 MuJoCo raw contact pair 放进 CEM reward。raw contact 只用于 strict eval 的 RL-safe hard gate。

## 5. Manifest / launcher / evaluator

新增：

```text
workspace/core4d/scripts/experiments/E164/build_bimanual_global_mask_reward_manifest.py
workspace/core4d/scripts/launch/active/run_E164_local.sh
workspace/core4d/scripts/launch/active/run_E164_remote.sh
workspace/core4d/scripts/launch/active/pull_E164_remote_results.sh
workspace/core4d/scripts/eval/runners/eval_E164_bimanual_global_mask_reward.py
workspace/core4d/scripts/eval/wrappers/eval_E164_bimanual_global_mask_reward.sh
```

Manifest 只包含三个 E164 rows，并记录：

```text
case_id
person_idx
source_e163_row
source_mask_path
e164_mask_path
override_path
split
run_status
```

建议三卡并行分配：

| split | case | 理由 |
|---|---|---|
| local-gpu0 | `box004_082_p1` | 主失败 case，便于本地快速检查日志/视频 |
| remote-gpu0 | `box023_person2` | RL 关键回归 case |
| remote-gpu1 | `box026_139_p1` | 泛化检查 |

运行规则：

```text
WAIT_FOR_GPU_IDLE=0
不 kill 其他程序
已有完整产物则 skip
OOM/资源冲突只记录失败并回报
```

## 6. 评测计划

E164 评测至少输出：

```text
workspace/core4d/results/E164/bimanual_global_mask_reward/eval/full/
```

主比较对象：

```text
SPIDER+rubberhand
E161 releaseDecay
E163 narrowSurfaceBand
E164 bimanualGlobalMaskReward
```

主指标仍保持 E162/E163 的 RL-safe 口径：

```text
raw hand_object_physics_contact_in_mask_frac
clean 3mm / clean 5mm contact in mask
physical penetration 3mm / 5mm
geometric penetration 2mm / 5mm
Table4 tracking metrics
fall
```

hard gate：

```text
E164 raw contact in mask 相对同 case SPIDER+rubberhand 下降不能超过 0.05
success_tracked = true
fall = false
Table4 tracking fields 非 NaN
```

已确认：如果 E164 使用重处理后的 global mask 做 reward，则 baseline 的 raw contact hard gate 也必须在同一 E164 global mask 上重算，避免 denominator 不一致。

因此 E164 evaluator 会在评测时对以下方法统一使用 E164 global mask 重算 masked contact/penetration 指标：

```text
SPIDER+rubberhand
E161 releaseDecay
E163 narrowSurfaceBand
E164 bimanualGlobalMaskReward
```

Table4 tracking、fall、非 mask 相关诊断仍按原轨迹和原定义计算。

## 7. 成功判据

### C1: mask 语义正确

三个 case 的 E164 mask 都满足：

```text
L mask == R mask
L/R 均等于 fill_holes(max(original_L, original_R))
raw/spider 处理后的 active segment 可视化已生成
原 E143/E163 mask 文件未被覆盖
```

### C2: reward 语义正确

`config_act.yaml` 和 trajectory diagnostics 必须证明：

```text
surface_band_bimanual_required = true
surface_band_left/right_sdf 存在
surface_band_bimanual_gate 存在
任一手不满足 band 时 surfaceBand 不给分
```

`contact_hdmi_bimanual_required = true` 已确认，`config_act.yaml` 必须能证明该项开启。

### C3: 产物完整

每个 case 必须有：

```text
root npz
trajectory_mjwp_act.npz
config_act.yaml
full mp4
E164 processed mask npz
mask before/after PNG
```

### C4: RL-safe hard gate

三个 case 均需：

```text
raw contact delta vs SPIDER+rubberhand >= -0.05
success_tracked = true
fall = false
```

## 8. 已确认实现口径

E164 按以下已确认口径实现：

```text
mask keys = raw_contact_mask_3cm + spider_contact_mask_3cm
mask hole fill = fill all internal holes between first active and last active
mask person scope = target person_idx only; non-target person unchanged
E164 CEM time axis = spider, no auto eval-mask fallback
reward bimanual contact = reward-level bimanual condition, not MuJoCo raw contact pair
surfaceBand condition = both hands inside [-1mm, 3mm]
surfaceBand 双手 score 聚合 = min(left_score, right_score)
contact_hdmi_rew 也加双手 AND
active window 内任一手不满足 = 该 reward term 不给分；不额外加 penalty
strict eval 使用 E164 global mask 重算所有对比方法的 masked contact/penetration 指标
```

## 9. 执行顺序

按以下顺序执行：

1. 写 E164 mask builder，生成三 case processed masks 和 PNG/TSV 诊断。
2. 加默认关闭的 bimanual reward config 与 reward diagnostics。
3. 写 E164 manifest/override/launcher/pull/evaluator。
4. 静态检查：`py_compile`、`bash -n`、`git diff --check`。
5. 先本地 smoke `box004_082_p1`，确认 config/mask/reward diagnostics。
6. 三卡 full 并行，不 kill 其他程序。
7. pull + strict eval + 写结果 log + 更新 tracker。
