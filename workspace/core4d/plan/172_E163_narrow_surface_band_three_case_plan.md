# E163 — narrow symmetric surfaceBand three-case RL-safe probe

日期：2026-06-14

## 0. 背景

E162 重新评测后确认，E161 `surfaceBandReleaseDecay` 的主要问题不是 aggregate clean 指标，而是
raw in-mask physical contact 退化。典型 case 是：

```text
box023_person2:
  SPIDER+rubberhand raw contact = 0.9077
  E161 releaseDecay raw contact = 0.7538
  delta = -0.1538
```

这会直接破坏下游 RL：RL 主要需要 hand-object contact gate 存在，对穿透改善不敏感。因此 E163 的目标不是
继续压 penetration，而是先修复 surfaceBand 对 raw contact 的欠约束。

E159-E161 当前 surfaceBand 的有效区间太宽：

```text
-1mm <= sdf <= +30mm
score = exp(-max(sdf, 0) / sigma)
```

它会奖励离物体很远的“接近表面”，并且对 `-1mm..0mm` 的轻微穿入给满分。这会让 CEM 偏向“干净但接触不稳定”的
解，尤其伤害 `box023_person2` 这种 baseline raw contact 很高的 case。

## 1. 实验目标

E163 只做一个小而硬的三 case probe：

```text
base method = E161 surfaceBandReleaseDecay
change = narrow symmetric surfaceBand
cases = box023_person2, box021_029_p2, box004_083_p2
resource = local-gpu0 + remote-gpu0 + remote-gpu1
```

目标判断很简单：

1. `box023_person2` raw contact 必须从 E161 releaseDecay 明显恢复。
2. 三个 case 都不能相对 `SPIDER+rubberhand` 触发 raw contact hard regression。
3. tracking/fall 不能回退。
4. penetration/release false contact 作为 secondary diagnostics，不能覆盖 raw contact 失败。

## 2. 方法定义

### 2.1 E163 主配置

沿用 E161 `surfaceBandReleaseDecay` 的 hand gate、posture gate、release decay 和 contact mask 设置，只替换
surfaceBand 的区间和 score：

```text
surface_band_min_sdf_m = -0.001
surface_band_width_m = 0.003      # 沿用现有变量名；这里表示 upper bound = +3mm
surface_band_sigma = 0.0015
surface_band_score_mode = symmetric_abs

surface_band_score = exp(-abs(sdf) / sigma)
in_band = (-0.001 <= sdf <= 0.003)
score = in_band ? surface_band_score : 0
```

`sigma=0.0015m` 是 E163 主值，不先做 sweep：

```text
sdf = 0mm   -> score = 1.000
sdf = -1mm  -> score ~= 0.513
sdf = +3mm  -> score ~= 0.135
```

这个设计表达的是：0mm 附近最优；轻微穿入不是满分；3mm 仍有弱信号；超过 3mm 不再奖励。

### 2.2 保持不变的 E161 设置

```text
surface_band_rew_scale = 1.5
surface_band_penalty_scale = 0.0
surface_band_decay_frac = 0.15

contact_hdmi_mask_source = core4d_3cm
contact_hdmi_mask_path = <case-specific raw_contact_mask_3cm.npz>
contact_hdmi_mask_person_idx = <case-specific person_idx>
contact_hdmi_mask_time_axis = auto

cem_hand_gate_enabled = true
cem_hand_gate_min_sdf_m = -0.010
cem_hand_gate_max_violation_pct = 0.10
cem_hand_gate_hard_floor_m = -0.020

cem_posture_gate_enabled = true
cem_posture_gate_mean_z_err_m = 0.10
cem_posture_gate_terminal_z_err_m = 0.12
cem_posture_gate_max_z_drop_m = 0.18
cem_posture_gate_terminal_frac = 0.15
cem_posture_gate_min_valid_frac = 0.05
cem_posture_gate_fallback_lambda = 5.0
```

实现时必须保证 E158-E161 可复现：旧实验默认仍使用 `one_sided` 公式，只有 E163 override 打开
`symmetric_abs`。

## 3. Claims

| Claim | 判据 |
|---|---|
| C1: narrow band 能恢复 `box023_person2` raw contact | `box023_person2` raw contact 至少超过 E161 releaseDecay `0.7538`；真正 pass 需要 `>=0.8577` |
| C2: 不再触发 RL contact hard regression | 三个 case 的 raw contact delta vs `SPIDER+rubberhand` 均 `>= -0.05`，逐 case 判，不看 aggregate |
| C3: 不牺牲基础稳定性 | 三个 case `success_tracked=true` 且 `fall=false` |
| C4: penetration 清理不完全丢失 | 物理/几何 penetration 作为诊断；若 raw contact pass 但 penetration 回到 rubberhand 水平，需要标记为 tradeoff，不直接 promote |
| C5: 实现不污染历史实验 | E158-E161 默认 score 行为不变；E163 通过显式 config flag 使用新公式 |

### 3.1 硬阈值

E163 使用 E162 分层交集口径的 `SPIDER+rubberhand` 作为 baseline。三个 case 的 raw contact hard gate
直接写死，避免再次出现 aggregate 好看但 RL 不可用：

| case | `SPIDER+rubberhand` raw contact | E163 pass 下限 |
|---|---:|---:|
| `box023_person2` | 0.9077 | 0.8577 |
| `box021_029_p2` | 0.3818 | 0.3318 |
| `box004_083_p2` | 0.5323 | 0.4823 |

实现/产物 gate：

```text
config_act.yaml must contain:
  surface_band_min_sdf_m = -0.001
  surface_band_width_m = 0.003
  surface_band_sigma = 0.0015
  surface_band_score_mode = symmetric_abs
  contact_hdmi_mask_source = core4d_3cm

each case must have:
  root npz
  trajectory_mjwp_act.npz
  config_act.yaml
  full mp4
```

缺任何一个产物，不进入结论。

## 4. 运行矩阵

只新跑 E163 三条 full CEM。

| split | GPU | case | 理由 |
|---|---|---|---|
| local | GPU0 | `box023_person2` | 关键 RL failure case，优先本地看日志和视频 |
| remote | GPU0 | `box021_029_p2` | E159/E160 posture/fall 风险 case |
| remote | GPU1 | `box004_083_p2` | release false / clean contact 风险 case |

先做一次轻量 smoke 只验证 plumbing，不作为效果结论：

```text
smoke case = box023_person2
checks = config_act 写入 symmetric_abs / narrow band, trajectory 有 surface_band_score/sdf/rew diagnostics
```

smoke 通过后立即三卡并行 full。

## 5. 文件规划

建议新增：

| 类型 | 路径 |
|---|---|
| manifest builder | `workspace/core4d/scripts/experiments/E163/build_narrow_surface_band_manifest.py` |
| local launcher | `workspace/core4d/scripts/launch/active/run_E163_local.sh` |
| remote launcher | `workspace/core4d/scripts/launch/active/run_E163_remote.sh` |
| remote pull | `workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh` |
| eval runner | `workspace/core4d/scripts/eval/runners/eval_E163_narrow_surface_band.py` |
| eval wrapper | `workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh` |
| result root | `workspace/core4d/results/E163/narrow_surface_band/` |
| logs | `logs/core4d/E163_narrow_surface_band/` |

代码改动建议：

```text
spider/config.py
  add surface_band_score_mode: str = "one_sided"

spider/simulators/mjwp.py
  if mode == "one_sided":
      score = exp(-clamp(sdf, min=0) / sigma)
  elif mode == "symmetric_abs":
      score = exp(-abs(sdf) / sigma)
```

## 6. 评测设计

E163 不重新跑 rubberhand baseline。baseline 和历史方法使用 E162 分层交集里已经整理出的同 case 轨迹。

主比较方法：

| 方法 | 作用 |
|---|---|
| `OmniRetarget` | SPIDER 输入参考 |
| `SPIDER+rubberhand` | 统一 baseline |
| `+gateA` | 当前 raw contact 相对安全的 clean8 参考 |
| `E155 decay` | release smooth transition 参考 |
| `E158 surfaceBand-A` | 宽 band + penalty 的失败参考 |
| `E159 surfaceBand-A2` | 宽 band、无 posture 的失败参考 |
| `E160 postureRerankA` | posture 修 fall 后的参考 |
| `E161 releaseDecay` | E163 直接 base |
| `E163 narrowSurfaceBand` | 新方法 |

主表只放少量指标：

```text
raw接触 = hand_object_physics_contact_in_mask_frac
clean3接触 = hand_object_physics_contact_3mm_in_mask_frac
物理穿透3mm = hand_object_physics_penetration_3mm_frame_frac
几何穿透2mm = hand_geom_penetration_2mm_frac
release误接触3mm = hand_object_release_false_contact_3mm_frac
success_tracked / fall
SPIDER Table4 tracking: joint deg, eef pos/orient, root pos/orient, object pos/orient
```

失败判据：

```text
if raw_contact_delta_vs_SPDRubberhand < -0.05:
    case_status = contact_regression_fail
elif success_tracked is false:
    case_status = tracking_fail
elif fall is true:
    case_status = fall_fail
elif any Table4 tracking field is NaN:
    case_status = tracking_metric_missing
else:
    case_status = pass
```

## 7. 预期结果与解释

### 7.1 理想结果

```text
box023_person2 raw contact:
  E161 releaseDecay = 0.7538
  E163 target >= 0.8577

all 3 cases:
  raw contact delta vs rubberhand >= -0.05
  tracked 3/3
  fall 0/3
```

如果成立，E163 才有资格进入 clean8 probe；不直接升级为默认候选。

### 7.2 可能失败模式

| 失败模式 | 含义 | 下一步 |
|---|---|---|
| raw contact 仍低 | surfaceBand 本身不是 raw contact 的充分代理 | 增加显式 raw contact/hold-contact 项，而不是继续调 band |
| reward 太 sparse，tracking 变差 | `sigma=1.5mm` 或 `+3mm` 上界过窄 | 只在三 case 上加一个 E163b `sigma=2mm` 或 `upper=5mm` 对照 |
| raw contact 恢复但 penetration 回升 | narrow band 把物理清理能力拿掉了 | 保留为 tradeoff，不 promote；考虑把 hand gate 或 shallow penetration penalty 分离 |
| box004 release false 回来 | narrow band 恢复接触但尾部吸附也恢复 | 继续保留 E161 releaseDecay，必要时加强 release-window gate |

## 8. 执行顺序

1. 实现 config flag 和 E163 manifest/launcher/evaluator。
2. 本地 smoke `box023_person2`，只检查 plumbing。
3. 启动 full：

```bash
bash workspace/core4d/scripts/launch/active/run_E163_local.sh full
bash workspace/core4d/scripts/launch/active/run_E163_remote.sh full
```

4. 回收远程结果：

```bash
bash workspace/core4d/scripts/launch/active/pull_E163_remote_results.sh full
```

5. 严格评测：

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E163_narrow_surface_band.sh full
```

6. 写 log，更新 tracker。若三 case 未通过 raw contact hard gate，不进入 clean8。
