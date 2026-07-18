# E169 实验计划：Lower-body / Object Penetration 三轴因子消融

日期：2026-07-18

实验方向：`core4d`

Phase：32

状态：已执行完成；结论见 `log/228_E169_lowerbody_object_factorial_results.md`

---

## 0. 决策摘要

E169 的首要目标是修复 E168 Box021 中最主要、最剧烈的失败：下肢穿箱、踩箱和借箱支撑。用户提出的三个方向正确，但三者作用不同，不能视为互相替代：

| 轴 | 作用 | 单独使用的主要风险 |
|---|---|---|
| `P`：下肢-物体 MuJoCo 物理碰撞 | 改变动力学，使下肢不能无代价穿过箱体 | 可能把“穿箱”变成真实踩箱、箱体被踢飞或接触冲量导致失稳 |
| `R`：下肢-物体 soft reward penalty | 在候选之间软排序，偏好更大 clearance | 可被 hand/object tracking reward 换取；E115/E117 已出现避腿但丢手或塌姿态 |
| `G`：CEM lower-body candidate gate | 定义硬可行域，过滤持续贴箱、踩箱和深穿透候选 | 过严时可能没有有效候选并长期 fallback |

因此 E169 不做三个孤立 arm，而采用完整 `2×2×2` 因子设计，测量 `P/R/G` 主效应及交互。冻结并复用 4 条 E168 baseline，只新跑其余 7 arm：

```text
4 cases × (8 factorial cells - 1 reused baseline) = 28 new full CEM runs
```

执行资源固定为远程 A100 `GPU 0,1,2,3`。不使用本地 GPU、A6000 或 A100 `4,5,6,7`。

---

## 1. Context

### 1.1 E168 的核心失败事实

E168 Box021 人工审查为 `13 USE / 15 DO_NOT_USE`，失败率 `53.6%`。失败不是 release threshold 过严造成的假象：

- `14/15` 失败轨迹至少出现 lower-body/object 非法接触或非法支撑症状；
- `9/15` 以该问题为首要失败类型；
- 失败组 `leg_penetration_frac` 中位数为 `0.378`，可用组为 `0.048`；
- `9/15` 失败 case 的现有 CEM fallback mean `<=0.10`；
- `033_p1` 明显跨箱并站箱，但 fallback mean 仅约 `0.001`。

这说明许多失败不是“CEM 找不到合法解”，而是当前 objective/gate 将任务上非法的动作当作合法解。增加 sample 数或 iteration 数不能修复错误的可行域。

### 1.2 E168 effective config 的缺口

4 条代表 case 的 E168 effective config 一致：

```yaml
leg_object_penalty_scale: 0.0
leg_object_penalty_geom_names: []

cem_safety_gate_enabled: true
# 仅 head/torso/pelvis/shoulder/elbow，不包含 lower body

foot_slip_enabled: false
foot_ground_enabled: false
cem_smooth_enabled: false
```

scene 中下肢 collision geoms 存在，但继承 `contype=0, conaffinity=0`。显式 MuJoCo pair 只有：

- 下肢/脚与 floor；
- 手与 object；
- object 与 floor；
- **没有任何下肢/脚与 object pair**。

所以 E168 视频里的“站在箱上”并不表示箱体真正物理支撑了机器人，而是 actuator 控制下的几何重叠。离线 `leg_penetration_frac` 只是生成后的 SDF 评测指标，不是运行时 constraint。

### 1.3 历史实验边界

| 实验 | 结论 | E169 约束 |
|---|---|---|
| E115 | uniform leg penalty 可消除部分 lower-body interference，但主 case hand contact `65.3% -> 12.0%`；增强 contact 后穿透又回来；`0` release candidate | 不再做单纯 reward weight sweep |
| E117 | phase/state-gated leg penalty 仍通过丢 hand contact 或 posture 解决冲突；`0` release candidate | `R-only` 只保留为因果对照，不作为预设答案 |
| E119 | upright/support guard 说明姿态与 support 约束需要独立处理 | E169 只回答 lower-body/object，不顺带调 root/support |
| E152 | dedicated hand gate 能在不明显损失 near contact 时大幅降低深穿透 | lower-body gate 也应是独立语义轴 |
| E153 | hand gate 的 `min_sdf/hard_floor/max_violation` 解耦能避免 fallback collapse | lower-body gate 必须输出独立 valid/fallback/violation 诊断 |
| E168 | lower-body/invalid support 覆盖 `14/15` 失败，且低 fallback 失败大量存在 | 优先补硬可行性，而不是先加 smooth 或算力 |

### 1.4 固定不变的 E168 基线

除 `P/R/G` 三个轴外，所有 arm 必须继承相同的 E168 `E167A_zOnlyBody` effective config、输入轨迹、raw contact mask、rubber hand scene、seed、CEM budget 和 horizon。尤其保持：

```text
retarget_variant_id = omnirt_v1
target_variant_id = ref_fk
hand_collision_variant_id = rubber_hull
spider_method_id = E167A_zOnlyBody
seed = 0
foot_slip = off
foot_ground = off
cem_smooth = off
reference retiming = off
```

E169 不重跑 OmniRetarget，不更换 target route，也不修改 E168 原 scene 或结果。

---

## 2. Claims

| Claim | 最低证据 |
|---|---|
| C0：三轴 plumbing 正确且互相独立 | 8 个 cell 的 effective config、scene SHA 和 runtime diagnostics 与位编码一致；非目标配置无漂移 |
| C1：`P` 确实改变下肢-物体动力学 | P-on scene 编译后 16 个 lower-body/object pair 全部存在；接触时有 MuJoCo contact/impulse，P-off 无对应 pair |
| C2：`R` 是 soft preference，不是充分硬约束 | 固定 `scale=2.0` 后报告 reward/clearance 变化及 contact/posture trade-off；不以单个视觉改善宣称成功 |
| C3：`G` 能过滤非法候选且不是 fallback-only | 输出独立 leg gate valid fraction、selected valid fraction、violation 和 fallback；`fallback_mean<=0.10` 且 last-iteration valid fraction `>=0.05` |
| C4：三轴至少有一个组合修复主失败 | 3 条失败 case 中至少 `2/3` 同时通过 lower-body、hand contact、body-z、fall 和视觉标准；目标为 `3/3` |
| C5：修复不是通过丢手、塌姿态或踢飞物体实现 | 通过 raw hand contact、body-z、root/EEF tracking、object motion 与视频联合审查 |
| C6：对可用 control 不产生明显回归 | `023_p2` 满足 control guard，且没有新增下肢接触、踩箱、object kick 或姿态失稳 |
| C7：能够区分 `P/R/G` 主效应与交互 | 对每个连续指标报告 case-level factorial contrasts；对 3 条失败和 1 条 control 分开报告，不做伪显著性检验 |

---

## 3. 代表 Case

### 3.1 固定 case 集

| GPU | Case | 角色 | E168 关键证据 | 选择原因 |
|---:|---|---|---|---|
| 0 | `box021_20231018_033_p1` | pure loophole | leg penetration `0.302`；root error `41.4cm`；hand contact `0.222`；fallback `0.001` | 最清楚的 reward-valid 非法捷径，验证 gate 是否补上错误可行域 |
| 1 | `box021_20231018_032_p1` | high-dynamic failure | leg penetration `0.427`；jerk `1328.8`；ankle acc `108.4`；hand contact `0.829` | 验证碰撞约束在高速/激烈纠错下是否导致 object kick 或失稳 |
| 2 | `box021_20231020_020_p2` | isolated lower-body failure | leg penetration `0.419`；hand contact `0.814`；body-z p95 `0.145m` | 手部接触和 body-z 尚可，最适合隔离 lower-body 修复是否有效 |
| 3 | `box021_20231020_023_p2` | accepted regression control | 用户 `USE/NO_ISSUE`；leg penetration `0.055`；root error `6.5cm`；hand contact `0.788` | 防止算法只会修坏例，同时破坏本来可用的轨迹 |

### 3.2 代表性边界

本批是 mechanism probe，不声称覆盖全部 Box021 分布：

- 3 条失败覆盖低-fallback loophole、高动态失败、单一 lower-body 失败；
- 1 条 control 保护已可用轨迹；
- 没有加入唯一 fall case、非 Box021 object 或多 seed；
- E169 的结论只能决定三轴机制和下一步默认候选，不能直接宣称全量 recall 提升。

---

## 4. `2×2×2` 因子矩阵

### 4.1 Cell 定义

| Cell | P | R | G | Scene | 用途 | Full run |
|---|:---:|:---:|:---:|---|---|---|
| `B0` | 0 | 0 | 0 | frozen E168 rubber scene | 原始基线 | 复用 E168，不重跑 |
| `P` | 1 | 0 | 0 | E169 physical sidecar | 物理碰撞主效应 | 新跑 |
| `R` | 0 | 1 | 0 | frozen no-P snapshot | soft penalty 主效应 | 新跑 |
| `G` | 0 | 0 | 1 | frozen no-P snapshot | hard gate 主效应 | 新跑 |
| `PR` | 1 | 1 | 0 | E169 physical sidecar | physics × reward | 新跑 |
| `PG` | 1 | 0 | 1 | E169 physical sidecar | physics × gate | 新跑 |
| `RG` | 0 | 1 | 1 | frozen no-P snapshot | reward × gate | 新跑 |
| `PRG` | 1 | 1 | 1 | E169 physical sidecar | 完整组合 | 新跑 |

每条结果必须记录：

```text
case_id, cell_id, p_enabled, r_enabled, g_enabled,
base_scene_sha256, effective_scene_sha256,
trajectory_sha256, mask_sha256, effective_config_sha256,
seed, result paths, remote host, gpu, start/end time
```

### 4.2 Factorial contrast

连续指标对每个 case 计算：

```text
P main = mean(P-on cells) - mean(P-off cells)
R main = mean(R-on cells) - mean(R-off cells)
G main = mean(G-on cells) - mean(G-off cells)

PR / PG / RG / PRG interaction = 对应标准 effect-coded contrast
```

只报告 per-case contrast、3 个失败 case 的方向一致性和 control 结果。本批 `n=4` 不做显著性检验，也不把 control 与 failure 混成一个均值掩盖回归。

---

## 5. 三轴实现契约

### 5.1 `P`：下肢-箱体显式物理碰撞

#### Geom 集

三轴统一使用同一组 16 个 lower-body geoms：

```yaml
- left_hip_collision
- right_hip_collision
- left_thigh_collision
- right_thigh_collision
- left_shin_collision
- right_shin_collision
- left_linkage_brace_collision
- right_linkage_brace_collision
- lf0
- lf1
- lf2
- lf3
- rf0
- rf1
- rf2
- rf3
```

#### Pair 契约

不得修改共享 base scene 的 `contype/conaffinity`，以免引入 robot self-collision 或改变其他 contact。对每个 case 从 E168 rubber scene 生成 E169-scoped physical sidecar，仅新增上述 16 个 geom 到 `object_collision` 的显式 pair：

```xml
<pair name="E169_<lower_geom>_object"
      geom1="<lower_geom>"
      geom2="object_collision"
      solref="0.008 1"
      margin="0"
      gap="0"
      condim="1" />
```

固定 `condim=1`，第一轮只引入法向阻挡，不同时增加摩擦/扭转维度。这样仍可能产生真实支撑或踢箱，因此 `P-only` 是必要诊断，不预设为生产答案。

scene builder 必须使用 XML parser，禁止字符串拼接；对 pair name、geom 存在性、重复 pair、pair 数量和 physics fingerprint 做断言。

#### P 轴 preflight

每条 scene 在 GPU launch 前必须通过：

1. E168 input scene SHA 与 imported snapshot 一致；
2. P-off 的 lower-body/object pair 数为 `0`，P-on 为 `16`；
3. MuJoCo compile 成功；
4. 初始姿态和 reference 前几帧没有超过 `5mm` 的 lower-body/object 初始深重叠；
5. 单步/短 rollout 无 NaN、contact buffer overflow 或异常大 object impulse；
6. 除新增 `<pair>` 外，P scene 的 XML 语义 diff 为零。

若存在初始深重叠，停止对应 P arm 并记录 blocker；不得用瞬移、临时关闭碰撞或 warm-up 穿模静默绕过。

### 5.2 `R`：固定 soft lower-body penalty

复用现有 `leg_object_penalty_*` plumbing，不新增第二套 reward：

```yaml
leg_object_penalty_scale: 2.0
leg_object_penalty_margin_m: 0.02
leg_object_penalty_geom_names: <16-geoms-above>
leg_object_penalty_geom_ids: []
leg_object_penalty_gate_source: always
leg_object_penalty_start_eval_time: 0.0
leg_object_penalty_end_eval_time: 999.0
```

设计约束：

- E169 只使用 `scale=2.0`，不做 `2/4/8` 权重 sweep；
- 不提高 `contact_hdmi_gain` 补偿 reward 冲突，否则多出第四个轴；
- R-off 保持 `scale=0.0` 且 geom list 为空；
- `leg_object_penalty_mean/max/gate` 必须写入每步 diagnostics。

### 5.3 `G`：独立 lower-body candidate gate

不要把 lower-body geoms 追加到现有 `cem_safety_gate_geom_names`。现有 body gate、hand gate 与 lower-body gate 的语义、阈值和健康度不同，合并后无法判断哪一类约束导致 fallback。

新增独立字段：

```yaml
cem_leg_gate_enabled: true
cem_leg_gate_geom_names: <16-geoms-above>
cem_leg_gate_geom_ids: []
cem_leg_gate_min_sdf_m: 0.005
cem_leg_gate_max_violation_pct: 0.02
cem_leg_gate_hard_floor_m: -0.005
cem_leg_gate_min_valid_frac: 0.02
cem_leg_gate_fallback: least_violation
```

候选有效性定义：

```text
valid = (minimum lower-body/object SDF >= -0.005m)
        AND
        (fraction[SDF < +0.005m] <= 0.02)
```

解释：

- `+5mm` 是 semantic clearance band，用于把持续贴箱/踩箱也视为非法，而不只拦深穿透；
- 最多 `2%` 的短暂违反用于容忍离散时间和 contact embedding；
- 任意时刻深于 `-5mm` 直接 invalid；
- `least_violation` 只保证搜索不中断，不代表该候选合格；持续 fallback 的 arm 按 gate health failure 处理。

必须新增并分别保存：

```text
leg_gate_valid_frac
leg_gate_selected_valid_frac
leg_gate_fallback_used
leg_gate_violation_pct
leg_gate_min_sdf_min_m
leg_gate_min_sdf_p05_m
selected_leg_gate_valid
```

G-off 必须保持全部字段为 disabled/empty，并验证 sampling 数值与 E168 baseline 路径一致。

---

## 6. Scene 与输入快照

E169 启动前必须创建：

```text
workspace/core4d/results/E169/s0_environment/
workspace/core4d/results/E169/scene_snapshot/no_physics/<case_id>/
workspace/core4d/results/E169/scene_snapshot/lowerbody_physics/<case_id>/
```

每个 case 保存：

- E168 rubber scene 原文件；
- E169 physical sidecar；
- trajectory、mask、object asset 的路径与 SHA256；
- XML canonical fingerprint、pair audit 和 semantic diff；
- source E168 result/config/video 的 immutable reference；
- Git commit、Holosoma commit、MuJoCo version 和 builder version。

活跃 scene sidecar 必须按 experiment skill 的 scene snapshot 规则纳入 git；不得只依赖 `example_datasets/` 下被忽略的工作副本或 mtime。

---

## 7. 需要实现的文件

当前阶段只规划，以下文件尚未创建或修改。

| # | Canonical path | 计划改动 |
|---:|---|---|
| 1 | `spider/config.py` | 增加默认关闭的 `cem_leg_gate_*` 配置；旧实验数值行为不变 |
| 2 | `spider/optimizers/sampling.py` | 增加独立 leg gate、effect-coded diagnostics 和 fallback 状态 |
| 3 | `spider/simulators/mjwp.py` | 暴露 leg gate/reward 与 lower-body/object physical contact diagnostics |
| 4 | `workspace/core4d/scripts/experiments/E169/build_lowerbody_collision_scenes.py` | 用 XML parser 创建/审计 P sidecar 与 scene snapshot |
| 5 | `workspace/core4d/scripts/experiments/E169/build_lowerbody_object_factorial_manifest.py` | 固化 4 case × 8 cell、输入 SHA、override 和 A100 queue |
| 6 | `workspace/core4d/scripts/experiments/E169/variants.tsv` | 32 个分析 cell；标记 4 个 reused B0 与 28 个 new full |
| 7 | `examples/config/override/core4d_E169_*.yaml` | 生成 case/cell override，除 P/R/G 外与 E168 effective config 一致 |
| 8 | `workspace/core4d/scripts/launch/active/run_E169_remote_a100.sh` | fixed allowlist preflight、canary/full、4 个串行 worker |
| 9 | `workspace/core4d/scripts/launch/active/pull_E169_remote_a100_results.sh` | 只回收 execution manifest 登记的 artifact 并校验 SHA |
| 10 | `workspace/core4d/scripts/launch/active/watch_E169_remote_a100.sh` | 可选 watcher；只监控/回收，不启动额外 GPU worker |
| 11 | `workspace/core4d/scripts/eval/runners/eval_E169_lowerbody_factorial.py` | E168 对齐指标、leg gate health、物理 contact/impulse 和 factorial contrasts |
| 12 | `workspace/core4d/scripts/eval/wrappers/eval_E169_lowerbody_factorial.sh` | smoke/full eval 入口 |
| 13 | `workspace/core4d/scripts/eval/reports/gen_E169_lowerbody_factorial_xlsx.py` | 完整 per-cell 指标、contrasts、gate health、人工审核表 |
| 14 | `workspace/core4d/scripts/experiments/E169/render_factorial_results.py` | 本地离线渲染 28 条新 full 结果与 case-wise montage |

所有新字段默认关闭；必须先跑旧 gate 单元测试和 E168 config replay regression，确认未改变历史行为。

---

## 8. A100 执行计划

### 8.1 固定资源契约

远程 profile：

```text
host = tianyiyun-A100
root = /home/dataset-assist-0/xiayb/workspace/spider
ALLOWED_GPUS = 0 1 2 3
workers = 4
one serial worker per GPU
render = false
```

用户最新指令固定使用 `0,1,2,3`，并明确允许与现有程序叠加运行。启动前必须：

1. 确认 GPU `0,1,2,3` 均在本轮允许集合内；
2. 保存全机 `nvidia-smi` 和 compute process 快照，用于解释运行时波动；
3. tmux 启动前再次检查四卡；
4. 不以显存占用或已有 compute process 阻止 launch，不 kill/暂停/修改任何现有程序；
5. 不自动缩成 3 卡，不改用 `4,5,6,7`，不 kill 或抢占其他任务。

A100 在 E168 中 `EGL/OSMesa/GLFW` 均无法渲染，因此 E169 只做 compute。`save_video=false`；结果回收后在本地离线渲染。

### 8.2 固定分片

| GPU | Case | Full queue（串行） |
|---:|---|---|
| 0 | `033_p1` | `P -> R -> G -> PR -> PG -> RG -> PRG` |
| 1 | `032_p1` | `P -> R -> G -> PR -> PG -> RG -> PRG` |
| 2 | `020_p2` | `P -> R -> G -> PR -> PG -> RG -> PRG` |
| 3 | `023_p2` | `P -> R -> G -> PR -> PG -> RG -> PRG` |

每张卡同一时刻只能有一个 E169 worker。每个 run 的 root NPZ、outdir NPZ、config、log 和状态文件路径必须唯一。

### 8.3 Canary

先在四张卡各跑一条 `PRG` smoke：

| GPU | Canary |
|---:|---|
| 0 | `033_p1/PRG` |
| 1 | `032_p1/PRG` |
| 2 | `020_p2/PRG` |
| 3 | `023_p2/PRG` |

Canary 只验证：

- scene compile；
- 16 个 pair 和三个 axis 位正确；
- GPU/runtime 能完整退出；
- 无 NaN、OOM、contact buffer error；
- root NPZ、outdir NPZ、config 和 log 齐全。

**Canary 不计算质量门槛，不要求量化改善，不作为 full 结果复用。** 四条都跑通后才启动 28 条 full。

### 8.4 计划命令

实现阶段应固化并使用以下入口，不在 shell 中临时拼接生产命令：

```bash
# local: build and preflight only
python workspace/core4d/scripts/experiments/E169/build_lowerbody_object_factorial_manifest.py \
  --stage preflight

# remote A100 GPU0-3: runtime canary
bash workspace/core4d/scripts/launch/active/run_E169_remote_a100.sh canary
bash workspace/core4d/scripts/launch/active/pull_E169_remote_a100_results.sh canary

# remote A100 GPU0-3: 28 new full rows
bash workspace/core4d/scripts/launch/active/run_E169_remote_a100.sh full
bash workspace/core4d/scripts/launch/active/pull_E169_remote_a100_results.sh full

# local evaluation and rendering
bash workspace/core4d/scripts/eval/wrappers/eval_E169_lowerbody_factorial.sh full
python workspace/core4d/scripts/experiments/E169/render_factorial_results.py --stage full
```

---

## 9. 评测与成功标准

### 9.1 主指标

沿用 E168/E167A 口径，不重新发明 release metric：

| 类别 | 指标 | 阈值/用途 |
|---|---|---|
| Lower body | `leg_penetration_frac` | **主失败指标，`<=0.10`** |
| 视觉 | lower-body/object overlap/support | 无可见穿插、踩箱或借箱支撑 |
| Hand contact | raw in-mask physics contact | `>=0.50` |
| Body-z | `body_z_err_p95_m` | `<=0.20m`；peak 只作诊断 |
| Fall | `fall_flag` | `false` |
| Hand penetration | >3mm physics penetration frame frac | `<=0.30`，保持 E168 口径 |
| Tracking | root/EEF/object pos+ori error | 与 B0 报 delta，防止用整体偏离换 clearance |
| Motion health | trackbody/ankle accel+jerk | 诊断 collision impulse 和激烈纠错，不在 4 case 上拟新阈值 |
| Physics | lower-body contact/impulse、object acc/speed | 诊断踩箱与 object kick |
| Gate health | dedicated leg valid/fallback/violation | 防止 hard gate 退化为 fallback-only |

### 9.2 失败 case 成功条件

`033_p1`、`032_p1`、`020_p2` 的某个 cell 必须同时满足：

```text
leg_penetration_frac <= 0.10
raw hand contact in mask >= 0.50
body_z_err_p95_m <= 0.20
fall_flag = false
无可见下肢-箱体穿插、踩箱、借箱支撑
无明显 object kick / 爆姿 / 高频碰撞抖动
```

E169 mechanism success 的最低门槛是同一 cell 在 `2/3` 失败 case 通过；目标是 `3/3`。不同 case 各自挑不同 cell 只能说明 per-case rescue，不能形成统一默认方法。

### 9.3 Control guard

`023_p2` 除统一 hard gates 外，还要求相对 B0：

```text
hand contact delta >= -0.10
root position error delta <= +5cm
EEF position error delta <= +5cm
leg_penetration_frac <= 0.10
no new lower-body contact/support or object kick
```

### 9.4 Gate health

所有 G-on cell 必须分别满足：

```text
cem_leg_gate_fallback_used_mean <= 0.10
cem_leg_gate_valid_frac_last_iter_mean >= 0.05
selected_leg_gate_valid = true for accepted trajectories
```

现有 body/hand/posture gate 的 fallback 不能代替 leg gate 健康度。若 leg gate 频繁 fallback，即使最终离线指标偶然通过，也不能把该 cell 提升为默认配置。

### 9.5 人工视频审查

本地为 28 条新 full 结果生成：

- ref/sim side-by-side 视频；
- 每 case 8-cell montage；
- 接触发生前、最大 penetration、最大 impulse、末帧关键帧；
- 统一人工字段：`USE / DO_NOT_USE`、`NO_ISSUE / MINOR_ACCEPTABLE / MAJOR_ISSUE`、失败 taxonomy。

人工审查优先判断“是否还在踩箱/借箱支撑”，不能只看最终 `leg_penetration_frac`。

---

## 10. 决策规则

| 结果模式 | 解释 | 下一步 |
|---|---|---|
| `G`/`RG` 最好且 gate 健康 | 硬可行域是主要缺口，物理 pair 非必要 | 以最小有效 gate 方案进入跨 case 验证 |
| `PG`/`PRG` 明显优于无 P 组合 | physical blocking 与 gate 有正交贡献 | 保留 P，并重点检查 object kick/control 回归 |
| `P` 降穿透但出现踩箱或 object kick | 物理碰撞有效但不具语义充分性 | P 不单独推广，必须与 G 组合 |
| `R` 降穿透但 contact/posture 回归 | 重现 E115/E117 trade-off | 不调更大权重；R 不作为默认 |
| `PRG` 修复失败但 control 回归 | 完整约束过强或 physical impulse 有害 | 不进入全量；根据 contrasts 确定移除哪一轴 |
| G-on 长期 fallback | gate clearance/threshold 与可达域不匹配 | E169 判 gate contract 失败；后续另立 calibration，不在本轮结果后改阈值 |
| 所有 cell 均无法在同一配置通过 `2/3` | lower-body 不是唯一充分根因 | 转向 stance-foot/root support 或 reference feasibility，不扩大 E169 全量 |

不允许根据 full 结果临时改变 reward scale、gate threshold 或 pair 参数后仍归入同一 E169 factorial。任何参数修订必须新建后续实验/版本。

---

## 11. 风险与缓解

| 风险 | 影响 | 计划控制 |
|---|---|---|
| 初始 lower-body/object overlap | 开启 P 后第一步产生爆炸冲量 | CPU/MuJoCo preflight 检查初始和 reference 前几帧；深于 `5mm` 直接 block |
| 真实踩箱替代几何穿箱 | P 指标改善但语义仍错误 | G 使用正 clearance band；视频检查踩箱/借支撑 |
| 箱体被脚踢飞 | object tracking、hand contact 和姿态恶化 | 记录 contact impulse/object acc/speed；control guard；P 不预设为答案 |
| Soft reward 丢手或塌姿态 | 重现 E115/E117 失败 | 固定单一 scale，保留 R-only 因果对照，同时检查 contact/body-z/root |
| Gate 无有效候选 | fallback-only 伪 hard constraint | 独立 leg gate health；fallback 超阈值即失败，不把输出当 release |
| 共享 gate 诊断混淆 | 无法定位 body/hand/leg 哪个约束生效 | dedicated `cem_leg_gate_*`，不扩写 safety gate geom list |
| contact 数增加导致 buffer overflow/OOM | Canary 或 full 中断 | Canary 检查；如只需提高容量，所有 E169 new arm 使用同一容量并记录，不改变动力学参数 |
| scene 漂移或远程副本不一致 | 因子对比不可复现 | 双 scene snapshot、SHA、semantic XML diff、remote preflight |
| A100 与其他程序叠加 | runtime 变慢或 OOM | 固定只用 0-3 并保存启动快照；按用户指令允许叠加，绝不 kill 其他程序；失败时记录现场后再决定恢复策略 |
| A100 无渲染 | 无法在远程做视觉 QA | compute-only，pull 后本地离线 render |
| 4 case 过拟合 | 机制在全量 Box021 上不成立 | 本轮只做 mechanism selection；通过后另做跨 sequence/object 验证 |

---

## 12. 明确不在 E169 范围内

- 不启动或修改下游 RL；
- 不做 RL export；
- 不重跑 OmniRetarget，不切换 `omnirt_v2`；
- 不启用 foot-slip、foot-ground、stance-foot、support polygon；
- 不新增 root roll/pitch、terminal upright 或 stability reward/gate；
- 不启用 CEM smooth/B2 postprocess；
- 不做 reference retiming、adaptive horizon 或 reach projection；
- 不改变 hand reward、surface band、body-z、object reward 或 CEM budget；
- 不跑 Box004、Bucket004、其他 Box021 case 或多 seed；
- 不在 A100 上渲染；
- 不使用本地 GPU、A6000 或 A100 `4,5,6,7`。

这些方向可能是 E169 后续的必要补充，但若同时加入会破坏 `P/R/G` 的因果归因。

---

## 13. 产物与完成定义

计划执行完成后应有：

```text
workspace/core4d/results/E169/
  s0_environment/
  scene_snapshot/
  canary/
  cem/full/
  eval/full/
  render/full/
  execution_manifest.json
  artifact_manifest.json
```

E169 只有在以下项目全部完成后才能标记实验完成：

1. 4 条 canary runtime 通过；
2. 28 条 new full run 与 4 条 frozen B0 组成完整 32-cell 分析表；
3. artifact SHA 与 effective config audit 全部通过；
4. E168 对齐指标、leg gate health 和 factorial contrasts 完整；
5. 28 条新视频完成本地渲染和人工审查；
6. 输出 xlsx、结果 log，并更新 tracker；
7. 明确选择一个统一 cell，或明确判定三轴方案未达到 `2/3 + control` 门槛。

当前只完成实验计划；以上实现、canary 和 full run 均未开始。
