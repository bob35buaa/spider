# E169 lower-body / object 三轴因子实验结果

日期：2026-07-18

状态：实验、严格回收、评测、渲染和人工审查全部完成；当前参数不晋级全量，也不导出 RL。

## Scope

- Case：3 条 E168 Box021 人工失败轨迹 + 1 条人工可用 control。
- 因子：`P`=下肢-物体显式物理碰撞，`R`=lower-body soft reward，`G`=CEM lower-body candidate gate。
- 设计：完整 `2^3` 因子矩阵；复用 4 条 E168 `B0`，在 A100 `0,1,2,3` 新跑 28 条 full CEM。
- 冻结项：输入、seed、CEM budget、hand/object/body-z 配置、OmniRetarget variant 与 target route 均未改变。

## Factor Definitions and Implementation

### Cell 命名

`P/R/G` 是三个独立开关，cell 名称表示开启开关的并集：

| Cell | P | R | G | 含义 |
|---|:---:|:---:|:---:|---|
| `B0` | 0 | 0 | 0 | E168 原始 baseline；直接复用，不重跑 |
| `P` | 1 | 0 | 0 | 仅增加下肢-物体物理碰撞 |
| `R` | 0 | 1 | 0 | 仅增加 lower-body soft reward penalty |
| `G` | 0 | 0 | 1 | 仅增加 CEM lower-body candidate gate |
| `PR` | 1 | 1 | 0 | 物理碰撞 + soft reward |
| `PG` | 1 | 0 | 1 | 物理碰撞 + candidate gate |
| `RG` | 0 | 1 | 1 | soft reward + candidate gate |
| `PRG` | 1 | 1 | 1 | 三轴全部开启 |

三个轴使用同一组 16 个 lower-body collision geom：左右 hip、thigh、shin、linkage brace，以及左右脚 `lf0-lf3/rf0-rf3`。R/G 在 config load 时将 geom name 解析为 MuJoCo geom id；P 则为每个 geom 创建与 `object_collision` 的显式 pair。

### P：Physical collision

**目标**：改变真实 MuJoCo 动力学，使下肢碰到箱体时产生法向接触，而不是几何体无代价互穿。P 不直接修改 reward 或 CEM candidate 排序。

**实现**：P-on 将 `scene_name` 从 E168 的 `scene_act_E168_rubber_hull` 切换为 E169 sidecar `scene_act_E169_lowerbody_physics`。sidecar 只新增 16 个 XML `<pair>`，其余 scene 结构必须与 E168 scene 一致：

```xml
<pair geom1="<lower_body_geom>"
      geom2="object_collision"
      solref="0.008 1"
      margin="0"
      gap="0"
      condim="1" />
```

参数与控制入口：

| 参数 | P-off | P-on | 作用 |
|---|---|---|---|
| `scene_name` | `scene_act_E168_rubber_hull` | `scene_act_E169_lowerbody_physics` | 选择是否包含显式 pair 的 scene |
| pair `solref` | 无 pair | `0.008 1` | MuJoCo 接触求解器时间常数/阻尼比 |
| pair `margin` / `gap` | 无 pair | `0 / 0` | 接触激活边界 |
| pair `condim` | 无 pair | `1` | 仅法向接触约束，不额外引入切向摩擦维度 |

生成与审计代码为 `scripts/experiments/E169/build_lowerbody_collision_scenes.py`。每个 P-on scene 编译后必须精确含 16 个 pair，semantic diff 只能包含这些 pair；`qpos0` 和 reference 前 5 帧若已有深于 `-5mm` 的 overlap，则 preflight 直接拒绝。

### R：Reward penalty

**目标**：不改变物理接触，只在 CEM reward 中软性偏好更大的下肢-箱体 clearance。候选仍可以用其他 reward 收益换取这项惩罚，因此 R 不是硬约束。

**实现**：`spider/simulators/mjwp.py` 每帧计算 16 个 lower-body geom 到箱体的最小 SDF：

```text
leg_hinge = max(0, margin - min_leg_object_sdf)
leg_object_penalty = -scale * leg_hinge * gate
```

E169 固定参数：

| 配置字段 | R-off | R-on | E169 含义 |
|---|---:|---:|---|
| `leg_object_penalty_scale` | `0.0` | `2.0` | 线性 hinge penalty 权重 |
| `leg_object_penalty_margin_m` | `0.02` | `0.02` | 希望保持 `2cm` clearance；仅 R-on 时生效 |
| `leg_object_penalty_geom_names` | `[]` | 16 个 lower-body geom | 参与 SDF 最小值计算的几何体 |
| `leg_object_penalty_gate_source` | `always` | `always` | 全时段启用，不依赖 hand contact mask |
| `leg_object_penalty_start_eval_time` | `0.0` | `0.0` | 时间窗下界；`always` 模式下不裁剪 |
| `leg_object_penalty_end_eval_time` | `999.0` | `999.0` | 时间窗上界；覆盖完整轨迹 |

例如 SDF 为 `0` 时单帧惩罚为 `-2.0×0.02=-0.04`，SDF 为 `-5mm` 时为 `-2.0×0.025=-0.05`。该 reward term 与原有 qpos、contact、task body/object 等 reward 相加，其他 reward 权重均冻结为 E168 值。

### G：CEM candidate gate

**目标**：在每轮 CEM 的 sampled rollout 中定义 lower-body 硬可行域，优先只用合法候选更新 control distribution；这与 R 的软排序不同。

**实现**：`spider/simulators/mjwp.py` 输出每帧 lower-body/object SDF，`spider/optimizers/sampling.py` 将整段 rollout 汇总为 sample-level mask。E169 中一个候选仅在以下两项同时满足时被 lower-body gate 判为 valid：

```text
整段 rollout 的 min_sdf >= -0.005m
SDF < +0.005m 的帧比例 <= 0.02
```

即允许最多 2% 的帧进入 `0-5mm` clearance band 或发生不深于 `5mm` 的轻微穿入，但任意一帧深于 `-5mm` 都直接无效。

| 配置字段 | G-off | G-on | 作用 |
|---|---:|---:|---|
| `cem_leg_gate_enabled` | `false` | `true` | lower-body gate 总开关 |
| `cem_leg_gate_geom_names` | `[]` | 16 个 lower-body geom | gate 的 SDF 监控集合 |
| `cem_leg_gate_min_sdf_m` | `0.005` | `0.005` | `+5mm` 目标 clearance，定义违规帧 |
| `cem_leg_gate_max_violation_pct` | `0.02` | `0.02` | 违规帧最多占 rollout 的 2% |
| `cem_leg_gate_hard_floor_m` | `-0.005` | `-0.005` | 任意帧不得深于 `-5mm` |
| `cem_leg_gate_min_valid_frac` | `0.02` | `0.02` | 至少 2% sample population 合法才走 hard-gate 分支 |
| `cem_leg_gate_fallback` | `least_violation` | `least_violation` | 合法候选不足时按违规深度/比例最小回退 |

Full CEM 的 `num_samples=1024`，所以 2% 对应每轮至少 `ceil(1024×0.02)=21` 个 combined-valid candidates。G mask 会与 E168 已开启的 body/hand gate 做逻辑 AND：combined-valid 数量达到门槛时，只从合法候选中选择 elite；不足时进入 `least_violation` fallback，并以 reward 作小权重 tie-breaker。

运行时记录 `cem_leg_gate_valid_frac`、`selected_valid_frac`、`fallback_used`、min-SDF 和 violation fraction。E169 的 `0/16 gate-health pass` 正是依据这些诊断，而不是只看最终离线 penetration。

## Artifacts

- 完整指标表：`results/E169/eval/full/E169_lowerbody_object_factorial_metrics.xlsx`
- 逐轨迹指标：`results/E169/eval/full/e169_case_metrics.tsv`
- Cell 汇总：`results/E169/eval/full/e169_cell_summary.tsv`
- E168 核心指标：`results/E169/eval/full/e169_e168_key_metrics.tsv`
- 指标统计：`results/E169/eval/full/e169_metric_summary.tsv`
- 最差样本：`results/E169/eval/full/e169_worst_case_rankings.tsv`
- 因子 contrasts：`results/E169/eval/full/e169_factorial_contrasts.tsv`
- 人工审查：`results/E169/eval/full/manual_review_template.tsv`
- 视频：`results/E169/render/full/`；4 个 8-cell montage 位于 `results/E169/render/full/montage/`
- 严格回收证据：`results/E169/artifacts/full/artifact_summary.json`

完整性：canary `4/4`；full `28/28`；artifact files `84`；evaluation `32/32`；not-ready `0`；errors `0`。28 条单视频和 4 个 montage 全部生成。xlsx 含 12 个 sheet、1577 个公式，LibreOffice 重算错误为 `0`。

### E168 指标完整性对齐

2026-07-18 复查发现，首版 E169 的逐轨迹 TSV 虽已计算 E168 核心数值，但 `Cell Summary/Factorial Contrasts` 只纳入 11 项，也缺少 E168 工作簿的“指标统计”和“最差样本”视图，因此展示层不完整。修订后：

- `Complete Metrics` 为 `32×287`，覆盖 E168 完整表的全部 192 个字段，并保留 E169 新增字段；
- `E168 Key Metrics` 为 `32×50`，显式包含 E168 定义的 23 项核心指标、门控/人工状态及 E169 leg physics/gate diagnostics；
- `Cell Summary` 为 8 个 cell × 47 列，包含 tracking、z、contact、release、penetration、lower-body、fall、manual 与 gate-health 汇总；
- `Metric Summary` 为 216 行，对 overall 和 8 个 cell 报告 24 项指标的 `n/mean/median/p95/min/max`；
- `Worst Cases` 为 60 行，按 E168 的 12 个 ranking metric 各列 top-5；
- `Factorial Contrasts` 从 11 项扩展为 24 项指标，共 840 行，包含 E168 23 项核心指标和 E169 leg physics contact；
- `hand_object_release_false_contact_3mm_frac` 在无 release window 的轨迹上按 E168 口径保留 N/A，不伪填 0。

## Quantitative Results

自动 gate 汇总：

| 指标 | 结果 |
|---|---:|
| numeric release pass | `8/32` |
| E169 acceptance pass | `2/32` |
| G-on gate-health pass | `0/16` |
| 任一统一 cell 的 failure-case pass | `0/3` |

跨 4 case 的平衡主效应（on - off；lower-body penetration 越低越好）：

| 因子 | leg penetration | leg physics contact | hand contact | 解释 |
|---|---:|---:|---:|---|
| `P` | `+0.016` | `+0.156` | `-0.047` | 单独没有降低穿透，主要把几何重叠变成真实接触，并轻微损失手部接触 |
| `R` | `-0.080` | `-0.001` | `+0.040` | 有稳定 soft preference，但不足以形成统一可用方案 |
| `G` | `-0.126` | `-0.072` | `-0.021` | lower-body 改善最大，但当前 candidate selection contract 不健康 |

平均 leg penetration 最低的 cell 为 `RG=0.035`，其次为 `PG=0.079`、`R=0.087`、`PRG=0.089`。这些均值不能覆盖 gate-health 和 per-case 失败。

Gate 失败不是 leg-valid pool 全空：`16/16` G-on 的 last-iteration leg-valid fraction 均 `>=0.05`，但 `selected_all_valid_mean` 为 `0.209-0.926`，没有一条达到 `1.0`；同时仅 `8/16` 满足 leg fallback mean `<=0.10`。这里不能直接推断 non-fallback 分支混入了 invalid candidate：实际 elite selection 使用 `body ∧ hand ∧ leg` 的 combined mask，leg-valid pool 足够时，combined-valid pool 仍可能因 body/hand gate 不足而进入 combined fallback。无论触发源是哪一个 gate，最终 selected candidates 没有维持 leg-valid invariant，因此当前 G-on 输出仍不能作为硬约束结果。

## Visual Review

人工审查覆盖 4 个 montage 的 20 个关键帧、28 条新视频的 sim-only 8-frame contact sheet，以及 control 的 4 条 P-on 轨迹的 20-frame dense sheet。

| Case | 新 cell 人工结果 | 结论 |
|---|---|---|
| `033_p1` | `0 USE / 7 DO_NOT_USE` | 全部仍有明显跨箱、站箱或借箱支撑；raw hand contact 也仅 `0.185-0.278` |
| `032_p1` | `0 USE / 7 DO_NOT_USE` | R/G 系显著降低腿部穿透，但全部仍有明显手部深穿；P/PR 还保留下肢接触 |
| `020_p2` | `3 USE / 4 DO_NOT_USE` | `PG/RG/PRG` 无明显 lower-body overlap/support；其余仍有剧烈下肢干涉 |
| `023_p2` control | `7 USE / 0 DO_NOT_USE` | 无 object kick、爆姿或失稳；P-on 四条存在短时 near-threshold 接触，标为 `MINOR_ACCEPTABLE` |

全部 28 条新轨迹合计 `10 USE / 18 DO_NOT_USE`。其中 failure-case 的 3 条可用结果全部集中在 `020_p2`，不存在跨 case 的统一方案。

## Claims

| Claim | 结论 | 证据 |
|---|---|---|
| C0 plumbing 独立 | 支持 | 32-cell config audit 通过；P/R/G diagnostics 与 cell 位编码一致；非目标配置无漂移 |
| C1 P 改变动力学 | 支持 | 每个 P-on scene 精确含 16 个 lower-body/object pair，SHA 匹配；P-on 出现真实 physics contact，P-off 无对应 pair |
| C2 R 是 soft preference | 支持 | R 主效应使 leg penetration `-0.080`，但没有统一 failure-case pass，证明其不是充分硬约束 |
| C3 G 健康过滤候选 | 不支持 | `0/16` gate-health pass；selected-all-valid 全部小于 1，且 `8/16` fallback 超阈值 |
| C4 至少一个组合修复 `2/3` | 不支持 | 所有统一 cell 均为 `0/3`；只有 `020_p2` 的 PG/RG/PRG 被逐 case rescue |
| C5 不靠丢手/塌姿态/踢箱修复 | 部分支持 | `020_p2` 的三条 rescue 无明显 kick/失稳且 contact/body-z 达标；`032_p1` 仍手部深穿，`033_p1` contact 不足 |
| C6 control 无明显回归 | 部分支持 | R 通过自动 control guard；G/RG 视觉正常但 gate-health 失败；P-on 四条引入短时 lower-body contact |
| C7 可区分主效应与交互 | 支持 | 840-row case-level factorial contrast 表覆盖 E168 核心指标，failure/control 分开报告，未做伪显著性检验 |

## Decision

E169 当前参数矩阵不进入全量 Box021，不设为默认配置，也不导出 RL。主要结论是：

1. `G` 是 lower-body 改善的最大主效应，`R` 次之；`P` 单独贡献弱，且可能把穿透变成真实踩箱/接触。
2. 当前首要工程问题是 G 的 selected-valid/fallback 契约，而不是继续增大 reward scale 或默认打开 P。
3. `033_p1` 不是单一 lower-body 问题：低 hand contact 与跨箱/站箱共同存在，需要单独检查 reference feasibility、stance/root support 与手部接触目标冲突。
4. `032_p1` 的 lower-body 已可被 R/G 系改善，但 hand penetration 成为新的主瓶颈，应与 lower-body 机制分开处理。

下一实验应先同时记录每次 CEM iteration 的 per-gate valid 数、combined-valid 数、elite 需求数、combined fallback 触发源和 fallback 后各 gate 的 selected-valid fraction；再决定是修订 combined fallback、保持 per-gate invariant，还是校准不可兼容的 body/hand/leg 可行域。之后用同一 4-case panel 复测。阈值、reward scale 和 pair 参数不在 E169 内追改。

## Execution Notes

- A100 运行期间按用户要求固定使用 GPU `0,1,2,3`，允许与既有任务叠加，未 kill 其他程序。
- 远端 full tmux 在 28 条完成后自然退出；本地 watcher 和 postprocess watcher 均已结束。
- 收尾时再次连接 A100 的冗余检查因本机临时 DNS 解析失败未执行；严格 pull summary、远端 session 自然退出记录和本地进程审计均无遗留 E169 作业。
