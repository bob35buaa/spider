# Post-E020 优化总览计划：E022-E025 root-cause 分线

日期：2026-05-20

## Context

E020 已把 E018b 13 个 case 做完 S1-S6 failure attribution。用户明确要求：

- `desk021_p1` 直接忽略。
- `box021_p1` / `box021_p2` 暂按数据问题忽略。
- `E021` 已被 `plan/22_E021_holosoma_rl_export_plan.md` 占用，后续优化实验从 `E022` 开始编号。
- 后续按 root cause 分类依次优化；每个实验继续按 `experiment-planning-zh` 的 Plan -> Implement -> Train -> Evaluate -> Log -> Tracker 流程展开。

E019 unified eval 可先沿用；本轮不先修 FPS P2。涉及 smoothness / foot-skating velocity 时在 log 中标注 FPS caveat；object tracking、contact preservation、deep penetration、fall/upright 等本轮主指标不依赖该常量。

## Scope

### 排除 / 保留

| 类别 | Case | 处理 |
|---|---|---|
| 用户指定忽略 | `desk021_p1` | 不进入优化成功标准；保留在 E020 证据中 |
| 用户指定暂忽略 | `box021_p1`, `box021_p2` | 不进入 E024 stability 成功标准；后续若要恢复需先做数据审计 |
| 已通过 | `box025_p2` | 不优化；作为 regression guard 可选 |
| 本轮剩余 | 9 cases | 按 root cause 进入 E022-E025 |

### 剩余 case 分组

| Root cause | Cases | E020/E018b 基线 | 后续实验 |
|---|---|---|---|
| `contact_mask` | `box023_p1` | object pass: Epos `0.043m`, Erot `1.96deg`; contact `22.5%`; deep pen `0%`; mask overclaim `54.4%`; no fall | E022 |
| `retarget_kinematic` | `box025_p1`, `bucket007_p2` | ref leg/object interference `66.5%` / `66.3%`; mean contact `47.9%`; no fall | E023 |
| `algo_stability` | `bucket001_p1`, `bucket001_p2` | object pass; fall `2/2`; pelvis min evidence from E020/E018b; contact `0.0%` / `66.3%` | E024 |
| `algo_contact` | `box023_p2`, `bucket005_s2_p1`, `bucket005_s2_p2`, `bucket007_p1` | fall `0/4`; mean contact `75.3%`; mean deep pen `58.6%`; artifact/shortcut dominant | E025 |

## 执行顺序

### E022: contact mask semantics repair

**目标 case**：`box023_p1`

**为什么先做**：这是最小闭环的数据/语义修复；如果 mask gate 仍错，后续 controller sweep 会对错误 contact phase 过拟合。

**做什么**：

1. 复用 E018b derived task、canonical anchor、scene XML，不改 object-side support proxy。
2. 新建 E022 manifest/overrides，显式复制 contact mask 到 `results/E022/contact_masks/`，避免污染 E018b。
3. 对比 mask 版本：E018b baseline、raw 3cm per-EEF、raw 3cm + 短时 dilation/gap-closing、必要时强制 `contact_hdmi_mask_time_axis={eval,spider}` 排查 axis。
4. 新增 E022 eval 字段：mask active pct、mask overclaim/mismatch pct、E018b paper metrics、fall/artifact gates。

**成功标准**：

- `box023_p1` contact preservation `22.5% -> >=70%`。
- mask overclaim `54.4% -> <20%`。
- deep penetration 维持 `<=10%`，最好保持 `0%`。
- no fall；object Epos `<0.10m`、Erot `<25deg`。

### E023: retarget kinematic geometry repair

**目标 case**：`box025_p1`, `bucket007_p2`

**做什么**：

1. 先不做 reward sweep；复用 E020 S2 ref physics，对 kinematic ref 的 leg/object interference 做定位。
2. 生成 corrected ref / scene diagnostic variant，优先修 ref-side leg-object interference，而不是让 physics 控制器追错误参考。
3. 修复后 rerun physical E018b-style rollout，并用 E020 S2 + E018b eval 双重验证。

**成功标准**：

- ref leg/object interference 从约 `66%` 降到 `<15%`。
- sim deep penetration ok，contact preservation `>=70%`。
- object gate 不退化：Epos `<0.10m`、transport pass。

### E024: robot stability / fall control

**目标 case**：`bucket001_p1`, `bucket001_p2`

**做什么**：

1. 保持 object-side canonical support proxy 不变。
2. Sweep `stability_penalty_scale` / `stability_penalty_threshold` / root-local tracking sigma / contact gain，避免 bucket case 通过摔倒或低 pelvis 姿态运输。
3. 如果 reward sweep 失败，再考虑新增 upright/root-pose terminal gate 或更强 lower-body tracking。

**成功标准**：

- `bucket001_p1/p2` fall false。
- pelvis min `>=0.45m`，目标 `>=0.55m`。
- object Epos `<0.10m` 且 transport pass。
- contact preservation 提升，至少不牺牲 deep penetration / floor-leg gate。

### E025: robot-side contact closure + collision penalty

**目标 case**：`box023_p2`, `bucket005_s2_p1`, `bucket005_s2_p2`, `bucket007_p1`

**做什么**：

1. 先跑 reward-only contact closure sweep：`hold_contact_rew_scale`、`contact_hdmi_gain`、`contact_hdmi_sigma`、orientation gate。
2. 若 artifact 仍由 leg/object 或 hand/object penetration 主导，则新增 explicit robot-object collision penalty（现有代码只有 eval 指标，没有训练 reward 参数）。
3. 四个 case 用远程 2 GPU 并行，按 case 分队列，不把 `box021_*` 混入成功标准。

**成功标准**：

- deep penetration duration 均 `<15%`，max penetration `<=5cm`。
- contact preservation 保持 `>=70%`。
- no fall；object Epos `<0.10m`、transport pass。
- 至少 `2/4` case 达 strict pass，且未通过 case 有明确下一步。

## 统一实现约束

| 项 | 规则 |
|---|---|
| 编号 | 优化实验从 `E022` 起；plan 文件 NN 继续按目录递增，不等于实验号 |
| 基线 | 复用 E018b pipeline：derived task、canonical support proxy、online MP4、E018b eval schema |
| 结果目录 | 每个实验写到 `workspace/core4d_collab_retarget/results/E0NN/` |
| 训练脚本 | 每个实验必须有 `scripts/train/train_E0NN.sh` |
| 远程脚本 | 当独立 variant >=3 时创建 `scripts/run_E0NN_remote.sh`，按 remote-execution.md 使用 2 GPU |
| scene snapshot | 训练脚本第一步调用 `scripts/convert/snapshot_scenes.sh E0NN <case...>` |
| eval | 每个实验复制/参数化 E018b evaluator 到 `scripts/eval/eval_E0NN.py`；必要时追加 `unified_eval.py` postprocess |
| FPS caveat | E019 P2 未修前，不用 smoothness / foot-skating velocity 作主成功标准 |
| log | 每个实验完成后写 `log/{NN}_E0NN_*.md` 并更新 tracker |

## 第一批落地

下一步先展开 E022 计划：`plan/25_E022_contact_mask_semantics_repair_plan.md`。
