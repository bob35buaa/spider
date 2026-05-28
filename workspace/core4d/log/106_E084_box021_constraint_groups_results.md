# E084 Results: Box021 constraint groups main gate

日期：2026-05-28

对应计划：`workspace/core4d/plan/89_E084_box021_constraint_groups_plan.md`

## 结论

E084 按 main-gate 策略完成了 3 组约束/奖励实验，只跑 `d003_box021_20231018_029_p2` main case：

- E084A safety penalty：完成；
- E084B upright / ctrl trust：首次远程因 `task_body_rew` 与 full-body FK shape 不兼容失败，修复后 retry 完成；
- E084C semantic hand contact + lift：完成；
- `box023_p2` guard 未运行，因为 3 个 main 全部未通过 gate。

核心结论：

1. A/C 能让手不再撑地、pelvis 不再深度塌陷，但仍退化成“蹲抱/压箱/不 lift”，上半身-箱体 penetration 仍约 `80%+`，object-floor 仍约 `94-97%`。
2. B 降低了 object mean error 和 object bottom gap，但牺牲了接触和姿态：sim contact 只有 `9.3%`，左手撑地 `39.5%`，视觉上箱体明显翻倒。
3. 这不是单个 penalty 缺失的问题。当前 CEM reward 小调参只能在两个坏局部解之间切换：稳定但不抬，或更激进但翻箱/失控。
4. E084 没有产生可进入 guard 或 RL seed 的正例。下一步应停止 Box021 的 CEM 小调参，转向数据/seed 可行性、kinematic/support seed、hard-gate staged CEM 三条路线。

## 改动

代码改动：

| 文件 | 改动 |
|---|---|
| `spider/config.py` | 新增 `hand_floor_penalty_*`、`object_lift_rew_*`、`object_floor_penalty_*` 配置，并解析 hand-floor geom ids |
| `spider/simulators/mjwp.py` | reward 中新增 hand-floor penalty、object lift reward、object floor penalty；修复 `task_body_rew` 在 full-body `body_xpos_ref` 下的 shape mismatch |

新增 E084 脚本：

| 类型 | 路径 |
|---|---|
| variants | `workspace/core4d/scripts/E084/variants.tsv` |
| override 生成 | `workspace/core4d/scripts/E084/generate_e084_overrides.py` |
| 预处理 | `workspace/core4d/scripts/run_E084_preprocess.sh` |
| 训练 | `workspace/core4d/scripts/train/train_E084.sh` |
| 远程启动 | `workspace/core4d/scripts/run_E084_remote.sh` |
| 远程 worker | `workspace/core4d/scripts/E084/run_remote_inside.sh` |
| 远程回收 | `workspace/core4d/scripts/pull_E084_remote_results.sh` |
| 评估 | `workspace/core4d/scripts/eval/eval_E084.py` |
| contact sheet | `workspace/core4d/scripts/eval/extract_E084_contact_sheets.sh` |

## 实验矩阵

| Variant | 组别 | 运行位置 | 状态 |
|---|---|---|---|
| `E084A_d003_box021_20231018_029_p2_safety` | A safety penalty | local GPU0 | 完成 |
| `E084B_d003_box021_20231018_029_p2_upright` | B upright / ctrl trust | remote GPU0 retry | 完成 |
| `E084C_d003_box021_20231018_029_p2_semantic` | C semantic hand contact + lift | remote GPU1 | 完成 |

Guard variants 已生成但未运行：

```text
E084A_box023_p2_safety_guard
E084B_box023_p2_upright_guard
E084C_box023_p2_semantic_guard
```

原因：merged eval 后 `accepted_main_variants=[]`、`guard_splits_to_run=[]`。

## 结果路径

| 类型 | 路径 |
|---|---|
| 汇总 | `workspace/core4d/results/E084/comparison.csv` |
| gate summary | `workspace/core4d/results/E084/e084_gate_summary.json` |
| aggregate | `workspace/core4d/results/E084/aggregate_summary.json` |
| upper-body 诊断 | `workspace/core4d/results/E084/upperbody_diagnostics.csv` |
| 视频/轨迹 | `workspace/core4d/results/E084/E084*.mp4`, `workspace/core4d/results/E084/E084*.npz` |
| keyframes | `workspace/core4d/results/E084/keyframes/` |
| contact sheets | `workspace/core4d/results/E084/keyframes/contact_sheets/` |
| logs | `logs/E084/` |

## 量化结果

Gate summary:

```json
{
  "accepted_groups": [],
  "accepted_main_variants": [],
  "guard_splits_to_run": [],
  "num_main_gate_accept_for_guard": 0,
  "num_main_results": 3,
  "num_results": 3,
  "stop_before_guard": true
}
```

主指标：

| Variant | Obj mean | Pelvis min | Contact | Object floor | Bottom gap | Upper pen | Head/Torso pen | Hand floor L/R | Ctrl Linf | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E084A safety | `0.637m` | `0.596m` | `66.7%` | `93.8%` | `-11.9cm` | `80.6%` | `39.5/11.6%` | `0/0%` | `1.79` | FAIL |
| E084B upright | `0.417m` | `0.529m` | `9.3%` | `86.0%` | `-4.3cm` | `47.3%` | `45.7/0.0%` | `39.5/1.6%` | `2.20` | FAIL |
| E084C semantic | `0.672m` | `0.562m` | `66.7%` | `96.9%` | `-11.9cm` | `82.2%` | `44.2/11.6%` | `0/0%` | `2.01` | FAIL |

对 E083 `20231018_029_p2` 的相对变化：

| 项 | E083 | E084A | E084B | E084C | 解释 |
|---|---:|---:|---:|---:|---|
| obj mean | `0.608m` | `0.637m` | `0.417m` | `0.672m` | B 物体误差最好，但接触/姿态坏 |
| pelvis min | `0.478m` | `0.596m` | `0.529m` | `0.562m` | A/C 修复深度塌陷，B 仍低于 0.55 |
| sim contact | `69.0%` | `66.7%` | `9.3%` | `66.7%` | B 牺牲接触 |
| object floor | `94.6%` | `93.8%` | `86.0%` | `96.9%` | 3 组都没真正 lift |
| bottom gap | `-11.8cm` | `-11.9cm` | `-4.3cm` | `-11.9cm` | B 改善高度但箱体翻/接触失控 |
| upper pen | `82.2%` | `80.6%` | `47.3%` | `82.2%` | A/C 基本没解决上身压箱 |
| LH floor | `10.1%` | `0.0%` | `39.5%` | `0.0%` | A/C 对手撑地有效，B 反退化 |

## 可视化复核

Contact sheets:

- `workspace/core4d/results/E084/keyframes/contact_sheets/E084_all_cases_sheet.jpg`
- `workspace/core4d/results/E084/keyframes/contact_sheets/E084A_d003_box021_20231018_029_p2_safety_sheet.jpg`
- `workspace/core4d/results/E084/keyframes/contact_sheets/E084B_d003_box021_20231018_029_p2_upright_sheet.jpg`
- `workspace/core4d/results/E084/keyframes/contact_sheets/E084C_d003_box021_20231018_029_p2_semantic_sheet.jpg`

Subagent high 视觉复核：

| Variant | 视觉标签 | 是否可作为 guard/RL seed | 关键失败现象 |
|---|---|---|---|
| E084A safety | 稳定但保守；贴箱/蹲抱；未 lift | 否 | 箱体基本 upright 和落地，但 sim 长时间蹲在箱体后方/侧后方，缺少抬起和搬运语义；像“抱住/压住箱子”而不是 lift。 |
| E084B upright | 明显翻箱；upright 失守；接触失控 | 否 | 多帧箱体大角度 pitch/roll，甚至靠边角支撑；人体姿态被箱体牵引到异常位置，接触像推翻/挂住箱子。 |
| E084C semantic | 语义接近 A；轻微 lift 意图但仍未完成 | 否 | 相比 A 有更多贴近和手部接触，但箱体仍主要落地，后段仍是蹲抱/压箱式局部最优。 |

## Claims 验证

| Claim | 结果 |
|---|---|
| A 能抑制错误物理接触 | 部分通过：hand-floor 降为 0，pelvis 稳定；但 upperbody penetration 仍 `80.6%`，object floor `93.8%` |
| B 能通过 upright/ctrl trust 防止趴箱 | 失败：object err 降低，但 contact 只有 `9.3%`，LH floor `39.5%`，箱体翻倒 |
| C 能把接触语义转为 hand-led lift | 失败：与 A 类似，contact 高但 object floor `96.9%`，bottom gap `-11.9cm` |
| 任一 main 通过后再跑 guard | 通过执行规则：无 main 通过，因此未跑 guard |
| 结果可视化 + subagent high 复核 | 通过 |

## 遇到的错误

| 错误 | 影响 | 解决 |
|---|---|---|
| E084B 首次远程 run 报 `size of tensor a (8) must match tensor b (32)` | B 初跑中断 | `task_body_rew` 在 `use_local_frame_reward=true` 时收到 full-body `body_xpos_ref`，已在 reward 中按 `task_body_ids` 切回目标 body；optimizer smoke 通过后远程 retry 完成 |
| 远程 A/B/C 并行 snapshot manifest 出现重复行 | 只影响 manifest 可读性，不影响 scene 或训练 | B retry 时单独 snapshot 生成了干净 manifest；log 中记录该瑕疵 |
| MuJoCo EGL 析构 warning | 训练结束清理阶段 warning，产物存在 | 按 `.npz/.mp4`、eval 和脚本状态确认 run 成功 |

## 机制分析

E084 的失败说明 Box021 `20231018_029_p2` 已经超出“加一个 penalty 修局部解”的范围。

1. A/C 的 penalty 能惩罚手撑地和低 pelvis，但不能把“接触”变成“承重 lift”。CEM 仍可通过蹲抱/压箱获得较高 contact reward，箱子保持落地。
2. C 的 object lift reward 是 soft shaping，不是 hard constraint；在当前 object dynamics 和 contact 搜索空间里，它不足以战胜箱体落地局部最优。
3. B 证明单纯增强 upright/trust 会切断 contact：身体更受限后，CEM 找不到稳定手-箱承重解，转而翻箱或接触消失。
4. 三组都无法同时满足：手部语义接触、上身不压箱、箱体离地、姿态稳定。这指向 seed/数据可行性问题，而不是继续调同一 reward stack。

## 决策

E084 不进入 guard，不产生 RL seed。

下一步见 `workspace/core4d/plan/90_E085_box021_exit_gate_seed_routes_plan.md`。核心策略：

1. 先做 Box021 可行性/seed audit，确认 raw/OmniRetarget/E018/E084 哪个环节把任务推入不可行动作；
2. 再做 kinematic/support seed，先生成“身体姿态合理、箱体由外部支撑”的 seed；
3. 只有在 seed 可行后，才做 hard-gate staged CEM；如果 hard gate 仍找不到接触，就停止把该 Box021 case 当正例。
