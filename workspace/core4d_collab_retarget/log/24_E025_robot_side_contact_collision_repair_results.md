# E025 结果：robot-side contact closure + collision penalty

日期：2026-05-20

## 初始目标

按照 `plan/28_E025_robot_side_contact_collision_repair_plan.md`，E025 处理 E020/E022 后剩余的 robot-side contact / collision 问题：

- `box023_p1`：E022 已修复 contact mask semantics，但 strict contact preservation 仍只有约 `25%`。
- `box023_p2`：E020 `algo_contact`，contact closure low。
- `bucket005_s2_p1/p2`、`bucket007_p1`：object-side transport 已成立，但 high contact 来自 hand/object deep penetration 或 leg/object shortcut。

E025 的实现目标是新增默认关闭的训练期 robot-object / leg-object penetration penalty，并测试它是否能在不破坏 E018b canonical support proxy 与 object tracking 的前提下，把 contact closure 和 penetration artifact 同时推过 strict gate。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/28_E025_robot_side_contact_collision_repair_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E025/variants.tsv` |
| Reward implementation | `spider/config.py`, `spider/simulators/mjwp.py` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E025.sh`, `workspace/core4d_collab_retarget/scripts/run_E025_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E025.py` |
| Results | `workspace/core4d_collab_retarget/results/E025/` |
| Comparison | `workspace/core4d_collab_retarget/results/E025/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E025/aggregate_summary.json` |
| Online videos | `workspace/core4d_collab_retarget/results/E025/online_video/` |
| Video-frame skill output | `workspace/core4d_collab_retarget/results/E025/video_frames_skill/` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E025/scene_snapshot/manifest.txt` |

## 执行状态

- [x] Reward knobs：新增默认 `0.0` 的 `robot_object_penalty_scale` 与 `leg_object_penalty_scale`，旧实验默认不受影响。
- [x] Geometry resolution：E025 hand penalty 解析 2 个 hand geoms；leg guard 解析 16 个 lower-body geoms。
- [x] Preprocess：8 个 overrides 生成。
- [x] Smoke：8/8 4-step smoke 通过；smoke 只验证 wiring。
- [x] Full：8/8 full variants 完成并写出 NPZ + MP4。
- [x] Eval：`eval_E025.py --all` 完成，`num_results=8`。
- [x] Visual artifacts：8/8 online MP4 到位，`video-frames` skill 抽取 8 张代表帧。

## 通俗解释：E025 做了什么、为什么没过

E025 处理的是 E022-E024 剩下的 robot-side 问题。到这里 object-side support proxy 基本已经稳定，所以 E025 不再改 anchor 或物体托举方式，而是看机器人手、腿和物体之间的接触质量。

这批 case 其实分成两类：

| 类型 | Case | 直观问题 | E025 的处理 |
|---|---|---|---|
| 低接触 | `box023_p1`、`box023_p2` | 手没有持续贴住物体，contact preservation 低 | 加强 contact reward：更强的 hold-contact、更大的 contact gain、更宽的接触范围，并加一点 near-field 手掌朝向约束 |
| 高接触但穿透 | `bucket005_s2_p1/p2`、`bucket007_p1` | contact 数字很高，但很多时候是手伸进物体里面 | 新增训练期 penetration penalty，惩罚手/机器人进入物体内部；另外给 `bucket005_s2_p1` 试了 leg guard |

这不是全组合 sweep，而是按问题类型选了 8 个关键 variants：

| Group | Variants | 目的 |
|---|---|---|
| `box023` contact closure | `box023_p1`、`box023_p2` 各 1 个 high-contact 版本 | 看更强 contact reward 能不能把低接触拉到 `>=70%` |
| bucket hand penalty lite | `bucket005_s2_p1/p2`、`bucket007_p1` 各 1 个 hand penalty scale `2` 版本 | 看轻量穿透惩罚能不能降低 hand-object deep penetration |
| bucket stronger hand penalty | `bucket005_s2_p2`、`bucket007_p1` 各 1 个 scale `4` 版本 | 如果 scale `2` 不够，看更强 penalty 有没有方向性收益 |
| leg guard | `bucket005_s2_p1` 1 个 hand+leg penalty 版本 | 专门看 lower-body/leg shortcut 能不能压下去 |

结果分组看更清楚：

| Case / group | 结果 | 说明 |
|---|---|---|
| `box023_p1` | contact 只有 `28.51%`，但 no-fall、penetration 都还可以 | E022 修完 mask 以后，问题已经不是“该不该接触”的标签，而是手的目标位置/时序没有真正闭合到物体表面 |
| `box023_p2` | contact 到 `52.38%`，但仍不到 `70%`，而且出现 fall 和 penetration regression | 强拉 contact 会把机器人姿态拉坏，不能继续简单加 contact gain |
| `bucket005_s2_p1` | hand penalty scale `2` 后 contact `99.47%`，deep penetration 仍 `92.89%` | penalty 太弱，优化器仍然选择“高接触 + 手在物体里”的捷径 |
| `bucket005_s2_p1` leg guard | leg penetration 从 `17.54%` 降到 `14.22%`，但 hand deep penetration 完全没变 | leg guard 对腿部捷径有一点用，但解决不了主要的手-物体穿透 |
| `bucket005_s2_p2` | scale `2` deep pen `77.83%`，scale `4` 降到 `64.53%` | 更强 penalty 有方向性改善，但离 `<15%` 还很远 |
| `bucket007_p1` | scale `2` deep pen `51.01%`，scale `4` 降到 `35.57%`，max pen 从 `10.55cm` 降到 `5.16cm` | 这是 E025 最清楚的正向信号，但仍没有过 strict gate |

因此 E025 的结论是：新增 penalty 的工程实现是成功的，8/8 full 都能跑，旧实验默认不受影响，8/8 object tracking 也没有回退；但这个 penalty 还是太“软”。它能在 `bucket005_s2_p2` 和 `bucket007_p1` 上看到改善方向，却挡不住优化器继续用“伸进物体内部”换高 contact。

最重要的负结论是：后面不应该再简单加 contact gain 或重复 mask sweep。`box023` 更应该查 contact target 的时序和动态目标；bucket cases 则需要更硬的几何约束，比如 SDF barrier、CEM sample rejection/projection、或者把目标改成“接近物体表面但不能进入物体内部”。

## 与基线对比：contact 提升了吗，穿透下降了吗

读表规则：

- `↑` 表示越高越好，`↓` 表示越低越好。
- `Δ = E025 结果 - 进入 E025 前的 baseline`；百分比指标的 Δ 单位是 pp。
- `box023_p1` 的 baseline 用 E022 mask 修复后的 best-contact 结果；其他 case 用 E018b canonical baseline。

低接触 case：contact 有一点提升，但代价或幅度不合格。

| Case | 指标 | 方向 | Baseline | E025 结果 | Δ | 结论 |
|---|---|---|---:|---:|---:|---|
| `box023_p1` | Contact 5cm | ↑ | `25.30%` | `28.51%` | `+3.21pp` | 只小幅提升，远低于 `70%` |
| `box023_p1` | Deep penetration | ↓ | `0.00%` | `4.11%` | `+4.11pp` | 变差，但仍在 guard 内 |
| `box023_p1` | Max penetration | ↓ | `1.81cm` | `3.37cm` | `+1.56cm` | 变差，但仍低于 `5cm` |
| `box023_p2` | Contact 5cm | ↑ | `28.57%` | `52.38%` | `+23.81pp` | 明显提升，但仍未达 `70%` |
| `box023_p2` | Deep penetration | ↓ | `3.33%` | `20.67%` | `+17.34pp` | 明显变差，超过 guard |
| `box023_p2` | Max penetration | ↓ | `2.37cm` | `6.17cm` | `+3.80cm` | 明显变差 |
| `box023_p2` | No fall | pass | pass | fail | 变差 | 强拉 contact 把姿态拉坏 |

高接触但穿透 case：penalty 有方向性，但没有压到目标。

| Case / variant | 指标 | 方向 | Baseline | E025 结果 | Δ | 结论 |
|---|---|---|---:|---:|---:|---|
| `bucket005_s2_p1` lite | Contact 5cm | ↑ | `97.59%` | `99.47%` | `+1.87pp` | 维持高 contact |
| `bucket005_s2_p1` lite | Deep penetration | ↓ | `88.15%` | `92.89%` | `+4.74pp` | 变差，penalty 没压住 |
| `bucket005_s2_p1` leg guard | Leg penetration | ↓ | `23.70%` | `14.22%` | `-9.48pp` | 腿部有改善 |
| `bucket005_s2_p1` leg guard | Deep penetration | ↓ | `88.15%` | `92.89%` | `+4.74pp` | 手部穿透仍没改善 |
| `bucket005_s2_p2` scale 2 | Deep penetration | ↓ | `74.38%` | `77.83%` | `+3.45pp` | 变差 |
| `bucket005_s2_p2` scale 4 | Deep penetration | ↓ | `74.38%` | `64.53%` | `-9.85pp` | 有改善，但仍远高于 `<15%` |
| `bucket007_p1` scale 2 | Deep penetration | ↓ | `68.46%` | `51.01%` | `-17.45pp` | 有明显改善，但不够 |
| `bucket007_p1` scale 4 | Deep penetration | ↓ | `68.46%` | `35.57%` | `-32.89pp` | 最明显改善，但仍 fail |
| `bucket007_p1` scale 4 | Max penetration | ↓ | `8.40cm` | `5.16cm` | `-3.24cm` | 明显改善，但仍略高于 `5cm` |

这张对比表说明：E025 不是完全没有信号，`bucket007_p1` 和 `bucket005_s2_p2` 的 stronger penalty 确实让 deep penetration 下降；但下降幅度还不够。`box023_p2` 的 contact 虽然提升了，却用 fall/penetration 换来的，所以不能算成功。

## 量化结果

| Variant | Case | Role | Penalty | Contact 5cm | Deep pen | Max pen | Leg pen | No fall | Object | Strict |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `E025_box023_p1_hc2_gain8_sigma20_ori_nf` | `box023_p1` | contact closure | hand `0`, leg `0` | `28.51%` | `4.11%` | `3.37cm` | `0.00%` | true | true | false |
| `E025_box023_p2_hc2_gain8_sigma20_ori_nf` | `box023_p2` | contact closure | hand `0`, leg `0` | `52.38%` | `20.67%` | `6.17cm` | `19.33%` | false | true | false |
| `E025_bucket005_s2_p1_penalty_lite_hc1` | `bucket005_s2_p1` | penetration guard | hand `2`, leg `0` | `99.47%` | `92.89%` | `5.03cm` | `17.54%` | true | true | false |
| `E025_bucket005_s2_p1_leg_guard_penalty` | `bucket005_s2_p1` | leg guard | hand `2`, leg `2` | `99.47%` | `92.89%` | `5.04cm` | `14.22%` | true | true | false |
| `E025_bucket005_s2_p2_penalty_lite_hc1` | `bucket005_s2_p2` | penetration guard | hand `2`, leg `0` | `97.25%` | `77.83%` | `8.13cm` | `15.76%` | true | true | false |
| `E025_bucket005_s2_p2_penalty_s4_hc1` | `bucket005_s2_p2` | penetration guard | hand `4`, leg `0` | `96.42%` | `64.53%` | `7.06cm` | `16.26%` | true | true | false |
| `E025_bucket007_p1_penalty_lite_hc1` | `bucket007_p1` | penetration guard | hand `2`, leg `0` | `82.66%` | `51.01%` | `10.55cm` | `24.16%` | true | true | false |
| `E025_bucket007_p1_penalty_s4_hc1` | `bucket007_p1` | penetration guard | hand `4`, leg `0` | `84.13%` | `35.57%` | `5.16cm` | `16.11%` | true | true | false |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `8` |
| `num_E025_strict_success` | `0` |
| `num_object_no_regression_pass` | `8` |
| `num_no_fall_pass` | `7` |
| `num_contact_closure_pass` | `6` |
| `num_penetration_guard_pass` | `1` |
| `num_leg_guard_pass` | `2` |
| `best_contact_pct` | `99.47%` (`bucket005_s2_p1_penalty_lite`) |
| `best_deep_penetration_pct` | `4.11%` (`box023_p1`, but contact fail) |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 low-contact case closure | 不通过：`box023_p1=28.51%`、`box023_p2=52.38%`，均未达 `>=70%`；p2 还引入 fall / penetration regression |
| C2 bucket deep penetration 降到 `<15%` | 不通过：bucket variants deep penetration 仍 `35.57-92.89%` |
| C3 object-side 不回退 | 通过：8/8 object no-regression pass，Epos `0.039-0.064m`，Erot `2.33-7.03deg` |
| C4 no stability regression | 部分通过：7/8 no-fall，`box023_p2` high contact closure variant fall |
| C5 至少部分泛化成立 | 不通过：strict success `0/8` |

## 可视化观察

`video-frames` skill 已抽取：

- `E025_box023_p1_hc2_gain8_sigma20_ori_nf_t0240.jpg`
- `E025_box023_p1_hc2_gain8_sigma20_ori_nf_t0340.jpg`
- `E025_box023_p2_hc2_gain8_sigma20_ori_nf_t0208.jpg`
- `E025_box023_p2_hc2_gain8_sigma20_ori_nf_t0390.jpg`
- `E025_bucket005_s2_p2_penalty_lite_hc1_t0300.jpg`
- `E025_bucket005_s2_p2_penalty_s4_hc1_t0300.jpg`
- `E025_bucket007_p1_penalty_lite_hc1_t0260.jpg`
- `E025_bucket007_p1_penalty_s4_hc1_t0260.jpg`

视觉观察与量化一致：box023 的 high contact reward 可以制造更多近接触，但 strict 5cm preservation 仍不足，且 `box023_p2` 在接触拉拽下发生低 pelvis / fall。bucket variants 保持高 object tracking 与高 contact，但接触仍主要来自手伸入物体，`robot_object_penalty_scale=4` 只把 `bucket007_p1` deep penetration 从 `51.01%` 降到 `35.57%`，没有形成干净的 surface contact。

## 结论

E025 是负结果，但给出清晰约束：

1. E025 的工程 wiring 是成功的：默认关闭的 penetration penalty 不影响旧实验，E025 override 能正确解析 hand / leg geoms，8/8 full 都能跑完并保留 object-side behavior。
2. object-side canonical support proxy 非常稳定：E022-E025 中所有 full variants 基本都保持 object no-regression，这说明当前主要瓶颈已经不是 anchor 或 object transport。
3. contact closure 不能靠继续加 `hold_contact` / `contact_hdmi_gain` 解决。`box023_p1/p2` 仍未达 `70%` strict contact，p2 还出现 fall 与 penetration regression。
4. 当前 penetration penalty 口径太弱：它能在 `bucket005_s2_p2`、`bucket007_p1` 上产生方向性改善，但仍远高于 `<15%` 目标。特别是 `bucket005_s2_p1` hand deep penetration 完全不变，说明 CEM 仍能找到“高 contact + inside object”的低成本路径。
5. leg guard 有局部价值：`bucket005_s2_p1` leg penetration 从 `17.54%` 降到 `14.22%`，但只解决 lower-body 一小部分，不能修复 hand-object shortcut。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 远程 SSH reset / timeout | 多次 | 低频重试；最终主要依赖本地 GPU 顺序完成 |
| 远程 E025 run_mjwp stale | 2 | 为 `run_E025_remote.sh` 加 `RUN_TIMEOUT_SECONDS=2400` / `RUN_STALL_TIMEOUT_SECONDS=300`；stale 后清理 remote tmux/process |
| 远程 `bucket005_s2_p2_s4` stall at `38/296` | 1 | 本地重跑并完成 full |
| MP4/EGL teardown 报 `EGL_NOT_INITIALIZED` | 多次 | wrapper 已成功复制 NPZ/MP4 并退出完成；记录为 viewer teardown，不影响 rollout/eval |

## 下一步

- `box023_p1/p2`：转 dynamic target / contact timing diagnosis。重点检查 mask active window、EEF target time alignment、near-field orientation target 是否在正确接触阶段驱动，而不是继续加 contact gain 或重复 mask sweep。
- bucket penetration：把 penalty 从 soft reward 提升为更强的 geometric barrier / feasibility constraint。候选包括 signed-distance barrier with steeper hinge、CEM sample rejection / projection、surface-normal target + inside-object hard penalty、或将 contact target 改为“贴近表面但 SDF 非负”的双条件。
- lower-body artifact：leg guard 可保留，但需要与真实 lower-body collision geometry 一起做；单独 leg reward 只能轻微降低 leg shortcut。
- stability：`bucket001_p1` 仍需单独实验，方向是 upright/root terminal gate、foot support / lower-body control regularizer，或重新检查 p1 reference/support timing。
