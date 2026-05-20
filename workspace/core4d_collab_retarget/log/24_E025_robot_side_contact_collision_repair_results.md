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
