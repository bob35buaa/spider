# E080 Plan: box025 大物体边界/负控复查

日期：2026-05-15

## Context

E079 证明 E077 的 CORE4D 数据处理 pipeline 可以推广到 10+ 个 single-person case，但也暴露了两个问题：

1. 只用 3cm contact quality proxy 不足以代表动力学质量。
2. case-specific window 修正了 fixed `post2=2.0s-3.6s` 的误伤，但三阈值仍会把 `box023_p1`、`bucket005_s2_p1` 等视觉失败样本误判为成功。

用户要求下一步先跑 `box025` 看看。这里必须保留 E054 的历史结论：`box025_p1`、`box025_p2`、`box025_s2_p1` 都是 Tier 3 / drop，主要原因是 `dim_max=0.89m > 0.70m`，单 G1 的臂展/站位结构上不适合单人搬运。E080 因此不是把 `box025` 当作预期成功正例，而是把它作为大物体边界/负控 case，用 E079 的最新数据和评估口径复查。

本地核验结果：

- `box025_person1` 与 `box025_person2` 都已有 SPIDER case。
- 两者 `trajectory_kinematic.npz` 均为 124 帧。
- 两者与 Holosoma `retarget_replace_batch_trimmed/20231011-048-person{1,2}-Box025_with_obj_original.npz` 完全一致。
- Holosoma untrimmed→trimmed 窗口均为 `trim_start=38, trim_frames=124`。

## Claims

| ID | Claim | 成功标准 |
|----|-------|----------|
| C1 | E077/E079 数据 pipeline 能以显式 trim window 处理 box025 p1/p2 | 生成 `workspace/core4d/results/E080/contact_masks/box025_person{1,2}/raw_contact_mask_3cm.npz`，`trim_start=38`，`T_spider=124` |
| C2 | E079 no-hold CEM 配置可直接跑 box025 p1/p2 | 两个 variant 都产出 `.npz`、`.mp4`、per-variant eval summary，不崩溃 |
| C3 | box025 作为 Tier 3 负控不应被误读为泛化成功 | 若 case-window 三阈值为 True 但视频仍明显趴箱/接触假/物体未真实搬运，则记录为 metric false positive；若数值与视觉均失败，则说明当前判据能识别大物体负控 |
| C4 | p1/p2 差异能帮助判断失败来自数据质量还是结构性几何 | 对比 p1/p2 的 contact mask、pelvis、object error、visual 失败模式，判断是否同属大物体不可解 |
| C5 | 远程执行只使用 `spider-remote` GPU1 | remote 脚本硬编码/默认 `REMOTE_GPU=1`，不触碰 GPU0 |

## 改动计划

### 1. E080 数据列表

新增：

- `workspace/core4d/data_preprocess/cases_E080_box025.tsv`

内容包括：

- `box025_person1`: raw `20231011/048`, `person1`, `Box025`, trim `38/124`
- `box025_person2`: raw `20231011/048`, `person2`, `Box025`, trim `38/124`

本轮不重跑 Holosoma retarget，也不覆盖已有 SPIDER scene，只生成/刷新 E080 contact mask。

### 2. E080 variant 与 override

新增：

- `workspace/core4d/scripts/E080/variants.tsv`
- `workspace/core4d/scripts/E080/generate_e080_overrides.py`

两个主 variant：

| Variant | Task | Split | Role |
|---------|------|-------|------|
| `E080_box025_p1` | `box025_person1` | local | main |
| `E080_box025_p2` | `box025_person2` | remote | main |

Override 继承 E079 主验证口径：

- base: `core4d_e074a_box023`
- `contact_hdmi_mask_source=core4d_3cm`
- `contact_hdmi_mask_path=workspace/core4d/results/E080/contact_masks/<slug>/raw_contact_mask_3cm.npz`
- `hold_contact_rew_scale=0.0`
- `hold_contact_start_eval_time=0.0`
- `hold_contact_end_eval_time=0.0`
- palm normal 由 `compute_palm_normal.py` 对当前 task 自动计算，不复用 box023 常量。

### 3. 训练与远程

新增：

- `workspace/core4d/scripts/run_E080_preprocess.sh`
- `workspace/core4d/scripts/train/train_E080.sh`
- `workspace/core4d/scripts/run_E080_remote.sh`
- `workspace/core4d/scripts/pull_E080_remote_results.sh`

调度：

- 本机 RTX5090：`E080_box025_p1`
- 远程 `spider-remote` GPU1：`E080_box025_p2`

远程启动前必须 commit + push，保证远程代码/config 与本地一致。

### 4. 评估与可视化

新增：

- `workspace/core4d/scripts/eval/eval_E080.py`

评估口径复用 E079：

- case-specific window = 3cm mask 左/右手 OR 的 first/last active frame，前后 padding 10 frame。
- 保留 fixed post2 仅作历史诊断，不作为主结论。
- 输出 `comparison.csv`、`aggregate_summary.json`、`eval_summary_E080_*.json/csv` 和 timeseries。

可视化：

- 每个 `.mp4` 抽取 `f50/f75/f100/f115/f120/f125/f145/f160/f180`，超长/越界帧跳过。
- 结果分析时使用 subagent 观察视频/关键帧，避免单人视觉判断偏差。

## 执行顺序

1. 写入 E080 plan，更新 `progress.md`。
2. 新增 E080 TSV、override 生成脚本、preprocess/train/remote/pull/eval 脚本。
3. 静态检查：`py_compile`、`bash -n`、`git diff --check`。
4. 运行 `run_E080_preprocess.sh` 生成 box025 p1/p2 3cm mask。
5. 生成 E080 overrides，并做 Hydra/短 horizon smoke（如时间允许）。
6. commit + push 后启动远程 GPU1；本地同时跑 p1。
7. 回收远程 p2，统一 eval。
8. 抽帧/视频观察，写 E080 结果日志并更新 tracker。

## 风险与回退

| 风险 | 处理 |
|------|------|
| box025 明知 Tier 3，结果大概率失败 | 失败本身是负控证据；不要为了让 box025 成功去调 reward |
| case-window 三阈值误判成功 | 记录为 E079 判据 false positive，并推动加入 object max/orientation/visual_usable 指标 |
| 远程未同步 ignored 数据 | `example_datasets/` 默认被 ignore；远程启动前必须 `git add -f` 纳入 E080 活跃 case 的 `scene.xml/scene_act.xml/scene_act_meta.json/task_info.json/0/trajectory_kinematic.npz` 和 `example_datasets/processed/core4d/assets/objects/box025/box025_m.obj`，不能假设远程已有本地残留数据 |
| 远程 GPU1 被占用 | 先跑本地 p1，远程 p2 等 GPU1 空闲后启动 |
