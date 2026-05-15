# E079 Plan: CORE4D 10+ 高接触质量 case 泛化验证

## Context

E078 的核心结论是：现有 CEM/MJWP 算法在高质量 reference/contact 输入下可以工作。`box023_person2` 的 3cm per-EEF mask 与 ref 接触质量好，sim 动作接近可用；`box023_person1` 仍出现 f120-f125 右腿相位偏差，更像数据/retarget 质量问题。

E054 已完成 21 个 CORE4D case 的几何可达性和 intent 分级，6 个 B+C 原始序列是首批候选：

| 序列 | 已有 p1 task | raw | object | E054 备注 |
|------|--------------|-----|--------|-----------|
| box021 | `box021_person1` | `20231018/030` | `Box021` | Tier2, both-hand |
| box023 | `box023_person1` | `20231008/045` | `Box023` | Tier1, both-hand；p1 是反例，p2 是 E078 正例 |
| bucket001 | `bucket001_person1` | `20231030/094` | `bucket001` | Tier1, single-left |
| bucket005_s2 | `bucket005_s2_person1` | `20231002/004` | `bucket005` | Tier1, both-hand |
| bucket007 | `bucket007_person1` | `20231020/055` | `Bucket007` | Tier2, both-hand |
| desk021 | `desk021_person1` | `20231008/007` | `Desk021` | Tier2, both-hand |

为了达到 10+ 样本，本轮实验单位定义为 single-person case：每个 raw 序列构造 `person1/person2` 两个单人 SPIDER case，共 12 个候选。

Post-hoc role correction（2026-05-15）：`box023_person2` 是 E078 已验证成功的 positive/calibration guard，不计入 main 成功率；`box023_person1` 是已知失败/数据质量反例，但仍作为 main 中的负例/false-positive 检查样本。

## Claims

| ID | Claim | 成功标准 |
|----|-------|----------|
| C1 | E077 数据处理 pipeline 可泛化到 6 个 B+C 序列的 p1/p2 单人 case | 12 个候选中至少 10 个完成 3cm mask、Holosoma trim、SPIDER `scene.xml/scene_act.xml/trajectory_kinematic.npz` 校验 |
| C2 | 3cm contact audit 能筛出高质量 reference/contact case | 每个候选都有 `raw_contact_mask_3cm.npz`、`audit_summary_3cm.json` 和 quality CSV；高质量样本定义为 intent 窗口内目标手 3cm active ≥80%，且非目标手不被错误强制 |
| C3 | 通用 CEM 配置在高质量 case 上具有泛化能力 | 至少 10 个候选中 ≥7 个满足：post2 pelvis_z_min ≥0.55m、post2 sim contact ≥50%、post2 obj_err_mean ≤0.20m、视频无明显摔倒/大跨步失真 |
| C4 | `box023_person1` 仍作为反例被识别 | E079 对 `box023_person1` 不应把它误判为成功；如果数值过关但视频仍右腿相位错，应记为 data-quality false positive |
| C5 | 本轮不新增 case-specific hand-crafted contact window | 主验证配置不依赖固定 `hold_contact_start/end_eval_time`；如保留 E075B/E078B 作为 calibration，只作为对照，不作为泛化主结论 |
| C6 | 远程调度符合当前资源约束 | 远程只使用 `spider-remote` GPU1；GPU0 不启动任务；本机 RTX 5090 与远程 GPU1 分摊任务 |

## 改动计划

### 1. 数据 pipeline 泛化

- 新增通用 scene 生成脚本：
  `workspace/core4d/data_preprocess/create_spider_scene_from_template.py`
  - 从 `source_scene_task` 复制 `scene.xml`。
  - 用 trimmed retarget qpos 第一帧更新 object 初始 `pos/quat`。
  - 写 `task_info.json`，记录 raw sequence、person、object、source_scene、trimmed npz。
  - 可选生成 `scene_act.xml`。
- 修改 `workspace/core4d/data_preprocess/pipeline.sh`，从 E077 专用 `create_box023_person2_scene.py` 切到通用脚本。
- 新增 `workspace/core4d/data_preprocess/cases_E079_12.tsv`，覆盖 6 序列 × p1/p2。
- p1 已存在的 SPIDER case 不强制重建；但仍要生成/校验 3cm contact mask，并保留 scene snapshot。

### 2. Case/contact 质量审计

- 新增 E079 audit 脚本：
  `workspace/core4d/scripts/eval/eval_E079_contact_quality.py`
- 输入：
  - E054 `case_tier_classification.csv`
  - E079 contact masks
  - 每个 SPIDER case 的 `trajectory_kinematic.npz`
- 输出：
  - `workspace/core4d/results/E079/contact_quality.csv`
  - 每 case JSON summary
- 审计字段：
  - mask left/right active pct
  - intent/post2 窗口 active pct
  - raw min distance mean/max
  - single-hand case 的非目标手 active pct
  - p1/p2 trim window 是否一致

### 3. 通用 CEM 配置

主验证不继续使用 box023 固定 `hold_contact` window。新增 E079 通用配置生成器：

- `workspace/core4d/scripts/E079/generate_e079_overrides.py`
- 每个 case 生成一个 override：
  - defaults: E074/E073/E062 链路中的通用部分
  - `task=<case>`
  - `contact_hdmi_mask_source=core4d_3cm`
  - `contact_hdmi_mask_path=workspace/core4d/results/E079/contact_masks/<slug>/raw_contact_mask_3cm.npz`
  - `contact_hdmi_mask_person_idx=0/1`
  - `contact_hdmi_palm_normal_left/right` 自动由 `compute_palm_normal.py` 算得
  - `hold_contact_rew_scale=0.0`

保留一个 calibration 组：

- `E079_calib_box023_p2_hold`: 复现 E078B 口径，用于确认 E079 脚本没有改变 E078B 结果。
- calibration 不计入 C3 泛化成功率。

### 4. 训练/运行脚本

新增脚本：

| 类型 | 路径 |
|------|------|
| 数据预处理 | `workspace/core4d/scripts/run_E079_preprocess.sh` |
| 本地训练 | `workspace/core4d/scripts/train/train_E079.sh` |
| 远程启动 | `workspace/core4d/scripts/run_E079_remote.sh` |
| 远程回收 | `workspace/core4d/scripts/pull_E079_remote_results.sh` |
| 评估 | `workspace/core4d/scripts/eval/eval_E079.py` |

调度策略：

- 本机 RTX 5090：跑 6 个优先 case，包括 `box023_person1` 反例和若干短序列。
- 远程 `spider-remote` GPU1：跑剩余候选，单 GPU 串行。
- 远程脚本硬性只用 `CUDA_VISIBLE_DEVICES=1`，不触碰 GPU0。
- 每个实验完成后立即复制 `trajectory_mjwp_act.npz` 到 `workspace/core4d/results/E079/`，避免 output_dir 覆盖。

### 5. 可视化与结果记录

- 每个 case 输出 `.mp4`，并抽取固定关键帧：
  `f50, f75, f100, f115, f120, f125, f145, f160, f180`，超出长度则跳过。
- `eval_E079.py` 输出：
  - `comparison.csv`
  - `eval_summary_<variant>.json`
  - `timeseries_<variant>.csv`
  - timeline plots
- 实验完成后写：
  `workspace/core4d/log/100_E079_core4d_generalization_10plus_results.md`
- 更新 `EXPERIMENT_TRACKER.md` 与 `progress.md`。

## 执行顺序

1. 写入 plan，更新 progress。
2. 实现通用数据/scene 脚本、E079 case TSV、override 生成器、train/eval/remote/pull 脚本。
3. 本地静态验证：
   - `py_compile` 覆盖新增 Python。
   - `bash -n` 覆盖新增 shell。
   - `pipeline.sh --dry-run` 覆盖 E079 TSV。
4. 运行/复用 E079 preprocessing，至少先完成 mask + case verify。
5. 生成 E079 overrides，Hydra compose/smoke test 1-2 个 case。
6. git add/commit/push，远程 GPU1 启动 E079 remote；本机同步跑本地分片。
7. 回收远程结果，本地统一 eval。
8. 抽帧/视频观察，写结果日志，更新 tracker。

## 风险与回退

| 风险 | 处理 |
|------|------|
| p1/p2 Holosoma trim window 不一致 | 单人验证允许各自窗口；若要做双人同场景，另起 common-scale/common-window 实验 |
| 某个 p2 retarget 失败 | 记录为 preprocessing fail；仍继续其余 case，C1 要求至少 10/12 完成 |
| 某 case 没有可用 source_scene_task | 按 `SCENE_TEMPLATE_GUIDE.md` 先补模板；不临时复制不匹配物体 |
| 通用 no-hold 配置显著差于 E078B | 先跑 calibration，若 box023_p2 no-hold 明显失败，结果中区分“算法依赖 hold window”与“数据质量泛化”两件事，不把 window 配置当作泛化结论 |
| 远程 GPU0 占用高 | E079 remote 脚本只启动 GPU1；本机承担另一半 |
