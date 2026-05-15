# E079 Results: CORE4D 10+ 高接触质量 case 泛化验证

日期：2026-05-15

对应计划：`workspace/core4d/plan/84_E079_core4d_generalization_10plus_plan.md`

## 结论摘要

E079 验证了 E077 数据处理 pipeline 可以从 `box023` 推广到 10+ 个 CORE4D single-person case。根据用户指出的评估窗口问题，已将量化评估从固定 `box023`-derived `post2=2.0s-3.6s` 改为 case-specific contact/intent window 复算。

核心结果：

- 数据 pipeline：`11/12` 个候选可运行。`desk021_person2` 在 Holosoma retarget 阶段 `CVXPY solve failed: infeasible`，按 plan 记为 preprocessing fail；其余 11 个 case 完成 mask/scene/scene_act/trajectory 校验。
- 接触 mask：11 个可用 case 都生成 3cm per-EEF contact mask 和 contact-quality 审计文件，且不依赖手写 hold-contact window。
- CEM 泛化：case-specific window 下，10 个 main case 中 `E079_success_case_window=True` 为 `6/10 = 60%`；固定 `post2` 旧口径为 `2/10 = 20%`，仅保留作历史诊断参考。
- Guard：`E079_box023_p2` 是 E078 已验证成功的 positive/calibration guard，本轮 case-window 与 fixed-post2 均复现成功；它不计入 main 成功率。
- 视觉质量：用户复查后重新校准：`box021_p1` 视觉质量很好，但该序列 ref 接触位置本身异常；`bucket007_p2`、`bucket005_s2_p2` 是相对更可信的正例/近正例；`desk021_p1` 前段没抬起来、后段才相对正常；`bucket005_s2_p1` 物体持续受力旋转，不应算 near-pass；`box023_p1` 是已知 p1 失败/数据质量反例；`bucket007_p1` 更像 trim/ref 质量问题导致初始后退，后续算法有弥补。
- 失败模式：`box021_p2` 和 `bucket001_p2` 是明确倒地失败；`bucket001_p1` 是低姿态/跪姿补偿；`bucket007_p1` 是前摇/trim 未剔干净 + ref 支撑脚误差造成初始位置后退，而不是单纯 CEM 接触失败。

因此，E079 的主结论是：**E077 数据处理 pipeline 已验证可用；case-specific window 复算后，main 数值成功率从 fixed-window `2/10` 升到 `6/10`，但仍低于计划中的 `>=7/10`。`box023_p2` positive guard 复现成功；`box023_p1` 是明确 false positive；`bucket007_p1` 更像 trim/ref 支撑质量导致的初始偏移，而非算法完全失败。下一步应改进成功判据，并检查 trim/前摇剔除与 reference feasibility/stability audit。**

## 实验配置

### 数据与 mask

- 数据入口：`workspace/core4d/data_preprocess/pipeline.sh`
- E079 case TSV：
  - `workspace/core4d/data_preprocess/cases_E079_existing_p1.tsv`
  - `workspace/core4d/data_preprocess/cases_E079_existing_p2.tsv`
  - `workspace/core4d/data_preprocess/cases_E079_build_p2.tsv`
- Contact mask：`workspace/core4d/results/E079/contact_masks/*/raw_contact_mask_3cm.npz`
- Contact quality：`workspace/core4d/results/E079/contact_quality.csv`

### 训练与调度

- 本地 RTX5090：`bash workspace/core4d/scripts/train/train_E079.sh local 0`
- 远程 A6000：`bash workspace/core4d/scripts/run_E079_remote.sh`
- 远程约束：只使用 `spider-remote` GPU1，GPU0 未启动任务。
- 回收：`bash workspace/core4d/scripts/pull_E079_remote_results.sh`
- 评估：`workspace/core4d/scripts/eval/eval_E079.py`

主验证配置：

- `contact_hdmi_mask_source=core4d_3cm`
- `contact_hdmi_mask_key=eval_contact_mask_3cm`
- `hold_contact_rew_scale=0.0`
- 不使用 case-specific `hold_contact_start_eval_time/end_eval_time`

## 结果路径

| 类型 | 路径 |
|------|------|
| 结果汇总 | `workspace/core4d/results/E079/comparison.csv` |
| 聚合摘要 | `workspace/core4d/results/E079/aggregate_summary.json` |
| Contact quality | `workspace/core4d/results/E079/contact_quality.csv` |
| NPZ/MP4 | `workspace/core4d/results/E079/E079_*.npz`, `workspace/core4d/results/E079/E079_*.mp4` |
| 关键帧 | `workspace/core4d/results/E079/keyframes/*/f*.jpg` |
| Contact sheets | `workspace/core4d/results/E079/keyframes/contact_sheets/*_sheet.jpg` |
| 本地日志 | `logs/E079/E079_*_p1.log`, `logs/E079/local_gpu0.log` |
| 远程日志 | `logs/E079/E079_*_p2.log`, `logs/E079/remote_gpu1.log` |
| Scene snapshot | `workspace/core4d/results/E079/scene_snapshot/*/` |

## 量化指标

### Case-Specific Window

本轮修正后的主量化口径是 case-specific contact/intent window：

- 来源：每个 case 当前训练使用的 `eval_contact_mask_3cm`。
- person：使用该 variant 的 `contact_hdmi_mask_person_idx`。
- hand：左右手取 OR，表示该 person 至少一只手与物体 3cm 几何接触。
- window：从首次 active frame 到末次 active frame，并向前/后各 padding 10 frame；若末端越界则截断。
- 统计：在该 case-specific window 内统计 object error、sim/ref contact、hand SDF、ctrl deviation、pelvis min。

case-window 数值成功标准仍沿用 E079 原三阈值：

```
case_window_pelvis_z_min_m >= 0.55
case_window_sim_contact_frames_pct >= 50
case_window_obj_err_mean_m <= 0.20
```

对应字段：

- `E079_success_case_window`：case-specific window 下的主量化成功标签。
- `main_case_window_success_pct`：只统计 `role=main` 的 case-window 成功率。

### Fixed Post2 Diagnostic

固定 `post2` 口径保留作历史兼容诊断：

```
post2_pelvis_z_min_m >= 0.55
post2_sim_contact_frames_pct >= 50
post2_obj_err_mean_m <= 0.20
```

- `post2=2.0s-3.6s` 是 E072/E078 在 `box023` 上定位 post-2s hold/place failure 时引入的窗口；它大致对应 `box023` 的搬运后半段/放置阶段。
- 这个窗口不是 CORE4D 多 case 的通用接触/任务窗口。不同 case 的接触开始、释放、放置和动作长度不同，因此不能直接用固定 `2.0s-3.6s` 判断其他 case 是否成功。
- 因此，`E079_success_numeric` 和 `main_numeric_success_pct=30%` 应理解为“沿用 box023 诊断窗口的 fixed-window 参考指标”，不再作为主成功率。

指标来源：

- 表格数据来自 `workspace/core4d/scripts/eval/eval_E079.py` 生成的 `workspace/core4d/results/E079/comparison.csv`。
- `eval_E079.py` 复用 `workspace/core4d/scripts/eval/eval_E078.py::evaluate_variant()`：逐个读取 `workspace/core4d/results/E079/E079_*.npz`，加载对应 override/ref trajectory 和 scene model，replay sim/ref qpos 后统计物体、pelvis、接触、SDF、ctrl 差异。
- `CaseWin` 是 `eval_E079.py` 写入的 `E079_success_case_window`，是本轮修正后的主量化标签。
- `Numeric` 是 `eval_E079.py` 写入的旧 `E079_success_numeric`，只对固定 post2 窗口的量化指标负责，保留作历史诊断参考。
- `Visual` 来自 `.mp4` 抽帧后的 contact sheet 观察和 subagent 复核，不参与 `Numeric` 计算。
- `post2_*` 统计窗口为 eval time `2.0s-3.6s`，即 50Hz 下约 frame `100-180`；这是从 `box023` 诊断继承的固定窗口，不保证覆盖其他 case 的真实任务阶段。

表格列含义：

| 列 | 含义 |
|----|------|
| `Variant` | 本轮运行名，对应 `workspace/core4d/scripts/E079/variants.tsv` 和同名 `E079_*.npz/.mp4`。 |
| `Role` | `main` 计入 E079 泛化成功率；`guard` 是 calibration/回归检查，不计入主成功率。本轮 guard 为 E078 已验证成功的 `box023_p2`。 |
| `Split` | 运行位置，`local` 为本机 RTX5090，`remote` 为 `spider-remote` GPU1。 |
| `CaseWin` | 是否同时满足 case-specific window 下的三阈值；这是修正后的主量化标签。 |
| `Window s` | case-specific window 的起止 eval time，单位秒。 |
| `Numeric` | 是否同时满足固定 post2 窗口下的 `post2_pelvis_z_min_m >= 0.55`、`post2_sim_contact_frames_pct >= 50`、`post2_obj_err_mean_m <= 0.20`；仅为诊断参考。 |
| `Obj mean/max m` | 对应窗口内 sim object center 与 ref object center 的欧氏距离均值/最大值，单位米；越小越好。 |
| `Pelvis min m` | 对应窗口内 sim pelvis 世界坐标 z 最小值，单位米；低于阈值通常表示蹲跪、倒地或严重失衡。 |
| `Ref/Sim contact %` | 对应窗口内 ref/sim 至少一只手与物体发生几何接触的帧比例；sim 太低表示脱手。 |
| `Hand SDF mean m` | 对应窗口内 sim 左/右手到物体 signed-distance 的较小值均值，单位米；越小代表手离物体越近，但不单独等价于真实抓握。 |
| `Ctrl Linf` | 对应窗口内 robot ctrl 与 ref ctrl 的逐帧 L-infinity 差异最大值，单位为对应 actuator ctrl 单位；过大通常表示 CEM 为满足接触/物体目标而偏离 ref 动作。 |
| `Visual` | 人工/视频关键帧复核的简短结论，用于补充量化指标不能覆盖的倒地、隔空接触、物体漂移和动作语义问题。 |

合并评估结果：

```json
{
  "num_results": 11,
  "num_main_results": 10,
  "num_main_case_window_success": 6,
  "main_case_window_success_pct": 60.0,
  "num_main_numeric_success": 2,
  "main_numeric_success_pct": 20.0,
  "guard_results": ["E079_box023_p2"]
}
```

### Case-Specific Window 结果

| Variant | Role | Split | CaseWin | Window s | Obj mean/max m | Pelvis min m | Ref/Sim contact % | Hand SDF mean m | Ctrl Linf | Visual |
|---|---|---|---|---|---|---|---|---|---|---|
| `E079_box021_p1` | main | local | False | 0.42-3.30 | 0.240/0.598 | 0.683 | 9.0/41.4 | 0.086 | 0.854 | visual good：视觉质量好，但 ref 接触位置本身异常 |
| `E079_box023_p1` | main | local | True | 0.54-3.44 | 0.167/0.280 | 0.656 | 89.0/68.5 | 0.066 | 0.654 | known fail：p1 数据/retarget 质量反例，case-window 数值 false positive |
| `E079_bucket001_p1` | main | local | False | 0.46-3.28 | 0.289/0.407 | 0.182 | 88.7/0.0 | 0.357 | 1.898 | fail：跪姿/低 pelvis |
| `E079_bucket005_s2_p1` | main | local | True | 0.42-4.62 | 0.170/0.408 | 0.651 | 92.9/72.5 | 0.048 | 0.853 | visual fail/weak：物体持续受机器人力而旋转 |
| `E079_bucket007_p1` | main | local | True | 1.86-4.82 | 0.198/0.396 | 0.705 | 100.0/67.8 | 0.148 | 0.978 | data/trim issue：前摇和 ref 支撑脚误差导致初始后退，后续弥补 |
| `E079_desk021_p1` | main | local | True | 0.42-4.04 | 0.145/0.335 | 0.727 | 96.2/62.1 | 0.059 | 0.713 | partial：前段没有抬起来，后段才相对正常 |
| `E079_box021_p2` | main | remote | False | 0.00-2.70 | 0.536/0.961 | 0.131 | 81.6/36.8 | 0.081 | 1.098 | fail：早期倒地 |
| `E079_box023_p2` | guard | remote | True | 0.46-3.44 | 0.162/0.317 | 0.679 | 88.0/69.3 | 0.071 | 0.669 | positive guard：E078 已验证成功，本轮复现成功 |
| `E079_bucket001_p2` | main | remote | False | 0.54-4.48 | 0.426/0.743 | 0.102 | 85.9/85.9 | 0.041 | 1.363 | fail：倒地/侧躺 |
| `E079_bucket005_s2_p2` | main | remote | True | 0.52-4.56 | 0.159/0.353 | 0.635 | 93.1/89.2 | 0.033 | 0.960 | visual near-pass：稳定但接触略假 |
| `E079_bucket007_p2` | main | remote | True | 0.76-3.78 | 0.117/0.243 | 0.753 | 100.0/94.7 | 0.052 | 0.599 | pass visual：可用，有相对位置偏移 |

### Fixed Post2 诊断结果

| Variant | Role | Split | Numeric | Obj mean/max m | Pelvis min m | Ref/Sim contact % | Hand SDF mean m | Ctrl Linf | Visual |
|---|---|---|---|---|---|---|---|---|---|
| `E079_box021_p1` | main | local | False | 0.277/0.598 | 0.719 | 0.0/46.1 | 0.114 | 0.854 | visual good：视觉质量好，但 ref 接触位置本身异常 |
| `E079_box023_p1` | main | local | False | 0.173/0.280 | 0.700 | 80.2/48.1 | 0.096 | 0.729 | known fail：p1 数据/retarget 质量反例，case-window 数值 false positive |
| `E079_bucket001_p1` | main | local | False | 0.213/0.395 | 0.144 | 70.4/0.0 | 0.338 | 1.868 | fail：跪姿/低 pelvis |
| `E079_bucket005_s2_p1` | main | local | False | 0.238/0.408 | 0.729 | 100.0/85.2 | 0.028 | 0.529 | visual fail/weak：物体持续受机器人力而旋转 |
| `E079_bucket007_p1` | main | local | False | 0.200/0.396 | 0.705 | 100.0/54.3 | 0.192 | 0.978 | data/trim issue：前摇和 ref 支撑脚误差导致初始后退，后续弥补 |
| `E079_desk021_p1` | main | local | True | 0.142/0.335 | 0.730 | 100.0/87.7 | 0.042 | 0.655 | partial：前段没有抬起来，后段才相对正常 |
| `E079_box021_p2` | main | remote | False | 0.711/0.939 | 0.131 | 48.0/18.0 | 0.253 | 0.976 | fail：早期倒地 |
| `E079_box023_p2` | guard | remote | True | 0.157/0.317 | 0.698 | 79.0/71.6 | 0.096 | 0.672 | positive guard：E078 已验证成功，本轮复现成功 |
| `E079_bucket001_p2` | main | remote | False | 0.593/0.743 | 0.144 | 91.4/100.0 | 0.013 | 1.255 | fail：倒地/侧躺 |
| `E079_bucket005_s2_p2` | main | remote | False | 0.246/0.353 | 0.726 | 100.0/100.0 | 0.023 | 0.394 | pass visual：稳定但接触略假 |
| `E079_bucket007_p2` | main | remote | True | 0.154/0.243 | 0.753 | 100.0/96.3 | 0.059 | 0.557 | pass visual：可用，有相对位置偏移 |

## 可视化观察

可视化基于每个 `.mp4` 抽取的关键帧 `f50/f75/f100/f115/f120/f125/f145/f160/f180`，并合成 contact sheet。三个 subagent 分组复核了 p1、p2 和成功/失败对比。

### 正例或接近正例

- `E079_box023_p2`：positive guard，机器人基本保持站立，箱子没有明显掉落或发散；手-物接触大体可信，但搬运中箱体姿态和脚步有一定不自然，后段放下动作略僵硬。
- `E079_bucket007_p2`：机器人保持站立，大桶稳定，整体时序和物体位置一致性明显好于 p1；后段前倾和接触位置仍有偏差。
- `E079_bucket005_s2_p2`：视觉上相对最好之一，几乎全程站稳，桶没有明显掉落；但桶像被约束在身体前，抓握真实感不足，因此数值上因 obj mean error 未过 0.20m。
- `E079_box021_p1`：用户复查认为视觉质量很好；但该序列 ref 的接触位置本身比较奇怪，因此它更像“视觉可用但 reference/contact 语义需单独解释”的样本，而不是由当前 contact 指标直接判定的失败。

### 负例

- `E079_box021_p2`：从 f50 起 sim 已倒地/贴地，后续箱子与人体分离，动作语义完全失败。
- `E079_bucket001_p2`：多帧倒地或侧躺，桶压在/靠近身体，后续没有恢复稳定站立交互。
- `E079_bucket001_p1`：未完全飞走，但 pelvis 很低，多次蹲跪或半跪，像用低姿态补偿接触，不能作为可用结果。
- `E079_bucket007_p1`：用户复查指出开头是人类第一视角录制前按摄像头/摸头的前摇动作，trim 没有剔除干净；同期 ref 右腿也疑似没有稳定接地，属于重定向或动捕支撑脚误差。sim 为避免单脚不稳选择左脚后撤一步，导致整体初始位置比 ref 向后，早期没有接触到物体；后续 CEM 有一定弥补，后半段接触基本恢复。因此它不应简单归为“算法接触失败”，而应标为 trim/ref data-quality issue。
- `E079_box023_p1`：作为已知 p1 失败/数据质量反例，前中段稳定，但末段箱子离手后落地/漂移；和 E078A 一致，仍不应被误判为成功。case-window 三阈值会把它判 True，说明当前量化判据仍不够。
- `E079_desk021_p1`：用户复查认为不是特别好，前段没有把桌子抬起来，后段才相对正常；因此从 pass 降级为 partial。
- `E079_bucket005_s2_p1`：用户复查指出物体一直受机器人力并持续旋转，不应视为 near-pass；case-window 数值 True 在这里高估了视觉质量。

## Claims 验证

| Claim | 结果 | 说明 |
|------|------|------|
| C1 pipeline 可泛化到 6 个 B+C 序列 p1/p2 | 部分通过 | `11/12` 可运行，超过至少 10 个的最低要求；`desk021_person2` retarget infeasible。 |
| C2 3cm contact audit 能筛出高质量 ref/contact | 通过但需改进 | 11 个可用 case 均有 mask 和 quality CSV；但 high-quality proxy 只说明几何接触存在，不能保证动力学稳定或真实抓握。 |
| C3 通用 CEM 配置在高质量 case 泛化 | 未通过 | case-specific window main 数值成功 `6/10`，低于计划的 `>=7/10`；用户复查后视觉近可用更保守，主要保留 `box021_p1`、`bucket007_p2`、`bucket005_s2_p2`，另有 `box023_p2` positive guard。`box023_p1`、`bucket005_s2_p1`、`desk021_p1` 说明三阈值会高估视觉质量；`bucket007_p1` 更像 trim/ref data-quality issue。 |
| C4 `box023` known-case sanity | 部分通过 | `box023_p2` 是 E078 已验证成功 positive guard，本轮 case-window 与 fixed-post2 均为 True，视频也可用；`box023_p1` 是已知失败/数据质量反例，视觉仍失败，但 case-window 数值为 True，说明三阈值会误判。 |
| C5 不新增 hand-crafted contact window | 通过 | 主验证 `hold_contact_rew_scale=0.0`，无固定 hold window。 |
| C6 远程调度符合资源约束 | 通过 | 远程仅 GPU1 跑 p2 分片，本地 RTX5090 跑 local 分片。 |

## 分析

### 1. E077 pipeline 成功，但 contact quality proxy 不等于动力学质量

E079 证明 3cm per-EEF mask 和通用 scene 构造可以跑过 10+ case。这个结论重要，因为 E078B 不再是单个 `box023_person2` 的偶然样本。

但 `contact_quality.csv` 的 high-quality proxy 主要衡量 ref/mask 的几何接触覆盖率；它无法保证：

- retarget 后人体支撑姿态合理；
- CEM 能在接触 reward 和稳定性之间找到同一动作语义；
- 物体相对人体的位置和朝向不会漂移；
- 单手/双手语义与物体动力学一致。

`box021_p2`、`bucket001_p2` 都有较高 ref/contact 指标，但 sim 直接倒地，说明下一步需要加入 data-quality 中的 stability/pose feasibility audit，而不是只看 hand-object distance。

### 2. Case-specific window 改变了结论，但没有让 C3 通过

`bucket005_s2_p2` 视觉最好之一，在 fixed post2 口径下因为 `post2_obj_err_mean_m=0.246m` 超过 0.20m 被判 False；改用 case-specific window 后变为 True，说明用户指出的问题是实质性的：固定 `box023` 窗口确实会误伤其他 case。

但 case-specific window 也暴露了另一个问题：三阈值仍不足以完整表达动作语义。

- `bucket005_s2_p1` 在 case-window 中过阈值，但用户复查显示物体一直受机器人力并持续旋转，说明 contact/object mean 指标没有捕捉物体姿态和受力语义。
- `desk021_p1` 在 case-window 中过阈值，但用户复查显示前段没有抬起来，后段才相对正常，说明只看窗口均值会掩盖阶段性失败。
- `bucket007_p1` 在 case-window 中 obj/contact/pelvis 都过阈值；用户复查显示早期偏差来自前摇未 trim 干净和 ref 支撑脚误差，sim 后撤避免摔倒后导致初始位置偏后，后续算法弥补。因此它更应进入 trim/ref feasibility audit，而不是直接归因于 CEM 接触失败。
- `box023_p1` 是已知失败/数据质量反例，case-window 数值也过阈值，但视频仍显示后段掉落/漂移。

因此，case-specific window 是必要修正，但还需要加入接触语义连续性、object max error、visual usable / near-pass 标签，才能成为正式 C3 判据。

### 3. 失败主要分两类

第一类是支撑稳定性崩溃：

- `box021_p2`
- `bucket001_p2`
- `bucket001_p1`

这些 case 后续 contact/obj 指标没有太大解释价值，因为身体已经换成了倒地/跪姿语义。

第二类是接触弱或量化 false positive：

- `box023_p1`
- `box021_p1`

这些 case 没有完全倒地，但量化指标与视觉语义不完全一致，后续更适合做 contact target、eef offset、palm normal 或 data alignment 诊断。

第三类是 trim/ref data-quality issue：

- `bucket007_p1`

该 case 的主要问题不是单纯接触 reward 失败，而是开头前摇没有剔除干净，加上 ref 支撑脚疑似不稳定，导致 sim 为保持平衡后撤一步，初始位置偏离 ref。后续 CEM 已经有一定弥补。

## 下一步建议

1. 将 E079 正例/近正例分成三组继续分析：
   - main visual positive/near-positive：`box021_p1`、`bucket007_p2`、`bucket005_s2_p2`
   - positive guard：`box023_p2`
   - case-window false positive / overestimated：`box023_p1`、`bucket005_s2_p1`、`desk021_p1`
   - data/trim issue：`bucket007_p1`
   - negative：`box021_p2`、`bucket001_p2`、`bucket001_p1`
2. 检查 Holosoma / SPIDER trim 逻辑，重点看 `bucket007_p1` 这类第一视角开头按摄像头/摸头前摇是否未被 `trim_no_contact.py` 或显式 trim window 剔除；必要时加入 pre-contact gesture trimming 或人工 audit 标记。
3. 继续改进 E079 评估脚本：case-window 已接入，但下一版应增加 `case_window_obj_err_max_m` 阈值、接触连续性/断触次数、早期接触建立检查、物体旋转/姿态误差，以及 `visual_usable` / `near_success` 手工或半自动标签。
4. 新增 data-quality audit：评估 retarget reference 的 pelvis height、foot support、knee/foot contact、object-human relative trajectory，不再只用 hand-object 3cm mask 作为质量代理。
5. 对 `bucket005_s2_p2` 做单独诊断：它视觉最稳但 fixed-window obj mean error 未过阈值，适合用来区分“窗口不匹配”与“物体跟踪仍不够准”。
6. 对倒地负例先做 reference feasibility/initial support audit，再决定是否进入 CEM reward 调参；不要在明显数据/支撑失败 case 上继续调 contact reward。

## 复现命令

```bash
bash workspace/core4d/scripts/run_E079_preprocess.sh
bash workspace/core4d/scripts/train/train_E079.sh local 0
bash workspace/core4d/scripts/run_E079_remote.sh
bash workspace/core4d/scripts/pull_E079_remote_results.sh
python workspace/core4d/scripts/eval/eval_E079.py
```
