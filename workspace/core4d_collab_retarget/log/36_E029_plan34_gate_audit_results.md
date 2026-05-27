# E029 计划 34 gate 审计结果

日期：2026-05-27

## 目标

`plan/34_E029_cola_d6_support_body_redesign_plan.md` 明确要求：

- 先完成 E018/E028 的 support 语义审计；
- 再完成 5 个 E028 候选 case 的 axis/contact 预检查；
- 再跑不训练的 D6 support load-path sanity；
- 只有候选 case 的 D6 sanity 达到 `>=4/5` 时才进入 full CEM。

本轮补齐计划中列出的评估入口 `scripts/eval/eval_E029.py`，把 E029 所有 sanity 输出汇总成一个可复核的 gate 决策，避免只凭单个日志手动判断。

## 代码变更

新增：

- `scripts/eval/eval_E029.py`

功能：

- 读取 `results/E029/audit/e028_candidate_modes.csv`；
- 读取 `results/E029/preflight/axis_contact_summary.csv`；
- 统计 `d6/freejoint/connect/multiconnect` 的 manifest；
- 扫描所有 `results/E029/*/sanity/candidates_*_summary.csv`；
- 计算最佳候选 sanity 结果，并判断是否达到 `>=4/5` 的 full gate；
- 将 baseline modes `direct-object-wrench-control`、`mocap-weld-current` 标为不参与 gate，避免把非 dynamic-D6 对照误算为 full gate；
- 检查 axis/contact 面板、sanity 视频、抽帧是否存在；
- 写出：
  - `results/E029/eval/eval_summary.md`
  - `results/E029/eval/eval_summary.json`
  - `results/E029/eval/sanity_overview.csv`

运行命令：

```bash
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E029.py --all
```

静态检查：

```bash
python -m py_compile workspace/core4d_collab_retarget/scripts/eval/eval_E029.py
```

通过。

## Gate 审计

评估输出：

- `results/E029/eval/eval_summary.md`
- `results/E029/eval/eval_summary.json`
- `results/E029/eval/sanity_overview.csv`

汇总：

| 项目 | 证据 |
|---|---:|
| 候选 case 数 | `5` |
| 审计行数 | `5` |
| 预检查行数 | `5` |
| 扫描到的候选 sanity 汇总 | `30` |
| 最佳候选 sanity | `3/5` |
| full CEM 决策 | `stop_before_full_sanity_failed` |

最佳候选 sanity：

- 汇总文件：`results/E029/d6/sanity/candidates_d6-locked-support_raw_ref_candidates_locked_raw_summary.csv`
- 通过数：`3/5`
- object mean 平均/最差：`0.116/0.127m`
- support drift mean 平均/最差：`0.007/0.010m`
- force saturation 最大值：`0`
- NaN 总数：`0`

Manifest 覆盖：

| 分支 | 行数 |
|---|---:|
| scalar D6 | `10` |
| freejoint | `10` |
| point connect | `5` |
| multiconnect | `20` |

Baseline mode 覆盖：

| 模式 | 通过数 | Object mean 平均/最差 | 是否参与 gate |
|---|---:|---:|---|
| `mocap-weld-current` | `0/5` | `0.078/0.091m` | 否 |
| `direct-object-wrench-control` | `0/5` | `0.187/0.214m` | 否 |

可视化覆盖：

- axis/contact 面板：`5`
- sanity 视频：`12`
- 抽帧：`11`

## 对照计划 34 的完成度审计

| 要求 | 证据 | 状态 |
|---|---|---|
| 只使用 `results/E028/candidates.json` 中的 case | eval 中候选数为 `5`；所有 manifest/summary 都使用这 5 个 E028 variants | 完成 |
| 审计 E018/E028 support 语义 | `results/E029/audit/current_support_semantics.md`, `e028_candidate_modes.csv` | 完成 |
| Axis/contact 预检查 | `results/E029/preflight/axis_contact_summary.csv`，5 个面板 | 完成 |
| 生成 D6-equivalent scene/data | scalar D6 manifest `10` 行，freejoint manifest `10` 行 | 完成 |
| Baseline sanity modes | `baseline/sanity/candidates_mocap-weld-current_*`, `baseline/sanity/candidates_direct-object-wrench-control_*` | 完成 |
| Dynamic D6/load-path sanity | `eval_E029.py` 扫描 30 个候选 sanity 汇总 | 完成 |
| 可视化 | 5 个 axis 面板，12 个 sanity 视频，11 个抽帧 | 完成 |
| 候选 D6 sanity `>=4/5` | 最佳 gate-eligible 候选 summary 为 `3/5` | 失败 |
| Full CEM | 受 C5 gate 控制；候选 sanity 失败，所以不允许启动 | 已正确停止 |
| Tracker/progress/log 更新 | tracker 行、关键指标行、logs 30-36、progress 记录 | 完成 |

## 声明验证

| 声明 | 状态 | 证据 |
|---|---|---|
| C1: E018/E028 不是 COLA dynamic support body | 通过 | 审计显示 `5/5 mocap_pad`，`0/5 cola_d6_match` |
| C2: E029 已尝试 dynamic support body | 通过 | scalar D6/freejoint manifest 存在；non-mocap support 分支已编译并运行 |
| C3: 已尝试 D6-equivalent 连接 | 已尝试但未过 gate | scalar weld、freejoint weld、point-connect、multiconnect 均已运行；没有候选配置达到 `>=4/5` |
| C4: 不再假设固定 canonical-z 轴 | 通过 | preflight 覆盖 `5/5`；Box021 高度轴全部为 local `y` |
| C5: 不训练 load path 已被证明可用 | 失败 | 最佳候选 sanity 为 `3/5`，低于要求的 `>=4/5` |
| C6: full retarget 是否允许启动 | 不允许 | C5 失败 |
| C6: full retarget 是否正确停止 | 通过 | E029 D6 分支没有启动 full CEM |
| C7: 可视化是否完成 | 通过 | panels/videos/frames 已存在，并在 logs 30/32/34/35 中记录观察 |

## 决策

E029 现在已经按计划 34 执行到 full gate，但它**没有满足 D6 sanity 成功标准**。因此 full CEM 继续停止。

这个结论比早先的人工判断更强，因为：

- 旧 E018/E028 的 anchor/preprocessing 问题已经被验证；
- corrected endpoint 和 preflight 覆盖已经被验证；
- 多种 support-object load path 近似都已经尝试；
- 最佳候选结果仍是 `3/5`，不是 `>=4/5`；
- 因此失败不是因为缺少可视化或缺少 eval 步骤。

如果继续 E029，下一步必须是新的方法分支：true per-axis D6 / relative-pose impedance support-object controller。它应该先单独写计划再实现，因为继续重复 equality-weld/connect anchor 或 gain sweep 会违反“不要重复同一失败配置”的规则。
