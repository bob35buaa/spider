# E163 下游 RL 结果分析与 SPIDER 评测优化启发

日期：2026-06-17

## Summary

- E163 `narrowSurfaceBand` 在 SPIDER/MuJoCo 侧确实修复了 E161 的 raw contact 回退，并且三 case handoff 是 3/3 `RL_EXPORT_READY`；但下游 RL 表明，`raw contact / clean contact / penetration / tracking` 仍不是充分条件。
- 当前没有证据表明本轮 RL 失败来自训练中断或 eval 未跑完。5 组本轮实验都有 `model_5999.pt`，rollout log 都显示 64 env 评测流程完成；失败组的问题是没有任何 env 保存成 `trajectory_complete`。
- 此前指出的 SUGAR 诊断缺口已经修复，并用修复后的 rollout-only eval 复跑三组 0-completion case 验证：每组都落盘 64 条 `failed_windows.csv` 记录和 64 个失败 npz。
- RL 结果分成三类：`box004_r161/spider_e163` 是 target/progress 成功但 height gate 失败；`box021_r160/omniretarget` 直接证据显示 `obj_pos` 为主、少量 `anchor_pos/ee_body_pos` 早停；`box023_r158/spider_e163` 直接证据显示 `ee_body_pos` 为主。
- 对 SPIDER 评测的直接启发是：下游预测指标必须从“MuJoCo raw contact 单轴 hard gate”扩展成 `SUGAR/Isaac filtered contact + net-filter gap + dynamic feasibility + lift/height preservation + termination-risk` 的组合判据。

## 1. Evidence

主要证据来源：

| 类型 | 路径 |
|---|---|
| RL 汇总 | `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/docs/plan/CORE4D_E163_OMNIRT_AND_SPIDER_E163_RL_RESULTS_CN.md` |
| SUGAR RL 输出 | `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e163_refiner_rl/` |
| Box004 OmniRT RL 输出 | `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d_e163_omnirt_refiner_eval/` |
| SUGAR 失败字段复跑 | `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e163_refiner_rl/*/*/eval_failed_fields_rerun_20260617_170731/` |
| E163 SPIDER 三 case 结果 | `workspace/core4d/log/206_E163_narrow_surface_band_results.md` |
| E163 RL handoff | `workspace/core4d/log/207_E163_narrow_surface_band_rl_export_results.md` |
| E163 clean8 扩展 | `workspace/core4d/log/208_E163_narrow_surface_band_clean8_results.md` |
| Box004 clean8 失败分析 | `workspace/core4d/log/209_E163_box004_082_p1_failure_analysis.md` |
| E163 RL export table | `workspace/core4d/results/E163/narrow_surface_band_rl_export/s6_downstream/rl_export/rl_export_input.tsv` |

本报告只分析已有结果，不启动新训练。

## 2. Bug And Issue Audit

### 2.1 训练和 eval 完整性

本轮 5 组新增实验均满足：

| check | 结果 |
|---|---|
| `model_5999.pt` | 5/5 存在 |
| 训练日志 | 5/5 到 `Learning iteration 5999/6000` |
| eval summary | 5/5 存在 |
| rollout log | 5/5 显示 `All 64 envs completed` |
| policy video | 5/5 存在 |

结论：三组 0-completion 不是 eval 中断，而是 policy rollout 在所有窗口内都没有达到 `trajectory_complete` 保存条件。

### 2.2 诊断 bug 复核：失败 rollout 字段已修复并验证

SUGAR 代码路径：

```text
/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py
```

当前修复内容已经覆盖此前缺口：

- `MotionCommand._record_rollout_termination` 对非 timeout 终止调用 `_save_failed_window`。
- 每个失败窗口写入 `failed_windows.csv`，字段包括 `termination_reason`、`all_termination_reasons`、`last_anchor_pos_error`、`last_ee_error`、`last_obj_pos_error`、`last_obj_ori_error` 和对应 threshold。
- 失败片段也保存为 npz，目录按 reason 分组，例如 `raw_npz/obj_pos/*.npz`、`raw_npz/ee_body_pos/*.npz`。
- `evaluate_refiner_rollout.py` 会读取 `failed_windows.csv` 和失败 npz，生成 `analysis/failed_windows.csv`、`analysis/failed_trajectory_metrics.csv`，并在 0-completion summary 中输出 failure reason 分布。

为避免覆盖原始 RL 结果，本次只对三组 0-completion case 做 rollout-only 复跑，输出 tag 为：

```text
eval_failed_fields_rerun_20260617_170731
```

复跑结果：

| case | source | complete | failed windows | failure reasons | mean frames | mean duration ratio | key last error |
|---|---|---:|---:|---|---:|---:|---|
| `box021_r160` | `omniretarget` | 0/64 | 64 | `obj_pos=49`, `anchor_pos=9`, `ee_body_pos=6` | 81.2 | 0.647 | `last_obj_pos_error=0.308 > 0.3` |
| `box023_r158` | `omniretarget` | 0/64 | 64 | `obj_ori=59`, `obj_pos=5` | 59.8 | 0.260 | `last_obj_ori_error=0.817 > 0.8` |
| `box023_r158` | `spider_e163` | 0/64 | 64 | `ee_body_pos=60`, `obj_pos=3`, `anchor_pos=1` | 88.0 | 0.392 | `last_ee_error=0.325 > 0.3` |

结论：

- 诊断 bug 对新生成的 rollout 已解决；后续 0/64 不再是黑盒。
- 原始 `eval/analysis/summary.md` 仍是修复前 artifact，因此其中三组 0-completion 的 `End reasons: {}` 不能再作为“没有原因”的证据。
- 修复没有改变成功率：三组仍是 0/64 complete，只是失败机制从训练日志间接推断变成了 rollout 级直接证据。
- 代码注释中仍有一句旧说明“非 timeout 不保存 trajectory”，但实际代码已经保存失败 npz；建议后续顺手改注释，避免维护者误读。

### 2.3 路径复现性隐患

远程训练拉回来的 `params/env.yaml` 中，部分 `motion_folder` 是远程绝对路径：

```text
/home/xiayb/pHRI_workspace/Loco-Manipulation-projects/SUGAR-private/data/...
```

本次本地 eval 没有被污染，因为 `watch_and_eval_core4d_e163_refiner.sh` 显式传入了本地 `--motion_folder` 和 `--object_usd`。但如果后续直接用 checkpoint 目录中的 `params/env.yaml` 复现，会路径不可用。

建议：SUGAR 训练保存的 params 中同时写入 `motion_folder_rel`、`object_usd_rel`，并在 eval 脚本优先使用显式 CLI override。

### 2.4 Box004 资产缺失已修复，但应加入 preflight

远程 `box004_r161/spider_e163` 首次启动失败，原因是远程缺少：

```text
descriptions/objects/Box004/Props/instanceable_meshes.usd
descriptions/objects/Box004/config.yaml
```

后续已同步完整 `descriptions/objects/Box004/` 并重启成功。这个不是最终结果的污染源，但暴露出 preflight 不够严格。

建议：

- SUGAR launcher 在训练前 hard check `object_usd`、object config、instanceable mesh。
- 不要在 object-specific USD 缺失时静默依赖通用或错误 `object_urdf_fallback`。
- banner 里仍显示 `object_urdf_fallback` 指向 Box021，这个字段至少需要改名为 fallback-only diagnostic，避免误读。

### 2.5 口径问题不是 bug，但会导致结论误读

必须分开报告：

| 口径 | 含义 |
|---|---|
| `completion_rate` | 是否保存 `trajectory_complete` rollout |
| `SUGAR target success 0.3m` | final object 到目标 0.3m 内 |
| `progress_only_success` | 沿目标方向移动进度是否达标 |
| `height_success` | Holosoma-like lift/carry 高度 gate |
| `Holosoma-like success` | progress + height |

`box004_r161/spider_e163` 是典型反例：completion、target、progress 都是 64/64，但 height 是 0/64，因此严格 Holosoma-like success 是 0/64。不能把它简单写成“RL 成功”或“RL 失败”，必须说明是哪一种口径。

### 2.6 clean8 与三 case RL 不应混用

E163 clean8 扩展结论是 7/8 pass，`box004_082_p1` 因 raw contact 退化失败；但本次下游 RL 批次消费的是 E163 三 case export：

```text
box023_person2
d003_box021_20231018_029_p2
e091_box004_20231003_2_083_p2
```

因此不能把 clean8 的 `box004_082_p1` blocker 直接解释成本轮 RL 的 `box004_r161` 现象。它们都提示 raw-contact 口径脆弱，但不是同一个 case。

## 3. Upstream E163 Result Recap

E163 三 case 在 SPIDER 侧通过了原定 hard gate：

| short case | Core4D case | raw contact in mask | clean3 contact | physPen3 | geomPen2 in mask | tracked | fall | obj err m |
|---|---|---:|---:|---:|---:|---|---|---:|
| `box023_person2` | `box023_person2` | 0.8769 | 0.8000 | 0.0368 | 0.0923 | true | false | 0.0069 |
| `box021_029_p2` | `d003_box021_20231018_029_p2` | 0.7455 | 0.3818 | 0.2667 | 0.0727 | true | false | 0.0098 |
| `box004_083_p2` | `e091_box004_20231003_2_083_p2` | 0.6290 | 0.5161 | 0.0667 | 0.0645 | true | false | 0.0064 |

三 case export 也通过：

| item | result |
|---|---|
| `rl_export_input.tsv` | 3/3 `RL_EXPORT_READY` |
| `scene_act / trajectory / contact_mask / cem_result_npz` | 3/3 exists |
| `source_exp_id` | `E163` |
| `spider_method_id` | `gateA_surfaceBandA2_postureRerankA_narrowSurfaceBandReleaseDecay` |
| `target_variant_id` | `ref_fk` |
| partner OmniRetarget | 3/3 pass |

上游结论边界：这些指标证明 E163 是更好的 CEM/retarget 候选，但不等价于 downstream RL success。

## 4. RL Results

Holosoma-like 聚合结果：

| case | source | complete | H-like | progress | height | target0.3 | z mean / th | z max / th | final err | contact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `box004_r161` | `omniretarget` | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 0.217/0.180 | 0.542/0.372 | 0.098 | 0.597 |
| `box021_r160` | `omniretarget` | 0/64 | 0/64 | 0/64 | 0/64 | 0/64 | n/a | n/a | n/a | n/a |
| `box023_r158` | `omniretarget` | 0/64 | 0/64 | 0/64 | 0/64 | 0/64 | n/a | n/a | n/a | n/a |
| `box004_r161` | `spider_e163` | 64/64 | 0/64 | 64/64 | 0/64 | 64/64 | 0.137/0.164 | 0.191/0.338 | 0.041 | 0.540 |
| `box021_r160` | `spider_e163` | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 0.234/0.196 | 0.418/0.343 | 0.096 | 0.717 |
| `box023_r158` | `spider_e163` | 0/64 | 0/64 | 0/64 | 0/64 | 0/64 | n/a | n/a | n/a | n/a |

训练末期日志提供了更细的失败机制：

| case | source | reward | ep len | hoi | err anchor | err body | err obj | err obj rot | term complete | term ee | term obj pos | term obj ori | term anchor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `box004_r161` | `omniretarget` | 6.01 | 116.53 | 0.0727 | 0.1518 | 0.1363 | 0.1109 | 0.1064 | 0.8952 | 0.0452 | 0.0352 | 0.0206 | 0.0039 |
| `box004_r161` | `spider_e163` | 5.26 | 100.52 | 0.0650 | 0.2310 | 0.1807 | 0.0819 | 0.5902 | 0.8010 | 0.1342 | 0.0462 | 0.0072 | 0.0113 |
| `box021_r160` | `omniretarget` | 3.26 | 59.66 | 0.0370 | 0.2496 | 0.1180 | 0.2464 | 0.2612 | 0.0000 | 0.1551 | 0.4395 | 0.0354 | 0.3700 |
| `box021_r160` | `spider_e163` | 5.27 | 85.42 | 0.0515 | 0.1573 | 0.0903 | 0.1217 | 0.1386 | 0.9238 | 0.0122 | 0.0460 | 0.0166 | 0.0015 |
| `box023_r158` | `omniretarget` | 3.34 | 51.42 | 0.0343 | 0.2046 | 0.1101 | 0.2829 | 0.6932 | 0.0000 | 0.0336 | 0.2425 | 0.6884 | 0.0355 |
| `box023_r158` | `spider_e163` | 3.94 | 61.50 | 0.0430 | 0.1921 | 0.1517 | 0.2336 | 0.2169 | 0.0000 | 0.8621 | 0.1074 | 0.0031 | 0.0275 |

这张训练末期表说明：0/64 不是同一种失败。修复后 rollout-only 复跑进一步把该判断落到每个失败窗口上：

- `box021_r160/omniretarget`：rollout 失败窗口为 `obj_pos=49/64`、`anchor_pos=9/64`、`ee_body_pos=6/64`，主因是 object position tracking，anchor/body 是次级风险。
- `box023_r158/omniretarget`：rollout 失败窗口为 `obj_ori=59/64`、`obj_pos=5/64`，主因是物体姿态跟踪。
- `box023_r158/spider_e163`：rollout 失败窗口为 `ee_body_pos=60/64`、`obj_pos=3/64`、`anchor_pos=1/64`，主因是末端/肢体 tracking。

## 5. Case-Level Interpretation

### 5.1 `box004_r161/omniretarget`: clean strict success

这是最干净的成功正例：

- complete 64/64。
- progress 64/64。
- height 64/64。
- target 0.3m 64/64。
- training 末期 `trajectory_complete≈0.895`。

它说明 same-person OmniRetarget 在某些 case 上仍然可以直接成为 RL 可执行 reference。E163 不能被解释成“总是优于 OmniRetarget”，更准确的判断是 case-dependent。

### 5.2 `box021_r160/spider_e163`: clean strict success

这是 E163 的核心成功正例：

- complete 64/64。
- Holosoma-like success 64/64。
- contact_hands_fraction 0.7166，是六组里最高。
- training 末期 `trajectory_complete≈0.924`，`obj_pos / obj_ori / anchor_pos` 终止都很低。

它说明 E163 对 `box021_029_p2` 这类 case 不只是改善上游接触，也能转化成下游 RL 可执行性。

### 5.3 `box004_r161/spider_e163`: target 成功但 height 失败

这是最重要的口径反例：

- complete 64/64。
- progress 64/64。
- target 0.3m 64/64。
- final target error 0.041m，比 `box004_r161/omniretarget` 的 0.098m 还好。
- 但 height 0/64，`z_mean=0.137 < 0.164`，`z_max=0.191 < 0.338`。

解释：E163 在这个 case 上学到了能把物体送到目标附近的策略，但没有复现 Holosoma-like lift/carry 高度。若任务定义是“搬起再移动”，这是严格失败；若任务定义允许推/低位滑移，则它不是行为失败，而是 height gate 与任务语义不完全一致。

对 SPIDER 的启发：E163 当前 reward 更偏向接触清洁、目标附近和低穿透，不保证 lift/height profile。后续如果 downstream 需要 lift/carry，SPIDER/CEM selection 必须显式加入 height/lift preservation。

### 5.4 `box021_r160/omniretarget`: 接触好但 RL 失败

RL 汇总中的严格数据质量表显示，该组初始接触探针并不差：

- source contact 0.680。
- Isaac any contact 0.600。
- any-contact IoU 0.627。
- net-filter gap 0.013。

但 RL 末期：

- complete 0/64。
- `term_obj_pos≈0.440`。
- `term_anchor_pos≈0.370`。
- `term_complete=0`。

修复后 rollout-only 复跑给出直接证据：

- `failed_windows=64/64`。
- `termination_reason`: `obj_pos=49`、`anchor_pos=9`、`ee_body_pos=6`。
- 平均失败时间约 81.2 帧，`duration_ratio≈0.647`。
- `last_obj_pos_error≈0.308m`，刚越过 0.3m hard threshold。
- 失败片段的 `hands_contact_ratio≈0.659`，`object_z_height_mean≈0.241m`，说明它不是接触量或高度完全缺失。

解释：这里失败不主要由接触量不足解释，而更像 object path 动态、anchor/object tracking、速度/加速度或策略探索问题。继续只调 raw contact label 不会解决这个 case。

### 5.5 `box023_r158/omniretarget`: 物体姿态动态风险

训练末期：

- `term_obj_ori≈0.688`。
- `term_obj_pos≈0.243`。
- `err_obj_rot≈0.693`。
- complete 0/64。

修复后 rollout-only 复跑给出直接证据：

- `failed_windows=64/64`。
- `termination_reason`: `obj_ori=59`、`obj_pos=5`。
- 平均失败时间约 59.8 帧，`duration_ratio≈0.260`，比另外两组更早。
- `last_obj_ori_error≈0.817`，刚越过 0.8 rad hard threshold。
- 失败片段 final target error 均值约 1.41m，说明它在很早阶段就偏离到无法形成有效完整 rollout。

解释：这个 case 的主要失败信号是 object orientation，而不是单纯 hand contact。结合 RL 汇总中 Box023 较大的 net-filter gap，可能存在物体姿态跟踪、接触施力方向、asset/filter 语义不一致的叠加风险。

### 5.6 `box023_r158/spider_e163`: 接触表面不差，但身体/末端不可执行

训练末期：

- `term_ee_body_pos≈0.862`。
- `err_body≈0.152`，高于 `box021/spider_e163` 的 0.090。
- complete 0/64。

修复后 rollout-only 复跑给出直接证据：

- `failed_windows=64/64`。
- `termination_reason`: `ee_body_pos=60`、`obj_pos=3`、`anchor_pos=1`。
- 平均失败时间约 88.0 帧，`duration_ratio≈0.392`。
- `last_ee_error≈0.325m`，越过 0.3m hard threshold；`last_obj_pos_error≈0.242m`、`last_obj_ori_error≈0.204` 尚未到主要阈值。
- 失败片段 `hands_contact_ratio≈0.663`、`object_z_height_mean≈0.216m`，说明接触/高度表面上并不差，但身体/末端已经先失稳。

解释：E163 对 Box023 修复了 SPIDER/MuJoCo raw contact，但 RL policy 在 Isaac 中首先撞到末端/身体 tracking hard termination。这提示上游评测不能只看物体和接触，还要看 reference 对机器人本体是否可执行，特别是手腕、脚踝、root/anchor 的动态余量。

## 6. Contact Probe Lessons

RL 汇总中的 strict data quality 表给出了更接近 SUGAR reward 的接触诊断：

| case | source | strict RL | target/progress | height | source contact | Isaac any | any IoU | net-filter gap | observation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `box004_r161` | `omniretarget` | 64/64 | 64/64 | 64/64 | 0.581 | 0.571 | 0.635 | 0.000 | clean success |
| `box021_r160` | `omniretarget` | 0/64 | 0/64 | 0/64 | 0.680 | 0.600 | 0.627 | 0.013 | 接触好但动态失败 |
| `box023_r158` | `omniretarget` | 0/64 | 0/64 | 0/64 | 0.449 | 0.235 | 0.476 | 0.346 | net/filter 风险 |
| `box004_r161` | `spider_e163` | 0/64 | 64/64 | 0/64 | 0.609 | 0.036 | 0.058 | 0.077 | target 成功但 height 失败 |
| `box021_r160` | `spider_e163` | 64/64 | 64/64 | 64/64 | 0.778 | 0.479 | 0.615 | 0.017 | clean success |
| `box023_r158` | `spider_e163` | 0/64 | 0/64 | 0/64 | 0.486 | 0.324 | 0.667 | 0.315 | IoU 尚可但 net/filter gap 大 |

关键结论：

1. Isaac/SUGAR filtered contact 与 strict RL 成功相关，但不是充分条件。
2. `net-filter gap` 对 Box023 很有价值：它揭示未过滤物理接触和 SUGAR reward 使用的 hand-to-Obj filtered contact 不一致。
3. `any-contact IoU` 不能单独作为通过标准；`box023_r158/spider_e163` 的 IoU 不低，但 RL 仍 0/64。
4. `box021_r160/omniretarget` 说明接触强也可能失败，必须加入 dynamic feasibility。
5. `box004_r161/spider_e163` 说明 contact probe 差也不必然不能到目标，但可能变成低位推送/低高度搬运。

## 7. Implications For SPIDER Evaluation

### 7.1 评测指标要分层，而不是一个 hard gate

建议将 downstream-readiness 分成四层：

| layer | 指标 | 作用 |
|---|---|---|
| L1 Geometry | `geomPen2/5`, `physPen3/5`, release false | 排除明显穿透和 release-side 吸附 |
| L2 Contact Semantics | MuJoCo raw contact, clean contact, RL mask contact | 保证 source mask 内确实有可用接触 |
| L3 Isaac/SUGAR Compatibility | filtered hand-to-Obj force contact, any/both IoU, net-filter gap | 检查 SPIDER 接触是否能被 SUGAR reward 看见 |
| L4 Dynamic Feasibility | object/anchor/ee/body termination-risk, velocity/acceleration/height | 预测 RL policy 是否能完整执行 |

E163 证明 L1/L2 通过还不够；Box021/Omni 和 Box023/SPIDER 分别卡在 L4。

### 7.2 `raw contact` 应保留，但不能是唯一 RL-safe 口径

E162/E163 的 raw in-mask hard gate仍然有价值，尤其能抓住 E161 `box023_person2` 的 raw contact regression。但它有两个盲区：

- 小正间隙 near-contact 会被判非 raw contact，例如 `box004_082_p1`。
- MuJoCo raw contact 不等于 SUGAR filtered force contact，例如 Box023 的 net-filter gap。

建议保留 raw gate 作为最低门槛，同时新增：

- `isaac_filtered_any_contact_frac`
- `isaac_filtered_both_contact_frac`
- `source_vs_isaac_any_iou`
- `source_vs_isaac_both_iou`
- `net_filtered_contact_gap`
- `hand_obj_dist_when_source_contact`

### 7.3 必须加入 failure-mode aware 指标

从 RL 训练日志和修复后的 `failed_windows.csv` 看，失败类型可以由终止项区分：

| failure mode | 代表 case | 上游应补的预测指标 |
|---|---|---|
| object position tracking failure | `box021_r160/omniretarget` | object velocity/acceleration, object path curvature, target displacement/time |
| object orientation tracking failure | `box023_r158/omniretarget` | object angular velocity/acceleration, orientation jump, grasp torque leverage |
| end-effector/body tracking failure | `box023_r158/spider_e163` | wrist/ankle/root tracking margin, EEF acceleration, body pose deviation |
| height/lift gate failure | `box004_r161/spider_e163` | object z mean/max/final relative to source, lift duration, carry height profile |

这些指标应进入 SPIDER eval 表，而不是等 RL 6000 iter 后才发现。

落地上，SPIDER/SUGAR handoff 后的评测表应强制保存两类失败证据：

- 训练末期平均终止统计，用于判断 policy 在训练分布里的主要失败模式。
- rollout eval 的 per-window `failed_windows.csv`，用于判断最终 checkpoint 在固定 64-attempt 评测中的直接失败原因。

## 8. Algorithm Optimization Directions

### 8.1 E163 后续不是继续单调增强 contact

`box021_r160/omniretarget` 已经说明“接触强但 RL 失败”存在。继续只增强 contact 或 raw-contact hard gate，会错过动态可执行性问题。

建议下一版 CEM/rerank 目标写成多目标：

```text
score = tracking
      + contact_quality
      + isaac_filtered_contact_proxy
      + lift_height_preservation
      - penetration
      - net_filter_gap_risk
      - dynamic_termination_risk
```

其中 `dynamic_termination_risk` 可以先用离线 reference 诊断近似，不必一开始就内嵌 Isaac。

### 8.2 对 `box004_r161/spider_e163`：补 height/lift preservation

现象：target/progress 成功，但 height 不达标。

建议：

- 在 CEM selection/rerank 中加入 object z profile 约束：`z_mean/z_max` 不低于 source 的一定比例。
- 对 carry/lift 类 case 增加 `lift_duration` 和 `min_lift_height`，避免把搬运退化成低位推/滑。
- 不要只用 final target error 选择 winner；`box004/spider_e163` final error 很好，但严格 carry 不成功。

### 8.3 对 `box023_r158/spider_e163`：补身体/末端可执行性

现象：`term_ee_body_pos≈0.862`，说明 RL 先被末端/身体 tracking hard termination 击穿。

建议：

- 在 SPIDER eval 中加入 `max_eef_accel`、`max_wrist_speed`、`eef_path_curvature`、`root/anchor error margin`。
- 在 CEM rerank 中惩罚高 EEF jerk 和 reference 中突然的 hand-side switch。
- 对 Box023 类 net-filter gap 高的 case，先修 Isaac filtered contact 对齐，再判断是否需要更强 bimanual reward。

### 8.4 对 `box021_r160/omniretarget`：补动态可执行性筛选

现象：接触探针强，但 `term_obj_pos` 和 `term_anchor_pos` 高。

建议：

- 对 OmniRetarget baseline 也跑同一套 dynamic feasibility probe，不要只把它当参考。
- 分析 object path velocity/acceleration 与 anchor path drift，判断是否是 reference 太激进或 policy exploration 不足。
- 若继续训练，优先做 curriculum 或 longer-horizon stabilization，而不是调 contact mask。

### 8.5 对 Box023 net-filter gap：修 reward-visible contact

Box023 两种 source 都失败，且 net-filter gap 大。这个问题不是 SPIDER raw contact 一侧能完全解决的。

建议：

- 用 Isaac kinematic replay 标准化检查 `filtered force to Obj`，并作为 handoff 前必跑项。
- 对 object asset/filter 配置做 preflight，确保 hand contact 进入 SUGAR `hands_contact` reward 的过滤目标。
- 如果 net contact 高但 filtered contact 低，优先修 asset/filter 或 sensor target，而不是继续调整 CEM surfaceBand。

## 9. Recommended Next Steps

优先级从高到低：

1. 将 SUGAR rollout failure logging 固化成必检项。当前代码已修复并验证；后续所有 0-completion eval 都必须保存 `failed_windows.csv`、失败 npz 和 `failed_trajectory_metrics.csv`，并把 reason counts 汇总进主报告。
2. 为 E163 三 case 和 clean8 统一生成 `SPIDER + Isaac/SUGAR contact compatibility` 表，至少包含 filtered any/both contact、IoU、net-filter gap。
3. 在 `core_metrics.py` 或新 eval runner 中加入 dynamic feasibility 指标：object linear/angular velocity、acceleration、jerk、object z profile、EEF/root tracking margin。
4. 做两个小 ablation，而不是直接扩量：
   - `box004_r161/spider_e163`: 加 height/lift rerank，验证能否从 target-only success 变成 strict success。
   - `box023_r158/spider_e163`: 加 EEF/body feasibility rerank 或 smoother hand trajectory，验证能否降低 `ee_body_pos` termination。
5. 将 downstream result 记录回 Core4D S6 evidence，但字段上明确区分 `completion`, `sugar_target_success`, `progress_success`, `height_success`, `holosoma_like_success`，不要把其中任一项单独写成“RL 成功”。

## 10. Final Interpretation

E163 的上游方法仍然是有效的：它修复了 E161 在 `box023_person2` 上的 raw contact regression，并且在 `box021_r160/spider_e163` 上转化成了严格 RL 成功。但 RL 结果证明，当前 SPIDER 评测还缺三类 downstream-critical 信号：

1. 接触是否能被 SUGAR/Isaac filtered reward 看见。
2. 轨迹动态是否能避免 object/anchor/EEF/body hard termination。
3. 目标到达是否保持了任务所需的 lift/height 语义。

因此，E163 不应被简单降级为“无效”，也不应被升级为默认 RL-safe 方法。更准确的结论是：E163 是一个强上游候选，但要进入下游默认路线，必须增加 RL-aware evaluation 和 case-specific rerank，尤其针对 Box004 height、Box023 filtered contact/body feasibility、Box021 Omni dynamic feasibility 这三类 failure mode。

## Reproducibility Notes

- E163 SPIDER 三 case eval root: `workspace/core4d/results/E163/narrow_surface_band/eval/full/`
- E163 RL export root: `workspace/core4d/results/E163/narrow_surface_band_rl_export/`
- SUGAR RL main output root: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e163_refiner_rl/`
- Failed-fields rerun tag: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e163_refiner_rl/*/*/eval_failed_fields_rerun_20260617_170731/`
- Holosoma-like summary: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/outputs/core4d/e163_refiner_rl/holosoma_like_summary.csv`
- Rollout evaluator: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/evaluate_refiner_rollout.py`
- Holosoma-like evaluator: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/scripts/sugar_rl/evaluate_refiner_holosoma_like.py`
- Rollout saving logic: `/home/ubuntu/Workspace/Loco-Manipulation/SUGAR/source/sugar_rl/sugar_rl/tasks/locomanip/mdp/commands.py`
