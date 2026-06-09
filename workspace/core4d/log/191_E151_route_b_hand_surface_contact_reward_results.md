# E151 路线 B：手表面接触奖励实验结果

日期：2026-06-10 CST

计划：`workspace/core4d/plan/159_E151_route_b_hand_surface_contact_reward_plan.md`

## 范围

E151 测试 E150 接触锚点诊断后的路线 B：保留 rubber-hand collision 场景，但让 reward 轴更直接地读真实手/物体几何。

Benchmark case：

- `box021_029_p2`
- `box004_083_p2`
- `box023_person2`

实验矩阵：

| 方法 | target | reward | 运行状态 |
|---|---|---|---|
| `baseline` | ref_fk/off05 | 现有 E147/E148 rubber 设置 | 复用，不重跑 |
| `b2_sup` | external adaptive support surface | `contact_hdmi` | full CEM |
| `b2_tip` | external fingertip-aware target | `contact_hdmi` | full CEM |
| `b1_mesh` | ref_fk/off05 | `contact_hdmi + hand_support(mesh)` | full CEM |
| `B1+B2` | 最优 external target | 加 `hand_support(mesh)` | 未跑；触发条件未满足 |

## 实现

核心代码改动在 `spider/simulators/mjwp.py` 的 `_geom_box_sdf_min`：

- primitive geom 保留旧 SDF 路径。
- mesh geom 新增固定、确定性的 mesh 顶点采样路径，与 E147 evaluator 的几何口径对齐。
- `MESH_SDF_SAMPLE_COUNT=800`，与 `eval_E147` 的 mesh sample count 保持一致。

新增 E151 固定入口：

- `workspace/core4d/scripts/E151/build_route_b_manifest.py`
- `workspace/core4d/scripts/E151/check_mesh_sdf_and_targets.py`
- `workspace/core4d/scripts/train/train_E151_route_b_hand_surface_contact.sh`
- `workspace/core4d/scripts/run_E151_local.sh`
- `workspace/core4d/scripts/run_E151_remote.sh`
- `workspace/core4d/scripts/pull_E151_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E151_route_b_hand_surface_contact.py`
- `workspace/core4d/scripts/eval/eval_E151_route_b_hand_surface_contact.sh`
- `examples/config/override/core4d_E151_*.yaml`

Smoke 阶段发现一个 evaluator 路径 bug：manifest 中保存的是 full-stage 路径，所以 smoke eval 一开始会去找缺失的 `cem/full` 路径。已修复 evaluator：按传入的 `stage` 重写 E151 run rows 的 `result_npz/outdir_npz/video`，同时保留 E147/E148 baseline 复用路径不变。

## 前置检查

Manifest：

- 总行数：12
- baseline 复用：3
- 新 full CEM：9
- split：local-gpu0 3，remote-gpu0 3，remote-gpu1 3

Preflight：

| 检查 | 结果 |
|---|---:|
| mesh SDF parity rows | 9 |
| mesh max abs diff vs E147 evaluator | 0.0m |
| primitive regression rows | 2 |
| primitive max abs diff | 0.0m |
| target validation rows | 6 |

Box021 clean target gate：

- B2-sup old-vs-clean max diff：0.3280478m
- B2-tip old-vs-clean max diff：0.06922875m
- 两者均超过 5mm 阈值，因此 E151 选择 clean 重生成的 box021 target。

## 执行

Smoke：

- 本地 tmux：`E151_local_smoke_230945`
- 远端 tmux：`E151_smoke_230945`
- pull 后 artifacts：9/9 root NPZ、9/9 MP4、9/9 outdir trajectory
- smoke eval：`method_rows=12`、`delta_rows=9`、`missing=0`、`visual_sheets=9`
- B1 reward plumbing 已确认非零；smoke 是低迭代检查，不作为最终成功/失败结论。

Full：

- 本地 tmux：`E151_local_full_232329`
- 远端 tmux：`E151_full_232329`
- remote GPU0：`box004_083_p2` B2-sup、B2-tip、B1-mesh
- remote GPU1：`box023_person2` B2-sup、B2-tip、B1-mesh
- local GPU0：`box021_029_p2` B2-sup、B2-tip、B1-mesh

没有 kill 任何无关进程。已有 GPU 负载保持不动，E151 本地 split 叠加跑在本地 GPU 上。SSH 监控过程中偶发 `kex_exchange_identification` 断开，但 tmux/log/artifact 检查均显示运行健康。

Full artifact 计数：

| artifact | count |
|---|---:|
| root NPZ | 9/9 |
| MP4 | 9/9 |
| outdir `trajectory_mjwp_act.npz` | 9/9 |
| run logs | 9/9 |
| visual sheets | 9 |
| extracted video frames | 6 |

Full eval：

```text
E151 eval: method_rows=12 delta_rows=9 missing=0 visual_sheets=9
```

## 量化结果

### 方法均值

| 方法 | cases | 5cm | 10cm | hand pen | physics contact | leg pen | obj err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 3 | 0.6150 | 0.6484 | 0.3397 | 0.3491 | 0.0450 | 0.0083 |
| `b2_sup` | 3 | 0.7113 | 0.7273 | 0.5168 | 0.5326 | 0.1047 | 0.0088 |
| `b2_tip` | 3 | 0.7137 | 0.7326 | 0.6091 | 0.6145 | 0.0576 | 0.0093 |
| `b1_mesh` | 3 | 0.6939 | 0.7184 | 0.5471 | 0.5649 | 0.0375 | 0.0089 |

### 相对 baseline 的均值变化

| 方法 | success | 5cm delta | 10cm delta | hand pen delta | physics delta | leg pen delta | obj err delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b2_sup` | 0/3 | +0.0963 | +0.0786 | +0.1771 | +0.1835 | +0.0597 | +0.0005 |
| `b2_tip` | 0/3 | +0.0987 | +0.0842 | +0.2694 | +0.2654 | +0.0126 | +0.0010 |
| `b1_mesh` | 0/3 | +0.0789 | +0.0700 | +0.2074 | +0.2158 | -0.0075 | +0.0006 |

事前定义的主成功判据：

> mean `hand_geom_near_5cm` delta >= +0.05，且 `hand_geom_penetration` 不高于 baseline，同时不摔倒，object/leg guard 可接受。

三种方法都提高了 5cm 接触，但三种方法都同步提高了手-物体穿透。因此 primary success 均为 0/3。

### 逐 case 结果

| case | 方法 | 5cm delta | hand pen delta | fall | 备注 |
|---|---|---:|---:|---|---|
| `box004_083_p2` | `b2_sup` | +0.1619 | +0.2190 | true | 接触增加，但姿态不稳 |
| `box004_083_p2` | `b2_tip` | +0.1619 | +0.3333 | false | 最强 contact/penetration tradeoff |
| `box004_083_p2` | `b1_mesh` | +0.0952 | +0.2571 | false | 身体稳定，但手仍穿入 |
| `box021_029_p2` | `b2_sup` | +0.0533 | +0.2533 | true | 近场接触增加，但 fall 且 leg/object 干扰增加 |
| `box021_029_p2` | `b2_tip` | +0.0533 | +0.3867 | false | 稳定，但手穿透显著加深 |
| `box021_029_p2` | `b1_mesh` | +0.0533 | +0.2400 | false | 稳定，但 hand penetration 仍上升 |
| `box023_person2` | `b2_sup` | +0.0735 | +0.0588 | false | 小幅改善，但 penetration 上升 |
| `box023_person2` | `b2_tip` | +0.0809 | +0.0882 | false | 小幅改善，但 penetration 上升 |
| `box023_person2` | `b1_mesh` | +0.0882 | +0.1250 | false | 稳定，但 penetration 上升 |

## B1 机制检查

B1 hand-support 诊断在 full run 中非零：

| case | `hand_support_sdf_mean_mean` | `hand_support_score_mean_mean` | `hand_support_rew_mean_mean` |
|---|---:|---:|---:|
| `box021_029_p2` | 0.0075 | 0.8452 | 2.4889 |
| `box004_083_p2` | 0.0930 | 0.6738 | 1.8942 |
| `box023_person2` | 0.1188 | 0.5428 | 1.6119 |

结论：B1 reward plumbing 确实被激活并参与优化，但当前设置下它把 rubber hand 拉向/拉进物体，而不是产生无穿透的表面贴合改善。

## 视觉观察

生成的并排 visual sheets：

- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box004_083_p2_f25_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box004_083_p2_f55_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box004_083_p2_f85_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box021_029_p2_f25_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box021_029_p2_f55_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box021_029_p2_f85_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box023_person2_f25_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box023_person2_f55_baseline_b2_b1.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/box023_person2_f85_baseline_b2_b1.jpg`

额外抽帧：

- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/video_frames/E151_box021_029_p2_b1_mesh_full_f75.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/video_frames/E151_box004_083_p2_b1_mesh_full_f105.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/video_frames/E151_box023_person2_b1_mesh_full_f136.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/video_frames/E151_box021_029_p2_b2_tip_full_f75.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/video_frames/E151_box004_083_p2_b2_tip_full_f105.jpg`
- `workspace/core4d/results/E151/route_b_hand_surface_contact/visual_inspection/video_frames/E151_box023_person2_b2_tip_full_f136.jpg`

观察：

- `box021_029_p2`：B1 保持机器人站立，但 sim 手/前臂主要从 box 近侧面/边缘压入。B2-tip 稳定，但穿入更深。B2-sup 会 fall。
- `box004_083_p2`：B1 视觉上把 rubber hand 压进 box 侧面。B2-tip 的手-物体重叠最强，与最大 penetration delta 一致。
- `box023_person2`：B1 姿态最干净，但掌面仍明显沉入物体侧面。

视觉结果与指标一致：E151 提高了 near-contact / physics-contact fraction，但提升主要来自穿透，不是 clean surface contact。

## B1+B2 条件分支决策

B1+B2 事前定义为条件触发：

> 只有 B2 和 B1 单独都 pass 时，才跑 B1+B2。

本轮条件未满足：

- B2-sup success：0/3
- B2-tip success：0/3
- B1-mesh success：0/3

因此没有启动 B1+B2。

## Claims 验证

| claim | 结果 | 证据 |
|---|---|---|
| Mesh SDF reward path 与 evaluator 对齐 | PASS | preflight mesh parity `max_abs_diff=0.0m`；primitive regression `max_abs_diff=0.0m`。 |
| B2 external target pipeline 可用 | PASS | log 中出现 `E085 external contact target` 与 `E040 dynamic target source=external`；B2 full artifacts 6/6。 |
| B1 hand-support reward 被激活 | PASS | 三个 B1 full run 的 `hand_support_rew_mean_mean` 均非零。 |
| 本地 1 卡 + 远端 2 卡完成 full matrix | PASS | full root NPZ/MP4/outdir/logs = 9/9/9/9，远端结果已 pull。 |
| 主成功判据 | FAIL | 三方法 mean 5cm 均提高超过 5pp，但 hand penetration 全部上升；variant-case success 0/9。 |
| B1+B2 trigger | NOT RUN | 因 B1/B2 单独均未 pass，未触发。 |

## 结论

E151 没有验证当前 B2/B1 形式的路线 B。

实验确认 external target projection 和 mesh-aware hand-support reward 都能把优化器拉向更多手-物体接触。但是在没有硬反穿透项或真实手-物体物理碰撞的情况下，额外接触主要通过 rubber hand 压入物体实现。B1 是三者中相对最有保留价值的方向，因为它没有增加 mean leg penetration，并且三个 full run 都稳定；但它仍然不满足主判据，因为 mean hand penetration 增加 +0.2074。

建议下一步：

1. 在尝试 B1+B2 前，先加入显式 mesh hand-object penetration penalty 或非对称 SDF 目标。
2. 保留 B1 mesh SDF plumbing，因为该路径已验证可用且 reward 信号有效。
3. 不把 B2 单独作为 release 路径；它复现了 E116 风格问题，即通过非 clean penetration / posture tradeoff 换取更高 contact。

## 验证

- `py_compile`：E151 evaluator 与 preflight scripts 通过。
- shell syntax：E151 run/pull/eval scripts 通过。
- smoke：`method_rows=12`、`delta_rows=9`、`missing=0`、`visual_sheets=9`。
- full artifacts：root NPZ/MP4/outdir/logs = 9/9/9/9。
- full eval：`method_rows=12`、`delta_rows=9`、`missing=0`、`visual_sheets=9`。
- 视觉检查：已检查 side-by-side sheets 与抽帧；定性结论与 penetration 指标一致。
