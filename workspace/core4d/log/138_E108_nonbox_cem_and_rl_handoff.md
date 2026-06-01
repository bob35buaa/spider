# E108 Phase 4-6 非 box bucket004 到 CEM/RL handoff 结果

## 目标

承接 `workspace/core4d/plan/117_E108_nonbox_to_rl_data_pipeline_plan.md`，验证非 box case 在显式 template review 后能否进入 Stage2b、target gate、visual QC、full CEM，并形成可追踪的 RL handoff 候选。

## 数据与 template

首批只 release `bucket004_person1`，不自动 release 其它 bucket source template。

| 项目 | 结果 |
|---|---|
| source scene | `example_datasets/processed/core4d/unitree_g1/humanoid_object/bucket004_person1/scene.xml` |
| collision policy | `bucket_wall_proxy_aabb`：底面 + 四侧壁 proxy |
| template review | high subagent `APPROVE_CLEAN` |
| template 状态 | `clean_reviewed` |
| 质量/惯量策略 | `assumed_uniform_5kg_v3_no_real_mass_source` |

审查要点：MuJoCo load OK，`nq=43/nv=41/nu=29`；未发现 `mass=29.632` robot inertial 污染；proxy 仅作为 review-approved nonbox template，不等价于真实 bucket 精确物理。

## Stage2b 与 visual QC

`bucket004_person1` 的 4 条 5cm pass rows 均完成 Stage2b execute 与 target gate：

| case | Stage2b | target gate | visual QC | 结论 |
|---|---|---|---|---|
| `bucket004_20231003_1_012_p1` | pass | pass | pass | 进入 CEM |
| `bucket004_20231002_022_p1` | pass | pass | pass | 进入 CEM |
| `bucket004_20231002_021_p1` | pass | pass | pass | 进入 CEM |
| `bucket004_20231003_1_013_p1` | pass | pass | reject | 不进入 CEM |

被拒绝的 `013_p1` 是真实 visual QC 问题：初始 bucket 离地且离机器人较远，后续物体飞起/跳变；object z 最高约 `1.18m`，frame 7 到 8 位置跳变约 `2.62m`。

## full CEM

本地 1 卡 + 远程 2 卡并行完成 3 条 full CEM。

| variant | case | T | obj mean | obj max | contact | lower-body | 原始 strict CEM |
|---|---|---:|---:|---:|---:|---|---|
| `E108B01_bucket004_20231003_1_012_p1_ref_fk_nonbox` | `bucket004_20231003_1_012_p1` | 125 | 0.013m | 0.038m | 44.0% | pass | fail |
| `E108B02_bucket004_20231002_022_p1_ref_fk_nonbox` | `bucket004_20231002_022_p1` | 184 | 0.008m | 0.024m | 46.2% | pass | fail |
| `E108B03_bucket004_20231002_021_p1_ref_fk_nonbox` | `bucket004_20231002_021_p1` | 145 | 0.008m | 0.029m | 55.9% | fail | fail |

原始 strict fail 主要来自 `lie_on_box_frac`，这是 box-era 顶面规则。high subagent 复核 CEM 视频后给出 bucket-aware 结论：

| case | visual CEM decision | 说明 |
|---|---|---|
| `012_p1` | pass | `lie_on_box` 属于 bucket 上沿/桶壁 proximity false positive；lower-body 干涉低；适合作为 RL smoke 候选 |
| `022_p1` | pass | `lie_on_box` 属于 bucket 规则误报；lower-body 干涉为 0；适合作为 RL smoke 候选 |
| `021_p1` | reject | 视觉存在 lower-body/bucket interference，`leg_box_interference_frac=31.0%`，不适合作为 RL smoke 候选 |

因此 E108 记录两套口径：

- `E108_cem_eval_summary.md` 保留原始 box-era strict 数值，不改写；
- S6 downstream evidence 对 `012_p1`、`022_p1` 写入 `bucket_aware_visual_cem_pass`，并在 notes 中明确 strict box `lie_on_box` 失败是 bucket false positive override。

## S6 状态管理

初始 CEM 证据目录：

```text
/tmp/core4d_dcv3_E108_nonbox/downstream_evidence_bucket004_person1/
```

补充 RL smoke 后的证据目录：

```text
/tmp/core4d_dcv3_E108_nonbox/downstream_evidence_bucket004_person1_rl_smoke/
```

registry 目录：

```text
/tmp/core4d_dcv3_E108_nonbox/registry_bucket004_person1_final/
```

最终 registry 为 8 行：4 个 `omnirt_v1/ref_fk` rows、4 个 shared raw rows。S6 只记录 CEM/RL 下游证据，不反向改写 raw contact、template、Stage2b、target gate 或 visual QC。

| case | variant | current decision | CEM | RL | downstream decision | failure mode |
|---|---|---|---|---|---|---|
| `bucket004_20231003_1_012_p1` | `omnirt_v1/ref_fk` | `VISUAL_QC_PASS` | pass | pass | `DOWNSTREAM_RL_PASS` | `bucket_aware_visual_cem_pass` |
| `bucket004_20231002_022_p1` | `omnirt_v1/ref_fk` | `VISUAL_QC_PASS` | pass | not_run | `DOWNSTREAM_CEM_PASS` | `bucket_aware_visual_cem_pass` |
| `bucket004_20231002_021_p1` | `omnirt_v1/ref_fk` | `VISUAL_QC_PASS` | fail | not_run | `DOWNSTREAM_CEM_FAIL` | `lowerbody_bucket_interference` |
| `bucket004_20231003_1_013_p1` | `omnirt_v1/ref_fk` | `REJECT_VISUAL_QC` | not_run | not_run | empty | visual QC reject |

## RL 入口判断

Holosoma 侧新增了 bucket004 专用 RL export、reward/config 与固定训练入口，不复用 bucket005 的尺寸参数，也不伪造 partner motion。

新增代码：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/scripts/export_e108_bucket004_rl.py
/home/ubuntu/Workspace/holosoma/workspace/v3/scripts/train/train_core4d_e108_bucket004_nonbox.sh
```

Holosoma 配置：

```text
exp:g1-29dof-wbt-w-object-e108-bucket004-handbox-v4-3
```

关键约束：

- `MotionConfig.ignore_partner=True`，`partner_urdf_path=None`；
- motion export 从 E108 CEM 的 SPIDER `scene_act`/trajectory 转成 Holosoma `_mj_w_obj.npz`；
- `bucket004` handbox half-extents 使用 mesh AABB `0.161566625 / 0.231058755 / 0.15230013m`；
- smoke 仅证明 RL 入口可启动，不等价于完整训练收敛。

已导出两条 CEM-pass motion：

| case | export |
|---|---|
| `bucket004_20231003_1_012_p1` | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/results/e108_bucket004_rl/exports/E108B01_bucket004_20231003_1_012_p1_mj_w_obj.npz` |
| `bucket004_20231002_022_p1` | `/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction/results/e108_bucket004_rl/exports/E108B02_bucket004_20231002_022_p1_mj_w_obj.npz` |

已对 `bucket004_20231003_1_012_p1` 启动 RL smoke：

| 项目 | 结果 |
|---|---|
| run id | `E108B01-smoke` |
| 规模 | `2` iterations, `64` envs |
| 状态 | pass |
| total timesteps | `3072` |
| final mean reward | `1.87` |
| checkpoint | `/home/ubuntu/Workspace/holosoma/logs/core4d_e108_bucket004_handbox_v4_3_smoke/20260601_204343-e108b01_bucket004_handbox_v4_3_smoke-locomotion/model_00001.pt` |
| train log | `/home/ubuntu/Workspace/holosoma/logs/core4d_e108_bucket004_handbox_v4_3_smoke/e108b01_bucket004_handbox_v4_3_smoke_train.log` |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 自动筛出非 box 候选 | 通过：40 个 bucket smoke 中 5cm `34 pass / 1 review / 5 fail` |
| C2 非 box proxy template 只生成 review | 通过：9 个 bucket proxy 均 load/render pass，但默认 `manual_review_required` |
| C3 review override 后进入 Stage2b/target gate/S5 | 通过：`bucket004_person1` 4 条进入 Stage2b，3 条 `HANDOFF_READY` |
| C4 至少 1 个非 box case 形成 RL-ready handoff | 部分通过：2 条 bucket-aware CEM pass；原始 box strict 因 bucket `lie_on_box` false positive 失败 |
| C5 RL smoke 启动 | 通过：`bucket004_20231003_1_012_p1` 完成 Holosoma no-partner smoke，写回 S6 `DOWNSTREAM_RL_PASS` |

## 结论

E108 证明 v3 数据管线已经可以从原始 CORE4D 自动挖掘非 box bucket 候选，经 template review 后进入 Stage2b、target gate、visual QC、full CEM，并把下游证据写回 S6 registry。补充 RL smoke 后，至少 1 条非 box case 已经进入 Holosoma RL 阶段并保存 checkpoint。

当前可继续做完整 RL 训练的候选是：

```text
bucket004_20231003_1_012_p1
bucket004_20231002_022_p1
```

`bucket004_20231002_021_p1` 因 lower-body/bucket interference 拒绝；`bucket004_20231003_1_013_p1` 因 visual QC reject 不进入 CEM/RL。
