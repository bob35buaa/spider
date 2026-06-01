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

S6 证据目录：

```text
/tmp/core4d_dcv3_E108_nonbox/downstream_evidence_bucket004_person1/
```

registry 目录：

```text
/tmp/core4d_dcv3_E108_nonbox/registry_bucket004_person1_final/
```

最终 registry 为 8 行：3 个 `omnirt_v1/ref_fk` CEM rows、3 个 shared raw rows、1 个 visual reject row、1 个 shared reject-case raw row。

| case | variant | current decision | CEM | downstream decision | failure mode |
|---|---|---|---|---|---|
| `bucket004_20231003_1_012_p1` | `omnirt_v1/ref_fk` | `VISUAL_QC_PASS` | pass | `DOWNSTREAM_CEM_PASS` | `bucket_aware_visual_cem_pass` |
| `bucket004_20231002_022_p1` | `omnirt_v1/ref_fk` | `VISUAL_QC_PASS` | pass | `DOWNSTREAM_CEM_PASS` | `bucket_aware_visual_cem_pass` |
| `bucket004_20231002_021_p1` | `omnirt_v1/ref_fk` | `VISUAL_QC_PASS` | fail | `DOWNSTREAM_CEM_FAIL` | `lowerbody_bucket_interference` |
| `bucket004_20231003_1_013_p1` | `omnirt_v1/ref_fk` | `REJECT_VISUAL_QC` | not_run | empty | visual QC reject |

## RL 入口判断

Holosoma 侧可复用的训练入口是 v3 handbox PPO 路线，例如：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/scripts/train/train_core4d_r103_bucket005_from_scratch.sh
/home/ubuntu/Workspace/holosoma/workspace/v3/scripts/train/train_core4d_r119_r121_box021_omnirt_threecase.sh
```

但 E108 不能直接把 bucket004 CEM 输出喂给 RL，原因是：

1. Holosoma `MotionLoader` 需要 `_mj_w_obj_w_partner.npz` 格式，包含 `joint_pos/body_pos_w/object_pos_w/body_names/joint_names` 等字段；
2. E108 CEM 输出是 SPIDER `trajectory_mjwp_act.npz`，`scene_act.xml` 中 object 为 `pos_x/pos_y/pos_z/rot_z/rot_y/rot_x` actuated joints，不是 Holosoma 直接读取的 motion 格式；
3. Holosoma 已有 `bucket005` reward/config，但没有严格匹配 `bucket004` half-extents 的 handbox reward/config；
4. 用 `bucket005` 配置硬套 `bucket004` 会污染“能进入 RL”的结论。

所以 E108 的阶段性结论是：数据构建到 S6 已产生 2 条 bucket-aware CEM-pass 的 RL smoke 候选；真正启动 Holosoma RL 前，需要新增 bucket004 RL export + bucket004 handbox config。该工作应作为下一步独立实验，不能在 E108 中伪装完成。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 自动筛出非 box 候选 | 通过：40 个 bucket smoke 中 5cm `34 pass / 1 review / 5 fail` |
| C2 非 box proxy template 只生成 review | 通过：9 个 bucket proxy 均 load/render pass，但默认 `manual_review_required` |
| C3 review override 后进入 Stage2b/target gate/S5 | 通过：`bucket004_person1` 4 条进入 Stage2b，3 条 `HANDOFF_READY` |
| C4 至少 1 个非 box case 形成 RL-ready handoff | 部分通过：2 条 bucket-aware CEM pass；原始 box strict 因 bucket `lie_on_box` false positive 失败 |
| C5 RL smoke 启动 | 未执行：缺 bucket004 Holosoma export/config，下一步单独实现 |

## 结论

E108 证明 v3 数据管线已经可以从原始 CORE4D 自动挖掘非 box bucket 候选，经 template review 后进入 Stage2b、target gate、visual QC、full CEM，并把下游证据写回 S6 registry。

当前可进入下一步 RL export/config 的候选是：

```text
bucket004_20231003_1_012_p1
bucket004_20231002_022_p1
```

`bucket004_20231002_021_p1` 因 lower-body/bucket interference 拒绝；`bucket004_20231003_1_013_p1` 因 visual QC reject 不进入 CEM/RL。
