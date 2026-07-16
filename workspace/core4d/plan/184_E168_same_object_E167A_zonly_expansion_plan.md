# E168 实验计划：同物体多轨迹 E167A z-only 数据扩展

日期：2026-07-16

实验方向：`core4d`

当前分支：`experiment/E161-surface-release-ablation`

状态：planning-only。本计划只定义范围、契约、脚本和执行门；当前不实现代码，不运行 S1-S6，不启动 CEM/RL。

---

## 1. Context

### 1.1 下游需求

下游 RL 需要同一个物体的多条有效轨迹。当前上游 SPIDER 数据集中，`box004`、`box021` 和 `bucket004` 已有少量 seed，但同物体的其他 CORE4D raw case 尚未系统召回、按统一版本重定向和导出。

用户提供的 seed 为：

| 用户短名 | canonical raw case |
|---|---|
| `box004_083_p1` | `box004_20231003_2_083_p1` |
| `box004_083_p2` | `box004_20231003_2_083_p2` |
| `box021_029_p2` | `box021_20231018_029_p2` |
| `box021_035_p1` | `box021_20231011_035_p1` |
| `box021_035_p2` | `box021_20231011_035_p2` |
| `bucket004_022_p1` | `bucket004_20231002_022_p1` |

这些 seed 的作用是确定精确物体身份：

```text
object_key in {box004, box021, bucket004}
```

seed 不是自动质量正例，也不计入 E168 的“新召回”数量。特别是 E167 中 `box021_035_p1/E167A` 虽然 tracked 且未 fall，但因 contact/penetration regression 没通过 SPIDER release gate。

### 1.2 动作范围

本轮只处理 CORE4D 官方 action label：

```text
move1_obs0, move2_obs0
move1_obs1, move2_obs1
move1_obs3, move2_obs3
```

官方语义：

| 维度 | 含义 |
|---|---|
| `move1` | 两人协作搬运，只有一人知道目标位置 |
| `move2` | 两人协作搬运，两人都知道目标位置 |
| `obs0` | 无障碍 |
| `obs1` | 稀疏障碍 |
| `obs3` | 密集障碍 |

明确不处理：

```text
pass*, raise*, rot*, strike*
```

本轮不为这些非 move 动作建专用 contact/motion gate，也不把它们放入 CEM 或 RL handoff。

### 1.3 只读 inventory 盘点

基于当前挂载的 `CORE4D_Real`，三个目标物体共有 102 条 person-case：

| object_key | raw person-case | move1/2 | 排除 seed 后 | 非 seed obs0 | 非 seed obs1/3 |
|---|---:|---:|---:|---:|---:|
| `box004` | 20 | 14 | 12 | 2 | 10 |
| `box021` | 50 | 36 | 33 | 13 | 20 |
| `bucket004` | 32 | 22 | 21 | 7 | 14 |
| **总计** | **102** | **72** | **66** | **22** | **44** |

因此 E168 的 candidate upper bound 是 66，不等于最终 CEM 数量。每条 case 必须依次通过 raw contact、template、Stage2b、target gate、visual QC 后才允许进入 E167A CEM。

### 1.4 历史可复用证据

E168 不应把所有 66 条都当作完全从零开始：

| 来源 | 可复用价值 |
|---|---|
| E167 | `box004_082_p1` 已有 E167A full CEM 产物，可在 provenance/config/artifact audit 后导入 E168 bank，不默认重跑 |
| E107 | Box021 已有 13 条 clean target-gate evidence，其中排除 3 条 seed 后约 10 条可作为高优先输入 |
| E145 | Bucket004 已有 6 条 `raw_mask_ref_fk` handoff-ready，排除 `bucket004_022_p1` seed 后约 5 条可作为高优先输入 |
| E143/E148 | 部分 case 已有 3cm mask、rubber hand sidecar 或历史 CEM，可用作审计/基线；不能隐式扫描 legacy 目录 |

历史状态只作为显式 import evidence。E168 必须写出自己的 imported snapshot、registry 和 source reference，不回写或覆盖历史实验。

### 1.5 E167A 版本事实

E167 已有的实际结果：

| 指标 | E167A |
|---|---:|
| cases | 7 |
| tracked | 7/7 |
| fall | 0/7 |
| SPIDER release gate | 6/7 |
| Holosoma z gate | 7/7 |
| 失败 case | `box021_035_p1` contact/penetration regression |

E168 是数据扩展实验，不重新声称“E167A 优于 baseline”。核心问题是：能否将同一套 E167A method contract 稳定扩展到更多同物体 move case，并产出可审核的 RL handoff。

---

## 2. 固定版本契约

### 2.1 Variant 轴

E168 必须把四个正交概念分开记录：

| 字段 | primary | rescue/备注 |
|---|---|---|
| `retarget_variant_id` | `omnirt_v1` | v1 infeasible 时用 `omnirt_v2` |
| `target_variant_id` | `ref_fk` | 不使用 `adaptive/fingertip_aware` |
| `hand_collision_variant_id` | `rubber_hull` | 必须使用 sidecar scene |
| `spider_method_id` | `E167A_zOnlyBody` | 不启用 B1/B2 |
| `source_exp_id` | `E168` | 历史复用 row 仍保留原始 source ref |
| `contact_mask_label` | `3cm` | `5cm` 只作 review/diagnostic |

禁止把 `E167A_zOnlyBody` 写进 `target_variant_id`。它是 SPIDER/CEM 方法，不是 target route。

### 2.2 OmniRetarget v2 rescue，不启用 replace

E168 采用以下 retarget 版本口径：

| id | 含义 | 必须记录的关键参数 |
|---|---|---|
| `omnirt_v1` | 原版 OmniRetarget / baseline route | Phase4 flags 全关，`replace_wrist_with_fingertip=false` |
| `omnirt_v2` | 改进版 OmniRetarget | Phase4 flags 开启，`replace_wrist_with_fingertip=false` |
| `omnirt_v2_replace` | 改进版 OmniRetarget + fingertip replacement | Phase4 flags 开启，`replace_wrist_with_fingertip=true`；本版 E168 不使用 |

其中 Phase4 flags 指 Holosoma `workspace/v1/README.md` 中的改进项：

```text
--retargeter.enable-constraint-relaxation
--retargeter.enable-foot-z-constraint
--retargeter.foot-slide-penalty-weight 1.0
--retargeter.enable-contact-preservation
--retargeter.object-penetration-tolerance-scale 0.8
```

`replace` 只是 input rewrite 后缀：从 SMPL-X 左右手 5 个 fingertip joint 计算均值，并替换 `global_joint_positions[:,20]` / `[:,21]`。它不能单独代表 v2，也不进入本版 E168 的生产或 rescue。

当前 v3 文档/registry 中的旧 id `omnirt_v1_fingertip_replacement` 只表达 fingertip replacement，不足以表达 `omnirt_v2`。implementation 阶段必须新增或显式注册 `omnirt_v2`，让 Stage2b adapter 传 Phase4 flags，同时保持 `replace_wrist_with_fingertip=false`，不得传 `REPLACE_WRIST_WITH_FINGERTIP=1`。

E168 rescue 规则：

```text
omnirt_v1 pass
  -> 使用 omnirt_v1 产物

omnirt_v1 omniretarget_infeasible
  -> 进入 omnirt_v2 rescue

rescue pass
  -> 继续 ref_fk target gate 和 E167A_zOnlyBody

rescue infeasible
  -> 标记 REJECT_DUAL_OMNIRT_INFEASIBLE
```

不允许：

- 对 raw-contact fail、template fail、target-gate fail 使用 `omnirt_v2`“绕过”上游事实；
- 用 `omnirt_v2` 覆盖 v1 输出；
- 把 `omnirt_v2` 成功 row 标为 `omnirt_v1`；
- 把只开 replace、未开 Phase4 flags 的 row 标为 `omnirt_v2`；
- 在本版 E168 中启用 replace / fingertip replacement；
- 因 rescue 改用 `fingertip_aware` target route。

首批已知 v1 infeasible probe：

| case | 历史状态 |
|---|---|
| `box004_20231003_2_082_p2` | E095/E167 partner path: OmniRetarget CVXPY infeasible/missing |
| `box021_20231011_034_p2` | E107: OmniRetarget infeasible |
| `box021_20231018_028_p1` | E107: OmniRetarget infeasible |

在批量 rescue 前，使用一个已知 v1-success seed 做 `omnirt_v2` adapter canary。该 canary 只验证 pipeline，不计入新数据产量。

### 2.3 E167A effective config

E168 不依赖“每个新 case 都已有 E163 case-specific override”。应把 E167A 拆为：

```text
generic E167A method profile
+ case-specific task/scene/mask/person layer
```

必须与 E167A reference 对齐的关键字段：

```yaml
e167_body_z_enabled: true
e167_body_z_names:
  - left_ankle_roll_link
  - right_ankle_roll_link
  - left_wrist_yaw_link
  - right_wrist_yaw_link
e167_body_z_weight: 2.0
e167_body_z_threshold_m: 0.25

e167_ground_z_enabled: true
e167_ground_z_names:
  - left_ankle_roll_link
  - right_ankle_roll_link
e167_ground_z_weight: 2.0
e167_ground_contact_height_m: 0.05

foot_slip_enabled: false
foot_slip_weight: 0.0
foot_ground_enabled: false
foot_ground_weight: 0.0
local_frame_ankle_weight: 1.0

cem_smooth_enabled: false
cem_hand_gate_enabled: true
cem_hand_gate_min_sdf_m: -0.010
cem_hand_gate_max_violation_pct: 0.10
cem_hand_gate_hard_floor_m: -0.020

surface_band_rew_scale: 1.5
surface_band_penalty_scale: 0.0
surface_band_width_m: 0.003
surface_band_min_sdf_m: -0.001
surface_band_sigma: 0.0015
surface_band_score_mode: symmetric_abs
surface_band_gate_source: contact_mask

cem_posture_gate_enabled: true
cem_posture_gate_mean_z_err_m: 0.10
cem_posture_gate_terminal_z_err_m: 0.12
cem_posture_gate_max_z_drop_m: 0.18
cem_posture_gate_terminal_frac: 0.15
cem_posture_gate_min_valid_frac: 0.05
cem_posture_gate_fallback_lambda: 5.0
```

每条 resolved config 需要生成参数摘要和 SHA256。Axis audit 必须证明：

- 没有 B1 `cem_smooth_axis=z`；
- 没有 B2 postprocess；
- 没有 XY foot-slip reward；
- 没有 3D ankle extra weight；
- object tracking 仍保持 3D；
- body/ground executability penalty 只读取 z。

---

## 3. Claims

| Claim | 最低证据 |
|---|---|
| C0-recall-complete | recall manifest 恰有 102 条 raw row；move scope 72；seed 6；非 seed candidate 66；按 object/obs 计数与本计划表一致 |
| C1-scope-pure | production/rescue/CEM manifest 中 action 全部匹配 `^move[12]_obs[013]$`；`pass/raise/rot/strike` rows = 0 |
| C2-object-identity | 每条 row 的 `object_key` 精确属于 `{box004,box021,bucket004}`；同 object bank 的 mesh path 与 SHA256 一致 |
| C3-contact-authority | production row 必须 `raw_contact_3cm_status=pass`；mask 帧数与 raw/trim mapping 可证明，禁止静默 resize/interpolate；5cm-only 不自动晋级 |
| C4-variant-provenance | primary 与 rescue 输出分别绑定完整 retarget variant、solver/converter SHA、params JSON、converted/retargeted/trimmed path；不存在 variant 覆盖 |
| C5-v2-rescue-answer | 三条已知 infeasible case 和本轮新增 v1 infeasible rows 均有明确 `omnirt_v2` rescue 结果；至少能回答 Phase4 flags 在不启用 replace 时是否改变可行性。若 0 条救回，claim 结论为 fail 而非重复 v1 |
| C6-E167A-parity | 所有 CEM row 的 resolved method config 通过 E167A key audit；`spider_method_id=E167A_zOnlyBody`；无 B1/B2 串入 |
| C7-stage-gates | 进入 CEM 的 row 全部满足 template clean/clean_reviewed、Stage2b pass、target gate pass、visual QC pass |
| C8-motion-quality | release row 全部 tracked、无 fall、Holosoma body-z peak `<=0.25m`、terminal pelvis-z error `<=0.08m`，并通过 contact/penetration/lower-body gate |
| C9-obstacle-clarity | obs1/obs3 rows 保留 `source_obstacle_level` 和 `obstacle_context_not_reconstructed`；报告不将其声称为 obstacle-aware RL scene |
| C10-yield | 每个 object 至少得到 2 条非 seed `RL_EXPORT_READY` trajectory；不足时保持 gate，不用 5cm 或 failed row 填数量 |
| C11-reproducibility | E168 结果只写 `results/E168`；scene snapshot、manifest、registry、NPZ、MP4、config、eval、handoff 路径完整且可校验 |

---

## 4. Candidate 分层

### 4.1 Tier A：obs0 production-first

共 22 条非 seed case：

| object | case group |
|---|---|
| `box004` | `20231003_2/082 p1,p2` |
| `box021` | `20231011/034 p1,p2`; `20231018/028 p1,p2`; `029 p1`; `030 p1,p2`; `031 p1,p2`; `20231020/019 p1,p2`; `020 p1,p2` |
| `bucket004` | `20231002/021 p1,p2`; `022 p2`; `20231003_1/012 p1,p2`; `013 p1,p2` |

Tier A 优先原因：

- 与历史 E107/E145/E167 的无障碍 carry 分布最接近；
- 已有较多 target/template evidence；
- 最适合作为 generic E167A builder 的首批验证。

Tier A 不等于自动通过。已知 infeasible、raw-contact reject、visual reject 必须保留原始失败事实，并按本计划的限定 rescue 规则处理。

### 4.2 Tier B：obs1/obs3 review-to-production

共 44 条非 seed case：

| object | case group |
|---|---|
| `box004` | `20231002/048 p1,p2`; `20231003_2/084,085,086,087 p1,p2` |
| `box021` | `20231011/036,037,038,039 p1,p2`; `20231018/032,033,034,035 p1,p2`; `20231020/022,023 p1,p2` |
| `bucket004` | `20231002/017,018,020 p1,p2`; `20231003_1/014,015,016,017 p1,p2` |

当前 SPIDER scene 不重建 CORE4D 原始障碍几何。Tier B 的解释是：

> 使用带 obstacle-conditioned 人类动作的轨迹，在 robot + object 场景中验证其能否作为稳定重定向参考。

Tier B 必须额外检查：

- 绕行动作在空场景中是否仍自然；
- 脚步/抬腿是否产生 ground penetration 或 fall；
- 人体是否出现“为了跨越不存在障碍而异常抬腿”的明显视觉问题；
- 物体 path 是否连续，是否存在 obstacle support 才能成立的动作；
- target scene 不得声称包含原始障碍。

### 4.3 Excluded action accounting

其余 30 条 person-case 只进入 recall accounting：

```text
scope_status=excluded_non_move_action
execution_decision=OUT_OF_SCOPE_E168
```

它们不进入 raw-contact production manifest、OmniRetarget、SPIDER 或 CEM。

---

## 5. Pipeline 与 Gate

### Phase 0：环境、编号和 provenance preflight

目标：

- 固定 run root 为 `workspace/core4d/results/E168/`；
- 记录 spider/holosoma/solver git SHA 和 dirty 状态；
- 确认 `CORE4D_RAW_ROOT`、`SMPLX_MODEL_DIR`、本地 GPU、A6000 2 卡、A100 8 卡动态空闲 GPU 选择规则；
- 确认两台远程机 SSH alias 和 repo/raw/model 路径，禁止在脚本中猜主机名；
- 审计 E167 实际产物与 tracker 的 planning-only 记录差异；
- 对 E167A reference resolved config 生成 immutable profile snapshot。

Phase 0 hard stop：

- 任何 repo SHA/solver variant 无法解释；
- A100/A6000 远程路径或 A100 空闲 GPU 选择脚本未确认；
- E167A reference config 无法完整解析；
- 正式 run root 不是 `results/E168`。

### Phase 1：全量 recall 与 S1 raw contact

步骤：

1. 从 raw 重建全量 inventory。
2. 按大小写归一化的精确 `object_key` 召回三个物体。
3. 写出 102-row recall accounting。
4. 用 action regex 筛出 72 条 move。
5. 标记并排除 6 条 seed，得到 66 条 candidate。
6. 对 move candidate 同时计算 3cm/5cm raw-contact proxy。
7. 3cm pass 进入下一阶段；3cm review/5cm-only 留在 review，不自动 production。

3cm proxy 的含义：

- 每帧采样 object mesh surface；
- 任一 SMPL-X hand vertex 到 surface `<0.03m` 时标记该手 contact；
- current clean-carry pass 要求双手、左右平衡和 partner support 达到阈值；
- motion rotation/lift gate 只作为 move/carry 生产条件，不推广到本轮已排除的其他 action。

Phase 1 输出必须分别保存 3cm/5cm，不允许用 5cm 覆盖 3cm：

```text
s1_raw_contact/inventory/inventory.tsv
s1_raw_contact/recall/e168_same_object_recall.tsv
s1_raw_contact/raw_contact/raw_contact_candidates_3cm.tsv
s1_raw_contact/raw_contact/raw_contact_pass_3cm.tsv
s1_raw_contact/raw_contact/raw_contact_candidates_5cm.tsv
s1_raw_contact/raw_contact/raw_contact_pass_5cm.tsv
s1_raw_contact/raw_contact/per_sequence/*/raw_contact_proxy.npz
```

### Phase 2：S2 template audit

Box：

- `box004_person1/person2`、`box021_person1/person2` 走当前 clean template audit；
- 缺失时按 E103 clean base 流程补建；
- 必须检查 MuJoCo load、robot inertial、object mass/inertia、collision extents。

Bucket：

- `bucket004_person1/person2` 只能使用 `clean_reviewed` template；
- 可显式导入 E144/E145 review evidence，但必须写 imported snapshot 和 source ref；
- 若重新生成 proxy，必须做 mesh/collision overlay review；
- 不能仅因 MuJoCo load pass 自动 release 非 box template。

### Phase 3：S3 primary OmniRetarget 与 rescue

Primary：

```text
retarget_variant_id=omnirt_v1
target_variant_id=ref_fk
```

Primary 的每条 row 必须记录：

- converter/solver git SHA；
- converted NPZ；
- raw OmniRetarget output；
- trimmed NPZ；
- trim window/policy；
- contact mask mapping；
- Stage2b/SPIDER target task；
- verify summary；
- infeasible traceback/status。

Rescue：

- 只消费 `stage2b_status=omniretarget_infeasible` 的 v1 rows；
- 使用 `omnirt_v2/ref_fk`；
- 输出到独立目录；
- 成功后仍需重新经过完整 target gate/visual QC；
- v1/v2 两条 registry row 并存。

### Phase 4：S4 target gate 与 visual QC

机器 gate：

- target scene/scene_act 可 load；
- qpos/qvel/ctrl/contact 帧数一致；
- trimmed qpos 与 SPIDER input 一致；
- object pose 第一帧 patch 正确；
- robot inertial 无历史污染；
- target tracking、inside/penetration、lower-body interference 不越界；
- contact mask raw-to-trim mapping 有明确切片证据。

Visual QC：

- 每条 target 生成 replay MP4 和 keyframe sheet；
- obs1/obs3 单独标记并重点检查脚步、抬腿和空场景合理性；
- bucket 做 object orientation、hand/bucket wall proximity 和 lower-body interference 检查；
- 审查结论必须是具体观察，不允许空白或“待补充”。

只有：

```text
target_gate_status=pass
visual_qc_status=pass
```

才进入 E167A CEM manifest。

### Phase 5：S5 scene adapter 与 E167A manifest

对每条 ready row：

1. 创建/复用 source target scene。
2. 使用 `patch_hand_collision.py` 创建 rubber-hull sidecar。
3. 禁止覆盖源 `scene_act.xml`。
4. 生成 case-specific override。
5. 解析并审计 effective E167A config。
6. 生成 scene snapshot 和 sha256 manifest。
7. 分配唯一 run id、output path、machine、GPU、queue order。

建议状态：

```text
READY_REUSE_E167
READY_PRIMARY_V1
READY_RESCUE_V2
REJECT_RAW_CONTACT
REJECT_TEMPLATE
REJECT_OMNIRT_DUAL_INFEASIBLE
REJECT_TARGET_GATE
REJECT_VISUAL_QC
```

### Phase 6：canary 与 full CEM

Canary 顺序：

| canary | 目的 |
|---|---|
| E167 已知成功 seed 一条 | 验证 generic E167A profile 与历史 resolved config 一致 |
| `bucket004_022_p1` 或同模板 bucket row | 验证非 box + rubber sidecar + E167A 通路 |
| 已知 v1-success seed 的 `omnirt_v2` | 验证 rescue adapter 能正常执行，不评价 rescue 效果 |
| 一条 obs1/obs3 | 验证 obstacle tag、空场景 replay 和评测字段 |

Canary 全部通过后才能启动 full。

Full CEM 只跑一个 production arm：

```text
E167A_zOnlyBody
```

不运行：

```text
E167A_B1
E167A_B2
E163 baseline sweep
reward/threshold sweep
```

### Phase 7：评测、视觉签收与 release

评测必须直接 import：

```python
from eval.core.core_metrics import evaluate_sequence, EvalConfig
```

不得动态 import E147/E162/E167 evaluator。

Release hard gates：

| 类别 | 条件 |
|---|---|
| artifacts | root NPZ、outdir `trajectory_mjwp_act.npz`、`config_act.yaml`、full MP4 全部存在 |
| config | E167A parity/axis audit pass |
| source | 3cm raw-contact pass；mask mapping exact |
| tracking | `fall_flag=false`；terminal pelvis-z error `<=0.08m` |
| z executability | monitored body-z peak `<=0.25m` |
| contact | physics contact in raw mask `>=0.25` |
| release | false contact outside mask `<=0.30` |
| penetration | physics penetration >3mm frame frac `<=0.25` |
| lower body | lower-body/object interference `<=0.05` |
| relative gate | 有可比 historical baseline 时，raw in-mask contact delta `>=-0.05` |
| visual | 无 fall、趴物体、明显穿模、物体飞离、错物体、错人、异常障碍动作 |

没有可比 historical baseline 的新 case 使用 absolute gate + visual QC，不能伪造 baseline delta。

Fail row 仍保留 metrics、video 和 failure mode，但不进入 RL export。

### Phase 8：S6 object trajectory bank 与 RL handoff

最终生成：

```text
s6_downstream/rl_export/rl_export_input.tsv
s6_downstream/rl_export/object_trajectory_bank.tsv
s6_downstream/rl_export/summary.{json,md}
```

每条 RL-ready row 至少包含：

- `object_key`
- `case_id`
- `source_action`
- `source_obstacle_level`
- `obstacle_context_not_reconstructed`
- `retarget_variant_id`
- `target_variant_id`
- `hand_collision_variant_id`
- `source_exp_id`
- `spider_method_id`
- `scene_act`
- `trajectory`
- `contact_mask`
- `cem_result_npz`
- `cem_status`
- `visual_qc_status`
- `rl_export_decision`
- source/result SHA256

下游只消费：

```text
rl_export_decision=RL_EXPORT_READY
```

E168 不启动 RL 训练，不把 CEM pass 描述为 RL success。

---

## 6. 多机动态 GPU 执行策略

本节按 `.codex/skills/experiment-planning-zh/remote-execution.md` 执行。E168 最多同时使用 `local GPU0 + A6000 2 GPU + A100 动态最多 4 GPU`，即最多 7 个 worker；实际 A100 worker 数由启动时空闲卡决定。

### 6.1 机器角色与 GPU 选择

| worker pool | GPU 来源 | 选择规则 | 主要角色 |
|---|---|---|---|
| `local-gpu0` | 本地 GPU0 | 仅在人工确认空闲后使用 | canary、快速诊断、failed-case recovery、最后尾单 |
| `a6000-gpu0/1` | A6000/RTX 6000 Ada 远程两卡机 | `GPU0/GPU1`，每卡同时最多 1 个 CEM | Box004、Bucket004 主队列，小规模 v2 rescue |
| `a100-gpu${id}` | 8 卡 A100 远程机 | 显存占用 `<5000MB` 的 GPU，哪个空闲用哪个，最多 4 张 | Box021 主队列、长序列、跨物体 overflow |

A100 不固定 `GPU0-3` 或 `GPU4-7`。启动时如果 GPU `2,3,6,7` 满足显存占用 `<5000MB`，则本轮 A100 worker 就是 `a100-gpu2/a100-gpu3/a100-gpu6/a100-gpu7`。如果只找到 1-3 张空闲卡，就只使用这些卡；如果 0 张空闲卡，则跳过 A100 队列，不 kill 其他进程。

A100 选择命令必须固化在 `run_E168_remote_a100.sh`，并把完整 snapshot 写入 `workspace/core4d/results/E168/s0_environment/a100_gpu_selection.tsv`：

```bash
A100_GPU_MEM_USED_LIMIT_MB=5000
A100_MAX_GPUS=4
A100_SELECTED_GPUS="$(
  ssh "$A100_HOST" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v limit=$A100_GPU_MEM_USED_LIMIT_MB '{gsub(/ /,\"\",\$1); gsub(/ /,\"\",\$2); if (\$2+0 < limit) print \$1}' | head -n $A100_MAX_GPUS | paste -sd, -"
)"
test -n "$A100_SELECTED_GPUS"
```

A100/A6000 的真实 SSH alias、repo path、raw root、SMPL-X path、Holosoma/SPIDER commit、dirty 状态和 GPU snapshot 必须在 Phase 0 写入 environment manifest；计划中不猜测别名。

### 6.2 Execution manifest 字段

S4 target gate + visual QC 完成后才生成最终 CEM execution manifest。每条 scheduled row 必须包含：

| 字段 | 含义 |
|---|---|
| `case_id/object_key/source_action/source_obstacle_level` | 数据身份 |
| `retarget_variant_id/target_variant_id` | `omnirt_v1/ref_fk` 或 v1 infeasible 后的 `omnirt_v2/ref_fk` |
| `trimmed_frames/raw_contact_frames/contact_mask_sha256` | 调度估算和 mask provenance |
| `machine_profile/remote_host/remote_root` | `local`、`a6000-2gpu` 或 `A100-8gpu` |
| `gpu_id/worker_id/queue_order` | 具体 GPU 与同卡串行顺序 |
| `run_id/output_npz/outdir/log_path/video_path` | 唯一路径，禁止覆盖 |
| `scene_act/override/config_sha256/scene_snapshot_sha256` | 复现证据 |
| `status/failure_mode/retry_of` | 完成、失败、rescue 或重跑关系 |

### 6.3 调度方式

禁止按初始 66 条 candidate 静态启动。实际 ready 数量只有 S1-S4 后才确定，调度步骤固定为：

1. 运行 canary：本地 GPU0 先跑 E167 已知成功 seed、bucket row、v2 adapter smoke、obs1/obs3 smoke。
2. 采集 worker pool：A6000 固定 `0,1`；A100 启动时按显存占用 `<5000MB` 选最多 4 张；本地 GPU0 只在 canary/recovery/tail 使用。
3. 对 `target_gate_status=pass` 且 `visual_qc_status=pass` 的 rows，按 `trimmed_frames` 降序做 Longest Processing Time 近似均衡。
4. 初始 soft preference：Box004/Bucket004 优先 A6000；Box021、长序列和 overflow 优先 A100；若某 pool 不可用，则按当前可用 worker 重新均衡。
5. v1 primary 先跑；只有 `stage2b_status=omniretarget_infeasible` 的 rows 进入 `omnirt_v2/ref_fk` rescue 队列，rescue 成功后重新经过 S4/S5，再进入 CEM manifest。
6. 同一 GPU 内严格串行，不同 GPU 并行；每个 tmux session 只消费一个 worker queue。
7. 每条 run 的 NPZ/video/outdir/log 路径唯一，路径中必须包含 `case_id`、`retarget_variant_id`、`worker_id` 或唯一 run id。

如果 A100 空闲 GPU 在 selection 和 tmux 启动之间被占用，launcher 必须重新查询并重建 A100 worker pool；不能盲目沿用过期 `A100_SELECTED_GPUS`。

### 6.4 远程同步与 preflight

每台远程机必须按 `remote-execution.md` 做 fast-forward git 同步，`.gitignore` 忽略的大输入只同步本次 manifest 需要的文件：

- active case scene/task；
- rubber sidecar；
- object asset；
- contact mask；
- override/config；
- execution manifest；
- E168 scene snapshot manifest；
- S3 retarget/trimmed target 输入；
- 必要的 baseline/reference evidence，不同步整个历史 `logs/` 或 `workspace/*/results/`。

每个 worker 启动前必须执行 preflight：

1. `git rev-parse HEAD` 等于本地计划记录的 commit；
2. `scene_act`、object asset、contact mask、override、target trajectory 全部存在；
3. `gpu_id` 属于当前 worker pool，A100 时还必须满足显存占用 `<5000MB`；
4. run output path 不存在或处于可恢复的 incomplete 状态；
5. `HOLOSOMA_ROOT`、`HOLOSOMA_DEPS_DIR`、`CORE4D_RAW_ROOT`、`SMPLX_MODEL_DIR` 均为 environment manifest 中记录的值。

推荐 tmux session 命名：

```text
E168_<profile>_gpu<gpu_id>_<mode>_<YYYYmmdd_HHMMSS>
```

### 6.5 结果回收与多机完成确认

结果回收必须使用独立 pull 脚本，并按 worker 汇总：

- root NPZ count；
- outdir `trajectory_mjwp_act.npz` count；
- `config_act.yaml` count；
- MP4 count；
- sha256/size sanity；
- remote tmux/train log；
- failed row 的 `failure_mode`、traceback 摘要和是否 eligible for retry。

`watch_E168_same_object_queue.sh` 只负责 pull、artifact audit、eval 触发和缺口报告；不直接新增未在 execution manifest 中登记的 run。

---

## 7. 计划修改的文件

以下文件在 implementation 阶段创建；本轮 planning-only 不创建：

| # | 文件 | 作用 |
|---:|---|---|
| 1 | `workspace/core4d/scripts/experiments/E168/build_same_object_recall_manifest.py` | 精确 object/action/seed 召回与 102/72/66 accounting |
| 2 | `workspace/core4d/scripts/experiments/E168/build_omnirt_rescue_manifest.py` | v1 infeasible 到 `omnirt_v2` rescue queue |
| 3 | `workspace/core4d/scripts/experiments/E168/build_e167a_manifest.py` | generic E167A profile + case layer + split |
| 4 | `workspace/core4d/scripts/experiments/E168/export_rl_handoff.py` | object trajectory bank 与 RL export |
| 5 | `workspace/core4d/scripts/train/train_E168_same_object_e167a.sh` | 固定 CEM single/full 入口，启动前 snapshot |
| 6 | `workspace/core4d/scripts/launch/active/run_E168_local.sh` | local canary/recovery |
| 7 | `workspace/core4d/scripts/launch/active/run_E168_remote_a6000.sh` | A6000 2-GPU queue |
| 8 | `workspace/core4d/scripts/launch/active/run_E168_remote_a100.sh` | A100 dynamic idle-GPU queue，显存占用 `<5000MB`、最多 4 张 |
| 9 | `workspace/core4d/scripts/launch/active/pull_E168_remote_a6000_results.sh` | A6000 结果回收 |
| 10 | `workspace/core4d/scripts/launch/active/pull_E168_remote_a100_results.sh` | A100 结果回收 |
| 11 | `workspace/core4d/scripts/launch/active/watch_E168_same_object_queue.sh` | 多机完成确认、pull、artifact audit、eval |
| 12 | `workspace/core4d/scripts/eval/runners/eval_E168_same_object_e167a.py` | 公共 metrics 驱动的 per-case release eval |
| 13 | `workspace/core4d/scripts/eval/wrappers/eval_E168_same_object_e167a.sh` | 固定 eval 入口 |
| 14 | `workspace/core4d/scripts/eval/reports/gen_E168_same_object_report.py` | object/obs/variant/yield summary 和结果 log |

若现有 v3 orchestration 缺少“按 exact input manifest 停在指定 stage”的能力，优先扩展通用参数，不复制整条 pipeline。任何 core 修改必须默认关闭或保持旧行为。

---

## 8. 结果目录

正式结果只允许写入：

```text
workspace/core4d/results/E168/
├── s0_environment/
├── s1_raw_contact/
│   ├── inventory/
│   ├── recall/
│   └── raw_contact/
├── s2_templates/
├── s3_retarget/
│   ├── omnirt_v1/ref_fk/
│   └── omnirt_v2/ref_fk/
├── s4_gate_visual_qc/
├── s5_handoff/
│   ├── hand_collision/
│   ├── overrides/
│   └── handoff_manifest.tsv
├── s6_downstream/
│   ├── cem/
│   │   ├── smoke/
│   │   └── full/
│   ├── eval/
│   ├── visual_qc/
│   └── rl_export/
├── registries/
├── imported_snapshots/
└── scene_snapshot/
```

正式产物不得放 `/tmp` 或 legacy data-construction 目录。`results/E168` 不进 git。

---

## 9. 计划命令

以下命令在 implementation 阶段固化脚本后执行；本轮不执行。

### 9.1 生成脚本骨架

```bash
python workspace/core4d/scripts/gen_experiment.py \
  --exp-id E168 \
  --description "same_object_E167A_zonly_expansion" \
  --splits "local-gpu0,a6000-gpu0,a6000-gpu1,a100-dynamic" \
  --dry-run
```

Generator 仅用于预览通用骨架。A100 的真实 GPU id 必须由 `run_E168_remote_a100.sh` 启动时查询，不在 dry-run split 中写死。

### 9.2 Recall/S1-S4 preflight

```bash
bash workspace/core4d/scripts/train/train_E168_same_object_e167a.sh recall
bash workspace/core4d/scripts/train/train_E168_same_object_e167a.sh stage2b
bash workspace/core4d/scripts/train/train_E168_same_object_e167a.sh target-gate
bash workspace/core4d/scripts/eval/wrappers/eval_E168_same_object_e167a.sh preflight
```

### 9.3 Canary

```bash
bash workspace/core4d/scripts/launch/active/run_E168_local.sh canary
bash workspace/core4d/scripts/eval/wrappers/eval_E168_same_object_e167a.sh canary
```

### 9.4 Full 多机

```bash
bash workspace/core4d/scripts/launch/active/run_E168_local.sh full

A6000_HOST=<confirmed_alias> \
  bash workspace/core4d/scripts/launch/active/run_E168_remote_a6000.sh full

A100_HOST=<confirmed_alias> \
A100_GPU_MEM_USED_LIMIT_MB=5000 \
A100_MAX_GPUS=4 \
  bash workspace/core4d/scripts/launch/active/run_E168_remote_a100.sh full
```

若 A100 启动时 `nvidia-smi` 显示 GPU `2,3,6,7` 的显存占用均 `<5000MB`，`run_E168_remote_a100.sh` 应写出 `A100_SELECTED_GPUS=2,3,6,7` 并只在这四张卡上生成 worker queue。

### 9.5 Pull/eval/export

```bash
A6000_HOST=<confirmed_alias> \
  bash workspace/core4d/scripts/launch/active/pull_E168_remote_a6000_results.sh full

A100_HOST=<confirmed_alias> \
  bash workspace/core4d/scripts/launch/active/pull_E168_remote_a100_results.sh full

bash workspace/core4d/scripts/eval/wrappers/eval_E168_same_object_e167a.sh full
python workspace/core4d/scripts/experiments/E168/export_rl_handoff.py
```

---

## 10. 成功标准

### 10.1 Pipeline completion

| 项目 | 完成标准 |
|---|---|
| Recall | 102 raw / 72 move / 6 seed / 66 non-seed，计数和 case identity 可复核 |
| Scope | production manifest 中没有非 move action |
| Raw contact | 3cm/5cm 分开；所有 production row 为 3cm pass |
| Retarget | 每条 candidate 有 v1 最终状态；所有 v1 infeasible 有 `omnirt_v2` rescue 状态 |
| Target | 每条 CEM row target gate + visual QC pass |
| CEM | execution manifest 中所有 scheduled rows 都有完成或明确 failure |
| Eval | 每条完成 CEM 有公共指标、release decision、failure mode |
| Visual | 每条 release row 有 MP4/关键帧和具体观察 |
| Export | 只导出 `RL_EXPORT_READY` |
| Reproducibility | scene/config/code/data SHA 和持久路径完整 |

### 10.2 Data yield

最低数据目标：

```text
box004: >= 2 non-seed RL_EXPORT_READY
box021: >= 2 non-seed RL_EXPORT_READY
bucket004: >= 2 non-seed RL_EXPORT_READY
```

扩展目标：

- Tier A obs0 ready rows 尽量全部生产；
- Tier B obs1/obs3 在视觉与物理 gate 通过后全部纳入；
- `omnirt_v2` 至少救回 1 条已知 v1 infeasible case，则支持 rescue hypothesis；
- 如果 `omnirt_v2` 0 条救回，实验仍完整结束，但 C5-rescue 的效果结论为 fail。

不得为了达到数量目标：

- 使用 5cm-only row；
- 忽略 visual QC；
- 导出 fall/tracking fail；
- 降低 E167A z gate；
- 把 seed 重跑计为新数据；
- 把同一 trajectory 的不同路径/副本重复计数。

---

## 11. 风险与处置

| 风险 | 处置 |
|---|---|
| 66 条只是 recall upper bound，实际 ready 较少 | 先执行 S1-S4，再生成 CEM manifest；报告逐 stage yield |
| `omnirt_v2` 不能解决 CVXPY infeasible | 记录 dual-infeasible，不做第三次相同参数重跑 |
| `omnirt_v2` 改善可行性但损害 contact/target | 仍走完整 target/CEM gate，variant-specific report |
| obs1/obs3 原始障碍未重建 | 明确 tag，增加空场景视觉审查，不声称 obstacle-aware |
| bucket proxy/碰撞语义不正确 | 强制 clean_reviewed + mesh/collision overlay |
| 新 case 缺 rubber sidecar | 用标准 patch 工具生成，不覆盖源 scene |
| 新 case 没有历史 baseline | 使用 absolute gate + visual QC，不伪造 delta |
| mask/raw/trim 帧错位 | 只允许显式 trim slice；禁止补零、resize、interpolate |
| 多机结果覆盖 | variant/case/run path 唯一；同 GPU 串行；pull 后 sha/count audit |
| A100/A6000 环境不一致 | Phase 0 记录 repo/env/raw/model SHA/path/GPU snapshot；canary 后再 full |
| A100 空闲卡选择过期 | launch 前按显存 `<5000MB` 重查；GPU 不再空闲则重建 worker pool，不沿用旧 selection |
| E167 tracker 与实际 artifacts 不一致 | E168 显式引用实际 evidence；执行前补 E167 provenance reconciliation |

---

## 12. No-Go Rules

- 不处理 `pass/raise/rot/strike`。
- 不从 66 candidate 直接启动 66 条 CEM。
- 不把 5cm near-contact 当作 3cm production contact。
- 不修改或覆盖 E167/E107/E145 结果。
- 不覆盖 source `scene_act.xml`。
- 不把 `omnirt_v2` 标为 v1。
- 不启用 `replace` / fingertip replacement。
- 不把 E167A 写入 target variant。
- 不启用 E167A+B1/B2。
- 不静默修剪、补零或 resize contact mask。
- 不把 target gate pass 等价为 CEM pass。
- 不把 CEM pass 等价为 RL success。
- 不启动下游 RL 训练。
- 不把正式结果写到 `/tmp` 或 legacy workspace。

---

## 13. 结束记录

执行完成后必须：

1. 写新的 E168 result log，不修改历史 completed log。
2. 生成 `log/INDEX.md`。
3. 更新 `EXPERIMENT_TRACKER.md` 的 E168 状态和 log 链接。
4. 更新 `progress.md`，保留关键错误、决策、yield 和路径，删除纯轮询噪声。
5. 在 Claims 表逐项给出 pass/fail/partial。
6. 如果所有 required Claims 通过，按实验规则 commit + push。
