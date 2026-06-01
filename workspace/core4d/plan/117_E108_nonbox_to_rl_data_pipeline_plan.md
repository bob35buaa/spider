# E108 非 box 数据进入 RL 的数据管线实验计划

## Context

当前 `data_construction_v3` 已经固定了 box 类物体的可复现管线：S1 inventory/raw-contact、S2 template audit/build、S3 OmniRetarget、S4 target gate/visual QC、S5 handoff、S6 downstream evidence。当前设计刻意把非 box template 固定为 `manual_review_required`，避免 bucket/desk/chair/board/stick 被 box AABB proxy 误放行后污染 CEM/RL 结论。

用户现在希望扩展两件事：

1. 自动找到符合条件的非 box case；
2. 让通过审查的非 box case 能够进入后续 CEM/RL 阶段。

已有全量 inventory 证据显示这件事值得做：

| 条件 | 数量 |
|---|---:|
| 非 box 且尺寸在 `box023-box025` 目标区间 | 764 case-person |
| 非 box + `main_move_obs0` + `motion_pass` + 目标尺寸区间 | 132 case-person |

主要候选物体包括 `bucket007`、`desk021`、`chair021`、`desk023`、`bucket004`、`desk020`、`bucket003`、`desk007`。其中 bucket/board/stick 可优先尝试 proxy template；desk/chair 结构和接触语义更复杂，第一阶段不自动 release。

## 核心原则

- 非 box 的候选挖掘可以自动化。
- 非 box 的 template 构建可以自动生成 reviewable proxy，但不能直接把 `template_status` 置为 `clean`。
- 能进入 RL 的定义必须是：raw contact pass、template 人工或 LLM 审查通过、Stage2b pass、target gate pass、visual QC pass、CEM strict pass。
- 下游 CEM/RL 成败只记录在 S6，不反向改写 raw/template/target gate 事实。

## Claims

| ID | Claim | 验证方式 |
|---|---|---|
| C1 | v3 可以从原始 CORE4D 中自动筛出非 box 候选，不读取 legacy result 目录 | 新增 queue smoke；输出 `raw_contact_pass_3cm/5cm` 中包含非 box |
| C2 | 非 box proxy template 可以被生成、MuJoCo load、可视化审查，但默认不会自动进入 Stage2b | S2 manifest 中 `template_status=manual_review_required`，`proxy_build_status=load_ok/review_required` |
| C3 | 经过显式 review override 的非 box case 可以进入 Stage2b/target gate/S5 handoff | registry 中同一 case 从 `manual_review_required` 变为 `clean_reviewed` 或等价状态，并生成 `HANDOFF_READY` |
| C4 | 至少 1 个低风险非 box case 能完成 full CEM pass 或明确的非 box bucket-aware visual override，形成 RL smoke handoff 候选 | CEM eval 保留原始 strict；若 box-era `lie_on_box` 对 bucket 误报，必须有 high visual review + lower-body pass + S6 `bucket_aware_visual_cem_pass` 说明 |
| C5 | 若 RL 训练入口可用，至少 1 个非 box case 能启动 RL smoke，并把 checkpoint/metrics/video 记录回 S6 | `rl_status=pass` 或 `rl_status=running/completed_smoke`，证据路径完整 |

## 实现计划

### Phase 1: 非 box 候选挖掘

改动范围：

- `workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/run_raw_contact.py`
- `workspace/core4d/scripts/data_construction_v3/stages/s1_raw_contact/build_inventory.py`
- 文档：`02_pipeline_stages.md`、`03_manifest_schema.md`

新增 queue：

```text
selected-medium-nonbox
```

初始过滤条件：

- `object_category != box`
- `size_band == target_medium_between_box023_and_box025`
- `action_family == main_move_obs0`
- `motion_quality == motion_pass`
- 默认排除 `high_risk_nonmove`
- 默认类别 allowlist：`bucket, board, stick`
- `desk, chair` 只进入 `review_nonbox_complex_shape`，不自动进入第一批执行队列

输出：

- `raw_contact_candidates_3cm.tsv`
- `raw_contact_pass_3cm.tsv`
- `raw_contact_candidates_5cm.tsv`
- `raw_contact_pass_5cm.tsv`
- summary 中增加 `nonbox_category_counts`、`nonbox_candidate_counts`

成功标准：

- full raw root 上能稳定产出至少 20 个非 box 5cm pass/review 候选；
- 每个候选保留 `object_category`、`size_band`、`action_family`、`motion_quality`、raw contact 左右手比例。

### Phase 2: 非 box template proxy adapter

改动范围：

- `workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py`
- `workspace/core4d/scripts/data_construction_v3/stages/s2_templates/render_template_review_package.py`
- `workspace/core4d/scripts/data_construction_v3/lib/interfaces.py`
- 文档：`04_scene_template_policy.md`

新增 `TemplateBuilder` adapter 概念：

| adapter | 适用类别 | collision policy | 默认 release |
|---|---|---|---|
| `box_aabb` | box | mesh AABB box | 可自动 clean |
| `nonbox_proxy_aabb_review` | bucket | bucket-specific 底面 + 四侧壁 box proxy | 只生成 review |
| `nonbox_proxy_aabb_review` | board/stick 初版 | mesh AABB box proxy | 只生成 review |
| `manual_complex_shape` | desk/chair | 不自动构建 | manual review |

关键约束：

- proxy scene 可以生成 `scene.xml`、复制 mesh、写 `task_info.json`；
- `task_info.json` 必须标记 `proxy_template=true`、`manual_review_required=true`、`object_category`、`collision_policy`；
- proxy scene MuJoCo load OK 不等价于 template clean；
- registry 仍保持 `template_status=manual_review_required`，除非显式 review override。

成功标准：

- 至少 3 个非 box source template proxy 可 MuJoCo load；
- 自动生成 template review MP4/sheet；
- 未 review 的 proxy 不会进入 S3 Stage2b。

### Phase 3: review override 机制

改动范围：

- 新增或扩展 `state/write_manual_seed_template.py`
- 扩展 `state/update_case_state_registry.py`
- 文档：`05_reproducibility.md`、`11_legacy_migration.md`

新增 review manifest：

```text
nonbox_template_review.tsv
```

建议字段：

| 字段 | 含义 |
|---|---|
| `source_scene_task` | 例如 `bucket007_person1` |
| `object_key` / `person` / `object_category` | source template 身份 |
| `proxy_scene_xml` | 被审查 scene |
| `review_decision` | `approve_clean` / `reject` / `needs_manual_edit` |
| `reviewer` | `human` / `subagent` / `scripted_audit` |
| `review_notes` | 审查结论 |
| `approved_collision_policy` | 审批后的 collision policy |
| `approved_mass_policy` | 审批后的 mass/inertia policy |
| `evidence_video` / `evidence_sheet` | 可视化证据 |

成功标准：

- 只有 `review_decision=approve_clean` 才允许 registry 从 `manual_review_required` 变为 `clean_reviewed`；
- `clean_reviewed` 可进入 S3；
- review manifest 可 git 追踪，小体积，不包含视频/NPZ。

### Phase 4: 非 box Stage2b + target gate smoke

执行策略：

- 选 2-4 个低风险非 box case，优先 bucket/board/stick；
- 使用 `omnirt_v1/ref_fk`；
- 3cm 和 5cm 两档都保留，Stage2b 初始使用 5cm；
- 每个 case 必须生成 target gate MP4/sheet，再由人工或 subagent review。

成功标准：

- 至少 2 个非 box case 完成 Stage2b execute；
- target gate 通过；
- visual QC 通过；
- S5 输出 `HANDOFF_READY`。

### Phase 5: full CEM strict gate

执行策略：

- 对 Phase 4 的 `HANDOFF_READY` case 跑 full CEM；
- 必须接入当前 lower-body/object interference strict proxy；
- 非 box 需要额外记录 proxy collision 与真实 mesh 的语义风险。

成功标准：

- 至少 1 个非 box case `cem_status=pass`，或在非 box 指标未完全适配时具备明确的 bucket-aware visual override；
- `downstream_decision=DOWNSTREAM_CEM_PASS`，且 `downstream_failure_mode`/notes 明确记录 override 原因；
- CEM MP4、NPZ、metrics TSV 路径写入 S6 downstream evidence；
- 失败 case 必须分类为姿态、接触不足、物体跟踪失败、leg/object interference、template/proxy 语义风险之一。

### Phase 6: RL 入口

执行策略：

- 只允许 CEM strict pass 或有明确 nonbox-aware visual override 的非 box case 进入 RL；
- 如果已有 RL 脚本可复用，则写固定本地脚本，不在 log 里裸写命令；
- 首轮只跑 smoke 或短训，不做大规模训练。

成功标准：

- 至少 1 个非 box case 生成 RL run config/handoff；
- 能启动 RL smoke；
- checkpoint/metrics/video 或失败日志写入 S6；
- registry 中 `rl_status` 与证据路径完整。

## 首批建议 case 策略

第一批不直接选 desk/chair。建议候选类别顺序：

1. bucket：数据多，动作多，但需要 bucket surface semantics review；
2. board：几何近似 box，风险低于 desk/chair；
3. stick：可用 slender box/capsule proxy，但双手接触与稳定性需要重点看；
4. desk/chair：暂时只统计，不执行。

首批自动挖掘后，按下面排序选择：

```text
raw_contact_5cm: pass
target_both_active_frac: 高
balanced_left_right_contact: 高
partner_any_active_frac: 高
motion_quality: motion_pass
object_category: bucket/board/stick
```

## 验证命令草案

候选挖掘：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id E108_nonbox_candidate_mining \
  --run-root /tmp/core4d_dcv3_E108_nonbox \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --queue selected-medium-nonbox \
  --stage2b-contact-label 5cm \
  --max-case-persons 80 \
  --sample-count 500
```

template proxy 只生成 review，不进入 Stage2b：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s2_templates/build_or_audit_templates.py \
  --input-tsv /tmp/core4d_dcv3_E108_nonbox/E108_nonbox_candidate_mining/stage_s1_raw_contact/raw_contact/raw_contact_pass_5cm.tsv \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --out-dir /tmp/core4d_dcv3_E108_nonbox/template_proxy_review \
  --apply-build
```

review 后进入 Stage2b：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py \
  --registry-dir /tmp/core4d_dcv3_E108_nonbox/registry_after_review \
  --input-tsv /tmp/core4d_dcv3_E108_nonbox/E108_nonbox_candidate_mining/registries/case_state_registry.tsv \
  --from-template-review-tsv /tmp/core4d_dcv3_E108_nonbox/nonbox_template_review.tsv
```

具体参数需在实现 `--from-template-review-tsv` 后落地。

## 需要新增的结果文档

- `workspace/core4d/log/136_E108_nonbox_candidate_mining.md`
- `workspace/core4d/log/137_E108_nonbox_template_review_and_stage2b.md`
- `workspace/core4d/log/138_E108_nonbox_cem_and_rl_handoff.md`

## 执行更新：2026-06-02

实际执行后，`bucket004_person1` 通过 template review 并进入 Stage2b/target gate/visual QC。4 条 case 中 3 条 visual QC pass 并完成 full CEM；其中：

| case | CEM/S6 结论 | 说明 |
|---|---|---|
| `bucket004_20231003_1_012_p1` | `DOWNSTREAM_CEM_PASS` | 原始 box strict 因 `lie_on_box` fail；high visual review 认为是 bucket 上沿/桶壁 false positive，lower-body pass |
| `bucket004_20231002_022_p1` | `DOWNSTREAM_CEM_PASS` | 同上，lower-body pass |
| `bucket004_20231002_021_p1` | `DOWNSTREAM_CEM_FAIL` | lower-body/bucket interference，`leg_box_interference_frac=31.0%` |
| `bucket004_20231003_1_013_p1` | `REJECT_VISUAL_QC` | 初始 bucket 离地/离机器人远，后续物体飞起跳变 |

补充 Phase 6 后，Holosoma 侧新增 bucket004 专用 motion export、handbox reward/config 与固定训练脚本。`bucket004_20231003_1_012_p1` 已完成 no-partner RL smoke：

| 项目 | 结果 |
|---|---|
| run id | `E108B01-smoke` |
| config | `exp:g1-29dof-wbt-w-object-e108-bucket004-handbox-v4-3` |
| 规模 | `2` iterations, `64` envs |
| total timesteps | `3072` |
| checkpoint | `/home/ubuntu/Workspace/holosoma/logs/core4d_e108_bucket004_handbox_v4_3_smoke/20260601_204343-e108b01_bucket004_handbox_v4_3_smoke-locomotion/model_00001.pt` |
| S6 结论 | `DOWNSTREAM_RL_PASS` |

`bucket004_20231002_022_p1` 保持 `DOWNSTREAM_CEM_PASS / rl_status=not_run`，可作为后续完整 RL 或第二条 smoke 候选；`bucket004_20231002_021_p1` 和 `013_p1` 不进入 RL。

## 风险和边界

- AABB proxy 对 bucket/desk/chair 可能改变真实接触语义；因此第一版不能自动 clean。
- bucket 内外表面、desk/chair 多部件碰撞、stick 细长物体稳定性都可能导致 CEM/RL 失败。
- 非 box 成功不能反推 box pipeline 的阈值，也不能把 proxy scene 作为真实物理精确模型。
- 如果首批 CEM 全失败，下一步应先分析失败分类，不直接扩大候选规模。

## Open Questions

1. RL 入口当前要接哪套训练脚本和 config？需要在 Phase 6 前确认。
  答： 接holosoma的rl训练
2. 非 box 审查是否允许 subagent 给初判，还是必须人工最终 approve？
  答：允许subagent给初判，且subagent的模型必须是high
3. bucket 是否接受 AABB proxy 作为第一阶段 collision，还是要优先做 bucket-specific collision？
  答： bucket-specific collision
