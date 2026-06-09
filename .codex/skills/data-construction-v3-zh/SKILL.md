---
name: data-construction-v3-zh
description: Core4D data_construction_v3 数据构建工作流。用于从 CORE4D_Real 原始 mocap 构建 inventory/raw-contact/template/OmniRetarget/target gate/visual QC/CEM-RL handoff，维护 case_state_registry，归档 workspace/core4d/results/E###，以及处理非 box template review、retarget variant、target route 和 S6 downstream evidence。触发词：数据构建、data_construction_v3、CORE4D raw、raw contact、scene template、Stage2b、target gate、visual QC、handoff、S6 registry、CEM/RL evidence、非 box 候选。
allowed-tools: "Read Write Edit Bash Glob Grep"
metadata:
  version: "1.0.0"
---

# Core4D 数据构建 v3

用于执行、审查和维护 `workspace/core4d/scripts/data_construction_v3/` 数据管线。此 skill 关注可复现的数据构建，不替代 RL 实验计划；如果任务同时涉及实验设计、CEM/RL 结果分析或 tracker 更新，也应遵循 `experiment-planning-zh`。

## 先恢复上下文

开始前优先读取：

```text
workspace/core4d/docs/data_construction_v3/README.md
workspace/core4d/docs/data_construction_v3/02_pipeline_stages.md
workspace/core4d/docs/data_construction_v3/03_manifest_schema.md
workspace/core4d/docs/data_construction_v3/04_scene_template_policy.md
workspace/core4d/docs/data_construction_v3/08_retarget_variants.md
workspace/core4d/docs/data_construction_v3/10_diagnostic_contracts.md
workspace/core4d/docs/data_construction_v3/12_completion_audit.md
workspace/core4d/docs/data_construction_v3/14_release_readiness.md
workspace/core4d/docs/data_construction_v3/15_hand_collision_variants.md
workspace/core4d/EXPERIMENT_TRACKER.md
workspace/core4d/progress.md
```

只读取与当前问题有关的文档；不要把全部 docs 一次性塞进上下文。

## 结果路径硬规则

正式数据构建实验结果必须放在对应实验目录：

```text
workspace/core4d/results/E###/
```

示例：E108 的 run root 是 `workspace/core4d/results/E108/`。

不要把正式结果放在：

- `/tmp/...`
- `workspace/v3/data_construction`
- `workspace/v3/data_construction_v2`
- `workspace/core4d/data_preprocess`

`/tmp` 只允许用于可丢弃的 smoke/release check 临时产物。若为了快速试跑临时用了 `/tmp`，任务结束前必须复制或重跑到 `workspace/core4d/results/E###/`，并在 log/progress 中记录最终持久路径。

`workspace/core4d/results/` 不进 git；用户会用外部同步工具同步。不要因为结果重要就 `git add -f` results。

## 标准阶段

v3 pipeline 阶段：

```text
S0  environment/config check
S0b case_state_registry init
S1  raw inventory + raw contact 3cm/5cm
S2  source scene template build/audit
S3  OmniRetarget/SPIDER preprocess by retarget_variant_id x target_variant_id
S4  target gate + visual QC
S5  candidate bank + handoff manifest
S6  CEM/RL downstream evidence
```

S0-S2 可共享；S3 之后必须带 `retarget_variant_id` 和 `target_variant_id`。S5/CEM 之后还可带 `hand_collision_variant_id`。不同 variant/route/collision 输出不能互相覆盖。

## 默认路线和 contract

默认 production route：

```text
retarget_variant_id=omnirt_v1
target_variant_id=ref_fk
```

Contract 规则：

- E098 是所有 route 的基础 contract。
- E099-E101 只属于 `target_variant_id=fingertip_aware`。
- `fingertip_aware` 与 `ref_fk`、`adaptive` 平级，不是默认路线。
- 若选择 `fingertip_aware`，必须提供或生成 E099-E101 route diagnostic evidence；否则不能进入 Stage2b。

OmniRetarget 参数必须显式版本化。至少区分：

- 原始算法
- `omnirt_v1`
- `omnirt_v1 + REPLACE_WRIST_WITH_FINGERTIP`

算法参数改变就是新 variant，不要覆盖旧结果。

## 手部碰撞体 variant

机器人手部碰撞体是 S5/CEM 的独立轴：

```text
hand_collision_variant_id=sphere5cm   # 默认旧行为
hand_collision_variant_id=rubber_hull # rubber hand mesh convex hull
```

规则：

- 该轴只改机器人侧 `lh/rh` hand collision geom，不改 OmniRetarget、不改 target route、不改物体侧 `collision_policy`。
- registry 主键为 `(case_id, retarget_variant_id, target_variant_id, hand_collision_variant_id)`；旧 row 缺省 `sphere5cm`。
- `rubber_hull` 必须通过 sidecar scene 注入，例如 `scene_act_E147_rubber_hull.xml`；禁止覆盖源 `scene_act.xml`。
- adapter 入口是 `stages/s5_handoff/patch_hand_collision.py`；CEM override 通过 `scene_name=<sidecar basename>` 指向 sidecar。
- rubber mesh 评估必须使用 mesh-aware SDF（采样 mesh 顶点或等价方法），不能只用 geom center/rbound 近似。

## Template 规则

source scene template 是数据基础，错误 template 会污染后续 CEM/RL。

Box 类：

- 可按当前 clean template 流程自动构建/审计。
- 必须检查 MuJoCo load、robot inertial、object mass/inertia、collision policy。
- 遇到 source scene missing，不是跳过理由；需要补 template 或进入 backlog。

非 box：

- 候选挖掘可以自动化。
- template 可生成 reviewable proxy。
- 未经显式人工或 subagent review，不能直接置为 `clean`。
- 通过 review 后用 `clean_reviewed` 或等价状态进入 S3。
- bucket/board/stick 可优先做 proxy；desk/chair 使用 tight surface voxel multi-box review proxy，而不是标准桌/椅语义模板。
- desk/chair proxy 规范：从 OBJ 表面 voxelization 生成 `object_collision` + `object_collision_voxel_*` 多个 local AABB boxes；默认 policy 为 `desk_surface_voxel_multibox_proxy_draft` / `chair_surface_voxel_multibox_proxy_draft`；必须保持 `manual_review_required`，不能仅凭 MuJoCo load 或 render pass 自动 release。
- desk/chair review evidence 必须包含 mesh/collision overlay 或 object-only mesh/collision sheet；用 `stages/s2_templates/render_template_mesh_collision_review_package.py --object-only` 生成；确认 proxy 没有明显大一圈、没有套错物体拓扑、没有把圆面三脚凳/侧板 U 架/非标准 chair 误建成标准桌椅。

已知 template 风险：

- 不要把历史污染的 `box021_person1` 作为 base。
- 警惕 robot link inertial 被 object mass/inertia 污染，例如所有 robot link 同 mass/惯量。
- 对新 template 必须保留审查证据、notes 和 source_ref。

## 状态管理

核心 registry：

```text
<run_root>/registries/case_state_registry.tsv
<run_root>/registry_*/case_state_registry.tsv
```

更新状态时使用脚本，不手工改 registry：

```bash
workspace/core4d/scripts/data_construction_v3/state/update_case_state_registry.py
```

S6 downstream evidence 只记录 CEM/RL 后验结果，不反向改写 raw contact、template、Stage2b、target gate 或 visual QC 事实：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s6_downstream/record_downstream_evidence.py
```

已有历史结果要纳入时，优先生成小型 seed/import snapshot；不要隐式读取 legacy result 目录。

## 运行入口

初始化：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/init_workspace.sh <run_id> \
  --holosoma-repo "$HOLOSOMA_REPO" \
  --run-root "workspace/core4d/results/E###" \
  --core4d-raw-root "$CORE4D_RAW_ROOT"
```

从 raw 跑：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode full-from-raw \
  --run-id <run_id> \
  --run-root "workspace/core4d/results/E###" \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR"
```

从 registry/snapshot 继续：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_pipeline.py \
  --mode resume-from-summary \
  --run-id <run_id> \
  --run-root "workspace/core4d/results/E###" \
  --resume-registry "<path>/case_state_registry.tsv"
```

如果当前脚本默认建议 `${HOLOSOMA_REPO}/workspace/v3/data_construction_v3_runs`，对正式 Core4D 实验应改成 `workspace/core4d/results/E###`。

## CEM/RL handoff

CEM 结果、MP4、summary、downstream evidence input 也归属于同一个实验目录，例如：

```text
workspace/core4d/results/E108/s5_handoff/handoff_manifest.tsv
workspace/core4d/results/E108/s6_downstream/cem/full/
workspace/core4d/results/E108/s6_downstream/evidence/
workspace/core4d/results/E108/s6_downstream/rl_export/rl_export_input.tsv
workspace/core4d/results/E108/registries/
```

正式实验根目录必须按 `s0_environment/`、`s1_raw_contact/`、...、`s6_downstream/` 组织。整理前的 smoke/临时目录只能放在 `archive_legacy/`，不得作为新脚本默认输入。

RL motion export 不直接扫描 CEM 目录，也不只读 S5 handoff。CEM evidence 产生后，必须用 `stages/s6_downstream/export_rl_inputs.py` 生成 `s6_downstream/rl_export/rl_export_input.tsv`；下游只消费其中 `rl_export_decision=RL_EXPORT_READY` 的 rows。该表必须同时带 `scene_act`、`trajectory`、`contact_mask`、`cem_result_npz`、`cem_status`，以保证 RL 转换使用的 `scene_act` 与 CEM 所属 target case 对齐。

Holosoma RL 输出可以留在 Holosoma 仓库自己的 `workspace/v3/data_construction/results/` 和 `logs/` 下，但 Core4D S6 registry 必须记录这些路径。

不要把 `DOWNSTREAM_CEM_PASS` 夸大为 RL 成功；不要把 RL smoke 夸大为完整训练收敛。分别记录 `cem_status`、`rl_status`、`downstream_decision` 和 evidence path。

## 验证

代码或管线改动后至少跑：

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | xargs -0 python3 -m py_compile
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh --run-root workspace/core4d/results/E###_release_check --no-smoke
git diff --check
```

若有 raw data 和 SMPL-X，优先追加 compact smoke：

```bash
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root workspace/core4d/results/E###_release_check_with_smoke \
  --core4d-raw-root "$CORE4D_RAW_ROOT" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" \
  --with-smoke
```

正式实验完成后检查：

- 结果目录在 `workspace/core4d/results/E###/`
- registry 行数和关键状态可解释
- CEM/RL evidence 路径存在
- MP4/NPZ/TSV/JSON summary 都在持久目录
- 文档和 tracker 指向持久目录，不指向 `/tmp`
- `git status` 不包含 results 产物
