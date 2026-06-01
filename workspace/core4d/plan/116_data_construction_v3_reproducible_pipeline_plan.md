# Core4D 数据构建 v3 可复现流程固定化计划

日期：2026-06-01

## 1. 背景

当前 E091-E107 已经基本验证了从 CORE4D 原始数据到候选 case、OmniRetarget/SPIDER preprocess、target gate、visual QC、CEM/RL handoff 的主链路。E103 进一步发现并修复了 source scene template 污染问题，说明数据构建阶段必须被当作基础设施固定下来；否则下游 CEM/RL 优化会建立在不可信数据上。

本计划目标不是新增一个实验 trick，而是把整套数据构建流程产品化为可复现流程：任意机器只要提供原始 `CORE4D_Real` 路径、两个 repo 和对应环境，就能按文档从原始 mocap 数据重建候选与下游 handoff 数据。

## 2. 已确认口径

| 项目 | 决策 |
|---|---|
| 标准化入口 | 先放在 `spider` repo 内 |
| 支持模式 | 同时支持 `full-from-raw` 和 `resume-from-summary` 两种模式 |
| 原始数据 | 用户自己提供 `CORE4D_Real` 路径 |
| 环境 | SPIDER 与 OmniRetarget/hsretargeting 环境保持分离；OmniRetarget 环境由已有环境变量控制 |
| OmniRetarget 参数 | 所有求解器参数和 input-conversion 参数必须显式暴露，并作为 retarget variant 进入版本管理 |
| OmniRetarget 分叉 | OmniRetarget 之前的 raw inventory/contact/template 可共用；从 conversion/retarget 开始按 variant 分叉 |
| box template | box 类物体基于 E103 后的 clean source-template 流程自动生成/审计 |
| 非 box 物体 | 不自动放行，必须人工审查 template/collision/mass/inertia |
| source scene missing | 不能跳过；必须生成 template，或进入明确 template backlog |
| visual/subagent review | 机器检查作为硬 gate；人工/LLM review 作为 release checklist，不作为唯一 gate |
| 旧目录策略 | `workspace/v3/data_construction` 和 `workspace/v3/data_construction_v2` 只作为 legacy，不删除、不改写；新流程默认不写入，也尽量不读取 |
| 结果存储 | 在哪台机器运行就写到哪台机器本地新结果目录；本计划不处理跨机器同步 |

## 3. 命名、目录边界与旧目录隔离

为避免和 Holosoma 旧目录 `workspace/v3/data_construction`、`workspace/v3/data_construction_v2` 混淆，spider 内的规范文档采用：

```text
workspace/core4d/docs/data_construction_v3/
```

这里的 `v3` 表示第三版可复现流程规范。新规范不在旧的两个 Holosoma 目录中继续追加文件。

旧目录定位：

| 目录 | 新流程定位 | 默认行为 |
|---|---|---|
| `${HOLOSOMA_REPO}/workspace/v3/data_construction` | legacy v1：旧脚本和旧结果参考 | 不写入；默认不读取 |
| `${HOLOSOMA_REPO}/workspace/v3/data_construction_v2` | legacy v2：E091-E107 历史工作区 | 不写入；默认不读取 |

如果新流程必须依赖某个 legacy 结果，有两种允许方式：

1. 显式暴露 legacy path 参数，并在 run manifest 中记录；
2. 把需要依赖的文件复制/导入到新工作区，形成新的 imported snapshot。

不能让新流程隐式读取旧目录，也不能继续把新结果写入旧目录。

新流程默认结果目录：

```text
${DATA_CONSTRUCTION_RUN_ROOT:-${HOLOSOMA_REPO}/workspace/v3/data_construction_v3_runs}/<run_id>/
```

代码和文档目录：

建议新增脚本入口：

```text
workspace/core4d/scripts/data_construction_v3/
```

这些脚本只做 wrapper、检查、manifest 管理和状态管理，不复制 OmniRetarget 或 SPIDER 的核心算法。

代码进入 git 管理；运行结果目录、可视化、NPZ/MP4/CSV 等大产物不进 git。

## 3.1 `workspace/core4d/data_preprocess` 的定位

`workspace/core4d/data_preprocess` 不是单纯文档目录，它包含 E077-E080 之后形成的历史可运行 Stage2b 入口：

```text
workspace/core4d/data_preprocess/
  README.md
  SCENE_TEMPLATE_GUIDE.md
  pipeline.sh
  generate_core4d_contact_masks.py
  infer_holosoma_trim_window.py
  write_trim_window.py
  create_spider_scene_from_template.py
  verify_processed_case.py
  cases_*.tsv
```

新 v3 流程对它的定位：

| 文件/功能 | v3 定位 |
|---|---|
| `README.md` | 历史说明，不作为新用户入口 |
| `SCENE_TEMPLATE_GUIDE.md` | 可迁移到 v3 template policy，但必须补 E103 后的 clean-base/audit 规则 |
| `pipeline.sh` | 历史 Stage2b orchestrator，作为迁移参考，不作为 canonical 入口 |
| `generate_core4d_contact_masks.py` | contact proxy 逻辑可迁移，但必须支持 3cm/5cm 命名和 manifest |
| `infer_holosoma_trim_window.py` / `write_trim_window.py` | 可迁移，作为 trim window provenance 工具 |
| `create_spider_scene_from_template.py` | 可迁移为 target scene patcher；不是 source template 生成器 |
| `verify_processed_case.py` | 可迁移为基础 verify，但必须补 scene inertial/collision/contact-site audit |
| `cases_*.tsv` | legacy case list；不能作为 v3 自动候选源 |

和 v3 的重叠：

- `pipeline.sh` 覆盖 v3 的 S3 主链路：convert -> retarget -> trim -> contact mask -> target scene -> `spider/process_datasets/core4d.py` -> `scene_act` -> verify。
- `SCENE_TEMPLATE_GUIDE.md` 覆盖 v3 的 S2 部分 template 规则。
- `generate_core4d_contact_masks.py` 覆盖 v3 S1/S3 中 contact mask 的部分逻辑。

和 v3 的冲突/不足：

1. `pipeline.sh` 默认写到 `workspace/core4d/results/data_preprocess`，不符合 v3 独立 run root 和状态 registry 口径。
2. `pipeline.sh` 默认 `REPLACE_WRIST_WITH_FINGERTIP=1`，不符合 medium box 之后“不默认启用 fingertip replacement、必须显式 variant”的口径。
3. `generate_core4d_contact_masks.py` 虽然有 `--threshold` 参数，但输出文件名/key 仍固定为 `3cm`，不满足 v3 3cm/5cm 双阈值并行输出要求。
4. `create_spider_scene_from_template.py` 只复制 `source_scene_task` 并 patch target object pose，不会重建 mesh/collision/mass/inertia；因此它不能解决 source scene missing，只能在 S2 clean template 已存在后用于 S3 target scene。
5. `verify_processed_case.py` 只检查 basic shape、MuJoCo load 和 qpos match，不包含 E103 后必须的 robot inertial、object collision extents、污染质量/惯量、hand contact sites 等硬审计。
6. `cases_*.tsv` 中包含 bucket/desk 等非 box case；v3 对非 box 要求人审，不能沿用这些 TSV 作为自动执行列表。
7. 该目录没有 `case_state_registry`、run manifest、config hash、stage manifest，也没有 S4 target gate、S5 handoff、S6 downstream 状态管理。

处理原则：

- 不删除、不改写 `workspace/core4d/data_preprocess`，避免破坏历史复现。
- v3 文档中明确标记它为 legacy Stage2b implementation。
- 后续把可复用代码迁移/包装到 `workspace/core4d/scripts/data_construction_v3/`，迁移后新代码才是长期入口。
- 新流程不得隐式调用 `workspace/core4d/data_preprocess/pipeline.sh`；如短期必须调用，要通过 v3 wrapper 显式记录 legacy dependency，并写入 run manifest。

## 3.2 E098-E101 的纳入方式

E098-E101 不是同一种层级的检查。v3 需要分清：

- **E098 是全路线基础 contract**：无论 `target_variant_id=ref_fk / adaptive / fingertip_aware`，都必须继承 E098 的 face helper、`contact_pos` provenance、replay gate、hard-block 规则。
- **E099-E101 是 fingertip-aware target route 的 contract**：它们主要服务于 raw fingertip vote、fingertip-aware external target 生成、target gap/active mask 和 route-level CEM guard。默认 `ref_fk` route 不应被强制要求跑完 E099-E101。

| 实验 | 必须纳入 v3 的结论 | v3 落点 |
|---|---|---|
| E098 | `face_label` 必须使用全 3D 六面 argmax，不能再用 xy-only；`anchor_face_review=true` 且未 refit 必须 hard block；`contact_pos` 是 G1 FK palm site，不是 raw mocap fingertip；replay gate 包含 pelvis_end_z / pelvis_tilt_end / lie_on_box | **全 route 必选**：S1/S4/S5；`TargetGate`；failure taxonomy；manifest diagnostic 字段 |
| E099 | raw 5 指尖信息流进入 audit；fingertip vote 与 palm vote 可不同；CORE4D box family 不能默认使用 world-up face 投影；raw contact 3D 可视化是 release evidence | **fingertip-aware route 必选**；其它 route 可作为 diagnostic，不作为 hard gate |
| E100 | fingertip-aware external contact target 可作为 target variant；target 生成必须记录 vote face、quat audit、active mask、target gap；守门 case face_changed=False 时 target swap 应为 0 | **fingertip-aware route 必选**；定义 `target_variant_id=fingertip_aware` 的 target provenance |
| E101 | E100 fingertip target 不退化 box004 guard，但不能救 box021 D003；box021 D003 失败应归为 posture/upright/motion-level binding 等下游失败，不应继续作为“target face 修复可解决”的数据正例；torch compile 需要可关闭 | **fingertip-aware route 的 route-level evidence/negative prior**；S6 downstream evidence；failure taxonomy |

具体 contract：

1. `face_utils` 逻辑必须进入 v3 公共几何工具，所有 face label / face projection / support-face 判定统一调用，禁止各脚本各自实现。
2. v3 中任何使用 `contact_pos` 的指标都必须标注来源：`fk_palm_site` / `raw_fingertip` / `external_target`，不能再把 FK palm 当 raw contact。
3. raw fingertip vote、palm vote、face_changed、quat audit 是 `target_variant_id=fingertip_aware` 的必填字段；对 `ref_fk` / `adaptive` route 可为空或作为 diagnostic 缓存。
4. world-up projection 不能作为 CORE4D box 的 fingertip/adaptive target 默认路径；只有通过 quat audit 明确允许时才能启用。`ref_fk` route 不做 external face projection，不受这条硬约束影响。
5. E098 replay gate 中 `pelvis_end_z` 和 `lie_on_box` 可作为硬 gate；`pelvis_tilt_end` 在后续 E107 口径中降为 diagnostic，不再单独作为硬失败归因。v3 文档必须写清楚这个版本差异。
6. E100 external target NPZ 的 `active` 信息要进入 v3 schema。若下游 CEM/RL 暂不读 active，也必须在 handoff manifest 中保留；该要求只对 external target route 生效。
7. E101 的负结论要进入 registry seed：box021 D003 的 `fingertip_aware` route 不能因为 face target 修复而自动列为 positive；box004 guard 可作为 `fingertip_aware` route 不退化的证据。

## 4. 总体目标

### 4.1 必须做到

1. 任意机器可通过本地配置指定 repo、原始数据和结果目录。
2. 从原始 CORE4D mocap 可重新生成 inventory、raw-contact 候选、template、Stage2b 输出、target gate 结果和 handoff manifest。
3. 所有阶段都有明确输入、输出、manifest schema、失败分类和可重跑命令。
4. template 缺失不再被当作数据失败；必须进入 template backlog 并补 template。
5. E103 暴露的 robot inertial / scene 污染类问题进入硬审计，不能再次混入下游。
6. `results/` 不进 git 的前提下，仍能靠 config、manifest、git sha、环境检查和命令记录重建结果。
7. 建立数据状态管理机制，记录每个 case 在 raw contact、template、Stage2b、target gate、CEM、RL 等阶段的状态，支持跳过已完成阶段和从头重建两种使用方式。
8. 建立 retarget algorithm/variant 管理机制，把 OmniRetarget 求解器版本、输入改写参数、trim 参数和下游 SPIDER 生成参数全部写入 manifest。
9. 代码结构预留过滤器接口和算法接口，后续新增筛选规则或 retarget 版本时不再重复造一套 pipeline。
10. 把 E098 的基础诊断 contract 固化为全 route 可执行 gate/schema；把 E099-E101 固化为 `fingertip_aware` target route 的可选扩展 contract，而不是只保留在实验日志中。

### 4.2 暂不做

1. 不合并 SPIDER 与 OmniRetarget 环境。
2. 不设计跨机器结果同步。
3. 不把 CEM/RL 成败反向定义为数据构建唯一成败。
4. 不自动推广到 chair/table/bucket 等非 box 复杂物体。

## 5. 两种运行模式

### 5.1 full-from-raw

从 `CORE4D_Real` 原始目录开始，完整重建：

```text
raw mocap + object mesh
  -> full inventory
  -> raw contact 3cm/5cm
  -> case state registry
  -> candidate bank
  -> template backlog / template generation
  -> retarget variants
       -> Stage2b
       -> target gate
       -> visual QC
  -> handoff manifest
```

这是最终的标准可复现模式。

### 5.2 resume-from-summary

从新流程已有的 `case_state_registry`、stage manifest 或显式导入的 legacy snapshot 继续：

```text
case_state_registry / imported snapshot
  -> candidate bank
  -> template audit/generation
  -> retarget variants
       -> Stage2b
       -> target gate
       -> visual QC
  -> handoff manifest
```

这个模式用于提高迭代效率，但必须在输出 manifest 中标记 upstream source，不允许伪装成 full-from-raw。

允许导入 legacy 状态，但必须显式执行 import：

```text
legacy data_construction / data_construction_v2
  -> imported snapshot
  -> case_state_registry
```

导入后的 registry 是新流程自己的状态文件；后续运行不再回写 legacy 目录。

## 6. 管线阶段

### S0: 配置与环境检查

输入：

- `SPIDER_REPO`
- `HOLOSOMA_REPO`
- `CORE4D_RAW_ROOT`
- `DATA_CONSTRUCTION_RUN_ROOT`
- `PYTHON_BIN`
- OmniRetarget/hsretargeting 相关环境变量
- GPU / MuJoCo / EGL / ffmpeg 配置

输出：

- `environment_check.json`
- `environment_check.md`
- 当前 git sha、dirty 状态、Python executable、关键 package version

成功标准：

- repo 路径存在；
- raw data 路径存在；
- MuJoCo 能 load 一个 guard scene；
- ffmpeg 可用；
- OmniRetarget 命令可定位；
- 结果根目录可写。

### S0b: 数据状态注册表初始化

输入：

- 新流程已有 `case_state_registry`；
- 可选：人工确认的已知 case 状态；
- 可选：显式导入的 legacy snapshot。

输出：

- `case_state_registry.tsv`
- `case_state_registry.json`
- `case_state_summary.md`

状态注册表至少覆盖阶段：

| 阶段 | 示例状态 |
|---|---|
| raw_inventory | `not_seen` / `present` / `invalid_raw` |
| raw_contact_3cm | `pass` / `reject` / `not_run` |
| raw_contact_5cm | `pass` / `reject` / `not_run` |
| template | `clean` / `backlog` / `audit_fail` / `manual_review_required` |
| stage2b_by_variant | `pass` / `omniretarget_infeasible` / `preprocess_fail` / `not_run` |
| target_gate_by_variant | `pass` / `review` / `reject` / `not_run` |
| visual_qc_by_variant | `pass` / `review` / `reject` / `not_run` |
| cem_by_variant | `pass` / `fail` / `not_run` / `not_required` |
| rl_by_variant | `pass` / `fail` / `not_run` / `not_required` |

关键规则：

- registry 可以写入已经确定的历史 case 状态，用于提高效率；
- registry 中每个非 `not_run` 状态必须有 evidence path、source、日期和 version；
- 从头跑时可以选择忽略 registry 的 skip 建议，但仍应在结束后更新 registry；
- resume 时只允许跳过 evidence 完整且 schema/version 兼容的阶段；
- legacy 状态必须标记 `source_type=legacy_import`，不能伪装成 v3 直接产物。
- S3 之后的状态必须带 `retarget_variant_id` 和 `target_variant_id`；同一个 case 在不同 retarget variant / target route 组合下可以有不同状态。

### S1: 原始数据 inventory 与 raw contact mining

输入：

- `CORE4D_Real/human_object_motions`
- `CORE4D_Real/object_models`

输出：

- 全量 case-person inventory；
- object mesh extents / volume；
- raw contact `3cm` 与 `5cm` 两档结果；
- fingertip preflight 统计；
- raw fingertip face vote / palm face vote / face_changed；
- object quat audit / `disable_world_up`；
- reject/pass reason 分布；
- raw contact 可视化。

关键规则：

- 3cm 和 5cm 必须分别输出候选表；
- raw contact 只是 contact proxy，不是人工真值；
- 阈值、帧数、person/hand 映射必须写入 manifest；
- 不能因为 source scene missing 而跳过 raw mining。
- 所有 face 相关计算必须使用 E098 后的全 3D 六面 face helper。
- 对 CORE4D box，world-up face 投影默认关闭；只有 quat audit 明确允许时才启用。
- raw fingertip face vote / palm vote / quat audit 对 `fingertip_aware` route 为必填；对 `ref_fk` route 不是进入 CEM/RL 的硬门槛。

### S2: Source scene template backlog、生成与审计

输入：

- candidate manifest；
- object mesh；
- clean base template；
- E103 后的 source-template policy。

输出：

- template backlog；
- 新建或复用的 source template；
- scene inertial audit；
- collision geometry audit；
- MuJoCo load audit；
- source template visual review sheet / mp4；
- `task_info.json` provenance。

关键规则：

- box 类物体按 E103 clean base 流程生成；
- 非 box 物体必须人工审查；
- canonical source template 只保留 clean source scene 责任；
- target object pose 由 trimmed qpos 第一帧 patch；
- 旧 runtime artifact 不能混入 source template。

硬失败：

- MuJoCo load fail；
- robot inertial 不匹配 clean base；
- hand contact sites 缺失；
- object collision extents 与 mesh AABB 不一致；
- template 里残留已知污染质量/惯量。

### S3: Stage2b OmniRetarget + SPIDER preprocess

S3 是第一个强制按 retarget variant 与 target route 双轴分叉的阶段。S0-S2 的 raw inventory、raw contact、source template 可以共用；S3 之后的 converted NPZ、retargeted NPZ、trimmed NPZ、SPIDER target task、target gate、visual QC、CEM/RL 状态都必须带 `retarget_variant_id` 和 `target_variant_id`。

固定流水线：

```text
convert_core4d_to_omniretarget.py
  -> robot_retarget.py
  -> trim_no_contact.py
  -> generate_core4d_contact_masks.py
  -> create_spider_scene_from_template.py
  -> spider/process_datasets/core4d.py
  -> generate scene_act
  -> verify
```

每条 case 必须产出：

- converted NPZ；
- OmniRetarget 输出 NPZ；
- trimmed NPZ；
- contact mask；
- target `scene.xml`；
- `trajectory_kinematic.npz`；
- `scene_act.xml`；
- verify summary；
- run log；
- stage manifest row。

每条 Stage2b row 必须记录：

- `retarget_variant_id`；
- `target_variant_id`；
- `solver_family`；
- `solver_version`；
- `solver_git_sha`；
- `solver_repo_path`；
- `converter_version`；
- `converter_git_sha`；
- `replace_wrist_with_fingertip`；
- `include_fingertip_centers`；
- `trim_policy`；
- `retarget_config_json`；
- `variant_parent_id`；
- `branch_stage=conversion`。

关键规则：

- OmniRetarget 原始输出位置必须记录，不能只记录转成 SPIDER 输入后的文件；
- CVXPY infeasible 是 Stage2b 失败，不等同 raw data 失败；
- source scene 缺失不允许在这里静默 skip，必须回到 S2。
- `REPLACE_WRIST_WITH_FINGERTIP` 不是 OmniRetarget 求解器内部参数，而是 conversion/input rewrite 参数：它把 SMPL-X wrist joint `20/21` 替换为左右手 5 个 fingertip 的均值。它会改变 OmniRetarget 的 wrist/eef source target，因此必须作为 retarget variant 管理。

初始 retarget variants：

| variant id | 含义 | solver | conversion/input semantics | 说明 |
|---|---|---|---|---|
| `omnirt_original` | 原始算法 | Holosoma `9c238cf80f531c0e65d818348c3c1a5cc2764f5b` Initial public release | 原始 conversion 语义；当前 CORE4D converter 不在该提交内 | 用于回溯或严格对照；执行需单独 checkout/adapter |
| `omnirt_v1` | v1 优化算法 | 当前 v1/优化后的 OmniRetarget | wrist 不替换为 fingertip | medium box 默认候选路线 |
| `omnirt_v1_fingertip_replacement` | v1 + fingertip replacement | 当前 v1/优化后的 OmniRetarget | `replace_wrist_with_fingertip=true` | reach-risk / 大物体 guard 或显式 ablation |

注意：retarget variant 只管理 OmniRetarget/input rewrite 轴；`target_variant_id=ref_fk/adaptive/fingertip_aware` 是另一个独立轴。默认 target route 是 `ref_fk`。只有显式选择 `fingertip_aware` target route 时，才要求 E099-E101 route diagnostics 通过。

这三个只是初始内置 variant；后续新增 top-face rewrite、contact preservation、不同 trim policy 等，都按同一机制注册为新 variant，而不是新建一套 pipeline。

### S3b: retarget variant 注册表

新增独立注册表：

```text
retarget_variant_registry.tsv
retarget_variant_registry.json
```

核心字段：

| 字段 | 说明 |
|---|---|
| `retarget_variant_id` | 稳定唯一 id |
| `display_name` | 人读名称 |
| `solver_family` | `omniretarget` / future algorithm |
| `solver_version` | 原始 / v1 / future |
| `solver_repo_path` | 实际执行 repo |
| `solver_git_sha` | 执行时 git sha |
| `solver_dirty` | 是否 dirty |
| `converter_script` | conversion 脚本路径 |
| `converter_git_sha` | conversion 脚本所在 repo sha |
| `params_json` | 所有显式参数 |
| `input_rewrite_policy` | wrist / fingertip / topface / future |
| `branch_from_stage` | 通常是 `conversion` |
| `created_by` / `created_at` | provenance |

variant registry 是数据状态的一部分。`case_state_registry` 不只按 case 记录状态，还要按 `(case_id, retarget_variant_id, target_variant_id)` 记录 S3 之后的状态。

### S4: Target gate 与 visual QC

输入：

- Stage2b 输出；
- target scene；
- trajectory；
- contact masks；
- source template audit。

输出：

- target gate summary；
- replay metrics；
- OmniRetarget 可视化；
- MuJoCo replay 可视化；
- raw fingertip / palm / target overlay（`fingertip_aware` 必选，其它 route 可选）；
- object-local hand/support overlay；
- external target gap summary（external target route 必选）；
- visual QC manifest；
- release checklist。

硬 gate 建议：

- scene/source audit clean；
- target qpos/scene qpos layout 一致；
- object pose patch 正确；
- inside/object penetration 不超阈值；
- lower-body/object collision 不超阈值；
- contact mask coverage 合理；
- replay 无明显趴箱/穿箱/错物体。
- external target 若启用，必须通过 shape/hash/active-mask/target-gap 检查。

diagnostic only：

- `pelvis_tilt_end` 等已确认不应作为硬失败归因的指标，只保留诊断列。
- `contact_pos` 派生指标必须标注其来源是 FK palm site，不能命名为 raw contact。

### S5: Candidate bank 与 handoff manifest

输出标准候选库：

```text
PASS
REVIEW
REJECT_RAW_CONTACT
REJECT_RAW_FINGERTIP_MISMATCH  # only for fingertip-aware route
REJECT_TEMPLATE_BACKLOG
REJECT_TEMPLATE_AUDIT
REJECT_OMNIRETARGET
REJECT_TARGET_GATE
REJECT_VISUAL_QC
QUARANTINE_LEGACY_TEMPLATE_BUG
DOWNSTREAM_POSTURE_FAIL
DOWNSTREAM_MOTION_BINDING_FAIL
```

handoff manifest 必须包含：

- case id；
- object/person/date/seq；
- source scene；
- target scene；
- trajectory path；
- scene_act path；
- contact mask path；
- raw contact 阈值档位；
- template audit id；
- Stage2b run id；
- target gate decision；
- visual QC decision；
- git sha；
- config hash。

同时更新 `case_state_registry`：

- raw contact 失败的 case 保留为可解释 reject，不再重复进入 Stage2b；
- template backlog 的 case 等 template 补齐后可重新激活；
- Stage2b pass 但 CEM 未跑的 case 保持 `cem=not_run`；
- CEM pass 但 RL 未跑的 case 保持 `rl=not_run`；
- 全链路 pass 的 case 可作为 downstream positive，但仍保留各阶段 evidence。

### S6: CEM / RL 下游入口

数据构建只提供 clean handoff，不直接把 RL 成败当作上游数据唯一标签。

下游输入：

- clean target task；
- clean trajectory；
- clean `scene_act.xml`；
- contact mask；
- override config；
- handoff manifest row。

下游输出应回写为 downstream evidence：

- CEM metrics；
- lower-body contact metrics；
- videos；
- RL train/eval results；
- 是否推荐进入 RL-ready set。

## 7. 文档交付

拟新增：

```text
workspace/core4d/docs/data_construction_v3/
  README.md
  00_environment.md
  01_data_layout.md
  02_pipeline_stages.md
  03_manifest_schema.md
  04_scene_template_policy.md
  05_reproducibility.md
  06_failure_taxonomy.md
  07_troubleshooting.md
  08_retarget_variants.md
  09_extension_interfaces.md
  10_diagnostic_contracts.md
  11_legacy_migration.md
```

各文件职责：

| 文件 | 内容 |
|---|---|
| `README.md` | 总入口，说明两种模式、最小运行路径和常用命令 |
| `00_environment.md` | SPIDER 环境、OmniRetarget 环境、MuJoCo/EGL/ffmpeg、GPU 检查 |
| `01_data_layout.md` | raw data、repo、workspace、results、manifest 的目录约定 |
| `02_pipeline_stages.md` | S0-S6 阶段的输入、输出、命令、失败处理 |
| `03_manifest_schema.md` | inventory、raw contact、template、Stage2b、gate、handoff 字段定义 |
| `04_scene_template_policy.md` | E103 后 clean template policy，box 自动/非 box 人工审查边界 |
| `05_reproducibility.md` | git sha、config hash、环境检查、checksum、重跑策略 |
| `06_failure_taxonomy.md` | 失败模式分类、硬 gate 与 diagnostic 指标区别 |
| `07_troubleshooting.md` | 常见错误：missing raw path、CVXPY infeasible、scene load fail、qvel missing、污染 inertial 等 |
| `08_retarget_variants.md` | OmniRetarget 算法版本、input rewrite 参数、variant 分叉和对照策略 |
| `09_extension_interfaces.md` | 过滤器接口、retarget 算法接口、gate 接口、visualizer 接口 |
| `10_diagnostic_contracts.md` | E098 全局 contract 与 E099-E101 `fingertip_aware` route contract 如何进入 v3 |
| `11_legacy_migration.md` | 旧目录、旧脚本和 imported snapshot 的迁移边界 |

## 8. 脚本交付

拟新增：

```text
workspace/core4d/scripts/data_construction_v3/
  README.md
  lib/
  orchestration/
  stages/s0_environment/
  stages/s1_raw_contact/
  stages/s2_templates/
  stages/s3_retarget/
  stages/s4_gate_visual_qc/
  stages/s5_handoff/
  stages/s6_downstream/
  state/
  migration/
  qa/
```

第一阶段先做 wrapper 和检查，不重写成熟算法：

- 复用现有 E091/E102/E103/E104/E106/E107 中已验证的逻辑；
- 把硬编码路径改为参数或环境变量；
- 每个脚本输出 machine-readable JSON/TSV 和人读 summary；
- 每个阶段都能 dry-run。
- 从 legacy 目录迁移来的代码进入新脚本目录后再使用；旧目录脚本不作为长期入口。

接口预留：

| 接口 | 职责 | 最小输入 | 最小输出 |
|---|---|---|---|
| `CandidateFilter` | 新增候选过滤规则 | inventory/raw-contact/template rows | pass/reject/reason rows |
| `RetargetAdapter` | 新增 retarget 算法或 OmniRetarget variant | case row + source template + variant config | converted/retargeted/trimmed paths + status |
| `TemplateBuilder` | 新增物体 template 构造策略 | object mesh + clean base + policy | source scene + audit |
| `TargetGate` | 新增几何/动力学前 gate | target scene + trajectory + masks | pass/review/reject + metrics |
| `Visualizer` | 新增可视化产物 | stage output + metrics | png/mp4/manifest |

这些接口第一版可以是 Python dataclass/protocol + CLI registry，不需要过度抽象；关键是参数和输出 schema 稳定。

## 9. Manifest 与 provenance

每次运行至少生成：

```text
run_manifest.json
config_resolved.json
environment_check.json
git_state.json
case_state_registry.tsv
case_state_registry.json
retarget_variant_registry.tsv
retarget_variant_registry.json
stage_manifest_*.tsv
stage_manifest_*.json
```

其中 `run_manifest.json` 记录：

- run id；
- start/end time；
- mode: `full-from-raw` 或 `resume-from-summary`；
- config path；
- raw data root；
- output root；
- spider git sha；
- holosoma git sha；
- command line；
- stage decisions；
- known warnings。

`case_state_registry` 记录每个 case 的最新状态和 evidence：

| 字段 | 说明 |
|---|---|
| `case_id` | 规范化 case id |
| `object_key` / `date` / `seq` / `person` | CORE4D 身份信息 |
| `retarget_variant_id` | S3 之后必填；S0-S2 可为空或 `shared` |
| `target_variant_id` | S3 之后必填；默认 `ref_fk`，可选 `adaptive` / `fingertip_aware` / future |
| `raw_contact_3cm_status` / `raw_contact_5cm_status` | 两档 raw contact 状态 |
| `fingertip_vote_status` | raw fingertip vote 是否可用；`fingertip_aware` 必填，其它 route 可空 |
| `palm_vote_status` | palm/FK vote 是否可用；`fingertip_aware` 必填，其它 route 可空 |
| `face_changed_l` / `face_changed_r` | palm face 与 fingertip face 是否不同；`fingertip_aware` 必填 |
| `disable_world_up` | quat audit 后是否禁用 world-up 投影；external face projection route 必填 |
| `template_status` | source template 状态 |
| `stage2b_status` | OmniRetarget/SPIDER preprocess 状态 |
| `target_gate_status` | target gate 状态 |
| `visual_qc_status` | 可视化审查状态 |
| `target_variant_id` | external target / ref_fk / adaptive / future |
| `target_active_mask_status` | external target active mask 是否存在并被下游读取；external target route 必填 |
| `cem_status` | CEM 状态 |
| `rl_status` | RL 状态 |
| `current_decision` | 当前整体决策 |
| `evidence_root` | 证据目录 |
| `source_type` | `v3_run` / `manual_seed` / `legacy_import` |
| `source_ref` | run id、legacy path 或人工记录 |
| `updated_at` | 更新时间 |

结果目录不进 git，但上述 manifest 可以作为复现索引。

## 10. 实施顺序

### Phase A: 文档与 contract 先行

1. 创建 `docs/data_construction_v3/`。
2. 写 `README.md`、`01_data_layout.md`、`02_pipeline_stages.md`。
3. 写 `04_scene_template_policy.md`，把 E103 经验固化。
4. 写 `06_failure_taxonomy.md`，统一失败模式命名。
5. 写 `03_manifest_schema.md`，定义 `case_state_registry` 和各 stage manifest。
6. 写 `11_legacy_migration.md`，说明 `workspace/core4d/data_preprocess`、Holosoma `data_construction`、Holosoma `data_construction_v2` 的 legacy 定位和迁移边界。
7. 写 `08_retarget_variants.md`，把 `omnirt_original`、`omnirt_v1`、`omnirt_v1_fingertip_replacement` 三个初始 variant 和参数 schema 固化。
8. 写 `09_extension_interfaces.md`，定义候选过滤、retarget adapter、template builder、gate、visualizer 的扩展接口。
9. 写 `10_diagnostic_contracts.md`，逐条映射 E098 的全局检查，以及 E099-E101 对 `fingertip_aware` route 的代码/结论到 v3 schema 和 gate。

验收：

- 新人能读文档知道从哪里开始；
- 能区分 source template、target scene、scene_act；
- 能区分数据失败、template backlog、OmniRetarget 失败、下游 CEM/RL 失败。

### Phase B: 配置和环境检查

1. 新增 `check_environment.py`。
2. 新增 `init_workspace.sh`。
3. 新增 `register_retarget_variant.py` 的最小版本。
4. 新增 `update_case_state_registry.py` 的最小版本。
5. 输出 `environment_check.{json,md}`、空/seed registry 和初始 retarget variant registry。

验收：

- 在缺 raw path、缺 ffmpeg、MuJoCo load fail、OmniRetarget env 缺失时给出明确错误。

### Phase C: Stage wrapper 固化

1. 从 `workspace/core4d/data_preprocess` 和其它 legacy 脚本中迁移必要逻辑到 `workspace/core4d/scripts/data_construction_v3/`。
2. 把 `pipeline.sh` 拆成 v3 wrapper 阶段，不再作为长期入口。
3. 把 raw contact 3cm/5cm 输出固定为两份候选，修复旧脚本阈值可变但文件名/key 固定 `3cm` 的语义问题。
4. 把 template backlog 逻辑固定为不可跳过。
5. 把 Stage2b wrapper 改为完全参数化，并把 `REPLACE_WRIST_WITH_FINGERTIP` 改成显式 variant，不设隐式 production 默认。
6. Stage2b 输出按 `retarget_variant_id` × `target_variant_id` 分叉，variant/route 组合之间共享 S0-S2，不能互相覆盖输出。
7. 把 target gate/visual QC 输出统一 manifest。
8. 每个阶段更新 `case_state_registry`。

验收：

- 不再依赖 `/home/ubuntu/...` 硬编码；
- 每个阶段能单独重跑；
- 每个阶段失败都能进入标准 failure taxonomy。
- 新流程不写入 `workspace/v3/data_construction` 或 `workspace/v3/data_construction_v2`。

### Phase D: 端到端 smoke

选择小样本验证：

- 一个已知 positive：box004；
- 一个 E103 后 clean 的 box026；
- 一个 E107 clean Box021 case。

验收：

- `full-from-raw` 或尽可能接近 raw 的模式能生成 handoff manifest；
- `resume-from-summary` 能从已有 manifest 继续；
- 输出路径、本地结果、可视化、失败分类都符合文档。
- registry 能正确表示：raw contact fail、Stage2b pass 但 CEM 未跑、CEM pass 但 RL 未跑、全链路 pass 等不同状态。

## 11. 风险与处理

| 风险 | 处理 |
|---|---|
| 历史脚本散落在旧目录和 E091/E102/E103 等目录 | 迁移到 `data_construction_v3` 脚本目录后再作为长期入口 |
| `workspace/core4d/data_preprocess` 仍是可运行旧入口 | 保留历史目录；v3 不隐式调用，必要逻辑迁移/包装后再使用 |
| 旧 contact mask 支持 `--threshold` 但文件名/key 固定 `3cm` | v3 contact mask 输出必须按阈值命名，如 `raw_contact_mask_3cm` 与 `raw_contact_mask_5cm` |
| 旧 `pipeline.sh` 默认 fingertip replacement | v3 把 fingertip replacement 作为显式 variant，不能作为中等 box production 默认 |
| OmniRetarget 参数没有版本化导致结果混用 | 每个 Stage2b 结果必须绑定 `retarget_variant_id`、`target_variant_id`、git sha 和完整 `params_json` |
| 新过滤规则/新算法反复复制 pipeline | 通过 `CandidateFilter` / `RetargetAdapter` / `TargetGate` 等接口接入，核心 orchestrator 不复制 |
| raw contact 旧逻辑依赖历史 summary | full-from-raw 必须从 raw 重建；resume 只能读 v3 registry 或显式导入 snapshot |
| template 自动生成再次污染 | E103 audit 变成硬 gate，非 box 必须人工审查 |
| 结果不进 git 导致不可追踪 | manifest、config、git sha、环境检查必须进结果目录 |
| LLM/subagent review 不可复现 | 只作为 release checklist，机器 gate 保持独立 |
| 下游 CEM 失败被误判成数据失败 | 数据构建阶段与下游阶段分离，CEM/RL 只作为 downstream evidence |
| registry 状态过期导致错误跳过 | 每个状态带 schema/version/evidence；版本不兼容时必须重跑对应阶段 |

## 12. 当前不确定项

目前主要口径已确认。剩余只需要在实施时按实际代码确认：

1. D001/D002 raw-contact 的最干净入口脚本到底复用哪一个，还是需要从 E104 多阈值脚本整理成 canonical 入口。
2. `omnirt_original` 已 pin 到 Holosoma initial public release，但执行 strict original 对照仍需要单独 checkout/adapter；新增 variant 的 exact solver repo/commit/path 仍需逐项确认，不能靠名字推断。
3. 首批 `case_state_registry` 的 manual seed 应写入哪些已确定 case，需要根据 E091-E107 的最终可信结论逐条整理。
4. 对非 box 物体的人工审查表单字段需要等第一批非 box case 出现后再补充。

## 13. 下一步

确认本计划后，先执行 Phase A：创建 `workspace/core4d/docs/data_construction_v3/` 并写中文文档骨架；随后做 Phase B 的环境检查脚本和状态注册表最小版本。实现过程中不移动、不删除、不改写 `workspace/v3/data_construction` 和 `workspace/v3/data_construction_v2`，新结果写入独立 run root，不上传 `results/`。
