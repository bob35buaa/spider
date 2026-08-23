# E203 — CORE4D v2 人体动作数据 orig 重定向全流程（box + bucket）

## Context（为什么做）

此前所有 CORE4D 重定向数据都源自 `CORE4D_Real`（v1）的 `person{N}_poses.npz`。CORE4D 官方释放了 `CORE4D_Real_human_object_motions_v2`（v2）——**同一批真实捕捉、同帧率/同长度、但 SMPLX 人体拟合被重新优化**（同 case 核实：帧数一致、betas 一致、位姿 transl 差≈0.6cm/关节差≈0.9cm）。v2 人体动作更干净，值得用现有 `data-construction-v3` 管线跑一版重定向数据，为下游 CEM/RL 提供更高质量 source。

本实验 **E203**：用 v2 人体动作，对 **box001/box004/box021/box023/box024/box026 + bucket 类（排除 bucket006/008）** 走完整 v3 管线 S0–S6（含 full CEM，orig，不做 object augmentation），产出与 v1 **物理隔离** 的一版数据（`dataset_name=core4d_v2`）。

已确认的关键事实：
- v2 `person_{N}/result.npz` 的键与 v1 `person{N}_poses.npz["arr_0"]` **完全一致**（vertices/joints(T,127,3)/betas/…/body_pose/hand_pose），T-major。
- v2 **不含物体位姿**；同 case 帧数与 v1 对齐（抽 40/40 通过），故物体位姿 + `object_metadata.json` 逐帧复用 v1。
- 管线 raw loader（`s1_raw_contact/build_inventory.py`、`run_raw_contact.py`）与 legacy `data_preprocess/pipeline.sh` 硬编码 v1 布局（`human_object_motions/*/*` + `{person}_poses.npz` + `smooth_objposes.npy` + `object_metadata.json`）与 `dataset_name=core4d`。
- ⚠️ 管线里 `omnirt_v2` 已是**求解器 rescue 变体**（Phase4 松弛），与"v2 人体动作数据"是两回事，**绝不可混用该 id**；E203 用 `retarget_variant_id=omnirt_v1`（默认，不替换 wrist）。

## 范围（用户确认）

- **物体**：box001、box004、box021、box023、box024、box026 + bucket 类（排除无资产/无审查模板的 **bucket006、bucket008**）。
- **数据版本**：v2 人体动作 + 复用 v1 物体位姿。
- **只跑 orig**：不做 OmniRetarget object translation/rotation augmentation。
- **深度**：S0–S6，含 **full CEM**。CEM 配置 **复用 E199**（rubber_hull 手部碰撞 + PRG + contact-aligned proxy，见 `workspace/core4d/plan/228_E199_*`）。
- **隔离**：新 `dataset_name=core4d_v2`，输出树与 v1 完全并行。
- **CEM 优先级**：`box001, box004, box024` → `box021, box023` → `bucket`。

**In-scope 规模（v2 已覆盖的 person-sequences，共 605）**：
| 组 | 物体 | seq 数 |
|---|---|---|
| P1 | box001(98)+box004(6)+box024(46) | 150 |
| P2 | box021(50)+box023(46) | 96 |
| P3 | bucket001/003/004/005/007/009/010 | 315 |
| P4 | box026 | 44 |

> ⚠️ **待确认**：用户 CEM 优先级里列了 box021/box023 但**未提 box026**（box026 在原始物体清单里）。本计划默认 box026 **仍在范围内**、CEM 排最后（P4）。若应剔除 box026，请说明。

## 隔离设计（core4d_v2 数据集）

`core4d` 在 5 处硬编码。策略：**共享只读资产/模板用 symlink，输出用 dataset 参数**，对 v1 完全向后兼容（参数默认 `core4d`）。

1. **新建并行输出树** `example_datasets/processed/core4d_v2/`：
   - `assets` → symlink → `../core4d/assets`（物体几何相同，共享）
   - `unitree_g1/dual_humanoid_object` → symlink → `../../core4d/unitree_g1/dual_humanoid_object`（**source scene 模板共享**，与 source 版本无关）
   - `unitree_g1/humanoid_object/` → **真实空目录**（v2 target 任务落这里，`dcv3_omnirt_v1_ref_fk_<obj>_<date>_<seq>_pN`，与 v1 同名任务因树不同而不冲突）
2. **v2 raw 物化根** `${V2_RAW_ROOT}`（新脚本，见下）：把 v2 `result.npz` 重打包为 v1 布局的 `person{N}_poses.npz`（`np.savez(arr_0=<dict>)`），并 symlink v1 的 `smooth_objposes.npy`/`object_metadata.json`；`object_models` → symlink v1。仅物化 in-scope cases。

## 代码改动（隔离、可逆、须 git 追踪；默认值保持 v1 行为）

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/data_construction_v3/stages/s3_retarget/run_stage2b.py` | 加 `--spider-dataset`（默认 `core4d`）；`spider_task_dir` 用该值替换硬编码 `core4d`（:271）；通过 env `SPIDER_DATASET` 传给 pipeline.sh 命令（:223-243 command 列表）。 |
| `workspace/core4d/data_preprocess/pipeline.sh` | 读 `SPIDER_DATASET`（默认 core4d）：source_scene 路径（:285）、`core4d.py --dataset-name`（:473）、`create_spider_scene_from_template.py`（:462/481）与 `verify_processed_case.py`（:494）的 dataset/task 路径统一走该变量。 |
| `workspace/core4d/data_preprocess/create_spider_scene_from_template.py`、`verify_processed_case.py` | 若输出路径内含 `core4d` 硬编码，加 `--dataset-name` 参数（默认 core4d）。 |
| `workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/export_cem_overrides.py` | override YAML 内 task/scene 路径引用 `core4d_v2`；文件名保持 `core4d_dcv3_...`（或加 dataset 前缀），确保指向 core4d_v2 任务目录。 |

**新增脚本（不改核心 loader）**：
- `workspace/core4d/scripts/data_construction_v3/migration/materialize_v2_raw_root.py`：输入 in-scope case 清单 + v1/v2 根 → 产出 `${V2_RAW_ROOT}`（含 manifest 记录 source sha/软链）。校验每个 case v2 `result.npz` 键完整、T 与 v1 `smooth_objposes` 一致（不一致的 case 记 skip，不静默）。

> `spider/process_datasets/core4d.py` **无需改**（已有 `--dataset-name`）。核心 raw loader（build_inventory/run_raw_contact）**无需改**——它们读的是物化后的 v1 兼容布局。

## 执行步骤（→ verify）

1. **物化 v2 raw 根** → verify：`${V2_RAW_ROOT}` 下 in-scope cases 都有 `person{1,2}_poses.npz` + 物体软链；抽 3 case 用 v1 converter 加载成功、joints shape=(T,22,3)。
2. **建 core4d_v2 输出树 + symlink** → verify：`ls -l` 确认 assets/dual_humanoid_object 为软链、humanoid_object 为真实空目录。
3. **代码改动 + py_compile + 无参回归** → verify：`find … -name '*.py' | xargs py_compile`；用 `--spider-dataset core4d`（默认）跑 1 个 v1 case dry-run，确认 v1 行为不变（可逆性）。
4. **S0 init + S0b registry**：`init_workspace.sh E203 --run-root workspace/core4d/results/E203 --core4d-raw-root ${V2_RAW_ROOT}`；注册 `omnirt_v1`。
5. **S1 inventory + raw contact 3cm/5cm**（queue 限 in-scope 物体）→ verify：inventory 覆盖 605 seq；raw_contact_pass_3cm 非空。
6. **S2 模板**：source 模板经 symlink 复用 v1 已有 clean（box）/ reviewed（bucket，E177/E202 proxy）→ 用 import/seed 把 in-scope 物体 `template_status` 置 clean/clean_reviewed → verify：`build_or_audit_templates.py` audit 全 pass（MuJoCo load、robot inertial 无 mass=29.632 污染、object mass/inertia）。
7. **S3 retarget**（`--retarget-variant-id omnirt_v1 --target-variant-id ref_fk --spider-dataset core4d_v2 --execute --allow-legacy-stage2b-wrapper`）→ verify：stage2b_manifest 每 row 有 converted/retargeted/trimmed/spider_task_dir/contact_mask；task 落在 `core4d_v2/.../humanoid_object/`。
8. **S4 target gate + visual QC** → verify（硬 gate）：scene/scene_act MuJoCo load、qpos/layout 一致、trimmed↔spider qpos 一致、穿透/趴箱/腿干涉不超阈；生成 replay MP4 供人工/LLM 复核。
9. **S5 handoff + CEM override（core4d_v2 路径）** → verify：handoff_manifest 只含 HANDOFF_* rows；override YAML 指向 core4d_v2 任务。
10. **S6 full CEM（复用 E199 配置，按 P1→P2→P3→P4 优先级）** → 每组产 CEM metrics/MP4/summary；`export_rl_inputs.py` 生成 rl_export_input.tsv → verify：见评估标准。

## 评估标准（量化，含视觉，参考 `.claude/rules/experiment.md` §5）

- **机器 gate（S4）**：每 case 必须过 clean scene/qpos 一致/穿透/趴箱/腿干涉硬 gate；报告 pass 率（分物体 mean/worst）。
- **CEM（S6）**：报告 obj_pos 误差、eef/手穿透、腿穿透、gate 通过率、fall 率的 **mean+std+worst**，分物体、全 605 分布（不 cherry-pick）。
- **v1↔v2 A/B（核心验收）**：对同 case 的 v1-sourced（现有）与 v2-sourced 结果，做**同 clip 并排视频** + 指标对比，判定"v2 人体更干净"是否转化为下游更优/不劣（tracking、接触保真、稳定性）。
- **视觉强制**：用 `/video-frames` 抽 grasp/contact/transition 关键帧，检查穿透/漂浮/抖动/趴箱；reward 曲线单独不作结论。
- **成功判据（预声明）**：in-scope 中 v2-sourced 的 S4 gate pass 率 **≥ v1 同 case 基线**，且 CEM 关键指标（obj_pos、腿穿透、fall）**不劣于 v1**（等价或更优）；A/B 视频无 v2 特有 artifact。

## 快照与可复现（`.claude/rules/experiment.md` §7）

- 训练前对 in-scope 每 case 的 scene.xml/scene_act.xml + JSON meta 做 `scene_snapshot/` 快照 + `manifest.txt`（git HEAD + sha256），存 `workspace/core4d/results/E203/scene_snapshot/`。
- core4d_v2 的 active-case scene XML 按规则 `git add -f`。
- 代码改动一次 commit：`exp(core4d): E203 core4d_v2 dataset 隔离 + v2 raw 物化 + S0-S6 orig`。

## 交付物

- `workspace/core4d/results/E203/`（S0–S6 全阶段产物、registry、CEM evidence、rl_export_input.tsv，不进 git）。
- `example_datasets/processed/core4d_v2/...`（v2 重定向数据）。
- 实验日志 `workspace/core4d/log/NNN_E203_core4d_v2_orig_retarget_results.md` + tracker 行。
- 本计划将复制到 `workspace/core4d/plan/NNN_E203_*.md`（experiment-planning-zh 规范）。

## 风险

- **代码改动波及 legacy pipeline.sh**：必须保证 `--spider-dataset` 默认 core4d 时 v1 行为逐字不变（步骤 3 回归）。
- **算力**：605 seq full CEM 很重；按优先级分批，先 P1 出结论再继续。
- **bucket 模板**：复用 E177/E202 reviewed proxy；若某 bucket 实例的 proxy 缺失需先补审查（bucket006/008 已排除）。
- **box004 样本少**（v2 仅 6 seq，7 case 在 v2 缺失），统计力弱，仅作定性。
