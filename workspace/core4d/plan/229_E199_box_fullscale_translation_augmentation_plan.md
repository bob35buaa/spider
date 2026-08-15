# plan229 · E199 全量：box 类 s6 full-CEM case 平移增强（放量数据生成）

_Core4D · Phase 62 · Run **R286** · 承接 [plan228](228_E199_omniretarget_object_augmentation_full_cem_plan.md) / [log286](../log/286_E199_omniretarget_object_augmentation_results.md) · 2026-08-16 · **状态：计划态**_

## Context / 目标

pilot（plan228/log286，8 物体各 1 case）已**证明上游 OmniRetarget object augmentation 在本管线有效可信**：平移档 obj_pos 误差仅 +19.9%（<25% 阈），接触保持 / 腿穿透 / 12-gate 通过率不劣于甚至优于 orig，零跌倒，视觉无 artifact；**旋转档（±45° yaw）8/8 物体系统性不可达**。

本计划把已验证的**平移增强**放量到 **box 类所有已进入 s6 full CEM 阶段的 case**，为下游 RL 训练批量扩数据（每 case ≈ +3× 平移变体）。

- **只做平移**（trans0/1/2），**不做旋转**（pilot 已证 rot 不可达，放量无意义）。
- **实验号仍 E199**（同一方向的放量阶段，非新方向）；同分支 `experiment/E199-omniretarget-object-augmentation`。
- **已跑完的 case 跳过**（pilot 5 个代表 case 的 orig+trans0/1/2 = 20 条 box run 已完成）。

## 关键决策（用户 2026-08-16 确认）

1. **orig 同条件基线 = 复用现有 A0/PRG full CEM**（不重跑 orig）。87 个 box case 已有 E198 A0 arm（即 E173/PRG full CEM，`E173_{case}_PRG.npz`）作为 orig 基线，eval 直接复用。
   - **代价（诚实报告）**：orig 用 omnirt_v1 retarget，aug 用 omnirt_v2 —— 存在轻微 retarget 变体混淆。**但 pilot 已在完全同条件（orig 也重跑 omnirt_v2）下证明可信度**；放量目标是产数据而非再做一次严格 ablation，故复用即可。eval 报告须显式标注此 confound。
2. **执行范围 = 全量 87 case 一次性排队**，8 卡 priority queue（resume-safe，不抢占），跳过 pilot 已完成。

## 范围：87 个 box s6-full-CEM case（权威 = E198 A0 arm）

「进入 s6 full CEM」的权威定义 = 在 E198 four-arm eval 中拥有 **A0（PRG）arm 完成 rollout** 的 case。逐物体计数（来自 `results/E198/s6_downstream/eval/full_factorial/e198_arm_cache.tsv`，arm=A0）：

| 物体 | s6 full-CEM case | pilot 已跑(跳过) | 本轮新增 case | ×3 trans |
|---|---|---|---|---|
| box001 | 28 | 1 | 27 | 81 |
| box004 | 6 | 1 | 5 | 15 |
| box021 | 28 | 1 | 27 | 81 |
| box023 | 16 | 1 | 15 | 45 |
| box024 | 9 | 1 | 8 | 24 |
| **合计** | **87** | **5** | **82** | **246** |

- **新增 full CEM 上限 = 246 条 aug**（trans0/1/2 × 82 新 case）。实际 ≤246：部分档位上游 IK 不可行 / 参考初始腿-物穿透被 PRG 运行时拦截时跳过该档（同 pilot bucket007 trans2）。
- **orig = 0 条新 run**（复用 87 个 A0/PRG）。
- 预计 8×A100 priority queue ~30–40h（单例 full 1024×32 ≈ 45min）。

## 冻结不变量（与 pilot 逐字段一致）

| 项 | 值 |
|---|---|
| CEM | seed=0, num_samples=1024, max_num_iterations=32, use_torch_compile=false |
| retarget（aug） | omnirt_v2/ref_fk（`OMNIRT_V2_ENV`：relaxation+foot_z+contact_preservation on, slide=1.0, penetration_tol=0.8） |
| arm | E199 rubber_hull + 16 lowerbody-pair PRG（`scene_act_E199_rubberHull_PRG`），reward base = E167A |
| 增强 config | holosoma object_interaction 原生 trans_0/1/2（前/左/右 0.2m），逐字节沿用；**不加载 rot_0/1** |
| 接触掩码 | 每 case 由 v2 `_original` trim 窗重算 1 份 3cm 掩码，3 平移变体复用（固定窗 trim，按 orig trim_start 切片对齐） |
| evaluator | 公共 `eval.core.core_metrics`（rule 13） |
| orig 基线 | 复用 E198 A0 arm（E173 `_PRG.npz`，omnirt_v1）——只读，不重跑 |

## Claims（放量验收）

- **C0 链路放量**：82 个新 box case 各产出 ≤3 平移 SPIDER task（scene + trajectory + `scene_act_E199_rubberHull_PRG` 齐全）；base task yaml + E199 PRG override 全生成，manifest 0 blocker；scene sidecar 全快照（git HEAD + sha256）。
- **C1 case 权威一致**：87-case 注册表逐条与 E198 A0 arm case_id 对齐（object 计数 28/6/28/16/9），无重复、无缺失；每 case 能解析出 date/seq/person/object/model + 基础几何模板 scene。
- **C2 执行闭合**：≤246 条 aug full CEM 完成、打分成功，error / non-finite / diverged / fall = 0；跳过档位有明确 infeasible 记录（不静默）。
- **C3 增强正确性**：所有 trans 变体接近段 pose 相对 orig 偏移 = 0.20m（±容差），操作终点偏移 ≤ 接近段 25%（指数衰减锚定生效）。
- **C4 物理可信度（放量分布）**：aug vs 复用 orig（同 case）核心指标 mean+std+worst 全分布报告（不 cherry-pick）：obj_pos mean 增幅 ≤25%，接触保持 / 腿穿透 / 12-gate 通过率不劣于 orig 分布下限，无新增 fall/diverged。逐物体分层。
- **C5 可行性分布**：报告每物体 trans0/1/2 的可行 / 不可行（IK / 初始穿透）计数，作为放量产能的一等结论。
- **C6 视觉复核（强制，rule 9）**：每物体随机抽 ≥2 个新 case 的平移变体渲染关键帧，检查穿模 / 漂浮 / 抖动 / 跌倒；side-by-side vs 该 case 的 orig。观察写入 log287「实际观察」，不得留空。

**判定**：≥90% 可行 aug 满足 C4 且 C6 视觉无致命 artifact → 放量数据可用于下游 RL 扩充；否则定位破坏物理的 case/物体，收窄放量集。

## 改动文件

> 复用 pilot 的 E199 全部脚本骨架（`scripts/experiments/E199/*`、`launch/active/run_E199_local_8gpu.sh`、`eval/*/eval_E199_augmentation.*`、`render_qc.py`）。放量只需**扩 case 注册表 + 关旋转 + resume 跳过**，核心逻辑不改。

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E199/e199_common.py` | ① 新增 `BOX_FULLSCALE = True` 开关 / 或独立 `load_fullscale_cases()`：从 `results/E198/.../e198_arm_cache.tsv`（arm=A0）派生 87 个 box case 注册表（case_id→object_key/date/seq/person/object_name/object_model_rel/base_scene），替代硬编码 8-case `CASES`；② `VARIANTS` 放量集去掉 rot0/rot1（仅 orig+trans0/1/2），`AUG_VARIANTS` 随之只剩 3 平移。orig 在放量中**不重跑**（仅作 eval 参照，见下）。 |
| `scripts/experiments/E199/build_augmented_tasks.py` | ① case 元数据解析改走 `load_fullscale_cases()`——base_target_task 从 arm_cache `scene_xml` 父目录派生（71 个 v1 + 16 个 v2，均含 task_info.json+scene.xml，本机全在），读 task_info 取 date/seq/person/object/model，从 CORE4D raw root 定位原始序列跑上游增强；`aug_task_name()` 幂等处理已是 v2 的 base；② 只 build trans0/1/2（不建 orig task）；③ 保留逐变体 try/except 容错（infeasible / 初始穿透跳过）；④ artifacts TSV 增 `object_key` 分组便于放量合并。 |
| `scripts/experiments/E199/build_aug_manifest.py` | manifest 只含 trans 行（tier P1）；**orig 行改为指向复用的 E198 A0/PRG rollout**（`result_npz`/`outdir_npz`/`scene_act` 指 E173 PRG，status=`reused_a0`，不进 CEM 队列，仅供 eval 配对）。skip 集：pilot 已完成的 20 条 box run。 |
| `scripts/experiments/E199/run_local_priority_queue.py` | 无需改（manifest 驱动 + resume skip 已支持）；`reused_a0` 行自动跳过（非 pending 状态）。 |
| `scripts/eval/runners/eval_E199_augmentation.py` | orig 配对源改为 E198 A0 arm_cache（omnirt_v1，标注 confound）；按 (object, variant) 放量聚合 mean+std+worst + 逐物体分层 + 可行性分布（C5）。 |
| `scripts/launch/active/run_E199_local_8gpu.sh` / `scripts/train/train_E199.sh` | 加 `SCOPE=box_fullscale` env 透传；train 首步 `snapshot_scenes.sh` 对新 case scene 快照（rule 10b）。 |
| scene 快照 | `results/E199/scene_snapshot/cem_sidecars/<task>/`（每 sidecar 自动快照）+ manifest.txt。 |

> **决策记录**：orig 复用 A0 时，`build_aug_manifest.py` 不应误把 A0 rollout 当 E199 队列任务重跑；用独立 `reused_a0` 状态隔离。这是本轮相对 pilot 的唯一结构性改动，须隔离可逆（rule 3）。

## 执行命令

```bash
# 0) 前置：pilot 剩余 5 条 bucket CEM 跑完 + pilot eval 补全（不阻塞放量数据构建，但放量 CEM 队列启动前确认 GPU 有空档）

# 1) 全量数据构建（82 新 box case × 3 平移；跳过 pilot 已建；orig 不重建）
SCOPE=box_fullscale bash workspace/core4d/scripts/train/train_E199.sh

# 2) 全量 full CEM（8 卡 priority 队列，resume-safe，跳过 reused_a0 + pilot-done）
GPUS=0,1,2,3,4,5,6,7 SCOPE=box_fullscale \
  bash workspace/core4d/scripts/launch/active/run_E199_local_8gpu.sh   # run_in_background

# 3) 放量评估（aug vs 复用 A0/PRG orig，逐物体分层 + 可行性分布）
bash workspace/core4d/scripts/eval/wrappers/eval_E199_augmentation.sh

# 4) 视觉复核（每物体 ≥2 新 case 平移变体关键帧）
.venv/bin/python workspace/core4d/scripts/experiments/E199/render_qc.py --objects box001,box004,box021,box023,box024
```

## 风险 / 缓解

| 风险 | 缓解 |
|---|---|
| base task 目录命名不一致（71 个 `dcv3_omnirt_v1_ref_fk_*`、16 个 `dcv3_omnirt_v2_ref_fk_*`） | **已核实 87/87 case 的 scene_xml + orig rollout 均在本机，0 缺失**（2026-08-16 校验）。base_target_task 从 arm_cache `scene_xml` 父目录派生，不硬套 v1 前缀；`aug_task_name()` 的 `omnirt_v1→v2` 替换须幂等（v2 base 保持 v2）。16 个 v2 命名 case 见下注。 |
| orig omnirt_v1 vs aug omnirt_v2 混淆 | pilot 已用同条件证明可信；eval 显式标注，结论只声称「放量数据物理可信/可用」，不声称严格 ablation。 |
| 246 条 full CEM 占卡时长 | resume-safe queue，与他人 job 叠加不抢占；分物体 tier 内按 case 派发，可中断续跑。 |
| 某物体平移大面积不可行 | C5 可行性分布是一等结论；若某物体 <50% 可行，报告并从放量集剔除该物体，不强行产不可信数据。 |
| 接触掩码复用错位（增强改初始位姿） | 固定窗 trim（按 orig trim_start），augmentation 锚定操作段 → 时间维掩码不变，与 pilot 同保障。 |

## 成功标准（量化）

1. C0/C1：82 新 case 注册表与 A0 arm 逐条对齐（28/6/28/16/9），全 task + override + manifest 生成，0 blocker。
2. C2：≤246 aug full CEM 完成，0 error/non-finite/diverged/fall；infeasible 档位全记录。
3. C4：obj_pos mean 增幅 ≤25%；接触/腿穿透/12-gate 不劣于 orig 分布；0 新增 fall。
4. C5：逐物体 trans0/1/2 可行性计数报告。
5. C6：每物体 ≥2 case 视觉复核，观察入 log287，无致命 artifact。
6. 全部通过 → 更新 log287 + EXPERIMENT_TRACKER + progress；数据交下游 RL。

## 下一步（Phase 2，仍延后）

object scale（长宽高）增强需上游 holosoma 为 object_interaction 增 scale 增强 + 重算接触，另开计划。
