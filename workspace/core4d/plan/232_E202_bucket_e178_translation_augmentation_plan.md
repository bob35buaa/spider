# plan232 · E202：bucket 类 s6 full-CEM case 平移增强（碰撞体 + CEM 全用 E178）

_Core4D · Phase 63 · Run **R290** · 承接 [plan228](228_E199_omniretarget_object_augmentation_full_cem_plan.md)（pilot）/ [plan229](229_E199_box_fullscale_translation_augmentation_plan.md)（box 放量）/ [plan194](194_E178_bucket_contact_aligned_top_segment_plan.md)（bucket 碰撞体权威）· 2026-08-19 · **状态：计划态（待批准）**_

## Context / 目标

E199/E200 已经把 OmniRetarget object augmentation 的**平移增强**放量到 **box 类**所有 s6 full-CEM case（plan229，×3 平移，omnirt_v2）。本计划把同一套已验证的平移增强机制放量到 **bucket 类**，为下游 RL 扩数据。

与 box 放量的**唯一实质差别 = 碰撞体与 CEM 栈整体改用 E178**（用户 2026-08-19 明确要求「bucket 的碰撞体用 E178，其他 CEM 设置都用 E178」）：

- **碰撞体（E178）**：每物体 contact-aligned 五段实心 box proxy（`union` SDF，`batch_groups=true`）——bucket003 lower 0.94 / top 0.95×0.95、bucket007 lower 0.82 / top 0.97×0.885（5 geom → 90 pair）、bucket004 单 mesh-AABB box（1 geom → 18 pair）。这与 E199 box 用的 **16-pair 单 geom PRG** 是**不同的碰撞设置**。
- **CEM 栈（E178 = E174 PRG arm）**：rubber_hull 手 + PRG 下体物理 + `E170_PRG_lowerbodyPhysics_softPenalty_candidateGate` reward/gate；`scene_name` 走 E178 contact-aligned proxy + `union` SDF；`1024×32`、seed=0；3cm 接触掩码。

> **实验号 = 新 E202**（碰撞体/CEM 栈与 E199 不同，是独立组合，不并入 E199）。**同分支** `experiment/E199-omniretarget-object-augmentation`（同 core4d 方向、同 object-augmentation 主题，数据集/思路未变，不新建分支）。

### E199 pilot 对 bucket 的已知结论（log286）

- **可行性（omnirt_v2）**：8 物体一致 = **3 平移全可行、±45° 旋转全不可行**；bucket007 trans2（右移）参考首帧腿-桶穿透 15mm，被 PRG 运行时重叠保护正确拦截。→ **本计划只做平移，不做旋转**。
- pilot bucket 用的是 **E199 16-pair PRG**（非 E178 五段 proxy），故 pilot 已跑的 3 个 bucket case **不能跳过**，E202 必须在 E178 碰撞体下重跑。

## 关键决策（待用户 2026-08-19 确认）

1. **只做平移**（trans0/1/2 前/左/右 0.2m），**不做旋转**（pilot 已证 rot 全不可达）。
2. **retarget 变体 = omnirt_v2（仅用于 aug）**。E178 orig 用 omnirt_v1，但 object augmentation 在 v1 下常 IK 不可达（物体移出可达域），必须用 v2（E199 已证）。碰撞体在 orig 与 aug 间**完全相同（都是 E178）**，只有 retarget 变体不同 —— 这是唯一 confound，eval 显式标注。
3. **orig 同条件基线 = 复用现有 E178 27-case full-CEM 结果**（omnirt_v1，`E178_{case}_..._contactAlignedTop_full`），**不重跑 orig**。理由与 plan229 一致：目标是放量产数据而非再做严格 ablation；且碰撞体一致，confound 仅在 retarget 变体。
   - **可选（若用户要严格 ablation）**：orig 也在「omnirt_v2 + E178 碰撞体」下重跑 27 条 —— 代价翻倍，默认不做，作为待选记入风险表。
4. **执行范围 = 全量 27 个 E178 bucket case 一次性排队**，8 卡 priority queue（resume-safe，不抢占）。

## 范围：27 个 E178 bucket s6-full-CEM case（权威 = E178 full manifest）

权威 = `results/E178/s6_downstream/manifests/semantic_bucket_full_manifest.tsv`（27 unique case，逐条含 date/seq/person/object + `scene_act_E178_contactAlignedTop` + proxy 几何参数）：

| 物体 | E178 full case | proxy geom / pair | ×3 trans（上限） |
|---|---:|---:|---:|
| bucket003 | 9 | 5 / 90 | 27 |
| bucket004 | 4 | 1 / 18 | 12 |
| bucket007 | 14 | 5 / 90 | 42 |
| **合计** | **27** | — | **81** |

- **新增 full CEM 上限 = 81 条 aug**。实际 ≤81：部分档位上游 IK 不可行 / 参考首帧腿-桶穿透被 PRG 运行时拦截时跳过该档（同 pilot bucket007 trans2）。
- **orig = 0 条新 run**（复用 27 个 E178 full-CEM）。
- 预计 8×A100 priority queue ~10–15h（bucket 单例 full 1024×32 ≈ 45min，81 条）。

## 冻结不变量（逐字段对齐 E178，仅换 retarget=v2 + 相对 orig 平移物体位姿）

| 项 | 值（来源） |
|---|---|
| CEM | seed=0, num_samples=1024, max_num_iterations=32（E174/E178 契约） |
| retarget（aug） | **omnirt_v2/ref_fk**（`OMNIRT_V2_ENV`：relaxation+foot_z+contact_preservation on, slide=1.0, penetration_tol=0.8） |
| 碰撞体 | **E178 contact-aligned 五段 proxy**（per-object：003=0.94/0.95×0.95，007=0.82/0.97×0.885，004=单 mesh-AABB），`object_collision_sdf_mode=union`，`object_collision_sdf_batch_groups=true` |
| arm / reward | **E174 PRG arm**（rubber_hull 手 + PRG 下体物理 90/18 pair + `E170_PRG_lowerbodyPhysics_softPenalty_candidateGate`），`scene_name=scene_act_E202_bucketAlignedTop_PRG` |
| 增强 config | holosoma object_interaction 原生 trans_0/1/2（前/左/右 0.2m），逐字节沿用；**不加载 rot_0/1** |
| 接触掩码 | 每 case 由 v2 `_original` trim 窗重算 1 份 3cm 掩码，3 平移变体复用（固定窗 trim，按 orig trim_start 切片对齐）——同 E199 |
| evaluator | 公共 `eval.core.core_metrics`（rule 13） |
| orig 基线 | 复用 E178 27-case full-CEM（omnirt_v1，contactAlignedTop）——只读，不重跑 |

## Claims（放量验收）

- **C0 链路放量**：27 个 bucket case 各产出 ≤3 平移 SPIDER task（scene + trajectory + `scene_act_E202_bucketAlignedTop_PRG` 齐全）；base task yaml + E202 PRG override 全生成，manifest 0 blocker；scene sidecar 全快照（git HEAD + sha256）。
- **C1 碰撞体几何一致（E202 核心）**：每个 aug 变体的 `scene_act` object_collision geom（数量 + 尺寸 + `union`/`batch_groups`）与该物体 E178 orig proxy **逐字段一致**（bucket003/007 = 5 geom / 90 pair、bucket004 = 1 geom / 18 pair）；增强只改物体位姿轨迹，不改 proxy 几何。preflight 断言 geom-parity，任一不符停止。
- **C2 case 权威一致**：27-case 注册表逐条与 E178 full manifest case_id 对齐（object 计数 9/4/14），无重复、无缺失。
- **C3 执行闭合**：≤81 条 aug full CEM 完成、打分成功，error / non-finite / diverged / fall = 0；跳过档位有明确 infeasible 记录（不静默）。
- **C4 增强正确性**：所有 trans 变体接近段 pose 相对 orig 偏移 = 0.20m（±容差），操作终点偏移 ≤ 接近段 25%（指数衰减锚定生效）。
- **C5 物理可信度（放量分布）**：aug vs 复用 E178 orig（同 case）核心指标 mean+std+worst 全分布报告（不 cherry-pick）：obj_pos mean 增幅 ≤25%，接触保持 / 腿穿透 / 12-gate 通过率不劣于 orig 分布下限，无新增 fall/diverged。逐物体分层。
- **C6 可行性分布**：报告每物体 trans0/1/2 的可行 / 不可行（IK / 初始穿透）计数，作为放量产能的一等结论。
- **C7 视觉复核（强制，rule 9）**：每物体随机抽 ≥2 个 case 的平移变体渲染关键帧，检查穿模 / 漂浮 / 抖动 / 跌倒；side-by-side vs 该 case 的 E178 orig。观察写入 log291「实际观察」，不得留空。

**判定**：≥90% 可行 aug 满足 C5 且 C7 视觉无致命 artifact → bucket 放量数据可用于下游 RL 扩充；否则定位破坏物理的 case/物体，收窄放量集。

## 改动文件（隔离于 E202，E199/E178 历史零覆盖 — rule 3）

> 复用 E199 的 aug 流水线骨架（pipeline.sh aug 分支、trim/掩码、manifest 队列、eval 聚合），**唯一实质替换 = 碰撞体/scene 构建从 E199 16-pair PRG 换成 E178 五段 proxy + E174 90-pair union PRG**。E178 proxy 几何是 per-object、mesh 派生、与位姿无关（`build_contact_aligned_boxes`），可直接迁移到 aug scene。

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E202/e202_common.py`（新） | 契约单一真源：① `load_e178_bucket_cases()` 从 E178 full manifest 派生 27 case 注册表（case_id→object/date/seq/person/base_target_task/proxy 参数 + orig_* 指向 E178 full-CEM rollout 供 eval 配对）；② `TRANS_VARIANTS`（仅 trans0/1/2）+ `OMNIRT_V2_ENV`；③ `SCENE_NAME=scene_act_E202_bucketAlignedTop_PRG`；④ 从 E174 `e174_common` / `build_prg_cem_manifest` 导入 union/90-pair PRG 契约 + `E170_PRG` reward/gate 常量（单一真源，禁止硬编码），从 E178 `contact_aligned_bucket_proxy` 导入 `build_contact_aligned_boxes`（per-object proxy 几何）；⑤ CEM 1024×32 seed0 从 E174 契约取。 |
| `scripts/experiments/E202/build_augmented_tasks.py`（新，仿 E199） | 每 case：跑 `pipeline.sh` aug（`RETARGET_AUGMENTATION=1 --skip-spider` + omnirt_v2 env，convert→parallel retarget→trim original→3cm 掩码）→ 固定窗 trim 平移变体 → build SPIDER task `{base_v2}__aug_{trans}`（scene.xml→trajectory via `core4d.py`）→ **对 aug scene 施加 E178 五段 proxy 几何 + E174 90/18-pair union PRG + rubber_hull → `scene_act_E202_bucketAlignedTop_PRG`**（替代 E199 的 `build_prg_scene`）→ C1 geom-parity 断言（对齐该物体 E178 orig proxy）→ C4 pose-diff。保留逐变体 try/except 容错（infeasible / 初始穿透跳过）。 |
| `scripts/experiments/E202/build_aug_manifest.py`（新，仿 E199） | manifest 只含 trans 行（tier P1）；**orig 行指向复用的 E178 full-CEM rollout**（`result_npz`/`outdir_npz`/`scene_act` 指 E178 contactAlignedTop，status=`reused_e178`，不进 CEM 队列，仅供 eval 配对）；写 E202 priority manifest + authority TSV。 |
| `scripts/experiments/E202/run_local_priority_queue.py` | 复用 E199 实现（manifest 驱动 + resume skip；`reused_e178` 行自动跳过）——薄封装或直接 import。 |
| `scripts/eval/runners/eval_E202_bucket_augmentation.py`（新） | 直接 `from eval.core.core_metrics import ...`（rule 13）；orig 配对源 = E178 full-CEM（omnirt_v1，标注 confound）；按 (object, variant) 放量聚合 mean+std+worst + 逐物体分层 + 可行性分布（C6）。 |
| `scripts/eval/wrappers/eval_E202_bucket_augmentation.sh`（新） | shell 入口。 |
| `scripts/train/train_E202.sh`（新） | 首步 `snapshot_scenes.sh E202 <27 case>`（rule 10b）→ 调 `build_augmented_tasks.py` + `build_aug_manifest.py`。 |
| `scripts/launch/active/run_E202_local_8gpu.sh`（新） | 8 卡 priority 队列启动，GPU 占位符。 |
| scene 快照 | `results/E202/scene_snapshot/cem_sidecars/<task>/`（每 sidecar 自动快照）+ `manifest.txt`（git HEAD + sha256）。 |

> **决策记录**：orig 复用 E178 时，`build_aug_manifest.py` 用独立 `reused_e178` 状态隔离，队列不误重跑（同 plan229 的 `reused_a0`）。E202 全部新增文件，不改 E199/E178/E174 任一脚本（如需从它们 import，只读 import，不改其行为）。

## 执行命令

```bash
# 1) 数据构建（27 bucket case × 3 平移；orig 不重建；首步自动 snapshot）
bash workspace/core4d/scripts/train/train_E202.sh

# 2) 全量 full CEM（8 卡 priority 队列，resume-safe，跳过 reused_e178）
GPUS=0,1,2,3,4,5,6,7 \
  bash workspace/core4d/scripts/launch/active/run_E202_local_8gpu.sh   # run_in_background

# 3) 放量评估（aug vs 复用 E178 orig，逐物体分层 + 可行性分布）
bash workspace/core4d/scripts/eval/wrappers/eval_E202_bucket_augmentation.sh

# 4) 视觉复核（每物体 ≥2 case 平移变体关键帧，side-by-side vs E178 orig）
.venv/bin/python workspace/core4d/scripts/experiments/E202/render_qc.py --objects bucket003,bucket004,bucket007
```

## 风险 / 缓解

| 风险 | 缓解 |
|---|---|
| **E178 proxy 迁移到 aug scene 出错**（几何/pair 数不对） | proxy 几何是 per-object mesh 派生、与位姿无关，可直接复用 `build_contact_aligned_boxes`；C1 preflight geom-parity 断言（geom 数/尺寸/pair 数逐字段对齐 E178 orig），不符即停，不进 CEM。 |
| orig omnirt_v1 vs aug omnirt_v2 retarget 混淆 | 碰撞体在 orig/aug 完全一致（都是 E178），confound 仅在 retarget 变体；pilot 已在同条件证可信；eval 显式标注；结论只声称「放量数据物理可信/可用」，不声称严格 ablation。若用户要严格：另跑 omnirt_v2+E178 的 orig（默认不做）。 |
| bucket 在严格 12-gate 下本就难（lower_body 主导，log286 已记） | 12-gate 通过率与增强变量正交 —— 以「aug 不劣于同 case E178 orig 分布下限」为判据，而非绝对 gate 通过率；tracking/穿透/接触为主指标。 |
| bucket007 trans2 类初始腿-桶穿透 | PRG 运行时重叠保护拦截 → 记为 infeasible（C6），不静默、不产不可信数据。 |
| 接触掩码复用错位（增强改初始位姿） | 固定窗 trim（按 orig trim_start），augmentation 锚定操作段 → 时间维掩码不变，同 E199 保障。 |
| 某物体平移大面积不可行 | C6 可行性分布是一等结论；若某物体 <50% 可行，报告并从放量集剔除该物体。 |

## 成功标准（量化）

1. C0/C1/C2：27 case 注册表与 E178 full manifest 逐条对齐（9/4/14），全 task + override + manifest 生成，proxy geom-parity 27/27 通过，0 blocker。
2. C3：≤81 aug full CEM 完成，0 error/non-finite/diverged/fall；infeasible 档位全记录。
3. C4：接近段偏移 0.20m（±容差），操作终点偏移 ≤ 接近段 25%。
4. C5：obj_pos mean 增幅 ≤25%；接触/腿穿透/12-gate 不劣于 E178 orig 分布；0 新增 fall。逐物体分层。
5. C6：逐物体 trans0/1/2 可行性计数报告。
6. C7：每物体 ≥2 case 视觉复核，观察入 log291，无致命 artifact。
7. 全部通过 → 更新 log291 + EXPERIMENT_TRACKER（R290）+ progress；数据交下游 RL（可接 E200 式 RL-export）。

## 下一步（延后）

- object scale（长宽高）增强需上游 holosoma 为 object_interaction 增 scale 增强 + 重算接触，另开计划（与 box 放量共用同一 Phase 2 待办）。
- bucket aug 数据的 RL-export（仿 E200 三版导出），待 C1–C7 通过后另开。
