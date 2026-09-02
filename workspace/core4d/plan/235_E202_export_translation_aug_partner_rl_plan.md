# plan235 · E202-export：USE13 bucket 平移增强 → partner 配对 + Holosoma RL-ready 导出

_Core4D · Phase 63 S6 · Run **R290**（E202 导出阶段，无新 CEM）· 承接 [log291](../log/291_E202_bucket_e178_translation_augmentation.md)（E202 源侧平移增强）/ [log264](../log/264_E178_manual_use_partner_rl_export_results.md)（E178-export orig 配对导出契约）· 2026-08-27 · **状态：计划态（待批准）**_

## Context / 为什么做

E202（log291）已把 **OmniRetarget 物体平移增强**（trans0/1/2，omnirt_v2）放量到全部 **27** 个 E178 full-CEM bucket case 的**源侧**，并证明物理有效（obj_pos −0.6%、跌倒/穿透不劣于 orig、73/81 变体可行、73/73 CEM 0 err）。但 E202 只到**源侧 CEM + 物理有效性评估**，**没做 partner 配对，也没做 Holosoma RL 导出**。

E178-export（log264）已经为 **人工终审 `USE=13`** 的 bucket case（bucket003×5、bucket004×1、bucket007×7）产出过 **orig** 的 partner-complete RL 资产（source=E178 CEM，partner=对手人 retarget，Holosoma 26/26 过 pre-train gate）。

**本计划（E202 的 RL 导出阶段）**：**基于 E202 已跑好的源侧 aug rollout**，为这 **13 个 USE case** 导出**平移增强**版的 RL-ready 资产。**不新开实验号**（属 E202 S6 导出）。核心新增工作 = **partner（对手人）在同一物体扰动下 aug retarget**，再按 log264 契约配对 + Holosoma 导出。

### 复用 vs 新增

| 环节 | 来源 | 本计划 |
|---|---|---|
| 源侧 aug retarget + **CEM** rollout | E202（27 case，USE13 ⊂ 27） | **复用，零新 CEM** |
| 源侧物理有效性 | E202 `full_augmentation` | 复用 + USE13 子集分层报告 |
| **partner aug retarget（同扰动）** | — | **新增**（retarget-only，无 CEM） |
| source–partner 配对 | log264 契约 | **新增**（沿用 pair-complete） |
| Holosoma RL 导出 + pre-train gate | log264 exporter | **新增**（复用 exporter，扩 aug 维度） |
| pair/一致性/视觉评估 | — | **新增** |

> **本阶段无新 CEM 算力**：源侧 rollout 全部复用 E202；partner 侧与 log264 一样只 retarget（不 CEM）。工作量 = partner aug retarget（≤39 次，分钟级/条）+ 打包 + 评估。

## 范围（用户 2026-08-27 确认）

- **基底数据**：E202 源侧 aug rollout（`s6_downstream/cem/full/E202_*_aug_trans{0,1,2}_PRG*.npz`）。
- **选择 authority**：`results/E178/s6_downstream/eval/full/user_manual_review_filled.tsv`（SHA `d430a8ef…6c9f`，USE=13）—— 只读。
- **物体**：仅 bucket（bucket003×5、bucket004×1、bucket007×7 = 13 case）。
- **增强档位**：仅平移 **trans0/1/2**（前/左/右 0.2m）。不做旋转。
- **产出**：13 case 的平移增强 RL-ready 资产（orig 已由 log264 导出，本阶段补 trans 变体）。

## 核心技术点：partner 必须在同一物体扰动下 retarget

OmniRetarget 增强扰动的是**两人共享的物体接近段位姿**（trans_k + 指数衰减，manipulation 终点锚定，残差 ≈0.027m）。E202 只对**源侧那一人**施加了 trans_k。要产出物理自洽的配对，**partner（对手人）必须在完全相同的 trans_k 物体扰动下 retarget**，否则两人在接近段对物体位置产生分歧。

- 两人 task 共享同一 object 轨迹 + 同一 `object_metadata.json`，对各自 task 施加**同一 trans_k config** → 产生**逐字节相同的扰动物体轨迹**（确定性）。这是核心可验证不变量（C2）。
- partner 侧用 **omnirt_v2**（与源侧一致），**只 retarget、不 CEM**（同 log264：source 是 CEM 侧，partner 是真实 retarget 侧），trim 到 `common_raw_window`。

## 冻结不变量

| 项 | 值（来源） |
|---|---|
| 选择 authority | `user_manual_review_filled.tsv`（SHA `d430a8ef…6c9f`，USE=13）——只读 |
| 源侧 rollout | **复用 E202** `E202_{case}_aug_{trans}_PRG*.npz`（1024×32 seed0，E178 碰撞 + E174 PRG + `E170_PRG` gate；见 `e202_bucket_priority_manifest.tsv`） |
| 增强 config | holosoma object_interaction 原生 trans_0/1/2（前/左/右 0.2m），逐字节沿用；不加载 rot |
| partner retarget | **omnirt_v2/ref_fk**（`OMNIRT_V2_ENV`，与源侧一致），同一 trans_k 扰动，`common_raw_window` 对齐（复用 log264/E174 Stage2b 共窗逻辑，逐 case 重解析） |
| 导出 exporter | 复用 log264 `E178/export_manual_use_partner_rl.py`（→ E187 base），扩增 aug 变体维度；不改 orig 导出行为 |
| evaluator | 公共 `eval.core.core_metrics`（rule 13）；源侧指标直取 E202 `full_augmentation` + `e178_case_metrics.tsv` |

## Claims（可验收）

- **C0 选择保真**：authority TSV SHA == `d430a8ef…6c9f`，USE==13（5/1/7）；DO_NOT_USE 零进入；不符即停。
- **C1 源侧复用完整**：13 个 USE case 的每个可行 trans 变体在 E202 manifest 命中 `run_complete_pending_eval`（result_npz+outdir+scene_act SHA 齐全）；报告 USE13 子集 trans 可行/不可行分布。
- **C2 物体扰动两人一致（核心）**：每个可行 trans 变体，source-aug（E202）与 partner-aug 的扰动后物体轨迹逐帧 max-abs 差 ≤ 容差；接近段偏移 = 0.20m±容差、终点残差 ≤ 接近段 25%。不符 → 该变体不导出并记录（不静默）。
- **C3 partner 可行性 + 配对闭合**：每个可行 trans 变体，partner 在同扰动下 retarget 成功（IK 可达、无初始穿透）→ `pair complete`；**禁止 source-only / 零 partner / 复制 source fallback**（同 log264）。报告 partner 逐档可行/不可行（可能异于源侧）。
- **C4 物理质量门（增强特有）**：只导出满足 **fall_flag=0 且未发散** 的源侧 aug 变体。E202 已知 `bucket003_20231018_005_p1` 三档均 fall —— 若属 USE，其 trans 档全排除、仅 orig 保留，log 明列。numeric gate 保留为 provenance（不作删除依据，同 log264）；物理 fall/发散是硬排除。
- **C5 Holosoma 导出闭合**：对导出集（orig 复用 log264 + 通过 C2/C3/C4 的 trans 变体）跑 exporter，每 case ×2 target source（cem/trajectory），产出 paired motion 并**全部过 registry pre-train gate**；motion 字段（`partner_hand_pos_w/quat_w` 等）完整；报告最终 motion 计数。
- **C6 物理有效性（USE13 分层）**：源侧 aug vs orig（同 case，E202 `full_augmentation` + `e178_case_metrics.tsv`）在 USE13 子集上 mean+std+worst 全分布对比（obj_pos/接触/腿手穿透/gate/fall），逐物体分层，不 cherry-pick。
- **C7 视觉复核（强制，rule 9）**：每物体随机抽 ≥2 个 USE case 的 trans 变体，渲染 **source + partner 同场景配对**关键帧（grasp/contact/transition），检查两人-物体接近段一致性、穿模/漂浮/抖动/趴箱；side-by-side vs 该 case 的 orig 配对。`/video-frames` 抽帧，观察写入 log，不得留空。

**判定**：C0–C5 全过 + C2 两人一致 + C7 视觉无致命 artifact + C6 不劣于 orig 分布下限 → 增强 RL 导出集可用于下游扩数据；否则定位破坏配对/物理的 case/档位，收窄导出集并明列排除清单。

## 产能预估

| 物体 | USE case | ×3 trans 上限 | 预计可行(引 E202 源侧) | 说明 |
|---|--:|--:|--:|---|
| bucket003 | 5 | 15 | ~15 | E202 trans 全可行（1 fall case 需 C4 排除） |
| bucket004 | 1 | 3 | ~3 | E202 trans 全可行 |
| bucket007 | 7 | 21 | ≤~19 | E202 部分档腿-桶穿透不可行 |
| **合计** | **13** | **39** | **≤~37** | 最终取决于 C3 partner 可行性（可能更少） |

- 最终 RL 导出 = `13 orig`（复用 log264）+ `≤~37 trans`（新增），每条 ×2 target source。
- 无新 CEM。partner 增强 retarget ≤39 次 retarget-only，本地串行或小规模并行。

## 改动文件（隔离于 E202 导出阶段，E178/E202/log264 历史零覆盖 — rule 3）

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E202/e202_export_common.py`（新） | 契约单一真源：① `load_use13()` 从 authority TSV 派生 USE13 注册表（case_id→object/date/seq/person + orig source/partner 指向 log264 rl_export，SHA 校验）；② 从 `e202_bucket_priority_manifest.tsv` 定位每 USE case 可行 trans 源侧 rollout；③ `TRANS_VARIANTS`(trans0/1/2) + `OMNIRT_V2_ENV`；④ `common_raw_window` 解析复用 log264/E174 Stage2b。 |
| `scripts/experiments/E202/build_partner_aug.py`（新，仿 E199/E202 aug 分支） | 对每 USE case 每可行 trans 变体：跑 holosoma aug retarget（`RETARGET_AUGMENTATION=1 --skip-spider` + omnirt_v2 + trans_k）**对 partner task**，固定窗 trim 到 `common_raw_window`；产出 partner aug NPZ + provenance/SHA；逐变体 try/except 容错（infeasible 跳过并记录）。 |
| `scripts/experiments/E202/check_object_traj_parity.py`（新） | C2 断言：source-aug（E202）vs partner-aug 扰动后物体轨迹逐帧 max-abs 差 ≤ 容差；偏移/终点残差校验；不符列表输出。 |
| `scripts/experiments/E202/export_aug_partner_rl.py`（新，薄封装 log264 exporter） | 复用 `E178/export_manual_use_partner_rl.py`（→ E187 base）：source = E202 aug rollout（按 C4 过滤 fall/发散），partner = `build_partner_aug` 产物；orig + 通过 C2/C3/C4 的 trans 变体统一打包 → `results/E202/s6_downstream/rl_export_aug/`；每 case ×2 target source Holosoma 导出 + pre-train gate。 |
| `scripts/eval/runners/eval_E202_export_augmentation.py`（新） | 直接 `from eval.core.core_metrics import ...`（rule 13）；源侧指标取 E202 `full_augmentation` + `e178_case_metrics.tsv`，按 USE13 子集 (object,variant) 分层 mean+std+worst（C6）+ 可行性/配对/导出计数汇总。 |
| `scripts/launch/active/run_E202_export_partner_aug.sh`（新） | 真实入口：C0 校验 → build_partner_aug（可选并行）→ check_object_traj_parity → export → eval → 快照。 |

> 复用 E202/E199 aug 流水线骨架与 log264 导出契约；**唯一新增逻辑 = partner 侧同扰动 retarget + 两人物体轨迹一致性断言 + aug 变体维度配对导出**。不改 `pipeline.sh` 语义，不改 E178/E202 任何历史产物。

## 执行步骤（→ verify）

1. **C0 authority 校验** → verify：TSV SHA == 期望、USE==13、object 计数 5/1/7。
2. **定位源侧可行 trans（复用 E202）** → verify：USE13 每 case trans0/1/2 命中 E202 `run_complete_pending_eval`；输出可行分布 + fall/发散标记（C1/C4）。
3. **partner aug retarget（新）** → verify：每可行 trans 变体 partner NPZ 生成、trim 窗对齐、provenance/SHA 写入；infeasible 明确记录（C3）。
4. **C2 物体轨迹一致性断言** → verify：max-abs 差 ≤ 容差；不符变体剔除并列出。
5. **配对 + C4 物理过滤 + Holosoma 导出** → verify：pair-complete（无 source-only）；fall/发散档排除；paired motion 全过 pre-train gate（C3/C4/C5）。
6. **eval（USE13 分层 C6）** → verify：obj_pos/接触/穿透/gate/fall mean+std+worst 不劣于 orig 分布下限，逐物体分层。
7. **C7 视觉复核** → verify：每物体 ≥2 case 的 source+partner 配对关键帧渲染，`/video-frames` 抽帧，观察写入 log。
8. **快照 + git** → 见下。

## 快照与可复现（rule 7 / rule 10b）

- 导出前对导出集每 case 每变体用到的 scene_act（含 E202 `scene_act_E202_bucketAlignedTop_PRG`）+ **新增 partner aug scene** 做 `scene_snapshot/` 快照 + `manifest.txt`（git HEAD + sha256），存 `workspace/core4d/results/E202/scene_snapshot_export/`（不覆盖 E202 训练期快照 `results/E202/scene_snapshot/`）。
- 复用的 E202 源侧 scene 记录 SHA 指针（指向 E202 训练快照），不重复拷贝。

## Git 管理（rule 11）

- 属 **E202 S6 导出阶段**，同 exp_name=core4d、同 object-augmentation 主题、数据集未变 → 不触发强制沟通，沿用当前分支。
- 完成后：`exp(core4d): E202-export — USE13 bucket 平移增强 partner 配对 + Holosoma RL 导出 {一句话结论}`。

## 交付物

- `workspace/core4d/results/E202/s6_downstream/rl_export_aug/`（partner aug NPZ、C2 一致性报告、配对清单、Holosoma paired motions、eval、scene_snapshot_export；不进 git）。
- 实验日志 `workspace/core4d/log/NNN_E202_export_translation_aug_partner_rl_results.md` + tracker（E202-export 行）。
- 本计划 `workspace/core4d/plan/235_E202_export_translation_aug_partner_rl_plan.md`。

## 风险与待确认

1. **partner 侧可行性未知**：E202 只验证源侧；partner 在同扰动下可能有不同 IK/穿透结果（尤其 bucket007）→ C3 如实报告，不足则如实缩量（缩量是用户决定）。
2. **fall case 处理**：E202 已知 `bucket003_20231018_005_p1` 三档全 fall；若属 USE，其 trans 档全排除、仅 orig 保留（C4）。
3. **omnirt_v2 partner confound**：aug partner 用 v2、orig partner（log264）用 v1，碰撞体/物体扰动一致，唯一变量是 retarget 变体 → eval 显式标注（同 E202 诚实处理）。
