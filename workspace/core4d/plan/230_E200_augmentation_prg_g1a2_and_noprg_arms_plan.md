# plan230 · E200：OmniRetarget 平移增强放量到 **PRG+G1+A2** 与 **noPRG(E167A)** 两 arm（+ 三 arm 横向对比）

_Core4D · Phase 63 · Run **R287**(PRG+G1+A2) + **R288**(noPRG) · 承接 [plan229](229_E199_box_fullscale_translation_augmentation_plan.md) / [log286](../log/286_E199_omniretarget_object_augmentation_results.md) · 2026-08-17 · 分支 `experiment/E199-omniretarget-object-augmentation`（同方向延伸，不新建分支）· **状态：计划态，待批准**_

## Context / 目标

E199 已在 **A0/PRG arm**（`scene_act_E199_rubberHull_PRG`，reward base=E167A_zOnlyBody + 16 lower-body/object 碰撞对）下证明**上游 OmniRetarget 平移增强**在本管线物理可信可用（pilot obj_pos +5.0%、腿/手穿透 aug 更优、0 跌倒、视觉无 artifact；放量 87 case × ≤3 平移已建 249 aug task）。

**E200 把已验证的平移增强复用到另外两个下游 CEM arm**，目的（用户确认「两者都要」）：
1. **产数据**：为下游 RL 额外产出 **PRG+G1+A2** 与 **noPRG(E167A)** 两种 arm 的增强数据变体。
2. **横向择优**：在同 case 上对 **noPRG / PRG(E199) / PRG+G1+A2** 三 arm 的增强数据做 paired 物理质量对比，为下游选最优 arm 提供依据（复用 E194 三 arm 对比框架口径）。

### 核心洞察（决定 E200 极低成本）

> **平移增强只存在于 retarget 阶段产出的参考轨迹里，与下游 SPIDER CEM 的 arm(scene_act/reward override) 完全解耦。**

E199 每个 aug task 目录（`.../dcv3_omnirt_v2_ref_fk_{case}__aug_trans{0,1,2}/`）**已同时含**：
- `0/trajectory_kinematic.npz` —— **arm-independent** 的增强参考轨迹（trans 扰动已烘焙其中）；
- `scene_act.xml` —— **pre-PRG 的 base scene**（= **noPRG scene**，无 16 碰撞对）；
- `scene_act_E199_rubberHull_PRG.xml` —— **PRG scene**（含 16 lower-body/object 碰撞对）。

因此 E200 **不重跑上游增强、不重建任何轨迹**，只需：
- **noPRG arm**：直接用 aug `scene_act.xml` + **E167A base override** 跑 CEM；
- **PRG+G1+A2 arm**：从 aug `scene_act_E199_rubberHull_PRG.xml` 建 **G1 gravcomp sidecar**（object body `gravcomp` 0→1 单变量 diff）+ 挂 **A2_GATE** override 跑 CEM。

## 关键决策（用户 2026-08-17 确认）

| # | 决策 | 内容 |
|---|---|---|
| 1 | **目标** | 两者都要：产两 arm 增强数据 **且** 三 arm 横向 paired 择优。 |
| 2 | **noPRG 定义** | **= E167A（`E167A_zOnlyBody` base arm）**，即 pre-PRG base scene + E167A reward，`leg_object_penalty_scale=0` / `leg_object_penalty_geom_names=[]` / `cem_leg_gate_enabled=false`，**无 G1/A2**。 |
| 3 | **noPRG orig 基线** | **复用已有 E190 38-case noPRG**（不新跑 orig）。逐物体：box001=13 / box004=4 / box021=11 / box023=7 / box024=3。来源 E189(box001/004/024) + E179(box023) + E168/E167(box021)。 |
| 4 | **case 范围** | **全 87 box case**（两 arm 各产 87 case 的 aug，复用 E199 已建 249 aug task）。 |

## Arm 精确定义（三 arm 对照，权威见 e173/e198_common）

| Arm | scene_act | object gravcomp | hand-gate(A2) | leg-object 碰撞/leg gate | reward base | orig 基线来源 |
|---|---|---|---|---|---|---|
| **noPRG (E167A)** — Arm C / R288 | aug `scene_act.xml`（base，无碰撞对） | 0 | 无 | `leg_object_penalty_scale=0`, `cem_leg_gate_enabled=false` | E167A_zOnlyBody | **E190 noPRG（38 case）** |
| **PRG (E199)** — 参照 | aug `scene_act_E199_rubberHull_PRG.xml` | 0 | 无 | 16 对 + leg gate on | E167A_zOnlyBody | E199 已跑（87 case aug + E198 A0 orig） |
| **PRG+G1+A2** — Arm B / R287 | **新建 G1 gravcomp sidecar**（从 PRG scene，object gravcomp 0→1） | **1** | **A2_GATE** | 16 对 + leg gate on | E167A_zOnlyBody | **E198 G1A2 arm（87 case）** |

**A2_GATE**（逐字节沿用 E198/E192，audit 校验相等）：
`cem_hand_gate_min_sdf_m=-0.010`、`cem_hand_gate_max_violation_pct=0.05`、`cem_hand_gate_hard_floor_m=-0.015`。

**G1 单变量 diff**：sidecar == PRG scene，仅 `object` body 的 `gravcomp` 由 0/缺失 → 1，其余树结构逐节点相等（沿用 E198 `assert_gravcomp_diff` 断言）。

## 范围与 GPU 预算

| Arm | aug case | aug CEM 上限 | orig 基线 | orig 新跑 |
|---|---|---|---|---|
| PRG+G1+A2 (R287) | 87 | **≤249**（复用 E199 feasible aug 集） | E198 G1A2（87，只读复用） | 0 |
| noPRG (R288) | 87 | **≤249**（复用 E199 feasible aug 集） | E190 noPRG（**38**，只读复用） | 0 |
| **合计** | — | **≤498 aug full CEM** | — | **0 orig** |

- 单例 full 1024×32 ≈ 45min；8×A100 priority queue（resume-safe，不抢占）≈ **40–50h**。
- **0 条新 orig run**（两 arm 的 orig 全部复用 E198 G1A2 + E190 noPRG）。
- **0 条新上游 retarget / trajectory build**（复用 E199 aug）。

### 覆盖口径（诚实报告，写入 log）

- **PRG+G1+A2 aug vs E198 G1A2 orig**：87 case **全配对**（clean，同 87 universe）。
- **noPRG aug vs E190 noPRG orig**：仅 **38 case 可配对**（E190 基线覆盖面）；其余 49 noPRG aug 为 **produce-only**（供下游 RL，无同 case orig 基线，仅做视觉 QC）。
- **三 arm 横向 paired 对比**：在 **noPRG(38) ∩ PRG(87) ∩ PRG+G1+A2(87) = 38 case 交集**上做（box001=13/004=4/021=11/023=7/024=3）。

## 冻结不变量

| 项 | 值 |
|---|---|
| CEM | seed=0, num_samples=1024, max_num_iterations=32, use_torch_compile=false（与 E199/E198 逐字段一致） |
| 增强参考轨迹 | **复用 E199 fullscale aug**（omnirt_v2/ref_fk，trans0/1/2，0.2m 前/左/右）；不重跑、不加 rot |
| 接触掩码 | 复用 E199 每 case 3cm 掩码（固定窗 trim，arm 无关） |
| reward base（三 arm 同） | E167A_zOnlyBody（PRG 与 noPRG 差异仅在 scene 碰撞对 + leg gate；G1/A2 为 PRG+G1+A2 的增量） |
| evaluator | 公共 `eval.core.core_metrics`（rule 13），EvalConfig 与 E198/E199 同 contract |
| orig 基线 | E198 G1A2（PRG+G1+A2）、E190 noPRG（noPRG）——**只读，不重跑** |

## Claims（放量 + 对比验收）

- **C0 数据复用正确**：两 arm 各 87 case 的 aug scene_act 均**源自 E199 aug task 目录**（sha256 追溯）；trajectory/contact_mask 与 E199 逐字节一致（arm 无关性证明）。0 条新上游 retarget、0 条新 trajectory build。
- **C1 arm 构造单变量正确**：
  - PRG+G1+A2 的 G1 sidecar 相对 aug PRG scene 是 **object gravcomp 0→1 的单变量 diff**（`assert_gravcomp_diff` 通过）；A2 override 三字段 == A2_GATE。
  - noPRG 的 scene 无 16 碰撞对痕迹、override 含 `leg_object_penalty_scale=0` + `cem_leg_gate_enabled=false`（`prg_negative_config_check` 通过，复用 E190 审计口径）。
- **C2 执行闭合**：≤498 条 aug full CEM 完成、打分成功，error / non-finite / diverged / fall = 0；infeasible/跳过档位有明确记录（不静默）。
- **C3 各 arm vs 各自 orig（放量分布，目标①）**：mean+std+worst 全分布、逐物体分层：
  - PRG+G1+A2 aug vs E198 G1A2 orig（87 paired）：obj_pos mean 增幅 ≤25%，接触/腿穿透/12-gate 不劣于 orig 分布下限，无新增 fall/diverged。
  - noPRG aug vs E190 noPRG orig（38 paired）：同上阈值。
- **C4 三 arm 横向 paired 对比（目标②）**：在 38 case 交集上，对 orig 与 aug 分别出 **noPRG / PRG / PRG+G1+A2** 的 12-gate 通过率、obj_pos/ori、接触保持、腿穿透 paired 对比（McNemar/bootstrap，复用 E194 workbook 口径），给出**每物体 arm 排名 + 推荐 arm**。
- **C5 可行性分布**：报告两 arm 逐物体 trans0/1/2 的可行/不可行计数（本轮 infeasible 主要应来自 CEM diverge/gate，而非上游 IK——上游已在 E199 定档）。
- **C6 confound 标注**：显式记录 orig(omnirt_v1) vs aug(omnirt_v2) retarget 混淆；**noPRG box021 额外口径差异**（E168 CEM 超参/评分与统一流程不一致 + 2 bridge case 未跑过 PRG，继承 E190 审计 note），box021 三 arm 对比结论须单列、不与 box001/004/023/024 混算。
- **C7 视觉复核（强制，rule 9）**：每物体每 arm 随机抽 ≥2 个新 case 的平移变体渲染关键帧，检查穿模/漂浮/抖动/跌倒；三 arm side-by-side（同 case 同 trans）。观察写入 log288「实际观察」，不得留空。

**判定**：≥90% 可行 aug 满足 C3 且 C7 无致命 artifact → 两 arm 增强数据可用于下游 RL；C4 给出可复现的 per-object arm 推荐 → 目标②达成。否则定位破坏物理的 arm×物体，收窄该 arm 放量集。

## 改动文件

> 复用 E199 全部脚本骨架 + E198 的 G1/A2 arm 构造工具 + E190 的 noPRG 负配置口径 + E194 三 arm 对比 workbook。E200 的新增逻辑**只有「按 arm 换 scene_act/override 重跑 CEM + 三 arm 配对 eval」**，核心增强/轨迹/掩码逻辑一行不改。

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E200/e200_common.py`（新建） | E200 scope/paths；`ARMS = ("noprg", "prg_g1a2")`；从 E199 `e199_fullscale_aug_artifacts.tsv` 派生 249 aug task 注册表（trajectory/base scene_act/PRG scene_act/contact_mask 直接引用，不复制）；import E198 `A2_GATE`/`a2_overrides`/`assert_gravcomp_diff` 与 E173 `BASE_REWARD_METHOD`、E190 negative-config 断言（单一真源，不复制常量）。 |
| `scripts/experiments/E200/build_arm_scenes.py`（新建） | per aug task per arm 产 scene sidecar + override：① **prg_g1a2**：从 aug PRG scene 建 G1 gravcomp sidecar（单变量 diff，`assert_gravcomp_diff` 自检）+ A2_GATE override payload；② **noprg**：直接指向 aug `scene_act.xml` + E167A base override（含 leg_object_penalty=0 / leg_gate off），`prg_negative_config_check` 自检。写 per-arm manifest。 |
| `scripts/experiments/E200/build_manifest.py`（新建） | 两 arm 的 8-GPU priority manifest（trans-only，无 orig 行）；orig 行以 `reused_g1a2`/`reused_e190_noprg` 状态**指向 E198 G1A2 / E190 noPRG rollout**（只读，仅供 eval 配对，不进 CEM 队列）；skip 集=已跑完。 |
| `scripts/experiments/E200/run_local_priority_queue.py`（新建，仿 E199） | manifest 驱动 + resume skip；`reused_*` 行自动跳过；记录落卡 GPU id。 |
| `scripts/eval/runners/eval_E200_arm_augmentation.py`（新建） | ① 各 arm aug vs 各自 orig 放量分布（C3，逐物体分层 + 可行性 C5）；② 38-case 交集三 arm paired 对比（C4，复用 `eval.core.core_metrics` + E194 McNemar/bootstrap 口径）。orig 配对源：prg_g1a2→E198 G1A2 arm_cache、noprg→E190 rl_export。 |
| `scripts/eval/reports/gen_E200_three_arm_workbook.py`（新建，仿 E194） | 三 arm 12-gate + tracking 对比 workbook（noPRG/PRG/PRG+G1+A2），per-object 排名 + 推荐 arm。 |
| `scripts/experiments/E200/render_qc.py`（新建，仿 E199） | 三 arm side-by-side 关键帧（同 case 同 trans）。 |
| `scripts/train/train_E200.sh` / `scripts/launch/active/run_E200_local_8gpu.sh` / `pull_E200_*.sh` | 用 `gen_experiment.py` 模板生成；`ARM=noprg|prg_g1a2` 透传；train 首步 `snapshot_scenes.sh` 对**新建的 G1 sidecar** 快照（rule 10b；noPRG 复用 E199 已快照的 base scene，PRG+G1+A2 的 G1 sidecar 为新产物必须快照）。 |
| scene 快照 | `results/E200/scene_snapshot/g1_sidecars/<task>/` + `manifest.txt`（git HEAD + sha256）。 |

> **决策记录（rule 3 隔离可逆）**：E200 全部为**新建 E200/ 目录 + 新 results/E200/**，不改 E199/E198/E190 任何脚本或产物；orig 复用以 `reused_*` 状态隔离，绝不把 E198/E190 rollout 当队列任务重跑。

## 执行命令

```bash
# 0) 前置：确认 E199 fullscale aug DATA BUILD 完成（87 case 的 aug trajectory+scene 齐全；
#    注意——只依赖 E199 的「数据构建」，不依赖 E199 的 CEM 是否 100% 跑完）。
#    校验：results/E199/.../e199_fullscale_aug_artifacts.tsv 应含 249 行且 scene_act.xml + 
#    scene_act_E199_rubberHull_PRG.xml + trajectory 全部在盘。

# 1) 建两 arm 的 scene sidecar + override（G1 gravcomp 单变量 diff + A2 override / E167A negative config）
.venv/bin/python workspace/core4d/scripts/experiments/E200/build_arm_scenes.py --arms noprg,prg_g1a2

# 2) 建 8-GPU priority manifest（trans-only，orig 行 reused_*）
.venv/bin/python workspace/core4d/scripts/experiments/E200/build_manifest.py --arms noprg,prg_g1a2

# 3) 两 arm aug full CEM（8 卡 priority 队列，resume-safe，跳过 reused_*/已跑完）  # run_in_background
GPUS=0,1,2,3,4,5,6,7 ARM=prg_g1a2 bash workspace/core4d/scripts/launch/active/run_E200_local_8gpu.sh
GPUS=0,1,2,3,4,5,6,7 ARM=noprg    bash workspace/core4d/scripts/launch/active/run_E200_local_8gpu.sh

# 4) 评估：各 arm vs 各自 orig（C3/C5） + 38-case 交集三 arm paired（C4）
bash workspace/core4d/scripts/eval/wrappers/eval_E200_arm_augmentation.sh
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E200_three_arm_workbook.py

# 5) 视觉复核（三 arm side-by-side，每物体每 arm ≥2 case）
.venv/bin/python workspace/core4d/scripts/experiments/E200/render_qc.py --objects box001,box004,box021,box023,box024
```

## 风险 / 缓解

| 风险 | 缓解 |
|---|---|
| noPRG orig 仅 38 case，无法与 87 aug 全配对 | 明确分层：87 aug 用于**产数据**，38 用于**验证/三 arm 对比**，其余 49 produce-only + 视觉 QC。诚实写入 log（C6）。 |
| box021 noPRG 口径不一致（E168 超参 + 2 bridge） | 继承 E190 审计 note；box021 三 arm 对比结论**单列**，不与其余 4 物体混算（C6）。 |
| orig omnirt_v1 vs aug omnirt_v2 混淆 | 沿用 E199 结论（同条件 pilot 已证可信）；eval 显式标注，只声称「增强数据物理可信/可用」，不声称严格 retarget ablation。 |
| G1 sidecar 建错（非单变量） | `assert_gravcomp_diff` 逐节点断言（复用 E198）；建完即自检，失败即 abort，不进队列。 |
| noPRG scene 误带 PRG 碰撞对 | `prg_negative_config_check`（复用 E190）逐 case 核 scene_name/leg penalty/leg gate。 |
| ≤498 条占卡 40–50h | resume-safe queue，与他人 job 叠加不抢占；分 arm×物体 tier 派发，可中断续跑；必要时先跑 38-case 交集（保对比结论）再补 produce-only。 |
| 某 arm×物体大面积 diverge | C5 可行性分布是一等结论；若某 arm 某物体 <50% 可行，报告并从该 arm 放量集剔除，不强产不可信数据。 |

## 成功标准（量化）

1. C0/C1：两 arm 各 87 case aug scene 追溯到 E199 aug task（sha256）；G1 单变量 diff + A2==A2_GATE + noPRG negative-config 全自检通过；0 新 retarget/trajectory。
2. C2：≤498 aug full CEM 完成，0 error/non-finite/diverged/fall；infeasible 全记录。
3. C3：两 arm 各自 aug vs orig，obj_pos mean 增幅 ≤25%、接触/腿穿透/12-gate 不劣于 orig 分布、0 新增 fall（PRG+G1+A2 用 87 paired，noPRG 用 38 paired）。
4. C4：38-case 交集三 arm paired workbook 产出，per-object arm 排名 + 推荐 arm（box021 单列）。
5. C5：两 arm 逐物体 trans0/1/2 可行性计数。
6. C6：confound + box021 口径差异显式标注。
7. C7：每物体每 arm ≥2 case 三 arm side-by-side 视觉复核，观察入 log288，无致命 artifact。
8. 全部通过 → 写 log288 + 更新 EXPERIMENT_TRACKER + progress + build_log_index；两 arm 数据交下游 RL。

## 下一步

- 下游 RL：三 arm（noPRG / PRG / PRG+G1+A2）× (orig + aug) 数据的训练对比（选 arm + 验增强增益）。
- Phase 2（object scale 增强）仍延后（需上游 holosoma 为 object_interaction 增 scale 增强 + 重算接触）。
