# plan234 · E204 (noPRG) / E205 (G1A2)：复用 E178 的 27 bucket case 做两 arm 重定向 + 三 arm 对比

_Core4D · Phase 64 · Run **R289**(E204 noPRG) + **R290**(E205 G1A2) · 承接 [E178 contact-aligned bucket](../log/237_E178_bucket_contact_aligned_proxy_gates.md) / [plan230 E200 三 arm 口径](230_E200_augmentation_prg_g1a2_and_noprg_arms_plan.md) · 2026-08-23 · 分支 `experiment/E203-core4d-v2-orig-retarget`（同 core4d 方向，v1 arm ablation）· **状态：脚本已实现，待用户在 8 卡机执行 CEM**_

## Context / 目标

E178 已在 **PRG arm**（leg 惩罚 + leg gate）+ **contact-aligned 五段物体代理**（`scene_act_E178_contactAlignedTop`, `object_collision_sdf_mode=union`）下，对 27 个 bucket case（bucket003=9 / bucket004=4 / bucket007=14）跑通 full CEM（1024×32 seed0, omnirt_v1 ref_fk；1/27 为 omnirt_v2 rescue case），产出 `results/E178/.../e178_case_metrics.tsv`。

本实验在**完全相同的 27 case、相同参考轨迹、相同 3cm 接触掩码、相同 CEM 预算、相同 contactAlignedTop 五段物体代理**下，只改 reward arm，产两条新 arm 数据并做三 arm 择优（用户 2026-08-23 确认 4 项决策）：

- **E204 (noPRG / E167A)** — R289：去 leg-PRG（场景删 16 腿↔物碰撞对 + `leg_object_penalty_scale=0` / `cem_leg_gate_enabled=false`），object `gravcomp=0`，无 A2。**保留五段物体代理**。
- **E205 (G1A2 = PRG+G1+A2)** — R290：E178 PRG 之上加 **G1**（object body `gravcomp` 0→1 单变量 sidecar）+ **A2**（hand-gate `min_sdf=-0.010 / max_violation=0.05 / hard_floor=-0.015`）。

**解耦洞察**：arm 差异只在 `scene_act` + reward/gate override，与 arm 无关的轨迹/掩码/CEM 预算完全复用 E178，不重跑上游 retarget、不重建轨迹。中间 PRG arm 直接读用 E178 现成结果。

## 已核实的继承链 + 关键事实

```
core4d_E178_<case>_contactAlignedTop → core4d_E174_<case>_PRG → core4d_dcv3_omnirt_v{1,2}_ref_fk_<case> → core4d_E167_..._E167A → core4d_E163_..._narrowSurfaceBand
```
- **hand-gate 在 E163 层已启用**（`cem_hand_gate_enabled=true`, geoms `["lh","rh"]`, 默认 `max_viol=0.10/hard_floor=-0.020`）→ E205 的 A2 三字段 retune 生效；E204 不动它。
- **object_collision_sdf=union 在 E178 层声明** → E204 不继承 E178，须显式重声明。
- **26/27 case 是 omnirt_v1，1 case（`bucket007_20231020_055_p1`）是 omnirt_v2**（v2-rescue）。base override 逐 case 从 source 行推导（`Path(scene_act).parent.name`），不硬编码 v1。
- E178 contactAlignedTop 场景含 114 pair = 80 腿↔物（16 腿 geom × 5 seg）+ 10 手↔物（2 手 × 5）+ 24 腿-地/自碰撞。E204 只删 80 腿↔物（用 `base.is_robot_object_pair` 精确识别），保留手↔物 + 腿-地/自碰撞。

## Arm 精确定义（三 arm 对照）

| Arm | scene_act | 腿↔物碰撞对 | leg penalty/gate | object gravcomp | hand-gate | 继承 |
|---|---|---|---|---|---|---|
| **noPRG (E204)** | `scene_act_E204_contactAlignedTop_noPRG`（五段代理，删 80 腿↔物） | 0 | 0 / false | 0 | E163 默认 | `dcv3_*_<case>` |
| **PRG (E178, 参照)** | `scene_act_E178_contactAlignedTop` | 80 | 2.0 / true | 0 | E163 默认 | E174 PRG |
| **G1A2 (E205)** | `scene_act_E205_contactAlignedTop_gravcomp`（=E178 + gravcomp 单变量） | 80 | 2.0 / true | **1** | **A2** | E178 |

## 改动文件（全部新建于 `scripts/experiments/E204_E205/`，隔离可逆，不改 E178/E174/E198/E200）

| 文件 | 作用 | 验证状态 |
|---|---|---|
| `e204e205_common.py` | 27-case 源加载（复用 E178 production 过滤）、arm 定义、路径、逐 case task/base 推导（含 v2 case）；import A2_GATE/assert_gravcomp_diff(E198/E200)、HAND/LOWER_BODY_GEOMS(E175)、CEM 冻结量(E198) | ✅ import + 27 源 9/4/14 通过 |
| `build_arm_scenes.py` | 复用 E178 `build_scene` 再生 canonical E178 场景（byte-parity）；派生 E205 gravcomp（`assert_gravcomp_diff` 单变量）+ E204 noPRG（删 80 腿↔物，断言 0 残留 / 10 手对留存 / gravcomp 0）；快照(rule 10b) | ✅ 27×2 场景 + 快照生成 |
| `build_overrides.py` | 写 27×2 override YAML + Hydra compose 审计（E204: penalty=0/gate off/union/hand-gate=E163默认；E205: hand-gate==A2_GATE/PRG on/union） | ✅ 27×2 审计 PASS |
| `run_e204e205_cem.py` | 54=27×2 full CEM 8-GPU 驱动（一卡一 slot、skip-already-done、**逐 arm 独立 output_dir 防碰撞**、KEEP_GOING、summary json） | ✅ dry-run 54 job；canary 待确认 |
| `../../launch/active/run_E204_E205_8gpu.sh` | 自包含入口：STEP0 建场景+快照 → STEP1 建+审 override → STEP2 54 full CEM | ✅ 已写 |
| scene 快照 | `results/E204/scene_snapshot/` + `results/E205/scene_snapshot/`（git HEAD + sha256） | ✅ 265 行 manifest |

**输出路径**：`results/E{204,205}/s6_downstream/cem/full/E{204,205}_<case>_{noPRG,G1A2}/trajectory_mjwp_act.npz`（逐 arm 隔离，skip 检测键此文件）。

## Claims 与量化成功标准

- **C0 数据复用正确**：两 arm 轨迹 + 3cm 掩码 == E178 对应 case（复用 dcv3 base，0 新上游 retarget/轨迹）。
- **C1 arm 单变量正确**：E204 场景 0 腿↔物对 + composed penalty=0/gate false + 五段代理保留 + gravcomp 0；E205 gravcomp 单变量 diff + hand-gate==A2_GATE + PRG on + 五段代理 union 保留。**（build 期已断言，27×2 override 审计 PASS）**
- **C2 执行闭合**：54 条 full CEM 完成、打分成功，error/non-finite/diverged/fall=0；infeasible 记录不静默。
- **C3 三 arm paired（核心）**：27 case、逐物体（bucket003/004/007）、full 分布（mean+std+worst），6 门（fall / object_pos≤20cm / object_ori≤10° / contact≥0.50 / hand_pen≤0.30 / leg_pen≤0.10）逐门 paired 通过率（noPRG vs PRG=E178 vs G1A2）。
- **C4 per-object 推荐 arm**：每物体 arm 排名 + 推荐，附依据。
- **C5 视觉复核（强制 rule 9）**：每物体每 arm ≥2 case 三 arm side-by-side，无穿模/漂浮/抖动/跌倒；观察入 log。

**判定**：C0–C2 全过 + C3 三 arm workbook 产出 + C4 per-object 推荐 + C5 无致命 artifact → 达成。

## 执行命令

```bash
# 用户在 8 卡机（先 git pull，.venv 就绪）：一键跑全部 54 条
bash workspace/core4d/scripts/launch/active/run_E204_E205_8gpu.sh
# 冒烟：LIMIT=1 NUM_SAMPLES=64 MAX_ITERS=4 bash .../run_E204_E205_8gpu.sh
# 若 egl 崩：E204E205_MUJOCO_GL=osmesa bash .../run_E204_E205_8gpu.sh

# 结果回共享盘后（本会话或后续）：三 arm 评估
bash workspace/core4d/scripts/eval/wrappers/eval_E204E205_arm_ablation.sh   # 待建
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E204E205_three_arm_workbook.py  # 待建
.venv/bin/python workspace/core4d/scripts/experiments/E204_E205/render_qc.py  # 待建，三 arm side-by-side
```

## 风险 / 缓解

| 风险 | 缓解 |
|---|---|
| G1 gravcomp 在 bucket（质量 2–120kg，vs box 5kg）效应差异大 | 逐物体报告；单变量断言；重物体重点视觉复核 |
| 五段代理 × G1/A2 在 bucket 上未验证（此前只 box） | canary 先冒烟；C5 视觉强制 |
| noPRG 漏删/误删腿↔物对 | `base.is_robot_object_pair` 精确识别，断言删 16×N、留 2×N 手对、gravcomp 0 |
| E204/E205 共享 task 目录，rollout 覆盖 | 逐 arm 独立 `output_dir`，skip 键各自 npz |
| v1/v2 混淆 | 固定复用 E178（26 v1 + 1 v2 rescue），log 显式标注；不与 E203 core4d_v2 混算 |
| 1 case（055_p1）为 v2-rescue | base override 逐 case 从 source 推导，已在 override 审计中验证正确继承 v2 base |

## 下一步

- 用户执行 54 full CEM → 结果回收 → 建 eval/workbook/render_qc → 写 log292 + 更新 tracker（R289/R290）。
- 下游 RL：三 arm（noPRG/PRG/G1A2）× 27 bucket 数据训练对比，选 arm。
