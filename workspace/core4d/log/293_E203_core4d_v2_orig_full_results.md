# E203 — CORE4D v2 人体动作 orig 重定向 + full CEM 全量结果（box + bucket）

- **日期**: 2026-08-25
- **分支**: `experiment/E199-omniretarget-object-augmentation`（E203 提交其上）
- **计划**: `workspace/core4d/plan/233_E203_core4d_v2_orig_retarget_plan.md`
- **P1 阶段日志**: `workspace/core4d/log/292_E203_p1_core4d_v2_orig_cem_results.md`
- **运行根**: `workspace/core4d/results/E203/`（**不进 git**，外部同步盘；数据树 `example_datasets/processed/core4d_v2/` 也 **不进 git**，按用户约定）

## 目的

用 **v2 人体动作**（`CORE4D_Real_human_object_motions_v2`，同捕捉、SMPLX 重拟合更精）替代 v1，走完整 data-construction-v3 管线到 full CEM，产出与 v1 **物理隔离** 的一版数据（`dataset_name=core4d_v2`），orig（不做 object augmentation）。范围：box001/004/021/023/024/026 + bucket 类（排除 bucket006/008）。

## 管线与隔离

- v2 raw 物化 → v1 布局（复用 v1 物体位姿，逐帧对齐）；输出树 `core4d_v2`（assets/模板 symlink 自 core4d，输出独立）。
- retarget：`omnirt_v1`（默认）→ CVXPY 不可行 case 用 `omnirt_v2`（Phase-4 松弛）**rescue**。
- CEM：复用 E199 PRG 契约（rubber_hull 手 + 16 下肢 pair + leg gate），1024×32，**8 卡 task 级并行**，torch.compile 开（osmesa）。
- **评估**：冻结 E173 评估器 + E201 14-gate funnel（`run_E203_p1_eval.py --object-keys ""`，全 171 case，不重仿真），输出 `results/E203/s6_downstream/eval_full/`。

## 漏斗产量（全量）

| 阶段 | 数量 |
|---|---|
| SPIDER 落地任务（core4d_v2 树） | 195 |
| CEM 成功（`cem_ok`，有 `trajectory_mjwp_act.npz`） | **171**（omnirt_v1 137 + omnirt_v2 rescue 34）|
| PRG 预检拒绝（起始帧腿-物穿透 > 硬底） | 24（P1 10 + rest 14）|

**分物体 CEM 完成数**：box001 35 / box021 30 / box023 18 / box024 19 / box026 12 / bucket001 3 / bucket003 6 / bucket005 4 / bucket007 42 / bucket010 2。

**掉出范围**：box004（0，v2 未过 3cm 接触门）；bucket004 / bucket009（0，无可行重定向/接触样本，整体 drop）；bucket006/008（计划内排除）。

## 结果（171 case，全量分布，不 cherry-pick）

### 分物体核心指标

| 物体 | n | obj_pos(cm) mean±std / worst | eef_pos(cm) | fall | leg_pen frac | 6-gate 过 |
|---|---|---|---|---|---|---|
| box001 | 35 | 11.9±3.3 / 19.0 | 33.4 | 2 | 0.068 | 15 |
| box021 | 30 | 15.3±4.8 / 32.5 | 20.6 | 0 | 0.066 | 17 |
| box023 | 18 | 13.5±4.1 / 26.6 | 17.6 | 1 | 0.045 | 11 |
| box024 | 19 | 13.0±2.9 / 19.1 | 23.9 | 1 | 0.083 | 9 |
| box026 | 12 | 16.8±6.5 / 30.2 | 31.7 | 0 | 0.052 | 8 |
| **bucket001** | 3 | **81.7±19.2 / 108.2** | 84.4 | 0 | 0.180 | 0 |
| bucket003 | 6 | 9.0±1.6 / 11.6 | 11.8 | 0 | **0.387** | 0 |
| bucket005 | 4 | 10.9±1.7 / 13.5 | 14.3 | 0 | 0.011 | 3 |
| **bucket007** | 42 | 9.8±4.2 / 25.9 | 15.9 | 0 | **0.312** | 6 |
| bucket010 | 2 | 7.6 / 7.7 | 10.3 | 0 | 0.135 | 1 |
| **全部** | **171** | **13.7±10.5 / 108.2** | — | 4 | — | **70 (41%)** |

- 变体：omnirt_v1 obj_pos 12.2±6.2 / worst 63.4；omnirt_v2 rescue **19.3±19.1 / worst 108.2**（rescue 集是 v1 难解硬 case，明显更差，符合预期）。

### 释放门 & 14-gate funnel

- **6-gate 数值门**：70/171 过（**41%**）。失败模式：**lower_body（腿穿透）67**、**contact（接触占比不足）53**、hand_penetration 12、body_z 11、release 8、fall 4。
- **14-gate（E201 funnel）**：L1_reject 118 / L2_review 16 / **L3_auto 37**（strict narrow 全过 = 37/171，22%）。

### 视觉 QC（强制，抽检 3 代表 case）

- **box001（108_p2，obj_pos 11.8）**：机器人俯身、双手压箱顶、箱体直立贴合,sim 跟 ref 姿态一致,**无穿透/跌倒,物理可信**。✓
- **bucket001（v2 rescue 010_p1，obj_pos 108cm）**:sim 侧机器人上身塌陷前扑、头near地,**姿态崩溃/跟踪失败**。✗
- **bucket007（007_p2，leg_pen 0.91）**:大桶,sim 侧机器人**小腿/膝明显插入桶体**（91% 帧腿穿透）——obj 跟踪虽好(8.6cm)但腿-物穿透严重,物理不可信。✗

### 与 v1 对照（同物理 case，matched 75/171；v1 基线 E170 26 + E173 49；仅 box001/box024 有配对）

| 指标 | 方向 | E203 v2 mean | v1 mean | Δ | improved/n |
|---|---|---|---|---|---|
| track_obj_pos_err_cm | ↓ | 13.0 | 13.34 | **−0.34** | 40/75 |
| track_obj_ori_err_deg | ↓ | 5.70 | 5.89 | −0.19 | 40/75 |
| track_root_pos_err_cm | ↓ | 20.9 | 18.89 | **+2.01** | 35/75 |
| track_eef_pos_err_cm | ↓ | 19.62 | 18.14 | **+1.48** | 35/75 |
| hand_pen_3mm_frame_frac | ↓ | 0.139 | 0.207 | **−0.068** | 52/75 |
| hand_contact_in_mask_frac | ↑ | 0.705 | 0.751 | **−0.046** | 27/75 |
| release_false_contact_frac | ↓ | 0.079 | 0.057 | +0.021 | 13/72 |
| body_z_err_p95_m | ↓ | 0.117 | 0.125 | −0.009 | 43/75 |
| leg_penetration_frac | ↓ | 0.063 | 0.072 | −0.009 | 25/75 |

> 注:matched 集里 v2 侧混入了 omnirt_v2 rescue 的硬 case,对 v2 略不利(非严格同求解器口径)。

## 结论

1. **端到端跑通**:v2 人体动作 物化→converter→OmniRetarget→SPIDER→PRG→CEM 全链在 `core4d_v2` 隔离树内正确运行,v1 树未污染。171 full-CEM 结果,`omnirt_v2` rescue 贡献 34 个可行任务。
2. **box 是本轮真正可交付的数据**(121 rest + 54 P1):obj_pos **12–17cm**、fall≈0、视觉干净;box001/024 与 v1 **打平**——在手-物穿透(−6.8pp)、obj/body_z 上略优,在 root/eef 位置跟踪上略差(+1.5–2cm)。**"v2 SMPLX 更干净" 未转化为 CEM 层面的决定性提升,总体持平**。
3. **bucket 本轮不达交付标准**:
   - bucket003/007 **腿-物穿透 0.31/0.39**(个案高至 0.91)——粗粒度 E199 单 object_collision PRG 挡不住腿插大桶(这正是选 E199-而非-E202 per-seg union 时预警的风险)。
   - bucket001 **跟踪崩溃**(obj_pos 73–108cm,姿态塌陷)。
   - 仅 bucket005/010 勉强可用(n 少)。
4. **绝对质量**:全量 6-gate 41% / 14-gate strict 22%——受 bucket 腿穿透与接触不足拖累;剔除 bucket 后 box 组明显更高。

## 预声明成功判据回看

- 判据:v2 同 case S4/CEM 关键指标(obj_pos、腿穿透、fall)**不劣于 v1**。
- 结果(matched box 集):obj_pos −0.34cm(略优)、leg_pen mean −0.009(略优)、fall 持平 → **"不劣于" 达成**;但 root/eef 位置 +1.5–2cm(略劣)。**判定:box 组达到"不劣于 v1"(持平),但无显著增益;bucket 组不达标。**

## 未决 / 下一步

- **bucket 重跑**:用 E202 per-segment union PRG(bucket003/004/007 有定义)重做 bucket CEM,压腿穿透;bucket001 需查 proxy/模型或从范围剔除。
- **box004**:v2 未过接触门,若需覆盖要放宽 3cm 门限或核对物体位姿。
- **A/B 严格化**:matched 对照混入 rescue,若要干净结论需对同 case 同求解器(仅 omnirt_v1)配对。
- box 数据可直接作为下游 RL export 候选(质量与 v1 持平)。

## 交付物 / 产物

- 数据:`example_datasets/processed/core4d_v2/unitree_g1/humanoid_object/dcv3_omnirt_v{1,2}_ref_fk_*`(195 任务,171 含 CEM 输出;**不进 git**)。
- 评估:`workspace/core4d/results/E203/s6_downstream/eval_full/`(`e203_case_metrics.tsv` 171 行、`summary.json`、`e203_vs_prg_comparison.tsv`、`e203_funnel_rollout.tsv`、`scene_snapshot/` 172 项)。
- CEM evidence:各任务 `0/trajectory_mjwp_act.npz` + `visualization_mjwp_act.mp4`(171 视频)。
- 脚本:`run_e203_cem.py`(8 卡调度)、`run_E203_p1_eval.py`(全量评估,`--object-keys ""`)、`run_E203_rest_all.sh`(P2/P3/P4 自动链)。
- 环境:`python3.12-dev`(torch.compile)、`MUJOCO_GL=osmesa`。
