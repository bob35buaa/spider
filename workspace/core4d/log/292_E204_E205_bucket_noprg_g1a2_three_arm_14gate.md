# log292 · E204(noPRG)/E205(G1A2) vs E178(PRG) — 三 arm 14-gate 对比 (27 bucket)

_Core4D · Phase 65 · Run R289(E204 noPRG) + R290(E205 G1A2) · 承接 [plan234](../plan/234_E204_E205_bucket_noprg_g1a2_arm_ablation_plan.md) · 2026-08-24 · 分支 `experiment/E203-core4d-v2-orig-retarget`_

## Context

复用 E178 的 27 bucket case（bucket003=9/004=4/007=14），在**同 contactAlignedTop 五段物体代理 + omnirt_v1 ref_fk 轨迹 + 3cm 掩码 + CEM 1024×32 seed0** 下，只改 reward arm，做三 arm 对比：**noPRG(E204) / PRG(E178,复用) / G1A2(E205)**。用户在 4 卡机跑完 54 条 full CEM（`E204E205_MUJOCO_GL=disable` + `+use_torch_compile=false`，该机无 GL 库、无 python3.12-dev）。

## 执行 / 数据

- CEM：54/54 `cem_ok`（E204 27 + E205 27），0 fail。
- 评测口径：**E201 14-gate 漏斗**（4 硬门 fall/body_z≤0.20/ankle_jerk<1000/obj_speed<3 + 10 带门 wide/narrow）。PRG 读 E178 canonical `e178_case_metrics.tsv`（同 public core_metrics）；E204/E205 用 `evaluate_sequence + run_health + body_z_p95` 现评。同一尺子。
- 产物：`results/E204/s6_downstream/eval/three_arm/three_arm_rollout.tsv`（81 行）+ `E204E205_three_arm_14gate.xlsx`（README/14-Gate Summary/Per-Object/Per-Case/Dist&Paired 5 sheet）。

## 结果（14-gate, narrow 口径, n=27）

| arm | hard | wide_all | **narrow_all** | L3_auto |
|---|---|---|---|---|
| PRG (E178) | 21/27 | 12/27 | **9/27 (33%)** | 9 |
| G1A2 (E205) | 23/27 | 9/27 | **8/27 (30%)** | 8 |
| noPRG (E204) | 22/27 | 11/27 | **7/27 (26%)** | 7 |

**逐门 narrow 通过计数（关键差异）**：

| gate | PRG | noPRG | G1A2 | 读法 |
|---|---|---|---|---|
| **leg_pen** | 27 | **24** | 27 | noPRG 输在腿穿透（mean 0.083 vs PRG 0.021，4×；paired 15/27 case 更劣）——与 E200 box 结论一致 |
| obj_ori | 25 | 27 | 26 | noPRG 略优；G1A2 mean 最低(4.92° vs 5.63°, paired 20/7 胜) |
| root_ori | 22 | 20 | 23 | G1A2 最好（G1 gravcomp 稳姿态） |
| ankle_jerk | 22 | 22 | 24 | G1A2 最平滑 |
| eef_pos | 17 | 17 | 19 | G1A2 略优 |
| contact | 24 | 24 | 23 | PRG/noPRG 略高；G1A2 gravcomp 把物体略拉离手→接触−0.013 |
| release | 21 | 23 | 22 | noPRG 略优 |
| 其余(fall/body_z/obj_speed/obj_pos/hand_pen/root_pos) | | | | ±1~2 内近似 |

**paired 分布（vs PRG，mean / win-lose）**：
- obj_pos：noPRG −0.18cm(19/8), **G1A2 −0.91cm(22/5)** ——G1A2 物体位置最好。
- obj_ori：G1A2 −0.71°(20/7) 最好。
- hand_pen：G1A2 −0.016(12/10) 最低。
- **leg_pen：noPRG +0.062(仅2/27胜,15劣)** ——noPRG 显著更差；G1A2 −0.008 与 PRG 持平。
- **noPRG 存在灾难性离群**：root_pos worst **145cm**、eef_pos worst 148cm、body_z worst 0.82（std 24.8）——个别 case 腿穿透→根跟踪崩，拖低严格通过率。PRG/G1A2 无此离群（root_pos std ~9）。

**逐物体 narrow_all**：bucket003(凹,最难) noPRG1/PRG2/G1A2 1；bucket004 三者均 2/4；bucket007 noPRG4/PRG5/G1A2 5。

## Claims 验证

- C0 数据复用 ✅（同 omnirt_v1 轨迹+掩码，0 新 retarget；1 例 055_p1 为 omnirt_v2 rescue，base 逐 case 推导）。
- C1 arm 单变量 ✅（build 期断言 + 27×2 override compose 审计 PASS；config_act 确认 E204 penalty0/gate off、E205 penalty2.0/gate on/gravcomp1/hand-gate=A2）。
- C2 执行闭合 ✅（54/54 cem_ok, 0 fail/diverge）。
- C3 三 arm paired ✅（本 log 表 + xlsx）。
- C4 per-object 推荐 ✅（见下）。
- **C5 视觉复核 ⏳ 未完成**（rule 9）：viser 三 arm 并排回放脚本 `viser_replay_arms.py` 已实现并验证可启动（server 绑定、场景构建、arm 加载均通过；qpos (T,2,42)→取 sim 切片已修）；但当前机器被 8 个外部 python@100% CPU 饱和（load~17），重导入/FK 进程被饿死/杀（exit144），**全量渲染未跑成**。待机器空闲或在 4 卡机运行补 C5。

## 结论 / per-object 推荐

1. **leg_pen 是 PRG 的决定性优势**：noPRG 去掉腿-物约束后腿穿透 4×，并诱发个别 case 根跟踪灾难性发散 → 严格 14-gate 最低(26%)。**下游若对腿穿透敏感，不推荐 noPRG。**
2. **G1A2 = 最佳物体跟踪 + PRG 级腿安全**：G1 gravcomp 显著改善 obj_pos/obj_ori/root_ori/ankle_jerk、hand_pen 最低，leg_pen 与 PRG 持平；代价是接触略降(−0.013)。严格通过 30%，仅略低于 PRG 33%。
3. **PRG(E178) 严格通过最高(33%)**：接触保持最好(0.705)、无离群。
4. **推荐**：bucket003/004 三者接近（004 全 2/4，003 全弱）；bucket007 PRG≈G1A2>noPRG。**综合：物体跟踪优先选 G1A2；接触保真/最稳选 PRG；noPRG 不推荐（腿穿透+离群）。** 与 E200 box 三 arm 结论方向一致。

## 结果路径

- CEM：`results/E{204,205}/s6_downstream/cem/full/E{204,205}_<case>_{noPRG,G1A2}/trajectory_mjwp_act.npz`
- 评测：`results/E204/s6_downstream/eval/three_arm/{three_arm_rollout.tsv, E204E205_three_arm_14gate.xlsx}`
- 场景快照(rule 10b)：`results/E{204,205}/scene_snapshot/`
- 脚本：`scripts/eval/runners/eval_E204E205_arm_ablation.py`、`scripts/eval/reports/gen_E204E205_three_arm_workbook.py`、`scripts/experiments/E204_E205/viser_replay_arms.py`

## 下一步

- 补 C5 视觉复核（机器空闲后跑 viser 三 arm 并排；每物体 ≥2 case，重点看 noPRG 的 leg 穿透离群 case + G1A2 的接触是否目视变松）。
- 下游 RL：三 arm × 27 bucket 数据训练对比选 arm（倾向 G1A2 或 PRG）。
