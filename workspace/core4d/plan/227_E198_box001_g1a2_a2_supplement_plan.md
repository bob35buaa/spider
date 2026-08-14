# plan227 · E198 补充：box001 的 G1+A2 + A2（E196 修正参考基线）

_Core4D · Phase 61 · 2026-08-13 · 承接 [plan226](226_E198_g1xa2_factorial_and_E192_a2_expansion_plan.md) / [log284](../log/284_E198_g1xa2_factorial_results.md) · evaluator `core4d-e154-physics-contact-v1`_

## 0. 一句话

把 E198 的 G1×A2 完整 2×2 因子从 4 物体（box004/021/023/024）**扩到第 5 个物体 box001**，只新跑 box001 的 **G1+A2（28）+ A2（28）= 56 条 Full CEM**，A0/G1 基线**复用历史 rollout 并在 E196 修正参考下重打分**。**实验号仍为 E198，不新开。**

## 1. Context / 背景与动机

- E198（log284）已刻画 4 物体的 G1×A2 交互：两干预**非可加、主要相互抵消副作用**；obj_ori 上呈**双向物体特异救援**（box023 A2 救 G1、box004 G1 救 A2）。box001 是历史上朝向长尾最严重的物体，是检验"救援方向是否可预测"的关键第 5 点（见 log284 §5.4 下一步）。
- **box001 从未有 A2 / G1+A2**：A2 只在 box024/004（E192）+ box021/023（E192-ext）；G1+A2 只在 box004/021/023/024（E198）。
- **参考基线的关键约束（用户已定）**：box001 的 28 例中，**21 例**在 E194 有 runtime/compiled **Euler convention 错配**，orientation 参考被污染；[E196](../log/278_E196_reference_metadata_integrity_fix_results.md) 已修复 `scene_act_meta.json`（从 compiled hinge axes 解析约定、缺失即 hard-fail）并**重跑 corrected G1**（21 例，`seed=0/1024×32`），证明 corrected G1 在 obj_ori 与 obj z/3D 上优于 contaminated G1、strict 12-gate 通过数 6→10。其余 **7 例**在 E194 本就干净。
- **用户三项决策（本次 AskUserQuestion）**：
  1. **参考版本 = E196 修正版**：21 例用 E196 corrected 参考元数据、7 例用 E194 干净版；新跑的 A2/G1+A2 全部对齐修正后参考。
  2. **基线 = 复用+重打分**：复用 PRG(E173) + G1（21 E196 corrected rollout + 7 E194 clean rollout），用同一公共 evaluator 在修正参考下重打分，**不重跑基线**；仅新跑 56 条。
  3. **范围 = 全部 28 例**（21 受影响 + 7 干净）。

## 2. 冻结不变量（沿用 E198，见 plan226）

| 项 | 冻结值 |
|---|---|
| G1 干预 | object body `gravcomp="1"`，`kp_pos=500`、`kp_rot=50`（scene sidecar） |
| A2 干预 | `cem_hand_gate_min_sdf_m=-0.010` / `cem_hand_gate_max_violation_pct=0.05` / `cem_hand_gate_hard_floor_m=-0.015`（config override，no-gravcomp 时即 A2；叠 gravcomp 即 G1+A2） |
| A0 基线 gate | `-0.010 / 0.10 / -0.020`（PRG，no-gravcomp） |
| CEM | `seed=0`、`num_samples=1024`、`max_num_iterations=32`、Full |
| retarget variant | 逐 case 保留（box001 v1=21 / v2=7） |
| evaluator | 公共 `eval.core.core_metrics`，`core4d-e154-physics-contact-v1` |
| **参考元数据** | **E196 修正版**：21 例用 corrected `scene_act_meta.json`（`reference_fix_case_authority.tsv` 权威）、7 例用 E194 原始（本就正确） |

**单变量纪律**：G1 与 A0/A2 之间**唯一差异**是 object gravcomp（+kp）；A2 与 A0 之间唯一差异是 3 个 hand-gate 字段。scene physics XML（`scene_act_E194_G1_expansion_rubberHull_PRG_gravcomp.xml`）在 21 例上与 E196 corrected G1 **逐字节一致**（修正只在 meta sidecar，不动 physics）。

## 3. 执行范围与优先级

**新跑 56 条 Full CEM**（本地 8 卡 priority 队列，与他人 job 叠加共跑、**不 kill/不抢占**，查空闲显存派发）：

| Tier | 臂 | Cases | 干预 | 结果目录 |
|---|---|---|---:|---|
| **P0-b1**（先跑） | G1+A2 | box001 ×28 | gravcomp + A2 gate | `results/E198/s6_downstream/cem/full_g1a2_box001/` |
| **P1-b1**（后跑） | A2 | box001 ×28 | no-gravcomp + A2 gate | `results/E192/s6_downstream/cem/full_a2_box001/` |

- 优先级**在现有 E198 P0–P3 之后追加**：本补充自身内部 **G1+A2（P0-b1）先于 A2（P1-b1）**，与用户"先跑 G1+A2 后跑 A2"一致。
- GPU：`PER_GPU_MEM_MIB=5000`、每卡 1 run（沿用 E198 默认），8 卡 0–7。

**复用+重打分（不占新 GPU）**：
- **A0(PRG)**：28 例 E173 PRG rollout。
- **G1**：21 例 E196 corrected G1 rollout（`results/E196/.../full_reference_fix/E196_box001_*_G1_reference_fix_outdir/trajectory_mjwp_act.npz`，已验证 0 缺失）+ 7 例 E194 clean G1 rollout（`results/E194/.../full_g1_expansion/`）。
- 4 臂统一用公共 evaluator 在修正参考下重打分为 by_case / by_object。

## 4. 关键前置（reproducibility，rule 7 / rule 10b）

1. **修正 meta 恢复+校验**：对 21 例，从 `report/E196/provenance/scene_snapshot/reference_fix/{case}/` 恢复/核验 corrected `scene_act_meta.json` 到 live `example_datasets` 路径，sha256 对齐 `reference_fix_case_authority.tsv`；7 例校验 E194 meta 已正确。**新跑 A2/G1+A2 前必须确保 live scene 的 meta 是修正版**（否则新 arm 又被污染，破坏 C1/C3）。
2. **单变量审计**：G1+A2 vs A2 的 scene 仅差 gravcomp（ET 签名 diff）；A2 vs A0 仅差 3 个 gate 字段；`A2_GATE==E192`。
3. **SHA parity**：trajectory / scene / override / contact_mask 逐 case sha256 记入 `e198_box001_authority.tsv`。
4. **scene 快照**：56 条用到的 scene XML + meta 快照到 `results/E198/scene_snapshot/g1a2_box001/` 与 `results/E192/scene_snapshot/a2_box001/`，含 `manifest.txt`（git HEAD + sha256）。
5. **C3 baseline parity**：box001 A0+G1 重打分 vs **E196 corrected 冻结表**逐 case 核验（corrected G1 的 obj_ori 应与 E196 报告一致，非 E194 contaminated 值）。

## 5. Claims / 可验证声明

| Claim | 判据 |
|---|---|
| C0 scope/provenance | box001 ×28 唯一 case；A0/G1/A2/G1+A2 四臂来源全可追溯（E173/E196+E194/新跑/新跑）；SHA parity 全过 |
| C1 单变量 intervention | 56 新 run：G1+A2 单变量 gravcomp 审计过；A2 仅改 3 gate 字段；`A2_GATE==E192`；21 例 meta==E196 corrected |
| C2 execution/numeric closure | 56/56 Full 完成；box001 四臂 112/112 scored；error/non-finite/diverged=0 |
| C3 baseline parity（修正基线） | box001 A0+G1 重打分 vs E196 corrected 冻结表逐 case z/obj_ori 差 `≤1e-6`（**对齐 corrected，不是 contaminated**） |
| C4 交互项估计 | box001 2×2 完整；六指标 INT+95%CI+四主效应；并入 5 物体 by_object |
| C5 承重接触保留 | box001 G1+A2 in-mask contact vs A0 不显著回退（阈值同 E198 C5） |
| C6 物理安全 | box001 无新增 fall/non-finite/diverged |
| C7 gate migration | box001 4 transition × 12 门 P→F/F→P + exact McNemar |
| C8 device confound | 队列按空闲卡派发，落卡 GPU id 逐 case 记录 |
| C9 证据闭合 | 数值/gate/interaction 闭合；4-cell MP4 + viser（`--arm` 可筛）交互复核 |

## 6. 交互项与假设

- 主关注 **obj_ori**：在**修正参考**下检验 box001 的救援方向。假设 H1：box001 与 box023 同属"G1 单用抬高朝向误差、A2 纠正"（A2 救 G1，INT<0）；H0：无显著交互。用 paired bootstrap 95% CI + exact McNemar 判定。
- 其余指标预期与 E198 一致：z 由 G1 独占（INT≈0）、穿透次可加。
- **决策**：本补充**不改变 A2/G1+A2 的 governance**（仍为诊断性因子探索）；仅把 5 物体的交互结构补全，更新 log284 判决为 5 物体版 `FACTORIAL_CHARACTERIZED`。

## 7. 改动文件（待批准后实现）

| 文件 | 改动 |
|---|---|
| `scripts/experiments/E198/e198_common.py` | 新增 box001 expansion 分支 + corrected-meta 解析（21 corrected / 7 clean）；`build_cells()` 追加 box001 56 cell |
| `scripts/experiments/E198/build_g1a2_manifest.py` | 生成 `e198_box001_authority.tsv` + `e198_box001_*_manifest.tsv`；meta 恢复/校验；单变量审计；快照；git add -f |
| `scripts/experiments/E198/run_local_priority_queue.py` | 消费 box001 tier（P0-b1 G1+A2 先、P1-b1 A2 后）；resume-safe |
| `scripts/launch/active/run_E198_local_8gpu.sh` | 增加 `SCOPE=box001` 入口（canary→full） |
| `scripts/eval/runners/eval_E198_factorial.py` | `arm_rows()` 增加 box001：A0→E173、G1→E196 corrected(21)+E194(7)、A2→E192 box001 manifest、G1A2→E198 box001 manifest |
| `scripts/eval/reports/gen_E198_xlsx.py` | 5 物体（box001 加入 `OBJ_ORDER`/`OBJECTS`）；逐 case sheet 含 box001 |
| `scripts/experiments/E198/render_g1a2.py` | box001 4-cell（`--all` 自动含，续跑守卫已在） |
| `log/284`（更新）/ `log/283`（A2 box001 并入）/ 新 log 条目 | 5 物体结果回填 |

## 8. 运行命令（待批准）

```bash
# 1. 构建 box001 authority + manifest + meta 恢复/校验 + 单变量审计 + 快照
MUJOCO_GL=osmesa .venv/bin/python workspace/core4d/scripts/experiments/E198/build_g1a2_manifest.py \
  --scope box001 --apply --snapshot
# 2. 本地 8 卡 priority 队列（canary 后 Full；G1+A2 先、A2 后）
MODE=canary SCOPE=box001 bash workspace/core4d/scripts/launch/active/run_E198_local_8gpu.sh
MODE=full  SCOPE=box001 GPUS=0,1,2,3,4,5,6,7 PER_GPU_MEM_MIB=5000 MAX_PER_GPU=1 \
  bash workspace/core4d/scripts/launch/active/run_E198_local_8gpu.sh
# 3. 5 物体 factorial eval + xlsx + report
bash workspace/core4d/scripts/eval/wrappers/eval_E198_factorial.sh
PYTHONPATH=workspace/core4d/scripts .venv/bin/python workspace/core4d/scripts/eval/reports/gen_E198_xlsx.py
# 4. 渲染 + viser 复核（G1+A2 only）
MUJOCO_GL=osmesa PYTHONPATH=workspace/core4d/scripts .venv/bin/python \
  workspace/core4d/scripts/experiments/E198/render_g1a2.py --all
bash workspace/core4d/scripts/eval/wrappers/review_player.sh E198 --arm G1A2 --port 8082
```

## 9. 成功标准（量化）

- **执行**：56/56 Full 完成、0 失败；box001 112/112 scored、0 error。
- **C3**：box001 A0+G1 重打分对齐 E196 corrected 表，obj_ori 差 `≤1e-6°`（若对齐到 contaminated 值则判 FAIL，说明 meta 未修正）。
- **C4**：box001 六指标 INT+CI 全部产出；obj_ori 的 H1/H0 判定明确（CI 是否含 0）。
- **可视化**：box001 28 例 4-cell MP4 + 逐例 12-gate；无摔倒/飞散/脱手。
- **结论**：5 物体交互结构写入 log284；A2/G1+A2 是否升级的 governance **维持不变**，除非 box001 出现与 4 物体定性相反的强证据（届时单独讨论）。

## 10. 风险与红线

- **meta 污染风险**（最高）：若新跑时 live scene 的 `scene_act_meta.json` 不是 E196 修正版，新 arm 会重新引入 Euler 污染 → C1/C3 FAIL。前置步骤 4.1 强制恢复+校验，且训练脚本首步快照。
- **不 kill/不抢占**他人 GPU job（沿用 E198 C8 纪律）。
- **不重跑基线**（用户已定复用）：若发现 E196 corrected rollout 与其冻结表打分不一致，停下报告，不擅自重跑。
- **未获批准前不写脚本、不占 GPU、不改 scene/meta。**
