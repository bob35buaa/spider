# E153 — 轴1 杠杆 A 续：CEM hand gate 阈值扫（min_sdf × max_violation，先解耦再扫）

## 0. 背景与动机

E152 证明 CEM hand gate（轴1）能压手-物体物理深穿透，gateA_b1（reward B1 + gate）在 3 case 上深穿透 `con<−5mm` 全降、近场接触保持/上升；引入 2mm/5mm 深度分档几何穿透后，gateA_b1 在深度感知判据下 **3/3 strict**。但 E152 暴露两个待解问题：

1. **`max_violation_pct` 是死旋钮**（自 E088 身体 gate 起一直惰性）：`sampling.py:_build_gate_masks` 的 `valid_mask = (min_sdf ≥ min_sdf_m) AND (violation_pct ≤ max_violation_pct)`，两项共用同一 `min_sdf_m` → 第一项（硬地板，任一帧深于阈值即废）已强制 violation_pct=0 → 第二项恒真，`max_violation_pct` 不起作用。要扫它必须先解耦。
2. **gate 阈值未调优**：E152 固定 `min_sdf_m=−0.010 / max_violation_pct=0.05`。box023 gateA_b1 fallback 升到 0.096、valid 掉到 0.769（逼近 R1 退化边缘），box021 gateA 在 2mm 档几何穿透还微升 +0.013——说明阈值对不同 case 偏紧/偏松不一，需要按网格扫出甜点。

E153 = 先修 gate 解耦（让 max_violation_pct 真正生效），再在 3 case 上扫 `min_sdf_m × max_violation_pct`，目标把 gateA_b1 推到稳定 3/3 strict + gate 健康（无 fallback 退化、无塌陷）。

## 1. Claims（事前定义，可证伪）

- **C1（解耦正确性，前置）**：Stage0 解耦后，
  - (a) **回归**：`cem_hand_gate_hard_floor_m` 未设（默认 sentinel → =min_sdf_m）时，E152 某 gateA_b1 case 重算 valid_mask/CEM 结果**数值不变**（钉死纯增量，E088–E152 不受影响）。
  - (b) **激活**：设 `hard_floor_m=−0.020`（深于所有扫描 min_sdf_m）后，固定 min_sdf_m、只变 max_violation_pct，`cem_hand_gate_valid_frac` / 结果**随之变化** → 证明旋钮已生效。
- **C2（主判据，调优目标）**：存在某 `(min_sdf_m, max_violation_pct)` 设定，使 gateA_b1 相对 b1（E151 复用）在**全 3 case** 上同时满足 **深度感知 success**（`success_pen2mm_down_contact_keep`=true，即 2mm 档真穿透 Δ≤0 + near-5cm Δ≥−0.02 + obj_err Δ≤0.02 + 不摔），且物理深穿透 `con<−5mm` 不回升、物理接触不掉。即 3/3 strict。
- **C3（gate 健康）**：该设定下 3 case 的 `cem_hand_gate_valid_frac` 不长期塌到 `min_valid_frac`、`cem_gate_fallback_used` 低（目标 ≤0.05），修掉 E152 box023 的 fallback=0.096 边缘问题。

## 2. Stage0：gate 解耦（隔离、可逆、默认关、回归钉死）

改 `spider/config.py` + `spider/optimizers/sampling.py`（及 `sampling_fast.py` 同步）：

- config 新增（身体 + 手 gate 各一，默认 sentinel `nan` → 回退旧行为）：
  - `cem_safety_gate_hard_floor_m: float = float("nan")`
  - `cem_hand_gate_hard_floor_m: float = float("nan")`
- `_build_gate_masks.add_gate(...)` 增参 `hard_floor_m`，逻辑改为：
  ```
  floor = hard_floor_m if math.isfinite(hard_floor_m) else min_sdf_m   # nan → 旧行为
  valid_mask = (min_sdf >= floor) AND (violation_pct <= max_violation_pct)
  ```
  - `floor == min_sdf_m`（默认）：与现状完全等价（max_violation 仍惰性）→ **E088–E152 数值不变**。
  - `floor` 设更深（如 −0.020）：第一项放宽为"绝对单帧底线"，`violation_pct ≤ max_violation_pct` 成为正常区间的约束 → **两个旋钮都生效**。语义 = "手每帧不得深于 `min_sdf_m`（软违规阈值），允许至多 `max_violation_pct` 帧违规，但任一帧绝不得深于 `hard_floor_m`（绝对底线）"。
- 身体 gate 与手 gate **共用同一 add_gate**，一次改两边。
- **范围约束**：只动 `_build_gate_masks` 的 floor 逻辑 + 两个新字段；不碰 reward、不碰 SDF 计算、不碰 elite 加权。

**验证（C1）**：
- 单元/回归：对 `E152_box004_083_p2_gateA_b1`（default hard_floor=nan）重跑，与 E152 结果逐指标 max_abs_diff=0。
- 激活检查：smoke 跑同 case `hard_floor=−0.02`，对比 `max_violation∈{0.05,0.10}` 的 `cem_hand_gate_valid_frac` 不同。

## 3. 扫描矩阵（gateA_b1，3 case）

固定 `cem_hand_gate_hard_floor_m = −0.020`（绝对单帧底线，深于所有扫描 min_sdf_m，激活 max_violation）。

| 轴 | 取值 | tag |
|---|---|---|
| `cem_hand_gate_min_sdf_m`（软违规阈值） | −0.005 / −0.010 / −0.015 | sdf005 / sdf010 / sdf015 |
| `cem_hand_gate_max_violation_pct`（允许违规帧占比） | 0.05 / 0.10 | v05 / v10 |

→ 6 组/case × 3 case = **18 新 full CEM**。复用 E152 的 baseline（gate 关）+ b1（gate 关、reward only）作参考，不重跑（gate-off，不受 Stage0 影响）。

variant 命名：`E153_{case}_gateA_b1_{sdfTAG}_{vTAG}`，base override 复用 `core4d_E152_{case}_gateA_b1.yaml`，3 个 gate 参数走 Hydra CLI 覆盖（不新建 18 个 override）。

| case | task | base override | GPU |
|---|---|---|---|
| box021_029_p2 | d003_box021_20231018_029_p2_e107_clean | core4d_E152_box021_029_p2_gateA_b1 | **local-gpu0** |
| box004_083_p2 | e091_box004_20231003_2_083_p2_e092_dyn | core4d_E152_box004_083_p2_gateA_b1 | **remote-gpu0** |
| box023_person2 | box023_person2_legobj | core4d_E152_box023_person2_gateA_b1 | **remote-gpu1** |

## 4. 并行执行（本机 1 卡 + 远程 2 卡，按 case 分卡）

每个 case = 一个脚本，跑该 case 的 6 组 ablation（串行同 GPU）：
- `scripts/E153/run_case_box021.sh`（local GPU0）
- `scripts/E153/run_case_box004.sh`（remote GPU0）
- `scripts/E153/run_case_box023.sh`（remote GPU1）
- 三脚本共用 `scripts/E153/_sweep_lib.sh`（6 组循环 + run_mjwp 调用 + 产物拷贝 + 关键帧），避免逻辑漂移。

编排：本机 tmux 跑 box021；远程 `run_E153_remote.sh`（待加，镜像 E152）同步代码/依赖后，远程 tmux GPU0/GPU1 并行跑 box004/box023；完成后 `pull_E153_remote_results.sh` 回收。

## 5. 评测与成功判据

复用 E152 evaluator（已含 2mm/5mm 深度分档几何穿透 + `success_pen2mm_down_contact_keep` + 物理 `con<−5mm`）。新增 E153 manifest/eval 固定入口（`scripts/E153/build_*` + `scripts/eval/eval_E153_*`），口径同 E152：
- 每组 vs b1 报：near-5cm Δ、几何穿透 0/2mm/5mm Δ、物理接触 Δ、深穿透 `con<−5mm` Δ、leg穿透 Δ、obj_err Δ、fall、gate_valid/fallback。
- **主判据 = `success_pen2mm_down_contact_keep`（深度感知）**，0mm 仅作参考。
- 选出每 case 最优组，再看是否存在**全 3 case 通用**的最优 `(min_sdf, max_violation)`（C2）。
- 报 per-case + 3-case mean/std/worst，禁 cherry-pick；A/B 同 clip 关键帧并排。

## 6. 风险与对策

- **R1 解耦改动影响存量**：默认 sentinel nan → floor=min_sdf_m，行为不变；C1(a) 回归钉死 max_abs_diff=0。
- **R2 hard_floor=−0.02 仍太松/太紧**：先 smoke 一组确认 valid_frac 合理、有真实剔除；必要时调 hard_floor（但本实验固定，避免三轴爆炸）。
- **R3 网格无通用甜点**：若不存在全 3-case 通用最优，退而给"每 case 最优 + 通用次优"两套结论，并把 case 依赖性如实记录。
- **R4 box021 本地中断**（E152 教训）：本地脚本支持已完成 row skip（`outputs_exist`），中断可续；GPU0 当前空闲。

## 7. 数据管线 / 快照

- 沿用 E152 rubber 场景（pair 已存在，scene 不动）。每脚本训练前调 `snapshot_scenes.sh E153 <case>` 快照该 case scene XML 到 `results/E153/.../scene_snapshot/`（experiment.md §7）。
- 复用 baseline(E147/E148) + b1(E151) 轨迹，不重跑。

## 8. 范围约束

- 只动 Stage0 gate floor 解耦（2 新字段 + add_gate floor 逻辑）+ 扫 min_sdf/max_violation；不碰 reward、SDF、solref（轴1 杠杆 B 留作后续 E）、target、稳定/object actuator/partner。
- 所有改动 git 追踪 + C1 回归；scene/manifest per-exp snapshot。

## 9. 记录

- `log/<N>_E153_*.md`：Stage0 回归/激活验证 + 18 组 ×（双穿透口径 + gate 健康 + success_pen2mm）对比表 + 每 case 最优 + 通用甜点 + 视频 + 结论。
- `EXPERIMENT_TRACKER.md` 加 E153 行；progress.md 全程记录。
- 若 C2 成立（找到 3/3 strict 设定）→ commit；该设定作为后续 RL 参考数据/默认 gate 配置候选。
