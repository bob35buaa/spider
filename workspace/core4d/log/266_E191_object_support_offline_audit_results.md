# E191 结果：物体支撑建模离线审计（box024 物体 tracking / 手物穿透根因）

_Core4D · 2026-08-08 · **零 CEM 算力**，仅对已落盘 rollout 重新打分 · plan: [216](../plan/216_E191_object_support_offline_audit_plan.md)_

> **实验编号说明**：E190 已被「38-case noPRG RL export」占用，本实验顺延为 E191。

---

## 1. 目的

R018 侧分析把 SPIDER→RL 的问题收敛到 provenance 混杂与「上游 gate 预测不了下游崩塌」两条。
在此之上用户点出一个未归因现象：**box024 的物体非机器人一侧高度远低于参考轨迹**
（`track_obj_pos_err_cm_mean` 13.64 vs box004 12.01 / box001 11.26），且**手物穿透异常高**
（PRG 0.378 / noPRG 0.391 vs box004 0.14–0.15、box001 0.18–0.21），只有 `027_p1/p2` 好一些。

E191 不跑仿真。它给 eval 补上缺失的度量维度，并在已完成的 E172/E173/E174/E189 rollout 上重新打分，
目的是把三条**与物体尺寸完美共线、现有实验无法区分**的机制分开：

| | 机制 | 随尺寸如何变 |
|---|---|---|
| (a) | 固定米制 reward/gate 几何 | 阈值不变、物体变大 |
| (b) | 物体伺服过软 + partner 未建模 | 力臂变长 |
| (c) | 大箱无几何闭合，只能压平面 | 抓取拓扑变 |

## 2. 参数与运行命令

```bash
# 度量扩展（纯追加）后，离线重打分 + 回归校验
.venv/bin/python workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py --workers 32
# 假设判定 + 配置 provenance 审计
.venv/bin/python workspace/core4d/scripts/eval/reports/gen_E191_object_support_report.py
```

`unset MUJOCO_GL`（沿用 E189 launcher 的 glfw fallback 约定）。
新增度量的默认参数（`EvalConfig`）：`object_pos_actuator_gain=500.0`、`object_rot_actuator_gain=50.0`、
`object_lift_threshold_m=0.05`、`hand_gate_hard_floor_m=-0.020`。前两者**不可从 scene XML 恢复**
（XML 里 `kp="0"`，增益由 `examples/run_mjwp.py` 运行时注入），取值取自 E163→E167A→E172/E173/E189 resolved 链。

## 3. 改动文件

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/eval/core/core_metrics.py` | 纯追加 `E191_SUPPORT_FIELDS`（22 列）+ `_object_support_metrics()` + `_box_corners_world()`；`EvalConfig` 增 6 个 E191 参数；主循环增 `hand_frame_min_con_dist` 逐帧最小接触距 |
| `workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py` | 新增，离线重打分 + 逐列回归校验 |
| `workspace/core4d/scripts/eval/reports/gen_E191_object_support_report.py` | 新增，H1–H6 判定 + 物体内力臂回归 + 配置 provenance 审计 |
| 产物 | `results/E191/audit/{e191_object_support_audit.tsv, E191_object_support_report.md, e191_config_provenance.tsv, e191_coverage.json, frames/}` |

**无 scene snapshot**：E191 不跑物理仿真，按 `.claude/rules/experiment.md` §7 不适用。

## 4. 覆盖与回归校验

- **141/169 行成功**。28 行全部是 E170（box021）——`dcv3_omnirt_v1_ref_fk_box021_*` 数据目录
  在本 checkout 中**完全缺失**（0 个），scene XML 与参考轨迹都取不到。**box021 无法离线审计**，
  这是一个必须记录的覆盖缺口：R018 §9.1 里那个「原生」物体恰好是唯一取不到的。
- **回归零差异 PASS**：E189 全部 43 行、每一个既有列逐值 bit-identical（`e191_coverage.json`）。
- **交叉校验 PASS**：`obj_side_*` 与本轮独立复算一致（box024 `026_p1` near/far = −1.96/−17.61 cm；
  `ref_hand_geom_penetration_frac` 0.6387，独立算得 0.639）。
- 覆盖 **10 个物体**（4 box + 5 bucket + 1 desk），比原计划的 5 个多一倍，跨物体回归样本因此显著变好。

## 5. 结果

### 5.1 用户报的现象：确认，且比预期更系统

box024 逐 case 分侧高度（E189 noPRG，抬起帧）：

| case | lift frac | 机器人侧 Δz | **非机器人侧 Δz** | 不对称 | objpos | handpen3mm |
|---|---:|---:|---:|---:|---:|---:|
| 026_p1 | 0.56 | −2.0 | **−17.6** | 15.6 | 14.44 | 0.546 |
| 026_p2 | 0.58 | −4.1 | **−13.1** | 9.0 | 10.37 | 0.313 |
| 027_p1 | 0.47 | −6.7 | **−11.5** | 4.8 | 9.07 | 0.348 |
| 027_p2 | 0.50 | −3.8 | **−12.1** | 8.3 | 8.48 | 0.137 |
| 028_p1 | 0.51 | −11.0 | −11.1 | 0.1 | 12.02 | 0.347 |
| 028_p2 | 0.54 | −6.0 | **−13.2** | 7.2 | 14.43 | 0.200 |
| 030_p1 | 0.47 | −3.0 | **−14.1** | 11.1 | 13.94 | 0.520 |
| 031_p1 | 0.50 | −10.2 | **−14.5** | 4.3 | 22.76 | 0.632 |
| 031_p2 | 0.46 | −12.0 | **−16.2** | 4.2 | 15.77 | 0.473 |

**⚠️ 对用户表述的一处修正**：`027_p1/p2` 的远端下沉是 −11.5 / −12.1 cm，与其余 case 同量级
（全 9 例远端一致落在 **−11 ~ −18 cm**）。**远端下沉是 box024 全量普遍现象，没有例外**；
`027` 之所以看起来好，是**手物穿透和总位置误差**低，不是远端不塌。

跨物体对照（E189/E173 侧）：

| object | 最长半轴 | 机器人侧 Δz | 非机器人侧 Δz | 不对称 |
|---|---:|---:|---:|---:|
| box023 | 0.196 | −9.93 | −10.31 | **0.38** |
| box004 | 0.224 | −7.70 | −8.48 | **0.78** |
| box001 | 0.406 | −7.19 | −11.20 | **4.01** |
| **box024** | **0.491** | −6.53 | **−13.71** | **7.18** |

**目视核验**（`results/E191/audit/frames/`，左 ref 右 sim）：
`b024_026p1_2.4s.png` —— 参考里箱体水平，sim 里**远端明显向地面栽下去**，机器人端翘起；
`b004_086p1_1.6s.png` —— box004（短力臂）ref 与 sim 几乎重合，箱体保持水平。视觉与数值完全一致。

### 5.2 预注册假设判定（**两条被证伪，照实记录**）

| # | 假设 | 判定 | 关键数字 |
|---|---|---|---|
| H1 | 伺服下垂主导物体位置误差 | **REFUTE** | 抬起帧 z 误差确实系统性存在（−4.2 ~ −10.4 cm），但 **z 占比只有 0.21–0.41**，总误差由**水平分量主导** |
| H2 | 不对称量由力臂驱动（跨物体） | **INCONCLUSIVE** | ρ = 0.72（p=0.0076，n=13）。方向明确且显著，但未达预注册的 ρ≥0.8 |
| H3 | 穿透主要是参考自带 | **REFUTE** | 对 **box 类成立**（ref 0.45–0.63 ≥ run 0.30–0.34），但 bucket/desk **完全反向**（bucket007 ref 0.11 / run 0.55） |
| H4 | box024 顶死 CEM 手门地板 | **PASS** | 饱和帧占比 box024 0.0187 vs box023 0.0113 / box001 0.0047 / box004 0.0023 |
| H5 | Omni 侧不共享此伪影 | **NOT TESTABLE** | 见 §5.4 |
| H6 | 物体内力臂回归分离 (b) | **LOW POWER** | box024 的 9 例力臂跨度仅 **0.039 m**，无功效；box001（跨度 0.401 m，n=28）ρ=0.19 p=0.33 |

### 5.3 H1b（**事后分析，非预注册**）：下垂量级精确符合 m·g/kp

审计意外发现**物体质量并非常量**：box 全部钉死 5.0 kg，但 E174 的 bucket/desk 跨 **2.0–120.4 kg**，
而 `init_pos_actuator_gain` 对所有 case 恒为 500 N/m。这构成一个尺寸之外的天然变量：

| object | 质量 | m·g/kp 预测 | 实测抬起帧 z 误差 |
|---|---:|---:|---:|
| bucket007 | 2.0 kg | −3.9 cm | **−4.2 cm** |
| bucket010 | 3.0 kg | −5.9 cm | −5.1 cm |
| bucket003 | 3.3 kg | −6.5 cm | −5.1 cm |
| box004 | 5.0 kg | −9.8 cm | −8.5 cm |
| box001 | 5.0 kg | −9.8 cm | −9.2 cm |
| box024 | 5.0 kg | −9.8 cm | −9.7 cm |
| box023 | 5.0 kg | −9.8 cm | −10.4 cm |
| desk007 | 69.1 kg | −135.6 cm | −9.2 cm（离群，见下） |

ρ = 0.66（p=0.0154, n=13）；剔除 desk007 后 ρ = 0.68（p=0.0178, n=12），
**观测/预测比中位数 0.93、范围 [0.60, 1.08]**。desk007 是 69 kg 的桌子，伺服根本抬不动它，
由地面承重，因此不在同一 regime。

→ **伺服下垂的量级由 m·g/kp 精确预测**（H1b），但它**不是**物体位置误差的主要来源（H1 被证伪，水平分量更大）。
这两条必须一起读：伺服建模缺陷真实存在、可预测、且被 partner 缺失放大成远端下沉，
但把它全部修好也**只能消掉约 1/3 的 `track_obj_pos_err`**。

### 5.4 H5 与 E174 身份核实：R018 的一处引用错误

R018 母分析把 `results/E174/.../e174_case_metrics.tsv` 标为「OmniRetarget bucket 对照」。核实结果：

- E174 全部 39 行的 `method` = `E174_E170PRG_nonbox_candidate_r1`；
- `plan/190_E174_bucket_desk_move2_full_pipeline_plan.md` §1 明写它是把 E170–E173 冻结重定向算法
  **首次扩展到非 box 物体**的 SPIDER 实验，「算法与冻结配置完全沿用 E170/E171/E172/E173」。

**E174 是 SPIDER 自己的 bucket/desk 流水线，不是 Omni 对照。** 本仓库中**不存在任何 OmniRetarget 侧的上游
rollout 表**，因此 H5 无法检验，且 R018 说的「对称审计缺口」比原以为的更大——不是缺一份 gate 审计，
而是 Omni 侧上游整个不存在。

### 5.5 配置 provenance 审计（A6）

`e191_config_provenance.tsv`（141 case）独立复现了链路审计：

- 所有 case 的 defaults 链均为 **17 节**；外来（他物体）case 级祖先：
  **box024/box001/bucket/desk 各 11 个**、box023 7 个、box004 6 个；
- 27 个被追踪的米制/尺度敏感参数中，**27 个在全部 10 个物体上完全相同，0 个随物体变化**——
  跨越 2.0–120.4 kg 质量与 0.196–0.491 m 最长半轴；
- 全部 156 个 per-case config 继承同一个 **`core4d_E167_box004_082_p1_E167A`**（box004 case 级 override），
  reward 几何实际由 `core4d_E163_box004_082_p1_narrowSurfaceBand` 与 `core4d_E156_box004_082_p1_gateA` 设定。

## 6. 结论

1. **用户报的远端下沉现象成立且是系统性的**：box024 全 9 例远端一致下沉 −11~−18 cm，机器人侧只 −2~−12 cm；
   不对称量随最长半轴单调（box023 0.38 → box004 0.78 → box001 4.01 → box024 7.18 cm）。目视确认。
2. **`027_p1/p2` 并不是远端不塌**，它们只是穿透和总位置误差低。原表述需修正。
3. **伺服 + 无 partner 的建模缺陷被实证**（H1b，跨 2–5 kg 质量范围观测/预测比 0.93），
   但**它只解释约 1/3 的物体位置误差**（H1 证伪），不能作为唯一杠杆。
4. **穿透的来源是分物体的**（H3 证伪）：box 类主要继承自参考轨迹（ref ≥ run），bucket/desk 类反过来由物理产生。
   box024 是唯一显著顶到 CEM 手门硬地板的物体（H4 PASS）。
5. **(a)(b)(c) 仍未分离**：H2 方向对但不达标，H6 因 box024 力臂跨度仅 0.039 m 而无功效。
   **零算力路线到此为止，要分离必须做 Stage B 的 config-only A/B**（A1/A2 查 (b)、A4 查 (a)，两者可并行）。
6. **两处对 R018 的勘误**：E174 不是 Omni 对照（§5.4）；配置层对所有物体一致，
   **不能**解释 §9.1 的原生/跨物体差距（§5.5）。

## 7. 局限

- **box021 完全缺席**（数据目录缺失），而它正是 R018 结论里的「原生」锚点。
- H1b 是**事后分析**，不是预注册假设；n=13 个 exp×object 组，且质量只有 2/3/3.3/5/69 kg 五档。
- 伺服增益 500/50 取自 resolved config，不是从 rollout 反推的；若某些 case 实际增益不同，H1b 会失真。
- `grip_far_arm_m` 用手部 geom 中心沿物体最长局部轴投影计算，对非长条物体（bucket/box023）语义弱。
- 未跑 `ruff`：本环境未安装（`ruff`/`uvx`/`python -m ruff` 均不可用），仅通过 `py_compile`。

## 8. 下一步

Stage B（E192，需另行批准）：box024(9) + box004(6) 共 15 例 canary，四臂并行——
`partner_force_scale=0.5`、`init_pos/rot_actuator_gain` 提高（查 (b)）、尺寸自适应门阈值（查 (a)）。
偏心 partner 支撑需先改 `_apply_partner_force`（`mjwp.py:3547` 对 `nq_obj=6` 硬读四元数，会越界）。
