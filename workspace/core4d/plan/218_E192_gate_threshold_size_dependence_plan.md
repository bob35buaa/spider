# E192 实验计划：中度收紧 CEM 手门——是否存在 box024 特异阈值效应（机制 (a)）

_Core4D · Phase 55 · 参数与设计已获用户确认，脚本未创建、CEM 未启动 · 承接 [E191](../log/266_E191_object_support_offline_audit_results.md)_

---

## 📋 Context

E191 离线审计（零算力，141 条 rollout 重打分）确认了 box024 的远端下沉与高穿透，但把根因留在三条与物体身份、尺寸和抓取方式共线的机制上：

| 机制 | 当前证据 | 本轮处理 |
|---|---|---|
| (a) 固定米制 CEM gate / reward 阈值 | box024 的物理接触深度更常接近旧地板 | **E192 的干预对象** |
| (b) 物体伺服过软 + partner 未建模 | E194 的 G1 已证明 gravcomp 可修平移下垂并降低部分穿透；位置错与伺服力的主判别仍未闭合 | E192 冻结为 **no-gravcomp**，不叠加 G1 |
| (c) 抓取拓扑缺少几何闭合 | box001/box024 对照仍待 E193 检验 | E192 不改抓握目标 |

E191 §5.5 的 provenance 审计给出 (a) 的事实基础：

- **156/156** 个 per-case config 继承同一个 `core4d_E167_box004_082_p1_E167A`（box004 case 级 override）
- 追踪的 27 个米制/尺度敏感参数在全部对象上完全相同
- `plan/189` §147 要求过 per-object base，但实际未创建
- `log/233:131` 将尺寸自适应 hand-collision margin 划出范围，至今没有对应干预实验

### 证据边界：物理接触深度不等于 CEM gate SDF

E191 的 `hand_object_con_dist_min_m` 和 `hand_gate_floor_saturation_frac` 来自 MuJoCo 物理接触 `con.dist`；CEM gate 实际使用另一套保守 SDF。二者相关但不是同一个量，不能仅凭 `con.dist≈-0.020` 宣称 CEM 已把解顶到硬地板。

基线 NPZ 的候选池诊断提供了更直接但仍不完整的证据：

- box024 `028_p1` / `031_p2` 的候选池最低 hand-gate SDF 分别约为 `-0.0212 / -0.0262 m`，确实越过旧 `-0.020 m` 地板
- box004 六例的全局最低候选 SDF 约为 `-0.0127 m`
- 旧产物没有直接记录被选候选的 hand-gate 最小 SDF，因此 E192 必须增加 observation-only 的 selected-candidate 诊断，不能继续用物理 `con.dist` 代替 gate 内部读数

### (a) 的两个相反子机制

| 子机制 | 含义 | 干预方向 | E192 处理 |
|---|---|---|---|
| (a1) gate 饥饿 | 门太紧，没有合法候选，fallback 到劣解 | 放松 | 不设独立实验臂；作为 gate-health 否决条件 |
| (a2) gate 过松 | 深穿候选仍可合法入选 | 收紧 | **本轮中度收紧策略包** |

PRG 基线侧的总体 gate-health 没有显示 box024 更饥饿：box024 / box004 的 `cem_gate_valid_frac` 为 `0.7954 / 0.7790`，`cem_posture_gate_fallback_used` 为 `0.1427 / 0.1530`。因此本轮不额外开“放松门”实验臂，但若 A2 导致合法候选塌缩，必须判为 `INCONCLUSIVE_GATE_COLLAPSE`，不能判阈值无效。

### 本轮 estimand：box024 特异，不是尺寸因果

box024 与 box004 除了尺寸，还同时存在物体几何、抓取拓扑、动作分布和物体身份差异。两物体的差分效应最多支持“box024 对该阈值策略响应更强”，不能单独证明响应由尺寸引起。

本轮正式 estimand 冻结为：

```text
reduction(object) = mean_case(A0 penetration - A2 penetration)
DiD = reduction(box024) - reduction(box004)
```

判别含义：

```text
box024 改善显著大于 box004  → BOX024_SPECIFIC_THRESHOLD_EFFECT
两物体同向同幅改善           → GLOBALLY_SUBOPTIMAL
gate 健康但两物体都不改善     → THRESHOLD_POLICY_NOT_EFFECTIVE
gate 合法候选或基线复现塌缩   → INCONCLUSIVE，不下机制结论
```

box004 是阴性对照，但不是“尺寸因果”的充分对照。box001/box023 留给后续多对象或拓扑实验，本轮不据此声称 size dependence。

---

## 🎯 Claims

| # | Claim | 最低证据（预注册，事后不得调整） |
|---|---|---|
| C1 | 中度收紧 hand-gate 策略能降低 box024 穿透 | A2 下 box024 `hand_object_physics_penetration_3mm_frame_frac ≤ 0.20`（A0 为 `0.3776`），且至少 7/9 例改善 |
| C2 | 降穿透不是靠把手拿开换来的 | box024 `hand_object_physics_contact_3mm_in_mask_frac ≥ 0.3079`，且 `hand_object_physics_contact_in_mask_frac ≥ 0.75`；视觉若确认失去承重接触，则 C2 失败，即使均值过线 |
| C3 | 效应对 box024 特异，而非全局同幅调参收益 | `reduction=A0-A2`；`DiD=reduction(box024)-reduction(box004) ≥ 0.10`，不取绝对值，且分层 bootstrap 95% CI 下界 `>0` |
| C4 | 不产生跨门安全回退 | box004 六例的任一 case×gate 不得由 PASS→FAIL；box024 `leg_penetration_frac` 相对 A0 不得升高 `>0.05` |
| C5 | 已删除独立的 (a1) 实验臂 | `cem_hand_gate_*` / `cem_gate_*` / `cem_leg_gate_*` / `cem_posture_gate_*` 全部必报，并由 C7 判定 gate 是否健康 |
| C6 | gate 内部读数与物理结果方向一致 | 非 fallback 步中，被选候选必须满足新 `-0.015 m` 硬地板；box024 固定口径的 `hand_gate_fixed_depth_15mm_frame_frac` 必须低于 A0，且至少 7/9 例改善。若 C1 过但该固定深度指标不降，判 `MECHANISM_MISMATCH` |
| C7 | 干预得到有效执行，而不是退化为 fallback | 每个物体的 A2 `cem_hand_gate_valid_frac ≥ 0.60`，且 `cem_gate_fallback_used ≤ A0+0.15`；任一不满足则判 `INCONCLUSIVE_GATE_COLLAPSE`，禁止判 C1/C3 不成立 |

> `hand_gate_fixed_depth_{10,15,20}mm_frame_frac` 对 A0/A2 使用相同绝对深度计算，避免把每个 arm 按自己的地板重算后得到不可比的“饱和率”。原 `hand_gate_floor_saturation_frac` 继续报告，但不再单独承担 C6 判决。

---

## ⚙️ 实验设计

冻结不变量：PRG 开启；E167A_zOnlyBody profile；`rubber_hull` 手部碰撞体；CEM `seed=0 / 1024 samples / 32 opt steps`；`gravcomp=0`；`partner_force_scale=0`；抓握目标保持 `ref_fk`。

| 臂 | 配置 | Full 数量 | 作用 |
|---|---|---:|---|
| A0-history | E172 box004×6 + E173 box024×9 | 0（只读） | 历史基线 |
| A0-sentinel | 当前代码、历史 A0 配置 | 3 | 检查 box SDF 修正后旧基线能否复现 |
| A2 | 中度收紧 hand gate | 15 | 正式干预 |

一臂两物体：box024×9 是待检验对象，box004×6 是阴性对照。正式结论依赖 signed DiD，不从 box024 单独改善推出“特异效应”。

### A2 参数：中度收紧策略包

```yaml
cem_hand_gate_min_sdf_m: -0.010          # 保持不变：深于 10 mm 的帧计为违规
cem_hand_gate_max_violation_pct: 0.05   # 基线 0.10：违规帧上限 10% → 5%
cem_hand_gate_hard_floor_m: -0.015      # 基线 -0.020：绝对深度下限 20 mm → 15 mm
```

对于 100 帧候选，A2 允许最多 5 帧位于 `[-0.015,-0.010)`，但任何一帧都不得深于 `-0.015 m`。这是预先选定的两参数策略包，不用于区分 `max_violation_pct` 与 `hard_floor_m` 各自的贡献，也不用于搜索最优阈值。

### 严格冻结项

- `cem_leg_gate_*`、`cem_posture_gate_*`、`cem_safety_gate_*`
- `surface_band_*`、`object_lift_sigma`、`leg_object_penalty_*`、全部 reward scale
- `init_pos_actuator_gain=500`、`init_rot_actuator_gain=50`
- object body 不含 `gravcomp`；不开 `partner_force_scale` 或 support proxy
- `contact_hdmi_target_source` 与 `ref_fk` 抓握目标不变
- scene、trajectory、contact mask、selected retarget variant 与 A0 逐 case hash 对齐

### Case 集合

- box024×9：`20231011_{026_p1,026_p2,027_p1,027_p2,028_p1,028_p2,030_p1,031_p1,031_p2}`
- box004×6：`20231003_2_{082_p1,082_p2,083_p1,083_p2,086_p1,086_p2}`

box001/box023 本轮不跑，因此本轮明确不下尺寸依赖结论。

### 当前代码 A0 哨兵

本轮三条 Full A0 哨兵固定为：

| 对象 | Case | 选择理由 |
|---|---|---|
| box024 | `026_p1` | 历史穿透最高端 |
| box004 | `082_p1` | 常规阴性对照 |
| box004 | `086_p2` | 历史高 fallback 压力对照 |

`box024_20231011_027_p2` 已从本轮 A0-sentinel 明确丢弃；不以替代
case 补齐。A2 case 集与后续实验授权保持不变。

A0-sentinel 相对历史同 case 必须满足：

- resolved config、scene/trajectory/contact-mask SHA 与冻结基线一致
- 12 个二值 gate 逐项一致
- penetration/contact/leg 三类 fraction 的绝对差均 `≤0.03`
- position tracking 绝对差 `≤1.0 cm`，orientation tracking 绝对差 `≤1.0°`
- `cem_hand_gate_valid_frac` 与 `cem_gate_fallback_used` 绝对差均 `≤0.05`

任一哨兵失败即停止为 `INCONCLUSIVE_BASELINE_DRIFT`。本计划不自动补跑其余 11 条 A0；如需完整当前代码 A0，必须另行取得用户算力批准。

### 预算与顺序

- A0-sentinel：3 条 Full CEM（`027_p2` excluded）
- A2 canary：3 条 `64 samples × 4 opt_steps`，只检查 plumbing 与严重 gate collapse，不用于效果判决
- A2 Full：15 条 Full CEM
- 已批准 Full 上限：**18 条**；不含额外 11 条 A0

E192 的资源编排冻结为 **三卡并行**：本地 GPU0 一条、远程 A100 GPU4/GPU5
各一条。A0-sentinel 一卡一条；A2 canary 同样一人一条；A2 Full 的 15 条按
`local-gpu0 / a100-gpu4 / a100-gpu5 = 5 / 5 / 5` 串行队列分配。GPU6/GPU7
不进入本轮 E192 的任何 manifest 或 launcher。

这是叠加式启动合同：E192 的 local/remote launcher 只创建自己的 session、日志和结果目录，**不得调用 `pkill`/`kill`/`killall`，不得停止、重排、抢占或等待现有程序**；即使目标 GPU 已有负载，也直接启动 E192 worker。若 E192 自身某 worker OOM 或失败，只记录该 worker 的失败并保留其他任务和 worker，不自动杀进程、不跨卡迁移、不静默重试。

A2 canary 固定为 box024 `026_p1`、box024 `027_p2`、box004 `082_p1`。执行顺序必须是 scene/config 审计 → A0-sentinel → A2 canary → A2 Full。

---

## 📊 基线与统计

### 历史 A0 汇总

| 指标 | box004（E172，n=6） | box024（E173，n=9） |
|---|---:|---:|
| 12 门通过 | 2/6 | 2/9 |
| `hand_object_physics_penetration_3mm_frame_frac` | 0.1469 | **0.3776** |
| `hand_object_physics_contact_3mm_in_mask_frac` | 0.4507 | 0.3079 |
| `hand_object_physics_contact_in_mask_frac` | 0.6944 | 0.8188 |
| `hand_gate_floor_saturation_frac`（旧 `-0.020 m` 口径） | 0.0046 | **0.0136** |
| `hand_object_con_dist_min_m` | -0.0141 | **-0.0201** |
| `leg_penetration_frac` | 0.0565 | 0.0386 |
| `track_obj_pos_err_cm_mean` | 12.0105 | 13.6421 |
| `track_obj_ori_err_deg_mean` | 11.5344 | 6.2508 |
| `obj_side_z_asym_cm` | 1.1588 | **7.3322** |
| `cem_gate_valid_frac` | 0.7790 | 0.7954 |
| `cem_posture_gate_fallback_used` | 0.1530 | 0.1427 |

### 必报 gate 机制读数

- `cem_hand_gate_selected_min_sdf_m` 与 selected mean/p05
- `cem_hand_gate_valid_frac`、`cem_hand_gate_selected_valid_frac`
- `cem_gate_valid_frac`、`cem_gate_fallback_used`
- `cem_leg_gate_*`、`cem_posture_gate_*`
- `hand_gate_fixed_depth_10mm_frame_frac`
- `hand_gate_fixed_depth_15mm_frame_frac`
- `hand_gate_fixed_depth_20mm_frame_frac`
- 原 `hand_gate_floor_saturation_frac`，仅作兼容诊断

12 门、历史基线表全部指标与 E191 的 22 个 support 列仍需逐物体报告，禁止把两个物体合并成单一均值。

### 统计规则

- 连续指标以 case 为单位，在物体内做 paired bootstrap，种子 0、10000 次
- DiD bootstrap 分别在 box024/box004 内重采样 case 后做差，报告 95% CI
- C3 必须同时满足点估计 `≥0.10` 与 95% CI 下界 `>0`
- 不对 12 个语义不同的 gate 合并做一个“门级 McNemar”
- 每个 gate 分别报告 `PASS→PASS / PASS→FAIL / FAIL→PASS / FAIL→FAIL`
- case-level numeric release pass 可报告 exact McNemar，但 n=6/9 时只作辅助证据

---

## 🛡️ 判决与 stop-loss

### 判决优先级

| 优先级 | 条件 | 判定 |
|---:|---|---|
| 1 | A0-sentinel 任一复现门失败 | `INCONCLUSIVE_BASELINE_DRIFT` |
| 2 | C7 失败或 canary 严重塌缩 | `INCONCLUSIVE_GATE_COLLAPSE` |
| 3 | C1/C2 过但 C6 失败 | `MECHANISM_MISMATCH` |
| 4 | C1+C2+C3+C4+C6+C7 全过 | `BOX024_SPECIFIC_THRESHOLD_EFFECT` |
| 5 | C1+C2+C4+C6+C7 过但 C3 不过 | `GLOBALLY_SUBOPTIMAL` |
| 6 | C1 不过且 C7 通过 | `THRESHOLD_POLICY_NOT_EFFECTIVE` |

`BOX024_SPECIFIC_THRESHOLD_EFFECT` 不等于 size dependence，也不自动把 A2 升级为 production 默认。E193 继续使用原始 gate、no-gravcomp 配置，保持与 E192/E194 正交；三个机制实验完成后再另设组合实验。

### Canary stop-loss

- 3 例中若至少 2 例 CEM 崩溃、override 未生效或 artifact 不完整，停止，不进 Full
- 3 例中若至少 2 例 `cem_hand_gate_valid_frac <0.05` 或 `cem_gate_fallback_used >0.80`，停止为 `INCONCLUSIVE_GATE_COLLAPSE`
- Canary 的 12 门 migration 只作诊断；不得因 box004 的低预算 Canary 出现新 FAIL 而删除 box004 对照或改跑 box024-only
- A2 Full 启动后不得事后修改阈值、case 集、C1-C7 或统计定义

---

## 🔧 拟新增或修改文件

| 文件 | 用途 |
|---|---|
| `workspace/core4d/scripts/experiments/E192/e192_common.py` | 冻结 A0-sentinel/A2、18 条 Full 上限、case 集、阈值与 method ID |
| `workspace/core4d/scripts/experiments/E192/build_hand_gate_manifest.py` | 生成当前代码 A0-sentinel 与 A2 manifest，并执行 snapshot |
| `workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py` | 确认 A2 仅改 hand-gate 两字段，且 no-gravcomp、其余配置与 A0 逐字段相同 |
| `spider/optimizers/sampling.py` | observation-only：记录 selected hand-gate SDF；不得改变 reward、gate、selection 或 control |
| `spider/optimizers/sampling_fast.py` | 与标准 sampling 保持诊断字段一致；不得改变选择逻辑 |
| `workspace/core4d/scripts/eval/core/core_metrics.py` | 新增固定 10/15/20 mm 的物理深穿帧率公共指标 |
| `workspace/core4d/scripts/launch/active/run_E192_hybrid_5gpu.sh` | 三卡总入口：本地 GPU0 + 远程 A100 GPU `4,5`；`baseline_sentinel / canary / full` 三模式入口 |
| `workspace/core4d/scripts/launch/active/run_E192_remote_A100.sh` | 在远程 A100 主机上分别绑定物理 GPU `4,5`，启动两个独立 worker；只追加启动，不触碰现有进程 |
| `workspace/core4d/scripts/launch/active/pull_E192_remote_A100_results.sh` | 按唯一 worker/case manifest 从远程回收 E192 结果，禁止覆盖非 E192 路径 |
| `workspace/core4d/scripts/eval/runners/eval_E192_hand_gate_arm.py` | A0-history/A0-sentinel/A2 的配对评测，直接 import 公共 metrics |
| `workspace/core4d/scripts/eval/wrappers/eval_E192_hand_gate_arm.sh` | canonical shell 入口 |
| `workspace/core4d/scripts/eval/reports/gen_E192_arm_comparison.py` | A0 复现审计、C1-C7、signed DiD 与判决优先级 |
| `workspace/core4d/results/E192/scene_snapshot/` | 训练前 scene 双重快照 |

优先用 `gen_experiment.py --exp-id E192` 生成骨架；manifest builder 与判决逻辑按本计划实现。新增核心诊断必须是纯追加、默认不影响存量实验，并补单元测试证明选择结果不变。

## 🚀 执行入口

```bash
# 0. scene 快照、manifest 与正向审计
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E192 <15 cases>
.venv/bin/python workspace/core4d/scripts/experiments/E192/build_hand_gate_manifest.py --apply --snapshot
.venv/bin/python workspace/core4d/scripts/experiments/E192/audit_gate_overrides.py --require-all

# 1. 当前代码 A0 哨兵，先过复现门（三卡并行；不停止现有任务）
MODE=baseline_sentinel bash workspace/core4d/scripts/launch/active/run_E192_hybrid_5gpu.sh

# 2. A2 plumbing canary；通过后才进 Full
MODE=canary bash workspace/core4d/scripts/launch/active/run_E192_hybrid_5gpu.sh
MODE=full bash workspace/core4d/scripts/launch/active/run_E192_hybrid_5gpu.sh
# 远程结果回收（只拉取 E192 自己的 manifest 行）
bash workspace/core4d/scripts/launch/active/pull_E192_remote_A100_results.sh

# 3. 评测、报告与渲染
bash workspace/core4d/scripts/eval/wrappers/eval_E192_hand_gate_arm.sh full
```

结果路径：`workspace/core4d/results/E192/s6_downstream/{cem,eval,render}/`；远程临时 staging 必须按 `E192/<worker_id>/<case_id>/` 隔离，回收后不得覆盖其他实验。

### 三卡 manifest contract

每个 E192 manifest row 必须固定且唯一地记录 `worker_id`、`host`、物理 `gpu_id`、`case_id`、`arm`、`seed`、resolved-config SHA、scene snapshot SHA、远程 staging 路径和本地最终结果路径。固定映射为：`local-gpu0`（本地 GPU0）、`a100-gpu4`、`a100-gpu5`；同一 case 不得被两个 worker 领取，A0 的 `027_p2` 不进入 sentinel manifest。launcher 必须使用独立 tmux/session 名称和 `E192_*` 日志前缀，以保证叠加运行时可审计。

## 🔍 本轮 corrected sentinel 执行结果

- 三条 A0 sentinel 均完成并回收：`026_p1`（local GPU0）、`082_p1`
  （A100 GPU4）、`086_p2`（A100 GPU5）；`027_p2` 未启动、未进入评测。
- 评测 `evaluated=3/3`、`errors=0`，但 replay tolerance 仅 `026_p1` 全部满足。
  `082_p1` 的 penetration/contact/orientation 差分别为 `+0.0826/-0.0492/+17.33°`；
  `086_p2` 的 leg/position/orientation 差分别为 `+0.0822/+3.26cm/-1.78°`。
- 依照优先级 1，正式判决为 `INCONCLUSIVE_BASELINE_DRIFT`；A2 canary 与 A2
  Full 不得启动。完整表格见 [log269](../log/269_E192_corrected_box_sdf_sentinel_results.md)。

### 2026-08-09 用户执行覆盖：跳过 A0，改用 Ada6000

用户明确要求本次继续执行 A2 并跳过 A0；随后又明确指定“不使用 A100，改用远程
Ada6000 两卡并行”。该执行授权覆盖上面的 A0 stop-loss 和三卡资源编排，但不改动
A2 阈值、case 集、canary stop-loss 或 C1–C7 定义：

- A0 sentinel 不重复启动，历史/纠正后的 A0 产物只作为记录，不作为可比基线授权。
- A2 canary 使用远程 `a6000-2gpu` 的 RTX 6000 Ada GPU0/1，worker 为
  `ada-gpu0` / `ada-gpu1`，队列为 2/1。
- 用户在 Full 启动后补充要求使用本机 GPU0 + 远程 Ada GPU0/1 三卡叠加；
  Full 因此固定为 `local-gpu0 / ada-gpu0 / ada-gpu1 = 5/5/5` 串行队列。
- Ada canary 仍必须通过原注册的 stop-loss；若触发，仍禁止启动 Full。

### 2026-08-09 用户 stop-loss 豁免

Ada canary 触发 `INCONCLUSIVE_GATE_COLLAPSE` 后，用户明确指示“直接上 Full”。
因此 Full 作为用户豁免 canary stop-loss 后的诊断性执行启动；该豁免不把 canary
改判为 PASS，也不恢复原预注册判决链的因果有效性。Full 仍保持 15-case、
`1024 samples × 32 iterations × seed 0` 与 A2 阈值不变；资源按后续用户指令
修订为本机 GPU0 + Ada6000 GPU0/1 三卡 5/5/5。

## 👁️ 可视化

- 15/15 A2 self MP4，并与冻结 A0 生成 paired 对照视频
- 对每个发生 numeric migration 的 case 用 `/video-frames` 抽取首次接触、抬起峰值、搬运中段和放下帧
- box024 `026_p1` 与 `027_p2` 两端必须查看
- 重点区分“贴面承重”“沿面滑动”和“失去接触”；失去承重接触时 C2 失败，不得由汇总均值覆盖视觉证据

## 🚫 Non-goals

- 不测尺寸因果；正式结论上限是 box024 特异效应
- 不拆分 `max_violation_pct` 与 `hard_floor_m` 的单独贡献；A2 是一个冻结策略包
- 不做阈值 sweep，不搜索最优参数
- 不叠加 E194 gravcomp；E192 的 object body 保持 `gravcomp=0`
- 不测 partner/support proxy，不改变物体 actuator gain
- 不改 E193 抓取拓扑或 `contact_hdmi_target_source`
- 不做 PRG on/off 消融
- 不跑 box001/box023/bucket/desk
- 不下 RL/Holosoma 结论

## ✅ 执行前 checklist

- [x] 用户确认 estimand 降为 `BOX024_SPECIFIC_THRESHOLD_EFFECT`
- [x] 用户确认 A2：`hard_floor -0.020→-0.015`、`max_violation_pct 0.10→0.05`、`min_sdf=-0.010` 不变
- [x] 用户同意增加直接 CEM selected-candidate 与固定深度物理诊断
- [x] 用户批准本轮 3 条当前代码 A0-sentinel（排除 `027_p2`）；Full 总上限相应为 18 条
- [x] 用户同意 `INCONCLUSIVE_BASELINE_DRIFT` / `INCONCLUSIVE_GATE_COLLAPSE` 与 signed DiD 统计规则
- [x] 用户确认 E192 在 no-gravcomp 基础上执行
- [x] 用户确认本轮 sentinel 使用本地 GPU0 + 远程 A100 GPU4/GPU5 并行（2026-08-09 更新）
- [x] 实现脚本和 observation-only 指标，并通过单元/合同测试
- [x] scene snapshot、SHA、resolved config 与 3 条 sentinel manifest 审计通过
- [x] corrected 3-case sentinel 完成并完成 baseline replay 评测
- [x] baseline drift stop-loss 触发；不启动 A2 canary/full
