# E191 — box024 物体 tracking / 手物穿透根因诊断与度量补齐

## Context

R018 侧分析（`SUGAR-private/docs/analysis/R018-9_to_R018-13_...md` §9/§10）已经把 SPIDER→RL 的问题
收敛到两条：provenance 混杂、以及"上游 gate 预测不了下游崩塌"。承接它的 SPIDER 侧行动计划
（`docs/plan/R018_spider_side_followup_plan_CN.md`）要求做两件事：动作 A 建新的 RL 导向筛选口径，
动作 B 优化 E167A 基底重定向。

在此之上，用户点出一个尚未归因的现象：**box024 的物体 tracking 差** —— 大部分 case 的物体
**非机器人一侧高度远低于参考轨迹**（`track_obj_pos_err_cm_mean` 13.64 vs box004 12.01 / box001 11.26），
且**手物穿透异常高**（PRG 0.378 / noPRG 0.391 vs box004 0.14–0.15、box001 0.18–0.21），只有
`027_p1/p2` 明显好一些。

本次调研已经把这两个现象归到**同一个物理建模缺陷**上，而且这个缺陷**不是 box024 独有的，
是全部 CORE4D box 共有的、只是被 box024 的几何放大**。本计划的目标是：先用零 CEM 算力的离线诊断
把这个缺陷量化坐实、把缺失的度量补进 eval，再据此决定要不要动物理模型。

> 用户已确认的边界：**本轮只做诊断与度量，不重跑 CEM**；后续若重跑，范围定为 box024(9) + box004(6) 对照；
> 新的分侧高度指标**只观测、不进硬门**。

---

## 已查明的机制（本次调研的核心结论）

### 发现 1：物体不是自由体，而是被一套**过软的 6-DoF 位置伺服**钉在参考轨迹上

`scene_act*.xml` 把 object 的 `<freejoint>` 换成 6 个 slide/hinge 关节 + 6 个 `<position>` 执行器，
运行时由 `examples/run_mjwp.py:1208-1252` 注入增益，且每步把 object 通道的 ctrl 硬复位到参考 qpos
（`examples/run_mjwp.py:1350-1356`）。E189 实跑的 resolved config：

```yaml
contact_guidance: true
init_pos_actuator_gain: 500.0    # N/m，P-only，无重力前馈
init_rot_actuator_gain: 50.0     # N·m/rad
guidance_decay_ratio: 1.0        # 不衰减，最终 rollout 仍是全增益
residual_gain_ratio: 1.0
partner_force_scale: 0.0         # partner 完全没建模
```

所有 CORE4D box 的质量都被写死成 **5.0 kg**（与体积无关）。于是稳态下垂

```
sag = m·g / kp_pos = 5.0 × 9.81 / 500 = 9.81 cm
```

**实测（E189 noPRG，MuJoCo FK 复算，抬起帧）与该预测吻合**：

| object | 质心下垂 | 机器人侧底缘 Δz | 非机器人侧底缘 Δz | 不对称量 | 最长半轴（力臂） |
|---|---:|---:|---:|---:|---:|
| box004 | −8.1 cm | **−7.69 cm** | **−8.46 cm** | 0.77 cm | 0.224 m |
| box001 | −9.2 cm | −7.14 cm | −11.20 cm | 4.06 cm | 0.406 m |
| **box024** | **−10.1 cm** | **−6.43 cm** | **−13.69 cm** | **7.26 cm** | **0.491 m** |

→ **用户观察到的"非机器人一侧远低于参考"被完整复现并量化了**，而且不对称量与力臂严格单调。

### 发现 2：partner 完全没有建模，且 5 套已实现的 partner 机制**全部关闭**

box024/004/001 的 scene 里没有 equality、没有 mocap、只有一个机器人。仓库里存在
`partner_force`（`mjwp.py:3474`）、`support_proxy`（`:3738`）、`support_dynamic`、weld anchor、
`mocap_partner`（`:4292`）五套 partner 机制，且在 `workspace/core4d_collab_retarget/`（E004–E018b）
被验证过，但**没有一套接进 E163→E167A→E172/E173/E189 这条生产链**。一个机器人独自托 5 kg 的箱子，
差额全部由那套过软的伺服兜底 —— 这就是下垂的来源。

box024 的 0.98 m 长轴水平放置，机器人抓在离一端 0.29 m 处，**远端力臂 1.11 m**，
重力力矩由 `kp_rot = 50 N·m/rad` 独自承担 → 额外 2–6° 下倾 → 远端多掉 7 cm。

### 发现 3：`track_obj_pos_err_cm_mean` 目前主要在测建模伪影，不是重定向质量

误差分解（cm）：box024 dz −4.75 / dxy 10.75；box004 dz −3.22 / dxy 8.94；box001 dz −3.54 / dxy 8.52
（全帧均值；抬起帧的 dz 达 −8～−10 cm，p10 达 −13 cm）。所有 box 的总误差都落在 10–13 cm 这个
由 `m·g/kp` 设定的地板上，20 cm 的 `object_pos` 门只剩 ~7–10 cm 余量。
**这直接解释了 R018 §9.6① "上游 tracking 指标分不开崩塌组与存活组"** —— 该指标被一个恒定伪影占满了。

### 发现 4：手物穿透**大部分是参考轨迹自带的**，不是 CEM 产生的

用 eval 自己的 SDF 探针（`core_metrics.py:435 geom_object_sdf`）跑**参考轨迹本身**：

| object | 参考 pen<0 帧占比 | 参考 pen<−3mm | 参考最深 | 物理 run pen<0 | run 最深 |
|---|---:|---:|---:|---:|---:|
| box004 | 0.508 | 0.436 | −1.11 cm | 0.288 | −0.86 cm |
| box024 | 0.602 | 0.476 | −0.94 cm | 0.387 | −1.20 cm |
| box001 | 0.628 | 0.531 | −1.29 cm | 0.322 | +13.9 cm |

→ 参考里手就已经陷在箱子里 50–63% 的帧、最深约 1 cm；**物理仿真其实是在减轻穿透，不是制造穿透**。

### 发现 5：手物穿透在 reward 里**根本没有惩罚项**，唯一约束是 CEM 硬门，而 box024 顶在门上

E167A 链的 resolved 值：`hand_object_deep_penalty_scale: 0.0`、`surface_band_penalty_scale: 0.0`、
`robot_object_penalty_geom_names` 只含 head/torso/pelvis/shoulder/elbow（**不含 lh/rh**）。
唯一的界是 CEM 精英门 `cem_hand_gate_hard_floor_m: -0.020` / `max_violation_pct: 0.10`。

实测 `hand_object_con_dist_min_m`：**box024 = −0.0200（正好顶死硬地板）**，box004 −0.0115、box001 −0.0096。
配合接触质量：box024 `contact_in_mask` 0.816（最高）但 `contact_3mm_in_mask` 0.288（最低）——
即 §9.6③ 的"只跟不抬"指纹在 box024 上最极端。

### 发现 6：`crossObject_candidate` 是**配置标签**，不是数据谱系 —— R018 §9.1 的分层不存在

**字符串本身是硬编码常量**（`e173_common.py:104`），由 `build_prg_cem_manifest.py:294` 原样写进
`spider_method_id`；box001/023/024 共 53 个 case 共用同一个值，不可能携带 per-case 的数据来源。

**"跨"的是配置**，三条证据：
1. 紧邻的 `SOURCE_CONFIG_ID = "E170_PRG_lowerbodyPhysics_softPenalty_candidateGate"` 与 E170 自己的
   `METHOD_ID`（`export_box021_user_approved_rl.py:25`）逐字节相同；区块注释即
   `# --- Frozen CEM / scene contract (E170 PRG cross-object candidate) ---`。被搬运的是那份冻结的
   CEM/scene contract（16 腿 geom 对、penalty 2.0、margin 0.02 m、gate floor、CEM 预算）。
2. `plan/189` 表格行明令：`| E173 method ID | ... | **不写入 target_variant_id** |`
   —— `target_variant_id` 才是数据谱系字段。同 plan §72：「E173 将 PRG 定义为**跨物体候选配置**，非已验证 default」。
3. `_candidate_` = 该配置只在 box021 验证过，搬到别的物体只算候选；`_r1` = revision 1。

**数据实际全是 box024 自己的**（`task_info.json`）：
`source_qpos = .../trimmed/20231011-027-person1-Box024_with_obj_original.npz`、`object_name = Box024`、
`object_model_rel = box/box024_m.obj`、`source_scene = .../box024_person1/scene.xml`；
contact mask 指向 `.../contact_masks/dcv3_omnirt_v2_ref_fk_box024_.../raw_contact_mask_3cm.npz`；
hand target 是 `ref_fk` + 空路径（由本 case 自己的 reference FK 导出）。
（`omnirt_v1/v2` 也不是别的物体：v2 是同 case 的救援路线，`e173_common.py:97`
`V1_INFEASIBLE_STATUS = "omniretarget_infeasible"  # only status eligible for v2 rescue`；box024 9 例中 4 例走 v2。）

**R018 §9.1 那张表比的是两代命名口径，不是两种数据来源**（全量 dump `method` 列所得）：

| 实验 | object | `method` 列 | `target_variant_id` | 实际有 PRG？ |
|---|---|---|---|---|
| E170 | box021 (28) | `E167A_zOnlyBody`（记的是 **base reward 名**）| `ref_fk` | **有**，override 就叫 `core4d_E170_box021_*_PRG` |
| E172 | box004 (6) | `E172_E170PRG_crossObject_candidate_r1`（**复合方法 ID**）| `ref_fk` | 有 |
| E173 | box001/023/024 (53) | `E173_E170PRG_crossObject_candidate_r1` | `ref_fk` | 有 |

`target_variant_id` 五物体全为 `ref_fk`，完全一致；box021 同样有 PRG，只是它那一代没把 PRG 写进 method 字符串。

**净影响：** §9.1 的 provenance 分层要撤，连带 §9.6④「原生 vs 跨物体候选是 object 级最干净的判别量」失去依据；
R018 动作 B 里「为 box004/001/023/024 产出原生重定向以消除 provenance 混杂」这条 **P1 行动实际无事可做**
（它们本来就是原生的），资源应转向发现 7/8 指出的真实杠杆。

> 📌 **待核**：R018 把 `E174/.../e174_case_metrics.tsv` 标为「OmniRetarget bucket 对照」，但其 method 列是
> `E174_E170PRG_nonbox_candidate_r1`，对应 `plan/190_E174_bucket_desk_move2_full_pipeline_plan.md`
> —— 看起来是 SPIDER 侧 bucket/desk 流水线而非 Omni 对照。这会影响「对称审计」结论，Stage A 顺带核实。

### 发现 7：reward 几何是**全物体共用的一套固定米制阈值**，且与尺寸完美共线

`examples/config/override/` 下 **156/156** 个 `core4d_dcv3_omnirt_v*_ref_fk_*.yaml`
（box001×30 / box004×6 / box021×28 / box023×16 / box024×10 / box026×17）全部继承同一个
`core4d_E167_box004_082_p1_E167A`。**box021 也不例外** —— 它的链同样穿过全部 5 个 box004 case 级祖先，
其中 `core4d_E163_box004_082_p1_narrowSurfaceBand`（定全部 `surface_band_*`）和
`core4d_E156_box004_082_p1_gateA`（定全部 `cem_*_gate_*`）是 reward 几何的真正来源。

> ⚠️ **对上一稿的撤回**：不存在"box024/box001 被污染而 box021 干净"的差异。配置层对五个物体**完全一致**，
> 因此**不能**用来解释 R018 §9.1 的原生/跨物体差距。SPIDER 内部各物体的对比是内部公平的。

真实问题在别处：**34 个带长度量纲的字段在 7× 体积跨度（0.036→0.256 m³）上逐字节相同，从未做过 per-object 重标**。
`plan/189` §147 书面要求过建 `core4d_E167_{object}_*_E167A`，但 box024/box001 的**根本不存在**；
且该要求本身写明"reward 字段级 diff 必须为空"——是 provenance 卫生，不是重标定
（旁证：`core4d_E167_box021_035_p1_E167A.yaml` 等确实存在、零引用、与 box004 版逐字段相同）。
`log/233:19` 把 `object-agnostic` 当作正当性——这对文件出身意图成立，对参数量纲语义不成立。

**有实测 binding 证据的参数（只列这些，不搞"34 个都有问题"）：**

| 参数 | 值 | box024 证据 |
|---|---|---|
| `cem_hand_gate_hard_floor_m` | −0.020 | `hand_object_con_dist_min_m` = **−0.0200** 顶死；box004 −0.0115 / box001 −0.0096 |
| `cem_leg_gate_hard_floor_m` | −0.005 | 3 条大箱 PRG scene reject 全撞此线（起始重叠 −0.016/−0.011/−0.009，`log/233:54`）|
| `cem_posture_gate_*` | 0.10/0.12/0.18 m | `cem_posture_gate_fallback_used` **0.0888** vs box004 0.0367 / box001 0.0204（2.4–4.4×）|
| CEM 门整体 | — | `cem_gate_valid_frac` **0.8485** 最低；`cem_hand_gate_selected_valid_frac` 0.8725 最低 |
| `object_lift_sigma` | 0.05 m | 伺服下垂已达 10 cm → lift 奖励在 box024 上基本失效（与发现 1 耦合）|

`surface_band_width_m: 0.003` **不**列入——3mm 表面带是局部接触判据，无证据表明随尺寸失效。

### 发现 8（本次调研最重要的方法学结论）：三重共线，现有设计无法归因

"大箱失败"目前有三个候选机制，**与物体尺寸完美共线**：

| | 机制 | 随尺寸如何变 |
|---|---|---|
| (a) | 固定米制 reward/gate 几何相对容差过紧（发现 7） | 阈值不变、物体变大 |
| (b) | 物体伺服过软 + partner 未建模，力臂放大（发现 1/2） | 力臂 0.224→0.491 m |
| (c) | 大箱无几何闭合，只能压平面（E189 改口的"抓取拓扑"） | 拓扑随尺寸变 |

只有 5–6 个物体、每物体一套固定阈值 → **该设计从来没有能力区分 (a)(b)(c)**。
因此 **E173 "大箱退化主因 hand_penetration（手陷大平面）"（`log/233:113`）与
E189 "真正决定的是抓取姿态/接触拓扑"（`log/265:240-245`）两条结论都应降级为未定论**。
`log/233:131` 自己提过"尺寸自适应 hand-collision margin"并明确划出范围，至今无实验做过。

---

## 计划

### Stage A（E191）：离线诊断 + 度量补齐 —— 零 CEM 算力 ← 本轮唯一要执行的

复用已落盘的 `results/{E170,E172,E173,E174,E189}/s6_downstream/**` npz，不跑任何仿真。

#### A1. 扩展 `core_metrics.py`（**纯追加**，旧列逐列零差异回归）

改动文件：`workspace/core4d/scripts/eval/core/core_metrics.py`
（在 `_table4_tracking_metrics`（:561）与 `evaluate_sequence`（:873）里追加；复用现成的
`geom_sample_points`（:405）/ `signed_point_box`（:383）/ `geom_object_sdf`（:435），不新写几何代码）

| 新列 | 定义 | 诊断什么 |
|---|---|---|
| `track_obj_z_err_m_mean` / `_p10` | 物体质心 z 的**有符号**误差（现只有 L2 范数，方向被吃掉了） | 发现 1/3 |
| `track_obj_xy_err_cm_mean` | 水平分量 | 误差分解 |
| `obj_side_near_z_err_m` / `obj_side_far_z_err_m` / `obj_side_z_asym_cm` | 底缘分侧高度误差；侧向轴取 pelvis→物体质心的水平单位向量，取物体 8 角点投影分侧后各侧最低点；只在抬起帧（`ref_z > z0+5cm`）统计 | **用户报的现象本身**；与 SUGAR 侧 R010-6 的探针口径对齐 |
| `obj_lifted_frame_frac` | 上面几项的分母 | 可解释性 |
| `object_guidance_force_z_N_p95` / `object_guidance_torque_Nm_p95` | `kp × (ctrl_target − 实际关节值)`，P-only 执行器可精确反推；与 `m·g = 49.05 N` 对照 | 直接把发现 1 变成一列数 |
| `ref_hand_geom_penetration_frac` / `_3mm_frac` / `_min_m` | 同一 SDF 探针跑在**参考轨迹**上 | **发现 4**：区分"目标自带"与"物理产生" |
| `hand_gate_floor_saturation_frac` | `min con dist ≤ −0.0195` 的帧占比（硬地板 −0.020） | 发现 5 |
| `object_mass_kg` / `object_half_extents_m` / `object_max_half_extent_m` / `grip_far_arm_m` | 物体几何 + 抓握中点到远端的距离 | 力臂回归的自变量 |

#### A2. 离线审计 runner + 全量表

新增 `workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py`
（照 `eval_E189_boxes_e167a_vs_prg.py` 的结构写），遍历 E170(box021 原生) / E172(box004) /
E173(box001/023/024) / E189(noPRG) / **E174(OmniRetarget 对照)** 的既有 npz，产出
`results/E191/audit/e191_object_support_audit.tsv` + `E191_object_support_report.md`。

**顺带补上 R018 计划点名的"对称审计缺口"**：Omni 侧 box/bucket target 第一次有了同口径的上游表。

#### A3. 预注册的可证伪判据（Stage A 的成败标准）

| # | 假设 | 通过 | 证伪 |
|---|---|---|---|
| H1 | 伺服下垂主导物体位置误差 | 6 组（5 box + Omni）质心 z 误差均值 ∈ [−13, −7] cm，且 \|dz\| 在抬起帧占总误差 ≥ 60% | 任一组 z 分量 < 40% |
| H2 | 不对称量由力臂驱动 | `obj_side_z_asym_cm` vs `object_max_half_extent_m` 跨 5 物体 Spearman ρ ≥ 0.8 | ρ < 0.5 |
| H3 | 穿透主要是参考自带 | 每个物体 `ref_hand_geom_penetration_frac` ≥ 0.45 且 ≥ run 侧同列 | run 侧显著高于 ref 侧 |
| H4 | box024 顶死 CEM 手门地板 | box024 `hand_gate_floor_saturation_frac` ≥ box001/box004 的 2× | 三者相当 |
| H5 | Omni 侧不共享此伪影 | Omni（E174）质心 z 误差显著小于 SPIDER 侧 → 证明这是 SPIDER 特有建模选择 | Omni 同样下垂 ~10 cm（则该伪影与方法无关，R018 的对照仍成立） |

> H5 是本轮最有价值的一格：如果 Omni 侧没有这个下垂，那么 R018 里"Omni 物体 tracking 更好"这句话
> **有相当一部分是 SPIDER 的伺服伪影，而不是重定向质量差异** —— 会直接改写母分析的归因。

#### A4. 目视核验（按 `.claude/rules/experiment.md` §5，强制）

用 `/video-frames` 从已渲染视频抽帧，做 A/B 对照：
- box024 `026_p1`（不对称 −15.7 cm，穿透 0.521，最差）vs `027_p2`（不对称 −8.2 cm，穿透 0.065，最好）
- box024 `026_p1` vs box004 `086_p1`（同样过不了门，但力臂短）
抽帧点：首次接触、抬起峰值、搬运中段、放下。确认"远端拖地"与"手陷进大平面"两个视觉特征。

#### A5. 混杂分离：物体内力臂回归（**本轮判别力最高的一步，零算力**）

发现 8 说明跨物体比较永远分不开 (a)(b)(c)。出口在于：**在单个物体内部，物体几何与全部阈值都是常量，
唯一变化的是每个 case 的抓握位置 → 力臂**。（已量到某 box024 case 的抓握中点离一端 0.29 m、离另一端 1.11 m。）

对全部 87 个 box case（5 物体）计算 `grip_far_arm_m`（抓握中点沿最长轴到远端的距离），做两层回归：

| 层 | 自变量 | 因变量 | (b) 成立的预测 | (a) 成立的预测 |
|---|---|---|---|---|
| **物体内**（box024 n=9、box001 n=28、box021 n=28 分别做） | `grip_far_arm_m` | `obj_side_z_asym_cm` | 显著正相关（ρ ≥ 0.6） | 无关（物体内阈值恒定） |
| **物体间**（n=5） | `object_max_half_extent_m` | `cem_posture_gate_fallback_used`、`hand_gate_floor_saturation_frac` | 弱 | 显著正相关 |

**判据 H6**：若物体内力臂回归显著 → (b) 被独立支持，且不依赖 (a)(c)；
若物体内不显著、而门触发率只随物体尺寸变 → 指向 (a)。两者都显著则需 Stage B 才能定量拆分。
若 9 个 box024 case 的 `grip_far_arm_m` 方差过小（< 0.1 m）导致回归无功效，**照实记录功效不足，不强行下结论**。

#### A6. 配置 provenance 审计表（永久产物）

把发现 7 的链路审计固化成 `results/E191/audit/e191_config_provenance.tsv`：
每个 case 一行，列出 17 节链路、外来 case 级祖先数、以及那 34 个米制字段的取值。
用途：(i) 让"全物体共用一套固定阈值"这件事在任何后续实验里一眼可见；
(ii) 给 R018 动作 B 的"原生对原生"提供真正的配置层证据，而不是靠方法 ID 字符串判断。

#### A7. 产出

- `workspace/core4d/log/266_E191_object_support_offline_audit_results.md`（实验记录：目的/参数/命令/结果/结论）
- `workspace/core4d/EXPERIMENT_TRACKER.md` + `progress.md` 增行
- 一份**决策备忘**：Stage B 是否值得开、开哪一臂；以及 (a)/(b) 的相对权重
- 给 R018 侧的勘误四条：
  1. **发现 6 —— §9.1 的 provenance 分层不存在**：`crossObject_candidate` 是配置标签；
     `target_variant_id` 五物体全为 `ref_fk`；E170 与 E172/E173 只是两代命名口径不同，box021 同样有 PRG。
     连带 §9.6④「原生 vs 跨物体候选是 object 级最干净的判别量」失去依据；
     动作 B 的 P1「产出原生重定向」无事可做。另需核实 E174 是否真是 Omni 对照。
  2. 发现 3 —— `track_obj_pos_err` 是伺服伪影主导，为 §9.6① 「上游 tracking 分不开崩塌/存活」提供机制解释
  3. 发现 7 —— 156/156 配置共用一套固定米制阈值，**不能**解释 §9.1 的物体间差异（对所有物体一致）
  4. 发现 8 —— (a)(b)(c) 三重共线，E173/E189 关于「大箱失败主因」的两条结论降级为未定论

---

### Stage B（E192，**预备，需另行批准后才启动**）：15 例 canary 证伪

范围按用户确认：**box024 全部 9 例 + box004 全部 6 例作对照**（box004 力臂最短，预期几乎不动 →
若 box004 也大幅变化，说明改动不是走力臂通道，假设被证伪）。

| 臂 | 改动 | 现在能不能直接跑 | 预测 / 检验哪条机制 |
|---|---|---|---|
| A0 | E189 noPRG / E173 PRG 现状 | 已落盘，无需跑 | 基线 |
| A1 | `partner_force_scale: 0.5` | **能**，纯 Hydra override；无 point/spring 时走 `mjwp.py:3532` 的 COM 上托分支，不碰 freejoint 假设 | 检验 **(b)**。下垂 ~10→~5 cm；**倾斜不变**（COM 施力无力矩） |
| A2 | `init_pos_actuator_gain: 500→2500`，`init_rot_actuator_gain: 50→250` | **能**，纯 override | 检验 **(b)**。下垂 ~2 cm、倾斜 ~1°；**风险：更硬的弹簧把箱子更用力顶进手里，穿透可能变差** |
| **A4** | **尺寸自适应阈值**：只放宽 A5/发现 7 里有 binding 证据的三个——`cem_hand_gate_hard_floor_m`、`cem_leg_gate_hard_floor_m`、`cem_posture_gate_max_z_drop_m`，按 `object_max_half_extent_m / 0.224`(box004 基准) 线性缩放 | **能**，纯 override | 检验 **(a)**。这是 `log/233:131` 提过但从未做的"尺寸自适应 margin"。**box004 应几乎不动**（缩放系数=1），box024 系数 2.19 |
| A3 | 偏心 partner 支撑（在 partner 抓握点施力，同时修下垂与倾斜） | **不能**，需改核心代码 | 检验 **(b)** 的完全版，唯一能同时修下垂与倾斜的臂 |

**A1/A2（机制 b）与 A4（机制 a）都是纯 config，可以并行跑，这就把发现 8 的三重共线在实验层拆开。**
（(c) 抓取拓扑无法用 config 检验，只能作为 A1/A2/A4 全部无效后的残差解释。）

**A3 的实现障碍（已定位，需在 Stage B 前评估）**：
- `_apply_partner_force` 的 point/spring 分支在 `mjwp.py:3547` 硬读
  `qpos[obj_qadr+3 : obj_qadr+7]` 当四元数 —— 对 `nq_obj=6` 的 contact_guidance 物体是越界/错读。
  需按 `scene_act_meta.json` 的 `euler_convention: XZY` 从 6 个 slide/hinge 关节重建位姿（~30 行，隔离可逆）。
- `support_proxy_enabled` 在 `mjwp.py:3763-3766` **显式拒绝** `contact_guidance`，走这条路要先把物体换回 freejoint，
  改动面远大于 A3，不建议。

判据（预注册，box024 9 例）：`obj_side_z_asym_cm` ≤ 3.0 且 `track_obj_pos_err_cm_mean` ≤ 6.0，
≥7/9 例达成；同时 box004 6 例任一 12-gate 门**不得回退**；且必须过 A4 同款目视核验。

---

### Stage C（E193，**预备，gated on B**）：手物穿透

**必须放在 B 之后** —— 因为发现 1/5 表明穿透部分是伺服把箱子顶进手里造成的，先调穿透会调错对象。

| 顺序 | 改动 | 依据 |
|---|---|---|
| C1 | `cem_hand_gate_hard_floor_m` −0.020→−0.010，`max_violation_pct` 0.10→0.05 | box024 顶死现有地板（发现 5） |
| C2 | `hand_object_deep_penalty_scale` 0→5.0，`threshold_m` 0.01（E119/E120 用过的值，本链里是死代码） | reward 里目前零惩罚 |
| C3 | **不是**去建 `core4d_E167_box024_*_E167A`（`plan/189` §147 那条要求是纯 provenance 卫生，"reward 字段级 diff 必须为空"，建了也不改数值）。真正要做的是把 A4 里被证实有效的尺寸缩放**固化**成 per-object base，并在 provenance 审计表里显式标注缩放系数 | 发现 7/8：问题是量纲，不是文件名 |
| C4（P2） | 若 C1–C3 压不下去，回到 IK/`ref_fk` 阶段做手-物表面投影 | 发现 4：50–63% 穿透是参考自带的 |

判据：box024 `hand_object_physics_penetration_3mm_frame_frac` ≤ 0.20（现 0.378），
**同时** `hand_object_physics_contact_3mm_in_mask_frac` ≥ 0.288（不许靠"把手拿开"刷分）。

---

## 与 R018 两个动作的接口

- **动作 A（RL 导向筛选口径）**：`obj_side_z_asym_cm` 与 `ref_hand_geom_penetration_frac` 加入口径，
  **按用户决定只观测、不设硬门**，先用现有 by-case 表离线标定（呼应 R018 计划"避免过拟合到 38-case"的边界）。
  R010-6 已在 SUGAR 侧证明"远端拖地"直接决定 dual-side lift 成败，这是该指标的下游相关性依据。
- **动作 B（E167A 基底优化）**：Stage A 的 H1/H2/H5/H6 决定"基底优化"该往哪打。若 H5 成立
  （Omni 无此下垂），则动作 B 的首要目标从"降物体朝向误差"改为"修物体支撑建模"，优先级更高。
  **动作 B 的 P1「为 box004/001/023/024 产出原生（非 cross-object 候选）重定向」应删除** ——
  发现 6 证明它们本来就是原生的，`target_variant_id` 全为 `ref_fk`，无混杂可消除。
  释放的资源转向 (a) 阈值量纲（Stage B(E192) 的 A4 臂）和 (b) 物体支撑建模（A1/A2/A3 臂）。
- **勘误四条**：见 A7 产出。发现 6 直接撤销 R018 §9.1 与 §9.6④ 的 provenance 分层。

---

## 改动文件清单（Stage A）

| 文件 | 改动 |
|---|---|
| `workspace/core4d/scripts/eval/core/core_metrics.py` | 追加 A1 的新列（纯追加） |
| `workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py` | 新增，离线审计 runner（含 A5 力臂回归、A6 provenance 审计） |
| `workspace/core4d/scripts/eval/wrappers/eval_E191_object_support_audit.sh` | 新增，入口（照 E189 wrapper，`PYTHON_BIN=.venv/bin/python`、`unset MUJOCO_GL`） |
| `workspace/core4d/log/266_E191_object_support_offline_audit_results.md` | 新增，实验记录 |
| `workspace/core4d/plan/216_E191_object_support_offline_audit_plan.md` | 新增，本计划归档 |
| `workspace/core4d/EXPERIMENT_TRACKER.md` / `progress.md` | 增行 |

Stage A **不跑物理仿真**，因此按 `.claude/rules/experiment.md` §7 **不需要 scene snapshot**；
Stage B/C 启动时必须先调 `scripts/convert/snapshot_scenes.sh`。

---

## 验证

1. **回归零差异**（最关键）：在 E189 的 43 行上跑新旧 `core_metrics.py`，
   断言全部 158 个既有列逐值 bit-identical，只多出新列。不通过则改动不合格。
2. **交叉验证新列**：`obj_side_*` 与本次调研的独立复算数值一致
   （box024 near/far = −6.43/−13.69 cm，box004 −7.69/−8.46，box001 −7.14/−11.20，误差 < 0.5 cm）。
3. **量纲自检**：`object_guidance_force_z_N_p95` 在抬起帧应落在 `m·g = 49.05 N` 量级；
   `track_obj_z_err ≈ −force/500` 应自洽。
4. **判据表**：H1–H5 逐条判 PASS/REFUTE 并写进 log，**证伪的假设照实记录**，不改判据凑结论。
5. **目视**：A4 的抽帧对照贴进 log。
6. 跑 `ruff check .` / `ruff format .`。

```bash
# Stage A 全流程
bash workspace/core4d/scripts/eval/wrappers/eval_E191_object_support_audit.sh
# 回归检查
.venv/bin/python workspace/core4d/scripts/eval/runners/eval_E191_object_support_audit.py --regress-against \
  workspace/core4d/results/E189/s6_downstream/eval/full/e189_case_metrics.tsv
```
