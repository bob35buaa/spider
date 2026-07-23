# E174 实验计划：Bucket / Desk 非 box 物体 move2 全流程筛选及 Full CEM

_Core4D Phase 37 · 2026-07-23 · production screening plan（planning-only，待用户确认后执行）_

计划前身：E170 [plan/186](186_E170_box021_prg_full_validation_plan.md) · E171 [plan/187](187_E171_box022_box026_full_pipeline_plan.md) · E172 [plan/188](188_E172_box004_full_pipeline_plan.md) · E173 [plan/189](189_E173_box024_box023_box001_full_pipeline_plan.md)

---

## 📋 1. 决策摘要

E174 把 E170–E173 的冻结重定向算法**首次扩展到非 box 物体**：对尺寸介于 box023(0.036 m³) 与 box025(0.333 m³) 之间的 **bucket 类 + desk 类**物体，仅取 **move2（双人协作搬运）**动作，从 CORE4D raw authority 重新执行 data_construction_v3 `S0–S6` 主链路，并对本轮真正通过 S5 handoff gate 的全部 person-case 运行 Full CEM。算法与冻结配置**完全沿用 E170/E171/E172/E173**（`omnirt_v1→v2 rescue` + `ref_fk` + `rubber_hull` + `E170 PRG` 跨物体候选），不做 reward/route/geometry sweep。

固定决策如下：

- **物体范围（7 个，move2-only 入围）**：`bucket004 / bucket009 / bucket010 / bucket007 / bucket003`（5 个 bucket）+ `desk005 / desk007`（2 个 desk）。这是「尺寸 ∈ [box023, box025] 且存在 move2 序列」的全部 bucket/desk 物体。chair 类本轮不做（见 §2.3）。
- **动作策略（move2-ONLY，比 E17x 更严）**：只有 `move2*` 进入 production 全流程。`move1*` 本实验**明确 out-of-scope**（在 inventory 后作 `EXCLUDE_ACTION_NOT_MOVE2` 排除，保留终态在 registry 但不进 S2–S6）；`join/leave/rot/pass/strike/raise` 仍由 dcv3 `HIGH_RISK_ACTION_PREFIXES` 作 Stage0 `reject_action_not_first_line`。
- **优先级与调度**：GPU 队列与执行顺序按 **bucket（小→大体积）→ desk**，即 `bucket004 → bucket009 → bucket010 → bucket007 → bucket003 → desk005 → desk007`。小体积 bucket 最先出结果，早期确认非 box 几何是否系统性失败，再决定 desk（大体积 + 桌腿）是否值得全量。任一物体 DATA_NEGATIVE 不阻断其余。
- **raw 入口**覆盖 7 物体**完整 120 sequence / 240 person-case** 做 authority 闭合；production expected = **37 move2 seq / 74 move2 pc**；最终 authority 以 E174 fresh inventory 为准。
- S1 同时重算 3cm/5cm raw contact；3cm 主入口，5cm-only 进 recall review，不自动放行。
- **不导出 RL / partner**：E174 只产 case-level Full CEM 结果与用户标签（与 E172/E173 一致）。
- S2 遵循 dcv3 **非 box template 策略**（doc 04）：每物体 `{obj}_person1/2` template 默认 `manual_review_required`，**不能仅凭 MuJoCo load / render 自动置 clean**，必须经 `nonbox_template_review.tsv` 显式 `review_decision=approve_clean` → `clean_reviewed` 才进 S3。bucket 用 `nonbox_proxy_aabb_review` adapter（`bucket_wall_proxy_aabb`：底面+四侧壁）；desk 用 `nonbox_surface_voxel_review` adapter（`desk_surface_voxel_multibox_proxy_draft`：OBJ 表面 voxelize→多 box）。**bucket009_person1 与 desk005_person1 目录缺失 → 按该 proxy-review 流程补建**（desk005 走 surface voxel proxy）；多个 base 未 git-tracked → review 通过后 `git add -f`；所有 `_freejoint_legobj_e0**` / `_s2_` 历史污染变体**禁止**作 base。
- **物体碰撞几何（object 侧）与手部碰撞体（robot 侧）是两条正交轴**：物体凹几何由上述 `collision_policy`（bucket_wall / desk_surface_voxel）表达并冻结，不做 sweep；`hand_collision_variant_id=rubber_hull` 只改机器人手 `lh/rh`（doc 15），与物体形状无关。S3/S4/S5/S6 contract 与 E173 字段级一致；S5 的 rubber_hull 走 E174-scoped sidecar，不覆盖 source `scene_act.xml`。
- S6 用 E170 PRG 冻结配置作跨物体候选，先条件式 canary 再对全部 S5-ready 做 Full CEM，全部跑**本机 8×L20Y**。
- 用户负责最终人工标签与是否进入后续 RL/partner export；本轮不启动 RL。

> ⚠️ **解释边界**：「对 7 物体做全流程」表示所有 240 raw rows 都必须得到可追溯终态，不表示绕过数据 gate 强产 CEM row。若 fresh S1–S5 后某物体 0 ready，这是有效的 `DATA_NEGATIVE`。

> 🔬 **本实验核心科学看点**：这是 E170 PRG + rubber_hull 冻结算法**首次用于非凸非 box 物体**。凹几何的挑战落在**物体侧 `collision_policy` proxy 的保真度**——bucket 用 `bucket_wall_proxy_aabb`（底面+四侧壁近似空心筒）、desk 用 `desk_surface_voxel_multibox_proxy_draft`（表面 voxel 多 box 近似桌面+四腿）；proxy 若比 mesh 大一圈或桥接桌腿间隙，会产生 phantom collision / 错误接触面。**注意 `rubber_hull` 是机器人手侧碰撞体、与物体形状正交，不参与物体凹几何的表达**（早期草稿曾误把二者混为一谈，已更正）。E174 不改任何几何参数，而是通过 S2 object-only mesh/collision overlay 审查 + canary + Full CEM 视觉核验，**量化 proxy 失配的严重度**，作为「PRG+rubber_hull+非 box proxy 是否适用于凹几何物体」的第一份证据。

## 🔍 2. 历史证据与边界

### 2.1 Raw inventory prior（fresh 盘点，2026-07-23）

从 live raw root `.../CORE4D/CORE4D_Real/human_object_motions` 按 `object_metadata.json`（key=`obj_name`）精确召回；action 来自 `action_labels.json`（key=`date/seqid`）。7 物体 mesh 在 raw `object_models/{bucket,desk}/` 均存在。数字用于 S0 authority 漂移检查，不代替 E174 fresh inventory。

| 物体 | 类别 | 体积 m³ | raw seq | **move2 seq** | **move2 pc（production）** | move1 seq（排除） | 其它 seq（Stage0） | 日期 |
|---|---|---|---|---|---|---|---|---|
| bucket004 | bucket | 0.045 | 16 | 5 | **10** | 6 | 5 | 20231002, 20231003_1 |
| bucket009 | bucket | 0.092 | 8 | 3 | **6** | 2 | 3 | 20231002 |
| bucket010 | bucket | 0.120 | 16 | 7 | **14** | 3 | 6 | 20231002, 20231003_2, 20231011 |
| bucket007 | bucket | 0.179 | 18 | 6 | **12** | 6 | 6 | 20231003_1, 20231003_2 |
| bucket003 | bucket | 0.192 | 16 | 6 | **12** | 5 | 5 | 20231018, 20231020 |
| desk005 | desk | 0.238 | 9 | 3 | **6** | 3 | 3 | 20231023 |
| desk007 | desk | 0.241 | 37 | 7 | **14** | 7 | 23 | 20231023, 20231030 |
| **合计** | — | — | **120** | **37** | **74** | **32** | **51** | — |

move2 序列明细（date/seqid）：

- **bucket004**: 20231002/017,021; 20231003_1/012,014,016
- **bucket009**: 20231002/056,058,060
- **bucket010**: 20231002/029,031,033; 20231003_2/055,057,059; 20231011/115
- **bucket007**: 20231003_1/021,023,025; 20231003_2/021,023,025
- **bucket003**: 20231018/001,003,005; 20231020/064,066,068
- **desk005**: 20231023/028,030,032
- **desk007**: 20231023/046,048,050; 20231030/028,030,032,034

### 2.2 物体尺寸（mesh AABB，跨物体对照，非 gate）

| 物体 | 长×宽×高 (m) | 体积 m³ | 对角线 m | 量级 | 几何拓扑 |
|---|---|---|---|---|---|
| bucket004 | 0.32×0.46×0.30 | 0.045 | 0.64 | 小 | 空心筒 |
| bucket009 | 0.38×0.61×0.39 | 0.092 | 0.82 | 中小 | 空心筒 |
| bucket010 | 0.40×0.74×0.40 | 0.120 | 0.94 | 中 | 空心筒 |
| bucket007 | 0.55×0.57×0.57 | 0.179 | 0.98 | 中 | 空心筒 |
| bucket003 | 0.54×0.76×0.47 | 0.192 | 1.05 | 中大 | 空心筒 |
| desk005 | 0.40×0.74×0.80 | 0.238 | 1.16 | 大 | 桌面+四腿 |
| desk007 | 0.43×0.70×0.80 | 0.241 | 1.15 | 大 | 桌面+四腿 |
| _参考 box023(E173)_ | 0.39×0.30×0.30 | 0.036 | 0.58 | 小 | 凸箱 |
| _参考 box004(E172)_ | 0.45×0.35×0.26 | 0.041 | 0.63 | 小 | 凸箱 |
| _参考 box026(E171)_ | 0.63×0.47×0.39 | 0.116 | 0.88 | 中 | 凸箱 |
| _参考 box024/001(E173)_ | ~1.0×0.5×0.5 | 0.25 | 1.2 | 大 | 凸箱 |

E174 的两条科学轴：**(a) 尺寸** — bucket 覆盖 0.045–0.19 m³（对标已跑的小/中 box），desk 0.24 m³（对标大 box）；**(b) 凹几何 proxy 保真度** — 非 box 物体侧 `collision_policy` 用 proxy（bucket_wall / desk_surface_voxel）表达凹形，是独立于尺寸的全新失败轴。

**基于 E173 KEY FINDING 的先验预测**（PRG numeric pass 随尺寸单调下降：小箱~80%、大箱~35%）：若非 box 只是「尺寸效应」，预期 bucket ~60–80%、desk ~35%；若物体 proxy（bucket 侧壁 / desk 表面 voxel）失配引入额外接触/穿透误差，实际会**显著低于该尺寸先验**。二者之差即「凹几何 proxy 惩罚」的量化。

### 2.3 为何本轮不做 chair

chair 类（chair021 等，0.21–0.29 m³）虽然也在尺寸区间且 move2 序列最多，但椅子有**靠背 + 细椅腿**，凹度比 bucket/desk 更极端，surface voxel proxy 失配预计最严重。E174 先用 bucket（缓凹）+ desk（中凹）建立非 box 基线证据，chair 留待非 box 可行性确认后的下一批（潜在 E175）。

### 2.4 历史 overlap：仅作 warning prior，不作 completion 证据

7 物体在 E016–E081（Phase ≤10）与 E125–E145（"RL Bridge & Nonbox"）阶段有大量 **pre-PRG / pre-rubber_hull** 处理痕迹，表现为 `{obj}_person*_freejoint_legobj_e0**`、`{obj}_s2_*` 等污染变体目录。这些是旧算法（无 move2-only funnel、无 rubber_hull、无 E170 PRG 冻结配置）的产物：

| 类别 | 可复用 | E174 不得继承 |
|---|---|---|
| `{obj}_person{1,2}` clean base（scene.xml/task_info） | 作 S2 review 的 base（须经 object-only overlay review → `clean_reviewed` + SHA + git add -f） | 其历史 CEM/yield 结论、历史 review_decision |
| `{obj}_person*_freejoint_legobj_e0**` | ❌ 禁止作 base | 全部（pre-PRG 污染） |
| `{obj}_s2_*` | ❌ 禁止作 base | 全部 |

### 2.5 E170 PRG + rubber_hull 的复用边界

与 E172/E173 一致：PRG 对 lower-body 有定向收益但伴随 contact trade-off、gate-health 恒 0。E174 将 PRG + rubber_hull 定义为**跨物体候选配置**，非已验证 default。不做 P/R/G 消融、不做 hull-vert sweep、不根据 canary 质量临时调参；所有 S5-ready rows 用同一冻结配置，实验后按物体/尺寸/凹度报告可迁移性。

## 🎯 3. Scope、authority 与固定配置

### 3.1 In scope

- 7 物体全量 raw inventory（120 seq/240 pc）与 action/inventory 筛选（move2-only）
- 3cm/5cm raw-contact 重算、summary、timeline、reject taxonomy
- 每物体 `person1/2` clean source template fresh build/audit（含 bucket009_p1 / desk005_p1 补建）、visual package、SHA snapshot
- `omnirt_v1/ref_fk` primary Stage2b + 限定 `omnirt_v2/ref_fk` rescue
- target gate、kinematic replay、visual QC、S5 handoff
- rubber-hull + E170 PRG cross-object candidate 的 canary 与 Full CEM（本机 8×L20Y）
- 全量 metrics、分组分析（**bucket vs desk、按体积、非 box vs 已跑 box 对照、凹几何失配定性**）、视觉抽查、用户人工标签准备
- S0–S6 registry、completion audit、结果日志、tracker 更新

### 3.2 Out of scope

- `move1` 及所有非 move2 动作的下游处理
- chair 类及其它未列物体
- `fingertip_aware`/`adaptive`/`omnirt_original`/`omnirt_v2_replace`、任何 wrist-to-fingertip replacement
- raw-contact threshold、reward、gate、CEM budget、**hull vertex / 物体碰撞几何** 或 scene physics sweep
- 为增加 positive 改变 action policy、target route，或对非 `omniretarget_infeasible` row 用 v2
- RL 训练 / RL-ready export / partner export / 默认配置升级
- 把任何历史（pre-PRG bucket/desk 结果、`_e0**`/`_s2_` 变体）计入 E174 fresh completion count

### 3.3 Authority 定义

| Authority | 唯一来源 | 必须断言 |
|---|---|---|
| Raw authority | E174 `s1_raw_contact/inventory/inventory.tsv` | object ∈ 7 targets；case_id 唯一；预期 120 seq/240 pc |
| Action scope | E174 inventory action 列 | 只有 move2 标 `production`；move1 标 `EXCLUDE_ACTION_NOT_MOVE2`；其余标 Stage0 reject |
| S1 authority | E174 fresh 3cm/5cm manifests | 每 raw row 两档均有状态，不静默缺失 |
| Stage2b authority | E174 v1 primary + v2 rescue manifests | 每 eligible row 有 v1 终态；每 v1 infeasible row 有 v2 终态与唯一 selected variant |
| S5 authority | E174 registry + handoff manifest | 只含通过 template/Stage2b/target/visual gate 的 row |
| Full CEM authority | `S5_READY_SET` 精确快照 | `full_expected = count(S5_READY_SET)`，不预硬编码 |
| 人工 authority | E174 user review TSV | Codex 不写/覆盖 `manual_*` 字段 |

`S5_READY_SET` 定义：

```text
inventory/action eligible (move2 only)
AND selected raw-contact route approved
AND template audit pass
AND selected Stage2b variant pass
AND target gate pass
AND visual QC pass
AND rubber-hull/PRG sidecar contract pass
```

### 3.4 Raw-contact promotion policy

与 E172/E173 字段级一致（单 label，禁 3cm/5cm 双份 target 覆盖）：

1. `3cm pass` → 直接进 production Stage2b 候选
2. `3cm review` → raw-contact evidence review 决定 pass/reject
3. `3cm fail + 5cm pass/review` → `5cm_recall_review`，仅明确双手/目标人物接触证据才晋级
4. `5cm fail` → 终态 `REJECT_RAW_CONTACT`
5. 已晋级 3cm row 不再生成 5cm duplicate

> ⚠️ **非 box raw-contact 注意**：bucket 空心 / desk 桌面—人手接触点分布与 box 抓握不同。3cm surface-distance 用的是 mesh 表面距离，非 box mesh 已正确加载即可直接复用，无需改阈值；但 raw-contact review 时须留意「手接触桌腿 / bucket 内壁」是否被 mesh 正确表达。

### 3.5 固定 retarget fallback 与 CEM 配置（与 E171/E172/E173 字段级一致）

| 轴 | E174 冻结值 | 说明 |
|---|---|---|
| Retarget primary | `omnirt_v1` | 所有 S3 eligible row 必先运行 |
| Retarget rescue | `omnirt_v2` | 仅 v1 `omniretarget_infeasible` 时运行；非 A/B sweep |
| Target route | `ref_fk` | E098 公共 contract |
| **Object collision policy** | bucket=`bucket_wall_proxy_aabb`；desk=`desk_surface_voxel_multibox_proxy_draft`（`target_cells=26`+inward shrink） | 物体侧凹几何 proxy，冻结不 sweep（doc 04） |
| Base reward/method | `E167A_zOnlyBody` | 与 E170–E173 PRG 基座一致；per-object base override 见下 |
| Hand collision | `rubber_hull` | 机器人手 `lh/rh` mesh 凸包，maxhullvert=64，mesh-aware SDF；与物体正交（doc 15）；E174 sidecar |
| Lower-body physics | 16 geoms ↔ object pair | 沿用 E169/E170 P 配置 |
| Lower-body penalty | scale `2.0`、margin `0.02m` | 沿用 E169/E170 R 配置 |
| Candidate gate | min SDF `0.005m`、max violation `0.02`、hard floor `-0.005m` | fallback=`least_violation` |
| CEM budget | seed `0`、samples `1024`、opt steps `32` | canary smoke `samples=64, opt_steps=4` |
| Source config ID | `E170_PRG_lowerbodyPhysics_softPenalty_candidateGate` | 冻结来源 |
| E174 method ID | `E174_E170PRG_nonbox_candidate_r1` | 不写入 `target_variant_id` |

**Per-object base reward override**：每物体以该物体一个 canonical move2 case 生成对应 `core4d_E167_{object}_{seq}_{person}_E167A` base override（reward 权重字段与 `E167A_zOnlyBody` 逐字一致，仅换 case-specific scene/trajectory 路径）；选取 case 记录在 registry，reward 字段级 diff 必须为空。

`omnirt_v2` 使用 E168/E171/E172/E173 同款 Phase4 rescue contract（`enable_constraint_relaxation/enable_foot_z_constraint/foot_slide_penalty_weight=1.0/enable_contact_preservation/object_penetration_tolerance_scale=0.8/replace_wrist_with_fingertip=false/include_fingertip_centers=false`）；`omnirt_v1` 保持 primary contract（Phase4 flags 全关）。v1/v2 选择规则、独立目录/registry、`rescue_of`/SHA 记录、Full CEM effective config 字段级一致等约束**与 E173 §3.5 完全相同**。

### 3.6 物体碰撞几何决策（非 box，按 dcv3 proxy 策略，仍不 sweep）

E174 物体侧碰撞几何**遵循 dcv3 非 box template 策略**（doc 04），**不走 box 的 mesh-AABB box builder**：

- **bucket**（bucket004/009/010/007/003）：`nonbox_proxy_aabb_review` adapter，`collision_policy=bucket_wall_proxy_aabb`（底面 + 四侧壁 box geoms，近似空心筒）。
- **desk**（desk005/007）：`nonbox_surface_voxel_review` adapter，`collision_policy=desk_surface_voxel_multibox_proxy_draft`（OBJ 表面 voxelize，`target_cells=26` + 轻微 inward shrink，合并为 `object_collision` + `object_collision_voxel_*` 多 box geoms，避免比 mesh 大一圈）。

现有 7 物体 template 已带上述 proxy policy（已核实 bucket004/007=`bucket_wall_proxy_aabb`、desk007=`desk_surface_voxel_multibox_proxy_draft`），E174 **冻结复用这些 proxy policy，不做 target_cells / shrink / geom 数量 sweep**。这两条 policy 是本实验唯一表达物体凹几何的机制；`rubber_hull` 只作用于机器人手 `lh/rh`（doc 15），与物体形状正交，二者不得混淆。

proxy 失配（比 mesh 大一圈 / 桥接桌腿间隙 / 套错拓扑）在 **S2 object-only mesh/collision overlay review** 时先行拦截（见 §5.3），残余的 runtime phantom collision 通过 canary + Full CEM 视觉核验暴露和量化；若发现系统性失配，作为 `DATA_NEGATIVE`/`CEM_NEGATIVE` 记录并上报用户，**不在本实验内改几何参数**（改 proxy 是独立的后续实验）。

## 📊 4. Claims

| Claim | 最低证据 |
|---|---|
| C0：raw authority 完整 | fresh inventory 覆盖 7 物体 120 seq/240 pc；差异有 raw path/hash 解释 |
| C1：action scope 正确 | 37 move2 seq 标 production；32 move1 标 EXCLUDE_ACTION_NOT_MOVE2；51 其它标 Stage0；均留 registry |
| C2：两档 raw contact 可复现 | 240 rows（或 production 子集）均有 3cm/5cm 状态、proxy path、metrics、terminal route；5cm 不覆盖 3cm |
| C3：非 box template 合规且可信 | 每物体 `person1/2` 走 dcv3 非 box 流程：proxy adapter 正确（bucket_wall / desk_surface_voxel）、`template_status` 经 `nonbox_template_review.tsv` `approve_clean`→`clean_reviewed`（禁自动 clean）、含 object-only mesh/collision overlay evidence、MuJoCo/inertial/mass-inertia/contact-site + SHA；bucket009_p1、desk005_p1 fresh proxy-build 有构建+review 证据；无 `_e0**`/`_s2_` 变体作 base |
| C4：route/variant 语义未污染 | target route 始终 `ref_fk`；v1/v2 provenance 分离 |
| C5：v1→v2 fallback 完整 | 所有 S3 eligible row 有 v1 终态；所有 fresh v1 infeasible row 有 v2 终态；非该状态 v2 attempts=0 |
| C6：S3–S5 无静默缺失 | 每 selected v1/v2 row 在 target gate/visual QC/handoff 均有 pass 或明确 terminal failure |
| C7：Full CEM 覆盖精确 | 每 `S5_READY_SET` row 有完整 Full CEM artifact 或明确 terminal runtime failure；expected/completed/failed 闭合 |
| C8：质量证据完整 | 每 CEM-complete row 有统一 metrics、gate-health、MP4、关键帧、Codex verification |
| C9：分层结论可解释 | 按 object、person、date、sequence、contact label、selected variant 报告 funnel/yield/failure taxonomy/指标分布 |
| C10：凹几何 proxy 证据可解释 | 明确量化 bucket（`bucket_wall_proxy_aabb`）/desk（`desk_surface_voxel_multibox_proxy_draft`）物体 proxy 下的接触/穿透/phantom collision 行为，对照 E173 尺寸先验区分「尺寸效应」与「凹几何 proxy 惩罚」；rubber_hull（手侧）与物体 proxy 分开报告 |
| C11：历史比较不越界 | pre-PRG bucket/desk、`_e0**`/`_s2_` 变体只作 warning，明确 scene/method 差异，不声称因果 A/B |
| C12：用户 authority 独立 | Codex 只写核验列；用户完成全部 CEM-complete row 的最终 `USE/DO_NOT_USE` |
| C13：结果可复现 | config/git/SHA、scene snapshot、registry、commands、NPZ、metrics、video、audit 均在 `results/E174/` |

## 🔄 5. S0–S6 工作流

7 物体共用同一 S0–S6 链路，按 §1 优先级顺序推进（S0–S5 可穿插并行构建，S6 GPU 队列按优先级排序）。单物体流程：

```
Fresh raw (120 seq/240 pc) → S0 env/authority preflight
  → S1 inventory + 3cm/5cm contact → {move2? → 否: move1→EXCLUDE / 其它→Stage0 reject / 是: raw route?}
  → {否: case reject / 是: S2 非 box proxy template build + review}
  → {template clean_reviewed? → 否: object block / 是: S3 omnirt_v1}
  → {v1 pass: S4 / v1 infeasible: S3 omnirt_v2 → {v2 pass: S4 / v2 fail: reject} / v1 other fail: reject}
  → S4 target+visual gate → S5 rubber+PRG handoff
  → S6 canary → {runtime healthy? → 否: scoped block / 是: Full CEM} → verify → user review
  → 所有终态进 completion audit → 发布 E174 log
```

### 5.1 S0：环境、配置与 authority preflight

保存 repo path/git SHA/dirty diff、raw root/SMPL-X path、Python/MuJoCo/ffmpeg/GPU contract、resolved config/run ID、240-row expected identity 与 fresh inventory identity diff、code/config/schema SHA。任何 raw root 误指、case set 漂移、公共 config 不一致或结果目录覆盖风险，运行 S1 前作 global blocker。

**环境（复用 E171–E173 verified，见 memory `e171-data-paths-and-env`）：**
- CORE4D raw root = `/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real`（**不用** dead mount）
- SMPLX_MODEL_DIR = `.../mocap_data/human_model_files`（converter 内部追加 `smplx/`）
- S3 retarget env = conda `hsretargeting`
- S6 CEM 跑**本机 8×L20Y**（`MUJOCO_GL=osmesa`）；如遇 `.cache/run.py` GPU-filler 占卡，按已授权先释放，**不 kill 其它任务**

### 5.2 S1：fresh inventory 与 3cm/5cm raw contact

对 `object_keys=bucket004,bucket009,bucket010,bucket007,bucket003,desk005,desk007` 运行完整 inventory，不用 `max-case-persons` 截断。所有 row 含 object/date/sequence/person/action/source path/hash。**action scope 过滤在 inventory 后应用**：move2→production；move1→`EXCLUDE_ACTION_NOT_MOVE2`（留 registry，不进 S2–S6）；join/leave/pass/rot/strike/raise→dcv3 Stage0 `reject_action_not_first_line`。S1 同次 surface-distance 输出 3cm/5cm mask、候选、summary、timeline、per-sequence NPZ。

### 5.3 S2：非 box source template proxy build + 人工 review（doc 04，含 bucket009_p1 / desk005_p1 补建）

**关键：非 box template 不能像 box 那样自动 clean**。所有 7 物体的 `person1/2` template 都必须走 dcv3 非 box 流程，`template_status` 默认 `manual_review_required`，仅在显式 review 通过后置 `clean_reviewed` 方可进 S3。

- **proxy adapter 与 policy（冻结）**：
  - bucket（004/009/010/007/003）→ `nonbox_proxy_aabb_review`，`collision_policy=bucket_wall_proxy_aabb`（底面+四侧壁）。
  - desk（005/007）→ `nonbox_surface_voxel_review`，`collision_policy=desk_surface_voxel_multibox_proxy_draft`（表面 voxelize `target_cells=26`+inward shrink→多 box），**不用标准桌语义模板**。
- **已有 template 复核**：现有 clean base（bucket004/010/007/003 双人、bucket009_p2、desk005_p2、desk007 双人）已带正确 proxy policy（已核实），但仍须**重新走 object-only overlay review** 确认其 `template_status`；若历史未标 `clean_reviewed` 或缺 review evidence，本轮补做 review。忽略所有 `_freejoint_legobj_e0**` / `_s2_` 污染变体。
- **缺失需补建**：`bucket009_person1`（proxy_aabb）、`desk005_person1`（surface voxel proxy）目录不存在 → 用对应 proxy adapter 生成 review-only proxy template（含 task_info），保持 `proxy_template=True`、`manual_review_required=True`，构建证据/notes/source_ref 全保存。
- **review evidence（硬要求）**：每个 template 生成 review package，至少含 ① source template orbit sheet/mp4（`render_template_review_package.py`）；② **object-only mesh/collision overlay sheet**（`stages/s2_templates/render_template_mesh_collision_review_package.py --object-only`），确认 proxy 没有明显大一圈、没有桥接桌腿间隙、没有套错拓扑。Codex/subagent review 后在 `nonbox_template_review.tsv` 写 `review_decision=approve_clean` → 脚本置 `template_status=clean_reviewed`。**禁止仅凭 MuJoCo load / render pass 自动 release**。
- **硬失败**（doc 04）：MuJoCo load fail、`nq/nv/nu` 不符、hand contact sites 缺失、robot inertial 与 clean base 不符（警惕历史 `29.632` 污染）、object collision extents 明显偏离 mesh AABB、残留旧 runtime artifact → object blocker。
- 通过 review 的 clean_reviewed template 文件 + git HEAD + SHA256 存到 `results/E174/scene_snapshot/source_templates/` 并 `git add -f`（bucket009_p2/bucket003_p1p2/bucket010_p2/desk007_p1p2 等 untracked base 同理）。

### 5.4 S3：v1 primary 与 v2 rescue 真执行

只对 approved raw-contact + move2 rows 执行 Stage2b（预期上界 74 move2 pc，实际以 raw-contact pass 为准）。流程、manifest 归一化（CVXPY infeasible → `omniretarget_infeasible`）、rescue manifest 构建、v2 adapter canary、独立目录/SHA 记录**与 E173 §5.4 完全一致**。任何物体历史 infeasible prior 只作 warning，须由 fresh v1 evidence 重新确认。

### 5.5 S4：target gate 与 visual QC

机器 gate 消费 selected v1/v2 产物，检查 target/source scene、qpos layout、object pose patch、inertial、penetration、lower-body interference、replay contract。Codex 审查所有 target-gate-pass row keyframe sheet + 全部 warning/boundary row 的 replay MP4，其余按 object×person×date×contact-label 分层抽查。**非 box 重点**：bucket 内壁/desk 桌腿处的手接触是否合理、桌腿与下肢是否穿插、desk surface-voxel proxy 是否桥接桌腿间隙、bucket_wall proxy 是否比 mesh 大一圈导致漂浮。发现穿透/趴物/错误接触人/瞬移/爆姿时扩查完整视频。只有 visual QC pass 进 S5；Codex 结论存独立 verification 字段，不写 `manual_*`。

### 5.6 S5：rubber-hull 与 PRG sidecar handoff

从 S4-pass target `scene_act.xml` 生成 rubber-hull sidecar，再加 16 个 lower-body/object collision pair。sidecar 用 E174 专属名 `scene_act_E174_rubberHull_PRG.xml`，禁止覆盖原始 `scene_act.xml`。semantic diff 只能含预期 hand mesh collision + 16 pair；object mass/friction/actuator/world/source trajectory 不变。S5 保存 frozen `S5_READY_SET`、每行 `selected_retarget_variant_id`、v1→v2 provenance、CEM override、effective scene SHA、hand-collision audit、candidate/reject manifest。进 canary 前所有 S5-ready target 的 `scene.xml`、原始 `scene_act.xml` 与 E174 sidecar XML 必须 `git add -f` 并复制到实验 snapshot 记录 SHA（双重保障）。

### 5.7 S6：conditional canary 与 Full CEM（本机 8×L20Y，按物体优先级排队）

Canary 从 S5-ready 确定性选择：每个存在 ready row 的 `object×person` 至少 1 条；若存在 v2-rescued S5-ready row 至少覆盖 1 条 v2；**bucket 与 desk 各至少 1 条**（确保两种拓扑都被 runtime 验证）。Canary 只验证 runtime/scene/config/artifact contract（`samples=64, opt_steps=4`）；数值差/0 positive 不阻断，只有公共 runtime contract 不健康才按分级方案暂停。Canary 健康后对 frozen `S5_READY_SET` 全部跑 production `samples=1024, opt_steps=32` Full CEM。

**GPU 调度**：8 卡按 `assigned_gpu` sharding；队列按物体优先级 `bucket004→bucket009→bucket010→bucket007→bucket003→desk005→desk007` 填充，同物体内按 case_id 排序。CEM 是 GPU-bound，**1 case/GPU，不做 48-wide packing**（见 memory E173 execution lessons）；用 8-wide serial local runner。

## 🛡️ 6. 分级阻断与 stop-loss

### 6.1 分级阻断

| 级别 | 典型条件 | 阻断范围 | 解除条件 |
|---|---|---|---|
| Global blocker | raw authority 漂移、公共 SHA 不一致、路径覆盖、环境契约失败、公共 runtime 失败 | 暂停 E174 全批 | 修复公共 contract 并重跑 S0/canary |
| Object blocker | 某物体 template audit/build、sidecar semantic diff、object-level scene load 失败 | 只暂停该物体 | 该对象 contract 修复并重审；其余继续 |
| Case blocker | action/raw contact fail、v1 非 rescue 类失败、v1/v2 dual-infeasible、target gate fail、visual reject、单 case artifact fail | 只淘汰该 person-case | 保留 terminal evidence；其余继续 |
| Warning | pre-PRG bucket/desk 历史、`_e0**`/`_s2_` 变体、template 缺失、5cm-only、历史 infeasible prior、**非 box proxy 失配（大一圈/桥接间隙）** | 不自动阻断 | report 保留 warning/provenance |

「任一 preflight 失败即阻断全批」不适用于 E174。全批只由 global contract 失败阻断；object/case 失败必须隔离——**尤其 bucket009_p1/desk005_p1 若不可 clean build，只阻断该物体**。

### 6.2 Canary stop-loss

与 E173 一致：同 signature 双 canary fail → object blocker；单 case v1 infeasible → v2 rescue（非 blocker）；异常初始姿态/输入损坏/非 solver-infeasible v1 fail → case blocker（不试 v2）；OOM/资源争用 → 降并发/重分 shard，不改算法配置、不 kill 其它任务；canary 质量差/数值 fail/视觉不理想 → 不作执行 stop-loss，继续 full 测真实 yield。**非 box proxy phantom collision 属"质量差"，同样不 stop-loss，交由 full + 视觉核验量化**。三次失败协议照旧，每次失败输入/signature/尝试/结果写 `progress.md` 与 E174 evidence。

### 6.3 Full completeness contract

```text
full_expected = full_completed + full_terminal_failed   （逐物体 + 全批各闭合一次）
```

每 completed row 须 root/outdir NPZ 一致、qpos finite、effective config、scene SHA、PRG diagnostics、metrics、video；每 terminal failed row 须 error log、failure mode、最后尝试记录。目录无文件且 registry 无终态属未完成。

## 🔧 7. 实现与固化入口

### 7.1 计划阶段修改文件

| 文件 | 本次改动 |
|---|---|
| `plan/190_E174_bucket_desk_move2_full_pipeline_plan.md` | 新建 E174 正式计划 |
| `EXPERIMENT_TRACKER.md` | 新增 E174 Phase 37 计划入口 |
| `progress.md` | 记录 authority、非 box 边界、模板差异、move2-only 决策与计划验收 |

### 7.2 执行前拟新增文件（优先 fork E173 脚本，仅换 object scope / action filter / 路径）

| 文件 | 作用 |
|---|---|
| `scripts/experiments/E174/e174_common.py` | 公共 contract（fork E173；`OBJECT_KEYS=(bucket004,bucket009,bucket010,bucket007,bucket003,desk005,desk007)`、per-object expected、优先级顺序、`ACTION_SCOPE="move2_only"`、RAW_INVENTORY_PRIOR、SCENE_NAME=`scene_act_E174_rubberHull_PRG`、method ID=`E174_E170PRG_nonbox_candidate_r1`） |
| `scripts/experiments/E174/build_pipeline_authority.py` | 从 S1–S5 生成 frozen authority、per-object funnel、terminal state audit（含 move2-only action scope 断言 + 非 box `template_status=clean_reviewed` 断言） |
| `scripts/experiments/E174/build_nonbox_template_review.py` | 调用 dcv3 `stages/s2_templates` 非 box proxy adapter（bucket=`nonbox_proxy_aabb_review`、desk=`nonbox_surface_voxel_review`）+ `render_template_mesh_collision_review_package.py --object-only`，生成 `nonbox_template_review.tsv` review queue（bucket009_p1/desk005_p1 补建） |
| `scripts/experiments/E174/build_omnirt_rescue_manifest.py` | 只从 fresh v1 `omniretarget_infeasible` 构建 v2 canary/rescue queue |
| `scripts/experiments/E174/run_stage2b_queue.py` | 执行 v1 primary 与 v2 rescue |
| `scripts/experiments/E174/build_prg_cem_manifest.py` | 生成 rubber-hull/PRG sidecar、override、canary/full manifest（dcv3 override 复制到 Hydra 搜索路径，避免 E172 遇到的假 `missing_dcv3_override`） |
| `scripts/experiments/E174/run_cem_queue.py` | 按 manifest 执行 canary/full，物体优先级排队，1 case/GPU |
| `scripts/experiments/E174/render_cem_results.py` | Full CEM MP4、关键帧、review package |
| `scripts/experiments/E174/audit_completion.py` | 审计 S0–S6、逐物体 + 全批 Full completeness、Codex/user authority |
| `scripts/launch/active/run_E174_data_pipeline.sh` | fresh S0–S5 固化入口（7 物体） |
| `scripts/launch/active/run_E174_local_8gpu_cem.sh` | 本机 8×L20Y canary/full 入口（fork E173 local runner，8-wide serial，assigned_gpu sharding + 物体优先级；释放已授权 GPU-filler） |
| `scripts/launch/active/watch_E174_cem.sh` | 监控、增量回收、terminal completion |
| `scripts/launch/active/postprocess_E174_after_full.sh` | strict completion 后 render/eval/report/audit |
| `scripts/eval/runners/eval_E174_nonbox.py` | 直接 import `eval.core.core_metrics` 的统一 evaluator |
| `scripts/eval/wrappers/eval_E174_nonbox.sh` | 固化评测入口 |
| `scripts/eval/reports/gen_E174_nonbox_report.py` | 精炼 Markdown 主报告 + 详细 TSV/JSON（per-object 分段 + bucket/desk 分组） |

若现有 dcv3 / E173 入口已提供某能力，E174 wrapper 只冻结参数和路径，不复制管线实现。实验特有逻辑放 `scripts/experiments/E174/`，真实 launch 放 `scripts/launch/active/`。

### 7.3 固化执行入口（执行前先实现并 code review）

```bash
bash workspace/core4d/scripts/launch/active/run_E174_data_pipeline.sh

MODE=canary bash workspace/core4d/scripts/launch/active/run_E174_local_8gpu_cem.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E174_local_8gpu_cem.sh

bash workspace/core4d/scripts/launch/active/watch_E174_cem.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E174_nonbox.sh
```

GPU ID 只表示计划默认 shard；启动前只读检查实际占用并保存快照。GPU 不可用时等待或调整 E174 shard，不暂停/kill/重排其它任务。

## 📈 8. Full CEM、评测与人工审查

### 8.1 Artifact 硬契约

与 E173 §8.1 一致：每 Full CEM row 验证 root/outdir NPZ qpos shape/数值/SHA 一致；qpos + reward/gate diagnostics finite；effective config 与 frozen E174 config 精确一致；scene name/path/SHA 与 manifest 一致；trajectory/contact mask/source/target scene/override 有 SHA；MP4 可解码、非零帧、帧数可解释；case_id/person/contact label/selected variant/object 无错配；v1 failure/v2 eligibility/`rescue_of`/最终 selected variant 精确闭合。

### 8.2 指标核验

Evaluator 直接 import `eval.core.core_metrics`，不动态加载其它实验 evaluator。阈值 launch 前冻结。维度同 E173（Tracking / Contact / Hand safety / Body safety / Dynamics / Gate health / Completeness）。

报告至少按以下维度分组：**object（7 个）**；**bucket vs desk**；`person1` vs `person2`；date；sequence；3cm primary vs 5cm recall；`omnirt_v1` vs `omnirt_v2`；S1/S3/S4/S5 failure taxonomy；**按体积（bucket 0.045→0.19 递增 vs desk 0.24）**；以及与 box021(E170)/box026(E171)/box004(E172)/box023·box024·box001(E173) 的跨物体对照。**新增「凹几何失配」定性维度**：统计 bucket 内壁/desk 桌腿处的 phantom contact、穿透、漂浮案例数。历史 overlap 只报同 case 方向，不做严格 paired causal claim。

### 8.3 视觉抽查与用户终审

Full CEM 后 Codex：查看所有 case keyframe sheet；完整查看所有 numeric fail、gate fallback、阈值边界、worst-case MP4；从 numeric pass 分层抽查；记录穿透/非法支撑/趴物/fall/object kick/爆姿/接触丢失/reference infeasible/**物体 proxy phantom collision**；独立填 `codex_verification.tsv`（无 `manual_*`）。生成供用户审核的精炼 Markdown 主报告，回答：

1. 240 条 raw row 最终分别停在哪一层（含 32 move1 EXCLUDE + 51 其它 Stage0，按物体拆分）
2. 每物体有多少 S5-ready、Full complete、numeric pass
3. 哪些 case 可用/不可用及原因
4. v1 infeasible 多少、v2 救回多少、dual-infeasible 哪些（按物体）
5. **非 box 物体 proxy 是否引入超出尺寸先验的失败**：对照 E173 尺寸-通过率曲线，bucket/desk 实际通过率 vs 尺寸先验预测的差值 = 凹几何 proxy 惩罚量化
6. bucket（`bucket_wall_proxy_aabb`）vs desk（`desk_surface_voxel_multibox_proxy_draft`）失败模式差异；两种物体 proxy 对两种拓扑的具体影响（rubber_hull 手侧影响单独报告）

用户对所有 CEM-complete rows 给出 fresh `USE/DO_NOT_USE`。拍板前 machine recommendation 固定 `PENDING_USER_REVIEW`，不生成 RL-ready/partner export。

## ✅ 9. 成功标准与结果分级

### 9.1 Pipeline completion 成功标准

| 检查项 | 通过标准 |
|---|---|
| Raw coverage | 预期 240 rows 全部有 inventory 与 3cm/5cm 终态，或 authority drift 已解释并经用户确认 |
| Action scope | 37 move2 全进 production 候选；32 move1 全标 EXCLUDE；51 其它全 Stage0；无静默漏项 |
| Template coverage | 每物体 2/2 source template 走非 box proxy 流程 + object-only overlay review → `clean_reviewed`（`nonbox_template_review.tsv` 有 `approve_clean`）+ SHA snapshot（bucket009_p1、desk005_p1 含 fresh proxy-build + review 证据） |
| Eligible Stage2b | 100% 有 v1 终态；每 v1 infeasible row 100% 有 v2 终态；非 eligible v2 attempts=0 |
| S4/S5 coverage | 所有 selected v1/v2 pass row 均有 target/visual/handoff 终态 |
| Full CEM coverage | 逐物体 + 全批 `full_expected = completed + terminal_failed`，missing=0 |
| Metrics/visual | 所有 CEM-complete row 有 metrics、video、关键帧、Codex verification |
| User authority | 所有 CEM-complete row 有用户最终标签后才形成 final 结论 |
| Reproducibility | release checks、completion audit、SHA/scene/config contract 全 pass |

### 9.2 Scientific yield 与执行成功分离

E174 不设「至少 N 条 positive」作为执行完成门槛。逐物体 `0 positive` 甚至 `0 S5-ready` 都可能是正确筛选结果（尤其非 box 几何可能系统性失败）。yield 分级**逐物体**判定：

| 结果类型 | 定义 | 含义 |
|---|---|---|
| `PIPELINE_INCOMPLETE` | authority/terminal state/Full/证据缺失 | 执行失败，不能下科学结论 |
| `DATA_NEGATIVE` | 某物体 0 S5-ready 且 S0–S5 完整 | 该物体数据/target gate 未发现可下游 row |
| `CEM_NEGATIVE` | 有 S5-ready 但 0 user USE | downstream method 对该物体无产出 |
| `PARTIAL_YIELD` | ≥1 条 user USE，未覆盖全分层 | 保留 case-level positives，不升级默认 |
| `OBJECT_YIELD` | 某物体出现用户 USE 且 trade-off 可解释 | 支持继续评估该物体/拓扑跨物体 PRG |

无论哪类都必须同时报告 funnel denominator（每物体 raw/move2/S5/CEM 总数），不能只报 positive。

### 9.3 配置晋级边界

E174 只决定是否产生可保留的 case-level CEM 结果。即使达到 `OBJECT_YIELD`，也不能自动宣称 PRG/rubber_hull 为全物体/全拓扑默认。**非 box 证据（bucket/desk）若成立，是把 PRG 适用范围从 box 扩展到凹几何的第一份数据，但仍不构成默认化依据**；若失败，则明确界定 PRG/rubber_hull 的适用边界止于凸 box。

## 💾 10. 结果路径与复现

### 10.1 标准结果树

```text
workspace/core4d/results/E174/
├── s0_environment/
├── s1_raw_contact/{inventory/,raw_contact/}
├── s2_templates/
├── scene_snapshot/{source_templates/,cem_sidecars/}
├── s3_retarget/{omnirt_v1/ref_fk/,omnirt_v2/ref_fk/,rescue/}
├── s4_gate_visual_qc/{omnirt_v1/ref_fk/,omnirt_v2/ref_fk/}
├── s5_handoff/{hand_collision/,cem_overrides/}
├── s6_downstream/{cem/canary/,cem/full/,eval/,evidence/}
├── registries/          # per-object + 全批 funnel/authority
└── completion_audit/
```

`results/E174/` 不进 git。代码、计划、log、tracker，以及 S5-ready 的 active `example_datasets/.../*.xml`（含 bucket009_p1/desk005_p1 补建的 clean template + 各 untracked base 的 git add -f）进入 git；`scene_snapshot/` 与大体积 NPZ/MP4 由外部结果同步管理，manifest 摘要与 SHA 写入最终 log。

### 10.2 启动前验证

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | xargs -0 python3 -m py_compile
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root workspace/core4d/results/E174_release_check --no-smoke
git diff --check
```

### 10.3 最终 completion audit

至少验证：7 物体 raw/S1/S2/S3/S4/S5/S6 row-set 逐物体闭合；move2-only action scope 断言通过（move1/其它未误入 S2–S6）；3cm/5cm promotion 无 duplicate/overwrite；每 S3 eligible row 有 v1 终态、v2 rescue set 与 fresh v1 infeasible set 精确相等；v1/v2 registry/output 并存、所有 v2 row 有 `rescue_of`、无 v2 覆盖 v1 或非 eligible v2 attempt；每物体 2 个 template（含补建）与全部 CEM sidecar SHA 可恢复；S5-ready 与 Full expected set 精确相等；逐物体 + 全批 completed + terminal failed = expected，missing=0；metrics/MP4/keyframes/Codex/user review 对 CEM-complete set 覆盖完整；variant/route/collision/method id 语义未混用；S6 failure 未反向改写 S1–S5；未经用户批准无 RL-ready/partner export。

只有 completion audit 通过且用户标签齐全后，E174 才从「执行完成待终审」更新为最终结果，并新建分析型实验日志（log/234）；tracker 只保留一句话摘要与 log 链接。

## 🔗 11. 与已跑实验的关系

E174 是 E170(box021)/E171(box022+box026)/E172(box004)/E173(box024+box023+box001) 的**跨物体扩展第 5 批**，也是**首次离开 box 拓扑**：算法冻结不变，物体换成 5 bucket（空心，0.045–0.19 m³）+ 2 desk（桌腿，0.24 m³），动作收紧到 move2-only。与 E173 的差异：物体侧改用 dcv3 非 box proxy collision policy（`bucket_wall_proxy_aabb` / `desk_surface_voxel_multibox_proxy_draft`）+ 强制 object-only overlay review → `clean_reviewed`（不可自动 clean）、move2-only action scope、bucket009_p1/desk005_p1 proxy 补建、报告新增 bucket/desk/凹几何 proxy 分组。E174 独立推进，不继承 pre-PRG bucket/desk 历史 yield，也不启动 RL 导出——是否进入 RL/partner export 由用户在 case-level yield 与终审后另行决定。E174 的结果将决定 chair 类（更极端凹几何）是否值得作 E175。

---

## 待用户确认后执行

本文件为 **planning-only**。确认后按 §7.3 顺序实现脚本 → code review → 跑 S0–S5 → canary（bucket+desk 各覆盖）→ Full CEM（本机 8 卡，bucket→desk）→ 评测 → 视觉核验 → 写 log/234 → 更新 tracker。
