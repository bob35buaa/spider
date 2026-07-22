# E173 实验计划：Box024 / Box023 / Box001 全流程筛选及 Full CEM（move-only）

_Core4D Phase 36 · 2026-07-22 · production screening plan（planning-only，待用户确认后执行）_

计划前身：E170 [plan/186](186_E170_box021_prg_full_validation_plan.md) · E171 [plan/187](187_E171_box022_box026_full_pipeline_plan.md) · E172 [plan/188](188_E172_box004_full_pipeline_plan.md)

---

## 📋 1. 决策摘要

E173 对三个 box 物体 `box024`、`box023`、`box001` 从 CORE4D raw authority 重新执行 data_construction_v3 的 `S0–S6` 主链路（**move-only**），并对本轮真正通过 S5 handoff gate 的全部 person-case 运行 Full CEM。算法与冻结配置**完全沿用 E170/E171/E172**（`omnirt_v1→v2 rescue` + `ref_fk` + `rubber_hull` + `E170 PRG` 跨物体候选），不做 reward/route sweep，也不是旧结果搬运。

固定决策如下：

- **物体范围与优先级**：三物体，GPU 调度与执行顺序按用户指定优先级 **① box024 → ② box023 → ③ box001**。box024 数据小、最先完；box023 次之；box001 数据最大放最后。任一物体 DATA_NEGATIVE 不阻断其余物体。
- **动作策略（move-only，与 E168/E172 一致）**：只有 `move1/move2` 进入 production 全流程；`join/leave/rot/pass1/pass2/strike/raise` 在 S1 即作为 Stage0 非首选动作 reject（dcv3 `HIGH_RISK_ACTION_PREFIXES` 自动归类为 `reject_action_not_first_line`），**保留终态在 registry** 但不进入 S2–S6，不为其设计 gate。
- raw 入口覆盖三物体**完整 97 sequence / 194 person-case** 以做 authority 闭合；production expected = **88 move pc**；最终 authority 以 E173 fresh inventory 为准。
- S1 同时重算 3cm/5cm raw contact；3cm 是 production 主入口，5cm-only row 进入 recall review，不自动放行。
- **不导出 RL / partner**：E173 只产 case-level Full CEM 结果与用户标签，本轮明确不做 RL-ready/partner export（与 E172 一致）。
- S2 对每个物体的 `boxNNN_person1/2` source template 做 fresh build/audit 和 SHA snapshot。**box023 已有 git-tracked clean person1/2 模板**；**box024/box001 仅 person1 存在且 untracked、无 task_info、person2 缺失** → 这两个物体的 S2 必须**先补建 person2 及缺失 task_info**（dcv3 clean template 自动构建流程），而非仅审计。
- S3 与 E172 一致：所有 eligible row 先跑 `omnirt_v1/ref_fk`；只有 fresh `stage2b_status=omniretarget_infeasible` 才进入 `omnirt_v2/ref_fk` rescue。
- S4 运行 target gate、kinematic replay 和 visual QC；Codex 负责指标核验与视觉抽查。
- S5 固定 `hand_collision_variant_id=rubber_hull`，使用 E173-scoped sidecar scene，不覆盖 source `scene_act.xml`。
- S6 采用 E170 PRG 冻结配置作为跨物体候选配置，先做条件式 canary，再对全部 S5-ready rows 做 Full CEM，全部跑**本机 8×L20Y**。
- 用户负责最终人工标签和是否进入后续 RL/partner export 的拍板；E173 本轮不启动 RL，也不自动导出 partner。

> ⚠️ **解释边界：** “对三物体做全流程”表示所有 194 raw rows 都必须得到可追溯的终态，不表示绕过数据 gate 强制产生 CEM row。若 fresh S1–S5 后某物体/某分层为 0 ready，这是有效的数据层负结果（`DATA_NEGATIVE`）。

## 🔍 2. 历史证据与边界

### 2.1 Raw inventory prior（fresh 盘点，2026-07-22）

从 live raw root `.../CORE4D/CORE4D_Real/human_object_motions` 按 `object_metadata.json`（key=`obj_name`，大小写混用 `box024`/`Box024` 已合并）精确召回；action 来自 `action_labels.json`（key=`date/seqid`）。三物体 mesh（`box024_m.obj`/`box023_m.obj`/`box001_m.obj`）在 raw + processed assets 均存在。这些数字用于 S0 authority 漂移检查，不代替 E173 fresh inventory。

| 物体 | raw seq | raw pc | move seq | **move pc（production）** | 非首选 seq | 非首选 pc | 非首选动作构成 | 日期 |
|---|---|---|---|---|---|---|---|---|
| **box024** | 23 | 46 | 5 | **10** | 18 | 36 | join×9 / leave×8 / pass2×1 | 20231011, 20231108 |
| **box023** | 23 | 46 | 16 | **32** | 7 | 14 | pass2×3 / pass1×2 / rot×2 | 20231008, 20231011, 20231020 |
| **box001** | 51 | 102 | 23 | **46** | 28 | 56 | join×9 / leave×9 / pass1×4 / pass2×4 / rot×2 | 20231003_1/2, 20231011, 20231020, 20231023, 20231108 |
| **合计** | **97** | **194** | **44** | **88** | **53** | **106** | — | — |

关键观察：
- **box024（P1）move 占比极低**：23 seq 里只有 5 条是 move，其余 18 条是 `join`/`leave`（协作交接）——production 只有 10 pc。这与 box004（14 pc move）同量级，是本批最小的 production 集。
- **box023（P2）** move 占比高（16/23），production 32 pc，是三者中 move 数据最“干净充足”的。
- **box001（P3）** raw 最大（102 pc）但过半是 join/leave/pass/rot，production 46 pc。

### 2.2 物体尺寸（mesh AABB，用于跨物体对照，非 gate）

| 物体 | 长×宽×高 (m) | 体积 m³ | 对角线 m | 量级 |
|---|---|---|---|---|
| box023 | 0.391 × 0.304 × 0.303 | 0.036 | 0.58 | 小（近立方） |
| box024 | 0.982 × 0.509 × 0.505 | 0.253 | 1.22 | 大（细长高） |
| box001 | 0.813 × 0.626 × 0.502 | 0.256 | 1.14 | 大 |
| _参考 box004(已跑)_ | 0.447×0.348×0.264 | 0.041 | 0.63 | 小 |
| _参考 box026(已跑)_ | 0.629×0.469×0.394 | 0.116 | 0.88 | 中 |

E173 首次把 PRG 跨物体证据扩展到**大箱（box024/box001，~0.25 m³，对角线 >1.1m）**。已跑三物体（box021 0.059 / box026 0.116 / box004 0.041）都 ≤0.12 m³，均为中小箱。大箱对搬运姿态（起始更可能低位抱举/贴身）、手-物 SDF、下肢-物碰撞的影响是 E173 的核心科学看点；box023 作小箱对照锚点。

### 2.3 历史 overlap：仅作 warning prior，不作 completion 证据

| 物体 | 历史来源 | 可复用的 contract/evidence | E173 不得继承的结论 |
|---|---|---|---|
| box023 | E055–E072、E079、E163、E162（Phase 11–18/27） | 存在大量 pre-PRG 旧 scene（`box023_person*_freejoint_legobj_e0**`）与失败诊断 | **pre-PRG、pre-rubber_hull、无 move-only funnel**；box023 旧算法多次 FAIL（E059 1.5/6、E064 2/5）不能作 E173 基线或 completion 证据 |
| box025（近邻大箱） | E041/E080 | “sphere 过拟合” reward 诊断、大物体边界复查 | 仅定性参考大箱难点；E173 不跑 box025 |
| box024 | 无 | 无任何历史实验（干净） | — |
| box001 | 无 PRG-era 实验 | 仅 processed template 半成品 | — |

已知 template 风险（warning prior）：box024/box001 的 `person1` 模板 untracked 且无 `task_info.json`，`person2` 目录缺失；box023 有历史污染变体目录（`_e0**`），**不得**把任何 `_e0**` 派生目录当 base，只用 clean `box023_person1/2`。所有 template 必须由 E173 fresh build/audit 并保留证据与 SHA。

### 2.4 E170 PRG 的复用边界

与 E172 一致：E170(box021 18/28, gate-health 0/28)、E171(box026 5/12, 0/12)、E172(box004 5/6, 0/6) 均显示 PRG 对 lower-body 有定向收益但伴随 contact trade-off、gate-health 恒 0。E173 将 PRG 定义为**跨物体候选配置**，非已验证 default。E173 不做 P/R/G 消融，不根据 canary 质量临时调参；所有 S5-ready rows 用同一冻结配置，实验结束后按物体/分层报告可迁移性。

## 🎯 3. Scope、authority 与固定配置

### 3.1 In scope

- 三物体全量 raw inventory（97 seq/194 pc）与 action/inventory 筛选
- 3cm/5cm raw-contact 重算、summary、timeline 和 reject taxonomy
- 每物体 `person1/2` clean source template 的 fresh build/audit、visual package 与 SHA snapshot（box024/box001 含 person2 补建）
- `omnirt_v1/ref_fk` primary Stage2b、限定 `omnirt_v2/ref_fk` rescue 及其 terminal failures
- target gate、kinematic replay、visual QC 和 S5 handoff
- rubber-hull + E170 PRG cross-object candidate 的 canary 与 Full CEM（本机 8×L20Y）
- 全量 metrics、分组分析（含大箱 vs 小箱、跨已跑物体对照）、视觉抽查、用户最终人工标签准备
- S0–S6 registry、completion audit、结果日志和 tracker 更新

### 3.2 Out of scope

- `fingertip_aware`、`adaptive`、`omnirt_original`、`omnirt_v2_replace` 或任何 wrist-to-fingertip replacement
- raw-contact threshold、reward、gate、CEM budget 或 scene physics sweep
- 为增加 positive 数量而改变 action policy、target route，或对非 `omniretarget_infeasible` row 使用 v2
- RL 训练、RL-ready export、partner export 和默认配置升级
- 把任何历史（box023 pre-PRG 结果、E091/E167 等）计入 E173 fresh completion count
- box025 或其它未列物体（本批只跑 box024/box023/box001）

### 3.3 Authority 定义

| Authority | 唯一来源 | 必须断言 |
|---|---|---|
| Raw authority | E173 `s1_raw_contact/inventory/inventory.tsv` | object ∈ {box024,box023,box001}；case_id 唯一；预期 97 seq/194 pc |
| S1 authority | E173 fresh 3cm/5cm manifests | 每个 raw row 在两档均有状态，不能静默缺失 |
| Stage2b authority | E173 v1 primary + v2 rescue manifests | 每个 eligible row 有 v1 终态；每个 v1 infeasible row 有 v2 终态和唯一 selected production variant |
| S5 authority | E173 registry + handoff manifest | 只含通过 template/Stage2b/target/visual gate 的 row |
| Full CEM authority | `S5_READY_SET` 的精确快照 | `full_expected = count(S5_READY_SET)`，不预先硬编码 |
| 人工 authority | E173 user review TSV | Codex 不写或覆盖 `manual_*` 字段 |

`S5_READY_SET` 定义（与 E172 一致）：

```text
inventory/action eligible (move1/move2)
AND selected raw-contact route approved
AND template audit pass
AND selected Stage2b variant pass
AND target gate pass
AND visual QC pass
AND rubber-hull/PRG sidecar contract pass
```

### 3.4 Raw-contact promotion policy

与 E172 字段级一致（单 label、禁 3cm/5cm 双份 target 覆盖）：

1. `3cm pass` → 直接进入 production Stage2b 候选
2. `3cm review` → 经 raw-contact evidence review 决定 pass/reject
3. `3cm fail + 5cm pass/review` → `5cm_recall_review`，仅明确双手/目标人物接触证据才晋级
4. `5cm fail` → 终态 `REJECT_RAW_CONTACT`
5. 已晋级 3cm row 不再生成 5cm duplicate

被晋级的 5cm-only row 必须在 manifest 记录 `stage2b_contact_label=5cm`、reviewer、review notes 和 evidence path。

### 3.5 固定 retarget fallback 与 CEM 配置（与 E171/E172 字段级一致）

| 轴 | E173 冻结值 | 说明 |
|---|---|---|
| Retarget primary | `omnirt_v1` | 所有 S3 eligible row 必须先运行 |
| Retarget rescue | `omnirt_v2` | 仅 v1 `omniretarget_infeasible` 时运行；不是 A/B sweep |
| Target route | `ref_fk` | E098 公共 contract |
| Base reward/method | `E167A_zOnlyBody` | 与 E170/E171/E172 PRG 基座一致；per-object base override 见下 |
| Hand collision | `rubber_hull` | mesh-aware SDF，maxhullvert=64；E173 sidecar |
| Lower-body physics | 16 geoms ↔ object pair | 沿用 E169/E170 P 配置 |
| Lower-body penalty | scale `2.0`、margin `0.02m` | 沿用 E169/E170 R 配置 |
| Candidate gate | min SDF `0.005m`、max violation `0.02`、hard floor `-0.005m` | fallback=`least_violation` |
| CEM budget | seed `0`、samples `1024`、opt steps `32` | canary smoke `samples=64, opt_steps=4` |
| Source config ID | `E170_PRG_lowerbodyPhysics_softPenalty_candidateGate` | 冻结来源 |
| E173 method ID | `E173_E170PRG_crossObject_candidate_r1` | 不写入 `target_variant_id` |

**Per-object base reward override**：E170/E171/E172 的 base reward override 是 case-scoped（如 `core4d_E167_box004_082_p1_E167A`）。E173 每物体需以该物体一个 canonical move case 生成对应的 `core4d_E167_{object}_{seq}_{person}_E167A` base override（reward 权重字段与 E167A_zOnlyBody 逐字一致，仅换 case-specific scene/trajectory 路径）。base override 的选取 case 记录在 registry，reward 字段级 diff 必须为空。

`omnirt_v2` 使用 E168/E171/E172 同款 Phase4 rescue contract：

```text
enable_constraint_relaxation = true
enable_foot_z_constraint = true
foot_slide_penalty_weight = 1.0
enable_contact_preservation = true
object_penetration_tolerance_scale = 0.8
replace_wrist_with_fingertip = false
include_fingertip_centers = false
```

`omnirt_v1` 保持 primary contract（上述 Phase4 flags 全关，`replace_wrist_with_fingertip=false`、`include_fingertip_centers=false`）。v1/v2 选择规则、独立目录/registry、`rescue_of`/SHA 记录、Full CEM effective config 字段级一致等约束**与 E172 §3.5 完全相同**，此处不复述。

## 📊 4. Claims

| Claim | 最低证据 |
|---|---|
| C0：raw authority 完整 | fresh inventory 覆盖三物体 97 seq/194 pc；任何差异有 raw path/hash 解释 |
| C1：两档 raw contact 可复现 | 194 rows 均有 3cm/5cm 状态、proxy path、metrics 和 terminal route；5cm 不覆盖 3cm |
| C2：template 基础可信 | 每物体 `person1/2` 均完成 MuJoCo/inertial/mass-inertia/collision/contact-site/visual 和 SHA audit；box024/box001 person2 fresh build 有构建证据 |
| C3：route/variant 语义未污染 | target route 始终 `ref_fk`；v1/v2 provenance 分离；box023 历史 `_e0**` 变体不作 base |
| C4：v1→v2 fallback 完整 | 所有 S3 eligible row 有 v1 终态；所有 fresh v1 infeasible row 有 v2 终态；非该状态 v2 attempts=0 |
| C5：S3–S5 无静默缺失 | 每个 selected v1/v2 row 在 target gate、visual QC 和 handoff 均有 pass 或明确 terminal failure |
| C6：Full CEM 覆盖精确 | 每个 `S5_READY_SET` row 均有完整 Full CEM artifact 或明确 terminal runtime failure；expected/completed/failed 闭合 |
| C7：质量证据完整 | 每个 CEM-complete row 均有统一 metrics、gate-health、MP4、关键帧和 Codex verification |
| C8：分层结论可解释 | 按 object、person、action、date、sequence、contact label、selected variant 报告 funnel/yield/failure taxonomy/指标分布 |
| C9：历史比较不越界 | box023 pre-PRG、box025 邻近大箱只作 warning/定性参考，明确 scene/method 差异，不声称因果 A/B |
| C10：用户 authority 独立 | Codex 只写核验列；用户完成全部 CEM-complete row 的最终 `USE/DO_NOT_USE` 拍板 |
| C11：结果可复现 | config/git/SHA、scene snapshot、registry、commands、NPZ、metrics、video 和 audit 均在 `results/E173/` |
| C12：PRG 跨物体（含大箱）可解释 | 明确报告三物体 PRG 的 lower-body 收益与 contact/gate-health trade-off，**首次含大箱（box024/box001 ~0.25m³）**，与 box021/box026/box004 及小箱 box023 对照 |

## 🔄 5. S0–S6 工作流

三物体共用同一 S0–S6 链路，按 §1 优先级 box024 → box023 → box001 顺序推进（S0–S5 可穿插并行构建，但 S6 GPU 队列按优先级排序）。单物体流程与 E172 图一致：

```
Fresh raw (97 seq/194 pc) → S0 env/authority preflight
  → S1 inventory + 3cm/5cm contact → {move? → 否: Stage0 reject / 是: raw route?}
  → {否: case reject / 是: S2 template build+audit}
  → {template clean? → 否: object block / 是: S3 omnirt_v1}
  → {v1 pass: S4 / v1 infeasible: S3 omnirt_v2 → {v2 pass: S4 / v2 fail: case reject} / v1 other fail: case reject}
  → S4 target+visual gate → S5 rubber+PRG handoff
  → S6 canary → {runtime healthy? → 否: scoped block / 是: Full CEM} → verify → user review
  → 所有终态进 completion audit → 发布 E173 log
```

### 5.1 S0：环境、配置与 authority preflight

保存 repo path/git SHA/dirty diff、raw root/SMPL-X path、Python/MuJoCo/ffmpeg/GPU contract、resolved config/run ID、194-row expected identity 与 fresh inventory identity diff、code/config/schema SHA。任何 raw root 误指、case set 漂移、公共 config 不一致或结果目录覆盖风险，在运行 S1 前作为 global blocker。

**环境（复用 E171/E172 verified，见 memory `e171-data-paths-and-env`）：**
- CORE4D raw root = `/mnt/ali-sh-1/usr/xiayibo/xyb_data_tidal_alsh/other-datasets/mocap_data/CORE4D/CORE4D_Real`（**不用** dead mount `/mnt/a0ccc676-...`）
- SMPLX_MODEL_DIR = `.../mocap_data/human_model_files`（converter 内部追加 `smplx/`）
- S3 retarget env = conda `hsretargeting`
- S6 CEM 跑**本机 8×L20Y**（`MUJOCO_GL=osmesa`）；如遇 `.cache/run.py` GPU-filler 占卡，按 E171 已授权先释放，**不 kill 其它任务**

### 5.2 S1：fresh inventory 与 3cm/5cm raw contact

对 `object_keys=box024,box023,box001` 运行完整 inventory，不用 `max-case-persons` 截断。所有 row 含 object/date/sequence/person/action/source path/hash。Stage0 非首选动作（106 pc）reject 是合法结果但必须留 registry。S1 同次 surface-distance 计算输出 3cm/5cm mask、候选、summary、timeline 和 per-sequence NPZ。

### 5.3 S2：source template fresh build/audit（含 box024/box001 person2 补建）

- **box023**：`box023_person1/2` 已 git-tracked clean，做 fresh audit + SHA snapshot；忽略所有 `_e0**` 历史污染变体。
- **box024 / box001**：`person1` untracked、无 `task_info.json`，`person2` 缺失 → 用 dcv3 clean template 自动构建流程**补建 person2 及缺失的 task_info**，person1 也重新规范化并纳入 audit。构建证据、notes、source_ref 全保存。
- 启动时把实际使用的 template 文件、git HEAD、SHA256 存到 `results/E173/scene_snapshot/source_templates/`，构建/规范化后的 clean template `git add -f` 纳入主 git。检查 MuJoCo load、robot inertial、object mass/inertia、collision policy、contact-site。

### 5.4 S3：v1 primary 与 v2 rescue 真执行

只对 approved raw-contact + move rows 执行 Stage2b（预期上界 88 move pc，实际以 raw-contact pass 为准）。流程、manifest 归一化（CVXPY infeasible → `omniretarget_infeasible`）、rescue manifest 构建、v2 adapter canary、独立目录/SHA 记录**与 E172 §5.4 完全一致**。任何物体的历史 infeasible prior 只作 warning，须由 fresh v1 evidence 重新确认。

### 5.5 S4：target gate 与 visual QC

机器 gate 消费 selected v1/v2 产物，检查 target/source scene、qpos layout、object pose patch、inertial、penetration、lower-body interference、replay contract。Codex 审查所有 target-gate-pass row keyframe sheet + 全部 warning/boundary row 的 replay MP4，其余按 object×person×action×contact-label 分层抽查；发现穿箱/趴箱/错误接触人/物体瞬移/爆姿时扩查完整视频。**大箱（box024/box001）重点看起始抱举姿态与 box 底部离地**。只有 visual QC pass 进入 S5；Codex 结论存独立 verification 字段，不写 `manual_*`。

### 5.6 S5：rubber-hull 与 PRG sidecar handoff

从 S4-pass target `scene_act.xml` 生成 rubber-hull sidecar，再加 16 个 lower-body/object collision pair。sidecar 用 E173 专属名 `scene_act_E173_rubberHull_PRG.xml`，禁止覆盖原始 `scene_act.xml`。semantic diff 只能含预期 hand mesh collision + 16 pair；object mass/friction/actuator/world/source trajectory 不变。S5 保存 frozen `S5_READY_SET`、每行 `selected_retarget_variant_id`、v1→v2 provenance、CEM override、effective scene SHA、hand-collision audit 和 candidate/reject manifest。进 canary 前所有 S5-ready target 的 `scene.xml`、原始 `scene_act.xml` 和 E173 sidecar XML 必须 `git add -f` 并复制到实验 snapshot 记录 SHA（双重保障）。

### 5.7 S6：conditional canary 与 Full CEM（本机 8×L20Y，按物体优先级排队）

Canary 从 S5-ready 确定性选择：每个存在 ready row 的 `object×person` 至少 1 条；若存在 v2-rescued S5-ready row 至少覆盖 1 条 v2。Canary 只验证 runtime/scene/config/artifact contract（`samples=64, opt_steps=4`）；数值差/0 positive 不阻断，只有公共 runtime contract 不健康才按分级方案暂停。Canary 健康后对 frozen `S5_READY_SET` 全部跑 production `samples=1024, opt_steps=32` Full CEM。

**GPU 调度**：8 卡按 `assigned_gpu` sharding；队列按物体优先级 **box024 → box023 → box001** 填充，同物体内按 case_id 排序。box024 先跑先出结果，便于早期发现大箱系统性问题再决定 box001 是否值得全量。

## 🛡️ 6. 分级阻断与 stop-loss

### 6.1 分级阻断

| 级别 | 典型条件 | 阻断范围 | 解除条件 |
|---|---|---|---|
| Global blocker | raw authority 漂移、公共 SHA 不一致、路径覆盖、环境契约失败、公共 runtime 失败 | 暂停 E173 全批 | 修复公共 contract 并重跑 S0/canary |
| Object blocker | 某物体 template audit/build、sidecar semantic diff、object-level scene load 失败 | 只暂停该物体 | 该对象 contract 修复并重审；其余物体继续 |
| Case blocker | action/raw contact fail、v1 非 rescue 类失败、v1/v2 dual-infeasible、target gate fail、visual reject、单 case artifact fail | 只淘汰该 person-case | 保留 terminal evidence；其余继续 |
| Warning | box023 pre-PRG 历史、box024/box001 template 半成品、box025 大箱难点、5cm-only、历史 infeasible prior | 不自动阻断 | report 保留 warning/provenance |

“任一 preflight 失败即阻断全批”不适用于 E173。全批只由 global contract 失败阻断；object/case 失败必须隔离——**尤其 box024/box001 template 若不可 clean build，只阻断该物体，不牵连 box023**。

### 6.2 Canary stop-loss

与 E172 一致：同 signature 双 canary fail → object blocker；单 case v1 infeasible → v2 rescue（非 blocker）；异常初始姿态/输入损坏/非 solver-infeasible v1 fail → case blocker（不试 v2）；OOM/资源争用 → 降并发/重分 shard，不改算法配置、不 kill 其它任务；canary 质量差/数值 fail/视觉不理想 → 不作执行 stop-loss，继续 full 测真实 yield。三次失败协议照旧，每次失败输入/signature/尝试/结果写 `progress.md` 与 E173 evidence。

### 6.3 Full completeness contract

```text
full_expected = full_completed + full_terminal_failed   （逐物体 + 全批各闭合一次）
```

每 completed row 须 root/outdir NPZ 一致、qpos finite、effective config、scene SHA、PRG diagnostics、metrics 和 video；每 terminal failed row 须 error log、failure mode 和最后尝试记录。目录无文件且 registry 无终态属未完成。

## 🔧 7. 实现与固化入口

### 7.1 计划阶段修改文件

| 文件 | 本次改动 |
|---|---|
| `plan/189_E173_box024_box023_box001_full_pipeline_plan.md` | 新建 E173 正式计划 |
| `EXPERIMENT_TRACKER.md` | 新增 E173 Phase 36 计划入口 |
| `progress.md` | 记录 authority、历史边界、模板差异和计划验收 |

### 7.2 执行前拟新增文件（优先 fork E172 脚本，仅换 object scope/路径）

| 文件 | 作用 |
|---|---|
| `scripts/experiments/E173/e173_common.py` | 路径、schema、case ID、SHA 与状态公共 contract（fork E172；`OBJECT_KEYS=("box024","box023","box001")`，per-object expected 计数、优先级顺序、RAW_INVENTORY_PRIOR、SCENE_NAME=`scene_act_E173_rubberHull_PRG`、method ID=`E173_E170PRG_crossObject_candidate_r1`） |
| `scripts/experiments/E173/build_pipeline_authority.py` | 从 S1–S5 生成 frozen authority、per-object funnel 和 terminal state audit |
| `scripts/experiments/E173/build_omnirt_rescue_manifest.py` | 只从 fresh v1 `omniretarget_infeasible` 构建 v2 canary/rescue queue |
| `scripts/experiments/E173/run_stage2b_queue.py` | 执行 v1 primary 与 v2 rescue |
| `scripts/experiments/E173/build_prg_cem_manifest.py` | 生成 rubber-hull/PRG sidecar、override、canary/full manifest（dcv3 override 复制到 Hydra 搜索路径，避免 E172 遇到的假 `missing_dcv3_override`） |
| `scripts/experiments/E173/run_cem_queue.py` | 按 manifest 执行 canary/full，物体优先级排队 |
| `scripts/experiments/E173/render_cem_results.py` | Full CEM MP4、关键帧和 review package |
| `scripts/experiments/E173/audit_completion.py` | 审计 S0–S6、逐物体 + 全批 Full completeness、Codex/user authority |
| `scripts/launch/active/run_E173_data_pipeline.sh` | fresh S0–S5 固化入口（三物体） |
| `scripts/launch/active/run_E173_local_8gpu_cem.sh` | 本机 8×L20Y canary/full 入口（fork E172 local runner，按 assigned_gpu sharding + 物体优先级；释放已授权 GPU-filler） |
| `scripts/launch/active/watch_E173_cem.sh` | 监控、增量回收和 terminal completion |
| `scripts/launch/active/postprocess_E173_after_full.sh` | strict completion 后 render/eval/report/audit |
| `scripts/eval/runners/eval_E173_boxes.py` | 直接 import `eval.core.core_metrics` 的统一 evaluator |
| `scripts/eval/wrappers/eval_E173_boxes.sh` | 固化评测入口 |
| `scripts/eval/reports/gen_E173_boxes_report.py` | 精炼 Markdown 主报告 + 详细 TSV/JSON（per-object 分段，避免 E172 遇到的单物体硬编码 KeyError） |

若现有 dcv3 / E172 入口已提供某能力，E173 wrapper 只冻结参数和路径，不复制管线实现。实验特有逻辑放 `scripts/experiments/E173/`，真实 launch 放 `scripts/launch/active/`。

### 7.3 固化执行入口（执行前先实现并 code review）

```bash
bash workspace/core4d/scripts/launch/active/run_E173_data_pipeline.sh

MODE=canary bash workspace/core4d/scripts/launch/active/run_E173_local_8gpu_cem.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E173_local_8gpu_cem.sh

bash workspace/core4d/scripts/launch/active/watch_E173_cem.sh
bash workspace/core4d/scripts/eval/wrappers/eval_E173_boxes.sh
```

GPU ID 只表示计划默认 shard；启动前只读检查实际占用并保存快照。GPU 不可用时等待或调整 E173 shard，不暂停/kill/重排其它任务。

## 📈 8. Full CEM、评测与人工审查

### 8.1 Artifact 硬契约

与 E172 §8.1 一致：每 Full CEM row 验证 root/outdir NPZ qpos shape/数值/SHA 一致；qpos + reward/gate diagnostics finite；effective config 与 frozen E173 config 精确一致；scene name/path/SHA 与 manifest 一致；trajectory/contact mask/source/target scene/override 有 SHA；MP4 可解码、非零帧、帧数可解释；case_id/person/contact label/selected variant/object 无错配；v1 failure/v2 eligibility/`rescue_of`/最终 selected variant 精确闭合。

### 8.2 指标核验

Evaluator 直接 import `eval.core.core_metrics`，不动态加载其它实验 evaluator。阈值 launch 前冻结。维度同 E172（Tracking / Contact / Hand safety / Body safety / Dynamics / Gate health / Completeness）。

报告至少按以下维度分组：**object（box024/box023/box001）**；`person1` vs `person2`；action；date；sequence；3cm primary vs 5cm recall；`omnirt_v1` vs `omnirt_v2`；S1/S3/S4/S5 failure taxonomy；**大箱（box024/box001）vs 小箱（box023）**；以及与 box021(E170)/box026(E171)/box004(E172) 的 PRG 跨物体对照。历史 overlap 只报同 case 方向，不做严格 paired causal claim。

### 8.3 视觉抽查与用户终审

Full CEM 后 Codex：查看所有 case keyframe sheet；完整查看所有 numeric fail、gate fallback、阈值边界和 worst-case MP4；从 numeric pass 分层抽查；记录穿透/非法支撑/趴箱/fall/object kick/爆姿/接触丢失/reference infeasible；独立填 `codex_verification.tsv`（无 `manual_*`）。生成供用户审核的精炼 Markdown 主报告，回答：

1. 194 条 raw row 最终分别停在哪一层（含 106 条非首选动作，按物体拆分）
2. 每物体有多少 S5-ready、Full complete、numeric pass
3. 哪些 case 可用/不可用及原因
4. v1 infeasible 多少、v2 救回多少、dual-infeasible 是哪些（按物体）
5. **PRG 的 lower-body 收益是否在大箱（box024/box001）成立、contact/gate-health trade-off 是否比中小箱更重**，与 box021/box026/box004/box023 对照
6. box023 fresh 结果与其 pre-PRG 历史失败的方向差异（仅定性）

用户对所有 CEM-complete rows 给出 fresh `USE/DO_NOT_USE`。拍板前 machine recommendation 固定 `PENDING_USER_REVIEW`，不生成 RL-ready/partner export。

## ✅ 9. 成功标准与结果分级

### 9.1 Pipeline completion 成功标准

| 检查项 | 通过标准 |
|---|---|
| Raw coverage | 预期 194 rows 全部有 inventory 与 3cm/5cm 终态，或 authority drift 已解释并经用户确认 |
| Template coverage | 每物体 2/2 source template fresh build/audit + visual + SHA snapshot 完成（box024/box001 含 person2 补建证据） |
| Eligible Stage2b | 100% 有 v1 终态；每 v1 infeasible row 100% 有 v2 终态；非 eligible v2 attempts=0 |
| S4/S5 coverage | 所有 selected v1/v2 pass row 均有 target/visual/handoff 终态 |
| Full CEM coverage | 逐物体 + 全批 `full_expected = completed + terminal_failed`，missing=0 |
| Metrics/visual | 所有 CEM-complete row 有 metrics、video、关键帧和 Codex verification |
| User authority | 所有 CEM-complete row 有用户最终标签后才形成 final 结论 |
| Reproducibility | release checks、completion audit、SHA/scene/config contract 全 pass |

### 9.2 Scientific yield 与执行成功分离

E173 不设“至少 N 条 positive”作为执行完成门槛。逐物体的 `0 positive` 甚至 `0 S5-ready` 都可能是正确筛选结果。yield 分级**逐物体**判定：

| 结果类型 | 定义 | 含义 |
|---|---|---|
| `PIPELINE_INCOMPLETE` | authority/terminal state/Full/证据有缺失 | 执行失败，不能下科学结论 |
| `DATA_NEGATIVE` | 某物体 0 S5-ready，且其 S0–S5 完整 | 该物体数据/target gate 未发现可下游 row |
| `CEM_NEGATIVE` | 有 S5-ready 但 0 user USE | downstream method 对该物体无产出 |
| `PARTIAL_YIELD` | ≥1 条 user USE，未覆盖全分层 | 保留 case-level positives，不升级默认 |
| `OBJECT_YIELD` | 某物体出现用户 USE 且 trade-off 可解释 | 支持继续评估该物体/尺寸档跨物体 PRG |

无论哪类都必须同时报告 funnel denominator（每物体 raw/move/S5/CEM 总数），不能只报 positive。

### 9.3 配置晋级边界

E173 只决定是否产生可保留的 case-level CEM 结果。即使达到 `OBJECT_YIELD`，也不能自动宣称 PRG 为全物体/全尺寸默认。默认升级需另一个含跨对象对照、gate-health 修复和 RL evidence 的实验。**大箱证据（box024/box001）若成立，是把 PRG 适用尺寸区间从中小箱扩展的第一份数据，但仍不构成默认化依据。**

## 💾 10. 结果路径与复现

### 10.1 标准结果树

```text
workspace/core4d/results/E173/
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

`results/E173/` 不进 git。代码、计划、log、tracker，以及 S5-ready 的 active `example_datasets/.../*.xml`（含 box024/box001 补建的 clean template）进入 git；`scene_snapshot/` 与大体积 NPZ/MP4 由外部结果同步管理，manifest 摘要和 SHA 写入最终 log。

### 10.2 启动前验证

```bash
find workspace/core4d/scripts/data_construction_v3 -name '*.py' -print0 | xargs -0 python3 -m py_compile
workspace/core4d/scripts/data_construction_v3/orchestration/run_release_checks.sh \
  --run-root workspace/core4d/results/E173_release_check --no-smoke
git diff --check
```

### 10.3 最终 completion audit

至少验证：三物体 raw/S1/S2/S3/S4/S5/S6 row-set 逐物体闭合；3cm/5cm promotion 无 duplicate/overwrite；每 S3 eligible row 有 v1 终态、v2 rescue set 与 fresh v1 infeasible set 精确相等；v1/v2 registry/output 并存、所有 v2 row 有 `rescue_of`、无 v2 覆盖 v1 或非 eligible v2 attempt；每物体 2 个 template（含补建）与全部 CEM sidecar SHA 可恢复；S5-ready 与 Full expected set 精确相等；逐物体 + 全批 completed + terminal failed = expected，missing=0；metrics/MP4/keyframes/Codex/user review 对 CEM-complete set 覆盖完整；variant/route/collision/method id 语义未混用；S6 failure 未反向改写 S1–S5；未经用户批准无 RL-ready/partner export。

只有 completion audit 通过且用户标签齐全后，E173 才从“执行完成待终审”更新为最终结果，并新建分析型实验日志；tracker 只保留一句话摘要和 log 链接。

## 🔗 11. 与已跑实验的关系

E173 是 E170(box021)/E171(box022+box026)/E172(box004) 的**跨物体扩展第 4 批**，算法冻结不变，首次纳入**大箱（box024/box001，~0.25 m³）**并以小箱 box023 作对照锚点。与 E172 的差异仅在：三物体、优先级排队、box024/box001 需 S2 补建 person2 模板、报告新增大箱 vs 中小箱分组。E173 独立推进，不继承 box023 pre-PRG 历史 yield，也不启动任何 RL 导出——是否进入 RL/partner export 由用户在 case-level yield 与终审后另行决定。

---

## 待用户确认后执行

本文件为 **planning-only**。确认后按 §7.3 顺序实现脚本 → code review → 跑 S0–S5 → canary → Full CEM（本机 8 卡，box024→box023→box001）→ 评测 → 视觉核验 → 写 log/233 → 更新 tracker。
