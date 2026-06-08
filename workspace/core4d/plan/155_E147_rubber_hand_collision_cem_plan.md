# E147 — 手部碰撞体变体（rubber 凸包）纳入 v3 管线 + 重跑 CEM A/B

## 背景

E146 可视化已证（log 186 / `project_hand_collision_geom`）：SPIDER 重定向当前手部碰撞体是 **5cm 球**（`hand_collision` class，scene `lh`/`rh`），rubber_hand mesh 仅 visual。静态回放下 rubber 接触不输球、干净大箱穿透 64%→4%，但小箱/坏轨迹引入深穿透——因为那条轨迹是**为球优化的**。下游 RL 侧已独立观察到 rubber mesh 提升 RL 成功率。

**本实验目的**：把手部碰撞体作为一个**正式的数据管线维度**纳入 data_construction_v3（新字段 + handoff 注入 + docs + skill），并在该维度下用 rubber 凸包碰撞体**重跑 SPIDER CEM**，A/B 对照球版，验证净收益（接触↑ + 穿透↓ 同时成立）。

**核心设计判断（回答"是不是要多一个字段"）**：**是，要新增一个变体轴。**
- v3 已有两个变体轴：`retarget_variant_id`（OmniRetarget/输入改写）、`target_variant_id`（SPIDER target route）。registry 主键是 `(case_id, retarget_variant_id, target_variant_id)`。
- 手部碰撞体是**第三个正交轴**：它不改 retarget、不改 target route，只改 CEM 物理 scene 里机器人手的碰撞几何。任何 retarget×target 组合都能配任意手碰撞体。
- 注意区别于已有的 `collision_policy`（那是 **物体侧** 碰撞代理，如 `bucket_wall_proxy_aabb`，TemplateBuilder 产出）。手部是机器人侧，必须单独命名。
- **新字段名：`hand_collision_variant_id`**，取值 `sphere5cm`（默认，现状）/ `rubber_hull`（本实验）/ future。

## 已验证的技术可行性（前置风险已排除，实测）

| 风险 | 验证结果 |
|---|---|
| MJWP/Warp 能否用 mesh 碰撞 | ✅ mujoco_warp 支持凸多面体(nmeshpoly)+CONVEX 碰撞；`put_model` 接受含 mesh 碰撞的模型 |
| rubber STL 直接做碰撞体 | ❌ 原始 STL `geom_rbound=0`，broadphase 跳过 |
| **解法：凸包** | ✅ asset 加 `maxhullvert="64"` → `geom_rbound=0.0998`，mesh-box 接触实测 `ncon=1`。**单凸包即可，无需离线 VHACD** |
| 替换先例 | ✅ `workspace/core4d/scripts/convert/patch_hand_3box.py`（sphere→3box，scene 自包含、改 default+geom+contact pairs） |
| CEM handoff 注入点 | ✅ `stages/s5_handoff/export_cem_overrides.py:write_override` 生成 override YAML（写 contact_hdmi_target_source 等），scene 由此 adapter 决定 |

## A. 数据管线接入（data_construction_v3）—— 本实验的主体

把手部碰撞体做成一等公民，不散在 E147 一次性脚本里。

### A1. 新字段 + schema
- `lib/interfaces.py`：在 CEM-relevant result（或 registry 写入路径）加 `hand_collision_variant_id: str = "sphere5cm"`。
- `lib/common.py`：注册取值常量 `HAND_COLLISION_VARIANTS = {"sphere5cm", "rubber_hull"}` + 每个 variant 的 scene patch 规格（geom 类型/mesh/pos/maxhullvert）。
- registry 主键：S6/CEM 维度从 `(case_id, retarget_variant_id, target_variant_id)` 扩成 **`(case_id, retarget_variant_id, target_variant_id, hand_collision_variant_id)`**。默认 `sphere5cm` 保持向后兼容（旧 row 视为 sphere5cm）。
- `state/update_case_state_registry.py`：key 元组加第 4 维；缺省填 `sphere5cm`。

### A2. HandCollisionAdapter（新扩展接口）
- 在 `lib/interfaces.py` 加 `HandCollisionAdapterResult`（继承 ExtensionResult + `hand_collision_variant_id`、`base_scene_act`、`patched_scene_act`、`patch_params_json`）。
- 新脚本 `stages/s5_handoff/patch_hand_collision.py`（照 `patch_hand_3box.py` 结构）：
  - 输入：handoff row 的 `scene_act`（球版基线）+ `hand_collision_variant_id`。
  - `sphere5cm`：原样透传（no-op）。
  - `rubber_hull`：asset 加 `maxhullvert="64"`；删 `hand_collision` sphere default；`lh`/`rh` 改成 `type="mesh" mesh="left/right_rubber_hand"`（pos 用源 rubber visual 偏移）；contact pairs（`lh/rh × floor/object`）名不变、必要时调 condim/solref/friction。
  - 输出 patched scene 到 v3 产物区 + sha256/manifest。

### A3. CEM override 注入
- `stages/s5_handoff/export_cem_overrides.py:write_override`：根据 `hand_collision_variant_id` 选 base scene → 调 A2 patch → override YAML 的 scene 指向 patched scene（或写 `scene_name`）。
- override / handoff / downstream manifest 全部带上 `hand_collision_variant_id` 列。

### A4. 产物与状态
- `data_construction_v3/`：registry/manifest 增列 `hand_collision_variant_id`；`existing_cases.tsv` schema_version bump，旧行补 `sphere5cm`。
- migration（`migration/build_existing_cases_seed.py`）：旧 row 默认 `sphere5cm`。

## B. 实验执行（10 case，跨物体 × 成功/失败）

数据来源：`data_construction_v3/existing_cases.tsv`（权威 CEM 历史）+ E145 nonbox 结果（desk/bucket 未注册进表）。**现实约束**：跑过 full CEM 的物体只有 box(004/021/023/025/026)+bucket+desk；chair/board/stick 全 not_run。所以"跨物体×成功+失败"只能在 box+bucket 间取（desk 仅 fail，无 pass 基线）。

5 对 pass/fail × 5 类物体，npz 全部已确认存在：

| # | case_id | 物体 | 原 CEM | 来源 |
|---|---|---|---|---|
| 1 | e091_box004_20231003_2_083_p1 | box004 | pass | E096b |
| 2 | e091_box004_20231003_2_082_p1 | box004 | pass | E096b |
| 3 | d003_box021_20231011_035_p1 | box021 | pass | E107 |
| 4 | d003_box021_20231011_035_p2 | box021 | fail | E107 |
| 5 | box023_person2 | box023 | pass | E081 |
| 6 | box023_person1 | box023 | fail | E079 |
| 7 | e091_box026_20231020_134_p1 | box026 | pass | E106 |
| 8 | e091_box026_20231020_134_p2 | box026 | fail | E106 |
| 9 | bucket004_20231003_1_012_p1 | bucket004 | pass(RL pass) | E108 |
| 10 | bucket004_20231002_021_p1 | bucket004 | fail | E108 |

平衡：5 pass + 5 fail；5 类物体；4 组同物体 A/B（21/23/26/bucket，其中 box026_134 p1/p2 同序列同动作不同人，控制变量最佳）。

### 执行步骤
0. **逐 case 解析 scene/override**（前置）：10 case 来自 6 个实验（E079/E081/E096b/E106/E107/E108），scene_act/override 分散。脚本 `scripts/E147/resolve_cases.py` 从 existing_cases.tsv 的 `cem_result_npz`/`evidence_root` 反查每 case 的 processed task dir、scene_act（优先各实验 `scene_snapshot/`）、原 override、mask、task/data_id/person_idx → `E147/cases_manifest.tsv`，**人工核对齐全**。
1. 经 A2/A3 为每 case 生成 `rubber_hull` 的 patched scene + CEM override（`sphere5cm` 不重跑，复用旧 npz）。
2. **单 case 烟测**（box021_035_p1）：rubber 版能进 CEM、不崩、出轨迹；查日志 `hand_object_deep_penalty`/`cem_gate_min_sdf`/`sample_gate_violation` 正常收敛（mesh 过穿应触发 deep_penalty 把手推开）；确认 `scene_name=` 旁路生效。
3. **全 10 case 只跑 rubber_hull**（决策1：球版复用旧 npz）。命令照 `run_box025_3box_regression.sh:104`（`run_mjwp.py +override= scene_name=<rubber scene> ... output_dir=`），分卡并行。A/B = 旧球版 npz vs 新 rubber npz。
4. **双指标评测**（`unified_replay_eval.py`，照 experiment.md §5）：
   - 接触：`hand_geom_near_{3,5,8,10}cm_frac`（越高越好）
   - 穿透：`hand_geom_penetration_frac`、`hand_geom_deep_penetration_2cm_frac`
   - 稳定/任务：pelvis fall、object 跟踪误差、leg/body 穿透、object_floor(掉箱)
   - 报 mean+std+worst，禁 cherry-pick；A/B 同 clip 并排视频。
   - 成功判据（事前定义）：rubber 相比 sphere **接触 5cm 不降(±2pp)且 深穿透 2cm 显著下降**，pelvis 不摔、不掉箱；并看 rubber 能否把原 fail 救成 pass；分物体看大箱/小箱/桶差异。

## C. 文档与 SKILL 更新（强制，和管线改动同 PR）

- `docs/data_construction_v3/03_manifest_schema.md`：registry 主键加 `hand_collision_variant_id`；字段表加该列（取值 `sphere5cm`/`rubber_hull`/future）。
- **新建 `docs/data_construction_v3/15_hand_collision_variants.md`**（独立 doc，不并入 08）：定义手部碰撞体轴，与 retarget/target 正交；说明 `sphere5cm`（默认 5cm 球）vs `rubber_hull`（rubber 凸包 maxhullvert=64，含 maxhullvert 概念解释）；区别于物体侧 `collision_policy`；patch 机制 + contact pair 注意事项；旁路注入约定（不覆盖源 scene）。
- `docs/data_construction_v3/02_pipeline_stages.md`：S5 handoff 增 HandCollisionAdapter 步骤。
- `docs/data_construction_v3/09_extension_interfaces.md`：加 HandCollisionAdapter 接口契约。
- `.codex/skills/data-construction-v3-zh/SKILL.md`（+ agents/）：把手部碰撞体维度、新字段、patch 脚本、A/B 跑法写进 skill，使后续 agent 知道这个轴。

## D. 记录
- log `workspace/core4d/log/<N>_E147_*.md`：双指标 A/B 表（10 case × sphere/rubber）、按物体分组结论、视频。
- 更新 memory `project_hand_collision_geom`：从"几何潜力好"升级到"物理重优化净收益数据"。

## 风险与对策
- **凸包太粗**（手指凹陷被填）：若 box023 小箱仍过穿，记录，留凸分解为后续，不本轮处理。
- **contact 参数失配**（mesh 接触点多→抖）：Step2 烟测发现就调，改动隔离进 patched scene + variant 规格。
- **registry key 扩维兼容**：默认 `sphere5cm`，旧 row/旧脚本不受影响；migration 补缺省。
- **scene 路径分散**：Step0 必须先人工核对清单。
- **老实验缺 override**（E079/E081）：从 source_ref 的 eval_summary 重建最小 override。

## 范围约束
- 只改手部 `hand_collision` 几何 + 必要 contact 参数 + 管线字段/adapter/docs/skill；**不改 SPIDER reward/算法**。
- 所有 patched scene + 字段改动 git 追踪 + per-exp snapshot（experiment.md §7）。
- 不做凸分解（单凸包已验证够用）。
- 不碰 object actuator / partner / 物体侧 `collision_policy`。
- chair/board/stick 不纳入（无 CEM 历史）。

## 已定决策（用户拍板，2026-06-08）
1. **球版直接复用旧 npz**（existing_cases.tsv 里 `cem_result_npz` 已存在）——不重跑 sphere5cm，只跑 rubber_hull 一遍；A/B = 旧球版 npz vs 新 rubber npz。注意：旧 npz 可能是不同 CEM 代码版本，评测时如发现球版指标异常需复核，但默认信任旧结果。
2. **scene 注入用旁路，不覆盖源**：patched rubber scene 写到 v3 产物区，CEM 用 `scene_name=` 指向它；**绝不覆盖 processed dir 的 scene_act.xml**。需确认 config.py:874 `scene_name` 对 humanoid_object 生效（执行时验）。
3. **`hand_collision_variant_id` 进 registry 主键**：S6/CEM 维度 key = `(case_id, retarget_variant_id, target_variant_id, hand_collision_variant_id)`，同 case 可并存 sphere5cm/rubber_hull 两条 CEM 证据。A1 落实 `update_case_state_registry.py` 的 key 元组扩维 + 旧 row 默认 `sphere5cm`。
4. **新建独立 doc `docs/data_construction_v3/15_hand_collision_variants.md`**（不并入 08）。
5. **maxhullvert = 凸包顶点上限**，固定 **64**（已有几何数据依据，不扫参）。MuJoCo 把 rubber mesh（完整凸壳 3191 顶点）简化成 ≤N 顶点凸壳做碰撞（算法只吃凸体）。几何扫参实测（最远点采样模拟简化，简化壳 vs 完整壳的方向支撑缩水）：

   | maxhullvert | 缩水 mean | p90 | max | 碰撞耗时 |
   |---|---|---|---|---|
   | 8 | 5.9mm | 13.5mm | 18.9mm | 4.5µs |
   | 32 | 1.3mm | 3.2mm | 9.0mm | 5.2µs |
   | **64** | **0.5mm** | **1.3mm** | **2.8mm** | 5.5µs |
   | 128 | 0.3mm | 0.8mm | 1.6mm | 5.6µs |
   | 256 | 0.2mm | 0.5mm | 1.5mm | 7.5µs |

   结论：64 缩水亚毫米（远小于 cm 级接触带），正在收益递减拐点后；16~128 碰撞速度几乎无差异（无"用小值省时间"动机）。**不需扫参，更不需 full CEM 扫**——maxhullvert 是已饱和的精度旋钮，对 CEM 速度无影响。真正的几何不确定性在"凸包 vs 真实凹形手"（搬箱是手掌外凸面贴箱，凸包本就贴合，影响小），留到 CEM 结果若异常再查凸分解。

## 待执行时确认（非阻塞）
- config.py:874 `scene_name` 旁路对 humanoid_object 实际生效（Step2 烟测验）。
- 老实验（E079/E081）若缺 per-case override，从 source_ref 的 eval_summary 重建最小 override。
