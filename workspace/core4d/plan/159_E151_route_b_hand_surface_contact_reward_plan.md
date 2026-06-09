# E151 — 路线B：让接触奖励"看见真实几何"（reward 轴，B2→B1→B3）

## 0. 当前现状：橡胶手是"哑几何"

E150（log `190`）证伪路线A 后，核实了橡胶手 `lh`/`rh` 在三层的约束状态：

| 层 | `lh`/`rh` 状态 | 证据 |
|---|---|---|
| 物理碰撞 | ❌ 关 | rubber 场景仅 `floor`+`object_collision` 可碰；`lh`/`rh` 是 `contype/conaffinity=0/0`。手插箱物理引擎不阻挡 |
| 读手 mesh 的 reward | ❌ 全关 | `hand_support_rew_scale=0`、`hand_object_deep_penalty_scale=0`（scale=0 时连 `geom_ids` 都不解析，`config.py:1095`） |
| 点对点 `contact_hdmi` | ⚠️ 开（gain=5.0）但**不读手** | 读的是 `wrist+0.05` 虚拟点，不知道橡胶手 mesh 存在 |

→ **橡胶手 mesh 物理不碰、reward 不读，纯装饰。手往哪去全由埋在腕里的 0.05 虚拟点决定。** 例外：`cem_safety_gate`/`robot_object_penalty` 读 `head/torso/pelvis/shoulder/elbow_collision` 的 SDF，**唯独不含手**。

## 1. 两个正交实验轴（解耦，本轮只做轴2）

| 轴 | 改什么 | 机制 | 解决 | 本轮 |
|---|---|---|---|---|
| **轴1 物理碰撞** | scene.xml `lh`/`rh` contype 0/0→1/1 | 物理引擎硬阻挡（被动） | 防穿透 | ❌ 后续单独立 E 号 |
| **轴2 reward** | reward 读真实几何 | 奖励吸引手贴箱（主动） | 提升接触 | ✅ 本轮 E151 |

解耦理由：两轴解决不同失败模式（穿透 vs 接触），合改无法归因（experiment.md §5）；且轴1 开物理碰撞有 mesh-box 仿真稳定性风险，是独立的坑。

## 2. reward 轴内部：B2 → B1 → B3，work 了再叠加 B1+B2（用户拍板）

### 关键历史发现：B2 不是新东西，基础设施已存在

SPIDER 早有 **external target** 机制专门做"把接触目标挪到箱面"，B2 = 复用它，**不改 reward 代码**：
- 机制：`contact_hdmi_target_source="external"`（`config.py:211`）+ `contact_hdmi_target_path` → 从 NPZ 加载 object-local target（`run_mjwp.py:964-1008`）。
- 历史链：E093（诊断 `wrist+5cm` 离 raw 接触 >20cm，=我们 E150 结论的更早版本）→ E094（`adaptive_support` 投支撑面）→ E100（fingertip 投指尖面）→ E116（external target 实跑 4-5 case）。
- **E116 负面教训**：external surface target 单独跑，"release candidates: 0"——能把接触拉上去，但 CEM 用非语义姿态满足 target（手压上去但下半身穿物体/姿态崩/穿透增）。**但 E116 当时是 sphere/wrist 几何、没有橡胶手 mesh。** 这正是为什么需要 B1 补"让 reward 看见手面"——B2 修靶子位置、B1 让 reward 读手形，两者正交，E116 失败因为只有 B2。

### B2（先做）：external target 投影，两种投影都试

复用 external 机制 + 橡胶手场景（E116 没做过的组合）。两种投影变体（用户要求都试）：

| 变体 | 投影策略 | 生成器 | 3 case 产物 |
|---|---|---|---|
| **B2-sup** adaptive_support | raw 接触点 → 物体 world-up **支撑面**（顶面轴 clip ±half、面内 inset 1.5cm；非 raw-active 帧回退 ref_fk） | `E094/build_handbox_target_projection.py --reward-mode adaptive_support`（依赖 E093 raw-mask manifest） | ✅ 全现成 |
| **B2-tip** fingertip | palm → **指尖投票面** | `E100/build_fingertip_aware_target.py` | ✅ 全现成 |

NPZ 格式一致（`spider_contact_target_object_local`，`(T,2,3)` object-local），都走 `target_source=external`。
- config（override overlay，**不改基链语义**）：`contact_hdmi_target_source: external`、`contact_hdmi_target_path: <B2-sup 或 B2-tip 的 npz>`、`contact_hdmi_dynamic_target: false`（external 是预生成 per-frame）、橡胶手场景沿用 E147/E148 sidecar。
- **保留** rubber 场景 + `contact_hdmi_gain`（B2 只换 target 内容，reward 公式/锚点不动）。

### B1（B2 之后）：开 `hand_support_rew` + 修 `_geom_box_sdf_min` mesh 分支

让 reward 真正读橡胶手 mesh 表面：
- **已存在休眠 reward** `hand_support_rew`（`mjwp.py:1469-1488`）：`exp(-(|hand_geom→box SDF|-margin)/sigma) * gate`，配置齐全（`geom_names=['lh','rh']`、`margin=0.01`、`sigma=0.015`、`gate_source=contact_mask`），只是 scale=0。
- **真正卡点 `_geom_box_sdf_min`（`mjwp.py:68-130`）把 mesh 退化成包围球**：对 `type=mesh` 只读 `geom_size[0]`=2.58cm 当球半径 → 即使开 `hand_support` 读到的也是埋在腕里的小球，和路线A 同病。**必须加 mesh 顶点分支。**
- **改动**（隔离、可逆、git 追踪）：
  1. `_geom_box_sdf_min` 加 `mesh` 分支——按评测端同款（`eval_E147.geom_object_sdf`，`mjwp.py` 外）采凸壳顶点（下采样固定点数控开销）逐点算 box SDF 取 min，半径置 0。**纯增量**：sphere/capsule 老路径不动，默认 `scale=0` 不触发。
  2. config（override）：`hand_support_rew_scale: >0`（待 smoke 定，初值 2.0~5.0）、`hand_support_margin_m`（0.005~0.01）、`hand_support_sigma`（0.015）、`hand_support_gate_source: contact_mask`。
- **B1 子决策（用户关注的"是否抛弃 ref 接触点"）**：
  - **B1-a 叠加**（建议先）：留 `contact_hdmi`（粗对位，保接触时序/左右手）+ 加 `hand_support`（精贴合）。风险：两项可能打架（一个要腕复刻参考穿透位、一个要手面贴箱）。
  - **B1-b 替换**：`contact_hdmi_gain`→0，只靠 `hand_support`+mask。无冲突但丢参考时序信息。
  - 先 B1-a，冲突明显退 B1-b。

### B3（备选）：替换 contact_point 为 mesh↔box 最近点对（`mjwp.py:1033-1044`），可微/开销/CEM 稳定性风险大，仅 B1/B2 都不够时考虑。

### 叠加：B1+B2（两者各自 work 后）
B2 修靶子位置 + B1 让 reward 读手面，正交互补。仅当 B2 与 B1 **各自单独有提升**（见判据）后才跑叠加。

## 3. Benchmark：relaxed8 选 3 case（用户拍板"用现成3个"）

| case | task | 物体 | off05 基线 | B2-sup | B2-tip |
|---|---|---|---|---|---|
| box021_029_p2 | d003_box021_20231018_029_p2_e107_clean | box021 | ✅ E148 rubber | ⚠️ 见 §3.1 | ⚠️ 见 §3.1 |
| box004_083_p2 | e091_box004_20231003_2_083_p2_e092_dyn | box004 | ✅ E148 rubber | ✅ E094 现成可用 | ✅ E100 现成可用 |
| box023_person2 | box023_person2_legobj | box023 | ✅ E147 rubber | ✅ E094 现成可用 | ✅ E100 现成可用 |

三物体覆盖、两种投影 + 基线零额外数据准备。

### 3.1 box021 数据有效性：E103 scene template bug 的影响（已调查）

**bug（E103/E107）**：`box021_person1/scene.xml` 等的 robot link inertial 被污染成 object 质量（mass=29.632kg，本应只属 object）。受影响前缀：box021(11)、d003_box021(68)、box026(7)；**box023/box004 不受影响**（E103 plan 实证）。

**对 target 的影响判定**（已核实，非假设）：
- bug 污染的是 **robot inertial（动力学）**，不是 target 投影逻辑（几何代码无误）。
- E094/E100 box021 target 从 **kinematic 参考 qpos + object pose 在 object-local 系**算出，meta 写的是污染期 task `d003_box021_20231018_029_p2`。
- E107 clean 重建：**复用同一份 qpos**（`old_traj` 直接复制，`qpos_matches_legacy_spider=True` 13/13），只换干净 inertial scene + 重 patch object pose。本轮已验证 clean ref 与 old 的 pelvis root 逐元素吻合。
- → qpos 相同 ⟹ object-local target **理论一致**，但 E107 重 patch object pose 时若 quat 约定不同会让 target 偏 → **必须校验，不假设**（experiment.md §5）。

**处理（用户拍板：前置校验 + 按需重生成）**：B2 开跑前对 box021_029_p2 做硬前置——
1. 用 **clean scene**（`d003_box021_20231018_029_p2_e107_clean`，远端;本地有 E107 scene_snapshot 可参照）+ 同一 qpos，重跑 E094/E100 生成器（`--case-ids`，**生成器逻辑不改**），得到 clean target。
2. 与现存污染期 NPZ 比对：若逐帧 object-local 坐标一致（<5mm）→ 现存可用；若不一致 → 用 clean 重生成版。
3. box023/box004 不受 bug 影响，现存 target 直接用。

校验产物记入 log（旧/新 target 差异表）。

## 4. 实验矩阵（每 case）

| 跑法 | target | reward | 用途 |
|---|---|---|---|
| baseline | ref_fk（off05） | contact_hdmi | 复用 E148/E147 rubber，不重跑 |
| B2-sup | external adaptive_support | contact_hdmi | 投影变体① |
| B2-tip | external fingertip | contact_hdmi | 投影变体② |
| B1 | ref_fk | contact_hdmi + hand_support(mesh) | 读手面 |
| B1+B2 | external（B2 最优那种） | + hand_support(mesh) | 叠加（条件触发） |

新跑量：3 case ×（B2-sup + B2-tip + B1）= 9，叠加视情况 +3。

## 5. 数据管线对应处理（用户强调：不破坏原有）

- **external target NPZ** 走 `contact_hdmi_target_path`，**与 ref_fk 路径并存**——只切 `target_source`，原 ref_fk 管线零改动。
- **`_geom_box_sdf_min` mesh 分支纯增量**：`scale=0` 或 geom 非 mesh 时走老路径，sphere/capsule 数值不变（回归测试钉死）。
- **scene** 复用 E147/E148 rubber sidecar，不重新 patch；per-exp snapshot 进 `results/E151/scene_snapshot/` + manifest.txt（git HEAD+sha256，experiment.md §7）。
- **target NPZ 复用** E094/E100 现成产物，不重生成（若需重生成，调原生成器 `--case-ids`，不改生成器逻辑）。
- **评测**复用 `eval_E147` 顶点级 mesh SDF（已是几何精确口径）。
- override 走 overlay（`defaults: [基链, _self_]`），不改 8 case 基链语义。

## 6. 评测与成功判据（事前定义，experiment.md §5）

- **主判据**：相对 off05 rubber 基线，**`hand_geom_near_5cm` 均值 ↑ ≥ +0.05 且 `hand_geom_penetration` 不升（≤baseline）**，pelvis 不摔、`obj_err` 不恶化、leg_pen 不显著升。报 mean+std+worst（3 case），禁 cherry-pick。
- **B1 机制判据**：训练 `hand_support_sdf`/`hand_support_score` 曲线收敛到 margin 带内（reward 真在拉手面）。
- **叠加触发条件**：B2（任一投影）与 B1 **各自单独**满足主判据，才跑 B1+B2。
- **视觉（mandatory）**：grasp/contact 关键帧 A/B（off05 vs 各方案）并排，手面（掌/指）是否真贴箱、有无穿透/悬浮/姿态崩（呼应 E116 教训：警惕"手压上去但下半身穿物体"）。

## 7. 风险与对策

- **R1 mesh SDF 开销**：顶点级 × N × T × geom。对策：固定下采样点数（32~64，复用评测端 `MESH_SAMPLE_COUNT`）；smoke 测单 case step 时间；必要时只用指尖/掌心关键点。
- **R2 穿透反升**：margin 项 `|sdf|-margin` 双侧理论停表面；扫 margin；评测看 `hand_geom_penetration`/`deep2`。
- **R3 contact_hdmi vs hand_support 冲突**：见 B1-a/b 子决策。
- **R4 E116 姿态作弊复现**：external target 可能被非语义姿态满足（下半身穿物体）。对策：沿用 `contact_mask` gate；评测 + 视觉同看 leg_pen/pelvis/obj_err；这正是 B1（读手面）要补的。
- **R5 改 SPIDER 核心 reward 路径**（B1）：改动隔离、可逆、git 追踪；mesh 分支纯增量；回归测试老用例。

## 8. 验证（端到端）

1. **单元（B1 前置，关键）**：构造橡胶手贴箱已知姿态，新 `_geom_box_sdf_min`(mesh) 输出 vs `eval_E147.geom_object_sdf` 同姿态一致（<mm）。钉死"reward 口径==评测口径"。
2. **回归**：sphere/capsule 老用例 SDF 数值不变。
3. **B2 smoke**：1 case 切 external target，确认 NPZ 加载日志（`E085 external contact target: ...`）、CEM 收敛、出轨迹。
4. **B1 smoke**：1 case 开 `hand_support_rew_scale`，确认 config 吃进、step 时间可接受、`hand_support_score` 非零、出轨迹。
5. **全量**：3 case × {B2-sup, B2-tip, B1} eval；条件满足跑 B1+B2。
6. **结论**：B2（投影）/ B1（读手面）各自能否突破 E150 暴露的 reward 天花板；叠加是否补全 E116 的缺口。

## 9. 范围约束

- 只动 reward 轴；不碰物理碰撞轴（轴1）、跟踪/稳定/object actuator/partner/物体侧 collision_policy。
- 不改基链 override 语义、不改 target 生成器逻辑、不改 ref_fk 管线。
- 所有改动 git 追踪 + 回归；scene/manifest per-exp snapshot。

## 10. 记录

- `log/<N>_E151_*.md`：3 case × 多方案双指标表 + 视频 + 结论。
- `EXPERIMENT_TRACKER.md` 加 E151 行。
- memory `project_contact_anchor_misalignment`：从"路线A 证伪 + 路线B 方案"升级到"路线B 实测"。
