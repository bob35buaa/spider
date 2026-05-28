# E090 Plan: H2-First Fingertip Replacement Ablation, Then Original OmniRetarget

日期：2026-05-28
触发问题：`workspace/exp_diagnostic/my_thoughts.md`
依赖诊断：`workspace/exp_diagnostic/diagnostic_report.md`、`workspace/exp_diagnostic/findings/08_B_path_omniretarget_audit.md`、`workspace/core4d/log/111_E089_g1_feasibility_AB_results.md`

## Context

当前现象有三层证据：

1. `workspace/core4d_collab_retarget` 的 E028/E030 与 `workspace/core4d` 的 E082-E088 都在 D003 Box021 上失败；同一 SPIDER reward 栈下，`box023_person2` 通过，`box025_person2` 至少是 partial positive。
2. `workspace/exp_diagnostic/diagnostic_report.md` 显示 D003 Box021 的 G1 wrist/eef target 在 object local frame 中高度异常：`d003_box021_20231018_029_p2` 的右手 FK target 33.3% 帧位于 box 内部，多个 case 的双手都在侧面/下沿附近而不是 world-up top face。
3. E089 已经补上动态验证：`box021_person1` 这种 gate-pass 的老 Box021 数据在 SPIDER full CEM 中 head/upper/hand-floor 全 0%；D003 13 case 做 post-IK top-face 修复后 world-up-face-frac 到 99-100%，top-2 smoke 的 head_pen 都是 0%。

因此可以确认：Box021 失败不是单纯 reward/mass/hard gate 问题，而是上游 retarget 后的 G1 接触几何对下游 SPIDER 不可执行。

但还不能直接确认“你的 OmniRetarget 改进导致穿模”。原因：

- 当前 D003 production 调用 `examples/robot_retarget.py` 时没有显式传入 Phase 4 flags（`enable_constraint_relaxation`、`enable_contact_preservation`、`enable_foot_z_constraint` 等），而这些默认值在 `RetargeterConfig` 中仍是 false。
- D003 pipeline 确实使用了当前 Holosoma 代码和 `--replace_wrist_with_fingertip`，这会把 `global_joint_positions[:,20:22]` 从 SMPLX wrist 改成 5 个 fingertip 的均值。这个选择更可能直接影响 wrist/eef target 的 box-local 语义。
- “原始 OmniRetarget”可能指两个不同层面：原始求解器代码，或原始完整数据转换语义。E090 必须把这两层拆开，否则无法归因。

2026-05-28 用户补充：`--replace_wrist_with_fingertip` 当时是为了解决 Box025 物体太大、G1 臂展不够的问题，相当于把 retarget source 从 wrist 推到 fingertip cluster 来“加长”接触 reach。这个动机对 Box025 是合理的，但也加强了 H2：同一个策略在 Box021 低位合抱/侧面接触上可能把目标推到箱体内部或下沿。因此 E090 调整为 **先验证 H2，再做 original OmniRetarget 对照**。

### 术语澄清：`world-up top face`

本文里的 `world-up top face` 不是 object local `+z` 面，也不是图像坐标里的“上方”。它指每一帧物体 collision box 的 6 个外表面中，外法线与 world/MuJoCo `+Z` 方向（重力反方向）夹角最小的那一面。

因此对 XML 里带初始旋转的 box，`world-up top face` 可能对应 local `+x/-x/+y/-y/+z/-z` 中任意一个面。Box021 有约 90° around +X 的姿态，继续 hardcode local `+z` 会把侧面误当成“上表面”。E090 的严格 `world_up_face_frac` 用这个定义；为了不误杀 Box025 这类原本通过的高侧壁/legacy-local-z 接触，gate 另外保留 `legacy_local_z_face_frac`，并用 `support_face_frac=max(world_up_face_frac, legacy_local_z_face_frac)` 做 production pass/fail。

## 根因假设

### H1: D003 hand target 语义错误或对 G1 不可达

D003 Box021 的人手目标在 raw/converted 阶段就偏向侧面、下沿、角点或物体内部。G1 短臂和低 pelvis 约束会把这个问题放大，下游 CEM 只能在“追手但穿箱”和“不穿但丢 contact”之间选局部解。

### H2: `--replace_wrist_with_fingertip` 是主要触发因素

当前 D003 conversion 把 wrist target 替换为 fingertip-mean。这个策略本来服务于 Box025：物体大、手臂 reach 不够时，fingertip mean 比 wrist 更接近真实接触点，能缓解 G1 臂展不足。

Box021 D003 的风险正好相反：原始 wrist 可能仍在箱体外，但 fingertip/contact cluster 在双人低位合抱动作里更靠近箱体侧面、角点、下沿，甚至落入 collision box。OmniRetarget 后续把 `global_joint_positions[:,20:22]` 作为 G1 `left/right_wrist_yaw_link` 的 Laplacian source，等价于把 G1 wrist/eef 拉向这个更“深入物体”的点。

所以 E090 不把 `--replace_wrist_with_fingertip` 视为“错误参数”，而是验证它是否应该变成条件策略：大箱子/臂展不足时启用，小中箱体/低位合抱/inside-risk case 禁用或改写到 support face。若 no-fingertip 变体显著减少 `inside_frac` 并提高 `world_up_face_frac`，则说明当前 D003 全局默认不合适，但不能推出 Box025 也应关闭该参数。

### H3: 你的 Phase 4 改进不是 D003 主因，但可能改变 pass/fail 边界

Phase 4 的 constraint relaxation 可以把原本 infeasible 的 case 解出来，同时允许小穿透。README 中 Phase 4 的 penetration duration/depth 确实高于 Phase 3。不过 D003 production 没有显式启用这些 flags，所以它不是当前 D003 几何异常的首要嫌疑。仍需通过原始 commit/worktree 对照验证。

### H4: top-face constrained pre-IK target 是最直接修复

E089 的 post-IK lite 已验证“把 wrist/eef target 放到 world-up face +5cm”能显著改善 gate 与 SPIDER smoke。长期应在 pre-IK converted NPZ 层改写 `global_joint_positions[:,20:22]`，让 OmniRetarget 的 Laplacian solver 自然分配手臂/肩/躯干姿态。

## Claims

| Claim | 最低证据 |
|---|---|
| C1 `--replace_wrist_with_fingertip` 是否是关键变量 | same solver 下 no-fingertip 变体相对 fingertip baseline 在至少 2/3 canonical case 上同时改善 `inside_frac`、`signed_dist`、`world_up_face_frac`，且不降低 trim/contact mask 可用性 |
| C2 如果 no-fingertip 通过 G1 gate，则 SPIDER dynamic 应明显优于 E082-E088 | no-fingertip top case 24-step smoke `head_pen <= 10%`、`upper_pen <= 10%`、`hand_floor <= 10%`；full CEM 至少 1 个 case 达 `pelvis_min >= 0.55m` 且 safety 三项 <= 5% |
| C3 原始 OmniRetarget 能否进一步消除 D003 Box021 几何异常 | 三个 canonical case 中至少 2 个在 original 变体下 `inside_frac <= 5%` 且 `world_up_face_frac >= 60%`，并优于 current/no-fingertip |
| C4 如果 no-fingertip 与 original 都 fail，但 top-face constrained pass，则问题是 contact target 语义，不是 solver 或 fingertip 单一因素 | current/no-fingertip/original 均 `world_up_face_frac < 20%` 或 `inside_frac > 10%`，top-face constrained 语义 gate ≥ 5/13 pass，且 SPIDER smoke safety 明显改善 |
| C5 Phase 4 flags 是否会引入下游不可用穿模 | 当前代码 + Phase 4 flags 的 geometry/smoke 比当前默认更差，或出现更多 wrist/object penetration；若没有变差，则 Phase 4 不是本案主因 |
| C6 H2 修复不能破坏 Box025 reach guard | winning policy 在 `box025_person2` guard 上仍保持 `support_face_frac` pass；若 no-fingertip 让 Box021 变好但 Box025 变差，结论必须是条件化启用，而非全局删除 |

## 2026-05-28 Phase 1A/2B 证据更新

已完成 current no-fingertip、topface-preIK canonical retarget，以及 `box023_person2`/`box025_person2` topface-preIK guard。几何结果写入 `workspace/core4d/results/E090/geometry/geometry_summary.csv`。

| Variant / task | Gate | 关键指标 | 结论 |
|---|---|---|---|
| `d003_box021_20231018_029_p2_nofing_e090` | reject | `inside=0/0%`，`support=17.3/14.7%`，`T=75` | no-fingertip 消除 inside，但没有把手放到可支撑面 |
| `d003_box021_20231011_035_p2_nofing_e090` | reject | `inside=0/0%`，`support=27.1/21.8%` | 接近 support 阈值但仍不够 |
| `d003_box021_20231020_019_p1_nofing_e090` | infeasible | Holosoma `CVXPY solve failed: infeasible` | 不重复同配置，记为 no-fingertip 可行性风险 |
| `d003_box021_20231018_029_p2_btop_preik_e090` | reject | `support=100/100%`，仅因 `T=78<80` reject | 几何语义已修正，但该片段太短，不作为第一批 smoke 主力 |
| `d003_box021_20231011_035_p2_btop_preik_e090` | pass | `inside=0/0%`，`support=100/100%`，`T=135` | Box021 topface-preIK 第一 smoke 候选 |
| `d003_box021_20231020_019_p1_btop_preik_e090` | pass | `inside=4.0/4.0%`，`support=98/100%`，`T=101` | Box021 topface-preIK 第二 smoke 候选 |
| `box023_person2_btop_preik_e090` | pass | `inside=0/0%`，`support=100/100%` | 常规成功箱体 guard 通过 |
| `box025_person2_btop_preik_e090` | reject | `inside=48.1/51.9%`，signed distance 均值约 `-1/-7mm`，但 `support=98.8/96.9%` | topface-preIK 不能全局应用；Box025 需要保留 reach-aware/fingertip 语义 |

当前判定：

- C1 只得到“部分支持”：关闭 `--replace_wrist_with_fingertip` 能把已生成的 Box021 no-fingertip 产物从 inside 中拉出来，但无法稳定进入 support face，且第三个 canonical case 求解 infeasible。因此不能把 no-fingertip 当成最终修复。
- C4 得到强支持：Box021 上 no-fingertip 不够，topface-preIK 明显修复 support-face 语义，说明主问题是 hand target semantic，而不是单一 solver 或 fingertip flag。
- C6 已经触发约束：Box025 topface-preIK guard 失败，符合用户提出的原始动机。最终策略必须按物体尺寸、reach margin、inside-risk/contact face 条件化，而不是全局删除 fingertip replacement 或全局启用 topface rewrite。

更新后的主线：先用 Box021 topface-preIK 的两个 gate-pass case 做 SPIDER smoke，验证几何修复是否转化为动态可训练；同时把 Box025 视作 negative guard，后续只验证 baseline/fingertip 或 reach-aware 条件策略，不再把 Box025 topface-preIK 纳入正向 smoke。

## 2026-05-28 Phase 3/4 SPIDER 证据更新

Phase 3 已按计划生成 m10 + leg/upper-body-object collision 的 SPIDER smoke 派生 task，结果写入 `workspace/core4d/results/E090/smoke/smoke_eval_summary.csv`。Phase 4 只推进 smoke-pass 的 S1 到 full CEM，结果写入 `workspace/core4d/results/E090/full/full_eval_summary.csv`。

| Stage | Variant | Gate | 关键指标 | 视觉观察 | 结论 |
|---|---|---|---|---|---|
| smoke | `E090S1_box021_20231011_035_p2_btop` | pass | `contact=95.6%`、`obj_mean=0.001m`、`pelvis_min=0.538m`、head/upper/hand-floor 全 `0%` | 没有明显穿箱/手撑地，但弯腰且头部贴近箱面 | 进入 full CEM |
| smoke | `E090S2_box021_20231020_019_p1_btop` | reject | `contact=40.6%`、`pelvis_min=0.180m`、`LH_floor=56.4%`、head/upper `0%` | 后段左手/身体落地并翻箱 | 不进入 full |
| full | `E090S1_box021_20231011_035_p2_btop` | reject | `contact=57.8%`、`obj_mean=0.009m`、`pelvis_min=0.134m`、head/upper/hand-floor 全 `0%` | 安全穿模指标继续为 0，但机器人趴低/跪低，头部贴近箱面 | topface-preIK 修复几何安全，但 full CEM 仍有低 pelvis 局部解 |

关键解释：

- topface-preIK 确实把原先 E082-E088 的 head/upper/hand-floor 问题大幅消掉，C4 的“contact target semantic 是主因之一”继续成立。
- 但 Phase 4 full 未达到 `pelvis_z_min >= 0.55m`，说明单独修 wrist/support-face 还不足以保证动态姿态质量。
- 这不是 S1 的 retarget reference 先天低 pelvis：轻量 preflight 中 `load_data` 转换后的 scene-act ref pelvis min 约 `0.657m`，full rollout 的 sim pelvis min 为 `0.134m`。失败来自 CEM/reward 在更充分优化后转向低姿态局部解，而不是 pre-IK retarget 几何本身。
- 下一步不应重复 S2 full，也不应回到 reward/mass 小 sweep；应该加入 pelvis/upright/stability 约束或 elite gate，与 topface-preIK 一起验证。

## 实验矩阵

先跑 3 个 canonical failure case。第一批只做 A/B/F，直接检验 H2 与 top-face 修复；C/D/E 作为第二批，在 A/B 结果不能解释现象或需要严格归因到 solver 时再跑。

| ID | 批次 | 目的 | Retargeter | Converted NPZ | 关键差异 | 输出 task suffix |
|---|---|---|---|---|---|
| A current-baseline | 0 | 复用 D003 现状作对照 | 当前 Holosoma 默认 flags | `--replace_wrist_with_fingertip` | 已存在，不重跑 | 原始 D003 task |
| B current-no-fingertip | 1 | 隔离 fingertip replacement | 当前 Holosoma 默认 flags | 不替换 wrist | 只改 conversion，最快验证 H2 | `_nofing_e090` |
| F topface-preik | 1 | 正式修复候选 | 当前 Holosoma 默认 flags | pre-IK 改写 wrist target 到 world-up face +5cm | 语义修复上限 | `_btop_preik_e090` |
| C original-solver-same-input | 2 | 隔离求解器代码改动 | original/official OmniRetarget worktree | 与 A 相同 converted NPZ | 同输入，不同 solver | `_origsolver_e090` |
| D original-full | 2 | 验证完整原始 pipeline | original/official OmniRetarget worktree | 原始 conversion 语义 | 输入和 solver 都原始 | `_origfull_e090` |
| E current-phase4 | 2 | 验证你的 Phase 4 flags | 当前 Holosoma + Phase 4 flags | 与 A 相同 converted NPZ | 开启 relaxation/contact/foot-z | `_phase4_e090` |

Canonical case:

| Case | 历史用途 | 失败特征 |
|---|---|---|
| `d003_box021_20231018_029_p2` | E082-E088 主验证 | R wrist 33.3% inside box，T=75 |
| `d003_box021_20231011_035_p2` | E082/E083/E030 对照 | pelvis min 低，无 top-face hand |
| `d003_box021_20231020_019_p1` | E082/E083/E030 对照 | side/edge target，动态倒伏 |

若 canonical 阶段 B 或 F 任一通过，先补 Box025/Box023 guard，再扩展到 D003 13 case。只有当 B 无法解释失败，或需要回答“是否必须回滚到原始 OmniRetarget”时，才跑 C/D/E。

## Phase 0: 修正 gate 口径

E089 已发现 `g1_feasibility_gate.py` 的 `top_face_frac` hardcoded local +z，对 Box021 的 90° around +X rotation 会误判。E090 前必须先修，否则 original/current/topface 的几何对比不可信。

改动：

| 文件 | 改动 |
|---|---|
| `workspace/exp_diagnostic/scripts/g1_feasibility_gate.py` | 计算每帧或首帧 object rotation 下最接近 world-up 的 local axis/sign，用 world-up face 替代 local +z |
| `workspace/exp_diagnostic/scripts/gate_compare_b_path.py` | 与新 gate 保持一致，删除只用于补救的临时语义判定 |

验证：

```bash
.venv/bin/python workspace/exp_diagnostic/scripts/g1_feasibility_gate.py \
  --tasks d003_box021_20231018_029_p2 d003_box021_20231018_031_p2_btop box023_person2 box025_person2
```

期望：

- `box023_person2`、`box025_person2` 仍 pass；
- D003 original 失败 case 仍 reject；
- E089 B-path `_btop` case 不再因 local +z bug 被 false reject。

## Phase 1: H2-first 产物生成

新增脚本：

| 文件 | 作用 |
|---|---|
| `workspace/core4d/scripts/E090/variants.tsv` | 记录 case、variant、converted_dir、retargeted_dir、trimmed_dir、target_task |
| `workspace/core4d/scripts/E090/rewrite_wrist_top_face_preik.py` | 在 converted NPZ 层改写 `global_joint_positions[:,20:22]` 到 world-up face +5cm |
| `workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh` | 对 3 case 先跑 B/F 变体，支持 `--force`、`--include-original` |
| `workspace/core4d/scripts/E090/collect_retarget_metadata.py` | 记录 git commit、CLI flags、conda env、每个 NPZ sha256 |

### Phase 1A: 必跑，验证 `replace_wrist_with_fingertip`

只用当前 Holosoma 默认 solver，避免 original worktree 和依赖问题干扰。

```bash
bash workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh \
  --case-set canonical \
  --variants current_no_fingertip,topface_preik
```

最小结论：

- B 优于 A：H2 在 Box021 上成立，先考虑按物体尺寸、接触面、inside-risk 条件关闭 `--replace_wrist_with_fingertip`，但不能全局删除；
- B 仍 fail 但 F pass：不是简单 fingertip 问题，需要 top-face semantic rewrite；
- B/F 都 fail：再跑 original 对照和 Phase 4 flags。

### Phase 1B: 条件执行，严格归因 original/Phase4

Original worktree 策略：

1. 优先使用 official/original OmniRetarget commit 或上游 tag；
2. 若本地没有 official tag，创建 Holosoma worktree 到 Phase 4 改动前的 commit，例如 `1a43cf2` 或更早可运行版本；
3. 如果 original worktree 缺少当前 CORE4D helper，只复用当前 conversion 产物作为 `--data_path`，明确标记为 `original-solver-same-input`；
4. 若 original worktree 不可运行，则不伪造结论，只保留 current-no-fingertip 与 topface-preik 两个可执行对照。

Retarget 命令模板：

```bash
bash workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh \
  --case-set canonical \
  --variants original_solver_same_input,original_full,current_phase4 \
  --include-original
```

## Phase 2: 几何评估

新增脚本：

| 文件 | 作用 |
|---|---|
| `workspace/core4d/scripts/eval/eval_E090_retarget_geometry.py` | 对每个 retargeted/trimmed NPZ 生成 SPIDER trajectory 后计算 gate 指标 |
| `workspace/core4d/scripts/eval/render_E090_wrist_overlay.py` | 生成 object-local 三视图 overlay，和诊断报告中的 `04_overlay_*.png` 对齐 |

核心指标：

| 指标 | 通过阈值 | 解释 |
|---|---:|---|
| `inside_frac_l/r` | <= 5% | wrist/eef 不应落在 box 内 |
| `signed_dist_mean_l/r` | >= 0.03 m | 至少离表面外 3cm |
| `world_up_face_frac_either` | >= 60% | Box021 严格目标：至少一手稳定在 world-up face/上沿 |
| `support_face_frac_either` | >= 30% | 通用 production gate：world-up face 或 legacy local-z support 任一成立，保护 Box025 guard |
| `wrist_below_pelvis_gap_mean` | <= 0.20 m | 避免强迫机器人深度趴低 |
| `pelvis_z_min_ref` | >= 0.60 m | ref 姿态不能先天倒伏 |
| `T` | >= 80 | 序列长度足够 CEM |

输出：

| 产物 | 路径 |
|---|---|
| 几何汇总 CSV | `workspace/core4d/results/E090/geometry/geometry_summary.csv` |
| 每 case JSON | `workspace/core4d/results/E090/geometry/per_case/*.json` |
| overlay PNG | `workspace/core4d/results/E090/geometry/overlays/*.png` |

判定：

- 若 B 明显优于 A，说明 `--replace_wrist_with_fingertip` 是 Box021 主要风险点，original 对照降为确认项；随后必须跑 Box025/Box023 guard，确认条件化策略不会牺牲已有正例；
- 若 B 不优于 A，但 C/D 明显优于 A/B，说明 current solver 或改进代码改变了 retarget 质量；
- 若 A/B/C/D/E 都 fail 而 F pass，说明必须做 target semantic repair，不能只回滚 OmniRetarget；
- 若 E 明显更差，Phase 4 flags 对 SPIDER 生产不安全，应保留在离线可视化/可行性探索，不进入 D003/D005b 主线。

### Phase 2B: Box025 reach guard

当 canonical Box021 上出现可用 winner 后，立刻对 `box025_person2` 和 `box023_person2` 做 guard 验证：

| Guard | 目的 | 最低要求 |
|---|---|---|
| `box025_person2` baseline | 保留 `--replace_wrist_with_fingertip` 原始动机的正例 | current baseline 仍 pass，作为 reach guard 下界 |
| `box025_person2` winner policy | 检查 no-fingertip/topface 是否破坏大箱子 reach | `inside_frac <= 5%`、signed distance 不为负、`support_face_frac` pass，SPIDER smoke 不出现新 head/upper/floor 失败 |
| `box023_person2` winner policy | 检查常规成功箱体不被回归 | geometry gate pass，smoke safety 不差于已有结果 |

2026-05-28 guard 结果已经显示：Box023 topface-preIK pass，但 Box025 topface-preIK 出现 `inside=48.1/51.9%`，即使 support 很高也不合格。因此 Box025 guard 的核心不是“能否贴到某个 support face”，而是不能为了接触面语义把 G1 wrist 推进大箱体内部。

若 Box021 的 winner policy 在 Box025 guard 上失败，则最终实现必须是条件化策略，例如按 box half-extent、human contact face、inside-risk score 或 G1 reach margin 决定是否启用 fingertip replacement 或 topface rewrite。

## Phase 3: SPIDER smoke

只把 Phase 2 通过 gate 且不违反 guard 的 variant 接 SPIDER。基于 2026-05-28 Phase 1A/2B 结果，第一批 smoke 范围收敛为：

| Smoke group | Task | 原因 |
|---|---|---|
| S1 | `d003_box021_20231011_035_p2_btop_preik_e090` | topface-preIK canonical gate pass，`T=135` |
| S2 | `d003_box021_20231020_019_p1_btop_preik_e090` | topface-preIK canonical gate pass，`T=101` |
| Hold-out | `d003_box021_20231018_029_p2_btop_preik_e090` | support 已到 100/100%，但 `T=78<80`，先不作为 smoke 主力 |
| Negative guard | `box025_person2_btop_preik_e090` | inside `48.1/51.9%`，禁止作为正向策略推广 |

所有 Box021 smoke 使用同一 scene 策略，避免 mass/collision 混杂：

- Box021 mass 固定为 10kg；
- 保留 E083 upper-body-object pairs 和 E081 leg/foot-object pairs；
- 使用 E089/E088 reward stack，不再 sweep reward；
- `contact_hdmi_target_source=ref_fk` 作为第一轮 smoke，避免 raw external target 与不同 retarget 语义不一致；
- 每个 scene 启动前 snapshot 到 `workspace/core4d/results/E090/scene_snapshot/`。

新增脚本：

| 文件 | 作用 |
|---|---|
| `workspace/core4d/scripts/E090/build_spider_tasks.py` | 从通过 gate 的 trimmed NPZ 生成 derived task、scene_act、override |
| `workspace/core4d/scripts/train/train_E090_smoke.sh` | 每个 variant 跑 `max_num_iterations=4` |
| `workspace/core4d/scripts/eval/eval_E090.py` | 汇总 geometry + smoke + full CEM 指标 |

Smoke 命令：

```bash
.venv/bin/python workspace/core4d/scripts/E090/build_spider_tasks.py --stage smoke
bash workspace/core4d/scripts/train/train_E090_smoke.sh local 0
.venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage smoke
```

Smoke 通过标准：

| 指标 | 目标 |
|---|---:|
| `head_pen_frac` | <= 10% |
| `upper_pen_frac` | <= 10% |
| `LH/RH_floor_frac` | <= 10% |
| `obj_err_mean` | <= 0.10 m |
| 视觉 | 双手在箱体 world-up face/上沿附近，无头胸压箱、无手撑地捷径 |

若 S1/S2 smoke 的 safety 明显优于 E082-E088，则进入 full CEM；若 smoke 仍出现头胸压箱或手撑地，下一步不是回到 no-fingertip，而是降低 topface offset（+5cm → +3cm）、只投影 active contact frames，或在 pre-IK rewrite 中加入 wrist signed-distance margin。

2026-05-28 实际结果：S1 smoke safety pass，S2 因手撑地 fail；S1 full safety 仍为 0% penetration/floor，但 pelvis collapse。后续 Phase 4B 应只围绕 S1 做一个姿态约束验证，例如 `pelvis_z`/torso-upright elite gate、加强 qpos/upright tracking，或在 CEM cost 中显式惩罚 pelvis 低于 0.55m。

## Phase 4: Full CEM

只对 smoke top-2 做 full CEM：

1. 最优 current-no-fingertip 变体一条，如果 Phase 2 显著改善；
2. 最优 topface-preik 变体一条；
3. 可选 original 变体一条，只有当 Phase 1B 显示 original 明显优于 B/F 时执行。

2026-05-28 实际执行：只对 smoke-pass 的 `E090S1_box021_20231011_035_p2_btop` 跑 full。结果 safety 三项通过但 pelvis 失败，因此 E090 不进入 13 case 扩展，先做 Phase 4B 姿态约束。

训练命令：

```bash
bash workspace/core4d/scripts/train/train_E090_full.sh local 0
.venv/bin/python workspace/core4d/scripts/eval/eval_E090.py --stage full
```

Full CEM 成功标准：

| 指标 | E082-E088 baseline | E090 目标 |
|---|---:|---:|
| `head_pen_frac` | 18-89% | <= 5% |
| `upper_pen_frac` | 53-89% | <= 5% |
| `LH/RH_floor_frac` | 10-81% | <= 5% |
| `pelvis_z_min` | 常低至 0.17-0.64m | >= 0.55m |
| `obj_err_mean` | 0.61-0.78m | <= 0.10m |
| `contact_either` | 70-83% 但靠压箱 | >= 45%，且视觉为手部接触 |

## Phase 5: 13 case 批量扩展

触发条件：

- no-fingertip 或 topface-preik 在 canonical full CEM 中至少 1 个通过；
- 或者 geometry pass rate 在 canonical 阶段达到 2/3。

批量目标：

```bash
bash workspace/core4d/scripts/E090/run_holosoma_retarget_ablation.sh \
  --case-set d003_13 \
  --variants <winning_variants>
.venv/bin/python workspace/core4d/scripts/eval/eval_E090_retarget_geometry.py --case-set d003_13
```

判定：

| 结果 | 下一步 |
|---|---|
| no-fingertip 13 case 中 >=5 pass，SPIDER top-2 smoke pass | D003/D005b 对 Box021/低位合抱禁用 `--replace_wrist_with_fingertip`，或改成条件化启用 |
| original 13 case 中 >=5 pass，SPIDER top-2 smoke pass | 回滚或固定 original retarget path 进入 D005b |
| topface-preik >=8 pass，SPIDER top-2 full 至少 1 pass | 集成 pre-IK target rewrite 到 Holosoma D005b/D006 |
| 所有变体均低于 3/13 pass | Box021 D003 作为数据池整体降级，转 Box026/Box022 新箱型 |

## 风险与处理

| 风险 | 处理 |
|---|---|
| no-fingertip 改善 Box021 但可能损害 Box025 | 后续必须在 Box025/Box026 guard 上复测；结论只允许“条件化”，不能全局删除 fingertip replacement |
| original OmniRetarget worktree 无法在当前数据/依赖上运行 | 记录 blocked reason，不用它做因果结论；保留 no-fingertip 和 topface-preik 对照 |
| original conversion 与当前 pipeline API 不兼容 | 拆成 `original-solver-same-input` 与 `original-full`，能跑哪个用哪个，结论标清 |
| topface-preik 改写导致手臂 IK residual 大或 Box025 inside 回归 | 与 E089 post-IK lite residual 对照；对 Box021 可先只投 active contact frames 或降低 +5cm 到 +3cm；对 Box025 不使用全局 topface-preIK，改用 reach-aware/fingertip 语义 |
| ref_fk target 降低 contact 指标 | safety 优先；contact 作为视觉确认后的二级指标，必要时增加 raw/external target 复评 |
| 13 case 批量耗时过长 | canonical 3 case 先决策；批量只跑 winning variants |

## 不做

- 不继续对 E082-E088 失败输入做 reward weight/mass/hard-gate sweep；
- 不把 D6 locked support scaffold 作为主线；
- 不用 hardcoded local +z gate 判断旋转 box；
- 不把 Phase 4 “retarget success”直接等价为 SPIDER trainability。

## 预期决策树

1. 已观察到 no-fingertip 只解决 inside，不解决 support face，且有一个 canonical case 求解 infeasible。因此 no-fingertip 不作为 E090 第一修复主线，只保留为 H2 的部分证据。
2. 已观察到 Box021 topface-preIK canonical 2/3 gate pass，剩余 1/3 只因 `T=78<80` reject。下一步优先验证 topface-preIK 是否带来 SPIDER dynamic safety 改善。
3. 已观察到 Box025 topface-preIK guard 失败。最终实现必须是条件策略：Box021/低位合抱/inside-risk 使用 support-face semantic repair；Box025/大箱子 reach-risk 保留 fingertip replacement 或 reach-aware 目标。
4. original OmniRetarget 对照降为确认项：只有当 topface-preIK 的 SPIDER smoke/full 不能转化为动态收益，或需要对论文/工程回滚做严格 solver 归因时再跑 C/D/E。
5. 已观察到 topface-preIK full 没有 canonical pass：S1 safety 通过但 pelvis collapse，S2 smoke 手撑地失败。因此不扩展 D003 13 case；下一步是 S1-only 的姿态约束验证。如果姿态约束后仍无法过 full，则 Box021 D003 单 G1 数据池降级，转 dual-G1/mocap partner 或 Box026/Box022。
