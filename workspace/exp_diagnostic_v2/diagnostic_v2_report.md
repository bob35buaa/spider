# 实验诊断 v2 — E088–E097 复盘 + 接触面审计 + OmniRetarget 硬骨头角度

日期：2026-05-30（含当日修订：B1 影响范围下调、新增 B6 接触语义错位、box021 quat 90° 几何解释）
范围：复盘 E088–E097（diagnostic v1 之后的 10 个实验）；审计 anchor-face 选择逻辑与可视化；列出"硬啃 OmniRetarget"的可行角度。
执行：spider 主仓库本地 GPU 节点，read-only（未改任何 pipeline 代码）。
配套细化报告（三份，全中文）：

- `workspace/exp_diagnostic_v2/findings/01_e088_e097_audit.md` — E088–E097 vs `exp_diagnostic/diagnostic_report.md` 的逐项核对、冲突列表、未验证项
- `workspace/exp_diagnostic_v2/findings/02_face_selection_audit.md` — anchor face 选择 + 接触语义 全 codebase 审计（B1-B5 旧 anchor pipeline bug + 新 B6 接触语义错位）
- `workspace/exp_diagnostic_v2/findings/03_omniretarget_hardbone_angles.md` — OmniRetarget 13 个候选改造角度、Top-3 路线、未知项

辅助：`workspace/exp_diagnostic_v2/scripts/check_face_selection.py`（验证 B1 bug 的最小脚本）。

## 本次修订摘要（2026-05-30 晚）

回应用户的两个尖锐问题：

1. **"box023_p2 L 面也错了，为什么 full CEM 还正常？"**
   - 触发对 B1 影响范围的全 codebase 复核：grep `support_proxy_enabled: true` 只在 E006-E008 出现，**E082-E097 阶段所有 box021/box023/box025/box004/Box026 的 config 都没启用 support weld**。B1 在 E082-E097 阶段的真实影响降级为**统计污染**（让 `anchor_face_review=true` flag 失去诊断价值、让 face 统计被错读），**不是这些实验失败的因果驱动**。
   - box023_p2 能过的真正原因是 v1 §3.3 / §3.4 的纯几何量：wrist FK 在 box 外 +18-28 cm、hand z 高过 pelvis；CEM 直立就能到。box021 D003 wrist FK 0%–33% 在 box 内、hand z 比 pelvis 低 12-25 cm；CEM 无可行位姿。两 case 的成败差异由"基础几何"决定，**与面选择正交**。
   - box023_p2 L 的 z 面占比只有 51%（边界 case），xy-only 把它判到 +y 严格说只是"二选一里赌错了一边"，不像 box021 那种 +z 占 65-76% 的清晰错判。

2. **"wrist 不是真实接触点，CORE4D 5 指尖在 SMPL-X 转换中已被丢，G1 又没有手指 DoF，这是潜在的接触语义问题"**
   - 用户洞察完全成立。该问题**单独成为 B6**，已写入 02 §4。其本质是**整条 OmniRetarget → SPIDER pipeline 中没有任何位置真正代表"指尖在哪里"**。
   - 三层信息损失：(1) SMPL-X 22-joint 标准把 5 指尖丢成 1 个 wrist；(2) `--include_fingertip_centers` 在 D003 production 默认 OFF，即便开了也只剩 1 个均值点；(3) G1 标准 URDF 的 `left_rubber_hand_link` 是无 DoF 的球。
   - 量化证据：E093 的 `wrist+5cm → raw mean` ≥ 20 cm 全 case、Box026/Box025 49-64 cm。
   - **对 v1 H1 的修正解读**：v1 §3.3 报 box021 18029_p2 R wrist FK "33% 帧 INSIDE box"，其中**至少一部分是 wrist 内陷的几何假象**——人指尖贴 +y 侧外，wrist 被反向弯曲带到 box 几何中心方向，FK 落点看上去 INSIDE 但人手并没穿模。所以 H1 方向正确、粒度太粗；真正的根本错位发生在 STAGE A（wrist 抽样代替 contact），IK FK INSIDE 是次生现象。

3. **顺带发现 box021 quat 90° X 的几何坑（解释 E090 反退化）**
   - box021 18029_p2 `obj quat = 90° around +X`，旋转后 **local +z → world −y（水平方向）、local +y → world +z（向上）**。
   - 02 §3 实测表里 box021 R wrist "主接触面 = local +z" 对应世界里是**水平长侧面，不是顶面**。
   - v1 §6 / E090 实施的 "topface-preIK" 把 contact target 投到 local +z + 5 cm —— 对 box021 等价于推到 world −y 水平方向，**反着推**。
   - 解释 E090 失败模式（safety 全 0% 但 pelvis 0.134 m）：IK 推 G1 去够侧面而 ref motion 是蹲箱旁按局部 +z，姿态对不上；reward 必须跟 ref motion → pelvis 塌下去。
   - 同理解释 Box025 反退化（E090 `inside 48.1/51.9%`）：Box025 在 0.31 m 高凳上人侧抓侧壁，world-up 投影把 contact 推到顶面 → IK 不可行。
   - **结论**："world-up = top" 假设只对 quat ≈ identity 的对象成立。任何"投到 box 顶面"的启发式都必须由 raw mocap 实际接触面决定方向——这正是 Angle 8 应该驱动 Angle 2 的几何依据。

---

## TL;DR

**Q1（E088–E097 是否解释/解决 v1 诊断的问题？）**

- v1 的核心主张 H1（box021 D003 IK 目标在几何上不可行）是 **方向上被证实，但 v1 §6 设计的"同一 case、同一 reward、仅改 target"的干净 A/B（B2 / B3）从未真正执行**；E089A 换了 `box021_person1`（动作 / 时间窗口 / person 全换），E089B 写的是 qpos 改写（B-2 lite），E090 用 topface-preIK 跑的是另外的 case (`035_p2` / `019_p1`)。因此"H1 verified" 的结论方向正确，但实验设计上不干净。
- **新发现：几何修复是必要不充分** —— E090 S1 full、E091 box004 smoke、E094 box026_039 三个独立证据都出现"safety penetration 全 0%、但 pelvis 塌（0.08-0.44 m）/ 身体压箱"的失败模式，是 v1 H1 没预测到的"上肢/姿态层失败"。
- E094 C2 出现 v 字反直觉：target repair 让所有数值指标都改善（contact 33%→80%、obj_err 0.009→0.002 m、pelvis 0.083→0.440 m），但视觉上变成"机器人趴在箱子上"——当前 head/upper/floor gate 检测不到这种 invalid behavior。
- E097 的"8 个新候选"在视觉复审后归零（5/6 已有 D004 历史结论，1/6 D003 CVXPY infeasible）——说明 mining 漏过了 legacy outcome 列。
- 还存在 E088→E089→E090 的口径漂移：E089B 报 9/13 PASS gate（B-2 lite），E090 报 2/3 PASS gate（B-1-like），两套 gate 语义不同、case 集不同，但 EXPERIMENT_TRACKER 并列展示，造成"yield 在涨"的错觉。

**Q1（面选择 / 接触语义 / 可视化）**：审计出 **6 个 bug，其中 4 个属于 "wrong-target / 语义错位"**：

- **B1**：`E017.audit_select_anchors.py:48,192` 起的 anchor 选择器只用 xy 二维 argmax，**完全屏蔽 ±z 面**；以 box021/box023 实测，true top face 为 +z 的 8 个 hand-case 中 7 个被错判为 ±x/±y。**影响范围已修正**：是 9/13 D003 box021 `anchor_face_review=true` flag 的根因、是 E006-E028 weld-based 实验的支架错位根因；但 **E082-E097 阶段所有 box021 config 都没启用 support weld，B1 在这一阶段降级为「统计污染」（让 face 解读错），不是失败驱动**。
- **B2 / B3**：E028b、E029 的 `_project_to_face` 把 ±z 面静默映射到 ±y / 默认回退 `+x`，目前因为上游已经在 B1 错成侧面，所以暂未触发；修 B1 之后立即变 wrong-target，必须同步修。
- **B4**：`spider/process_datasets/core4d.py:128-144` 写到 `trajectory_kinematic.npz` 的 `contact_pos` 其实是 IK FK palm（不是 raw mocap），但下游 E017-E028 的所有"raw contact face 统计"都按 raw mocap 解读。v1 诊断 §3.2/§3.5 也踩了这个坑。
- **B5**：`anchor_face_review=true` 只是 informational，标了之后 case 依然进入 full CEM。
- **B6（新增，核心）**：**接触语义错位 (wrist ≠ contact)** —— 整条 OmniRetarget → SPIDER pipeline 中**没有任何位置真正代表"指尖在哪里"**。三层信息损失（人 → SMPL-X → OmniRetarget → G1 → spider）让 5 指尖坍缩成 1 个无 DoF 的 sphere。**E082-E097 box021 D003 失败的真正主因之一**：v1 §3.3 报告的"R wrist FK 33% INSIDE box" 至少一部分是 wrist 内陷的几何假象（人指尖贴外面、wrist 被弯曲带到 box 几何中心方向），所以 v1 H1 方向对、粒度太粗。根本修复在 STAGE A 输入端补 fingertip 信息，不是在 SPIDER reward 端摇 offset。详见 02 §4。
- 现代路径（E093 / E094 / `check_d005_handbox_surface_gate.py` / `examples/run_mjwp.py` runtime / OmniRetarget 本身）**面选择无 bug**，但**都受 B6 影响**——它们做的"全 3D 面投票"输入还是 wrist / palm site FK，不是 raw 指尖。
- 可视化方面：现有 PNG 渲染脚本对代码忠实（绘制实际选中的面），所以"代码错 ↔ 图也跟着错"，肉眼不会发现。`data_construction_v2/visualizations/raw_contact/*.png` 的生成脚本**两个 repo 里都没有**，不可复核。

**box023 为何能过 vs box021 为何过不了**（针对用户疑问的直接答复）：
- box023 的 wrist FK 信号良好（signed dist +18-28 cm 在 box 外、hand z 高过 pelvis），**底层几何本来就可行**，所以即便 B1 把面标错也不影响 reward 把 G1 收敛到合理姿态。
- box021 D003 的 wrist FK 0%-33% 在 box 内、hand z 比 pelvis 低 12-25 cm，**底层几何就不可行**——而其中"INSIDE box" 部分实际由 B6（wrist 不能代表 contact）放大，并非物理上人手穿模。
- 这两个 case 的成败差异由 v1 §3.3 / §3.4 的**纯几何量**决定（无面概念），与 B1 的面错判正交。

**Q2（OmniRetarget 硬骨头 —— 不绕开如何啃？）**：13 个候选角度，按 Impact × Feasibility 排序后 Top-3 路线：

1. **A2 + A8（pre-IK 手腕 + 接触面投影，用 raw mocap 指尖投票打 face label）** — 推广 E094 `adaptive_support` 启发式到非世界向上面，主案 raw mocap 接触面而非世界 up 假设。修复"hand-in-box + wrong face"两类失败的同时保留 box023/box025 (side grip) 守门。1-2 工程日。
2. **A1（在 IK 求解器内加 wrist-object SDF 软/硬约束）** — 直接堵 33% R-wrist INSIDE box 这条根因；`mj_jacBody` 已经在用，2 工程日。
3. **A4（pelvis-height floor 软约束）** — 直接堵 E090/E091/E094 那一类"safety 全 0%、pelvis 塌"的失败模式；1 工程日。

如果三条做完 box021_D003 pelvis 仍 < 0.55 m，**A3（per-limb 分段 smpl_scale）** 是自然的 Top-4，因为它在根因上解释了"为什么 G1 不得不蹲下"。

被显式拒绝 / 推迟的：A11（学习式 retargeter，数据不足，等 PASS pool > 200 再说）、SPIDER 端再做 reward 调参（E060-E088 已证伪）。

---

## Part 1 — E088–E097 vs v1 诊断的核对

### 1.1 每个实验做了什么 / 结论是什么

详见 `findings/01_e088_e097_audit.md` §1（每实验一段，包含 setup + 关键数字 + 当时结论）。简表：

| Exp | 主要变化 | 结果 | 一句话定位 |
|---|---|---|---|
| E088 | hard CEM gate + absolute clearance reward | FAIL（3 变体 head/upper pen 仍 27-80%）| 证明问题不在 reward 调参 |
| E089A | 切到 `box021_person1`（feasibility gate PASS）| WORK（pelvis 0.687 m，safety 全 0%）| 证明"几何可行的同对象 case 可解"，但 confound H2 |
| E089B | 13 case post-IK qpos 改写（B-2 lite）| 几何全部修复 inside→0%，世界向上面占比 4-14%→99% | 计划中的 B-1（重跑 OmniRetarget）从未执行 |
| E090 | topface-preIK + 去 fingertip 替换 ablation | S1 full pelvis 0.134 m FAIL，box025 gate 由 PASS 变 REJECT | **几何对，姿态仍塌**；topface 不能全局用 |
| E091 | data_construction_v2 medium box（Box026 / box004）| Box026 三 case 全 reject；box004_083_p2 PASS gate，smoke pelvis 0.079 m REVIEW | 第一个 box004 可行 seed |
| E092 | C1/C2/C3 full CEM + 直接 OmniRetarget RL 对照 | C1 box004 WORK，C2/C3 box026 FAIL | box004 第一份 CEM 正样 |
| E093 | 7 case contact target 几何审计 | wrist+5cm vs raw mean ≥ 20 cm 全 case；box026 49-64 cm | target 语义偏差量化 |
| E094 | `adaptive_support` 外部 target | C1 box004 WORK；**C2 box026 metrics 全改善但视觉为"趴箱"**；C3 反退步 | 反直觉点 1，gate 漏检 invalid behavior |
| E095 | feature-based candidate mining | 2 个新 box004 case Stage2b+D005b PASS | box004 case 库扩到 3 |
| E096 / E096b | box004 first batch full CEM（mask off / on）| P1/P2 WORK；mask 切换 Δcontact ≈ -1 pp | box004 3-case 稳健正样 |
| E097 | feature 路径再 mine 候选 | 6 enabled 后被视觉复审驳回到 0 | mining 漏 legacy outcome |

### 1.2 v1 诊断 §5 主假设的最新状态

| H | v1 主张 | v2 状态 | 关键证据 |
|---|---|---|---|
| H1 | box021 D003 wrist 目标几何不可行 | **方向 VERIFIED，干净 A/B 未做；粒度需修正** | E089A WORK on `person1`（confounded）；E089B 修几何到 99% 世界向上面；**E090 S1 full 几何 OK 但 pelvis 0.134 m FAIL** → H1 必要不充分。**粒度修正**：v1 §3.3 报 "R wrist 33% INSIDE box" 至少部分是 B6 wrist 内陷的几何假象（人指尖贴外面、wrist 被弯曲带到 box 中心方向、FK 落点看似 INSIDE 但实际人手没穿模），真正的根本错位在 STAGE A wrist 抽样代替 contact，IK INSIDE 是次生 |
| H2 | box021 D003 是双人协作动作，G1 单人不行 | **对象层 REFUTED；动作层 PARTIAL** | E089A 同对象的 `person1` 单 G1 PASS；但 E090 S1 full / B4 smoke pelvis 0.087-0.18 m 暗示某些 D003 动作仍 motion-level 不可行 |
| H3 | D003 person / time window / smpl_scale 选错 | **PARTIAL，但 wrist→fingertip 通道是更深层证据** | E090 confirm `--replace_wrist_with_fingertip` 改变几何但非干净赢；**E093 量化 wrist+5cm vs raw mean 差 20-64 cm 是 B6 接触语义错位在 H3 维度的直接证据**；E097 028_p1 D003 CVXPY infeasible；person/window 轴未复测 |
| H4 | mass 异常引发反作用 | **REFUTED 为主因** | E091/E092 5 kg 标准化后 Box026 仍 fail；E094 C2 obj_mean 0.002 m 说明 mass 不是 binding |

完整对照表（含 v1 §4 候选根因表 + §6 Phase B/C action items + `data_filter_recommendation.md` §5）见 `findings/01_e088_e097_audit.md` §2。

### 1.3 冲突 / 反直觉点（共 11 条，挑 6 条要点）

1. **E089A 被当成"H1 verified"用，实际同时改了 (a) target 几何 (b) ref motion (c) time window (d) person**。该有的 same-case A/B（B2 vs B3）从未跑。`findings/01_e088_e097_audit.md` §3.2。
2. **E094 C2 反直觉**：所有定量指标改善 → 视觉变差（趴箱）。当前 gate 测不到这种 invalid behavior，是最大的评估盲区。§3.3。
3. **E094 C3 target repair 让 contact -16 pp、RH floor +5 pp**——"修几何只会更好"的假设被局部反例打破。§3.4。
4. **E090 topface-preIK 让 Box025 gate 由 PASS 变 REJECT**（inside 48/52%）+ E090 S1 box021 full 也 pelvis 塌——v1 推荐的 wrist clamp 不能全局应用。**新增几何解释**：box021 18029_p2 `obj quat = 90° around +X`，旋转后 local +z = world −y（水平方向），"topface"投影把 contact 推到 box 水平长侧面而非世界顶面，对 box021 是反方向推；Box025 是 0.31 m 高凳上侧抓，world-up 投影同样反方向。结论：world-up = top 只对 quat ≈ identity 成立。§3.5 + §3.11。
5. **B1 影响范围下调**（本日新增）：grep `support_proxy_enabled` 确认 E082-E097 阶段 box021 config 都没开 weld；B1 在该阶段是「面统计污染」而非「失败驱动」。原"B1 解释 9/13 box021 D003 失败"被改正为"B1 解释 9/13 `anchor_face_review=true` flag + E006-E028 weld 错位"。§3.10。
6. **B6 接触语义错位**（本日新增）：5 指尖 → 1 wrist → 1 无 DoF 球，三层信息损失；E093 量化 wrist vs raw mean 差 20-64 cm；v1 §3.3 "wrist FK 33% INSIDE box" 至少部分是 wrist 内陷的几何假象。详 `findings/02_face_selection_audit.md` §4。

其余 5 条（E089B vs E090 yield 口径漂移、E096 mask 配错却结果稳健、E097 phantom candidates、E088 clearance 被 CEM hack 翻箱、E089A 把 case 替换当成验证）见 01 §3.6-3.9。

### 1.4 v1 诊断 §6 Phase B/C 中没真正执行的事

| 项目 | 状态 | 缺失内容 |
|---|---|---|
| B1: 写 `repaired_contact_target_object_local.npz` 给 `contact_hdmi_target_source=external` 用 | **SUBSTITUTED** | E089B 改的是 qpos，不是外部 target NPZ；E094 才出了 adaptive_support 但不是 `18029_p2`。|
| B2 / B3: same case 18029_p2 仅改 target 的 24-step CEM A/B | **NOT EXECUTED** | E089A 换 case；E090 换 case；干净对照空缺。|
| C1: ≥50 pp 收益后全量 13 case + full CEM | **PARTIAL** | E090 做了类似动作，S1 full 失败在 pelvis 而非 head pen，C1 触发器从未被显式评估。|
| C2: 转 dual-G1 / Mocap partner 路径 | **NOT TESTED** | E089A 的 person 切换是数据侧绕开，不是 dual-G1。|
| C3: 真重跑 OmniRetarget pipeline | **NOT EXECUTED** | hsretargeting conda env 没装上，B-1 仍是 in-spirit；audit doc 已注明 caveat。|
| Pelvis collapse 机制 ablation | **NOT TESTED** | E090 / E091 都出现 pelvis 塌，但 ref_pelvis_z 对比 / pelvis floor gate 未做。|
| Box022 raw-contact preflight | **NOT DONE** | 6 个 Box022 case 自 E091 至今仍在 `raw_contact_preflight_disabled`。|
| Mask source ablation 反向（mask 切换能否救 fail case）| **NOT DONE** | E096b 只在 box004（已 WORK）上换；fail case 没换过。|

完整列表（10 项）见 §4。

---

## Part 2 — 接触面 + 接触语义审计

### 2.1 面（face）选择 + 接触代理点的全 codebase 分布

`findings/02_face_selection_audit.md` §1 给出全部 file:line。简单分群：

- **SPIDER 主仓库 / reward / runtime**：没有"面"的概念。reward 消费的是 3D 点（`contact_pos_ref` / 外部 `*_contact_target_object_local.npz`）。**但所有 3D 点都来自 wrist FK / palm site，不来自 raw 指尖**——这是 B6 的入口。
- **OmniRetarget**：没有面的概念，wrist 当作一个普通 mesh vertex（仅有 climbing 模式的 z-weighted surface 采样）；JOINTS_MAPPINGS 中 `L_Wrist → left_wrist_yaw_link` 永远启用，`L_Fingertip_Center → left_rubber_hand_link` 仅在 core4d_v2+`--include_fingertip_centers` 时启用，**D003 production 默认 OFF**。
- **`core4d_collab_retarget/scripts/E0NN/` 老 anchor pipeline**：是面选择的真正发生地（E017/E018b/E020/E028/E028b/E029），**B1-B5 全在这里**。
- **现代路径**（E093/E094 + holosoma `check_d005_handbox_surface_gate` + bucket semantics）：全 3D 面逻辑正确，**但都受 B6 影响**——它们的输入仍是 wrist / palm site FK，不是 raw 指尖。

### 2.2 6 个 bug（详细见 02 §3 和 §4）

| # | 位置 | 性质 | 影响范围 | 修法 |
|---|---|---|---|---|
| **B1** | `E017.audit_select_anchors.py:48,192`（被 E018/E018b/E020/E028 继承）| wrong-target（E006-E028 阶段）/ statistical pollution（E082-E097 阶段，已修正）| xy-only argmax 屏蔽 ±z；box021 D003 实测 8/9 case 真 top face 为 +z 被错判为 ±x/±y；**解释 9/13 `anchor_face_review=true` flag 和 E006-E028 weld 错位；不直接解释 E082-E097 阶段失败**（这些阶段无 weld）| 改 `FACE_ORDER` 加 ±z，`face_label` 用全 3D argmax |
| **B2** | `E028b/build_e028b_manifest.py:164-174` | wrong-target（latent）| `_project_to_face` 把 ±z 静默当作 ±y；目前上游已被 B1 错成侧面，未触发 | `axis = "xyz".index(face[1])` 或 assert |
| **B3** | `E029/generate_e029_d6_assets.py:166` | wrong-target（latent）| 非 side face 静默回退 `+x` | 同上 |
| **B4** | `spider/process_datasets/core4d.py:128-144` | misleading | `contact_pos` 是 IK FK palm site（阈值 0.15 m）不是 raw mocap；E017-E028b 全部"raw contact"统计实际跑在 IK FK 上；**v1 诊断 §3.2/§3.5 也踩了这个坑** | rename 为 `contact_pos_fk` 或同时存 raw + FK 两路 |
| **B5** | `E028/manifest.tsv` `anchor_face_review=true` 仅 informational | 流程 | 标了的 case 依然进 E082-E088 full CEM 队列（虽然 weld 没用，但 manifest 过滤仍据此放行）| gate 上 hard block，直到面 refit 重做 |
| **B6**（新） | 整条 OmniRetarget → SPIDER pipeline | wrong-target（根本性）| **接触语义错位 (wrist ≠ contact)** ：5 指尖被 SMPL-X 22-joint 丢成 wrist；`--include_fingertip_centers` D003 OFF；G1 rubber_hand 无 DoF；spider contact_pos = palm site FK。**整条 pipeline 中没有任何位置真正代表"指尖在哪"**。E093 量化 gap ≥ 20 cm 全 case、Box026/Box025 49-64 cm。v1 H1 "wrist FK 33% INSIDE" 至少部分是 wrist 内陷的几何假象 | STAGE A 启用 `--include_fingertip_centers` + 从 raw mocap 重算 5 指尖 + face/reward target 由 5 指尖投票驱动 |

### 2.3 可视化忠实度

| 渲染物 | 是否与代码选中的面一致 | 备注 |
|---|---|---|
| `findings/04_overlay_*.png`（v1）| 是 | 散点忠实展示存储的 contact_pos / FK wrist 在 object local 的位置；没有面标签，所以画的就是真相。**但 npz 本身受 B4/B6 污染** |
| `data_construction_v2/visualizations/raw_contact/*.png` | **不可复核** | 生成脚本在 spider/holosoma 两 repo 都不存在；只有 PNG 输出和 manifest |
| `data_construction_v2/visualizations/d005b/*_object_local_overlay.png` | 应一致（面选择层）| 对应的 `check_d005_handbox_surface_gate.py` 用全 6 面逻辑；**但输入仍是 wrist / palm site，不是 raw 指尖** |
| `core4d_collab_retarget/results/E028/anchor_visual/` | **画的是代码选中的（错的）面** | 渲染忠实 ↔ 但代码本身是 xy-only 错的，所以"图也跟着错"，肉眼复核不会发现 |

**新增风险（来自 B6）**：所有现存的"面 / 接触可视化"都画的是 IK FK palm site 的分布，不是人指尖的分布。要让可视化恢复诊断价值，需要在 PNG 上同时叠加 raw 5 指尖位置（在 STAGE A enable fingertip_centers 之后才能拿到）。

### 2.4 推荐的修复顺序

`findings/02_face_selection_audit.md` §6（9 条）；最优先：

1. **立即（同一天）**：修 B1+B2+B3（必须同步，否则 B1 修后 B2/B3 立即变 wrong-target）；写 deprecation warning 到 `core4d.py:128` 标注 `contact_pos` 是 FK；加 pelvis-collapse detector 到 CEM gate。
2. **短期（1-2 工程日，根治 B6）**：OmniRetarget D003 production 永久启用 `--include_fingertip_centers` + core4d_v2；face 决策端从 wrist 改成 raw 5 指尖投票（per-frame，distance < threshold 才投，允许 corner/edge 标签）；contact reward target 从 wrist+5cm 改成 fingertip-vote-face + 投票面上最贴近接触的指尖位置（E094 `adaptive_support` 的非 world-up 推广）。
3. **中期（结构整理）**：PNG 加面标签 + 叠加 raw 指尖；统一 6 处不同的 `face_label` 到一个 helper；hard block `anchor_face_review=true` case。

---

## Part 3 — OmniRetarget 硬骨头：13 个角度 + Top-3

完整内容（含 pipeline 拆解、每个失败模式归到 OmniRetarget 的哪一步、每个角度的 What / Why / Cost / Risk / Quick falsifying test）见 `findings/03_omniretarget_hardbone_angles.md`，574 行。

### 3.1 OmniRetarget 当前 pipeline 简图

```
CORE4D (SMPL-X 22 joints + object pose) 
  → STAGE A: convert_core4d_to_omniretarget.py  (rows 20/21 = wrist；可选 22/23 fingertip center)
  → STAGE B: InteractionMeshRetargeter (per-frame CVXPY DiffIK + Laplacian deformation cost)
              cost ← mesh 形变；constraint ← ground/object 非穿透 + foot-stick + joint limit + trust region
              **object trajectory LOCKED**, robot is the only thing being solved for
              **wrist 是一个普通 mesh vertex；没有 object-SDF wrist cost；没有 pelvis floor cost**
  → STAGE C: trim_no_contact.py  (IGL signed distance contact mask)
  → STAGE D: spider create_spider_scene_from_template.py
  → STAGE E: spider/process_datasets/core4d.py  (写 trajectory_kinematic.npz)
  → SPIDER MJWP sampling MPC
```

关键洞察：STAGE B 的 wrist 唯一信号是 `global_joint_positions[i, 20:21, :]`。**没有面监督，没有 wrist-vs-object-SDF cost，没有 pelvis floor**。SPIDER 端要把所有这些不可行 burden 自己扛——E082-E088 fail 的根本机制。

### 3.2 失败模式归因到 STAGE

| 失败模式 | 注入步骤 | 机制 |
|---|---|---|
| hand-in-box (R 33% INSIDE) | STAGE B | 没有 obj-collision cost on wrist；Laplacian 只优化形变，源 wrist 在 box 内时 IK 跟着进去 |
| wrist 在 pelvis 下方 25 cm | STAGE A→B | smpl_scale 是单一标量，"人蹲箱下"几何被等比缩到"机器人蹲箱下" |
| wrong contact face | STAGE A（无面监督）+ STAGE B（无面 cost）| wrist 是单点无法表达"贴哪一面" |
| pelvis 塌（<0.55 m）| STAGE A（mocap 蹲）+ STAGE B（无 pelvis floor）| 没有任何项告诉 retargeter "G1 必须直立搬箱"|
| 全局 smpl_scale 错配 | STAGE B init | G1 与 SMPL-X 的 arm/height 比例不同；按身高缩臂太长 |
| `anchor_face_review=true` 反复 | spider 下游面选择不稳 | wrist 在面交界处时面选择 frame-to-frame 翻转 |
| CVXPY infeasible | STAGE B 硬约束冲突 | 极端帧上 joint limit + foot-stick + 非穿透不可同时满足 |

### 3.3 Top-3 推荐路线（执行优先级）

| 排名 | 角度 | Impact | Feasibility | Score | 改动 |
|---|---|---:|---:|---:|---|
| **1** | **A2 + A8 合并：pre-IK 手腕投影 + raw mocap 面标签** | 5 | 4 | 20 | OmniRetarget 输入 |
| **2** | **A1：IK 求解器内 object-SDF wrist 软/硬约束** | 5 | 4 | 20 | OmniRetarget |
| **3** | **A4：pelvis-height floor 软约束** | 4 | 5 | 20 | OmniRetarget |
| 4 | A3：per-limb scale 校准 | 5 | 3 | 15 | OmniRetarget |
| 5 | A7c：用 E089 gate 自动 route 单/双 G1 | 3 | 5 | 15 | OmniRetarget 输入 |
| 6 | A9：最小 wrist-AABB 投影（A1 廉价版）| 3 | 5 | 15 | OmniRetarget 输入 |
| 7 | A6：pipeline 内嵌物理 MPC refinement | 5 | 2 | 10 | OmniRetarget |
| 8-10 | A10 时间平滑 / A13 课程激活 / A5 reachability map | | | | |
| 11 | A12：intent 重写变体（top-hold ↔ side-grip ↔ under-grip）| 3 | 2 | 6 | OmniRetarget 输入 + label |
| 12 | A11：学习式 retargeter | 5 | 1 | 5 | **DEFER** —— PASS pool 太小 |

#### Top-1 / 2 / 3 的成功判定（必须可证伪，照搬 03 §4）

- **Top-1（A2+A8）**：geometric: ≥10/13 D003 case PASS（修复后的世界向上面感知 gate）；box023_p2 守门保持；**box025_p2 面投票结果与侧抓 intent 一致（即不投到 +z 顶面）**；**Box021 IK FK INSIDE box 占比从 v1 §3.3 的 33% 降到 < 5%（B6 wrist 内陷假象部分被消除）**。SPIDER 24-step CEM on `18029_p2`: head/upper/floor ≤ 10/10/5%。**首要补做 v1 §6 没做的 B2/B3 干净 A/B**。
- **Top-2（A1）**：post-retarget wrist-inside-box = 0% on 13 D003 + box023；CVXPY infeasible-frame frac ≤ 5%；stack on Top-1 后 SPIDER full CEM `035_p2_btop` pelvis_min ≥ 0.45（今天 0.134）。
- **Top-3（A4）**：post-retarget pelvis_min ≥ 0.50 on ≥10/13；stack on Top-1+2 后 SPIDER full CEM pelvis_min ≥ 0.55；box023_p2 守门保持 WORK。

> 03 新增 **Angle 2.5（quat 偏离 identity 普查）** 作为防退化前置：在 STAGE A 写 fingertip 投影前，先对所有 box 跑 `obj quat` vs identity 偏角统计，>30° 的 case（box021、Box022 部分）必须用 raw mocap 面而非世界 up 决定方向。

Top-3 全部做完仍 pelvis < 0.55 → 启动 **A3（per-limb scale）**，因为它从几何根因解释"为什么 G1 不得不蹲"。

### 3.4 实施前必须先答的 6 个未知项（§5）

最便宜的两条（< 1 小时）应在写代码前做：

- **#3** per-frame per-link IK residual 日志（30 LOC）—— 决定"infeasible 是均匀差还是集中在几帧"，影响 A1 是否需要 slack。
- **#5** Phase 4 `enable_contact_preservation` 打开后看 gate（5 LOC + 重跑 STAGE B）—— 如果已经部分解决 wrist-inside-box，A1 可大幅简化。

其余 4 个（reachability-aware？smpl_scale 与对象 mesh 的交互？base placement 是否 binding？per-frame residual 分布？）见 §5。

---

## Part 4 — 立刻应做（按风险/收益排）

1. **修 face-selection B1 + B2 + B3 同步**（半天）—— 三者必须一起改（B1 改完 B2/B3 立即变 wrong-target）。是 v1 §3.2/§3.5 文字描述的实际错误的源头，也是 9/13 `anchor_face_review=true` flag 的根因；E082-E097 阶段虽然没用 weld（影响降级为统计污染），但任何对面的诊断解读都受其污染。
2. **修 B4：标注 `contact_pos` 是 FK 不是 raw mocap**（半小时）—— `spider/process_datasets/core4d.py:128` 加 deprecation comment + 同步在 v1 诊断报告 §3.2/§3.5 加 errata 引用 B4/B6。
3. **启用 `--include_fingertip_centers` 并写 5 指尖投票脚本（B6 第一步）**（1 工程日）—— 这是根治 B6 最小成本一步：先在 STAGE A 让 fingertip 信号流进 OmniRetarget，再写 face/contact 用 raw 指尖投票的 helper。不需要改 SPIDER reward 即可获得诊断价值（先用来重新审计 box021/Box025 的真实接触面）。
4. **加 pelvis-collapse detector + lie-on-box detector 到 CEM gate**（半天）—— `pelvis_min_world_z < 0.40` 或 `pelvis_pitch > 60°` 或 `torso_world_z - box_top_z < 0.10` 直接 FAIL，否则 E094 C2 "趴箱"那类失败模式仍会被错判为 WORK。
5. **跑 v1 §6 真正的 B2/B3 干净 A/B**（1 GPU 日）—— 同 case `18029_p2`、同 reward stack、只切 target source（E085 raw vs B6 修复后的 fingertip-vote 外部 NPZ）。这是把"H1 verified" 从 directional 变成 conclusive 的最小一步。
6. **写 OmniRetarget Top-1（A2+A8）**（2-3 工程日）—— 当前最高 ROI；修好后能给 SPIDER 提供一批新的 PASS case，反向减小 A11（学习式 retargeter）所需数据缺口。
7. **跑 Angle 2.5 quat 偏离 identity 普查**（1 小时脚本）—— 在投资 Top-1 投影前，先列出所有 `obj quat` 偏角 > 30° 的 case；这些 case 不能用 world-up 投影。
8. **找出 / commit `data_construction_v2/visualizations/raw_contact/` 的生成脚本** —— 视觉评审依赖的 PNG 是诊断证据链的一环；脚本不在 repo 里就没法判定真假。修 B6 之后这些 PNG 应同时叠加 raw 指尖位置。
9. **冻结"E089/E090 多套 gate 并列展示"的口径**：把"yield"按 gate-version 分桶，否则 TRACKER 的 18-22 行号比较是无意义的。

---

## Part 5 — 一些值得记下的、并非问题但需要注意的发现

- box004 三 case (083_p2 / 083_p1 / 082_p1) 是当前 SPIDER 端**唯一稳健正样集合**（E092/E094/E096/E096b 4 次独立确认；mask on/off 一致）。要 RL 接管时应以 E096b 作为引用。
- E088C 的 absolute clearance reward 机制本身验证为正确，**但被 CEM 用翻箱方式 hack 掉**；v1 §6 C4 说"作为默认保留"在下游 inherit 中执行了，但功劳应记给 E089 的数据侧干预，不是 reward。
- box021_person1 (E089A) 与 box021 D003 person2 是 **不同 ref motion**；不应在比较中混用。它的成功说明对象本身可达，但不说明 D003 动作可达。
- E090 揭示 `--replace_wrist_with_fingertip` 单独打开 / 关闭都有 trade-off（inside↓ 但 support↓；某些 case CVXPY infeasible），不是一个干净的全局开关——应纳入 Angle 8 的面标签流程一并解决。
- `data_construction_v2` 的 mining 目前正向 feature-based 演化（E095/E097），方向是对的；E097 的"漏 legacy outcome"是一次性 bug，已在 121 中修，可以继续延用。

---

## 附录 — 工作区

```
workspace/exp_diagnostic_v2/
├── diagnostic_v2_report.md            ← 本文件（顶层综合）
├── findings/                          ← 三份细化报告，全中文
│   ├── 01_e088_e097_audit.md          ← E088-E097 vs v1 诊断逐项核对（含 §3.10 B1 修正 + §3.11 quat 90° 几何 + §4.11 wrist→fingertip 链路验证）
│   ├── 02_face_selection_audit.md     ← 面 + 接触语义审计（B1-B5 + 新 B6 单独成 §4）
│   └── 03_omniretarget_hardbone_angles.md ← 13 个角度 + Top-3 + 新 Angle 2.5（quat 普查）
└── scripts/
    └── check_face_selection.py        ← 验证 B1 bug 的最小脚本
```
