# 技术报告：CORE4D 双人协作动力学重定向 — Outline & Skeleton (v0)

> 状态：骨架草稿。每节列出 (a) 要点、(b) 素材源 `file:line`、(c) 必备图表 placeholder。
> 上游计划：`plan/23_tech_report_plan.md`
> 中文 v0.5 全文草稿：`report/01_v0.5_draft.md`（subagent 扩写中）
> 数据可用性：E018 (2 case) ✓、E014 (6 case) ✓、E018b (13 case) ✓ 全部就绪

---

## 1. 引言 / Introduction

### 1.1 单人物理重定向的成熟与双人协作的空白

**要点**
- 物理一致 retargeting（SPIDER / DynaRetarget）在单人 manipulation / 单 humanoid 已经稳定 work
- 但 humanoid + 大物体 + 双人协作（CORE4D 类）下，单人化会让 CEM 把"另一端人类的支撑"伪装成"腿推 + 旋转 shortcut"
- 我们的工作专门攻这个 gap

**素材**
- SPIDER 已能做什么：`paper/Pan 等 - 2026 - SPIDER ...pdf`，`README.md`，`paper_notes/01_E001_literature_synthesis.md:9-24`
- 单人 spider 在 CORE4D 上的失败证据：`workspace/core4d_collab_retarget/log/02:1`（E002 main 0.703m / guard 0.830m），`docs/01_direction_review_2026-05-18.md:12-15`

### 1.2 研究问题与贡献摘要

**Research Question**：能否在不依赖 `scene_act` 6-DoF object actuator 的前提下（即 true freejoint），用单 G1 humanoid + 虚拟 partner，对 CORE4D 双人协作数据做物理一致的动力学重定向，并保持 object 跟踪精度与 E081 单人 baseline 同级？

**贡献**（详见 §4 / §5）
1. 把 SPIDER 从单人 manipulation 扩展到 CORE4D 双人，并坚持 true-freejoint object
2. 诊断并定性证明 spring/force coupling 的结构性 lag（E004-E012, τ=√(m/k)）
3. **COLA-B soft-weld 范式（核心贡献）**：用 kinematic support + MuJoCo weld equality 替代力反馈
4. Canonical Support Proxy Anchor：把 E014 手工 anchor 自动化成 `face center + 0.62·half_z` 规则
5. Paper-aligned 分层评测协议

---

## 2. 相关工作 / Related Work

### 2.1 物理一致重定向
- **SPIDER** (Pan et al. 2026)：GPU 并行 CEM + virtual contact guidance — 单人 manipulation。`paper_notes/01:9-24`
- **DynaRetarget** (Dhedin et al. 2026)：sampling-based trajectory opt + short-horizon shooting — 在 long-horizon contact 切换上易"早期错一步、后续不可恢复"。`paper_notes/01:33-51`

### 2.2 运动学重定向
- **OmniRetarget**：interaction-mesh SOCP — 我们的运动学基线（参考 holosoma v1）。`holosoma/workspace/v1/README.md:8-34`

### 2.3 双人 / 协作控制
- **COLA / Du 2025**：support body + 6-DoF joint coupling，避免在 COM 上施加 ref-tracking force。`paper_notes/02_E005_du2025_cola_virtual_force_analysis.md`
- **It Takes Two / Harmanoid** (Liu et al. 2025)：partner 几何/接触/相对 root pose 必须进入优化。`paper_notes/01:54-74`

---

## 3. 数据与运动学预处理 / Data & Kinematic Preprocessing

### 3.1 CORE4D 数据特性
- 真实双人 mocap，每 demo 有 person1/person2 完整 SMPL-X + 物体 6-DoF
- 6 类物体（chair/desk/board/box/bucket/stick），箱体最长边 0.4-1.0m，超出单 G1 末端工作空间（`holosoma/workspace/v1/README.md:107-119`）
- 接触语义复杂：手 + 前臂 + 偶尔躯干

### 3.2 基于 OmniRetarget 的 CORE4D 适配（Phase 4a-4d）
1. CORE4D Y-up → OmniRetarget Z-up（`convert_core4d_to_omniretarget.py:13-50`）
2. 约束松弛 + 4 级自适应回退（成功率 45.8% → 100%）
3. 高度门控接触检测：`contact = (|v_xy|<thresh) ∧ (z_foot<z_min+0.02)` + 中值滤波
4. 显式手-物接触保持代价：`C_contact = w·Σ max(0, ‖p_wrist - p_obj‖ - d_ref)`
5. 凹形物体自适应点云 + 容差收紧

输出：OmniRetarget 风格 `(T, nq)` 轨迹，作为动力学 pipeline 初始化与 reference。

[**Fig.1 placeholder: Pipeline overview — CORE4D raw → OmniRetarget kin → SPIDER dyn (COLA-B)**]

---

## 4. 方法 / Method

### 4.1 问题设定
- True-freejoint object（`nq_obj=7`）
- 单 G1 humanoid（`nu=29`）
- 虚拟 partner（不消耗 actuator，但提供 object 端支撑约束）
- CEM-based sampling MPC

### 4.2 为什么 spring/force coupling 不够 — 结构性 lag 诊断
**经验**：E004-E012 七轮 sweep（COM gravity/spring/hold-contact/dual-point/contact-pad），best `E011_box025_p2_com_xyz_k100` obj `0.340/0.673m`，仍是 E081 的 2.4×（`docs/01_direction_review_2026-05-18.md:83-105`）

**物理推导**：spring time constant τ=√(m/k)。要把 lag 压到 5cm 需 k≥320 N/m，但触发 robot hand 脱开（E009）或数值爆解（E012/E015）。

**结论**：单点 off-COM wrench 必然 rotation shortcut；spring/force 范式有天然上限。

[**Fig.2 placeholder: E011 obj-pos vs time, sim vs ref, with spring-lag annotation**]

### 4.3 COLA-B：kinematic support + soft weld 位置约束
- **核心思想**：把 partner-object coupling 从 *force* 换成 *position constraint*
- **实现**：kinematic support body（mocap，不进 nq）+ 与 object 间 MuJoCo `weld` equality with soft `solref/solimp`；non-COM anchor，不打 direct wrench（`log/14:106`）
- **关键参数**：`solref="0.02 1"`、`solimp="0.95 0.99 0.001"`、anchor 在 object 表面 face 上

**结果**：E014 6/6 freejoint parity ok，4/4 main 过 E013 soft target；**best `t02` obj `0.056/0.087m`**，hand 86.7%、floor 51.4%、leg 0%、rot 2.2°（`log/14:69-74`）。注意 E081 baseline 用的是 `scene_act` 6-DoF object actuator + contact guidance（`nq_obj=6`，**不是 freejoint**），其 obj `0.143/0.271m` 是在 object 无真实惯性约束下取得的；**E014 在更严格的 true-freejoint 设定（`nq_obj=7`）下，把 object 跟踪精度做到 E081 的 ~2×**。

[**Fig.3 placeholder: COLA-B geometry — support body + weld equality + anchor on object face**]
[**Fig.4 placeholder: E014 vs E081 side-by-side keyframes** — 用 `results/E014/keyframes_skill/E014_box025_p2_jointB_t02_sheet.jpg`]

### 4.4 Canonical Support Proxy Anchor
- **问题**：E014 anchor `[0,0.38,0.30]` 是手工 per-case；E016 自动化用 `mask_active_ref_palm_centroid_surface_clamp` 在 `box023_p2` 错配（`+X` raw vs `+Y` predicted）
- **规则**：`if face is ±X: [±half_x, 0, 0.62·half_z]`；同理对 ±Y / ±Z
- **验证**：E018 GT gate `box023_p2` anchor dist `1.17cm`、`box025_p2` `0.94cm`（`log/18:9-15`）
- **泛化**：E018b 13/13 case anchor 生成成功，2/2 GT case dist <1.2cm

[**Fig.5 placeholder: E016 mask-derived anchor vs E018 canonical anchor on box023/box025**]

### 4.5 分层评测协议
三层 gate：
1. **Object-side**（SPIDER Table 4 Pos/Ori Err、DynaRetarget success、carry progress、transport success）
2. **Robot-side**（OmniRetarget contact preservation、deep penetration、leg interference）
3. **Visual stability**（pelvis_z_min、fall detection）

明确提出："object-only success ≠ 完整 retargeting 成功"（`log/19:43-50`）

---

## 5. 实验 / Experiments

### 5.1 实验设置
- 硬件：本地 RTX 5090 + 远程 2× RTX 6000 Ada；CEM `num_samples=1024`, `opt_steps=32` (full), `opt_steps=4` (smoke)
- Case split：见 `EXPERIMENT_TRACKER.md`
- 评测口径：见 §4.5 + `paper_metrics.py`

### 5.2 关键节点对照：E081 vs E014 vs E018b

[**Tab.2 — 三节点 metrics 对照** — 数字见 `plan/23_*.md` §4，可直接搬过来]

要点：E081 是 `scene_act` actuator-guided（`nq_obj=6`，object 无真实惯性约束），E014 / E018b 才是 true freejoint（`nq_obj=7`，object 完全由物理约束 + robot 接触 + virtual partner 决定）。E014 在更严格的 freejoint 设定下做到 obj `0.056/0.087m`，比 E081 actuator-guided `0.143/0.271m` 提升 ~2×，且 leg interference 0%、rot 2.2° — 这是双重严格的成功。

### 5.3 13-case 泛化结果与失败模式

[**Tab.3 — E018b 13 case 完整 paper 指标表** — 数字见 `log/19:67-83`]

**Aggregate**：13/13 SPIDER/Dyna/transport object success；mean Epos `0.054m`、Erot `5.22°`；**contact preservation ok 5/13、deep penetration ok 7/13、robot upright 9/13、strict generalization 1/13**。

**失败模式分布**：
- robot_fall 4/13（box021_p1/p2, bucket001_p1/p2）
- contact_preservation_gap 5/13（box023_p1/p2, box025_p1, bucket007_p2, desk021_p1）
- artifact_failed 2/13（bucket005_s2_p2, bucket007_p1）
- push_or_leg_shortcut 1/13（bucket005_s2_p1）
- pass 1/13（box025_p2）

[**Tab.4 — 失败模式归因表** — 由 E020 强化后回灌]
[**Fig.6 placeholder: 13-case montage** — 待 E018b 数据恢复]

### 5.4 消融
- **E013 oracle**：object 完全按 ref kinematic override 时 obj `0.011/0.038m`（`log/13`）— 证 ref 本身可达，是上限
- **E015 dynamic support + PD**：升级到 COLA A+B 反而 obj `0.311/0.418m`、rot `50.3°`、PD clamp 250N、2/3 数值不稳（`log/15:7-9`）— B-only 已足够，A 引入开销
- **E016 anchor-from-mask**：自动 anchor 用 `mask_active_palm_centroid` 在 `box023_p2` 错配，failure 凸显 E018 canonical 规则的必要性

---

## 6. 讨论 / Discussion

### 6.1 object-side success ≠ 完整 retargeting 成功
- 13/13 object 跟踪通过、5/13 contact、1/13 strict pass — 这是诚实定位
- 当前 CEM 在 weld 约束下学到 "靠 weld 就行，手不必贴紧"，是 robot-side reward 设计缺陷
- 4 个 fall case 暴露 spider framework 缺少 fall gate / leg collision penalty / partner-reaction 建模

### 6.2 下一步
- **Robot-side stability**：fall gate、leg collision penalty、stance reward（→ E022+）
- **Contact 闭环**：hold-contact reward 重新评估，或上游 mask 修正（→ E021/E022）
- **Multi-agent**：bucket 类 case 单 G1 本质不可行，需要 dual-G1 协作建模
- **RL prior 衔接**：把 work 的 case（box025_p2 等）导给 holosoma RL（详见 `plan/22_E021_*.md`）

---

## 7. 结论与未来工作 / Conclusion & Future Work

### 7.1 结论
- 在 CORE4D 真 freejoint 双人协作设定上，**COLA-B soft-weld + canonical anchor 是首个做到 object-side 跨 13 case 全过、且单 case object 精度逼近单人 actuator-guided baseline 2×** 的方案
- 同时诚实揭示：完整双人协作 retargeting 仍受限于 robot-side stability 与 contact preservation

### 7.2 未来工作
- 修复 E018b 失败 case（fall / contact gap） — 见 `plan/21_E020_*`
- 跨方法对比（spider physical vs OmniRetarget kin） — 见 `plan/20_E019_*`
- RL prior 端到端验证 — 见 `plan/22_E021_*`
- Multi-agent partner（bucket 类） — 全新方向

---

## 附录建议

- A. 完整实验链 tracker（`EXPERIMENT_TRACKER.md`）摘录
- B. 所有 18 个 log 的一句话索引
- C. 主要配置 / Hydra override（COLA-B 关键字段）
- D. Reproducibility：commit hash + scene snapshot + manifest 路径

---

## TODO（按 plan/23）

- [ ] Fig.1 / Fig.2 / Fig.3 schematic 绘制
- [ ] E018b 数据恢复 → Fig.6 / Tab.3 落实
- [ ] E019 跑出 OmniRetarget kin-only 对照 → Tab.5
- [ ] E020 归因 → Tab.4 强化
- [ ] 起草中文 v1 全文
