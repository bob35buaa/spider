# 计划：CORE4D 双人协作动力学重定向 — 技术报告（task_afterE018 §3）

> 上游：`task_afterE018.md` §3 — 总结 `workspace/core4d`（主要 E081 及其继承之前的实验）+ `workspace/core4d_collab_retarget`，写一份详细技术报告：背景 / 方法 / 结果。
> 输出位置：`workspace/core4d_collab_retarget/report/`
> 关联：本计划不分配 E0NN 编号（不是物理实验），但建议与 E019/E020 并行推进 — E019/E020 产出新指标和归因后回灌报告。

---

## 0. 关键事实

1. **故事主轴**：单人 baseline 失败 → spring 范式诊断性证伪 → COLA-B 位置约束突破 → canonical anchor 泛化 → 暴露 robot-side gap。
2. **核心 selling 数字**：E014 obj `0.056/0.087m` vs E081 `0.143/0.271m`，**E014 在更严格的 true-freejoint 设定下，仍把 object 跟踪精度做到 E081 (actuator-guided) 的 ~2×**。
3. **设定区分（务必清楚）**：
   - **E081 是 `scene_act` 6-DoF object actuator + contact guidance，不是 freejoint**（`nq_obj=6` slide+euler）。E081 的"成功"很大程度依赖 actuator 给 object 注入 inertial-free guidance（参见 `EXPERIMENT_TRACKER.md:41` 的 E001 审计结论）。
   - **E014 / E018 / E018b 才是 true freejoint**（`nq_obj=7`，由 MJCF `<freejoint/>` 给出，无 object actuator）。
   - 所以"E014 在 true-freejoint 下做到 obj `0.056m`"是双重严格的：(a) 物理设定更严格；(b) 数字仍比 E081 好 2×。
4. **当前 strict generalization 仅 1/13**，是论文的诚实弱点；建议把"object-side 成功 / robot-side 仍 open"做成 Discussion 核心观点，反而显得严谨。
5. **运动学是 OmniRetarget**，动力学是 SPIDER；明确分工和适配在 method 章节。
6. **数据已就绪**：E018b 13 NPZ + 13 MP4 + aggregate + comparison.csv 全在 `results/E018b/`，Fig.6 / Tab.3 不再阻塞。

---

## 1. 报告章节大纲（中英对照）

```
1. 引言 / Introduction
   1.1 单人物理重定向的成熟与双人协作的空白
       Single-Agent Maturity vs. Multi-Agent Gap
   1.2 研究问题与贡献摘要
       Research Question & Contribution Summary

2. 相关工作 / Related Work
   2.1 物理一致重定向: SPIDER / DynaRetarget
       Physics-Consistent Retargeting
   2.2 运动学重定向: OmniRetarget
       Kinematic Retargeting
   2.3 双人 / 协作控制: COLA, It Takes Two / Harmanoid
       Collaborative Whole-Body Control

3. 数据与运动学预处理 / Data & Kinematic Preprocessing
   3.1 CORE4D 数据特性
   3.2 基于 OmniRetarget 的 CORE4D 适配 (4a–4d)

4. 方法 / Method
   4.1 问题设定：true-freejoint object + 单 humanoid + 虚拟 partner
   4.2 为什么 spring/force coupling 不够 — 结构性 lag 诊断
   4.3 COLA-B：kinematic support + soft weld 位置约束
   4.4 Canonical Support Proxy Anchor
   4.5 分层评测协议

5. 实验 / Experiments
   5.1 实验设置（硬件、CEM 预算、case split）
   5.2 关键节点对照：E081 vs E014 vs E018b
   5.3 13-case 泛化结果与失败模式
   5.4 消融：E013 oracle / E015 dynamic A / E016 anchor-from-mask

6. 讨论 / Discussion
   6.1 object-side success ≠ 完整 retargeting 成功
   6.2 下一步：robot-side stability / contact / RL prior 衔接

7. 结论与未来工作 / Conclusion & Future Work
```

---

## 2. 必备图表清单

| # | 名称 | 现状 | Action |
|---|---|---|---|
| Fig.1 | Pipeline overview（OmniRetarget kin → SPIDER dyn with COLA-B） | 无 | **新做** schematic 一张 |
| Fig.2 | spring-lag 物理示意 + E011 obj-vs-time 曲线 | `results/E011/` 有 NPZ | **新做曲线** |
| Fig.3 | E014 anchor 几何示意（support body 在 object face） | 无 | **新做** schematic |
| Fig.4 | E014 vs E081 keyframe 并排 | `results/E014/keyframes_skill/*.jpg` 已有 | 直接用 |
| Fig.5 | E016 anchor (mask-derived) vs E018 canonical anchor | `results/E016/visual/*.mp4` + `results/E018/anchor_visual/` 已有 | 拼接 |
| Fig.6 | E018b 13-case montage（online MP4 截帧） | `results/E018b/E018b_*_canonical_t02.mp4` 13 个已就绪 | ffmpeg 抽帧 + 拼图即可 |
| Tab.1 | 指标演进表（E001 → E018b, 8-10 行） | `EXPERIMENT_TRACKER.md:39-62` 完整 | 直接用 |
| Tab.2 | E081 / E014 / E018b 三节点对照 | 数字已抽出，见下文 | 直接用 |
| Tab.3 | E018b 13-case 完整 paper 指标 | log 已有 | 直接用，但需视频补齐 |
| Tab.4 | 失败模式归因表（4 类 × case） | log 已有；E020 会强化 | 用 + E020 增强 |
| Tab.5（可选）| OmniRetarget kinematic-only baseline 对照 | `holosoma/workspace/v1/scripts/eval_paper_metrics.py` 可跑 | **新做**（E019 unified_eval 副产物） |

---

## 3. 章节素材索引（file:line 风格）

### 1. 引言 / 相关工作
- SPIDER paper：`paper/Pan 等 - 2026 - SPIDER ...pdf`
- DynaRetarget paper：`paper/Dhedin 等 - 2026 - DynaRetarget ...pdf`
- COLA / It Takes Two：`paper/Liu 等 - 2025 - It Takes Two ...pdf`
- 已有文献综合笔记：`paper_notes/01_E001_literature_synthesis.md`, `02_E005_du2025_cola_virtual_force_analysis.md`
- 项目 README：`README.md`、`CLAUDE.md`

### 2. 数据与运动学
- OmniRetarget 适配公式与四个 Phase 4 改动：`holosoma/workspace/v1/README.md:8-106`
- CORE4D 转换器：`holosoma/workspace/v1/scripts/convert_core4d_to_omniretarget.py:1-60`

### 3. 方法
- 问题设定 + true-freejoint vs scene_act：`workspace/core4d_collab_retarget/log/02:1`，`docs/01_direction_review_2026-05-18.md:12-15`
- spring lag 诊断（τ=√(m/k) 推导）：`docs/01_direction_review_2026-05-18.md:83-105`
- E014 COLA-B 实现：`workspace/core4d_collab_retarget/log/14_E014_cola_b_kinematic_weld_results.md`
- canonical anchor 规则：`workspace/core4d_collab_retarget/log/18_E018_canonical_support_proxy_anchor_results.md:9-15`
- 分层评测协议：`workspace/core4d_collab_retarget/scripts/eval/paper_metrics.py:395`

### 4. 实验
- E081 baseline 数字：`workspace/core4d/log/102_E081_leg_object_collision_results.md:189`
- E014 / E015 / E016 / E018 / E018b 详细数字：`log/14`-`log/19`
- 实验链 tracker：`EXPERIMENT_TRACKER.md`

### 5. 讨论
- "object-side 成功 ≠ 完整成功"：`log/19:43-50`
- 4 类失败模式：`log/19:67-83`，并由 E020 强化

---

## 4. 三节点关键 metrics 对照（已抽数字）

| 指标 | E081 main `box025_p2_legobj` (scene_act, 单 case) | E014 main `box025_p2_jointB_t02` (true freejoint, 单 case) | E018b main `box025_p2` (canonical, 13 case 之一) |
|---|---:|---:|---:|
| Object 控制范式 | scene_act 6-DoF actuator + contact guidance | kinematic support body + soft weld | 同 E014，anchor 改 canonical face-center |
| nq_obj | 6 (slide+euler) | 7 (freejoint) | 7 (freejoint) |
| obj mean / max (m) | `0.143 / 0.271` | **`0.056 / 0.087`** | `0.056 / —` (Epos 0.056m, Erot 1.9°) |
| Hand contact % | `89.0` | `86.7` | (contact preservation 5cm `86.9%`) |
| Floor contact % | `59.5` | `51.4` | — |
| Leg interference % | `7.5` | **`0.0`** | `0.0` |
| Object rot err (°) | — | `2.2` | `1.9` |
| xy ratio (transport) | — | `0.999` | (transport success ✓) |
| 视觉 fall | no | no | no (pelvis min `0.760m`) |

---

## 5. 实施步骤

| Step | 内容 | 产出 |
|---|---|---|
| 1 | 写 `report/00_outline.md`（本计划同时落盘） | outline + 数字 + 素材索引 |
| 2 | 准备 Fig.1 / Fig.2 / Fig.3 schematic（draw.io / matplotlib） | 3 张矢量 + PNG |
| 3 | 等待 / 推动 E019 unified_eval 输出，补 Tab.5（OmniRetarget kin 对照） | `tables/table_all_methods.xlsx` 一列 |
| 4 | 等待 / 推动 E018b 数据恢复，做 Fig.6 13-case montage | `report/figs/fig6_e018b_montage.jpg` |
| 5 | 等待 / 推动 E020 归因，回灌 Tab.4 强化失败模式 | `report/tables/tab4_failure_modes.md` |
| 6 | 按 outline 分章节起草（建议先英文骨架后翻中文，或并行）| `report/01_intro.md` … `report/07_conclusion.md` |
| 7 | 整合成 `report/spider_core4d_collab_retarget_v1.md` + 同名 `.pdf`（用 pandoc / typst） | 一份完整 v1 |
| 8 | 内部 review + 校对 | v1 → v1.1 |

---

## 6. 篇幅建议

- **总长**：4500-6500 字（中文）— 正文 12-16 页 + 附录
- **图**：6-8 张（Fig.1-6 必备 + 2 张消融示意备选）
- **表**：4-5 张（Tab.1-4 必备 + Tab.5 OmniRetarget 对照）
- **比例**：背景+相关工作 ~20%、方法 ~30%、实验+结果 ~35%、讨论+结论 ~15%

---

## 7. 已知前置工作

1. **E019 unified_eval**：影响 Tab.5（OmniRetarget kin-only 对照）。本轮与报告 v0.5 并行做。
2. **E020 归因**：影响 Discussion 6.1 的可信度（4 类失败模式如果有量化归因，论证强很多）。本轮不做，留 v1。
3. **统计严格性**：当前多数指标是 single rollout，按 `.claude/rules/experiment.md:48-58` 应给 mean ± std + worst case（跑 ≥3 seeds）。可在 final 报告前补一轮。

---

## 8. 成功标准

- ✅ outline + 数字 + 素材索引全部完整（**本次先交付这部分**）
- ✅ E019/E020 数据到位后，正文 v1 在 1 周内成稿
- ✅ Tab.2 三节点对照 + Tab.3 13-case 表 + Fig.4/5 E014 vs E018 anchor 三件套至少先出
- ✅ Discussion 明确点出 "object-side 成功 / robot-side 仍 open" 的诚实定位
