# After-E018 计划索引

> 上游：`workspace/core4d_collab_retarget/task_afterE018.md`
> 当前实验编号到 E018b，本批次新计划编号 E019 / E020 / E021 + 一份独立报告计划。
> 编写时间：2026-05-20

---

## 总体路线

```
                   ┌───────────────────────────────────────┐
                   │   task_afterE018.md 4 个收尾子任务     │
                   └───────────────────────────────────────┘
                                     │
        ┌────────────────┬───────────┴──────────┬──────────────────┐
        ▼                ▼                      ▼                  ▼
    §1 全面评测       §2 失败归因           §3 技术报告        §4 RL 导出
    plan/20_E019      plan/21_E020          plan/23 + report/   plan/22_E021
        │                │                      │                  │
        │                │                      │                  │
        ▼                ▼                      ▼                  ▼
  unified_eval.py    E020_audit/         report/spider_     scripts/export/
  paper_metrics      root_cause CSV      core4d_collab_v1   (放 holosoma 侧)
  docs/eval_metrics  attribution panel
        │                │                      ▲                  │
        └──────► 回灌 ───┴──────► 数字/归因 ────┘                  │
                                                                   │
                                            box025_p2 → holosoma RL ◀
```

---

## 4 个独立计划

| Plan | 文件 | 任务 | 阻塞 | 估时 |
|---|---|---|---|---|
| **E019** | `plan/20_E019_unified_eval_framework_plan.md` | 统一评测脚本 + SPIDER T4 + OmniRetarget T2 + xlsx + `docs/eval_metrics.md` | 无 | 2.5d |
| **E020** | `plan/21_E020_failure_attribution_audit_plan.md` | 13 case root_cause 归因 + 6 步 protocol + diagnostic visualizations | 无 | 2.0d |
| **E021** | `plan/22_E021_holosoma_rl_export_plan.md` | spider E018b → holosoma RL 训练格式（shim + batch driver + 首批 2-3 case 验证） | holosoma `models/{obj}/` 资产核对 | 2.0d |
| **技术报告** | `plan/23_tech_report_plan.md` + `report/00_outline.md` + `report/01_v0.5_draft.md` | 背景 / 方法 / 结果三段式技术报告（中文） | E019/E020 输出回灌 v1 | 1 周 |

**用户已决策（2026-05-20）**：本轮先做 E019 完整 P0 + 报告中文 v0.5，E020/E021 留下一轮。

---

## 数据就绪状态（2026-05-20 已更新）

**E018b 13 NPZ + 13 MP4 + aggregate_summary.json + comparison.csv 全部就绪**，在 `results/E018b/`。
- 4 个计划全部可立即推进，无阻塞。
- 仅 `online_video/` 子目录视情况补齐（13 个 root MP4 已有，能用）。

---

## 推进顺序建议

不要严格串行 — 4 个计划有大量并行机会：

| Week 1 | E019 step 1-4（adapter + SPIDER T4 + penetration） | + E020 step 1-3（S1/S2/S3 不依赖 sim NPZ） | + E021 step 1-2（shim + 资产核对） | + 报告 outline 落盘（本次已完成） |
| Week 2 | E019 step 5-10（kin 对照 + xlsx + docs） | + E020 step 4-7（S4/S5/S6 + 报告） | + E021 step 3-4（box025_p2 端到端） | + 报告 v0 框架 + Tab.2 |
| Week 3 | （等数据） | （等数据） | E021 step 5-8（13 case 全跑 + RL load） | 报告 v1 全文 |

---

## 文件清单（本批新增）

```
workspace/core4d_collab_retarget/
├── plan/
│   ├── AFTER_E018_INDEX.md                              # 本文件
│   ├── 20_E019_unified_eval_framework_plan.md
│   ├── 21_E020_failure_attribution_audit_plan.md
│   ├── 22_E021_holosoma_rl_export_plan.md
│   └── 23_tech_report_plan.md
└── report/
    └── 00_outline.md                                    # 技术报告骨架
```

---

## 与现有路线的关系

- 与 E018b 后续可能的 algorithm 实验（如 fall gate / leg collision penalty / partner-reaction modeling）**互不阻塞**：本批是"收尾 + 工具化"，不修算法
- E020 的归因输出会自然孕育下一批算法实验（暂未编号），E020 step 7 要求 "至少 3 条 actionable 下一步实验建议"
- E019 的 unified_eval 成为后续所有实验的默认评测入口

---

## 用户决策记录（2026-05-20）

1. **E018b 数据策略**：13 NPZ + 13 MP4 + aggregate 已就绪，直接用
2. **优先级**：本轮 E019 完整 P0 + 报告中文 v0.5；E020/E021 留下一轮
3. **报告语言**：中文
4. **执行方式**：立即并行实施 E019 + 报告 v0.5

## E081 vs E018b 设定澄清（务必正确表述）

- **E081 baseline**（`workspace/core4d` 工作区，单 case `box025_p2_legobj`）：使用 `scene_act` 6-DoF object actuator + contact guidance，**object 不是 freejoint**，`nq_obj=6` (slide + euler)
- **E014 / E018 / E018b**（本工作区）：**真正的 true freejoint**，`nq_obj=7`（pos 3 + quat 4），由 MJCF `<freejoint/>` 给出，无 object actuator
- 因此 "E014 obj `0.056/0.087m` vs E081 `0.143/0.271m`" 的对比意义是：**我们在更严格的物理设定下（true freejoint），仍把 object 跟踪精度做到 E081 (actuator-guided) 的 ~2×**。任何地方提到 "freejoint 下接近 E081" 都要注明 E081 不是 freejoint。
