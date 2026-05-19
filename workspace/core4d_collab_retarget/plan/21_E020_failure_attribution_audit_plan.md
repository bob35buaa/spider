# E020 计划：E018b 失败 case 归因分析（task_afterE018 §2）

> **依赖**：E019 unified_eval（建议但非阻塞 — 即便 E019 没做完，本计划也可用现有 eval_E018b.py 产物推进）。
> 上游：`task_afterE018.md` §2 — 基于全面数值指标 + 仔细可视化（视频 + 运动曲线）找出失败 case 是数据质量问题还是算法问题。
> 关键先例：`workspace/core4d/log/97_E076_contact_source_audit.md`。

---

## 0. 关键前置事实

1. **E018b 数据已就绪**（13 NPZ + 13 MP4 + aggregate + comparison.csv），无阻塞，可直接对 13 case 跑完整 S1-S6 protocol。
2. **E018b log 记录的 13 case 分类已经定性**（log 19_E018b_…:67-83）：4 fall / 5 contact gap / 2 penetration artifact / 1 leg shortcut / 1 pass。本计划要把这个定性归因升级成**可证伪的 root-cause CSV**。
3. **E076 方法论可复用**：分层证据 + 时间对齐 — raw mocap → 几何 proxy → retargeted ref → MJWP mask 四层独立计算后对齐到同一 horizon。
4. **E081 不是 freejoint**（是 `scene_act` actuator-guided）；归因 protocol 不对比 E081，只对比 ref（kinematic retarget）和 raw（SMPL-X mocap）。

---

## 1. 目标

A. **每 case 唯一 root_cause**：`{raw_data, retarget_kinematic, contact_mask, algo_tracking, algo_contact, algo_stability}`，附 S1-S5 pass/fail 证据列。
B. **数据质量候选清单**：哪些 case 应该回到 OmniRetarget kin pipeline 修；哪些应该在 SPIDER 物理侧修；哪些是 SPIDER 框架根本不适合的（需 multi-agent 才能解）。
C. **失败模式归因 protocol**：6 步可复现 audit pipeline，未来新增 case 可直接套用。
D. **诊断可视化**：曲线 + 3D anchor 标注 + penetration heatmap + joint Err heatmap，每 case 一个 `attribution_panel.png`。

---

## 2. E018b 13 case 失败分类总览（log 已知）

| Case | diag (log) | 推测主因（待 S1-S5 验证） |
|---|---|---|
| box021_p1 | robot_fall | algo (fall gate + leg) |
| box021_p2 | robot_fall | algo (fall + contact loss) |
| box023_p1 | contact_preservation_gap | **data (E076 已证 R 手仅边界接触，mask 全 1 过强)** |
| box023_p2 | contact_preservation_gap | algo (anchor OK，contact reward 弱) |
| box025_p1 | contact_preservation_gap | algo (E017 face cancellation) |
| box025_p2 | **pass** | — |
| bucket001_p1 | robot_fall | data + algo (小物 partner 强支撑) |
| bucket001_p2 | robot_fall | algo (leg collision penalty 缺) |
| bucket005_s2_p1 | push_or_leg_shortcut | algo (reward shortcut) |
| bucket005_s2_p2 | artifact_failed | algo (face cancellation + 无碰撞 cost) |
| bucket007_p1 | artifact_failed | algo |
| bucket007_p2 | contact_preservation_gap | data (partner 强支撑) |
| desk021_p1 | contact_preservation_gap | data (desk 大、mask 全 1 过强) |

---

## 3. 归因 Protocol（受 E076 启发的 6 步）

| Step | 检查 | 数据 / 脚本 | Pass 判据 |
|---|---|---|---|
| S1 | **Anchor vs raw**：raw frame 上算 partner-hand → object surface 真实接触点分布，与 E018b canonical anchor `(face, 0.62·half_z)` 比对 | 新 `audit_anchor_vs_raw.py`，复用 E076 的 SMPL-X + KDTree | anchor 距 raw partner contact centroid < 8cm |
| S2 | **Kinematic ref 物理合理性**：在 ref 轨迹（不跑 sim）上检测 hand-object dist、object-floor、leg-object pen、pelvis_z 时间序列 | 新 `audit_ref_physics.py` 读 `trajectory_kinematic.npz` + scene XML | 无 >3cm 持续穿透；pelvis_z > 0.55；mask=1 时 hand-object < 3cm |
| S3 | **Per-EEF contact mask vs raw**：raw 2cm/3cm proxy → 当前 mask `(T,)` → 检查 "mask 全 1 但 raw 仅边界接触" 错配 | 新 `audit_mask_vs_raw.py`，输出 timeline 三联图 | mismatch < 15% 帧 |
| S4 | **Sim vs Ref 对齐**：E018b online rollout NPZ 与 ref 同时间窗叠加：object pose / pelvis / wrist / contact | 扩展 `render_E016_visuals.py` 加曲线面板 | Epos/Erot 物理合理；pelvis 不突降 |
| S5 | **Failure mode 归因决策**：S1-S4 任一 fail → 数据问题；全 pass 但 sim fail → 算法问题；S2 fail 且 raw 合理 → retarget 问题 | 新 `decide_root_cause.py` | 每 case 唯一 root_cause |
| S6 | **A/B 视觉验证**：fall / pen case 用 `/video-frames` 抽 grasp / transport / release 三联，旁标 S1-S4 数值 | 现有 `render_E018_anchor_videos.py` 扩展 | 视觉与归因一致 |

---

## 4. 新增诊断指标 / 可视化

现有 sheet 1440x480 三视图只看动作不看数值。新增：

- **时间序列叠加图**（必备）：`pelvis_z`、`object_z`、`hand-object dist (L/R)`、`mask`、`contact force proxy` —— sim 实线 + ref 虚线 + raw 点线，同一时间轴
- **3D anchor-on-object 标注图**：物体 mesh 上画 E018b canonical anchor + E014 GT anchor + raw partner contact heatmap，三色对比
- **Joint-level Err 热力图**：per-joint Epos/Erot 时间×关节矩阵，定位失败时哪个关节 blow up
- **Penetration heatmap**：mesh 顶点穿透深度时序，区分 "腿顶桶" vs "手指穿箱"
- **Mask mismatch timeline**：raw 2cm proxy vs 当前 mask vs ref geom contact 三条 0/1 timeline（仿 E076）

---

## 5. 数据质量问题先验清单

| Case | 怀疑数据问题 | 证据来源 |
|---|---|---|
| box023_p1/p2 | mask 全 1 过强，R 手仅边界接触 | E076 raw 分析 |
| bucket001_p1/p2 | bucket 小、partner 双手强支撑，单 G1 物理上无法独立完成 | E018b fall + contact 0% |
| bucket007_p2 | 同 partner-support 缺失 | contact 29.7% but object pass |
| desk021_p1 | desk 大、canonical face-center 不一定是 partner 接触点 | contact 52% / leg 0.5% / object 完美 |
| box025_p1 | E017 标 centroid_cancellation | E017 anchor_audit.csv |
| bucket005_s2_p1/p2 | 同 cancellation + `face_support_frac=0` | E017 audit |

未怀疑（数据干净）：`box021_p1/p2`、`box025_p2`、`box023_p2` — 失败多为算法侧。

---

## 6. 目录结构

```
workspace/core4d_collab_retarget/
├── plan/21_E020_failure_attribution_audit_plan.md           # 本文件
├── log/20_E020_failure_attribution_audit_results.md         # 实施时新建
├── scripts/E020_audit/
│   ├── audit_anchor_vs_raw.py        # S1
│   ├── audit_ref_physics.py          # S2
│   ├── audit_mask_vs_raw.py          # S3
│   ├── overlay_sim_ref_curves.py     # S4
│   ├── decide_root_cause.py          # S5
│   ├── render_attribution_keyframes.py  # S6
│   └── plot_attribution_panel.py     # 汇总 5-in-1 PNG
└── results/E020_audit/
    ├── per_case/{variant}/
    │   ├── anchor_vs_raw.png
    │   ├── ref_physics_timeline.png
    │   ├── mask_vs_raw_timeline.png
    │   ├── sim_ref_overlay.png
    │   ├── penetration_heatmap.png
    │   ├── joint_err_heatmap.png
    │   └── attribution_panel.png         # 6-in-1 汇总
    ├── root_cause_attribution.csv         # 13 行: case, S1-S5 pass/fail, root_cause, evidence
    ├── attribution_summary.md             # 跨 case 总表
    └── scene_snapshot/                    # 按规范快照
```

---

## 7. 实施步骤

| Step | 内容 | Verify |
|---|---|---|
| 1 | 实现 S1 audit_anchor_vs_raw，先在 box023_p2（已知 face 错→改对了）+ box025_p2（已知 pass）上验证 | 输出 anchor_vs_raw.csv，预期 box023_p2 此前 S1 fail、E018b 已 pass；box025_p2 全 pass |
| 2 | 实现 S2 audit_ref_physics，扫 13 case kinematic ref | 输出 S2 fail 集合；预期 bucket001/007 有 ref 物理问题 |
| 3 | 实现 S3 audit_mask_vs_raw，复用 E076 SMPL-X 计算 | 输出 mask mismatch 比例；预期 box023_p1 mismatch >20% |
| 4 | 实现 S4 overlay_sim_ref_curves，每 case 输出 6 曲线图 | 视觉 review 13/13 |
| 5 | 实现 S5 decide_root_cause，输出 root_cause_attribution.csv | 13 行唯一归因；与 §2 推测对照 |
| 6 | S6 关键帧 + plot_attribution_panel.py 汇总 | 13 个 attribution_panel.png |
| 7 | 写 `log/20_*.md`，按 root_cause 分组列建议下一步实验 | 至少 3 条 actionable 下一步（如 "E021: 修 box023 mask"、"E022: 加 leg collision penalty"、"E023: 多 agent partner-coupling"） |

---

## 8. 成功标准

- ✅ 13 case 每 case 一个 root_cause CSV 行 + 一个 attribution_panel.png
- ✅ 至少 3 个 case 的归因结论与 log 19 中 "推测主因" 一致或纠正
- ✅ 输出至少 3 条可执行的下一步实验建议（命名为 E021 / E022 / ...）
- ✅ E076 的方法论被显式扩展记录到 `docs/audit_protocol.md`

## 9. 已知风险

- SMPL-X 22 关节获取路径需先确认（同 E019）
- "mask 全 1 过强" 类问题修复需回到上游 OmniRetarget / contact mask 生成流程，超出本工作区范围 — 本计划只输出**证据**和**修改建议**，不修上游

## 10. 时间预算

| Phase | 内容 | 时间 |
|---|---|---|
| A | S1+S2 实现（不依赖 sim NPZ） | 0.5 day |
| B | S3 mask vs raw（需 SMPL-X 流程） | 0.5 day |
| C | S4+S5 + plot panel | 0.5 day |
| D | S6 + 13 case 全跑 + 报告 | 0.5 day |
| **合计** | | **2.0 day** |
