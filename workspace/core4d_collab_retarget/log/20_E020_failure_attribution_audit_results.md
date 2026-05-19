# E020 结果：E018b failure attribution audit

日期：2026-05-20

## 初始目标

按照 `plan/21_E020_failure_attribution_audit_plan.md`，把 E018b 13 个 case 的定性失败分类升级为可复现 root-cause audit：每 case 有 S1-S5 证据列、唯一归因、诊断可视化 panel，并沉淀复用协议。

E019 unified eval 未实施；E020 按计划使用现有 E018b eval 产物推进，无阻塞。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/21_E020_failure_attribution_audit_plan.md` |
| Scripts | `workspace/core4d_collab_retarget/scripts/E020_audit/` |
| Protocol doc | `workspace/core4d_collab_retarget/docs/audit_protocol.md` |
| Results | `workspace/core4d_collab_retarget/results/E020_audit/` |
| Root-cause CSV | `workspace/core4d_collab_retarget/results/E020_audit/root_cause_attribution.csv` |
| Summary | `workspace/core4d_collab_retarget/results/E020_audit/attribution_summary.md` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E020_audit/scene_snapshot/manifest.txt` |

## 执行状态

- [x] S1 `audit_anchor_vs_raw.py`：13/13 raw SMPL-X contact centroid audit，输出 `anchor_vs_raw.csv` 和 `anchor_vs_raw.png`。
- [x] S2 `audit_ref_physics.py`：13/13 kinematic ref physics audit，输出 `ref_physics.csv` 和 timeline。
- [x] S3 `audit_mask_vs_raw.py`：13/13 current contact field vs raw per-EEF mask audit，输出 `mask_vs_raw.csv` 和 timeline。
- [x] S4 `overlay_sim_ref_curves.py`：13/13 sim/ref overlay、penetration heatmap、joint err heatmap。
- [x] S5 `decide_root_cause.py`：输出 13 行唯一 attribution。
- [x] S6 keyframes + panel：13/13 `keyframe_triplet.jpg` 和 13/13 `attribution_panel.png`。
- [x] `docs/audit_protocol.md` 已记录 E076-style 分层证据 protocol。

## 量化汇总

| root_cause | count |
|---|---:|
| `algo_stability` | 4 |
| `algo_contact` | 4 |
| `retarget_kinematic` | 2 |
| `contact_mask` | 1 |
| `raw_data` | 1 |
| `pass` | 1 |

## Case 表

| Case | E018b diag | Root cause | 关键证据 | 下一步 |
|---|---|---|---|---|
| `box021_p1` | `robot_fall_visual_fail` | `algo_stability` | object success but pelvis min `0.126m`, deep pen `53.1%`, leg `35.2%` | `E022_stability_leg_collision` |
| `box021_p2` | `robot_fall_visual_fail` | `algo_stability` | pelvis min `0.260m`, contact `10.3%`, leg `18.4%` | `E022_stability_leg_collision` |
| `box023_p1` | `contact_preservation_gap` | `contact_mask` | current contact all-on vs raw any `46.3%`, mismatch `54.4%` | `E021_per_eef_mask_repair` |
| `box023_p2` | `contact_preservation_gap` | `algo_contact` | GT anchor gate pass/object pass, but contact only `28.6%` | `E023_robot_side_contact_closure` |
| `box025_p1` | `contact_preservation_gap` | `retarget_kinematic` | ref leg/object interference `66.5%` before sim failure | `E021_ref_geometry_repair` |
| `box025_p2` | `paper_generalization_pass` | `pass` | only strict pass: contact `86.9%`, no fall/deep pen/leg artifact | none |
| `bucket001_p1` | `robot_fall_visual_fail` | `algo_stability` | object pass but sim contact `0%`, pelvis min `0.145m` | `E022_stability_leg_collision` |
| `bucket001_p2` | `robot_fall_visual_fail` | `algo_stability` | pelvis min `0.439m`, deep pen `64.6%` | `E022_stability_leg_collision` |
| `bucket005_s2_p1` | `push_or_leg_shortcut` | `algo_contact` | contact high `97.6%` but deep pen `88.2%`, leg `23.7%` | `E023_robot_side_contact_closure` |
| `bucket005_s2_p2` | `artifact_failed` | `algo_contact` | contact high `95.9%` but deep pen `74.4%` | `E023_robot_side_contact_closure` |
| `bucket007_p1` | `artifact_failed` | `algo_contact` | contact `79.0%` but deep pen `68.5%` | `E023_robot_side_contact_closure` |
| `bucket007_p2` | `contact_preservation_gap` | `retarget_kinematic` | ref hand contact `100%` but ref leg/object interference `66.3%` | `E021_ref_geometry_repair` |
| `desk021_p1` | `contact_preservation_gap` | `raw_data` | large-object/partner-side mismatch; raw anchor evidence disagrees with canonical support face | `E024_multi_agent_or_data_filter` |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 13 case 每 case 一个 root-cause CSV 行 | 通过：`root_cause_attribution.csv` 13 行 |
| C2 每 case 一个 attribution panel | 通过：`panel_index.csv` 13 行，13 个 `attribution_panel.png` |
| C3 至少 3 个 case 与 log 19 推测一致或纠正 | 通过：fall 4 case 归为 stability；`box023_p1` 归为 contact_mask；`box023_p2` 归为 algo_contact；bucket artifact/shortcut 3 case 归为 algo_contact；`box025_p2` 保持 pass |
| C4 至少 3 条下一步实验建议 | 通过：E021/E022/E023/E024 |
| C5 E076 方法论显式扩展到文档 | 通过：`docs/audit_protocol.md` |

## 结论

E020 证实 E018b 的 object-side transport 成功与完整 retarget 成功分离。13/13 object/transport pass 之后，主要失败源分成四类：

1. **robot stability**：`box021_p1/p2`、`bucket001_p1/p2`。
2. **robot-side contact/collision artifact**：`bucket005_s2_p1/p2`、`bucket007_p1`，以及 GT-anchor 成立但 contact gap 仍在的 `box023_p2`。
3. **kinematic ref geometry**：`box025_p1`、`bucket007_p2` 的 ref leg/object interference 已很高。
4. **data/contact-mask issue**：`box023_p1` all-on contact mask mismatch 最明确；`desk021_p1` 更像大物体/partner support 不适合单 G1 的数据筛选或 multi-agent 问题。

下一轮不应继续扫 E018b anchor。优先级建议：

- `E021`: 修 per-EEF contact mask 与 ref geometry，先处理 `box023_p1`、`box025_p1`、`bucket007_p2`。
- `E022`: 加 robot stability / leg-object collision control，处理四个 fall case。
- `E023`: 加 robot-side contact closure 与 collision penalty，处理 artifact/shortcut case。
- `E024`: 对 partner-heavy 或大物体 case 做 multi-agent 可行性/data filter。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 初版 S4 用 `contact_preservation_5cm >= 50%` 作为 pass，导致 `box025_p1`/`desk021_p1` 误判为 S4 pass | 1 | 改为复用 E018b 严格布尔门槛：`paper_omniretarget_contact_preservation_ok`、deep-penetration ok、floor/leg ok |
