# E017 结果：anchor audit + face-cluster selection

日期：2026-05-19

## 初始发现记录

E014 与 E016 的关键差异已确认：

- E014 anchor 是手工指定的 object-local support point，用来隔离验证 COLA-B soft-weld 结构本身。
- E014 `box025_p2` anchor 为 `[0.0, 0.38, 0.30]`，贴近 box025 `+Y` 面且偏上。
- E014 `box023_p2` anchor 为 `[0.16, 0.0, 0.10]`，贴近 box023 `+X` 面且偏上。
- E016 只继承了 E014 的 weld 结构、`solref=0.02 1`、`solimp=0.9 0.95 0.001` 和 no-direct-wrench 口径；anchor 改为 `mask_active_ref_palm_centroid_surface_clamp` 自动推断。
- `box023_p2` 中 E016 自动 anchor 为 `[0.030, 0.157, 0.150]`，落在 `+Y` 面；E014 成功 anchor 落在 `+X` 面。

因此 E016 的结论需要拆分：

1. E014 B-only weld 结构的 object-side tracking 泛化成立：13/13 SPIDER/Dyna object success 与 transport success。
2. 当前 centroid anchor 自动选择不成立：`box023_p2` 已经暴露 face selection 错误，robot-side 姿态和接触形态明显差于 E014。

## 待执行

- 实现 E017 anchor audit / face-cluster selector。
- 对 13 个 E016 case 输出 anchor confidence 和 likely anchor-wrong 分类。
- 对疑似 anchor 错误的 subset 生成 E017 candidate scene/override，并做 quick 验证。

## Anchor Audit 结果

结果文件：

- `workspace/core4d_collab_retarget/results/E017/anchor_audit.csv`
- `workspace/core4d_collab_retarget/results/E017/anchor_method_audit.csv`
- `workspace/core4d_collab_retarget/results/E017/e016_anchor_failure_attribution.csv`
- `workspace/core4d_collab_retarget/results/E017/anchor_audit_summary.json`
- `workspace/core4d_collab_retarget/results/E017/manifest.tsv`
- `workspace/core4d_collab_retarget/results/E017/manifest_validation.tsv`

汇总：

| 类别 | 数量 | 说明 |
|------|------|------|
| `likely_anchor_wrong_manual_mismatch` | 1 | 与 E014 已验证 anchor face 明确不一致 |
| `possible_anchor_wrong_centroid_cancellation` | 2 | active palm 分布在对侧 face 双峰，centroid 被抵消到 unsupported face |
| `ambiguous_low_confidence_centroid` | 1 | centroid face support 为 0，但 E016 失败更像 robot artifact，需要验证 |
| `manual_seed_face_matches_current` | 1 | E016 与 E014 同 face，但 anchor 高度可疑 |
| `anchor_face_plausible` | 7 | 当前 anchor face 有足够 contact/mask 支撑 |
| `likely_non_anchor_robot_artifact` | 1 | E016 已有较高 contact preservation，失败更像 leg/floor shortcut |

关键 case：

| Case | E016 anchor face | Selected-person contact face distribution | 结论 | E017 candidate |
|------|------------------|---------------------------|------|----------------|
| `box023_p2` | `+Y` | `+X 51.2% / -X 48.8% / +Y 0%` | 明确 anchor 选错；E014 是 `+X` | `face_cluster=[0.153,0.089,0.115]`，`e014_seed=[0.16,0,0.10]` |
| `box025_p1` | `+Y` | `-X 53.2% / +X 46.1% / +Y 0.7%` | 疑似 centroid cancellation，但无 E014 seed 佐证 | `face_cluster=[-0.377,0.181,-0.117]` |
| `bucket005_s2_p2` | `+Y` | `+X 50.8% / -X 49.2% / +Y 0%` | 疑似 centroid cancellation；E016 artifact failed | `face_cluster=[0.158,0.093,-0.058]` |
| `bucket005_s2_p1` | `+Y` | `+X 50.8% / -X 49.2% / +Y 0%` | 低置信，但 E016 contact preservation 已很高，可能不是 anchor 主因 | `face_cluster=[0.158,0.106,0.067]` |
| `box025_p2` | `+Y` | `-X/+X` 双峰，E014 也选 `+Y` | face 对，E016 的 z `0.399` 偏高；应对照 E014 `z=0.30` | `centroid_v2=[0.006,0.378,0.305]`，`e014_seed=[0,0.38,0.30]` |

已暂停 subset 实验：用户指出应先审核再跑；远端 tmux 和本地 E017 进程已停止，停止时没有产生 E017 NPZ。

## 另一侧语义复核

用户指出 anchor 应该是“另一侧/partner support 与物体的接触点”。复查代码后结论如下：

- E016 `generate_e016_assets.py::_load_mask()` 使用 `row["person_idx"]`，`infer_support_point()` 再用同一条单人 task 的 `left_palm/right_palm` 坐标生成 centroid anchor。
- E017 auto 原实现继承同一口径，`_load_support_points()` 也是基于 `row["person_idx"]` 的 selected-person active palm points 做 face-cluster 或 z-corrected centroid。
- 因此 E016 与 E017 auto 的算法输入都不是显式的另一侧 contact，而是 `selected_person_contact_mask`。只有 E014 seed 是人工指定且已验证的 support anchor。

已修改 audit 输出，新增 counterpart-person 弱证据通道：

- `anchor_audit.csv` 新增 `anchor_algorithm_contact_side=selected_person_contact_mask`。
- 新增 `partner_source_task`、`partner_top_face`、`partner_top_face_frac`、`selected_partner_top_relation`、`current_partner_face_support_frac`、`selected_partner_face_support_frac` 等列。
- `anchor_method_audit.csv` 新增 `partner_side_status`，区分 anchor face 是否与 counterpart-person dominant contact face 一致。
- `e016_anchor_failure_attribution.csv` 新增 `possible_anchor_error_not_partner_side`，用于无 E014 GT 且 E016 face 与 counterpart-person dominant face 不一致的 case。

13-case 复核汇总：

| 项 | 结果 |
|----|------|
| Auto anchor 输入侧 | `selected_person_contact_mask` |
| selected/partner dominant face 相同 | 8 case |
| selected/partner dominant face 不同 | 3 case |
| selected/partner dominant face 对侧 | 1 case |
| 缺 counterpart task | 1 case (`desk021_p1`) |
| E014 GT 与 counterpart-person dominant face | 2/2 都不同 |

关键注意：counterpart-person mask 不能直接覆盖 E014 GT。`box023_p2` 中 counterpart-person dominant face 是 `+Y`，刚好会“支持”E016 的错误 `+Y` anchor，但 E014 已验证 GT 是 `+X`；`box025_p2` 中 counterpart-person dominant face 是 `-X`，而 E014 GT 是 `+Y`。这说明 E014 的 support anchor 是人工验证的 object-local support point，不等价于简单取另一人的 raw dominant palm face。

## 修正后的审核口径

用户明确要求：E014 的 `box023_p2` 与 `box025_p2` anchor 作为已验证 GT；E016 与 E017 auto anchor 都需要被审核。按此口径新增逐方法审核：

| Case | Method | Face | GT status / weak status | 结论 |
|------|--------|------|-------------------------|------|
| `box023_p2` | E016 centroid | `+Y` | `gt_face_mismatch`，GT dist `0.210m` | 明确 anchor 选错 |
| `box023_p2` | E017 auto | `+X` | `gt_near_with_offset`，GT dist `0.090m` | face 已修正，需验证 y offset 是否可接受 |
| `box025_p2` | E016 centroid | `+Y` | `gt_near_with_offset`，GT dist `0.099m`，主要是 z 偏高 | 可能 anchor 高度导致问题 |
| `box025_p2` | E017 auto | `+Y` | `gt_pass`，GT dist `0.008m` | 与 E014 GT 对齐 |
| `box025_p1` | E016 centroid | `+Y` | `centroid_cancellation_unsupported`，support `0.7%` | 无 GT，可能 anchor 错误 |
| `bucket005_s2_p2` | E016 centroid | `+Y` | `centroid_cancellation_unsupported`，support `0%` | 无 GT，可能 anchor 错误 |
| `bucket005_s2_p1` | E016 centroid | `+Y` | `centroid_cancellation_unsupported`，但 E016 contact preservation 高且失败类为 push/leg shortcut | 弱证据，不进第一批验证 |

E016 失败归因分层：

| 层级 | Case | 验证 variants |
|------|------|---------------|
| 明确 anchor 错误 | `box023_p2` | `E017_box023_p2_face_cluster`、`E017_box023_p2_e014_seed` |
| 可能不是 partner-side anchor | `box021_p2`、`box023_p1` | 无；E017 auto 仍是 selected-person based，不能作为 partner-side 验证 |
| 可能 anchor 错误 | `box025_p1` | `E017_box025_p1_face_cluster` |
| 可能 anchor 高度/位置偏差 | `box025_p2` | `E017_box025_p2_centroid_v2`、`E017_box025_p2_e014_seed` |
| 可能 anchor 错误 | `bucket005_s2_p2` | `E017_bucket005_s2_p2_face_cluster` |
| 弱证据，暂不跑 | `bucket005_s2_p1` | 无 |

第一批验证 manifest：`workspace/core4d_collab_retarget/results/E017/manifest_validation.tsv`。

## Algorithm Validation

运行方式：

- 本地 GPU0：`manifest_validation.tsv` 中 4 个 local variants。
- 远端 GPU0/GPU1：显式同步本地 `manifest_validation.tsv`，各跑 1 个 remote variant。
- 统一本地 eval：`logs/core4d_collab_retarget/E017/eval_validation_all.log`。

结果文件：

- `workspace/core4d_collab_retarget/results/E017/comparison.csv`
- `workspace/core4d_collab_retarget/results/E017/aggregate_summary.json`
- `workspace/core4d_collab_retarget/results/E017/anchor_validation_delta.csv`
- `workspace/core4d_collab_retarget/results/E017/visual/visual_eval.md`

6 个 validation variants 汇总：

| Variant | Source | Policy | Epos | Erot | Contact 5cm | Deep pen | Leg | Pass | 结论 |
|---------|--------|--------|------|------|-------------|----------|-----|------|------|
| `E017_box023_p2_face_cluster` | `E016_box023_p2` | auto face_cluster | `0.0417` | `1.85` | `5.6%` | `0.0%` | `0.0%` | False | E017 auto face 已对齐 GT face，但 quick 指标/视觉未改善 |
| `E017_box023_p2_e014_seed` | `E016_box023_p2` | E014 GT seed | `0.0413` | `1.84` | `4.4%` | `2.7%` | `0.0%` | False | 即使用 GT anchor，E016 quick 预算下仍未恢复 E014 视觉 |
| `E017_box025_p1_face_cluster` | `E016_box025_p1` | auto face_cluster | `0.0600` | `3.76` | `27.5%` | `0.0%` | `4.5%` | False | 不支持 anchor 是主要失败源 |
| `E017_box025_p2_centroid_v2` | `E016_box025_p2` | auto z-corrected | `0.0563` | `1.98` | `56.2%` | `0.0%` | `0.0%` | True | 支持 E016 z 偏高是失败因素 |
| `E017_box025_p2_e014_seed` | `E016_box025_p2` | E014 GT seed | `0.0561` | `2.02` | `59.8%` | `0.0%` | `0.0%` | True | GT seed 与 auto z-corrected 一致，通过 |
| `E017_bucket005_s2_p2_face_cluster` | `E016_bucket005_s2_p2` | auto face_cluster | `0.0424` | `4.82` | `94.8%` | `65.0%` | `11.3%` | False | 仍是 deep penetration/artifact，不支持 anchor 主因 |

与 E016 baseline 对比后的归因：

| Source case | Audit level | Validation result | 更新后的归因 |
|-------------|-------------|-------------------|--------------|
| `box023_p2` | 明确 E016 anchor face 错 | E017 auto / E014 seed 都未在 quick 预算下改善 contact/visual | anchor 选择错误成立，但不是 E016 quick 失败的充分原因；需要 full-budget 复查才可与 E014 成功直接对齐 |
| `box025_p2` | 可能 anchor 高度偏差 | auto z-corrected 与 E014 seed 均从 E016 fail 变为 pass | anchor 高度是主要原因 |
| `box025_p1` | 可能 centroid cancellation | face_cluster 无改善 | 不支持 anchor 主因 |
| `bucket005_s2_p2` | 可能 centroid cancellation | face_cluster 无改善，deep penetration 仍高 | 不支持 anchor 主因，主要是 robot-object artifact |

重要控制变量：E014 full run 的 `opt_steps=32`，而本轮 E016/E017 validation 与 E016 13-case baseline 对齐使用 quick `max_num_iterations=4`。因此 `box023_p2` 的结论只说明“在 E016 quick 预算下，修 anchor 不足以恢复”；它不能否定 E014 GT anchor 在 full-budget 下有效。

## 可视化观察

视频参数：`1440x480 @ 50fps`，front-camera ref/sim side-by-side，重放时恢复 `support_weld_anchor` mocap 位置。

实际观察：

- `box023_p2_face_cluster`：face 已从 E016 `+Y` 改到 `+X`，但后段 sim 仍出现人形失稳/倒地，视觉上没有恢复 E014 full run 的稳定搬运。
- `box023_p2_e014_seed`：使用 E014 GT anchor 后仍在 quick budget 下失稳，说明失败不只是 anchor 坐标。
- `box025_p2_centroid_v2` 与 `box025_p2_e014_seed`：sim 与 ref 姿态接近，腿/箱干扰消失，和量化 pass 一致。
- `box025_p1_face_cluster`：物体跟踪仍可，但手部接触保持没有明显改善。
- `bucket005_s2_p2_face_cluster`：仍有明显 robot-object penetration/artifact，和 deep penetration 指标一致。

## Anchor 位置视频可视化

新增脚本：`workspace/core4d_collab_retarget/scripts/eval/render_E017_anchor_videos.py`。

可视化口径：

- 输出为 `1440x480 @ 50fps`，左侧 front view，右侧 top view。
- marker 颜色：E014 GT 绿色，E016 centroid 红色，E017 auto 蓝色。
- 只渲染 reference object trajectory 上的 anchor world position，不混入 rollout 成败。
- 索引文件：`workspace/core4d_collab_retarget/results/E017/anchor_visual/anchor_visual_eval.md`。

已生成 7 个 anchor-position videos：

| Case | 视频 |
|------|------|
| `E016_box023_p2` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_box023_p2_anchor_positions.mp4` |
| `E016_box025_p2` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_box025_p2_anchor_positions.mp4` |
| `E016_box021_p2` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_box021_p2_anchor_positions.mp4` |
| `E016_box023_p1` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_box023_p1_anchor_positions.mp4` |
| `E016_box025_p1` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_box025_p1_anchor_positions.mp4` |
| `E016_bucket005_s2_p1` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_bucket005_s2_p1_anchor_positions.mp4` |
| `E016_bucket005_s2_p2` | `workspace/core4d_collab_retarget/results/E017/anchor_visual/E016_bucket005_s2_p2_anchor_positions.mp4` |

实际观察：

- `box023_p2`：红色 E016 centroid anchor 明确在 `+Y` 上侧，绿色 E014 GT 在 `+X` 侧，蓝色 E017 auto 也在 `+X` 侧但相对 GT 有约 `0.089m` 的 y offset。视频确认 E016 face 语义错，E017 只修正到同 face，未完全复现 GT 点。
- `box025_p2`：绿色 E014 GT 与蓝色 E017 auto 基本重合在 `+Y` 侧，红色 E016 anchor 在同侧但明显更高；视频确认该 case 是 anchor 高度偏差。
- `box025_p1` / `bucket005_s2_p2`：E017 auto 从红色 `+Y` centroid 改到 selected-person top face，但 quick rollout 未改善，说明“anchor 点几何更合理”不是这些 case 的充分失败原因。
- `box021_p2` / `box023_p1`：无 E014 GT；视频只展示 E016/E017 auto anchor 的实际落点，不能证明 partner/support-side 正确。

## E017 方法复查：是否需要优化

基于用户重新澄清的 E014 语义，E017 auto 不能作为最终 anchor 选择方法，只能作为 audit/debug baseline。

关键原因：

1. E014 anchor 是 object-local partner-side **proxy support point**，不是另一人手掌的真实接触点。
2. E017 `face_cluster` 仍从 selected-person active palm contact points 取 median，再 snap 到 face。它能修正 E016 centroid cancellation 的 face，但仍保留真实接触的切向偏移和高度。
3. E014 两个 GT 都符合“面中心 + 上侧高度”的 proxy pattern：
   - `box025_p2`: `[0, +half_y, 0.64*half_z]`
   - `box023_p2`: `[+half_x, 0, 0.57*half_z]`
4. E017 当前 `face_cluster` 对无 GT case 可能产生低位 anchor，例如 `box025_p1` 的 z 为 `-0.117m`、`bucket005_s2_p2` 的 z 为 `-0.058m`；这不符合 E014 的上侧 support proxy 语义。

对比一个 canonical proxy rule：先沿用 audit/GT 选 face，但点固定为该 face 的中心上侧：

```text
if face is ±X: [±half_x, 0, 0.62*half_z]
if face is ±Y: [0, ±half_y, 0.62*half_z]
```

| Case | E017 auto | Canonical proxy | 与 GT 距离 |
|------|-----------|-----------------|------------|
| `box023_p2` | `[0.153, 0.089, 0.115]` | `[0.153, 0.000, 0.109]` | `0.012m` |
| `box025_p2` | `[0.006, 0.378, 0.305]` | `[0.000, 0.378, 0.291]` | `0.009m` |
| `box025_p1` | `[-0.377, 0.181, -0.117]` | `[-0.377, 0.000, 0.291]` | no GT |
| `bucket005_s2_p2` | `[0.158, 0.093, -0.058]` | `[0.158, 0.000, 0.143]` | no GT |

结论：E017 的 face audit 有价值，但 anchor point placement 需要优化。下一版应把 E017 auto 改成 `support_proxy_canonical`：

- face 来源：E014 GT/template 优先；无 GT 时才用 audit face 作为弱证据。
- point placement：face center + upper support height，不保留 palm median 的 tangential offset。
- z band：强制在上侧 support band，例如 `0.55-0.70 * half_z`，禁止负 z / 底部 anchor。
- pre-retarget gate：有 GT case 要先通过 GT distance；无 GT case 至少输出 confidence，低置信不自动跑。

## Claims 验证

| Claim | 结果 |
|-------|------|
| C0 E016/E017 auto 是否真的使用另一侧 contact | 未通过：代码和 audit 均确认 auto anchor 输入是 `selected_person_contact_mask`，不是显式 partner-side contact |
| C1 anchor audit 能识别低置信 E016 centroid anchors | 通过：输出 E016/E017 方法审核与 GT 对齐 |
| C2 face-cluster selector 避免 centroid cancellation unsupported face | 部分通过：静态 anchor face 支持提升，但算法验证只在 `box025_p2` 转化为通过 |
| C3 anchor 相关失败能与纯 robot-side contact 失败分离 | 通过：`box025_p2` 支持 anchor 主因；`box025_p1`/`bucket005_s2_p2` 不支持；`box023_p2` 需要 full-budget 控制 |
| C4 `box023_p2` 生成 E014-consistent candidate | 通过：E017 auto face 为 `+X`，E014 seed 已生成并验证 |

## 下一步

1. 对 `box023_p2` 做 full-budget 控制实验：`E017_box023_p2_face_cluster` 与 `E017_box023_p2_e014_seed` 使用 E014 同等 `opt_steps=32`，判断自动 anchor 是否接近 E014 full 视觉。
2. 下一版不能再把 selected-person face-cluster 称为 partner/support anchor；若要自动化 support anchor，需要新建 role-aware selector，并把 E014 GT 作为校准优先级。
3. 将 `box025_p2` 的 z-corrected rule 纳入下一版自动 anchor，但标注为 selected-person/GT-calibrated rule，而不是 partner-side rule。
4. 对非 anchor 主因 case 转向 robot-side posture/contact reward 或 penetration/artifact 约束。
