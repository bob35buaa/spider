# E018 结果：canonical support proxy anchor

日期：2026-05-19

## 初始状态

本实验按用户建议先只验证两个 E014 GT case：

- `box023_p2`
- `box025_p2`

实验目标不是继续把 palm contact point 当 anchor，而是把 E014 的人工逻辑自动化成 canonical support proxy：

```text
if face is ±X: [±half_x, 0, 0.62*half_z]
if face is ±Y: [0, ±half_y, 0.62*half_z]
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/18_E018_canonical_support_proxy_anchor_plan.md` |
| Results | `workspace/core4d_collab_retarget/results/E018/` |
| Logs | `logs/core4d_collab_retarget/E018/` |
| Manifest | `workspace/core4d_collab_retarget/results/E018/manifest.tsv` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E018/scene_snapshot/` |

## 执行状态

- [x] 实现 E018 canonical assets / overrides / train / eval / visual 脚本。
- [x] 运行 preprocess + smoke。
- [x] full-budget 跑 `box023_p2` 和 `box025_p2`。
- [x] eval + 可视化。
- [x] 两个 GT gate 已通过；下一步可规划 E018b 10+ case 泛化。

## 运行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E018_preprocess.sh --force
bash workspace/core4d_collab_retarget/scripts/train/train_E018.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E018.sh one 0 E018_box023_p2_canonical_t02
bash workspace/core4d_collab_retarget/scripts/train/train_E018.sh one 0 E018_box025_p2_canonical_t02
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E018.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/render_E018_visuals.py --force
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/render_E018_anchor_videos.py --force
```

## Anchor gate

| Variant | E018 canonical anchor | E014 GT anchor | dist | face | gate |
|---------|------------------------|----------------|------|------|------|
| `E018_box023_p2_canonical_t02` | `[0.1531, 0, 0.109492]` | `[0.16, 0, 0.10]` | `0.0117m` | `+x` | pass |
| `E018_box025_p2_canonical_t02` | `[0, 0.3778, 0.290904]` | `[0, 0.38, 0.30]` | `0.00936m` | `+y` | pass |

注意：E018 不是逐字复用 E014 手工点，而是把 E014 的 proxy pattern canonicalize 成 face center + upper support band。结论应写作“E014 GT-face canonicalized anchor 复现 E014 proxy 语义”。

## Full 结果

`aggregate_summary.json`:

```json
{
  "num_results": 2,
  "num_config_ok": 2,
  "num_gt_anchor_pass": 2,
  "num_soft_target_pass": 2,
  "num_gt_gate_pass": 2,
  "num_paper_spider_success": 2,
  "num_paper_dynaretarget_success": 2,
  "num_transport_success": 2,
  "num_contact_preservation_ok": 1,
  "num_deep_penetration_ok": 2,
  "num_generalization_pass": 1,
  "mean_paper_Epos_case_m": 0.0493,
  "mean_paper_Erot_case_deg": 2.10,
  "mean_contact_preservation_5cm_pct": 60.0,
  "mean_deep_penetration_duration_pct": 2.0,
  "max_gt_anchor_dist_m": 0.0117
}
```

| Variant | GT gate | Obj mean/max | E014 t02 obj mean/max | Hand % | Floor % | Leg % | Omni contact 5cm % | Deep pen % | Foot skate % | Diagnosis |
|---------|---------|--------------|------------------------|--------|---------|-------|--------------------|------------|--------------|-----------|
| `box023_p2` | pass | `0.0425 / 0.0793m` | `0.0426 / 0.0800m` | `75.3` | `33.3` | `0.0` | `32.1` | `4.0` | `82.5` | `contact_preservation_gap` |
| `box025_p2` | pass | `0.0562 / 0.0871m` | `0.0562 / 0.0868m` | `87.9` | `51.4` | `0.0` | `87.9` | `0.0` | `38.4` | `e018_gt_gate_pass` |

解释：

- 两个 case 都达到 E014 soft target，且 object tracking 与 E014 `t02` 几乎重合。
- `box023_p2` 的 E018 soft gate pass，但 paper-aligned OmniRetarget contact preservation 只有 `32.1%`，foot-skating 也高；这说明 canonical anchor 修复了 E016 的 anchor 语义/physics gate，但不代表完整 robot-side contact artifact 已解决。
- `box025_p2` 在 anchor、soft target、paper generalization 三层都通过。

## 可视化

| 类型 | 路径 |
|------|------|
| Comparison index | `workspace/core4d_collab_retarget/results/E018/visual/visual_eval.md` |
| Anchor index | `workspace/core4d_collab_retarget/results/E018/anchor_visual/anchor_visual_eval.md` |
| `box023_p2` comparison | `workspace/core4d_collab_retarget/results/E018/visual/E018_box023_p2_canonical_t02_comparison.mp4` |
| `box025_p2` comparison | `workspace/core4d_collab_retarget/results/E018/visual/E018_box025_p2_canonical_t02_comparison.mp4` |
| `box023_p2` anchor | `workspace/core4d_collab_retarget/results/E018/anchor_visual/E018_box023_p2_canonical_t02_anchor_positions.mp4` |
| `box025_p2` anchor | `workspace/core4d_collab_retarget/results/E018/anchor_visual/E018_box025_p2_canonical_t02_anchor_positions.mp4` |

`ffprobe` 验证 4 个视频均为 `1440x480 @ 50fps`；帧数：`box023_p2=272`、`box025_p2=248`，与 E014 对齐。

实际观察：

- `box023_p2` comparison：sim object 与 ref object 在关键帧基本重合，guard 没有 E016 可视化里的明显箱子漂走；姿态稳定，腿/箱干涉未见明显 shortcut。手部接触在若干帧不够贴合，这与 paper contact preservation 低一致。
- `box025_p2` comparison：sim/ref object 基本重合，搬运过程中没有明显 rotation shortcut；手-箱接触保持好，腿/箱干涉为 0。
- `box023_p2` anchor video：E018 yellow marker 与 E014 green marker同在 `+X` 上侧 proxy 区域；E016 red marker仍在错误的 `+Y` 侧，验证 E016 失败中存在明确 anchor face 错。
- `box025_p2` anchor video：E018 yellow marker 与 E014 green marker在 `+Y` 上侧基本重合；E016 red marker同侧但高度偏高，符合 E017/E018 对 `box025_p2` 的高度偏差诊断。

## Claims 验证

| Claim | 结果 |
|-------|------|
| C1 canonical anchor 复现 E014 GT proxy 语义 | pass：2/2 face 一致，dist `<=0.012m` |
| C2 canonical scene 保持 E014 B-only 约束 | pass：2/2 `config_ok`，`nu=29`、`nq_obj=7`、no object actuator、no direct wrench、无 `object_target` |
| C3 两个 GT case full-budget 达到 E014 级 object tracking | pass：2/2 object mean/max 与 E014 `t02` 对齐 |
| C4 可视化确认 anchor 在 partner-side 上侧 proxy 位置 | pass：4 个视频已生成并观察；anchor marker 与 E014 GT 重合 |
| C5 先过两例 GT gate，再扩展 10+ case | pass：本轮未直接全量；可以进入 E018b 泛化 |

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|------|---------|----------|
| 远程 `box025_p2` 在 `226/248` 左右退出且未写 NPZ | 1 | 检查远端 `df -h` 发现 `/` 100% 满，判定为远端磁盘问题；改为本地重跑 `box025_p2` full |
| anchor-position 视频初版用 `source_variant` 命名成 `E016_box...` | 1 | 修正 `render_E018_anchor_videos.py`，输出改为 E018 variant 命名并重刷 |

## 下一步

进入 E018b：把 canonical support proxy 扩展到 E016 的 10+ case。建议规则：

1. GT/template face 优先：已验证 case 直接用 `box023:+X`、`box025:+Y` 模板。
2. 无 GT case 用 E017 audit face 作为弱证据，但 point placement 统一为 face center + `0.62*half_z`，禁止 palm median 的切向 offset 和负 z。
3. 重定向前先跑 anchor audit gate：face confidence、是否 bottom/low-z、是否与 E016 centroid 强冲突、是否存在 counterpart weak evidence。
4. E018b 评测继续分层汇报：anchor gate / object SPIDER-Dyna gate / OmniRetarget artifact gate，不把 object success 与完整 retarget success 混为一谈。
