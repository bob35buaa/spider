# E018 Plan: canonical support proxy anchor

日期：2026-05-19

## Context

E014 证明 COLA-B kinematic support body + soft weld/equality 结构成立，但 anchor 是人工指定的 object-local partner-side proxy support point。E016/E017 尝试自动 anchor 时暴露了语义偏差：

- E016 `mask_active_ref_palm_centroid_surface_clamp` 会把 palm contact centroid 当 anchor，`box023_p2` 因 `+X/-X` 双峰抵消后错误落到 `+Y`。
- E017 `face_cluster` 能修正部分 face，但仍从 selected-person palm contact median 取点，保留真实接触的切向偏移和低/底部高度。
- E014 anchor 不是另一侧真实接触点，而是“partner-side proxy support point”：固定在 object local，面中心附近，上侧高度，避免 COM oracle 和极端角点。

E017 方法复查显示 canonical proxy rule 同时贴近两个 E014 GT：

```text
if face is ±X: [±half_x, 0, 0.62*half_z]
if face is ±Y: [0, ±half_y, 0.62*half_z]
```

对 GT case：

- `box023_p2`: canonical `[0.153, 0, 0.109]`，到 E014 `[0.16, 0, 0.10]` 约 `0.012m`。
- `box025_p2`: canonical `[0, 0.378, 0.291]`，到 E014 `[0, 0.38, 0.30]` 约 `0.009m`。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 canonical anchor 复现 E014 GT proxy 语义 | `box023_p2` / `box025_p2` 的 anchor-to-GT dist 分别 `<=0.03m`，face 与 GT 一致，z 在 `0.55-0.70 * half_z` |
| C2 canonical scene 仍保持 E014 B-only 约束 | `contact_guidance=false`、`nu=29`、`nq_obj=7`、object actuator empty、no direct wrench、非 COM relpose、无 `object_target` |
| C3 两个 GT case 在 full-budget 下达到 E014 级 object tracking | 使用 E014 同等 full CEM budget；`box025_p2` 过 E014 soft target，`box023_p2` guard stable |
| C4 可视化确认 anchor 点在 partner-side 上侧 proxy 位置 | 生成 comparison video 与 anchor-position video，实际观察不得留空 |
| C5 只有两例 GT gate 通过后才扩展 10+ case | 本轮不直接全量；若 C1-C4 通过，再开 E018b 13-case canonical 泛化 |

## 改动

1. 新增 E018 canonical anchor generator：
   - 输入 E016 13-case variants 和 E017 audit face。
   - Phase A 只写两个 GT variants：`box023_p2`、`box025_p2`。
   - face 优先用 E014 GT/template；无 GT case 后续才用 audit face。
   - point placement 使用 face center + `canonical_z_frac=0.62`。

2. 新增 E018 scene/override/train/eval/visual：
   - `scripts/E018/generate_e018_assets.py`
   - `scripts/E018/generate_e018_overrides.py`
   - `scripts/run_E018_preprocess.sh`
   - `scripts/train/train_E018.sh`
   - `scripts/eval/eval_E018.py`
   - `scripts/eval/render_E018_visuals.py`
   - `scripts/eval/render_E018_anchor_videos.py`

3. 新增通用 scene snapshot 脚本：
   - `scripts/convert/snapshot_scenes.sh`
   - E018 train 入口先调用它，快照 scene XML 和 manifest sha256。

## Variants

| Variant | Source task | Face source | Face | Anchor rule | 预计 anchor |
|---------|-------------|-------------|------|-------------|-------------|
| `E018_box023_p2_canonical_t02` | `box023_person2` | E014 GT/template | `+x` | `[+half_x, 0, 0.62*half_z]` | `[0.153, 0, 0.109]` |
| `E018_box025_p2_canonical_t02` | `box025_person2` | E014 GT/template | `+y` | `[0, +half_y, 0.62*half_z]` | `[0, 0.378, 0.291]` |

两者都沿用 E014 的 `solref=0.02 1`、`solimp=0.9 0.95 0.001`、`support_proxy_gravity_scale=0.5`、`hold_contact_rew_scale=0`。

## 成功标准

| Gate | 目标 |
|------|------|
| Config | 2/2 `E018_config_ok=true` |
| GT anchor | 2/2 `anchor_gt_dist_m <= 0.03` 且 `anchor_gt_face_match=true` |
| Object tracking | `box025_p2` main 过 E014 soft target；`box023_p2` guard stable |
| Comparison to E017 | `box023_p2` full-budget canonical 明显优于 E017 quick face-cluster；`box025_p2` 不低于 E017 z-corrected |
| Visualization | 2/2 comparison video + anchor-position video，`1440x480 @ 50fps` |

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E018_preprocess.sh --force
bash workspace/core4d_collab_retarget/scripts/train/train_E018.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E018.sh local_gt 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E018.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/render_E018_visuals.py --force
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/render_E018_anchor_videos.py --force
```

若两个 GT case 通过，再创建 E018b manifest 并按本地 1 卡 + 远程 2 卡扩展到 10+ case。

## 结果路径

| 产物 | 路径 |
|------|------|
| Plan | `workspace/core4d_collab_retarget/plan/18_E018_canonical_support_proxy_anchor_plan.md` |
| Log | `workspace/core4d_collab_retarget/log/18_E018_canonical_support_proxy_anchor_results.md` |
| Results | `workspace/core4d_collab_retarget/results/E018/` |
| Logs | `logs/core4d_collab_retarget/E018/` |
| Manifest | `workspace/core4d_collab_retarget/results/E018/manifest.tsv` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E018/scene_snapshot/` |
