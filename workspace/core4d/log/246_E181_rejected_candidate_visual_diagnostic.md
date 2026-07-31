# E181 补充日志：rejected CoACD candidate 三维/二维诊断

_Core4D Phase 44 · 2026-07-31 · plan
[199](../plan/199_E181_coacd_canonical_geometry_plan.md)_

## 0. 一句话结论

已为 bucket003/004/007 各冻结一个代表性 rejected candidate，生成 E178
风格的四视角三维图和六切面二维占据图。图像清楚区分 `M*`、convex hull
union 与 phantom solid，人工检查 `3/3 APPROVED_DIAGNOSTIC`；它们解释了
Gate B 的失败模式，但不改变 **`ASSET_REJECTED / NO_C_STAR`** 结论。

## 1. 范围与安全边界

本补充只做 post-terminal、dev-only 可视化诊断：

- 输入仅为 cleaned `M*`、已构建 CoACD candidate 和 Gate B dev fixture；
- 不读取 heldout24/full27 query；
- 不修改 Gate B verdict、canonical selection 或 production P/R/G；
- 不生成或冒充 `C*`、`D_C`；
- 不启动 physics、canary 或 CEM；
- 每张图片显式标注 `REJECTED / NOT C*`。

## 2. 冻结 case

| Object | Candidate | 选择理由 | Hulls | Broader cavity |
|---|---|---|---:|---:|
| bucket003 | `t020_k32_v032` | 本物体最低值 | 32 | 19.070% |
| bucket004 | `t010_k32_v064` | closest-to-PASS，仅失败 cavity | 32 | 7.570% |
| bucket007 | `t020_k32_v032` | 本物体最低值 | 32 | 4.625% |

注意 bucket004 的 7.570% 是为可解释性冻结的 closest-to-PASS candidate；
E181 全 18 个 candidate 的最低值仍是正式日志中的 5.950%。两者均远高于
`≤0.1%` 硬门。

## 3. 可视化合同

三维图复用 E178 视觉语义：

- 固定 azimuth=`35/125/215/305°`；
- 左列为蓝色 oracle `M*`；
- 中列为 32 个分色 convex parts；
- 右列为 overlay，洋红点表示已知 free cavity 中被 `C` 错误占据的样本。

二维图为 object-local `XY/XZ/YZ` 六切面：

- 蓝色：`M* only`，即 collider miss；
- 橙色：`M* ∩ C`；
- 洋红色：`C only`，即 phantom solid。

## 4. 实际观察

### bucket003 · `t020_k32_v032`

- `XZ -Y base` 与 `XZ +Y end` 存在大面积连通的洋红 phantom；
- convex parts 在开口/端段跨接并形成伪封闭块；
- 该观察与 `3814/20000=19.07%` broader-cavity false occupancy，以及
  C→M、must-cover、core、normal 多门失败一致。

### bucket004 · `t010_k32_v064`

- 外表面 overlay 整体贴合，符合该 candidate 通过其余八个 fidelity/task
  gates；
- `XZ -Y base` 的凹口附近有连续 phantom bridge，`+Y end` 也有局部误占；
- 六切面 C-only 分别为
  `1.04/0.43/3.41/0.26/0.33/0.95%`；
- 问题是局部凹腔跨接，不是均匀 surface-fit 偏差；与
  `1514/20000=7.57%` broader-cavity reject 一致。

### bucket007 · `t020_k32_v032`

- `XY mid Z` 与 `YZ mid X` 出现大块蓝色 `M* only`，说明 collider
  coverage 明显不足；
- `XZ +Y end` 出现 `14.50%` 的洋红 phantom ring，端部开口/凹腔被错误
  封闭；
- 该模式与 `925/20000=4.625%` broader-cavity，以及 C→M p90/p99、core、
  normal gate 失败一致。

## 5. 审核与可复现性

| 项目 | 结果 |
|---|---|
| Renderer status | `PASS` |
| Candidate rows | 3 |
| 人工视觉审核 | `3/3 APPROVED_DIAGNOSTIC` |
| False broad 重算 | `3814 / 1514 / 925`，与 Gate B exact |
| Heldout status | `SEALED_NO_C_STAR` |
| Canonical status | `NO_C_STAR` |
| 3D montage SHA256 | `2938938a...1d885` |
| 2D montage SHA256 | `a6154f00...43f9b` |

静态检查通过：ruff、ruff format-check、`py_compile`、launcher `bash -n`
和 `git diff --check`。最终 renderer 复跑保持图片与 manifest SHA exact，
证明固定输入下输出可复现。

## 6. 结论

这批图支持以下诊断：

1. CoACD 没有“运行失败”：三个代表 candidate 均有 32 个合法 convex
   parts；
2. bucket004 证明单看外轮廓 overlay 会漏判，必须检查 cavity free-space；
3. bucket003/007 显示当前 `K≤32` 分解会在关键开口产生 phantom solid，
   bucket007 还伴随明显 collider miss；
4. 因而图片解释并强化 Gate B reject，不构成放宽门槛或继续 Full 的依据。

若继续，应在新实验号中测试更高 hull budget 或 task-aware decomposition；
E181 保持终止状态。

## 7. 结果路径

```text
workspace/core4d/results/E181/s2_asset_eval/rejected_candidate_visuals/
├── bucket003_t020_k32_v032_3d_overlay.png
├── bucket003_t020_k32_v032_2d_cross_sections.png
├── bucket004_t010_k32_v064_3d_overlay.png
├── bucket004_t010_k32_v064_2d_cross_sections.png
├── bucket007_t020_k32_v032_3d_overlay.png
├── bucket007_t020_k32_v032_2d_cross_sections.png
├── e181_rejected3_3d_montage.png
├── e181_rejected3_2d_montage.png
├── render_manifest.json
└── visual_review.json
```

可复现入口：

```bash
bash workspace/core4d/scripts/launch/active/run_E181_local.sh asset-visual
```
