# E182 阶段日志：bucket003 task-aware pre-segmentation v8 结果

_Core4D Phase 45 · 2026-08-01 · plan
[200](../plan/200_E182_task_conditioned_coacd_full_cem_plan.md) · 前置
[248](248_E182_bucket003_p_decomposition_v5_v7_results.md)_

## 0. 一句话结论

v8 按冻结协议完整构建了
`2 task-derived planes × threshold {5,10,20mm} × K {16,32}=12`
个 bucket003 candidate，但 static P topology gate 为 **`0/12 PASS`**。最佳
near-miss 是 `plane1/t010/K32`：
`TP18/phantom7/missed9, precision=0.720, recall=0.667`；满足 precision，
但未恢复第19个真接触。能恢复 `TP≥19` 的候选最少仍有12个 phantom，未形成计划要求的
`TP≥19 AND phantom≤8` 第三工作点。

因此 v8 的冻结动作是：

```text
STOP_V8_NO_PLANE_OR_FLOOR_CHANGES
```

本轮不进入完整 P/R/G、grid-SDF、production freeze、heldout24 或 Full CEM。
这不是 CoACD 软件构建失败：12/12 candidate 均构建成功；失败项是 task-distribution
上的 P contact precision–recall launch floor。

## 1. 背景与假设

log 248 已证明三类自由度均无可用 candidate：

| 路线 | 搜索规模 | P PASS |
|---|---:|---:|
| v5 global threshold | 9 | 0 |
| v6 per-segment threshold hybrid | 3×6561 | 0 |
| v7 segment3 Bell(4) partition | 2×15 | 0 |

v7 precision-side 到 recall-side 只新增3个 contact pose：1个 TP 与2个 phantom；三点
在 object-local `x` 上相隔约48.65mm，而 `y/z` range 仅4.91/5.23mm。因此 v8 的
假设是：用三点在 dominant axis 上的全部相邻中点，对原 segment3 做 task-aware exact
local split，可能分离“需要恢复的 TP”和“不应新增的 phantom”。

## 2. 冻结协议

### 2.1 自动 topology

| 项目 | 冻结值 |
|---|---|
| transition labels | `TP, PHANTOM, PHANTOM` |
| dominant axis | `x`，相对第二轴 range=`9.30×` |
| planes | `-0.14311002844145554m`, `-0.11878629238288715m` |
| source topology | x2×y4 的8 segments |
| v8 topology | 每个 candidate 只用一条 plane 替换 segment3，最终9 segments |
| unchanged reuse | 7 segments × 3 thresholds = 21 frozen v4 manifests |
| K | `16,32`；不运行 K8/K4 |
| threshold | `5,10,20mm` |
| static P authority | 882 poses，27 oracle-contact poses |
| hard gate | precision/recall均`≥0.70`，等价于`TP≥19 AND phantom≤8` |

两条 exact boolean split 均为2/2 watertight、winding-consistent、positive-volume；
volume closure delta 分别为 `4.77535e-11m³` 与 `-1.27153e-12m³`。

### 2.2 可复现性闭环

冻结前完成：

- v8 direct contracts `14/14 PASS`；
- core/v1–v7 parent matrix `60/60 PASS`；
- ruff、format、compileall、repo `diff --check` 全 GREEN；
- v8 root 在 protocol freeze 前为空；
- protocol 直接绑定8个 runtime source roles，包含 v4 reducer/check/export source；
- segment/base/candidate/static aggregate 均有 strict SHA/tamper/resume validator；
- complete stage resume 为 validation-only no-op，不重跑 CoACD/reducer/evaluator。

protocol SHA：

```text
d1b19069c4be0b7f4c33884ca7aab6539ad9080ce0624a84d8eb78cf9e45fdd8
```

## 3. 构建结果

### 3.1 CoACD bases

只对两条 plane 的两个新 child、三个 threshold 运行 CoACD，共12个 fresh subprocess；
每个 child 返回4 hulls。21个 unchanged v4 base manifest 只复用，不重建。

| 指标 | 结果 |
|---|---:|
| new children | 12/12 BUILD_PASS |
| composite bases | 6/6 BUILD_PASS |
| base totals（每plane t005/t010/t020） | `36/32/22` |
| 4-worker stage wall | `128.42s` |
| child wall sum | `348.54s` |
| GPU / heldout access | `0 / 0` |

CoACD 多次报告“4 convex hull limitation 下 residual concavity 超 threshold”。本协议已把
`max_convex_hull=4`定义为 soft request，并保留最多16 hull的安全上限；本轮12个 child
均返回4，因此这些 warning 不是 build failure，candidate 是否可用由冻结 static P 决定。

### 3.2 Final candidates

| base | K16 | K32 |
|---|---:|---:|
| t005（36 hulls） | 16（20 merges） | 32（4 merges） |
| t010（32 hulls） | 16（16 merges） | 32（0 merges） |
| t020（22 hulls） | 16（6 merges） | 22（0 merges） |

两条 plane 的 count pattern 相同。12/12 candidate 都满足：

- 9个 nonempty segments；
- no cross-presegment merge；
- actual hulls `≤K`；
- ordered asset SHA 唯一；
- final part max vertices `≤256`，实际每row最大值为
  `73/38/68/32/55/32/74/40/69/32/67/32`；
- candidate reducer wall sum `4.67s`。

## 4. Static P 结果

| Candidate | Plane | t(mm) | K | Actual | TP/phantom/missed | Precision | Recall | Wall(s) | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `taskpreseg_v8_p00_t005_k16_v256` | 0 | 5 | 16 | 16 | 22/24/5 | 0.478 | 0.815 | 0.919 | FAIL |
| `taskpreseg_v8_p00_t005_k32_v256` | 0 | 5 | 32 | 32 | 22/22/5 | 0.500 | 0.815 | 0.949 | FAIL |
| `taskpreseg_v8_p00_t010_k16_v256` | 0 | 10 | 16 | 16 | 19/17/8 | 0.528 | 0.704 | 0.923 | FAIL |
| `taskpreseg_v8_p00_t010_k32_v256` | 0 | 10 | 32 | 32 | 19/15/8 | 0.559 | 0.704 | 0.947 | FAIL |
| `taskpreseg_v8_p00_t020_k16_v256` | 0 | 20 | 16 | 16 | 15/11/12 | 0.577 | 0.556 | 0.946 | FAIL |
| `taskpreseg_v8_p00_t020_k32_v256` | 0 | 20 | 32 | 22 | 15/11/12 | 0.577 | 0.556 | 0.992 | FAIL |
| `taskpreseg_v8_p01_t005_k16_v256` | 1 | 5 | 16 | 16 | 8/6/19 | 0.571 | 0.296 | 0.934 | FAIL |
| `taskpreseg_v8_p01_t005_k32_v256` | 1 | 5 | 32 | 32 | 8/3/19 | 0.727 | 0.296 | 0.924 | FAIL |
| `taskpreseg_v8_p01_t010_k16_v256` | 1 | 10 | 16 | 16 | 18/10/9 | 0.643 | 0.667 | 0.902 | FAIL |
| `taskpreseg_v8_p01_t010_k32_v256` | 1 | 10 | 32 | 32 | 18/7/9 | **0.720** | **0.667** | 0.933 | FAIL |
| `taskpreseg_v8_p01_t020_k16_v256` | 1 | 20 | 16 | 16 | 19/15/8 | 0.559 | 0.704 | 0.930 | FAIL |
| `taskpreseg_v8_p01_t020_k32_v256` | 1 | 20 | 32 | 22 | 19/12/8 | 0.613 | 0.704 | 0.971 | FAIL |

static-P total query wall sum=`11.27s`；aggregate=`0/12`。

### 4.1 Trade-off 解释

- **Plane0/t005** 是 high-recall 端：`TP22/missed5`，但 phantom=`22–24`；
- **Plane1/t005** 是 high-precision/low-recall 端：K32 precision=`0.727`，但只保留8个 TP；
- **Plane1/t010/K32** 是最佳平衡点：把 phantom 降到7，却仍只有18个 TP；
- **Plane1/t020/K32** 恢复到19个 TP，但 phantom 回升到12；
- K32通常减少同 topology 的 phantom，但无法独立改变 recall signature；增加 hull count
  不是缺失第三工作点的充分条件。

所以 v8 证明：**单条 task-derived plane 会移动离散 P 工作点，但仍无法同时保留
TP19 与 phantom≤8。**

## 5. 可视化

### 5.1 命令

```bash
uv run python \
  workspace/core4d/scripts/experiments/E182/render_task_aware_preseg_v8.py
```

renderer 是 static-P 后的只读 diagnostic，不修改 frozen builder；选取：

1. `best_balance`: plane1/t010/K32，`18/7/9`；
2. `high_recall`: plane0/t005/K32，`22/22/5`；
3. `tp19_precision_best`: plane1/t020/K32，`19/12/8`。

每个代表均输出四视角3D和XY/XZ/YZ+timeline 2D：

- [best balance 3D](../results/E182/s2_task_query_eval/attempt2_task_aware_preseg_v8/visual_diagnostic/best_balance_task_pose_3d.png)
- [best balance 2D](../results/E182/s2_task_query_eval/attempt2_task_aware_preseg_v8/visual_diagnostic/best_balance_task_pose_2d.png)
- [high recall 3D](../results/E182/s2_task_query_eval/attempt2_task_aware_preseg_v8/visual_diagnostic/high_recall_task_pose_3d.png)
- [high recall 2D](../results/E182/s2_task_query_eval/attempt2_task_aware_preseg_v8/visual_diagnostic/high_recall_task_pose_2d.png)
- [TP19 precision-best 3D](../results/E182/s2_task_query_eval/attempt2_task_aware_preseg_v8/visual_diagnostic/tp19_precision_best_task_pose_3d.png)
- [TP19 precision-best 2D](../results/E182/s2_task_query_eval/attempt2_task_aware_preseg_v8/visual_diagnostic/tp19_precision_best_task_pose_2d.png)

### 5.2 实际观察

已逐张查看6张原分辨率图，而非只检查文件存在：

- 三个代表的最深 phantom 都是 `reference pose148`，位于同一 bucket 顶部端部/rim
  的手部查询点；不是远处 body/leg 噪声；
- 该 pose 的 oracle clearance 约 `+0.465mm`，三个候选分别给出
  `-4.01/-4.37/-4.77mm`，即局部 convex envelope 向外跨过接触临界面约4–5mm；
- 3D 总体 bucket 外形仍肉眼合理，但 rim 邻域的 hull 边界仍跨过原 mesh 的薄壁/唇缘；
- best-balance 仅7个 phantom，但 timeline 显示漏掉9个真实 contact；
- high-recall 在 reference 早段产生一串额外 candidate-contact 窗，TP提高到22的同时
  phantom也提高到22；
- TP19代表需要12个 phantom 才换回第19个 TP；
- phantom 同时出现在 reference 与 E178-final 的多个离散窗口，不是一个可忽略的偶发 pose；
- 标题、坐标、图例均完整可读。

视觉证据与数字结论一致：问题仍是上缘局部 convex 外包的 task-contact trade-off。

## 6. Claims 验证

| Claim | 结果 |
|---|---|
| V8-C1 自动 split family 在看新分数前唯一冻结 | **PASS** — 两plane、12 rows、8 source roles与SHA闭合 |
| V8-C2 exact topology 与 build 可复现 | **PASS** — 4 split children、12 CoACD children、12 candidates全部通过严格审计 |
| V8-C3 产生第三P工作点 `TP≥19 AND phantom≤8` | **FAIL** — `0/12`；TP19最少phantom=12 |
| V8-C4 可进入完整 P/R/G | **FAIL / FORBIDDEN** — static P hard prerequisite未通过 |
| V8-C5 不污染 heldout/Grid/Full | **PASS** — heldout/GPU/grid/Full access均为0 |
| V8-C6 负结果有实际3D/2D视觉证据 | **PASS** — 3 representatives / 6 images均已实际查看 |

## 7. 结果路径与 SHA

统一结果根目录：

```text
workspace/core4d/results/E182/s2_task_query_eval/
  attempt2_task_aware_preseg_v8/
```

| Artifact | SHA-256 |
|---|---|
| `attempt2_v8_protocol_manifest.json` | `d1b19069c4be0b7f4c33884ca7aab6539ad9080ce0624a84d8eb78cf9e45fdd8` |
| `segments/segment_summary.json` | `5101cae592e44b2e1566ca883094aa6d994eaedefb783da85cf46f119dbde25e` |
| `base_summary.json` | `417de337b12fdd381bbce33a6a32dfcae91ee1bbf381803f19559a27a6bcfb16` |
| `build_summary.json` | `3c1d8d8022b6cf8ec637a34d08612b4fcb0502c89478dd59456c42c9268e057f` |
| `static_p_screen/static_p_aggregate.json` | `10666ec4d61668b528838cdab7cef9e8cb5748c2a884b2b10a827d7d694df60b` |
| `visual_diagnostic/visual_manifest.json` | `1d1ce64ea6e97f44476dc4580c4511cf0eb6b9efe600e877c44fcf9e9558a5d3` |

结果根共388个文件；protocol/build/static/visual strict validators均已重跑 PASS。

## 8. 失败协议与下一步

### 已冻结的停止项

- 不移动两条 plane；
- 不增加手工切面；
- 不降低 P floor；
- 不因 K32 precision 较好而强送完整 P/R/G；
- 不访问 heldout24；
- 不启动 grid-SDF 或 Full CEM；
- 不使用本地或远程 GPU。

### 后续方向需要重新决策

v5–v8 已连续排除 global threshold、per-segment threshold、partial merge 与单-plane
task-aware split。继续沿相同自由度扫参违反三次失败协议。若要继续 E182，需要在新计划/
新协议中选择实质不同的结构，例如：

1. **同时使用两条自动导出的 plane**，把三个 transition x 区域完全隔开；这必须作为
   新 topology（不是回改 v8）预注册，并说明为何不构成事后挑 cut；
2. **rim-aware 非平面分区/局部原始mesh collision primitive**，直接保留唇缘曲面；
3. **放弃“同一 CoACD C 同时承担全部 P”**，采用 MuJoCo SDF/plugin 或更高保真局部碰撞
   路线；仍保持 D_M 只作 oracle，不能数据泄漏选型。

在用户确认新的方法自由度前，E182 状态为：

```text
P_LAUNCH_FLOOR_NOT_CLOSED
FULL_NOT_AUTHORIZED
V8_STOPPED_COMPLETE_NEGATIVE_RESULT
```
