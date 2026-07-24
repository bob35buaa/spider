# E177 结果日志：五段无盖低 geom Bucket Proxy

_Core4D Phase 40 · 2026-07-24 · plan [193](../plan/193_E177_bucket_semantic5_proxy_plan.md)_

## 0. 一句话结论

E177 将后续 bucket scope 收敛为 bucket003/004/007 共 27 条，并冻结
`5/1/5` 个 collision boxes：bucket003/007 使用沿 object-local Y 的五段
实心桶身且不建独立盖子，bucket004 使用单个 mesh AABB。27/27 scene
contract 与最终 mesh/proxy 视觉审查通过，用户于 2026-07-24 明确批准使用
该版本；本日志不包含 contact fidelity、canary 或 Full CEM 结果。

## 1. Scope 与最终几何

| Object | Cases | Proxy | Geoms | Robot–object pairs |
|---|---:|---|---:|---:|
| bucket003 | 9 | local-Y 五段实心 box，无独立盖子 | 5 | 90 |
| bucket004 | 4 | mesh AABB 单实心 box | 1 | 18 |
| bucket007 | 14 | local-Y 五段实心 box，无独立盖子 | 5 | 90 |
| **总计** | **27** | — | — | — |

bucket003/007 的每一段按对应 Y band 的 robust XZ 截面拟合，段间保留
4mm overlap。为避免 box 比原始圆角/圆截面 mesh 明显偏大，XZ 截面以中心
做 inward scale：

- bucket003：`0.94`；
- bucket007：`0.82`。

第 5 个 body box 覆盖 `+Y` 端面，不再创建 lid 或 lid-strip geom。桶身保持
实心，以阻止 E174 中出现过的腿进入空心 bucket proxy。

## 2. 静态与几何结果

| Object | Geoms | Pairs | Mesh→union p90 | Union→mesh p90 |
|---|---:|---:|---:|---:|
| bucket003 | 5 | 90 | 2.43 cm | 2.14 cm |
| bucket004 | 1 | 18 | 2.21 cm | 3.03 cm |
| bucket007 | 5 | 90 | 3.36 cm | 3.65 cm |

- E174 39-row authority 筛选后精确得到 `9/4/14=27` 条。
- 27/27 sidecar MuJoCo compile PASS。
- pair matrix 精确为 `18 × object_geom_count`，无 missing/extra/duplicate。
- override parity 与 non-proxy scene signature PASS。
- 三个对象的 union 双向 p90 均不超过 4cm。
- PRG 保持 `object_collision_sdf_mode=union` 与
  `object_collision_sdf_batch_groups=true`，没有引入 non-box SDF 盲区。

## 3. 可视化

### 证据

- 四视角 mesh/collision/overlay：
  `workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/proxy_visual_evidence/e177_3_object_proxy_montage.png`
- XY/XZ/YZ、`-Y base` 与 `+Y end` 截面：
  `workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/proxy_visual_evidence/e177_bucket_cross_section_montage.png`

### 实际观察

- bucket003/007 均只包含 5 个沿 local-Y 连续排列的 body boxes，图中无独立
  lid/lid-strip。
- 五段 proxy 跟随桶身由小到大的截锥趋势，4mm overlap 未形成轴向碰撞漏缝。
- bucket007 圆截面处的 box 主动 inward，避免回到整块外接 AABB 的大角点
  phantom；局部 mesh 超出 proxy 属于控制在约 4cm gate 内的显式取舍。
- bucket004 单 AABB 与 mesh 外包络一致，保留用户指定的低精度单 box 决策。
- 用户最终确认：“ok，就用这版吧”。

## 4. 验证

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E177/test_semantic_bucket_proxy.py
.venv/bin/python \
  workspace/core4d/scripts/experiments/E177/build_semantic_bucket_production.py \
  --preflight --overwrite
.venv/bin/python \
  workspace/core4d/scripts/experiments/E175/render_nonbox_proxy_overlay.py \
  --experiment-label E177 \
  --input-tsv workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/proxy_visual_review.tsv \
  --out-dir workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/proxy_visual_evidence
.venv/bin/python workspace/core4d/scripts/experiments/E175/test_multigeom_sdf.py
```

结果：E177 proxy test、27-case builder、overlay render、E175 multi-box union
回归和 `git diff --check` 均 PASS。

## 5. Claims

| Claim | 结果 |
|---|---|
| C1 scope 27，分布 9/4/14 | **PASS** |
| C2 geom 数 5/1/5 | **PASS** |
| C3 pairs 90/18/90 | **PASS** |
| C4 union proxy→mesh p90≤4cm | **PASS** |
| C5 full-Y mesh→body union p90≤4cm | **PASS** |
| C6 ref-contact→proxy p90≤8cm | **未执行** — 下一 gate |
| C7 三对象 visual review | **PASS** — 用户明确批准 |
| C8 authority parity | **PASS** |
| C9 canary runtime≤3s | **未执行** |

## 6. 结果路径

| 类型 | 路径 |
|---|---|
| E177 root | `workspace/core4d/results/E177/` |
| scene snapshot | `workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/` |
| Full manifest | `workspace/core4d/results/E177/s6_downstream/manifests/semantic_bucket_full_manifest.tsv` |
| Canary manifest | `workspace/core4d/results/E177/s6_downstream/manifests/semantic_bucket_canary_manifest.tsv` |
| Builder summary | `workspace/core4d/results/E177/s6_downstream/manifests/semantic_bucket_manifest_summary.json` |

## 7. 结论与下一步

E177 的 low-geom proxy 版本已冻结，但只完成了本地几何、物理 contract 与视觉
批准。下一步应先跑 27 条 ref-FK contact fidelity；通过后再运行三对象 canary，
最后才可决定是否在 GPUs `2,3,6,7` 启动 27-case Full CEM。
