# E177 实验计划：三种 Bucket 语义 1/5-geom Proxy

_Core4D Phase 40 · 2026-07-24 · 继承 E176 multi-geom physics/PRG 与 group batching_

前置证据：[E176 结果](../log/235_E176_lowgeom_proxy_canary_results.md) ·
[E176 计划](192_E176_lowgeom_nonbox_proxy_plan.md)

## Context

E176 已将 5 个 bucket + desk007 压到 6–9 collision geoms，并验证
multi-geom physics、PRG union SDF、runtime config 与 group batching
正确。但 Full-ready case 分布很不均衡：

| Object | Full-ready |
|---|---:|
| bucket003 | 9 |
| bucket004 | 4 |
| bucket007 | 14 |
| bucket009 | 1 |
| bucket010 | 2 |
| desk007 | 9 |

用户决定 desk 暂不考虑，并按 case 数弃掉 bucket009/bucket010；主要保留
bucket003、bucket004、bucket007。用户后句明确把 bucket007 列入主要优化
对象，因此本计划把前句“bucket007 和 bucket009 太少”解释为
“bucket010 和 bucket009 太少”这一明显笔误。若该解释不对，E177 在任何
CEM 启动前可直接收缩 scope。

E177 authority 固定为：

```text
bucket003  9
bucket004  4
bucket007 14
total     27
```

E177 不修改 E174–E176 的 scene、manifest、结果或完成日志。

## Geometry Design

### bucket004：单实心 AABB

- 使用 visual mesh 在 object-local frame 下的精确 AABB。
- 输出单一 `object_collision` box。
- 这是用户明确授权的低精度近似；允许填充物体内部。
- 预期 `1 geom / 18 robot-object pairs`。

### bucket003 / bucket007：无独立盖子的五段实心桶身

本地首版可视化发现两者的真实桶轴都是 object-local `+Y`，不是 `Z`。
第二轮视觉 review 否决了 3/4 段外接 AABB：轴向台阶太少，整块 lid
也明显外扩。第三轮曾验证 6 段桶身 + 分条 lid，用户最终决定取消独立
lid，并将桶身统一收敛为 5 段。PRG union SDF 当前只支持 box，因此沿
完整 `Y` 范围切成 5 段，并对每段 XZ 截面做 inward fit：

- bucket003 的 XZ 截面沿 Y 约从 `0.343m` 增至 `0.524m`：
  `5 body boxes = 5 geoms`；
- bucket007 的 XZ 截面沿 Y 约从 `0.402m` 增至 `0.528m`：
  `5 body boxes = 5 geoms`。

每个 body box 先拟合对应 Y 分层的 robust XZ 外轮廓，再以截面中心做
object-specific inward scale；层间保留毫米级 overlap。第5段直接覆盖
`+Y` 端面，不再创建任何 lid geom。

离线候选筛选冻结参数：

| Object | Body | Body XZ scale | Lid |
|---|---:|---:|---:|
| bucket003 | 5 | 0.94 | none |
| bucket007 | 5 | 0.82 | none |

body boxes 有意填充桶腔，理由是：

1. 用户已允许 bucket004 使用单实心 box；
2. E174 的主要失败之一正是腿进入空心 proxy；
3. “3–4 个 box 估计梯形圆柱”更自然地对应分层实心截面；
4. 若改成 3 个薄侧壁会留下整面碰撞缺口，4 个薄侧壁则仍保留腿可进入的
   空腔。

参数从 object-local mesh 离线拟合后按 object 冻结，不在运行时重新拟合。
预期 pair 数：

```text
bucket003  5 geoms / 90 pairs
bucket007  5 geoms / 90 pairs
```

## Claims

| Claim | 可验证标准 |
|---|---|
| C1 scope | Full authority 精确为 27，分布 9/4/14 |
| C2 complexity | bucket003/004/007 精确为 5/1/5 geoms |
| C3 physics | compiled pairs 精确为 90/18/90，无 missing/extra/duplicate |
| C4 stepped hull | body 分层跟随截锥，union proxy→mesh p90≤4cm |
| C5 body | full-Y mesh→body union p90 ≤ 4cm |
| C6 ref contact | active ref-FK contact target→proxy grouped p90 ≤ 8cm |
| C7 visual | 3/3 object-only overlay 人工 `approve_clean` |
| C8 authority parity | 除 scene/proxy/pair/实验ID外，不漂移 reward、轨迹、mask、seed |
| C9 runtime | 后续 canary 3/3 runtime PASS、median plan time≤3s |

## Gates

### Gate A：本地建模

- 三个 mesh 均可加载且 local frame/AABB 可解释；
- geom counts 精确为 `5/1/5`；
- pair counts 精确为 `90/18/90`；
- 27/27 sidecar MuJoCo compile；
- stripped scene signature 仅 proxy/pair 轴变化；
- bucket003/007 各 Y 分层 robust XZ coverage 与整体 AABB gate 通过；
- 不存在独立 lid geom，第5段必须覆盖 mesh 的 `+Y` 端；
- union 外表面双向 p90 过线，不能用包含内部重叠面的旧 proxy→mesh 指标
  代替外扩审计。

### Gate B：本地视觉与 contact fidelity

- 生成每对象四视角 mesh/proxy/overlay；
- bucket003/007 额外生成 XY/XZ/YZ 截面，明确展示阶梯实心近似；
- bucket004 明确标注“solid AABB by user decision”；
- `manual_review_required` 保持到用户/人工批准；
- 27/27 ref-FK contact audit 可计算，三个 object-grouped p90≤8cm。

### Gate C：未来 canary

只有 Gate A/B 全部通过后，才允许每个对象 1 条 `64×4` canary：

```text
bucket003_20231018_001_p1
bucket004_20231002_021_p1
bucket007_20231020_055_p1
```

必须 3/3 runtime PASS 且逐 case median plan time≤3s，才允许 27 条
`1024×32` Full。当前用户请求只授权本地 proxy 设计/可视化，本轮不启动
CEM。

## Artifacts

```text
workspace/core4d/results/E177/
├── s0_environment/
├── s2_proxy/
├── s5_handoff/
├── scene_snapshot/semantic_bucket_proxy/
└── s6_downstream/manifests/
```

主要实现：

```text
workspace/core4d/scripts/experiments/E177/semantic_bucket_proxy.py
workspace/core4d/scripts/experiments/E177/build_semantic_bucket_production.py
workspace/core4d/scripts/experiments/E177/test_semantic_bucket_proxy.py
```

Sidecar：

```text
scene_act_E177_semanticBucketProxy.xml
```

## 执行顺序

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E177/test_semantic_bucket_proxy.py
.venv/bin/python workspace/core4d/scripts/experiments/E177/build_semantic_bucket_production.py \
  --preflight
.venv/bin/python workspace/core4d/scripts/experiments/E175/render_nonbox_proxy_overlay.py \
  --experiment-label E177 \
  --input-tsv workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/proxy_visual_review.tsv \
  --out-dir workspace/core4d/results/E177/scene_snapshot/semantic_bucket_proxy/proxy_visual_evidence
```

视觉批准和 canary 属于后续显式步骤；不从本计划自动串联 Full。
