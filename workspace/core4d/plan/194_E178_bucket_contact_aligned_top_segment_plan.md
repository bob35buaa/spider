# E178 实验计划：Bucket Contact-Aligned Top Segment Proxy

_Core4D Phase 41 · 2026-07-24 · 继承 E177 5/1/5 low-geom proxy_

前置证据：[E177 proxy 结果](../log/236_E177_five_step_no_lid_bucket_proxy_results.md) ·
[E177 计划](193_E177_bucket_semantic5_proxy_plan.md)

## Context

E177 的 27-case ref-FK contact fidelity 已完成，27/27 可计算、0 errors，但
object-grouped target→proxy p90 gate 只通过 bucket004：

| Object | Active rows | Contact→proxy p90 | 8cm gate |
|---|---:|---:|---|
| bucket003 | 1885 | 8.059cm | borderline FAIL |
| bucket004 | 575 | 7.480cm | PASS |
| bucket007 | 1677 | 11.258cm | FAIL |

bucket007 的 1677 个 active targets 中有 1665 个最近第 5 段，contact local-Y
p05/p95 为 `0.1965/0.2879m`；581 rows 相对 visual mesh under-cover 超过
3cm。失配集中在 `+Y` 端段的 XZ 截面，不是五段接缝、下方四段、physics pair
或 PRG union 漏接。

只读敏感性审计显示，在不增加 geom、不恢复 lid 的前提下，只扩大第 5 段：

- bucket003：XZ scale `0.94→0.95`，contact p90 `8.059→7.828cm`；
- bucket007：XZ scale `0.82→0.94`，contact p90 `11.258→7.993cm`。

E178 保留 E177 作为已完成历史，不覆盖其 sidecar、override、manifest、结果或
log。

## Geometry Design

| Object | Lower 4 segments | Top segment X/Z | Total geoms |
|---|---:|---:|---:|
| bucket003 | XZ scale 0.94 | 0.95 / 0.95 | 5 |
| bucket004 | user-authorized mesh AABB | same | 1 |
| bucket007 | XZ scale 0.82 | 0.97 / 0.885 | 5 |

bucket007 最初的等比 `0.94/0.94` 候选虽使 contact p90 达到 7.993cm，但
exposed proxy→mesh p90 为 4.031cm，未过 4cm gate。预注册的分轴候选
`0.97/0.885` 在同一 deterministic audit 下为：

```text
contact p90              7.913cm
union mesh→proxy p90     3.315cm
union proxy→mesh p90     3.952cm
```

它仍是单个 axis-aligned top box；分轴 scale 仅用于减少圆形截面角点外扩，
不新增旋转、lid 或额外 geom。

约束保持：

- 仍为 local-Y 五段实心 boxes；
- 不创建 lid/lid-strip；
- 相邻段保留 4mm overlap；
- physics pair 与 PRG box-union 使用完全相同的 `5/1/5` geoms；
- `object_collision_sdf_mode=union`；
- `object_collision_sdf_batch_groups=true`。

## Claims

| Claim | 可验证标准 |
|---|---|
| C1 scope | authority 精确为 27，分布 9/4/14 |
| C2 complexity | geom 数保持 5/1/5，pair 数保持 90/18/90 |
| C3 geometry | 三对象 union mesh→proxy 与 exposed proxy→mesh p90 均≤4cm |
| C4 contact | 27/27 可计算，三个 object-grouped target→proxy p90 均≤8cm |
| C5 localization | bucket007 active target 最近 geom 仍主要为第 5 段，且 undercoverage 指标相对 E177 改善 |
| C6 visual | 新 overlay/截面无明显大一圈、phantom bridge 或接缝；人工重新批准 |
| C7 runtime | 视觉批准后 3-case `64×4` canary 3/3 PASS，逐 case median plan time≤3s |

## Gates

### Gate A：本地 candidate

- 新 E178 sidecar/override/results，不覆盖 E177；
- 27/27 MuJoCo compile、authority parity、pair matrix PASS；
- 双向 exposed-union p90 全部≤4cm；
- E178 ref-contact gate 三对象 p90 全部≤8cm；
- overlay 与主轴截面生成完成。

若任一项失败，停止在本地诊断，不启动 CEM。

### Gate B：人工视觉 review

E178 review 初始必须为 `manual_review_required/PENDING_CODEX_REVIEW`。只有用户
或人工明确重新批准 candidate，才可进入 canary。

### Gate C：canary

沿用三条：

```text
bucket003_20231018_001_p1
bucket004_20231002_021_p1
bucket007_20231020_055_p1
```

必须 3/3 runtime PASS 且逐 case median plan time≤3s，才允许 Full。

### Gate D：Full

仅 Gate A–C 全过后，在 A100 GPUs `2,3,6,7` 启动 27-case `1024×32`。
用户已明确授权固定使用这些卡，不因其他程序动态换卡；execution manifest 仍
保存启动时 GPU/process snapshot。

## Artifacts

```text
workspace/core4d/results/E178/
├── s0_environment/
├── s2_proxy/contact_fidelity/
├── scene_snapshot/semantic_bucket_proxy/
└── s6_downstream/manifests/
```

实现入口：

```text
workspace/core4d/scripts/experiments/E178/contact_aligned_bucket_proxy.py
workspace/core4d/scripts/experiments/E178/build_contact_aligned_production.py
workspace/core4d/scripts/experiments/E178/test_contact_aligned_bucket_proxy.py
workspace/core4d/scripts/eval/runners/eval_E178_contact_fidelity.py
workspace/core4d/scripts/eval/wrappers/eval_E178_contact_fidelity.sh
```

## 执行顺序

1. 单元测试与 E177 默认行为回归；
2. E178 builder preflight，生成 27-case sidecar/manifest；
3. 双向几何 gate；
4. 27-case ref-contact gate；
5. 生成 overlay/截面，等待人工 review；
6. review 通过后才运行 canary；
7. canary 通过后才启动 Full。
