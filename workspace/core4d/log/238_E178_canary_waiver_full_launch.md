# E178 运行日志：Canary、吞吐豁免与 Full 启动

_Core4D Phase 41 · 2026-07-24 · plan
[194](../plan/194_E178_bucket_contact_aligned_top_segment_plan.md)_

## 0. 当前结论

E178 三条 `64×4` canary 已完整回收：runtime `3/3 PASS`，吞吐
`2/3 PASS`。唯一超门项 bucket004 的 median plan time 为 `3.0215s`，
比预注册 `3.0s` 阈值高 `0.0215s`（约 `0.72%`）。用户于 2026-07-24
明确认为该差异不影响正式运行并批准启动 Full。

原始 throughput gate 保持 `status=fail`，没有改写指标。launcher 使用默认
关闭、必须填写原因且写入 execution manifest 的显式 waiver；runtime gate
仍不可豁免。27-case `1024×32` Full 已在固定 GPUs `2,3,6,7` 启动。

## 1. Canary 结果

| Object / case | Geoms | Pairs | Records | Median | P90 | Runtime | 3s gate |
|---|---:|---:|---:|---:|---:|---|---|
| bucket003 / `20231018_001_p1` | 5 | 90 | 202 | 2.93525s | 2.9717s | PASS | PASS |
| bucket004 / `20231002_021_p1` | 1 | 18 | 139 | 3.0215s | 3.0754s | PASS | FAIL |
| bucket007 / `20231020_055_p1` | 5 | 90 | 77 | 2.9799s | 3.0489s | PASS | PASS |

产物：

- runtime gate：
  `workspace/core4d/results/E178/s6_downstream/cem/canary/canary_runtime_gate.json`
- throughput gate：
  `workspace/core4d/results/E178/s6_downstream/cem/canary/canary_throughput_gate.json`
- canary manifest：
  `workspace/core4d/results/E178/s6_downstream/manifests/semantic_bucket_canary_manifest.tsv`
- 三条 result/outdir NPZ 与 config 均已回收；runtime validator
  `passed_rows=3`、`failed_rows=0`。
- bucket003 canary 视频未生成，但视频在 pull contract 中为 optional，不影响
  runtime/throughput 判定。

## 2. 用户授权的 Gate C 例外

本次只豁免 throughput gate 的轻微超门：

```text
bucket004 excess = 3.0215 - 3.0 = 0.0215s
relative excess  = 0.0215 / 3.0 = 0.7167%
```

不豁免以下 contract：

- canary runtime 必须 `3/3 PASS`；
- Full 精确为 27 rows、`1024` samples、`32` iterations；
- 固定使用用户指定的 GPUs `2,3,6,7`；
- exact-file sync、remote preflight、GPU/process snapshot；
- Full 完成后的 scoped pull 与 artifact validator。

为避免篡改历史 gate，generic A100 launcher 新增：

```text
E176_ALLOW_THROUGHPUT_GATE_WAIVER=1
E176_THROUGHPUT_GATE_WAIVER_REASON=<required reason>
```

默认仍严格拒绝 throughput FAIL；开关为 1 但 reason 为空也会拒绝。
execution manifest 记录 waiver boolean 和完整原因。

## 3. Full 启动

| 字段 | 值 |
|---|---|
| Session | `e178_a100_full_20260724_120137` |
| Remote root | `/home/dataset-assist-0/xiayb/workspace/e178_spider_runs/e178_a100_full_20260724_120137` |
| GPUs | `2,3,6,7` |
| Rows | 27 |
| CEM | `1024×32` |
| Frame loads | GPU2=780 / GPU3=854 / GPU6=838 / GPU7=826 |
| Sync | 1934/1934 files PASS |
| Watch interval | 30s |

启动前 rsync 两次遇到远端 timeout/broken pipe；bounded retry 自动恢复，
没有创建重复 tmux。最终 remote sync verification、dry-run preflight 和
tmux 创建全部成功。

execution evidence：

```text
workspace/core4d/results/E178/s0_environment/
  a100_full_e178_a100_full_20260724_120137/execution_manifest.json
```

## 4. 当前状态与下一步

截至 2026-07-24 12:07，Full tmux 已由 watcher 确认存活。Full 结束后执行：

1. 连续两次确认远程 tmux 消失；
2. 仅回收 execution manifest 登记的 27 rows；
3. 校验 result NPZ、outdir NPZ、config、log 与 manifest 状态；
4. 运行正式 E178 evaluator，记录各 object/case 的 numeric、contact 与
   penetration 结果；
5. 更新本日志、Tracker 和 S6 downstream evidence。
