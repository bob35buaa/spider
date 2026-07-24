# E178 追加计划：RTX 5090 / A100 Hybrid Throughput Rebalance

_Core4D Phase 41 · 2026-07-24 · 追加到
[E178 主计划](194_E178_bucket_contact_aligned_top_segment_plan.md)_

## Context

E178 27-case `1024×32` Full 已在 A100 GPUs `2,3,6,7` 启动。四个远端
worker 的初始 frame loads 为 `780/854/838/826`，但用户观察到实际推进偏慢，
并明确授权使用本地 GPU。

本地只读快照显示 GPU0 为 `NVIDIA GeForce RTX 5090 32GB`，显存占用
`150MiB`、利用率 0%，没有其他 `run_mjwp/run_cem_queue`。远端当前仍是
0 个正式 result NPZ，四个 worker 均在首条 case。

远端 runner 在启动时把 shard rows 读入内存，不能靠修改 TSV 安全删除正在
排队的 rows；强杀 worker 会损失当前 case。另一方面，runner 在每条 row
开始前会检查 `result_npz + outdir_npz + config_act`：三件产物完整且验证通过
时会执行 `[skip-complete]`。因此 hybrid 采用“本地先完成远端队尾 row，再把
完整产物同步到本次 remote run root”的无中断方案。

## Claims

| Claim | 可验证标准 |
|---|---|
| H1 isolation | speed probe 使用独立 variant/result/outdir/log，E178 Full manifest 与正式产物 sha 不变 |
| H2 comparability | 本地先跑与 A100 canary 相同的 bucket007 case、`64×4`、seed 0，并报告 wall time 与 optimized-record median/p90 |
| H3 production scaling | canary probe PASS 后追加同 case `1024×2` 独立探针，估计 production sample-density 下的 plan time |
| H4 authority | hybrid allocation 对 27 个 case 精确分区；local-owned 与 remote-only 不重叠，已 running/completed case 不迁移 |
| H5 race safety | 仅当远端 shard row 仍为 `not_run` 时同步本地产物；若已为 `running/complete`，禁止覆盖远端 |
| H6 runtime | 本地正式 row 也通过 E176 multi-geom runtime validator，产物 sha 与 outdir 一致 |
| H7 benefit | 预测 hybrid makespan 比保持四张远端卡至少缩短 5%，否则不迁移正式 rows |

## Phase 1：本地速度探针

基准 case：

```text
bucket007_20231020_055_p1
```

理由：已有 A100 canary 基准（median `2.9799s`、77 records），输入、scene、
proxy 与 reward 完全相同；83 frames 较短，能快速测出 5090 相对速度。

两个独立 probe：

1. `64×4`：与 A100 canary 直接可比；
2. `1024×2`：只在第一个 probe runtime PASS 后执行，用于估计 Full 的
   sample-density；不作为 E178 正式 Full 结果。

持久路径：

```text
workspace/core4d/results/E178/s0_environment/local_speed_probe_<timestamp>/
workspace/core4d/results/E178/s6_downstream/benchmark/local_speed_probe_<timestamp>/
logs/E178/benchmark/local_speed_probe_<timestamp>/
```

入口必须固化为：

```text
workspace/core4d/scripts/launch/active/run_E178_local_speed_probe.sh
```

## Phase 2：吞吐估计与分配

从以下证据估计速度：

- A100 canary 的逐 case median plan time；
- A100 Full 每个已完成 row 的 wall time、frames、object geom count；
- RTX 5090 两个 probe 的 wall time、records、median/p90；
- 当前远端四个 shard 的 row 顺序与剩余 frame load。

以 `predicted_seconds(case, device)` 为权重做 LPT 分配，目标最小化五个 worker
的最大预测完成时间。若只能得到相对速度，则用：

```text
predicted cost = frames × production plan-time factor(object) / device speed
```

local-owned rows 只从各远端 shard 的队尾选择，为同步产物留出安全提前量。
任何 `running/run_complete_pending_eval` row 都固定留在远端。

## Phase 3：无中断 Hybrid 执行

1. 生成独立 allocation manifest，记录 row owner、来源 remote shard、
   queue position、预测成本和选择时间；
2. 本地单 worker 串行运行 local-owned rows，使用原始 production row 的
   `1024×32` 和正式 E178 输出路径；
3. 每条本地 row 完成后运行完整 runtime validation；
4. 再次读取远端 shard：只有该 row 仍为 `not_run` 才把
   `result_npz/outdir_npz/config_act/log` 同步到本次 remote run root；
5. 远端 runner 到达该 row 后自行 `[skip-complete]` 并原子更新 shard；
6. watcher 仍以远端 execution manifest 为最终 pull authority。

禁止：

- 杀掉当前 A100 case；
- 修改/覆盖正在运行的远端 row；
- 用 benchmark 产物冒充正式 Full；
- 同一正式 row 在本地和远端同时写同一文件系统；
- 在没有 5% makespan 收益时为了“多一张卡”强行迁移。

## 成功标准

- 两个 probe 的输入 sha、参数和结果路径可审计；
- 得到 5090/A100 的可解释速度比与误差边界；
- hybrid allocation 的 27-row authority parity PASS；
- 本地与远端最终合计恰好 27 个有效 Full rows；
- 无 artifact overwrite、无 duplicate authority、无因 rebalancing 丢失的
  running case；
- 预测和实测 makespan 均记录到后续 E178 log。

## 回退

若 probe 失败、RTX 5090 生产密度明显更慢、或远端队尾已接近启动，则保持现有
A100 四 worker 不变。probe 作为环境/吞吐诊断保留，不重复同配置。
