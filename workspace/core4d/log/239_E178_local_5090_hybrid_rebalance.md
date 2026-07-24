# E178 运行日志：RTX 5090 测速与 Hybrid Rebalance

_Core4D Phase 41 · 2026-07-24 ·
[主计划 194](../plan/194_E178_bucket_contact_aligned_top_segment_plan.md) ·
[追加计划 195](../plan/195_E178_hybrid_local_remote_rebalance_plan.md)_

## 0. 当前结论

本地 GPU0 为 RTX 5090 32GB，测速前仅占用 150MiB。与 A100 使用相同
bucket007 case 的 `64×4` probe 显示，本地 optimized-record median 为
`1.0995s`，A100 为 `2.9799s`，本地快 `2.7102×`。

追加 `1024×2` production-density probe 后，本地 median 为 `0.6857s`；
按 iterations 线性外推 `1024×32` 为 `10.9712s/record`。与此同时，正在运行
的远端 Full 四条 row 实测为 `25.6089–30.1505s/record`。因此本地 5090
适合作为第五个、且相当于约 2.3–2.75 张当前 A100 worker 的加速节点。

12:50 live allocation preflight 在 20% 本地 slowdown safety factor 下选择
本地接管 11 条远端队尾 rows、远端保留 16 条，预测剩余 makespan
`5.996h→3.986h`，缩短 `33.52%`。

## 1. 环境失败与修复

第一次 probe 沿用历史本地脚本的 `MUJOCO_GL=osmesa`，在 MuJoCo import
阶段立即失败：

```text
AttributeError: 'NoneType' object has no attribute 'glGetError'
```

该尝试 wall time 为 0s，manifest 保持 `not_run`，没有进入 CEM、没有写正式
Full 路径。随后对 `egl/glfw/disable` 做 import preflight，三者均通过；
`egl` 进一步成功加载 E178 bucket007 sidecar，因此正式 probe/hybrid 使用
与远端一致的 EGL，没有重复 osmesa 失败配置。

失败证据：

```text
workspace/core4d/results/E178/s0_environment/
  local_speed_probe_canary_20260724_123931/
```

## 2. Isolated probes

| Probe | Samples×iters | Wall | Records | Median | P90 | Runtime |
|---|---:|---:|---:|---:|---:|---|
| RTX 5090 canary | `64×4` | 92s | 77 | 1.0995s | 1.1063s | PASS |
| A100 canary | `64×4` | — | 77 | 2.9799s | 3.0489s | PASS |
| RTX 5090 density | `1024×2` | 59s | 77 | 0.6857s | 0.7037s | PASS |

两个本地 probe 均使用独立 manifest/result/outdir/log，未修改：

- E178 Full manifest；
- 正式 Full result/outdir；
- 远端 shard；
- A100 watcher/pull authority。

结果：

```text
workspace/core4d/results/E178/s6_downstream/benchmark/
  local_speed_probe_canary_20260724_124046/
  local_speed_probe_density_20260724_124230/
```

## 3. 远端 Full 实测

live shard/log snapshot 中四条 running row：

| GPU | Object | Full median plan time |
|---:|---|---:|
| 2 | bucket003 | 29.8947s |
| 3 | bucket003 | 29.1567s |
| 6 | bucket003 | 30.1505s |
| 7 | bucket004 | 25.6089s |

当前远端 runner 已把 shard 读入内存，不能直接改 TSV 删除队列；强杀会损失
首条正在运行的 case。因此不停止 A100 worker，只使用 runner 的
`output_complete→skip-complete` contract 对队尾 rows 做本地提前完成。

## 4. Hybrid allocation preflight

12:50 snapshot：

| 项目 | 结果 |
|---|---:|
| Authority | 27 rows，bucket003/004/007=`9/4/14` |
| Remote-only predicted makespan | 5.996h |
| Local safety factor | 1.20 |
| Tail counts GPU2/3/6/7 | `2/3/3/3` |
| Local / remote rows | `11 / 16` |
| Hybrid predicted makespan | 3.986h |
| Predicted improvement | 33.52% |
| Minimum remote deadline margin | 2237s |

11 条本地 rows 全部位于对应远端 shard 的位置 5–7。执行顺序按远端预计到达
deadline 排序；即使本地比 probe 外推慢 20%，最小仍预留约 37 分钟。

preflight：

```text
workspace/core4d/results/E178/s0_environment/
  hybrid_local_20260724_1250_preflight/
```

## 5. Race-safe handoff contract

每条 local-owned row：

1. 启动前读取 live remote shard，必须仍为 `not_run`；
2. 本地单 worker 运行原始 E178 production row 的 `1024×32`；
3. 单 row E176 multi-geom runtime validator 必须 PASS；
4. promotion 前再次确认同一远端 row 仍为 `not_run`；
5. 产物先 rsync 到 remote staging，再快速 promote
   `result_npz/outdir_npz/config_act/log/row_manifest`；
6. 远端 runner 对该 row 立即执行一次 `[skip-complete]` 验证；
7. 原始 A100 worker 后续到达该 row 时再次自行 skip。

任一 live status 已变为 `running/complete`，立即停止 promotion，不覆盖远端。

## 6. 实现与验证

```text
workspace/core4d/scripts/launch/active/run_E178_local_speed_probe.sh
workspace/core4d/scripts/experiments/E178/build_hybrid_rebalance.py
workspace/core4d/scripts/launch/active/run_E178_local_hybrid.sh
```

已通过：

- launcher `bash -n`；
- builder/runner/validators `py_compile`；
- scoped `git diff --check`；
- hybrid `PREP_ONLY`；
- 27-row authority、object distribution、local/remote disjoint、deadline margin
  和预测收益 gates。

## 7. 下一步

正式 hybrid 启动时重新抓 live snapshot 并重算一次 allocation，不能直接复用
12:50 preflight。启动后监控首条 local production row 的实际
`1024×32` timing；若偏离 safety estimate 或 race margin 不足，停止分配后续
local rows，远端原队列继续兜底。

## 8. 正式启动更新

12:54 live snapshot 重算结果：

| 项目 | 正式启动值 |
|---|---:|
| Remote-only remaining makespan | 5.954h |
| Hybrid remaining makespan | 3.944h |
| Predicted improvement | 33.75% |
| Local / remote rows | 11 / 16 |
| Tail counts GPU2/3/6/7 | `2/3/3/3` |

Hybrid ID：

```text
hybrid_local_20260724_125411
```

首条 local-owned row `bucket007_20231023_075_p2` 于 12:54:14 启动；启动前
live remote row 仍为 `gpu7.tsv / not_run / position 5`。RTX 5090 首轮正式
`1024×32` plan time 为 `11.14–11.43s`，与 density 外推 `10.9712s`
偏差很小，明显低于预留的 20% slowdown safety。显存约 2.0GiB、利用率约
61%，未见 OOM 或 numeric error。

正式执行证据：

```text
workspace/core4d/results/E178/s0_environment/
  hybrid_local_20260724_125411/
```

## 9. Full 运行中间结果：本地离线视频

用户要求先查看已经完成的 Full rows。15:20 以启动时五项完整性快照
（primary NPZ、outdir NPZ、config、scene、source trajectory）冻结出 8 条
bucket007；仍在计算或尚未回收的 rows 不进入本轮渲染。

渲染入口：

```text
workspace/core4d/scripts/launch/active/run_E178_render_completed_local.sh
```

结果：

| 项目 | 结果 |
|---|---:|
| Selected / rendered / failed | `8 / 8 / 0` |
| FPS | `50` |
| Frames per video | `118–212` |
| Duration | `2.36–4.24s` |
| Video size | `0.77–1.44MB` |

持久路径：

```text
workspace/core4d/results/E178/s6_downstream/render/full/
workspace/core4d/results/E178/s6_downstream/render/full/
  keyframes_completed_20260724_152018/
logs/E178/render/full/render_completed_20260724_152018.log
```

### 可视化 → 实际观察

已按 `video-frames` 从 8 条视频各抽取中间帧并检查 montage。8/8 均正确显示
E178 bucket007 蓝色桶 mesh、G1 和 ref/physics 左右对照；未见黑帧、空 scene、
模型拓扑串 case、明显镜头裁切或编码损坏。中间帧可见不同程度的机器人姿态/
足部偏差，`bucket007_20231020_059_p1` 的桶在中段呈倾斜状态；这属于需要用户
结合完整时间序列判断的物理结果，而非 renderer 故障。本轮只确认 replay
可用，不把 midpoint QC 夸大为 CEM quality pass。

## 10. Full 最终回收

本地 Hybrid 于 15:48 完成 `11/11` local-owned rows，每条 runtime gate
均 PASS，且远端对应 shard 后续全部执行 `[skip-complete]`，未发生双写或
覆盖。A100 四个 worker 于约 16:53 清空计算队列；watcher 连续确认 session
结束后执行唯一一次最终权威 pull。

| 项目 | 最终结果 |
|---|---:|
| Full authority rows | `27/27` |
| Local promotion / A100 compute | `11 / 16` |
| Primary result NPZ | `27/27` |
| Outdir trajectory NPZ | `27/27` |
| Config | `27/27` |
| Runtime gate | `27/27 PASS` |
| Pull/eval errors | `0` |

最终 runtime evidence：

```text
workspace/core4d/results/E178/s6_downstream/cem/full/full_runtime_gate.json
workspace/core4d/results/E178/s0_environment/
  a100_full_e178_a100_full_20260724_120137/pull_summary.json
```

## 11. Full 统一评估

新增 E178 canonical 入口：

```text
workspace/core4d/scripts/eval/runners/eval_E178_lowgeom.py
workspace/core4d/scripts/eval/wrappers/eval_E178_lowgeom.sh
```

入口复用参数化后的 E176 low-geom evaluator；E176 的默认 experiment id、
expected rows 和输出命名保持不变。E178 明确固定为 `27` rows、输出前缀
`e178`，并把离线 render MP4 写入 metrics 的 `video` 列。

```bash
bash workspace/core4d/scripts/eval/wrappers/eval_E178_lowgeom.sh \
  full --require-all
```

| Object | Numeric pass | 主要失败计数 |
|---|---:|---|
| bucket003 | `5/9` | hand penetration 2；release/fall/body-z/lower-body 各 1 |
| bucket004 | `3/4` | contact 1；release 1 |
| bucket007 | `8/14` | hand penetration 4；contact 2；lower-body 2；release 1 |
| **Overall** | **`16/27`** | hand penetration 6；contact 3；release 3；lower-body 3；fall/body-z 各 1 |

完整性为 evaluated/paired=`27/27`，not-ready/errors/missing-baseline=`0/0/0`，
summary status=`pass`。`gate_health_pass=0/27` 是 result NPZ 内部 PRG
leg-gate health 的独立诊断，不能解释为最终 numeric pass=`0/27`；最终统一
numeric gate 仍为 `16/27`。

结果：

```text
workspace/core4d/results/E178/s6_downstream/eval/full/
  e178_case_metrics.tsv
  e178_vs_e174_paired_deltas.tsv
  e178_group_summary.tsv
  summary.json
```

## 12. 全量视频与 review player 注册

完成最终 pull 后重跑增量 renderer：已存在 `8` 条自动 skip，新渲染 `19`
条，最终 selected/rendered-or-existing/failed=`27/27/0`。逐文件 ffprobe
确认 27 条均为 H.264、`1440×480`、50fps，帧数 `118–432`；metrics 中
outdir/scene/trajectory/video 均为 `27/27 exists`。

```text
workspace/core4d/results/E178/s6_downstream/render/full/
workspace/core4d/results/E178/s6_downstream/render/full/
  keyframes_all_20260724_165645/
```

### 可视化 → 实际观察

按 `video-frames` 对 27 条视频各取 duration 中点，按 bucket003/004/007
生成三张 montage 并逐张打开检查。27/27 均有清晰的 ref/physics 双画面、
G1 与正确 bucket mesh，未见黑帧、空 scene、串物体拓扑、编码破损或明显
镜头裁切。bucket003 的中点姿态/桶位差异较分散；bucket004 至少一条中点
物理桶位与参考差异明显；bucket007 若干条有人体/物体偏差，至少一条桶明显
倾斜。以上只证明 renderer/replay 可用，不替代完整时间序列人工质量裁决。

review player 的真实 Python 实现位于：

```text
workspace/core4d/scripts/eval/review/viser_review_player.py
```

canonical 启动入口仍为：

```bash
bash workspace/core4d/scripts/eval/wrappers/review_player.sh
```

E178 已加入 `review_index.DEFAULT_EXPS`。Headless check 结果为 E178
indexed/evaluated/numeric-pass/playable=`27/27/16/27`，全库
indexed/playable=`165/165`。另对首条 E178 做真实 loader smoke：
MuJoCo `nq=42`，sim/ref shape=`(416,42)/(466,42)`，playback 416 frames、
50fps，确认注册后的 3D 数据可实际加载。

## 13. Full-validation XLSX

按 E173/E174 既有 full-validation workbook 结构生成：

```text
workspace/core4d/results/E178/s6_downstream/eval/full/
  E178_buckets_prg_full_validation.xlsx
```

可复现入口：

```bash
python3 workspace/core4d/scripts/eval/reports/gen_E178_bucket_prg_xlsx.py
```

工作簿包含 Overview、Case Metrics、Paired Deltas、Group Summary、Worst
Cases、Manual Review、Codex Verification、Not Ready、Eval Errors 共 9 个
sheet。Overview 使用 Excel 公式汇总 evaluated/numeric/pass-rate、三个
bucket 的分项结果、E174→E178 numeric transition、gate health 和人工审阅
状态；Manual Review 预置 27 条 `PENDING` row，并绑定 27 个正式 MP4。

LibreOffice 强制重算结果：

```text
status=success
total_formulas=554
total_errors=0
```

data-only 复核为 evaluated/numeric=`27/16`、pass rate=`59.259%`、
bucket003/004/007=`5/9,3/4,8/14`、E174 fail→E178 pass=`16`、
E174 pass→E178 fail=`1`。Case Metrics、Manual Review、Codex Verification
中的视频路径与 27 个 keyframe 路径均 `27/27 exists`；XLSX zip integrity
PASS。
