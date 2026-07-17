# E168 S5/CEM Canary 结果

日期：2026-07-17

状态：A6000 exact-input canary 4/4 artifact 完成；2026-07-17 17:xx 按用户决策改用 A100 0/1/2/3，A100 compute-only canary 4/4 完成；随后按新指令重分片为 A6000 2 卡 + A100 4 卡的 6 卡 full production。

## 结论

E168 S5 handoff 与 CEM canary 执行链路已经跑通：

- `rubber_hull` handoff：40/40 `HANDOFF_READY`
- E167A config audit：40/40 pass
- CEM production manifest：40 rows，全部 `not_run`
- A6000 canary manifest：4 rows，全部 `run_complete_pending_eval`
- canary artifacts：root NPZ / outdir `trajectory_mjwp_act.npz` / `config_act.yaml` / MP4 均为 4/4

本轮原计划只验证 A6000 canary；后续按用户新决策，canary 能跑通即可，不再等待量化 canary eval，A100 0/1/2/3 直接放行 full production。

## Canary Rows

| case | object | obs | variant | status | final pos | final quat | visual note |
|---|---|---:|---|---|---:|---:|---|
| `box004_20231003_2_082_p2` | box004 | obs0 | `omnirt_v2` | complete | 0.1570 | 0.3715 | 中后段有倒地/趴倒帧 |
| `box004_20231003_2_086_p1` | box004 | obs3 | `omnirt_v1` | complete | 0.0915 | 0.1362 | 中后段有蹲倒/倒地帧 |
| `box021_20231011_036_p1` | box021 | obs1 | `omnirt_v1` | complete | 0.1646 | 0.1702 | 有翻倒/躺倒帧 |
| `bucket004_20231002_018_p1` | bucket004 | obs3 | `omnirt_v1` | complete | 0.0824 | 0.0791 | 有躺倒帧 |

这些视觉观察来自 1fps smoke sheet，不是 full-budget CEM 质量结论。它们足以说明当前 64 samples × 4 iterations smoke 不能作为 release-quality gate。

## 关键修复

初次 A6000 canary 失败于 contact mask key：

```text
missing trimmed_stage2b_output_contact_mask_3cm
```

根因是 S5 handoff 的 `contact_mask_time_axis=trimmed_stage2b_output` 是 provenance label，不是 `run_mjwp.py` 支持的 runtime mask key 前缀。已修复：

- `export_cem_overrides.py` 将未知 runtime axis 映射为 `auto`
- `audit_e167a_cem_configs.py` 对同一映射做 config parity audit
- 重建并安装 40 个 E168 override

修复后 `run_mjwp.py` 按目标长度选择 `eval_contact_mask_3cm`，4 条 canary 均执行完成。

## 结果路径

| artifact | path |
|---|---|
| canary manifest | `workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_canary_manifest.tsv` |
| production manifest | `workspace/core4d/results/E168/s6_downstream/cem/manifests/cem_production_manifest.tsv` |
| A100 canary shards | `workspace/core4d/results/E168/s6_downstream/cem/manifests/a100_canary_E168_a100_canary_20260717_170653/` |
| A100 production shards | `workspace/core4d/results/E168/s6_downstream/cem/manifests/a100_production_E168_a100_production_20260717_171706/` |
| canary root | `workspace/core4d/results/E168/s6_downstream/cem/canary/` |
| production root | `workspace/core4d/results/E168/s6_downstream/cem/full/` |
| canary logs | `logs/E168/cem/canary/` |
| production logs | `logs/E168/cem/full/` |
| video sheets | `workspace/core4d/results/E168/s6_downstream/cem/canary/video_review/` |
| A6000 selection | `workspace/core4d/results/E168/s0_environment/a6000_canary_gpu_selection.tsv` |
| A100 selection | `workspace/core4d/results/E168/s0_environment/a100_*_gpu_selection_*.tsv` |

## 2026-07-17 A100 放行更新

用户明确放行 A100 `0,1,2,3` 四卡，并要求 canary 能跑通即可，不需要先补量化 canary eval。执行结果：

- A100 直连路径：`batchcom@61.172.170.106:/home/dataset-assist-0/xiayb/workspace/spider`
- 首次 A100 canary 使用 `MUJOCO_GL=egl`，4/4 在视频渲染阶段失败，错误为 `Cannot initialize a EGL device display`。
- A100 环境 `osmesa` 和 `glfw` 也不可用；因此 A100 runner 改为 compute-only：`save_video=false`，并以 root NPZ / outdir `trajectory_mjwp_act.npz` / `config_act.yaml` 作为 canary 跑通证据。
- A100 compute-only canary session `E168_a100_canary_20260717_170653`：4/4 `run_complete_pending_eval`，GPU0-3 各完成 1 条。
- 初始 A100-only full session `E168_a100_production_20260717_171706` 已停止；停止时无 root/outdir NPZ，仅有 4 个 partial `config_act.yaml`，因此未保留为完成 row。
- production manifest 已重排为 `box021 -> box004 -> bucket004`。

## 2026-07-17 6 卡重分片更新

用户要求本地暂不使用，改为 A6000 2 卡 + A100 4 卡并行正式 full CEM。已执行：

- 新 shard root：`workspace/core4d/results/E168/s6_downstream/cem/manifests/remote6_production_20260717_173420/`
- A100 session：`E168_6gpu_a100_20260717_173420`，GPU `0,1,2,3`，compute-only `save_video=false`
- A6000 session：`E168_6gpu_a6000_20260717_173420`，GPU `0,1`，正常 full 输出
- 首批 running rows：
  - A100 GPU0：`box021_20231011_034_p1`
  - A100 GPU1：`box021_20231011_034_p2`
  - A100 GPU2：`box021_20231011_036_p1`
  - A100 GPU3：`box021_20231011_036_p2`
  - A6000 GPU0：`box021_20231011_037_p1`
  - A6000 GPU1：`box021_20231011_037_p2`
- `queue_order.tsv` 前 28 条为 `box021`，随后 `box004`，最后 `bucket004`，满足优先级要求。

注意：A100 full 当前不生成 MP4；视频如需要，应在可渲染环境中后处理或转 A6000/本地生成，不阻塞本次 full CEM 计算启动。

## 下一步

1. 监控并回收 6 卡 full production：`bash workspace/core4d/scripts/launch/active/pull_E168_remote_6gpu_results.sh production 20260717_173420`。
2. full 产物完成后再做 release/evidence 判断；A100 本轮只保证 CEM 计算产物，不提供视频。
3. CEM/evidence release 后再做 partner queue 和最终 `RL_EXPORT_READY + PAIR_COMPLETE` export。
