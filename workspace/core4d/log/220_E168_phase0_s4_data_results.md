# E168 Phase 0-S4 数据构建结果

日期：2026-07-17

状态：S0-S4 完成；S5/CEM/eval/paired RL export 未完成，未启动下游 RL。

## 结论

E168 对 `box004/box021/bucket004` 的同物体 move 数据完成 recall、3cm raw-contact、template、OmniRetarget 和 S4 审查。40 条 E168 新 source-person row 通过 target gate 与 visual QC；另有 `box004_20231003_2_082_p1` 直接复用 E167 `RL_EXPORT_READY`，但不得在 E168 中计为 newly generated。

OmniRetarget v1 在 40 条 production row 中 `36 pass / 4 infeasible`。隔离 canary 通过后，v2 Phase4 rescue 对 4 条 v1 infeasible 全部救回，且 `replace_wrist_with_fingertip=false`：

- `box004_20231003_2_082_p2`
- `box021_20231011_034_p2`
- `box021_20231018_028_p1`
- `box021_20231020_019_p1`

因此 v2 rescue hypothesis 当前为 pass；没有 dual-infeasible row。

## Stage Yield

| Stage | 结果 |
|---|---:|
| recall | 102 source-person / 72 move / 6 seed / 66 candidate |
| S1 3cm | 41 pass / 5 contact fail / 20 motion reject |
| E167 direct import exclusion | 1 |
| S3 new production | 40 |
| v1 | 36 pass / 4 infeasible |
| v2 production rescue | 4/4 pass |
| effective S4 target gate | 40/40 pass |
| effective S4 visual QC | 40/40 pass |

S4 pass 分布：`box004=3`、`box021=28`、`bucket004=9`；`person1=18`、`person2=22`；Tier A obs0 `18`、Tier B obs1/obs3 `22`。action 仅包含 `move1/move2 + obs0/obs1/obs3`。

## Visual QC

v1 36 条和 v2 4 条均生成 target replay MP4 与 8-frame sheet。六页 review montage 已逐页检查，未观察到错物体/错模板、机器人爆姿、物体瞬移或大尺度穿插。显式 review 记录：

```text
workspace/core4d/results/E168/s4_gate_visual_qc/visual_qc_review.tsv
workspace/core4d/results/E168/s4_gate_visual_qc/review_montages/
```

该结论只表示 source trajectory 通过 S4，不表示 CEM 或 RL 成功。

## Pair Coverage

当前 source bank 为 40 条 E168 S4 pass + 1 条 E167 direct import。已有 S3 artifact 可直接反向复用时，35 条 source 已覆盖 opposite-person OmniRetarget；仍缺 6 个唯一 partner artifact：

```text
box004_20231003_2_082_p1
box021_20231018_029_p2
box021_20231018_031_p1
box021_20231018_034_p1
box021_20231018_035_p1
bucket004_20231002_022_p1
```

这里的缺失只用于下一阶段 pair queue 规划。任何 row 在 CEM release 和 partner artifact 均通过前都不能标为 `RL_EXPORT_READY + PAIR_COMPLETE`。

## Provenance

- v1 manifest：`workspace/core4d/results/E168/s3_retarget/omnirt_v1/ref_fk/stage2b_manifest_omnirt_v1_ref_fk.tsv`
- v2 manifest：`workspace/core4d/results/E168/s3_retarget/omnirt_v2/ref_fk/stage2b_manifest_omnirt_v2_ref_fk.tsv`
- registry：`workspace/core4d/results/E168/registries/case_state_registry.tsv`
- S4：`workspace/core4d/results/E168/s4_gate_visual_qc/`
- v2 canary：`workspace/core4d/results/E168/s3_retarget/canary/omnirt_v2/ref_fk/`

A100 因缺少 policy allowlist 保持禁用；本阶段没有启动远程 GPU 或 CEM/RL。

## 下一步

1. 生成只含 S4 pass row 的 `rubber_hull` S5 handoff，并为每个新 target task 生成不覆盖 `scene_act.xml` 的 sidecar。
2. 固化 E167A config parity/axis audit 后生成 CEM canary 与 full execution manifest。
3. CEM/eval release 后仅对实际待导出 source 形成 partner reuse/v1/v2 queue。
4. 最终只导出 `RL_EXPORT_READY + PAIR_COMPLETE`，不启动下游 RL。
