# E005 结果：partner-force ref_dt 显式化与 support-site 几何

日期：2026-05-17

## 状态

Setup 与 smoke 已通过。E005 在 E004 基础上做一个代码级修正：`partner_force` 的 reference frame indexing 不再写死为 30Hz，而是使用每个 task 显式给出的 `partner_force_ref_dt`。`box025` 使用 task_info 中的 `0.03333333333333333`；`box023` 使用默认 `0.02`。

本实验还新增 off-COM support-site：在 object-local 支撑点施加等效 wrench `F, r x F`，用来测试“协作者支撑点几何”是否比 COM-only force 更符合双人搬运语义。完整 full 指标将在本地/远程 full run 结束并拉回结果后填写。

## 计划执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E005_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E005_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E005.py --all
```

## 配置

变体设计：

- corrected/explicit-ref-dt COM spring (`com_s20/s40`)：作为 E004 COM-force 的对照；
- off-COM support-site force：`box025` 使用 object-local `-Y/+Y`，`box023` 使用 object-local `-X/+X`，方向来自 preprocess 计算的 ref contact normal；
- `box023_p2` guard 变体用于检查稳定性；
- 1 个 hold-contact 变体用于检查机器人参与度。

所有变体必须保持 true-freejoint parity：`scene.xml`、`contact_guidance=false`、`scene_name=""`、`object_action_dims=0`、`object_actuator_ids=[]`、`kp_rot=0`。

## Smoke

执行命令：

```bash
bash workspace/core4d_collab_retarget/scripts/run_E005_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E005.sh eval --all
```

Smoke 使用 `max_sim_steps=4`，只验证 wiring、CUDA/Warp 路径与 parity，不作为任务成功依据。

汇总：

```json
{
  "num_results": 9,
  "num_freejoint_parity_ok": 9,
  "num_main_results": 6,
  "num_guard_results": 3,
  "num_guard_stable_proxy": 3
}
```

## 结果路径

| 产物 | 路径 |
|------|------|
| Results | `workspace/core4d_collab_retarget/results/E005/` |
| Logs | `logs/core4d_collab_retarget/E005/` |
| Overrides | `examples/config/override/core4d_collab_E005_*.yaml` |
| Variants | `workspace/core4d_collab_retarget/scripts/E005/variants.tsv` |

## 待补充结果表

- config parity / smoke 细表；
- COM explicit-ref-dt 对照指标；
- support-site side ablation 指标；
- guard 稳定性指标；
- 关键帧/视频观察；
- Claims C1-C5 验证；
- E006 决策。
