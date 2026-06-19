# E166 pathology-split Phase1 prep

日期：2026-06-18
计划：`workspace/core4d/plan/180_E166_pathology_split_phase1_continuation_plan.md`
前置结论：`box023_person2` 归为 self-collision/init-penetration pathology，E166 主线采用 7-label pathology-split gate，不补 `box026_139_p1` SUGAR label。

## 结论

E166 可以从 Phase0 进入 SPIDER Phase1/2 准备，但本轮只完成默认关闭实现和 CPU 验证，未启动 CEM/RL/GPU。

Phase0 依据仍为：

- all-case C-R3 未通过，不能声称全体 case 红线通过。
- 排除 `box023_person2` pathology 后，非 pathology 子集脚/平滑信号成立：`ankle_acc`、`trackbody_jerk_p95`、`obj_speed_max`、`foot_slip` 的 `|rho|` 达到继续主线的标准。
- `box026_139_p1` 保持 unlabeled，只参与 Tier-1 metric，不参与 downstream correlation。

## 实现

### `spider/config.py`

新增默认关闭/中性字段：

- `cem_smooth_enabled=False`
- `cem_smooth_body_names=["left_ankle_roll_link", "right_ankle_roll_link"]`
- `cem_smooth_body_ids=[]`
- `cem_smooth_accel_weight=0.0`
- `cem_smooth_jerk_weight=0.0`
- `local_frame_ankle_ids=[7, 13]`
- `local_frame_ankle_weight=1.0`
- `foot_slip_enabled=False`
- `foot_slip_weight=0.0`
- `foot_slip_contact_height_m=0.05`
- `foot_ground_enabled=False`
- `foot_ground_weight=0.0`

`process_config` 只在 `cem_smooth_enabled=True` 时解析 `cem_smooth_body_names`，默认不触发。

### `spider/simulators/mjwp.py`

新增 A2 ankle extra-weight hook，完全镜像 E044 wrist extra-weight：

- 只在 `local_frame_ankle_weight != 1.0` 且 `local_frame_ankle_ids` 非空时执行。
- 默认 `local_frame_ankle_weight=1.0`，因此 baseline reward 数值路径不变。

### `spider/postprocess/smooth_handoff.py`

新增 E166-B2 CPU-only 后处理骨架：

- 读一个 handoff `.npz`，写独立输出 `.npz`，不覆盖输入。
- 默认平滑 floating arrays 的时间轴，跳过 `mask/contact/phase/time/frame` 等相位数组。
- 支持 `--pin-mask-key`，在 mask on/off 边界附近恢复原值，避免接触相位边界被平滑移动。
- 无 `scipy` 时 fallback 到 moving average。

## 验证

已通过：

```bash
python -m py_compile spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/postprocess/smooth_handoff.py
python -m py_compile workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py
bash -n workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
git diff --check
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh
```

E166 redline 仍输出到：

- `workspace/core4d/results/E166/redline_predictive/redline_case_metrics.tsv`
- `workspace/core4d/results/E166/redline_predictive/redline_correlations.tsv`
- `workspace/core4d/results/E166/redline_predictive/redline_summary.md`

B2 smoke：

- 构造临时 `.npz`
- 运行 `spider/postprocess/smooth_handoff.py`
- 验证输出保留 `contact_mask` 不变，`qpos` 被识别为 smoothed key

## 未做

- 未启动 SPIDER CEM。
- 未启动 SUGAR/RL。
- 未实现 `sampling.py` 内的 jerk/accel reward 扣分。
- 未实现 foot-slip / foot-ground reward 数值项。
- 未生成 E166 Phase3 manifest。

## 下一步

1. 在 SPIDER 内补 E166 B1/A/A+B manifest builder，明确 3 case × 5 臂中哪些复用、哪些新跑、哪些走 B2 后处理。
2. 实现 B1 sample-level smoothness aggregation，优先只影响 `rews`，不做 hard valid gate。
3. 实现 A1/A3 脚约束 reward，并保持默认关闭。
4. 写本地/远程 launch/pull 脚本后，再按本机 1 卡 + 远程 2 卡叠加跑。
