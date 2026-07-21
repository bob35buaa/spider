# E166 continuation plan: pathology-split gate 后进入 SPIDER Phase1/2

日期：2026-06-18
前置：
- `log/214_E166_redline_predictive_results.md`
- `log/215_E166_R3b_redline_pathology_split_results.md`
- `log/216_E166_R3c_two_stage_redline_results.md`

## Context

用户已确认 `box023_person2` 应视为特例：它的失败主因是 self-collision / init-penetration pathology，而不是 E166 主线关心的“脚可执行性 + 平滑度”。因此 E166 不再为了补 `box026_139_p1` downstream label 跳到 SUGAR；Phase0 采用当前 7 个已有 downstream label。

修订后的 gate 口径：

- all-case C-R3 没有通过：`max |rho|=0.505 < 0.6`。
- 但排除 `box023_person2` 后，脚/平滑/速度信号成立：
  - `ankle_acc rho=-0.783`
  - `trackbody_jerk_p95 rho=-0.725`
  - `obj_speed_max rho=-0.667`
  - `foot_slip_max_m rho=-0.638`
- `box023_person2` 从 E166 A/B 主实验排除，单独归为 pathology/self-collision 修复方向。

本计划继续 E166，但必须限定在 SPIDER 主线内；不再启动 SUGAR 补标签任务，不补 box026 downstream。

## Claims

| # | Claim | 最低证据 |
|---|---|---|
| C-GATE-revised | pathology split 后，E166 主线可进入 Phase1/2 | `box023_person2` 被 self-collision pathology 捕获；非 box023 子集 `ankle_acc/jerk/obj_speed/foot_slip` 至少一项 `|rho|>=0.6` |
| C-impl-safe | 默认关闭实现不改变 E163 baseline 行为 | 新增 config 默认值全关或权重为中性；现有脚本 py_compile/import 通过 |
| C-A2-prep | ankle extra weight 可独立打开并默认等价原行为 | `local_frame_ankle_weight=1.0` 时 reward 逻辑与旧版一致 |
| C-B2-prep | handoff 后处理可作为 CPU-only 独立工具准备 | 新工具默认不覆盖输入，缺依赖/不支持格式时显式失败 |

## Scope

### 进入本轮

1. 更新实验记录：E166 后续使用 `pathology_split_gate`，明确排除 `box023_person2`。
2. 做默认关闭的 SPIDER 实现准备：
   - `spider/config.py` 新增 E166 字段，默认关闭/中性。
   - `spider/simulators/mjwp.py` 先实现 A2 ankle extra weight，完全镜像 wrist extra weight，默认 `1.0` 不改变 baseline。
   - 新建 `spider/postprocess/smooth_handoff.py` CPU-only B2 工具骨架，输入/输出独立，不覆盖原文件。
3. 只做本机 CPU 验证和 py_compile；不启动 CEM/RL GPU。

### 不进入本轮

1. 不启动 SUGAR 训练或 eval。
2. 不启动 E166 9 条 CEM / 12 条 RL。
3. 不修改 E165D peak-margin 逻辑，不复用 `cem_peak_margin_*`。
4. 不把 box023 纳入 E166 A/B 主实验。

## 成功标准

| 项 | 标准 |
|---|---|
| 记录 | tracker/progress 标明 R3b pathology-split gate 作为继续依据 |
| 默认行为 | 所有新增开关默认不改变原 reward |
| 代码验证 | `python -m py_compile spider/config.py spider/simulators/mjwp.py spider/optimizers/sampling.py spider/postprocess/smooth_handoff.py` |
| E166 eval | `bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh` 仍输出 7-label Phase0 结果 |

## 后续 GPU 条件

本轮完成后，若用户确认继续 GPU：

1. 先补 E166 manifest：3 case × 5 臂，排除 box023。
2. 再写 `workspace/core4d/scripts/train/train_core4d_E166*.sh` 和 `scripts/launch/active/run_E166_remote.sh`。
3. 启动前必须 snapshot scene，并保留 contact/penetration 一票否决。
