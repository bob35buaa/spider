# E011 计划：soft object tether 诊断 E081 所需外部 coupling

日期：2026-05-18

## Context

E006 的可视化现象“物体不平移，只旋转”已经被量化确认：参考 `box025_p2` object 水平净位移约 `1.57m`、旋转约 `2deg`，E006 main 实际水平位移只有 `0.21-0.50m`，旋转却达到 `20.6-47.6deg`。后续实验给出了一条失败链：

1. E006 direct off-COM wrench 同时有 timebase/speed 截断和 `r x F` 力矩捷径，容易旋转替代平移；
2. E008 修正 timebase/speed 后，best `ypos_k20_vmax2` 能到 xy ratio `0.724`、rot `13.6deg`，但仍差 E081；
3. E009 说明单纯加 robot-side hold-contact reward 不能闭合支撑，强 HC 会引入旋转/腿干涉；
4. E010 说明单个 mocap contact pad 不足以把 support 端运动通过 MuJoCo contact 传给 object，5/5 main proxy gate 过但 xy ratio 只有 `0.17-0.33`。

E011 不把 soft tether 当最终算法，而是诊断实验：用 object COM reference spring / optional weak rotation spring 测出“达到 E081-like transport 至少需要多强的外部 coupling”。如果弱 tether 就能恢复 E081 指标，说明当前瓶颈主要是 partner-object coupling 机制；如果强 tether 仍不能恢复，说明 robot-side retarget/search 目标本身也不支持搬运姿态。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 COM-level soft tether 能消除 E006 的 off-COM 旋转捷径 | rotation 不应超过 `15deg`，xy ratio 应随 kp 单调上升 |
| C2 若外部 coupling 是主瓶颈，中等 kp 可把 main 拉近 E081 | obj mean/max、floor、xy、hand 与 E081 gate 对比 |
| C3 若 robot-side 仍不参与，tether 成功也不能判 work | hand contact 需 `>=80%`；若 hand 很低则标记为“external-only success” |
| C4 需要记录外力代价 | 保存并评估 `partner_force_force/torque` mean/max，避免用无限外力伪成功 |
| C5 true-freejoint parity 不破坏 | `nu=29`、`nq_obj=7`、object actuator empty、`contact_guidance=false` |

## 实现改动

1. `spider/simulators/mjwp.py`
   - `step_env` 在 `partner_force_spring_kp>0` 或 `partner_force_spring_kp_rot>0` 时也调用 `_apply_partner_force`，支持纯 spring diagnostic；
   - object `xfrc_applied` 每 step 只清一次，partner force 与 support proxy 可累加；
   - 记录 `partner_force_last_force/torque`。

2. `examples/run_mjwp.py`
   - 当 partner force / spring 启用时，保存 `partner_force_force` 与 `partner_force_torque` 到 NPZ。

3. E011 scripts
   - `workspace/core4d_collab_retarget/scripts/E011/variants.tsv`
   - `workspace/core4d_collab_retarget/scripts/E011/generate_e011_overrides.py`
   - `workspace/core4d_collab_retarget/scripts/run_E011_preprocess.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E011.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E011_remote_tmux.sh`
   - `workspace/core4d_collab_retarget/scripts/run_E011_remote.sh`
   - `workspace/core4d_collab_retarget/scripts/pull_E011_remote_results.sh`
   - `workspace/core4d_collab_retarget/scripts/eval/eval_E011.py`

## Variant Grid

E011 分两组：先测 COM tether 本身，再测 E008 best 上加少量 COM tether 是否能跨过最后距离。

| Variant | Task | partner spring | support proxy | Role | 意义 |
|---------|------|----------------|---------------|------|------|
| `E011_box025_p2_com_xyz_k20` | `box025_person2_freejoint_legobj` | kp `20`, gravity `0.5`, rot `0` | off | main | 复查 E004/E005 COM spring，但用 sim_dt ref |
| `E011_box025_p2_com_xyz_k50` | same | kp `50`, gravity `0.5`, rot `0` | off | main | 中等 coupling |
| `E011_box025_p2_com_xyz_k100` | same | kp `100`, gravity `0.5`, rot `0` | off | main | 强 coupling 上界 |
| `E011_box025_p2_com_xyz_k50_g1` | same | kp `50`, gravity `1.0`, rot `0` | off | main | 检查缺的是竖直支撑还是水平 coupling |
| `E011_box025_p2_com_xyz_k50_rot1` | same | kp `50`, gravity `0.5`, rot `1` | off | main | 弱 orientation tether 是否压低 residual rotation |
| `E011_box025_p2_ypos_k20_vmax2_com_k25` | same | kp `25`, gravity `0`, rot `0` | E008 best `ypos/k20/vmax2` | main | 测 E008 best 只差少量 COM coupling 是否过门槛 |
| `E011_box025_p2_ypos_k20_vmax2_com_k50` | same | kp `50`, gravity `0`, rot `0` | E008 best `ypos/k20/vmax2` | main | E008+中等 coupling |
| `E011_box023_p2_com_xyz_k50` | `box023_person2_freejoint_legobj` | kp `50`, gravity `0.5`, rot `0` | off | guard | guard 稳定性 |
| `E011_box023_p2_com_xyz_k100` | same | kp `100`, gravity `0.5`, rot `0` | off | guard | guard 强 coupling 上界 |

`partner_force_ref_dt` 固定为 `sim_dt=0.0166667`，因为运行时 `qpos_ref` 已经插值到 sim_dt；这是与 E006/E007 时间基准 bug 对齐后的诊断口径。

## Parallel Execution

| GPU | Queue |
|-----|-------|
| local GPU0 | `com_xyz_k20` -> `ypos_k20_vmax2_com_k25` -> `ypos_k20_vmax2_com_k50` |
| remote GPU0 | `com_xyz_k50` -> `com_xyz_k100` -> `com_xyz_k50_g1` |
| remote GPU1 | `com_xyz_k50_rot1` -> `box023_com_xyz_k50` -> `box023_com_xyz_k100` |

## 成功标准

E011 的“work”仍以 E081 main 口径为准：

- `E011_freejoint_parity_ok=true`
- `case_window_obj_err_mean_m <= 0.20`
- `case_window_obj_err_max_m <= 0.40`
- `case_window_sim_contact_frames_pct >= 80%`
- `case_window_sim_object_floor_contact_frames_pct <= 75%`
- `E011_object_xy_disp_ratio >= 0.75`
- `E011_object_rot_deg <= 15deg`
- `E011_case_window_partner_force_norm_n_mean <= 120N`
- `E011_case_window_partner_force_norm_n_max <= 250N`

诊断分类：

- `physical_candidate`: 过 E081 gate 且 effort reasonable、hand contact `>=80%`；
- `external_only_success`: object 过 E081 gate 但 hand contact `<80%` 或 effort 过大；
- `robot_side_blocked`: 强 tether 仍不过 object gate；
- `rotation_shortcut`: obj mean 看似改善但 rot `>30deg` 或 floor `>80%`。

## 预授权命令

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E011.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E011_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E011_remote_results.sh __codex_auth_probe__
```

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E011_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E011.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E011.py --all

# full
bash workspace/core4d_collab_retarget/scripts/run_E011_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E011.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E011_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E011.py --all
```

## 决策规则

- 若 `E008 best + com_k25/k50` 达到 E081 gate 且 effort 合理：E012 做真实 partner mocap hands / 双点接触，把 COM tether 降解成物理可学的 contact mechanism。
- 若 `COM-only k50/k100` 能达 object gate 但 hand contact 很低：说明外部约束能搬物，但机器人没有协作，E012 应转 robot-side hand/support pose shaping。
- 若 `k100` 仍不过 object gate：说明当前 robot trajectory/search 与 reference object transport 不兼容，下一步应重新审计 reward/horizon/action space，而不是继续 virtual force。
- 若出现 rotation shortcut：禁止沿该参数继续加 kp，优先降低 rot/torque 或回到 E008 best。
