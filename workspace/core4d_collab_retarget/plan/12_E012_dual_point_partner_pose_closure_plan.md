# E012 计划：dual-point partner pose closure 诊断

日期：2026-05-18

## Context

E011 已经把 E006 的“只旋转、不平移”问题拆开：

1. E006 的 off-COM support-point wrench 容易形成 `r x F` 旋转捷径，并且 E006 还有 `support_proxy_ref_dt=0.0333` 误用 sim_dt 插值参考的 timebase 截断；
2. COM-level spring 能明显恢复平移，`k100` 达到 xy ratio `0.905`、rot `3.5deg`，并略优于 E008 best；
3. 但 `k100` 的 object error 仍是 `0.340/0.673m`，离 E081 `0.143/0.271m` 很远；
4. `g1` 能把 floor contact 降到 `22.5%`、xy ratio 提到 `0.971`，但 hand contact 掉到 `75.1%`，像外部支撑拉走物体，不是协作搬运；
5. E008+COM k25/k50 没有修好，k25 还把 rotation 拉到 `24.8deg`。

因此 E012 不再继续单 COM kp/gravity sweep。下一步做 dual-point partner-side pose closure：在 partner 侧同一物体面上放两个 local feature points，让两个点都跟随 reference object 的对应点。它仍然是诊断性外部 coupling，不是最终算法；目标是验证“姿态/高度/端点闭合”是否足以把 E011 的 COM 平移改善推进到 E081 级 object error，同时不让机器人手端脱开。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 dual-point spring 比 single off-COM support point 更少旋转捷径 | object rot `<=15deg`，不出现 E006 的 `20-48deg` 旋转 |
| C2 dual-point closure 可显著降低 E011 k100 的 object error | main obj mean/max 明显低于 `0.340/0.673m`，目标接近 `<=0.20/0.40m` |
| C3 若 robot-side 仍不闭合，dual-point 成功也会暴露为 hand contact 低 | hand contact 必须 `>=80%`，否则标记 `external_only_success` |
| C4 effort 必须合理 | partner force mean/max `<=120/250N`，torque max `<=30Nm` |
| C5 true-freejoint parity 不破坏 | `nu=29`、`nq_obj=7`、object actuator empty、`contact_guidance=false` |

## 实现改动

1. `spider/config.py`
   - 新增 `partner_force_points_local: list[list[float]]`，默认空；
   - 若非空，优先于 `partner_force_point_local`，表示多个 object-local feature points。

2. `spider/simulators/mjwp.py`
   - 在 `_apply_partner_force()` 中支持 multi-point spring：
     - 对每个 local point `p_i` 计算当前点 `x + R p_i` 和 ref 点 `x_ref + R_ref p_i`；
     - 每点施加 `F_i = kp/N * (ref_i - cur_i) - kd/N * v_i`；
     - 累加 net force 和 `sum(r_i x F_i)` 到 `xfrc_applied`；
     - 保持 object true-freejoint，不新增 action/object actuator；
   - 继续保存 net `partner_force_force/torque` 诊断字段。

3. E012 scripts
   - `workspace/core4d_collab_retarget/scripts/E012/variants.tsv`
   - `workspace/core4d_collab_retarget/scripts/E012/generate_e012_overrides.py`
   - `workspace/core4d_collab_retarget/scripts/run_E012_preprocess.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E012.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E012_remote_tmux.sh`
   - `workspace/core4d_collab_retarget/scripts/run_E012_remote.sh`
   - `workspace/core4d_collab_retarget/scripts/pull_E012_remote_results.sh`
   - `workspace/core4d_collab_retarget/scripts/eval/eval_E012.py`

## Variant Grid

Main `box025_person2_freejoint_legobj` 使用 partner 侧 `y=+0.38`，两个点沿 box x 方向分开，近似 partner 双手/双点支撑：

| Variant | Points local | kp | gravity | extra shaping | Role | 意义 |
|---------|--------------|----|---------|---------------|------|------|
| `E012_box025_p2_dualy_x20_k50_g05` | `[+/-0.20, +0.38, 0.30]` | 50 | 0.5 | none | main | 中等 dual-point，和 E011 k50 对比 |
| `E012_box025_p2_dualy_x20_k100_g05` | same | 100 | 0.5 | none | main | 和 E011 k100 对比 |
| `E012_box025_p2_dualy_x30_k100_g05` | `[+/-0.30, +0.38, 0.30]` | 100 | 0.5 | none | main | 更大力臂，检查姿态闭合 |
| `E012_box025_p2_dualy_x20_k100_g08` | `[+/-0.20, +0.38, 0.30]` | 100 | 0.8 | none | main | 竖直支撑介于 E011 k100 与 g1 |
| `E012_box025_p2_dualy_x20_k150_g05` | same | 150 | 0.5 | none | main | 强 coupling 上界，检查是否只差 stiffness |
| `E012_box025_p2_dualy_x20_k100_g05_obj3` | same | 100 | 0.5 | object reward x3 | main | 看 reward 是否能让 robot-side 跟上 |

Guard `box023_person2_freejoint_legobj` 使用 object x 侧双点，避免 E011 guard k100 摔倒：

| Variant | Points local | kp | gravity | Role | 意义 |
|---------|--------------|----|---------|------|------|
| `E012_box023_p2_dualx_y10_k50_g05` | `[+0.16, +/-0.10, 0.10]` | 50 | 0.5 | guard | 稳定性 guard |
| `E012_box023_p2_dualx_y10_k100_g05` | same | 100 | 0.5 | guard | 强 coupling guard |

## Parallel Execution

| GPU | Queue |
|-----|-------|
| local GPU0 | `box025_dualy_x20_k100_g05` -> `box025_dualy_x20_k100_g05_obj3` |
| remote GPU0 | `box025_dualy_x20_k50_g05` -> `box025_dualy_x20_k150_g05` -> `box023_dualx_y10_k50_g05` |
| remote GPU1 | `box025_dualy_x30_k100_g05` -> `box025_dualy_x20_k100_g08` -> `box023_dualx_y10_k100_g05` |

## 成功标准

Main work gate 仍对齐 E081：

- `E012_freejoint_parity_ok=true`
- `case_window_obj_err_mean_m <= 0.20`
- `case_window_obj_err_max_m <= 0.40`
- `case_window_sim_contact_frames_pct >= 80%`
- `case_window_sim_object_floor_contact_frames_pct <= 75%`
- `E012_object_xy_disp_ratio >= 0.75`
- `E012_object_rot_deg <= 15deg`
- `E012_case_window_partner_force_norm_n_mean <= 120N`
- `E012_case_window_partner_force_norm_n_max <= 250N`
- `E012_case_window_partner_torque_norm_nm_max <= 30Nm`

诊断分类：

- `physical_candidate`: 过 E081 gate 且 hand/floor/effort 合理；
- `external_only_success`: object gate 过但 hand contact `<80%` 或 effort 过大；
- `pose_closure_helped`: 相对 E011 k100 obj mean/max 至少改善 `0.08/0.15m`，但未过 E081；
- `rotation_shortcut`: rot `>30deg` 或 floor `>80%`；
- `guard_unstable`: guard pelvis min `<0.55m` 或 leg interference 超 E081 guard `+5pp`。

## 预授权命令

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E012_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E012_remote_results.sh __codex_auth_probe__
```

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E012_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E012.py --all

# full
bash workspace/core4d_collab_retarget/scripts/run_E012_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E012.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E012_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E012.py --all
```

## 决策规则

- 若 dual-point 过 E081 gate 且 effort 合理：E013 将 dual-point spring 物理化为 partner mocap hands / 双点 contact constraint。
- 若 dual-point 只改善 obj error 但 hand 仍掉出门槛：E013 转 robot-side hand/support pose shaping，而不是继续加 partner effort。
- 若 dual-point 强 coupling 仍不过 E011 k100：说明 reward/search 对 robot-object 协同姿态不兼容，下一步重审 robot retarget objective/horizon，而不是继续 virtual force。
- 若 guard k100 继续摔倒：保留 guard 稳定约束，main 任何成功都不能直接判为通用方案。
