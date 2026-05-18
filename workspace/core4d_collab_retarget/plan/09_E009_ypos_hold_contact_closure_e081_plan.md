# E009 计划：ypos best + robot-side hold-contact 闭环

日期：2026-05-18

## Context

E008 已经解决了 E006/E007 的主要“只旋转不平移”问题：

- proxy support tracking 在 5/5 main full 中通过，best proxy ratio `0.998`；
- best `E008_box025_p2_ypos_k20_vmax2` 达到 object xy ratio `0.724`、object rotation `13.6deg`；
- floor contact `64.7%`、leg-box interference `2.9%` 已接近或优于 E081 margin；
- 但 hand contact 只有 `78.0%`，obj mean/max 仍为 `0.363/0.684m`，未达到 E081 `0.143/0.271m`。

因此 E009 不再继续扫 `support_proxy_max_xy_speed` 或 `kp`。本轮目标是补 robot 手端闭环：在 E008 best `ypos_k20_vmax2` 周围加入 hold-contact reward，测试是否能把手部接触和 object tracking 推过 E081 transport gate。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 E008 best 的主要剩余瓶颈是 robot-side hand/support 闭环 | hold-contact 后 hand contact 上升，object xy ratio/obj error 同步改善 |
| C2 hold-contact 不能以腿/箱干涉或摔倒换 tracking | leg interference 不高于 E081+margin，pelvis stable |
| C3 speed/timebase 不再是主因 | 所有 main 仍需 `E009_proxy_support_tracking_ok=true` |
| C4 eval 继续对齐 E081，而不是 E005 | 保留 E008 的 E081 transport gate 和 majority gate |
| C5 若 hold-contact 无法改善，则下一轮必须进入 contact pad/soft constraint | E009 log 中按结果明确 E010 方向 |

## 实现改动

1. 新增 E009 脚本与 overrides：
   - `workspace/core4d_collab_retarget/scripts/E009/variants.tsv`
   - `workspace/core4d_collab_retarget/scripts/E009/generate_e009_overrides.py`
   - `workspace/core4d_collab_retarget/scripts/run_E009_preprocess.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E009.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E009_remote_tmux.sh`
   - `workspace/core4d_collab_retarget/scripts/run_E009_remote.sh`
   - `workspace/core4d_collab_retarget/scripts/pull_E009_remote_results.sh`
   - `workspace/core4d_collab_retarget/scripts/eval/eval_E009.py`

2. 本轮不改 MJWarp 施力公式，只通过 overrides 调整 hold-contact reward：
   - `hold_contact_start_eval_time=0.64`
   - `hold_contact_end_eval_time=4.08`
   - `hold_contact_require_ref_contact=true`
   - sweep `hold_contact_rew_scale`

## Variants

| Variant | Role | Queue | 目的 |
|---------|------|-------|------|
| `E009_box025_p2_ypos_k20_vmax2_hc05` | main | remote_gpu0 | best + 轻量 hold-contact，防止过强干扰 |
| `E009_box025_p2_ypos_k20_vmax2_hc1` | main | local | best + 标准 hold-contact，主验收 |
| `E009_box025_p2_ypos_k20_vmax2_hc2` | main | remote_gpu0 | best + 更强 hold-contact，测试 hand contact 上限 |
| `E009_box025_p2_ypos_k20_vmax0_hc1` | main | local | 不限速 best-side + hold-contact，对照 vmax2 |
| `E009_box023_p2_xpos_k10_vmax0_hc1` | guard | remote_gpu1 | stable guard + hold-contact，检查是否摔倒 |
| `E009_box023_p2_xpos_k10_vmax0_hc2` | guard | remote_gpu1 | stronger guard hold-contact |

本轮跳过 `box023_xneg`，因为 E008 full 在 `96/272` 卡住；guard 先用 `xpos` 稳定方向。

## E081 对齐成功标准

继承 E008：

1. `E009_proxy_support_tracking_ok`
   - `proxy_xy_disp / ref_support_xy_disp >= 0.95`
   - final proxy-to-ref-support gap `<=0.08m`

2. `E009_reaches_E081_transport_proxy`
   - role 为 main；
   - true-freejoint parity ok；
   - proxy support tracking ok；
   - `case_window_obj_err_mean_m <= 0.20`
   - `case_window_obj_err_max_m <= 0.40`
   - `case_window_sim_contact_frames_pct >= 80%`
   - `case_window_sim_object_floor_contact_frames_pct <= 75%`
   - `E009_object_xy_disp_ratio >= 0.75`
   - `E009_object_rot_deg <= 15deg`

3. `E009_improves_E008_best`
   - 与 `E008_box025_p2_ypos_k20_vmax2` 比：
     - obj mean/max 降低；
     - hand contact `>=80%`；
     - xy ratio `>=0.75` 或至少不低于 E008 best；
     - leg interference `<=7.5% + 5pp`。

## 预授权命令

```bash
bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh __codex_auth_probe__ 0
bash workspace/core4d_collab_retarget/scripts/run_E009_remote.sh __codex_auth_probe__
bash workspace/core4d_collab_retarget/scripts/pull_E009_remote_results.sh __codex_auth_probe__
```

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E009_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E009.py --all

# full
bash workspace/core4d_collab_retarget/scripts/run_E009_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E009.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/pull_E009_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E009.py --all
```

## 决策规则

- 若 hold-contact 提升 hand contact 但 object error 不降：说明仅靠 reward close-loop 不够，E010 转 contact pad / soft contact constraint。
- 若 hold-contact 造成 leg interference 或摔倒：保留 E008 best，E010 改成 contact pad，而不是继续加 reward。
- 若某个 E009 变体达到或接近 `xy>=0.75`、hand `>=80%`、rot `<=15deg`：围绕该配置小范围加密。
