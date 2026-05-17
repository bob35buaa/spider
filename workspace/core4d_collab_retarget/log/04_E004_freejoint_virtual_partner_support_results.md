# E004 结果：true-freejoint 虚拟协作者支持 sweep

日期：2026-05-17

## 状态

E004 Wave A/B 已完成。结论：COM 级虚拟协作者支持没有恢复 true-freejoint 协作搬运。9 个已评估变体都保持 `scene.xml` freejoint 口径：`contact_guidance=false`、`nu=29`、`nq_obj=7`、无 object actuator；但 main `box025_p2` 没有任何变体达到 useful proxy。

准备 E005 时发现一个后验注意点：`_apply_partner_force` 里参考帧 timestep 写死为 `1/30`。这与 `box025_person2_freejoint_legobj` 的 `task_info.ref_dt=1/30` 一致，因此 E004 main `box025` 仍是有效的 COM-force 结果；但 `box023_person2_freejoint_legobj` 没有 `ref_dt` 字段，会使用 config 默认 `0.02`，所以 E004 guard 的 spring timing 可能存在滞后。E005 会显式写入 `partner_force_ref_dt`，并继续测试 support-site 几何。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E004_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/run_E004_remote.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh one 0 E004_box023_p2_s20
bash workspace/core4d_collab_retarget/scripts/train/train_E004.sh one 0 E004_box023_p2_s10_hc
bash workspace/core4d_collab_retarget/scripts/pull_E004_remote_results.sh
```

远程 GPU1 在两个 `box023` run 临近结束时卡住：`box023_s20` 卡在 `250/272`，`box023_s10_hc` 卡在 `194/272`，GPU 利用率为 0 且未写出 trajectory。两者都已终止，并改为本地补跑完成。远程 GPU0 完成了所有 `box025` 变体。

## 配置与 Smoke

本实验生成了 9 个 Wave A/B 变体。Wave C rotation probe 保持注释状态，没有运行。

全部 9 个变体的 smoke 均通过，smoke 使用 `max_sim_steps=4`，只验证 wiring 与 parity，不代表任务成功。eval 显示所有变体 `E004_freejoint_parity_ok=True`。

## 完整指标

| Variant | 角色 | Wave | kp | hold | object mean/max | hand contact | floor contact | leg intf | bottom mean | pelvis min | Proxy |
|---------|------|------|----|------|-----------------|--------------|---------------|----------|-------------|------------|-------|
| `E004_box025_p2_g05` | control | A | 0 | 0 | `0.660/1.288m` | `77.5%` | `71.1%` | `0.0%` | `-0.068m` | `0.770m` | control failed |
| `E004_box025_p2_s10` | main | A | 10 | 0 | `0.770/1.566m` | `91.9%` | `83.8%` | `9.2%` | `-0.074m` | `0.778m` | useful False |
| `E004_box025_p2_s20` | main | A | 20 | 0 | `0.771/1.577m` | `90.2%` | `82.1%` | `6.9%` | `-0.075m` | `0.775m` | useful False |
| `E004_box025_p2_s40` | main | A | 40 | 0 | `0.769/1.551m` | `90.2%` | `82.1%` | `0.6%` | `-0.077m` | `0.772m` | useful False |
| `E004_box025_p2_s20_hc` | main | B | 20 | 1 | `0.770/1.552m` | `89.0%` | `79.2%` | `10.4%` | `-0.076m` | `0.775m` | useful False |
| `E004_box025_p2_s40_hc` | main | B | 40 | 1 | `0.768/1.549m` | `91.9%` | `68.2%` | `15.6%` | `-0.068m` | `0.779m` | useful False |
| `E004_box023_p2_s10` | guard | A | 10 | 0 | `0.814/1.494m` | `65.3%` | `78.7%` | `0.0%` | `0.024m` | `0.687m` | guard stable True |
| `E004_box023_p2_s20` | guard | A | 20 | 0 | `0.806/1.505m` | `60.7%` | `70.0%` | `0.0%` | `0.040m` | `0.684m` | guard stable True |
| `E004_box023_p2_s10_hc` | guard | B | 10 | 1 | `0.812/1.471m` | `67.3%` | `62.7%` | `10.7%` | `0.018m` | `0.040m` | guard stable False |

汇总：

```json
{
  "num_results": 9,
  "num_freejoint_parity_ok": 9,
  "num_main_useful_proxy": 0,
  "num_main_strong_proxy": 0,
  "num_guard_stable_proxy": 2
}
```

## 可视化观察

关键帧拼图：

- `workspace/core4d_collab_retarget/results/E004/contact_sheet_main.jpg`
- `workspace/core4d_collab_retarget/results/E004/contact_sheet_guard.jpg`

main `box025` 观察：

- gravity-only (`g05`) 不能形成 transport；sim object 仍明显滞后 ref，且基本仍是 floor-supported。
- translation spring (`s10/s20/s40`) 能让机器人保持在箱子附近，并带来较高手部接触率，但箱子仍处在同一类失败模式：被拖/推在地面附近，而不是沿 ref 轨迹被搬运。
- hold-contact (`s20_hc/s40_hc`) 没有改善 object tracking 或 carry 语义。`s40_hc` 虽然把 floor-contact metric 降到 `68.2%`，但 leg interference 升到 `15.6%`，不是干净收益。

guard `box023` 观察：

- `s10/s20` 机器人保持站立且没有腿箱干涉，但 object tracking 仍差，mean 约 `0.81m`；物体更像被放下/留在后方，而不是被运输。
- `s10_hc` 视觉上无效：机器人在 f204 前后摔倒，对应 `post2_pelvis_z_min=0.040m` 与 `guard stable False`。

## Claims 验证

| Claim | 结果 |
|-------|------|
| C1 虚拟 translational support 是缺失因素 | 当前 COM-force 实现下拒绝。没有 main 变体超过 E003 best (`0.400/0.795m`)；所有 main spring 都停在约 `0.77/1.55m`。 |
| C2 gravity-only 不充分 | 支持。`g05` 失败 (`0.660/1.288m`)，且仍是 floor-supported。 |
| C3 有效 translational spring 区间在 `kp=10-40` | 拒绝。`kp=10/20/40` 都相近且较差；更高 kp 主要改变 artifact，没有带来 transport。 |
| C4 不能让虚拟协作者独自完成任务 | 支持为诊断要求。本实验中虚拟协作者本身也没有完成任务；高 hand contact 不等于因果搬运。 |
| C5 rotation torque 不应进入主线 | 支持。位置/支撑目标已经明显失败，因此 Wave C rotation probe 没有依据；不应重复 E030 式 torque sweep。 |

## 解释

E004 表明：在 object COM 上施加虚拟协作者外力，对 CORE4D 协作搬运来说仍然太弱或语义不足。它可以提供一定 support，但没有创造缺失的交互几何：机器人仍缺少稳定抓握/支撑关系，无法把物体轨迹通过真实接触传递出来。

主要问题不是手是否靠近物体。多个失败 main 变体都有 `89-92%` hand contact。真正缺失的是 force/contact 语义：协作者支撑哪里、机器人应该支撑哪里，以及 object orientation/support point 如何与双手耦合。

## 下一步

不要运行 E004 Wave C rotation probes。位置/支撑目标已经失败，且旧 E030 表明 torque 容易造成 NaN/正反馈。

下一步应离开 COM xfrc，转向以下方向之一：

1. 在物体协作者一侧施加显式 support-point/site force，而不是 COM-only；
2. dual-agent/proxy partner，通过接触或连接约束形成第二支撑者；
3. equality/contact constraint，使物体拥有物理上有意义的第二支撑点，同时保持优化目标为单机器人 + 人类协作者。

E005 优先测试显式 support geometry 或 dual-agent proxy，不继续单纯调 `partner_force_spring_kp`。
