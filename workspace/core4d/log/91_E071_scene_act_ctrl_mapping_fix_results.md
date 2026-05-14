# E071 结果: 修复 scene_act ctrl mapping 后 early drift 消失，但 post-2s 搬放阶段失败

## 实验目的

验证 E070 定位出的根因：`examples/run_mjwp.py` 在 `contact_guidance` 且 `ctrl_ref.shape[1] != config.nu` 时，把 `qpos_ref[:, :config.nu]` 误当成 actuator ctrl，导致 floating-base qpos 混入 robot actuator ctrl。

E071 改动为保留原始 29-dim robot ctrl，并让 scene_act ref conversion 补齐末 6 维 object actuator ctrl。

## 本次改动

- 修改 `examples/run_mjwp.py`
  - 删除 `ctrl_ref = qpos_ref[:, : config.nu]` fallback。
  - 当 ctrl 维度不等于 `nu` 时仅记录日志，保留 raw ctrl。
  - 后续 scene_act conversion 继续生成 35-dim ctrl: 前 29 维 robot ctrl，末 6 维 object slide/euler ctrl。
- 新增配置: `examples/config/override/core4d_e071w02_box023.yaml`
- 新增训练入口: `workspace/core4d/scripts/train/train_E071.sh`
- 新增评估脚本: `workspace/core4d/scripts/eval/eval_E071.py`

## 运行命令

```bash
bash workspace/core4d/scripts/train/train_E071.sh 0
```

## 输出

- `workspace/core4d/results/E071/E071W02_box023.npz`
- `workspace/core4d/results/E071/E071W02_box023.mp4`
- `workspace/core4d/results/E071/eval_summary.csv`
- `workspace/core4d/results/E071/keyframes/f010_t020.jpg`
- `workspace/core4d/results/E071/keyframes/f030_t060.jpg`
- `workspace/core4d/results/E071/keyframes/f040_t080.jpg`
- `workspace/core4d/results/E071/keyframes/f075_t150.jpg`
- `logs/E071/E071W02_box023.log`

## 关键日志确认

启动阶段确认修复路径生效：

```text
Preserving raw ctrl reference for contact guidance (ctrl dims: 29 -> 35); scene_act conversion will pad object controls when applicable.
E027b: converted ref nq 43 -> 42 (quat->XZY euler, body_pos=[0.155, -0.124, 0.31])
```

末尾 EGL destructor 有 non-fatal ignored exception，但进程 exit 0，npz/mp4/eval_summary 均已保存。

## 指标

| 指标 | E069-W02 | E071-W02 | 判定 |
|------|----------|----------|------|
| t=0.017 yaw err | 12.40 deg | 0.574 deg | PASS |
| t=0.033 yaw err | 22.16 deg | 1.075 deg | PASS |
| vs E070 orig t=0.017 | N/A | -0.0004 deg | PASS |
| vs E070 orig t=0.033 | N/A | +0.0005 deg | PASS |
| warmup robot ctrl diff | 0.0 rad | 0.0 rad | PASS |
| warmup object ctrl diff | 0.0 | 0.0 | PASS |
| B1 max foot z [0,2s] | 0.222 m | 0.069 m | PASS |
| B2 single-foot runs | 未解决 | 2 | 数值触发但视频无初始 lunge |
| pelvis_min_intent | 0.578 m | 0.674 m | PASS |
| pelvis_mean_intent | 未记录 | 0.723 m | PASS |
| post-2s obj_err max/mean | 未统计 | 0.308 / 0.133 m | FAIL |
| first post-2s obj_err > 25cm | 未统计 | frame 100 / 2.00s | FAIL |
| post-2s pelvis body z min | 未统计 | 0.200 m | FAIL |
| first post-2s pelvis z < 45cm | 未统计 | frame 166 / 3.32s | FAIL |

`eval_summary.csv`:

```text
B1_pre_contact_max_foot_z_m=0.0690272
T=272
yaw_err_t0017_deg=0.5735647
yaw_err_t0033_deg=1.0754782
robot_ctrl_max_diff_warmup=0.0
object_ctrl_max_diff_warmup=0.0
pelvis_min_intent=0.6736118
pelvis_mean_intent=0.7231964
post2_obj_err_max_m=0.3079629
post2_obj_err_mean_m=0.1330441
post2_pelvis_body_z_min_m=0.2002778
first_post2_obj_err_gt_25cm_time_s=2.0
first_post2_pelvis_z_lt_45cm_time_s=3.32
```

## 视频复核

关键帧:

- t=0.20s: sim 与 ref 站姿方向一致，无 E069 初始大转身。
- t=0.60s / 0.80s: sim 正常弯腰接近箱子，无单脚 lunge。
- t=1.50s: sim 已按 ref 搬箱，整体姿态稳定。
- t=2.00s: sim 已开始落后/偏离 ref，物体位置误差约 0.31m。
- t=2.30s: sim 手部没有稳定托住箱子，箱体相对 ref 仍明显偏移。
- t=2.60s: 箱子开始离手，robot 姿态向后/侧向补偿。
- t=2.90s: 箱子接近地面，sim 用腿/身体顶住箱子，已偏离真实搬放动作。
- t=3.20s: robot 坐到/压到箱子上。
- t=3.60s / 4.00s: robot 摔倒，足部高度异常，pelvis body z 最低到 0.20m。

## 结论

E071 验证 E070 对 **early drift** 的根因完全成立：box023 初始 yaw/lunge 不是 MJWarp physics、不是 object actuator gains、不是 CEM warmup，而是 scene_act 下 ctrl_ref preprocessing 把 `qpos_ref[:, :nu]` 当成 actuator ctrl 的映射错误。

修复后：

- early yaw 从 `12.40 / 22.16 deg` 降到 `0.57 / 1.08 deg`；
- 与 E070 `orig_ctrl` parity 逐位一致；
- pre-contact foot lift 从 `0.222m` 降到 `0.069m`；
- 保存链路正常。

但 E071 **不是整体成功**。2s 后进入搬运/放下阶段，机器人没有稳定拿住箱子，并在约 3.3s 后摔倒。之前只看 0-1.5s 的结论过于乐观，需要按 post-2s failure 重新进入下一步诊断。

## 下一步

E072 不应先做 box025 regression，也不应继续调 warmup。当前主问题已经转移到 post-2s hold/place failure，应做定向诊断：

1. 标注 2.0-3.6s 的 hand-object contact、hand distance、object tracking、pelvis/body stability。
2. 区分失败来自 grasp/hold contact 丢失，还是 putdown 阶段 CEM 稳定性崩溃。
3. 如果是 hold contact 丢失，优先做 hold/contact reward 或 object-hand coupling 诊断。
4. 如果是 putdown 稳定性崩溃，再考虑 CEM delta clamp / stability prior。
