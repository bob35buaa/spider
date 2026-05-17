# E006 Plan: COLA-style support-body proxy for true-freejoint retargeting

日期：2026-05-17

状态：计划中。E005 已证明直接 object COM / support-site 外力路线到达上限；E006 迁移 COLA 的 support body 范式，但首轮不直接新增 freejoint body，避免破坏现有 MJWarp 管线的 object-last 布局。

## Context

COLA 论文里的虚拟协作者不是一个直接拉 object COM 的外力，而是位于物体远端的 support body。它由速度/高度/yaw 命令控制，并通过 6-DoF 连接把力传给物体；support-object interaction force 还被用作 human effort 指标。

当前 `core4d_collab_retarget` 管线有一个硬约束：`humanoid_object` true-freejoint 场景默认 `nq=43,nv=41,nu=29,nq_obj=7`，且 object freejoint 位于 `qpos` 末尾。`spider/simulators/mjwp.py` 的 reward、termination、DR offset、kinematic override、eval 脚本都依赖 `qpos[:, -nq_obj:]` 指向 object。如果直接给 XML 加一个独立 freejoint support body，会导致 model `nq/nv` 与 `trajectory_kinematic.npz` 不一致，或者让 object 不再是末尾 slice。因此 E006 首轮采用“虚拟 support body/controller + soft connector wrench”：

- object 仍是真 freejoint passive body；
- robot action 仍只有 G1 的 29 维；
- support proxy 不进入 `qpos/qvel/ctrl`，而是在 `step_env` 内根据 reference support trajectory 计算 proxy pose/velocity；
- proxy 与 object-local support point 之间用 spring-damper connector 传力，写入 object body 的 `xfrc_applied`；
- 记录 support proxy force/torque、proxy/support point gap 和 object end-height 指标，避免把结果误判为 object actuator。

这不是最终的 dynamic support body，但它比 E005 更接近 COLA：力不再直接由 `ref_point-current_point` 一步生成，而是由一个独立 proxy trajectory 和 connector 生成，并明确输出 partner effort。

## Claims

| Claim | 验证证据 |
|-------|----------|
| C1 support proxy 可以在不破坏 true-freejoint parity 的情况下接入 MJWarp | smoke/full eval 均满足 `contact_guidance=false`、`scene_name=""`、`nu=29`、`nq_obj=7`、object actuator empty |
| C2 proxy connector 比 E005 direct support-site 更接近 COLA support body 语义 | NPZ 保存 `support_proxy_force/torque/pos/support_point_pos`；eval 输出 partner effort 与 connector gap |
| C3 速度/高度命令式 proxy 不能只靠“虚拟人替机器人搬” | useful 判定要求 object tracking 改善同时 hand contact 不低于 `80%`，且 partner force 均值/峰值不过大 |
| C4 main `box025_p2` 至少应超过 E005 direct support-site | 同侧同 kp 的 obj mean 或 floor contact 明显优于 E005 `yneg_s20/ypos_s20`；否则 support-body 近似无效 |
| C5 guard `box023_p2` 不应被 proxy 破坏 | pelvis min `>=0.55m`、leg interference `<=5%`，并记录若远程再次卡住的具体位置 |

## Code Changes

1. `spider/config.py`
   - 新增 `support_proxy_enabled` 开关。
   - 新增 `support_proxy_point_local`、`support_proxy_gravity_scale`、`support_proxy_connector_kp/kd`、`support_proxy_xy_velocity_scale`、`support_proxy_max_xy_speed`、`support_proxy_height_tau`、`support_proxy_ref_dt`、force/torque clamp。

2. `spider/simulators/mjwp.py`
   - 在 `setup_env` 中为 true-freejoint object 预计算 support proxy trajectory：
     - object ref support site = `ref_obj_pos + R_ref * point_local`；
     - XY 按 reference support-site velocity command 积分，可 clip 到 COLA 风格速度范围；
     - Z 按 reference height 低通跟踪；
     - proxy velocity 由 proxy position 差分得到。
   - 新增 `_apply_support_proxy_force`：
     - current support point = object COM + `R_obj * point_local`；
     - point velocity = object linear velocity + `omega x r`；
     - connector force = gravity share + `kp*(proxy_pos-current_point) + kd*(proxy_vel-current_point_vel)`；
     - object wrench = `F, r x F`，写入 `xfrc_applied[:, object_body_id, :6]`；
     - 保存 last force/torque/proxy/support point 到 env，供轨迹 NPZ 记录。
   - 新增 `get_support_proxy_state`，让 `examples/run_mjwp.py` 在 committed rollout step 后写入 `trajectory_mjwp.npz`。

3. `examples/run_mjwp.py`
   - 若 `support_proxy_enabled`，每个 committed sim step 记录：
     - `support_proxy_force`
     - `support_proxy_torque`
     - `support_proxy_pos`
     - `support_proxy_vel`
     - `support_point_pos`
     - `support_point_vel`

4. E006 scripts
   - `scripts/E006/variants.tsv`
   - `scripts/E006/generate_e006_overrides.py`
   - `scripts/run_E006_preprocess.sh`
   - `scripts/train/train_E006.sh`
   - `scripts/train/train_E006_remote_tmux.sh`
   - `scripts/run_E006_remote.sh`
   - `scripts/pull_E006_remote_results.sh`
   - `scripts/eval/eval_E006.py`

## Variant Grid

所有 variant 使用 E002 true-freejoint leg-object tasks，保持 `scene_name=""`、`contact_guidance=false`、`object_action_dims=0`、`object_actuator_ids=[]`、`partner_force_scale=0`。

| Variant | Task | point local | kp | xy vel scale | max xy speed | hold | Role |
|---------|------|-------------|----|--------------|--------------|------|------|
| `E006_box025_p2_yneg_k20_v1` | `box025_person2_freejoint_legobj` | `[0,-0.38,0.30]` | 20 | 1.0 | 0.8 | 0 | main anchor |
| `E006_box025_p2_yneg_k40_v1` | same | `[0,-0.38,0.30]` | 40 | 1.0 | 0.8 | 0 | stiffness |
| `E006_box025_p2_yneg_k20_v05` | same | `[0,-0.38,0.30]` | 20 | 0.5 | 0.8 | 0 | less actuator-like velocity |
| `E006_box025_p2_ypos_k20_v1` | same | `[0,0.38,0.30]` | 20 | 1.0 | 0.8 | 0 | side ablation |
| `E006_box025_p2_yneg_k20_v1_hc` | same | `[0,-0.38,0.30]` | 20 | 1.0 | 0.8 | 1 | robot participation |
| `E006_box023_p2_xneg_k10_v1` | `box023_person2_freejoint_legobj` | `[-0.16,0,0.10]` | 10 | 1.0 | 0.8 | 0 | guard |
| `E006_box023_p2_xpos_k10_v1` | same | `[0.16,0,0.10]` | 10 | 1.0 | 0.8 | 0 | guard side ablation |

## Parallel Execution

| GPU | Queue |
|-----|-------|
| local GPU0 | `box025_yneg_k20_v1` -> `box025_yneg_k20_v1_hc` |
| remote GPU0 | `box025_yneg_k40_v1` -> `box025_yneg_k20_v05` -> `box025_ypos_k20_v1` |
| remote GPU1 | `box023_xneg_k10_v1` -> `box023_xpos_k10_v1` |

## Commands

```bash
bash workspace/core4d_collab_retarget/scripts/run_E006_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E006.sh smoke 0
git add <E006 files> && git commit -m "exp(core4d_collab_retarget): E006 support proxy setup" && git push
bash workspace/core4d_collab_retarget/scripts/train/train_E006.sh local_wave 0
bash workspace/core4d_collab_retarget/scripts/run_E006_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E006_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E006.py --all
```

## Success Criteria

| Area | Useful | Strong |
|------|--------|--------|
| Parity | every result `nu=29`, `nq_obj=7`, no object actuator | same plus `support_proxy_enabled=true` and `partner_force_scale=0` |
| Main tracking | beats E005 same-side support-site by `>=0.10m` mean or reduces floor contact by `>=15pp` | reaches/approaches E003 best: obj mean `<=0.40m`, max `<=0.80m` |
| Robot participation | hand contact `>=80%` or HC variant improves contact without worsening object err | video/keyframes show robot and proxy share support |
| Partner effort | horizontal force mean/peak recorded and not explosive | mean `<80N`, max `<160N`, torque max `<25Nm` |
| Guard | pelvis min `>=0.55m`, leg interference `<=5%` | stable and object mean improves vs E002/E005 guard |

## Decision Rules

- If E006 improves over E005 but force is too high, next experiment should penalize partner effort or reduce connector stiffness rather than increasing kp.
- If E006 improves tracking but hand contact collapses, next experiment should combine support proxy with stronger hold/contact reward or robot-side contact target.
- If E006 does not improve over E005 support-site, stop force-only proxy work and attempt an XML-level mocap contact pad / soft equality proxy with a smaller smoke first.
- If adding proxy metrics or CUDA graph interaction causes instability, revert to metric-free smoke first, then reintroduce recording after the dynamics path is stable.
