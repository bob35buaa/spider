# E015 Plan: COLA A+B dynamic support + PD command

日期：2026-05-19

## 背景

E014 已证明 COLA-B 位置约束成立：6/6 full 通过 E013 soft target，4/4 main lag-free 且 push-vs-carry 通过，2/2 guard stable。根据 `docs/03_agent_execution_plan_E013_E016.md`，E014b 不触发，下一步进入 E015：在 B-only 的 support-object soft weld 基础上加入 COLA 特征 A，把 kinematic/mocap support anchor 升级为 dynamic support body + PD command，并记录 partner effort。

E015 的目标不是再证明 object trajectory 可达；E014 已证明。E015 要回答：把 support 从 kinematic mocap anchor 换成受 PD 驱动的 dynamic body 后，是否仍能接近 E014/E013 soft target，并给出合理的 partner effort / compliance 诊断。

## 关键实现口径

E015 不使用 freejoint support body。新增 support body 使用 3 个 slide + 3 个 hinge 标量关节：

- model dims 从 E014 的 `nq=43/nv=41/nu=29` 变为 `nq=49/nv=47/nu=29`。
- object 仍在 qpos 最后 7 维，因此 `config.nq_obj=7` 和 object eval 口径保持一致。
- support qpos 插入在 robot qpos 和 object qpos 之间：`robot(36) + support(6) + object(7)`。
- support qvel 插入在 robot qvel 和 object qvel 之间：`robot(35) + support(6) + object(6)`。
- `ctrl_ref` 仍为 robot 29 维；support 不新增 actuator，不由 CEM 直接优化。
- support PD 通过 `qfrc_applied` 写 6 个 generalized forces，跟踪由 object ref pose + object-local support point 派生出的 support target。
- soft weld equality 仍保留：`body1="object"`，`body2="support_dynamic_anchor"`，`relpose="<support_point_local> 1 0 0 0"`。

该口径避免把 support freejoint qpos/quaternion 混入现有 humanoid reward 的 qpos/nv 映射，同时仍是动态 body，不是 mocap body。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 E015 保持 object true-freejoint，不回退到 `scene_act` / object actuator | eval 检查 `contact_guidance=false`、`nq_obj=7`、object last 7、object actuator empty、`nu=29` |
| C2 E015 support 是 dynamic 6-DoF body，不是 mocap/kinematic oracle | scene 编译检查 `support_dynamic_anchor` 非 mocap、6 个 joints、`nq=49/nv=47/nu=29` |
| C3 support PD 通过 effort 传递，不使用 direct object wrench 或 object kinematic override | config 检查 partner force=0、object_kinematic=false、support mode=`dynamic_weld` |
| C4 dynamic support + soft weld 仍能过 E013 soft target | main/guard 指标对齐 E014 target |
| C5 partner effort 合理 | support PD force mean `<=100N`、max `<=250N`，torque mean/max 单独记录；若打 clamp 标记 overdrive |
| C6 guard 稳定 | guard pelvis min `>=0.55m`，floor/leg/hand soft target 通过 |

## 初始变体

| Variant | Case | Role | support mass | pos kp | pos kd | rot kp | rot kd | solref | 说明 |
|---------|------|------|--------------|--------|--------|--------|--------|--------|------|
| `E015_box025_p2_m2_kp500` | `box025_person2_freejoint_legobj` | main | `2.0kg` | `500` | auto critical | `80` | auto critical | `0.02` | 默认动态 support |
| `E015_box025_p2_m1_kp500` | `box025_person2_freejoint_legobj` | main | `1.0kg` | `500` | auto critical | `80` | auto critical | `0.02` | 轻 support |
| `E015_box025_p2_m2_kp1000` | `box025_person2_freejoint_legobj` | main | `2.0kg` | `1000` | auto critical | `120` | auto critical | `0.02` | 更硬 PD |
| `E015_box023_p2_m2_kp500` | `box023_person2_freejoint_legobj` | guard | `2.0kg` | `500` | auto critical | `80` | auto critical | `0.02` | guard 稳定性 |

Support points 继承 E014：

- main: `[0.0, 0.38, 0.30]`
- guard: `[0.16, 0.0, 0.10]`

## 实现计划

1. 新增 E015 scene/data generator：
   - 基于 task `scene.xml` 生成 `scene_e015_dyn_<slug>.xml`。
   - 在 `object` body 前插入 `support_dynamic_anchor`，包含 3 slide + 3 hinge joints 和 non-colliding geom/site。
   - 添加 `e015_support_weld` equality。
   - 生成 augmented trajectory NPZ 到 `workspace/core4d_collab_retarget/results/E015/data/<variant>/trajectory_kinematic.npz`。
   - 编译验证 `nq=49`、`nv=47`、`nu=29`，且 object body joint 仍是最后 7 qpos。

2. 扩展 runtime：
   - `support_proxy_mode` 增加 `dynamic_weld`。
   - `_load_support_proxy` 继续预计算 support target pos/quat，并新增 support target Euler / velocity。
   - `step_env` 在 dynamic mode 下写 `qfrc_applied` 到 `support_dynamic_anchor` 的 6 个 joint dof。
   - 保存 diagnostics：support PD force/torque、support body pos/vel、support point pos/vel、PD target index、support gap。

3. 新增 E015 scripts/eval：
   - `scripts/E015/variants.tsv`
   - `scripts/E015/generate_e015_assets.py`
   - `scripts/E015/generate_e015_overrides.py`
   - `scripts/run_E015_preprocess.sh`
   - `scripts/train/train_E015.sh`
   - `scripts/run_E015_remote.sh` / pull / remote tmux
   - `scripts/eval/eval_E015.py`

4. 运行顺序：
   - 静态检查 + preprocess。
   - 4-step smoke 先验证 qpos/qvel/ctrl dims、dynamic scene、PD diagnostics。
   - full：本地跑 default main，远程两卡跑剩余 3 条。
   - full 后总评、视频关键帧检查、log/tracker/progress 更新。

## 成功标准

沿用 E013/E014 soft target：

Main：

- obj mean `<=0.193m`
- obj max `<=0.371m`
- hand contact `>=73.6%`
- floor contact `<=69.5%`
- leg-object interference `<=12.5%`
- lag-free obj mean `<0.28m`
- push-vs-carry：floor `<=70%` 且 leg `<=15%`

Guard：

- obj mean `<=0.214m`
- obj max `<=0.417m`
- hand contact `>=61.7%`
- floor contact `<=44.7%`
- leg-object interference `<=7.7%`
- pelvis min `>=0.55m`

E015 additional：

- support PD force mean `<=100N`
- support PD force max `<=250N`
- support dynamic config/parity ok
- no object kinematic override, no partner-force spring, no object actuator

## 决策规则

- 若 E015 过 soft target 且 effort 合理：COLA A+B 成立，优先进入 pipeline 对接；E015b 不触发。
- 若 E015 object 仍过 target 但 effort overdrive：记录为 B-only 可用、A 需要 effort tuning；触发 E015b mass/PD sweep。
- 若 E015 明显差于 E014 且未过 target：先分析 dynamic support lag / PD saturation，再决定 E015b；不跳 E016。
- 只有在 E015/E015b 全套失败后，或需要正交验证时，才考虑 E016 contact partner mocap hands。
