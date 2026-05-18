# E013-E016 Agent Execution Plan

日期：2026-05-19

本文件把 `docs/01_direction_review_2026-05-18.md` 中的路线压缩成可执行 agent 计划。E012 已完成，结论与预期一致：spring / multi-point force 范式不再继续扫参。后续按 E013 -> E014/E014b -> E015/E015b(/E015c) -> E016 的条件链推进。

## 总目标

建立 true-freejoint object 下“接近 E081”的软 target，然后优先验证 COLA 的位置约束范式：

1. E013 先做 true-freejoint object oracle，量化 E081 gate 的可达软阈值。
2. E014 验证 COLA 特征 B：kinematic support body + 6-DoF / weld 位置约束，测试能否消除 E011/E012 的 xy lag。
3. E014b 只在 E014 明显改善但未过门时触发，继续扫 B-only stiffness / solimp。
4. E015 在 B 有效后增加 COLA 特征 A：dynamic support body + PD command。
5. E015b/E015c 只在 E015 继续改善但未过门时触发，扫 mass / PD / joint 参数。
6. E016 只作为全套 COLA sweep 后的回退或正交验证，不能在 E014 一轮失败后直接跳转。

## 当前已知事实

- E081 baseline 是 `scene_act` actuator-guided object，不是真 freejoint 搬运；但它仍作为软参考目标。
- E011 best `E011_box025_p2_com_xyz_k100` 在 hand / floor / leg / xy ratio / rotation 上接近或优于 E081，主要残差是 xy 位置滞后：obj `0.340/0.673m`。
- E012 dual-point force 未解决 E011 残差，强 coupling 产生 rotation shortcut；main `0/6` 过 E081 transport，guard `0/2` 稳定。
- 因此下一步必须离开 spring force 范式，先用 oracle 定义软 target，再测位置约束。

## E013: true-freejoint object oracle

### 目的

在 true-freejoint scene 上把 object 按 reference 直接放到位，CEM 只优化 robot；输出 hand contact / floor / leg interference / pelvis / walking 等指标的 oracle 上限。该结果用于 finalize E014 之后的软 target。

### 实现口径

现有 `object_pd_override` 只适用于 `scene_act` 的 6 维 object actuator，不适用于 freejoint scene。E013 应使用显式 freejoint object kinematic oracle：

- `contact_guidance=false`
- `scene_name=scene`
- `nq_obj=7`, `nu=29`
- `object_action_dims=0`, `object_actuator_ids=[]`
- `object_kinematic_override=true`
- 每个 step 前后把 object freejoint qpos/qvel 写为插值后的 ref object qpos/qvel

### 变体

| Variant | Case | Role | 说明 |
|---------|------|------|------|
| `E013_box025_p2_obj_oracle` | `box025_person2_freejoint_legobj` | main | E081 main 的 true-freejoint oracle |
| `E013_box023_p2_obj_oracle` | `box023_person2_freejoint_legobj` | guard | E081 guard 的 true-freejoint oracle |

### 成功输出

- E013 plan / scripts / eval / log
- `workspace/core4d_collab_retarget/results/E013/comparison.csv`
- `workspace/core4d_collab_retarget/results/E013/aggregate_summary.json`
- 两个 full NPZ、视频、关键帧
- E014 soft target JSON：`workspace/core4d_collab_retarget/results/E013/e014_soft_targets.json`

### E014 软 target 定义

对 main：

```
obj_mean_target = max(E081_obj_mean + 0.05, E013_oracle_obj_mean + 0.05)
obj_max_target  = max(E081_obj_max  + 0.10, E013_oracle_obj_max  + 0.10)
hand_target     = min(E081_hand_pct, E013_oracle_hand_pct) - 5pp
floor_target    = max(E081_floor_pct, E013_oracle_floor_pct) + 10pp
leg_target      = max(E081_leg_intf_pct, E013_oracle_leg_intf_pct) + 5pp
```

同时保留硬诊断门：

- lag-free：main obj mean 必须显著低于 E011 k100 的 `0.340m`；若 E014 best `>=0.28m`，判定位置约束未消除 lag。
- 推 vs 搬：leg-object contact `<=15%` 且 floor contact `<=70%`。
- partner overdrive：partner/support 反力 mean `>200N` 或 max `>800N` 标记为 overdrive。

## E014: COLA B-only kinematic support + weld/equality

### 触发

E013 完成并产出软 target 后启动。

### 目标

把 partner-object coupling 从 force feedback 换成位置约束，优先验证“位置约束是否能消除 xy lag”。先不加 dynamic support body，减少工程变量。

### 初始 variants

| Variant | solref timeconst | gravity | hold-contact | Role |
|---------|------------------|---------|--------------|------|
| `E014_box025_p2_jointB_t02` | 0.02 | 0.5 | no | main |
| `E014_box025_p2_jointB_t05` | 0.05 | 0.5 | no | main |
| `E014_box025_p2_jointB_t02_g08` | 0.02 | 0.8 | no | main |
| `E014_box025_p2_jointB_t02_hc1` | 0.02 | 0.5 | yes | main |
| `E014_box023_p2_jointB_t02` | 0.02 | 0.5 | no | guard |
| `E014_box023_p2_jointB_t02_hc1` | 0.02 | 0.5 | yes | guard |

### 判定

- 过 E013 soft target 且推/搬门通过：B-only 成立，进入 pipeline 对接或 E015 验证完整 COLA。
- 明显改善但未过门：触发 E014b。
- 改善有限或数值不稳：仍先分析失败，再决定是否加 A；不能直接跳 E016。

## E014b: B-only stiffness sweep

触发条件：E014 best obj mean 从 E011 `0.340m` 压到 `<=0.27m`，但未过 E013 soft target。

预算 4-5 runs：`t01`, `t01_g08`, `t01_hc2`, `t02_solimp_tight`, `t005_unstable_probe`。若提升 `<0.02m`，判定 B-only 饱和。

## E015: COLA B+A dynamic support + PD

触发条件：E014/E014b 证明 B 有效，或者 B 改善明显但仍需 compliance / effort metric。

目标：把 kinematic support 升级为 dynamic support body + PD command，同时保留 6-DoF joint coupling。

初始 variants：`m2_kp500`, `m1_kp500`, `m2_kp1000`, `box023_m2_kp500`。判定门沿用 E013 soft target，并增加 partner effort mean `<=100N`、max `<=250N`。

## E015b / E015c

触发条件：E015 相对 E014 best 又改善 `>=0.03m` 但未过 soft target。

先扫 mass + PD kp/kd，预算 `<=10` runs；若仍有明确趋势再进 E015c 扩 joint limits / friction / mass。

## E016: kinematic + contact partner mocap hands

仅在以下条件启动：

1. E014 + E014b + E015 + E015b(/E015c) 全部未过 soft target，且 best 仍未接近 target；或
2. COLA 已有 work/near-work 配置，需要用 contact paradigm 做正交验证。

禁止触发：E014 一轮失败就跳 E016；E015 数值不稳就跳 E016。

E016 复用 holosoma v4.x 几何经验：双 capsule mocap hands，radius `7cm`，length `15cm`，palm offset `12cm`，纯 ref-driven relocalization，启用 hand-object contact pair，不用 weld。

## 立即下一步

1. 创建 E013 计划。
2. 实现 freejoint object kinematic oracle 的显式配置口径。
3. 生成 E013 overrides / train / eval 脚本。
4. 先跑 E013 smoke；通过后跑 main + guard full。
5. 写 E013 log，更新 tracker，并生成 `e014_soft_targets.json`。
