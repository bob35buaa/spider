# E014 Plan: COLA-B kinematic support + soft weld/equality

日期：2026-05-19

## 背景

E013 已完成 true-freejoint object oracle。它证明 `box025_person2_freejoint_legobj` / `box023_person2_freejoint_legobj` 的 reference object trajectory 在 true-freejoint 数据与 eval 口径下本身可达：main obj `0.011/0.038m`，guard obj `0.017/0.064m`。因此 E014 不再放宽为“freejoint 本身不可达”，而是按 `docs/03_agent_execution_plan_E013_E016.md` 测试 COLA 特征 B：kinematic support body + 6-DoF / weld equality position constraint。

E014 不能直接使用旧 `scene_weld.xml` 口径。旧口径把 `object_target` mocap body 放在 object COM，并以 `relpose=0 0 0 1 0 0 0` 直接跟踪完整 object ref pose，等价于接近 E013 的对象位姿 oracle。E014 的验证口径必须是 partner/support 侧锚点：

- object 仍是真 freejoint：`contact_guidance=false`，`scene_name` 指向 freejoint XML，`nq_obj=7`，`nu=29`。
- support body 是 mocap body，不新增 actuator，不新增 object action dim。
- soft weld 连接 `object` 与 support mocap body，`relpose` 为 object-local support point，而不是 COM 零偏移。
- support mocap body 的位置来自 `qpos_ref` object pose 加 local support point：`p_support_ref = p_obj_ref + R_obj_ref * p_local`。
- mocap orientation 跟随 object ref orientation；这保留 6-DoF weld 测试，但通过 offset anchor 明确区别于 COM object oracle。

## 实验问题

E011 COM spring 可把 main xy ratio 推到 `0.905`，但 obj 仍 `0.340/0.673m`；E012 dual-point force 没有解决 residual，反而产生 rotation shortcut。E014 要回答：

1. 位置约束能否把 main obj mean 从 E011/E012 的 `0.30-0.34m` 区间压到 E013 soft target 附近？
2. soft weld 是否能保持 push-vs-carry 指标健康，不靠 floor/leg 或对象旋转捷径通过？
3. guard case 是否稳定，避免 E012 的 guard collapse？

## 变体

| Variant | Case | Role | solref timeconst | gravity | hold-contact | support point |
|---------|------|------|------------------|---------|--------------|---------------|
| `E014_box025_p2_jointB_t02` | `box025_person2_freejoint_legobj` | main | `0.02` | `0.5` | no | `[0.0, 0.38, 0.30]` |
| `E014_box025_p2_jointB_t05` | `box025_person2_freejoint_legobj` | main | `0.05` | `0.5` | no | `[0.0, 0.38, 0.30]` |
| `E014_box025_p2_jointB_t02_g08` | `box025_person2_freejoint_legobj` | main | `0.02` | `0.8` | no | `[0.0, 0.38, 0.30]` |
| `E014_box025_p2_jointB_t02_hc1` | `box025_person2_freejoint_legobj` | main | `0.02` | `0.5` | yes | `[0.0, 0.38, 0.30]` |
| `E014_box023_p2_jointB_t02` | `box023_person2_freejoint_legobj` | guard | `0.02` | `0.5` | no | `[0.16, 0.0, 0.10]` |
| `E014_box023_p2_jointB_t02_hc1` | `box023_person2_freejoint_legobj` | guard | `0.02` | `0.5` | yes | `[0.16, 0.0, 0.10]` |

`gravity` 通过 `support_proxy_gravity_scale` 保持可选：E014 默认不施加 direct wrench，只用该字段标记/后续兼容；若实现中需要重力补偿，必须在结果中明确标记，不得混入 B-only 成功结论。

## 实现计划

1. 新增 E014 scene generator：
   - 基于每个 task 的 `scene.xml` 生成 `scene_e014_jointB_<slug>.xml`。
   - 添加 mocap body `support_weld_anchor`，含非碰撞可视 geom/site。
   - 添加 `<equality><weld ...>`，`body1="object"`，`body2="support_weld_anchor"`，`relpose="<support_point_local> 1 0 0 0"`，`solref="<timeconst> 1"`，`solimp` 随 variant 写入。
   - 用 MuJoCo 编译检查确认 `nq=43`、`nu=29`、`nmocap>=1`。

2. 复用并扩展 `support_proxy` runtime：
   - `_load_support_proxy` 已能从 object ref + local point 预计算 support ref position。
   - 新增 `support_proxy_mocap_quat_mode`，E014 设置为 `object_ref`，使 mocap anchor quat 跟随 ref object quat。
   - `_update_support_proxy_mocap_pad` 写入 `support_weld_anchor` 的 mocap pos/quat。
   - 保存已有 diagnostics：`support_proxy_pos`、`support_point_pos`、gap、force/torque 占位。

3. 新增 E014 overrides/scripts/eval：
   - `scripts/E014/variants.tsv`
   - `scripts/E014/generate_e014_scenes.py`
   - `scripts/E014/generate_e014_overrides.py`
   - `scripts/run_E014_preprocess.sh`
   - `scripts/train/train_E014.sh`
   - `scripts/eval/eval_E014.py`
   - 根据 6 个 variants 决定本地/远程拆分；先跑 smoke，再跑 full。

## 验收指标

E014 使用 E013 生成的 `workspace/core4d_collab_retarget/results/E013/e014_soft_targets.json`：

Main soft target：

- obj mean `<=0.193m`
- obj max `<=0.371m`
- hand contact `>=73.6%`
- floor contact `<=69.5%`
- leg-object interference `<=12.5%`
- lag-free diagnostic：obj mean `<0.28m`
- push-vs-carry：floor `<=70%` 且 leg `<=15%`

Guard soft target：

- obj mean `<=0.214m`
- obj max `<=0.417m`
- hand contact `>=61.7%`
- floor contact `<=44.7%`
- leg-object interference `<=7.7%`
- pelvis min `>=0.55m`

额外 parity / diagnostic：

- `E014_freejoint_parity_ok=true`：`contact_guidance=false`、`nq_obj=7`、`nu=29`、`object_action_dims=0`、object actuator ids empty。
- `E014_anchor_not_com_oracle=true`：`support_proxy_point_local` 非零且 scene 中没有 `object_target`/`scene_weld` COM zero-relpose 口径。
- `E014_support_gap_mean/max` 记录 support anchor 与 object support point gap。
- 若有 direct support wrench 或 object kinematic override，必须判为 config violation。

## 决策规则

- 若 main best 过 E013 soft target 且 push-vs-carry 通过，COLA-B 成立，进入 pipeline 对接或 E015 A+B 验证。
- 若 main best obj mean `<=0.27m` 但未过全部 target，触发 E014b stiffness / solimp sweep。
- 若 main best obj mean `>=0.28m` 或数值不稳，先分析失败形态；不能直接跳 E016，应按总路线判断是否进入 E015。
- 若 guard 全部不稳定，E014 不可作为 work 配置，即使 main 接近 target。
