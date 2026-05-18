# E013 计划：true-freejoint object oracle

日期：2026-05-19

## Context

E012 full 已完成。结果显示 dual-point partner force 没有修复 E011 的 xy lag，强 coupling 主要产生 rotation shortcut；6 条 main 没有任何一条达到 E081 transport，2 条 guard 均不稳定。

根据 `docs/03_agent_execution_plan_E013_E016.md`，下一步不是继续 spring / force sweep，而是先做 true-freejoint object oracle。E013 的任务是回答：如果 object 在 true-freejoint scene 下被 oracle 完美放到 reference 位置，robot / hand / leg / floor / pelvis 等指标的上限是什么。这个结果将用于定义 E014 之后的软 target。

注意：现有 `object_pd_override=true` 只覆盖 `scene_act` 的 6 维 object actuator ctrl；freejoint scene 没有 object actuator。因此 E013 不能直接复用 `object_pd_override`，需要用显式 freejoint object kinematic override。

## Claims

| Claim | 验证方式 |
|-------|----------|
| C1 E013 是 true-freejoint scene，不回退到 `scene_act` object actuator | `contact_guidance=false`、`scene_name=scene`、`nq_obj=7`、`nu=29`、`object_actuator_ids=[]` |
| C2 object oracle 能把 object trajectory 基本压到 ref | obj mean/max 应远低于 E011 `0.340/0.673m`，并记录 oracle 自身残差 |
| C3 oracle 后的 hand / floor / leg / pelvis 指标能给 E014 软 target 提供依据 | 输出 main/guard 全指标和 `e014_soft_targets.json` |
| C4 oracle 不应该被误判为物理 work | 结果日志明确标记为 oracle/cheating upper bound，不作为最终算法成功 |
| C5 脚本/评估口径可复现 | plan、variants、overrides、train/eval 脚本、结果路径全部落盘 |

## 实现改动

1. `spider/config.py`
   - 新增 `object_kinematic_override: bool = False`
   - 新增 `object_kinematic_ref_dt: float = -1.0`，`<=0` 时使用 `sim_dt`，因为 qpos_ref 已插值到 sim step
   - 新增 `object_kinematic_set_qvel: bool = True`

2. `examples/run_mjwp.py`
   - 当 `object_kinematic_override=true` 时，预加载 ref object qpos/qvel 到 env。
   - 要求 `embodiment_type=humanoid_object` 且 `nq_obj=7`。

3. `spider/simulators/mjwp.py`
   - 增加 helper，在 step 前和 step 后把 object freejoint qpos 写成 ref qpos。
   - 若 `object_kinematic_set_qvel=true`，同步写 object qvel，避免 physics step 内残留速度污染 contact。
   - 保留 legacy `partner_force_spring_kp < 0` 的兼容行为，但 E013 使用新字段。

4. E013 scripts
   - `workspace/core4d_collab_retarget/scripts/E013/variants.tsv`
   - `workspace/core4d_collab_retarget/scripts/E013/generate_e013_overrides.py`
   - `workspace/core4d_collab_retarget/scripts/run_E013_preprocess.sh`
   - `workspace/core4d_collab_retarget/scripts/train/train_E013.sh`
   - `workspace/core4d_collab_retarget/scripts/eval/eval_E013.py`

## Variant Grid

| Variant | Case | Queue | Role | 目的 |
|---------|------|-------|------|------|
| `E013_box025_p2_obj_oracle` | `box025_person2_freejoint_legobj` | local | main | E081 main 的 true-freejoint oracle |
| `E013_box023_p2_obj_oracle` | `box023_person2_freejoint_legobj` | local | guard | E081 guard 的 true-freejoint oracle |

E013 只有 2 条 full，不需要远程并行。

## 成功标准

- Smoke:
  - 2/2 variants 产出 4-step NPZ；
  - eval 能读取 summary；
  - `E013_freejoint_oracle_config_ok=true`。
- Full:
  - 2/2 variants full NPZ 覆盖 smoke；
  - `aggregate_summary.json` 包含 main/guard；
  - `e014_soft_targets.json` 生成；
  - 视频和关键帧完成；
  - log 中写明实际观察。

## E014 Soft Target 输出规则

Main 使用 E081 main 与 E013 main 共同定义：

```text
obj_mean_target = max(E081_obj_mean + 0.05, E013_oracle_obj_mean + 0.05)
obj_max_target  = max(E081_obj_max  + 0.10, E013_oracle_obj_max  + 0.10)
hand_target     = min(E081_hand_pct, E013_oracle_hand_pct) - 5pp
floor_target    = max(E081_floor_pct, E013_oracle_floor_pct) + 10pp
leg_target      = max(E081_leg_intf_pct, E013_oracle_leg_intf_pct) + 5pp
```

Guard 同理使用 E081 guard 与 E013 guard。

## 执行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E013_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E013.sh smoke 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E013.py --all

# full
bash workspace/core4d_collab_retarget/scripts/train/train_E013.sh full 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E013.py --all
```

## 决策规则

- 若 E013 oracle 本身 obj error 仍接近 E011 (`>=0.28m`)：E014 的目标必须显著放宽，并优先审查 SPIDER reward/horizon 而不是直接推进 COLA。
- 若 E013 oracle obj error 接近 E081：E014 软 target 仍接近 E081，说明主要问题是当前 force/spring 范式的 lag。
- 无论 E013 数值如何，E014 仍按位置约束路线启动；E013 只改变阈值，不改变主线顺序。
