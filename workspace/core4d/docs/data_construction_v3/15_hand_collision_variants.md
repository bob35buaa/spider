# 15 手部碰撞体 variant

## 定义

`hand_collision_variant_id` 是 S5/CEM 阶段的机器人侧碰撞体轴，和 `retarget_variant_id`、`target_variant_id` 正交。

| 字段 | 作用 |
|---|---|
| `retarget_variant_id` | OmniRetarget / input rewrite 版本。 |
| `target_variant_id` | SPIDER target route，例如 `ref_fk`、`adaptive`、`fingertip_aware`。 |
| `hand_collision_variant_id` | CEM scene 中机器人手 `lh/rh` 的碰撞几何。 |

该轴不同于 `collision_policy`。`collision_policy` 描述物体侧 proxy/template；`hand_collision_variant_id` 只描述机器人手侧碰撞体。

## 当前取值

| variant | 语义 | scene 行为 |
|---|---|---|
| `sphere5cm` | 旧默认。每只手一个 5cm sphere，geom 名称为 `lh/rh`。 | no-op，沿用源 `scene_act.xml`。 |
| `rubber_hull` | rubber hand visual mesh 作为凸包碰撞体。 | 在 mesh asset 上设置 `maxhullvert=64`，把 `lh/rh` 替换为 `type="mesh"` 的 convex hull geom。 |

`maxhullvert=64` 是 MuJoCo mesh 凸包顶点上限。E147 前置几何检查显示 64 已处在精度收益递减区间，缩小误差低于厘米级 contact band，因此不作为 full CEM sweep 参数。

## Patch 规则

入口：

```bash
workspace/core4d/scripts/data_construction_v3/stages/s5_handoff/patch_hand_collision.py \
  --base-scene-act "$TASK_DIR/scene_act.xml" \
  --hand-collision-variant-id rubber_hull \
  --scene-name scene_act_rubber_hull \
  --install-dir "$TASK_DIR" \
  --out-dir "$RUN_DIR/s5_handoff/hand_collision/<case_id>"
```

`rubber_hull` 必须满足：

- 不覆盖源 `scene_act.xml`；
- sidecar scene 与源 task 放在同一目录，例如 `scene_act_E147_rubber_hull.xml`；
- `lh/rh` geom 名称保持不变；
- `left_rubber_hand/right_rubber_hand` mesh asset 加 `maxhullvert="64"`；
- `lh` 使用 `mesh="left_rubber_hand"`、`pos="0.0415 0.003 0"`；
- `rh` 使用 `mesh="right_rubber_hand"`、`pos="0.0415 -0.003 0"`；
- MuJoCo load 后 `lh/rh` 必须是 mesh，且 `geom_rbound > 0`。

CEM override 只需要设置：

```yaml
scene_name: scene_act_E147_rubber_hull
```

`spider/config.py` 会把 `scene_name` 解析为当前 task 目录旁的 `<scene_name>.xml`。

## A/B 评估约定

默认 A/B：

- `sphere5cm` 复用历史 full CEM NPZ；
- `rubber_hull` 重跑 full CEM；
- 两者用同一 case、同一 retarget/target route、同一 reward/算法，只换 scene hand collision。

评估必须同时报告：

- contact：`hand_geom_near_3cm/5cm/8cm/10cm_frac`；
- penetration：`hand_geom_penetration_frac`、`hand_geom_deep_penetration_2cm_frac`；
- 稳定性：pelvis/fall、object floor contact；
- 任务：object tracking error；
- 干扰：lower-body/body penetration。

`rubber_hull` 的 SDF 不能按 geom center/rbound 近似；评估应采样 rubber mesh 顶点或使用等价 mesh-aware SDF。
