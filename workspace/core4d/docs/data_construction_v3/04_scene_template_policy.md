# 04 Scene template 策略

source scene template 是数据构建的基础层。E103 已经证明 template 污染会让后续 CEM/RL 结论失效，因此 v3 把 template 审计作为硬 gate。

## 责任边界

| 对象 | 责任 |
|---|---|
| source template `scene.xml` | clean robot skeleton、object mesh、collision、mass/inertia、hand contact sites、placeholder object pose |
| target scene `scene.xml` | 从 source template 派生，并用 trimmed qpos 第一帧 patch object pose |
| `scene_act.xml` | target trajectory 生成后再生成，用于 CEM/MJWP |
| `trajectory_kinematic.npz` | SPIDER target qpos/qvel/ctrl/contact/contact_pos |

source template 不能把 target case 的 object pose 当成数据真值。target pose 必须来自 trimmed qpos。

source template 的可视化审查由 `render_template_review_package.py` 生成，输出 MP4、keyframe sheet 和 `template_visual_manifest.tsv/json`。该可视化只用于审查 source template 的 mesh/collision/robot skeleton 是否合理，不替代 S2 的机器审计。

## Box 类物体

box 类物体可使用 E103 后 clean-base 策略：

1. 选择 clean base scene，例如已审计的 `box023_person1`。
2. 复制 robot/world/contact skeleton。
3. 替换 object mesh。
4. 用 mesh AABB 生成 box collision half-extents。
5. 写入明确的 mass/inertia policy。
6. MuJoCo load。
7. 运行 robot inertial audit、collision audit、contact-site audit。
8. 生成 source template visual sheet/mp4。

## 非 box 物体

bucket、desk、chair 等非 box 不能自动放行。必须人工审查：

- collision proxy 是否合理；
- mass/inertia 是否有依据；
- object body pose 和 mesh frame 是否一致；
- hand/object contact proxy 是否可解释；
- 是否需要多个 collision geom 或非 box proxy。

非 box 的 `template_status` 默认应为 `manual_review_required`，直到审查通过。v3 允许生成 review 用 proxy template，但不能仅凭 proxy 生成或 MuJoCo load 成功自动置为 clean。

当前 proxy adapter：

| adapter | 类别 | collision policy | 默认状态 |
|---|---|---|---|
| `nonbox_proxy_aabb_review` | bucket | `bucket_wall_proxy_aabb`：底面 + 四侧壁 box geoms | `manual_review_required` |
| `nonbox_proxy_aabb_review` | board/stick | `mesh_aabb_box_proxy` | `manual_review_required` |
| `nonbox_surface_voxel_review` | desk/chair | `desk_surface_voxel_multibox_proxy_draft` / `chair_surface_voxel_multibox_proxy_draft`：OBJ 表面 voxelization 后合并成多 box proxy | `manual_review_required` |

### desk/chair surface voxel proxy

desk/chair 在 CORE4D 中不是稳定语义类别：同一个 `desk` 类可能包含侧板/U 型架、小圆几/三脚凳、管架等形态；`chair` 也可能不是标准 seat/back/legs 拓扑。因此 v3 不再用“标准桌/椅”语义模板自动构建 collision。

缺失 desk/chair source template 时，S2 可生成 review-only surface voxel proxy：

1. 读取 raw OBJ mesh，保持 object-local frame，不对 mesh 做自动旋转。
2. 用 `trimesh.voxelized()` 对 mesh 表面 voxelize。
3. 将 occupied surface voxels 贪心合并成一组 local AABB box geoms。
4. 使用较高分辨率 `target_cells=26` 和轻微 inward shrink，避免 collision 比 mesh 明显大一圈。
5. 输出 `object_collision` + `object_collision_voxel_*` 多个 box geoms。

该 proxy 的目标是生成可审查、可加载、比单 AABB 更贴近 mesh 的碰撞草稿；它不是最终人工语义建模，也不能自动 release。通过 review 前必须保持：

```text
template_status=manual_review_required
proxy_template=True
manual_review_required=True
```

review package 必须至少包含：

- source template orbit sheet/mp4；
- mesh/collision overlay 或 object-only mesh/collision sheet，可由 `render_template_mesh_collision_review_package.py --object-only` 生成；
- `nonbox_template_review.tsv` 中的显式 `review_decision=approve_clean` 才能进入 `clean_reviewed`。

审查通过后，必须通过 `nonbox_template_review.tsv` 显式写入 `template_status=clean_reviewed`。`clean_reviewed` 才能进入 Stage2b。

## 硬失败

- MuJoCo load fail；
- `nq/nv/nu` 与预期不一致；
- hand contact sites 缺失；
- robot inertial 与 clean base 不一致；
- 出现已知污染质量或惯量，例如历史 `29.632` robot inertial；
- object collision extents 与 mesh AABB 明显不一致；
- source template 残留旧 runtime artifact 并被误用。

## source scene missing

`source_scene_task` missing 不是数据 reject，也不能跳过。必须进入 template backlog：

```text
template_status=backlog
current_decision=REJECT_TEMPLATE_BACKLOG
```

补齐 template 并通过 audit 后，case 才能进入 S3。
