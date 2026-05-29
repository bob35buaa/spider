# E093 Plan: Contact Target Geometry Audit Before CEM/RL

日期：2026-05-29

## Context

E092 full CEM 纠偏后，三条 medium-box SPIDER dynamic 结果不是 smoke 阶段的全失败：

| case | task | full result | 关键现象 |
|---|---|---|---|
| C1 | `e091_box004_20231003_2_083_p2` | WORK | pelvis `0.663m`，contact `64.8%`，head/upper/floor `0%` |
| C2 | `e091_box026_20231018_039_p2` | FAIL | pelvis `0.083m`，support face 低，低髋/半跪 |
| C3 | `e091_box026_20231020_135_p2` | FAIL | pelvis `0.177m`，right-inside `12.2%`，RH floor `17.1%` |

用户提出的下一步不是继续盲跑 CEM/RL，而是在 CEM 之前深究 contact target 几何：

1. 当前 `ref_fk` dynamic target 本质是 `wrist_yaw_link + [0.05, 0, 0]`，不一定等于真实 raw contact point。
2. 需要在已有 `box023 / box025 / box004 / box021 / box026` case 上量化这个偏移。
3. 需要验证 G1 sphere hand geometry 是否让接触位置不准。
4. 需要重新把 HDMI 3-box 几何、Holosoma 后来的 handbox 几何作为 proxy 对照可视化，而不是只讨论日志结论。
5. 本实验优先产出可视化，并用 high-reasoning subagent 观察。

相关历史证据：

- E039 引入 `contact_hdmi_eef_offset=[0.05,0,0]` 作为 G1 wrist 到 palm proxy。
- E073 只修了 dynamic target 与 reward 的口径一致性：target 也从 wrist origin 改为 `wrist+5cm`，但没有证明该点等于 raw contact。
- E084/E085 已在 D003 Box021 main 上发现 old G1 pseudo target 与 raw surface target 平均相差约 `27cm`。
- E061/E075/E076 相关日志说明 3-box 曾失败，主要是几何末端和 `[0.05,0,0]` reward point 错位，而非 3-box 这个概念本身必然无效。
- Holosoma R084-R086 使用 `main_mesh_collision_handbox_m5.urdf`，handbox link 固定到 wrist_yaw_link，Box023 smoke/preflight 可跑通，但仍需要放回 SPIDER/Core4D case 做几何对照。

## Scope

E093 是诊断实验，不跑 full CEM/RL，不改变训练 reward。输出必须回答：

1. `wrist+5cm` 与 raw CORE4D contact surface target 的偏移有多大？
2. 偏移方向是否系统性落在错误 face、inside box、低位/非支撑面？
3. 当前 sphere 的真实接触面与 reward point 的偏差有多大？
4. 3-box proxy 和 Holosoma handbox proxy 是否比 sphere/reward point 更接近 raw contact，或只是把物理接触提前/推远？
5. box004/box023 为什么看起来更容易 work，box021/box026 为什么更容易 fail？

## Cases

第一批覆盖 7 条已有 case，其中 5 类物体都必须出现：

| group | task | raw mask source | 目的 |
|---|---|---|---|
| box023 | `box023_person2` | `workspace/core4d/results/E079/contact_masks/box023_person2/raw_contact_mask_3cm.npz` 或 E081/E084 同名 | known positive/guard，小箱/中箱 |
| box025 | `box025_person2` | `workspace/core4d/results/E080/contact_masks/box025_person2/raw_contact_mask_3cm.npz` | 大箱 reach-hack 背景，partial positive |
| box004 | `e091_box004_20231003_2_083_p2` | Holosoma v3 `stage2b_medium/results/contact_masks/...` | E092 C1 WORK，对照 box023 pattern |
| box021-a | `box021_person1` | `workspace/core4d/results/E079/contact_masks/box021_person1/raw_contact_mask_3cm.npz` | box021 旧 positive/control |
| box021-b | `d003_box021_20231018_029_p2` | `workspace/core4d/results/E084/contact_masks/d003_box021_20231018_029_p2/raw_contact_mask_3cm.npz` | known fail，E084/E085 27cm 偏差复核 |
| box026-a | `e091_box026_20231018_039_p2` | Holosoma v3 `stage2b_medium/results/contact_masks/...` | E092 C2 FAIL，support low |
| box026-b | `e091_box026_20231020_135_p2` | Holosoma v3 `stage2b_medium/results/contact_masks/...` | E092 C3 FAIL，right-inside risk |

若 raw mask 只含 binary/dist，不含 raw target，脚本必须复用 E085 的 raw surface target 生成逻辑，从 audit summary 中的 raw sequence/object mesh 重新生成每帧 raw contact centroid；不能退化成只看 `trajectory_kinematic.npz/contact_pos`。`contact_pos` 只能作为 SPIDER-upstream reference point 的辅助对照。

## Geometry To Compare

每帧、每只手计算以下点/体在 object-local 和 world frame 下的位置：

| name | 定义 | 用途 |
|---|---|---|
| `raw_surface` | raw SMPL-X hand vertices 对 object mesh 的 close-contact centroid，投影到 MuJoCo collision box face | 真实接触语义 proxy |
| `contact_pos` | `trajectory_kinematic.npz` 内的 upstream contact point（若存在） | 检查 SPIDER 输入中的 contact reference |
| `wrist_origin` | `left/right_wrist_yaw_link` body origin | retarget wrist 本体 |
| `reward_wrist_5cm` | `wrist_origin + wrist_rot @ [0.05,0,0]` | 当前 `ref_fk` dynamic target/reward point |
| `sphere_center` | 当前 `lh/rh` hand sphere center，预期 `wrist+10cm` | 当前物理碰撞中心 |
| `sphere_surface_to_raw` | sphere center 到 raw surface 的方向距离减半径 | sphere 是否能以合理表面接触 raw target |
| `contact_site` | `contact_left/right_hand` site，预期约 `wrist+8cm` | reward/site/geom 三者一致性 |
| `threebox_proxy` | 历史 HDMI 3-box：wrist cuff / palm / finger pad，使用日志与 patch 脚本中的几何参数；重点画 box3 tip/centroid | 验证 3-box 是否只是比 reward point 更前而提前撞箱 |
| `handbox_proxy` | Holosoma `main_mesh_collision_handbox_m5.urdf` 中 `left/right_handbox_link` box：fixed joint origin `~[0.1074, ±0.0116, 0.0102]`，box size `~[0.1418,0.0766,0.1165]` | 检查 handbox 面/中心是否更接近 raw contact |

三类几何体都要在可视化里用不同颜色显示：

- reward point / wrist point：点或小球。
- sphere：透明球和最近 raw-contact 表面点。
- 3-box / handbox：透明 box wireframe，显示中心和最接近 raw surface 的 box face/vertex。

## Metrics

每个 case 输出 per-hand summary：

| metric | 说明 |
|---|---|
| `raw_active_frac` | raw mask active 帧比例 |
| `wrist5_to_raw_mean/median/p90/max_m` | 当前 target 到 raw surface 的距离 |
| `wrist5_vs_contact_pos_mean_m` | 与 SPIDER `contact_pos` 的距离（若存在） |
| `raw_face_counts` / `wrist5_face_counts` | object collision face 分布 |
| `raw_support_frac` / `wrist5_support_frac` | world-up 或 legacy +z support face 占比 |
| `wrist5_inside_frac` | `wrist+5cm` 落入 collision box 内比例 |
| `sphere_surface_gap_mean/p90_m` | sphere 表面到 raw surface 的 signed gap，负值代表 raw target 在 sphere 内/穿透需求 |
| `threebox_surface_gap_mean/p90_m` | 3-box proxy 到 raw surface 的最近 surface gap |
| `handbox_surface_gap_mean/p90_m` | handbox proxy 到 raw surface 的最近 surface gap |
| `best_proxy_by_distance` | raw target 下哪种 proxy 最近 |
| `bad_proxy_flags` | inside、wrong-face、low-support、surface-gap-too-large 等 |

跨 case 输出：

- `box004/box023` 是否同 pattern：尺寸接近、wrist5 outside/support、raw-to-wrist5 偏差小或同 face。
- `box026` fail 是否主要由尺寸/reach、face switch、inside risk 或 sphere surface mismatch 解释。
- `box021` old positive vs D003 fail 的几何差异。

## Artifacts

输出根目录：`workspace/core4d/results/E093/contact_geometry/`

| artifact | 内容 |
|---|---|
| `case_manifest.tsv` | case、task_dir、mask_path、person_idx、raw source |
| `geometry_summary.{json,csv,md}` | case-level 指标 |
| `per_frame_points.csv` | 逐帧点位与 face/gap |
| `visuals/object_local/*_overlay.png` | object-local 三视图：raw / contact_pos / wrist / wrist+5cm / sphere center / 3-box / handbox |
| `visuals/timeline/*_timeline.png` | 偏移距离、inside/support、face switch、sphere/box gap 时间线 |
| `visuals/mujoco/*_keyframes.png` | MuJoCo reference qpos 关键帧 + colored markers/transparent proxies |
| `visuals/mujoco/*_geometry.mp4` | MuJoCo 短视频，显示随时间变化的目标和几何 proxy |
| `visuals/dashboard/contact_geometry_dashboard.png` | 跨 case 汇总图 |
| `visual_review/high_subagent_review.md` | high subagent 对可视化的逐 case 观察 |

所有 PNG 必须做非空检查；MP4 至少用 `ffmpeg -v error -i ... -f null -` 验证可解码。

## Implementation Plan

1. 新增 `workspace/core4d/scripts/E093/build_contact_geometry_manifest.py`
   - 固化 7 条 case 的 task_dir / raw mask / person_idx。
   - 自动检查 scene、trajectory、mask、audit_summary、object mesh 是否存在。

2. 新增 `workspace/core4d/scripts/E093/audit_contact_geometry.py`
   - 复用 E085 raw target 生成逻辑，但泛化到 E093 manifest。
   - 用 MuJoCo `mj_forward(qpos_ref[t])` 计算 wrist/site/sphere。
   - 从 Holosoma handbox URDF 解析 handbox fixed joint origin/box size。
   - 从历史 3-box log/patch 或 scene snapshot 中解析/重建 3-box proxy；若无法找到完整 patch 源码，则使用日志中明确的 wrist+17.5cm tip/centroid proxy，并在 summary 标明 `threebox_proxy_source=log_approximation`。
   - 输出 CSV/JSON/MD 和 2D 可视化。

3. 新增 `workspace/core4d/scripts/E093/render_contact_geometry_mujoco.py`
   - 读取 `per_frame_points.csv`。
   - 在 MuJoCo scene 上用 viewer marker 或临时 mocap/site geom 渲染 wrist/raw/sphere/box proxies。
   - 输出关键帧 sheet 和 mp4。

4. 运行全量诊断：
   - `python workspace/core4d/scripts/E093/build_contact_geometry_manifest.py --force`
   - `python workspace/core4d/scripts/E093/audit_contact_geometry.py --manifest ...`
   - `python workspace/core4d/scripts/E093/render_contact_geometry_mujoco.py --manifest ...`

5. 使用 high-reasoning subagent 复核：
   - 输入 dashboard、每 case overlay/timeline/keyframes。
   - 要求观察：raw 与 wrist+5cm 是否同 face；sphere 是否能合理接触；3-box/handbox 是否改善或更糟；box004/box023 pattern 与 box026/box021 fail pattern 是否可视上成立。

6. 写正式结果：
   - `workspace/core4d/log/115_E093_contact_target_geometry_audit_results.md`
   - 更新 `workspace/core4d/progress.md`
   - 更新 `workspace/core4d/EXPERIMENT_TRACKER.md`

## Claims

| Claim | 判定方式 |
|---|---|
| C1: `wrist+5cm` 与 raw contact 的偏移在 fail case 显著大于 work/guard case | 比较 `wrist5_to_raw_mean/p90`：box021-D003、box026 > box004/box023 |
| C2: sphere 几何会让接触位置语义不准 | 若 `sphere_surface_gap` 在 fail case 大、且 raw target 经常落在 sphere 无法自然接触的 face/方向，则支持 |
| C3: 3-box 旧失败可由 reward/geometry mismatch 解释，而不是“面接触无价值” | 若 3-box proxy tip/face 明显越过 raw surface 或与 `wrist+5cm` 差距 >10cm，支持 |
| C4: handbox proxy 至少在 Box023/box004 pattern 上比 sphere/reward point 更接近 raw support contact | 比较 `handbox_surface_gap` 与 `sphere_surface_gap`，并由可视化确认 |
| C5: box004 和 box023 属于同一个容易 work 的 contact pattern | 二者尺寸/face/support/inside/raw-wrist 偏移相近，且可视化显示接触点在外侧支撑区域 |

## Success Criteria

E093 完成的最低标准：

1. 7 条 case 全部有 `geometry_summary` 结果；若某条缺 raw mesh/audit summary，必须明确标记原因并仍输出 wrist/sphere/handbox 指标。
2. 至少 7 张 object-local overlay、7 张 timeline、7 张 MuJoCo keyframe sheet 非空。
3. 至少覆盖 3 个 proxy：sphere、3-box、handbox；其中 handbox 必须从 Holosoma URDF 解析参数。
4. high subagent 完成可视化复核，复核文本写入结果目录。
5. log 115 对 C1-C5 逐条给出 PASS/PARTIAL/FAIL，并给出是否进入下一轮 CEM/RL 的明确建议。

## Non-goals

- 不在 E093 内修改 reward 或训练配置。
- 不用 E093 结果直接宣称 handbox/3-box 能 work；这里只判断几何对齐，不判断策略训练成败。
- 不重复 E092 Stage B/C RL；E092 full correction 需另行修正 log/tracker，但 E093 的主要任务是 contact geometry。
