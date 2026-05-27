# E083 Results: upper-body-object collision pairs on Box021 failures

日期：2026-05-28

对应计划：`workspace/core4d/plan/88_E083_upperbody_object_collision_plan.md`

## 结论

E083A 完整跑完 3 个 D003 Box021 main case + `box023_p2` guard。结果是：

- Box021 main：`0/3` 通过；
- `box023_p2` guard：通过，且没有上半身穿模或手撑地退化；
- upper-body-object collision pairs 是安全的物理补丁，但**不足以解决 Box021 弯腰搬箱失败**。

核心结论：

1. E083A 把 E082 的“头/躯干深度栽进箱体、腿箱大穿插”部分改成了“被箱子挡住后趴箱/压箱/卡箱沿”。
2. 3 个 Box021 main 仍全部视觉不可用：不是合理抱箱/搬箱，而是趴箱、头部浅穿、低髋倒伏、手撑地或腿部干涉。
3. `box023_p2` guard 未退化，说明新增 upper-body pairs 本身不是普遍有害；失败集中在 Box021 的 reach / support / contact semantic 上。
4. 下一步不应继续只加 collision pair。必须进入 reward/constraint 层：upperbody 禁穿惩罚、hand-floor 惩罚、姿态稳定/控制 trust region、手部语义接触和 object lift/floor-contact 约束。

## 改动

E083 为每个 source task 新建派生 task，不修改原始 task：

```text
d003_box021_20231018_029_p2_upperobj_e083
d003_box021_20231011_035_p2_upperobj_e083
d003_box021_20231020_019_p1_upperobj_e083
box023_person2_upperobj_e083
```

每个派生 `scene_act.xml` 增加：

- 16 个腿/脚-`object_collision` pair；
- 7 个 upper-body-`object_collision` pair：

```text
head_collision
torso_collision
pelvis_collision
left/right_shoulder_yaw_collision
left/right_elbow_yaw_collision
```

MuJoCo scene 检查：4 个派生 task 均为 `nq/nv/nu/npair=42/41/35/49`，且 `leg=16`、`upper=7`。

## 运行与路径

| Split | Variant | GPU | 状态 |
|---|---|---:|---|
| local | `E083_d003_box021_20231018_029_p2_upperobj` | local GPU0 | 完成 |
| remote-gpu0 | `E083_d003_box021_20231011_035_p2_upperobj` | remote GPU0 | 完成 |
| remote-gpu1 | `E083_d003_box021_20231020_019_p1_upperobj` | remote GPU1 | 完成 |
| remote-gpu1 | `E083_box023_p2_upperobj_guard` | remote GPU1 | 完成 |

结果路径：

| 类型 | 路径 |
|---|---|
| 汇总 | `workspace/core4d/results/E083/comparison.csv` |
| upper-body 诊断 | `workspace/core4d/results/E083/upperbody_diagnostics.csv` |
| 聚合 | `workspace/core4d/results/E083/aggregate_summary.json` |
| 视频/轨迹 | `workspace/core4d/results/E083/*.mp4`, `workspace/core4d/results/E083/*.npz` |
| keyframe sheets | `workspace/core4d/results/E083/keyframes/contact_sheets/` |
| logs | `logs/E083/` |

## 量化结果

Aggregate:

```json
{
  "num_results": 4,
  "num_main_results": 3,
  "num_main_case_window_success": 0,
  "main_case_window_success_pct": 0.0,
  "num_main_upperbody_physical_proxy_success": 0,
  "main_upperbody_physical_proxy_success_pct": 0.0,
  "guard_results": ["E083_box023_p2_upperobj_guard"]
}
```

主表：

| Case | Role | Obj mean/max | Pelvis min | Sim contact | Leg intf | Obj floor | Bottom gap | Upper pen | Head/Torso pen | Hand floor | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20231018_029_p2` | main | `0.608/0.963m` | `0.478m` | `69.0%` | `0.0%` | `94.6%` | `-11.8cm` | `82.2%` | `38.8/27.9%` | L `10.1%`, R `0%` | FAIL |
| `20231011_035_p2` | main | `0.903/1.632m` | `0.563m` | `5.7%` | `76.2%` | `100.0%` | `-18.8cm` | `92.5%` | `21.1/0.0%` | `0/0%` | FAIL |
| `20231020_019_p1` | main | `0.866/1.458m` | `0.193m` | `41.2%` | `7.4%` | `97.3%` | `-9.7cm` | `83.1%` | `29.1/0.0%` | L `65.5%`, R `0%` | FAIL |
| `box023_p2` | guard | `0.160/0.315m` | `0.675m` | `70.7%` | `0.0%` | `34.7%` | `-4.3cm` | `0.0%` | `0/0%` | `0/0%` | PASS |

相对 E082 的主要变化：

| Case | 改善 | 退化/仍失败 |
|---|---|---|
| `20231018_029_p2` | obj mean `0.680→0.608m`，pelvis min `0.355→0.478m`，contact `14.7→69.0%`，leg intf `69.8→0%` | object-floor `94.6%`，bottom gap 更差到 `-11.8cm`，上半身仍 `82.2%` 帧浅穿/压箱 |
| `20231011_035_p2` | pelvis min `0.183→0.563m`，不再 deep collapse | obj mean `0.457→0.903m`，contact `37.4→5.7%`，leg intf `52.9→76.2%`，箱子全程贴地 |
| `20231020_019_p1` | leg intf `31.8→7.4%`，obj max 略好 | pelvis min 仍 `0.193m`，LH floor `65.5%`，箱子几乎全程贴地 |

## 可视化复核

Contact sheets:

- `workspace/core4d/results/E083/keyframes/contact_sheets/E083_all_cases_sheet.jpg`
- `workspace/core4d/results/E083/keyframes/contact_sheets/E083_d003_box021_20231018_029_p2_upperobj_sheet.jpg`
- `workspace/core4d/results/E083/keyframes/contact_sheets/E083_d003_box021_20231011_035_p2_upperobj_sheet.jpg`
- `workspace/core4d/results/E083/keyframes/contact_sheets/E083_d003_box021_20231020_019_p1_upperobj_sheet.jpg`
- `workspace/core4d/results/E083/keyframes/contact_sheets/E083_box023_p2_upperobj_guard_sheet.jpg`

Subagent high 视觉复核：

| Case | 视觉标签 | 观察 |
|---|---|---|
| `20231018_029_p2` | 不可用：趴箱/穿箱型失败 | 多数帧上身压在箱体上，头和躯干仍进入箱体区域；有接触但不是合理抱箱/推箱。 |
| `20231011_035_p2` | 不可用：趴箱 + 腿部干涉严重 | 动作退化成身体覆盖箱体，upper-body penetration 很高，leg interference 是额外硬伤。 |
| `20231020_019_p1` | 不可用：跌倒/手撑地 + 头部穿箱 | 多帧低髋、半跪或趴地，左手频繁撑地，箱子接触不是稳定操控。 |
| `box023_p2 guard` | 可用：guard 未退化 | 站姿和箱体关系保持较好，无上半身穿箱，无手撑地。 |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 不污染 source task | 通过；只新增 `*_upperobj_e083` 派生目录 |
| C2 leg/foot pair 保持 | 通过；4/4 scene 均 `leg=16` |
| C3 upper-body pair 生效 | 通过；4/4 scene 均 `upper=7` |
| C4 三个 Box021 main 完成 full CEM | 通过；3/3 `.npz/.mp4/eval/keyframes` |
| C5 guard 完成且不明显退化 | 通过；`box023_p2` numeric/visual guard 均通过 |
| C6 失败机制被重新判定 | 通过；从深穿箱转为趴箱/浅穿/错误接触语义 |
| C7 可视化和 subagent high 复核完成 | 通过 |

## 机制解释

E083A 证明“缺 upper-body-object pair”是 E082 的一个真实物理漏洞，但不是 Box021 失败的唯一原因。

碰撞 pair 加上后，CEM 不再容易把身体深插进箱子内部；但当前 reward 仍允许更隐蔽的局部解：

- 用头、肩、胸、手臂贴压箱体来获得 contact/object objective；
- 用低髋/半跪/趴地维持接触；
- 用手撑地作为额外支撑；
- 让箱子继续贴地，靠压/推而不是 lift/support。

当前 E083 配置仍是：

```text
stability_penalty_scale = 0.0
task_body_rew_scale = 0.0
hold_contact_rew_scale = 0.0
contact_hdmi_gain = 5.0
task_obj_pos/rot_rew_scale = 1.0/1.0
```

因此 CEM 会优先追手-箱 proximity 和物体 actuator tracking，而不会显式避免“上身压箱/趴箱/手撑地/物体贴地”。

## 决策

E083A 不可作为后续 RL seed。下一步进入 E084，做 3 组受控实验：

1. 安全惩罚组：upperbody-object penetration penalty + hand-floor penalty + stability penalty；
2. 姿态/控制组：stronger ctrl_ref_guard + task_body/upright tracking + 降低 contact/object 牵引；
3. 语义接触/lift 组：hand-only semantic contact gate + object lift/floor-contact penalty。

下一步计划：`workspace/core4d/plan/89_E084_box021_constraint_groups_plan.md`
