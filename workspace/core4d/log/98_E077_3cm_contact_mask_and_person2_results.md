# E077 结果: CORE4D 3cm contact mask 清洗 + box023_person2 数据构造

## 状态

E077 完成两件事：

1. 用 CORE4D 官方 visualization 同类的 3cm 几何阈值，生成 `box023` raw/spider/eval 三个时间轴的 per-person/per-hand contact mask proxy。
2. 从 raw `20231008/045` 构造 `box023_person2` 单人 SPIDER case，并完成 shape、scene、trim 与 object 对齐核验。

本轮不训练 MJWP；只做数据与 mask 层准备。

## 结果路径

| 类型 | 路径 |
|------|------|
| Plan | `workspace/core4d/plan/82_E077_core4d_3cm_contact_mask_and_box023_person2_plan.md` |
| 3cm mask npz | `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.npz` |
| 3cm mask csv | `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.csv` |
| 3cm summary | `workspace/core4d/results/E077/contact_masks/box023/audit_summary_3cm.json` |
| person2 converted | `workspace/core4d/results/E077/holosoma_box023_person2/converted/` |
| person2 retargeted | `workspace/core4d/results/E077/holosoma_box023_person2/retargeted/20231008-045-person2-Box023_with_obj_original.npz` |
| person2 trimmed | `workspace/core4d/results/E077/holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz` |
| person2 SPIDER case | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/` |
| person2 verify | `workspace/core4d/results/E077/box023_person2_verify_summary.json` |

## 脚本

| 脚本 | 用途 |
|------|------|
| `workspace/core4d/scripts/E077/generate_core4d_contact_masks.py` | raw 3cm contact proxy 生成 |
| `workspace/core4d/scripts/E077/build_box023_person2.sh` | person2 构造入口 |
| `workspace/core4d/scripts/E077/trim_box023_person2.py` | retargeted `[42:178]` 裁剪 |
| `workspace/core4d/scripts/E077/create_box023_person2_scene.py` | scene.xml / scene_act.xml 生成 |
| `workspace/core4d/scripts/E077/verify_box023_person2.py` | person2 核验 |

## 3cm contact mask

mask 口径：

- raw sequence: `CORE4D_Real/human_object_motions/20231008/045`
- object: `Box023`
- threshold: `0.03m`
- object surface sample: `20000`
- hand proxy:
  - left broad range `4700:5500`
  - right broad range `7500:8150`
- axis:
  - person: `person1`, `person2`
  - hand: `left`, `right`

输出数组：

| 字段 | shape | 含义 |
|------|-------|------|
| `raw_contact_mask_3cm` | `(178, 2, 2)` | raw 30Hz 全帧 p1/p2 L/R mask |
| `spider_contact_mask_3cm` | `(136, 2, 2)` | raw `[42:178]`，对应 SPIDER trajectory |
| `eval_contact_mask_3cm` | `(227, 2, 2)` | 50Hz eval 映射 |
| `*_min_dist_m` | 同 mask 时间轴 | hand proxy 到 object surface 最小距离 |
| `*_vertex_count_lt_thresh` | 同 mask 时间轴 | 小于 3cm 的 hand proxy 顶点数 |

关键窗口结果：

| eval window | raw window | p1 L | p1 R | p2 L | p2 R |
|-------------|------------|------|------|------|------|
| 100-114 | 102-110 | 15/15, mean 0.81cm | 13/15, mean 2.42cm | 15/15, mean 0.11cm | 15/15, mean 0.11cm |
| 115-130 | 111-120 | 16/16, mean 0.74cm | 16/16, mean 2.12cm | 16/16, mean 0.17cm | 16/16, mean 0.11cm |
| 131-145 | 121-129 | 5/15, mean 13.46cm | 4/15, mean 11.14cm | 2/15, mean 25.04cm | 5/15, mean 15.26cm |

解释：

- 3cm 口径下，E075 f115-f130 对应窗口里 `person1_right` 是持续接触，但 mean distance 2.12cm，属于接近阈值的边界接触。
- `person2_left/right` 是强接触，均值约 1-2mm 量级。
- f131-f145 开始，两个人的 contact 都快速下降，这与 putdown/release 阶段一致。

## box023_person2 构造

构造步骤：

1. `convert_core4d_to_omniretarget.py`
   - `--date 20231008 --seq 045 --person person2 --with_object --replace_wrist_with_fingertip`
   - 输出 `converted/20231008-045-person2-Box023_with_obj.npz`
2. `robot_retarget.py`
   - task: `20231008-045-person2-Box023_with_obj`
   - 输出 `retargeted/..._original.npz`
3. 固定 slice `[42:178]`
   - 输出 136 帧 `trimmed/..._original.npz`
4. 生成 SPIDER case
   - `box023_person2/scene.xml`
   - `box023_person2/0/trajectory_kinematic.npz`
   - `box023_person2/scene_act.xml`

核验结果：

| 项 | 结果 |
|----|------|
| retargeted qpos | `(178,43)` |
| trimmed qpos | `(136,43)` |
| SPIDER qpos | `(136,43)` |
| SPIDER qvel | `(136,41)` |
| SPIDER ctrl | `(136,29)` |
| SPIDER contact | `(136,2)`, 全 1 |
| `trimmed == retargeted[42:178]` | `true` |
| `scene.xml` | MuJoCo load OK, `nq=43,nv=41,nu=29` |
| `scene_act.xml` | MuJoCo load OK, `nq=42,nv=41,nu=35`, euler `XZY` |

## 重要发现: p1/p2 object qpos 不能直接合并

核验时发现：

- converted 层 `person1/person2` 的 `object_poses` 完全一致：`maxdiff=0`。
- 但 retarget/SPIDER 层 `person1/person2` 的 object qpos 不完全一致：
  - max abs diff: `0.0621m`
  - position diff mean: `[0.00018, 0.02700, -0.00697]`
  - quaternion diff max: `0`

原因来自 Holosoma retarget preprocess：

```python
human_joints = human_joints * scale
object_poses[:, -3:-1] = object_poses[:, -3:-1] * scale
dz_scale = (object_poses[:, -1] - object_z0) * scale
```

`person1` 与 `person2` 的 SMPL height/scale 不同，所以同一 raw object trajectory 在各自单人 retarget qpos 中被缩放到不同坐标尺度。

结论：

- `box023_person2` 作为单人 SPIDER case 可以使用。
- 但不能把现有 `box023_person1` 和 `box023_person2` 的 retargeted qpos 直接拼成双机器人同场景；双人合成前必须做 common-scale/common-world alignment。

## Claims 验证

| ID | Claim | 结果 | 判定 |
|----|-------|------|------|
| C1 | 生成可复现 3cm per-person/per-hand mask | npz/csv/json 已生成，关键窗口复现 | PASS |
| C2 | 3cm mask 映射到 SPIDER 和 eval 时间轴 | 输出 `(136,2,2)` 和 `(227,2,2)` | PASS |
| C3 | 构造 `box023_person2` 单人 SPIDER case | qpos/qvel/ctrl/contact/scene/scene_act 生成 | PASS |
| C4 | person2 构造不是错帧/错物体 | trim slice 正确；converted object 同源；retarget qpos 有 scale 差异并已记录 | PASS with caveat |
| C5 | 不把 3cm mask 说成真实物理真值 | log 明确标注为 geometry proxy | PASS |

## 决策

下一步建议：

1. 将 MJWP 的 contact mask 从 scalar 改成 per-EEF `(T,2)`。
2. 增加配置开关，允许从 E077 的 `spider_contact_mask_3cm[:, person_idx, :]` 读取 mask：
   - `person_idx=0`: person1。
   - `person_idx=1`: person2。
3. 先在 `box023_person1` 上跑一个最小变体：
   - base: E075B/E074A 之后的当前 best。
   - 只替换 contact mask，不改 hold_contact window。
   - 对比 f115-f130 right leg phase mismatch 是否减轻。
4. `box023_person2` 可以单独做 sanity run，但不要把 p1/p2 retarget qpos 直接合并成双机器人，除非先解决 common-scale alignment。
