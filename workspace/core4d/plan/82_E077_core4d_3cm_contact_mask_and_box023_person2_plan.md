# E077 实验计划: CORE4D 3cm contact mask 清洗 + box023_person2 数据构造

## Context

E076 修正了 E075 讨论中的过强推断：

- CORE4D raw 没有人工逐帧 hand-contact 真值。
- 官方 benchmark contact 是 SMPL-X/object 几何 proxy：
  - `prepare_hho.py` / `prepare_behave.py`: 2cm。
  - `visualization.py`: 3cm。
- 对 `box023_person1` 的 E075 `f115-f130`，raw 几何显示 person1 右手是 2cm 阈值边界、3cm 阈值持续接触；person2 双手强接触。
- 当前 SPIDER `trajectory_kinematic.npz` 的 `contact=(T,2)` 是转换器全 1，不是 raw/HDMI contact label；MJWP 的 contact mask 也是从单 G1 ref 估计的 scalar proxy。

用户提出两个下一步：

1. 用 `/home/ubuntu/Workspace/CORE4D-Instructions` 里的 3cm 阈值，清洗几个 case 的 contact mask，先把它当成 contact mask 真值。
2. 在 `box023` 上试 person2 数据；构造数据时要仔细核验。

## Claims

| ID | Claim | 成功标准 |
|----|-------|----------|
| C1 | 可以从 CORE4D raw 生成可复现的 3cm per-person/per-hand contact mask | `box023_person1` 输出 `raw_contact_mask_3cm.npz/csv/json`，含 p1/p2 L/R，帧映射明确，E076 关键窗口复现 |
| C2 | 3cm mask 能映射到 SPIDER 30Hz ref 和 MJWP 50Hz eval | 输出 `spider_mask_3cm`、`eval_mask_3cm`，记录 `raw_frame = 42 + ref_frame` 和 `eval_frame/50 -> ref_frame` |
| C3 | `box023_person2` 可以从 raw `20231008/045` 构造成 SPIDER 单人 case | 生成 `example_datasets/.../box023_person2/{scene.xml,task_info.json,0/trajectory_kinematic.npz}`，qpos=(136,43)，MuJoCo FK 成功 |
| C4 | person2 构造不是错帧/错物体 | person2 trimmed qpos 与 retargeted slice 对齐；object trajectory 与 person1 同源；首末帧 object pose/mesh/scene load 通过 |
| C5 | 不把 3cm mask 直接当最终物理真值 | log 明确记录 3cm 是官方可视化 proxy，不是人工真值；后续 reward 接入前必须保留 audit |

## 设计

### A. raw 3cm contact mask generator

新增脚本：

```text
workspace/core4d/scripts/E077/generate_core4d_contact_masks.py
```

输入：

- raw sequence: `20231008/045`
- object mesh: `CORE4D_Real/object_models/box/box023_m.obj`
- threshold: 默认 `0.03`
- sample count: 默认 20000 object surface points
- trim start: 默认 `42`
- fps: raw/ref `30Hz`，eval `50Hz`

输出：

```text
workspace/core4d/results/E077/contact_masks/box023/
  raw_contact_mask_3cm.npz
  raw_contact_mask_3cm.csv
  audit_summary_3cm.json
```

mask 字段：

- `raw_contact_mask_3cm`: `(T_raw, 2, 2)`，axis = person(p1,p2), hand(L,R)。
- `spider_contact_mask_3cm`: `(T_spider, 2, 2)`，使用 raw `[42:178]`。
- `eval_contact_mask_3cm`: `(ceil(T_spider/30*50), 2, 2)`，用 `round(eval_f/50*30)` 映射。
- `min_dist_m`: 对应最小距离，便于软权重设计。
- `n_vertices_lt_thresh`: hand proxy 顶点数，便于判断边界接触。

hand proxy 暂用 E076 审计口径：

- left broad range: `4700:5500`
- right broad range: `7500:8150`
- fingertip ids 只用于 audit，不作为 mask 主判定。

### B. box023_person2 data construction

新增脚本：

```text
workspace/core4d/scripts/E077/build_box023_person2.sh
```

步骤：

1. `convert_core4d_to_omniretarget.py`
   - `--date 20231008 --seq 045 --person person2 --with_object --replace_wrist_with_fingertip`
   - 输出到 `workspace/core4d/results/E077/holosoma_box023_person2/converted/`
2. `robot_retarget.py`
   - task name: `20231008-045-person2-Box023_with_obj`
   - 输出到 `.../retargeted/`
3. 生成 trimmed 数据
   - 若 retargeted 长度与 person1 相同，先使用同样 slice `[42:178]`，保证 person1/person2/object 同一时间窗。
   - 输出 `.../trimmed/20231008-045-person2-Box023_with_obj_original.npz`
4. 生成 SPIDER case
   - 复制/生成 `box023_person2/scene.xml` 与 `task_info.json`。
   - 用 `spider/process_datasets/core4d.py` 转成 `trajectory_kinematic.npz`。
   - 生成 `scene_act.xml`。

核验脚本：

```text
workspace/core4d/scripts/E077/verify_box023_person2.py
```

核验项：

- 文件存在、shape 正确。
- `trimmed == retargeted[42:178]`。
- person2 object qpos 与 person1 object qpos 同源/同窗口。
- MuJoCo load `scene.xml` 和 `scene_act.xml` 成功。
- hand site FK、object collision、contact mask 3cm 时间线输出一致。

## 非目标

- 本轮不直接训练 E077 MJWP。
- 本轮不直接把 3cm mask 接进 `examples/run_mjwp.py` reward。
- 本轮不把 3cm proxy 称为真实物理 contact，只作为官方可视化口径的临时监督。

## 执行命令

```bash
# A: 生成 3cm contact mask
env UV_CACHE_DIR=/tmp/uv-cache uv run python workspace/core4d/scripts/E077/generate_core4d_contact_masks.py

# B: 构造 box023_person2
bash workspace/core4d/scripts/E077/build_box023_person2.sh

# C: 核验
env UV_CACHE_DIR=/tmp/uv-cache uv run python workspace/core4d/scripts/E077/verify_box023_person2.py
```

## 结果路径

| 类型 | 路径 |
|------|------|
| contact masks | `workspace/core4d/results/E077/contact_masks/box023/` |
| person2 holosoma intermediates | `workspace/core4d/results/E077/holosoma_box023_person2/` |
| person2 SPIDER case | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/` |
| E077 log | `workspace/core4d/log/98_E077_3cm_contact_mask_and_person2_results.md` |

## 风险

| 风险 | 处理 |
|------|------|
| hand vertex range 不是官方 segmentation | log 中标为 proxy；后续可换成正式 SMPL-X hand segmentation |
| person2 retarget 可能失败或输出长度不同 | 先停在诊断，不强行构造 SPIDER case |
| scene.xml 复制 person1 可能漏掉 object 初态差异 | 构造后用 person2 qpos[0,36:43] 核验并必要时 patch object initial pose |
| 3cm mask 比 2cm 更宽，可能过度打开边界接触 | 同时保存 min distance 和 vertex count，后续 reward 可做 soft mask |
