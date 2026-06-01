# CORE4D 数据预处理 Pipeline

这个目录记录 E077 已验证成功、并在 E078 中实际使用的 CORE4D 数据预处理流程。当前已验证示例是 raw 序列 `20231008/045` 里的 `box023_person2`。

## E077 验证了什么

E077 验证了后续 CEM 实验需要的两类数据。

### 1. 3cm contact proxy mask

- 来源：CORE4D raw SMPL-X 顶点 + 物体表面 mesh。
- 输出 shape：
  - raw 轴：`(178, 2, 2)` = `(frame, person, hand)`
  - SPIDER trim 轴：`(136, 2, 2)`
  - eval 轴：`(227, 2, 2)`
- person 轴：`person1`, `person2`。
- hand 轴：`left`, `right`。
- 重要说明：这是与 CORE4D 官方可视化口径一致的几何 proxy，不是人工标注的物理接触真值。

### 2. person2 单人 SPIDER case

- Raw sequence: `CORE4D_Real/human_object_motions/20231008/045`
- Person: `person2`
- Object: `Box023`
- Mesh: `object_models/box/box023_m.obj`
- Holosoma retarget 输出按 `raw[42:178]` 裁剪，以对齐已有 `box023_person1` 的 SPIDER 时间窗口；这个窗口来自既有 `box023_person1` Holosoma trimmed 数据与 untrimmed retarget 数据的精确匹配，详见下文“固定裁剪窗口来源”。
- 最终 SPIDER case:
  `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2`

E078 进一步确认了这套 pipeline 是有效的：`box023_person2` 的 reference contact 质量明显好于 `person1`，CEM 得到的 sim 动作也更接近可用质量。

## FPS 速查

box023/E077 当前同步使用下面的 FPS 口径：

| 数据阶段 | FPS |
|----------|-----|
| CORE4D raw | 30Hz |
| OmniRetarget/Holosoma converted input | 30Hz |
| OmniRetarget/Holosoma retarget output | 30Hz |
| SPIDER `trajectory_kinematic.npz` | 30Hz |
| SPIDER/contact trim axis | 30Hz |
| eval contact mask axis | 50Hz |

## 路径配置

这套脚本按下面规则处理路径：

- 本项目内的路径使用相对 `REPO` 的相对路径，例如 `workspace/core4d/results/data_preprocess`。
- 外部项目或数据集路径使用绝对路径；脚本内提供本机默认值，换机器时通过环境变量覆盖，例如 Holosoma repo、CORE4D raw 数据、SMPL-X model。
- `REPO` 默认由 `git rev-parse --show-toplevel` 自动得到；通常不需要手动设置。

`pipeline.sh` 使用的路径变量如下：

| 变量 | 类型 | 默认值 / 要求 |
|------|------|---------------|
| `REPO` | 项目路径 | 当前 git repo |
| `HOLOSOMA_DIR` | 外部绝对路径 | 本机默认 `/home/ubuntu/Workspace/holosoma`；换机器时覆盖 |
| `CORE4D_REAL_ROOT` | 外部绝对路径 | 本机默认 `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/CORE4D/CORE4D_Real`；换机器时覆盖 |
| `SMPLX_MODEL_DIR` | 外部绝对路径 | 本机默认 `/mnt/a0ccc676-9496-49f8-a861-f8a1797dec52/mocap_data/smplx`；换机器时覆盖 |
| `RESULT_ROOT` | 项目内相对路径 | `workspace/core4d/results/data_preprocess` |
| `PYTHON_BIN` | 项目内相对路径或绝对路径 | `.venv/bin/python` |
| `RETARGET_PYTHON_BIN` | 绝对路径，可选 | OmniRetarget/hsretargeting Python；未设置时 source retargeting env 后使用 `$CONDA_PREFIX/bin/python` |
| `REF_FPS` | 标量 | `30.0` |
| `EVAL_FPS` | 标量 | `50.0` |
| `TRIM_MODE` | 字符串 | `holosoma` |

需要的环境：

- 当前 repo 的 Python 环境。默认使用 `.venv/bin/python`；如果需要其它解释器，可以覆盖 `PYTHON_BIN`。
- Holosoma repo，并且存在 `scripts/source_retargeting_setup.sh`。retargeting 阶段会使用 `$CONDA_PREFIX/bin/python` 或显式 `RETARGET_PYTHON_BIN`，避免外层 SPIDER `.venv` 抢占 `python`。
- CORE4D raw 数据和 object mesh。
- SMPL-X model 文件。
- Python 依赖：`numpy`, `trimesh`, `scipy`, `mujoco` 等。

## Box023 示例

在本机可以直接运行，不需要设置路径。换机器时，先设置外部路径；下面三个变量必须是目标机器上的绝对路径：

```bash
export HOLOSOMA_DIR=/abs/path/to/holosoma
export CORE4D_REAL_ROOT=/abs/path/to/CORE4D_Real
export SMPLX_MODEL_DIR=/abs/path/to/smplx
```

先 dry-run 一次，检查将要执行的命令：

```bash
bash workspace/core4d/data_preprocess/pipeline.sh \
  --case-file workspace/core4d/data_preprocess/cases_box023.tsv \
  --dry-run
```

实际运行：

```bash
bash workspace/core4d/data_preprocess/pipeline.sh \
  --case-file workspace/core4d/data_preprocess/cases_box023.tsv
```

指定其它 repo 侧 Python 解释器：

```bash
PYTHON_BIN=/path/to/python \
bash workspace/core4d/data_preprocess/pipeline.sh \
  --case-file workspace/core4d/data_preprocess/cases_box023.tsv
```

如果要覆盖 contact mask 的 FPS 口径：

```bash
REF_FPS=30.0 EVAL_FPS=50.0 \
bash workspace/core4d/data_preprocess/pipeline.sh \
  --case-file workspace/core4d/data_preprocess/cases_box023.tsv
```

如果要复现 E077 的输出目录布局，可以覆盖 `RESULT_ROOT`：

```bash
RESULT_ROOT=workspace/core4d/results/E077 \
bash workspace/core4d/data_preprocess/pipeline.sh \
  --case-file workspace/core4d/data_preprocess/cases_box023.tsv
```

E077 布局下的关键输出：

| 输出 | 路径 |
|------|------|
| 3cm mask npz | `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.npz` |
| 3cm mask csv | `workspace/core4d/results/E077/contact_masks/box023/raw_contact_mask_3cm.csv` |
| contact audit | `workspace/core4d/results/E077/contact_masks/box023/audit_summary_3cm.json` |
| Holosoma converted input | `workspace/core4d/results/E077/holosoma_box023_person2/converted/20231008-045-person2-Box023_with_obj.npz` |
| Holosoma retarget output | `workspace/core4d/results/E077/holosoma_box023_person2/retargeted/20231008-045-person2-Box023_with_obj_original.npz` |
| trimmed output | `workspace/core4d/results/E077/holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz` |
| SPIDER case | `example_datasets/processed/core4d/unitree_g1/humanoid_object/box023_person2/` |

## FPS 与时间轴

box023/E077 当前使用下面这组 FPS 和帧轴。这里的重点是：raw、OmniRetarget/Holosoma retarget 和 SPIDER kinematic reference 都是 30Hz；`eval_contact_mask_3cm` 是为了 MJWP/CEM 评估另行生成的 50Hz mask。

| 阶段 | FPS | box023 帧数 | 说明 |
|------|-----|-------------|------|
| CORE4D raw | 30Hz | 178 | `CORE4D_Real/human_object_motions/20231008/045` 的原始 SMPL-X/object 轨迹；3cm contact proxy 的 `raw_contact_mask_3cm` 也在这条轴上。 |
| OmniRetarget converted input | 30Hz | 178 | `convert_core4d_to_omniretarget.py` 生成的 Holosoma/OmniRetarget 输入；保留 raw 时间轴。 |
| Holosoma/OmniRetarget retarget output | 30Hz | 178 | retarget 后的 humanoid/object npz 内含 `fps=30`。 |
| trimmed output | 30Hz | 136 | 以 `raw[42:178]` 裁剪，用来对齐已有 `box023_person1` 的 SPIDER 窗口。 |
| SPIDER `trajectory_kinematic.npz` | 30Hz | 136 | `spider/process_datasets/core4d.py` 从 trimmed npz 生成；文件本身不保存 `fps` 字段，但 qvel 用源数据 30Hz 的 `dt=1/30` 差分得到。 |
| `spider_contact_mask_3cm` | 30Hz | 136 | 与 SPIDER reference 一一对齐的 contact mask。 |
| `eval_contact_mask_3cm` | 50Hz | 227 | 由 136 帧 30Hz SPIDER 窗口换算得到：`ceil(136 / 30 * 50) = 227`。每个 eval frame 用最近邻映射回 30Hz reference frame。 |
| MJWP/CEM runtime reference | `sim_dt` 决定 | E078 为 272 | `examples/run_mjwp.py::load_data` 会把 30Hz reference 插值到仿真步长，并追加 horizon/control tail；因此 E078 运行时把 227 帧 eval mask resize 到 `qpos_ref.shape[0] = 272`。 |

## Holosoma ref 来源

SPIDER CORE4D case 的 reference 来源应遵循 Holosoma 项目的 CORE4D Collaborate pipeline：

```text
CORE4D raw
  -> convert_core4d_to_omniretarget.py
  -> robot_retarget.py / retarget_core4d_obj_interaction.py
  -> trim_no_contact.py
  -> Holosoma trimmed qpos(T,43), fps=30
  -> spider/process_datasets/core4d.py
  -> SPIDER trajectory_kinematic.npz
```

对应文档：
`$HOLOSOMA_DIR/workspace/pipeline/README.md`

因此，泛化到其它 case 时，**不要**把固定 `[trim_start:trim_end)` slicer 当作默认 trim 逻辑。默认应使用 Holosoma `workspace/pipeline/trim_no_contact.py` 生成 trimmed retarget npz，然后再把这个 trimmed npz 转成 SPIDER reference。

本目录里的 `pipeline.sh` 当前默认：

- 用 Holosoma `convert_core4d_to_omniretarget.py` 做格式转换。
- 用 Holosoma `robot_retarget.py` 做 object interaction retarget。
- 用 Holosoma `workspace/pipeline/trim_no_contact.py` 做自适应 no-contact 前摇裁剪。
- 如果 TSV 显式写了 `trim_start/trim_frames`，直接把该窗口作为权威窗口，用于 contact mask raw/ref/eval 时间轴对齐。
- 如果 TSV 写 `auto`，用 `infer_holosoma_trim_window.py` 反查 trimmed qpos 在 untrimmed qpos 中的位置，得到 `trim_start/trim_frames`。

## 本地辅助脚本与边界

### `generate_core4d_contact_masks.py`

路径：
`workspace/core4d/data_preprocess/generate_core4d_contact_masks.py`

这个脚本已经从 E077 目录迁到 `data_preprocess`。它在 **CORE4D 两人搬运序列** 内是通用的，不绑定 box023/person2；但它不是任意数据集通用工具。

通用输入：

- `--seq-dir`：CORE4D raw sequence 目录，例如 `CORE4D_Real/human_object_motions/20231008/045`。
- `--mesh`：该序列物体 mesh，例如 `CORE4D_Real/object_models/box/box023_m.obj`。
- `--trim-start` / `--spider-frames`：Holosoma trimmed reference 对应的 raw 时间窗。这个窗口应由 `infer_holosoma_trim_window.py` 从 untrimmed/trimmed qpos 反查得到，或来自已经核验过的 Holosoma trim 记录。
- `--ref-fps` / `--eval-fps`：SPIDER reference 轴与 eval mask 轴的 FPS。

硬性假设：

- raw 目录里有 `person1_poses.npz`、`person2_poses.npz` 和 `smooth_objposes.npy`。
- pose 文件里有 SMPL-X `vertices`。
- person 轴固定是 `person1/person2`，hand 轴固定是 `left/right`。
- 手部区域使用当前脚本里的 SMPL-X vertex range 和 fingertip ids；这些是 E076/E077 使用的 proxy，不是 CORE4D 官方 contact annotation。
- 输出文件名和 key 仍沿用 `3cm` 命名，所以默认应使用 `--threshold 0.03`。如果改阈值，后续文件名/key 也应同步改，避免语义错位。

### `infer_holosoma_trim_window.py`

路径：
`workspace/core4d/data_preprocess/infer_holosoma_trim_window.py`

这个脚本不是 trim 工具，而是校验/反查工具。输入 Holosoma untrimmed retarget npz 和 Holosoma trimmed npz，检查：

- trimmed `qpos` 是否精确对应 untrimmed `qpos[start:end]`。
- `trim_start`、`trim_frames`、`trim_end` 分别是多少。
- 如果 TSV 里写了期望窗口，可以用它验证 TSV 是否和 Holosoma trimmed 数据一致。

示例：

```bash
.venv/bin/python workspace/core4d/data_preprocess/infer_holosoma_trim_window.py \
  --untrimmed workspace/core4d/results/E077/holosoma_box023_person2/retargeted/20231008-045-person2-Box023_with_obj_original.npz \
  --trimmed workspace/core4d/results/E077/holosoma_box023_person2/trimmed/20231008-045-person2-Box023_with_obj_original.npz \
  --expect-start 42 \
  --expect-frames 136
```

### `write_trim_window.py`

路径：
`workspace/core4d/data_preprocess/write_trim_window.py`

当 TSV 中已经有人工核验过的 `trim_start/trim_frames` 时，`pipeline.sh` 会用这个脚本写出 `trim_window.json`。这类显式窗口不会再被 Holosoma auto trim 覆盖，也不会强制要求当前 `RESULT_ROOT/holosoma_*` 下存在对应的 untrimmed/trimmed npz。

这个行为用于两类情况：

- 复用已经验证过的 SPIDER case，例如 E079 直接复用 E077/E078 的 `box023_person2`。
- 对 p1/p2 做 common raw window 对齐，并且这个窗口已经在实验日志里给出证据。

泛化到新 case 时，仍推荐先让 Holosoma `trim_no_contact.py` 自动生成 trimmed npz，再用 `infer_holosoma_trim_window.py` 反查窗口；只有窗口来源明确时才写死到 TSV。

### `create_spider_scene_from_template.py`

路径：
`workspace/core4d/data_preprocess/create_spider_scene_from_template.py`

这个脚本是 `pipeline.sh` 当前使用的通用 SPIDER scene 创建入口。它做三件事：

- 从 `source_scene_task` 对应的 `scene.xml` 复制出新 task 的 `scene.xml`。
- 读取 Holosoma trimmed qpos 的第一帧，把 object body 的初始 `pos/quat` 更新到 `qpos[0,36:43]`。
- 写 `task_info.json`，可选生成 `scene_act.xml`。

它不会自动生成新物体的 mesh/collision/mass/inertia；这些仍然来自 `source_scene_task` 模板。因此新物体如果没有合适模板，必须先按
[SCENE_TEMPLATE_GUIDE.md](SCENE_TEMPLATE_GUIDE.md)
制作可加载模板，再把该模板 task 写入 TSV 的 `source_scene_task` 字段。

## 批处理 case 文件

`pipeline.sh` 读取 tab 分隔的 TSV 文件。以 `#` 开头的行会被忽略。

列定义：

```text
enabled date seq person object_name object_model_rel source_scene_task target_task trim_start trim_frames data_id mask_slug
```

示例：

```text
1  20231008  045  person2  Box023  box/box023_m.obj  box023_person1  box023_person2  42  136  0  box023
```

字段含义：

| 字段 | 含义 |
|------|------|
| `enabled` | `1` 表示运行；其它值表示跳过 |
| `date` / `seq` | CORE4D raw 序列目录 |
| `person` | `person1` 或 `person2` |
| `object_name` | Holosoma 使用的物体名，例如 `Box023` |
| `object_model_rel` | 相对 `CORE4D_REAL_ROOT/object_models` 的物体 mesh 路径 |
| `source_scene_task` | 已存在的 SPIDER scene 模板，例如 `box023_person1` |
| `target_task` | 要生成的 SPIDER task 名 |
| `trim_start` / `trim_frames` | Holosoma trimmed reference 对应 untrimmed retarget 的窗口；实际 slice 是 `[trim_start, trim_start + trim_frames)`。可填 `auto`，由 `infer_holosoma_trim_window.py` 反查。 |
| `data_id` | SPIDER data id，通常是 `0` |
| `mask_slug` | contact mask 输出子目录名 |

E079 泛化验证新增了三个批处理列表：

| 文件 | 用途 |
|------|------|
| `workspace/core4d/data_preprocess/cases_E079_existing_p1.tsv` | 6 个已有 p1 SPIDER case，只跑 Holosoma trim/window 反查和 3cm contact mask；通常配合 `--skip-spider`，避免覆盖已有 scene。 |
| `workspace/core4d/data_preprocess/cases_E079_existing_p2.tsv` | 已经由 E077/E078 验证过的 p2 SPIDER case，例如 `box023_person2`；只重建 E079 contact mask，不覆盖已有 scene/trajectory。 |
| `workspace/core4d/data_preprocess/cases_E079_build_p2.tsv` | p2 单人 case 构造列表，用同物体 p1 scene 模板构造 `person2` SPIDER case；`desk021_person2` 已知 Holosoma retarget infeasible，默认禁用。 |

E079 推荐入口：

```bash
bash workspace/core4d/scripts/run_E079_preprocess.sh dry-run
bash workspace/core4d/scripts/run_E079_preprocess.sh all
```

### Holosoma 裁剪窗口来源

box023/E077 使用的窗口是：

```text
trim_start = 42
trim_frames = 136
trim_end = 178
slice = [42:178]
```

这个窗口的来源是 Holosoma pipeline 的 trimmed 数据，而不是 SPIDER 侧拍脑袋写死：

- Holosoma 里已有 `box023_person1` untrimmed retarget 输出，长度是 178 帧。
- Holosoma 里已有 `box023_person1` trimmed 输出，长度是 136 帧。
- 核验发现 trimmed `qpos` 精确等于 untrimmed `qpos[42:178]`。
- 现有 SPIDER `box023_person1/0/trajectory_kinematic.npz` 也是 136 帧。
- 因此 E077 构造 `box023_person2` 时先用同一窗口作为 TSV 期望值，并用 `infer_holosoma_trim_window.py` 校验 trimmed 输出确实对应 `[42:178]`。

30Hz 下，`42` 帧约等于 1.4s，`136` 帧约等于 4.53s。

新 case 的窗口应按下面顺序确定：

1. 先按 Holosoma pipeline 生成 untrimmed retarget 和 trimmed retarget。
2. 用 `infer_holosoma_trim_window.py` 反查 trimmed 对应 untrimmed 的 slice。
3. 如果要在 TSV 中显式记录窗口，写入反查得到的 `trim_start/trim_frames`；如果希望运行时自动反查，可写 `auto`。
4. 如果要构造 p1/p2 对照，必须额外检查两个 person 的 Holosoma trim window 是否一致；如果不一致，需要决定是尊重各自 Holosoma trim，还是为了双人对齐另做 common raw window，并把这个决定写入实验日志。
5. contact mask 的 `raw_frame = trim_start + ref_frame` 必须和最终 SPIDER reference 使用的 Holosoma trimmed qpos 一致。

### 与 Holosoma `trim_no_contact.py` 的区别

`$HOLOSOMA_DIR/workspace/pipeline/trim_no_contact.py` 是自适应裁剪工具：

- 它在 retarget 后的 robot/object MuJoCo 模型里做 FK。
- 用若干 hand body 点到 object mesh 的 signed distance 判断接触。
- 默认 `contact_threshold=0.05m`、`sustained_contact_duration=0.5s`、`no_contact_duration=0.5s`、`margin=0.5s`。
- 它只裁掉开头的 no-contact 段，输出 `[computed_trim_start:]`，通常没有固定 `trim_end`。
- 每个文件会独立估计 trim start，所以 p1/p2 可能得到不同窗口。

本地 `workspace/core4d/scripts/E077/trim_box023_person2.py` 只是 E077 历史兼容脚本：

- 它只服务 `box023_person2` 的历史复现，不是通用预处理入口。
- 新 case 不应依赖它。
- 泛化 pipeline 应使用 Holosoma `trim_no_contact.py`，再用 `infer_holosoma_trim_window.py` 做窗口核验。

### 如果没有 `source_scene_task` 模板

当前 E077 pipeline 还不是完全从零生成 MuJoCo scene。`source_scene_task` 不能为空，因为后续 SPIDER 转换需要先有：

- `scene.xml`：Unitree G1 robot + object freejoint + object mesh/collision geom + hand contact sites。
- `task_info.json`：推荐保留，用来记录模板来源、qpos 来源、object mesh、mass/inertia 等 provenance 信息。
- 后续 `scene_act.xml`：由 `scene.xml` 生成，用于 MJWP/CEM 的 actuator 版本场景。

现在的 `create_box023_person2_scene.py` 做的是“复制一个已有 `scene.xml`，再用 retarget qpos 的第一帧更新 object 初始 `pos/quat`”。这适合 `box023_person1 -> box023_person2` 这种同物体、同机器人、同场景结构的 case。

如果一个新 case 没有现成模板，需要先补一个 scene 生成步骤。推荐顺序是：

1. 优先找同物体或同类别物体的已有 SPIDER case 作为模板。
2. 如果只有近似模板，按 `workspace/core4d/scripts/convert/setup_new_cases.py` 的方式替换 object mesh、material、collision box、mass/inertia 和初始 pose。
3. 生成后必须用 MuJoCo load 校验 `scene.xml`，并确认 `nq=43`、hand contact sites 数量为 2、object qpos 对应 `qpos[:,36:43]`。
4. 再运行 `workspace/core4d/scripts/convert/generate_scene_act.py` 生成 `scene_act.xml`。

也就是说：没有模板时不能只在 TSV 里留空 `source_scene_task`；需要先创建一个可加载的 SPIDER scene，然后再把它填到 `source_scene_task`。

更详细的模板制作步骤、需要修改的字段、修改依据和校验清单见
[SCENE_TEMPLATE_GUIDE.md](workspace/core4d/data_preprocess/SCENE_TEMPLATE_GUIDE.md)。

## 处理步骤

对每个启用的 case，`pipeline.sh` 会依次执行：

1. 将 CORE4D raw 转成 Holosoma/OmniRetarget 输入格式：
   `holosoma/workspace/pipeline/convert_core4d_to_omniretarget.py`
2. 运行 Holosoma robot retarget：
   `holosoma_retargeting/examples/robot_retarget.py`
3. 裁剪 retarget 输出：
   `$HOLOSOMA_DIR/workspace/pipeline/trim_no_contact.py`
4. 确定 trim window：TSV 显式窗口直接写入，`auto` 窗口由 `infer_holosoma_trim_window.py` 从 Holosoma 输出反查。
5. 生成 3cm contact proxy：
   `workspace/core4d/data_preprocess/generate_core4d_contact_masks.py`
6. 基于已有 scene 模板创建 SPIDER scene：
   `workspace/core4d/data_preprocess/create_spider_scene_from_template.py`
7. 将裁剪后的 npz 转成 SPIDER `trajectory_kinematic.npz`：
   `spider/process_datasets/core4d.py`
8. 生成 `scene_act.xml`。
9. 校验 MuJoCo load 和 trajectory shape：
   `workspace/core4d/data_preprocess/verify_processed_case.py`

## 重要 caveat

E077 发现：converted 层的 CORE4D object pose 在 person1/person2 之间完全一致；但 Holosoma retarget preprocess 会按每个人的 `smpl_scale` 缩放 object motion。以 box023 为例，p1/p2 在 retarget 层的 object qpos 最大差约 6.2cm。

因此：

- `box023_person2` 可以作为 **单人 SPIDER case** 使用。
- 不要直接把 p1 和 p2 的 retarget qpos 合并成一个双机器人场景。
- 做双机器人同场景之前，需要先实现 common-scale / common-world alignment。
