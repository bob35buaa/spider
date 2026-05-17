# E002 Plan: freejoint leg-object control audit

日期：2026-05-17

## Context

E001 代码审计确认：E081 baseline 不是纯 freejoint 物体。当前 active E079-E081 链路继承 `core4d_e041c`，使用 `scene_act` + `contact_guidance`；object 从 freejoint ref 转为 3 slide + 3 hinge，6 个 object actuator 的 target 来自 ref object trajectory。CEM 基本是在 actuator-guided object trajectory 下优化 robot controls。

用户最先提出的问题是：物体状态是否是 GT；如果物体是 free joint、全靠机器人力和物理接触移动，这套方案还能不能 work。E002 直接做这个最小对照。

## Claims

| Claim | 最低证据 |
|-------|----------|
| C1 E002 真正移除了 object actuator guidance | 派生 task 使用 `scene.xml` freejoint object；override 中 `contact_guidance=false`、`scene_name=""`；运行日志/配置显示 `nu=29`、`nq_obj=7`，无 object actuator ids。 |
| C2 腿/脚-object 物理 contact 在 freejoint scene 中生效 | 派生 `scene.xml` 新增 16 个腿/脚-`object_collision` pair；eval 能统计 leg contact/interference。 |
| C3 box025_p2 在真 freejoint 下的能力边界可量化 | 与 E081 `box025_p2_legobj` 同口径比较 object mean/max、hand contact、leg interference、floor contact、bottom mean、视频关键帧。 |
| C4 box023_p2 guard 能判断 actuator 依赖程度 | 若 box023 freejoint 也明显退化，说明当前 pipeline 强依赖 object actuator guidance；若 guard 保持，说明小箱真实接触仍有可行性。 |

## 改动

### 1. 新增 E002 派生 task 生成脚本

**文件**: `workspace/core4d_collab_retarget/scripts/E002/create_freejoint_legobj_cases.py`

从原始 source task 复制：

- `scene.xml`
- `task_info.json`
- `0/trajectory_kinematic.npz`

只在派生 `scene.xml` 中追加腿/脚-`object_collision` contact pairs；不复制/不使用 `scene_act.xml`。

派生 task：

- `box025_person2_freejoint_legobj`
- `box023_person2_freejoint_legobj`

### 2. 新增 E002 overrides

**文件**:

- `examples/config/override/core4d_collab_E002_box025_p2_freejoint.yaml`
- `examples/config/override/core4d_collab_E002_box023_p2_freejoint.yaml`

继承 E081/E074 reward 口径，但强制：

```yaml
task: <derived_task>
scene_name: ""
contact_guidance: false
object_action_dims: 0
object_actuator_ids: []
object_actuator_names: []
contact_hdmi_mask_source: core4d_3cm
hold_contact_rew_scale: 0.0
```

保留：

- local-frame tracking
- task object reward
- 3cm per-EEF contact mask gate
- dynamic contact target / palm normal
- E074A robot ctrl guard

### 3. 新增 E002 train/eval 脚本

**文件**:

- `workspace/core4d_collab_retarget/scripts/E002/variants.tsv`
- `workspace/core4d_collab_retarget/scripts/run_E002_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/train/train_E002.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E002.py`

E002 eval 复用 E081 指标定义，但要支持 freejoint `scene.xml` / `nq_obj=7`，不能调用强制 scene_act 转换的 E078 `load_ref`。

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d_collab_retarget/plan/02_E002_freejoint_legobj_control_audit_plan.md` | 本计划 |
| 2 | `workspace/core4d_collab_retarget/scripts/E002/create_freejoint_legobj_cases.py` | 生成 freejoint 派生 task |
| 3 | `workspace/core4d_collab_retarget/scripts/E002/generate_e002_overrides.py` | 生成 overrides 并复制 mask |
| 4 | `workspace/core4d_collab_retarget/scripts/E002/variants.tsv` | variant 表 |
| 5 | `workspace/core4d_collab_retarget/scripts/run_E002_preprocess.sh` | 预处理入口 |
| 6 | `workspace/core4d_collab_retarget/scripts/train/train_E002.sh` | CEM 运行入口 |
| 7 | `workspace/core4d_collab_retarget/scripts/eval/eval_E002.py` | E081 口径 freejoint eval |
| 8 | `workspace/core4d_collab_retarget/log/02_E002_freejoint_legobj_control_audit_results.md` | 结果日志 |
| 9 | `workspace/core4d_collab_retarget/progress.md` | 进展记录 |
| 10 | `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md` | 完成后更新 |

## 运行命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E002_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E002.sh smoke 0
bash workspace/core4d_collab_retarget/scripts/train/train_E002.sh local 0
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E002.py
```

如果本地单卡太慢，可把 guard `box023_p2` 放到远程 GPU1，但 E002 只有 2 个独立实验，不强制远程并行。

## 成功标准

| 指标 | E081 baseline | E002 解释标准 |
|------|---------------|----------------|
| `nu` / object actuator | E081 `nu=35`, object actuator guided | E002 必须 `nu=29`，无 object actuator |
| box025 obj mean/max | `0.143/0.271m` | 若显著退化，说明 E081 的 box025 partial positive 依赖 object actuator；若接近，则真接触有希望 |
| box025 floor/bottom | floor `59.5%`, bottom `-0.075m` | 若 bottom 更低/落地更多，freejoint 失败；若改善则可进入 partner/contact 实验 |
| box025 leg intf | `7.5%` | 应保持 `<=7.5%`，否则物理 contact 未解决 leg/box 干涉 |
| box023 guard | strict proxy True | 若 guard 退化，说明 actuator guidance 对小箱也关键；若保持，后续可用小箱做 freejoint 方法开发 |
| 可视化 | E081 f100/f125/f160 | 必须抽取/检查同帧，确认不是腿/地板/穿模假支撑 |
