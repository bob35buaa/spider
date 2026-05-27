# E082 Plan: E081 leg-object pipeline on 3 D003 Box021 cases

日期：2026-05-27

## Context

`workspace/core4d_collab_retarget` 的 E029/E030 把问题推向了 freejoint/support-body 语义，复杂度很高，而且 E030 证明 D6 locked support scaffold 接 CEM 后会退化。用户判断：如果后续目标是给 RL 提供可用轨迹，不一定需要在 SPIDER 重定向阶段坚持 true-freejoint；可以回到 `workspace/core4d/log/102_E081_leg_object_collision_results.md` 的路线。

E081 的核心做法是：

- 保留 `scene_act.xml` 的 object actuator / kinematic-object 口径；
- 不直接修改原始 task；
- 新建 `*_legobj` 派生 task；
- 只在派生 `scene_act.xml` 里新增腿/脚-`object_collision` contact pairs；
- 使用 E079/E080 的 no-hold + 3cm per-EEF mask + case-specific window 评估；
- 用新增 `leg_box_interference`、`leg_object_contact`、`object_floor_contact`、`object_bottom_proxy` 判断物理合理性。

E082 把这条 E081 路线扩展到前面反复讨论的 3 个 D003 Box021 case：

| Variant | Source task | Person | 来源 |
|---|---|---:|---|
| `E082_d003_box021_20231018_029_p2_legobj` | `d003_box021_20231018_029_p2` | 1 | E030 local case |
| `E082_d003_box021_20231011_035_p2_legobj` | `d003_box021_20231011_035_p2` | 1 | E030 remote GPU0 case |
| `E082_d003_box021_20231020_019_p1_legobj` | `d003_box021_20231020_019_p1` | 0 | E030 remote GPU1 case |

本地检查确认这 3 个 source task 都已有：

- `scene.xml`
- `scene_act.xml`
- `scene_act_meta.json`
- `task_info.json`
- `0/trajectory_kinematic.npz`

且原始 `scene_act.xml` 目前只有 `left_hand_object`、`right_hand_object`、`object_floor` 三类物体相关 contact pair，没有腿/脚-物体 pair，正好符合 E081 派生策略。

3cm contact masks 已存在于：

```text
workspace/core4d_collab_retarget/results/E029/d6/contact_masks/<source_task>/raw_contact_mask_3cm.npz
```

E082 会复制到：

```text
workspace/core4d/results/E082/contact_masks/<source_task>/
```

## Claims

| Claim | 验证方式 | 成功/解释标准 |
|---|---|---|
| C1 E082 在 `workspace/core4d` 工作区完成 | plan/log/scripts/results 均在 `workspace/core4d`；训练日志在 `logs/E082` | 不在 `core4d_collab_retarget` 新增 E082 运行逻辑 |
| C2 E077-E081 数据链路被复用而不是重写 | 计划和日志记录 E077 masks、E078 per-EEF、E079 no-hold 泛化、E080/E081 leg-object 指标 | 明确 E082 是 E081 扩展，不是 support-body/freejoint 分支 |
| C3 不污染原始 D003 source task | diff/路径检查 | 原始 `d003_box021_* / scene_act.xml` 不改，只新增 `*_legobj_e082` 派生目录 |
| C4 腿/脚-物体 contact pair 生效 | scene inspection + eval | 每个派生 `scene_act.xml` 增加 16 个腿/脚-`object_collision` pair，eval 产生 `leg_object_contact` 字段 |
| C5 三个 case 都完成 full CEM | 文件检查 | 3/3 root `.npz`、3/3 outdir trajectory、3/3 `.mp4`、3/3 keyframes |
| C6 本地 1 卡 + 远程 2 卡并行 | tmux/log/GPU 分配 | 本地 GPU0 跑 `20231018_029_p2`，远程 GPU0/GPU1 分别跑另外两个 |
| C7 后续 RL 价值判断基于视觉 + E081 指标 | eval + subagent 视觉审查 | 不只看 object mean；同步看接触、腿干涉、floor-contact、bottom proxy、倒伏/穿插 |

## 改动

新增：

- `workspace/core4d/scripts/E082/variants.tsv`
- `workspace/core4d/scripts/E082/create_legobj_cases.py`
- `workspace/core4d/scripts/E082/generate_e082_overrides.py`
- `workspace/core4d/scripts/run_E082_preprocess.sh`
- `workspace/core4d/scripts/train/train_E082.sh`
- `workspace/core4d/scripts/run_E082_remote.sh`
- `workspace/core4d/scripts/pull_E082_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E082.py`

复用：

- `workspace/core4d/scripts/eval/eval_E081.py` 的 leg-object 指标逻辑；
- `workspace/core4d/scripts/eval/eval_E078.py` / `eval_E079.py` 的 scene_act replay、case-specific window；
- E029/E030 已生成的 D003 3cm contact masks。

## 运行安排

| Split | GPU | Variant |
|---|---:|---|
| local | local GPU0 | `E082_d003_box021_20231018_029_p2_legobj` |
| remote-gpu0 | remote GPU0 | `E082_d003_box021_20231011_035_p2_legobj` |
| remote-gpu1 | remote GPU1 | `E082_d003_box021_20231020_019_p1_legobj` |

远程同步说明：

当前 worktree 里另一个工作区有未提交 E030 记录；E082 不应为了远程运行强制清理或提交无关内容。因此 E082 远程启动脚本采用按需 `rsync -R` 同步本实验需要的脚本、overrides、派生 scene/data、contact masks 和 Box021 asset，而不是要求整个 git tree clean。

## 命令

```bash
# 1. 生成派生 case、复制 mask、生成 override
bash workspace/core4d/scripts/run_E082_preprocess.sh

# 2. 本地代表 case
bash workspace/core4d/scripts/train/train_E082.sh local 0

# 3. 远程 2 卡并行
bash workspace/core4d/scripts/run_E082_remote.sh

# 4. 回收远程结果并统一评估
bash workspace/core4d/scripts/pull_E082_remote_results.sh
bash workspace/core4d/scripts/train/train_E082.sh eval
```

## 成功标准

- 静态检查通过：新增 Python `py_compile`、新增 shell `bash -n`、`git diff --check`。
- 预处理完成：
  - 3 个 `*_legobj_e082` 派生 task 存在；
  - 3 个派生 `scene_act.xml` 包含 16 个新增腿/脚-物体 pair；
  - 3 个 E082 overrides 指向 `workspace/core4d/results/E082/contact_masks/...`。
- 运行完成：
  - 3/3 `.npz` 和 `.mp4` 存在；
  - 远程结果已回收到本地；
  - `workspace/core4d/results/E082/comparison.csv` 与 `aggregate_summary.json` 存在。
- 可视化完成：
  - 3/3 keyframes 或视频 sheet 存在；
  - subagent 给出逐 case 视觉审查。

## 风险

| 风险 | 处理 |
|---|---|
| scene_act 口径不是 true-freejoint | 本轮接受该限制；目标是给 RL 寻找更可用的动作种子，而不是证明纯物理搬运 |
| D003 case 不是 E079 原始 6-case canonical naming | E082 使用 source task 的 `task_info.json` 和已有 3cm mask，不重跑 raw preprocessing |
| 远程缺 ignored datasets | `run_E082_remote.sh` 显式 rsync 派生 task、overrides、masks、Box021 asset |
| 远程已有同名 tmux | 不 kill 其他 session；若 `E082_remote_d003_box021` 已存在，脚本应退出并提示 |
| E082 仍失败 | 记录为 E081 route 对 D003 Box021 的负结果，不回到 D6 support scaffold 小调参 |
