# E083 Plan: upper-body-object collision for E082 body-fall failures

日期：2026-05-28

## Context

`workspace/core4d/log/104_E082_body_fall_upperbody_collision_diagnosis.md` 已确认 E082 的三条 D003 Box021 失败不是预处理链路坏掉，而是当前派生 `scene_act.xml` 仍存在明确的物理约束缺口：

- E081/E082 只给腿/脚和 `object_collision` 增加了 contact pair；
- `head_collision`、`torso_collision`、`pelvis_collision`、shoulder/elbow 与 `object_collision` 没有 pair；
- 手-地面 pair 已存在，因此 CEM 可利用“头/躯干穿箱 + 手撑地”的局部解；
- 三个 Box021 失败 case 的 sim head/torso 穿入率分别为 `76.0/85.3%`、`17.2/51.1%`、`32.4/58.8%`，但 ref head/torso SDF 仍为正。

E083 直接沿 log 104 推荐的 E083A 推进：只改派生 scene 的 contact pairs，不先加 reward penalty。这样可以隔离验证“补上上半身-物体碰撞”本身是否足以消除头/躯干穿箱，并观察它会不会把失败模式转成撞箱、胸/肩推箱或更早摔倒。

## 实验矩阵

主实验跑 E082 的 3 个 D003 Box021 case；额外加 `box023_person2` guard，防止 upper-body pair 破坏此前较好的 box023_p2。

| Variant | Source task | Derived task | Person | Split | Role |
|---|---|---|---:|---|---|
| `E083_d003_box021_20231018_029_p2_upperobj` | `d003_box021_20231018_029_p2` | `d003_box021_20231018_029_p2_upperobj_e083` | 1 | local | main |
| `E083_d003_box021_20231011_035_p2_upperobj` | `d003_box021_20231011_035_p2` | `d003_box021_20231011_035_p2_upperobj_e083` | 1 | remote-gpu0 | main |
| `E083_d003_box021_20231020_019_p1_upperobj` | `d003_box021_20231020_019_p1` | `d003_box021_20231020_019_p1_upperobj_e083` | 0 | remote-gpu1 | main |
| `E083_box023_p2_upperobj_guard` | `box023_person2` | `box023_person2_upperobj_e083` | 1 | remote-gpu1 | guard |

远程执行遵循 `.codex/skills/experiment-planning-zh/remote-execution.md`：

- 本地 GPU0 跑最典型失败 `20231018_029_p2`；
- 远程 GPU0 跑 `20231011_035_p2`；
- 远程 GPU1 串行跑 `20231020_019_p1` 和 `box023_p2` guard；
- 不 kill 任何已有实验或 tmux；如果同名 E083 session 已存在，启动脚本直接退出。

## Claims

| Claim | 验证方式 | 成功/解释标准 |
|---|---|---|
| C1 不污染 source task | diff/路径检查 | 原始 `source_task/scene_act.xml` 不改，只新建 `*_upperobj_e083` 派生 task |
| C2 leg/foot pair 保持 | scene inspection | 每个派生 scene 仍有 16 个腿/脚-`object_collision` pair |
| C3 upper-body pair 生效 | scene inspection + eval | 每个派生 scene 有 head/torso/pelvis/shoulder/elbow 到 `object_collision` 的 pair |
| C4 三个 Box021 main 均完成 full CEM | 文件检查 | 3/3 `.npz`、`.mp4`、eval summary、keyframes/sheet 存在 |
| C5 guard 完成且不明显退化 | eval + 视觉 | `box023_p2` 不出现 head/torso 穿箱、手撑地或明显倒伏 |
| C6 失败机制被重新判定 | upperbody metrics + 视频 | 明确区分“穿箱被修掉但仍倒伏”和“仍有穿模/接触漏洞” |
| C7 可视化和 subagent high 复核完成 | keyframes + subagent report | log 中必须有逐 case 视觉观察，不只看数值 |

## 改动

新增：

- `workspace/core4d/scripts/E083/variants.tsv`
- `workspace/core4d/scripts/E083/create_upperobj_cases.py`
- `workspace/core4d/scripts/E083/generate_e083_overrides.py`
- `workspace/core4d/scripts/E083/run_remote_inside.sh`
- `workspace/core4d/scripts/run_E083_preprocess.sh`
- `workspace/core4d/scripts/train/train_E083.sh`
- `workspace/core4d/scripts/run_E083_remote.sh`
- `workspace/core4d/scripts/pull_E083_remote_results.sh`
- `workspace/core4d/scripts/eval/eval_E083.py`
- `workspace/core4d/scripts/eval/extract_E083_contact_sheets.sh`

复用：

- E082 的 D003 mask 来源：`workspace/core4d_collab_retarget/results/E029/d6/contact_masks/...`
- E079 的 `box023_person2` 3cm mask；
- `eval_E081.py` 的 leg-object / object-floor 指标；
- `diagnose_E082_body_fall.py` 的 head/torso/pelvis/hand-floor 诊断逻辑，扩展到 E083。

## 成功标准

基础完成：

- 静态检查通过：新增 Python `py_compile`、新增 shell `bash -n`、`git diff --check`；
- 预处理完成：4 个派生 task、4 个 override、4 份 scene snapshot；
- 本地+远程运行完成：4/4 `.npz` 与 `.mp4` 回收到本地；
- 统一评估完成：`comparison.csv`、`aggregate_summary.json`、`upperbody_diagnostics.csv` 存在；
- 可视化完成：4/4 keyframes 或 contact sheet 存在，subagent high 给出逐 case 视觉判断。

主判据：

- 三个 Box021 main 的 `case_window_sim_head_collision_object_penetration_pct` 与 `case_window_sim_torso_collision_object_penetration_pct` 应显著低于 E082；
- 若仍倒伏，需要确认失败是否从“穿箱漏洞”转为“被物体碰撞顶翻/胸肩推箱/手撑地”；
- 不把仅 object tracking 好但 hand-floor 或 upperbody collision 异常的结果接后续 RL。

## 风险

| 风险 | 处理 |
|---|---|
| 只加 pair 导致更早摔倒 | 这是 E083A 要验证的核心；若发生，进入 E083B reward penalty/stability guard |
| 碰撞 pair 太多导致 solver 慢或接触抖动 | 先复用 E081 solref/friction/condim；记录运行时间和视频 |
| hand-floor 仍是局部解 | eval 直接输出 `lh/rh_floor_contact_pct`，视觉复核必须标注 |
| guard 退化 | 若 box023 guard 明显变差，说明 upper-body pair 需要 margin/pair 范围收缩，而不能全局推广 |
