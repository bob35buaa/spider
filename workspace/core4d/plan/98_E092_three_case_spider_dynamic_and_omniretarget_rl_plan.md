# E092 Plan: three-case SPIDER dynamics and OmniRetarget RL comparison

日期：2026-05-29

关联输入：

- E091 结果：`workspace/core4d/log/113_E091_data_construction_v2_medium_box_results.md`
- E090 结果：`workspace/core4d/log/112_E090_h2_first_retarget_and_spider_smoke_results.md`
- E091 D005b summary：`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/d005b_g1_feasibility/d005b_summary.tsv`
- E091 OmniRetarget 可视化：`/home/ubuntu/Workspace/holosoma/workspace/v3/data_construction_v2/results/omniretarget_visuals/summary.md`

## 0. 计划定位

本计划是 E091 之后的下一步整体规划，目标不是继续扩大数据筛选，而是对已完成 OmniRetarget 可视化的三条 case 做两条输入路线的对比：

1. **SPIDER 动力学重定向路线**：每个 case 先跑 SPIDER/MJWP 动态重定向；只有动态结果达到 `WORK` 标准的 case，才把该动态序列作为 RL 输入。
2. **OmniRetarget 直接路线**：同三条 case 直接使用 OmniRetarget/Stage2b 产出的 SPIDER task/reference 跑 RL，作为对照。

核心问题：RL 失败到底主要来自 OmniRetarget 几何/reference 本身，还是来自后续动力学/姿态优化局部解？如果 SPIDER 动态序列能修好姿态，再作为 RL 输入显著优于直接 OmniRetarget，则后续数据构建应把 dynamic retarget 作为 RL 前置 filter；如果直接 OmniRetarget RL 已经可行，则不应把 E091 smoke 的 pelvis collapse 过度归因为上游数据。

## 1. 三条 case

本计划中的“三条 case”默认指 E091 已成功完成 OmniRetarget 可视化的三条 Stage2b case：

| ID | task | object | E091 D005b | 关键状态 | 本轮角色 |
|---|---|---|---|---|---|
| C1 | `e091_box004_20231003_2_083_p2` | `box004` | PASS | inside `0/0%`，support either `42.9%`，pelvis min `0.679m`；E091 minimal smoke pelvis collapse | positive seed / 主对照 |
| C2 | `e091_box026_20231018_039_p2` | `Box026` | REJECT | support either `19.5% < 30%`，inside/pelvis 均好 | low-support near-pass |
| C3 | `e091_box026_20231020_135_p2` | `Box026` | REJECT | support `62.2%`，但 right wrist inside `12.2% > 10%` | inside-risk near-pass |

明确不纳入本轮三 case：`e091_box026_20231018_040_p2`。它在 E091 no-fingertip OmniRetarget 中 CVXPY infeasible，没有可用 retargeted/trimmed NPZ，不能直接跑 OmniRetarget RL；除非单独开 retarget 修复实验。

## 2. Claims

| Claim | 验证方式 |
|---|---|
| C1: 至少一条 E091 case 可以通过 SPIDER 动力学重定向得到可用动态序列 | 三条 case 均跑 dynamic smoke；通过者再跑 dynamic full；`WORK` 定义见第 5 节 |
| C2: 对同一 case，RL from SPIDER-dynamic sequence 应优于 RL from direct OmniRetarget sequence | 对有 `WORK` 动态序列的 case，跑 paired RL：`rl_from_spider` vs `rl_from_omni`，比较 safety、pelvis、object tracking、contact continuity 和视频 |
| C3: Box026 near-pass 的失败类型可以被动态重定向区分 | C2/C3 若动态能修 support/inside 并跑出 RL，则 D005b 阈值/variant 需要调整；若仍失败，则 Box026 no-fingertip 路线暂不扩量 |
| C4: E091 box004 pelvis collapse 是动力学后续问题，而不是数据筛选彻底失败 | C1 若 dynamic 或 RL 任一路线达到 pass/review+，则保留 box004 为 medium seed；若两路都 pelvis collapse，则下一轮优先做姿态/upright 约束 |

## 3. 实验矩阵

### 3.1 Stage A: SPIDER 动力学重定向

每个 case 先跑一条动态重定向序列：

| variant | source task | derived dyn task | 预期 |
|---|---|---|---|
| `E092D1_box004_083_p2_dyn` | `e091_box004_20231003_2_083_p2` | `e091_box004_20231003_2_083_p2_e092_dyn` | 最可能 WORK；重点看 pelvis 是否从 E091 smoke 的 `0.079m` 拉回 |
| `E092D2_box026_039_p2_dyn` | `e091_box026_20231018_039_p2` | `e091_box026_20231018_039_p2_e092_dyn` | 验证 low support 是否能由动态接触补偿 |
| `E092D3_box026_135_p2_dyn` | `e091_box026_20231020_135_p2` | `e091_box026_20231020_135_p2_e092_dyn` | 验证 wrist-inside 风险是否能由动态安全项推出 |

执行分两层：

1. **dynamic smoke**：`max_num_iterations=4`，用于快速验证 load、碰撞、pelvis、object tracking 和视频。
2. **dynamic full**：只对 smoke `PASS` 或 `REVIEW+` 的 case 跑默认完整迭代。full 达到 `WORK` 才进入 `rl_from_spider`。

### 3.2 Stage B: RL from SPIDER dynamic sequence

只对 Stage A full 判定 `WORK` 的 case 跑。输入不是原始 OmniRetarget qpos，而是 SPIDER dynamic full 输出序列：

```text
workspace/core4d/results/E092/spider_dyn/<variant>.npz
```

每个通过 case 跑两档：

1. **RL smoke**：短训练/短 rollout，验证 task/override/checkpoint/eval 路径和 reward 无 NaN。
2. **RL main**：smoke 稳定后再跑正式训练；如果机器资源紧张，先每 case 1 seed，只有 winner 再补 2 个 seeds。

若 Stage A 没有任何 `WORK` case，本阶段跳过，不把 `REVIEW` 动态序列强行喂给 RL。

### 3.3 Stage C: RL from direct OmniRetarget sequence

三条 case 全部跑，作为必要对照。输入使用 E091 Stage2b 已生成的 SPIDER task/reference：

| variant | task | 备注 |
|---|---|---|
| `E092O1_box004_083_p2_omni` | `e091_box004_20231003_2_083_p2` | D005b pass，必须跑 |
| `E092O2_box026_039_p2_omni` | `e091_box026_20231018_039_p2` | D005b support reject，作为 low-support 对照 |
| `E092O3_box026_135_p2_omni` | `e091_box026_20231020_135_p2` | D005b inside reject，作为 inside-risk 对照 |

这里不因 D005b reject 预先跳过 C2/C3；目标正是验证 D005b reject 对 RL 是否有预测力。

## 4. 需要落地的脚本和产物

新增脚本建议：

| 类型 | 路径 | 作用 |
|---|---|---|
| case/task build | `workspace/core4d/scripts/E092/build_three_case_tasks.py` | 从三条 E091 task 生成 dynamic/RL 派生 task，注入 leg/upper-body object collision pairs，校验 `scene_act` |
| variant manifest | `workspace/core4d/scripts/E092/variants.tsv` | 统一记录 C1/C2/C3、source task、derived task、object、route、split |
| SPIDER dyn train | `workspace/core4d/scripts/train/train_E092_spider_dyn.sh` | 跑 dynamic smoke/full，先 snapshot scenes |
| SPIDER dyn eval | `workspace/core4d/scripts/eval/eval_E092_spider_dyn.py` | 计算 object/safety/pelvis/contact 指标，抽 keyframes |
| RL task adapter | `workspace/core4d/scripts/E092/build_rl_tasks.py` | 把 `omni_ref` 或 `spider_dyn_ref` 规范化成 RL 可读输入 |
| RL from dynamic | `workspace/core4d/scripts/train/train_E092_rl_from_spider.sh` | 对 Stage A `WORK` case 跑 RL |
| RL from Omni | `workspace/core4d/scripts/train/train_E092_rl_from_omni.sh` | 三条 direct OmniRetarget 对照 RL |
| RL eval | `workspace/core4d/scripts/eval/eval_E092_rl.py` | 同一套指标评估两条路线 |
| remote runner | `workspace/core4d/scripts/run_E092_remote.sh` | 三条 case 以上并行时使用 2-GPU 远程分发 |

产物路径：

```text
workspace/core4d/results/E092/
  scene_snapshot/
  spider_dyn/
    smoke/
    full/
    keyframes/
    spider_dyn_summary.{json,csv,md}
  rl_from_spider/
    smoke/
    main/
    eval_summary.{json,csv,md}
  rl_from_omni/
    smoke/
    main/
    eval_summary.{json,csv,md}
  comparison/
    paired_route_comparison.{csv,md}
    route_decision_tree.md

logs/E092/
  spider_dyn/
  rl_from_spider/
  rl_from_omni/
```

正式结果日志预留：`workspace/core4d/log/114_E092_three_case_spider_dynamic_and_omniretarget_rl_results.md`。

## 5. 通过标准

### 5.1 Dynamic `WORK` 标准

一条 SPIDER dynamic full 序列只有同时满足以下条件，才进入 `rl_from_spider`：

| 指标 | 阈值 |
|---|---:|
| load / rollout | task load 成功，rollout 产出 NPZ 和 MP4 |
| T | `>= 80` frames |
| object tracking | `obj_err_mean <= 0.10m`，`obj_err_max <= 0.30m` |
| pelvis | `pelvis_z_min >= 0.55m` |
| head penetration | `<= 5%` |
| upper-body penetration | `<= 5%` |
| hand floor | left/right 均 `<= 5%` |
| hand-object/contact continuity | either-hand contact/support proxy `>= 30%` 或人工复核确认不是悬空搬运 |
| visual QC | keyframes/视频非空；无明显跪地、趴地、头贴箱或翻箱 |

分级：

- `WORK`: 全部通过，进入 `rl_from_spider`。
- `REVIEW+`: 只差一个轻微指标，例如 pelvis `0.50-0.55m` 且视觉可接受；先不进 RL，除非用户确认。
- `FAIL`: load 失败、严重碰撞、pelvis collapse、object tracking 失控、视频明显不可用。

### 5.2 RL smoke 标准

| 指标 | 阈值 |
|---|---:|
| 训练稳定性 | 无 NaN、无 early crash、有 checkpoint/eval rollout |
| safety | head/upper/hand-floor 任一严重项不超过 `10%` |
| object tracking | `obj_err_mean <= 0.15m` |
| pelvis | `pelvis_z_min >= 0.45m` |
| visual | 不出现全程趴地/翻箱/穿箱 |

### 5.3 RL main 标准

沿用更严格阈值，与 dynamic `WORK` 对齐：

- `obj_err_mean <= 0.10m`
- `pelvis_z_min >= 0.55m`
- head/upper/hand-floor 均 `<= 5%`
- 视频和 keyframes 通过人工复核
- 若比较 paired routes，同一 case 中 `rl_from_spider` 至少需要在 pelvis、安全或 object tracking 上显著优于 `rl_from_omni`，才证明 dynamic 前处理有价值。

## 6. 执行顺序

1. **实现 E092 build/eval/train scaffold**
   - 建 `variants.tsv`；
   - 生成三条 dynamic 派生 task；
   - 生成三条 direct Omni RL task/override；
   - 对所有 scene 调 `snapshot_scenes.sh E092 ...`。

2. **先跑 C1 dynamic smoke**
   - C1 是唯一 D005b pass case，也是 E091 已知 pelvis collapse case；
   - 如果 C1 dynamic smoke 仍严重 pelvis collapse，先读视频和 metrics，再决定是否继续 C2/C3 dynamic full。

3. **并行跑三条 dynamic smoke**
   - 本地跑 C1；
   - 远程两卡跑 C2/C3；
   - 所有 smoke 完成后统一 eval 和可视化。

4. **只对 smoke 合格者跑 dynamic full**
   - 预期最多三条，实际可能只有 C1；
   - full 后按 `WORK/REVIEW+/FAIL` 决策是否进入 Stage B。

5. **跑 direct OmniRetarget RL 三条**
   - 三条都跑 RL smoke；
   - 若 smoke 均可运行，再跑 main；
   - C2/C3 即使 D005b reject 也保留，因为这正是 gate predictive test。

6. **跑 RL from SPIDER dynamic**
   - 仅对 Stage A `WORK` case 跑；
   - 与同 case 的 `rl_from_omni` 做 paired 对比。

7. **汇总决策**
   - 写 `comparison/paired_route_comparison.md`；
   - 写正式 log；
   - 更新 tracker；
   - 若 claims 通过再 commit。

## 7. 资源安排

触发远程并行规则：本轮至少 3 条 independent case，且后续有 RL 对照。

建议分配：

| 阶段 | 本地 | 远程 GPU0 | 远程 GPU1 |
|---|---|---|---|
| dynamic smoke | C1 box004 | C2 Box026 039 | C3 Box026 135 |
| dynamic full | smoke 最优者 | 其余 pass 者串行 | 其余 pass 者串行 |
| RL from Omni smoke/main | C1 | C2 | C3 |
| RL from SPIDER | 只跑 `WORK` case，按数量再分配 | 可复用 | 可复用 |

如果 RL 单次成本高，先执行 smoke + 1 seed main；只有出现路线差异或候选 winner 后再补 seeds。

## 8. 主要风险和应对

| 风险 | 观察信号 | 应对 |
|---|---|---|
| E091 式 pelvis collapse 在三条 dynamic 中复现 | pelvis min `<<0.55m`，视频跪地/趴地 | 不继续 full/RL；下一轮开姿态/upright elite gate 或 pelvis penalty |
| Box026 dynamic load 或 collision 不稳定 | `scene_act` load 失败、upper/hand-floor 高 | 先修 scene/碰撞/mass，不直接调 RL |
| Direct Omni RL 全部失败 | reward NaN、严重穿箱/趴地 | 说明 RL 对 raw Omni reference 不鲁棒；优先 dynamic 或 reference repair |
| Direct Omni RL 反而成功 | D005b reject case 也能 RL pass | 需要回看 D005b 阈值，尤其 support-face `30%` 和 inside `10%` 是否过严 |
| `RL` 入口与当前 SPIDER train harness 不完全一致 | 找不到明确 PPO/RL 脚本或 config | 先把 RL adapter 只做到输入/override/评估规范；等入口确认后接入，不混淆为 MJWP full |

## 9. 预期结论形态

最终不是只报一个成功/失败，而是给出路线决策：

| 结果模式 | 解释 | 下一步 |
|---|---|---|
| C1 dynamic WORK，`rl_from_spider` > `rl_from_omni` | E091 box004 数据有效，但需要动态前处理 | 扩 box004 / medium boxes 的 dynamic-filter pipeline |
| C1 direct Omni RL 已 pass，dynamic 无明显优势 | 上游 OmniRetarget 已够用，E091 smoke pelvis collapse 是训练配置问题 | 直接扩大 OmniRetarget RL，减少 dynamic 前处理 |
| Box026 dynamic/rl 任一路线 pass | Box026 near-reject 可救，D005b threshold/variant 需调整 | 对 Box026 做有限 H2/support-face variant，再扩 top7 |
| 三条 direct 和 dynamic 都失败 | 不是单 case 问题，是姿态/站立动态目标问题 | 停止扩数据，先开 pelvis/upright 约束实验 |
| D005b reject 的 C2/C3 RL 成功 | D005b 作为 hard gate 太保守 | 把 D005b 改成 ranking/filter，而非 hard reject |

## 10. 本轮暂不做

- 不把 `e091_box026_20231018_040_p2` 加入 RL；它没有可用 OmniRetarget 输出。
- 不在本轮扩大 Box026 top7 或 box004 person1。
- 不同时大改 reward；除非三条 case 都在同一类 pelvis collapse 上失败，再单独开姿态约束实验。
- 不把 Box025 的 `--replace_wrist_with_fingertip` reach hack 默认套到这三条 medium-box case。

## 11. 2026-05-29 执行结果

正式结果日志：`workspace/core4d/log/114_E092_three_case_spider_dynamic_and_omniretarget_rl_results.md`

### 11.1 Stage A `spider_dyn smoke`

已按本地 1 卡 + 远程 2 卡完成并回收：

| variant | case | status | pelvis min | object mean | head/upper | hand-floor |
|---|---|---|---:|---:|---:|---:|
| `E092D1_box004_083_p2_dyn` | C1 | FAIL | `0.074m` | `0.006m` | `0/0%` | `0/1.0%` |
| `E092D2_box026_039_p2_dyn` | C2 | FAIL | `0.191m` | `0.008m` | `0/0%` | `0/0%` |
| `E092D3_box026_135_p2_dyn` | C3 | FAIL | `0.186m` | `0.006m` | `0/0%` | `0/39.0%` |

结论：三条均未达到 smoke `PASS/REVIEW+`，因此不跑 Stage A full，也没有 Stage B `rl_from_spider` 输入。

### 11.2 Stage C `rl_from_omni smoke`

已按本地 1 卡 + 远程 2 卡完成并回收：

| variant | case | status | pelvis min | object mean | head/upper | hand-floor |
|---|---|---|---:|---:|---:|---:|
| `E092O1_box004_083_p2_omni` | C1 | FAIL | `0.073m` | `0.006m` | `0/0%` | `0/1.0%` |
| `E092O2_box026_039_p2_omni` | C2 | FAIL | `0.135m` | `0.008m` | `0/0%` | `0/0%` |
| `E092O3_box026_135_p2_omni` | C3 | FAIL | `0.189m` | `0.006m` | `0/0%` | `0/40.2%` |

结论：三条 direct OmniRetarget smoke 均未达到 pelvis `>=0.45m` 标准，因此不跑 Stage C main。

### 11.3 视觉复核与路线决策

- 6 个 smoke mp4、60 张 keyframes 和 6 张 contact sheets 均已生成。
- high subagent `019e6fc7-1fa9-7442-af87-b56c8f376236` 复核认为量化 FAIL 与画面一致：C1/C3 明显 pelvis collapse，C3 右手/右臂贴地，C2 低髋/半跪/压箱。
- 未观察到明确 head/upper-body 穿箱，说明 E083 之后的 upper-body pair/safety 不是本轮主瓶颈。
- spider_dyn 与 rl_from_omni 没有有意义的视觉差异；当前失败集中在 pelvis/upright/contact dynamics，而不是单纯的上游数据筛选是否通过。

最终决策：停止本轮 full/main，下一轮应先开 C1-only pelvis/upright/floor/contact 约束实验，再决定是否回到 Box026 或扩大 medium-box 数据。
