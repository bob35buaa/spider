# Agent 执行计划：E013 → E014 → E014b → E015 → E015b/c → E016

日期：2026-05-19
对应方向评估：`docs/01_direction_review_2026-05-18.md`（v4）
对应指标对比：`docs/02_E011_k100_vs_E081_full_metric_comparison.md`
本文档目的：把 v4 方向评估翻译成 **agent 可顺序执行的步骤清单**，每步包含触发条件、产出物、决策门、git 操作。

## 0. 全局执行规约（每个实验都必须遵守）

### 0.1 实验流程框架（强制遵循 experiment-planning-zh skill）

每个实验严格按 `Plan → Implement → Train → Evaluate → Log → Update Tracker → Git` 流程：

1. **Plan**：写 `workspace/core4d_collab_retarget/plan/NN_{exp}_plan.md`，包含 Context / Claims / 改动 / 成功标准 / 训练命令
2. **Implement**：按计划改代码，每次 Edit 后 `py_compile` 验证；改动尽量隔离在新文件/新 config，不污染共享路径
3. **Train**：本地 `scripts/train/train_{exp}.sh` + 远程 `scripts/run_{exp}_remote.sh` 并行；smoke (T=4) → full 两阶段
4. **Evaluate**：先 numeric eval (`scripts/eval/eval_{exp}.py`)，再视觉评估（详见 §0.4 必须用 subagent）
5. **Log**：写 `log/NN_{exp}_results.md`，包含 aggregate / 关键指标表 / Claims 验证 / 决策
6. **Update Tracker**：在 `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md` 加行 + 更新关键指标演进表
7. **Git**：每个实验完成（含 setup commit + results commit）必须 `commit + push`

**执行前必读**（每次开新 session 或恢复时）：

```bash
EXP_WS="workspace/core4d_collab_retarget"
cat "$EXP_WS/EXPERIMENT_TRACKER.md"
ls -1 "$EXP_WS/plan/"*.md | sort | tail -1 | xargs cat
ls -1 "$EXP_WS/log/"*.md | sort | tail -1 | xargs cat
cat "$EXP_WS/progress.md" 2>/dev/null
cat "$EXP_WS/docs/01_direction_review_2026-05-18.md"
```

### 0.2 GPU 并行策略（本地 1 卡 + 远程 2 卡）

- **本地**：`CUDA_VISIBLE_DEVICES=0`（RTX 5090），跑 1 个 main variant 串行
- **远程 spider-remote**：2× RTX 6000 Ada，tmux 内 GPU0/GPU1 各串行一个队列
- **远程标准流程**：参考 `.codex/skills/experiment-planning-zh/remote-execution.md`，步骤为：
  1. 本地 `git add + commit + push`
  2. `ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull"`
  3. `ssh spider-remote "tmux new-session -d -s E0NN; tmux send-keys -t E0NN 'bash workspace/core4d_collab_retarget/scripts/run_E0NN_remote.sh' Enter"`
  4. 监控：`ssh spider-remote "tmux capture-pane -t E0NN -p | tail -10"` + `ls workspace/.../results/E0NN/*.npz | wc -l`
  5. 回收：`bash workspace/core4d_collab_retarget/scripts/pull_E0NN_remote_results.sh`
- **分配原则**：main variants 优先本地（更快迭代）+ 远程 GPU0；guard variants 放远程 GPU1
- **预授权（如果没有的话）**：所有 `train_*.sh` / `run_*_remote.sh` / `pull_*_remote_results.sh` 在首次运行前必须用 `__codex_auth_probe__` 参数走预授权

### 0.3 子代理（subagent）使用规范

**何时必须用 subagent**：

1. **大规模代码搜索 / 跨文件审计**（>3 次 grep）→ 用 `Explore` subagent
2. **可视化生成与解读**（视频抽帧、关键帧拼图、指标曲线、雷达图等）→ **必须** spawn subagent（高模型），主 agent 不应直接处理大量帧/图片浪费 context
3. **设计实现层方案**（XML 改动 + config 字段 + sim 改动多文件协同）→ 先出 architectural plan
4. **大规模文档 / 论文阅读**（>1000 行）
5. **不确定的代码定位查询**（"X 函数在哪定义"）→ 用 `Explore` subagent

**模型选择**：复杂分析 / 设计 / 可视化解读 → `subagent_type` 高模型（默认 high）；简单查找 → medium

**禁止**：visualization 类任务（视频抽帧后看图、生成关键帧拼图、做雷达图）由主 agent 直接处理 — 必定爆 context

### 0.4 评估与可视化要求（强制）

每个 full 实验完成后，**两类评估都要做**，缺一不可：

#### A. Numeric Evaluation（主 agent 直接做）

- 用 `scripts/eval/eval_E0NN.py --all` 生成 `aggregate_summary.json` + `comparison.csv`
- 必须对齐 E081 口径报告：obj mean/max、hand contact %、leg interference %、floor contact %、pelvis min/err、bottom proxy、xy ratio、rotation deg、partner force mean/max
- 必须报告：相对 E011 k100 的 delta、相对 §Step 1 软 target（E081 + oracle 容忍带）的 pass/fail、"推 vs 搬" 门（leg-obj + floor 联合）的 pass/fail
- 必须做 Claims 表逐条验证

#### B. Visual Evaluation（**spawn subagent**）

**subagent task 描述模板**：

```
任务：评估 E0NN 实验结果的视觉效果，对比 baseline E081 / E011 k100。

输入文件：
- MP4 路径：workspace/core4d_collab_retarget/results/E0NN/E0NN_*.mp4
- baseline：workspace/core4d/results/E081/E081_box025_p2_legobj.mp4
- best E011：workspace/core4d_collab_retarget/results/E011/E011_box025_p2_com_xyz_k100.mp4

要做的事：
1. 用 video-frames skill 从每个 main MP4 抽 f32/f100/f130/f160/f204 五帧
   （对应 case-window 起始/中段/搬运/放置/结束）
2. 拼成 horizontal montage（每个 variant 一行，5 帧一列）输出到
   workspace/core4d_collab_retarget/results/E0NN/keyframes/E0NN_visual_montage.jpg
3. 对每个 main variant 写视觉评估（200 字内）：
   - 物体是否真实平移（不是原地翻转/贴地拖）
   - 机器人手是否持续接触（不是脱开/穿透）
   - 是否出现 holosoma 的"用腿推/趴箱推"失败模式
   - 与 E081/E011 k100 baseline 的肉眼差异
4. 输出汇总报告到 workspace/core4d_collab_retarget/results/E0NN/visual_eval.md
   不要回传图片内容（已存盘），只回传文字结论。

约束：subagent 内部直接读图，不要把 base64 / 大段帧数据回传给主 agent。
```

### 0.5 Git 提交规约（每个实验至少 2 commits）

每个实验完成必须产生至少 2 个 commit：

1. **Setup commit**（实验启动前）：
   - 包含 plan/、scripts/、scene XML 改动、config 改动、sim 代码改动
   - commit message: `exp(core4d_collab_retarget): set up E0NN <topic>`
2. **Results commit**（实验完成后）：
   - 包含 log/、results/{aggregate, comparison, keyframes 摘要}、tracker 更新、progress 更新、visual_eval.md
   - **不提交 NPZ / MP4**（gitignore），但提交 scene_snapshot/ 和 manifest.txt（参考 CLAUDE.md §7 双安全网）
   - commit message: `docs(core4d_collab_retarget): record E0NN <result-keyword>`
3. **每次 commit 必须紧跟 `git push origin exp/core4d-collab-retarget`**
4. 触发 sweep（E014b / E015b / E015c）的额外实验各自独立 setup + results commits

### 0.6 数据完整性双安全网

每个跑物理仿真的实验：

- **Safeguard 1**：active case 的 scene XML 必须 `git add -f`（active case 列表见每个实验 plan）
- **Safeguard 2**：训练脚本开头调用 `workspace/core4d_collab_retarget/scripts/convert/snapshot_scenes.sh E0NN <case1> [case2 ...]`，把 XML 快照到 `results/E0NN/scene_snapshot/` 并附 manifest.txt（git HEAD + sha256）

## 1. 决策树总览（执行顺序）

```
START
  │
  ├─[当前正在跑]── E012 dual-point spring sweep
  │     (按已有 plan 跑完，结论低预期，但作为完整诊断)
  │
  ├──────────────► E013 oracle ceiling (0.5 day, 必做)
  │                   │
  │                   └─► Finalize 软 target 容忍带
  │                       (写到 workspace/core4d_collab_retarget/docs/04_soft_target_definition.md)
  │
  ├──────────────► E014 COLA-B (kinematic support + weld equality)
  │                   │
  │      ┌────────────┴────────────┐
  │      │                         │
  │   过软门                改善但未过门
  │      │                         │
  │      ▼                         ▼
  │   RL pipeline 对接          E014b stiffness sweep
  │      (出工作区，本计划完结)     │
  │                       ┌─────────┴─────────┐
  │                    过软门              未过门
  │                       │                   │
  │                       ▼                   ▼
  │                    RL pipeline          E015
  │
  ├──────────────► E015 COLA B+A (dynamic support + 6-DoF joint + PD)
  │                   │
  │      ┌────────────┴────────────┐
  │      │                         │
  │   过软门                改善但未过门
  │      │                         │
  │      ▼                         ▼
  │   RL pipeline              E015b 参数 sweep
  │                       ┌─────────┴─────────┐
  │                    过软门         未过门但仍改进
  │                       │                   │
  │                       ▼                   ▼
  │                    RL pipeline         E015c (joint limits/friction 等更广)
  │                                            │
  │                                  全饱和后  │
  │                                            ▼
  └──────────────► E016 kinematic + contact (holosoma v4.x 风格回退/正交验证)
                       │
                       └─► 最终结论 + RL pipeline 对接或重审 baseline
```

## 2. 各实验 agent 执行清单

### 2.1 E012 收尾（按已有 plan 执行，不重写）

**状态**：plan 已存在 `plan/12_E012_dual_point_partner_pose_closure_plan.md`，按其执行。

**Agent 步骤**：

1. 跑完 8 个 variants（本地 + 远程并行，参考 plan §Parallel Execution）
2. **不要对 obj_mean 抱期待**（v4 文档已诊断 spring 范式 xy lag 结构性）
3. 完成后按 §0.4 做 numeric + visual eval（visual 必须 subagent）
4. 写 `log/12_E012_dual_point_results.md`，更新 tracker
5. Git: setup commit 已有，只需 results commit + push
6. **无论 E012 结果如何，下一步都是 E013**，不要因 E012 失败放弃方向

### 2.2 E013 oracle ceiling 诊断（0.5 day）

**目的**：建立"真 freejoint + 无穷强外力" 下的指标上限，作为后续软 target 容忍带的依据。

**Agent 步骤**：

#### 步骤 A：写 plan

- 写 `plan/13_E013_freejoint_object_pd_oracle_plan.md`
- Claims：
  - C1 启用 `object_pd_override=true` 后 object 完全跟随 ref（obj_mean < 0.05m）
  - C2 robot 在 ref 物体强制跟踪下达到的 hand contact / pelvis err / leg / floor 上界
  - C3 该上界与 E081 数字 gate 的差距 → 决定 §Step 1 软 target 是按 E081 还是按 oracle
- 改动：纯 config override，不改 sim 代码
- 成功标准：obj_mean < 0.05m，2 个 case 都跑出有效 NPZ

#### 步骤 B：实现 & smoke

- 在 `workspace/core4d_collab_retarget/scripts/E013/` 创建：
  - `variants.tsv`（2 个：box025_p2_oracle、box023_p2_oracle）
  - `generate_e013_overrides.py`：基于 `core4d_collab_E002` override 加 `object_pd_override=true`，**保持 nq_obj=7、freejoint scene、contact_guidance=false**
  - `run_E013_preprocess.sh`
  - `train/train_E013.sh`（无远程，本地 30 分钟跑完）
  - `eval/eval_E013.py`（复用 E011 eval 字段 + 输出 oracle delta 表）
- `py_compile` 通过
- 本地 GPU0 跑 smoke（T=4），确认 wiring

#### 步骤 C：Full + commit

- 本地 GPU0 串行跑 2 个 full（约 1h）
- numeric eval + visual eval（subagent 抽 5 帧 + 写 visual_eval.md）
- 写 log/13_E013_oracle_results.md
- 写 `docs/04_soft_target_definition.md`（基于 oracle 数据 finalize 容忍带）
- Setup commit + Results commit + push
- 更新 tracker

#### 决策

- 进入 E014，无分支

### 2.3 E014 COLA-B 主轮（核心实验，1-3 day）

**目的**：把 partner-object 连接从 spring force 改成 **6-DoF joint 约束**（用 MuJoCo weld equality + soft solref/solimp 等效实现），测试位置约束范式能否消除 xy lag。

**Agent 步骤**：

#### 步骤 A：架构设计（spawn Plan subagent，高模型）

- subagent task：阅读 `spider/simulators/mjwp.py`、`spider/config.py`、`example_datasets/.../box025_person2_freejoint_legobj/scene.xml`、`spider/simulators/mjwp_eq.py`（如存在 weld 参考），输出实现方案：
  - 怎么在 scene.xml 加 partner_support kinematic mocap body（pose 来自 person2 hand wrist mocap）
  - weld equality 的 solref/solimp 怎么暴露到 config
  - mocap pose 在 sim 内每步怎么更新（参考 E010 mocap pad 更新路径）
  - 关键不变量：`nq=43, nv=41, nu=29, nq_obj=7` 末尾保持
- subagent 输出 architectural plan 到 `workspace/core4d_collab_retarget/plan/14_E014_architecture.md`

#### 步骤 B：写实验 plan

- 写 `plan/14_E014_COLA_B_weld_equality_plan.md`
- Claims：
  - C1 6-DoF joint 约束消除 spring lag（obj_mean 显著低于 E011 k100 的 0.34m）
  - C2 位置约束传力不引发 robot 手脱开（hand contact 在容忍带内）
  - C3 不引发 holosoma 跨版本 "推 vs 搬" 失败（leg-obj ≤ 15%，floor ≤ 70%）
  - C4 不引发 partner 硬塞物体（反力 mean ≤ 200N，max ≤ 800N）
  - C5 nq/nv/nu/nq_obj 不变（freejoint parity ok）
- Variant 表（6 个，见 v4 doc §五 Step 2.1）
- 成功标准：§0.4 + §1 软 target 工作定义

#### 步骤 C：实现

- 改 `spider/config.py`：加 `partner_support_enabled` / `partner_support_weld_solref` / `partner_support_weld_solimp` / `partner_support_mocap_body_name` / `partner_support_pose_source`（取值如 `ref_partner_hand`）
- 改 `spider/simulators/mjwp.py`：
  - setup 时识别 partner_support body 并预计算 mocap pose 序列
  - step 时更新 mocap pose（方案 B：纯 ref-driven，不 relocalize）
  - 保持 `xfrc_applied` 路径与 partner_force 兼容（互不覆盖）
- 写新 scene XML 模板 `scene_partner_support.xml`（基于 `scene.xml` 加 partner_support body + weld）
- `generate_e014_overrides.py` + train/remote/pull/eval 脚本
- `py_compile` 全通过

#### 步骤 D：Smoke

- 本地 GPU0 跑 1 个 variant 4-step smoke
- 检查 NPZ 含 `freejoint parity ok` + `partner_support_pose` + 等效 wrench 诊断字段
- 任何字段缺失立即修，不进 full

#### 步骤 E：Full（本地 GPU0 + 远程 GPU0/1 并行）

- 提交 setup commit + push
- 启动远程 tmux session `E014`（GPU0 跑 main 3 个、GPU1 跑 guard 2 个）
- 本地 GPU0 跑 main 1 个
- 监控 GPU 利用率 + tmux 输出；卡住超 3 分钟无进展立即终止该 variant 并记录

#### 步骤 F：Eval + 决策

- §0.4 A numeric eval + B subagent visual eval
- 写 `log/14_E014_COLA_B_weld_results.md`，包含决策表：

| 判断 | 数据 | 行动 |
|------|------|------|
| 任一 main 过软 target | obj_mean ≤ §Step 1 阈值 + hand ≥ 容忍带 + 推 vs 搬 pass | **锁定该配置，跳过 E014b/E015，进入 RL pipeline 对接（出本计划）** |
| 改善明显未过门 | obj_mean ∈ [0.18, 0.27]m + lag-free 满足 + 推 vs 搬 pass | **进 E014b stiffness sweep** |
| 改善有限或 solver 不稳 | obj_mean ≥ 0.28m 或 solver 发散 | **跳过 E014b 直接进 E015**，看 dynamic body+PD 能否补 |

- Results commit + push + tracker 更新

### 2.4 E014b stiffness sweep（条件触发，1-2 day）

**触发**：E014 决策表第 2 行

**Agent 步骤**：

1. 写 `plan/15_E014b_stiffness_sweep_plan.md`
2. Variants（5 个，见 v4 doc §五 Step 2.1.5 表）
3. 不改 sim 代码，只新 generate_e014b_overrides.py 扫 solref/solimp/gravity/hc
4. 本地 + 远程并行；尽量复用 E014 train/remote scripts，加 `_e014b` 后缀
5. Eval + visual subagent
6. 写 `log/15_E014b_stiffness_sweep_results.md`
7. 决策表：

| 判断 | 行动 |
|------|------|
| 任一 E014b 过软 target | 锁定 → RL pipeline 对接 |
| best-E014b 比 best-E014 又改善 ≥ 0.03m 但未过门 | **进 E015**（趋势确认 B 还在改进） |
| E014b 相对 E014 提升 < 0.02m | **B 范式饱和，进 E015 看 A 能否补** |

8. Setup + Results commit + push

### 2.5 E015 COLA B+A 主轮（核心实验，2-4 day）

**目的**：把 E014 的 kinematic support body 升级为 **dynamic body + PD command**，对齐完整 COLA。

**Agent 步骤**：

#### 步骤 A：架构设计（spawn Plan subagent，高模型）

- subagent task：基于 E014 实现，设计如何把 partner_support 从 kinematic 改成 dynamic 并保留 nq_obj=7：
  - 把 partner_support 挂在 **object 的 kinematic tree 下**（不是世界）
  - 用 6-DoF joint（slide×3 + hinge×3）连接，每个 DOF 有 stiffness/damping/limit
  - PD target 来自 person2 hand mocap pose 每步推出
  - 力上限 clamp（≤ 150N / 30Nm）
  - **关键检查**：object freejoint 仍是 nq 末尾 7 维（avoid progress.md 17:05 警告的工程坑）
- subagent 输出 architectural plan 到 `plan/16_E015_architecture.md`，包含 scene XML 示意 + nq 布局示意 + PD 控制接入点

#### 步骤 B：写实验 plan

- 写 `plan/16_E015_COLA_BA_dynamic_support_plan.md`
- Claims：
  - C1 dynamic support body 不破坏 nq_obj=7 末尾假设
  - C2 PD command 追上 person2 hand mocap pose（partner-side tracking err 在容忍带内）
  - C3 partner-object interaction force 在合理范围（mean ≤ 100N，max ≤ 250N）
  - C4 obj_mean 不差于 E014 best；理想情况下 ≤ E014 best - 0.03m
  - C5 不出现 holosoma R037 训崩（NPZ 完整 124 帧 main / 136 帧 guard）

- Variants（4-5 个，见 v4 doc §五 Step 2.2 表）

#### 步骤 C：实现

- 改 scene XML 模板 `scene_partner_support_dynamic.xml`：partner_support 挂在 object body 下，6-DoF joint 各 DOF 加 stiffness/damping/limit/armature
- 改 `spider/config.py`：加 `partner_support_mode=dynamic_pd` 相关字段（mass / kp_pos / kd_pos / kp_rot / kd_rot / force_clamp / torque_clamp）
- 改 `spider/simulators/mjwp.py`：
  - dynamic body 路径下 PD target 每步从 ref hand pose 推出
  - PD 力施加到 partner_support 上（用 `xfrc_applied` 或者直接 joint actuator）
  - **不要**把 partner_support 加到 robot ctrl（nu=29 不变）
- `py_compile` 全通过

#### 步骤 D：Smoke

- 本地 1 variant 4-step smoke
- **关键 sanity check**：
  - `model.nq == 43 + N_dyn_partner_dof`（如果 partner 是 6-DOF + 挂 object 下，nq 应该 = 43 + 6 = 49，但 nq_obj 末 7 维仍是 object freejoint）
  - 在 mjwp 内 assert `qpos[:, -7:]` 仍指向 object pos+quat
  - partner_support 与 object 之间 6-DoF joint 在 qpos 中的位置 / `nq_obj` 定义需要更新到不破坏 object PD/eval 的口径
- 任何破坏立即修

#### 步骤 E：Full（本地 + 远程并行）

- Setup commit + push
- 启动远程 tmux session `E015`（GPU0 main 3、GPU1 guard 1-2）
- 本地 GPU0 main 1
- **训练崩溃监测**：ep_len 突然 < 50 步或 NPZ 写出 < 124 帧 → 立即停，**不算 E016 触发条件**，回去排查 PD 参数（kp_pos/kd_pos/mass 调整）

#### 步骤 F：Eval + 决策

- Numeric + visual subagent
- 写 `log/16_E015_COLA_BA_results.md`
- 决策表：

| 判断 | 行动 |
|------|------|
| 任一 main 过软 target | 锁定 → RL pipeline 对接 |
| 改善明显未过门（obj_mean ∈ [0.15, 0.25]m） | **进 E015b 参数 sweep** |
| 未过门也未比 E014 改善 | A 是冗余开销，**回退 E014 best 作为本阶段 reference**，启动 E016 正交验证 |
| 训练崩溃 | 排查参数后回到 E015 基线再判断；**不是 E016 触发** |

- Results commit + push + tracker

### 2.6 E015b 参数 sweep（条件触发，1-2 day）

**触发**：E015 决策表第 2 行

**Agent 步骤**：

1. 写 `plan/17_E015b_param_sweep_plan.md`
2. Sweep 策略（v4 doc §五 Step 2.2.5）：先 mass × PD kp 共 6 runs，仍未过门再扫 joint coupling 参数 4 runs，总 ≤ 10 runs / 1-2 day
3. 不改 sim 代码（除非缺暴露的参数），只 generate overrides
4. 本地 + 远程并行
5. Eval + visual subagent
6. 写 `log/17_E015b_param_sweep_results.md`
7. 决策表：

| 判断 | 行动 |
|------|------|
| 任一 E015b 过软 target | 锁定 → RL pipeline 对接 |
| best-E015b 比 best-E015 又改善 ≥ 0.03m 未过门 | **E015c 第三轮**（joint limits / friction / mass 更广范围，再消耗 ≤ 1 week） |
| E015b 提升 < 0.02m | **COLA 范式饱和**，锁定 best 作为 reference，进入 E016 正交验证 |

8. Setup + Results commit + push

### 2.7 E015c COLA 第三轮（条件触发，可选，1 week）

**触发**：E015b 决策表第 2 行

**简化原则**：避免无限扫参；E015c 失败后**必须**进入 E016 决策，不再有 E015d。

**Agent 步骤**：与 E015b 同结构，扫 joint limits（无 / 软限 / 软限+摩擦）+ 更广 mass 范围。3-6 runs。写 `plan/18_E015c_*` + `log/18_E015c_*`。

### 2.8 E016 kinematic + contact（回退 / 正交验证，2-3 day）

**触发条件**（v4 doc §五 Step 3 收紧）：

- A. **完全失败**：E014 + E014b + E015 + E015b + E015c 全部用完未过软 target，且 best < 0.03m 接近 target → COLA 范式被证伪
- B. **正交验证**：E015/E015b 锁定一个 work 但偏弱的 reference，验证 kinematic+contact 能否进一步改善

**不允许的触发**：E014 一轮失败、E015 训崩、各 sweep 没跑完。

**Agent 步骤**：

#### 步骤 A：架构设计（spawn Plan subagent）

- subagent 阅读 `holosoma/workspace/v2/log/10_v3.3_v4.x_results.md` § R023-fix2 实现细节、E010 scene_contact_pad XML / mocap pad 更新路径，输出实现方案：
  - 双 capsule mocap body（radius 7cm, length 15cm），pose 来自 person2 双手 wrist mocap
  - palm offset 12cm 沿 wrist→sim_object_center 方向
  - relocalization 方案 B：纯 ref-driven 不跟 robot/sim drift
  - hand-object contact pair 启用
- 输出到 `plan/19_E016_architecture.md`

#### 步骤 B：写实验 plan + 实现

- 写 `plan/19_E016_kinematic_contact_partner_hands_plan.md`
- Claims：
  - C1 双 capsule + palm offset 实测 penetration ≥ 4cm
  - C2 物体被双手夹持，obj_mean 改善
  - C3 hand contact 不脱开
  - C4 不出现 holosoma 推 vs 搬失败
- 实现：
  - 改 `spider/config.py`：加 `partner_kinematic_hands_enabled` / `partner_capsule_radius` / `partner_palm_offset_m`
  - 改 `spider/simulators/mjwp.py`：mocap pose 更新路径（参考 E010）
  - 写 `scene_partner_hands.xml`（基于 scene.xml 加 2 个 capsule mocap body + contact pair）
  - generate / train / remote / pull / eval 脚本

#### 步骤 C：Smoke + Full

- 标准流程
- Variants：5-7 个（radius 6/7/8cm × palm offset 10/12/15cm × ±robot HC × main+guard）

#### 步骤 D：Eval + 最终结论

- Numeric + visual subagent
- 写 `log/19_E016_kinematic_contact_results.md`
- **如果 E016 也未过软 target**：写最终结论文档 `docs/05_final_conclusion_2026-05-XX.md`，建议：
  - 切换 baseline 口径（接受 SPIDER freejoint 不可能达到 E081 actuator 数字）
  - 或回到下游 RL 框架（holosoma）评估能否直接吸收 SPIDER best 作为 reference
  - 或推迟到 SBTO / 双机器人 / 重写 sampler（长期工作）

## 3. 异常与中断处理

### 3.1 GPU 卡死 / SSH 断开 / NPZ 写不完整

参考 progress.md 已有处理模式：

- GPU 利用率 0% + log 超 3 分钟无更新 → 终止该 variant 进程，**保留 log**，单独本地补跑
- 远程 tmux pane 异常 → `tmux capture-pane -p` 查全文，必要时 `tmux kill-session` 重启
- NPZ 帧数不符（main 应 = 124，guard 应 = 136）→ 排除，**不要作为结果**，记录到 progress.md "遇到的错误" 表

### 3.2 决策门临界

如果 obj_mean 卡在容忍带边界（±0.02m 内），**不要自己决定**，写到 results log 里向 user 报告，等 user 确认是否锁定 / 升级。

### 3.3 Subagent 失败

如果 visualization subagent 没产出 visual_eval.md 或图片，**不要主 agent 接手** — 重新 spawn，必要时拆成更小任务（一次只评一个 variant）。

### 3.4 Git 提交冲突

远程可能有 dirty state（参考 progress.md E003 经验）：

- 不要在远程 `git stash`（容易丢 ignored 文件）
- `ssh spider-remote "cd ...; git status"` 先看
- 必要时手动通知 user 处理远程未提交内容，不擅自删除

## 4. 总预算与产出

| 阶段 | 最快 | 最慢 |
|------|------|------|
| E012 收尾 | 0.5 day | 1 day |
| E013 oracle | 0.5 day | 0.5 day |
| E014 主轮 | 1 day | 3 day |
| E014b sweep | 1 day | 2 day |
| E015 主轮 | 2 day | 4 day |
| E015b sweep | 1 day | 2 day |
| E015c（可选） | — | 7 day |
| E016 | 2 day | 3 day |
| **合计** | **7 day** (主轮过门) | **22 day** (全跑) |

预期典型路径：**E012 + E013 + E014 + E014b + E015 = 6-12 day**；如 E015 过门进入 RL pipeline 对接。

**最终产出**（无论结果好坏）：

1. 完整 E012-E016 log + tracker 更新
2. `docs/04_soft_target_definition.md`（E013 后产出）
3. `docs/05_final_conclusion_2026-05-XX.md`（E016 后产出，含 best reference 配置 + 已知限制 + RL pipeline 对接建议或重审 baseline 的建议）
4. 所有 commit 已 push 到 `exp/core4d-collab-retarget`

## 5. Quick Reference：常用命令

```bash
# 工作区根
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
export EXPERIMENT_WORKSPACE=workspace/core4d_collab_retarget

# 当前 git 状态
git status
git log --oneline -5

# 远程同步
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && git pull && nvidia-smi"

# 启动远程 tmux
ssh spider-remote "cd /home/xiayb/pHRI_workspace/spider && tmux new-session -d -s E0NN && tmux send-keys -t E0NN 'bash workspace/core4d_collab_retarget/scripts/run_E0NN_remote.sh' Enter"

# 监控
ssh spider-remote "tmux capture-pane -t E0NN -p | tail -20"
ssh spider-remote "ls workspace/core4d_collab_retarget/results/E0NN/*.npz | wc -l"

# 回收
bash workspace/core4d_collab_retarget/scripts/pull_E0NN_remote_results.sh

# Eval (numeric)
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E0NN.py --all

# Snapshot scene XMLs
bash workspace/core4d_collab_retarget/scripts/convert/snapshot_scenes.sh E0NN box025_person2_freejoint_legobj box023_person2_freejoint_legobj

# Git commit pattern
git add workspace/core4d_collab_retarget/{plan,log,results}/... spider/{config.py,simulators/mjwp.py}
git commit -m "exp(core4d_collab_retarget): set up E0NN <topic>

git push origin exp/core4d-collab-retarget
```
