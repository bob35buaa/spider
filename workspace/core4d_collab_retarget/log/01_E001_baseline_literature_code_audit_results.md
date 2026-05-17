# E001 Results: baseline/literature/code audit for collaborative retargeting

日期：2026-05-17

对应计划：`workspace/core4d_collab_retarget/plan/01_E001_baseline_literature_code_audit_plan.md`

## 结论摘要

E001 完成了三件事：

1. 明确 git / 工作区基线：`feat/dual-robot-retarget` 已 fast-forward 到远端 `main`，新方向分支为 `exp/core4d-collab-retarget`；本工作区编号从 E001 开始，`workspace/core4d` 的 E081 只作为 baseline。
2. 确认当前 E081 pipeline 的 object 控制口径：E081 不是纯 freejoint 物体。它继承 `core4d_e074a_box023 -> core4d_e073 -> core4d_e071w02 -> core4d_e062 -> core4d_e041c`，核心使用 `scene_act` + `contact_guidance`。物体从 freejoint ref 转成 3 slide + 3 hinge，并有 6 个 object actuator；object actuator target 来自 GT/ref 物体轨迹，commit 前还会恢复 actuator gain。
3. 论文综合给出三条可执行路线：SPIDER 指向更可靠的 virtual contact guidance，DynaRetarget 指向长 horizon segment refinement，Harmanoid 指向 partner proxy / interaction-aware 建模。

最重要的回答：如果把物体改成真正 freejoint、只靠机器人接触力/重力/外力移动，当前 E081 结果不能直接视为会继续 work。历史 E029 明确记录 `scene.xml` freejoint 模式下 CEM 不采样物体，物体只受物理力；E081 当前成绩有 object actuator guidance 的成分。下一个实验应先做 E002 freejoint 对照。

## 结果路径

| 类型 | 路径 |
|------|------|
| 论文综合 | `workspace/core4d_collab_retarget/paper_notes/01_E001_literature_synthesis.md` |
| E081 抽帧 | `workspace/core4d_collab_retarget/results/E001_baseline_frames/` |
| E081 baseline 表 | `workspace/core4d/results/E081/comparison.csv` |
| E081 aggregate | `workspace/core4d/results/E081/aggregate_summary.json` |

## 1. Object / Freejoint 控制口径

### 1.1 模式 A: `scene.xml` 真 freejoint

`scene.xml` 中 object 是自由关节：

- `example_datasets/.../box025_person2/scene.xml` 里 object body 使用 `<freejoint name="object_joint" />`，没有 object actuator。
- 同一文件 actuator 列表只包含 G1 关节 position actuator。
- contact pair 只有 `left_hand_object`、`right_hand_object`、`object_floor`，E080 原 scene 没有腿/脚-object pair。

代码侧：

- `spider/config.py` 在 `contact_guidance=false` 时设置 `nq_obj=7`，并选择 `scene.xml`。
- 历史 log 29 的结论是：`scene.xml` freejoint 模式下 `nu=29`，CEM 只采样 robot control；物体只受重力、碰撞、`xfrc_applied` 等物理作用。

判断：这是用户问题里“如果是 free joint、全靠机器人的力”的模式。它最物理真实，但历史上 E028/E029 已显示单靠 xfrc/接触难以稳定控制 orientation 和 lift。

### 1.2 模式 B: `scene_act.xml` contact guidance / actuator-guided object

E081 实际走的是这个模式：

- `core4d_e041c.yaml` 设置 `scene_name: scene_act`、`contact_guidance: true`、`init_pos_actuator_gain=500`、`init_rot_actuator_gain=50`、`object_pd_override=false`、`partner_force_scale=0`。
- `spider/config.py` 在 `contact_guidance=true` 时设置 `nq_obj=6`，选择 `scene_act.xml`，并解析 object actuator ids。
- `examples/run_mjwp.py` 始终加载 freejoint `trajectory_kinematic.npz`，然后在 contact guidance / object PD 模式下把 freejoint object `pos+quat` 转为 scene_act 的 `slide_pos+euler`。
- 同一转换还把 `ctrl_ref` 末 6 维补成 object slide/euler target。
- CEM 优化前会把 object actuator ctrl 重置为 ref ctrl；contact point delta 只会在 object position ctrl ids 上做很小的 clipped 修正。
- optimize 最后一轮可能把 object actuator gains 衰减到 0，但 commit 前代码会恢复 initial gains，使 PD actuator 在真实提交步继续驱动物体。
- `spider/config.py` 在 `contact_guidance && object_actuator_ids` 时会把 object actuator 维度的采样噪声置 0；因此虽然通用 sampler 形式上处理 `nu` 维控制，active E079-E081 中 object channels 不被有意义地随机采样。

因此：E081 的 object state 不是纯 GT kinematic override，但也不是纯 freejoint 物理搬运。更准确地说，它是 “scene_act 6DOF actuator + ref object target + contact guidance/PD gain” 的物理仿真；CEM 基本是在 actuator-guided object trajectory 下优化 robot controls。机器人、手接触、腿碰撞仍会影响状态，但 object actuator guidance 给了强先验。

### 1.3 模式 C: object PD override / partner force / kinematic override

历史代码还保留三类实验性 object 控制：

- `object_pd_override=true`：每步覆盖 object actuator ctrl 为 ref pos/euler，明确是 GT-like PD。
- `partner_force_scale/spring_kp`：通过 `xfrc_applied` 给 freejoint object 上力/弹簧，模拟 human partner support。
- `partner_force_spring_kp < 0`：物体 qpos kinematic override，是最不物理的 GT 物体轨迹。

这些模式不是 E081 主线，但为后续 partner proxy 实验提供入口。

## 2. E081 Baseline 验收口径

### 2.1 必须保留的 baseline cases

| Case | 角色 | E081 baseline |
|------|------|---------------|
| `E081_box025_p2_legobj` | main | case-window True，但 `E081_success_numeric=False`、`strict_proxy=False`；obj mean/max `0.143/0.271m`，hand contact `89.0%`，leg interference `7.5%`，floor contact `59.5%`，bottom mean `-0.075m` |
| `E081_box023_p2_legobj` | guard | case-window/numeric/strict proxy 都 True；obj mean/max `0.164/0.317m`，hand contact `66.7%`，leg interference `2.7%`，floor contact `34.7%`，bottom mean `0.144m` |

### 2.2 E002+ 必须报告的指标

- `E081_success_case_window`
- `E081_success_numeric`
- `E081_success_legobj_strict_proxy`
- case-window obj mean/max
- hand contact %
- pelvis min z
- leg SDF min/mean
- leg interference %
- leg near-2cm %
- leg-object contact %
- object-floor contact %
- object bottom mean
- object bottom gap vs ref

严格改善目标：

- box025 main 至少保持 case-window True。
- 若声称改善 lift/floor-contact，应让 floor contact 低于 E081 `59.5%`，bottom mean 高于 E081 `-0.075m`。
- 若声称 strict improvement，应让 strict proxy True：leg interference `<=5%`，且 bottom mean 不比 ref 低超过 `5cm`。
- box023 guard 必须保持三项 success flag 为 True，视觉不能崩坏。

指标代码位置：

- leg/foot geom 列表：`workspace/core4d/scripts/eval/eval_E081.py:L29`
- adjusted SDF：`eval_E081.py:L85`
- contact/floor/bottom row：`eval_E081.py:L160`
- summary metrics：`eval_E081.py:L231`
- strict proxy：`eval_E081.py:L285`
- post2 numeric：`eval_E081.py:L336`

### 2.3 可视化观察

E001 用 `video-frames` 从 E081 MP4 抽取 f100/f125/f160：

- `box025_p2_legobj`：f100/f125/f160 都显示 sim 手在箱侧，腿/脚相比 E080 没有明显深穿箱；但箱底仍接近地面，整体更像扶/推/partial carry，不是 strict lift。
- `box023_p2_legobj`：f100/f125 仍是可信抱箱/搬箱姿态，f160 进入弯腰放置阶段；有少量脚/箱接近，但不像主要支撑来源。

后续 E002+ 不允许只用数字判断成功；必须同步看同类关键帧或 MP4。

## 3. 论文综合后的实验映射

详见 `paper_notes/01_E001_literature_synthesis.md`。核心映射如下：

| 来源 | 对当前项目的可执行启发 | 风险 |
|------|------------------------|------|
| SPIDER | contact guidance 应维护 hand-object 相对接触 frame，并过滤短接触/漂移接触；不是单纯 distance reward | 当前 mask 来自 3cm heuristic，质量不足会放大错误接触 |
| DynaRetarget | E075-E081 的 hold/lift/place 失败是长 horizon 耦合；可先做短 segment micro-SBTO | E047 旧 SBTO port 失败过，必须从短 segment 和 E081 指标开始 |
| Harmanoid | CORE4D 大箱是双人协作，单人化会丢 partner dynamics；应引入 partner proxy 或 interaction-aware observation/reward | 双机器人 connect 历史上容易造假，面向 sim2real 应模拟真实人类另一端而不是训练双机器人策略 |

## 4. E002+ 候选路线图

| 优先级 | 实验 | 目标 | 成功标准 |
|--------|------|------|----------|
| P0 | E002 freejoint E081 control audit | 用 E081 的 `box025_p2/box023_p2` 复刻一组真 freejoint scene，不使用 object actuator guidance，验证 E081 是否依赖 GT-like object target | 给出同口径 E081 eval；若 box025/box023 均显著退化，确认 object actuator 是关键依赖；若 guard 仍 work，说明真实接触有希望 |
| P1 | E003 stable contact segment filter | 对 3cm mask 加稳定接触段/target drift 过滤，避免短接触和漂移 contact guidance | 不降低 box023 guard，box025 hand contact/leg interference 不退化 |
| P1 | E004 partner proxy force/spring diagnostic | 在 freejoint object 上用限幅 partner support 模拟真实人类另一端，先诊断 lift/floor-contact 缺口 | box025 bottom mean/floor contact 明显改善，且不引入腿/地板假支撑 |
| P2 | E005 micro-SBTO segment | 对 case-window 内 1.5-3.2s 做短段渐进 horizon refinement | 平滑度和 object lift/contact 优于 E081，计算成本可接受 |
| P2 | E006 RL-ready partner observation spec | 为后续单人 RL 定义 observation/reward：partner relative state、contact mask、object bottom/floor metric | 产出可训练接口，不一定跑 PPO |

## Claims 验证

| Claim | 结果 |
|-------|------|
| C1 object/freejoint 控制口径可被明确解释 | 通过。已区分 `scene.xml` freejoint、`scene_act` actuator-guidance、object PD/partner force/kinematic override。 |
| C2 E081 baseline 验收口径可复用 | 通过。已列出 main/guard baseline、必须指标、视觉检查和代码位置。 |
| C3 论文启发转化为本项目可执行实验 | 通过。已产出三论文综合和路线映射。 |
| C4 产出 E002+ 实验路线图 | 通过。已列出 5 个候选，P0 为 E002 freejoint audit。 |

## 下一步

写 E002 plan：创建不污染 E081 原派生任务的新 freejoint-legobj task，对 `box025_person2` 和 `box023_person2` 做 scene.xml freejoint + 腿/脚-object pair 对照；复用 E081 eval 指标和抽帧口径，直接回答“如果是 freejoint、只靠机器人/接触力，当前方案还能不能 work”。
