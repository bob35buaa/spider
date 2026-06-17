# E165 实验计划：下游 RL 可恢复性驱动的 SPIDER 评测杠杆与修复闭环

日期：2026-06-18
分支：`experiment/E161-surface-release-ablation`（exp_name: core4d，沿用，不新建分支）
上游依据：`workspace/core4d/docs/E163_RL_DEEP_INSIGHTS_CN.md`、`SUGAR-private/docs/SUGAR_EVAL_DETERMINISM_AND_STAGGERED_PHASE_CN.md`
RunID：Phase 0 为离线审计，不消耗 R（无新重定向）；Phase 2/3 产出新 R，开跑前再分配。

---

## Context

E163_narrow 在 SPIDER 自评的几何/物理接触/穿透/raw 接触上全面最优，但下游 RL（SUGAR refiner）成功率不随之单调最优。前序结论（已落档）：

1. **staggered-phase eval 已把二值 0/64↔64/64 改成连续成功率**：box021/spider 0.67、box004/spider 0.48、box023/spider 0.00 —— 证明原二值指标分辨率为零（box021 与 box004 原本都判 64/64）。
2. **可恢复性不对称（主线）**：RL 能自修复接触标签误差，但**不能**修复(a)离硬门的动态余量、(b)本体可执行性、(c)抬升语义——而 SPIDER 这三项都没度量。
3. **三种相反的接触病**：box021 健康；box004 接触标签虚高（手够不到箱）；box023 net 力虚假（手贴髋自碰撞穿透）。单一 net-filter gap 标量会把后两者混反。

### 根因分析

**关键事实（本轮代码确认，`spider/config.py:153-162`）：CEM 里物体是 GT。** 三种模式之一恒成立：`object_pd_override`（kp=2000 强 PD 跟 ref）/ `object_kinematic_override`（freejoint 直接写插值 ref，完全上轨）/ E015 软 weld。即 **CEM 只优化机器人，物体始终被拽着 follow GT 轨迹，机器人不需要真正承重/抬升**。

由此推出两条对评测框架的硬约束：

- **抬升语义无法在 CEM 选择端度量**：物体 z 恒等于 GT，每条 CEM rollout 都"完美抬升"，CEM 永远看不到抬不起来。→ 抬升必须作为**消费端（RL）**的 reward/约束，不能进 CEM rerank。
- **接触/穿透是 SPIDER 自评的 Goodhart**：在自家 MuJoCo-Warp、物体上轨条件下打分，数字能涨但接触不一定可迁移。唯一在本轮**真正预测了下游成败**的是 **Isaac 运动学回放探针（policy-free，物体仍按 ref on-rails）**。

### 关键 insight

把三个"RL 不可恢复"维度各配一个**廉价、可在训练前跑、与下游相关性已验证**的诊断/修复杠杆：

- **杠杆1（评测）**：Isaac on-rails 运动学接触探针 → handoff preflight 闸。**不拆 free-object、不考虑 free-joint z**（物体始终 on-rails），只报接触几何标量。
- **杠杆2（CEM 选择，后期）**：rerank 目标从"均值误差小"→"最坏帧离硬门余量大"。
- **杠杆3（RL reward）**：抬升语义进 RL reward（**不进 CEM**，因物体是 GT）。

---

## Claims

| # | Claim | 最低证据（量化） |
|---|-------|------------------|
| C-A | box004 接触标签虚高源于"手够不到箱" | 标签=接触的帧里，hand↔box 表面最近距离的 **中位数 > rubber-hand proxy 半径**（取 0.05–0.08m 上界 0.08m）；且这些帧 Isaac filtered 接触 recall < 0.2 |
| C-C | box023 frame0 自碰撞是**继承自源/姿态**而非 CEM 独有 | spider 与 omni 两条 handoff 的 frame0–10 **min(hand↔同侧髋) 均 < 0.10m**（两者都贴）→ 判继承；若仅 spider<0.10 而 omni≥0.10 → 判 CEM 引入 |
| C-E1 | on-rails 探针三标量能在 RL 前把三 case 分开，且与下游一致 | recall：box021>0.5、box004<0.2；max_init_net_force：box023>1000N、box021/box004<100N。排序与 staggered 成功率(0.67/0.48/0.00)方向一致 |
| C-B | 抬升语义进 RL reward 能救回 box004 height（杠杆3） | box004/spider 的 height 成功 **0/64 → ≥ 20/64**，且 staggered 完成率不低于 0.40、contact_ratio 不低于基线 0.50 |
| C-F | 消除 box023 手贴髋自碰撞能降低 ee_body 越门 | frame0 net 力 **→ <100N**；重训后 box023/spider staggered 完成率 **由 0.00 → >0** 或 ee_body 失败占比由 48/64 显著下降 |
| C-D | peak-margin rerank 能救回擦边失败（杠杆2，后期） | box021/omni（差 1.7cm）或 box023 的 obj_pos/ee_body **peak deviation 离门距离拉开**，重训后对应 case 成功率提升 |

---

## 实验结构（按"训练-free 先行、动核心后置"分 Phase）

### Phase 0 — 离线审计（无训练，hssim 环境，本机可全做）

| 子实验 | 对应 Claim | 脚本（待建） | 输入 | 产出 |
|---|---|---|---|---|
| **E165-A** box004 接触标签审计 | C-A | `scripts/eval/runners/eval_E165_box004_contact_audit.py` | `Core4D_E163N_Box004_R161/data_000/{robot_50hz.npz, contact_labels_50hz.npy}` + box ref 轨迹 | 逐帧 hand↔box-surface distance vs label 曲线 + 命中率表 |
| **E165-C** box023 初始穿透溯源 | C-C | `scripts/eval/runners/eval_E165_box023_penetration_trace.py` | spider `Core4D_E163N_Box023_R158` / omni `..._SamePersonOmniRT` / 源 Core4D | frame0–10 三方 min(hand↔同侧髋) 对比表 |
| **E165-E1** Isaac on-rails 探针成型为 preflight | C-E1 | `scripts/eval/runners/eval_E165_isaac_onrails_probe.py`（复用 `SUGAR-private/outputs/core4d_e163_threecase_contact_probe/*/isaac_contact_framewise.csv`） | 三 case isaac_contact_framewise.csv | 三标量 JSON：`filtered_contact_recall` / `phantom_force_rate` / `max_init_net_force` + 排序 vs staggered |

Phase 0 全部走 `eval.core.core_metrics`（如需公共指标）；纯分析脚本豁免 scene 快照（skill §10b 例外）。

### Phase 1 — 文档纠错（无训练）

- 重写 `docs/E163_RL_DEEP_INSIGHTS_CN.md` §4：杠杆3 改为 **RL reward**（删"CEM selection 加 z_max 约束"，因物体是 GT）；杠杆1 定稿为 **on-rails 三标量 preflight，不拆 free-object、不考虑 free-joint z**。
- 同步 SUGAR 文档对应段。
- 依赖 E165-A/C/E1 的结论数字回填。

### Phase 2 — RL 侧最小闭环（需 sugar GPU，远程；config 开关驱动、可逆）

| 子实验 | 对应 Claim | 改动面 | 判据 |
|---|---|---|---|
| **E165-B** height reward 进 RL | C-B | SUGAR reward cfg（新增 z-tracking/lift 项，开关默认关） | box004 height 0/64→≥20/64，完成率≥0.40，contact 不退化 |
| **E165-F** box023 自碰撞修复 | C-F | SUGAR：缩 rubber-hand proxy 半径 / 手↔髋 collision filter / 预抓取窗口不计 undesired_contacts | frame0 net<100N；box023 staggered>0 或 ee_body 失败占比降 |

依赖：E165-F 改动方向由 E165-C 溯源结论决定（上游源 vs CEM vs SUGAR proxy）。

### Phase 3 — CEM 核心（动上游，重训验证，最后）

| 子实验 | 对应 Claim | 改动面 |
|---|---|---|
| **E165-D** 动态余量 peak-margin rerank | C-D | CEM rerank 目标（core，git 隔离可逆）：对 obj_pos/obj_ori/ee_body/anchor 各算 peak deviation，惩罚接近下游阈值的峰值 |

**正交性说明**：杠杆2（D）治"擦边越门"，杠杆3（B）治"抬升缺失"，两者互不替代。

---

## 需要修改 / 新建的文件

| # | 文件 | 改动 | Phase |
|---|------|------|-------|
| 1 | `workspace/core4d/scripts/eval/runners/eval_E165_box004_contact_audit.py` | 新建（离线审计 A） | 0 |
| 2 | `workspace/core4d/scripts/eval/runners/eval_E165_box023_penetration_trace.py` | 新建（溯源 C） | 0 |
| 3 | `workspace/core4d/scripts/eval/runners/eval_E165_isaac_onrails_probe.py` | 新建（探针 E1，封装三标量 preflight） | 0 |
| 4 | `workspace/core4d/scripts/eval/wrappers/eval_E165_offline_audit.sh` | 新建（串起 A/C/E1） | 0 |
| 5 | `workspace/core4d/docs/E163_RL_DEEP_INSIGHTS_CN.md` | 纠错 §4（杠杆1/3） | 1 |
| 6 | `SUGAR-private/.../carry_box_refiner_env_cfg.py` 等 reward cfg | height reward 开关（默认关，可逆） | 2 |
| 7 | `SUGAR-private/.../assets/robots/unitree.py` 或 collision cfg | rubber-hand 半径/collision filter | 2 |
| 8 | CEM rerank 模块（`spider/optimizers/sampling.py` 或 config） | peak-margin rerank 选项 | 3 |

## 训练命令（Phase 2/3 开跑前固化为脚本）

```bash
# Phase 0（离线，无训练，本机 hssim）
bash workspace/core4d/scripts/eval/wrappers/eval_E165_offline_audit.sh

# Phase 2（远程 RL，开跑前生成 train 脚本，沿用 R158/R160/R161 handoff）
# bash workspace/core4d/scripts/train/train_core4d_E165B.sh R{XXX} {GPU}
```

## 成功标准（汇总）

| 指标 | 现状 | 本次目标 |
|------|------|----------|
| box004 标签接触帧 hand↔box 距离中位数 | 未量化 | **> 0.08m**（坐实 fiction） |
| box023 frame0 hand↔髋 (spider/omni) | spider 0.069–0.075m | **判定来源**（两者皆<0.10→继承） |
| on-rails 探针三标量排序 vs staggered | 雏形 | **方向一致**（recall/init-net 分开三 case） |
| box004 height 成功 | 0/64 | **≥ 20/64**（Phase 2） |
| box023 frame0 net 力 | 2357N（运动学探针） | **< 100N**（Phase 2） |

## 可视化（强制，skill §9）

- Phase 0：A/C 产出逐帧 distance/force 曲线图（matplotlib），写入 log「实际观察」；E1 三 case 接触热力对比。
- Phase 2：RL rollout 视频（远程 rerun 或离线渲染）+ `/video-frames` 抽关键帧（抓取/抬升时刻），核实 box004 是否真抬、box023 ee_body 是否缓解。

## 下一步

按用户优先级从 **Phase 0 E165-A** 开始（box004 接触标签审计，纯离线）。A/C/E1 三者无相互依赖，可并行实现。
