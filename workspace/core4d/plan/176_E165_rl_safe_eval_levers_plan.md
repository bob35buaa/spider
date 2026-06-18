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
| C-D | peak-margin rerank 能降低下游 hard-gate 擦边风险（杠杆2，后期） | CEM sample selection 后验显示 `ee_body_pos` / `anchor_pos` / root-z posture 的 **worst-frame margin** 拉开；box023/spider 的 `ee_body_pos` 失败占比显著下降，或对应 case staggered 完成率提升 |

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

| 子实验 | 对应 Claim | 改动面 | 判据 |
|---|---|---|---|
| **E165-D** 动态余量 peak-margin rerank | C-D | CEM sample-level elite selection（`spider/optimizers/sampling.py` / `sampling_fast.py`），新增默认关闭的 `cem_peak_margin_*` 选项 | box023/spider `ee_body_pos` hard-gate 失败占比下降，且 clean contact/penetration 不显著退化 |

#### Phase 3 设计约束（2026-06-18 讨论后定稿）

1. **rerank 对象不是离线结果文件**：E165-D 只改 CEM 内部每轮采样出的 `ctrls_samples[i]`。每个 sample rollout 得到一段 horizon 轨迹；在 elite selection 前计算 sample-level risk，再参与 top-k/softmax 权重或 fallback score。
2. **不把 `obj_pos/obj_ori` 作为 CEM sample rerank 主项**：CEM 中物体由 `object_pd_override` / `object_kinematic_override` / E015 soft weld 等机制 on-rails 跟随 GT，sample 间 `obj_*` 不是真自由物体可执行性信号。`obj_*` 继续作为 SUGAR 失败诊断和 reference 动态审计项，但不进入第一版 CEM rerank penalty。
3. **主 rerank 项只覆盖机器人可执行性与防塌姿态**：
   - `ee_body_pos`：SUGAR hard gate 对双腕/双踝的任意 body 位置误差，body list 固定为 `left_ankle_roll_link`、`right_ankle_roll_link`、`left_wrist_yaw_link`、`right_wrist_yaw_link`。
   - `anchor_pos`：SUGAR anchor body 为 `torso_link`，量 reference torso 与 robot torso 的世界坐标 L2 偏差。
   - `root-z posture`：沿用 E160 已验证有效的相对参考 root-z guard，防止 CEM 为了贴物体牺牲 body tracking/fall。
4. **`anchor_ori` 第一版只记录诊断，不作为强 penalty**：SUGAR 的 `anchor_ori` 不是完整 yaw/pitch/roll quaternion error，而是 projected gravity z 分量差，主要反映 torso roll/pitch 倾斜。除非 failed_windows 显示其成为主失败原因，否则不先纳入强 rerank。

#### Phase 3 sample-level 指标定义

对每个 CEM sample `i`、每个 rollout frame `t`：

```text
ee_body_err[i,t] =
  max over {left/right ankle, left/right wrist}
    || sugar_aligned_ref_body_pos[t,b] - sim_body_pos[i,t,b] ||

anchor_pos_err[i,t] =
  || ref_torso_link_pos[t] - sim_torso_link_pos[i,t] ||

root_z_err[i,t]  = abs(sim_root_z[i,t] - ref_root_z[t])
root_z_drop[i,t] = ref_root_z[t] - sim_root_z[i,t]
```

其中 `ee_body_err` 应尽量复刻 SUGAR 的 `body_pos_relative_w` 对齐语义：用 reference anchor 与当前 robot anchor 做 yaw/position 对齐后再比较 key body。若第一版在 MJWP 中直接复刻成本过高，允许先落一个 **diagnostic-only parity check**：对保存的 CEM trajectory 与 SUGAR failed-window 指标做离线对齐，确认 SPIDER 侧 proxy 与 SUGAR 排序一致后再开启强 penalty。

#### Phase 3 阈值与 margin 来源

| 项 | 下游 hard threshold 来源 | SUGAR hard gate | E165-D CEM safe threshold | rerank buffer | 说明 |
|---|---|---:|---:|---:|---|
| `ee_body_pos` | SUGAR `bad_motion_body_pos` | `0.30m` | `0.25m` | `0.03m` | 不改 SUGAR 终止门；CEM elite 选择比下游 hard gate 紧 5cm |
| `anchor_pos` | SUGAR `bad_anchor_pos` | `0.30m` | `0.25m` | `0.03m` | 防 torso/root 整体漂移贴近下游 hard gate |
| `root_z_mean_err` | E160 posture rerank | -- | `0.10m` | hard gate 复用 | 全 horizon root-z tracking |
| `root_z_terminal_err` | E160 posture rerank | -- | `0.12m` | hard gate 复用 | last 15% root-z tracking |
| `root_z_max_drop` | E160 posture rerank | -- | `0.18m` | hard gate 复用 | 只惩罚 sim 比 ref 低太多，不误杀正常弯腰 |

`0.25m` 是上游 CEM sample selection 的第一版 safe threshold，不是修改 SUGAR 的 `0.30m` hard termination。这样做的目的不是让 sample "刚好不越门"，而是在进入下游 RL 前预留 5cm 硬余量；再叠加 `0.03m` soft buffer 后，`ee_body_pos`/`anchor_pos` 的 peak 超过约 `0.22m` 就开始产生 rerank 扣分，超过 `0.25m` 进入明显高风险区。

`0.03m` buffer 来自 E163 failed-window 观察：下游失败多为刚越门的 cliff，如 box023/spider `ee_body_pos` 约 `0.325m > 0.30m`、box021/omni `obj_pos` 约 `0.317m > 0.30m`。E165-D 的 CEM 侧目标是保留足够动态余量，而不是贴线通过；若首轮发现 valid set 被 `0.25m` 过度压缩，再只调 `lambda/buffer` 或分 case 放宽，不默认回到 `0.30m`。

#### Phase 3 rerank 公式草案

```text
ee_peak      = max_t ee_body_err[t]
anchor_peak  = max_t anchor_pos_err[t]

ee_margin     = 0.25 - ee_peak
anchor_margin = 0.25 - anchor_peak

ee_violation =
  relu(0.03 - ee_margin) / 0.03

anchor_violation =
  relu(0.03 - anchor_margin) / 0.03

posture_violation =
    relu(mean(root_z_err) - 0.10) / 0.05
  + relu(mean_last15pct(root_z_err) - 0.12) / 0.05
  + relu(max(root_z_drop) - 0.18) / 0.05

peak_margin_violation =
    w_ee * ee_violation
  + w_anchor * anchor_violation
  + w_posture * posture_violation

rerank_score = reward - lambda_peak_margin * peak_margin_violation
```

初值建议：

```text
cem_peak_margin_enabled = false
cem_peak_margin_ee_threshold_m = 0.25
cem_peak_margin_anchor_threshold_m = 0.25
cem_peak_margin_buffer_m = 0.03
cem_peak_margin_w_ee = 1.0
cem_peak_margin_w_anchor = 0.5
cem_peak_margin_w_posture = 1.0
cem_peak_margin_lambda = 3.0
```

实现上应复用 E160 的 sample gate/fallback 思路：valid sample 足够时优先从 low-risk valid set 中按 reward 选 elite；valid 太少时使用 `rerank_score` fallback，而不是直接让 CEM 失败。

#### Phase 3 首轮范围

第一轮只跑 failure-focused 3 case，不直接扩 clean8：

| case | GPU 分配 | 目的 |
|---|---|---|
| `box023_person2` / spider E163 handoff | 本地 `gpu0` | 主要目标：降低 `ee_body_pos` 失败占比（E163 0/64 主因）；本地跑便于最快迭代阈值与 health 输出 |
| `box021_029_p2` 或对应 box021 E163 case | 远程 `spider-remote` GPU0 | 检查 anchor/root/posture margin，不回归 E160 曾修过的 fall |
| `box004_083_p2` 或 `box004_082_p1` | 远程 `spider-remote` GPU1 | 确认不破坏接触/抬升相关诊断；不期望 CEM rerank 单独解决 height reward |

#### Phase 3 执行与脚本约束

E165-D 首轮采用 **本地 1 卡 + 远程 2 卡并行**，对应 `.codex/skills/experiment-planning-zh/remote-execution.md`：

1. 本地 GPU 跑最关键的 `box023_person2`，优先观察 `ee_body_pos` peak/margin、`fallback_used` 和是否出现 posture 回归。
2. 远程机器使用 SSH alias `spider-remote`，项目路径 `/home/xiayb/pHRI_workspace/spider`，2x RTX 6000 Ada；GPU0 跑 `box021_029_p2`，GPU1 跑 `box004_*`，同一 GPU 内如后续扩 case 必须串行。
3. Phase3 开跑前必须固化脚本到 `workspace/core4d/scripts/launch/active/`，计划命名为：
   - `run_E165D_local.sh`
   - `run_E165D_remote.sh`
   - `pull_E165D_remote_results.sh`
   - 可选 `watch_and_pull_E165D.sh`
4. 远程启动走 tmux session，例如 `E165D_remote_<timestamp>`；本地启动也走 tmux session，例如 `E165D_local_<timestamp>`，避免 SSH/终端断开影响。
5. 远程代码同步优先走 git push/pull；如涉及 gitignored 的 handoff/npz/video 产物，使用 `pull_E165D_remote_results.sh` 回收，且每个 case 的 `video_output_path`、root npz copy 目标必须互不覆盖。

Phase3 结束前必须输出：

- CEM trajectory 中的 `cem_peak_margin_*` health 字段：`ee_peak/max/mean`、`anchor_peak/max/mean`、`margin_violation`、`selected_valid_frac`、`fallback_used`。
- strict eval 的 contact/penetration/tracking 表。
- SUGAR staggered rollout 的 failed_windows 对比：至少报告 `ee_body_pos`、`anchor_pos`、`obj_pos/obj_ori` 失败计数。`obj_*` 不作为 CEM rerank 主项，但必须确认是否成为新的主失败。

**正交性说明**：杠杆2（D）治"擦边越门"，杠杆3（B）治"抬升缺失"，两者互不替代。

#### Phase 3 执行记录（E165D，2026-06-18）

- 已按 `0.25m` safe threshold / `0.03m` buffer 实现并完成三 case full：本地 GPU0 `box023_person2`，远程 `spider-remote` GPU0 `box021_029_p2`，远程 GPU1 `box004_083_p2`。
- 产物回收后 root `.npz` / full mp4 / `trajectory_mjwp_act.npz` / `config_act.yaml` 均为 `3/3`；strict eval 输出 `metric_rows=6, missing=0, all_artifacts_ok=true`。
- 首轮结果：E165D 3/3 tracked、0 fall，mean EEF/root tracking 相对 E163 baseline 改善；但 clean 3mm contact 下降、3mm 物理穿透和 2mm 几何穿透上升，尤其 `box023_person2` raw contact 从 E163 `0.8769` 降到 `0.7077`。
- 因此 E165D 当前只证明 peak-margin sample-level rerank 管线可用，不直接推广为 RL handoff。Claim C-D 仍需 SUGAR staggered failed_windows 对比；若继续，应做 E165D2：降低 peak-margin 强度或作为 tie-breaker，并加入 contact-preservation guard。
- 诊断性 RL export 已完成：`rl_export_input.tsv` 为 `3/3 RL_EXPORT_READY`，partner OmniRetarget `3/3 pass`，输出在 `workspace/core4d/results/E165/peak_margin_rerank/rl_export/`。该 export 只是为后续 SUGAR failed_windows 对比准备输入，不改变“不推广”的判定。
- 详细结果见 [log/212_E165D_peak_margin_rerank_results.md](../log/212_E165D_peak_margin_rerank_results.md)。

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
