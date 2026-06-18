# E166 实验计划：基于 E163_narrow 的脚约束 + 平滑重定向优化（下游可执行性驱动）

日期：2026-06-18
分支：`experiment/E161-surface-release-ablation`（exp_name: core4d，沿用，不新建分支）
上游依据：
- `SUGAR-private/docs/CORE4D_E163_8CASE_E165D_DOWNSTREAM_ANALYSIS_CN.md`（重点 §2.5/§2.6/§8/§9）
- `workspace/core4d/docs/E163_RL_DEEP_INSIGHTS_CN.md`（可恢复性不对称 / 三杠杆）
- E165 系列：plan `176_E165_rl_safe_eval_levers_plan.md`、log `211/212/213`

RunID：Phase 0（E166-R3）为离线审计，不消耗 R（无新重定向）；Phase 1/2/3 产出新 R，开跑前再分配。E-num 本计划占用 **E166**，子实验以 `E166-{R3,B1,B2,A1,A2,A3,C}` 标识。

> **与 E165D 的边界（务必不混）**：E165D（杠杆2）优化的是"最坏帧离硬门**余量**"，§4 已证明它会牺牲接触、下游 box021 从 0.67 崩到 0/64，**不推广**。本计划（E166）优化的是参考运动的**平滑度 + 脚可执行性**——§2.5/§2.6 数据里真正区分下游成败的量。两条路线正交，E166 **不复用** `cem_peak_margin_*` 的 margin penalty，新增独立的 `cem_smooth_*` / `foot_*` 开关。

---

## Context

E163_narrow 在 SPIDER 自评的几何/物理接触/穿透/raw 接触上全面最优，但下游 RL（SUGAR refiner）的 staggered 完成率不随之单调最优。8case 扩展（log 213）把上一轮 n=3 的"可恢复性不对称"钉死为 n=7 的规律，下游分析（`CORE4D_E163_8CASE_E165D_DOWNSTREAM_ANALYSIS_CN.md`）给出四条最强证据：

1. **`ee_body_pos`（本体可执行性硬门）是 6/7 训练 case 的头号终止原因**（box023 48/64、box004_083_p1 36/64……），且这些失败 **97–100% 由脚（踝）驱动**（§2.5）。下游一刀切掉策略的是"机器人搬箱时脚跟不上参考步态/站位"，不是手够不到箱——一个**下半身 locomotion/平衡可执行性**问题，与 SPIDER 优化的操作接触基本正交。
2. **数据质量的真判别量是参考运动的平滑度/速度**（jerk、加速度、物体速度），不是物体类别（§2.6）：两个成功 case 的 accMax 56/60、jerk P95 1210/907；四个差 case 的 accMax 142–362、jerk P95 2956–3773（jerk ~3×、acc ~3–6×）。box021 自己另两条又抖又快的 clip 同样失败。
3. **上游接触不预测下游**：box021_035_p2 的 Isaac 接触比 035_p1 好 1.7×，staggered 反而差 14×。继续刷接触/穿透/tracking 的边际收益已接近零。
4. **E165D peak-margin 在唯一干净可比的 box021 上回退 0.67→0/64**，方向与上游接触退化一致；margin rerank 不是答案。

### 根因分析（code-grounded，基于 `spider/config.py` + `mjwp.py`）

下游失败的两大病灶在上游 SPIDER 完全没被约束：

| 维度 | SPIDER 当前做法（代码位置） | 缺口 |
|---|---|---|
| **脚/下半身** | 脚踝只作为通用 body-position 目标被跟踪：`local_frame_lower_ids = range(2,14)`（hips→ankles，`config.py:454`），与其它下身 body 同权重；reward 在 `mjwp.py:790` 的 `lower_pos_rew` | **无足-滑(foot-slip)惩罚、无足-地接触一致性、无 ankle 专项加权、无双脚支撑相位约束**。脚跟人体参考走，没人保证 g1 物理上踩得住 |
| **平滑** | 仅 `vel_rew_scale=0.0001`（`config.py:578`，`qvel_rew` 在 `mjwp.py:726`，且实际只在 dexmachina `use_rl_reward` 模式有意义）| **无 jerk/加速度惩罚、无输出轨迹时序平滑**。CEM 逐帧解的抖动原样进 handoff |
| **峰值余量** | E165D `cem_peak_margin_*`（含 ankle，`config.py:346-365`）已可选 | 是"离门**余量**"不是"**平滑度**"；§4 证明它牺牲接触、下游回退 |

即：**脚被当成"跟着人走"的被动目标，平滑完全没做**——这正是下游 97–100% 脚驱动失败 + 仅平滑 case 才成功的上游对应物。

### 关键 insight

把下游头号失败通道（脚 + 抖动）各配一个**config 开关驱动、默认关、可逆**的上游 CEM 干预，并先用**训练-free 的离线红线**验证"平滑度/脚一致性比接触更预测下游"这个前提，再花 GPU 重训验证：

- **杠杆 S0（最高）**：平滑参考运动，尤其脚/下半身（CEM jerk/accel 惩罚 + handoff 后平滑）。
- **杠杆 S1（脚约束）**：足-滑惩罚 + ankle 加权 + 足-地接触一致性。
- **杠杆 C（红线 preflight）**：对 CEM 输出轨迹做 Tier-1 纯运动学红线（jerk P95/accMax/足滑/足-地一致性/物体 Vmax），未训练即拦截注定失败的 handoff。

复用已有代码结构：smoothness/foot 的 sample-level gate 完全照搬 E160 posture gate / E165D peak_margin 在 `sampling.py` 的 `info_combined[...].max(dim=0)`（dim0=时间）聚合范式；ankle 加权照搬 E044 `local_frame_wrist_weight`（`mjwp.py:765`）。

---

## Claims

映射 §8.3 的 R1（平滑消融）/R2（脚约束消融）/R3（红线预测力），加可分主因的子项。每条都有量化最低证据，实验后逐条验证。

| # | Claim | 最低证据（量化） |
|---|-------|------------------|
| **C-R3** | jerk/acc 峰值对下游 staggered 的预测力 **强于** Isaac 接触 | 8 case 上 `jerk_P95`、`accMax`（被跟踪 body，尤其踝）与 staggered 完成率的 \|Spearman ρ\| ≥ 0.6，且**显著大于** Isaac `both@0.1`/IoU 与 staggered 的 \|ρ\|（§2.2 已暗示接触≈0 预测力）|
| **C-R3b** | 足-滑/足-地不一致同样有预测力 | 着地相位脚 XY 位移峰值 + 着地帧脚高度偏差与 staggered 的 \|ρ\| ≥ 0.5 |
| **C-B1** | CEM jerk/accel 惩罚能压低输出轨迹抖动且不牺牲接触 | 对 box021_035_p2 / box004_082_p1：CEM 输出 `jerk_P95` 从 3217/2956 **降到 ≤ 1500**、踝 `accMax` 显著下降；同时 clean 3mm contact / 穿透 **相对 E163 baseline 不退化**（容忍 ≤ 5% 相对劣化）|
| **C-B2** | handoff 后平滑（Savitzky-Golay）能进一步降抖且保接触相位 | 平滑后 `jerk_P95` 再降，接触 mask 的 on/off 相位边界位移 ≤ 1 帧（不破坏抓/放时刻）|
| **C-A1** | 足-滑惩罚 + ankle 加权降低下游踝部 ee_body 失败 | 重训后 3 case 的 `ee_body_pos` 失败中**踝主导占比**下降，且 staggered 上升：box021_035_p2 0.016→≥0.10、box004_082_p1 0.094→≥0.20、**box004_083_p2 0.484→≥0.60**（A 隔离 case，A 臂必须救它）|
| **C-A3** | 足-地接触一致性约束减少脚抬离/插地 | 着地相位脚高度偏差峰值下降；可视化无脚穿地/悬空 |
| **C-disentangle** | A/B1/B2 的下游贡献可被因子化消融分清（核心） | 预测矩阵成立：**box004_083_p2 上 B1/B2 臂 staggered 提升 ≈0（< +0.05）而 A 臂显著 >0**（脚约束独立有效、平滑不串味）；两条脚坏且抖 case 上 A+B > 单臂 |
| **C-B1vsB2** | CEM 内平滑(B1) 与事后平滑(B2) 的性价比可比 | 报告两条抖 case 上 B1 vs B2 的 jerk 降幅与下游 staggered 增益；若 B2 增益 ≥ B1 的 80%，结论"事后平滑足够、无需动 CEM" |
| **C-C** | Tier-1 红线能在训练前正确分出"注定失败"的 handoff | 用 C-R3 标定阈值对 8 case 二分类（staggered<0.10 为负），Tier-1（**须含 foot_slip/foot_ground，不能只看 jerk**）precision/recall 均 ≥ 0.75。注：box004_083_p2 平滑(jerk907)但脚坏，只有 foot 维度能标它——验证 Tier-1 多维度的必要性 |

> **正交性硬约束（E165D 教训）**：所有 C-B*/C-A* 实验的**接触/穿透不退化**是一票否决项。任何"平滑/脚约束提升了 tracking 但接触退化"的结果，按 §S2 判为**不推广**，与 E165D 同等对待。

---

## 实验结构（训练-free 先行，验证前提；再动 CEM；最后下游重训）

### Phase 0 — E166-R3 离线红线预测力（无训练，本机，最先做）

**目的**：在花任何 GPU 之前，先证明"平滑度/脚一致性比接触更预测下游"。若 C-R3 不成立，整条路线的前提就错了，必须先停下重想（skill §12）。

| 子实验 | 对应 Claim | 脚本 | 输入 | 产出 |
|---|---|---|---|---|
| **E166-R3** 8case 红线 vs staggered 相关性 | C-R3 / C-R3b / C-C | 扩展 `scripts/eval/runners/eval_E166_redline_predictive.py`（基于已有 `eval_E165_8case_handfoot_dataquality.py`）| 8 case handoff `robot_50hz.npz` + 各 case `success_vs_phase.csv`（staggered）+ Isaac `isaac_contact_framewise.csv` | 相关性表（Spearman ρ：jerk/acc/foot-slip/foot-ground vs staggered，对照 Isaac 接触）+ Tier-1 二分类 PR + 红线阈值标定 JSON |

- 复用 `eval.core.core_metrics`（如需公共指标）；纯分析脚本豁免 scene 快照（skill §10b 例外）。
- **Tier-1 指标定义**（纯运动学，毫秒级 CPU，对 handoff 数组直接算，不起 sim）：
  - `jerk_P95` / `accMax`：被跟踪 body 的逐帧二阶/三阶差分（尤其 `left/right_ankle_roll_link`、`local_frame_lower_ids`）。
  - `foot_slip`：参考着地相位（ref 脚高度 < 阈，如 0.05m）内脚踝世界 XY 位移峰值。
  - `foot_ground_consistency`：着地帧 g1 脚高度与 ref 脚高度偏差峰值。
  - `obj_Vmax`：物体速度峰值（§2.6 已列）。
- **门槛**：Phase 0 通过（C-R3 成立）才进 Phase 1；否则向用户报告并重新评估假设。

### Phase 1 — E166-B 时序平滑（动 CEM reward / handoff 后处理；config 开关，默认关）

| 子实验 | 对应 Claim | 改动面 | 判据 |
|---|---|---|---|
| **E166-B1** CEM jerk/accel 惩罚 | C-B1 | `config.py` 新增 `cem_smooth_*`（默认关）；`mjwp.py` get_reward 输出每帧 body 速度/加速度 info；`sampling.py` 在 horizon 上算 jerk/accel 并入 reward/gate | 输出 jerk_P95 降到 ≤1500、接触不退化 |
| **E166-B2** handoff 后平滑 | C-B2 | 新建 `spider/postprocess/smooth_handoff.py`（Savitzky-Golay/低通，保接触相位）。**纯后处理，不跑 CEM**：对已有 handoff 轨迹数组直接平滑，秒级 CPU | 再降 jerk，接触相位边界位移 ≤1 帧 |

> **B2 关键性质**：B2 是 handoff 轨迹的事后平滑，**不需要任何 SPIDER/CEM 优化**——套在 baseline（或任意臂）的 CEM 输出上即可。因此 B2-only 臂 = `baseline CEM 输出 + Savitzky-Golay`，无新 CEM。这也让 B1（CEM 内平滑，花算力）vs B2（事后平滑，零成本）的对比有意义：若 B2 就够，则无需动 CEM。

### Phase 2 — E166-A 脚部专项约束（动 CEM reward；config 开关，默认关）

| 子实验 | 对应 Claim | 改动面 | 判据 |
|---|---|---|---|
| **E166-A1** 足-滑惩罚 | C-A1 | `config.py` `foot_slip_*`；`mjwp.py` get_reward 在支撑相位惩罚踝世界 XY 速度 | 下游踝 ee_body 失败占比降、staggered 升 |
| **E166-A2** ankle 加权 | C-A1 | `config.py` `local_frame_ankle_ids`/`local_frame_ankle_weight`（镜像 E044 wrist，`config.py:462-466`）；`mjwp.py:790` lower_pos_rew 加权 | 同上 |
| **E166-A3** 足-地接触一致性 | C-A3 | `config.py` `foot_ground_*`；着地帧约束脚高度/法向一致 | 脚高度偏差降、无穿地/悬空 |
| **E166-A4**（可选/后置）双脚支撑相位 | — | 限制双脚同时离地腾空 | 仅在 A1–A3 不足时启用 |

### Phase 3 — E166-C 因子化消融 + 下游 staggered 验证 + 红线 gate 成型（需 GPU，本地+远程）

**设计：用 case 横跨 (脚×抖) 2×2，再用 5 臂消融拆 A/B1/B2。** 这批数据里 jerk 与脚坏强相关（§2.6），单靠选 case 拆不开 A/B，必须叠加同 case 消融臂。

#### 3 个代表性 case（2026-06-18 与用户讨论后定稿）

| case | 类别 | jerkP95 | 脚主导% | 绝对踝err [L,R] | object | 角色 |
|---|---|---:|---:|---|---|---|
| `box021_035_p2`（`Core4D_E163N8_d003_box021_..._035_p2`）| 脚坏且抖（最极端）| 3217 | 100% | [0.43, **0.60**] | box021 | 主战场 |
| `box004_082_p1`（`Core4D_E163N8_e091_box004_..._082_p1`）| 脚坏且抖（动力学最抖 acc362/objV5.7）| 2956 | 100% | [0.47, 0.44] | box004 | 跨物体复现 |
| `box004_083_p2`（=r161，`Core4D_E163N_Box004_R161`）| **脚坏但不抖**（A 隔离）| **907** | 72% | [0.30, 0.25] | box004 | 判别脚约束独立性 |

> 选 case 依据：`box004_083_p2` 平滑（jerk 907、acc 60，与成功 case 同级）但脚仍坏 → **B 在它身上该≈无效、A 才该起效**，是把脚约束(A)从平滑(B)里剥离的唯一干净 case。两条"脚坏且抖"覆盖两个物体。数据里**不存在"脚好但抖"的 case**（jerk↔脚坏相关，§2.6），故 B 无 case 级隔离，靠 B1/B2 消融臂取独立效应。`box004_083_p2` 与 `box004_082_p1` 同为 box004，构成同物体 jerk-轴对比（平滑 vs 抖，均脚坏）。

#### 5 臂消融（config 开关组合，全默认关、可逆）

| 臂 | 开启项 | 新 CEM | 新 RL | 含义 |
|---|---|:--:|:--:|---|
| **baseline** | 无（E163_narrow 原样）| ✗ 复用 | ✗ 复用 E163 staggered | 对照 |
| **B1** | `cem_smooth_*`（CEM jerk/accel 惩罚）| ✅ | ✅ | CEM 内平滑（花算力）|
| **B2** | handoff Savitzky-Golay 后平滑 | ✗ 纯后处理 | ✅ | 事后平滑（零成本）|
| **A** | `foot_slip_*` + `local_frame_ankle_weight` + `foot_ground_*` | ✅ | ✅ | 脚约束三项 |
| **A+B** | A + `cem_smooth_*` + 后平滑（全栈）| ✅ | ✅ | 合并增益 |

**compute**：每 case = 3 新 CEM（B1/A/A+B）+ B2 后处理（≈free）+ 4 新 RL。**3 case × = 9 CEM + 12 RL**（baseline 复用 E163，省 6 条）。

> baseline 复用前提：新臂只改 CEM reward / handoff 后处理，**下游 SUGAR RL 训练 recipe 必须与 E163 完全一致**，否则 baseline 须在同 recipe 下重跑以保公平（开跑前核对 RL config 是否与 log 213 一致）。

#### 可证伪预测矩阵（每格 = staggered 提升预期；✗≈无效）

| case \ 臂 | B1 | B2 | A | A+B | 关键证伪点 |
|---|:--:|:--:|:--:|:--:|---|
| `box004_083_p2`（脚坏不抖）| ✗ | ✗ | ✓✓ | ✓✓(≈A) | **若 B1/B2 救了已平滑的它 → 平滑机制串味/实现 bug**；若 A 救不了 → 脚约束无效 |
| `box021_035_p2`（脚坏且抖）| ✓ | ✓ | ✓ | ✓✓✓ | A+B 应 > 单臂；B1 vs B2 谁强=CEM内 vs 事后平滑之争；单臂增益大者=主因 |
| `box004_082_p1`（脚坏且抖）| ✓ | ✓ | ✓ | ✓✓✓ | 跨物体复现 box021 的主因排序与 B1/B2 强弱 |

#### GPU 分配（本地 1 卡 + 远程 2 卡，§8b）

- **CEM 重定向**（SPIDER）：本地 GPU 迭代最关键的 `box021_035_p2` 三臂；远程 `spider-remote` GPU0/GPU1 分跑两条 box004 的三臂。B2 后处理本机 CPU 批量做。
- **RL 重训 + staggered eval**（SUGAR）：远程 2 卡并行，12 条 RL 按 GPU 均分串行；baseline 复用不重训。
- 开跑前脚本固化到 `scripts/launch/active/run_E166_remote.sh` / `pull_E166_remote_results.sh`，video/npz 路径互不覆盖。

---

## 改动（code-grounded）

### 1. E166-R3 离线红线脚本（Phase 0，无 core 改动）

**文件**：`workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py`（新建，基于 `eval_E165_8case_handfoot_dataquality.py`）

- 对 8 case handoff 直接算 Tier-1 指标（jerk_P95/accMax/foot_slip/foot_ground/obj_Vmax）。
- 读各 case staggered `success_vs_phase.csv` 求完成率；读 Isaac `isaac_contact_framewise.csv` 求 both@0.1/IoU。
- 输出 Spearman ρ 表（红线指标 vs staggered，对照接触）+ Tier-1 二分类 PR + 阈值 JSON。

### 2. E166-B1 CEM jerk/accel 惩罚（Phase 1）

**文件**：`spider/config.py`（新增字段，默认关）

```python
# E166: temporal smoothness penalty on tracked body trajectories (ankle/lower-body focus)
cem_smooth_enabled: bool = False
cem_smooth_body_ids: list[int] = field(default_factory=list)   # resolved from names
cem_smooth_body_names: list[str] = field(default_factory=lambda: [
    "left_ankle_roll_link", "right_ankle_roll_link",
])
cem_smooth_accel_weight: float = 0.0   # per-frame body accel L2 penalty
cem_smooth_jerk_weight: float = 0.0    # horizon 3rd-diff jerk penalty
```

**文件**：`spider/simulators/mjwp.py`（`get_reward` ~726 附近）
- 输出每帧被跟踪 body 的线速度/加速度到 info（沿用 peak_margin 在 ~2178 处 info 输出范式）。

**文件**：`spider/optimizers/sampling.py`（~242 的 `info_combined[...].max(dim=0)` 范式）
- 对 horizon（dim0=时间）做二阶/三阶差分得 accel/jerk，按 sample 聚合（max/P95），乘权重并入 `reward`（不进 valid gate，避免 E165D 式硬过滤压缩 valid set）。

> 设计考量：jerk 是窗口量，必须在 `sampling.py` 的 horizon buffer 上算，不能在单帧 `get_reward` 里算——这正是 E165D peak_margin / E160 posture 已用的 `max(dim=0)` 聚合点。第一版只惩罚踝（最高杠杆），通过 `cem_smooth_body_names` 可扩到 `local_frame_lower_ids`。

### 3. E166-B2 handoff 后平滑（Phase 1）

**文件**：`spider/postprocess/smooth_handoff.py`（新建）
- 对 CEM 输出 `trajectory_kinematic.npz`/`robot_50hz.npz` 的 body/joint 轨迹做 Savitzky-Golay，**保接触相位**（接触 mask on/off 边界帧不平滑或单独处理）。
- 作为 handoff 导出前可选后处理步；输出独立文件，不覆盖原始 CEM 解。

### 4. E166-A1/A2/A3 脚约束（Phase 2）

**文件**：`spider/config.py`（新增，默认关）

```python
# E166: foot constraints
foot_slip_enabled: bool = False
foot_slip_weight: float = 0.0
foot_slip_contact_height_m: float = 0.05   # ref ankle height below -> "grounded"
foot_ground_enabled: bool = False
foot_ground_weight: float = 0.0
# A2: ankle extra weight (mirror E044 wrist_weight at config.py:462-466)
local_frame_ankle_ids: list[int] = field(default_factory=list)
local_frame_ankle_weight: float = 1.0
```

**文件**：`spider/simulators/mjwp.py`
- A1：`get_reward` 在 ref 着地相位惩罚踝世界 XY 速度（从 `data_wp` 速度/相邻帧位移）。
- A2：`mjwp.py:790` `lower_pos_rew` 加 ankle extra-weight 项，**完全镜像 765 行 wrist 写法**。
- A3：着地帧约束 g1 脚高度/法向与 ref 一致。

### 需要修改 / 新建的文件汇总

| # | 文件 | 改动 | Phase |
|---|------|------|-------|
| 1 | `workspace/core4d/scripts/eval/runners/eval_E166_redline_predictive.py` | 新建（红线预测力 R3）| 0 |
| 2 | `workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh` | 新建（R3 入口）| 0 |
| 3 | `spider/config.py` | 新增 `cem_smooth_*` / `foot_slip_*` / `foot_ground_*` / `local_frame_ankle_*`（全默认关）| 1/2 |
| 4 | `spider/simulators/mjwp.py` | get_reward 输出 body vel/accel info + 脚约束项 | 1/2 |
| 5 | `spider/optimizers/sampling.py` | horizon jerk/accel 聚合并入 reward | 1 |
| 6 | `spider/postprocess/smooth_handoff.py` | 新建（B2 后平滑，保接触相位）| 1 |
| 7 | `workspace/core4d/scripts/experiments/E166/build_*.py` | manifest（3 case × 5 臂，标注复用/新跑/后处理）| 3 |
| 8 | `workspace/core4d/scripts/train/train_core4d_E166{B1,A,C}.sh` | CEM 重定向入口（首步调 snapshot_scenes.sh；B2 走后处理不进此脚本）| 1/2/3 |
| 9 | `workspace/core4d/scripts/launch/active/run_E166_remote.sh` / `pull_E166_remote_results.sh` | 远程并行（CEM 9 条 + RL 12 条）| 3 |
| 10 | `workspace/core4d/scripts/eval/runners/eval_E166_ablation.py` + wrapper | 5 臂 staggered/接触/穿透/踝失败对比 + 预测矩阵核对 | 3 |
| 11 | `workspace/core4d/results/E166/scene_snapshot/` | scene XML 快照（skill §10b，训练前）| 1/2/3 |

## 训练命令（Phase 1/2/3 开跑前固化为脚本）

```bash
# Phase 0（离线，无训练，本机）
bash workspace/core4d/scripts/eval/wrappers/eval_E166_redline.sh

# Phase 1/2（SPIDER CEM 重定向，开跑前生成 train 脚本 + 首步 snapshot）
# bash workspace/core4d/scripts/train/train_core4d_E166B.sh R{XXX} {GPU} {case}

# Phase 3（本地+远程并行：SPIDER CEM 9条 + B2后处理 + SUGAR RL 12条重训 + staggered eval）
# bash workspace/core4d/scripts/launch/active/run_E166_remote.sh
# bash workspace/core4d/scripts/launch/active/pull_E166_remote_results.sh
```

## 成功标准（汇总，引 §2.6 基线数字）

| 指标 | 现状（E163 baseline）| 本次目标 |
|------|------|----------|
| C-R3：jerk/acc vs staggered \|ρ\| | 未量化 | **≥ 0.6 且 > Isaac 接触的 \|ρ\|** |
| box021_035_p2 CEM 输出 jerk_P95 | 3217 | **≤ 1500**（B1/B2/A+B）|
| box004_082_p1 CEM 输出 jerk_P95 | 2956 | **≤ 1500**（B1/B2/A+B）|
| box021_035_p2 下游 staggered | 0.016 | **≥ 0.10**（A+B 臂）|
| box004_082_p1 下游 staggered | 0.094 | **≥ 0.20**（A+B 臂）|
| **box004_083_p2 下游 staggered（A 隔离）** | 0.484 | **≥ 0.60（A 臂；B1/B2 臂应≈不变）** |
| 踝主导 ee_body 失败占比（3 case）| 72–100% | **显著下降** |
| clean 3mm contact / 穿透（含 box004 接触哨兵）| E163 基线 | **不退化（≤5% 相对劣化，否则一票否决）** |
| Tier-1 红线（多维度）二分类 precision/recall | — | **均 ≥ 0.75** |

## 可视化（强制，skill §9）

- **Phase 0**：8 case 的 jerk/acc/foot-slip vs staggered 散点 + ρ 标注图；Tier-1 红线 PR 曲线。写入 log「实际观察」。
- **Phase 1/2**：CEM 输出轨迹的踝部位置/速度/jerk 时序曲线（平滑前后对比图）。
- **Phase 3**：5 臂 RL rollout 视频（远程 rerun 或离线渲染）+ `/video-frames` 抽踝部关键帧（支撑/迈步/搬箱时刻），核实脚是否不再滑/不再抖、staggered 是否真改善、接触是否保住。**禁止留空**。重点看 box004_083_p2 的 A 臂 vs B 臂对比（验证 A 隔离预测）。

## 风险与三次失败预案（skill §12）

| 风险 | 预案 |
|---|---|
| C-R3 不成立（jerk 不预测下游）| **立即停**，向用户报告，重审"脚/平滑是根因"假设；不盲目进 Phase 1 |
| 平滑/脚约束牺牲接触（E165D 重演）| 一票否决，降权重 / 改为 tie-break / 只在接触达标候选内施加；不推广 |
| 单 case 方差掩盖效应（§2.4 同物体 40×）| 必须 staggered + 3 case 因子化设计；不用单 clip 排名 |
| 无"脚好但抖"case → B 无 case 级隔离 | B1/B2 独立效应靠消融臂；box004_083_p2 反证 B 不该串味（若 B 救平滑 case 则实现有 bug）|
| valid set 被脚约束过度压缩 | jerk/accel 入 reward 不入硬 gate；只 A1/A3 用软惩罚 |

## 下一步

从 **Phase 0 E166-R3** 开始（离线红线预测力，纯分析，复用 `eval_E165_8case_handfoot_dataquality.py`）。**C-R3 通过是进入 Phase 1 的硬门**。R3 与后续无 GPU 依赖，可立即实现并跑。
