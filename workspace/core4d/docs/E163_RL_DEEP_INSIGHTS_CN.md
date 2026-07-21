# E163 下游 RL 深度分析：为什么"上游最好"≠"下游最好"

日期：2026-06-17
作者：本轮重新分析（基于原始数据复算，非复述旧报告）
旧报告：`E163_RL_DOWNSTREAM_ANALYSIS_AND_SPIDER_IMPLICATIONS_CN.md`（作为指标目录保留；本文是其上层的洞察与纠错）

---

## TL;DR（一句话洞察）

> **E163 在 SPIDER 自己的物理引擎里把"接触/穿透/tracking"刷到了最好，但这恰恰是 RL 最能自我修复的维度；真正决定下游成败的三件事——接触能否在消费端物理里被复现、参考轨迹离硬终止门的动态余量、以及任务语义(抬升高度)是否进入了选择目标——SPIDER 一个都没度量。所以"上游赢"和"下游赢"脱钩，不是偶然，是结构性的。**

这条结论我称之为 **可恢复性不对称（recoverability asymmetry）**，是本文的主线。其余所有现象都是它的推论。

---

## 0. 本文的证据基线（都已用原始数据复算）

所有数字来自本地原始产物，不是转述：

| 来源 | 路径 |
|---|---|
| RL 逐轨迹指标 | `SUGAR-private/outputs/core4d/e163_refiner_rl/*/*/eval/analysis/per_trajectory_metrics.csv` |
| 失败窗口（含 margin） | `.../eval_failed_fields_rerun_20260617_170731/analysis/failed_windows.csv` |
| Holosoma-like 聚合 | `.../e163_refiner_rl/holosoma_like_summary.csv` |
| Isaac 接触运动学探针 | `SUGAR-private/outputs/core4d_e163_threecase_contact_probe/*/isaac_contact_framewise.csv` |
| Omni 接触探针 | `.../core4d_e163_omnirt_partner_contact_probe/` |

复算环境：`/mnt/public/usr/yancilin/work_dir/.holosoma_deps/miniconda3/envs/hssim`（pandas 2.3.3）。

---

## 1. 三个被旧报告忽略 / 弄反的关键事实

旧报告把六组结果分类得没错，但它停在"分类"，没有去逼问数据。下面三件事是我复算后才看清的，每一件都改变结论。

### 1.1 所有"0/64 失败"都是**贴着硬门擦边失败**，不是崩溃

复算每个失败窗口越过 hard threshold 的幅度：

| case/source | 主因 | 失败时误差 | 阈值 | 超出量(mean / max) |
|---|---|---:|---:|---:|
| box021_r160 / omni | obj_pos (49/64) | 0.317 | 0.30 m | **+0.017 / +0.046** |
| box023_r158 / omni | obj_ori (59/64) | 0.844 | 0.80 rad | **+0.044 / +0.063** |
| box023_r158 / spider | ee_body_pos (60/64) | 0.330 | 0.30 m | **+0.030 / +0.060** |

**洞察**：这三组不是"策略跑飞了"，是"几乎做对了、在硬门边缘多越了 1.5%–10% 就被一刀切掉"。`box021/omni` 的物体只差 1.7cm 就达标。这意味着：

- 下游的 binary success 是一个**悬崖型(cliff)信号**：参考轨迹动态上"激进 5%"就能让结果从 64/64 翻到 0/64。
- 用这种悬崖信号去给上游重定向方法排名，**统计上极其脆弱**。E163 是赢是输，可能只取决于某个 case 恰好落在悬崖的哪一侧。
- 它也是好消息：这些 case 大概率能靠 curriculum / 更长 horizon / 轻微平滑参考 救回来，**不需要重做上游方法**。

### 1.2 64 个 env 几乎完全一样——"64/64"其实是 **N≈1**，而且这是 SUGAR eval 设计特有的

复算成功组的逐轨迹分布：

| case/source | hands_contact_ratio | obj_final_err (m) | duration_ratio |
|---|---|---|---|
| box004/omni | 0.597 ± **0.0045** | 0.098 ± 0.015 | 0.994 ± **0.000** |
| box004/spider | 0.540 ± **0.0029** | 0.041 ± 0.013 | 0.994 ± **0.000** |
| box021/spider | 0.717 ± **0.0042** | 0.096 ± 0.011 | 0.992 ± **0.000** |

64 个 env 行为几乎完全一致（std ≈ 0.003，duration 完全相同）。

#### 根因（代码级，已定位）

不是 RL 的通病，是 **SUGAR rollout eval 的设计**导致的：

1. **单条 motion**：每个 case 的数据集只有 `data_000` 一条 clip。
2. **全部从 frame 0 起**：rollout 模式下 `motion_id[:] = arange % num_motion`、`time_steps[:] = 0`（`mdp/commands.py:556-557`）。窗口化（`rollout_start_distance`）只在每条 rollout 完成后**串行推进**；而这些 motion 很短（116–222 帧 ≈ 一个窗口），所以每个 env 只跑一个 frame-0 整段，64 个 env 是 **64 次并行的 frame-0 整段重放**。
3. **确定性策略**：`play.py:265,302` 用 `get_inference_policy`（取分布均值，不采样）。
4. **eval 关掉了仅有的两个扰动**：`RobotRolloutPlayEnvCfg.__post_init__`（`carry_box_refiner_env_cfg.py:153-161`）把 `push_robot=None`、`push_object=None`。
5. 剩余随机化（初始位姿 ±0.1、关节 ±0.1、物体质量 0.5–2×、摩擦）确实在跑（`obj_speed_max` env 间 ptp≈0.5 证明），**但参考跟踪策略是强收敛的**，短 horizon 内把这些扰动全吸收掉 → 接触/跟踪/成功饱和成一条。

#### 对照 Holosoma：同范式、不同 eval 设计 → 连续成功率

Holosoma WBT 用**同一类**跟踪 RL，但 eval 跑法不同（`workspace/v2/scripts/eval/eval_wbt_metrics.py` + `get_eval_config`）：跑 2000 步、撞 `bad_tracking`(0.25m) 就**就地 reset 继续**、到 clip 末尾**循环重放**。本地实测它的成功率是**连续谱**（R151 等其它 case，仅用于刻画 eval 行为，非 E163 排名）：

| Holosoma run | success_rate | env间 xy_std |
|---|---|---|
| box004_082_p1 / spider | 0/64 | 0.04–0.07 |
| box004_082_p1 / omnirt | 6/64 | 0.29 |
| box021_035_p1 / spider | 21→28/64 | 0.50–0.60 |
| box021_035_p1 / omnirt | 29→38/64 | **0.67–0.76** |
| bucket004 / omnirt | 46/64 | 0.58 |

机制：64 个 env 因 startup DR 不同，在 2000 步里**于不同时刻失败/reset → 去同步(desync)**，到末步各 env 状态各异 → success 快照呈连续分布。**数据自证**：0/64 的 run xy_std 小（还同步），连续成功率的 run xy_std 大（已去同步）。而 SUGAR 是短 episodic、撞硬门即终止，没有这个去同步过程 → 二值。

#### 结论

- "64/64 成功"= **一个行为模态对 DR 鲁棒**，不是 64 个独立难例。真正有效样本量 = **3 个 case**，每个近确定性 → 用来给 E163 vs OmniRetarget 排名是**欠功效**的。
- 结合 1.1：SUGAR 的下游成功**既是悬崖型、又是近确定性**——是信息量最低的一类排名信号。
- 关键纠正：这不是"RL 评测必然如此"，而是 SUGAR eval 可改。Holosoma 的连续成功率证明同范式能产出有判别力的信号。**因此在用一套能产出连续/分布式成功率的 eval 复测 E163 之前，不能对 E163 下游优劣下结论。** SUGAR 侧的具体改动方案见 SUGAR-private `docs/SUGAR_EVAL_DETERMINISM_AND_STAGGERED_PHASE_CN.md`。

#### 实测验证（staggered-phase eval，已落地并跑出三 case）

按上述方案实现了 staggered-phase eval（64 个 env 铺到不同参考起始相位，每 env 跑一条到 motion 末尾），对三个 spider case 实测：

| case | **staggered 完成率** | 原 frame-0 二值 | 主失败模式 | init 假象? |
|---|---|---|---|---|
| box021/spider | **0.67** (43/64) | 64/64 | obj_pos(物体跟丢) | 否(失败存活 8–74 帧) |
| box004/spider | **0.48** (31/64) | 64/64 完成* | ee_body/混合 | 否(6–82 帧) |
| box023/spider | **0.00** (0/64) | 0/64 | ee_body 48/64(结构性) | 否(5–90 帧) |

\* box004 原"完成 64/64"但高度 gate 0/64(低位推送)。三 case **均无前 5 帧失败** → staggered init 有效、数字可信。

相位曲线：box021 早相位(近起点)四分位 **1.00**、中段降到 ~0.50；box004 连早相位四分位也只 **0.56**。

**这组实测把本节论点钉死了**：

1. **二值 64/64 把两个不同质量的 case 判成了并列**——box021 与 box004 原来都是"64/64 完成"，staggered 一拉开就是 **0.67 vs 0.48**。二值指标分辨率为零，连"谁更好"都分不出（这正是"64 个 env 一样、不能真实反映"的实锤）。
2. **staggered 给出诚实下游排序**：`box021(0.67) > box004(0.48) > box023(0.00)`；再叠加高度 gate，box004 实际更差（完成的 48% 无一抬到位）。
3. box023 的 0/64 **不是相位运气、也不是 eval 确定性**，而是全相位 `ee_body` 结构性失败（与 §1.3 的手贴髋自碰撞一致）。

### 1.3 box023 的"net-filter gap"被旧报告**解释反了**

旧报告说 box023 的 net-filter gap 大，意味着"filtered hand-to-Obj reward 漏看了真实接触"。复算逐帧力数据后，事实相反：

box023 运动学回放中，**前 10 帧的 net 力均值 = 1420 N**，且 net 力从 **frame 0** 就存在；而抓取/源接触窗口是 frame 28–135。Obj-filtered 接触出现在 frame 29–122——**和源接触窗口几乎完全对齐**。

| box023/spider | 值 | 含义 |
|---|---|---|
| net 力 frame 0 / 5 / 10 | 614 → **1679 → 2357 N** | 力**先长大后消失**=穿透settling |
| net 力 >0.1N 起始帧 | **frame 0** | 初始即接触 |
| net 力消失帧 | ~frame 20 | 手臂伸出去够箱子后解除 |
| Obj-filtered 接触窗口 | frame 29–122 | 与源 28–135 对齐 ✓ |

**洞察（纠错）**：box023 的真问题不是"reward 漏看接触"，而是**自碰撞穿透**。Obj-filtered 传感器其实是**对的**；多看的是 net 传感器。

**已定位到具体碰撞对**（载入 box023 参考几何 `data_000/robot_50hz.npz` 复算）：frame 0–10 两手都在 **z≈0.68（髋高）**、距箱子 0.68m（根本没碰箱子）、距地面也远；**最近的身体是同侧髋部，仅 7cm**（L_hand→L_hip=0.075m，R_hand→R_hip=0.069m）。站立/预抓取姿态手垂在胯边，SUGAR 的 rubber-hand 碰撞代理（半径 ~5–8cm）与髋/大腿几何**重叠** → 自碰撞排斥力（运动学回放里被强制保持重叠，力一路涨到 2357N），直到手臂伸出去够箱子（~frame 20）才消失。对照：box021 预抓取 net 全 0（干净）；box004 全程 net≈0（手够不到任何东西）——三种相反的接触病。

→ 旧报告"修 filter 让它捕捉更多接触"方向完全错了；正确动作是**消除手贴髋自碰撞**（上游避免手垂贴髋 / SUGAR 缩小 rubber-hand 碰撞半径或对手↔髋加 collision filter / 预抓取窗口不计）。net-filter gap 作单一标量的危险就在于：它分不清"漏看真接触"和"多看自碰撞"，二者修复方向相反。

> **E165-C 回填（2026-06-18，溯源到底是 CEM 还是源数据）**：对比 spider 与 **OmniRetarget（同人）** 两条 handoff 的 frame0-10 手↔同侧髋距离——spider **0.069m**、omni **0.073m**，**两者几乎一致、都 <0.10m**。即 OmniRetarget 原始数据**同样**手贴髋。**结论：手贴髋自碰撞继承自源人体姿态（站立手垂胯边），不是 spider CEM 引入的问题**。→ 修复落在 g1 rubber-hand proxy 几何 / 源姿态处理，**确认不动 CEM**。（实测：`results/E165/C_box023_penetration/`）

#### net 接触是否影响 RL 训练？是——但不经接触奖励

| 通道 | 是否影响 | 依据 |
|---|---|---|
| 接触奖励 `hands_contact` | ❌ 不影响 | 用 filtered `force_matrix_w`（只对 Obj），自碰撞/net 不在内 |
| `undesired_contacts` 惩罚(-1.0) | ✅ 影响 | 读全身 `net_forces_w`，惩罚集合含髋；手碰髋的力记到髋上→超 0.1N→扣分。且参考姿本身让手贴髋，躲惩罚就得偏离参考，与 tracking 打架 |
| 物理扰动 | ✅ 影响 | 机器人 `enabled_self_collisions=True`（`assets/robots/unitree.py:44,57`），每次 reset 摆到手贴髋姿→排斥力扰动初始动力学→间接诱发 ee_body 终止 |

注意:probe 的 2357N 是**运动学强制重叠**值;RL 是动态、穿透会被几帧内解开,力更瞬态。"机制成立"已确认,"贡献多大"需在 RL rollout 里按 body 记髋接触帧数才能量化。这条只打 box023(box021/box004 预抓取 net≈0)。

---

## 2. 主线洞察：可恢复性不对称（为什么上游赢≠下游赢）

把六组结果按"RL 能不能自我修复该维度的误差"重排，主线就出来了：

| 上游误差维度 | RL 能否自我修复 | 证据 | 对下游的预测力 |
|---|---|---|---|
| **接触标签/接触量** | ✅ 能 | box004/spider：参考回放接触仅 0.036，RL rollout 却自学到 **0.54** 接触并把箱子推到目标 | **弱**（SPIDER 一直在刷它）|
| **动态可行性 vs 硬门余量** | ❌ 不能 | box021/omni 接触很好(IoU 0.63)，仍因 obj_pos 越门 1.7cm 而 0/64 | **强**（SPIDER 没测）|
| **本体可执行性(末端/身体)** | ❌ 不能 | box023/spider 接触表面不差，却被 ee_body 越门 3cm 击穿 | **强**（SPIDER 没测）|
| **任务语义(抬升高度)** | ❌ 不能（不在 reward 梯度里）| box004/spider xy 进度 0.98、final err 0.041(比 omni 还好)，但 z_max 仅 0.19 < 0.34，height 0/64 | **强**（SPIDER 没测）|

**这就是答案**：SPIDER 从 E161→E163 一路在优化、并用 hard gate 死卡 RL **最能原谅**的维度（接触清洁、穿透），而真正能翻转下游成败的三个维度（**离硬门的动态余量、本体可执行性、抬升语义**）一个都没进评测。

直接证据，`box004/spider` 一个 case 同时印证四条：
- 参考接触 0.036 → 策略接触 0.54：**RL 修复了接触**，所以上游接触指标对它毫无预测力。
- final target error 0.041 比 omni 的 0.098 **还好**，但严格成功 0/64：**只看 final error 选 winner 会选错**。
- z_max 0.19 vs omni 0.54（同 case 同目标）：narrowSurfaceBand/releaseDecay 学出了**低位推/滑**而非抬举；高度从未进入选择目标，RL 也就没有抬的动机。

> 推论：E163 的"接触更干净"很可能是用**牺牲抬升高度**换来的（手贴着 surface band 不脱离 → 倾向把物体保持在低位）。这是一个**可证伪的因果假设**，见 §4 实验 B。

---

## 3. 接触不是一个问题，是**三种相反的病**

旧报告用单一 "net-filter gap" 描述所有接触异常。复算逐帧力后，三个 spider case 是三种**互相矛盾**的病，需要三种相反的修法：

| case | 病症 | 逐帧证据 | 性质 | 修法 |
|---|---|---|---|---|
| **box021/spider** ✓ | 健康 | 源接触帧 filtered recall **0.615**；非接触帧幻象力 **0.000** | 接触干净迁移 | 不用修 |
| **box004/spider** | 源接触**未被重定向复现** | 源标 frame 45–147，Isaac filtered recall 仅 **0.058**，接触帧 mean 力 **1.08N**；标签是忠实的源人体 mask（连续区间，见 §3 脚注）| 手**没贴到箱子**（embodiment gap，非标签 bug）| 修 retarget 手部贴合 / 提高 rubber-hand proxy 容错 |
| **box023/spider** | 接触力**虚假** | frame0 net 1679→2357N，非接触帧均值 602N；手↔同侧髋仅 0.069m | **手贴髋自碰撞**（继承自源姿态）| 改 proxy 半径/collision filter（**非 CEM**，见 §1.3）|

**关键**：box004 是 Isaac 里接触**太少**（recall 0.06，重定向没把手送到箱），box023 是 Isaac 里接触**太多**（2357N 幻象，但来自自碰撞）。一个标量 gap 把这两种反向病混成一种，必然误导修复方向。

> **E165-A 回填（2026-06-18，已实测）**：几何**中心距对 box004/box021 无判别力**——两者标签接触帧 wrist↔箱面距离都 ~5cm（box004 中位 0.054m、box021 0.050m），但 Isaac filtered recall 是 **0.058 vs 0.615（10×）**。即"手离箱多远"分不出健康与虚高，**消费端 recall 才是接触质量的真判别量**。这也印证杠杆1：on-rails recall 比任何离线几何距离更可靠。
>
> **§3 脚注（标签 provenance，纠正早期误判）**：E163N 五个 case 的接触标签**全是连续区间**（box004=[45,147]、box023=[28,135]…），由 `convert_core4d_e163_manifest_to_sugar.py` 从**源人体接触 mask** 派生（gap-fill+resample），**不是**距离阈值产物。故 box004 的"虚高"是**重定向未达成源接触**，不是标签生成 bug。

补充观察——**box004 的接触对 hand placement 极度敏感**：同一个 box004 资产，spider 接触 0.036、omni_partner 0.112，但最终成功的那版 omni 却有 0.571。资产是共享的，差别全在**参考里手放在哪**。box004 是 fingertip/edge 型抓取，橡胶手 proxy 对它的容错极低 → 这是一个"retarget 手姿 × 接触 proxy"的交互问题，不是纯资产 bug。

---

## 4. 对 SPIDER 评测/算法的建设性意见（按杠杆排序）

不再给"指标清单"，而是给**最高杠杆的单点改动 + 可证伪实验**。

### 杠杆 1（最高）：把接触放到**消费端物理**里打分，而不是 SPIDER 自评

SPIDER 现在的接触/穿透是在它自己的 MuJoCo-Warp 里、用自己的 contact model 自评的——这是典型的 **Goodhart**：优化的指标和评判的模型是同一个，数字能涨而物理接触不可迁移。本轮唯一**真正预测了下游成败**的接触信号，是 **Isaac 运动学回放探针**（policy-free，只回放参考位姿）：

**E165-E1 实测三标量（2026-06-18，物体 on-rails）**：

| case | filtered recall | phantom force rate | max init net force | 病型 | 下游 staggered |
|---|---|---|---|---|---|
| box021 | 0.615 | 0.00 | 0 N | 干净迁移 | 0.67 |
| box004 | 0.058 | 0.197 | 0 N | 源接触未复现 | 0.48 |
| box023 | 0.667 | 0.614 | **2459 N** | 自碰撞穿透 | 0.00 |

→ **建议**：把这个 policy-free 的 Isaac 运动学探针做成 SPIDER handoff 前的**必跑闸**。它便宜（无需训练）、用消费端物理、且已被证明能在 RL 之前就把三个 case 分开。报**三个正交标量、别合成一个 gap**：
- `filtered_contact_recall`（源接触帧里 Isaac filtered 命中率）→ 抓 box004 型"源接触未复现"
- `phantom_force_rate`（非接触帧 net 力 >阈值占比）+ `max_init_net_force`（首 N 帧 net 力）→ 抓 box023 型"穿透/自碰撞"

**关键纠正（E165-E1 实测）**：**单标量不预测下游**——box023 的 recall 高达 **0.667 却 0/64**（败在自碰撞而非接触缺失），recall 单序 ≠ 下游序。必须**三标量联合分病**，这正是反对"合成单一 net-filter gap"的实锤。

**范围界定（重要，不要扩张这个杠杆）**：探针里物体保持 **on-rails（按 ref 驱动，与 CEM 同口径）**，**只量接触几何**（recall/phantom/init-net）。它**故意不**释放物体、**不**考虑 free-joint 的 z/承重——那类"抓握能否真的托住/抬起"的失败（如 box004 的低位推）由**杠杆3 在 RL 侧**负责。两个杠杆职责正交、互不合并：杠杆1 = on-rails 接触几何 preflight，杠杆3 = RL 抬升 reward。

### 杠杆 2：选择目标里加入**离硬门的动态余量**，而不只是均值误差

§1.1 证明失败都是擦边越门。SPIDER 的 CEM 现在按均值 tracking/contact 选 winner，但下游用的是**逐帧 hard gate**。两者错位。

→ **建议**：CEM rerank 的目标从"均值误差小"改成"**最坏帧离门的余量大**"。即对 obj_pos / obj_ori / ee_body / anchor 各算参考轨迹自身的 peak deviation，惩罚接近下游阈值的峰值。这能直接预防 box021/box023 的擦边失败，且无需 Isaac，离线参考诊断即可近似。

### 杠杆 3：抬升语义必须进 **RL reward**（不能进 CEM selection，因 CEM 里物体是 GT）

box004/spider 的 height 0/64 不是 RL 的错，是**抬升信号从未进入 RL 的梯度**。

**关键约束（本轮代码确认，`spider/config.py:153-162`）**：CEM 里物体是 **GT**——`object_pd_override`(kp=2000 强 PD 跟 ref) / `object_kinematic_override`(freejoint 直接写插值 ref，完全上轨) / E015 软 weld 三选一恒成立，物体始终被拽着 follow GT 轨迹，机器人不需要真正承重。因此**每条 CEM rollout 的物体 z 都完美等于参考**，CEM selection 永远看不到"抬不起来"——在 CEM 里加 `z_max` 约束是**恒满分的空操作**。抬升只有在物体**自由**（消费端 RL，有重力）时才会失败。

→ **建议（修正）**：抬升语义作为 **RL reward/约束**（z-tracking 或 lift bonus），**不**进 CEM selection。CEM 侧唯一相关的动作是**不要只用 final target error 选 winner**（避免选出"低位滑到位"的解），但抬不抬最终由 RL reward 决定。

### 可证伪实验（小而精，别扩量）

| 实验 | 假设 | 操作 | 成功判据 | 预测 |
|---|---|---|---|---|
| **A. box004 接触标签审计** | box004 接触标签虚高源于 retarget 手够不到箱 | 对 box004/spider 跑 hand-obj 表面距离逐帧统计，对比接触标签 | 若标签接触帧的手-面距离 > proxy 半径 → 标签 fiction 坐实 | 大概率坐实；指向重算标签阈值或修手姿 |
| **B. height reward 进 RL** | narrowSurfaceBand 用低位换了接触清洁 | 对 box004/spider 在 **RL** 加 z-tracking/lift reward（**非** CEM rerank，因物体 GT 恒满分），重训 | height 从 0/64 → >0，且 contact/pen 不显著退化 | 若成立 → 证明 §2 的因果假设 |
| **C. box023 初始穿透修复** | box023 失败含初始帧穿透 | 查 box023 初始位姿/场景资产，消除首帧 1400N net 力后重跑探针 | 首帧 net 力 → ~0；再训 RL 看 ee_body 越门是否缓解 | net 力可消；RL 是否救回需验证（擦边 3cm，可能需叠加杠杆 2）|
| **D. 动态余量 rerank** | 擦边失败可由 peak-deviation rerank 预防 | box021/omni + box023 加 peak-margin 惩罚重选/平滑 | 擦边 case 的 peak deviation 拉开离门距离，RL 翻正 | box021/omni 仅差 1.7cm，最可能先翻正 |

---

## 5. 待排查的 Bug / 工程隐患（按严重度）

1. **box023 初始帧 1400N 穿透**（最严重）：运动学回放首帧即对非 Obj 物体施 1420N。这是 scene/init-pose/资产穿透 bug，不是 reward 问题。必须在 handoff 前的 preflight 拦截（首帧 net 力阈值检查）。
2. **box004 接触标签与 Isaac 几何不一致**（5× 虚高）：`contact_labels_50hz.npy` 的接触判据与消费端橡胶手 proxy 不一致。需统一接触定义，或至少在 eval 报告 filtered_contact_recall。
3. **64-env eval 近确定性**（std≈0.003，duration 完全相同）：要么 env 随机化没生效（潜在 bug），要么 eval 本就确定性。无论哪种，**把 64/64 当 64 个独立成功上报是误导**，应注明有效样本 = case 数。
4. **硬门悬崖 + 近确定性 ⇒ 下游 binary success 不适合做上游方法 ranker**（方法论隐患）：当前用它给 E16x 排名，统计不成立。应改报"离门余量"等连续量。
5. **net-filter gap 单标量会误导修复方向**（见 §1.3）：必须拆成 recall（漏看真接触）和 phantom（多看假接触）两个正交量。
6. **远程 params 绝对路径 / box004 资产缺失 preflight**（旧报告 2.3/2.4 已记，仍有效）：保留。

---

## 6. 对"E163 到底好不好"的最终判断

- E163 **作为上游 CEM/retarget 候选是有效的**：它修了 E161 在 box023 的 raw contact 回退，并在 box021 上转化成了干净的下游成功。
- 但**不能**把 E163 升级为默认 RL-safe 方法，也**不能**因为 box004/box023 失败就降级它——因为：
  1. 当前下游信号是**悬崖型 + 近确定性 + n=3**，统计上撑不起任何方向性排名（§1.1、§1.2）；
  2. 两个失败 case 的真因（box023 初始穿透、box004 抬升缺失）**不是 E163 接触方法本身的问题**，是评测没覆盖的维度和工程 bug（§3、§5）。
- **正确的下一步不是继续刷接触**，而是：把接触挪到消费端物理打分（杠杆1）、把硬门余量和抬升语义放进选择目标（杠杆2/3），然后用实验 A–D 做小闭环验证。

> 一句话：E163 没有"不行"，是**评测的分辨率不够、且混进了两个工程 bug**。把这两件事修好，再谈 E163 在下游的真实排名。
</content>
</invoke>
