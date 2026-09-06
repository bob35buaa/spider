# plan241 · E211：desk007 gravcomp 强度救援（部分补偿 → 抓握重锚）

_承接 [E209/log298](workspace/core4d/log/298_E209_desk_chair_prg_g1_gravcomp.md)（R295, FAIL）与 [E210/log299](workspace/core4d/log/299_E210_bucket007_aug_g1only_gravcomp.md)（R296）· 分支 `feat/E207-bucket-g1only-gravcomp` · 拟用 log300 / plan241_

---

## Context

E209 把 `gravcomp=1` 单变量加到 E206 PRG 的 22 个 desk/chair case 上。物体侧全线收益（z bias −2.517→+1.041 cm，obj_pos −1.16 cm），但 14-gate 主门 C3 破（narrow 13→10、L1 3→11），判定 FAIL。

按物体拆开后，退化**几乎全部集中在 desk007 这 5 例**——chair006 是唯一每一项都改善的物体，desk007 是唯一每一项跟踪指标都恶化的物体：

| 物体 | n | narrow PRG→G1 | eef_ori | root_ori | contact | release |
|---|--:|---|---|---|---|---|
| chair006 | 5 | 3 → **5** | 12.93→11.75 | 5.23→5.44 | .892→**.907** | .121→**.037** |
| desk021 | 7 | 4 → 4 | 17.60→17.06 | 9.04→8.54 | .861→.877 | .020→.040 |
| desk023 | 4 | 4 → 1 | 16.96→21.28 | 8.01→9.47 | .910→.921 | .042→.054 |
| **desk007** | **5** | **2 → 0** | **14.67→25.62** | **6.98→14.79** | **.880→.764** | **.179→.307** |

本计划的目标：**只对 desk007 这 5 例**，在保住物体高度收益（z bias 归零、obj_pos/obj_ori 改善）的前提下，把 eef_ori / contact / release 拉回 PRG 附近。

---

## 一、诊断（已完成，本计划的依据）

### D1 · 物体是伺服驱动体，−z bias 是伺服下垂

`scene_act` 把物体的 freejoint 换成 3 slide + 3 hinge，各配 position actuator（运行时 `init_pos_actuator_gain=500`、`init_rot_actuator_gain=50`，见 `E209/audit_runtime_contract.py:64-68`）。−z bias = 未被手承担那份重力 / kp。`gravcomp` 抹掉重力项 → 下垂消失。这解释 E209 的收缩型传递函数。

MuJoCo 侧 `gravcomp` 是**浮点乘子**，不是布尔：`mujoco_warp/_src/passive.py:264-272` 里 `force = -gravity * body_mass * gravcomp`。部分补偿在运行时完全被支持，无需改 sim 代码。

### D2 · desk007 的「握」本身就是重力（机制假设）

抓握拓扑在 desk007 / chair006 之间零重叠（5/5 vs 5/5）：

| | desk007 | chair006 |
|---|---|---|
| 被握几何件第二维 | **5.2–8.1 cm** | **20.3–21.6 cm** |
| 被握件体积 | 1.2–2.1 L | 3.9–11.9 L |
| 握法 | 双手钩同一根桌沿细杆（`coarse_007..011`，45×5×5 cm 杆） | 两手分扣扶手 / 靠背（异件，形封闭） |
| COM 相对手 | 低 0.30 m、外伸 0.42 m ⇒ 19–22 N·m | 低 0.24 m、外伸 0.36 m |
| 参考躯干俯仰 | **全部前倾** +0.34…+4.76° | **全部后仰** −1.59…−3.72° |

desk007 是单向法向力钩握：掌面托不住任何几何，只因桌重把细杆压进掌心才成立；前倾躯干在配平那 20 N·m。chair006 是形封闭，与载荷无关。

补充事实：CORE4D 是**双人共抬**，场景里只编译了一个 G1 + 物体（31 bodies，无第二 agent）。物体 COM 落在单机器人抓握线外 0.32–0.44 m —— 伺服本来就在替缺席同伴出力。`gravcomp=1` 等于让伺服出 **100%**，而同伴份额只该是 50–70%。

### D3 · 卸载后总奖励净下降 —— 不是「换」，是问题变难了

精英样本逐项奖励（末次 CEM 迭代，逐帧均值，各组 5 例平均；直接从两臂 `trajectory_mjwp_act.npz` 读出）：

| 分项 | desk007 PRG→G1 | chair006 PRG→G1 |
|---|---:|---:|
| `contact_hdmi_rew` | 3.297 → 3.150 (**−0.147**) | 3.077 → 3.121 (+0.045) |
| `qpos_rew`(local_frame) | 2.808 → 2.696 (**−0.112**) | 2.904 → 2.910 (+0.006) |
| `surface_band_rew` | 0.864 → 0.840 (−0.025) | 0.706 → 0.733 (+0.027) |
| `task_obj_rew` | 0.636 → 0.651 (+0.014) | 0.614 → 0.651 (+0.037) |
| `object_lift_rew` | 0.331 → 0.403 (+0.073) | 0.039 → 0.043 (+0.004) |
| **总 `rew`** | 7.702 → **7.505 (−0.197)** | 6.771 → **6.902 (+0.131)** |

**desk007 上总奖励净降**。这修正了 log298 §4 的「CEM 把优化预算让渡出去」读法——不是主动权衡，是接触项失去物理锚点后 CEM 找到的解本身更差。

### D4 · 四个退化指标是同一原因的四个投影

- `contact` .880→.764：法向力→0，掌面与细杆只共动、不压紧，MuJoCo 接触事件真消失。
- `eef_ori` 14.7→25.6°：`contact_hdmi_ori`（掌法线·指向目标，weight 0.3）在无载荷时大量腕姿等价。**代码里没有 eef 朝向权重**——`local_frame_wrist_weight` 只作用于位置（`mjwp.py:1258-1276`），`local_frame_ori_sigma` 与下肢共用。
- `root_ori` 6.98→14.79°：配平 20 N·m 的前倾不再必要。
- `release` .179→.307（030_p2 .130→**.826**）：末段物体停在参考高度而非沉下去，手在 trailing 窗口持续蹭到它。PRG 的「干净松手」是下垂帮的忙。

### D5 · desk007 独有的放大器：CEM 安全门回退

desk007 是唯一 CEM 门真正咬住的物体族。`cem_gate_fallback_used`：chair006 / desk021 恰好 **0.0000**，desk007 = 0.064(PRG) → 0.076(G1)；`cem_body_gate_valid_frac` 0.942→0.897；028_p1 单例 0.925→**0.731**、回退帧 2→15。回退策略 `cem_safety_gate_fallback: least_violation`（`max_violation_pct=0.0` 零容忍），一旦触发**完全丢弃奖励排序**、只按穿透深度选。

但 034_p1（最坏例）回退反而变少（19→7），所以这是**放大器不是主因**。本计划不动门，只在报表里跟踪。

### D6 · 逐例 g* 高度一致

`g*_i = -pre_bias_i / delta_i`（delta 为 g=1 的实测修正量）：

| case | pre_bias | delta(g=1) | g* |
|---|--:|--:|--:|
| 028_p1 | −3.653 | +6.059 | 0.603 |
| 028_p2 | −2.736 | +2.823 | 0.969 |
| 030_p2 | −1.512 | +2.049 | 0.738 |
| 032_p2 | −2.557 | +3.600 | 0.710 |
| 034_p1 | −3.461 | +4.969 | 0.697 |

中位 **0.710**、均值 0.743。与 E209 全局收缩拟合的外推 g*≈0.71 独立吻合，也与「同伴份额 50–70%」的物理解释吻合。

---

## 二、实验设计 E211

**范围**：desk007 × 5 case（`20231030_028_p1/028_p2/030_p2/032_p2/034_p1`）。不加其它物体（已确认）。
**g 粒度**：物体级单值。逐例 g* 只作为**预测-实测证伪检查**，不作调参旋钮。

### Stage A — 部分补偿单变量扫描

新 arm `G07/G06/G04`：`<body name="object" gravcomp="0.8|0.6|0.4">`，其余与 E206 PRG 逐字节相同。

- 5 case × 3 档 = **15 条** CEM（1024×32 seed 0）。
- 与两条**冻结不重跑**的基线拼成 5 点曲线：g ∈ {0（E206 PRG）, 0.4, 0.6, 0.8, 1.0（E209 G1）}。
- **双机执行**：本机 8 卡跑 8 条 + 另一台 8 卡机跑 7 条，各一波并行，wall ≈ 单条时长（E209 desk/chair median 42.2 min）。详见 §四之二。

**预注册预测（可证伪，P0 冻结）**：

| # | 预测 | 证伪条件 |
|---|---|---|
| A-P1 | `z_bias(g) ≈ −2.784 + 3.900·g`（线性），5 点 OLS R² ≥ 0.90 | R² < 0.90 ⇒ 补偿力非线性传递，收缩模型不适用于中间档 |
| A-P2 | `contact(g)` 随 g **单调下降** | 非单调 ⇒ 「载荷即抓握」机制假设被证伪 |
| A-P3 | `eef_ori(g)` 随 g 单调上升 | 非单调 ⇒ 同上 |
| A-P4 | 存在 g 使 z 与机器人侧同时达标（见 §三） | 全档不达标 ⇒ 单靠 g 救不回，进 Stage B |

### Stage B — 抓握重锚（仅当 Stage A 未达标时）

用 Stage A 胜出的 `g*`，叠加 `hand_support_rew_scale ∈ {1.5, 3.0}` → **10 条**。

选这个项的理由：`hand_support_rew = scale · exp(−max(|sdf|−0.01, 0)/0.015) · gate`（`mjwp.py:2144-2182`）。当前被 E163 显式置 0，由 `surface_band`（width 3 mm、sigma 1.5 mm）接管。但 surface_band 在 2 cm 外已彻底饱和到 ~0（`exp(−0.02/0.0015)≈0`），**正好是 desk007 失载后手漂开的量级**；hand_support 在 2 cm 处仍给 `exp(−0.01/0.015)=0.51`，是唯一的长程召回项。3.0 是 E120/E121/E151/E156 的历史取值。

**Stage B 打破 E209 的 `ALLOWED_DIFF={"scene_name"}` 单变量契约**，必须显式扩键并新写审计（见 §四 P3）。

### Stage C — 契约兜底（仅当 Stage B 仍差 eef_ori）

`contact_hdmi_ori_weight: 0.3 → 0.5`，5 条。这是**唯一**能在 config 层碰 eef 朝向的旋钮，且它测的是掌法线点积、不是 gate 测的腕系测地角，只是弱代理。若 C 也不成，结论写成「eef_ori 需要 core 层新增权重项」，登记为缺口而非继续调。

---

## 三、评判标准（P0 冻结，不得事后改）

desk007 n=5 的两条基线（本计划已从 `e209_two_arm_rollout.tsv` / `e209_object_z_diff_by_case.tsv` 精确算出）：

| 指标 | 方向 | **PRG (g=0)** | **G1 (g=1)** |
|---|---|--:|--:|
| z_bias 宏平均 (cm) | →0 | −2.784 | +1.116 |
| mean\|z_bias\| (cm) | ↓ | 2.784 | 1.116 |
| z_mae (cm) | ↓ | 3.595 | 2.099 |
| obj_pos (cm) | ↓ | 9.078 | 8.120 |
| obj_ori (°) | ↓ | 5.310 | 4.828 |
| **eef_ori (°)** | ↓ | **14.671** | 25.622 |
| **contact** | ↑ | **0.8799** | 0.7644 |
| **release** (n=4) | ↓ | **0.1793** | 0.3065 |
| root_ori (°) | ↓ | 6.984 | 14.792 |
| root_pos (cm) | ↓ | 13.642 | 17.544 |
| eef_pos (cm) | ↓ | 12.609 | 14.504 |
| hand_pen | ↓ | 0.2540 | 0.2147 |
| body_z p95 (m) | ↓ | 0.0560 | 0.0800 |
| leg_pen | ↓ | 0.0318 | 0.0315 |
| ankle_jerk p95 | ↓ | 542.5 | 515.0 |
| **narrow** | ↑ | **2/5** | 0/5 |

### 主门 C1（必须全过才算 SUCCESS）

| 子句 | 判据 |
|---|---|
| C1a 高度收益保住 | mean\|z_bias\| ≤ **1.50 cm**（介于 G1 的 1.116 与 PRG 的 2.784 之间，取 G1 侧） ∧ z_mae ≤ **2.80 cm** |
| C1b 物体跟踪不退 | obj_pos ≤ **9.078**（PRG 值） ∧ obj_ori ≤ **5.310** |
| C1c **eef_ori** | ≤ **15.67°**（PRG + 1.0°） |
| C1d **contact** | ≥ **0.850**（PRG − 0.030） |
| C1e **release** | ≤ **0.229**（PRG + 0.050，同一 n=4 配对） |
| C1f narrow 不劣化 | ≥ **2/5** ∧ hard 5/5 |

### 副门（记录，不单独判 FAIL，但任一破必须在 log 里显式点名）

- C2 root_ori ≤ 7.98°（PRG+1.0）、root_pos ≤ 14.64 cm（PRG+1.0）、eef_pos ≤ 13.61 cm（PRG+1.0）
- C3 hand_pen ≤ 0.284（PRG+0.03）、body_z p95 ≤ 0.20（硬门）、leg_pen ≤ 0.20（硬门）、ankle_jerk < 1000（硬门）
- C4 吞吐：median wall ∈ [33, 55] min（E209 desk/chair 是 42.2）

### 反挑拣条款（rules §5）

- **全 5 例报表**，禁止只报最好的档或最好的 case；每个指标报 mean + std + worst-case。
- 逐例 g* 的预测-实测比对**只用于证伪 A-P1**，不得据此给每例挑不同的 g。
- Stage A 三档全部评测，即便某档明显更差也必须进表（否则 A-P2/A-P3 的单调性判据无意义）。
- Stage B 若开，必须在 **g\* 固定**的前提下比较，不得同时动 g。

### 视觉复核（rules §5，强制）

对胜出档 5/5 渲染并抽帧。**必须遵守 E209 F6 / E210 F3 的机位陷阱**：`_auto_video_camera`（`spider/viewers/__init__.py:262-291`）每帧用 sim∪ref 并集包围盒算 lookat/半径，sim 不同 ⇒ 机位不同 ⇒ **跨视频比姿态无效，连同一份参考都会渲成两个姿势**。唯一有效判据 = **同一视频内 sim vs ref**。抽帧位置必须先用逐帧数值定位分歧峰值再抽（E210 F4：按固定比例抽帧什么都看不出来）。

重点看：手是否仍钩在桌沿细杆上、腕姿与躯干前倾是否跟得上参考、末段是否完成放下并起身。

---

## 四、实施步骤

复用 E209/E210 已有骨架，尽量只写差异。

| P | 内容 | 关键文件 |
|---|---|---|
| **P0** | 编号 log300 / plan241；`e211_common.py`（照 `E209/e209_common.py` 改：`CASES` 只留 desk007 5 例、`ARMS = {"G08":0.8,"G06":0.6,"G04":0.4}`、`SCENE` 按 arm 派生）；冻结两条基线（从 E209 产物拷 z / 14-gate，**不重跑**） | 新 `workspace/core4d/scripts/experiments/E211/e211_common.py`；读 `E209/e209_common.py:300-301`（PRG 产物路径）、`E206/e206_common.py` |
| **P1** | **参数化的 gravcomp 断言**。`e200_common.assert_gravcomp_diff` 在 `:143` 只接受 base ∈ {None,"0","0.0"}、在 `:146` 硬编码 `"1"` —— `gravcomp="0.6"` 会被判 AssertionError。新写 `assert_gravcomp_diff_value(base, sidecar, value)`，**复用 `e200_common._signature`（:129-137）不重写**；`e200_common` 只读（E198/E200/E209/E210 都 import 它） | 新 `E211/e211_common.py`；参照 `E210/build_gravcomp_sidecars.py:57-79` 的**篡改样本反向自测**模板 |
| **P2** | 建 3×5=15 个 sidecar `scene_act_E211_lowgeom_PRG_gc{04,06,08}.xml`。保留 E209 的三道守卫：`gravcomp` token 恰好 1 次、MuJoCo 编译后 `ngeom/npair/nq/nv/nu/nbody` 与 base 全等、sidecar sha256 互不相同且不等于 base | 照 `E209/build_scenes.py:35-96` |
| **P3** | override：Stage A 仍是 1-key `scene_name`，`ALLOWED_DIFF={"scene_name"}` 不变。**Stage B 需扩为 `{"scene_name","hand_support_rew_scale","hand_support_geom_ids"}`**（`hand_support_geom_ids` 是 `config.py:1352-1362` 从 names 派生出来的，会跟着变，必须显式列入否则 compose diff 会误报） | 照 `E209/build_overrides.py:38,66-115` |
| **P4** | manifest + 输入 sha256 pin（轨迹 / 3cm 掩码 / base scene 必须与 E206 交付表逐条相等）。**同时产出 shardA / shardB 两份分片 manifest**（见 §四之二） | 照 `E209/build_manifest.py` |
| **P5** | **场景快照（rules §7 保障 2）**：`results/E211/scene_snapshot/` 覆盖 5 个 dcv3 task dir 全部 XML + `manifest.txt`（git HEAD + 每文件 sha256）。同时 `git add -f` 15 个新 sidecar。**Stage B 若改 XML 需重拍** | `workspace/core4d/scripts/convert/snapshot_scenes.sh` |
| **P6** | smoke 1 条（64×4）+ 运行时契约审计：`config_act.yaml` 全键 diff 仅 THE_VARIABLE；A0 hand-gate (0.10/−0.020)、`init_pos_actuator_gain=500`、`leg_object_penalty_scale=2.0`、`cem_leg_gate_enabled=true`、object `gravcomp` == 本档值 | 照 `E209/audit_runtime_contract.py:57-70`（`REQUIRED_RUNTIME` 加 gravcomp 值） |
| **P7** | Stage A full 15 条，**双机 8+7 并行**（§四之二）。**队列用 E199 版**（`E200/e200_common.TIER_RANK` 无 `"P0"` 会 KeyError，E209 F2）。**中断后必须先跑 `E210/reset_stale_rows.py`**（`run_local_priority_queue.py:30` 的 `ELIGIBLE` 不含 `running`，被打断的行会变墓碑并静默报 0 pending，E210 F1）——双机时**只对自己那份 shard 跑**，不要碰对方的 | `E199/run_local_priority_queue.py` + `E210/reset_stale_rows.py` |
| **P8** | 评测：5 档 × 5 例，14-gate 全门（`HARD_GATES + BANDED_GATES`，不能只写 `BANDED_GATES + body_z`，E209 F8）+ z 诊断 + A-P1..A-P4 判据。**逐指标方向表**（`contact` 越高越好，用统一 `Δ>0 = 变差` 会翻转结论，E210 F6）。**PRG 基线重打分必须逐位复现 E209 的 2/5 narrow**，否则 evaluator 漂移，先停 | 照 `workspace/core4d/scripts/eval/runners/eval_E209_g1_gravcomp.py:229-236`；门定义 `E201/funnel_config.py:32-51` |
| **P9** | 渲染 + 抽帧（osmesa）；按 §三 视觉口径复核 | 照 `E209/render_cem_results.py` |
| **P10** | 若 Stage A 未过 C1 → Stage B（P2'–P9' 同链路，只换 arm 定义）。Stage B 仍不过 → Stage C。三档都不过则如实判 FAIL 并登记 core 层缺口 | |
| **P11** | 写 log300、更新 `EXPERIMENT_TRACKER.md`（R297）、`progress.md`；commit 时**显式列路径**（E209 记过：并发提交者会把 `git add -f` 的文件扫进不相干 commit） | |

---

## 四之二、双机执行（Stage A 15 条 = 8 + 7）

### 为什么必须分片 manifest，不能两台机跑同一份

`E199/run_local_priority_queue.py` 每次状态变化都 `C.write_tsv(manifest, rows, fields)` **整文件重写**（:216、:230 等）。两台机器挂同一个 `/mnt`、指向同一份 manifest ⇒ 后写者用自己内存里的旧快照覆盖对方刚写的状态，双方都会把对方的行重新认领一遍或永久漏掉。这不是理论风险——E210 F1 已经在单机双队列上炸过一次（`ELIGIBLE` 不含 `running`，被打断的行变墓碑）。

处置：**P4 直接产出两份互不相交的分片 manifest**，各机只读写自己那份。CEM 输出目录按 `case × arm` 天然互不重叠，日志同理，所以除 manifest 外没有共享可写状态。

### 分片规则（确定性，写进 P4 并记入 log）

15 行按 `(case_id, arm)` 字典序排定后按索引奇偶轮转：

- `shardA` = 偶数索引 8 行 → **本机**
- `shardB` = 奇数索引 7 行 → **另一台机**

轮转而不是切两段，是为了让两片各自覆盖全部 5 个 case 和全部 3 个 g 档——万一一台机中途挂掉，剩下那片仍是一个有意义的、跨 case 跨档的子集，而不是「只有 028 系列」这种没法读的残局。

产物：
```
workspace/core4d/results/E211/s6_downstream/manifests/
  e211_stageA_full_manifest.tsv          # 15 行，仅供 P8 评测与审计对账，队列不写它
  e211_stageA_full_manifest.shardA.tsv   # 8 行，本机
  e211_stageA_full_manifest.shardB.tsv   # 7 行，另一台机
```
`build_manifest.py` 必须断言：两片行数 8/7、`case_id×arm` 键集合无交集、并集恰等于 15 行主表，且三份文件的输入 sha256 列逐行相等。

### 快照只拍一次（rules §7）

场景快照写共享路径 `results/E211/scene_snapshot/`，两台机同时拍会互相覆盖。约定：**本机（shardA）负责拍**，另一台机用 `SKIP_SNAPSHOT=1`，并由 `run_E211_local_8gpu.sh` 在跳过时**断言** `scene_snapshot/manifest.txt` 已存在且其中记录的 git HEAD == 当前 HEAD，不一致直接退出（避免另一台机在还没冻结场景、或场景已被改动的状态下开跑）。

### 命令

前置：P0–P6 已完成（sidecar / override / manifest / 快照 / smoke 全过），两台机器都能看到同一个 `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider`。

**本机（8 卡，8 条）**
```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
SHARD=A bash workspace/core4d/scripts/launch/active/run_E211_local_8gpu.sh
```

**另一台 8 卡机（7 条）—— 这条发给对面**
```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
SHARD=B SKIP_SNAPSHOT=1 bash workspace/core4d/scripts/launch/active/run_E211_local_8gpu.sh
```

先干跑确认认领的是自己那 7 条、且输入 sha 全部对得上：
```bash
SHARD=B DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E211_local_8gpu.sh
# 期望输出: [dry-run] 7 pending across gpus=['0'..'7'] mem>=5000MiB
```

`run_E211_local_8gpu.sh` 照 `run_E209_local_8gpu.sh` 改，新增 `SHARD`（A|B，默认 A）选分片文件，其余 env 旋钮沿用：`GPUS`（默认 `0,1,2,3,4,5,6,7`）、`PER_GPU_MEM_MIB`（5000）、`MAX_PER_GPU`（1）、`POLL_INTERVAL`（15）、`DRY_RUN`、`SKIP_SNAPSHOT`、`STAGE`（full|smoke）。脚本内已 `export TORCHDYNAMO_DISABLE=1`（本机无 python3.12-dev，triton JIT 会炸）和 `MUJOCO_GL=disable`（CEM 无头）。

若另一台机卡型/显存不同，用 `PER_GPU_MEM_MIB=<实测空闲下限>` 覆盖；队列只在空闲显存达标时派发，不会抢占别人的进程。

### 合流与对账（进 P8 之前必须做）

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E211/merge_shards.py --stage A
```
按 `(case_id, arm)` 主键把两片状态并回主表，并断言：
1. 15/15 `status == run_complete_pending_eval`，`problem rows: []`；
2. 两片没有同一主键的重复行（防止有人误在两台机上跑了同一份 shard）；
3. 每条产物 `config_act.yaml` 的 `scene_name` 与该行 `arm` 对应的 gravcomp 档一致（防止分片错配）；
4. 记录每行落在哪台机（`hostname`）与哪张卡，写进 manifest 的 `host` / `gpu` 列——**双机跑必须能回答「这条是在哪跑的」**，否则出了硬件相关差异无从追查。

对账不过就不许进 P8。

---

## 五、已知陷阱清单（从 E209/E210 继承，实施时逐条对照）

| 来源 | 陷阱 | 本计划的处置 |
|---|---|---|
| E209 F1 | `e200_common.build_gravcomp_sidecar` 输出名硬编码 E199 | 自写 writer，只复用 `_signature` |
| E209 F2 | `e200_common.TIER_RANK` 无 `"P0"` | 队列用 E199 版 |
| E209 F3 | 给 `e206_common.ARMS` 加 arm 会污染在跑的 E208 | `e206_common` / `e209_common` 全部只读 |
| E209 F5 | 运行时契约 NaN≠NaN 自比不等 | 用 `same()` 处理 NaN |
| E209 F6 / E210 F3 | 跨视频比姿态无效（auto camera 随 sim 变） | 只做同视频内 sim vs ref |
| E209 F8 | 逐门表漏 3 个硬门；`fall_flag` 布尔被 `_finite` 变 NaN | 14 门全列 + `num()` 强转 |
| E210 F1 | 队列对中断不 resume-safe | 派发前后跑 `reset_stale_rows.py` |
| E210 F2 | **代价有两种互斥形态**（接触塌陷 / 末段姿态崩溃），单一指标必漏一种 | C1c(eef_ori) 与 C1d(contact) **两条都留**，任一破即 C1 不过 |
| E210 F4 | 按固定比例抽帧看不出差异 | 先用逐帧数值定位峰值再抽 |
| E210 F6 | `contact` 方向相反；14-gate 的 contact 门与 RL-export 的 `_3mm_` 是两个字段 | 逐指标方向表 + 单一实现 |
| 本计划新增 | `assert_gravcomp_diff` 硬编码 `"1"`，部分补偿会被拒 | P1 新写参数化断言 + 篡改样本反向自测 |
| 本计划新增 | Stage B 的 `hand_support_geom_ids` 是派生键，会让 compose diff 误报 | 显式列入 `ALLOWED_DIFF` |
| 本计划新增 | 双机指同一份 manifest ⇒ 整文件重写互相覆盖状态 | 分片成 shardA/shardB，各机只写自己那份（§四之二） |
| 本计划新增 | 双机同时拍场景快照会互相覆盖 | shardA 拍，shardB `SKIP_SNAPSHOT=1` + 断言快照存在且 git HEAD 一致 |
| 本计划新增 | 跨机器硬件差异无法追查 | manifest 记 `host` / `gpu` 列，合流时对账 |

---

## 六、验证方式

1. **契约层（不跑 CEM 即可验）**：`python E211/build_scenes.py --dry-run` → 15/15 过参数化断言 + 编译不变量；篡改样本自测应报 AssertionError。`python E211/build_overrides.py --audit` → compose 全键 diff 恰为 `{scene_name}`，15/15。
2. **smoke**：1 条 64×4，`audit_runtime_contract.py` 逐项断言通过，其中 object `gravcomp` 读数 == 该档值（这是唯一能证明部分补偿真的进了 sim 的检查）。
3. **分片对账**：`SHARD=A/B DRY_RUN=1` 各自应报 8 / 7 pending 且两边主键无交集；跑完 `merge_shards.py --stage A` 必须 15/15 `run_complete_pending_eval`、无重复主键、`scene_name` 与 arm 逐行匹配。
4. **基线复现**：重打分 E206 PRG 的 desk007 5 例，必须逐位得到 narrow 2/5、eef_ori 14.671、contact 0.8799。不符即停。
5. **主判据**：§三 C1 六条子句 + A-P1..A-P4 四条预注册预测，全部落到 `results/E211/s6_downstream/eval/` 的 TSV/JSON。
6. **视觉**：胜出档 5/5 渲染，按 §三 口径抽帧复核，结论写进 log300。

---

## 七、预期结果与判读

- **最可能**：g=0.6 或 0.8 落在 g*≈0.71 两侧，z 达 C1a/C1b，contact/eef_ori 部分恢复但未必达 C1c/C1d → 进 Stage B。
- **若 A-P2/A-P3 非单调**：「载荷即抓握」假设被证伪，说明 desk007 的退化另有原因（最可能是 D5 的 CEM 门回退），此时应转向门参数而非补偿强度。
- **若三个 Stage 都不过**：如实判 FAIL。结论是 gravcomp 这条路线对**细杆钩握型抓握**不适用，需要在 core 层补两件东西：(a) 一条 hand-object 相对姿态门（E209 已提），(b) 一个真正的 eef 朝向权重（`local_frame_wrist_weight` 目前只管位置）。

注：desk007 与「细杆钩握」在数据里 100% 共线（n=5 vs n=5、每组一个物体），本计划**不做**打破混淆的对照。因此 §一 D2 的机制假设在本实验里只是**一致性证据**，不构成因果结论，log300 必须如实这样写。
