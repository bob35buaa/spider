# plan242 · E212：desk023 部分补偿扫描（gravcomp 0.4 / 0.6 / 0.8）

_承接 [E209/log298](../log/298_E209_desk_chair_prg_g1_gravcomp.md)（R295, FAIL）、[E211/log300](../log/300_E211_desk007_partial_gravcomp_sweep.md)（R297, FAIL）· 分支 `feat/E207-bucket-g1only-gravcomp` · 拟用 log301 / plan242 / **R298** / Phase 71_

---

## Context

E211 把可调 gravcomp 扫在 desk007 上，**主门 C1 五档全破**，但拿到一个可外推的机制结论：代价有两种形态，接触塌陷是载荷驱动（`r(g, contact) = −0.408`），姿态崩溃是 **CEM 安全门回退驱动**（`r(fallback, eef_ori) = +0.495`，控制 g 后偏相关 **+0.504**），而 `r(g, fallback) = +0.016` —— 两者近乎正交。

E212 把同一干预搬到 **desk023**。这不是「换个 case 再跑一遍」：desk023 与 desk007 处在**完全不同的 regime**，因此这是 E211 机制结论的**跨物体族外推检验**，也是一条独立的权衡曲线。

| | desk007（E211） | **desk023（E212）** |
|---|---|---|
| n | 5 | **4** |
| PRG narrow | 2/5（基线本就不好） | **4/4（基线完美）** |
| G1 narrow | 0/5 | 1/4 |
| z_bias PRG→G1 (cm) | −2.784 → **+1.116**（过冲） | −3.648 → **+0.353**（近乎归零） |
| 逐例 g\*（z 最优） | 中位 **0.71** | 中位 **0.93** |
| contact PRG→G1 | .880 → **.764（塌陷）** | .910 → **.921（反而改善）** |
| G1 破的门 | eef_ori + contact + release | **只有 eef_ori(3/4) + eef_pos/hand_pen(1/4)** |
| fallback 子系统 | body / hand gate | **leg gate** |

两条直接推论：

1. **desk023 没有接触塌陷形态**，只有姿态形态。E210 F2 的「两种互斥形态」在这里退化成一种。
2. **z 侧已经在 g=1.0 附近最优**（|bias| 0.413 cm）。降 g 只会**单调赔掉 z**（见 §二 A-P1 预测）。所以 E212 不是 E211 那种「两边都想救」，而是一条明确的**「拿 z 换 eef_ori」权衡曲线**——本计划的主门据此设计（用户口径，已确认）。

---

## 一、诊断（已完成，本计划的依据）

### D1 · 退化是纯粹的手/末端问题，其余全门无恙

E209 逐门读出（PRG 4/4 全门通过，故只列 G1 侧失败）：

| case | G1 narrow_failed | 说明 |
|---|---|---|
| 20231008_066_p1 | `eef_ori` | |
| 20231008_066_p2 | — | **唯一在 g=1 存活的一例** |
| 20231011_005_p1 | `eef_ori` | |
| 20231030_019_p1 | `eef_pos, eef_ori, hand_pen` | 最坏例 |

`fall` / `body_z` / `contact` / `release` / `lower_body` / `root_pos` / `root_ori` / `hand_pos` / `object_pos` / `object_ori` 在**两个端点、四个 case 上全部通过**。desk007 那种全面退化在 desk023 上没有发生。

### D2 · 逐例：fallback 解释 3/4，但 066_p1 是干净的反例

| case | eef_ori PRG→G1 | eef_pos (cm) | hand_pen | contact | **fallback** | narrow |
|---|---|---|---|---|---|---|
| 066_p1 | 16.71→**21.24** | 13.18→15.64 | 0.077→**0.303** | .929→.956 | **0.007→0.007（持平）** | T→F |
| 066_p2 | 15.86→16.05 | 13.58→**11.10** | 0.185→**0.106** | .896→.870 | **0.000→0.000** | **T→T** |
| 005_p1 | 18.78→22.25 | 13.48→14.01 | 0.237→**0.124** | .926→.941 | 0.148→**0.207** | T→F |
| 019_p1 | 16.47→**25.57** | 13.42→**22.18** | 0.161→**0.329** | .890→.918 | 0.048→**0.105** | T→F |

- **066_p2**：`cem_gate_fallback_used` 恒为 **0.000**，是四例里唯一在 g=1 保住 narrow 的，且 eef_pos / hand_pen **反而改善**。零回退 ⇒ 零损伤，与 E211 的 D5 完全一致。
- **019_p1**：fallback 涨幅最大（+0.057），也是损伤最重的一例（eef_pos 13.42→22.18）。一致。
- **005_p1**：fallback 涨 +0.059，eef_ori 中度恶化，但 eef_pos/hand_pen 改善。**部分**一致。
- **066_p1**：fallback **完全不动**（0.007→0.007），eef_ori 仍恶化 4.5°、hand_pen 恶化 4 倍。**这是 E211 机制的干净反例**——此处的损伤不可能由门回退解释，只能是载荷（D2 类）或别的东西。

desk023 的 fallback 又几乎全部来自 **leg gate**（`cem_leg_gate_fallback_used ≈ cem_posture_gate_fallback_used ≈ cem_gate_fallback_used`），而 desk007 的 034_p1 在 g=1 时 leg fallback 为 **0**、posture fallback 0.077 —— 是 body/hand 子系统。**同一个「回退」指标背后是不同的门**，log301 必须写清这一点，不能当成同一件事外推。

### D3 · 逐例 g\*（z 最优点）远高于 desk007

`g*_i = −pre_bias_i / delta_i`（delta 为 g=1 的实测修正量）：

| case | pre_bias (cm) | delta(g=1) (cm) | g\* |
|---|--:|--:|--:|
| 066_p1 | −3.774 | +3.653 | **1.033** |
| 066_p2 | −3.908 | +4.176 | 0.936 |
| 005_p1 | −3.274 | +3.554 | 0.921 |
| 019_p1 | −3.638 | +4.622 | 0.787 |

中位 **0.928**、均值 0.919。对比 desk007 的 0.710。**四例中三例 g\* > 0.9**，即 desk023 上全额补偿几乎不过冲。这正是「降 g 必赔 z」的来源。

---

## 二、实验设计 E212（Stage A）

**范围**：desk023 × 4 case（`20231008_066_p1`, `20231008_066_p2`, `20231011_005_p1`, `20231030_019_p1`）。不加其它物体。
**g 网格**：`{0.4, 0.6, 0.8}`，**与 E211 desk007 逐格一致**（用户口径，已确认）——代价是 g=0.4 大概率是废点，收益是两个物体族可以逐格对比，跨族外推最干净。
**粒度**：物体级单值。逐例 g\* 只作**预测-实测证伪检查**，不作调参旋钮。

新 arm `G04/G06/G08`：`<body name="object" gravcomp="0.4|0.6|0.8">`，其余与 E206 PRG **逐字节相同**。

- 4 case × 3 档 = **12 条** CEM（1024×32 seed 0）。
- 与两条**冻结不重跑**的基线拼成 5 点曲线：g ∈ {0（E206 PRG）, 0.4, 0.6, 0.8, 1.0（E209 G1）}。两个端点的产物均已核在位（4/4 × 2）。
- **双机执行**：本机 8 卡跑 8 条 + 另一台 8 卡机跑 4 条，各一波并行，wall ≈ 单条时长。详见 §四之二。

### 预注册预测（P0 冻结，**不参与主门判定**，独立报告）

主门是 C1（§三）。以下五条是机制层的可证伪预测，**不影响 SUCCESS/FAIL**，但必须逐条在 log301 里判定——E211 最大的价值恰恰来自这一层（它推翻了自己的 D2）。

| # | 预测 | 证伪条件 |
|---|---|---|
| **P1** | `z_bias(g) ≈ −3.648 + 4.001·g`（线性），5 点 OLS **R² ≥ 0.90** | R² < 0.90 ⇒ 收缩模型在 desk023 中间档不成立 |
| **P2** | `eef_ori(g)` 随 g **单调上升** | 非单调 ⇒ 与 E211 desk007 同样证伪，「载荷即姿态」跨族均不成立 |
| **P3** | **066_p2（fallback≡0）在全部 3 档保持 narrow PASS** | 任一档失败 ⇒ 零回退不足以保证无损伤，D5 机制不充分 |
| **P4** | 控制 g 后 `r(fallback, eef_ori)` 偏相关 **> +0.40**（E211 desk007 实测 +0.504） | ≤ +0.40 ⇒ E211 的 fallback 机制不能外推到 desk023（leg gate 与 body/hand gate 行为不同） |
| **P5** | **066_p1（fallback 恒定）的 eef_ori 随 g 单调上升**（此例只能是载荷驱动） | 非单调 ⇒ 该例两个候选机制**都**不成立，存在第三种未识别机制，必须登记为缺口 |

P1 的斜率与截距由两个冻结端点解析确定，所以 R² 度量的**只是三个中间档偏离直线的程度**，不是自我实现的拟合。

---

## 三、评判标准（P0 冻结，不得事后改）

desk023 n=4 的两条基线，均由本计划从 E209 交付产物 `e209_two_arm_rollout.tsv` / `e209_object_z_diff_by_case.tsv` 精确重算（P0 会再算一次并断言逐位相等）：

| 指标 | 方向 | **PRG (g=0)** | **G1 (g=1)** |
|---|---|--:|--:|
| z_bias 宏平均 (cm) | →0 | −3.6481 | **+0.3532** |
| mean\|z_bias\| (cm) | ↓ | 3.6481 | **0.4133** |
| z_mae (cm) | ↓ | 4.6158 | **3.1752** |
| obj_pos (cm) | ↓ | 10.3506 | **9.9442** |
| obj_ori (°) | ↓ | 5.2336 | **4.5995** |
| **eef_ori (°)** | ↓ | **16.9583** | 21.2777 |
| **eef_pos (cm)** | ↓ | **13.4131** | 15.7323 |
| **hand_pen** | ↓ | **0.1651** | 0.2152 |
| **contact** | ↑ | 0.9104 | **0.9211** |
| release | ↓ | **0.0421** | 0.0542 |
| root_ori (°) | ↓ | **8.0138** | 9.4703 |
| root_pos (cm) | ↓ | **14.5716** | 16.0636 |
| body_z p95 (m) | ↓ | **0.0809** | 0.0874 |
| leg_pen | ↓ | **0.0088** | 0.0095 |
| ankle_jerk p95 | ↓ | **598.03** | 720.34 |
| **narrow** | ↑ | **4/4** | 1/4 |
| hard | ↑ | 4/4 | 4/4 |

### 主门 C1（必须全过才算 SUCCESS）

| 子句 | 判据 | 取值来源 |
|---|---|---|
| C1a 高度收益保住 | mean\|z_bias\| ≤ **1.50 cm** ∧ z_mae ≤ **3.1752 cm** | z_mae 取 **G1 值**（最紧的一条 z 子句） |
| C1b 物体跟踪不退 | obj_pos ≤ **10.3506** ∧ obj_ori ≤ **5.2336** | PRG 值 |
| C1c **eef_ori** | ≤ **17.958°** | PRG + 1.0 |
| C1d **eef_pos / hand_pen** | eef_pos ≤ **14.413 cm** ∧ hand_pen ≤ **0.1951** | PRG + 1.0 cm / PRG + 0.03 |
| C1e **contact** | ≥ **0.8804** | PRG − 0.030 |
| C1f **narrow** | ≥ **3/4** ∧ hard **4/4** | PRG 是 4/4，允许掉一例 |

**任一档六条全过 = SUCCESS。** 三档皆不全过 = FAIL。

判据的重心说明（写在前面，避免事后解释）：C1b / C1e 在两个端点上**本来就都满足**，是防退化的兜底子句，不承担区分度；真正卡住的是 **C1a 的 z_mae 子句**、**C1c**、**C1d**、**C1f** 四条。也就是说 C1 实际在问：*有没有一个 g，能在 z_mae 不输给全额补偿的同时，把 eef_ori / eef_pos / hand_pen / narrow 保在 PRG 附近。*

### 副门（记录，不单独判 FAIL，但任一破必须在 log301 显式点名）

- **C2** root_ori ≤ **9.014°**（PRG+1.0）、root_pos ≤ **15.572 cm**（PRG+1.0）
- **C3** release ≤ **0.0921**（PRG+0.05）；硬门 body_z p95 ≤ 0.20、leg_pen ≤ 0.20、ankle_jerk < 1000
- **C4** 吞吐：median wall ∈ **[33, 55] min**（E209 desk/chair 42.2、E211 desk007 46.7；desk023 帧数 3104–4832，同量级）

### 反挑拣条款（rules §5）

- **全 4 例报表**，禁止只报最好的档或最好的 case；每个指标报 **mean + std + worst-case**。
- 三档**全部评测并进表**，即便某档（预期是 g=0.4）明显更差——否则 P2/P5 的单调性判据无意义。
- 逐例 g\* 的预测-实测比对**只用于证伪 P1**，不得据此给每例挑不同的 g。
- **066_p1 与 066_p2 必须分开报**：二者同源同一段动作的两个人，但一个 fallback≡0、一个是机制反例，合并会互相抵消。
- n=4 时单例权重 25%，**必须同时报「含/不含 019_p1（最坏例）」两套宏平均**（E211 §八-3 的教训：一个离群例会淹没其余的真实信号）。这不是挑拣——两套都报才诚实，主门仍以**含全部 4 例**的那套判定。

### 视觉复核（rules §5 强制）

用户口径：**本轮定位为纯诊断**，出片准入另议（若 C1 真有一档全过，再单开一次全量渲染 + 人工复核）。故本轮渲染范围收敛为：

- **胜出档（或 C1 最接近的一档）4/4** + **最坏例 019_p1 的全 3 档**，去重后约 6–7 条。
- **必须遵守机位陷阱**（E209 F6 / E210 F3）：`_auto_video_camera`（`spider/viewers/__init__.py:262-291`）每帧用 sim∪ref 并集包围盒算 lookat/半径 ⇒ sim 不同则机位不同 ⇒ **跨视频比姿态无效，连同一份参考都会渲成两个姿势**。mp4 内唯一有效判据 = **同一视频内 sim vs ref**。
- **跨档比对只能用 viser**：复用 `E211/viser_replay_arms.py`（同场景、同相机、同时间轴）。
- 抽帧位置**必须先用逐帧数值定位分歧峰值再抽**（E210 F4：按固定比例抽帧什么都看不出来）。
- 重点看：手是否仍贴在桌面/桌沿上、末端朝向是否跟得上参考、019_p1 的 eef_pos 22 cm 偏差在画面上是什么形态（手飞出去 or 整体平移）。

---

## 四、实施步骤

复用 E211 已有骨架，尽量只写差异。E211 的 9 个脚本 + 1 个 eval runner 构成完整链路，E212 逐个克隆并把 desk007/5/15/8+7 参数化为 desk023/4/12/8+4。

| P | 内容 | 关键文件 |
|---|---|---|
| **P0** | 编号 plan242 / log301 / **R298** / Phase 71；新建 `E212/e212_common.py`（照 `E211/e211_common.py`：`OBJECT_KEY="desk023"`、`EXPECTED_CASES=4`、`EXPECTED_ROWS=12`、`ARMS` 不变、`BASELINE_DESK023` 用 §三 表、`GATES` 用 §三 C1）；**P0 必须重算 §三 两条基线并断言与写死值逐位相等**（tol 同 E211：narrow 0、frac 类 5e-4、其余 5e-3） | 新 `workspace/core4d/scripts/experiments/E212/e212_common.py`；只读 import `e209_common` / `e200_common._signature` |
| **P1** | 复用 E211 的 `assert_gravcomp_diff_value(base, sidecar, value)`（`e200_common.assert_gravcomp_diff` 在 `:146` 硬编码 `"1"`，部分补偿会被拒；该函数被 E198/E200/E209/E210 共用，**只读不改**）。**同样带篡改样本反向自测**（值写错 / 值对但顺带改 mass 各一个，证明断言都会拒绝） | `E212/e212_common.py`；模板 `E211/e211_common.py:248-297` |
| **P2** | 建 3×4=**12** 个 sidecar `scene_act_E212_lowgeom_PRG_gc{04,06,08}.xml`，写在 4 个 desk023 dcv3 task dir 下（**已核实这些目录当前无任何 `scene_act_E21*` 文件，与 E211 零冲突**）。三道守卫：`gravcomp` token 恰好 1 次、MuJoCo 编译后 `ngeom/npair/nq/nv/nu/nbody` 与 base 全等、sidecar sha256 互不相同且不等于 base | `E212/build_scenes.py`，照 `E211/build_scenes.py` |
| **P3** | override：单变量 `ALLOWED_DIFF={"scene_name"}`。audit 必须验证 compose 全键 diff **恰为** `{scene_name}`，且**三个 arm 相互之间也只差这一键** | `E212/build_overrides.py` |
| **P4** | manifest + 输入 sha256 pin（轨迹 / 3cm 掩码 / base scene 必须与 E206 交付表逐条相等）。**同时产出 shardA(8) / shardB(4) 两份分片 manifest**（§四之二）。含 `host` / `gpu` 列 | `E212/build_manifest.py`。**注意 E211 F5：`from build_manifest import ...` 会撞到 E200 的同名模块，必须用显式路径 import（`_load_sibling`）** |
| **P5** | **场景快照（rules §7 保障 2）**：`results/E212/scene_snapshot/` 覆盖 4 个 dcv3 task dir 全部 XML + `manifest.txt`（git HEAD + 每文件 sha256）。同时 `git add -f` 12 个新 sidecar | `E212/snapshot_E212_scenes.sh` → `scripts/convert/snapshot_scenes.sh` |
| **P6** | smoke 1 条（64×4）+ 运行时契约审计：`config_act.yaml` 全键 diff 仅 THE_VARIABLE；A0 hand-gate (0.10/−0.020)、`init_pos_actuator_gain=500`、`leg_object_penalty_scale=2.0`、`cem_leg_gate_enabled=true`、**运行时读回 `MjModel.body_gravcomp` == 本档值**（唯一不能被文件命名伪造的证据——sha256 和 scene_name 在「0.6 写成 0.06」时都照样通过）。**完成判据必须用结果 npz，不能用 `config_act.yaml`**（E211 F4：它在进程启动时就写了，会把在飞的行报成 PASS） | `E212/audit_runtime_contract.py` |
| **P7** | Stage A full **12 条**，双机 **8+4** 并行（§四之二）。**队列用 E199 版**（`e200_common.TIER_RANK` 无 `"P0"` 会 KeyError，E209 F2）。**中断后必须先跑 `E210/reset_stale_rows.py`**（`run_local_priority_queue.py:30` 的 `ELIGIBLE` 不含 `running`，被打断的行会变墓碑并静默报 0 pending，E210 F1）——双机时**只对自己那份 shard 跑** | `E199/run_local_priority_queue.py` + `E210/reset_stale_rows.py`；`scripts/launch/active/run_E212_local_8gpu.sh` |
| **P8** | 评测：5 档 × 4 例 = **20 rollout**，14-gate 全门（`HARD_GATES + BANDED_GATES`，不能只写 `BANDED_GATES + body_z`，E209 F8）+ z 诊断 + C1 六条 + P1..P5 五条预测 + 「含/不含 019_p1」两套宏平均。**两端点重打分（而非从 E209 TSV 抬数）必须逐位复现 §三 冻结值**，否则 evaluator 漂移，先停。<br>**⚠ 本步不是常量替换**：E211 的 `c1_for()` 写死了 `C1d_contact` / `C1e_release` 两条子句，而 **E212 的 C1d 是 `eef_pos ∧ hand_pen`、C1e 是 `contact`**（§三），**必须改写函数体**而不只是改 `GATES` 字典。同理 E211 的 `A_P2/A_P3` **没有 `pass` 键**（控制台打印空白判定），E212 的 P2/P3/P4/P5 都要补上显式 `pass`。<br>**逐指标方向表**：E211 `DIRECTION` 16 项里只有 `contact` 是 `+1`，其余 `-1`，`z_bias_cm` 被**故意排除**在单调性之外（有符号偏差不是「越大越坏」）——这套可逐字复用（E210 F6） | `scripts/eval/runners/eval_E212_partial_gravcomp.py` + `scripts/eval/wrappers/eval_E212_partial_gravcomp.sh`（**E211 没有 wrapper**，本次按 rules §8 补上） |
| **P9** | 渲染（osmesa）约 6–7 条 + viser 跨档回放；按 §三 视觉口径复核，**实际观察不得留空** | `E212/render_cem_results.py`；复用 `E211/viser_replay_arms.py` |
| **P10** | **provenance 入库（rules §7，E211 F9/F10）**：`workspace/core4d/results` 是**指向外部存储的符号链接**，`git add -f` 会报 `beyond a symbolic link`。必须把快照 + 端点 + manifest + 评测 + 运行时契约镜像到 `workspace/core4d/report/E212/provenance/`（真实目录），并用 **`git add -f <dir>`**——`.gitignore` 含 `*.json` / `*.xlsx`，普通 `git add` 会**静默跳过**它们（E211 第一次就漏了全部 18 个 json + `manifest.txt`）。入库后**必须用 `git ls-files` 反查文件数**，不能只看命令没报错 | `workspace/core4d/report/E212/provenance/` |
| **P11** | 写 log301、更新 `EXPERIMENT_TRACKER.md`（R298）、`progress.md`；commit 时**显式列路径**（E209 记过：并发提交者会把 `git add -f` 的文件扫进不相干 commit）。**顺带**：`progress.md` 现 1806 行，远超 rules §14 的 200 行阈值，按 §14 归档 E209 及以前到 `progress_archive/` | |

**Stage B（不在本轮，仅登记）**：若 C1 三档全破，按 E211 §八-2 的建议走 **Stage B′**——固定胜出 g，扫 CEM 门参数（`cem_safety_gate_max_violation_pct: 0.0 → 0.02`，或 `cem_safety_gate_fallback: least_violation → 软惩罚`）。desk023 的 fallback 是 **leg gate** 主导，与 desk007 的 body/hand 不同，Stage B′ 的旋钮要相应改成 leg gate 侧，**不能照抄**。

### 四之一、克隆清单（E211 → E212 逐文件差异面）

E212 = E211 链路的参数化克隆。`e212_common.py` 是枢纽，其余每个文件都是它的薄壳（读 `C.<NAME>`）。**下表列出全部需要改的硬编码字面量**；未列出的一律逐字复用。

| 文件 | 必改 | 逐字复用 |
|---|---|---|
| `e212_common.py`（枢纽） | `EXP="E212"`、`OBJECT_KEY="desk023"`、`EXPECTED_CASES=4`；`BASELINE_DESK023`（§三 全部 34 个数）、`PREREG_PER_CASE`（§一 D3 四行）、`PREREG_Z_LINEAR`（截距 −3.6481 / 斜率 4.0013）、`GATES`（§三 C1）；`manifest_path` 里写死的 `e211_` 前缀 | `CASES`（按 `OBJECT_KEY` 派生）、`EXPECTED_ROWS`（派生）、`ARMS`（**与 E211 相同，不改**）、`scene_name()`（按 `EXP` 派生）、`gravcomp_str`、`sources()`、`assert_gravcomp_diff_value`、`build_sidecar`、`scene_gravcomp`、`audit()`、`BASELINE_TOL`；E209 权威（`BASE_SCENE`/`G1_SCENE`/CEM 预算/`EXPECTED_OBJECT_MASS_KG=5.0`/`task_dir`/…）全部只读继承 |
| `build_scenes.py` | sys.path、`e211_scene_audit.json` 文件名 | `MODEL_INVARIANTS`（`ngeom,npair,nq,nv,nu,nbody`）、`self_test`（**两个反向样本：值写错 / 值对但顺带改 mass**）、重复 sha 检查、「sidecar ≠ E209 g=1 场景」检查。条数由 `len(rows)×len(ARM_ORDER)` 自动变 12 |
| `build_overrides.py` | sys.path、`HEADER` 与注释里的 "E211" 字样 | `ALLOWED_DIFF={"scene_name"}`、`_compose`/`_drift`/`audit_case` 全部——里面写死的 `leg_object_penalty_scale==2.0`、`cem_leg_gate_enabled`、`len(leg_object_penalty_geom_names)==16`、`object_collision_sdf_mode=="union"`、`contact_hdmi_mask_path` 相等、以及 arm-vs-arm 两两只差 `scene_name` 的循环，**都是物体无关的物理契约，不是 desk007 事实** |
| `build_manifest.py` | sys.path、`EXPECTED_SHARD_ROWS={"A":8,"B":4}`、分片赋值改对角式（§四之二）、输出文件名 | 40 列 `FIELDS`（含 `host`/`gpu`）、6 项输入存在性、**C0 provenance**（轨迹/掩码/base scene sha 与 E206 交付表逐条相等）、基线 `config_act.scene_name==BASE_SCENE`、`check_shards` 其余断言、全部 `effective_scene_sha256` 互异且不等于 base |
| `freeze_baseline.py` | sys.path、输出名的 `e211_`/`_desk007`；**`BASELINE_RELEASE_N` 需重新推导** —— desk023 四例的 `release` **全部有限**（已核），故 **n=4**，不像 desk007 有一例释放窗为空 | `ROLLOUT_FIELDS`（12 项列名→基线键映射）、`measure()`（z_bias/z_abs_bias/z_mae/narrow 聚合）、arm 大小写映射 `prg↔PRG` / `g1↔G1` |
| `audit_runtime_contract.py` | sys.path、输出名 | `THE_VARIABLE={"scene_name","model_path"}`、`RUN_LOCAL_KEYS`、`STAGE_BUDGET_KEYS`、`REQUIRED_RUNTIME`（A0 hand gate `-0.010/0.10/-0.020`、`leg_object_penalty_scale=2.0`、`cem_leg_gate_enabled`、`init_{pos,rot}_actuator_gain=500/50`）、`same()` NaN 容错、`compiled_object_gravcomp()`（读 `MjModel.body_gravcomp[bid]`）、**以 result npz 而非 config_act 为完成判据**（E211 F4 已修） |
| `merge_shards.py` | sys.path、输出名、`_load_sibling` 里的 `sys.modules` 前缀 `e211_`→`e212_`（两个 runner 若同进程 import 会撞） | `_load_sibling`（**按 `__file__` 同目录的文件系统路径解析 + 前缀化模块名**，这是 E211 F5 的解法）、`DONE` 常量、`wall_minutes`（从 log header 解析 `started_at=` 与 npz mtime 求差）、四项检查、按 host 的 median 统计、C4 窗口读 `C.GATES` |
| `snapshot_E212_scenes.sh` | 两处内联 heredoc 的 sys.path/`import`、**`:31` 写死的 `-ne 5` → `-ne 4`** 及其报错文案、`snapshot_scenes.sh` 的实验 tag、manifest 路径 | 退出前的核验 heredoc：逐 case 要求 `{BASE_SCENE}.xml` 命中 `baseline_scene_sha256`、各 `{SCENE_BY_ARM[arm]}.xml` 命中 `effective_scene_sha256`、`{G1_SCENE}.xml` 仅验存在（其 hash 归 E209 所有） |
| `render_cem_results.py` | sys.path、`e211_render_summary.json` | `--arms` 默认 `"G06,G08"`（端点不重渲）、**`MUJOCO_GL==osmesa` 硬校验**（egl→EGLError、glfw→本机无 `_mjr_context`）、只渲 `outdir_npz` 已存在的行（可与 CEM 并发）、逐行异常隔离、端点 mp4 在位审计（`E206_{case}_prg.mp4` / `E209_{case}_G1.mp4`）、机位 caveat 文案 |
| `eval_E212_partial_gravcomp.py` | sys.path、`_load_by_path` 别名前缀、三个输出文件名、docstring 的「20 rollouts (4 desk023 × 5 g)」；**`c1_for()` 函数体必须改写**（见 P8）；`A_P2/A_P3` 补 `pass` 键 | `ARMS=("prg","G04","G06","G08","g1")` 与 `ARM_G` 映射（**不变**）、`arm_paths()` 端点 join、`DIRECTION` 16 项、`macro()`（只在**全 arm 皆有限**的公共 case 集上取均值）、`monotone()`、端点漂移守卫（tol 5e-3 + narrow 精确）、`GATE_FIELDS`、上游 `funnel_config`/`eval_E206_arm_ablation`/`eval.core.core_metrics` |
| `run_E212_local_8gpu.sh` | manifest 目录、快照路径、分片文件名、注释里的 "8 rows / B is 7" 与 "all 5 cases" | `SHARD`/`GPUS`/`PER_GPU_MEM_MIB`/`MAX_PER_GPU`/`POLL_INTERVAL`/`DRY_RUN`/`SKIP_SNAPSHOT`/`STAGE` 全部 env 旋钮、`TORCHDYNAMO_DISABLE=1`、`MUJOCO_GL=disable`、host 打戳、**已修正的短 sha HEAD 校验** |

**sys.path 顺序**：`e212_common` 需插入 `("E212","E209","E206","E200","E199")`。注意 `sys.path.insert(0, ...)` **后插者在前**，所以 E199 排第一——这正是 E211 F5 里 `from build_manifest import` 拿到别人模块的根因。凡是跨文件取符号，一律走 `_load_sibling` / `_load_by_path`。

---

## 四之二、双机执行（Stage A 12 条 = 8 + 4）

### 为什么必须分片 manifest

`E199/run_local_priority_queue.py` 每次状态变化都 `C.write_tsv(manifest, rows, fields)` **整文件重写**。两台机器挂同一个 `/mnt`、指向同一份 manifest ⇒ 后写者用自己内存里的旧快照覆盖对方刚写的状态，双方都会把对方的行重新认领或永久漏掉。这不是理论风险：E210 F1 已经在单机双队列上炸过一次。

处置：P4 直接产出两份**互不相交**的分片 manifest，各机只读写自己那份。CEM 输出目录按 `case × arm` 天然互不重叠，日志同理，所以除 manifest 外没有共享可写状态。

### 分片规则（确定性 Latin-square 对角，写进 P4 并记入 log301）

**必须改写 E211 的分片实现**。`E211/build_manifest.py:130` 用的是 `SHARDS[index % len(SHARDS)]`，即在排序后的 `(case_id, arm)` 序列上做**奇偶 parity**。因为每个 case 恰有 3 个 arm（奇数步长），parity 会跨 case 轮转，所以它对 12 行**依然能给出全 case 全 arm 的覆盖**——但比例是 **6/6**，不是用户要的 8/4。

E212 改用 Latin-square 对角。以 `case_idx ∈ [0,3]`（CASES 字典序）、`arm_idx ∈ [0,2]`（G04/G06/G08）：

```
(case_idx + arm_idx) % 3 == 2  ->  shardB（远端，4 行）
否则                            ->  shardA（本机，8 行）
```

| | shardA（本机 8 行） | shardB（远端 4 行） |
|---|---|---|
| 覆盖 case | **4/4** | **4/4** |
| 覆盖 arm | **G04, G06, G08 全** | **G04, G06, G08 全** |
| 明细 | 066_p1×{G04,G06}、066_p2×{G04,G08}、005_p1×{G06,G08}、019_p1×{G04,G06} | 066_p1×G08、066_p2×G06、005_p1×G04、019_p1×G08 |

为什么不是「取前 8 / 后 4」或「`i%3==2`」：`i%3==2` 会把**整个 G08 档全部放进 shardB**——而 G08 恰是预测最优档（|z_bias| 0.447），远端一挂就整档丢失，剩下的 8 条读不出曲线。对角规则保证**任一片单独存活都是一个跨 case 跨档的可读子集**。

改动点：`E212/build_manifest.py` 的 `EXPECTED_SHARD_ROWS`（E211 在 `:66` 写死 `{"A":8,"B":7}`）改为 `{"A":8,"B":4}`，`:130` 的 parity 赋值改为上式。`check_shards`（E211 `:164-189`）的其余断言可**逐字复用**：两片无交集、并集恰等于 12 行主表、每片各自覆盖 `set(C.CASES)` 与 `set(C.ARM_ORDER)`、三份文件输入 sha256 列逐行相等。

> **注意 E211 `:175-176` 的 early-return**：`if len(rows) != C.EXPECTED_ROWS: return` —— 子集运行时 `check_shards` 会**整段跳过**。这是给 `--cases` 调试留的口子，但意味着分片断言在非全量运行下**静默不生效**。P4 必须在全量（12 行）下至少跑一次并记录断言输出。

### 快照只拍一次（rules §7）

场景快照写共享路径 `results/E212/scene_snapshot/`，两台机同时拍会互相覆盖。约定：**本机（shardA）负责拍**，远端用 `SKIP_SNAPSHOT=1`，脚本在跳过时**断言** `scene_snapshot/manifest.txt` 已存在、且其中记录的 git HEAD == 当前 HEAD，不一致直接退出。

> **E211 F3 必须避免**：E211 的这条校验用 `grep -oE '[0-9a-f]{40}'` 找 40 位 sha，而 `snapshot_scenes.sh` 写的是**短 sha**，导致 `SNAP_HEAD` 恒为空、条件被静默跳过——**一条从来不触发的守卫比没有守卫更糟**。E212 沿用 E211 已修正的版本（显式解析 `# Git HEAD:` 行并按长度截取比对，解析不出来就退出），并在 P6 用一次故意改 HEAD 的负样本验证它真的会拒绝。

### 命令

前置：P0–P6 全部完成（sidecar / override / manifest / 快照 / smoke / 运行时契约全过）。两台机器共享同一个 `/mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider` 与同一个 `.venv`（用户已确认远端与 E211 同一台 `lshb-k8s-al-sh-gpu-rdma-prod-103`，配置不变）。

**本机（8 卡，8 条）—— 由 Claude 执行**
```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider
SHARD=A bash workspace/core4d/scripts/launch/active/run_E212_local_8gpu.sh
```

**另一台 8 卡机（4 条）—— 这条发给用户**
```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider && SHARD=B SKIP_SNAPSHOT=1 bash workspace/core4d/scripts/launch/active/run_E212_local_8gpu.sh
```

先干跑确认认领的是自己那 4 条：
```bash
cd /mnt/ali-sh-1/usr/xiayibo/work_dir/embodied/spider && SHARD=B DRY_RUN=1 bash workspace/core4d/scripts/launch/active/run_E212_local_8gpu.sh
# 期望输出: [dry-run] 4 pending across gpus=['0'..'7'] mem>=5000MiB
```

`run_E212_local_8gpu.sh` 照 `run_E211_local_8gpu.sh` 改（env 旋钮全部沿用）：`SHARD`（A|B，默认 A）、`GPUS`（默认 `0,...,7`）、`PER_GPU_MEM_MIB`（5000）、`MAX_PER_GPU`（1）、`POLL_INTERVAL`（15）、`DRY_RUN`、`SKIP_SNAPSHOT`、`STAGE`（full|smoke）。脚本内已 `export TORCHDYNAMO_DISABLE=1`（本机无 python3.12-dev，triton JIT 会炸）和 `MUJOCO_GL=disable`（CEM 无头）。

本机 8 卡（L20Y 80 GB）当前全空闲、无在跑 CEM 进程，8 条一波即可。远端 4 条也是一波。**预计 wall ≈ 单条时长 ≈ 40–50 min。**

### 合流与对账（进 P8 之前必须做）

```bash
.venv/bin/python workspace/core4d/scripts/experiments/E212/merge_shards.py --stage A
```
按 `(case_id, arm)` 主键把两片状态并回主表，并断言：
1. **12/12** `status == run_complete_pending_eval`，`problem rows: []`；
2. 两片没有同一主键的重复行（防止误在两台机上跑了同一份 shard）；
3. 每条产物 `config_act.yaml` 的 `scene_name` 与该行 `arm` 对应的 gravcomp 档一致（防分片错配）；
4. 每行记录落在哪台机（`host`）与哪张卡（`gpu`）——双机跑必须能回答「这条是在哪跑的」。

**对账不过就不许进 P8。**

---

## 五、已知陷阱清单（从 E209/E210/E211 继承，实施时逐条对照）

| 来源 | 陷阱 | 本计划的处置 |
|---|---|---|
| E209 F1 | `e200_common.build_gravcomp_sidecar` 输出名硬编码 E199 | 自写 writer，只复用 `_signature` |
| E209 F2 | `e200_common.TIER_RANK` 无 `"P0"` | 队列用 E199 版 |
| E209 F3 | 给 `e206_common.ARMS` 加 arm 会污染在跑的实验 | `e206_common` / `e209_common` / `e211_common` 全部只读 |
| E209 F5 | 运行时契约 NaN≠NaN 自比不等 | 用 `same()` 处理 NaN |
| E209 F6 / E210 F3 | 跨视频比姿态无效（auto camera 随 sim 变） | mp4 只做同视频内 sim vs ref；跨档一律用 viser |
| E209 F8 | 逐门表漏 3 个硬门；`fall_flag` 布尔被 `_finite` 变 NaN | 14 门全列 + `num()` 强转 |
| E210 F1 | 队列对中断不 resume-safe | 派发前后跑 `reset_stale_rows.py`，只跑自己那片 |
| E210 F4 | 按固定比例抽帧看不出差异 | 先用逐帧数值定位峰值再抽 |
| E210 F6 | `contact` 方向相反 | 逐指标 `DIRECTION` 表 |
| **E211 F1** | `assert_gravcomp_diff` 硬编码 `"1"`，部分补偿被拒 | 复用 E211 的参数化断言 + 篡改样本反向自测 |
| **E211 F3** | 快照 HEAD 校验用 40 位 sha 正则而 manifest 写短 sha ⇒ **守卫恒不触发** | 沿用已修正版；并在 P6 用负样本验证它真会拒绝 |
| **E211 F4** | 运行时契约以 `config_act.yaml` 为完成判据，把在飞的行报成 PASS | 完成判据改为**结果 npz**，未完成显式列 `pending` |
| **E211 F5** | `from build_manifest import` 会拿到 E200 的同名模块（E2xx 目录 sys.path 碰撞） | 一律显式路径 import（`_load_sibling`） |
| **E211 F9/F10** | `results` 是符号链接，`git add -f` 报 `beyond a symbolic link`；且 `.gitignore` 的 `*.json`/`*.xlsx` 让普通 `git add` **静默跳过**最要紧的文件 | 镜像到 `report/E212/provenance/` + `git add -f <dir>` + **`git ls-files` 反查文件数** |
| 本计划新增 | 双机指同一份 manifest ⇒ 整文件重写互相覆盖 | 分片 shardA/shardB，各机只写自己那份 |
| 本计划新增 | 朴素索引取模会把整个 G08 档放进远端片，远端一挂丢整档 | Latin-square 对角分片，两片各自覆盖全 case 全 arm |
| **本计划新增** | `check_shards` 在 `len(rows) != EXPECTED_ROWS` 时**整段 early-return**，子集调试运行下分片断言**静默不生效**（与 E211 F3 同类：一条不触发的守卫） | P4 必须在全量 12 行下至少跑一次并把断言输出记进 log301 |
| **本计划新增** | E212 的 C1d/C1e 与 E211 不是同一组指标，只改 `GATES` 字典会**用错门判 SUCCESS** | P8 改写 `c1_for()` 函数体；并用一个「故意把 eef_ori 调到 30°」的假 summary 反测 C1 确实会判 FAIL |
| **本计划新增** | E211 的 `A_P2/A_P3` 无 `pass` 键，控制台判定为空——预测写了等于没判 | P2..P5 全部补显式 `pass`，并在 summary JSON 里落盘 |
| **本计划新增** | desk023 的 fallback 是 **leg gate** 主导，desk007 是 **body/hand gate**。把两者当同一个「回退」指标外推会得出错误结论 | log301 必须分门报 `cem_{leg,body,hand,posture}_gate_fallback_used`，不能只报聚合的 `cem_gate_fallback_used` |
| **本计划新增** | n=4 时单例权重 25%，019_p1 一例可主导宏平均 | 强制同时报「含/不含 019_p1」两套；主门以含全部 4 例判定 |

---

## 六、验证方式

1. **契约层（不跑 CEM 即可验）**：`python E212/build_scenes.py --dry-run` → 12/12 过参数化断言 + 编译不变量；篡改样本自测应报 `AssertionError`。`python E212/build_overrides.py --audit` → compose 全键 diff 恰为 `{scene_name}`，12/12，且三 arm 相互也只差此键。
2. **基线复现（P0 门槛）**：重算 desk023 4 例的两条端点，必须逐位得到 §三 表——尤其 `narrow 4/4 → 1/4`、`eef_ori 16.9583 → 21.2777`、`z_bias −3.6481 → +0.3532`。**不符即停**。
3. **smoke**：1 条 64×4，`audit_runtime_contract.py` 逐项断言通过，其中运行时 `MjModel.body_gravcomp` 读数 == 该档值。
4. **分片对账**：`SHARD=A/B DRY_RUN=1` 各自应报 **8 / 4** pending 且两边主键无交集；跑完 `merge_shards.py --stage A` 必须 **12/12** `run_complete_pending_eval`、无重复主键、`scene_name` 与 arm 逐行匹配、`host`/`gpu` 列已填。
5. **主判据**：§三 C1 六条子句（含/不含 019_p1 两套）+ §二 P1..P5 五条预注册预测，全部落到 `results/E212/s6_downstream/eval/g_sweep/` 的 TSV/JSON。
6. **视觉**：按 §三 口径渲染 6–7 条 + viser 跨档回放，逐帧数值定位峰值后抽帧，结论写进 log301「实际观察」（**不得留空**）。
7. **provenance**：`git ls-files workspace/core4d/report/E212 | wc -l` 必须等于镜像目录实际文件数（E211 是 84/84）。

---

## 七、预期结果与判读

- **最可能（我的预判）**：**C1 三档全破，判 FAIL**。理由：z 侧 g\*≈0.93 意味着降 g 单调赔 z，C1a 的 `z_mae ≤ 3.1752` 子句在 g=0.6/0.4 上大概率破；而 eef_ori 侧若 E211 的 fallback 机制外推成立，则它**不是 g 的函数**，g=0.8 也救不回 C1c/C1f。两头都不达标是最可能的形态。
- **若 g=0.8 全过 C1**：说明 desk023 存在真实的权衡甜点，desk007 的失败是物体族特异（细杆钩握）而非 gravcomp 路线本身的问题。这会**部分推翻 E209「gravcomp 不用于 desk/chair 出片」**的结论——该结论是被 desk007 拖垮的宏平均。此时按用户口径单开出片准入。
- **若 P3 破（066_p2 在某档失守）**：零回退不足以保证无损伤，E211 的 D5 机制不充分，需要第三种解释。
- **若 P4 破（偏相关 ≤ +0.40）**：E211 的 fallback 机制**不能跨物体族外推**——很可能因为 leg gate 与 body/hand gate 的回退行为本就不同。这会直接改写 Stage B′ 的旋钮选择。
- **若 P5 破（066_p1 非单调）**：该例的损伤既不由载荷也不由回退解释，存在**第三种未识别机制**，必须登记为 core 层缺口而不是继续调参。
- **兜底判读**：desk023 在 **PRG（g=0）下本来就是 narrow 4/4**。如果全档 FAIL，对 desk023 的正确工程结论不是「继续调 g」，而是**直接用 PRG 出片、接受 −3.65 cm 的 z bias**——除非 z bias 本身是交付阻塞项。这一点必须在 log301 里明确写出，避免把一个「基线已经够用」的物体族误当成待救援对象。

**注**：desk023 的 4 例来自 3 段不同录制（066 的 p1/p2 是同一段的两个人）。n=4、且 fallback 与物体族在数据里高度共线，本计划**不做**打破混淆的对照。因此 §一 D2 的机制归因在本实验里只是**一致性证据**，不构成因果结论，log301 必须如实这样写。
