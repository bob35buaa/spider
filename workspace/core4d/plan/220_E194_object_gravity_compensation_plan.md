# E194 实验计划：物体重力补偿——伺服下垂是否可修、修了值不值（机制 (b)，不引入 partner）

_Core4D · Phase 57 · 计划态，**待用户批准后才创建脚本 / 跑 CEM** · 承接 [E191](../log/266_E191_object_support_offline_audit_results.md)，与 [E192](218_E192_gate_threshold_size_dependence_plan.md) / [E193](219_E193_grasp_topology_plan.md) 正交_

---

## 📋 Context

E191 把「大箱失败」留在三条与尺寸共线的机制上。E192 打 (a) 阈值、E193 打 (c) 抓取拓扑，
本实验打 **(b) 物体伺服过软**。

### (b) 的事实基础（E191 §5.3，H1b）

CORE4D 的物体不是自由体，而是被 6 个 slide/hinge 关节 + `<position>` 执行器钉在参考轨迹上，
增益由 `examples/run_mjwp.py:1208-1252` 运行时注入（XML 里是 `kp="0"`），**P-only，无重力前馈**：

```yaml
init_pos_actuator_gain: 500.0    # N/m
init_rot_actuator_gain: 50.0     # N·m/rad
guidance_decay_ratio: 1.0        # 不衰减
residual_gain_ratio: 1.0
```

于是稳态下垂 `sag = m·g / kp`。E174 引入的 bucket/desk 让物体质量跨 **2.0–120.4 kg**（box 全钉死 5.0 kg），
构成一个尺寸之外的天然变量，E191 的事后分析 H1b 用它验证了这条公式：

| object | 质量 | m·g/kp 预测 | 实测抬起帧 z 误差 |
|---|---:|---:|---:|
| bucket007 | 2.0 kg | −3.9 cm | **−4.2 cm** |
| bucket010 | 3.0 kg | −5.9 cm | −5.1 cm |
| bucket003 | 3.3 kg | −6.5 cm | −5.1 cm |
| box004 / box001 / box024 / box023 | 5.0 kg | −9.8 cm | −8.5 / −9.2 / −9.7 / −10.4 cm |

ρ = 0.68（p=0.0178, n=12，剔除 69 kg 从未被抬起的 desk007），**观测/预测比中位数 0.93、范围 [0.60, 1.08]**。

### ⚠️ 但 E191 的 H1 被证伪——这决定了本实验的期望上限

**抬起帧的 z 分量只占物体位置总误差的 0.21–0.41，水平分量才是主导。**

所以即使把下垂完全修掉，`track_obj_pos_err_cm_mean` **最多只能降约 1/3**
（box024 13.64 → 预期 ~9–10 cm，而不是 ~0）。本实验必须以此为预期基线，
**任何「修好伺服就解决 object tracking」的说法都与 E191 数据矛盾**。

### 关键 insight：2×2 把「箱子位置错了」和「伺服在推箱子」拆开

`sag = m·g / kp` 有两种改法，**但它们改的不是同一个物理量**。四格稳态解已用 MuJoCo 实测确认
（5.0 kg，slide joint + position actuator，与 CORE4D scene_act 同构）：

| 臂 | `kp` | `gravcomp` | 稳态位移 | **稳态伺服力** |
|---|---:|---:|---:|---:|
| **A0** 基线 | 500 | 0 | **−9.81 cm** | **49.05 N** |
| **G1** 重力补偿 | 500 | 1 | 0.00 cm | **0 N** |
| **G2** 加硬伺服 | 2500 | 0 | −1.96 cm | **49.05 N（不变！）** |
| **G3** 两者 | 2500 | 1 | 0.00 cm | 0 N |

> **位移由 `kp` 决定，力由 `gravcomp` 决定。** 基线里这两件事是混在一起的
> ——箱子既在错误的位置（低 9.8 cm），伺服又一直在用 49 N 顶它。2×2 正好把它们分离。

由此得到本实验最锋利的对照：

| 对照 | 隔离出什么 | 判别什么 |
|---|---|---|
| **G1 vs G2** | 两者都把箱子放回正确位置，**但只有 G1 去掉了 49 N 的伺服力** | 穿透到底是「箱子位置错」造成的，还是「伺服在推」造成的 |
| **G3 vs G1** | 稳态完全相同，只差**柔顺性**（箱子被手推时让不让） | 穿透里有多少来自箱子不肯让位 |
| **G2 vs A0** | 只修位置、不动力 | 位置误差单独的贡献 |

**如果穿透在 G1 下大降、在 G2 下不降** → 穿透是伺服顶出来的（机制 (b) 的强证据）。
**如果两者降幅相同** → 穿透只跟箱子位置有关，与伺服力无关。
**如果两者都不降** → 穿透与 (b) 无关，归因转向 (a)/(c)。

### 三条修法共同的局限

三臂**都在质心作用，都不产生力矩**。所以有一条尖锐预测：

> **三臂都应该消掉均匀下垂，但都不应该修好远端不对称**
> （box024 `obj_side_z_asym_cm = 7.33`）。倾斜来自「一端无支撑的力臂」，
> 要修它必须在**偏心点**施力——那正是 partner 模型，本实验明确不做。

这条如果成立，就把「平移下垂」和「远端倾斜」两个成分**在实验上分离**了，
是本实验最有价值的产出，即使一个 12 门都没修好。

### 边界：不引入 partner

按用户要求，本实验**不做任何 partner 建模**：不开 `partner_force_scale`、不开 `support_proxy_*`、
不加 mocap partner hands、不加 weld anchor、不改 `_apply_partner_force`。

> **命名澄清**：`partner_force_scale` 在无 point/spring 时走的分支（`mjwp.py:3532`）
> 数学上就是「质心向上托力 = scale × m·g」，即部分重力补偿。本实验**不用它**，
> 改用 MuJoCo 原生的 `gravcomp` body 属性——语义干净（是重力补偿，不是"有个人在扶"），
> 且不经过一段以 partner 命名、含 freejoint 硬假设的代码路径。

### 技术可行性（已实测确认，非推断）

MuJoCo **3.7.0**，`body gravcomp` 可用。最小复现：

```
gravcomp=1  稳态 qpos: 0.00000     (期望 ≈ 0)
无 gravcomp 稳态 qpos: -0.10782    (期望 ≈ -mg/kp = -0.0981)
```

**仓库已有先例**，不是新引入的机制：
- `spider/preprocess/generate_xml.py:429-431` — `gravcomp=(1 if len(left_object_files)==0 else 0)`，
  注释即「set gravcomp to 1 to avoid gravity」
- `spider/assets/robots/{allegro/right.xml:172, allegro/left.xml:151, mano/right.xml:82}` — 手部 palm 用 `gravcomp="1"`

当前 box024 的 `scene_act_E173_rubberHull_PRG.xml:339` 的 object body **没有** `gravcomp` 属性（默认 0），
所以 G1 就是在 scene sidecar 上加一个属性——与 E170 PRG 注入 `<contact><pair>` 完全同款的做法。

---

## 🎯 Claims

| # | Claim | 最低证据（预注册，事后不得调整） |
|---|---|---|
| C1 | 重力补偿/加硬伺服确实消掉伺服下垂 | G1/G3 下 `track_obj_z_err_m_lifted_mean` 绝对值 ≤ **0.02 m**；G2 下 ≤ **0.04 m**（基线 box004 −0.085 / box024 −0.097）|
| C2 | **下垂只是误差的一小部分**（复核 E191 H1） | G1 下 `track_obj_pos_err_cm_mean` 降幅在 **[15%, 45%]** 区间内。若降幅 > 60%，说明 E191 的 z/xy 分解有误，需回查 |
| C3 | **质心施力修不好远端倾斜** | 三臂下 box024 `obj_side_z_asym_cm` 相对基线 7.33 的改善均 < **2.0 cm**（即倾斜基本保留）|
| C4 | **重力补偿的代价：承重接触是否消失**（核心风险） | G1/G3 下 `hand_object_physics_contact_3mm_in_mask_frac` ≥ **0.3079**(box024) / **0.4507**(box004)。注意 `object_guidance_force_z_N_p95` 在 G1/G3 下**预期就该趋近 0**（实测稳态 0 N），因此**不能**再用它当承重判据——改用 3mm 硬接触占比 |
| C5 | **加硬伺服的代价：柔顺性下降是否推高穿透** | G2 下 `hand_object_physics_penetration_3mm_frame_frac` 相对基线升幅 ≤ **0.05**；若显著升高，则「箱子不肯让位」被证实为穿透来源之一 |
| **C6** | **主判别：穿透是「位置错」还是「伺服顶」造成的** | G1 与 G2 的穿透降幅之差 ≥ **0.08** → `PENETRATION_FROM_SERVO_FORCE`；差 < 0.03 → `PENETRATION_FROM_POSITION_ONLY`；两者都不降（降幅 < 0.05）→ `PENETRATION_NOT_FROM_B` |
| **C7** | **交互项：柔顺性的独立贡献** | G3 vs G1 稳态完全相同，唯一差别是 `kp`。若 G3 穿透显著高于 G1（差 ≥ 0.05），则柔顺性单独有害；若持平，则 `kp` 在有重力补偿时无影响 |
| C8 | 不产生跨门回退 | 三臂下 box004 六例 12 门任一门不得由 PASS→FAIL；`leg_penetration_frac` 升幅 ≤ 0.05 |
| C9 | 数值稳定 | 无新增 `fall_flag`；`qpos_jerk_l2_p95` 相对基线升幅 ≤ 30%（失重物体 + 硬弹簧都有抖动风险）|

> C2/C3/C4 是**预注册的负向预期**（预期「修不好」或「可能变差」），若结果反而正向，同样是重要发现，照实记录。
> C6/C7 才是本实验真正的判别项——它们只有靠 2×2 三臂齐全才能判。

---

## ⚙️ 实验设计

**冻结不变量**：PRG **开启**（生产默认）；E167A_zOnlyBody profile；`rubber_hull`；
CEM `seed=0 / 1024 samples / 32 opt steps`；**所有 gate/reward 阈值保持 E167A 原值**
（不与 E192 叠加）；**抓握目标保持 `ref_fk`**（不与 E193 叠加）。

| 臂 | 改动 | 载体 | 需要跑吗 |
|---|---|---|---|
| **A0** 基线 | — | — | **否** — E172(box004×6) + E173(box024×9) 已落盘 |
| **G1** 重力补偿 | object body `gravcomp="1"` | **scene sidecar** `scene_act_E194_rubberHull_PRG_gravcomp.xml` | 15 例 |
| **G2** 加硬伺服 | `init_pos_actuator_gain: 500 → 2500` | **纯 Hydra override** | 15 例 |
| **G3** 两者 | `gravcomp="1"` + `init_pos_actuator_gain: 2500` | sidecar + override | 15 例 |

**G1/G3 复用同一个 scene sidecar**，G2/G3 复用同一个 override 片段——所以只需要造 1 个 sidecar + 1 个 override 片段，
三臂由它们的组合构成。这也保证了「同一个 XML、同一个数值」，不会出现两臂的 gravcomp 实现不一致。

三臂**都只动平移增益**，`init_rot_actuator_gain` 保持 50 不动——因为 C3 要检验「质心施力修不好倾斜」，
若同时加硬旋转伺服就混淆了（旋转伺服变硬确实会压制倾斜，但那不是重力补偿的功劳）。

### Case 集合（15 例，与 E191/E192 完全一致，便于跨实验对照）

- **box024 × 9**：`20231011_{026_p1, 026_p2, 027_p1, 027_p2, 028_p1, 028_p2, 030_p1, 031_p1, 031_p2}`
- **box004 × 6**：`20231003_2_{082_p1, 082_p2, 083_p1, 083_p2, 086_p1, 086_p2}`

两个物体质量相同（5.0 kg），所以 G1 对两者的下垂修正量应当**相同**（约 9.8 cm）——
这本身是一条内部一致性检查：若 G1 对 box004 和 box024 的 z 误差修正量差异 > 2 cm，说明有别的东西在动。

### 预算

**45 条 Full CEM**（3 臂 × 15 例），与 E189 的 43 条同量级。8×L20Y 本地，按 `assigned_gpu` 分片。
先跑 **canary**（`64 × 4`）6 例：box024 `026_p1`（最差）/ `027_p2`（最好）+ box004 `082_p1`，**三臂各 2 例**，
确认 sidecar 与 override 都生效、且失重物体不发散，再放 Full。

> 三臂齐全是 C6/C7 的**必要条件**——少任何一臂都无法判「穿透来自位置错还是伺服顶」。

---

## 📊 基线（A0，已落盘，只读）

| 指标 | box004 (E172, n=6) | box024 (E173, n=9) |
|---|---:|---:|
| 12 门通过 | 2/6 | 2/9 |
| `track_obj_z_err_m_lifted_mean` (E191 新列) | −0.0826 | −0.1007 |
| `track_obj_xy_err_cm_lifted_mean` (E191 新列) | 17.6505 | 18.2009 |
| `track_obj_z_err_share_lifted` (E191 新列) | 0.3515 | 0.3692 |
| `track_obj_pos_err_cm_mean` | 12.0105 | 13.6421 |
| `obj_side_near_z_err_m` / `far` (E191 新列) | −0.0730 / −0.0846 | −0.0681 / −0.1414 |
| **`obj_side_z_asym_cm`** (E191 新列) | **1.1588** | **7.3322** |
| `object_guidance_force_z_N_p95` (E191 新列) | 106.09 | 79.59 |
| `object_guidance_torque_Nm_p95` (E191 新列) | 63.56 | 14.18 |
| `object_weight_N` | 49.05 | 49.05 |
| `hand_object_physics_penetration_3mm_frame_frac` | 0.1469 | 0.3776 |
| `hand_object_physics_contact_3mm_in_mask_frac` | 0.4507 | 0.3079 |
| `hand_object_physics_contact_in_mask_frac` | 0.6944 | 0.8188 |
| `leg_penetration_frac` | 0.0565 | 0.0386 |

全部取自 **PRG 侧**（E172/E173）的 E191 审计表，与本实验的 A0 基线同口径。
注意 `z_err_share_lifted` 只有 0.35/0.37 —— 这就是 C2 把降幅上限压在 45% 的依据。

### 必报指标

12 门全部 + 上表全部 + E191 的 22 个 support 列 + `qpos_jerk_l2_p95`。
逐物体分别报，**禁止合并成单一跨物体结论**。配对统计：McNemar（门级）+ bootstrap CI（连续量），种子 0、10000 次。

---

## 🛡️ 成功标准与 stop-loss

### 判决

**主判决由 C6 给出**（穿透的来源），其余 claim 决定它是否可用：

| C6 结果 | 判定 |
|---|---|
| G1 降幅 − G2 降幅 ≥ 0.08 | `PENETRATION_FROM_SERVO_FORCE` — 穿透主要是**伺服在顶箱子**造成的。机制 (b) 得到强证据，重力补偿进入候选 |
| 两者降幅之差 < 0.03（且都降） | `PENETRATION_FROM_POSITION_ONLY` — 只跟箱子位置有关，与伺服力无关。加硬伺服（不改物理真实性）即可，无需重力补偿 |
| 两臂降幅均 < 0.05 | `PENETRATION_NOT_FROM_B` — 穿透与 (b) 无关，归因压到 (a)/(c)，E192/E193 优先级上调 |

叠加判定：

| 结果 | 判定 |
|---|---|
| C1 过 + C3 过 + C4 过 | `SAG_FIXABLE_TILT_NOT` — **预期主线**。平移下垂可修、远端倾斜不可修，两成分实验分离 |
| C1 过 但 C4 不过 | `GRAVCOMP_HARMFUL` — 重力补偿把承重接触也消掉了，**G1/G3 不采用**；同时给 R018 §9.6③「只跟不抬」提供直接因果证据（此时若 C6 判 `PENETRATION_FROM_POSITION_ONLY`，G2 反而是唯一可用的修法）|
| C7：G3 穿透 ≥ G1 + 0.05 | 柔顺性单独有害 → 即使采用重力补偿，**也不要加硬 kp** |
| C7：G3 ≈ G1 | `kp` 在有重力补偿时无影响 → 后续统一用 G1（改动面更小）|
| C1 过 但 C3 不过（倾斜也修好了） | 意外结果，需回查：质心施力不应产生力矩，可能是 `gravcomp` 与接触耦合的二阶效应，**必须查清才能采信** |
| C1 不过 | `SAG_NOT_SERVO` — 下垂不是 `m·g/kp` 导致的，E191 H1b 被推翻，需重新归因 |
| C2 降幅 > 60% | E191 的 z/xy 分解有误，**先修 E191 的度量再谈结论** |

### Stop-loss

- canary 若 ≥2 例出现新的 `fall_flag` 或 CEM 发散，**停止**。失重物体 + 位置伺服是已知的潜在数值不稳定组合。
- 若 canary 上 G2 穿透暴涨（升幅 > 0.10）：**不终止 G2**——它对 C6/C7 是必需的对照臂，
  「G2 有害」本身就是一个有信息量的结果。只在 log 中标注并降低 G2 的推荐度。
- 若 canary 上 box004 三臂全部出现新的 12 门 FAIL，说明改动有全局代价，
  **降级为 box024-only（27 条）**，并记录 C8 失败；此时判决保留但可信度下调。

---

## 🔧 拟新增文件（**批准后才创建**）

| 文件 | 用途 |
|---|---|
| `workspace/core4d/scripts/experiments/E194/e194_common.py` | 冻结 contract：G1/G2/G3 三臂定义、15 例 case 集、CEM 预算、`E194_METHOD_ID` |
| `workspace/core4d/scripts/experiments/E194/build_gravcomp_manifest.py` | 生成 **1 个** scene sidecar（object body 加 `gravcomp="1"`，照 `E170/build_box021_prg_manifest.py` 的注入模式）+ **1 个** kp override 片段，三臂由二者组合而成；共 45 条 manifest 行 |
| `workspace/core4d/scripts/experiments/E194/audit_gravcomp_scenes.py` | **逐臂正向审计**：G1/G3 的 `body_gravcomp[object] == 1.0` 且**其余 model 字段与 A0 逐字段相同**（geom/mass/inertia/pair 全不变）；G2/G3 的 `init_pos_actuator_gain == 2500`；**三臂的 `init_rot_actuator_gain` 必须全为 50**；G1 的 kp 必须仍为 500 |
| `workspace/core4d/scripts/launch/active/run_E194_local_8gpu.sh` | canary / full（照 E189，`unset MUJOCO_GL`）|
| `workspace/core4d/scripts/eval/runners/eval_E194_gravcomp_arms.py` | A0 / G1 / G2 / G3 四格配对评测 + C6/C7 的臂间差值统计 |
| `workspace/core4d/scripts/eval/wrappers/eval_E194_gravcomp_arms.sh` | shell 入口 |
| `workspace/core4d/scripts/eval/reports/gen_E194_arm_comparison.py` | 报表 + C1–C9 逐条判定 + 2×2 交互表 |
| `workspace/core4d/results/E194/scene_snapshot/` | **训练前必做**：`snapshot_scenes.sh E194 <15 cases>`；G1 改了 XML，**必须重新快照** |

## 🚀 执行入口

```bash
# 0. scene 快照（第一步，强制；G1 会新增 sidecar，构建后需再快照一次）
bash workspace/core4d/scripts/convert/snapshot_scenes.sh E194 <15 cases>

# 1. 构建 + 审计
.venv/bin/python workspace/core4d/scripts/experiments/E194/build_gravcomp_manifest.py --apply --snapshot
.venv/bin/python workspace/core4d/scripts/experiments/E194/audit_gravcomp_scenes.py --require-all

# 2. canary → full
MODE=canary bash workspace/core4d/scripts/launch/active/run_E194_local_8gpu.sh
MODE=full   bash workspace/core4d/scripts/launch/active/run_E194_local_8gpu.sh

# 3. 评测 + 报表 + 渲染
bash workspace/core4d/scripts/eval/wrappers/eval_E194_gravcomp_arms.sh full
```

**结果路径**：`workspace/core4d/results/E194/s6_downstream/{cem,eval,render}/`

## 👁️ 可视化（强制）

- 45/45 self MP4 + 与 A0 的 paired 对照。
- 用 `/video-frames` 抽帧（首次接触 / 抬起峰值 / 搬运中段 / 放下），三件事必须逐帧确认：
  1. **箱体整体高度**是否回到参考位（C1 的视觉对应）
  2. **远端是否仍然下栽**（C3 的视觉对应）—— 这是本实验最关键的一张对照图
  3. **手是不是变成"扶着"而不是"托着"**（C4 的视觉对应）—— 失重物体下手可能只是虚搭
- box024 `026_p1`（基线远端 −17.6 cm，最差）必须**四格全看**（A0/G1/G2/G3 并排），并与 E191 已存的
  `results/E191/audit/frames/b024_026p1_2.4s.png` 直接对比。

## 🚫 Non-goals

- **不引入任何 partner 建模**：不开 `partner_force_scale` / `support_proxy_*` / `mocap_partner_trajectory` /
  `support_dynamic_*` / weld anchor，不改 `_apply_partner_force`。
- **不做偏心支撑** —— 远端倾斜本实验预期修不好（C3），修它需要偏心施力，属另一个实验。
- 不动 `init_rot_actuator_gain`（三臂全部保持 50，否则混淆 C3）。
- 不动 `init_rot_actuator_gain`（会混淆 C3）。
- 不测 (a) 阈值（E192）、不测 (c) 抓取拓扑（E193）。**三个实验互不叠加。**
- 不做增益 sweep：`kp` 只取 500/2500 两点，不搜参数空间；`gravcomp` 只取 0/1，不做部分补偿。
- 不改物体质量（5.0 kg 保持不动，那是另一条改法）。
- 不下 RL 结论：12 门 MJ-replay 口径，**不等于** RL/Holo 结论。

## ✅ 批准前 checklist

- [ ] 用户确认 `gravcomp="1"`（全补偿）而非部分补偿——部分补偿（如 0.5）数学上等价于
      「质心处的部分 partner 托力」，全补偿 1.0 语义更干净，也让 2×2 是真正的 on/off
- [ ] 用户确认 `init_pos_actuator_gain = 2500`（5×）这个取值
- [ ] 用户确认 45 条 Full CEM 的算力预算
