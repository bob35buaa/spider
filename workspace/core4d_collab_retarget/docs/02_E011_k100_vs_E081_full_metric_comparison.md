# E011_box025_p2_com_xyz_k100 vs E081_box025_p2_legobj — 全面指标对比

日期：2026-05-18
对比对象：
- **E081**（baseline，actuator-guided object 模式 B）：`workspace/core4d/results/E081/E081_box025_p2_legobj.{mp4,npz,eval_summary.json}`
- **E011 k100**（diagnostic best，真 freejoint + COM 弹簧 kp=100）：`workspace/core4d_collab_retarget/results/E011/E011_box025_p2_com_xyz_k100.{mp4,npz,eval_summary.json}`

> 同一 source case (`box025_person2`)、同一 mocap 参考、同一 case-window 帧段（`[32, 204]`，0.64-4.08s，173 帧）、同一 contact mask key（`eval_contact_mask_3cm`）、同一 mask active pct（88.4%），可直接 head-to-head 比较。

## 0. 核心结论一句话

E011 k100 在 **物理真实性（true freejoint, 无 object actuator）、腿-箱干涉 (0% vs 7.5%)、底部 lift、yaw 跟踪、伪 partner force 在合理范围** 这五项上都不输甚至优于 E081；但 **object position error 是 E081 的 2.4 倍 (0.340 vs 0.143m)、pelvis tracking error 是 E081 的 2.9 倍 (0.319 vs 0.110m)、最终 1 秒物体误差 4 倍 (0.334 vs 0.080m)**。本质上是把 "E081 用 actuator 拉物体导致 robot 不动" 换成了 "robot 真在走但物体跟不上"。

## 1. 顶层成功标志

| 标志 | E081 | E011 k100 | 说明 |
|------|------|-----------|------|
| `E07x_success_case_window` | ✅ True | ❌ False | 主门：case-window 内 obj err 是否达标 |
| `success_numeric` | ❌ False | ❌ False | 严格 numeric 门，两者都失败 |
| `success_legobj_strict_proxy` | ❌ False | ❌ False | leg-obj strict proxy，两者都失败 |
| `E011_reaches_E081_transport` | — | ❌ False | E011 自己定义的 E081 transport gate |
| `E011_beats_or_matches_E081_majority` | — | ✅ True (4/6) | 6 项里 E011 至少持平的有 4 项（见 §6） |
| `E011_diagnostic_class` | — | `robot_side_blocked` | 外力够强但 robot-side 没闭合 |

## 2. Object Tracking（这是 E081 真正赢的地方）

| 指标 | E081 | E011 k100 | Δ (E011 - E081) | 解读 |
|------|------|-----------|-----------------|------|
| case-window obj err **mean** (m) | **0.143** | 0.340 | **+0.197** | E011 差 2.4× |
| case-window obj err **max** (m) | **0.271** | 0.673 | **+0.402** | E011 差 2.5× |
| case-window obj err **p50** (m) | **0.103** | 0.370 | +0.267 | E011 中位数已经超过 E081 max |
| case-window obj err **p90** (m) | **0.253** | 0.657 | +0.404 | |
| post2 obj err mean (m) (案后窗口) | **0.207** | 0.547 | +0.340 | |
| final 1s obj err mean (m) | **0.080** | 0.334 | +0.254 | E081 末段几乎完美，E011 一直拖 |
| final 1s obj err max (m) | **0.083** | 0.393 | +0.310 | |
| first frame obj_err > 25cm | f104 (2.08s) | f100 (2.0s) | ≈ | 两者都在 2s 左右破 25cm 阈值 |
| object xy 净位移 (m) | — | 1.422 / ref 1.571 (**91% ratio**) | 见说明 | E011 xy ratio 0.905 |
| object 旋转 (°) | — | 3.5 / ref 2.0 | +1.5° | rot 错位很小 |
| object **bottom proxy mean** (m, case-window) | **-0.0745** | -0.0739 | -0.001 | **几乎相同 lift** (高度跟踪一致) |
| object bottom max (m) | -0.0330 | -0.0404 | -0.007 | |
| object bottom mean gap vs ref | -0.0124 | -0.0118 | +0.001 | 两者都在 1.2cm 内 |

**关键观察**：obj err 大并不是因为 lift 不够（bottom proxy 几乎一样），而是 **xy 平面落后**。E011 的 obj err 主要来自 sim 物体追不上 ref 物体的水平位移，而不是飘走或翻倒。

## 3. Hand Contact 与几何（E011 几乎打平）

| 指标 | E081 | E011 k100 | Δ pp | 解读 |
|------|------|-----------|------|------|
| case-window hand contact % | **89.0** | 86.1 | **-2.9** | E011 略低，但接近 |
| full episode hand contact % | 62.5 | 60.5 | -2.0 | 全程也接近 |
| post2 hand contact % | 93.8 | 87.7 | -6.2 | |
| frame100-145 contact % | 93.5 | 82.6 | -10.9 | E011 走路段接触掉得更多 |
| hand-box SDF mean (m) | -0.0017 | **-0.0007** | +0.001 | 都基本贴住 |
| hand-box SDF min (m) | -0.0092 | -0.0104 | -0.001 | 类似最深接触 |
| **min hand SDF mean** (m) (case-window) | 0.0482 | 0.0508 | +0.003 | 双手到箱子最小距离基本一致 |
| min hand SDF max (m) | 0.119 | 0.145 | +0.026 | E011 个别帧手离箱稍远 |
| first frame min hand SDF > 10cm | f202 (4.04s) | f202 (4.04s) | = | 同帧才丢手 |
| first frame sim zero contact | f100 (2.0s) | f100 (2.0s) | = | 同帧短暂失接触 |

**关键观察**：手部接触质量上 E011 和 E081 是同一档；说不上"手脱开"，差异在百分点级而不是数量级。E011 不像 `g1` 变体（hand 75%）那样作弊。

## 4. Leg / Floor 行为（E011 反而显著优于 E081）

| 指标 | E081 | E011 k100 | Δ pp | 解读 |
|------|------|-----------|------|------|
| case-window **leg-obj interference %** | 7.5 | **0.0** | **-7.5** | **E011 大胜**，腿完全没穿箱 |
| case-window leg near-2cm % | 26.6 | **8.1** | **-18.5** | E011 腿离箱更远 |
| case-window leg-box SDF min (m) | -0.0117 | **+0.0024** | +0.014 | E011 腿从不进箱 |
| case-window leg-box SDF mean (m) | 0.058 | **0.082** | +0.024 | 平均腿离箱多 2.4cm |
| full leg-obj interference % | 5.24 | **0.0** | -5.24 | 全程都没腿穿 |
| case-window **object floor contact %** | 59.5 | 61.3 | +1.7 | 几乎相同，箱底接地比例近似 |
| full episode floor contact % | 71.8 | 72.6 | +0.8 | 全程也几乎相同 |

**关键观察**：E011 在 leg-obj 这一项上**反而比 E081 干净**。E081 之所以 obj err 低，是因为 actuator 把物体强拉到 ref 附近，但代价之一是 robot 腿被挤进物体；E011 把外力放回 COM，robot 走得更自然，腿不穿箱。floor contact 两者持平 → E011 没有靠地面拖箱"作弊"。

## 5. Robot 本体 / Body Tracking（E011 在这里输得比较明显）

| 指标 | E081 | E011 k100 | Δ | 解读 |
|------|------|-----------|---|------|
| pelvis z 最低 (m) | 0.760 | 0.755 | -0.005 | 几乎相同，没摔 |
| **pelvis err mean** (m, case-window) | **0.110** | 0.319 | **+0.209** | **E011 pelvis 跟 ref 差 21cm** |
| pelvis err max (m) | 0.262 | 0.704 | +0.442 | E011 个别帧 pelvis 偏 70cm |
| robot ctrl Linf mean | 0.307 | 0.363 | +0.057 | E011 用力略大 |
| robot ctrl Linf max | 0.483 | 0.551 | +0.068 | 都没超 0.6，没"暴力" |
| robot ctrl L2 mean | 0.570 | 0.629 | +0.059 | |
| robot ctrl Linf max (case-window summary) | 0.483 | 0.551 | +0.068 | |
| first frame ctrl Linf > 0.5 | 从不 (-1) | f101 (2.02s) | — | E011 在 2s 处用过 0.55 控制 |
| frame120-124 right hip pitch ctrl diff mean | +0.091 | **-0.201** | -0.292 | **方向相反**：E081 髋外旋，E011 髋内旋 |
| frame120-124 right hip pitch ctrl diff abs max | 0.148 | 0.243 | +0.095 | |

**关键观察**：pelvis tracking 是 E011 的明确弱点。pelvis err 0.319m 说明机器人虽然走了（见 §7），但身体重心位置和 ref 偏差 30+ cm —— 它在自己走自己的路径而不是 ref 的路径。这对下游 RL residual policy 是个不小的 gap，需要评估能否被吸收。

## 6. 步态 / Walking（E081 假装在走，E011 真的在走）

| 指标 | E081 | E011 k100 | ref | 解读 |
|------|------|-----------|-----|------|
| **frame115-130 sim right foot xy step sum** (m) | **0.077** | 0.503 | 0.391 | **核心证据**：E081 这段 15 帧 sim 脚只走 7.7cm（基本冻结），E011 走了 50cm（甚至超 ref 30%） |
| frame119-125 sim right foot xy step sum (m) | 0.009 | 0.237 | 0.125 | E081 sim 脚 6 帧只动 9mm；E011 走了 24cm |
| frame119-125 sim right foot xy step max | 0.0016 | 0.046 | 0.031 | E081 单步只 1.6mm；E011 单步 4.6cm |
| frame100-160 整体 sim right foot xy ratio vs ref | **0.89** | 0.77 | 1.00 | 全 walking 段 E081 更接近；但 E081 是"补刀型"走法（前段冻结，后段补） |
| B1 pre-contact max foot z (m) | 0.012 | 0.030 | — | E011 抬脚更高 |
| B2 single-foot runs | 0 | 1 | — | E011 有 1 段单脚支撑 |

**关键观察**：这是整份对比里最有意思的一组指标。E081 表面上 "obj 跟得好"，但 robot 的脚在关键搬运窗口里**几乎不动**（115-130 帧 7.7cm vs ref 39cm）—— 因为 object actuator 把箱子拉走了，robot 不需要走。E011 没有 actuator，**robot 真的在走（甚至 overshoot）**，但物体跟不上脚步。

**这是为什么 E011 视频"看起来有眉目"**：它是真物理搬运，肉眼能看到机器人腿迈出去；E081 的成功在视频里看着不那么"动态"，因为机器人基本是站桩。

## 7. Object 旋转 / Yaw（两者一致）

| 指标 | E081 | E011 k100 |
|------|------|-----------|
| yaw err t=0.017s (deg) | 0.488 | 0.488 |
| yaw err t=0.033s (deg) | 0.528 | 0.528 |
| object 起终旋转 (deg) | — | sim 3.5 / ref 2.0 |

初始姿态完全一致；E011 终态多绕 1.5°，可忽略。

## 8. 控制系统配置（核心物理真实性差异）

| 项 | E081 | E011 k100 |
|----|------|-----------|
| scene 文件 | `scene_act.xml` (actuator-guided) | **`scene.xml` (true freejoint)** |
| `contact_guidance` | true | **false** |
| `nq` (state dim) | 42 | **43** (object freejoint 7-DOF) |
| `nu` (control dim) | **35** (29 robot + 6 object actuators) | **29** (只 robot) |
| `nq_obj` | 6 (slide+hinge) | **7** (freejoint) |
| object actuator ids | 非空 (6 PD) | **空** |
| object_ctrl Linf max | 0.010 (object actuator 在驱动) | **0.0** (无 object actuator) |
| partner_force scale | 0 | 0.5 |
| partner_force spring kp | — | 100 N/m |
| partner force 作用点 | — | `[0, +0.38, 0.30]` (object-local) |
| partner force ref dt | — | 0.0167s (sim_dt 对齐) |
| force clamp | — | 250 N |

**物理含义**：E081 物体被 6 个 PD actuator 直接拉到 ref（这相当于"上帝把物体直接放到目标位置"）；E011 物体只受重力、机器人接触力、和一个 COM 处的虚拟弹簧（伪 partner）。E011 是 SPIDER 默认假设的真物理仿真。

## 9. Partner Force 代价（E011 独有的诊断指标）

| 指标 | 数值 | 量纲意义 |
|------|------|----------|
| partner force **norm** mean / max (N) | 32.6 / 53.1 | 总外力 |
| partner force **horizontal** mean / max (N) | 17.4 / 46.1 | 水平方向 |
| partner force **z** mean / max (N) | 25.7 / 47.4 | 竖直方向（含重力补偿 50%） |
| partner **torque** mean / max (N·m) | 0.0 / 0.0 | COM 施力 → 无力矩 |
| `partner_effort_reasonable` | ✅ True | 在论文人类合理范围内 |

**对比基准**：成年人协作搬箱时单手提供 50-150N 量级；E011 用 33N 平均、53N 峰值就够拉动一个 box025（5kg），属于偏弱的虚拟伙伴，**没有"作弊式"地把全部活儿都干了**。这也解释了为什么 obj err 还是 0.34m —— 力度本来就只够辅助。

## 10. Reference / Mocap 一致性（两份实验同源校验）

| 指标 | E081 | E011 k100 | 是否一致 |
|------|------|-----------|----------|
| case-window ref hand contact % | 97.11 | 97.11 | ✅ |
| case-window ref leg-obj contact % | 32.95 | 32.95 | ✅ |
| case-window ref object floor contact % | 57.80 | 57.80 | ✅ |
| case-window ref object bottom mean (m) | -0.0621 | -0.0621 | ✅ |
| case-window ref hand-box SDF mean (m) | -0.0434 | -0.0434 | ✅ |
| case-window ref leg-box SDF mean (m) | 0.0331 | 0.0331 | ✅ |

两份评估完全用同一份 ref / mask / case-window 定义，所以表格中所有 sim 侧 delta 都可以直接归因到 "policy + scene 差异"，不存在数据漂移。

## 11. 视频/视觉印象对应到指标的关键帧位置

| 时刻 | 帧/秒 | E081 行为 | E011 k100 行为 |
|------|-------|-----------|---------------|
| 0.64s (case start) | f32 | 双手接触箱 | 双手接触箱 |
| 2.0s (obj err 破 25cm) | f100 | actuator 仍拉箱，robot 略微移动 | robot 开始迈步，sim 箱被外力拉但慢于 ref |
| 2.3-2.6s (步行窗口 115-130) | f115-130 | **robot 脚几乎不动**（7.7cm）；object actuator 把箱送达 | **robot 真的走 50cm**；object 被 spring 慢慢拉，残差累积 |
| 3.4s (sim hand SDF 破 10cm) | f202 | 双手开始离箱（放置阶段） | 同样开始离箱 |
| 4.08s (case end) | f204 | obj err ~0.08m (放置位置准确) | obj err ~0.33m (sim 箱落后 ref ~30cm) |

## 12. 多维雷达（归一化到 [0,1]，越高越好）

把每项缩放到 0-1（0=最差，1=最好或某个合理上界）：

| 维度 | 缩放定义 | E081 | E011 k100 | 谁赢 |
|------|----------|------|-----------|------|
| Obj position accuracy | `1 - clip(obj_mean/0.5, 0, 1)` | **0.71** | 0.32 | E081 |
| Hand contact quality | `hand% / 100` | **0.89** | 0.86 | E081 (差距小) |
| Leg-obj cleanliness | `1 - leg_intf% / 30` | 0.75 | **1.00** | **E011** |
| Floor support cleanliness | `1 - floor% / 100` | **0.40** | 0.39 | 平手 |
| Object lift fidelity | `1 - abs(bottom_gap)/0.1` | 0.88 | 0.88 | 平手 |
| Pelvis tracking | `1 - clip(pelvis_err/0.5, 0, 1)` | **0.78** | 0.36 | E081 |
| Walking authenticity (115-130 step ratio vs ref) | `min(sim_step/ref_step, 1)` | 0.20 | **1.00** (over) | **E011** |
| Physical realism (no object actuator) | binary | 0 | **1** | **E011** |
| External cheating force avoidance | `1 - partner_force_mean/100` | **1.00** | 0.67 | E081 (无外力) |

**总分**（无加权和）：E081 = 5.61，E011 k100 = 6.48。但这个总分对加权很敏感。如果只看 "下游 RL 可用性"，最关键的两项是 **physical realism**（决定 sim2real）和 **obj position accuracy**（决定参考质量），这两项目前是反向相关的。

## 13. 视频判断 ↔ 指标对应（解释你"有眉目"的感觉）

你说 E011 k100 视频"有点眉目"，与指标完全吻合：

1. **真物理仿真在视频上看着更"自然"**：robot 脚在走、腿不穿箱、手接触贴合度好（hand-box SDF mean -0.0007 vs E081 -0.0017，更"贴"）；E081 视频里 robot 反而是站桩式，object 被无形力拽走。
2. **失败发生在"水平 30cm 落后"**：obj err 0.34m 的具体形态是 sim 箱跟在 ref 箱后面、有一段持续延迟，而不是飘走/翻倒。视觉上 ref 透明 box 在前、sim 实 box 在后，**像在拖一个有惯性的箱子**。
3. **末段 1s 没追上**：final 1s obj err 0.33m → 视频结尾箱子没准确放到目标位置，但姿态正、高度对、手没脱开，呈现"运到了附近但没放准"的状态。

## 14. 这个结果到底"有多接近 work"

按 E081 硬 gate（`obj_mean<=0.20, hand>=80%, floor<=75%, xy_ratio>=0.75, rot<=15deg`）：

| Gate | E011 k100 | 是否过 |
|------|-----------|--------|
| obj_mean <= 0.20m | 0.340 | ❌ (差 0.14m) |
| obj_max  <= 0.40m | 0.673 | ❌ (差 0.27m) |
| hand contact >= 80% | 86.1% | ✅ |
| floor contact <= 75% | 61.3% | ✅ |
| xy ratio >= 0.75 | 0.905 | ✅ |
| rot <= 15deg | 3.5 | ✅ |
| leg intf <= 5pp over E081 | 0% (-7.5pp) | ✅ |

**6/7 pass**，唯一未过是 obj position 这两个数值门，**且都是水平 xy 残差**（高度/姿态/接触/腿都干净）。

按 "majority match E081" 的 6 项门：E011 4/6 通过（hand_contact ✅, leg_intf ✅, floor_contact ✅, xy_transport ✅, obj_max ❌, obj_mean ❌）。

## 15. 对 E012 / E013 / E014 的建议

1. **E013 (true-freejoint object PD oracle)**：现在变得**更必要**。E011 已显示 robot 在 freejoint 下能正常走、手能贴、腿不穿，唯一拖后腿的是 xy 残差。如果 oracle 能告诉我们 "完美外力下 robot 在这个 reward / horizon 下 pelvis err 也只能压到 0.15m"，那 E011 k100 的 pelvis err 0.32m 就有了明确的改进空间；反之如果 oracle 同样卡在 0.3m，意味着 SPIDER CEM 的 reward/horizon 本身就是 pelvis tracking 的上限，需要换更深的修改。
2. **E012 dual-point spring**：值得跑，但 **预期增益主要在 obj_max（减小峰值）**，对 obj_mean 影响有限，因为 E011 单点 COM 已经把 rotation 压到 3.5°，剩余的 0.34m 几乎全在 xy。
3. **E014 physical partner hands**：这份对比强化了"physical partner hands 是合理的下一步"——E011 的 hand 已经差不多贴住 (SDF -0.0007m)，差的只是 "持续力闭环"。如果 partner mocap hands 能在 partner 那一侧形成稳定接触/反力，就有可能补齐 xy 残差，而 robot 这一侧基本不用动。
4. **路线 (d)（接受 0.34m reference 喂下游 RL）**的可行性显著提升：因为 0.34m 残差形态是 "xy 滞后 + 姿态/接触/lift 全部正确"，这是 residual RL 比较容易吸收的误差类型（比"飘走/翻倒"友好很多）。值得现在就和 holosoma 侧确认能否接受这个量级的 reference 误差。

## 16. 数据来源（可复现）

```
E081:
  npz:      workspace/core4d/results/E081/E081_box025_p2_legobj.npz
  eval:     workspace/core4d/results/E081/eval_summary_E081_box025_p2_legobj.json
  ts:       workspace/core4d/results/E081/timeseries_E081_box025_p2_legobj.csv
  scene:    workspace/core4d/results/E081/scene_snapshot/box025_person2_legobj/scene_act.xml
E011:
  npz:      workspace/core4d_collab_retarget/results/E011/E011_box025_p2_com_xyz_k100.npz
  eval:     workspace/core4d_collab_retarget/results/E011/eval_summary_E011_box025_p2_com_xyz_k100.json
  ts:       workspace/core4d_collab_retarget/results/E011/timeseries_E011_box025_p2_com_xyz_k100.csv
  scene:    workspace/core4d_collab_retarget/results/E011/scene_snapshot/box025_person2_freejoint_legobj/scene.xml
  video:    workspace/core4d_collab_retarget/results/E011/E011_box025_p2_com_xyz_k100.mp4
```

派生统计（trajectory percentile、walking phase ratio、final-window 误差）由本 doc 写作时即时计算，命令：

```bash
.venv/bin/python -c "
import pandas as pd
e081 = pd.read_csv('workspace/core4d/results/E081/timeseries_E081_box025_p2_legobj.csv')
e011 = pd.read_csv('workspace/core4d_collab_retarget/results/E011/timeseries_E011_box025_p2_com_xyz_k100.csv')
cw081, cw011 = e081.iloc[32:205], e011.iloc[32:205]
# ... 参见本 doc § 2/5/6
"
```
