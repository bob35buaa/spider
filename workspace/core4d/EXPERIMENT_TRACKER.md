# CORE4D 动力学重定向实验跟踪器

## 实验总览

| Run | 日期 | Phase | 描述 | 状态 |
|-----|------|-------|------|------|
| E001 | 2026-04-30 | Phase 0 | 数据管线: holosoma → SPIDER 格式, Box025 场景 XML | 通过 |
| E002 | 2026-04-30 | Phase 1 | SPIDER MJWP 无引导 (Box025 p1): pelvis=0.10m, obj=0.83m (物体落地) | 完成 |
| E003 | 2026-04-30 | Phase 1 | SPIDER MJWP 有引导 (Box025 p1): pelvis=0.07m, obj=0.83m (物体仍落地) | 完成 |
| E004 | 2026-04-30 | Phase 1 | 强增益 kp=100/1000, decay=1.0: 物体仍落地 → 确认根因: 最终迭代归零设计 | 完成 |
| E005 | 2026-04-30 | Phase 4 | 混合轨迹导出: SPIDER机器人(物理)+运动学物体 → Holosoma格式 (52body,50fps) | 通过 |
| E006 | 2026-05-01 | Phase 1 | 前臂接触重定向: 3-box碰撞+contact_rew+高权重 → **视频证实箱子未离地** | 虚假突破(更正) |
| E007 | 2026-05-01 | Phase 1 | 路径Y物理对齐: physics_dt=0.005+Holosoma PD → PD过弱, obj_err 退化到 0.80+ | 失败 |
| E008 | 2026-05-01 | Phase 1 | 视频驱动诊断: obj z实测 max=0.307m(从未离地), 衰减PD引导期OK撤除即落 | 失败 |
| E009 | 2026-05-01 | Phase 1 | Person2力支撑5方案: 箱底面均≤8mm — **几何错位:双人对夹±x端** | 失败(几何洞察) |
| — | 2026-05-02 | Phase 2 | **规划**: Phase 2 路线图 (E010-E012), 死磕重定向 | 规划完成 |
| E010 | 2026-05-02 | Phase 2 | Connect/Kinobj重定向: **G1运动学可行性确认** (pelvis_min=0.733, 身体稳定) | **通过** |
| E011 | 2026-05-02 | Phase 2 | Mocap Partner协作: obj_z=0.460(有提升) 但pelvis崩溃; 架构限制 | 部分成功 |
| E012 | 2026-05-02 | Phase 2 | 导出Holosoma格式+partner数据: 格式完全匹配, pelvis_err=0.083m | **通过** |
| — | 2026-05-03 | Phase 3 | **规划**: Phase 3 路线图 (E013-E015), 修复E011架构限制+死磕重定向 | 规划完成 |
| E013 | 2026-05-03 | Phase 3 | Intra-rollout Mocap + Reward Sweep: 技术修复成功, 最佳r7偶尔C1+C2通过(obj=0.477,pelvis=0.712) **但高方差** | 部分成功 |
| E014 | 2026-05-03 | Phase 3 | 增大Partner碰撞体: 原始capsule从未碰到箱子(gap=0.175m), 增大后力太混乱 | 失败 |
| — | 2026-05-03 | Phase 4 | **规划**: Phase 4 路线图 (E015-E016), 小物体验证+双机器人 | 规划完成 |
| E015 | 2026-05-03 | Phase 4 | Bucket005小物体: body-only pelvis_err=0.129m, 物体未搬起; **修复scene_name bug** | 完成 |
| E016 | 2026-05-04 | Phase 4 | **双机器人Gibbs CEM**: nq=79 nu=58, obj_z=0.488(92%ref); **视频复查: connect约束强行拉物体, 机器人头歪/手穿模, 非真实搬运** | ❌ connect假象 |
| E017 | 2026-05-05 | Phase 4 | **双机器人 Soft 2-Connect**: obj_z_max=0.597(112%ref); **视频复查: 脚悬空+手张开, connect约束悬浮物体, 视觉不可用** | ❌ connect假象 |
| E018 | 2026-05-06 | Phase 4 | **Task-Space奖励(DynaRetarget)+Interaction(Harmanoid)**: obj_err↓49%; **视频复查: 头穿进箱子+身体扭曲, connect假指标; E031泛化4case全失败** | ❌ connect假象 |
| E020 | 2026-05-06 | Phase 5 | **多Case诊断(5物体×2模式)**: 全部stable=100%, 仅bucket005 pelvis_err<0.20m; chair022碰撞推飞 | 诊断完成 |
| E021 | 2026-05-06 | Phase 5 | **IK可达性+Anchor**: 行走是主因(87-96%), Pelvis XY Anchor使err↓47-72%, box025降至0.186m | **突破** |
| E022 | 2026-05-06 | Phase 5 | **Anchored+ObjRew**: desk005 lift=37%+视频确认手接触; box025臂展限制不变; bucket010 strong崩溃 | 部分成功 |
| E023 | 2026-05-06 | Phase 5 | **Full Anchor(XY+Yaw)**: bucket010 pelvis_err↓48%(0.15m), 但obj初始距离0.77m超臂展; 单人不可解 | 结构性结论 |
| E024 | 2026-05-07 | Phase 5 | **Partner Force(50-90%grav)**: 手从未主动碰物体, 90%下obj因失重飘起=偶发碰撞; CEM不产生接触 | 失败(结构性) |
| E025 | 2026-05-07 | Phase 6 | **Hand Approach Reward**: 手确实接近物体(dist 0.34→0.00), 但本质是推/碰静止物体, 非沿ref搬运; partner横向动态缺失 | 部分成功(技术有效, 目标未达) |
| E026 | 2026-05-07 | Phase 6 | **Sustained Contact (4096samp/24iter)**: bucket直立+持续接触, 但仍是推静止物体; 长horizon(2.4s)反而保守 | 同上 |
| E027 | 2026-05-07 | Phase 6 | **Partner Force Sweep + desk005**: 接触率不依赖pf强度(均64%); desk005也产生碰触但非搬运 | 同上 |
| E028 | 2026-05-07 | Phase 7 | **阻尼弹簧 4Case全覆盖**: desk005 z=84%/hand=90%/stable=100% 最佳指标; 但**所有case物体翻转**(orientation不受控); orientation spring/damping均不稳定 | C6 FAIL (视频不像搬运) |
| E029 | 2026-05-07 | Phase 7 | **Quasi-Kinematic(kp=100)**: pos跟踪改善(box025=0.07m), 但**仍翻转**; chair022 pelvis崩溃(27%); 结论:xfrc_applied无法控制freejoint orientation | FAIL (根本性限制) |
| E029-act | 2026-05-07 | Phase 7 | **PD Actuator+Kin Override**: contact_guidance CEM干扰obj ctrl; kin override不在rollout生效; **核心发现:scene.xml模式CEM不采样物体,问题是xfrc torque不稳定** | 方向明确(待debug torque) |
| E030 | 2026-05-07 | Phase 7 | **Orientation Debug+Hybrid Export**: CPU torque OK; Warp正反馈/weld被碰撞压倒; anchored body-only 4/4 stable但**hybrid export质量不足以支撑RL**——机器人姿态与搬运无因果关系 | 质量不足 |
| E031 | 2026-05-07 | Phase 8 | **双机器人Connect泛化4case**: 全部失败——行走位移+connect拽倒机器人; Gibbs在connect下有害; 58维CEM采样不够; E018的box025成功不可泛化 | FAIL (结构性) |
| E027b | 2026-05-07 | Phase 8 | **Object PD Override(scene_act+grav_comp+relative_euler)**: desk005 pos=0.10/rot=8.8°★★★, box025 rot=7.6°★★; 4/4 stable=100%; 修复3个bug(euler约定/slide偏移/body_quat相对旋转) | **desk005成功** |
| E027c | 2026-05-07 | Phase 8 | **Contact Guidance(OMOMO方案)on CORE4D**: 机器人走路(pelvis=1.6m)但物体不跟随; **关键发现:论文loco-manipulation用HDMI simulator不是MJWP**; MJWP contact_guidance的gain更新可能不生效 | FAIL (simulator限制) |
| E027d | 2026-05-08 | Phase 8 | **HDMI Physics+Debug**: 验证CUDA graph gains有效; 根因=object ctrl未重置为ref+noise_scale=0; 修复后仍需debug rollout内部state reset | 调试中 |
| E027d2 | 2026-05-08 | Phase 8 | **Body-Frame Fix+Commit Gain Restore**: 3个bug修复; box025=100%stable+1.46m; desk005=87%+1.54m; **body tracking有效(机器人站着走), 但物体通过PD actuator驱动而非真实接触搬运** | body tracking有效 |
| E032a | 2026-05-08 | Phase 9 | **Hand Approach+Reward Sweep**: HA=3提升contact(desk 9→83%, bucket 36→63%); task_body有害; base=10平衡stability/contact; PD sweep: 降低无效 | 完成 |
| E033 | 2026-05-08 | Phase 9 | **desk005 σ sweep+CEM budget**: σ=1.0达95%stable+91%<15cm(目标达成); 增加iter/horizon/samples反而恶化stability; tradeoff是根本性的 | **stable目标达成** |
| E034 | 2026-05-08 | Phase 10 | **HDMI-Style Reward: Stability Penalty**: bounded qpos失败(CEM无法区分站/倒); stability_penalty有效(不稳定时长↓86%, min_z 0.22→0.55m); **但desk005仍有t≈1s严重前倾(视频证实)**; contact mask对desk005无效(始终在范围内) | 改善但未解决 |
| E035 | 2026-05-08 | Phase 10 | **Local-Frame Body Tracking**: 移植HDMI yaw-local reward; desk005: pelvis_min=0.657m+Contact<10cm=94.4%(历史最佳); 3/3 cases零摔倒; **body tracking突破, 但视频显示机器人丢下桌子自己走了, 不是搬运** | body tracking突破 |
| E036 | 2026-05-08 | Phase 10 | **关闭hand_approach**: MPKPE 48cm→1.4cm; Contact<10cm暴跌(desk 94→7%); **确认: body tracking和contact是tradeoff, CEM无法同时优化** | body tracking突破 |
| E037-E039 | 2026-05-08~09 | Phase 11 | **Contact Reward系列**: box-SDF/HDMI-aligned设计; **发现config bug: contact reward从未执行** (E036关闭approach→body_ids=[]→reward条件false) | Bug发现 |
| E039b | 2026-05-09 | Phase 11 | **Config Bug Fix+Rotated SDF**: 修复后contact首次生效; box025=85%/bucket010=76%/desk005=81%; **但发现"手粘连物体"问题(固定target)** | 突破+新问题 |
| E040 | 2026-05-09 | Phase 11 | **Dynamic Per-Frame Target**: 动态target未解决手背接触问题; position-only reward根本缺陷=无方向约束; CEM用手背满足距离→不自然; Contact<10cm=64/66/4%; 需添加orientation reward | ❌ 不自然行为未消除 |
| E041 | 2026-05-09 | Phase 11 | **Orientation Reward(乘法门控)**: palm normal方向约束; 手掌朝向有所改善; 但乘法gating过严导致Contact/Stability退化(62/56%); body前倾问题仍存在; CEM难同时优化position+orientation | ⚠️ 方向正确但约束过严 |
| E041c | 2026-05-09 | Phase 11 | **Additive Ori(w=0.3)最佳变体**: Contact=66/57%+Stable=100%+MPKPE=1.4cm; **CEM reward sweep最佳, 但视频显示: box025趴在箱上, desk005丢下桌子走, bucket010手碰桶侧(推非搬)** | ★ CEM最佳(非搬运) |
| E042 | 2026-05-10 | Phase 11 | **Wrist Freeze(零化手腕噪声)**: 对齐HDMI做法; box025持平(64%), bucket010暴跌(48%); freeze阻止CEM补偿body tracking误差; HDMI成功因ref精确在把手; **确认contact上限≈64-66%(box025)** | ❌ 有害 |
| E043 | 2026-05-10 | Phase 11 | **原始OmniRetarget Ref对比**: Phase3(无松弛)反而更差; box025 64→52%, bucket010 66→57%; Phase4松弛版contact更优; desk005例外(stability改善) | Phase4更优 |
| E044a | 2026-05-10 | Phase 12 | **Wrist Weight=2.0**: box025 contact持平(67%)但stability崩溃(73%); pelvis_min=0.113m; 增强上半身权重→下半身stability退化 | ❌ stability退化 |
| E045 | 2026-05-10 | Phase 12 | **Sigma Sweep(0.3/0.15)**: 收紧sigma→contact全面下降(box025 66→38/46%, bucket010 57→20/23%); desk005局部改善(4→22%); **证实contact瓶颈不在tracking精度** | ❌ 无效 |
| E047a | 2026-05-10 | Phase 12 | **SBTO对齐DynaRetarget**: 修复5个偏差(Sigma EWMA/收敛准则/mean EWMA/elite fraction); 机器人摔倒(MPKPE=155cm); α_μ=0.95太保守+σ_min=0.01太紧; **SBTO+exp-kernel reward不兼容** | ❌❌ 失败 |
| E044b | 2026-05-10 | Phase 12 | **Wrist Weight=3.0**: box025 contact 59%+stable 100%(比w=2.0更稳但contact↓); bucket010 22%/92%; desk005 MPKPE=1.1cm最佳; **weight越大CEM越保守** | ❌ contact退化 |
| E047b | 2026-05-10 | Phase 12 | **SBTO放松参数(α_μ=0.5,σ_min=0.03)**: stability恢复98-100%(E047a=31%), 但MPKPE=69-87cm仍极差; **SBTO开环优化无法替代MPC闭环反馈** | ❌ tracking差 |
| — | 2026-05-11 | Bug Fix | **碰撞盒模板Bug修复**: 21/21 case碰撞盒全部修正为mesh AABB×1.05; box023从1.8x过大修正; bucket010 Y/Z互换修正; box025增大24% | 修复完成 |
| E048 | 2026-05-11 | Phase 13 | **碰撞盒修复后Baseline+HDMI对比**: 碰撞盒修复21 case; HDMI评估发现严重bug(内部ref漂移); **视频复查: 所有case均无搬运 — box023摔倒(ObjPos=14cm是假象), desk005丢下桌子走, box025趴着; 数值指标系统性误导** | ⚠️ 指标不可信 |
| E049 | 2026-05-11 | Phase 14 | **HDMI优化移植失败+eval修正**: 三优化(PD+damping+noise)直接移植使Stability崩(box025 98→57%); HDMI eval指标全部虚假(内部ref漂移); **之前"E041c object tracking优于HDMI 2-8x"结论不成立 — 都是假指标** | ❌ 移植失败 |
| E050 | 2026-05-11 | Phase 14 | **Euler Convention Fix尝试**: "xyz"→"XYZ"引发gimbal lock(box025 Y=89.4°); ObjPos 24→108cm; 已回退 | ❌ gimbal lock |
| E051 | 2026-05-11 | Phase 15 | **HDMI Scene物理配置全面诊断**: (1)Euler mismatch影响所有case(box023=178°,box025=140°); (2)修复euler反而恶化5×(内部自洽被打破); (3)**根因=scene物理配置:hand=1sphere(应3boxes),armature=1.0(应0.01),foot=4spheres(应7capsules)**; 需从suitcase模板重建scene | 方向明确 |
| E052a | 2026-05-11 | Phase 15 | **Suitcase模板+旧euler**: 3-box hand+低armature(0.01)+euler=xyz; ObjPos 37cm(比baseline 24cm更差); Joint 13.9°(比7.3°退化); **低armature损害body tracking, 错euler使好hand无效** | ❌ 单修scene不够 |
| E052c | 2026-05-12 | Phase 15 | **Suitcase模板+正确euler(XZY)**: ObjPos=98cm, Stab=32%(摔倒!); **2×2矩阵最差组合**; 结论: E048a baseline(24cm/7.3°/100%)是HDMI在CORE4D上的极限, euler/"错误"config实为CEM已适应的状态, 修正只会破坏 | ❌❌ 全矩阵失败 |
| E053 | 2026-05-12 | Phase 16 | **碰撞盒Margin Sweep(0.90/0.95/1.00×3case)**: box025上0.90最佳(pelvis_min 0.660 vs 1.05x的0.575); bucket010上1.05x反而最好(0.90/1.00 stability降至88-90%); desk005中间值(0.95-1.00)最差(Stab 66-78%); **不同物体形状需要不同margin, 无全局最优; 碰撞盒不是搬运失败的根因** | ⚠️ per-case策略 |

## 全局结论 (E001-E053, 50+ 实验)

### 视觉复查后的诚实评估

**50+ 实验, 跨 5 种方法, 没有任何实验产生视觉可接受的 loco-manipulation 结果:**

| 方法 | Body Tracking | 物体搬运 | 视觉质量 | 代表实验 |
|------|-------------|---------|---------|---------|
| E041c 单机器人 MJWP | 站着(desk)或摔倒(box) | ❌ 没搬 | ❌ | E048 |
| HDMI 单机器人 | ✅ 站着走 | ❌ 物体推歪 | 勉强(body OK) | E048a |
| 双机器人 CEM | ❌ 穿模变形 | ❌ connect假搬运 | ❌ | E016-E018 |
| 双机器人泛化 | ❌ 全部崩溃 | ❌ | ❌ | E031 |
| 阻尼弹簧/xfrc | — | ❌ 物体翻转 | ❌ | E028-E029 |

### 数值指标 vs 视觉真实的系统性偏差

| 指标 | 数值看起来 | 视觉实际 | 原因 |
|------|----------|---------|------|
| ObjPos=14cm (E041c box023) | "物体追踪好" | 机器人摔倒, 物体没动 | 物体静止=低误差 |
| Contact=93% (HDMI box023) | "手触碰物体" | 物体方向错 178° | euler 错配下的假接触 |
| obj_z=0.597m (E017d dual) | "箱子被抬起 112%" | connect 约束悬浮 | 非物理接触力 |
| Stability=100% (E041c desk) | "机器人稳定搬运" | 丢下桌子自己走了 | 只看 pelvis_z |

### 唯一有效的能力

**Body tracking (关节角度追踪)**: HDMI 7.3°, E035/E041c 15-20°
- 机器人能复现人体的站立、弯腰、行走等全身动作
- 但无法通过物理接触搬运物体

### 根本瓶颈

1. **CEM sampling-based MPC 无法产生 sustained contact**: 1024 samples × 32 iterations 的搜索空间不够
2. **Contact guidance decay**: 最后 iteration PD→0 后, CEM 没有维持接触的策略
3. **Connect 约束是 hack**: 绕过接触发现问题但引入穿模/变形
4. **CORE4D 数据是双人协作**: 单机器人物理上无法完成原始任务的搬运部分

## 关键指标演进

```
pelvis_err: E002(0.100) → E003(0.068) → E004(0.061) → E009a(0.115) → E012(0.083) → E015-d(0.129, bucket005)
E020 多Case pelvis_err: bucket005(0.157) < bucket010(0.660) ≈ desk005(0.647) ≈ box025(0.660) < chair022(0.787)
E020 多Case lift%: bucket005(60%) > bucket010(36%) > desk005(37%+obj) > chair022(130%碰撞推飞) > box025(1.5%)
E025 hand approach: bucket010 contact=55%(6/11), lift_max=0.095m; desk005 contact=40%(4/10), lift_max=0.057m
E026 sustained: bucket010 contact=64%(7/11), lift_max=0.093m, 4 consecutive >5cm
E027 generalization: bucket010 pf=50% still 64% contact; desk005 contact=80%, lift_max=0.117m (best overall)
E021 Anchored pelvis_err: box025(0.186, ↓72%) < desk005(0.279, ↓58%) < bucket010(0.293, ↓58%) < chair022(0.417, ↓47%)
joint_err:  E002(0.073) → E003(0.064) → E012(0.108 rad) → E015-d(0.202 rad, bucket005)
MPKPE (Phase 10+): E036=1.4cm → E039b=1.1-2.1cm → E040=1.2-1.9cm (全部<3cm, 优秀)
Contact<10cm演进:
  box025:  E036(56%) → E039b(85%,手粘连) → E040(64%,手背接触) | 均有不自然行为
  bucket010: E036(2%) → E039b(76%,手粘连) → E040(66%,手背接触) | 均有不自然行为
  desk005: E036(7%) → E039b(81%,stable=20%) → E040(4%,stable=81%)
  根因: position-only reward无方向约束 → CEM用手背满足距离 → 需orientation reward
Phase 12 探索(均未超越E041c baseline):
  E045a(σ=0.3): box025 38%↓ / bucket010 20%↓ / desk005 6%≈  ← sigma收紧有害
  E045b(σ=.15): box025 46%↓ / bucket010 23%↓ / desk005 22%↑  ← desk005局部改善
  E044a(w=2.0): box025 67%≈ / stability 73%↓ ← 上半身权重与stability tradeoff
  E047a(SBTO):  box025 0% / stability 31% ← SBTO+exp-reward不兼容,需调参
obj z 实测 (npz qpos):
  Phase 1: 所有实验 sim max ≤ 0.460m (E011), 实际未持续离地
  Phase 2 (kinobj): obj follows ref (PD驱动), pelvis stable
  Phase 3: E013-r7 obj_z=0.477(翻转非抬起), E013-ctrl obj_z=0.463, 高方差
  ref: 0.533m (峰值)
```

## Phase 2 结论

**G1 单人无法物理搬起 box025, 但运动学完全可行:**
- E010 证明: 当物体被驱动时, G1 身体保持稳定 (pelvis≥0.73m)
- E011 证明: Mocap partner 提供额外物理支撑 (obj_z=0.46)
- E012 证明: 可以导出高质量 hybrid 轨迹到 Holosoma RL

**最终产出**: `workspace/core4d/results/box025_person1_spider_holosoma_w_partner.npz`
- 机器人: SPIDER 物理合规 (pelvis_err=0.083m, 无脚滑)
- 物体: 运动学参考 (正确搬运曲线)
- Partner: 双手世界坐标 (供 RL interaction reward)

## Phase 1 结论 (E001-E009)

**G1 单人无法物理搬起 box025。根因不是力（gravcomp 也只离地 8mm），而是接触几何错位。**
- CORE4D box025 是双人协作任务，两人从 ±x 端对夹
- G1 从 -y 侧接近，接触方向完全不匹配
- 臂展 0.5m < box 长 0.61m，单人对夹不可解

## Phase 2 路线图

```
E010: Connect 约束 → "假设抓住了, G1 能搬吗?" (运动学验证)
  ├─ 成功 → E011: Mocap Partner → "有伙伴物理力, 能协作搬吗?"
  │         └─ 成功 → E012: 导出 + 泛化 (bucket/chair)
  └─ 失败 → 切换到 bucket005 (更小物体, 单人可行)
```

## Phase 3 路线图

```
E013: Intra-rollout Mocap Partner → "修复E011架构限制, rollout内更新partner"
  ├─ E013a: mjwp.py step_env 内 wp.copy mocap (最小改动)
  ├─ E013b: 切换到 mjwp_eq (利用已有per-step mocap更新)
  ├─ E013c: 奖励权重扫描
  │
  ├─ 成功 → E015: 导出+泛化
  └─ 失败 → E014: mjwp_eq Weld退火 + Mocap Partner
            ├─ 成功 → E015: 导出+泛化
            └─ 失败 → E015: 双机器人交替优化
```

## 关键教训
- **必须用 qpos 实测 + 视频验证** — reward 字段不可信 (E006/E008), 数值指标系统性误导 (E048-E052 全面复查)
- **几何分析先于参数调优** — 接触方向比力大小重要 (E009)
- **理解数据集语义** — 协作数据需要协作建模 (E009)
- **重定向 ≠ RL, PD 不需对齐** — 两阶段天然分离 (E007)
- **"修复" 可能破坏已适应的系统** — euler convention 修复使结果 5× 恶化 (E051b), CEM 已适应 "错误" 配置 (E052c)
- **碰撞盒大小无全局最优** — 不同物体形状需不同 margin, box025 需小(0.90), bucket010 需大(1.05) (E053)
- **connect 约束制造假指标** — 所有双机器人 "成功" 均为穿模变形 (E016-E018 视频复查)

## Logs

- E001: `workspace/core4d/log/01_E001_data_pipeline_results.md`
- E002-E004: `workspace/core4d/log/02_E002_E003_results.md`, `03_E004_strong_guidance_results.md`
- E005: `workspace/core4d/log/04_E005_holosoma_export_results.md`
- E006: `workspace/core4d/log/05_E006_forearm_contact_results.md` (更正:箱子未搬起)
- E007: `workspace/core4d/log/06_E007_path_Y_results.md`
- E008: `workspace/core4d/log/07_E008_real_lift_diagnosis_results.md`
- E009: `workspace/core4d/log/08_E009_person2_support_results.md`
- Phase 2 计划: `workspace/core4d/plan/09_phase2_retarget_roadmap.md`
- E010: `workspace/core4d/log/09_E010_kinematic_object_results.md`
- E011: `workspace/core4d/log/10_E011_mocap_partner_results.md`
- E012: `workspace/core4d/log/11_E012_export_results.md`
- Phase 3 计划: `workspace/core4d/plan/11_phase3_retarget_roadmap.md`
- E013: `workspace/core4d/log/12_E013_intra_mocap_results.md`
- E014: `workspace/core4d/log/13_E014_larger_partner_results.md`
- Phase 4 计划: `workspace/core4d/plan/14_phase4_small_object_dual_robot_plan.md`
- E015: `workspace/core4d/log/14_E015_bucket005_results.md`
- E016: `workspace/core4d/log/15_E016_dual_robot_results.md`
- 碰撞盒修复: `workspace/core4d/log/54_collision_box_bug_fix.md`
- E048-E049 评估修正: `workspace/core4d/log/57_E048_E049_eval_correction.md`
- E050 euler分析: `workspace/core4d/log/58_E050_hdmi_euler_analysis.md`
- E051 HDMI scene诊断: `workspace/core4d/log/59_E051_hdmi_scene_diagnosis.md`
- E052 suitcase模板: `workspace/core4d/log/60_E052_suitcase_template_results.md`
- E048-E052 视觉复查: `workspace/core4d/log/61_E048_E052_visual_reevaluation.md`
- E053 碰撞盒margin: `workspace/core4d/log/62_E053_collision_margin_sweep.md`

## 脚本

- 转换: `workspace/core4d/scripts/convert/convert_holosoma_to_spider.sh`
- 重定向(基线): `workspace/core4d/scripts/retarget/retarget_core4d_baseline.sh`
- 重定向(引导): `workspace/core4d/scripts/retarget/retarget_core4d_guidance.sh`
- 重定向(前臂): `workspace/core4d/scripts/retarget/retarget_core4d_forearm.sh`
- 重定向(前臂+act): `workspace/core4d/scripts/retarget/retarget_core4d_forearm_act.sh`
- 重定向(E009): `workspace/core4d/scripts/retarget/retarget_core4d_e009a.sh`
- 导出: `workspace/core4d/scripts/export/export_to_holosoma.sh`
- 评估(E009): `workspace/core4d/scripts/eval/eval_e009_lift.py`
