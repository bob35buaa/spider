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
| E016 | 2026-05-04 | Phase 4 | **双机器人Gibbs CEM**: nq=79 nu=58, 两G1均稳定(≥0.70), obj_z=0.488(92%ref) | **通过** |
| E017 | 2026-05-05 | Phase 4 | **双机器人 Soft 2-Connect**: obj_z>0.40持续98%帧, obj_z_max=0.597(112%ref) | **突破** |
| E018 | 2026-05-06 | Phase 4 | **Task-Space奖励(DynaRetarget)+Interaction(Harmanoid)**: obj_err↓49%, R1=95%/R2=100%stable | **最佳** |
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
- **必须用 qpos 实测 + 视频验证** — reward 字段不可信 (E006/E008)
- **几何分析先于参数调优** — 接触方向比力大小重要 (E009)
- **理解数据集语义** — 协作数据需要协作建模 (E009)
- **重定向 ≠ RL, PD 不需对齐** — 两阶段天然分离 (E007)

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

## 脚本

- 转换: `workspace/core4d/scripts/convert/convert_holosoma_to_spider.sh`
- 重定向(基线): `workspace/core4d/scripts/retarget/retarget_core4d_baseline.sh`
- 重定向(引导): `workspace/core4d/scripts/retarget/retarget_core4d_guidance.sh`
- 重定向(前臂): `workspace/core4d/scripts/retarget/retarget_core4d_forearm.sh`
- 重定向(前臂+act): `workspace/core4d/scripts/retarget/retarget_core4d_forearm_act.sh`
- 重定向(E009): `workspace/core4d/scripts/retarget/retarget_core4d_e009a.sh`
- 导出: `workspace/core4d/scripts/export/export_to_holosoma.sh`
- 评估(E009): `workspace/core4d/scripts/eval/eval_e009_lift.py`
