# CORE4D 研究进度

## 当前会话: 2026-05-07 (续2)

### E027b: Object PD Override — 突破性方案

**实验路径**:
1. 使用 `scene_act.xml` (6 position actuators: 3slide+3hinge, armature=1.0)
2. `_apply_object_pd_override()` 每步覆盖 object ctrl → PD 跟踪 ref
3. gravity compensation: `target_z += mg/kp`
4. CEM 只优化 robot body tracking + hand approach

**结果 (最终版)**:
| Case | stable | pos_err | rot_err° | 视觉 |
|------|--------|---------|----------|------|
| desk005 | 100% | **0.100** | **8.8°** | ★★★ 首次完整 6DOF 搬运 |
| box025 | 100% | 0.101 | **7.6°** | ★★ 方向正确但 z 不够 |
| bucket010 | 100% | 0.547 | 61.5° | ★ 仍有旋转偏移 |
| chair022 | 100% | 0.559 | 110.6° | ✗ 待 debug |

**修复的 3 个 Bug**:
1. Euler 约定: `"xyz"` → per-case 最佳 extrinsic 约定 (避免 gimbal lock)
2. Slide 偏移: `world_pos` → `world_pos - body_pos`
3. **Body_quat 相对旋转** (根本原因): `R_joint = R_body.inv() * R_world`

**关键发现**:
- CUDA graph 不更新 model params → gains 必须 bake 进 XML
- MuJoCo body_quat 非 identity 时, hinge 控制相对旋转

**待优化**:
1. 导出 desk005 hybrid 轨迹 → Holosoma RL
2. 增加 kp 解决 bucket/chair (需更大 armature)
3. 移除 hand_approach_rew (desk 不需要, box 有害)

---

### E029 Quasi-Kinematic (kp=100): 仍然翻转

- kp=100 + pf=1.0: box025 pos_err=0.07(优秀), desk005=0.13(好), 但**视频仍翻倒**
- chair022: pelvis 仅 27% stable (物体强拉导致 robot 失稳)
- **根本结论**: xfrc_applied 位置弹簧无法控制 freejoint 物体 orientation

**Phase 7 最终结论**:
> 位置弹簧 (E028/E029) 能驱动物体中心沿 ref 移动, 但无法阻止翻转。
> 这是 xfrc_applied + freejoint 的根本限制, 不是参数问题。
> 需要使用 MuJoCo joint-level 机制 (actuator 或 equality constraint) 来完整控制 6DOF。

**可行出路**:
1. 接受 kinematic object (E010 方案): robot body motion 正确, 物体用 ref → 导出 hybrid 轨迹
2. 用 scene_act.xml 的 position actuator 驱动物体 (需要生成新 scene XML)
3. 用 weld equality constraint + soft solref 让物体跟踪 mocap body

---

### E028 4Case 全覆盖 (position-only spring)

| Case | kp | stable | z_track | hand<10cm | 视频 |
|------|-----|--------|---------|-----------|------|
| box025 | 30 | 100% | 21% | 82% | 物体几乎不动, 手碰到 |
| bucket010 | 30 | 100% | 38% | 82% | 物体移动+手跟, 但翻转 |
| desk005 | 15 | 100% | 84% | 90% | **desk 翻倒** (C6 FAIL) |
| chair022 | 15 | 100% | 148%(过冲) | 73% | 椅子翻转/弹射 |

**核心问题**: 位置弹簧无法控制 orientation → 所有物体都翻转
- Orientation spring 实现了但导致数值不稳定 (quaternion torque issue)
- 纯位置弹簧: 拉物体中心但不阻止旋转
- **C6 (视频像搬运) 全部 FAIL**

**下一步**: 转向 E029 contact_guidance (PD actuator 同时控制 pos + rot)

---

### Phase 6 修正评估: Hand Contact Guidance (E025-E027)

**用户反馈**: 视频可视化证实 E025-E027 并非成功的搬运重定向:
1. bucket010: 参考中物体在 0-1s 被 partner 横向移动 (xy=0.32m, lift=13cm), sim 中物体静止 → 机器人碰到静止物体
2. desk005: 推/碰, 非协作搬运
3. 接触本身是手部 (非腿/脚), 但本质是 body tracking 中的偶发碰撞

**技术贡献 (有效)**:
- hand_approach_rew 确实让 CEM 产生手-物体接近 (0.34→0.00m)
- data_path override、表面距离近似、4096 samples 优化均有效

**未达成的目标**:
- 物体未沿参考轨迹运动 (partner 横向搬运动态完全缺失)
- 不构成"接触重定向" — 只是"body tracking 中碰到了物体"

**核心差距**: partner_force 只有竖直恒力, 缺少 partner 的横向搬运力。需要建模 partner 的完整搬运贡献。

---

### E025 Hand Approach Reward — CEM 手-物体接近 (技术有效, 目标未达)

**实验**: bucket010/desk005 + anchored + hand_approach_rew=5.0 + partner_force=0.85
**结果**:
- **bucket010 (E025-f)**: 55% frames contact, lift_max=9.5cm, 视频确认手伸向/抱住 bucket
- **desk005 (E025-g)**: 40% frames contact, lift_max=5.7cm, 跨 case 泛化成功
- **因果关系**: contact(t=3-5) → lift(t=5 peak 9.5cm), 完美时间相关

**机制**: `hand_approach_rew = scale * exp(-sigma * hand_to_surface_dist)`
- 与 contact_rew (只在接触后奖励) 不同: approach 从任意距离提供梯度
- 与 task_body_rew (跟踪参考手位) 不同: approach 直接拉向物体表面
- CEM 2048 采样中, 一部分"手离物体较近"的采样被 approach 放大 → 逐步收敛

**Phase 5 结论修正**: SPIDER CEM **可以做接触**, 但需要 approach reward 提供梯度引导。
之前 E024 失败是因为缺少手→物体的显式梯度, 不是 CEM 的绝对限制。

**代码改动**:
- `spider/config.py`: +hand_approach_rew_scale/sigma/body_names/body_ids/obj_half_extents, data_path 不再无条件覆盖
- `spider/simulators/mjwp.py`: +hand_approach_rew (surface distance + exp decay)
- `examples/run_mjwp.py`: +partner_force_ref_pos setup (for spring)

**后续优先级 (更新)**:
1. ~~Hand approach reward~~ → **已验证有效!**
2. 增加 samples (4096) + 更长 horizon (2.4s) → 更高接触率
3. 两阶段 CEM (body-first + approach) → 更稳定
4. 阻尼弹簧 → 修复 E026 不稳定
5. SBTO (可选进一步提升)

---

## 历史: 2026-05-07 (earlier)

### E024 失败: Partner Force — CEM 不产生主动接触

**实验**: box025 + anchored + partner force (50%/70%/90% gravity comp) + obj_rew=3.0
**结果**: 
- 90% force: obj lift=84% 但是失重飘移 (序列末尾, ref 已下降时箱子飘起)
- 视频确认: 所有配置中 G1 手始终在体侧, 未伸向物体
- pelvis tracking 良好 (0.15-0.21m), 但手不碰物体

**根因**: CEM 在关节空间采样, 无法发现"伸手→接触→施力"的因果序列。contact_rew 只在"已接触"后给奖励, 不引导"去接近"。

**Phase 5 总结论**: SPIDER CEM 能做 body tracking, 不能做 contact-rich manipulation。
问题是结构性的: 关节空间随机采样 + 短 horizon + 高维精确接触空间 = 概率极低。

**后续方向优先级**:
1. Hand position task-space tracking (强制手到参考位置)
2. Hand approach reward (手→物体距离梯度)
3. Spring + approach 组合
4. SBTO (全序列优化, DynaRetarget 路线)

---

### E023 Full Anchor (XY+Yaw)

**诊断三步**:
1. **Pelvis error 分解**: walk_err/pose_err = 3.4-15.5x → 行走贡献 87-96% 误差
2. **站定片段检测**: 全部 4 case 无站定帧 (均速 0.5-0.95 m/s) → 裁剪不可行
3. **手-物体表面距离**: 69-84% 帧手到物体 < 0.05m → 手部可达不是问题

**Pelvis XY Anchor 方案**: 每帧减去累积 pelvis xy 位移 → 原地搬运

| Case | Original → Anchored | 改善 |
|------|---------------------|------|
| box025 | 0.660m → **0.186m** | **↓72%** |
| bucket010 | 0.692m → **0.293m** | **↓58%** |
| chair022 | 0.787m → **0.417m** | **↓47%** |
| desk005 | 0.666m → **0.279m** | **↓58%** |

**结论**: SPIDER CEM 有局部姿态跟踪能力, 只是被行走需求拖累。Anchor 是有效的预处理。

### E020 诊断: 多 Case SPIDER 单人重定向能力验证

**5 个 Case × 2 种模式 (body-only / with-obj) = 10 runs**:

| Case | 物体特征 | body-only pelvis_err | body-only lift% | with-obj lift% |
|------|----------|---------------------|-----------------|----------------|
| box025 | 大箱 0.61×0.61×0.89m | 0.660 | 1.5% | 3.9% |
| bucket005 | 小桶 0.30×0.41×0.29m | **0.157** | **59.6%** | 29.0% |
| bucket010 | 中桶 0.40×0.74×0.40m | 0.692 | 36.2% | 3.8% |
| chair022 | 椅子 0.57×0.86×0.53m | 0.787 | 129.6% (推飞) | 127.4% |
| desk005 | 桌子 0.40×0.74×0.80m | 0.666 | 0.9% | **37.1%** |

**关键诊断结论**:
1. Body 稳定性全部 100% — SPIDER 核心能力 OK
2. Body tracking 精度普遍差 (仅 bucket005 < 0.20m) — **参考动作对 G1 运动学不友好是根因**
3. bucket005 是唯一表现好的 case (低 pelvis_err + 高 lift)
4. obj_rew 效果不一致: 对 desk005 有帮助, 对 bucket005/bucket010 反而有害
5. chair022 的 130% lift 是碰撞推飞, 非真实搬运

**回答核心问题**: "是 case 难度还是 SPIDER 算法?"
→ **是参考动作的 G1 运动学可达性问题**。当动作对 G1 友好时 (bucket005), body tracking 好, 碰撞位移也自然产生。当动作不友好时, 机器人选择"安全但不准确"的替代姿态, 手无法到达正确位置。

**产出**:
- 计划: `workspace/core4d/plan/20_E020_multicase_diagnosis_plan.md`
- 日志: `workspace/core4d/log/18_E020_multicase_diagnosis_results.md`
- 结果: `workspace/core4d/results/E020_multicase_diagnosis/{case}/{bodyonly,withobj}.{npz,mp4}`
- Metrics: `workspace/core4d/results/E020_multicase_diagnosis/metrics_summary.csv`
- 配置: `examples/config/override/core4d_{bucket010,chair022,desk005}.yaml`
- 新 case 数据: 3 × scene.xml + trajectory_kinematic.npz

---

**E018 结果 (双机器人 + connect2 + 多奖励)**:
- **物体跟踪误差减半**: obj_pos_err 0.599m → 0.308m (vs E017-d, ↓49%)
- **双机器人稳定性提升**: R1 86%→95%, R2 89%→100%
- **控制平滑度大幅改善**: E018-d 的 ctrl acc 仅为 E017-d 的 40%
- **最佳配置 E018-d2**: task-space body rewards + interaction reward + base_pos=15
- **无约束验证**: E018-noconnect 仅 11% 帧 obj_z>0.40, 印证 DynaRetarget "SBMPC 短视" 论点

**论文启发**:
- DynaRetarget (arXiv:2602.06827): task-space rewards (object_pos=40, torso=30, hand=5, foot=10)
- Harmanoid (arXiv:2510.10206): interaction reward (双机器人关键点相对距离 Eq.15)

**代码改动**:
- `spider/config.py`: +task-space reward fields, body name → id 解析
- `spider/simulators/mjwp.py`: get_reward 添加 task_body_rew/task_obj_rew/interact_rew
- `examples/run_mjwp.py`: ref_data 扩展为 6 元组, 预计算 body_xpos_ref
- 配置: `core4d_box025_e018_{connect,noconnect}.yaml`

**关键洞察**:
- task-space 奖励是 SBMPC 提升 tracking 质量的关键 (无需重写 optimizer)
- body stability vs obj tracking tradeoff: base_pos=15 是最佳平衡
- 短 horizon (1.6s) 是结构性瓶颈, 需要 E019 长 horizon 或 E020 SBTO

**文件产出**:
- `workspace/core4d/log/17_E018_taskspace_results.md`
- `workspace/core4d/results/E018d2_box025_dual_taskspace_basepos15.{npz,mp4}` (best)
- `workspace/core4d/results/E018{a,d,d2-noconnect}_*.{npz,mp4}`
- `workspace/core4d/results/E018d2_mpc{0,3,5,8,10}.png`

---

## 历史会话: 2026-05-05

### E017 突破性进展: 双机器人 Connect 约束协作搬箱

**E017 结果 (双机器人 + Connect 约束)**:
- **箱子首次持续离地**: E017-d obj_z > 0.40m 持续 198 连续帧 (98% 总帧数)
- **物体跟踪大幅提升**: obj_z_max=0.597 (ref peak=0.533, 112% 跟踪)
- **最佳方案: 柔约束 2-connect + 强 body reward**: solref=-200 -30, base_pos=15
- **硬约束导致过冲**: E017-a obj_z_max=1.021, 但机器人崩溃
- **CEM-only 无效**: 无约束时 obj_z > 0.40 仅 26/264 帧

**关键洞察**:
- Connect 约束将手"焊接"到箱面 → CEM 只优化身体 → 箱子自动跟随
- 从根本上解决了 CEM 无法通过随机采样产生有效接触力的问题
- 2-connect (6 eq) 优于 4-connect (12 eq): 减少过约束风险

**代码改动**:
- `workspace/core4d/scripts/generate_scene_dual_connect.py`: 生成 4-connect 和 2-connect 场景
- `examples/config/override/core4d_box025_e017.yaml`: E017 配置
- 场景: `scene_dual_robot_connect.xml`, `scene_dual_robot_connect2.xml`

**文件产出**:
- `workspace/core4d/log/16_E017_dual_connect_results.md`
- `workspace/core4d/results/E017{a-e}_*.npz/mp4`
- `workspace/core4d/results/E017d_mpc{0,3,5,8,10}.png`

---

## 历史会话: 2026-05-04

### Phase 4 实施完成 (E015-E016)

**E015 结果 (Bucket005 小物体)**:
- 数据管线通用: bucket005 scene.xml + 转换 + MJWP 全通
- Body-only retargeting 质量良好: pelvis_err=0.129m, pelvis_z≥0.607
- baseline (E015-a) 机器人直接摔倒 → 强 body reward 必需
- 物体从未搬起: 所有配置 obj_z_max ≤ 0.207m (ref max=0.247m)
- **scene_name bug 修复**: 之前 E013/E014 的 scene_name 被 filter_config_fields 丢弃

**E016 结果 (双机器人 Gibbs CEM)**:
- **首次实现双 G1 + 共享物体交替优化**: nq=79, nu=58
- 两机器人均保持稳定: R1 pelvis≥0.697, R2 pelvis≥0.715
- **obj_z_max=0.488 (ref peak=0.533, 92% 跟踪)** → 比单机器人 E013-r7(0.477) 略高
- 但仍为碰撞推动/翻转, 非协作搬运 (CEM horizon 限制不变)
- Object tracking reward 仍导致单侧崩溃 (E016-b: R1 crashed)

**代码改动**:
- `spider/config.py`: +scene_name 字段, +dual_humanoid_object 支持
- `spider/simulators/mjwp.py`: +dual_humanoid_object branches (5 个函数)
- `examples/run_mjwp.py`: Gibbs sampling → dual-humanoid robot1/robot2 split
- 新分支: `feat/dual-robot-retarget`

**E013 结果**:
- `wp.to_torch` 共享内存写入在 CUDA graph launch 前生效 → 技术修复成功
- Intra-rollout + 高 contact_rew(3.0) = 最佳组合 → obj_z=0.477, pelvis=0.712
- 但 CEM 高方差: C1+C2 同时通过率仅 40% (2/5)
- Intra-rollout 是双刃剑: 低 contact_rew 时反而恶化 (CEM 更难)

**E014 结果**:
- 原始 partner capsule (r=0.04) **从未触碰箱子** — gap=0.175m (wrist vs fingertip)
- 增大碰撞体 → 接触力太大太混乱 → 箱子被弹飞
- Mocap 无限刚性是根本限制: 太小→无接触, 太大→混沌

**关键新发现**:
1. partner_pos 数据是手腕坐标, 非指尖 → 距箱面 0.175m
2. CEM 0.4s horizon 不足以规划协作搬运序列
3. obj_z >0.4 的"抬起"实为翻转/翘起 (一角离地)

**Phase 3 结论**:
```
G1 + box025 物理协作搬运在 SPIDER CEM 框架内不可靠解决
→ 最佳策略: body-only retargeting (E012) + RL (Layer 2)
```

**文件产出**:
- `plan/11_phase3_retarget_roadmap.md`
- `log/12_E013_intra_mocap_results.md`
- `log/13_E014_larger_partner_results.md`
- `results/E013_box025_p1_intra_mocap.npz` (最佳物理尝试)
- `results/E014_box025_p1_bodyonly_scene.npz` (最佳 body-only)
           └─ 失败 → E015: 双机器人交替优化
```

**文件产出**:
- `plan/11_phase3_retarget_roadmap.md` — Phase 3 整体路线图

### 下一步: 实施 E013

### 关键教训 (Phase 1 总结)

1. **必须用 qpos 实测 + 视频验证** — reward 字段不可信
2. **几何分析先于参数调优** — 接触方向比力大小重要
3. **理解数据集语义** — 协作数据需要协作建模
4. **重定向 ≠ RL, PD 不需对齐** — 两阶段天然分离

### 脚本清单

| 脚本 | 用途 | 对应实验 |
|------|------|---------|
| `scripts/convert/convert_holosoma_to_spider.sh` | holosoma qpos → SPIDER NPZ | E001 |
| `scripts/retarget/retarget_core4d_baseline.sh` | MJWP 无引导 | E002 |
| `scripts/retarget/retarget_core4d_guidance.sh` | MJWP contact guidance | E003/E004 |
| `scripts/retarget/retarget_core4d_forearm.sh` | 前臂接触重定向 | E006 |
| `scripts/retarget/retarget_core4d_forearm_act.sh` | 前臂+PD引导 | E008 |
| `scripts/retarget/retarget_core4d_e009a.sh` | E009 力支撑 | E009 |
| `scripts/export/export_to_holosoma.sh` | 混合轨迹 → Holosoma 格式 | E005 |
| `scripts/eval/eval_e009_lift.py` | 底面离地客观评估 | E009+ |

---

## 历史会话: 2026-05-01

### E006-E009 实验汇总

- E006: 前臂接触 → reward 看似改善 61%, 视频证实箱子未离地(虚假突破)
- E007: 路径Y物理对齐 → 破坏重定向能力(PD冲突)
- E008: 视频驱动诊断 → 确认 E006 的 obj z 实测 max=0.307m
- E009: Person2 力支撑 5 方案 → 全部失败, 发现几何错位根因

---

## 历史会话: 2026-04-30

### Phase 0+1: E001-E005

- E001: 数据管线 OK (124帧, 30fps)
- E002-E004: 基线重定向, 物体落地, 确认归零设计
- E005: 混合轨迹导出 OK (52body, 50fps)
