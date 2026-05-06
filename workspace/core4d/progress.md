# CORE4D 研究进度

## 当前会话: 2026-05-06

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
