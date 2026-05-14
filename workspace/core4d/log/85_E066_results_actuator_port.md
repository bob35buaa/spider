# E066 Results — port HDMI object actuator gains (FAIL, reveals body-partition dilution)

## 状态: ❌ **FAIL — soft actuator 让 sim 完全不搬箱子 (C4 jumped to 112-138cm), pre-contact lunge 仍 persist. 揭示第二个 HDMI mismatch: body partition.**

**TL;DR**: log 84 假设"actuator stiffness (kp=500 vs HDMI 20) 是 lunge 元凶". E066 把 init_pos_actuator_gain 500→20, init_rot 50→0.3, decay 1.0→0.85, 测 A (drop task_obj) + D (exp form) 双变体. **结果**: pre-contact lunge **仍 persist** (E066-A B1=0.53, E066-D B1=0.75, 都比 E065 还差), 但 C4 final_obj_err **暴涨到 112-138cm** (E065-A 为 3.10cm, E065-D 1.75cm). 物理含义: soft actuator 让物体不再被 actuator "拽"着跟 ref, sim 没强 task_obj 也不去搬, 箱子留在原地 1m+ 远. 但 E066-A C1=0.34m (vs E065-A 0.12m) + C2=0.60m + C3=93% (大幅改善) + sim t=3-4s pelvis 起身到 0.85m 完全 recover — soft actuator 确实改善了稳定性, 但代价是任务失败. **第二个发现**: 即使 task_obj=0 + soft actuator (E066-A 双重保险), sim 仍 lunge B1=0.53m → 元凶不在 actuator 也不在 task_obj. **诊断**: MJWP `local_frame_lower_ids = [2..13]` (12 bodies, hip_yaw/roll/pitch + knee + ankle_pitch/roll), HDMI 用 6 bodies (`hip_pitch + knee + ankle_roll` × 2). MJWP `error.mean()` 把脚约束**稀释 2×** — 一只脚抬高, 12 个 body 平均后误差小, 6 个 body 平均后误差大 → MJWP 的 lower_pos_rew 不足以阻止单脚 lift.

## 1. 实验配置

| 变体 | task_obj | actuator | yaml |
|------|---------|----------|------|
| E066-A | drop (scale=0) | kp 20/0.3, decay 0.85 | core4d_e066a_box023.yaml |
| E066-D | HDMI exp form | kp 20/0.3, decay 0.85 | core4d_e066d_box023.yaml |

继承链: e066{a,d} → e065{a,d} → e062 → e041c (X1+sphere baseline).

## 2. 量化结果 (vs E065 R1 + E063 baseline)

| 指标 | E063 base | E065-A | E065-D | E066-A | E066-D | 阈值 |
|------|-----------|--------|--------|--------|--------|------|
| **B1** max foot_z [0-2s] | 0.48 | 0.30 | 0.69 | **0.53** ✗ | **0.75** ✗ | ≤0.10 |
| **B2** single-foot runs | 4 | 1 | 2 | 3 | 1 | =0 |
| **C1** pelvis_min_intent | 0.19 | 0.12 | 0.11 | **0.34** | 0.10 | ≥0.50 |
| C2 pelvis_mean_intent | 0.54 | 0.48 | 0.47 | **0.60** ✓ | 0.45 | ≥0.50 |
| C3 stable% intent | 67% | 64% | 60% | **93%** ✓ | 62% | ≥80% |
| **C4** final_obj_pos_err | 0.81cm | 3.10 | 1.75 | **112.01** ❌ | **138.18** ❌ | ≤30/20 |
| L palm contact% | 91% | 90% | 97% | **100%** | 83% | — |
| R palm contact% | 97% | 84% | 91% | **100%** | 100% | — |

**新现象**:
- E066-A: C1/C2/C3 全大幅改善 (sim 不再 deep fall), 但 box 落下 112cm
- E066-D: 全维度都比 E065-D 差, 包括 box 也丢得更远
- 两者 B1 都比 E065 还高 — soft actuator NOT 解决 lunge

## 3. 视觉观察 (规则 9 强制)

### E066-A (drop task_obj + soft actuator)
- t=0.4s: sim Lf z=0.21m (E065-A 也是 0.21), 起步差不多
- t=0.6s: **sim Rf 飙到 0.53m** — 单脚撑前倾, 比 E065-A 还高
- t=2.0s: ref 抱箱子前推, sim **superman dive** 头朝下双脚悬空 + 双手撑地推地 (push-up 起始姿), 箱子完全没动
- t=2.8s: pelvis 0.17m 触底
- t=3.2s: pelvis 飙升到 0.85m **完全起身**, 双脚回地
- t=4.0s: sim 单脚撑直立 (右脚悬), 箱子被丢在身边 1m+ 远

→ E066-A 是"前倾 dive → 撑地起身 → 弃箱直立"模式. C2/C3 高是因为后段恢复.

### E066-D (HDMI exp form + soft actuator)
- t=0.4-0.6s: sim Rf 飙到 0.75m, 比 E066-A 还激进
- t=0.8-2.0s: 持续单脚 lunge + 弯腰够箱
- t=2.8s: pelvis 0.10m 触底, 没起身
- t=4.0s: 仍 lunge stance, 箱子未搬

→ keyframes: `workspace/core4d/results/E066/keyframes/E066{A,D}_box023_kf*.jpg`
→ foot trace: `workspace/core4d/results/E066/foot_z_trace_E066{A,D}_box023.png`

## 4. 关键诊断 — MJWP body partition 稀释

```python
# MJWP (spider/config.py:158-163):
local_frame_upper_ids = list(range(14, 31))  # 17 bodies: waist_yaw → wrist_yaw
local_frame_lower_ids = list(range(2, 14))   # 12 bodies: hip_pitch → ankle_roll

# HDMI (spider/simulators/hdmi.py:819-823):
upper = [".*_shoulder_pitch_link", ".*_elbow_link", ".*_wrist_yaw_link"]  # 6 bodies (L+R × 3)
lower = [".*_hip_pitch_link", ".*_knee_link", ".*_ankle_roll_link"]       # 6 bodies (L+R × 3)
```

**Reward 计算**:
```python
return torch.exp(-error.mean(dim=1) / sigma)
```
两边都 `error.mean(dim=1)`, 但 MJWP 的 mean 跨 12 bodies, HDMI 跨 6.

**数学影响**: 假设 1 只脚 (1 body) 抬到 0.30m, 其他 11 个 lower body 误差 ~0.01m:
- MJWP: mean = (0.30 + 11*0.01) / 12 = 0.034 → reward = exp(-0.034/0.5) = 0.93 (几乎满分!)
- HDMI: mean = (0.30 + 5*0.01) / 6 = 0.058 → reward = exp(-0.058/0.5) = 0.89 (略低)

差距小但累积. 关键是 MJWP 把 hip_yaw/roll + ankle_pitch 这些 "跟着 hip_pitch + knee 几乎刚体连动" 的中间 link 也拉进 mean 平均, 它们的误差天然小, 把 outlier (脚 z) 稀释了.

类似的 dilution 也在 upper (17 vs 6) — waist + torso 这些"跟着 pelvis frame 自然在位"的 body 把 wrist 误差稀释了, 这或许解释了为什么 wrist_weight=2 在 E044 是错方向 (它加权过的是已稀释的版本).

## 5. R3 决策 — E067

**主推**: E067 = narrow body partition (port HDMI 6+6 ids) + (复用 E062 强 actuator + L2 task_obj 还是 D 还是 A 待定)

候选变体:
- **E067-N**: narrow ids ONLY, 其他保持 E062 baseline (强 actuator + L2 task_obj). 测试 partition 单变量影响.
- **E067-NS**: narrow ids + soft actuator + exp task_obj. 完整 HDMI clone.

双 GPU 并行. 评估: B1 主指标 + C4 (确认 box 仍跟随) + 视觉.

**Decision tree**:
- E067-N B1 ≤ 0.10 + C1 ≥ 0.50 + C4 ≤ 30: **🎉 CONFIRMED**, 改 spider/config.py 默认 lower_ids/upper_ids 到 narrow + permanently fixed
- E067-N FAIL but E067-NS PASS: 需要 narrow + soft actuator + exp 三件套
- 都 FAIL: 进 E068 = 直接跑 run_hdmi.py on box023 复现 ground-truth + foot_z trace 对比, 找剩余差异 (e.g. knot_dt 0.1 vs 0.2, num_envs, CEM noise schedule)

## 6. 教训 #13

**Reward `error.mean(dim=1)` 形式下, 增加 tracked bodies 数会反向稀释 outlier penalty**. 不是 "tracking 越多越好". 应该精选关键 bodies (HDMI: 6 lower 包含 hip_pitch/knee/ankle_roll, 不包含 hip_roll/yaw 这些被父子链约束的中间 link). MJWP 默认 `range(2,14)` 是不假思索"全部下肢", 是反 pattern.

**搜寻 reward 实现差异时, 不能只对比 sigma 和公式, 还要对比 body_id 列表 / mask / weights**. 这次教训: log 82 §10.2 我说 "body-tracking 公式完全相同 (MJWP 的 local_frame_rew 是 HDMI get_reward 的逐字 port)" — 公式对了, 但 body_ids 列表差 2× — **公式正确但 input 集差异也是 mismatch**.

## 7. 教训 #14

**Soft actuator 是 trade-off, 不是免费午餐**. 把 object actuator kp 从 500 降到 20 让 sim 不再被 box 反作用力推前 (好), 但同时 box 不再被 ref-trajectory 拖着走 (坏). 单变量改物理参数容易导致"换问题不解问题". 必须配套 **task_obj reward 加大** 来补偿失去的 actuator-driven tracking.

→ 这是 E066-D 失败的另一原因: exp task_obj 最大 reward 只有 1.0 (saturating), 不足以让 sim "主动" 搬已经不被 actuator 拖动的 box.

## 8. 改动文件 / 结果路径

| 类型 | 路径 |
|------|------|
| 训练脚本 | `workspace/core4d/scripts/train/train_E066.sh` |
| 评估脚本 | `workspace/core4d/scripts/eval/eval_E066.py` (clone E065) |
| 关键帧脚本 | `workspace/core4d/scripts/eval/extract_E066_keyframes.sh` (clone E065) |
| Yamls | `examples/config/override/core4d_e066{a,d}_box023.yaml` |
| Code | (无, 纯 yaml override) |
| Scene snapshot | `workspace/core4d/results/E066/scene_snapshot/` |
| 结果 | `workspace/core4d/results/E066/E066{A,D}_box023.{npz,mp4}` |
| Plots | `workspace/core4d/results/E066/{foot_z_trace,pelvis_obj}_E066{A,D}_box023.png` |
| Keyframes | `workspace/core4d/results/E066/keyframes/E066{A,D}_box023_kf*.jpg` |
| CSV | `workspace/core4d/results/E066/eval_summary.csv` |
