# E065 Results — task_obj_rew form ablation on box023 (FAIL, but reveals new culprit)

## 状态: ❌ **FAIL — 两个变体 B1/C1 均未通过, 但诊断指向 actuator stiffness 而非 reward form. → R2 (E066) port HDMI actuator gain.**

**TL;DR**: log 82 hypothesis "task_obj L2 unbounded form 是 pre-contact 抬腿元凶" 部分成立 — E065-A (drop task_obj) 把 B1 从 0.48m 降到 0.30m (改善 38%), 但仍未达 ≤0.10m 阈值, 且 C1 (pelvis_min_intent) 反而退化 (0.19→0.12m). E065-D (HDMI exp form) 则 B1 反恶化到 0.69m. 关键发现: **task_obj=0 全关掉时, sim 仍然抬腿 lunge** — 元凶不只是 task_obj. 唯一非零 reward 是 qpos_rew (body tracking, MJWP 跟 HDMI 字节一致), 所以 reward 形式不解释. 剩余假设: MJWP 用 kp=500 强 actuator, HDMI 用 kp=20, 物体太"硬"导致手要用力推 → COG 前移 → 单脚后伸 lunge 当反平衡.

## 1. 实验配置

| 变体 | task_obj_rew 形式 | scale | use_exp | sigma | yaml |
|------|------------------|-------|---------|-------|------|
| E065-A | (关闭) | 0.0/0.0 | n/a | n/a | core4d_e065a_box023.yaml |
| E065-D | scale·exp(-‖err‖/sigma) | 1.0/1.0 | true | 0.5 | core4d_e065d_box023.yaml |

两者均继承 E062 baseline (X1 only + sphere palm), 没有 E063/E064 的 stability/threshold 改动.

## 2. 量化结果

| 指标 | E063 baseline | E065-A | E065-D | 阈值 | 评估 |
|------|---------------|--------|--------|------|------|
| **B1** max foot_z [0-2s] | 0.48m | **0.30m** ✗ | 0.69m ✗ | ≤0.10m | A 改善 38% 仍 FAIL, D 退化 44% |
| **B2** single-foot runs | 4 | 1 | 2 | =0 | 都 FAIL, A 最优 |
| C1 pelvis_min_intent | 0.19m | **0.12m** ✗ | 0.11m ✗ | ≥0.50m | 都 FAIL, 比 baseline 还差 |
| C2 pelvis_mean_intent | 0.54m | 0.48m | 0.47m | ≥0.50m | 都 FAIL |
| C3 stable% intent | 67% | 64% | 60% | ≥80% | 都 FAIL |
| C4 final_obj_pos_err | 0.81cm | **3.10cm** ✓ | 1.75cm ✓ | ≤30/20 | 都 PASS — 物体追踪不依赖 task_obj reward |
| L palm contact% | 91% | 90% | 97% | — | D 略好 |
| R palm contact% | 97% | 84% | 91% | — | A 略差 |

C4 大惊喜: E065-A task_obj=0 时 final 物体 err 还有 3.10cm (相比 baseline 0.81cm), 说明物体追踪主要靠 ref ctrl + body tracking 拉手, **task_obj reward 在物体追踪中是 redundant signal**.

## 3. 视觉观察 (规则 9 强制不留空)

### E065-A (task_obj=0)
- t=0.4s: ref 双脚平地起步弯腰, sim **右脚悬空 0.30m** 已经单腿撑前倾
- t=0.8s: ref 已弯腰双手够箱 (双脚平地), sim 完全前倾几乎 prone, 头朝下, 右脚后伸悬空, 没有完成接触动作
- t=2.0s: pelvis 已掉到 0.13m (摔倒)
- t=2.5-4.0s: pelvis 在 0.10-0.40m 之间挣扎, 没有真正 recover

### E065-D (HDMI exp form)
- t=0.4s: 跟 A 类似, 右脚已悬空
- t=0.6s: sim 右脚悬空到 **0.69m** (远高于 A 的 0.30m), 极端 lunge
- t=0.8s: 类似 A, 接触阶段单脚撑伸手够箱
- t=2.0s: pelvis 掉到 0.07m, 几乎 prone
- t=4.0s: **pelvis recover 到 0.74m**, 但仍 lunge stance (一脚后伸), 箱子被推超过目标

D 跟 A 相比: 单脚抬更高但最终能回起立; A 抬得低但回不起来. 都不达标但 failure mode 不同.

→ keyframes: `workspace/core4d/results/E065/keyframes/E065{A,D}_box023_kf*.jpg`
→ foot_z trace: `workspace/core4d/results/E065/foot_z_trace_E065{A,D}_box023.png`

## 4. Reward 组成分析 (本实验关键诊断)

```
E065-A pre-contact (t=0-2s):
  qpos_rew_mean (= local_frame_rew):    +1.40
  task_body_rew_mean:                    0
  task_obj_rew_mean:                     0  (变体 A 设置)
  interact_rew_mean:                     0
  hand_approach_rew_mean:                0

E065-D pre-contact (t=0-2s):
  qpos_rew_mean (= local_frame_rew):    +1.45  
  task_body_rew_mean:                    0
  task_obj_rew_mean:                    +1.40  (saturating exp ∈ [0, 2])
  interact_rew_mean:                     0
  hand_approach_rew_mean:                0
```

**A 仅 qpos_rew 一个 reward 就足以让 sim lunge**. 这个 qpos_rew 计算用的 local_frame_rew, 包括 upper_pos/ori, lower_pos/ori, root_pos/ori, joint — 完全字节匹配 HDMI (sigma 0.5/1.0/0.5/0.25, W_TRACK=0.5, 见 mjwp.py:555-660 vs hdmi.py:1130-1185). 所以 reward 公式不是元凶.

## 5. 修正后的诊断 — log 82 hypothesis 部分推翻

| log 82 假设 | E065 数据 | 修正版 |
|------------|----------|--------|
| MJWP `task_obj=-L2` unbounded 是 pre-contact lunge 元凶 | E065-A 全关 task_obj 仍 lunge B1=0.30m | task_obj 是 contributor (38% improvement when removed), 不是 sole cause |
| HDMI saturating exp 应该 work | E065-D B1=0.69 worse | exp form 反而更糟, 因 [0, scale] 区间内"靠近就奖励"加强了急追物体倾向 |
| body-tracking 完全等价 → 不是元凶 | A 仅 qpos_rew 仍 lunge | body-tracking 公式没问题, 但**没强到能阻止其他扰动**; 其他扰动仍存在 |

**剩余唯一未对齐项**: actuator stiffness.
- HDMI: object actuator kp=20, kd=...  (hdmi.py:670 override)
- MJWP: object actuator kp=500 (scene_act.xml 默认)
- 物理含义: kp=500 时 object 几乎刚体 — 手要 push 必须用 25× 大力, 反作用力把 COG 推前 → 单脚撑

## 6. 决策树执行

按 log 83 §4.4 矩阵:
- E065-A: B1+C1 都 FAIL
- E065-D: B1+C1 都 FAIL

→ **❌❌ → R2 = E066 (port HDMI actuator gain)**

具体计划:
1. 改 `spider/simulators/mjwp.py` 加 actuator gain override 逻辑 (参考 hdmi.py:650-680)
2. 加 `spider/config.py` flag: `actuator_kp_override: dict | None = None` per-actuator group
3. 配 `examples/config/override/core4d_e066_box023.yaml` 继承 E065-D + actuator override (object actuators kp 500→20)
4. 双 GPU: GPU0 = E066 (D + actuator override), GPU1 = E066b (A + actuator override) 看 task_obj 全关 + 软 actuator 能不能 work
5. 评估同 R1, 加 actuator force trace

## 7. 教训 #12 (新增)

**reward 形式 ablation 没改善 → 怀疑物理参数 / actuator dynamics**. 当所有 reward 项都对齐 HDMI 但 sim 仍失败时, 元凶不在 reward, 而在动力学层. Object actuator 强度差异 (kp 500 vs 20) 改变了 COG 平衡所需的支撑力, 间接驱使 sim 选 lunge 姿态.

**关键泛化**: 调 reward 之前, 先确认 actuator/PD/joint limit 等物理参数 baseline 跟 reference workflow 是否对齐. **不要把 dynamics 问题当 reward 问题调.**

## 8. 改动文件 / 结果路径

| 类型 | 路径 |
|------|------|
| 训练脚本 | `workspace/core4d/scripts/train/train_E065.sh` |
| 评估脚本 | `workspace/core4d/scripts/eval/eval_E065.py` |
| 关键帧脚本 | `workspace/core4d/scripts/eval/extract_E065_keyframes.sh` |
| 一键脚本 | `workspace/core4d/scripts/run_E065.sh` |
| Yamls | `examples/config/override/core4d_e065{a,d}_box023.yaml` |
| Code | `spider/config.py` (use_exp fields), `spider/simulators/mjwp.py` (use_exp branch) — 已 commit 4b46d6d |
| Scene snapshot | `workspace/core4d/results/E065/scene_snapshot/` (manifest.txt sha256 + git HEAD 4b46d6d) |
| 结果 npz | `workspace/core4d/results/E065/E065{A,D}_box023.npz` |
| 视频 | `workspace/core4d/results/E065/E065{A,D}_box023.mp4` |
| Foot z plots | `workspace/core4d/results/E065/foot_z_trace_E065{A,D}_box023.png` |
| Pelvis/obj plots | `workspace/core4d/results/E065/pelvis_obj_E065{A,D}_box023.png` |
| Keyframes | `workspace/core4d/results/E065/keyframes/E065{A,D}_box023_kf*.jpg` |
| CSV | `workspace/core4d/results/E065/eval_summary.csv` |
