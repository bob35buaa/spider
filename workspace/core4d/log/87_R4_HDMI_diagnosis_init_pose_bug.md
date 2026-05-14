# R4-direct: HDMI vs MJWP box023 diagnosis — INIT POSE BUG smoking gun

## 状态: 🚨 **真元凶定位 — sim t=0 pelvis quat 偏 ref 22° (yaw 119.5° vs 97.8°), HDMI 仅 0.6°. 不是 reward / dynamics / solver, 而是 init 流程 bug. 待用户决策修复路径.**

**TL;DR**: 用现成 `workspace/core4d/results/E052/E052c_box023_euler_fix/trajectory_hdmi.npz` 跟 MJWP 直接对比. **关键发现**: HDMI 在 box023 完美 work (B1=0.066m ✓, pelvis 9s 稳在 0.62-0.75m, 物体 z 0.49→0.61m 真搬起来 + 维持). 三 round MJWP 失败的根因不在 reward/dynamics/solver, 而在: **MJWP sim t=0 pelvis quat 已经跟 ref 偏 22° yaw**, 接下来 2 帧再漂 30°. HDMI sim t=0 vs ref 仅 0.6°. 这是**初始化阶段就偏离的 bug**, 任何后续 reward 调参都治标不治本.

## 1. HDMI vs MJWP side-by-side trace (用户提供 trajectory_hdmi.npz)

**Foot z (pre-contact 0-2s)**:
| Source | B1 = max(Lf_z, Rf_z)[0:60 frames] | Notes |
|--------|-----------------------------------|-------|
| HDMI sim (E052c) | **0.066m** ✓ | 全程 0-9s 双脚 ≤0.07m, 真 walking |
| MJWP-E062 (E063 baseline) | 0.481m ❌ | t=0.5-1.5s 两脚交替抬 0.20-0.48m |
| MJWP-E065-A (drop task_obj) | 0.305m ❌ | 改善但仍 lunge |
| MJWP-E067-N (narrow partition) | **1.30m** ❌❌ | sim handstand |

**Pelvis z**:
| Source | min full | mean | t=2-3s 行为 |
|--------|---------|------|-------------|
| HDMI sim | 0.062m (t=9.5s) | 0.512m | 平稳 0.65-0.75 整段 |
| MJWP-E062 | 0.193m (t=2.3s) | 低于 0.50 半段 | 急摔 0.81→0.19 然后 push-up 起身 |

**Object z (carry quality)**:
| Source | obj z 0-9s 行为 | 评估 |
|--------|-----------------|------|
| HDMI | 0.49→0.61m 升 + 维持 0.55-0.63m → 0.66m place | ✓ 真搬运 |
| MJWP-E062 | 0.15→0.55→0.15m 起伏 | ❌ 不真搬, 是 actuator drag |

→ Plot: `workspace/core4d/results/hdmi_vs_mjwp_box023_diag.png`

## 2. HDMI ctrl 分析 — PPO learned 小残差跟 ref

```
HDMI ctrl 字段 |ctrl - ctrl_ref| stats:
  abs mean:  0.1724   (vs |ctrl_ref| mean 0.4791, ratio ~0.36)
  abs max:   1.7918  
  pre-contact (0-60 frames):
    |ctrl - ctrl_ref| mean: 0.1279   (vs |ctrl_ref| 0.4791, ratio 0.27)
```

HDMI's PPO policy 学到 **跟 ref ctrl + ~13% residual** 的策略. 这个 small residual 包含了"保持平衡"的 implicit prior.

MJWP CEM:
- 已经 init `ctrls = ctrl_ref[:horizon_steps]` (run_mjwp.py:1020), 用 ref ctrl 作 warm start ✓
- noise_scale = `logspace(0.5, 1.0, knots) × joint_noise_scale=0.05` = 0.025-0.05 per joint per knot
- 实际噪声 (~0.04) 比 HDMI residual (~0.13) **还要小**

→ 噪声不是元凶 (推翻 log 86 §5 candidate C 怀疑).

## 3. 真正元凶 — INIT POSE MISMATCH

| t=0 metrics | wxyz quat | euler (xyz deg) | yaw 偏离 ref |
|-------------|-----------|-----------------|-------------|
| **ref** (kinematic.npz) | (0.657, -0.029, 0.006, 0.753) | (-1.7, +3.0, **+97.8**) | 0° |
| **HDMI sim** (E052c) | (0.657, -0.034, 0.009, 0.753) | (-1.8, +3.6, **+97.7**) | **-0.1°** ✓ |
| **MJWP-E062 sim** (E063 npz) | (0.504, -0.005, 0.027, 0.863) | (+2.3, +2.1, **+119.5**) | **+21.7°** ❌❌ |

**这是 22° yaw rotation 在 sim 第 1 帧!** 然后接下来 2 帧 (t=0.03-0.07s) MJWP 又再漂 30° 到 yaw=148°. HDMI 同期保持 ±0.5°.

后续 sim 试图"扭回去 + 跟 ref"但已经偏离, body tracking reward 强行拉过来, 加上 freejoint pelvis 惯性 → 出现 lunge / handstand / superman 这些 CEM 找到的"代偿姿态".

## 4. 为什么 init 偏 22°?

**已 verify (排除)**:
- ref qpos[0, :7] = (-0.845, -1.555, 0.79, 0.657, -0.029, 0.006, 0.753) ✓
- 把 ref qpos[0, :42] 直接 assign 到 scene_act 的 d_act.qpos, mj_forward 后 pelvis xquat = (0.657, -0.029, 0.006, 0.753) — **完全对齐 ref** ✓
- 所以 init code path (mj_data.qpos[:] = qpos_ref[0]) 数学上对
- warmstart_qpos_path=''  (没 snap 改写)
- ref qvel[0] = 0  (没初始速度)

**未 verify (可能元凶)**:
- (a) MJWP 用 `wp.copy(env.data_wp.qpos, ...)` 初始化 GPU env, 这个过程是否 lossy 转换? — 检查 mjwarp `put_data` 是否做了什么转换
- (b) 第 1 frame 保存的 qpos[0,0,:] 是否是 init OR 已经 1 step physics? — 看 run_mjwp.py 控制流, 可能是 post-1-step
- (c) 强 robot actuator (kp=500) 对 joint=ctrl 时 zero-error 应该 zero-torque, 但如果有微小 ctrl_ref 跟 init joint 的 mismatch, 强 actuator 会瞬间发力 → pelvis swing
- (d) 物体 actuator 在 init 阶段从 qpos[36:42] (slider+euler) targeting ref obj pos, 但 env 物体 init 跟 ref 差 → 强 actuator 蹬出反作用力扭 robot pelvis (object 通过手到 pelvis 的 chain)

## 5. R5 候选方案 (按侵入性排序)

### A. 最小改动 (yaml only): 减弱 robot+obj actuator init kp
强 actuator (robot kp=500 in scene_act.xml + obj kp=20) 在 init 阶段如果 target ≠ state, 立即蹬出反作用力. 把 init kp 改弱 (e.g. obj kp 20→2, robot 500→100), warmup 几帧再 ramp up.
- 缺点: scene_act.xml 的 robot kp=500 不能 yaml override (硬编码在 XML 里), 需要改 XML 或加 runtime override

### B. 修 init 流程: 写 `init_kp_warmup` config (代码)
加 `spider/simulators/mjwp.py` 一个 init warmup phase: 第 1-N frames actuator kp ramp 0→full, 给 sim 时间 settle 到 ref pose.
- 中等改动, 1 个新 config flag + 几行 code

### C. 修 init bug: 把 sim 第 1 frame snap 到 ref qpos
跑 1 sim_step 后强制 `data_wp.qpos[:] = qpos_ref[1]` (re-snap), 再继续. 这绕过 init drift.  
- 简单粗暴, 验证假设最快

### D. 直接换 solver: 用 HDMI's PPO policy 在 box023 跑 inference
不是修 MJWP, 而是 fallback 到 HDMI workflow. 但 HDMI 在 contact 阶段也失败 (用户说 ref 左手发不上力). 不是真正解决 box023.

### E. 再做 trace 实验: 测 (a)-(d) 哪个是真原因
- 跑 1 个 minimal test: scene_act + ref qpos[0] + zero ctrl, 观察 N 步 后 pelvis quat. 不需要训练. 30 sec 跑完.
- 这样能精准定位是 actuator 问题还是 mjwarp 转换问题.

## 6. 教训 #17

**对比 sim 和 ref 的 init pose 必须包含 quaternion (不只 xyz)**. 我之前 (log 84) 说 "init 对齐 OK" 是因为只看 pelvis_z. 实际 quat 偏 22° 早就发生但没注意, 直到 R4-direct 才挖到. 后续 init pose 验证必须 dump qpos + quat 全维度跟 ref 比较.

## 7. 改动文件 / 结果

| 类型 | 路径 |
|------|------|
| Plot | `workspace/core4d/results/hdmi_vs_mjwp_box023_diag.png` |
| 诊断脚本 | (本 log §1-3 inline scripts, 没固化为 .py — 如有需要可固化) |
| Log | `workspace/core4d/log/87_R4_HDMI_diagnosis_init_pose_bug.md` (本文) |
