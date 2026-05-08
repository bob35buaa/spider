# E035: Local-Frame Body Tracking — HDMI 核心设计移植

## 状态: 稳定性解决，但 body tracking 远未达标 (MPKPE=48cm, 目标<15cm)

## 背景

E034 证明 stability_penalty 是 reactive 的（pelvis 已低才触发），无法预防摔倒。
HDMI 不摔的核心是 **local-frame body tracking**：在 pelvis yaw-only 坐标系中计算 body tracking error，CEM 可以选择"pelvis 稍偏但保持平衡"的方案。

## 代码改动

| 文件 | 改动 |
|------|------|
| `spider/simulators/mjwp.py` | +`_lf_yaw_quat`, `_lf_quat_mul` 等6个 helper; +`_local_pos_tracking`, `_local_ori_tracking`; get_reward 中添加 local-frame 分支 |
| `spider/config.py` | +`use_local_frame_reward`, `local_frame_*` sigma/ids/w_track 共8个字段 |
| `examples/run_mjwp.py` | 预计算全 body xpos(T,nbody,3) + xquat(T,nbody,4), 8-tuple ref_data |
| `examples/config/override/core4d_e035.yaml` | 新配置 |

## 全面评估结果 (Paper-Standard Metrics)

评估方法: `workspace/core4d/scripts/eval/eval_comprehensive.py`
指标定义: `workspace/core4d/docs/eval_metrics.md`

### E035 vs HDMI R013 基线 vs 论文

| 指标 | HDMI R013 | E035 desk005 | E035 box025 | E035 bucket010 | DynaRetarget | 合格线 |
|------|-----------|-------------|-------------|---------------|-------------|-------|
| **MPKPE (cm)** | **7.72** | 47.92 | 26.17 | 36.67 | 3.57 | <15 |
| **Joint Err (deg)** | **3.22** | 10.26 | 11.81 | 9.70 | — | <5 |
| **EEF Pos (cm)** | **7.88** | 48.36 | 36.38 | — | — | — |
| **EEF Ori (deg)** | **5.46** | 55.19 | 62.36 | — | — | — |
| **Root Pos (cm)** | **7.17** | 47.43 | 21.93 | 34.54 | — | <15 |
| **Root Ori (deg)** | **2.30** | 18.71 | 6.46 | — | — | — |
| **Obj Pos (cm)** | **5.39** | 22.55 | 16.80 | 15.05 | 8.81 | <12 |
| **Obj Ori (deg)** | **4.28** | 14.01 | 10.74 | — | 6.3 | <10 |
| **Stability >0.60m** | 84.8% | **100%** | **100%** | **100%** | — | >90% |
| **Penetration** | — | **0%** | **0%** | **0%** | — | <5% |
| **Foot Skating** | — | 10.9% | 27.2% | 15.7% | — | <10% |
| **Contact <10cm** | — | **94.8%** | 40.3% | 44.0% | — | >80% |
| **Smoothness (rad/s²)** | — | 9.8 | 10.0 | 9.4 | — | — |

**注**: DynaRetarget (G1 box kicking) 和 SPIDER OMOMO (G1 suitcase) 都是 humanoid loco-manipulation 任务，与我们直接可比。差距 6-13x 说明 reward/CEM 配置有根本性问题。

### desk005 详细分解

```
  A. BODY TRACKING
    MPKPE (all bodies):       47.92 ± 21.13 cm
    Joint Angle Error:        10.26 ± 5.26 deg
    EEF Position Error:       48.36 ± 15.20 cm
    EEF Orientation Error:    55.19 ± 27.14 deg

  B. ROOT TRACKING
    Root Position Error:      47.43 ± 23.69 cm
    Root Orientation Error:   18.71 ± 25.02 deg

  C. OBJECT TRACKING
    Obj Position Error:       22.55 ± 11.52 cm
    Obj Orientation Error:    14.01 ± 7.50 deg

  D. PHYSICAL PLAUSIBILITY
    Pelvis z: min=0.660m, mean=0.775m
    Stability >0.60m: 100%
    Penetration: 0%
    Foot Skating: 10.9%

  E. INTERACTION QUALITY
    Mean hand-obj surface dist: 5.32 cm
    <10cm: 94.8% (sustained: 106 frames / 3.53s)
    < 5cm: 39.7% (sustained: 39 frames / 1.30s)
    < 3cm: 23.3% (sustained: 19 frames / 0.63s)
    < 1cm: 14.7% (sustained: 17 frames / 0.57s)
    Contact Preservation (ref desired → sim <10cm): 99.0%

  F. SMOOTHNESS
    Mean |joint acceleration|: 9.8 rad/s²
```

## 可视化验证

### desk005

| 帧 | 时间 | 观察 |
|------|------|------|
| 0 (0s) | 起始 | ref/sim对齐良好，站立姿态正常 |
| 46 (0.8s) | 行走 | sim跟着ref走路，全程直立——与E034d的t=0.8s开始前倾形成对比 |
| 92 (1.5s) | 中前段 | sim站在桌旁，手搭在桌面上——姿态自然。这是E034d摔倒的时刻 |
| 139 (2.3s) | 中段 | sim走在桌旁，手触桌面——行走+接触同时进行 |
| 185 (3.1s) | 后段 | sim弯腰趴在桌面上，手贴桌——前倾较大但pelvis>0.70m |
| 231 (3.9s) | 结束 | sim站在桌旁，手搭桌面——稳定 |

**关键问题**: 视觉上机器人稳定且手在桌面，但**整体位移/朝向和ref差距极大** (Root Pos=47cm)。机器人在"做自己的事"而不是跟踪ref的locomotion轨迹。

### box025

| 帧 | 时间 | 观察 |
|------|------|------|
| 0 (0s) | 起始 | ref/sim对齐 |
| 50 (0.8s) | 弯腰 | ref/sim都弯腰趴箱顶，sim手搭箱面——姿态匹配 |
| 99 (1.7s) | 中段 | ref站在箱侧；sim也站着手伸向箱面——稳定 |
| 149 (2.5s) | 后中段 | ref推箱行走；sim站立手搭箱顶 |
| 198 (3.3s) | 后段 | ref弯腰趴箱顶；sim也弯腰趴箱顶——姿态跟踪好 |
| 247 (4.1s) | 结束 | ref手举起；sim站在箱旁——姿态偏离 |

### bucket010

| 帧 | 时间 | 观察 |
|------|------|------|
| 0-50 | 0-0.8s | ref/sim对齐，弯腰匹配 |
| 100-150 | 1.7-2.5s | sim站着，桶被PD推走距离太远 |
| 200-249 | 3.3-4.2s | sim弯腰，姿态跟踪好但桶太远 |

## Claims 验证 (修正)

1. ✅ **C1**: desk005 pelvis_z > 0.657m, >0.70m = 90.9% ≥ 90% — 达成
2. ✅ **C2**: 3/3 cases 无不稳定段 — 达成
3. ✅ **C3**: desk005 contact<10cm = 94.4% ≥ 70% — 达成
4. ❌ **隐含 C4**: body tracking 精度 — MPKPE=48cm，远超合格线(15cm)，**未达成**

## 根因分析: 为什么 HDMI MPKPE=7.7cm 而 E035=48cm?

### 对比 HDMI vs E035 的 reward 结构

| 差异 | HDMI | E035 |
|------|------|------|
| W_TRACK | 0.5 | 0.5 (相同) |
| hand_approach | **无** (用 contact mask 门控的 rew_contact) | **scale=5, σ=3, 全程激活** |
| object tracking | rew_obj_pos + rew_obj_ori (直接跟踪物体全局位姿) | **关闭** (task_body_rew_scale=0) |
| contact reward | **mask 门控** (只在 ref 标记接触帧激活, gain=5) | 全程激活 |
| stability_penalty | **无** (不需要) | scale=30, threshold=0.55 |

### 核心问题: hand_approach 全程激活 + 贡献过大

HDMI 的 `rew_contact` 有两个关键限制:
1. **Contact mask 门控**: 只在 ref 标记为"接触"的帧激活
2. **在 object_tracking 组内**: 与 rew_obj_pos + rew_obj_ori 并列，总贡献 max≈3.0

E035 的 `hand_approach_rew`:
1. **全程激活** (threshold=100m → 永远开启)
2. **scale=5, σ=3**: 当手距物体 10cm 时 reward ≈ 5*exp(-0.3)=3.7
3. **独立于 tracking 之外**: 直接加到 total reward

**计算**:
- E035 tracking max = 0.5 * 7 = 3.5 (body tracking 7 terms)
- E035 hand_approach max = 5.0 (当手贴物体时)
- **hand_approach 贡献 > tracking 贡献** → CEM 主要优化 hand_approach!

而 HDMI:
- tracking max = 3.5
- object_tracking max ≈ 3.0 (含 contact mask 门控)
- **tracking 和 object 相当** → CEM 同时优化两者

### 为什么 HDMI 不需要 stability_penalty

因为 HDMI 的 contact 有 mask 门控，不会在不该接触的时候拉手 → 不会导致前倾 → 不需要 penalty 来补救。

## 下一步计划 (E036)

### 方向: 对齐 HDMI reward 结构

1. **关闭/大幅降低 hand_approach_rew** — 它是造成 body tracking 退化的主因
2. **用 HDMI 风格的 rew_contact 替代** — contact mask 门控 + 在 object_tracking 组内
3. **开启 object global tracking** — rew_obj_pos + rew_obj_ori
4. **去掉 stability_penalty** — 如果 contact 有 mask，不应该需要

具体参数对齐:
```yaml
# E036 预期配置
use_local_frame_reward: true
local_frame_w_track: 0.5  # 保持
hand_approach_rew_scale: 0.0  # 关闭！
stability_penalty_scale: 0.0  # 关闭（不需要了）
# 新增: object global tracking (HDMI 风格)
task_obj_pos_rew_scale: 1.0  # rew_obj_pos
task_obj_rot_rew_scale: 1.0  # rew_obj_ori
# TODO: 添加 contact mask 门控的 rew_contact
```

**预期**: 去掉 hand_approach 后 CEM 会全力优化 body tracking → MPKPE 大幅下降。Contact 可能退化，但之后可以用 mask-gated contact 恢复。

### 验证目标

- MPKPE < 20cm (从48cm下降 >50%)
- Joint Err < 8deg
- Stability >0.60m 维持 >90%

## 结果路径

| 产出 | 路径 |
|------|------|
| desk005 | `workspace/core4d/results/E035/E035_desk005.{npz,mp4}` |
| box025 | `workspace/core4d/results/E035/E035_box025.{npz,mp4}` |
| bucket010 | `workspace/core4d/results/E035/E035_bucket010.{npz,mp4}` |
| 配置 | `examples/config/override/core4d_e035.yaml` |
| 全面评估脚本 | `workspace/core4d/scripts/eval/eval_comprehensive.py` |
| 评估指标文档 | `workspace/core4d/docs/eval_metrics.md` |
