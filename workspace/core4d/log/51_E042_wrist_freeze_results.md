# E042: Wrist Freeze — 零化手腕关节 CEM 噪声 (对齐 HDMI 做法)

## 状态: wrist freeze 未改善手背接触,反而降低 bucket010 contact; 前倾问题依旧

## 背景

核实 `run_hdmi.py` 发现: HDMI workflow 也是 CEM (非 RL), 但 HDMI 对 wrist 关节零化噪声 (line 112-118)。假设: wrist 零噪声使手腕保持 ref 姿态 → 手掌方向自然正确 → 解决手背接触。

## 核心改动

```python
# config.py: get_noise_scale() — 零化匹配关键词的关节噪声
if config.zero_noise_joint_keywords and hasattr(config, "_model_cpu_for_noise"):
    for ai in range(model.nu):
        aname = mj_id2name(model, mjOBJ_ACTUATOR, ai)
        if any(kw in aname for kw in config.zero_noise_joint_keywords):
            noise_scale[:, :, ai] *= 0.0
```

配置: `zero_noise_joint_keywords: [wrist_roll, wrist_pitch, wrist_yaw]`
→ 影响 6 个关节 (left/right × roll/pitch/yaw)

## 变体设计

| 变体 | Contact reward | Wrist freeze | 目的 |
|------|------|------|------|
| E042a | E040 (dynamic target, pos-only) | ✅ | 隔离 wrist freeze 对手背问题的效果 |
| E042b | E041c (dynamic target + additive ori) | ✅ | 组合 ori + wrist freeze |
| E042c | 无 (E036 baseline) | ✅ | wrist freeze 单独对 contact 的影响 |

## 结果

### box025

| 变体 | Contact<10cm | Stability | MPKPE | Preservation | 对比 |
|------|------|------|------|------|------|
| E036 (baseline, 无 freeze) | 56% | 100% | 1.4cm | - | - |
| E040 (pos-only, 无 freeze) | 64% | 100% | 1.3cm | 88.8% | - |
| E041c (additive ori, 无 freeze) | 66% | 100% | 1.4cm | 88.2% | - |
| **E042a (pos-only + freeze)** | 64% | 100% | 1.3cm | 83.2% | = E040 |
| **E042b (additive ori + freeze)** | 64% | 100% | 1.4cm | 84.0% | ≈ E041c |
| **E042c (无 contact + freeze)** | 48% | 100% | 1.4cm | 58.8% | < E036 (56%) |

### bucket010

| 变体 | Contact<10cm | Stability | MPKPE | Preservation | 对比 |
|------|------|------|------|------|------|
| E036 (baseline, 无 freeze) | 2% | 100% | 1.3cm | - | - |
| E040 (pos-only, 无 freeze) | 66% | 100% | 1.2cm | 95.4% | - |
| E041c (additive ori, 无 freeze) | 57% | 100% | 1.4cm | 81.6% | - |
| **E042a (pos-only + freeze)** | 48% | 100% | 1.3cm | 69.0% | ❌ < E040 (66%) |
| **E042b (additive ori + freeze)** | 34% | 100% | 1.3cm | 48.3% | ❌ < E041c (57%) |
| **E042c (无 contact + freeze)** | 6% | 100% | 1.3cm | 15.7% | ≈ E036 (2%) |

## 可视化分析 (t=2.0s, box025 — 手背接触关键帧)

### E042a (pos-only + wrist freeze)

sim: 弯腰前倾, 手在箱顶面。手腕冻结后手没有明显反转, 但**整体身体仍然过度前倾趴向箱面**。手掌方向难以从这个角度判断, 但前倾姿态与 E040 类似。

### E042b (additive ori + wrist freeze)

sim: 弯腰前倾更严重, 头几乎贴箱面。手在箱面上但身体姿态**比 E042a 更不自然**。ori reward + wrist freeze 的组合反而加剧了前倾。

### E042c (无 contact + wrist freeze)

sim: 站姿直立, 手在体侧/箱面附近。没有 contact reward 时身体姿态最自然, 但手没有主动接触物体。**wrist freeze 本身不产生接触行为**。

### E042a bucket010 (t=1.7s)

sim: 站在桶旁, 手在身体侧面, 没有伸向桶。Contact 从 E040 的 66% 暴跌到 48% — wrist freeze **反而阻碍了接触**。

## 关键分析

### 1. Wrist freeze 对 Contact 有害, 不是有益

| box025 | 无 freeze | + freeze | 变化 |
|--------|-----------|----------|------|
| pos-only | 64% | 64% | = |
| additive ori | 66% | 64% | -2% |
| 无 contact | 56% | 48% | **-8%** |

| bucket010 | 无 freeze | + freeze | 变化 |
|-----------|-----------|----------|------|
| pos-only | 66% | 48% | **-18%** |
| additive ori | 57% | 34% | **-23%** |
| 无 contact | 2% | 6% | +4% |

**bucket010 受损严重**: wrist freeze 阻止 CEM 调整手腕角度去接近物体 → 手保持 body tracking 给定的姿态, 无法额外伸向桶面。

### 2. 为什么 HDMI 的 wrist freeze 有效但 CORE4D 无效?

| | HDMI move_suitcase | CORE4D box025/bucket010 |
|---|---|---|
| 手的 ref 位置 | 精确在把手上 (ref 已经是正确接触) | 在物体附近但不精确 |
| Body tracking 精度 | 很高 (HDMI 专用 scene) | MPKPE=1.3cm (手位误差 ~5-10cm) |
| wrist freeze 的效果 | 手腕保持抓握姿态 → 手掌贴把手 | 手腕保持 ref 姿态 → 但 body tracking 误差使手不在正确位置 |

**核心差异**: HDMI 的 ref 中手已经精确在把手位置, wrist freeze 只需保持姿态。CORE4D 的 body tracking 有 1-5cm 误差, 需要 CEM 微调手腕来补偿 — wrist freeze 阻止了这个补偿。

### 3. 手背接触的真正根因 (更新理解)

经过 E041 + E042 的实验, 手背接触问题的根因不是:
- ❌ CEM 能力不足 (HDMI 也用 CEM 且成功)
- ❌ 缺少 orientation reward (HDMI 也没有)
- ❌ wrist 噪声扰动 (freeze 反而有害)

而是:
- ✅ **body tracking 误差 + 动态 target = 手在错误位置被 contact reward 拉扯**
- ✅ CEM 为了同时满足 body tracking 和 contact, 用前倾 + 手腕旋转来折中
- ✅ HDMI 成功是因为 ref 动作精确到把手位置, 不需要这种折中

### 4. 对 CORE4D contact 问题的最终认识

| 阶段 | 尝试 | 结果 | 教训 |
|------|------|------|------|
| E039b | 固定 target + config bug fix | 85%/76% 但手粘连 | 固定 target 不适合自由交互 |
| E040 | 动态 per-frame target | 64%/66% 但手背接触 | position-only 无方向约束 |
| E041 | orientation reward | 62-66% 部分改善 | CEM 难同时优化 pos+ori |
| E042 | wrist freeze | 64%/48% 有害 | 阻止 CEM 补偿 body tracking 误差 |

**结论**: 在 body tracking MPKPE=1.3cm 的条件下, contact reward 能达到的上限约为 **64-66% (box025)**, 且不可避免有一些不自然帧。进一步提升需要:
1. 提高 body tracking 精度 (尤其是手部)
2. 或接受当前结果, 用 E040/E041c 作为最终 contact 配置

## 改动文件

| 文件 | 改动 |
|------|------|
| `spider/config.py` | +`zero_noise_joint_keywords` field + get_noise_scale 零化逻辑 |
| `examples/config/override/core4d_e042{a,b,c}.yaml` | 3 种变体配置 |
| `workspace/core4d/scripts/run_E042_sweep_remote.sh` | 远程 2-GPU 并行脚本 |

## 结果路径

| 产出 | 路径 |
|------|------|
| E042a box025 | `workspace/core4d/results/E042/E042a_box025.{npz,mp4}` |
| E042a bucket010 | `workspace/core4d/results/E042/E042a_bucket010.{npz,mp4}` |
| E042b box025 | `workspace/core4d/results/E042/E042b_box025.{npz,mp4}` |
| E042b bucket010 | `workspace/core4d/results/E042/E042b_bucket010.{npz,mp4}` |
| E042c box025 | `workspace/core4d/results/E042/E042c_box025.{npz,mp4}` |
| E042c bucket010 | `workspace/core4d/results/E042/E042c_bucket010.{npz,mp4}` |
| 远程脚本 | `workspace/core4d/scripts/run_E042_sweep_remote.sh` |
