# E072 实验计划: box023 post-2s hold/place failure 诊断

## Context

E071 已修复 scene_act ctrl mapping 的 init bug：

- `run_mjwp.py` 不再把 `qpos_ref[:, :nu]` 当成 actuator ctrl。
- box023 0-2s early yaw/lunge 基本消失：
  - t=0.017/0.033s yaw err = 0.574 / 1.075 deg
  - B1 pre-contact max foot z = 0.069m
  - pelvis_min_intent = 0.674m

但 E071 不是整体成功。视频复核显示 2.0s 后 robot 没有稳定拿住箱子，箱子在 2.6-2.9s 开始脱手/落地，3.2s robot 坐压到箱子上，3.6-4.0s 摔倒。量化指标：

- first post-2s obj_err > 25cm = frame 100 / 2.00s
- post2_obj_err max/mean = 0.308 / 0.133m
- first post-2s pelvis_z < 45cm = frame 166 / 3.32s
- post2_pelvis_body_z_min = 0.200m

### 根因分析

当前还不能直接判断失败是：

1. **hold/contact 先丢**：手-箱距离或真实 hand-object contact 在 2.0s 附近先失效，随后物体偏离，最终姿态崩；
2. **putdown/stability 先崩**：手仍接近或接触箱子，但 CEM 在放下阶段让 pelvis/foot/support 失稳，导致后续脱手和摔倒；
3. **control divergence 先发生**：warmup 后 CEM ctrl 相对 ref ctrl 在手臂/下肢/物体 actuator 上突然偏离，驱动系统离开 ref manifold。

### 关键 insight

E071 的 `.npz` 已包含 `qpos/qvel/ctrl/time/trace_ref`，且保存了 `scene_snapshot/box023_person1/scene_act.xml`。因此 E072 可以不重新跑优化，直接对 E071 sim 与 converted ref 做 MuJoCo `mj_forward` replay，得到按帧同步的 contact/distance/stability/control 时间线。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1: 可复现并定位 first failure order | `diagnosis_summary.json` 给出 first hand distance/contact fail、first obj_err fail、first pelvis fall 的帧号/时间 |
| C2: 区分 hold/contact vs stability 先失败 | 时间线能判定 hand/object 指标是否早于 pelvis_z/foot/support 指标恶化至少 0.3s |
| C3: 区分 ref 本身可行还是 sim 偏离 | 同时输出 ref 与 sim 的 hand-to-box distance/contact count，并比较 post-2s 差异 |
| C4: 检查 ctrl divergence 是否是直接触发点 | 输出 post-warmup `ctrl - ctrl_ref` 的 robot/object 分组 norm/max，并标记 first large jump |
| C5: 可视化证据闭环 | 生成 2.0-3.6s 关键帧，log 中写明每帧实际观察 |

## 改动

### 1. 新增 E072 replay 诊断脚本

**文件**: `workspace/core4d/scripts/eval/eval_E072.py`

职责：

- 读取 `workspace/core4d/results/E071/E071W02_box023.npz`
- 读取 E071 scene snapshot: `workspace/core4d/results/E071/scene_snapshot/box023_person1/scene_act.xml`
- 使用 `_build_config()` + `_convert_scene_act_ref()` 得到同口径 ref qpos/ctrl
- 对 sim/ref 每帧 `mj_forward`
- 输出：
  - `timeseries.csv`: time、obj_err、pelvis_z、pelvis_err、L/R palm-to-box SDF distance、L/R hand-object contact count、foot z、ctrl norms
  - `diagnosis_summary.json`: first failure 时间、failure_order、初步分类
  - `contact_summary.csv`: 2.0-3.6s 窗口统计
  - `plots/post2_failure_timeline.png`

关键测量：

- hand sites: `left_palm`, `right_palm`
- hand collision geoms: `lh`, `rh`
- object body/geom: `object`, `object_collision`
- foot sites: `left_foot`, `right_foot`
- pelvis body: `pelvis`
- box half-size 直接从 `object_collision` geom size 读取，不 hardcode

### 2. 新增 E072 入口脚本

**文件**: `workspace/core4d/scripts/train/train_E072.sh`

虽然 E072 是 analysis-only，为遵守本地脚本规则，入口仍固化为 train script：

```bash
bash workspace/core4d/scripts/train/train_E072.sh
```

脚本只执行评估与关键帧抽取，不重新跑 CEM，不修改 scene。

### 3. 新增 E072 结果日志

**文件**: `workspace/core4d/log/92_E072_post2_hold_place_diagnosis_results.md`

记录：

- 结果路径
- 关键时间线表
- Claims 验证
- 视频/关键帧实际观察
- E073 决策树

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `workspace/core4d/scripts/eval/eval_E072.py` | 新增 replay 诊断 |
| 2 | `workspace/core4d/scripts/train/train_E072.sh` | 新增本地入口 |
| 3 | `workspace/core4d/log/92_E072_post2_hold_place_diagnosis_results.md` | 新增结果记录 |
| 4 | `workspace/core4d/EXPERIMENT_TRACKER.md` | 添加 E072 摘要 |
| 5 | `workspace/core4d/progress.md` | 记录执行进度 |

## Reward 权重

不适用。E072 不改 reward、不跑新 CEM，只分析 E071 已完成轨迹。

## 执行命令

```bash
bash workspace/core4d/scripts/train/train_E072.sh
```

## 成功标准

| 指标 | E071 | E072 目标 |
|------|------|-----------|
| first post-2s obj_err >25cm | 2.00s | 解释该时刻 hand distance/contact 是否已失效 |
| first pelvis_z <45cm | 3.32s | 判断是否晚于 contact/object failure |
| hand-object contact | 未统计 | 输出 L/R contact count 与 palm SDF distance |
| ref vs sim 差异 | 未统计 | 输出 ref/sim post-2s distance/contact 对照 |
| ctrl divergence | 未统计 | 输出 robot/object ctrl 分组偏差与 first jump |
| 可视化 | E071 keyframes | 生成并描述 2.0/2.3/2.6/2.9/3.2/3.6s 关键帧 |

## Decision Tree

| 结果 | 解读 | E073 下一步 |
|------|------|-------------|
| hand distance/contact 在 2.0s 前后先失效，pelvis 到 3.3s 才摔 | hold/contact 是一阶问题 | 加 hold/contact 约束或 hand-object target consistency，不先调 stability |
| hand contact 维持，但 pelvis/foot/support 先崩 | putdown stability 是一阶问题 | CEM delta clamp / support COM / pelvis upright prior |
| ctrl divergence 在 2.0s 附近先跳变 | CEM post-warmup trust region 问题 | 分组 ctrl trust-region 或 post-warmup delta clamp |
| ref 本身 2.0-3.6s hand distance/contact 也差 | 数据/几何不可行 | 回到 ref hand snap / contact face 修正，而不是调 CEM |

## 停止条件

- 不在 E072 中引入 reward 或配置改动。
- 如果 replay contact 受静态 `mj_forward` 限制，只把 contact count 作为辅助证据，以 SDF distance 和视觉关键帧为主。
- 如果 E071 结果文件缺失，先停止并要求重新生成 E071，而不是改实验假设。
