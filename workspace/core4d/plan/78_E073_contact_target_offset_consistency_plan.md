# E073 实验计划: 修复 dynamic contact target 的 eef_offset 口径并验证 hold/contact

## Context

E072 已定位 E071 的 post-2s failure order：

- frame 100 / eval 2.00s: obj_err=30.8cm，sim hand-object contact=0，而 ref 仍 contact=1。
- frame 166 / eval 3.32s: pelvis_z 才低于 45cm，明确摔倒。
- post2 contact frames: sim 44.4% vs ref 80.2%。
- object ctrl diff max 只有 0.01，不是 scene_act object ctrl mapping 新问题。

结论：当前一阶问题是 **hold/contact 先失效**，随后 CEM 追 object/body target 导致前扑和摔倒。

### 根因分析

复查 E040/E041c dynamic contact reward 发现一个几何口径不一致：

- target precompute 当前用 ref 的 `hand_pos = mj_data_ref.xpos[hid]`，其中 `hid` 是 `left_wrist_yaw_link/right_wrist_yaw_link` body 原点。
- reward 计算时 sim 端用的是 `contact_point = eef_pos + quat_apply(eef_quat, eef_offset)`，默认 `eef_offset=[0.05,0,0]`。

也就是说，reward 在让 **sim wrist+offset** 去追 **ref wrist origin**。这会把手部目标往腕部内侧偏 5cm，恰好和 E072 观察到的“手在箱附近但真实 contact=0”一致。

### 关键 insight

E073 不先加 stability penalty，也不先继续调 contact gain。先修几何口径，让 dynamic target 也使用 ref contact point：

```python
ref_contact_point = ref_eef_pos + quat_apply(ref_eef_quat, eef_offset)
target_local = obj_mat.T @ (ref_contact_point - obj_pos)
```

这样 reward 变成 “sim contact point 追 ref contact point”，和 E072 的 hand-object contact 诊断口径一致。

## Claims

| Claim | 最低证据 |
|-------|---------|
| C1: dynamic target 口径修正生效 | 训练日志打印 `dynamic target uses eef_offset`，config 保存字段为 true |
| C2: 0-2s early drift 不回归 | yaw t=0.017/0.033 < 2deg，B1 pre-contact max foot z <= 0.10m |
| C3: frame 100-145 hold/contact 改善 | E073 post2 sim contact frames > E071 44.4%，且 frame100-130 不再全为 0 contact |
| C4: object error 不在 frame100 即失控 | first obj_err >25cm 晚于 frame100，或 post2 obj_err max < E071 0.308m |
| C5: 不用摔倒来换 contact | first pelvis_z <45cm 不早于 E071 frame166；视频/关键帧不能出现更早前扑 |

## 改动

### 1. 新增 config 字段

**文件**: `spider/config.py`

新增：

```python
contact_hdmi_target_uses_eef_offset: bool = False
```

默认 false，保持历史实验可复现。E073 显式打开。

### 2. 修正 E040 dynamic target precompute

**文件**: `examples/run_mjwp.py`

当 `contact_hdmi_dynamic_target=true` 且 `contact_hdmi_target_uses_eef_offset=true` 时，target 从 ref wrist body origin 改为 ref contact point：

```python
eef_quat = mj_data_ref.xquat[hid]
contact_point = hand_pos + quat_apply(eef_quat, eef_offset)
target_np[t, ei] = obj_mat.T @ (contact_point - obj_pos)
```

### 3. 新增 E073 配置

**文件**: `examples/config/override/core4d_e073_box023.yaml`

继承 E071：

```yaml
defaults:
  - core4d_e071w02_box023
  - _self_

contact_hdmi_target_uses_eef_offset: true
```

### 4. 新增 E073 训练/评估脚本

**文件**:

- `workspace/core4d/scripts/train/train_E073.sh`
- `workspace/core4d/scripts/eval/eval_E073.py`

评估复用 E071/E072 指标，输出：

- early yaw/B1
- post2 hand contact/SDF/object/pelvis/ctrl metrics
- keyframes by frame index

## 需要修改的文件

| # | 文件 | 改动 |
|---|------|------|
| 1 | `spider/config.py` | 新增 `contact_hdmi_target_uses_eef_offset` |
| 2 | `examples/run_mjwp.py` | dynamic target 可选使用 ref eef_offset |
| 3 | `examples/config/override/core4d_e073_box023.yaml` | 新增 E073 override |
| 4 | `workspace/core4d/scripts/train/train_E073.sh` | 新增训练入口 |
| 5 | `workspace/core4d/scripts/eval/eval_E073.py` | 新增评估 |
| 6 | `workspace/core4d/log/93_E073_contact_target_offset_consistency_results.md` | 结果记录 |
| 7 | `workspace/core4d/EXPERIMENT_TRACKER.md` | 添加 E073 摘要 |
| 8 | `workspace/core4d/progress.md` | 记录进度 |

## Reward 权重

E073 不改 reward 权重，只修 dynamic target 几何口径。

| 类别 | E071 | E073 |
|------|------|------|
| contact_hdmi_gain | 5.0 | 5.0 |
| contact_hdmi_sigma | 0.3 | 0.3 |
| contact_hdmi_eef_offset | `[0.05,0,0]` | `[0.05,0,0]` |
| contact_hdmi_dynamic_target | true | true |
| contact_hdmi_target_uses_eef_offset | false | **true** |
| warmup_steps | 0.20 | 0.20 |

## 执行命令

```bash
bash workspace/core4d/scripts/train/train_E073.sh 0
```

## 成功标准

| 指标 | E071 | E073 目标 |
|------|------|-----------|
| yaw t=0.017/0.033 | 0.574 / 1.075 deg | < 2 deg |
| B1 pre-contact max foot z | 0.069m | <= 0.10m |
| post2 sim contact frames | 44.4% | > 44.4%，最好接近 ref 80.2% |
| frame100-130 contact | 全 0 | 至少部分帧非 0 |
| post2 obj_err max | 0.308m | < 0.308m 或 first >25cm 晚于 frame100 |
| first pelvis_z <45cm | frame166 | 不早于 frame166 |
| 视频 | 2s 后脱手前扑 | 2.0-2.9s 手-箱关系更稳定，无更早摔倒 |

## 可视化规则

训练完成后如果需要观察 keyframes/video，交给 subagent 执行视觉复核并回收结果。主线程只整合 subagent 观察到的具体帧描述，不直接用 `view_image` 做主观判断。

## Decision Tree

| 结果 | 解读 | 下一步 |
|------|------|--------|
| contact/object 改善且不更早摔 | 目标口径是关键因素 | E074 可加 robot ctrl trust-region guard 做稳定化 |
| contact 改善但更早摔 | contact reward 有效但需要 stability/support guard | E074 = contact target fix + ctrl/pelvis guard |
| contact 无改善 | 目标口径不是主因，CEM 搜索/动力学接触不足 | E074 转 robot ctrl trust-region 或真实 contact-count surrogate |
| early drift 回归 | 修正意外影响 ref/control pipeline | 回滚 E073 代码路径，重新对齐 E071 |

## 停止条件

- 不同时改 contact gain、stability penalty、task_obj 权重。
- 不做 box025 regression，除非 box023 hold/contact 明确改善。
- 如果训练失败，先修运行/保存链路，不把失败解释为 reward 结论。
