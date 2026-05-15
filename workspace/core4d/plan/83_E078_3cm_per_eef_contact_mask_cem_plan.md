# E078 实验计划: 3cm per-EEF contact mask 对齐 HDMI，并在 box023 p1/p2 上做 CEM 动力学重定向

## Context

E076/E077 已把 contact mask 的来源问题拆清楚：

- CORE4D raw 没有人工 hand-contact 真值。
- 官方可视化使用 3cm 几何阈值；E077 已按这个口径生成 `box023` 的 per-person/per-hand contact proxy。
- E077 输出：
  - `raw_contact_mask_3cm`: `(178,2,2)`
  - `spider_contact_mask_3cm`: `(136,2,2)`
  - `eval_contact_mask_3cm`: `(227,2,2)`
  - axis: person=`person1/person2`, hand=`left/right`
- 现有 MJWP 的 E039/E075 mask 仍然是 scalar `(T,)`，任意一只手接近就打开双手 reward，不是 HDMI 的 per-EEF mask。
- `box023_person2` 已构造成单人 SPIDER case，可以单独跑 CEM；但 p1/p2 retarget qpos 不能直接合成双机器人同场景，因为 Holosoma 按各自 `smpl_scale` 缩放 object 轨迹。

本轮目标是先做 **单人 p1 和单人 p2 两个 CEM 动力学重定向实验**，不是双机器人同场景合成。

## Claims

| ID | Claim | 成功标准 |
|----|-------|----------|
| C1 | MJWP contact_hdmi reward 支持 HDMI-style per-EEF mask | `approach_mask_val` 支持 `(N,2)`，reward 左右手独立 gated，不再 scalar broadcast |
| C2 | E077 3cm mask 可以按 case 读取 | p1 使用 `person_idx=0`，p2 使用 `person_idx=1`，mask 与 ref/eval 时间轴对齐并写入 log |
| C3 | p1 的 f115-f130 gait/putdown phase mismatch 减轻 | 对比 E075B：右腿 f120-f125 前跨幅度、`right_hip_pitch_joint` ctrl 偏离、post2 contact/object error 至少一项改善，且 early drift 不回归 |
| C4 | p2 单人 case 能完成 CEM sanity run | 生成 p2 npz/mp4/metrics；不要求一次成功，但必须能稳定加载 scene/scene_act 和 3cm mask |
| C5 | 不继续依赖 hand-crafted hold window | E078 主变量是 3cm per-EEF mask；不新增/调参 `hold_contact_start/end_eval_time` |

## 实验矩阵

| Run | Case | Base | Mask | 目的 |
|-----|------|------|------|------|
| E078A | `box023_person1` | E075B config | E077 3cm `person_idx=0` | 主验证：修正 p1 contact mask 后是否改善 f115-f130 |
| E078B | `box023_person2` | E075B-like p2 config | E077 3cm `person_idx=1` | p2 单人 CEM sanity + 对比 p2 强接触行为 |

E078A 使用当前最好的 p1 基线：

```text
core4d_e075b_box023
  -> core4d_e074a_box023
  -> core4d_e073_box023
  -> core4d_e071w02_box023
  -> core4d_e062_box023
  -> core4d_e041c
```

E078B 新增 p2 override，原则上继承同一 reward stack，但 task 改为 `box023_person2`。如果 E062 palm normal 是 p1 特化的 `[+x,+x]`，p2 必须先独立核验/自动计算 palm normal；不能盲目复用 p1 的 palm normal 当作“已对齐”。

## 实现计划

### 1. Config 增加 mask source

新增配置字段，保持默认不改变旧行为：

```python
contact_hdmi_mask_source: str = "rotated_sdf"  # "rotated_sdf" | "core4d_3cm"
contact_hdmi_mask_path: str = ""
contact_hdmi_mask_person_idx: int = 0
contact_hdmi_mask_time_axis: str = "spider"  # "spider" 或 "eval"，实现时以 run_mjwp 实际 ref 时间轴为准
contact_hdmi_mask_min_dist_key: str = "spider_min_dist_m"
```

默认 `rotated_sdf` 保持 E073/E075 兼容。

### 2. run_mjwp.py 读取 E077 mask

当前 E039b 逻辑生成 scalar：

```python
per_eef_mask_np = np.zeros(T_mask)
if any hand close:
    per_eef_mask_np[t] = 1.0
```

E078 改为：

- `rotated_sdf`: 生成 `(T, n_eef)`，每只手独立判断。
- `core4d_3cm`: 从 E077 npz 读取 `(T,2,2)`，选择 `person_idx` 后得到 `(T,2)`。
- 若 ref 长度与 mask 长度不同，必须显式记录映射策略：
  - 优先使用 `spider_contact_mask_3cm` 对齐 `trajectory_kinematic` 的 136 帧。
  - 若 `run_mjwp.py` 当前用的是 50Hz eval ref，则使用 `eval_contact_mask_3cm`。
  - 不允许 silent truncate；必须 log 原始长度、目标长度和 resize/index 规则。

### 3. mjwp.py reward 支持 per-EEF mask

当前 reward：

```python
rew_stack = torch.stack(per_eef_rew, dim=1)  # (N,2)
mask = approach_mask_val                   # scalar or (N,)
contact_hdmi_rew = (rew_stack * mask * gain + (1.0 - mask)).mean(dim=1)
```

E078 改为支持：

- scalar：旧行为。
- `(N,)`：旧 scalar-time mask，broadcast 到 eef。
- `(N,2)`：per-EEF mask，逐手 gated。

目标公式：

```python
mask_eef = normalize_mask_to_shape(mask, rew_stack)  # (N,2)
contact_hdmi_rew = (rew_stack * mask_eef * gain + (1.0 - mask_eef)).mean(dim=1)
```

### 4. 新增 E078 configs/scripts

新增 override：

```text
examples/config/override/core4d_e078a_box023_p1_3cm.yaml
examples/config/override/core4d_e078b_box023_p2_3cm.yaml
```

新增运行脚本：

```text
workspace/core4d/scripts/train/train_E078.sh
workspace/core4d/scripts/eval/eval_E078.py
workspace/core4d/scripts/run_E078_remote.sh
workspace/core4d/scripts/pull_E078_remote_results.sh
```

本机只有 1 GPU，远程有 2 张 A6000；E078A/E078B 可以远程并行：

- GPU0: E078A p1
- GPU1: E078B p2

## 评估指标

### 通用指标

| 指标 | 说明 |
|------|------|
| early yaw err | t=0.017/0.033s，确认 E071 ctrl mapping 不回归 |
| B1 pre-contact max foot z | 0-2s 下肢 early drift |
| first obj_err >25cm | 物体失控时间 |
| post2 contact ratio | 2s 后 hand-object contact |
| post2 obj_err max/mean | 物体 tracking |
| post2 pelvis_z min | 稳定性 |
| robot ctrl Linf | 是否仍有 f120-f125 大偏离 |

### p1 特化指标

重点复查 E075B 的 f115-f130：

- f119->f125 sim right foot XY step vs ref。
- f120-f124 `right_hip_pitch_joint` ctrl diff。
- f115-f130 per-hand mask：p1 L/R 是否按 3cm mask 独立 gating。
- 视频关键帧：f115-f130, f145, f160, f180。

### p2 特化指标

- scene/scene_act load。
- CEM 是否能完成全 horizon 并保存 npz/mp4。
- p2 mask 3cm 下 L/R 强接触阶段是否对应合理动作。
- 不把 p2 与 p1 直接合并，避免 common-scale 问题污染结论。

## 成功/失败判定

E078A 通过条件：

- early drift 不回归：yaw err <2deg，B1 <=0.10m。
- f115-f130 右腿 phase mismatch 至少一个核心指标改善：
  - right foot XY step 降低；
  - `right_hip_pitch_joint` ctrl diff 降低；
  - post2 contact/object error 改善。
- 不出现 E075A 那种 pelvis_z <0.45m 摔倒回归。

E078B 通过条件：

- 运行完整并保存结果。
- scene/scene_act/3cm mask 对齐无异常。
- 视频与数值能形成 p2 单人可用性判断。

如果 E078A 无改善但 E078B 明显更好，下一步不是直接切 p2，而是分析 p2 raw contact/retarget 几何是否更适合单 G1；仍要处理 p1/p2 common-scale 后才讨论双人合成。

## 风险与回滚

| 风险 | 处理 |
|------|------|
| `approach_mask_val` 形状影响其他 reward | mask normalization 只在 `contact_hdmi_rew` 内局部处理；旧 scalar 行为保持 |
| run_mjwp ref 时间轴不是 136 帧 | 实现时先打印 qpos_ref/mask length，显式选择 `spider` 或 `eval` |
| p2 palm normal 不适配 | 先自动核验 p2 palm normal；不盲用 p1 `[+x,+x]` |
| 3cm mask 过宽 | 同时记录 min_dist/vertex_count，后续可 soft mask |
| p2 object scale 与 p1 不一致 | 本轮 p2 只做单人 case；不做双人合成 |

## 待用户确认

确认后执行：

1. 修改 config/run_mjwp/mjwp，支持 `core4d_3cm` per-EEF mask。
2. 新增 E078A/E078B configs 和远程脚本。
3. push 后远程并行启动 E078A/E078B。
4. 回收结果，抽关键帧，写 `log/99_E078_3cm_per_eef_mask_results.md`。
