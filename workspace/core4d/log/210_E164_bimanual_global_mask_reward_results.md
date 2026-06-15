# E164 — bimanual global mask + bimanual reward results

日期：2026-06-15

计划文件：`workspace/core4d/plan/175_E164_bimanual_global_mask_reward_plan.md`

## 1. 目标

E164 验证两个用户指定的修正：

1. `raw_contact_mask_3cm` 和 `spider_contact_mask_3cm` 先取左右手 `max(L,R)`，再填充中间空洞，最后把左右手 mask 都设为这个全局 mask。
2. CEM reward 不能一只手满足就给分；`surfaceBand` 和 `contact_hdmi` 都必须双手同时满足才给分。

范围固定为三个 case：

```text
box023_person2
box004_082_p1
box026_139_p1
```

## 2. 方法

E164 以 E163 narrow surfaceBand 为 base：

```text
surface_band_min_sdf_m = -0.001
surface_band_width_m = 0.003
surface_band_sigma = 0.0015
surface_band_score_mode = symmetric_abs
surface_band_score = exp(-abs(sdf) / sigma)
```

E164 新增并打开：

```text
contact_hdmi_mask_time_axis = spider
contact_hdmi_bimanual_required = true
surface_band_bimanual_required = true
surface_band_bimanual_score_reduce = min
```

`surfaceBand` 分别计算左右手 SDF 和 score；只有左右手都在 `[-1mm, 3mm]` 内时才给分，分数取 `min(left_score, right_score)`。`contact_hdmi` 同样使用双手 AND 和瓶颈手聚合。

MuJoCo raw contact pair 不进入 CEM reward，只作为 strict eval 的 RL-safe hard gate。

## 3. 运行

预检和 smoke：

```text
py_compile: pass
bash -n: pass
git diff --check: pass
manifest: rows=3, to_run_total=3, preflight_ok=true
split_counts={local-gpu0:1, remote-gpu0:1, remote-gpu1:1}
smoke: box004_082_p1 pass
```

full 三卡并行，`WAIT_FOR_GPU_IDLE=0`，未 kill 其他程序：

| split | case |
|---|---|
| local-gpu0 | `box004_082_p1` |
| remote-gpu0 | `box023_person2` |
| remote-gpu1 | `box026_139_p1` |

本地和远程 full 均完成，回收后本地产物：

```text
root npz = 3/3
full mp4 = 3/3
trajectory_mjwp_act.npz = 3/3
config_act.yaml = 3/3
```

三条 `config_act.yaml` 均确认：

```text
contact_hdmi_mask_time_axis=spider
contact_hdmi_bimanual_required=true
surface_band_bimanual_required=true
surface_band_bimanual_score_reduce=min
surface_band_score_mode=symmetric_abs
surface_band_min_sdf_m=-0.001
surface_band_width_m=0.003
surface_band_sigma=0.0015
```

三条 trajectory 均包含：

```text
surface_band_bimanual_gate_mean
contact_hdmi_bimanual_gate_mean
```

strict eval：

```bash
.venv/bin/python workspace/core4d/scripts/eval/runners/eval_E164_bimanual_global_mask_reward.py full
```

输出：

```text
rows=12
missing=0
e164_pass=0/3
xlsx=workspace/core4d/results/E164/bimanual_global_mask_reward/eval/full/E164_bimanual_global_mask_reward_full_eval.xlsx
```

## 4. E164 结果

raw contact hard gate 仍按 E162 后固定口径：

```text
hand_object_physics_contact_in_mask_frac >= same-case rubberhand - 0.05
```

| case | rubberhand raw | hard lower bound | E163 raw | E164 raw | E164 Δ vs rubberhand | tracking | fall | status |
|---|---:|---:|---:|---:|---:|---|---|---|
| `box023_person2` | 0.9077 | 0.8577 | 0.8769 | 0.8462 | -0.0615 | true | false | fail |
| `box004_082_p1` | 0.6613 | 0.6113 | 0.5645 | 0.5645 | -0.0968 | true | false | fail |
| `box026_139_p1` | 0.5000 | 0.4500 | 0.6711 | 0.4474 | -0.0526 | true | false | fail |

失败原因全是 raw contact hard gate；三条都不是 tracking/fall 失败。

## 5. 方法级对比

| 方法 | case数 | 通过case | raw contact mean | clean3 contact mean | physPen3 mean | geomPen2 mean | worst raw Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SPIDER+rubberhand` | 3 | 3 | 0.6897 | 0.1849 | 0.2934 | 0.3171 | 0.0000 |
| `E161 releaseDecay` | 3 | 0 | 0.5906 | 0.4465 | 0.1167 | 0.0788 | -0.1538 |
| `E163 narrowSurfaceBand` | 3 | 2 | 0.7042 | 0.5696 | 0.1049 | 0.0767 | -0.0968 |
| `E164 bimanualGlobalMaskReward` | 3 | 0 | 0.6193 | 0.2857 | 0.2025 | 0.1725 | -0.0968 |

E164 的均值 raw contact 比 E163 低 `0.0848`，并且三个 case 都低于各自 hard lower bound。

## 6. 诊断

### 6.1 `box023_person2`

E163 已经把 E161 的 raw contact 回退修好：

```text
E161: 0.7538
E163: 0.8769
hard lower bound: 0.8577
```

E164 变成：

```text
E164: 0.8462
delta vs rubberhand: -0.0615
```

它只比下限低约 `0.0115`，但仍越过 hard gate。说明双手瓶颈 reward 没有提高 raw contact，反而牺牲了一部分原本 E163 保住的 contact 帧。

### 6.2 `box004_082_p1`

E163 已经在该 case fail：

```text
E163: 0.5645
hard lower bound: 0.6113
```

E164 仍是：

```text
E164: 0.5645
delta vs rubberhand: -0.0968
```

双手 global mask + 双手 reward 没有修复 E163 的 raw contact 缺口。该 case 的核心问题仍是：方法把接触从 rubberhand 的穿透式 raw contact 推向更少穿透的近接触/手侧切换，MuJoCo raw contact pair 没有增加。

### 6.3 `box026_139_p1`

这是最强负信号：

```text
rubberhand: 0.5000
E163: 0.6711
E164: 0.4474
hard lower bound: 0.4500
```

E163 在该 case 是明显通过，E164 则退回到低于下限。说明“必须双手同时满足”的瓶颈 reward 会在部分 case 上过度收紧优化空间，导致 raw contact 下降。

## 7. Claims

| Claim | 结果 | 说明 |
|---|---|---|
| C1 产物完整 | pass | 3/3 root npz、mp4、trajectory、config 齐全 |
| C2 mask/reward plumbing | pass | 三条 config 和 trajectory 诊断均证明 E164 口径生效 |
| C3 no missing eval | pass | strict eval 12 rows，missing=0 |
| C4 raw contact hard gate | fail | E164 0/3 pass |
| C5 tracking/fall | pass | E164 3/3 `success_tracked=true`，3/3 `fall=false` |

## 8. 结论

E164 是负结果，不应导出 RL-ready。

核心结论：

```text
artifact complete = 3/3
tracking pass = 3/3
fall = 0/3
raw contact hard gate = 0/3 pass
```

双手 global mask 的数据处理和双手 reward plumbing 都正确生效，但这个约束没有解决 raw contact pair 不足；它把 E163 已经恢复的 `box023_person2` 拉回 fail，并把 E163 强通过的 `box026_139_p1` 拉到下限以下。

下一步不建议继续沿着“更硬的双手同时 contact reward”加码。更合理的方向是直接补 raw-contact surrogate 或 final candidate rerank：在不放弃 E163 narrow surfaceBand 的穿透收益前提下，显式保护 active mask 内的 MuJoCo raw contact pair count。

## 9. 追加：downstream RL contact 口径

用户要求对齐 Holosoma 下游 exporter 的 `object_contact` 口径后，已在 shared eval metrics 和 E164 表中追加：

```text
rl_object_contact_ref_frac
rl_object_contact_filled_frame_count
hand_object_physics_contact_in_rl_mask_frac
hand_object_physics_contact_3mm_in_rl_mask_frac
hand_object_physics_contact_5mm_in_rl_mask_frac
```

口径来自：

```text
/home/ubuntu/Workspace/holosoma/workspace/v3/scripts/data/export_rl_motion_from_spider_tsv.py
```

其处理逻辑是：

```text
source = spider_contact_mask_3cm[:, person_idx, :]
any_hand = max(left, right)
person_mask = stack(any_hand, any_hand)
fill internal false gaps whose length <= 5 source frames
crop to aligned target window
nearest-neighbor resample to Holosoma output frames
object_contact_output_ratio = mean(object_contact)
```

CORE4D eval 没有 Holosoma 50Hz conversion/crop 上下文，因此在 SPIDER qpos 时间轴上加入等价指标：

```text
rl_object_contact_ref_frac = mean(max_lr + <=5 frame gap-fill mask)
hand_object_physics_contact_in_rl_mask_frac = raw MuJoCo hand-object contact inside that RL mask
```

E164 full eval 已重跑，表格已覆盖：

```text
workspace/core4d/results/E164/bimanual_global_mask_reward/eval/full/E164_bimanual_global_mask_reward_full_eval.xlsx
```

本轮 E164 strict eval 使用的是 E164 global spider mask；该 mask 已在构建时填满内部洞，所以三条 case 的 `rl_object_contact_filled_frame_count=0`，新增 RL contact 指标与原 `hand_object_physics_contact_in_mask_frac` 数值一致：

| case | 方法 | raw contact | RL contact | RL mask占比 | RL补洞帧数 |
|---|---|---:|---:|---:|---:|
| `box023_person2` | E164 | 0.8462 | 0.8462 | 0.4779 | 0 |
| `box004_082_p1` | E164 | 0.5645 | 0.5645 | 0.5688 | 0 |
| `box026_139_p1` | E164 | 0.4474 | 0.4474 | 0.5352 | 0 |

因此，downstream RL contact 口径不会改变 E164 结论：仍是 `0/3` pass。
