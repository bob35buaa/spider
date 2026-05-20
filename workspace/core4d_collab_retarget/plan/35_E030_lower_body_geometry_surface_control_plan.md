# E030 实验计划：Lower-Body Geometry / Surface-Control Repair

日期：2026-05-21

分支：`exp/core4d-collab-retarget-e030-geometry-surface-control`

## Context

E026 full eval 后，当前 P0 主要失败不再是 object tracking，而是 robot-side contact / penetration / geometry：

- `box025_p1`: best dynamic contact `66.07%`，deep pen `5.06%`，object `7.19cm`，no fall；但 E027 标记为 `retarget_questionable`，ref leg/object interference `66.5%`。
- `bucket007_p2`: best dynamic contact `29.75%`，deep pen `18.42%`，object `4.84cm`，no fall；E027 同样标记为 `retarget_questionable`，ref leg/object interference `66.3%`。
- `box023_p1/p2`: E027 timing panel 的 best shift 都是 `0`、expected gain `0pp`；这说明低接触不是简单 phase shift，更像 wrong-side / surface / geometry-control 问题。
- `bucket005_s2_p1`: E028 仍是高 contact + 高 penetration，contact `99.20%`，deep pen `92.89%`，适合作为 leg / body shortcut guard。
- `box025_p2`: 当前 strict pass guard，E026 best contact `86.93%`、deep pen `0%`，E030 不能破坏它。

E023 已证明 lower-body geometry 是部分根因：`lowerbody_proxy_min` 把 `box025_p1` ref leg/object interference 从 `66.53%` 降到 `25.40%`，把 `bucket007_p2` 从 `66.32%` 降到 `45.26%`，但仍未达到 `<15%`，也没有带来 strict success。`legpair_off` 是反例：`box025_p1` contact 到 `72.86%`，但 deep penetration 到 `32.02%`，不能作为修复。

E030 第一版不改 `spider/` 核心代码。它复用 E023 的派生 scene 机制、E028 的 SDF barrier / contact penetration gate，以及 E029 的 posture gate，先验证“更细 lower-body proxy + surface/control guard”是否能产生真实收益。如果 6 个 variants 仍没有有效信号，再决定是否进入新的 runtime surface target / candidate rejection 机制。

## 数据可用性原则

沿用 E027 的多证据弃用协议：

- 不能把 E028/E029/E030 的算法失败反向解释为数据差。
- `box025_p1`、`bucket007_p2` 只标为 `retarget_questionable`，不从分母中弃用；它们有 object tracking ok、no fall、Holosoma available 等 counter-evidence。
- `bucket001_p1/p2`、`bucket005_s2_p1/p2`、`bucket007_p1` 的 E028/E029 失败是 algorithmic / surface-control failure，不是 raw data discard evidence。
- 当前唯一 `discard_from_success_denominator=True` 的 case 仍是 `desk021_p1`。

## Claims

| Claim | 最低证据 |
|---|---|
| C1: 更细 lower-body proxy geometry 能进一步降低 `box025_p1` / `bucket007_p2` 的 ref leg/object interference | E030 target variants 的 full/case-window ref interference 相比 E023 `lowerbody_proxy_min` 继续下降，目标 `<15%` |
| C2: geometry 改善只有在不制造 sim leg/body shortcut 时才算有效 | `disabled_leg_object_pairs` 必须为空；sim leg interference 不高于 E026 best / E023 baseline；deep pen `<=15%`、max pen `<=5cm` |
| C3: `box023_p1/p2` 的低接触不是 phase shift，E030 应验证 surface/side-control 是否有信号 | 两个 diagnostic variants 输出 contact delta；若提升 `<10pp`，停止普通 contact gain sweep |
| C4: E030 不破坏 clean guard | `box025_p2` guard 保持 object/no-fall/contact/penetration，不允许变成 E028 那种 contact collapse |
| C5: bucket/leg shortcut guard 能识别高 contact 高 penetration 坏解 | `bucket005_s2_p1` variant 即使 contact 高，也必须因 penetration / sim leg artifact fail，不能进入 success |

## Scope

### Full variants

E030 限制在 6 个 full variants，方便本地 1 卡 + 远程 2 卡并行：

| Variant | Case | Source | Queue | Role | 机制 |
|---|---|---|---|---|---|
| `E030_box025_p1_tinygeom_surface_gate` | `box025_p1` | E018b canonical | local | target_geometry | 更细 lower-body proxy + hand/object barrier + contact gate |
| `E030_bucket007_p2_tinygeom_surface_gate` | `bucket007_p2` | E018b canonical | remote_gpu0 | target_geometry | 同上 |
| `E030_box023_p1_surface_hold_gate` | `box023_p1` | E022 best raw3 eval-axis | remote_gpu0 | diagnostic_surface | 不改 geometry，surface/side-control + penetration gate |
| `E030_box023_p2_surface_hold_gate` | `box023_p2` | E018b canonical | remote_gpu1 | diagnostic_surface | 不改 geometry，surface/side-control + penetration gate |
| `E030_bucket005_s2_p1_leg_guard_surface` | `bucket005_s2_p1` | E018b canonical | remote_gpu1 | shortcut_guard | leg/object penalty + contact penetration gate，识别 shortcut |
| `E030_box025_p2_guard_surface` | `box025_p2` | E018b canonical | local_after_main | clean_guard | 已通过 case 的 no-regression guard |

### 不纳入 full 的 case

- `bucket001_p1`: E029 已触发停止条件，下一步是 reachability/support timing audit，不继续 stability/contact reward sweep。
- `bucket001_p2`: 作为 E028/E029 的 bucket penetration 证据保留，E030 先用 `bucket005_s2_p1` 代表 leg/body shortcut guard。
- `desk021_p1`: 仍只保留 P1 caveat，不进 E030。

## 计划改动

### 1. E030 assets / overrides

新增脚本：

| 文件 | 作用 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/E030/variants.tsv` | 6 个 full variants 的结构化配置 |
| `workspace/core4d_collab_retarget/scripts/E030/generate_e030_assets.py` | 复制 source task、按 patch mode 派生 scene、复制 contact masks、写 manifest 与 override |
| `workspace/core4d_collab_retarget/scripts/run_E030_preprocess.sh` | 固化 E030 manifest/override 生成命令 |

Patch mode 规则：

| Patch mode | 含义 | 是否可算 success |
|---|---|---|
| `none` | 不改 scene，只改 reward/control override | 可算 success |
| `lowerbody_proxy_tiny` | 保留 leg/object collision pairs，仅把 16 个 lower-body/foot proxy 半径进一步缩小 | 可算 success，但必须通过 artifact guard |
| `legpair_off` | 禁用腿/脚-物体 pairs | E030 不生成；后续若做只能作为反例 |

### 2. 训练 / 远程执行脚本

新增脚本：

| 文件 | 作用 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/train/train_E030.sh` | 本地/远程/smoke/one/eval 入口 |
| `workspace/core4d_collab_retarget/scripts/train/train_E030_remote_tmux.sh` | 远程 GPU0/GPU1 队列 |
| `workspace/core4d_collab_retarget/scripts/run_E030_remote.sh` | `spider-remote` tmux launcher |
| `workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh` | 回收远程结果 |

并行策略：

- 本地 GPU：`E030_box025_p1_tinygeom_surface_gate`，随后 `E030_box025_p2_guard_surface`。
- 远程 GPU0：`E030_bucket007_p2_tinygeom_surface_gate` -> `E030_box023_p1_surface_hold_gate`。
- 远程 GPU1：`E030_box023_p2_surface_hold_gate` -> `E030_bucket005_s2_p1_leg_guard_surface`。

### 3. Evaluation / audit

新增：

| 文件 | 内容 |
|---|---|
| `workspace/core4d_collab_retarget/scripts/eval/eval_E030.py` | 复用 E018b evaluator，汇总 contact、deep/max penetration、object、fall、ref/sim interference、geometry patch audit |
| `workspace/core4d_collab_retarget/results/E030/case_scope.md` | E030 case 纳入/排除/guard rationale |
| `workspace/core4d_collab_retarget/results/E030/ref_interference.csv` | ref full/case-window leg/object interference before/after |
| `workspace/core4d_collab_retarget/results/E030/sim_interference.csv` | sim leg/object interference、robot/hand/leg penetration |
| `workspace/core4d_collab_retarget/results/E030/baseline_delta.csv` | 相对 E027/E026 best 的 delta |
| `workspace/core4d_collab_retarget/results/E030/comparison.csv` | variant-level full metrics |
| `workspace/core4d_collab_retarget/results/E030/aggregate_summary.json` | E030 aggregate |
| `workspace/core4d_collab_retarget/results/E030/keyframes/` | MP4 关键帧 |
| `workspace/core4d_collab_retarget/results/E030/scene_snapshot/` | active scene XML snapshots |

## 成功标准

### Target geometry

| 指标 | `box025_p1` | `bucket007_p2` |
|---|---:|---:|
| ref leg/object interference | `<15%` full 与 case-window | `<15%` full 与 case-window |
| sim leg artifact | 不高于 E023 baseline，目标 `<10%` | 不高于 E023 baseline，目标 `<10%` |
| contact 5cm | `>=70%` | `>=60%`，目标 `>=70%` |
| deep penetration | `<=15%` | `<=15%` |
| max penetration | `<=5cm` | `<=5cm` |
| object | `<=8cm` and ori `<=25deg` | `<=8cm` and ori `<=25deg` |
| fall | false | false |

### Diagnostic / guard

| Case | 标准 |
|---|---|
| `box023_p1/p2` | contact 相对 E026 best 提升 `>=10pp` 才算 surface-control 有信号；若不达标，停止普通 contact gain sweep |
| `bucket005_s2_p1` | 不允许 high-contact high-penetration 进入 success；deep pen 必须 `<=15%`、max pen `<=5cm` |
| `box025_p2` | object `<=8cm`、contact `>=80%`、deep pen `<=10%`、no fall |

Aggregate 最低目标：

- `num_target_geometry_success >= 1/2`；
- `num_clean_guard_pass = 1/1`；
- 没有任何 `disabled_leg_object_pairs` 的 success；
- 如果 target success 为 `0/2`，必须给出是否转 runtime surface target / candidate rejection 的明确结论。

## 停止条件

- 如果 `lowerbody_proxy_tiny` 把 ref interference 降低但 contact/penetration 没改善，E030 只记录为 geometry caveat，不继续 case-specific XML 微调。
- 如果任何变体通过删除 contact pair 或显著增加 penetration 获得 contact，提高项只作为反例。
- 如果 `box025_p2` guard contact 明显回退或 deep pen 超门限，暂停纳入该机制。
- 如果 `box023_p1/p2` surface-control contact 提升 `<10pp`，不继续接触权重 sweep，后续转更明确的 surface target / reference repair。

## 执行命令

### Preprocess / static checks

```bash
.venv/bin/python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E030/generate_e030_assets.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_E030.py

bash -n \
  workspace/core4d_collab_retarget/scripts/run_E030_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E030.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E030_remote_tmux.sh \
  workspace/core4d_collab_retarget/scripts/run_E030_remote.sh \
  workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh

bash workspace/core4d_collab_retarget/scripts/run_E030_preprocess.sh
```

### Smoke

```bash
RUN_TIMEOUT_SECONDS=600 RUN_STALL_TIMEOUT_SECONDS=180 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh smoke 0
```

### Full local + remote

```bash
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/train/train_E030.sh local 0

git push
RUN_TIMEOUT_SECONDS=2400 RUN_STALL_TIMEOUT_SECONDS=300 \
  bash workspace/core4d_collab_retarget/scripts/run_E030_remote.sh

bash workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E030.py --all
```

## Git strategy

1. Plan commit: `plan(core4d_collab): start E030 geometry surface control`
2. Implementation commit: `feat(core4d_collab): add E030 geometry surface control scripts`
3. Results commit: `log(core4d_collab): record E030 geometry surface control results`

当前分支已从 E029 结果分支切出，继承 E027/E028/E029 结果与默认关闭的核心 knobs。
