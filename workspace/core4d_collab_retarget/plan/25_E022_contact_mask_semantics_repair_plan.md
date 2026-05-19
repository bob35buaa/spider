# E022 实验计划：box023_p1 contact mask semantics repair

日期：2026-05-20

## Context

E020 对 E018b 13-case 做完 root-cause audit 后，用户要求忽略 `desk021_p1` 和 `box021_*`，其余 case 按 cause 分类优化；E021 已被 Holosoma RL export 占用，所以优化编号从 E022 开始。

E022 只处理 `contact_mask` 类失败：`box023_p1`。

### 根因分析

`box023_p1` 的 E018b object-side 已成立：

- Epos `0.043m`，Erot `1.96deg`。
- no fall，deep penetration `0.0%`。
- 失败集中在 contact preservation：`22.49%`。
- E020 S3 显示 current contact all-on vs raw 3cm selected-person contact only `46.3%`，mask overclaim `54.4%`。

这说明问题不在 canonical support proxy 或 object transport；优先验证 contact mask / reward gate 的语义，而不是先做 controller 大 sweep。

### 关键 insight

E018b pipeline 已有稳定的 derived task、canonical anchor、scene snapshot、online video 和 eval。E022 应只改 contact mask 语义与最小 contact reward gate，保持 object-side support proxy 不变，才能判断 `contact_mask` 归因是否成立。

## Claims

| Claim | 最低证据 |
|---|---|
| C1 mask 语义被显式修复 | E022 eval 输出 mask active/mismatch/overclaim；`box023_p1` overclaim 从 `54.4%` 降到 `<20%` |
| C2 contact preservation 改善 | `box023_p1` 5cm contact preservation 从 `22.49%` 提升到 `>=70%` |
| C3 object-side 不回退 | object Epos `<0.10m`，Erot `<25deg`，transport pass |
| C4 artifact 不新增 | robot fall false，deep penetration `<=10%`，floor/leg gate 不比 E018b 更差 |
| C5 E022 可复现 | 本地脚本、远程脚本、scene snapshot、manifest、eval CSV/JSON、online MP4 均落盘 |

## 改动

### 1. E022 variants / masks

**文件**：`workspace/core4d_collab_retarget/scripts/E022/variants.tsv`

计划首批 variants：

| Variant | mask | reward change | 目的 |
|---|---|---|---|
| `E022_box023_p1_baseline_replay` | E018b copied raw 3cm | 不 patch ref contact | 复现基线，排除脚本迁移误差 |
| `E022_box023_p1_raw3_eval_axis` | `eval_contact_mask_3cm` per-EEF | patch ref contact + same contact gain | 强制 eval axis，排查 auto axis |
| `E022_box023_p1_raw3_spider_axis` | `spider_contact_mask_3cm` per-EEF | patch ref contact + same contact gain | 强制 spider axis，排查插值/长度 |
| `E022_box023_p1_raw3_dilate3_hc1` | raw 3cm + 3-frame dilation | patch ref contact + `hold_contact_rew_scale=1.0` | 容忍短 mask gap，验证 contact closure |

如果首批 4 个都失败，第二批不得重复同配置；改为分析 video/curves 后决定是否需要修改 dynamic target 或进入 E025 contact closure。

### 2. Override generator

**文件**：`workspace/core4d_collab_retarget/scripts/E022/generate_e022_overrides.py`

设计：

- 读取 `results/E018b/manifest.tsv` 中 `E018b_box023_p1_canonical_t02` 的 row。
- 生成 `box023_person1_freejoint_legobj_e022_*` task copy；`scene_name` / canonical anchor 继承 E018b。
- 在 task copy 中 patch `trajectory_kinematic.npz::contact[:, :2]`，让 processed ref contact 不再 all-on；不修改 E018b 原始任务。
- 复制 mask 到 `results/E022/contact_masks/<variant>/`。
- 对每个 variant 写 `examples/config/override/core4d_collab_E022_*.yaml`。
- 保持 true-freejoint invariant：`contact_guidance=false`、`object_action_dims=0`、`object_actuator_ids=[]`。

### 3. Train / remote scripts

**文件**：

- `workspace/core4d_collab_retarget/scripts/train/train_E022.sh`
- `workspace/core4d_collab_retarget/scripts/run_E022_preprocess.sh`
- `workspace/core4d_collab_retarget/scripts/run_E022_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E022_remote_results.sh`

设计：

- `train_E022.sh smoke 0`：4 variant 4-step smoke。
- `train_E022.sh local 0`：本地跑 baseline replay 或最关键 variant。
- `run_E022_remote.sh`：远程 2 GPU 跑剩余 variants，同 GPU 串行。
- 训练前按实际 variant task 调用：`bash workspace/core4d_collab_retarget/scripts/convert/snapshot_scenes.sh E022 <box023_person1_freejoint_legobj_e022_*>`。

### 4. Eval

**文件**：`workspace/core4d_collab_retarget/scripts/eval/eval_E022.py`

设计：

- 复制 E018b evaluator 的核心字段，但 `RESULTS` / `MANIFEST` 指向 E022。
- 保留 paper metrics：object Epos/Erot、transport、contact preservation、deep penetration、fall/upright。
- 新增 mask 字段：mask source/key/time axis、left/right active pct、any active pct、raw/current mismatch/overclaim。
- 只在 E022 log 中使用 smoothness / foot-skating 作参考，不作为成败标准，因为 E019 FPS P2 尚未修。

## 需要修改的文件

| # | 文件 | 改动 |
|---|---|---|
| 1 | `workspace/core4d_collab_retarget/scripts/E022/variants.tsv` | 定义 4 个 contact-mask variants |
| 2 | `workspace/core4d_collab_retarget/scripts/E022/generate_e022_masks.py` | 复制 E018b task，patch ref contact，复制/派生 raw 3cm mask，生成 manifest |
| 3 | `workspace/core4d_collab_retarget/scripts/E022/generate_e022_overrides.py` | 生成 E022 Hydra overrides |
| 4 | `workspace/core4d_collab_retarget/scripts/run_E022_preprocess.sh` | 生成 masks + overrides |
| 5 | `workspace/core4d_collab_retarget/scripts/train/train_E022.sh` | smoke/local/remote queue/one/eval 入口 |
| 6 | `workspace/core4d_collab_retarget/scripts/train/train_E022_remote_tmux.sh` | 远程 GPU0/GPU1 队列入口 |
| 7 | `workspace/core4d_collab_retarget/scripts/run_E022_remote.sh` | 本地 SSH/tmux remote launcher |
| 8 | `workspace/core4d_collab_retarget/scripts/pull_E022_remote_results.sh` | 回收 remote NPZ/MP4/log |
| 9 | `workspace/core4d_collab_retarget/scripts/eval/eval_E022.py` | E022 专用 eval |
| 10 | `workspace/core4d_collab_retarget/log/21_E022_contact_mask_semantics_repair_results.md` | 实验完成后写结果 |
| 11 | `workspace/core4d_collab_retarget/EXPERIMENT_TRACKER.md` | 实验完成后更新状态与指标 |

## Reward / config 变更

| 类别 | E018b baseline | E022 首批 |
|---|---:|---:|
| `contact_hdmi_gain` | inherited `5.0` | 先保持 `5.0` |
| `contact_hdmi_sigma` | inherited `0.3` | 先保持 `0.3` |
| `contact_hdmi_dynamic_target` | inherited true | true |
| `contact_hdmi_target_uses_eef_offset` | inherited true | true |
| `hold_contact_rew_scale` | `0.0` | sweep `0.0` / `1.0` |
| `hold_contact_sigma` | `0.05` | `0.05` |
| `support_proxy_gravity_scale` | `0.5` | 保持 |
| `support_proxy_mode` | `mocap_pad` | 保持 |
| `stability_penalty_scale` | inherited `0.0` | 保持 `0.0`，E022 不做 stability sweep |

## 训练命令

```bash
bash workspace/core4d_collab_retarget/scripts/run_E022_preprocess.sh
bash workspace/core4d_collab_retarget/scripts/train/train_E022.sh smoke 0

# full: 本地 + 远程 2 GPU
bash workspace/core4d_collab_retarget/scripts/train/train_E022.sh local 0
bash workspace/core4d_collab_retarget/scripts/run_E022_remote.sh
bash workspace/core4d_collab_retarget/scripts/pull_E022_remote_results.sh

# eval
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E022.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/unified_eval.py \
  --method spider_E022 \
  --comparison workspace/core4d_collab_retarget/results/E022/comparison.csv \
  --out workspace/core4d_collab_retarget/results/E022/eval_unified
```

## 成功标准

| 指标 | E018b `box023_p1` | E022 目标 |
|---|---:|---:|
| Mask overclaim | `54.4%` | `<20%` |
| Contact preservation 5cm | `22.49%` | `>=70%` |
| Deep penetration duration | `0.0%` | `<=10%` |
| Robot fall | false | false |
| Object Epos | `0.043m` | `<0.10m` |
| Object Erot | `1.96deg` | `<25deg` |
| Transport | pass | pass |

## 决策规则

- 如果 raw axis/dilation 能把 mask overclaim 降到 `<20%`、contact 提到 `>=70%` 且不引入 artifact，E022 通过，进入 E023。
- 如果 ref contact overclaim 修复但 contact 仍低，E022 记录为“mask 数据修正不足以闭合 contact”，下一步把 `box023_p1` 转入 E025 contact closure，但不重复同一 mask sweep。
- 如果 object / stability 回退，优先看 online MP4 和 E020-style overlay，不能直接扩大 sweep。
