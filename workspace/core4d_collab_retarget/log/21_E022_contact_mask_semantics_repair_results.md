# E022 结果：box023_p1 contact mask semantics repair

日期：2026-05-20

## 初始目标

按照 `plan/25_E022_contact_mask_semantics_repair_plan.md`，只处理 E020 的 `contact_mask` case `box023_p1`。目标是验证 processed ref contact 的 all-on 语义是否是主要瓶颈：把 mask overclaim 从 `54.4%` 降到 `<20%`，并把 contact preservation 从 E018b `22.49%` 提升到 `>=70%`，同时保持 object transport / no-fall / no-deep-penetration 不回退。

用户指定 `desk021_p1` 和 `box021_*` 不进入本轮优化成功标准；E021 已被 Holosoma RL export 占用，因此本轮优化编号从 E022 开始。

## 结果路径

| 产物 | 路径 |
|---|---|
| Plan | `workspace/core4d_collab_retarget/plan/25_E022_contact_mask_semantics_repair_plan.md` |
| Variants | `workspace/core4d_collab_retarget/scripts/E022/variants.tsv` |
| Preprocess scripts | `workspace/core4d_collab_retarget/scripts/E022/` |
| Train scripts | `workspace/core4d_collab_retarget/scripts/train/train_E022.sh`, `workspace/core4d_collab_retarget/scripts/run_E022_remote.sh` |
| Eval script | `workspace/core4d_collab_retarget/scripts/eval/eval_E022.py` |
| Results | `workspace/core4d_collab_retarget/results/E022/` |
| Comparison | `workspace/core4d_collab_retarget/results/E022/comparison.csv` |
| Aggregate | `workspace/core4d_collab_retarget/results/E022/aggregate_summary.json` |
| Unified eval | `workspace/core4d_collab_retarget/results/E022/eval_unified/` |
| Online videos | `workspace/core4d_collab_retarget/results/E022/online_video/` |
| Keyframes | `workspace/core4d_collab_retarget/results/E022/keyframes/` |
| Video-frame skill output | `workspace/core4d_collab_retarget/results/E022/video_frames_skill/` |
| Scene snapshot | `workspace/core4d_collab_retarget/results/E022/scene_snapshot/manifest.txt` |

## 执行状态

- [x] E022 task copies：4 个 `box023_person1_freejoint_legobj_e022_*` task copy 生成，未修改 E018b 原始 task。
- [x] Contact patch：3 个 patched variants 写入 `trajectory_kinematic.npz::contact[:, :2]`，baseline replay 保持 all-on ref contact。
- [x] Overrides：4 个 `examples/config/override/core4d_collab_E022_*.yaml` 生成。
- [x] Smoke：4/4 variants 4-step smoke 通过；smoke 仅验证 wiring，不用于结论。
- [x] Full：本地 baseline + 远程 3 variants 全部完成，4/4 NPZ 与 4/4 MP4 到位。
- [x] Eval：`eval_E022.py --all` 完成，`num_results=4`。
- [x] Unified eval：`eval_unified/` CSV/JSON bundle 已生成。
- [x] Visual artifacts：4 个 MP4 均为 `272` frame / `5.44s`；train keyframes 与 `video-frames` skill f115 frame 均已落盘。

## Bug 来源与修正细节

E022 修复的是 E020 `contact_mask` 分线暴露出的 contact label 语义错误，而不是训练 runtime crash。问题集中在 `box023_p1`：E018b 复制任务中的 `trajectory_kinematic.npz::contact[:, :2]` 把双手 contact 参考帧几乎标成 all-on，baseline replay 里 `Ref contact any = 100.00%`；但原始 CORE4D 3cm per-hand mask 对选中 person 的实际接触帧只有约 `46%`。这意味着 reward/eval 被告知“几乎所有帧都应该保持手-物体接触”，其中约一半帧其实没有 raw 3cm 接触证据，形成 `54.41%` 的 mask overclaim/mismatch。

这个 bug 会污染 E020 的 contact 归因：如果 ref contact gate 本身是 all-on，那么 contact reward 不再只惩罚真实接触片段里的脱离，而会在大量非接触帧也拉手去贴物体。因此低 contact preservation 不能直接解释为策略没有学会接触闭合，也可能是目标 mask 语义错误导致 reward/eval 目标不干净。

E022 的修正策略是只改 copied task，不回写 E018b 原始任务：

1. 从源 E018b mask NPZ 中读取 `eval_contact_mask_3cm` 或 `spider_contact_mask_3cm`，选择 `box023_p1` 对应 person 的左右手 mask。
2. 将 raw 3cm mask resize 到 copied task 的 `trajectory_kinematic.npz::contact` 时间长度。
3. 对 patched variants 写回 `trajectory_kinematic.npz::contact[:, :2]`，使 ref contact 与 raw 3cm selected-person mask 对齐；`raw3_dilate3_hc1` 在此基础上做 3-frame dilation。
4. 同步生成 Hydra override，让 runtime contact reward 使用同一份 mask path、person index、time-axis 语义和 contact gain/sigma。
5. 在 `eval_E022.py` 中显式比较 patched ref contact 与实际使用的 raw mask，新增 overclaim/mismatch gate，避免再次把 all-on ref contact 当作有效接触目标。

修正后，3 个 patched variants 的 ref contact any 与 used mask any 对齐到 `45.81-48.90%` 区间，mask overclaim/mismatch 降到 `0-0.44%`；baseline replay 保留 all-on ref contact 作为 control，因此仍显示 `54.41%` overclaim/mismatch。这个对照证明：E022 的 mask semantics bug 已被修掉，但它只是必要修正，不是 contact preservation 失败的充分根因。

## 量化结果

| Variant | Ref contact any | Used mask any | Mask overclaim | Mask mismatch | Contact 5cm | Epos | Erot | Deep pen | Max pen | Fall | E022 success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `baseline_replay` | `100.00%` | `46.26%` | `54.41%` | `54.41%` | `24.50%` | `0.0436m` | `1.98deg` | `0.0%` | `1.73cm` | false | false |
| `raw3_eval_axis` | `45.81%` | `46.26%` | `0.22%` | `0.44%` | **`25.30%`** | `0.0435m` | `1.91deg` | `0.0%` | `1.81cm` | false | false |
| `raw3_spider_axis` | `46.32%` | `46.32%` | `0.00%` | `0.00%` | `22.18%` | `0.0436m` | `2.00deg` | `0.0%` | `1.85cm` | false | false |
| `raw3_dilate3_hc1` | `48.90%` | `48.90%` | `0.22%` | `0.22%` | `23.68%` | `0.0421m` | `1.98deg` | `0.0%` | `1.87cm` | false | false |

Aggregate:

| Metric | Value |
|---|---:|
| `num_results` | `4` |
| `num_mask_semantics_pass` | `3` |
| `num_contact_goal_pass` | `0` |
| `num_object_no_regression_pass` | `4` |
| `num_artifact_no_regression_pass` | `4` |
| `num_E022_success` | `0` |
| `best_contact_pct` | `25.30%` (`raw3_eval_axis`) |

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 mask 语义被显式修复 | 通过：3 个 patched variants overclaim/mismatch 均 `<1%`，baseline replay 保留 `54.41%` 作为 control |
| C2 contact preservation 改善到 `>=70%` | 不通过：best only `25.30%`，只比 E018b/E022 baseline 小幅波动 |
| C3 object-side 不回退 | 通过：4/4 Epos `<0.10m`、Erot `<25deg`、transport pass |
| C4 artifact 不新增 | 通过：4/4 no fall，deep penetration `0.0%`，max penetration `<2cm` |
| C5 E022 可复现 | 通过：scripts、manifest、scene snapshot、NPZ、MP4、keyframes、comparison、aggregate、unified eval 均落盘 |

## 可视化观察

所有 E022 MP4 均成功生成，长度 `272` frames / `5.44s`。`video-frames` skill 已从 baseline、best-contact `raw3_eval_axis` 和 `raw3_dilate3_hc1` 抽取 f115 frame 到 `results/E022/video_frames_skill/`；训练脚本也为 4 个 variants 抽取了 f32/f50/f75/f90/f100/f115/f130/f160/f180/f204/f250。

视觉检查与量化结论一致：object-side transport 姿态稳定，没有出现 E018b fall 类 case 的低 pelvis/fall artifact，也没有新增明显深穿透；但手/物体接触没有因为 ref contact mask 修正而形成持续闭合。也就是说，E022 修复了“什么时候应该 contact”的语义，却没有修复“物理控制如何把手留在物体表面”的闭环。

## 结论

E022 是一个负结果但完成了归因收缩：

1. `box023_p1` 的 processed ref contact all-on bug 已被修复：mask overclaim 从 `54.41%` 降到 `0-0.22%`。
2. 单纯修 mask / axis / 3-frame dilation + mild hold-contact 不能闭合 contact preservation：best 只有 `25.30%`，远低于 `>=70%` 目标。
3. object-side support proxy、transport、no-fall、no-deep-penetration 都没有回退，说明失败不是 E018b anchor/support proxy 被破坏。

因此 E020 的 `contact_mask` 归因需要细化：mask semantics 是真实 bug，但不是充分根因。`box023_p1` 后续不应重复同类 mask sweep，应并入 E025 的 robot-side contact closure / dynamic target / penetration-aware contact-control 分线。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| 远程首次 `run_E022_remote.sh` 被远端 dirty generated files 阻塞 | 1 | 在远端创建 stash `e022-remote-dirty-backup-20260520_042537` 后 fast-forward 到 `42d2bd6`，再启动 tmux |
| 本地初次中途 eval 混入旧 4-step smoke `raw3_dilate3_hc1.npz` | 1 | 删除 stale smoke NPZ/summary，等待 full remote artifact 回收后重新 `eval_E022.py --all` |
| 本地 baseline 退出时有 EGL cleanup warning | 1 | 进程 exit 0 且 NPZ/MP4 已写入；判定为 MuJoCo EGL destructor warning，不影响结果 |
| `task_info.json not found` warning for E022 copied box023_p1 tasks | 多次 | 与源 E018b `box023_p1` task 状态一致，eval 可完成；记录 warning，不作为本轮结论指标 |

## 下一步

- E023 已完成 setup+smoke，下一步可在 remote 2 GPU 跑 `box025_p1` / `bucket007_p2` lower-body geometry repair full。
- E024 已有 plan，处理 `bucket001_p1/p2` stability。
- E025 应吸收 `box023_p1` 与原 `algo_contact` 4 cases，新增 explicit robot/object penetration penalty，并做 contact closure sweep。
