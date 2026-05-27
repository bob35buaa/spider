# E030 D6 locked 3-case CEM 结果

日期：2026-05-27

## 目标

E029 的正式 gate 结论是 `stop_before_full_sanity_failed`：5 个 Box021 candidates 中，最佳 gate-eligible sanity 只有 `3/5`，低于 plan 34 要求的 `>=4/5`。

本轮按用户要求越过该 gate，只把已经通过 E029 D6 locked no-training sanity 的 3 个 case 接 full CEM，用本地 1 张 GPU + 远程 2 张 GPU 并行运行。这个实验用于回答一个实际问题：右侧 D6 locked support 结果如果接 CEM，能不能修掉物体漂移和机器人-物体脱离。

## 运行配置

| 分配 | Variant | Source task |
|---|---|---|
| 本地 GPU0 | `E029_d003_box021_20231018_029_p2_d6_locked` | `d003_box021_20231018_029_p2` |
| 远程 GPU0 | `E029_d003_box021_20231011_035_p2_d6_locked` | `d003_box021_20231011_035_p2` |
| 远程 GPU1 | `E029_d003_box021_20231020_019_p1_d6_locked` | `d003_box021_20231020_019_p1` |

脚本：

- `workspace/core4d_collab_retarget/scripts/train/train_E030.sh`
- `workspace/core4d_collab_retarget/scripts/run_E030_remote.sh`
- `workspace/core4d_collab_retarget/scripts/pull_E030_remote_results.sh`
- `workspace/core4d_collab_retarget/scripts/eval/eval_E030.py`

执行期间只检查 GPU 状态，没有 kill 或 reset 其他已有程序。远程仓库存在 dirty files，因此本轮没有 `git pull`，而是只用 `rsync` 同步 E030 需要的脚本、override、data、contact mask、scene 和 Box021 资产。

## 结果路径

| 类型 | 路径 |
|---|---|
| CEM 输出 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/` |
| 在线视频 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/online_video/` |
| 关键帧 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/keyframes/` |
| 视觉 sheet | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/visual_sheets/` |
| 量化汇总 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/comparison.csv` |
| 聚合指标 | `workspace/core4d_collab_retarget/results/E030/d6_locked_cem/aggregate_summary.json` |
| 运行日志 | `logs/core4d_collab_retarget/E030/` |

3/3 root `*.npz`、3/3 outdir `trajectory_mjwp.npz`、3/3 `online_video/*.mp4`、3/3 keyframes 都已回收到本地。

视频规格：

| Variant | 视频 |
|---|---|
| `20231011_035_p2` | `1440x480 @ 50fps`, 266 frames, 5.32s |
| `20231018_029_p2` | `1440x480 @ 50fps`, 150 frames, 3.00s |
| `20231020_019_p1` | `1440x480 @ 50fps`, 196 frames, 3.92s |

日志尾部有 MuJoCo EGL cleanup warning，但 shell 输出 `=== E029 one done ===`，结果文件完整；这些 warning 不影响本轮结论。

## 量化结果

`eval_E030.py` 只读取 3 个 E030 variants，并显式使用 `scene_e029_d6_locked_*.xml`，避免误用默认 `scene.xml`。

聚合结果：

| 指标 | 数值 |
|---|---:|
| 结果数 | `3` |
| case-window 成功数 | `0/3` |
| contact + object proxy 成功数 | `0/3` |
| case-window object error 均值 | `0.855m` |
| case-window contact pct 均值 | `64.08%` |
| carry progress ratio 均值 | `0.014` |
| object rotation error 均值 | `12.75deg` |

逐 case：

| Variant | Object mean/max | Contact pct | Pelvis min | Progress ratio | 标签 |
|---|---:|---:|---:|---:|---|
| `20231011_035_p2` | `0.909/1.636m` | `91.63%` | `0.333m` | `-0.021` | fail |
| `20231018_029_p2` | `0.703/1.151m` | `31.01%` | `0.363m` | `0.047` | fail |
| `20231020_019_p1` | `0.954/1.640m` | `69.59%` | `0.175m` | `0.015` | fail |

对比 E029 no-training sanity，同 3 个 case 的 sanity object mean 只有 `0.105-0.113m` 且都 pass gate；接 CEM 后 object mean 升到 `0.703-0.954m`。这说明 D6 locked load path 在 no-training sanity 中能拖动物体，但进入 full CEM 后，机器人动态、碰撞和 support-object coupling 会把系统推离参考。

## 视觉观察

主线程抽取了 6 帧 sheet：

- `visual_sheets/E029_d003_box021_20231011_035_p2_d6_locked_sheet.jpg`
- `visual_sheets/E029_d003_box021_20231018_029_p2_d6_locked_sheet.jpg`
- `visual_sheets/E029_d003_box021_20231020_019_p1_d6_locked_sheet.jpg`

独立 xhigh subagent 审查结论：

| Case | 观察 | 标签 |
|---|---|---|
| `20231011_035_p2` | 早期接近箱体，但中段开始机器人趴到箱体上，头/躯干/手臂与箱体严重穿插；后半程箱体更像被 support/pad 顶着，机器人不是有效抓扶。 | fail |
| `20231018_029_p2` | 初期手和箱体已有间隙/弱接触；中段箱体明显倾斜、旋转，机器人上身翻到箱体后方/上方；后期箱体被 support/pad 托住并翻起。 | fail |
| `20231020_019_p1` | 初期机器人离箱体较远，接触不稳定；中段短暂靠近后快速倒伏，腿/手/躯干和箱体互相穿插；最终机器人倒在箱体旁/下。 | fail |

视觉上 E030 没有修复 E029 D6 locked 的漂移/脱手，反而放大成倒伏、穿插、support 外露和非物理支撑。

## 为什么右侧不如左侧 OmniRetarget

三列对比里的右侧不是 CEM 后的最终优化，而是 E029 D6 locked no-training sanity。它的含义是：机器人大体走参考，support body 和物体通过 weld/equality load path 做物理积分，用来测试 support-object scaffold 能不能独立拖动物体。

因此它天然有两个问题：

- 物体运动可以主要由 support body 完成，机器人-物体真实接触不是因果闭环；
- CEM 进入 full dynamics 后，机器人姿态、碰撞、接触 reward 会和这个 support scaffold 竞争，优化器容易找到倒伏、穿插、靠 support 托物体的捷径。

这就是为什么你看到右侧物体漂移、机器人经常没接触，并且 E030 接 CEM 后进一步退化。这个现象不是单纯帧率或可视化问题，也不是某个 case 的偶然误差，而是当前 D6 locked support 分支的 load-path 语义不可靠。

## 声明验证

| Claim | 结果 | 证据 |
|---|---|---|
| C1 只运行 sanity 通过的 3 个 D6 locked case | 通过 | `train_E030.sh` 固化 3 variants |
| C2 不清理其他 GPU 程序 | 通过 | 只做 GPU 检查和 tmux 启动，没有 kill/reset |
| C3 本地 1 卡 + 远程 2 卡并行启动 | 通过 | 本地/远程日志均完成 |
| C4 结果可回收 | 通过 | `pull_E030_remote_results.sh` 已回收远程结果 |
| C5 full CEM 输出完整 | 通过 | 3/3 npz、trajectory、mp4、keyframes 存在 |
| C6 视觉审查由高推理子代理完成 | 通过 | 子代理给出 3/3 fail 观察 |
| C7 D6 locked 接 CEM 能改善右侧质量 | 失败 | 量化 `0/3` 成功，视觉 `3/3 fail` |

## 决策

当前 E030 D6 locked CEM 不应作为后续主线。

如果必须在 `original OmniRetarget`、`E018b`、`E029 D6 locked` 三者里选一个接下一轮 CEM，建议选 **original OmniRetarget** 作为起点/约束来源，而不是 E018b 或 E029 D6 locked：

- OmniRetarget 在三列对比中 robot-object 相对位置和接触语义最可信；
- E018b 在 Box021 上是坏基线，anchor/support proxy 泛化差；
- E029 D6 locked 比 E018b 稳一些，但稳定性来自 support scaffold，不来自真实 robot-object contact；E030 已验证它接 CEM 后会退化。

后续如果要继续“COLA-style support body”方向，需要另开方法分支：true per-axis D6 / relative-pose impedance support-object controller，并且先通过 no-training + short-CEM 双 gate；不能继续在当前 equality weld / connect anchor 上做小调参。
