# E028 结果：D003 Box021 13-case SPIDER dynamic retarget

日期：2026-05-27

## 初始目标

使用 Holosoma D003 中 `进入下一轮` 的 13 条 Box021 case-person，按 E018b 的 canonical support proxy 配置跑 SPIDER 动力学重定向。约束是本地 1 卡串行执行，不使用远端，不中断同时运行的 Holosoma `r097s_train`。

计划文件：

- `/home/ubuntu/Workspace/holosoma/workspace/v3/plan/13_E028_d003_box021_spider_dynamic_retarget_plan.md`

## 结果路径

| 产物 | 路径 |
|---|---|
| Manifest | `workspace/core4d_collab_retarget/results/E028/manifest.tsv` |
| Results | `workspace/core4d_collab_retarget/results/E028/` |
| Online videos | `workspace/core4d_collab_retarget/results/E028/online_video/` |
| Eval comparison | `workspace/core4d_collab_retarget/results/E028/comparison.csv` |
| Eval aggregate | `workspace/core4d_collab_retarget/results/E028/aggregate_summary.json` |
| Logs | `logs/core4d_collab_retarget/E028/` |

## 执行命令与过程

### 1. 实现与静态检查

新增脚本：

```text
workspace/core4d_collab_retarget/scripts/E028/build_e028_manifest.py
workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py
workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py
workspace/core4d_collab_retarget/scripts/run_E028_preprocess.sh
workspace/core4d_collab_retarget/scripts/train/train_E028.sh
workspace/core4d_collab_retarget/scripts/eval/eval_E028.py
workspace/core4d_collab_retarget/scripts/eval/index_E028_online_videos.py
workspace/core4d_collab_retarget/scripts/eval/watch_E028_eval_after_full.sh
```

命令：

```bash
cd /home/ubuntu/Workspace/spider
python -m py_compile \
  workspace/core4d_collab_retarget/scripts/E028/build_e028_manifest.py \
  workspace/core4d_collab_retarget/scripts/E028/generate_e028_assets.py \
  workspace/core4d_collab_retarget/scripts/E028/generate_e028_overrides.py \
  workspace/core4d_collab_retarget/scripts/eval/eval_E028.py \
  workspace/core4d_collab_retarget/scripts/eval/index_E028_online_videos.py

bash -n workspace/core4d_collab_retarget/scripts/run_E028_preprocess.sh
bash -n workspace/core4d_collab_retarget/scripts/train/train_E028.sh
bash -n workspace/core4d_collab_retarget/scripts/eval/watch_E028_eval_after_full.sh
git diff --check -- workspace/core4d_collab_retarget/scripts/E028 \
  workspace/core4d_collab_retarget/scripts/run_E028_preprocess.sh \
  workspace/core4d_collab_retarget/scripts/train/train_E028.sh \
  workspace/core4d_collab_retarget/scripts/eval/eval_E028.py \
  workspace/core4d_collab_retarget/scripts/eval/index_E028_online_videos.py \
  workspace/core4d_collab_retarget/scripts/eval/watch_E028_eval_after_full.sh
```

结论：

| 步骤 | 输入 | 进入下一步 | 边界/review | 抛弃/跳过 |
|---|---:|---:|---:|---:|
| 静态检查 | 8 个新增脚本 | 8 | 0 | 0 |

### 2. Preprocess / manifest

命令：

```bash
cd /home/ubuntu/Workspace/spider
bash workspace/core4d_collab_retarget/scripts/run_E028_preprocess.sh --force
```

产物：

- `workspace/core4d_collab_retarget/results/E028/manifest.tsv`
- 13 个 derived task：`<target_task>_freejoint_legobj_e028`
- 13 个 override：`examples/config/override/core4d_collab_E028_<target_task>_canonical_t02.yaml`

结论：

| 步骤 | 输入 | 进入下一步 | 边界/review | 抛弃/跳过 |
|---|---:|---:|---:|---:|
| Preprocess | 13 | 13 | 9 | 0 |

边界原因：9 条 `anchor_face_review=true`，主要是 side face margin 接近、top/bottom face 接触占比高或 side support weak。它们允许进入 smoke/full，但不能作为 clean case。

### 3. Smoke

命令：

```bash
cd /home/ubuntu/Workspace/spider
bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh smoke 0
```

结论：

| 步骤 | 输入 | 进入 full | 边界/review | 抛弃/跳过 |
|---|---:|---:|---:|---:|
| Smoke | 13 | 13 | 0 | 0 |

Smoke 只验证 scene/override/run_mjwp 最小启动与加载，不声明动力学成功。Smoke 产物已清理，避免与 full run 结果混淆。

### 4. 本地 1 卡串行 full run

命令：

```bash
cd /home/ubuntu/Workspace/spider
tmux new -d -s e028_full_local 'bash /tmp/e028_full_local_cmd.sh'
tmux new -d -s e028_eval_after_full 'bash workspace/core4d_collab_retarget/scripts/eval/watch_E028_eval_after_full.sh'
```

其中 `/tmp/e028_full_local_cmd.sh` 的核心命令是：

```bash
cd /home/ubuntu/Workspace/spider
bash workspace/core4d_collab_retarget/scripts/train/train_E028.sh local 0
```

日志：

- `logs/core4d_collab_retarget/E028/full_local_driver.log`
- `logs/core4d_collab_retarget/E028/<variant>.log`
- `logs/core4d_collab_retarget/E028/eval_after_full_driver.log`

full run 完成时间：

| # | variant | done |
|---:|---|---|
| 1 | `E028_d003_box021_20231011_034_p1_canonical_t02` | 03:14:55 |
| 2 | `E028_d003_box021_20231011_035_p1_canonical_t02` | 03:55:55 |
| 3 | `E028_d003_box021_20231011_035_p2_canonical_t02` | 04:37:56 |
| 4 | `E028_d003_box021_20231018_028_p2_canonical_t02` | 05:06:27 |
| 5 | `E028_d003_box021_20231018_029_p1_canonical_t02` | 05:28:57 |
| 6 | `E028_d003_box021_20231018_029_p2_canonical_t02` | 05:53:58 |
| 7 | `E028_d003_box021_20231018_030_p1_canonical_t02` | 06:22:59 |
| 8 | `E028_d003_box021_20231018_030_p2_canonical_t02` | 06:47:59 |
| 9 | `E028_d003_box021_20231018_031_p2_canonical_t02` | 07:19:30 |
| 10 | `E028_d003_box021_20231020_019_p1_canonical_t02` | 07:53:01 |
| 11 | `E028_d003_box021_20231020_019_p2_canonical_t02` | 08:28:01 |
| 12 | `E028_d003_box021_20231020_020_p1_canonical_t02` | 08:55:02 |
| 13 | `E028_d003_box021_20231020_020_p2_canonical_t02` | 09:23:33 |

结论：

| 步骤 | 输入 | 产出完整 | 边界/review | 抛弃/跳过 |
|---|---:|---:|---:|---:|
| Full run 产物 | 13 | 13 | 0 | 0 |

产物计数：

- root NPZ：13/13
- outdir `trajectory_mjwp.npz`：13/13
- online MP4：13/13
- `failed_local.log`：不存在
- `e028_full_local` 退出状态：0

### 5. Eval / index

watcher 自动执行：

```bash
cd /home/ubuntu/Workspace/spider
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/eval_E028.py --all
.venv/bin/python workspace/core4d_collab_retarget/scripts/eval/index_E028_online_videos.py --force
```

结论：

| 步骤 | 输入 | eval 产出 | 边界/review | 抛弃/跳过 |
|---|---:|---:|---:|---:|
| Eval/index | 13 | 13 | 13 | 0 |

审计结果：

- `comparison.csv`：13 行
- `aggregate_summary.json`：`num_results=13`
- per-case `eval_summary_*.json`：13/13
- per-case `eval_summary_*.csv`：13/13
- online video index：13/13 variants present
- `e028_eval_after_full`：已退出

## 量化汇总

来自 `workspace/core4d_collab_retarget/results/E028/aggregate_summary.json`。

| 指标 | 结果 |
|---|---:|
| manifest rows | 13/13 |
| config ok | 13/13 |
| canonical anchor pass | 13/13 |
| SPIDER object success | 10/13 |
| DynaRetarget object success | 10/13 |
| transport success | 13/13 |
| object transport pass | 10/13 |
| contact preservation ok | 2/13 |
| robot upright ok | 2/13 |
| robot fall detected | 11/13 |
| artifact ok | 0/13 |
| floor/leg ok | 5/13 |
| strict generalization pass | 0/13 |
| mean Epos / Erot | `0.0770m` / `8.63deg` |
| mean contact preservation 5cm | `29.56%` |
| mean deep penetration duration | `18.50%` |

诊断分布：

| diagnostic | count |
|---|---:|
| `anchor_face_review` | 9 |
| `robot_fall_visual_fail` | 3 |
| `contact_preservation_gap` | 1 |

## Final routing

这里区分 run-level 产物完成与 data-construction 可用性：

- run-level：13/13 都成功生成 SPIDER 动力学重定向产物，无运行失败。
- data-construction：0 条达到 clean `进入下一轮`；10 条可进入边界/视频复核；3 条因为 object transport pass 失败，暂不进入下一轮。

| 分组 | 数量 | case |
|---|---:|---|
| 进入下一轮 clean | 0 | - |
| 边界/review | 10 | `20231011_034_p1`, `20231011_035_p1`, `20231018_028_p2`, `20231018_029_p1`, `20231018_029_p2`, `20231018_030_p1`, `20231018_030_p2`, `20231020_019_p1`, `20231020_019_p2`, `20231020_020_p1` |
| 抛弃/不进下一轮 | 3 | `20231011_035_p2`, `20231018_031_p2`, `20231020_020_p2` |

抛弃/不进下一轮原因：这 3 条 `E028_object_transport_pass=false`，同时 `paper_spider_object_success=false` 和 `paper_dynaretarget_object_success=false`。

边界/review 的主因：

- 6 条 object pass 但 `anchor_face_review=true`。
- 3 条 object pass 且 anchor 不需 review，但触发 `robot_fall_visual_fail`。
- 1 条 `20231020_020_p1` object pass、robot upright、anchor 不需 review，但 contact preservation 太低，诊断为 `contact_preservation_gap`。

## Top 3 review bank

这 3 条不是 clean，只是下一步最值得先做 high reasoning 视频复核的边界样本：

| 排名 | case | 理由 |
|---:|---|---|
| 1 | `d003_box021_20231020_020_p1` | object pass、robot upright、anchor 不需 review、leg ok；主要短板是 contact preservation only `6.6%` |
| 2 | `d003_box021_20231020_019_p2` | object pass、robot upright、Epos `0.057m`；但 anchor face review、leg interference 与 contact gap 明显 |
| 3 | `d003_box021_20231011_035_p1` | object pass、Epos 最低 `0.043m`；但 anchor face review 且 robot fall，需要视频确认是否完全不可用 |

## Candidate anchor visualization follow-up

用户后续指定只看 `workspace/core4d_collab_retarget/results/E028/candidates.json` 中的 case。本次新增只覆盖这 5 条候选的当前 anchor 可视化：

| 产物 | 路径 |
|---|---|
| Anchor visual index | `workspace/core4d_collab_retarget/results/E028/anchor_visual/anchor_visual_eval.md` |
| Anchor summary CSV | `workspace/core4d_collab_retarget/results/E028/anchor_visual/anchor_summary.csv` |
| Contact cloud montage | `workspace/core4d_collab_retarget/results/E028/anchor_visual/candidate_anchor_cloud_montage.jpg` |
| Render script | `workspace/core4d_collab_retarget/scripts/eval/render_E028_anchor_visuals.py` |

可视化内容：

- `*_anchor_cloud.png`：object-local contact cloud，黄色星标为 E028 实际 support proxy anchor。
- `*_anchor_motion.mp4`：reference motion 中跟随 object 的 anchor marker，并叠加当前帧 active contact points。
- `*_anchor_motion_sheet.jpg`：每条视频的 6 帧 overview。
- 视频抽帧复核：`workspace/core4d_collab_retarget/results/E028/anchor_visual/video_frame_check_20231018_029_p2_t1.jpg`，确认 motion MP4 非空、视角正常；`-x` anchor 在部分 front/top 视角会被 object/robot 遮挡，因此 face 判断主要看 cloud PNG 与 sheet。

数值复核：

| Variant | Face | Review | Side frac / margin | Face counts `+x/-x/+y/+z/-z` | Anchor to selected-face centroid | Z offset |
|---|---|---|---:|---|---:|---:|
| `20231011_034_p1` | `+x` | `ambiguous_side_face_margin` | `0.434 / 0.028` | `124/116/34/12/0` | `0.143m` | `-0.071m` |
| `20231011_035_p1` | `+x` | `ambiguous_side_face_margin` | `0.407 / 0.016` | `105/101/52/0/0` | `0.143m` | `-0.062m` |
| `20231011_035_p2` | `-x` | `ambiguous_side_face_margin` | `0.402 / 0.019` | `102/107/47/0/10` | `0.388m` | `+0.356m` |
| `20231018_029_p2` | `-x` | `false` | `0.447 / 0.360` | `0/67/13/49/21` | `0.415m` | `+0.330m` |
| `20231020_019_p1` | `-x` | `weak_side_face_support,ambiguous_side_face_margin` | `0.327 / 0.071` | `47/64/50/0/35` | `0.429m` | `+0.417m` |

结论：

- 5 条候选里只有 `20231018_029_p2` 的 side-face 选择是清晰的；其余 4 条 face selection 本身就不稳。
- 即使 `20231018_029_p2` face 清晰，canonical anchor 的高度/横向中心仍与 selected-face contact cloud 明显错位。
- 因此后续不应直接沿用 E028 的 `canonical_z=0.62` 单点 anchor 当作 clean 输入；下一步应先做 anchor policy 修正或按可视化筛出稳定 face/height 的子集。

## Claims 验证

| Claim | 结果 |
|---|---|
| C1 D003 13 条可接入 SPIDER dynamic pipeline | 通过：manifest 13/13，full 13/13 |
| C2 每条生成 canonical support proxy anchor | 通过：13/13 canonical anchor pass；其中 9/13 标记 anchor face review |
| C3 true-freejoint parity | 通过：13/13 config ok、13/13 freejoint parity ok、13/13 no direct wrench |
| C4 本地 1 卡串行完成 13 条 full run | 通过：13/13 root NPZ、outdir trajectory、online MP4，`exit_status=0` |
| C5 eval 分离 object-side success 与 robot-side artifact | 通过：object transport pass 10/13，但 strict generalization pass 0/13；robot fall/contact/artifact 被单独暴露 |
| C6 不把 object tracking pass 误写成 clean trainable | 通过：最终 clean 0/13；10 条进入边界/review，3 条不进下一轮 |

## 结论

E028 的工程链路已经跑通：D003 13 条 Box021 输入全部完成 SPIDER 动力学重定向、全量 eval 和 online video index，没有 run-level 失败。

但按数据构建质量口径，本轮没有 clean 可用 case。主要问题不是生成链路，而是 dynamic retarget 后的 robot-side quality：11/13 触发 fall gate，9/13 anchor face 需要复核，contact preservation 均值只有 `29.56%`，strict artifact ok 为 0/13。后续如果继续推进，应先对 top 3 review bank 做视频复核，再决定是修 anchor face、contact preservation 还是 robot-side stability。

## 遇到的错误

| 错误 | 影响 | 处理 |
|---|---|---|
| 无 full run case 失败 | 无 | `failed_local.log` 不存在 |
| 初始单 case eval 预检曾写出 1 行 comparison | 可能误判为最终结果 | watcher 全量 eval 已覆盖，最终 `comparison.csv` 为 13 行、`aggregate_summary.json num_results=13` |
